//! AI voice conversion effect implementation.

use std::sync::Arc;
use std::path::PathBuf;
use std::collections::VecDeque;
use std::sync::Mutex;
use anyhow::Result;
use tracing::{warn, error, info};
use voclo_dsp::{Effect, EffectMetadata, ParameterValue, ProcessBlock, ProcessContext, Sample};
use crate::model::ModelManager;
use ort::{session::Session, value::{Value, ValueType}, execution_providers::CPUExecutionProvider, tensor::TensorElementType};

/// Factory for creating AI voice converter effects
pub struct AiVoiceConverterFactory {
    metadata: Arc<EffectMetadata>,
    model_manager: Arc<ModelManager>,
}

impl AiVoiceConverterFactory {
    pub fn new(metadata: Arc<EffectMetadata>, model_manager: Arc<ModelManager>) -> Self {
        Self {
            metadata,
            model_manager,
        }
    }
}

impl voclo_dsp::EffectFactory for AiVoiceConverterFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, sample_rate: u32, channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(AiVoiceConverter::new(
            Arc::clone(&self.metadata),
            Arc::clone(&self.model_manager),
            sample_rate,
            channels,
        )?))
    }
}

/// AI voice conversion effect
pub struct AiVoiceConverter {
    metadata: Arc<EffectMetadata>,
    model_manager: Arc<ModelManager>,
    sample_rate: u32,
    channels: usize,
    enabled: bool,
    model_name: Option<String>,
    model_path: Option<PathBuf>,
    pitch_shift: f32,
    mix: f32,
    // Audio buffers for accumulating samples per channel (models typically need larger chunks)
    input_buffers: Vec<VecDeque<Sample>>,
    output_buffers: Vec<VecDeque<Sample>>,
    buffer_size: usize,
    // ONNX session - using Arc<Mutex<>> for thread safety
    onnx_session: Option<Arc<Mutex<Session>>>,
    model_info: Option<crate::model::ModelInfo>,
    model_loaded: bool,
}

impl AiVoiceConverter {
    pub fn new(
        metadata: Arc<EffectMetadata>,
        model_manager: Arc<ModelManager>,
        sample_rate: u32,
        channels: usize,
    ) -> Result<Self> {
        info!("Creating AI voice converter at {} Hz, {} channels", sample_rate, channels);
        
        // Default buffer size: ~100ms at 44.1kHz
        let buffer_size = (sample_rate as f32 * 0.1) as usize;
        
        Ok(Self {
            metadata,
            model_manager,
            sample_rate,
            channels,
            enabled: true,
            model_name: None,
            model_path: None,
            pitch_shift: 0.0,
            mix: 1.0,
            input_buffers: (0..channels)
                .map(|_| VecDeque::with_capacity(buffer_size * 2))
                .collect(),
            output_buffers: (0..channels)
                .map(|_| VecDeque::with_capacity(buffer_size * 2))
                .collect(),
            buffer_size,
            onnx_session: None,
            model_info: None,
            model_loaded: false,
        })
    }
    
    pub fn load_model(&mut self, model_name: &str) -> Result<()> {
        let model_info = self.model_manager
            .get_model(model_name)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", model_name))?;
        
        if !model_info.path.exists() {
            return Err(anyhow::anyhow!("Model file not found: {:?}", model_info.path));
        }
        
        info!("Loading AI model '{}' from: {:?}", model_name, model_info.path);
        
        // Load ONNX model file into memory
        let model_data = std::fs::read(&model_info.path)
            .map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;
        
        // Load ONNX model
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {:?}", e))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| anyhow::anyhow!("Failed to set execution providers: {:?}", e))?
            .commit_from_memory(&model_data)
            .map_err(|e| anyhow::anyhow!("Failed to load model from memory: {:?}", e))?;
        
        info!("✅ ONNX session created successfully");
        
        // Get input/output shapes for debugging
        info!("Model has {} input(s) and {} output(s)", session.inputs.len(), session.outputs.len());
        for (idx, input) in session.inputs.iter().enumerate() {
            info!("Model input {}: name='{}', type={:?}", 
                  idx, input.name, input.input_type);
        }
        for (idx, output) in session.outputs.iter().enumerate() {
            info!("Model output {}: name='{}', type={:?}", 
                  idx, output.name, output.output_type);
        }
        
        self.model_name = Some(model_name.to_string());
        self.model_path = Some(model_info.path.clone());
        self.model_info = Some(model_info.clone());
        self.onnx_session = Some(Arc::new(Mutex::new(session)));
        self.model_loaded = true;
        
        info!("AI model '{}' loaded successfully (path: {:?})", model_name, self.model_path);
        info!("Model info: sample_rate={}, hop_length={}, n_fft={}", 
              model_info.sample_rate, model_info.hop_length, model_info.n_fft);
        
        Ok(())
    }
    
    
    fn process_audio_batch(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        // If no model is loaded, just pass through
        if !self.model_loaded || self.onnx_session.is_none() {
            // Log once that model is not loaded
            static mut LOGGED_NOT_LOADED: bool = false;
            unsafe {
                if !LOGGED_NOT_LOADED {
                    warn!("AI model not loaded - passing through audio unchanged");
                    LOGGED_NOT_LOADED = true;
                }
            }
            output.copy_from_slice(input);
            return Ok(());
        }
        
        let session = self.onnx_session.as_ref().unwrap();
        let mut session_guard = session.lock().map_err(|e| anyhow::anyhow!("Failed to lock ONNX session: {}", e))?;
        
        // Get input and output information first (before preparing inputs)
        if session_guard.inputs.is_empty() {
            return Err(anyhow::anyhow!("Model has no inputs"));
        }
        if session_guard.outputs.is_empty() {
            return Err(anyhow::anyhow!("Model has no outputs"));
        }
        
        // Collect input information to avoid borrowing issues
        let input_infos: Vec<(String, ValueType)> = session_guard.inputs.iter()
            .map(|input| (input.name.clone(), input.input_type.clone()))
            .collect();
        let output_name = session_guard.outputs.first()
            .ok_or_else(|| anyhow::anyhow!("Model has no outputs"))?
            .name.clone();
        
        // Prepare inputs based on model requirements
        // Check each input and create appropriate tensor based on its type
        let mut model_inputs: Vec<(&str, Value)> = Vec::new();
        let mut audio_input_found = false;
        
        for (input_name, input_type) in &input_infos {
            // Extract tensor element type from ValueType
            let element_type = match input_type {
                ValueType::Tensor { ty, .. } => {
                    // ty is TensorElementType
                    *ty
                }
                _ => {
                    warn!("Input '{}' is not a tensor (type: {:?}), skipping", input_name, input_type);
                    continue;
                }
            };
            
            match element_type {
                TensorElementType::Float32 => {
                    // Audio input - prepare as float32 tensor
                    let input_shape = vec![1, input.len()];
                    let input_data = input.to_vec();
                    let input_value = Value::from_array((input_shape, input_data))
                        .map_err(|e| anyhow::anyhow!("Failed to create float32 input tensor for '{}': {:?}", input_name, e))?
                        .into();
                    model_inputs.push((input_name.as_str(), input_value));
                    audio_input_found = true;
                }
                TensorElementType::Int64 => {
                    // Integer input - might be pitch, speaker ID, lengths, or other parameters
                    let default_value: i64 = if input_name.to_lowercase().contains("pitch") {
                        // Use pitch_shift parameter if available
                        self.pitch_shift.round() as i64
                    } else if input_name.to_lowercase().contains("speaker") || input_name.to_lowercase().contains("spk") {
                        // Default speaker ID
                        0
                    } else if input_name.to_lowercase().contains("length") {
                        // Length inputs typically expect the actual length value
                        input.len() as i64
                    } else {
                        // Default value for unknown int64 inputs
                        0
                    };
                    
                    // Determine shape based on input name
                    // Some inputs like "input_lengths" expect rank 1: [batch]
                    // Others might expect rank 2: [batch, value]
                    let (input_shape, input_data) = if input_name.to_lowercase().contains("length") {
                        // Length inputs typically have shape [1] (rank 1)
                        (vec![1], vec![default_value])
                    } else {
                        // Other int64 inputs might expect shape [1, 1] (rank 2)
                        // Try rank 2 first, but this might need adjustment based on model
                        (vec![1, 1], vec![default_value])
                    };
                    
                    let input_value = Value::from_array((input_shape, input_data))
                        .map_err(|e| anyhow::anyhow!("Failed to create int64 input tensor for '{}': {:?}", input_name, e))?
                        .into();
                    model_inputs.push((input_name.as_str(), input_value));
                }
                _ => {
                    warn!("Unsupported input type {:?} for input '{}', skipping", element_type, input_name);
                }
            }
        }
        
        if model_inputs.is_empty() {
            return Err(anyhow::anyhow!("No valid inputs prepared for model"));
        }
        
        if !audio_input_found {
            warn!("⚠️ No float32 audio input found in model - model may not be a voice conversion model");
        }
        
        // Run inference - ort's run() accepts Vec of tuples
        let outputs = session_guard.run(model_inputs)
            .map_err(|e| anyhow::anyhow!("Failed to run inference: {:?}", e))?;
        
        let output_value = outputs.get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("Output '{}' not found", output_name))?;
        
        // Extract tensor from value
        // try_extract_tensor returns (shape, slice)
        let (output_shape, output_slice) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract f32 tensor: {:?}", e))?;
        
        // Handle different output shapes
        // Convert shape from i64 to usize
        let output_shape_usize: Vec<usize> = output_shape.iter()
            .map(|&dim| dim.max(0) as usize)
            .collect();
        
        let output_samples = if output_shape_usize.len() == 2 {
            // Shape: [batch, samples]
            let batch_size = output_shape_usize[0];
            let sample_count = output_shape_usize[1];
            if batch_size > 0 && sample_count > 0 {
                // Flatten to 1D: take first batch
                let end_idx = sample_count.min(output.len());
                output[..end_idx].copy_from_slice(&output_slice[..end_idx]);
                
                // If output is shorter than input, pad with zeros
                if end_idx < output.len() {
                    output[end_idx..].fill(0.0);
                }
                sample_count
            } else {
                output.copy_from_slice(input);
                input.len()
            }
        } else if output_shape_usize.len() == 1 {
            // Shape: [samples]
            let sample_count = output_shape_usize[0].min(output.len());
            output[..sample_count].copy_from_slice(&output_slice[..sample_count]);
            
            // If output is shorter than input, pad with zeros
            if sample_count < output.len() {
                output[sample_count..].fill(0.0);
            }
            sample_count
        } else {
            // Unknown shape, pass through
            warn!("Unexpected output shape: {:?}, passing through", output_shape);
            output.copy_from_slice(input);
            input.len()
        };
        
        // Apply pitch shift if needed (post-processing)
        if self.pitch_shift.abs() > 0.01 {
            let pitch_factor = 1.0 + (self.pitch_shift / 12.0);
            let mut pitch_shifted = vec![0.0; output.len()];
            
            for i in 0..output.len() {
                let source_idx = (i as f32 / pitch_factor) as usize;
                if source_idx < output.len() {
                    let frac = (i as f32 / pitch_factor) - source_idx as f32;
                    if source_idx + 1 < output.len() {
                        pitch_shifted[i] = output[source_idx] * (1.0 - frac) + output[source_idx + 1] * frac;
                    } else {
                        pitch_shifted[i] = output[source_idx];
                    }
                } else {
                    pitch_shifted[i] = 0.0;
                }
            }
            output.copy_from_slice(&pitch_shifted);
        }
        
        // Reduce logging frequency to avoid spam
        static mut LOG_COUNTER: u32 = 0;
        unsafe {
            LOG_COUNTER += 1;
            if LOG_COUNTER % 1000 == 0 && input.len() > 0 {
                info!("🤖 ONNX inference: model='{}', input={} samples, output={} samples, pitch_shift={:.1} semitones", 
                      self.model_name.as_deref().unwrap_or("unknown"), 
                      input.len(), 
                      output_samples,
                      self.pitch_shift);
            }
        }
        
        Ok(())
    }
}

impl Effect for AiVoiceConverter {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, context: &ProcessContext) {
        if !self.enabled {
            return;
        }
        
        // If model is not loaded, pass through
        if !self.model_loaded || self.model_path.is_none() {
            return;
        }
        
        let data = block.data_mut();
        let frames = context.frame_count;
        let channels = context.channels;
        
        // Ensure we have enough buffers for all channels
        if self.input_buffers.len() != channels {
            self.input_buffers = (0..channels)
                .map(|_| VecDeque::with_capacity(self.buffer_size * 2))
                .collect();
            self.output_buffers = (0..channels)
                .map(|_| VecDeque::with_capacity(self.buffer_size * 2))
                .collect();
        }
        
        // Process each channel separately
        for ch in 0..channels {
            // Extract channel data and add to buffer
            {
                let input_buffer = &mut self.input_buffers[ch];
                for frame in 0..frames {
                    let idx = frame * channels + ch;
                    input_buffer.push_back(data[idx]);
                }
            }
            
            // Process when we have enough samples
            let output_batch = {
                let input_buffer = &mut self.input_buffers[ch];
                if input_buffer.len() >= self.buffer_size {
                    let input_batch: Vec<f32> = input_buffer
                        .drain(..self.buffer_size)
                        .collect();
                    
                    let mut output_batch = vec![0.0; input_batch.len()];
                    
                    // Process through AI model (no buffer references held here)
                    if let Err(e) = self.process_audio_batch(&input_batch, &mut output_batch) {
                        error!("AI processing error for channel {}: {}", ch, e);
                        // On error, pass through original
                        output_batch = input_batch;
                    } else {
                        // Debug: check if output differs from input
                        let input_rms: f32 = (input_batch.iter().map(|&x| x * x).sum::<f32>() / input_batch.len() as f32).sqrt();
                        let output_rms: f32 = (output_batch.iter().map(|&x| x * x).sum::<f32>() / output_batch.len() as f32).sqrt();
                        if (input_rms - output_rms).abs() < 1e-6 && input_batch.len() == output_batch.len() {
                            // Check if samples are identical
                            let identical = input_batch.iter().zip(output_batch.iter())
                                .all(|(a, b)| (a - b).abs() < 1e-6);
                            if identical {
                                warn!("⚠️ Model output is identical to input for channel {} - model may not be working", ch);
                            }
                        }
                    }
                    
                    Some(output_batch)
                } else {
                    None
                }
            };
            
            // Add processed output to buffer
            if let Some(output_batch) = output_batch {
                let output_buffer = &mut self.output_buffers[ch];
                for sample in output_batch {
                    output_buffer.push_back(sample);
                }
            }
            
            // Write output if available
            let output_buffer = &mut self.output_buffers[ch];
            let dry_mix = 1.0 - self.mix;
            for frame in 0..frames {
                let idx = frame * channels + ch;
                let dry = data[idx];
                
                if let Some(wet) = output_buffer.pop_front() {
                    data[idx] = dry * dry_mix + wet * self.mix;
                } else {
                    // Not enough processed samples yet, use dry signal
                    data[idx] = dry;
                }
            }
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "model_name" => {
                // Model name is stored as a float index in the parameter system
                // We'll need to handle this differently in the UI
                // For now, this is a placeholder
                warn!("Model name parameter update - use set_model() method instead");
            }
            "pitch_shift" => {
                self.pitch_shift = update.value.clamp(-12.0, 12.0);
                info!("AI voice converter pitch shift set to {} semitones", self.pitch_shift);
            }
            "mix" => {
                self.mix = update.value.clamp(0.0, 1.0);
            }
            _ => {
                warn!(
                    "parameter `{}` not found for effect `{}`",
                    update.id, self.metadata.id
                );
            }
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Downcast helper - allows GUI to access AiVoiceConverter methods
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}


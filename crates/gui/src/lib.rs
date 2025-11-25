//! GUI frontend built with `egui`/`eframe`.
//!
//! The UI exposes basic controls for the effect chain and provides placeholders
//! for forthcoming visualisations and preset management.

use std::{
    collections::VecDeque,
    fs,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context, Result};
use eframe::{egui, App};
use egui::{Color32, ComboBox, Context as EguiContext, Slider, SliderClamping, TopBottomPanel};
use egui_plot::{Line, Plot, PlotPoints};
use rustfft::{num_complex::Complex32, FftPlanner};
use serde::{Deserialize, Serialize};
use tracing::{error, warn, info};
use voclo_audio::{
    AudioEngine, AudioRecorder, DeviceDescriptor, ProfilingMetrics, ProfilingSnapshot,
    SharedPipeline, VisualizationReceiver,
};
use voclo_dsp::{EffectChain, EffectMetadata, ParameterRange, ParameterUnit, ParameterValue};
use voclo_effects::{EffectKind, EffectRegistry};
use voclo_ai::{ModelManager, AiVoiceConverterFactory, AI_VOICE_CONVERSION_ID};

mod settings;
use settings::AppSettings;

mod character_presets;
use character_presets::{CharacterPreset, PresetCategory};

const DEFAULT_PRESET_PATH: &str = "presets/default.json";
const PRESET_VERSION: u32 = 1;
const WAVEFORM_CAPACITY: usize = 4096;
const FFT_SIZE: usize = 1024;

pub struct GuiApp {
    audio_engine: Arc<AudioEngine>,
    pipeline: SharedPipeline,
    registry: EffectRegistry,
    chain_model: Vec<EffectSlot>,
    sample_rate: u32,
    channels: usize,
    preset_path: PathBuf,
    visualization: VisualizationReceiver,
    waveform: VecDeque<f32>,
    spectrum: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
    metrics: Arc<ProfilingMetrics>,
    last_metrics: ProfilingSnapshot,
    recorder: Arc<AudioRecorder>,
    recording: bool,
    last_recorded: Option<PathBuf>,
    recording_error: Option<String>,
    input_devices: Vec<DeviceDescriptor>,
    output_devices: Vec<DeviceDescriptor>,
    selected_input_device: Option<String>,
    selected_output_device: Option<String>,
    device_selection_error: Option<String>,
    settings: AppSettings,
    show_virtual_mic_help: bool,
    // AI model management
    model_manager: Arc<ModelManager>,
    available_models: Vec<String>,
    selected_model_for_slot: std::collections::HashMap<usize, String>,
}

impl GuiApp {
    pub fn new(
        audio_engine: Arc<AudioEngine>,
        pipeline: SharedPipeline,
        sample_rate: u32,
        channels: usize,
        visualization: VisualizationReceiver,
        recorder: Arc<AudioRecorder>,
        metrics: Arc<ProfilingMetrics>,
    ) -> Self {
        let preset_path = PathBuf::from(DEFAULT_PRESET_PATH);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        // Initialize AI model manager
        let model_manager = Arc::new(ModelManager::new());
        let available_models = model_manager.scan_models().unwrap_or_default()
            .into_iter()
            .map(|m| m.name)
            .collect();

        let device_manager = audio_engine.device_manager();
        let input_devices = device_manager.list_inputs().unwrap_or_default();
        let output_devices = device_manager.list_outputs().unwrap_or_default();

        let mut selected_input_device = audio_engine.current_input_device();
        let mut selected_output_device = audio_engine.current_output_device();

        let settings = AppSettings::load().unwrap_or_default();
        // Use saved settings if available, otherwise use current engine settings
        if selected_input_device.is_none() {
            selected_input_device = settings.input_device.clone();
        }
        if selected_output_device.is_none() {
            selected_output_device = settings.output_device.clone();
            
            // If no output device selected and virtual device available, suggest it
            if selected_output_device.is_none() {
                if let Ok(Some(virtual_device)) = device_manager.recommend_virtual_microphone_output() {
                    // Don't auto-select, but we'll show it as recommended
                    tracing::info!("Recommended virtual device: {}", virtual_device.name);
                }
            }
        }

        // Register AI voice conversion effect
        let mut registry = EffectRegistry::with_builtin();
        let ai_metadata = Arc::clone(&voclo_ai::AI_VOICE_CONVERSION_METADATA);
        let ai_factory = AiVoiceConverterFactory::new(
            ai_metadata,
            Arc::clone(&model_manager),
        );
        if let Err(e) = registry.register(Box::new(ai_factory)) {
            warn!("Failed to register AI voice conversion effect: {}", e);
        }

        Self {
            audio_engine,
            pipeline,
            registry,
            chain_model: Vec::new(),
            sample_rate,
            channels,
            preset_path,
            visualization,
            waveform: VecDeque::with_capacity(WAVEFORM_CAPACITY),
            spectrum: Vec::new(),
            fft,
            metrics,
            last_metrics: ProfilingSnapshot::default(),
            recorder,
            recording: false,
            last_recorded: None,
            recording_error: None,
            input_devices,
            output_devices,
            selected_input_device: selected_input_device.clone(),
            selected_output_device: selected_output_device.clone(),
            device_selection_error: None,
            settings,
            show_virtual_mic_help: false,
            model_manager,
            available_models,
            selected_model_for_slot: std::collections::HashMap::new(),
        }
    }

    pub fn run(self) -> eframe::Result<()> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1024.0, 720.0])
                .with_min_inner_size([640.0, 480.0]),
            ..Default::default()
        };

        eframe::run_native(
            "Voclo Voice Morpher",
            options,
            Box::new(move |_cc| Ok(Box::new(self))),
        )
    }

    fn add_effect(&mut self, kind: EffectKind) -> Result<()> {
        self.add_effect_by_id(kind.id())
    }
    
    fn add_effect_by_id(&mut self, effect_id: &str) -> Result<()> {
        let effect = self
            .registry
            .create(effect_id, self.sample_rate, self.channels)?;
        let metadata = self
            .registry
            .metadata_by_id(effect_id)
            .ok_or_else(|| anyhow!("metadata for effect `{}` not found", effect_id))?;
        let slot = EffectSlot::from_metadata(metadata);

        let mut effect = Some(effect);
        let slot_index = self.chain_model.len();
        self.pipeline.with_mut(move |chain| {
            if let Some(effect) = effect.take() {
                chain.add_effect(effect);
            }
        });
        self.chain_model.push(slot);
        
        // If this is an AI effect, try to load the first available model
        if effect_id == AI_VOICE_CONVERSION_ID && !self.available_models.is_empty() {
            if let Some(first_model) = self.available_models.first() {
                let model_name = first_model.clone();
                self.selected_model_for_slot.insert(slot_index, model_name.clone());
                // Load the model immediately
                self.load_model_for_effect(slot_index, &model_name);
            }
        }
        
        Ok(())
    }
    
    fn load_model_for_effect(&mut self, slot_index: usize, model_name: &str) {
        use voclo_ai::AiVoiceConverter;
        
        info!("🔄 Attempting to load model '{}' for effect at slot {}", model_name, slot_index);
        
        // First, make sure the model is registered in ModelManager
        if self.model_manager.get_model(model_name).is_none() {
            // Try to scan models again
            if let Ok(models) = self.model_manager.scan_models() {
                let model_names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
                self.available_models = model_names;
                info!("📋 Scanned models: {:?}", self.available_models);
            }
            
            if self.model_manager.get_model(model_name).is_none() {
                error!("❌ Model '{}' not found in ModelManager. Available models: {:?}", 
                       model_name, self.available_models);
                return;
            }
        }
        
        self.pipeline.with_mut(|chain| {
            if let Some(effect) = nth_effect_mut(chain, slot_index) {
                let effect_id = effect.metadata().id;
                info!("🔍 Found effect at slot {}: ID='{}', name='{}'", 
                      slot_index, effect_id, effect.metadata().name);
                
                // Check if this is the AI effect
                if effect_id != AI_VOICE_CONVERSION_ID {
                    warn!("⚠️ Effect at slot {} is not AI Voice Conversion (ID: '{}')", slot_index, effect_id);
                    return;
                }
                
                // Try to downcast to AiVoiceConverter
                if let Some(ai_effect) = effect.as_any().downcast_mut::<AiVoiceConverter>() {
                    info!("✅ Successfully downcast to AiVoiceConverter, loading model '{}'", model_name);
                    match ai_effect.load_model(model_name) {
                        Ok(()) => {
                            info!("✅ Successfully loaded model '{}' in AI effect at slot {}", model_name, slot_index);
                        }
                        Err(e) => {
                            error!("❌ Failed to load model '{}' in AI effect: {}", model_name, e);
                        }
                    }
                } else {
                    error!("❌ Failed to downcast effect at slot {} to AiVoiceConverter", slot_index);
                    error!("   Effect type: {:?}", std::any::type_name_of_val(effect));
                }
            } else {
                warn!("⚠️ No effect found at slot {}", slot_index);
            }
        });
    }

    fn set_effect_enabled(&mut self, index: usize, enabled: bool) {
        if index >= self.chain_model.len() {
            return;
        }
        self.chain_model[index].enabled = enabled;
        self.pipeline.with_mut(|chain| {
            if let Some(effect) = nth_effect_mut(chain, index) {
                effect.set_enabled(enabled);
            }
        });
    }

    fn set_effect_parameter(&mut self, index: usize, param_id: &'static str, value: f32) {
        self.pipeline.with_mut(|chain| {
            if let Some(effect) = nth_effect_mut(chain, index) {
                effect.update_parameter(ParameterValue { id: param_id, value });
            }
        });
        if let Some(slot) = self.chain_model.get_mut(index) {
            if let Some(control) = slot.parameters.iter_mut().find(|ctrl| ctrl.id == param_id) {
                control.value = value;
            }
        }
    }

    fn save_default_preset(&self) -> Result<()> {
        let preset = self.build_preset();
        if let Some(parent) = self.preset_path.parent() {
            fs::create_dir_all(parent).context("failed to create preset directory")?;
        }
        let data =
            serde_json::to_string_pretty(&preset).context("failed to serialize preset data")?;
        fs::write(&self.preset_path, data)
            .with_context(|| format!("failed to write preset file {}", self.preset_path.display()))
    }

    fn load_default_preset(&mut self) -> Result<()> {
        let data = fs::read_to_string(&self.preset_path)
            .with_context(|| format!("failed to read preset file {}", self.preset_path.display()))?;
        let preset: Preset =
            serde_json::from_str(&data).context("failed to deserialize preset data")?;
        self.apply_preset(preset)
    }

    fn load_character_preset(&mut self, character_preset: &CharacterPreset) -> Result<()> {
        let preset = Preset {
            version: PRESET_VERSION,
            effects: character_preset.get_effects().clone(),
        };
        self.apply_preset(preset)
    }

    fn build_preset(&self) -> Preset {
        Preset {
            version: PRESET_VERSION,
            effects: self
                .chain_model
                .iter()
                .map(EffectSlot::to_preset_effect)
                .collect(),
        }
    }

    fn apply_preset(&mut self, preset: Preset) -> Result<()> {
        if preset.version != PRESET_VERSION {
            warn!(
                "preset version {} does not match expected {}",
                preset.version, PRESET_VERSION
            );
        }

        self.pipeline.with_mut(|chain| chain.clear());
        self.chain_model.clear();

        for effect in preset.effects {
            let metadata = match self.registry.metadata_by_id(&effect.id) {
                Some(metadata) => metadata,
                None => {
                    warn!("preset references unknown effect `{}`", effect.id);
                    continue;
                }
            };

            let mut instance = self
                .registry
                .create(&effect.id, self.sample_rate, self.channels)
                .with_context(|| format!("failed to instantiate effect `{}`", effect.id))?;

            for parameter in &effect.parameters {
                if let Some(spec) = metadata
                    .parameters
                    .iter()
                    .find(|spec| spec.id == parameter.id.as_str())
                {
                    instance.update_parameter(ParameterValue {
                        id: spec.id,
                        value: parameter.value,
                    });
                } else {
                    warn!(
                        "preset references unknown parameter `{}` for effect `{}`",
                        parameter.id, effect.id
                    );
                }
            }

            instance.set_enabled(effect.enabled);

            let mut slot = EffectSlot::from_metadata(metadata.clone());
            slot.apply_preset(&effect);

            let mut instance = Some(instance);
            self.pipeline.with_mut(|chain| {
                if let Some(effect) = instance.take() {
                    chain.add_effect(effect);
                }
            });

            self.chain_model.push(slot);
        }

        Ok(())
    }

    fn refresh_visualization(&mut self) {
        let samples = self.visualization.drain(1024);
        for sample in samples {
            if self.waveform.len() >= WAVEFORM_CAPACITY {
                self.waveform.pop_front();
            }
            self.waveform.push_back(sample);
        }

        if self.waveform.len() >= FFT_SIZE {
            let start = self.waveform.len() - FFT_SIZE;
            let mut buffer: Vec<Complex32> = self
                .waveform
                .iter()
                .skip(start)
                .take(FFT_SIZE)
                .map(|&v| Complex32::new(v, 0.0))
                .collect();
            self.fft.process(&mut buffer);

            self.spectrum.clear();
            let half = FFT_SIZE / 2;
            self.spectrum.reserve(half);
            for bin in 0..half {
                let magnitude = buffer[bin].norm();
                self.spectrum.push(magnitude);
            }
        }

        self.last_metrics = self.metrics.snapshot();
    }

    fn show_visualizations(&self, ui: &mut egui::Ui) {
        let waveform_points = PlotPoints::from_iter(
            self.waveform
                .iter()
                .enumerate()
                .map(|(idx, value)| [idx as f64, *value as f64]),
        );
        Plot::new("waveform_plot")
            .allow_scroll(false)
            .allow_zoom(false)
            .height(140.0)
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new(waveform_points).color(Color32::LIGHT_GREEN));
            });

        if !self.spectrum.is_empty() {
            let spectrum_points = PlotPoints::from_iter(
                self.spectrum
                    .iter()
                    .enumerate()
                    .map(|(idx, value)| [idx as f64, (*value as f64)]),
            );
            Plot::new("spectrum_plot")
                .allow_scroll(false)
                .allow_zoom(false)
                .height(140.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(spectrum_points).color(Color32::LIGHT_BLUE));
                });
        } else {
            ui.label("Insufficient data for spectrum.");
        }
    }

    fn start_recording(&mut self) -> anyhow::Result<()> {
        if self.recording {
            return Ok(());
        }
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let path = format!("recordings/recording-{timestamp}.wav");
        self.recorder
            .start(&path, self.sample_rate, self.channels)?;
        self.recording = true;
        self.last_recorded = Some(PathBuf::from(path));
        self.recording_error = None;
        Ok(())
    }

    fn stop_recording(&mut self) -> anyhow::Result<()> {
        if !self.recording {
            return Ok(());
        }
        let path = self.recorder.stop()?;
        if let Some(path) = path {
            self.last_recorded = Some(path);
        }
        self.recording = false;
        Ok(())
    }

    fn refresh_devices(&mut self) {
        let device_manager = self.audio_engine.device_manager();
        if let Ok(inputs) = device_manager.list_inputs() {
            self.input_devices = inputs;
        }
        if let Ok(outputs) = device_manager.list_outputs() {
            self.output_devices = outputs;
            
            // If no output device selected and virtual device available, suggest it
            if self.selected_output_device.is_none() {
                if let Ok(Some(virtual_device)) = device_manager.recommend_virtual_microphone_output() {
                    // Auto-suggest virtual device but don't auto-select
                    tracing::info!("Found virtual device: {}", virtual_device.name);
                }
            }
        }
    }

    fn apply_device_selection(&mut self) {
        self.device_selection_error = None;
        
        // Stop current streams
        if let Err(err) = self.audio_engine.stop() {
            self.device_selection_error = Some(format!("Failed to stop audio: {}", err));
            return;
        }

        // Restart with new devices
        match self.audio_engine.start_with_devices(
            self.selected_input_device.as_deref(),
            self.selected_output_device.as_deref(),
        ) {
            Ok(stream) => {
                self.sample_rate = stream.sample_rate;
                self.channels = stream.channels as usize;
                self.device_selection_error = None;
                
                // Save settings
                self.settings.input_device = self.selected_input_device.clone();
                self.settings.output_device = self.selected_output_device.clone();
                if let Err(err) = self.settings.save() {
                    tracing::warn!("Failed to save settings: {}", err);
                }
            }
            Err(err) => {
                self.device_selection_error = Some(format!("Failed to start audio: {}", err));
            }
        }
    }
}

impl App for GuiApp {
    fn update(&mut self, ctx: &EguiContext, _frame: &mut eframe::Frame) {
        self.refresh_visualization();

        TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Voclo Voice Morpher");
                ui.add_space(16.0);
                if ui.button("Save preset").clicked() {
                    if let Err(err) = self.save_default_preset() {
                        error!("failed to save preset: {err:?}");
                    }
                }
                if ui.button("Load preset").clicked() {
                    if let Err(err) = self.load_default_preset() {
                        error!("failed to load preset: {err:?}");
                    }
                }
                ui.add_space(16.0);
                if ui.button("Refresh Devices").clicked() {
                    self.refresh_devices();
                }
                if ui.button("💡 Setup Virtual Mic").clicked() {
                    self.show_virtual_mic_help = !self.show_virtual_mic_help;
                }
                ui.add_space(16.0);
                if self.recording {
                    if ui.button("Stop Recording").clicked() {
                        if let Err(err) = self.stop_recording() {
                            error!("failed to stop recording: {err:?}");
                            self.recording_error = Some(err.to_string());
                        }
                    }
                } else if ui.button("Start Recording").clicked() {
                    if let Err(err) = self.start_recording() {
                        error!("failed to start recording: {err:?}");
                        self.recording_error = Some(err.to_string());
                    }
                }
            });
            ui.separator();
            ui.horizontal(|ui| {
                let current_input = self.selected_input_device.as_deref().unwrap_or("Default");
                ComboBox::from_label("Input Device")
                    .selected_text(current_input)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(
                            self.selected_input_device.is_none(),
                            "Default",
                        ).clicked() {
                            self.selected_input_device = None;
                        }
                        for device in &self.input_devices {
                            let selected = self.selected_input_device.as_ref()
                                .map(|s| s == &device.name)
                                .unwrap_or(false);
                            if ui.selectable_label(selected, &device.name).clicked() {
                                self.selected_input_device = Some(device.name.clone());
                            }
                        }
                    });
                ui.add_space(8.0);
                let current_output = self.selected_output_device.as_deref().unwrap_or("Default");
                ComboBox::from_label("Output Device")
                    .selected_text(current_output)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(
                            self.selected_output_device.is_none(),
                            "Default",
                        ).clicked() {
                            self.selected_output_device = None;
                        }
                        for device in &self.output_devices {
                            let selected = self.selected_output_device.as_ref()
                                .map(|s| s == &device.name)
                                .unwrap_or(false);
                            let display_name = if device.is_virtual_device() {
                                format!("🎤 {}", device.name)
                            } else {
                                device.name.clone()
                            };
                            if ui.selectable_label(selected, display_name).clicked() {
                                self.selected_output_device = Some(device.name.clone());
                            }
                        }
                    });
                ui.add_space(8.0);
                if ui.button("Apply Devices").clicked() {
                    self.apply_device_selection();
                }
            });
            
            // Show virtual microphone setup hint
            if let Some(output) = &self.selected_output_device {
                if let Some(device) = self.output_devices.iter().find(|d| &d.name == output) {
                    if device.is_virtual_device() {
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("💡 Virtual Microphone Mode:");
                            ui.label("Set this device as microphone in Telegram/Discord/other apps");
                        });
                        ui.horizontal(|ui| {
                            ui.label("📱 Telegram:");
                            ui.label("Settings → Privacy → Voice Messages → Microphone");
                        });
                        ui.horizontal(|ui| {
                            ui.label("🎮 Discord:");
                            ui.label("Settings → Voice & Video → Input Device");
                        });
                    }
                }
            }
            ui.label(format!(
                "Pipeline: {} effects @ {} Hz / {} ch",
                self.chain_model.len(),
                self.sample_rate,
                self.channels
            ));
            ui.label(format!(
                "Latency avg {:.2} ms | max {:.2} ms | CPU {:.1}%",
                self.last_metrics.average_latency_ms,
                self.last_metrics.max_latency_ms,
                self.last_metrics.average_cpu_percent
            ));
            if let Some(path) = &self.last_recorded {
                ui.label(format!("Last recording: {}", path.display()));
            }
            if let Some(err) = &self.recording_error {
                ui.colored_label(Color32::RED, err);
            }
            if let Some(err) = &self.device_selection_error {
                ui.colored_label(Color32::RED, format!("Device error: {}", err));
            }
        });
        
        // Show virtual microphone setup dialog
        if self.show_virtual_mic_help {
            egui::Window::new("Virtual Microphone Setup")
                .collapsible(false)
                .resizable(true)
                .default_size([500.0, 400.0])
                .show(ctx, |ui| {
                    ui.heading("🎤 How to use Voclo as Virtual Microphone");
                    ui.separator();
                    
                    ui.label("To use Voclo's processed voice in Telegram, Discord, or other apps:");
                    ui.add_space(10.0);
                    
                    ui.label("1️⃣ Install a virtual audio cable:");
                    ui.indent("virtual_mic_indent1", |ui| {
                        ui.label("• Windows: VB-Audio Cable (free) - https://vb-audio.com/Cable/");
                        ui.label("• macOS: BlackHole (free) - https://github.com/ExistentialAudio/BlackHole");
                        ui.label("• Linux: PulseAudio null-sink (built-in)");
                    });
                    
                    ui.add_space(10.0);
                    ui.label("2️⃣ Configure Voclo:");
                    ui.indent("virtual_mic_indent2", |ui| {
                        ui.label("• Select your real microphone as Input Device");
                        ui.label("• Select the virtual cable (🎤 device) as Output Device");
                        ui.label("• Click 'Apply Devices'");
                        ui.label("• Add effects and configure them");
                    });
                    
                    ui.add_space(10.0);
                    ui.label("3️⃣ Configure your app (Telegram/Discord/etc.):");
                    ui.indent("virtual_mic_indent3", |ui| {
                        ui.label("Telegram:");
                        ui.indent("telegram_indent", |ui| {
                            ui.label("Settings → Privacy → Voice Messages → Microphone");
                            ui.label("Select the virtual cable (🎤 device)");
                        });
                        ui.add_space(5.0);
                        ui.label("Discord:");
                        ui.indent("discord_indent", |ui| {
                            ui.label("Settings → Voice & Video → Input Device");
                            ui.label("Select the virtual cable (🎤 device)");
                        });
                    });
                    
                    ui.add_space(15.0);
                    ui.separator();
                    
                    let device_manager = self.audio_engine.device_manager();
                    if let Ok(virtual_outputs) = device_manager.find_virtual_outputs() {
                        if virtual_outputs.is_empty() {
                            ui.colored_label(Color32::YELLOW, "⚠️ No virtual audio devices found!");
                            ui.label("Please install a virtual audio cable (see step 1) and click 'Refresh Devices'");
                        } else {
                            ui.label("✅ Found virtual devices:");
                            for device in virtual_outputs {
                                ui.label(format!("  • 🎤 {}", device.name));
                            }
                        }
                    }
                    
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        if ui.button("Close").clicked() {
                            self.show_virtual_mic_help = false;
                        }
                        if ui.button("Refresh Devices").clicked() {
                            self.refresh_devices();
                        }
                    });
                });
        }

        egui::SidePanel::left("effect_library")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Effects");
                ui.separator();
                let metadata = self.registry.metadata();
                ComboBox::from_label("Add effect")
                    .selected_text("Select…")
                    .show_ui(ui, |ui| {
                        for meta in metadata {
                            let id = meta.id.to_string();
                            if ui
                                .selectable_label(false, meta.name)
                                .clicked()
                            {
                                if let Some(kind) = effect_kind_from_id(&id) {
                                    if let Err(err) = self.add_effect(kind) {
                                        error!("failed to add effect `{id}`: {err:?}");
                                    }
                                } else if id == AI_VOICE_CONVERSION_ID {
                                    // Handle AI voice conversion effect
                                    if let Err(err) = self.add_effect_by_id(&id) {
                                        error!("failed to add AI effect `{id}`: {err:?}");
                                    }
                                }
                            }
                        }
                    });
                
                ui.separator();
                ui.heading("Character Presets");
                
                let presets = CharacterPreset::all();
                
                // Group presets by category
                let anime_presets: Vec<_> = presets.iter().filter(|p| p.category == PresetCategory::Anime).collect();
                let game_presets: Vec<_> = presets.iter().filter(|p| p.category == PresetCategory::Game).collect();
                let robot_presets: Vec<_> = presets.iter().filter(|p| p.category == PresetCategory::Robot).collect();
                let monster_presets: Vec<_> = presets.iter().filter(|p| p.category == PresetCategory::Monster || p.category == PresetCategory::Demon).collect();
                let fantasy_presets: Vec<_> = presets.iter().filter(|p| p.category == PresetCategory::Fantasy).collect();
                
                ui.collapsing("🎌 Anime", |ui| {
                    for preset in anime_presets {
                        if ui.button(preset.name).clicked() {
                            if let Err(err) = self.load_character_preset(preset) {
                                error!("failed to load character preset `{}`: {err:?}", preset.name);
                            }
                        }
                        if !preset.description.is_empty() {
                            ui.label(format!("  {}", preset.description));
                        }
                    }
                });
                
                ui.collapsing("🎮 Game", |ui| {
                    for preset in game_presets {
                        if ui.button(preset.name).clicked() {
                            if let Err(err) = self.load_character_preset(preset) {
                                error!("failed to load character preset `{}`: {err:?}", preset.name);
                            }
                        }
                        if !preset.description.is_empty() {
                            ui.label(format!("  {}", preset.description));
                        }
                    }
                });
                
                ui.collapsing("🤖 Robot", |ui| {
                    for preset in robot_presets {
                        if ui.button(preset.name).clicked() {
                            if let Err(err) = self.load_character_preset(preset) {
                                error!("failed to load character preset `{}`: {err:?}", preset.name);
                            }
                        }
                        if !preset.description.is_empty() {
                            ui.label(format!("  {}", preset.description));
                        }
                    }
                });
                
                ui.collapsing("👹 Monster", |ui| {
                    for preset in monster_presets {
                        if ui.button(preset.name).clicked() {
                            if let Err(err) = self.load_character_preset(preset) {
                                error!("failed to load character preset `{}`: {err:?}", preset.name);
                            }
                        }
                        if !preset.description.is_empty() {
                            ui.label(format!("  {}", preset.description));
                        }
                    }
                });
                
                ui.collapsing("🧙 Fantasy", |ui| {
                    for preset in fantasy_presets {
                        if ui.button(preset.name).clicked() {
                            if let Err(err) = self.load_character_preset(preset) {
                                error!("failed to load character preset `{}`: {err:?}", preset.name);
                            }
                        }
                        if !preset.description.is_empty() {
                            ui.label(format!("  {}", preset.description));
                        }
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Effect Chain");
            ui.separator();
            if self.chain_model.is_empty() {
                ui.label("No effects in chain. Select an effect from the list to add one.");
            } else {
                for index in 0..self.chain_model.len() {
                    let mut changed_enabled = None;
                    let mut parameter_updates: Vec<(&'static str, f32)> = Vec::new();
                    {
                        let slot = &mut self.chain_model[index];
                        let mut enabled = slot.enabled;
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut enabled, "").changed() {
                                changed_enabled = Some(enabled);
                            }
                            ui.label(&slot.display_name);
                        });
                        if let Some(new_enabled) = changed_enabled {
                            slot.enabled = new_enabled;
                        }

                        for control in slot.parameters.iter_mut() {
                            ui.horizontal(|ui| {
                                let mut slider = Slider::new(
                                    &mut control.value,
                                    control.range.min..=control.range.max,
                                )
                                .text(control.label)
                                .clamping(SliderClamping::Always);

                                if control.range.step > 0.0 {
                                    slider = slider.step_by(control.range.step as f64);
                                }

                                if ui.add(slider).changed() {
                                    parameter_updates.push((control.id, control.value));
                                }

                                match control.unit {
                                    ParameterUnit::Percent => {
                                        ui.label(format!("{:.0}%", control.value * 100.0));
                                    }
                                    ParameterUnit::Decibels => {
                                        ui.label(format!("{:.1} dB", control.value));
                                    }
                                    ParameterUnit::Hertz => {
                                        ui.label(format!("{:.0} Hz", control.value));
                                    }
                                    ParameterUnit::Seconds => {
                                        ui.label(format!("{:.3} s", control.value));
                                    }
                                    ParameterUnit::Milliseconds => {
                                        ui.label(format!("{:.1} ms", control.value));
                                    }
                                    ParameterUnit::Ratio => {
                                        ui.label(format!("{:.2}", control.value));
                                    }
                                    ParameterUnit::Custom(ref unit) => {
                                        ui.label(format!("{:.2} {}", control.value, unit));
                                    }
                                    ParameterUnit::None => {
                                        ui.label(format!("{:.2}", control.value));
                                    }
                                }
                            });
                        }
                        
                        // Show model selector for AI voice conversion effects
                        if slot.effect_id == AI_VOICE_CONVERSION_ID {
                            ui.separator();
                            ui.label("🤖 AI Model:");
                            let current_model = self.selected_model_for_slot
                                .get(&index)
                                .cloned()
                                .unwrap_or_else(|| "No model selected".to_string());
                            
                            let mut model_to_load: Option<String> = None;
                            
                            ComboBox::from_id_source(format!("ai_model_{}", index).as_str())
                                .selected_text(&current_model)
                                .show_ui(ui, |ui| {
                                    if self.available_models.is_empty() {
                                        ui.label("No models found");
                                        ui.label("Place .onnx files in the 'models' directory");
                                    } else {
                                        for model_name in &self.available_models {
                                            let selected = self.selected_model_for_slot
                                                .get(&index)
                                                .map(|s| s == model_name)
                                                .unwrap_or(false);
                                            if ui.selectable_label(selected, model_name).clicked() {
                                                self.selected_model_for_slot.insert(index, model_name.clone());
                                                // Store model name to load after UI scope ends
                                                model_to_load = Some(model_name.clone());
                                            }
                                        }
                                    }
                                });
                            
                            // Load model after UI scope ends to avoid borrow issues
                            if let Some(model_name) = model_to_load {
                                self.load_model_for_effect(index, &model_name);
                            }
                            
                            if ui.button("🔄 Refresh Models").clicked() {
                                if let Ok(models) = self.model_manager.scan_models() {
                                    self.available_models = models.into_iter().map(|m| m.name).collect();
                                }
                            }
                        }
                    }
                    if let Some(new_enabled) = changed_enabled {
                        self.set_effect_enabled(index, new_enabled);
                    }
                    for (param_id, value) in parameter_updates {
                        self.set_effect_parameter(index, param_id, value);
                    }
                    ui.separator();
                }
            }
        });

        egui::SidePanel::right("visualizations")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.heading("Visualizations");
                ui.separator();
                self.show_visualizations(ui);
            });
    }
}

pub fn nth_effect_mut<'a>(
    chain: &'a mut EffectChain,
    index: usize,
) -> Option<&'a mut Box<dyn voclo_dsp::Effect>> {
    chain.iter_mut().enumerate().find_map(|(i, effect)| {
        if i == index {
            Some(effect)
        } else {
            None
        }
    })
}

fn effect_kind_from_id(id: &str) -> Option<EffectKind> {
    [
        EffectKind::Bypass,
        EffectKind::Gain,
        EffectKind::PitchShift,
        EffectKind::FormantShift,
        EffectKind::Robotizer,
        EffectKind::Demon,
        EffectKind::Reverb,
        EffectKind::Distortion,
        EffectKind::Filter,
    ]
    .into_iter()
    .find(|kind| kind.id() == id)
}

struct EffectSlot {
    effect_id: String,
    display_name: String,
    enabled: bool,
    parameters: Vec<ParameterControl>,
}

impl EffectSlot {
    fn from_metadata(metadata: Arc<EffectMetadata>) -> Self {
        let display_name = metadata.name.to_string();
        let effect_id = metadata.id.to_string();
        let parameters = metadata
            .parameters
            .iter()
            .map(|spec| ParameterControl {
                id: spec.id,
                label: spec.name,
                range: spec.range.clone(),
                unit: spec.unit.clone(),
                value: spec.default,
            })
            .collect();

        Self {
            effect_id,
            display_name,
            enabled: true,
            parameters,
        }
    }

    fn apply_preset(&mut self, preset: &PresetEffect) {
        self.enabled = preset.enabled;
        for control in &mut self.parameters {
            if let Some(parameter) = preset
                .parameters
                .iter()
                .find(|parameter| parameter.id == control.id)
            {
                control.value = parameter.value;
            }
        }
    }

    fn to_preset_effect(&self) -> PresetEffect {
        PresetEffect {
            id: self.effect_id.clone(),
            enabled: self.enabled,
            parameters: self
                .parameters
                .iter()
                .map(|control| PresetParameter {
                    id: control.id.to_string(),
                    value: control.value,
                })
                .collect(),
        }
    }
}

struct ParameterControl {
    id: &'static str,
    label: &'static str,
    range: ParameterRange,
    unit: ParameterUnit,
    value: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Preset {
    pub version: u32,
    pub effects: Vec<PresetEffect>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetEffect {
    pub id: String,
    pub enabled: bool,
    pub parameters: Vec<PresetParameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetParameter {
    pub id: String,
    pub value: f32,
}

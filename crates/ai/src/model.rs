//! Model management for AI voice conversion.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::{Context, Result};
use tracing::{info, warn};

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub sample_rate: u32,
    pub hop_length: usize,
    pub n_fft: usize,
}

/// Global model manager
pub struct ModelManager {
    models: Arc<RwLock<std::collections::HashMap<String, ModelInfo>>>,
    model_dir: PathBuf,
}

impl ModelManager {
    pub fn new() -> Self {
        let model_dir = Self::default_model_dir();
        
        // Create model directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(&model_dir) {
            warn!("Failed to create model directory: {}", e);
        }
        
        Self {
            models: Arc::new(RwLock::new(std::collections::HashMap::new())),
            model_dir,
        }
    }
    
    pub fn default_model_dir() -> PathBuf {
        // First, try to find models directory in the project root (for development)
        // Check if we're running from target/debug or target/release
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                // If running from target/debug or target/release, go up to project root
                if exe_dir.ends_with("target/debug") || exe_dir.ends_with("target/release") {
                    if let Some(project_root) = exe_dir.parent().and_then(|p| p.parent()) {
                        let models_dir = project_root.join("models");
                        if models_dir.exists() {
                            return models_dir;
                        }
                    }
                }
                // Otherwise, try models next to executable (for distribution)
                let models_dir = exe_dir.join("models");
                if models_dir.exists() {
                    return models_dir;
                }
            }
        }
        
        // Fallback: try current working directory
        if let Ok(cwd) = std::env::current_dir() {
            let models_dir = cwd.join("models");
            if models_dir.exists() {
                return models_dir;
            }
        }
        
        // Last fallback: relative to current directory
        PathBuf::from("models")
    }
    
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
    
    /// Scan for available models in the model directory
    pub fn scan_models(&self) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();
        
        info!("🔍 Scanning for models in: {:?}", self.model_dir);
        
        if !self.model_dir.exists() {
            warn!("⚠️ Model directory does not exist: {:?}", self.model_dir);
            return Ok(models);
        }
        
        for entry in std::fs::read_dir(&self.model_dir)
            .context("Failed to read model directory")? 
        {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                // Try to extract model info from filename
                // Format: name_sr{rate}_hop{hop}_nfft{nfft}.onnx
                // Or just: name.onnx (use defaults)
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                let (sample_rate, hop_length, n_fft) = Self::parse_model_filename(&name);
                
                let model_info = ModelInfo {
                    name: name.clone(),
                    path: path.clone(),
                    sample_rate,
                    hop_length,
                    n_fft,
                };
                
                // Register the model
                self.register_model(model_info.clone());
                
                models.push(model_info);
                info!("  ✓ Found model: '{}' at {:?}", name, path);
            }
        }
        
        info!("✅ Found {} model(s) total", models.len());
        Ok(models)
    }
    
    /// Parse model filename to extract parameters
    /// Format: name_sr44100_hop512_nfft2048.onnx
    fn parse_model_filename(name: &str) -> (u32, usize, usize) {
        let mut sample_rate = 44100;
        let mut hop_length = 512;
        let mut n_fft = 2048;
        
        // Try to extract parameters from filename
        for part in name.split('_') {
            if part.starts_with("sr") {
                if let Ok(sr) = part[2..].parse::<u32>() {
                    sample_rate = sr;
                }
            } else if part.starts_with("hop") {
                if let Ok(hop) = part[3..].parse::<usize>() {
                    hop_length = hop;
                }
            } else if part.starts_with("nfft") {
                if let Ok(nfft) = part[4..].parse::<usize>() {
                    n_fft = nfft;
                }
            }
        }
        
        (sample_rate, hop_length, n_fft)
    }
    
    /// Register a model
    pub fn register_model(&self, info: ModelInfo) {
        let mut models = self.models.write();
        models.insert(info.name.clone(), info.clone());
        info!("✅ Registered model: '{}' (total: {})", info.name, models.len());
    }
    
    /// Get a model by name
    pub fn get_model(&self, name: &str) -> Option<ModelInfo> {
        let models = self.models.read();
        let result = models.get(name).cloned();
        if result.is_none() {
            let available: Vec<String> = models.keys().cloned().collect();
            tracing::debug!("Model '{}' not found. Available: {:?}", name, available);
        }
        result
    }
    
    /// List all registered models
    pub fn list_models(&self) -> Vec<String> {
        let models = self.models.read();
        models.keys().cloned().collect()
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}


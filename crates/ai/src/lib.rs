//! AI-powered voice conversion for anime character voices.
//!
//! This module provides neural network-based voice conversion using ONNX models.
//! Supports RVC (Retrieval-based Voice Conversion) and similar models.

mod model;
mod voice_converter;

pub use model::{ModelManager, ModelInfo};
pub use voice_converter::{AiVoiceConverter, AiVoiceConverterFactory};

use voclo_dsp::{EffectMetadata, ParameterRange, ParameterUnit, ParameterSpec};
use std::sync::Arc;
use once_cell::sync::Lazy;

/// Effect kind for AI voice conversion
pub const AI_VOICE_CONVERSION_ID: &str = "ai_voice_conversion";

pub static AI_VOICE_CONVERSION_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        AI_VOICE_CONVERSION_ID,
        "AI Voice Conversion",
        &[
            ParameterSpec {
                id: "pitch_shift",
                name: "Pitch Shift (semitones)",
                range: ParameterRange {
                    min: -12.0,
                    max: 12.0,
                    step: 0.1,
                },
                default: 0.0,
                unit: ParameterUnit::None,
            },
            ParameterSpec {
                id: "mix",
                name: "Dry/Wet",
                range: ParameterRange {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default: 1.0,
                unit: ParameterUnit::Percent,
            },
        ],
    ))
});


//! Character presets for voice morphing.
//!
//! This module provides predefined effect chains for creating character voices
//! from anime, games, and other media.

use serde::{Deserialize, Serialize};

pub use super::{PresetEffect, PresetParameter};

/// Categories for character presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresetCategory {
    Anime,
    Game,
    Robot,
    Monster,
    Demon,
    Fantasy,
    SciFi,
    Other,
}

impl PresetCategory {
    pub fn name(&self) -> &'static str {
        match self {
            PresetCategory::Anime => "Anime",
            PresetCategory::Game => "Game",
            PresetCategory::Robot => "Robot",
            PresetCategory::Monster => "Monster",
            PresetCategory::Demon => "Demon",
            PresetCategory::Fantasy => "Fantasy",
            PresetCategory::SciFi => "Sci-Fi",
            PresetCategory::Other => "Other",
        }
    }
}

/// Metadata for a character preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub category: PresetCategory,
    pub author: &'static str,
    pub icon: &'static str,
}

impl CharacterPreset {
    /// Get all available character presets.
    pub fn all() -> Vec<&'static CharacterPreset> {
        vec![
            &ANIME_VOICE,
            &ANIME_GIRL_VOICE,
            &ANIME_BOY_VOICE,
            &CHIPMUNK_VOICE,
            &MONSTER_VOICE,
            &ROBOT_VOICE,
            &DEMON_VOICE,
            &DARTH_VADER_VOICE,
        ]
    }

    /// Get the effects chain for this preset.
    pub fn get_effects(&self) -> Vec<PresetEffect> {
        match self.id {
            "anime_voice" => anime_voice_effects(),
            "anime_girl_voice" => anime_girl_voice_effects(),
            "anime_boy_voice" => anime_boy_voice_effects(),
            "chipmunk_voice" => chipmunk_voice_effects(),
            "monster_voice" => monster_voice_effects(),
            "robot_voice" => robot_voice_effects(),
            "demon_voice" => demon_voice_effects(),
            "darth_vader_voice" => darth_vader_voice_effects(),
            _ => vec![],
        }
    }

    /// Find preset by ID.
    pub fn by_id(id: &str) -> Option<&'static CharacterPreset> {
        Self::all().into_iter().find(|p| p.id == id)
    }
}

// === Character Presets Definitions ===

/// Anime Voice - High-pitched, bright voice with elevated formants
/// Creates a typical anime character voice (cute girl, shounen protagonist)
pub const ANIME_VOICE: CharacterPreset = CharacterPreset {
    id: "anime_voice",
    name: "Anime Voice",
    description: "High-pitched, bright voice with elevated formants - perfect for anime characters",
    category: PresetCategory::Anime,
    author: "Voclo Team",
    icon: "🎌",
};

/// Anime Girl Voice - Very high-pitched, bright and cute voice
/// Optimized for female anime characters (moe, kawaii style)
pub const ANIME_GIRL_VOICE: CharacterPreset = CharacterPreset {
    id: "anime_girl_voice",
    name: "Anime Girl Voice",
    description: "Very high-pitched and bright - perfect for cute anime girls",
    category: PresetCategory::Anime,
    author: "Voclo Team",
    icon: "🌸",
};

/// Anime Boy Voice - Moderately high-pitched, energetic voice
/// Optimized for shounen anime protagonists
pub const ANIME_BOY_VOICE: CharacterPreset = CharacterPreset {
    id: "anime_boy_voice",
    name: "Anime Boy Voice",
    description: "Energetic and bright - perfect for shounen protagonists",
    category: PresetCategory::Anime,
    author: "Voclo Team",
    icon: "⚡",
};

/// Chipmunk Voice - Very high pitch with raised formants
/// Creates a cute, squeaky voice
pub const CHIPMUNK_VOICE: CharacterPreset = CharacterPreset {
    id: "chipmunk_voice",
    name: "Chipmunk Voice",
    description: "Very high-pitched voice with raised formants - cute and squeaky",
    category: PresetCategory::Fantasy,
    author: "Voclo Team",
    icon: "🐿️",
};

/// Monster Voice - Low, dark voice with distortion and reverb
/// Creates a scary monster voice
pub const MONSTER_VOICE: CharacterPreset = CharacterPreset {
    id: "monster_voice",
    name: "Monster Voice",
    description: "Low, dark voice with distortion and reverb - perfect for scary monsters",
    category: PresetCategory::Monster,
    author: "Voclo Team",
    icon: "👹",
};

/// Robot Voice - Robotic effect with low-pass filter
/// Creates a classic robot voice
pub const ROBOT_VOICE: CharacterPreset = CharacterPreset {
    id: "robot_voice",
    name: "Robot Voice",
    description: "Robotic effect with low-pass filter - classic sci-fi robot",
    category: PresetCategory::Robot,
    author: "Voclo Team",
    icon: "🤖",
};

/// Demon Voice - Spectral inversion with distortion and reverb
/// Creates an evil demon voice
pub const DEMON_VOICE: CharacterPreset = CharacterPreset {
    id: "demon_voice",
    name: "Demon Voice",
    description: "Spectral inversion with distortion and reverb - evil and menacing",
    category: PresetCategory::Demon,
    author: "Voclo Team",
    icon: "👹",
};

/// Darth Vader Voice - Low pitch, deep formants, low-pass filter
/// Creates a deep, menacing voice like Darth Vader
pub const DARTH_VADER_VOICE: CharacterPreset = CharacterPreset {
    id: "darth_vader_voice",
    name: "Darth Vader Voice",
    description: "Low pitch, deep formants, and low-pass filter - deep and menacing",
    category: PresetCategory::SciFi,
    author: "Voclo Team",
    icon: "🛸",
};

// === Effect Chain Builders ===

fn anime_voice_effects() -> Vec<PresetEffect> {
    vec![
        // Pitch shift up (improved: +5 semitones for more anime-like sound)
        PresetEffect {
            id: "pitch_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "semitones".to_string(),
                    value: 5.0, // +5 semitones (perfect fourth up) - more anime-like
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 1.0, // 100% wet
                },
            ],
        },
        // Formant shift up to make voice brighter and more youthful (increased)
        PresetEffect {
            id: "formant_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "shift".to_string(),
                    value: 1.45, // Shift formants up 45% - more pronounced anime character
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.85, // 85% wet for better balance
                },
            ],
        },
        // Light reverb for spatial presence (anime voices often have this quality)
        PresetEffect {
            id: "reverb".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "decay".to_string(),
                    value: 0.8, // Short decay for subtle space
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.15, // 15% wet - very subtle
                },
            ],
        },
        // Light gain boost for presence
        PresetEffect {
            id: "gain".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "gain_db".to_string(),
                    value: 2.5, // +2.5 dB for better presence
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 1.0,
                },
            ],
        },
    ]
}

fn anime_girl_voice_effects() -> Vec<PresetEffect> {
    vec![
        // Very high pitch shift for cute girl voice
        PresetEffect {
            id: "pitch_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "semitones".to_string(),
                    value: 6.0, // +6 semitones (tritone up) - very high and cute
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 1.0,
                },
            ],
        },
        // Strong formant shift for bright, youthful sound
        PresetEffect {
            id: "formant_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "shift".to_string(),
                    value: 1.5, // Shift formants up 50% - maximum brightness
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.9, // 90% wet
                },
            ],
        },
        // Subtle reverb for kawaii quality
        PresetEffect {
            id: "reverb".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "decay".to_string(),
                    value: 0.6, // Short, bright reverb
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.12, // 12% wet - very subtle
                },
            ],
        },
        // Gain boost for clarity
        PresetEffect {
            id: "gain".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "gain_db".to_string(),
                    value: 3.0, // +3 dB
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 1.0,
                },
            ],
        },
    ]
}

fn anime_boy_voice_effects() -> Vec<PresetEffect> {
    vec![
        // Moderate pitch shift for energetic boy voice
        PresetEffect {
            id: "pitch_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "semitones".to_string(),
                    value: 4.5, // +4.5 semitones - energetic but not too high
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 1.0,
                },
            ],
        },
        // Moderate formant shift for youthful but not overly bright
        PresetEffect {
            id: "formant_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "shift".to_string(),
                    value: 1.35, // Shift formants up 35%
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.8, // 80% wet
                },
            ],
        },
        // Light reverb for presence
        PresetEffect {
            id: "reverb".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "decay".to_string(),
                    value: 1.0, // Medium decay
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.18, // 18% wet
                },
            ],
        },
        // Gain boost
        PresetEffect {
            id: "gain".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "gain_db".to_string(),
                    value: 2.0, // +2 dB
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 1.0,
                },
            ],
        },
    ]
}

fn chipmunk_voice_effects() -> Vec<PresetEffect> {
    vec![
        // Very high pitch shift
        PresetEffect {
            id: "pitch_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "semitones".to_string(),
                    value: 8.0, // +8 semitones (octave up)
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 1.0,
                },
            ],
        },
        // Formant shift way up
        PresetEffect {
            id: "formant_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "shift".to_string(),
                    value: 1.5, // Shift formants up 50%
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.9,
                },
            ],
        },
    ]
}

fn monster_voice_effects() -> Vec<PresetEffect> {
    vec![
        // Pitch shift down
        PresetEffect {
            id: "pitch_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "semitones".to_string(),
                    value: -6.0, // -6 semitones (minor third down)
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 1.0,
                },
            ],
        },
        // Formant shift down for deeper voice
        PresetEffect {
            id: "formant_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "shift".to_string(),
                    value: 0.7, // Shift formants down 30%
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.8,
                },
            ],
        },
        // Heavy distortion for growl
        PresetEffect {
            id: "distortion".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "drive".to_string(),
                    value: 0.7, // High drive
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.6, // 60% wet
                },
            ],
        },
        // Reverb for cavernous effect
        PresetEffect {
            id: "reverb".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "decay".to_string(),
                    value: 2.5, // Long decay
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.4, // 40% wet
                },
            ],
        },
        // Low-pass filter to darken
        PresetEffect {
            id: "filter".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "cutoff".to_string(),
                    value: 4000.0, // Low cutoff
                },
                PresetParameter {
                    id: "resonance".to_string(),
                    value: 0.707,
                },
            ],
        },
    ]
}

fn robot_voice_effects() -> Vec<PresetEffect> {
    vec![
        // Robotizer effect (core of robot voice)
        PresetEffect {
            id: "robotizer".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "window".to_string(),
                    value: 30.0, // 30ms window
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.9, // 90% wet
                },
            ],
        },
        // Low-pass filter for muffled effect
        PresetEffect {
            id: "filter".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "cutoff".to_string(),
                    value: 3000.0, // Low cutoff
                },
                PresetParameter {
                    id: "resonance".to_string(),
                    value: 0.707,
                },
            ],
        },
        // Light pitch shift down for mechanical feel
        PresetEffect {
            id: "pitch_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "semitones".to_string(),
                    value: -2.0, // -2 semitones
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.6, // 60% wet
                },
            ],
        },
    ]
}

fn demon_voice_effects() -> Vec<PresetEffect> {
    vec![
        // Spectral inversion (core of demon voice)
        PresetEffect {
            id: "demon".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.8, // 80% wet
                },
            ],
        },
        // Heavy distortion
        PresetEffect {
            id: "distortion".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "drive".to_string(),
                    value: 0.8, // Very high drive
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.7, // 70% wet
                },
            ],
        },
        // Dark reverb
        PresetEffect {
            id: "reverb".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "decay".to_string(),
                    value: 3.0, // Very long decay
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.5, // 50% wet
                },
            ],
        },
        // Low-pass filter
        PresetEffect {
            id: "filter".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "cutoff".to_string(),
                    value: 3500.0, // Low cutoff
                },
                PresetParameter {
                    id: "resonance".to_string(),
                    value: 0.707,
                },
            ],
        },
    ]
}

fn darth_vader_voice_effects() -> Vec<PresetEffect> {
    vec![
        // Deep pitch shift down
        PresetEffect {
            id: "pitch_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "semitones".to_string(),
                    value: -8.0, // -8 semitones (octave down)
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.9, // 90% wet
                },
            ],
        },
        // Formant shift down for very deep voice
        PresetEffect {
            id: "formant_shift".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "shift".to_string(),
                    value: 0.65, // Shift formants down 35%
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.85, // 85% wet
                },
            ],
        },
        // Low-pass filter for muffled helmet effect
        PresetEffect {
            id: "filter".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "cutoff".to_string(),
                    value: 2500.0, // Very low cutoff
                },
                PresetParameter {
                    id: "resonance".to_string(),
                    value: 0.707,
                },
            ],
        },
        // Subtle reverb for echo effect
        PresetEffect {
            id: "reverb".to_string(),
            enabled: true,
            parameters: vec![
                PresetParameter {
                    id: "decay".to_string(),
                    value: 1.5, // Medium decay
                },
                PresetParameter {
                    id: "mix".to_string(),
                    value: 0.3, // 30% wet
                },
            ],
        },
    ]
}

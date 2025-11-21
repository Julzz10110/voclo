use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub version: u32,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
}

impl AppSettings {
    pub const CURRENT_VERSION: u32 = 1;
    const SETTINGS_PATH: &'static str = "voclo_settings.json";

    pub fn load() -> Result<Self> {
        if !Path::new(Self::SETTINGS_PATH).exists() {
            return Ok(Self::default());
        }

        let data = fs::read_to_string(Self::SETTINGS_PATH)
            .context("failed to read settings file")?;
        let settings: AppSettings = serde_json::from_str(&data)
            .context("failed to parse settings file")?;

        if settings.version != Self::CURRENT_VERSION {
            tracing::warn!(
                "Settings version {} does not match current version {}",
                settings.version,
                Self::CURRENT_VERSION
            );
        }

        Ok(settings)
    }

    pub fn save(&self) -> Result<()> {
        let data = serde_json::to_string_pretty(self)
            .context("failed to serialize settings")?;
        fs::write(Self::SETTINGS_PATH, data)
            .context("failed to write settings file")?;
        Ok(())
    }
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            input_device: None,
            output_device: None,
        }
    }
}



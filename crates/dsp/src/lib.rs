//! Core DSP abstractions and utilities for the Voclo real‑time voice morpher.
//!
//! The crate focuses on providing lock-free friendly data structures,
//! trait-based effect interfaces, and a configurable processing graph
//! that can be shared by both the audio engine and the higher-level GUI.

use std::{borrow::Cow, sync::Arc};

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Primary floating-point sample type used across the engine.
pub type Sample = f32;

/// Interleaved audio block wrapping a mutable slice of samples.
///
/// The block does not allocate and is suitable for use on the real-time thread.
pub struct ProcessBlock<'a> {
    data: &'a mut [Sample],
    channels: usize,
}

impl<'a> ProcessBlock<'a> {
    /// Creates a new process block from an interleaved buffer.
    ///
    /// # Panics
    ///
    /// Panics if `channels` is zero or if the sample count is not divisible by `channels`.
    pub fn new(data: &'a mut [Sample], channels: usize) -> Self {
        assert!(channels > 0, "channels must be non-zero");
        assert!(
            data.len() % channels == 0,
            "buffer length {} must be divisible by channels {}",
            data.len(),
            channels
        );
        Self { data, channels }
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    pub fn frames(&self) -> usize {
        self.data.len() / self.channels
    }

    #[inline]
    pub fn data(&self) -> &[Sample] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [Sample] {
        self.data
    }

    /// Returns the mutable samples for a particular channel.
    pub fn channel_mut(&mut self, index: usize) -> ChannelMut<'_> {
        assert!(index < self.channels, "channel index out of range");
        ChannelMut {
            data: self.data,
            channels: self.channels,
            index,
        }
    }
}

/// A view over a single channel inside an interleaved buffer.
pub struct ChannelMut<'a> {
    data: &'a mut [Sample],
    channels: usize,
    index: usize,
}

impl<'a> ChannelMut<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len() / self.channels
    }

    #[inline]
    pub fn iter_mut(&mut self) -> ChannelIterMut<'_> {
        ChannelIterMut {
            data: self.data,
            channels: self.channels,
            index: self.index,
            frame: 0,
        }
    }
}

pub struct ChannelIterMut<'a> {
    data: &'a mut [Sample],
    channels: usize,
    index: usize,
    frame: usize,
}

impl<'a> Iterator for ChannelIterMut<'a> {
    type Item = &'a mut Sample;

    fn next(&mut self) -> Option<Self::Item> {
        let frame_index = self.frame;
        if frame_index * self.channels + self.index >= self.data.len() {
            return None;
        }

        let ptr = self
            .data
            .as_mut_ptr()
            .wrapping_add(frame_index * self.channels + self.index);
        self.frame += 1;
        // SAFETY: we ensure unique mutable access by advancing the frame counter.
        Some(unsafe { &mut *ptr })
    }
}

/// Processing configuration for a block.
#[derive(Clone, Debug)]
pub struct ProcessContext {
    pub sample_rate: u32,
    pub channels: usize,
    pub frame_count: usize,
    pub time_since_start: f64,
}

/// Metadata describing a parameter exposed by an effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub range: ParameterRange,
    pub default: f32,
    pub unit: ParameterUnit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRange {
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterUnit {
    Decibels,
    Hertz,
    Ratio,
    Seconds,
    Milliseconds,
    Percent,
    Custom(Cow<'static, str>),
    None,
}

impl Default for ParameterUnit {
    fn default() -> Self {
        Self::None
    }
}

/// Lightweight value update used to avoid heap allocations on the audio thread.
#[derive(Debug, Clone, Copy)]
pub struct ParameterValue {
    pub id: &'static str,
    pub value: f32,
}

/// Shared metadata for an effect implementation.
#[derive(Debug, Clone)]
pub struct EffectMetadata {
    pub id: &'static str,
    pub name: &'static str,
    pub parameters: Arc<[ParameterSpec]>,
}

impl EffectMetadata {
    pub fn new(id: &'static str, name: &'static str, parameters: &[ParameterSpec]) -> Self {
        Self {
            id,
            name,
            parameters: Arc::from(parameters),
        }
    }
}

/// Trait implemented by every DSP effect in the engine.
pub trait Effect: Send {
    /// Returns effect metadata used by the host for introspection.
    fn metadata(&self) -> &EffectMetadata;

    /// Processes an interleaved buffer in-place.
    fn process(&mut self, block: &mut ProcessBlock<'_>, context: &ProcessContext);

    /// Applies a control parameter update.
    fn update_parameter(&mut self, update: ParameterValue);

    /// Enables or disables the effect in the processing chain.
    fn set_enabled(&mut self, enabled: bool);

    /// Returns whether the effect is currently active.
    fn is_enabled(&self) -> bool;
}

/// Trait for factories that create effect instances with runtime configuration.
pub trait EffectFactory: Send + Sync {
    fn metadata(&self) -> &EffectMetadata;
    fn create(&self, sample_rate: u32, channels: usize) -> anyhow::Result<Box<dyn Effect>>;
}

/// A sequential processing chain for real-time audio.
pub struct EffectChain {
    effects: Vec<Box<dyn Effect>>,
}

impl EffectChain {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    pub fn add_effect(&mut self, effect: Box<dyn Effect>) {
        self.effects.push(effect);
    }

    pub fn clear(&mut self) {
        self.effects.clear();
    }

    #[instrument(skip_all, level = "trace")]
    pub fn process(&mut self, block: &mut ProcessBlock<'_>, ctx: &ProcessContext) {
        for effect in self.effects.iter_mut() {
            if effect.is_enabled() {
                effect.process(block, ctx);
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn Effect>> {
        self.effects.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Effect>> {
        self.effects.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Gain {
        metadata: EffectMetadata,
        gain: Sample,
        enabled: bool,
    }

    impl Gain {
        fn new() -> Self {
            Self {
                metadata: EffectMetadata::new(
                    "gain",
                    "Unity Gain",
                    &[ParameterSpec {
                        id: "gain",
                        name: "Gain",
                        range: ParameterRange {
                            min: 0.0,
                            max: 2.0,
                            step: 0.01,
                        },
                        default: 1.0,
                        unit: ParameterUnit::Ratio,
                    }],
                ),
                gain: 1.0,
                enabled: true,
            }
        }
    }

    impl Effect for Gain {
        fn metadata(&self) -> &EffectMetadata {
            &self.metadata
        }

        fn process(&mut self, block: &mut ProcessBlock<'_>, _context: &ProcessContext) {
            for sample in block.data_mut() {
                *sample *= self.gain;
            }
        }

        fn update_parameter(&mut self, update: ParameterValue) {
            if update.id == "gain" {
                self.gain = update.value;
            }
        }

        fn set_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }
    }

    #[test]
    fn process_chain_applies_gain() {
        let mut chain = EffectChain::new();
        chain.add_effect(Box::new(Gain::new()));

        let mut samples = [1.0, -1.0, 0.5, -0.5];
        let mut block = ProcessBlock::new(&mut samples, 2);
        let ctx = ProcessContext {
            sample_rate: 48_000,
            channels: 2,
            frame_count: block.frames(),
            time_since_start: 0.0,
        };

        chain.process(&mut block, &ctx);
        assert_eq!(block.data(), &[1.0, -1.0, 0.5, -0.5]);

        for effect in chain.iter_mut() {
            effect.update_parameter(ParameterValue {
                id: "gain",
                value: 0.5,
            });
        }

        chain.process(&mut block, &ctx);
        assert_eq!(block.data(), &[0.5, -0.5, 0.25, -0.25]);
    }
}

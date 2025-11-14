//! Built-in effect registry and placeholder implementations.
//!
//! Each effect exposes metadata and constructors that integrate with the
//! trait-based DSP pipeline defined in `voclo-dsp`. The current
//! implementation focuses on scaffolding — the processing routines are
//! placeholders that will be replaced with real DSP algorithms in later
//! phases.

use std::{
    collections::{HashMap, VecDeque},
    f32::consts::{LN_10, PI},
    sync::Arc,
};

use anyhow::{anyhow, Result};
use num_complex::Complex32;
use once_cell::sync::Lazy;
use rustfft::{Fft, FftPlanner};
use tracing::warn;
use voclo_dsp::{
    Effect, EffectFactory, EffectMetadata, ParameterRange, ParameterSpec, ParameterUnit,
    ParameterValue, ProcessBlock, ProcessContext, Sample,
};

const TWO_PI: f32 = 2.0 * PI;
const VOCODER_FFT_SIZE: usize = 1024;
const VOCODER_OVERSAMPLE: usize = 4;
const MIN_PITCH_RATIO: f32 = 0.25;
const MAX_PITCH_RATIO: f32 = 4.0;
const SPECTRAL_FFT_SIZE: usize = 1024;
const SPECTRAL_HOP: usize = 256;

/// Logical identifiers for built-in effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectKind {
    Bypass,
    Gain,
    PitchShift,
    FormantShift,
    Robotizer,
    Demon,
    Reverb,
    Distortion,
    Filter,
}

impl EffectKind {
    pub fn id(self) -> &'static str {
        match self {
            EffectKind::Bypass => "bypass",
            EffectKind::Gain => "gain",
            EffectKind::PitchShift => "pitch_shift",
            EffectKind::FormantShift => "formant_shift",
            EffectKind::Robotizer => "robotizer",
            EffectKind::Demon => "demon",
            EffectKind::Reverb => "reverb",
            EffectKind::Distortion => "distortion",
            EffectKind::Filter => "filter",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            EffectKind::Bypass => "Bypass",
            EffectKind::Gain => "Output Gain",
            EffectKind::PitchShift => "Pitch Shift",
            EffectKind::FormantShift => "Formant Shift",
            EffectKind::Robotizer => "Robot Voice",
            EffectKind::Demon => "Spectral Inversion",
            EffectKind::Reverb => "Reverb",
            EffectKind::Distortion => "Distortion",
            EffectKind::Filter => "Low-pass Filter",
        }
    }
}

/// Global registry of available effect factories.
pub struct EffectRegistry {
    factories: HashMap<&'static str, Box<dyn EffectFactory>>,
}

impl EffectRegistry {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    pub fn with_builtin() -> Self {
        let mut registry = Self::new();
        registry
            .register(Box::new(GainFactory::new()))
            .expect("duplicate effect registration");
        registry
            .register(Box::new(PitchShiftFactory::new()))
            .expect("duplicate effect registration");
        registry
            .register(Box::new(FormantShiftFactory::new()))
            .expect("duplicate effect registration");
        registry
            .register(Box::new(RobotizerFactory::new()))
            .expect("duplicate effect registration");
        registry
            .register(Box::new(SpectralInversionFactory::new()))
            .expect("duplicate effect registration");
        registry
            .register(Box::new(LowPassFilterFactory::new()))
            .expect("duplicate effect registration");
        registry
            .register(Box::new(DistortionFactory::new()))
            .expect("duplicate effect registration");
        registry
            .register(Box::new(ReverbFactory::new()))
            .expect("duplicate effect registration");
        for kind in [EffectKind::Bypass] {
            registry
                .register(Box::new(PlaceholderFactory::new(kind)))
                .expect("duplicate effect registration");
        }
        registry
    }

    pub fn register(&mut self, factory: Box<dyn EffectFactory>) -> Result<()> {
        let id = factory.metadata().id;
        if self.factories.contains_key(id) {
            return Err(anyhow!("effect `{id}` already registered"));
        }
        self.factories.insert(id, factory);
        Ok(())
    }

    pub fn metadata(&self) -> Vec<Arc<EffectMetadata>> {
        let mut entries: Vec<_> = self
            .factories
            .values()
            .map(|factory| Arc::new(factory.metadata().clone()))
            .collect();
        entries.sort_by(|a, b| a.name.cmp(b.name));
        entries
    }

    pub fn metadata_by_id(&self, id: &str) -> Option<Arc<EffectMetadata>> {
        self.factories
            .get(id)
            .map(|factory| Arc::new(factory.metadata().clone()))
    }

    pub fn create(&self, id: &str, sample_rate: u32, channels: usize) -> Result<Box<dyn Effect>> {
        let factory = self
            .factories
            .get(id)
            .ok_or_else(|| anyhow!("effect `{id}` not registered"))?;
        factory.create(sample_rate, channels)
    }
}

impl Default for EffectRegistry {
    fn default() -> Self {
        Self::with_builtin()
    }
}

struct GainFactory {
    metadata: Arc<EffectMetadata>,
}

impl GainFactory {
    fn new() -> Self {
        Self {
            metadata: Arc::clone(&GAIN_METADATA),
        }
    }
}

impl EffectFactory for GainFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, _sample_rate: u32, _channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(GainEffect::new(Arc::clone(&self.metadata))))
    }
}

struct GainEffect {
    metadata: Arc<EffectMetadata>,
    gain_db: f32,
    gain_linear: f32,
    mix: f32,
    enabled: bool,
}

impl GainEffect {
    fn new(metadata: Arc<EffectMetadata>) -> Self {
        let mut effect = Self {
            metadata,
            gain_db: 0.0,
            gain_linear: 1.0,
            mix: 1.0,
            enabled: true,
        };
        effect.recompute_gain();
        effect
    }

    fn set_gain_db(&mut self, value: f32) {
        self.gain_db = value;
        self.recompute_gain();
    }

    fn set_mix(&mut self, value: f32) {
        self.mix = value.clamp(0.0, 1.0);
    }

    fn recompute_gain(&mut self) {
        self.gain_linear = 10.0f32.powf(self.gain_db / 20.0);
    }
}

impl Effect for GainEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, _context: &ProcessContext) {
        if !self.enabled {
            return;
        }

        let mix = self.mix;
        let dry_mix = 1.0 - mix;
        let gain = self.gain_linear;

        for sample in block.data_mut() {
            let input = *sample;
            let wet = input * gain;
            *sample = input * dry_mix + wet * mix;
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "gain_db" => self.set_gain_db(update.value),
            "mix" => self.set_mix(update.value),
            _ => warn!(
                "parameter `{}` not found for effect `{}`",
                update.id, self.metadata.id
            ),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

struct LowPassFilterFactory {
    metadata: Arc<EffectMetadata>,
}

impl LowPassFilterFactory {
    fn new() -> Self {
        Self {
            metadata: Arc::clone(&LOWPASS_METADATA),
        }
    }
}

impl EffectFactory for LowPassFilterFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, sample_rate: u32, channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(LowPassFilterEffect::new(
            Arc::clone(&self.metadata),
            sample_rate,
            channels,
        )))
    }
}

struct LowPassFilterEffect {
    metadata: Arc<EffectMetadata>,
    cutoff_hz: f32,
    resonance: f32,
    sample_rate: f32,
    alpha: f32,
    states: Vec<Sample>,
    enabled: bool,
}

impl LowPassFilterEffect {
    fn new(metadata: Arc<EffectMetadata>, sample_rate: u32, channels: usize) -> Self {
        let mut effect = Self {
            metadata,
            cutoff_hz: 1_000.0,
            resonance: 0.707,
            sample_rate: sample_rate.max(1) as f32,
            alpha: 0.0,
            states: vec![0.0; channels],
            enabled: true,
        };
        effect.recompute_alpha();
        effect
    }

    fn set_cutoff(&mut self, value: f32) {
        self.cutoff_hz = value;
        self.recompute_alpha();
    }

    fn set_resonance(&mut self, value: f32) {
        self.resonance = value;
        // Resonance is reserved for future enhancements.
    }

    fn recompute_alpha(&mut self) {
        let sr = self.sample_rate.max(1.0);
        let cutoff = self.cutoff_hz.clamp(10.0, sr * 0.45);
        let dt = 1.0 / sr;
        let rc = 1.0 / (2.0 * PI * cutoff);
        self.alpha = dt / (rc + dt);
    }
}

impl Effect for LowPassFilterEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, ctx: &ProcessContext) {
        if !self.enabled {
            return;
        }

        let channels = ctx.channels;
        if channels == 0 {
            return;
        }

        if self.states.len() != channels {
            // Avoid reallocating on the audio thread by logging and returning early.
            warn!(
                "channel mismatch for filter effect (expected {}, got {}); skipping processing",
                self.states.len(),
                channels
            );
            return;
        }

        let alpha = self.alpha;
        let data = block.data_mut();
        for frame in 0..ctx.frame_count {
            for channel in 0..channels {
                let idx = frame * channels + channel;
                let input = data[idx];
                let state = &mut self.states[channel];
                *state += alpha * (input - *state);
                data[idx] = *state;
            }
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "cutoff" => self.set_cutoff(update.value),
            "resonance" => self.set_resonance(update.value),
            _ => warn!(
                "parameter `{}` not found for effect `{}`",
                update.id, self.metadata.id
            ),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

struct PitchShiftFactory {
    metadata: Arc<EffectMetadata>,
}

impl PitchShiftFactory {
    fn new() -> Self {
        Self {
            metadata: Arc::clone(&PITCH_SHIFT_METADATA),
        }
    }
}

impl EffectFactory for PitchShiftFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, sample_rate: u32, channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(PhaseVocoderEffect::new(
            Arc::clone(&self.metadata),
            sample_rate,
            channels,
        )))
    }
}

struct SpectralInversionFactory {
    metadata: Arc<EffectMetadata>,
}

impl SpectralInversionFactory {
    fn new() -> Self {
        Self {
            metadata: Arc::clone(&DEMON_METADATA),
        }
    }
}

impl EffectFactory for SpectralInversionFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, sample_rate: u32, channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(SpectralInversionEffect::new(
            Arc::clone(&self.metadata),
            sample_rate,
            channels,
        )))
    }
}

struct SpectralInversionEffect {
    metadata: Arc<EffectMetadata>,
    enabled: bool,
    mix: f32,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    channels: Vec<SpectralInversionChannel>,
}

impl SpectralInversionEffect {
    fn new(metadata: Arc<EffectMetadata>, sample_rate: u32, channels: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let _ = sample_rate;
        let fft = planner.plan_fft_forward(SPECTRAL_FFT_SIZE);
        let ifft = planner.plan_fft_inverse(SPECTRAL_FFT_SIZE);

        let mut effect = Self {
            metadata,
            enabled: true,
            mix: 1.0,
            fft_forward: Arc::clone(&fft),
            fft_inverse: Arc::clone(&ifft),
            channels: Vec::new(),
        };
        effect.rebuild_channels(channels);
        effect
    }

    fn set_mix(&mut self, value: f32) {
        self.mix = value.clamp(0.0, 1.0);
    }

    fn rebuild_channels(&mut self, channels: usize) {
        self.channels = (0..channels)
            .map(|_| SpectralInversionChannel::new(
                Arc::clone(&self.fft_forward),
                Arc::clone(&self.fft_inverse),
            ))
            .collect();
    }
}

impl Effect for SpectralInversionEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, context: &ProcessContext) {
        if !self.enabled {
            return;
        }

        let channels = context.channels;
        if channels == 0 {
            return;
        }
        if self.channels.len() != channels {
            self.rebuild_channels(channels);
        }

        let frames = context.frame_count;
        let mix = self.mix;
        let dry_mix = 1.0 - mix;
        let data = block.data_mut();
        let sample_rate = context.sample_rate as f32;

        for ch in 0..channels {
            let state = &mut self.channels[ch];
            state.ensure_capacity(frames);
            {
                let input = state.input_buffer_mut(frames);
                for frame in 0..frames {
                    input[frame] = data[frame * channels + ch];
                }
            }

            state.process_samples(frames, sample_rate);

            let output = state.output_buffer(frames);
            for frame in 0..frames {
                let idx = frame * channels + ch;
                let dry = data[idx];
                let wet = output[frame];
                data[idx] = dry * dry_mix + wet * mix;
            }
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "mix" => self.set_mix(update.value),
            _ => warn!(
                "parameter `{}` not found for effect `{}`",
                update.id, self.metadata.id
            ),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

struct SpectralInversionChannel {
    fft_size: usize,
    hop: usize,
    latency: usize,
    rover: usize,
    window: Vec<f32>,
    in_fifo: Vec<f32>,
    out_fifo: Vec<f32>,
    output_accum: Vec<f32>,
    fft_buffer: Vec<Complex32>,
    temp_input: Vec<f32>,
    temp_output: Vec<f32>,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
}

impl SpectralInversionChannel {
    fn new(fft_forward: Arc<dyn Fft<f32>>, fft_inverse: Arc<dyn Fft<f32>>) -> Self {
        let fft_size = SPECTRAL_FFT_SIZE;
        let hop = SPECTRAL_HOP.min(fft_size / 2).max(1);
        let latency = fft_size - hop;
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                let phase = TWO_PI * i as f32 / fft_size as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        Self {
            fft_size,
            hop,
            latency,
            rover: latency,
            window,
            in_fifo: vec![0.0; fft_size],
            out_fifo: vec![0.0; fft_size],
            output_accum: vec![0.0; fft_size],
            fft_buffer: vec![Complex32::new(0.0, 0.0); fft_size],
            temp_input: Vec::with_capacity(fft_size),
            temp_output: Vec::with_capacity(fft_size),
            fft_forward,
            fft_inverse,
        }
    }

    fn ensure_capacity(&mut self, frames: usize) {
        if self.temp_input.len() < frames {
            self.temp_input.resize(frames, 0.0);
        }
        if self.temp_output.len() < frames {
            self.temp_output.resize(frames, 0.0);
        }
    }

    fn input_buffer_mut(&mut self, frames: usize) -> &mut [f32] {
        &mut self.temp_input[..frames]
    }

    fn output_buffer(&self, frames: usize) -> &[f32] {
        &self.temp_output[..frames]
    }

    fn process_samples(&mut self, frames: usize, _sample_rate: f32) {
        self.temp_output[..frames].fill(0.0);

        for i in 0..frames {
            let sample = self.temp_input[i];
            self.in_fifo[self.rover] = sample;
            let out_idx = self.rover - self.latency;
            if out_idx < self.out_fifo.len() {
                self.temp_output[i] = self.out_fifo[out_idx];
                self.out_fifo[out_idx] = 0.0;
            }
            self.rover += 1;

            if self.rover >= self.fft_size {
                self.process_frame();
                self.rover = self.latency;
            }
        }
    }

    fn process_frame(&mut self) {
        let half = self.fft_size / 2;

        for k in 0..self.fft_size {
            let value = self.in_fifo[k] * self.window[k];
            self.fft_buffer[k] = Complex32::new(value, 0.0);
        }
        self.fft_forward.process(&mut self.fft_buffer);

        for k in 1..half {
            self.fft_buffer.swap(k, self.fft_size - k);
        }

        self.fft_inverse.process(&mut self.fft_buffer);

        let scale = 1.0 / (self.fft_size as f32);
        for k in 0..self.fft_size {
            let value = self.fft_buffer[k].re * scale;
            self.output_accum[k] += value * self.window[k];
        }

        for k in 0..self.hop {
            self.out_fifo[k] = self.output_accum[k];
        }

        self.output_accum
            .copy_within(self.hop..self.fft_size, 0);
        self.output_accum[self.fft_size - self.hop..].fill(0.0);

        self.in_fifo.copy_within(self.hop..self.fft_size, 0);
        self.in_fifo[self.fft_size - self.hop..].fill(0.0);
    }
}

struct PhaseVocoderEffect {
    metadata: Arc<EffectMetadata>,
    pitch_ratio: f32,
    mix: f32,
    enabled: bool,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    channels: Vec<PhaseVocoderChannel>,
}

impl PhaseVocoderEffect {
    fn new(metadata: Arc<EffectMetadata>, sample_rate: u32, channels: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let _ = sample_rate;
        let fft = planner.plan_fft_forward(VOCODER_FFT_SIZE);
        let ifft = planner.plan_fft_inverse(VOCODER_FFT_SIZE);

        let mut effect = Self {
            metadata,
            pitch_ratio: 1.0,
            mix: 1.0,
            enabled: true,
            fft_forward: Arc::clone(&fft),
            fft_inverse: Arc::clone(&ifft),
            channels: Vec::new(),
        };
        effect.rebuild_channels(channels);
        effect
    }

    fn set_semitones(&mut self, semitones: f32) {
        let ratio = 2.0f32.powf(semitones / 12.0);
        self.pitch_ratio = ratio.clamp(MIN_PITCH_RATIO, MAX_PITCH_RATIO);
    }

    fn set_mix(&mut self, value: f32) {
        self.mix = value.clamp(0.0, 1.0);
    }

    fn rebuild_channels(&mut self, channels: usize) {
        self.channels = (0..channels)
            .map(|_| PhaseVocoderChannel::new(
                Arc::clone(&self.fft_forward),
                Arc::clone(&self.fft_inverse),
            ))
            .collect();
    }
}

impl Effect for PhaseVocoderEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, context: &ProcessContext) {
        if !self.enabled {
            return;
        }

        let channels = context.channels;
        if channels == 0 {
            return;
        }

        if self.channels.len() != channels {
            self.rebuild_channels(channels);
        }

        let frames = context.frame_count;
        let pitch_ratio = self.pitch_ratio;
        let mix = self.mix;
        let dry_mix = 1.0 - mix;
        let data = block.data_mut();

        for channel in 0..channels {
            let state = &mut self.channels[channel];
            state.ensure_capacity(frames);
            {
                let input = state.input_buffer_mut(frames);
                for frame in 0..frames {
                    input[frame] = data[frame * channels + channel];
                }
            }

            state.process_samples(frames, pitch_ratio, context.sample_rate as f32);

            let output = state.output_buffer(frames);
            for frame in 0..frames {
                let idx = frame * channels + channel;
                let dry = data[idx];
                let wet = output[frame];
                data[idx] = dry * dry_mix + wet * mix;
            }
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "semitones" => self.set_semitones(update.value),
            "mix" => self.set_mix(update.value),
            _ => warn!(
                "parameter `{}` not found for effect `{}`",
                update.id, self.metadata.id
            ),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

struct FormantShiftFactory {
    metadata: Arc<EffectMetadata>,
}

impl FormantShiftFactory {
    fn new() -> Self {
        Self {
            metadata: Arc::clone(&FORMANT_METADATA),
        }
    }
}

impl EffectFactory for FormantShiftFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, _sample_rate: u32, channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(FormantShiftEffect::new(
            Arc::clone(&self.metadata),
            channels,
        )))
    }
}

struct FormantShiftEffect {
    metadata: Arc<EffectMetadata>,
    enabled: bool,
    mix: f32,
    shift: f32,
    coeff: f32,
    channels: Vec<FormantChannelState>,
}

impl FormantShiftEffect {
    fn new(metadata: Arc<EffectMetadata>, channels: usize) -> Self {
        let mut effect = Self {
            metadata,
            enabled: true,
            mix: 1.0,
            shift: 1.0,
            coeff: 0.0,
            channels: Vec::new(),
        };
        effect.rebuild_channels(channels);
        effect.set_shift(1.0);
        effect
    }

    fn set_shift(&mut self, value: f32) {
        self.shift = value.clamp(0.5, 2.0);
        self.coeff = ((self.shift - 1.0) / (self.shift + 1.0)).clamp(-0.95, 0.95);
        for channel in &mut self.channels {
            channel.set_coeff(self.coeff);
        }
    }

    fn set_mix(&mut self, value: f32) {
        self.mix = value.clamp(0.0, 1.0);
    }

    fn rebuild_channels(&mut self, channels: usize) {
        if self.channels.len() > channels {
            self.channels.truncate(channels);
        } else {
            while self.channels.len() < channels {
                let mut state = FormantChannelState::new();
                state.set_coeff(self.coeff);
                self.channels.push(state);
            }
        }
    }
}

impl Effect for FormantShiftEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, context: &ProcessContext) {
        if !self.enabled {
            return;
        }

        let channels = context.channels;
        if channels == 0 {
            return;
        }
        if self.channels.len() != channels {
            self.rebuild_channels(channels);
            for channel in &mut self.channels {
                channel.set_coeff(self.coeff);
            }
        }

        let dry_mix = 1.0 - self.mix;
        let mix = self.mix;
        let data = block.data_mut();
        let frames = context.frame_count;

        for ch in 0..channels {
            let state = &mut self.channels[ch];
            for frame in 0..frames {
                let idx = frame * channels + ch;
                let dry = data[idx];
                let mut sample = dry;
                sample = state.process(sample);
                data[idx] = dry * dry_mix + sample * mix;
            }
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "shift" => self.set_shift(update.value),
            "mix" => self.set_mix(update.value),
            _ => warn!(
                "parameter `{}` not found for effect `{}`",
                update.id, self.metadata.id
            ),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

struct FormantChannelState {
    sections: [AllPassSection; 4],
}

impl FormantChannelState {
    fn new() -> Self {
        Self {
            sections: [
                AllPassSection::default(),
                AllPassSection::default(),
                AllPassSection::default(),
                AllPassSection::default(),
            ],
        }
    }

    fn set_coeff(&mut self, coeff: f32) {
        for section in &mut self.sections {
            section.set_coeff(coeff);
        }
    }

    fn process(&mut self, mut sample: f32) -> f32 {
        for section in &mut self.sections {
            sample = section.process(sample);
        }
        sample
    }
}

#[derive(Clone, Copy)]
struct AllPassSection {
    coeff: f32,
    prev_input: f32,
    prev_output: f32,
}

impl Default for AllPassSection {
    fn default() -> Self {
        Self {
            coeff: 0.0,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }
}

impl AllPassSection {
    fn set_coeff(&mut self, coeff: f32) {
        self.coeff = coeff;
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = -self.coeff * input + self.prev_input + self.coeff * self.prev_output;
        self.prev_input = input;
        self.prev_output = output;
        output
    }
}

struct RobotizerFactory {
    metadata: Arc<EffectMetadata>,
}

impl RobotizerFactory {
    fn new() -> Self {
        Self {
            metadata: Arc::clone(&ROBOT_METADATA),
        }
    }
}

impl EffectFactory for RobotizerFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, sample_rate: u32, channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(RobotizerEffect::new(
            Arc::clone(&self.metadata),
            sample_rate,
            channels,
        )))
    }
}

struct RobotizerEffect {
    metadata: Arc<EffectMetadata>,
    enabled: bool,
    mix: f32,
    window_ms: f32,
    sample_rate: f32,
    window_samples: usize,
    channels: Vec<RobotChannelState>,
}

impl RobotizerEffect {
    fn new(metadata: Arc<EffectMetadata>, sample_rate: u32, channels: usize) -> Self {
        let mut effect = Self {
            metadata,
            enabled: true,
            mix: 1.0,
            window_ms: 40.0,
            sample_rate: sample_rate.max(1) as f32,
            window_samples: 1,
            channels: Vec::new(),
        };
        effect.rebuild_channels(channels);
        effect.set_window_ms(40.0);
        effect
    }

    fn set_window_ms(&mut self, value: f32) {
        self.window_ms = value.clamp(5.0, 200.0);
        let samples = ((self.window_ms / 1_000.0) * self.sample_rate).round().max(1.0) as usize;
        self.window_samples = samples.max(1);
        for channel in &mut self.channels {
            channel.set_window(self.window_samples);
        }
    }

    fn set_mix(&mut self, value: f32) {
        self.mix = value.clamp(0.0, 1.0);
    }

    fn rebuild_channels(&mut self, channels: usize) {
        if self.channels.len() > channels {
            self.channels.truncate(channels);
        } else {
            while self.channels.len() < channels {
                let mut state = RobotChannelState::new(self.window_samples.max(1));
                state.set_window(self.window_samples.max(1));
                self.channels.push(state);
            }
        }
    }
}

impl Effect for RobotizerEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, context: &ProcessContext) {
        if !self.enabled {
            return;
        }

        let channels = context.channels;
        if channels == 0 {
            return;
        }
        if self.channels.len() != channels {
            self.rebuild_channels(channels);
            for state in &mut self.channels {
                state.set_window(self.window_samples.max(1));
            }
        }

        let dry_mix = 1.0 - self.mix;
        let mix = self.mix;
        let frames = context.frame_count;
        let data = block.data_mut();

        for ch in 0..channels {
            let state = &mut self.channels[ch];
            for frame in 0..frames {
                let idx = frame * channels + ch;
                let dry = data[idx];
                let robotic = state.process(dry);
                data[idx] = dry * dry_mix + robotic * mix;
            }
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "window" => self.set_window_ms(update.value),
            "mix" => self.set_mix(update.value),
            _ => warn!(
                "parameter `{}` not found for effect `{}`",
                update.id, self.metadata.id
            ),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

struct RobotChannelState {
    history: VecDeque<f32>,
    sum: f32,
    max_len: usize,
}

impl RobotChannelState {
    fn new(window_samples: usize) -> Self {
        let max_len = window_samples.max(1);
        Self {
            history: VecDeque::with_capacity(max_len),
            sum: 0.0,
            max_len,
        }
    }

    fn set_window(&mut self, window_samples: usize) {
        self.max_len = window_samples.max(1);
        while self.history.len() > self.max_len {
            if let Some(old) = self.history.pop_front() {
                self.sum -= old;
            }
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let abs = input.abs();
        self.history.push_back(abs);
        self.sum += abs;
        if self.history.len() > self.max_len {
            if let Some(old) = self.history.pop_front() {
                self.sum -= old;
            }
        }
        let len = self.history.len() as f32;
        let envelope = if len > 0.0 { self.sum / len } else { 0.0 };
        let sign = if input >= 0.0 { 1.0 } else { -1.0 };
        sign * envelope
    }
}

struct PhaseVocoderChannel {
    fft_size: usize,
    oversample: usize,
    step: usize,
    rover: usize,
    in_fifo: Vec<f32>,
    out_fifo: Vec<f32>,
    output_accum: Vec<f32>,
    fft_buffer: Vec<Complex32>,
    window: Vec<f32>,
    last_phase: Vec<f32>,
    sum_phase: Vec<f32>,
    ana_magn: Vec<f32>,
    ana_freq: Vec<f32>,
    syn_magn: Vec<f32>,
    syn_freq: Vec<f32>,
    syn_weight: Vec<f32>,
    temp_input: Vec<f32>,
    temp_output: Vec<f32>,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
}

impl PhaseVocoderChannel {
    fn new(fft_forward: Arc<dyn Fft<f32>>, fft_inverse: Arc<dyn Fft<f32>>) -> Self {
        let fft_size = VOCODER_FFT_SIZE;
        let oversample = VOCODER_OVERSAMPLE;
        let step = fft_size / oversample;
        let half = fft_size / 2;

        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                let phase = TWO_PI * i as f32 / (fft_size as f32);
                0.5 * (1.0 - (phase).cos())
            })
            .collect();

        Self {
            fft_size,
            oversample,
            step,
            rover: 0,
            in_fifo: vec![0.0; fft_size],
            out_fifo: vec![0.0; fft_size],
            output_accum: vec![0.0; fft_size],
            fft_buffer: vec![Complex32::new(0.0, 0.0); fft_size],
            window,
            last_phase: vec![0.0; half + 1],
            sum_phase: vec![0.0; half + 1],
            ana_magn: vec![0.0; half + 1],
            ana_freq: vec![0.0; half + 1],
            syn_magn: vec![0.0; half + 1],
            syn_freq: vec![0.0; half + 1],
            syn_weight: vec![0.0; half + 1],
            temp_input: Vec::with_capacity(fft_size),
            temp_output: Vec::with_capacity(fft_size),
            fft_forward,
            fft_inverse,
        }
    }

    fn ensure_capacity(&mut self, frames: usize) {
        if self.temp_input.len() < frames {
            self.temp_input.resize(frames, 0.0);
        }
        if self.temp_output.len() < frames {
            self.temp_output.resize(frames, 0.0);
        }
    }

    fn input_buffer_mut(&mut self, frames: usize) -> &mut [f32] {
        &mut self.temp_input[..frames]
    }

    fn output_buffer(&self, frames: usize) -> &[f32] {
        &self.temp_output[..frames]
    }

    fn process_samples(&mut self, frames: usize, pitch_ratio: f32, sample_rate: f32) {
        let pitch_ratio = pitch_ratio.clamp(MIN_PITCH_RATIO, MAX_PITCH_RATIO);
        let fft_size = self.fft_size;
        let step = self.step;
        let in_fifo_latency = fft_size - step;
        let freq_per_bin = sample_rate / fft_size as f32;
        let expct = TWO_PI * step as f32 / fft_size as f32;
        let oversample = self.oversample as f32;

        if self.rover == 0 {
            self.rover = in_fifo_latency;
        }

        self.temp_output[..frames].fill(0.0);

        for i in 0..frames {
            let sample = self.temp_input[i];
            self.in_fifo[self.rover] = sample;
            let out_idx = self.rover - in_fifo_latency;
            let sample_out = if out_idx < self.out_fifo.len() {
                let value = self.out_fifo[out_idx];
                self.out_fifo[out_idx] = 0.0;
                value
            } else {
                0.0
            };
            self.temp_output[i] = sample_out;
            self.rover += 1;

            if self.rover >= fft_size {
                self.process_frame(pitch_ratio, freq_per_bin, expct, oversample);
                self.rover = in_fifo_latency;
            }
        }
    }

    fn process_frame(&mut self, pitch_ratio: f32, freq_per_bin: f32, expct: f32, oversample: f32) {
        let fft_size = self.fft_size;
        let half = fft_size / 2;
        let step = self.step;

        for k in 0..fft_size {
            self.fft_buffer[k] = Complex32::new(self.in_fifo[k] * self.window[k], 0.0);
        }
        self.fft_forward.process(&mut self.fft_buffer);

        for k in 0..=half {
            let bin = self.fft_buffer[k];
            let magn = 2.0 * (bin.re * bin.re + bin.im * bin.im).sqrt();
            let phase = bin.im.atan2(bin.re);

            let mut delta_phase = phase - self.last_phase[k];
            self.last_phase[k] = phase;

            delta_phase -= (k as f32) * expct;
            let mut qpd = (delta_phase / PI).round() as i32;
            if qpd >= 0 {
                qpd += qpd & 1;
            } else {
                qpd -= qpd & 1;
            }
            delta_phase -= PI * qpd as f32;
            delta_phase = oversample * delta_phase / TWO_PI;
            delta_phase += k as f32;

            self.ana_magn[k] = magn;
            self.ana_freq[k] = delta_phase * freq_per_bin;
        }

        self.syn_magn.fill(0.0);
        self.syn_freq.fill(0.0);
        self.syn_weight.fill(0.0);

        for k in 0..=half {
            let index = ((k as f32) * pitch_ratio).round() as usize;
            if index <= half {
                self.syn_magn[index] += self.ana_magn[k];
                self.syn_freq[index] += self.ana_freq[k] * pitch_ratio;
                self.syn_weight[index] += 1.0;
            }
        }

        for k in 0..=half {
            if self.syn_weight[k] > 0.0 {
                self.syn_freq[k] /= self.syn_weight[k];
            } else {
                self.syn_freq[k] = (k as f32) * freq_per_bin;
            }
        }

        for k in 0..=half {
            let magn = self.syn_magn[k];
            let freq = self.syn_freq[k];
            let mut delta = freq - (k as f32) * freq_per_bin;
            delta /= freq_per_bin;
            delta = TWO_PI * delta / oversample;
            delta += (k as f32) * expct;
            self.sum_phase[k] += delta;
            let phase = self.sum_phase[k];
            let re = magn * phase.cos();
            let im = magn * phase.sin();

            if k == 0 || k == half {
                self.fft_buffer[k] = Complex32::new(re, 0.0);
            } else {
                self.fft_buffer[k] = Complex32::new(re, im);
                self.fft_buffer[fft_size - k] = Complex32::new(re, -im);
            }
        }

        for k in half + 1..fft_size {
            self.fft_buffer[k] = Complex32::new(0.0, 0.0);
        }

        self.fft_inverse.process(&mut self.fft_buffer);

        let scale = 1.0 / (fft_size as f32);
        for k in 0..fft_size {
            let value = self.fft_buffer[k].re * scale;
            self.output_accum[k] += value * self.window[k] * (2.0 / (half as f32 * oversample));
        }

        for k in 0..step {
            self.out_fifo[k] = self.output_accum[k];
        }

        self.output_accum.copy_within(step..fft_size, 0);
        self.output_accum[fft_size - step..fft_size].fill(0.0);

        self.in_fifo.copy_within(step..fft_size, 0);
        self.in_fifo[fft_size - step..fft_size].fill(0.0);
    }
}

struct DistortionFactory {
    metadata: Arc<EffectMetadata>,
}

impl DistortionFactory {
    fn new() -> Self {
        Self {
            metadata: Arc::clone(&DISTORTION_METADATA),
        }
    }
}

impl EffectFactory for DistortionFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, _sample_rate: u32, _channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(DistortionEffect::new(Arc::clone(&self.metadata))))
    }
}

struct DistortionEffect {
    metadata: Arc<EffectMetadata>,
    drive: f32,
    drive_gain: f32,
    post_gain: f32,
    mix: f32,
    enabled: bool,
}

impl DistortionEffect {
    fn new(metadata: Arc<EffectMetadata>) -> Self {
        let mut effect = Self {
            metadata,
            drive: 0.5,
            drive_gain: 1.0,
            post_gain: 1.0,
            mix: 0.75,
            enabled: true,
        };
        effect.update_drive_coeffs();
        effect
    }

    fn set_drive(&mut self, value: f32) {
        self.drive = value.clamp(0.0, 1.0);
        self.update_drive_coeffs();
    }

    fn set_mix(&mut self, value: f32) {
        self.mix = value.clamp(0.0, 1.0);
    }

    fn update_drive_coeffs(&mut self) {
        // Map [0,1] → [1, 21] pre-gain range.
        self.drive_gain = 1.0 + self.drive * 20.0;
        // Keep perceived loudness roughly stable.
        let compensation = (self.drive_gain.tanh()).max(1e-3);
        self.post_gain = 1.0 / compensation;
    }
}

impl Effect for DistortionEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, _context: &ProcessContext) {
        if !self.enabled {
            return;
        }

        let dry_mix = 1.0 - self.mix;
        let drive_gain = self.drive_gain;
        let post_gain = self.post_gain;
        let mix = self.mix;

        for sample in block.data_mut() {
            let input = *sample;
            let shaped = (input * drive_gain).tanh() * post_gain;
            *sample = input * dry_mix + shaped * mix;
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "drive" => self.set_drive(update.value),
            "mix" => self.set_mix(update.value),
            _ => warn!(
                "parameter `{}` not found for effect `{}`",
                update.id, self.metadata.id
            ),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

struct ReverbFactory {
    metadata: Arc<EffectMetadata>,
}

impl ReverbFactory {
    fn new() -> Self {
        Self {
            metadata: Arc::clone(&REVERB_METADATA),
        }
    }
}

impl EffectFactory for ReverbFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, sample_rate: u32, channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(ReverbEffect::new(
            Arc::clone(&self.metadata),
            sample_rate,
            channels,
        )))
    }
}

struct ReverbEffect {
    metadata: Arc<EffectMetadata>,
    decay: f32,
    mix: f32,
    states: Vec<ChannelReverbState>,
    enabled: bool,
}

impl ReverbEffect {
    fn new(metadata: Arc<EffectMetadata>, sample_rate: u32, channels: usize) -> Self {
        let sample_rate = sample_rate.max(1) as f32;
        let decay = 1.2;
        let mix = 0.35;
        Self {
            metadata,
            decay,
            mix,
            states: (0..channels)
                .map(|_| ChannelReverbState::new(sample_rate, decay))
                .collect(),
            enabled: true,
        }
    }

    fn set_decay(&mut self, value: f32) {
        self.decay = value.max(0.1);
        for state in &mut self.states {
            state.set_decay(self.decay);
        }
    }

    fn set_mix(&mut self, value: f32) {
        self.mix = value.clamp(0.0, 1.0);
    }
}

impl Effect for ReverbEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, block: &mut ProcessBlock<'_>, ctx: &ProcessContext) {
        if !self.enabled {
            return;
        }

        let channels = ctx.channels;
        if channels == 0 {
            return;
        }

        if self.states.len() != channels {
            warn!(
                "channel mismatch for reverb effect (expected {}, got {}); skipping processing",
                self.states.len(),
                channels
            );
            return;
        }

        let dry_mix = 1.0 - self.mix;
        let mix = self.mix;
        let frame_count = ctx.frame_count;
        let data = block.data_mut();

        for frame in 0..frame_count {
            for channel in 0..channels {
                let idx = frame * channels + channel;
                let input = data[idx];
                let state = &mut self.states[channel];
                let wet = state.process_sample(input);
                data[idx] = input * dry_mix + wet * mix;
            }
        }
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        match update.id {
            "decay" => self.set_decay(update.value),
            "mix" => self.set_mix(update.value),
            _ => warn!(
                "parameter `{}` not found for effect `{}`",
                update.id, self.metadata.id
            ),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

const COMB_TIMES: [f32; 4] = [0.0297, 0.0371, 0.0411, 0.0437];
const ALLPASS_TIMES: [f32; 2] = [0.005, 0.0017];
const ALLPASS_FEEDBACK: f32 = 0.5;

struct ChannelReverbState {
    combs: Vec<CombFilter>,
    allpasses: Vec<AllPassFilter>,
}

impl ChannelReverbState {
    fn new(sample_rate: f32, decay: f32) -> Self {
        let combs = COMB_TIMES
            .iter()
            .map(|&delay| CombFilter::new(sample_rate, delay, decay))
            .collect();
        let allpasses = ALLPASS_TIMES
            .iter()
            .map(|&delay| AllPassFilter::new(sample_rate, delay, ALLPASS_FEEDBACK))
            .collect();
        Self { combs, allpasses }
    }

    fn set_decay(&mut self, decay: f32) {
        for comb in &mut self.combs {
            comb.set_decay(decay);
        }
    }

    fn process_sample(&mut self, input: Sample) -> Sample {
        let mut sum = 0.0;
        for comb in &mut self.combs {
            sum += comb.process(input);
        }
        let mut output = if self.combs.is_empty() {
            sum
        } else {
            sum / self.combs.len() as Sample
        };
        for allpass in &mut self.allpasses {
            output = allpass.process(output);
        }
        output
    }
}

struct CombFilter {
    buffer: Vec<Sample>,
    index: usize,
    feedback: f32,
    delay_seconds: f32,
    sample_rate: f32,
}

impl CombFilter {
    fn new(sample_rate: f32, delay_seconds: f32, decay: f32) -> Self {
        let len = (delay_seconds * sample_rate).round() as usize;
        let len = len.max(1);
        let mut filter = Self {
            buffer: vec![0.0; len],
            index: 0,
            feedback: 0.0,
            delay_seconds,
            sample_rate,
        };
        filter.set_decay(decay);
        filter
    }

    fn set_decay(&mut self, decay: f32) {
        let decay = decay.max(0.1);
        let delay = self.delay_seconds.max(1.0 / self.sample_rate);
        // Derive the feedback coefficient so that the signal reaches -60 dB at the desired decay time.
        let exponent = -3.0 * delay / decay;
        self.feedback = (exponent * LN_10).exp();
    }

    fn process(&mut self, input: Sample) -> Sample {
        let output = self.buffer[self.index];
        self.buffer[self.index] = input + output * self.feedback;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        output
    }
}

struct AllPassFilter {
    buffer: Vec<Sample>,
    index: usize,
    feedback: f32,
}

impl AllPassFilter {
    fn new(sample_rate: f32, delay_seconds: f32, feedback: f32) -> Self {
        let len = (delay_seconds * sample_rate).round() as usize;
        let len = len.max(1);
        Self {
            buffer: vec![0.0; len],
            index: 0,
            feedback,
        }
    }

    fn process(&mut self, input: Sample) -> Sample {
        let buf_out = self.buffer[self.index];
        let output = -input + buf_out;
        self.buffer[self.index] = input + buf_out * self.feedback;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        output
    }
}

struct PlaceholderFactory {
    metadata: Arc<EffectMetadata>,
}

impl PlaceholderFactory {
    fn new(kind: EffectKind) -> Self {
        Self {
            metadata: Arc::new(placeholder_metadata(kind)),
        }
    }
}

impl EffectFactory for PlaceholderFactory {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn create(&self, _sample_rate: u32, _channels: usize) -> Result<Box<dyn Effect>> {
        Ok(Box::new(PlaceholderEffect::new(Arc::clone(&self.metadata))))
    }
}

struct PlaceholderEffect {
    metadata: Arc<EffectMetadata>,
    parameters: HashMap<&'static str, f32>,
    enabled: bool,
}

impl PlaceholderEffect {
    fn new(metadata: Arc<EffectMetadata>) -> Self {
        let parameters = metadata
            .parameters
            .iter()
            .map(|spec| (spec.id, spec.default))
            .collect();
        Self {
            metadata,
            parameters,
            enabled: true,
        }
    }
}

impl Effect for PlaceholderEffect {
    fn metadata(&self) -> &EffectMetadata {
        self.metadata.as_ref()
    }

    fn process(&mut self, _block: &mut ProcessBlock<'_>, _context: &ProcessContext) {
        if !self.enabled {
            return;
        }
        // No-op placeholder. Real DSP will be implemented in future phases.
    }

    fn update_parameter(&mut self, update: ParameterValue) {
        if let Some(value) = self.parameters.get_mut(update.id) {
            *value = update.value;
        } else {
            warn!("parameter `{}` not found for effect `{}`", update.id, self.metadata.id);
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// --- Metadata helpers -------------------------------------------------------------------------

static GAIN_PARAMS: Lazy<Vec<ParameterSpec>> = Lazy::new(|| {
    vec![
        ParameterSpec {
            id: "gain_db",
            name: "Gain (dB)",
            range: ParameterRange {
                min: -24.0,
                max: 24.0,
                step: 0.1,
            },
            default: 0.0,
            unit: ParameterUnit::Decibels,
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
    ]
});

static GAIN_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        EffectKind::Gain.id(),
        EffectKind::Gain.name(),
        GAIN_PARAMS.as_slice(),
    ))
});

static PITCH_SHIFT_PARAMS: Lazy<Vec<ParameterSpec>> = Lazy::new(|| {
    vec![
        ParameterSpec {
            id: "semitones",
            name: "Semitone Shift",
            range: ParameterRange {
                min: -12.0,
                max: 12.0,
                step: 0.01,
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
    ]
});

static PITCH_SHIFT_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        EffectKind::PitchShift.id(),
        EffectKind::PitchShift.name(),
        PITCH_SHIFT_PARAMS.as_slice(),
    ))
});

static FORMANT_PARAMS: Lazy<Vec<ParameterSpec>> = Lazy::new(|| {
    vec![
        ParameterSpec {
            id: "shift",
            name: "Formant Shift",
            range: ParameterRange {
                min: 0.5,
                max: 2.0,
                step: 0.01,
            },
            default: 1.0,
            unit: ParameterUnit::Ratio,
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
    ]
});

static FORMANT_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        EffectKind::FormantShift.id(),
        EffectKind::FormantShift.name(),
        FORMANT_PARAMS.as_slice(),
    ))
});

static ROBOT_PARAMS: Lazy<Vec<ParameterSpec>> = Lazy::new(|| {
    vec![
        ParameterSpec {
            id: "window",
            name: "Analysis Window (ms)",
            range: ParameterRange {
                min: 10.0,
                max: 80.0,
                step: 1.0,
            },
            default: 40.0,
            unit: ParameterUnit::Milliseconds,
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
    ]
});

static ROBOT_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        EffectKind::Robotizer.id(),
        EffectKind::Robotizer.name(),
        ROBOT_PARAMS.as_slice(),
    ))
});

static DEMON_PARAMS: Lazy<Vec<ParameterSpec>> = Lazy::new(|| {
    vec![ParameterSpec {
        id: "mix",
        name: "Dry/Wet",
        range: ParameterRange {
            min: 0.0,
            max: 1.0,
            step: 0.01,
        },
        default: 1.0,
        unit: ParameterUnit::Percent,
    }]
});

static DEMON_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        EffectKind::Demon.id(),
        EffectKind::Demon.name(),
        DEMON_PARAMS.as_slice(),
    ))
});

static REVERB_PARAMS: Lazy<Vec<ParameterSpec>> = Lazy::new(|| {
    vec![
        ParameterSpec {
            id: "decay",
            name: "Decay (s)",
            range: ParameterRange {
                min: 0.1,
                max: 10.0,
                step: 0.01,
            },
            default: 1.2,
            unit: ParameterUnit::Seconds,
        },
        ParameterSpec {
            id: "mix",
            name: "Dry/Wet",
            range: ParameterRange {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default: 0.35,
            unit: ParameterUnit::Percent,
        },
    ]
});

static REVERB_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        EffectKind::Reverb.id(),
        EffectKind::Reverb.name(),
        REVERB_PARAMS.as_slice(),
    ))
});

static DISTORTION_PARAMS: Lazy<Vec<ParameterSpec>> = Lazy::new(|| {
    vec![
        ParameterSpec {
            id: "drive",
            name: "Drive",
            range: ParameterRange {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default: 0.5,
            unit: ParameterUnit::Ratio,
        },
        ParameterSpec {
            id: "mix",
            name: "Dry/Wet",
            range: ParameterRange {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default: 0.75,
            unit: ParameterUnit::Percent,
        },
    ]
});

static DISTORTION_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        EffectKind::Distortion.id(),
        EffectKind::Distortion.name(),
        DISTORTION_PARAMS.as_slice(),
    ))
});

static FILTER_PARAMS: Lazy<Vec<ParameterSpec>> = Lazy::new(|| {
    vec![
        ParameterSpec {
            id: "cutoff",
            name: "Cutoff (Hz)",
            range: ParameterRange {
                min: 20.0,
                max: 20_000.0,
                step: 1.0,
            },
            default: 1_000.0,
            unit: ParameterUnit::Hertz,
        },
        ParameterSpec {
            id: "resonance",
            name: "Resonance (Q)",
            range: ParameterRange {
                min: 0.1,
                max: 10.0,
                step: 0.01,
            },
            default: 0.707,
            unit: ParameterUnit::None,
        },
    ]
});

static LOWPASS_METADATA: Lazy<Arc<EffectMetadata>> = Lazy::new(|| {
    Arc::new(EffectMetadata::new(
        EffectKind::Filter.id(),
        EffectKind::Filter.name(),
        FILTER_PARAMS.as_slice(),
    ))
});

fn clone_arc_metadata(meta: &Lazy<Arc<EffectMetadata>>) -> EffectMetadata {
    Arc::as_ref(&*meta).clone()
}

fn placeholder_metadata(kind: EffectKind) -> EffectMetadata {
    match kind {
        EffectKind::Bypass => EffectMetadata::new(EffectKind::Bypass.id(), EffectKind::Bypass.name(), &[]),
        EffectKind::PitchShift => unreachable!("pitch shift has a concrete implementation"),
        EffectKind::FormantShift => clone_arc_metadata(&FORMANT_METADATA),
        EffectKind::Robotizer => EffectMetadata::new(
            EffectKind::Robotizer.id(),
            EffectKind::Robotizer.name(),
            ROBOT_PARAMS.as_slice(),
        ),
        EffectKind::Demon => clone_arc_metadata(&DEMON_METADATA),
        EffectKind::Reverb => clone_arc_metadata(&REVERB_METADATA),
        EffectKind::Distortion => clone_arc_metadata(&DISTORTION_METADATA),
        EffectKind::Filter => clone_arc_metadata(&LOWPASS_METADATA),
        EffectKind::Gain => clone_arc_metadata(&GAIN_METADATA),
    }
}

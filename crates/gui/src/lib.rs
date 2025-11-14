//! GUI frontend built with `egui`/`eframe`.
//!
//! The UI exposes basic controls for the effect chain and provides placeholders
//! for forthcoming visualisations and preset management.

use std::{collections::VecDeque, fs, path::PathBuf, sync::Arc};

use anyhow::{anyhow, Context, Result};
use eframe::{egui, App};
use egui::{Color32, ComboBox, Context as EguiContext, Slider, SliderClamping, TopBottomPanel};
use egui_plot::{Line, Plot, PlotPoints};
use rustfft::{num_complex::Complex32, FftPlanner};
use serde::{Deserialize, Serialize};
use tracing::{error, warn};
use voclo_audio::{ProfilingMetrics, ProfilingSnapshot, SharedPipeline, VisualizationReceiver};
use voclo_dsp::{EffectChain, EffectMetadata, ParameterRange, ParameterUnit, ParameterValue};
use voclo_effects::{EffectKind, EffectRegistry};

const DEFAULT_PRESET_PATH: &str = "presets/default.json";
const PRESET_VERSION: u32 = 1;
const WAVEFORM_CAPACITY: usize = 4096;
const FFT_SIZE: usize = 1024;

pub struct GuiApp {
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
}

impl GuiApp {
    pub fn new(
        pipeline: SharedPipeline,
        sample_rate: u32,
        channels: usize,
        visualization: VisualizationReceiver,
        metrics: Arc<ProfilingMetrics>,
    ) -> Self {
        let preset_path = PathBuf::from(DEFAULT_PRESET_PATH);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        Self {
            pipeline,
            registry: EffectRegistry::with_builtin(),
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
        let effect = self
            .registry
            .create(kind.id(), self.sample_rate, self.channels)?;
        let metadata = self
            .registry
            .metadata_by_id(kind.id())
            .ok_or_else(|| anyhow!("metadata for effect `{}` not found", kind.id()))?;
        let slot = EffectSlot::from_metadata(metadata);

        let mut effect = Some(effect);
        self.pipeline.with_mut(move |chain| {
            if let Some(effect) = effect.take() {
                chain.add_effect(effect);
            }
        });
        self.chain_model.push(slot);
        Ok(())
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
            });
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
        });

        egui::SidePanel::left("effect_library")
            .resizable(true)
            .default_width(240.0)
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
                                }
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
struct Preset {
    version: u32,
    effects: Vec<PresetEffect>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PresetEffect {
    id: String,
    enabled: bool,
    parameters: Vec<PresetParameter>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PresetParameter {
    id: String,
    value: f32,
}

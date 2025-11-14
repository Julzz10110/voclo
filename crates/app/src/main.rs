use std::sync::Arc;

use anyhow::Context;
use tracing_subscriber::{fmt, EnvFilter};
use voclo_audio::{AudioEngine, RealtimePipeline};
use voclo_gui::GuiApp;

fn main() -> anyhow::Result<()> {
    init_tracing();

    let (pipeline_inner, viz_receiver) = RealtimePipeline::new();
    let pipeline = Arc::new(pipeline_inner);
    let audio_engine = AudioEngine::new(Arc::clone(&pipeline));
    let stream = audio_engine
        .start()
        .context("failed to start audio engine")?;

    let metrics = audio_engine.metrics();
    let gui_app = GuiApp::new(
        audio_engine.pipeline(),
        stream.sample_rate,
        stream.channels as usize,
        viz_receiver,
        audio_engine.recorder(),
        metrics,
    );

    gui_app
        .run()
        .context("failed to run GUI application")?;

    audio_engine.stop().context("failed to stop audio engine")?;
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = fmt().with_env_filter(filter).finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
}

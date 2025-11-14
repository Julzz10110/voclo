# Voclo Voice Morpher

Voclo is a cross‑platform, low‑latency voice morphing playground written in Rust.  
It targets near real‑time processing (<20 ms end‑to‑end) while keeping CPU usage low and exposing a modular DSP pipeline that can be extended with new effects and GUI controls.

## Features

- **Realtime audio I/O** via [`cpal`](https://github.com/RustAudio/cpal) (WASAPI/CoreAudio/ALSA/ASIO).  
- **Effect pipeline** implemented with trait‑based DSP modules (`voclo-dsp` + `voclo-effects`):
  - Pitch shifting (phase vocoder), formant shifting, robotizer, spectral inversion, filters, distortion, reverb, etc.
  - Dry/wet control and bypass per effect.
- **Hot parameter updates**: sliders in the GUI update the DSP chain without reallocations.
- **Presets**: save/load chains and parameters to JSON.
- **Visualization**: waveform + spectrum display, live latency/CPU read‑outs.
- **Modular architecture**:
  - `voclo-audio` – device management, input/output streams, profiling hooks.
  - `voclo-dsp` – processing traits, effect metadata, chaining.
  - `voclo-effects` – built‑in effect implementations (pitch/formant shift, robot, demon, etc.).
  - `voclo-gui` – egui/eframe‑based UI with plots and preset manager.

## Getting Started

### Prerequisites
- Rust toolchain (1.77+ recommended).
- Audio devices/drivers (microphone + output). On Windows you can use WASAPI loopback or ASIO; on macOS/Linux CoreAudio/ALSA works out of the box.

### Build & Run

```bash
cargo run -p voclo-app
```

When the GUI opens:
1. Ensure the proper system devices are selected (defaults are used for now).
2. Add effects from the left panel and tweak parameters.
3. Use *Save preset / Load preset* in the top bar to persist chains.

### Project Layout

```
Cargo.toml          # workspace definition
crates/
  app/              # entrypoint (sets up tracing, audio engine, GUI)
  audio/            # cpal integration, ring buffers, profiling
  dsp/              # shared DSP abstractions
  effects/          # effect implementations + registry
  gui/              # egui/eframe frontend with visualization & presets
presets/
  default.json      # example preset
```

## Roadmap

- Advanced “character” presets (anime voices, monsters, etc.).
- Additional composite effects (PSOLA, convolution reverb presets, modulation banks).
- Device picker and multi‑device routing.
- Long‑duration stability tests & benchmarking scripts.

## Contributing

1. Fork and clone the repository.
2. Create a branch, implement your feature, and run `cargo fmt && cargo clippy && cargo test`.
3. Submit a PR describing the change and how to reproduce/test it.

## License

This project is released under the MIT License. See `LICENSE` for details.


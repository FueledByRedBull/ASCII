# ASCII Engine Project Guide

Status: canonical project/development document. This file merges the former specification, roadmap, implementation, remediation, presets, and completion-plan notes.

## Product Goal

ASCII Engine is a deterministic, non-ML C++20 renderer that converts local images and video into terminal ASCII/ANSI output, text frame exports, encoded media, still images, and deterministic replay files.

The v1 baseline is intentionally conservative:

- no-OpenCV by default
- AVX2 disabled by default
- deterministic core behavior
- local image and video inputs
- terminal playback and file export
- replay write, inspect, playback, and text export

OpenCV, AVX2, webcam, audio polish, GPU compute paths, and richer packaging are optional or future-facing work unless explicitly promoted.

## Current Status

Verified locally on Windows/MSVC:

- `build_noopencv_check.cmd` configures and builds the no-OpenCV baseline.
- CTest passes the default no-OpenCV suite.
- Tests link the reusable `ascii-engine-core` target.
- Replay write/read/inspect/play/text-export paths are wired.
- CLI smoke tests cover image text export, still export, replay write/inspect, help, and unsupported output failure.
- Terminal renderer tests cover glyph, foreground, and block-art background diffs.
- Replay determinism is tested with byte-for-byte replay output comparison.

Not yet fully proven:

- OpenCV-enabled builds, because OpenCV is not installed in the current local dependency set.
- Hosted CI passing on GitHub runners, because workflow execution must happen remotely.

## Repository Layout

```text
src/
  core/        config, frame sources, pipeline, replay, temporal, motion, composition
  glyph/       font loading, glyph cache, character sets
  mapping/     glyph selection and color mapping
  render/      terminal, block, bitmap, video rendering
  terminal/    terminal capability and ANSI output
  audio/       audio decode/playback
  cli/         CLI parsing

tests/         unit, replay, renderer, CLI smoke, baseline notes
vendor/        vendored headers
images/        local sample media
```

## Build Policy

Defaults:

- `ASCII_USE_OPENCV=OFF`
- `ASCII_ENABLE_AVX2=OFF`
- `BUILD_TESTS=ON`

Windows recommended path:

```bat
setup_windows_deps.cmd
build_noopencv_check.cmd
```

Generic no-OpenCV build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF -DASCII_ENABLE_AVX2=OFF
cmake --build build --target ascii-engine -j
ctest --test-dir build --output-on-failure
```

Optional builds:

```bash
cmake -S . -B build-opencv -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=ON
cmake -S . -B build-avx2 -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF -DASCII_ENABLE_AVX2=ON
```

## Scope Tiers

### v1 Required

- Local video input.
- Local image input.
- Terminal playback.
- Text-frame export.
- Edge-aware glyph selection with temporal stability.
- Color modes: none, ANSI 16, ANSI 256, truecolor.
- Deterministic replay write, inspect, playback, and text export.
- Windows, Linux, and macOS build support.
- Basic performance instrumentation.
- CI build/test matrix.

### v1.x Quality

- Encoded video output.
- Still image output.
- Block-art mode.
- Image sequence input.
- Better profiling reports.
- Terminal compatibility matrix.

### v2 Advanced

- Webcam input polish.
- Audio support beyond best-effort behavior.
- Raw pipe input polish.
- Motion-compensated refinements beyond v1 stability needs.
- Plugin-style extension points.
- Additional glyph packs.
- GPU/compute-shader edge downscaling and stylized edge preprocessing.

## Runtime Behavior

### Output Matrix

| Source | Target | Behavior |
|---|---|---|
| image | none | render once to terminal |
| image | `.txt` | write one text file |
| image | video (`.mp4`, `.gif`, etc.) | encode a one-frame video/animation |
| image | still (`.jpg`, `.jpeg`, `.bmp`) | write one rendered still image |
| video | none | live terminal playback |
| video | `.txt` | write numbered text frames |
| video | video (`.mp4`, `.gif`, etc.) | encode all rendered frames |
| video | still (`.jpg`, `.jpeg`, `.bmp`) | write the first rendered frame only |
| any | unsupported extension | fail before processing with a clear error |

### Replay Commands

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4" --replay run.areplay
.\build-noopencv\ascii-engine.exe --inspect-replay run.areplay
.\build-noopencv\ascii-engine.exe --play-replay run.areplay
.\build-noopencv\ascii-engine.exe --play-replay run.areplay -o replay.txt
```

### Content Presets

Use `--profile` or `profile = "..."` in config.

- `natural`: balanced motion, temporal stability, and halftone detail for real-world video.
- `anime`: preserves line art and flat regions with stronger temporal flicker suppression.
- `ui`: prioritizes crisp text/shapes and disables halftone noise by default.

Profile values are applied before explicit CLI overrides.

## Algorithm Contract

The core renderer is classical image processing, not ML.

Per-frame pipeline intent:

1. Decode and normalize input.
2. Convert sRGB to linear-light.
3. Resize/crop using `fit`, `fill`, or `stretch`.
4. Build luminance and color buffers.
5. Apply blur and edge detection.
6. Compute multi-scale gradients, orientation, and adaptive edge data.
7. Run the deterministic CPU contour pass: Difference-of-Gaussians, Sobel, non-maximum suppression, adaptive thresholding, and per-cell tangent histograms.
8. Aggregate per-cell statistics.
9. Select glyphs using brightness, orientation, contrast, frequency, and texture signals.
10. Override non-`blockart` glyphs with active ASCII contour glyphs (`-`, `|`, `/`, `\`, `+`) while preserving normal color mapping.
11. Apply temporal smoothing, edge hysteresis, and transition costs.
12. Map color.
13. Render to terminal, text, bitmap, video, or replay.

Per-cell statistics should include:

- mean luminance
- luminance variance/stddev
- edge strength
- edge occupancy
- contour activation, glyph, and strength
- orientation histogram
- structure coherence
- mean color

Glyph modeling should include:

- brightness/density
- contrast
- orientation histogram
- edge suitability
- deterministic ordering

## Determinism Contract

- Same input bytes, config, and build profile should produce the same cell decisions.
- Core mapping must not depend on random behavior.
- Replay stores frame index, selected glyph/color cells, grid metadata, FPS, and config hash.
- Platform terminal color fallback differences must be documented when they affect visible output.

## Performance And Instrumentation

Runtime summaries are emitted on exit:

- `[PERF]`
- `[PERF_STAGES]`
- `[PERF_STAGES_PCT]`

Reference baseline:

- CPU: Intel Core i5-1240P / Ryzen 5 5600U class or better.
- RAM: 16 GB.
- Build: Release, no-OpenCV, OpenMP when available, no AVX2 requirement.
- Workload: local 1080p video input, `120x40`, truecolor, audio disabled.
- Target: at least 24 FPS processing throughput, startup under 2 seconds, resident memory under 512 MB.

`--strict-memory` enforces a conservative 512 MiB estimated-memory budget.

## Testing

Default no-OpenCV CTest coverage includes:

- algorithm/config/CLI unit coverage
- critical regression tests
- replay round-trip and determinism tests
- terminal renderer diff tests
- CLI smoke tests
- unsupported output validation

Recommended validation:

```powershell
cmd /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"" -arch=x64 -host_arch=x64 && cmake --build build-noopencv -j 4 && ctest --test-dir build-noopencv --output-on-failure"
```

Baseline notes live in `tests/baseline/README.md`.

## Release Checklist

- [x] Default no-OpenCV build configures locally.
- [x] Default no-OpenCV build compiles `ascii-engine`.
- [x] Default no-OpenCV tests build and pass locally.
- [x] Windows no-OpenCV script configures and builds locally.
- [x] AVX2 is opt-in.
- [x] Tests link `ascii-engine-core`.
- [x] Output mode matrix is implemented and documented.
- [x] Replay write/read/inspect/play behavior works.
- [x] Replay determinism test passes.
- [x] Terminal rendering tests cover glyph/fg/bg diffs.
- [x] CLI smoke tests pass locally.
- [x] Known limitations are documented.
- [ ] OpenCV build configures when OpenCV is installed.
- [ ] Hosted CI passes on supported platforms.

## Known Limitations

- On Windows, `cmake` and `ctest` may require the Visual Studio developer environment.
- Verified still-image targets for the no-OpenCV MSVC/vcpkg build are `.jpg`, `.jpeg`, and `.bmp`.
- `.png` output is intentionally rejected in this baseline because the FFmpeg still-image path crashed in local smoke testing.
- Webcam support is v2/deferred for the no-OpenCV baseline.
- Audio is best-effort/deferred and not a v1 release gate.
- OpenCV-enabled builds need separate validation with OpenCV installed.
- Hosted CI must run before cross-platform claims are fully proven.

## Future GPU Edge Downscaling

A compute-shader path is technically applicable, but not required for v1.

The proposed approach:

- Use tile-sized workgroups, ideally matching glyph/cell dimensions such as `8x8`.
- Build a group-shared histogram of edge direction/class values.
- Emit the dominant edge class only when a tile crosses an edge-density threshold.
- Keep a CPU reference path as the deterministic baseline.
- Document backend, supported platforms, determinism differences, fallback behavior, and visual parity tests.

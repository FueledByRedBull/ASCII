# ASCII Engine

Status: the no-OpenCV, non-AVX2 v1 baseline is complete. It passes 34 clean-build tests locally and in hosted Windows/Linux/macOS CI, and the Windows ZIP passes a separate clean-runner smoke test. Evidence is tracked in `PROJECT.md`.

ASCII Engine is a deterministic, non-ML C++20 renderer that converts video and images into ANSI/ASCII output for terminal playback and file export.

## Highlights

- Real-time terminal rendering from video/images.
- Output modes: live terminal, `.txt`, encoded video, verified still image formats, `.areplay` replay.
- Color modes: `none`, `16`, `256`, `truecolor`, `blockart`.
- Default CPU contour overlay maps strong local edges to `-`, `|`, `/`, `\`, and `+`.
- Content presets: `natural`, `anime`, `ui`.
- Deterministic replay capture with config hash.

## Performance Optimizations

Recent performance-oriented updates include:

1. Hierarchical motion estimation (coarse-to-fine pyramid refinement)
2. SIMD hot paths when explicitly enabled; the default release baseline is scalar-portable
3. Cache-aware tiling in edge/blur kernels
4. Optimized in-tree FFT phase-correlation (plan/twiddle caching, workspace reuse, rectangular FFT)
5. Stable-frame cache reuse for pipeline and cell stats
6. Parallel independent-cell composition, motion confidence evaluation, area resampling, and contour aggregation when OpenMP is available

These changes target better throughput without changing the external CLI.

## Repository Layout

```text
src/
  core/        pipeline, frame source, edge, motion, temporal, config, replay
  glyph/       font loading, glyph cache, character sets
  mapping/     glyph selection and color mapping
  render/      terminal/block/bitmap/video rendering and dithering
  terminal/    terminal capability and ANSI output
  audio/       audio decode/playback
  cli/         CLI parsing

tests/         unit/integration style targets
vendor/        vendored headers
assets/        assets
```

## Requirements

### Windows

- Visual Studio 2022 (Build Tools or Community) with C++ desktop workload
- CMake
- Ninja
- Git

Dependencies installed by script:

- SDL2
- FFmpeg
- zstd

### Linux/macOS

- C++20 compiler
- CMake
- SDL2 + FFmpeg + zstd development packages
- Optional OpenCV 4.x (`ASCII_USE_OPENCV=ON`)

## Build

### Windows quick start (recommended)

```bat
setup_windows_deps.cmd
build_noopencv_check.cmd
```

Binary:

```text
build-noopencv\ascii-engine.exe
```

### Generic CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF
cmake --build build --target ascii-engine -j
```

With OpenCV:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=ON
cmake --build build --target ascii-engine -j
```

Enable AVX2 kernel paths (optional):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF -DASCII_ENABLE_AVX2=ON
cmake --build build --target ascii-engine -j
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

Create the verified Windows x64 release ZIP from a built tree:

```powershell
.\package_windows_release.ps1 -BuildDirectory build-noopencv -Version 1.0.0
```

## Usage

Show help:

```powershell
.\build-noopencv\ascii-engine.exe --help
```

Render video:

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4"
```

Render image:

```powershell
.\build-noopencv\ascii-engine.exe ".\images\shot.png" --profile ui --cols 120 --rows 40
```

Webcam:

```powershell
.\build-noopencv\ascii-engine.exe webcam --fps 30
```

Webcam support is v2/deferred for the no-OpenCV baseline and requires an OpenCV-capable build.

Export video:

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4" -o out.mp4
```

Export animated GIF:

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4" -o out.gif
```

Export text:

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4" -o out.txt
```

Write replay:

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4" --replay run.areplay
```

Inspect replay:

```powershell
.\build-noopencv\ascii-engine.exe --inspect-replay run.areplay
```

Play replay or export replay text:

```powershell
.\build-noopencv\ascii-engine.exe --play-replay run.areplay
.\build-noopencv\ascii-engine.exe --play-replay run.areplay -o replay.txt
```

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

Requested file outputs are finalized through temporary files and return non-zero on open, write, encode, replay, or truncated-decode failure. PNG is intentionally not an output target in the verified no-OpenCV Windows baseline.

### Common Flags

- `--profile natural|anime|ui`
- `--char-set basic|blocks|line-art`
- `--color none|16|256|truecolor|blockart`
- `--fps N --cols N --rows N`
- `--edge-thresh X --blur X --temporal X`
- `--no-contours --contour-thresh X`
- `--motion-solve-div N --motion-reuse N --motion-still-thresh X`
- `--phase-interval N --phase-scene-trigger X`
- `--scale fit|fill|stretch`
- `--font <PATH>`
- `--no-audio`
- `--debug grayscale|edges|orientation`
- `--profile-live`
- `--strict-memory`
- `--fast` (disables costly analysis features for speed-focused preview)

### Performance Output

At program exit the engine prints a summary to `stderr`:

- `[PERF]` total frames, wall time, effective FPS, processing FPS
- `[PERF_STAGES]` absolute stage times (pipeline, motion, select, render, encode, misc)
- `[PERF_STAGES_PCT]` stage percentages of processing time

The 2026-07-16 reference measurement used a deterministic 60-frame 1920x1080 video, `120x40`, truecolor, no audio, Release MSVC/OpenMP, no OpenCV, and no AVX2 on a Ryzen 7 7800X3D:

| Mode | Processing | Peak working set |
|---|---:|---:|
| Full quality | 10.86 FPS | 136.43 MiB |
| `--fast --no-contours` | 25.70 FPS | 106.70 MiB |

One-image startup/process/exit measured 0.32 seconds and 75.91 MiB. The explicit speed path meets the 24 FPS reference target; full-quality mode does not, so use `--fast --no-contours` when throughput is the priority.

### Speed Tuning Example

For higher FPS with good quality balance on terminal playback:

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4" --cols 96 --rows 32 --motion-solve-div 4 --motion-reuse 5 --phase-interval 8 --motion-still-thresh 0.006
```

### Interactive Controls

- `Space`: pause/resume
- `q` or `Esc`: quit
- `c`: cycle color mode
- `+` / `-`: edge threshold up/down

## Config

Default config path:

- Linux: `~/.config/ascii-engine/config.toml`
- macOS: `~/Library/Application Support/ascii-engine/config.toml`
- Windows: `%APPDATA%\ascii-engine\config.toml`

Precedence:

1. built-in defaults
2. config file
3. CLI overrides

Preset details and development/release notes are documented in `PROJECT.md`.

### Font fidelity

`--font` controls glyph analysis and bitmap/video rendering. A terminal still draws emitted codepoints with the terminal application's own font, which can differ from the analysis font. Use still-image or video output when validating against a known font. The `natural`, `anime`, and `ui` profiles are deterministic hand-tuned presets, not claims that one is universally higher quality.

## Troubleshooting

### SDL2 not found during configure (Windows)

```bat
setup_windows_deps.cmd
build_noopencv_check.cmd
```

### Image fails to open

- Rebuild and run the newest executable.
- Verify path and extension (`png/jpg/jpeg/bmp/gif/tiff/webp`).

### Output looks too noisy

- Reduce grid size (`--cols`, `--rows`).
- Use `--profile ui` for screenshots/text.
- Increase `--edge-thresh` and/or `--blur`.
- Use `--color truecolor` or `--color none`.

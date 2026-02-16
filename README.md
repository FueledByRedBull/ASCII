# ASCII Engine

ASCII Engine is a deterministic, non-ML C++20 renderer that converts video and images into ANSI/ASCII output for terminal playback and file export.

## Highlights

- Real-time terminal rendering from video/images.
- Output modes: live terminal, `.txt`, encoded video, `.areplay` replay.
- Color modes: `none`, `16`, `256`, `truecolor`, `blockart`.
- Content presets: `natural`, `anime`, `ui`.
- Deterministic replay capture with config hash.

## Performance Optimizations

Recent performance-oriented updates include:

1. Hierarchical motion estimation (coarse-to-fine pyramid refinement)
2. SIMD hot paths (AVX2/SSE2 where available, scalar fallback)
3. Cache-aware tiling in edge/blur kernels
4. Optimized in-tree FFT phase-correlation (plan/twiddle caching, workspace reuse, rectangular FFT)
5. Stable-frame cache reuse for pipeline and cell stats

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

### Common Flags

- `--profile natural|anime|ui`
- `--char-set basic|blocks|line-art`
- `--color none|16|256|truecolor|blockart`
- `--fps N --cols N --rows N`
- `--edge-thresh X --blur X --temporal X`
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

Presets are documented in `PRESETS.md`.

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

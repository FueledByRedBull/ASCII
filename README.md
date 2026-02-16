# ASCII Engine

A non-ML, deterministic C++20 ASCII rendering engine for terminal playback and offline export.

This project implements the scope in `PLAN.md` / `IMPLEMENTATION.md`: edge-aware glyph mapping, temporal stabilization, perceptual color mapping, and separate text/block-art rendering paths.

## Current Highlights

- Input sources:
  - video files
  - single images
  - image sequences (wildcards)
  - raw pipe input (`pipe:WxH:rgb[:fps]`)
  - webcam (OpenCV-enabled builds)
- Output modes:
  - live terminal playback
  - text frame export (`.txt`)
  - encoded video output
  - deterministic replay files (`.areplay`)
- Color modes:
  - `none`, `16`, `256`, `truecolor`, `blockart`
- Content presets:
  - `natural`, `anime`, `ui`
- Core algorithms implemented in runtime:
  - linear-light processing + perceptual color distance (OKLab)
  - adaptive edge thresholds with global/local/hybrid modes
  - multi-scale edge analysis with normalized Laplacian scale selection
  - structure tensor + orientation/coherence stats per cell
  - unified glyph selection loss (brightness/orientation/contrast/frequency/texture)
  - frequency-domain glyph matching (8x16 DCT signatures)
  - texture-aware glyph selection (multi-frequency Gabor responses)
  - motion-aware temporal smoothing (phase-correlation assisted flow blending)
  - temporal wavelet flicker suppression (multi-level Haar shrinkage)
  - anisotropic diffusion (edge-preserving smoothing)
  - bilateral-grid-based color smoothing
  - quad-tree adaptive cell detail levels
  - block-art spectral palette quantization
  - halftone/blue-noise style dithering path

## Repository Layout

```text
src/
  core/        # pipeline, frame sources, edge, temporal, motion, config, replay
  glyph/       # font loading, glyph cache, char sets
  mapping/     # glyph selector, color mapper, bilateral grid
  render/      # terminal, block-art, bitmap, video encoder, dither
  terminal/    # terminal capability and ANSI output
  audio/       # audio decode/playback
  cli/         # argument parsing and help

tests/         # unit/integration-style executable targets
assets/        # project assets
docs: PLAN.md, IMPLEMENTATION.md, ROADMAP.md, QUESTIONS.md, AUDIT.md
```

## Requirements

## Windows (recommended path in this repo)

- Visual Studio 2022 Build Tools or Community (Desktop C++ workload)
- CMake
- Ninja
- Git

Dependencies are installed via local vcpkg by script:
- SDL2
- FFmpeg
- zstd

## Linux/macOS

- C++20 compiler
- CMake
- SDL2, FFmpeg, zstd development packages
- Optional: OpenCV 4.x (if building with `ASCII_USE_OPENCV=ON`)

## Build

## Windows quick build (no OpenCV)

From repository root:

```bat
setup_windows_deps.cmd
build_noopencv_check.cmd
```

This builds `ascii-engine.exe` in `build-noopencv`.

## Generic CMake build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF
cmake --build build --target ascii-engine -j
```

If OpenCV is available:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=ON
cmake --build build --target ascii-engine -j
```

## Usage

Show CLI help:

```bash
ascii-engine --help
```

Run video:

```bash
ascii-engine input.mp4
```

Run image:

```bash
ascii-engine image.png --cols 120 --rows 40 --profile ui
```

Webcam:

```bash
ascii-engine webcam --fps 30
```

Video output:

```bash
ascii-engine input.mp4 -o out.mp4
```

Text frame export:

```bash
ascii-engine input.mp4 -o out.txt
```

Replay export:

```bash
ascii-engine input.mp4 --replay run.areplay
```

### Useful Flags

- `--profile natural|anime|ui`
- `--char-set basic|blocks|line-art`
- `--color none|16|256|truecolor|blockart`
- `--fps N`, `--cols N`, `--rows N`
- `--edge-thresh X`, `--blur X`, `--temporal X`
- `--scale fit|fill|stretch`
- `--font PATH`
- `--no-audio`
- `--debug grayscale|edges|orientation`
- `--profile-live`
- `--strict-memory`

### Interactive Controls (playback)

- `Space`: pause/resume
- `q` or `Esc`: quit
- `c`: cycle color mode
- `+` / `-`: raise/lower edge threshold

## Config

Default config path:

- Linux: `~/.config/ascii-engine/config.toml`
- macOS: `~/Library/Application Support/ascii-engine/config.toml`
- Windows: `%APPDATA%\ascii-engine\config.toml`

Effective precedence:

1. built-in defaults
2. config file
3. CLI flags

Profile behavior (`PRESETS.md`):

- profile defaults are applied before explicit CLI overrides.

## Troubleshooting

## `SDL2 not found` during CMake configure

On Windows, run:

```bat
setup_windows_deps.cmd
```

Then rebuild with:

```bat
build_noopencv_check.cmd
```

## Single image fails to open

Ensure you are running a freshly rebuilt binary after pulling code changes. The image path is detected by extension and decoded through the image pipeline.

## Output looks too busy / noisy

- reduce grid density (`--cols`, `--rows`)
- use `--profile ui` for screenshots/text-heavy content
- raise `--edge-thresh` and/or `--blur`
- use `--color truecolor` or `--color none` for clean readability

## Development Notes

- Build settings and dependency wiring: `CMakeLists.txt`
- Windows dependency bootstrap: `setup_windows_deps.cmd`
- Windows no-OpenCV build helper: `build_noopencv_check.cmd`
- Preset documentation: `PRESETS.md`
- Roadmap and implementation contracts:
  - `PLAN.md`
  - `IMPLEMENTATION.md`
  - `ROADMAP.md`
  - `QUESTIONS.md`
  - `AUDIT.md`


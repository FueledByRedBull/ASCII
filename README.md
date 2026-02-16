# ASCII Engine

A deterministic, non-ML C++20 rendering engine that converts video, images, and webcam input into high-quality ANSI/ASCII art for real-time terminal playback and file export.

---

## Features

- **Real-time rendering** — video, images, and live webcam at up to 120 FPS
- **5 color modes** — `none` (monochrome), `16` (ANSI), `256` (extended), `truecolor` (24-bit), `blockart` (Unicode half-block)
- **Edge-aware glyph selection** — multi-feature matching using brightness, orientation, contrast, frequency, and texture signatures via histogram comparison
- **Temporal stabilization** — EMA smoothing, wavelet flicker suppression, phase-correlated motion compensation to reduce frame-to-frame jitter
- **Multi-scale edge detection** — Canny hysteresis with anisotropic diffusion, adaptive thresholding (per-tile or hybrid), and dark-scene floor correction
- **Content profiles** — `natural`, `anime`, `ui` presets with tuned defaults for different source types
- **Perceptual color quantization** — OkLab-space quantization with optional blue-noise halftone dithering and bilateral grid smoothing
- **Multiple output targets** — terminal (live), `.txt` (text dump), `.mp4` (encoded video via FFmpeg), `.areplay` (deterministic replay)
- **Audio sync** — SDL2-based audio playback synced to video frames
- **Interactive controls** — pause, cycle color modes, adjust edge threshold on the fly
- **Cross-platform** — Windows, Linux, macOS

---

## Architecture

### Rendering Pipeline

```
Input (video/image/webcam)
  │
  ├─► Frame Source (FFmpeg decode / stb_image / webcam capture)
  │
  ├─► Grayscale Conversion (BT.709 luminance)
  │
  ├─► Resize & Fit (fit/fill/stretch to grid dimensions)
  │
  ├─► Edge Detection
  │     ├── Multi-scale Gaussian blur (σ₀=0.8, σ₁=1.6)
  │     ├── Sobel gradient (Gx, Gy → magnitude + orientation)
  │     ├── Canny hysteresis (low/high threshold, NMS)
  │     ├── Adaptive thresholding (per-tile / global / hybrid)
  │     └── Optional anisotropic diffusion pre-filter
  │
  ├─► Cell Statistics Aggregation (per grid cell)
  │     ├── Mean luminance, variance, local contrast
  │     ├── Gradient orientation histogram (8-bin)
  │     ├── Frequency signature (8-bin DCT-like)
  │     ├── Gabor texture signature (8-bin)
  │     ├── Edge occupancy & strength
  │     └── Structure coherence
  │
  ├─► Motion Estimation (phase correlation, capped pixel displacement)
  │
  ├─► Temporal Stabilization
  │     ├── EMA smoothing (α=0.3)
  │     ├── Transition penalty (character change cost)
  │     ├── Wavelet flicker suppression
  │     └── Phase-correlated motion blending
  │
  ├─► Character Selection
  │     ├── Glyph cache (pre-rasterized via stb_truetype)
  │     ├── Multi-feature scoring:
  │     │     brightness (0.45) + orientation (0.40) +
  │     │     contrast (0.15) + frequency (0.20) + texture (0.15)
  │     └── Best-match from active character set
  │
  ├─► Color Mapping
  │     ├── OkLab perceptual quantization
  │     ├── Floyd-Steinberg dithering with error clamping
  │     ├── Optional blue-noise halftone
  │     └── Optional bilateral grid smoothing
  │
  └─► Rendering
        ├── Terminal renderer (ANSI escape sequences)
        ├── Block renderer (Unicode half-block art)
        ├── Bitmap renderer (off-screen rasterization)
        └── Video encoder (FFmpeg libavcodec → MP4/H.264)
```

### Source Layout

```
src/
  main.cpp                    Entry point, main loop, interactive controls
  core/
    types.hpp                 Fundamental types: Color, FrameBuffer, FloatImage,
                              EdgeData, GradientData, CellStats
    config.hpp / .cpp         Config structs, TOML loading, profile application,
                              CLI override merging, config hashing & validation
    pipeline.hpp / .cpp       Frame processing pipeline orchestration
    frame_source.hpp / .cpp   FFmpeg video decode, stb_image loader, webcam input
    edge_detector.hpp / .cpp  Multi-scale Canny, adaptive thresholds, NMS,
                              anisotropic diffusion
    cell_stats.hpp / .cpp     Per-cell feature aggregation (luminance, gradients,
                              orientation histograms, frequency/texture signatures)
    motion.hpp / .cpp         Phase-correlation motion estimation
    temporal.hpp / .cpp       EMA smoothing, wavelet flicker, transition penalty
    color_space.hpp / .cpp    OkLab/sRGB/linear conversions
    replay.hpp / .cpp         Deterministic replay capture/playback (.areplay, zstd)
  glyph/
    font_loader.hpp / .cpp    TTF font loading via stb_truetype
    glyph_cache.hpp / .cpp    Pre-rasterized glyph bitmaps with feature vectors
    char_sets.hpp             Character set definitions (basic, blocks, line-art)
    glyph_stats.hpp           Per-glyph feature descriptors
  mapping/
    char_selector.hpp / .cpp  Multi-feature glyph scoring and selection
    color_mapper.hpp / .cpp   Color quantization, dithering, halftone
    bilateral_grid.hpp / .cpp Bilateral grid color smoothing
  render/
    terminal_renderer.hpp/.cpp  ANSI escape output
    block_renderer.hpp / .cpp   Unicode half-block rendering with spectral palette
    bitmap_renderer.hpp / .cpp  Off-screen pixel rasterization
    video_encoder.hpp / .cpp    FFmpeg video encoding (H.264)
    dither.hpp / .cpp           Blue-noise halftone, Floyd-Steinberg error diffusion
  audio/
    audio_player.hpp / .cpp   SDL2 audio playback
  terminal/
    terminal.hpp / .cpp       Terminal capability detection, raw mode, size query
  cli/
    args.hpp / .cpp            CLI argument parsing and validation

tests/
  test_comprehensive.cpp      Core unit tests
  test_critical_fixes.cpp     Regression tests for critical bug fixes
  test_integration.cpp        End-to-end pipeline tests

vendor/                       Vendored single-header libraries
  stb_image.h                 Image loading (PNG, JPG, BMP, GIF, TIFF, WebP)
  stb_truetype.h              TrueType font rasterization
  toml.hpp                    TOML config file parsing
```

---

## Requirements

### Windows

- **Visual Studio 2022** (Build Tools or Community) with C++ desktop workload
- **CMake** ≥ 3.16
- **Ninja** (recommended generator)
- **Git**

The setup script will install:
- **SDL2** — audio playback
- **FFmpeg** (libavcodec, libavformat, libavutil, libswscale, libswresample) — video decode/encode
- **zstd** — replay file compression

### Linux / macOS

- C++20 compiler (GCC 11+, Clang 14+, Apple Clang 15+)
- CMake ≥ 3.16
- Development packages: `libsdl2-dev`, `libavcodec-dev`, `libavformat-dev`, `libavutil-dev`, `libswscale-dev`, `libswresample-dev`, `libzstd-dev`
- **Optional:** OpenCV 4.x (`-DASCII_USE_OPENCV=ON`)
- **Optional:** OpenMP for parallel cell aggregation

---

## Building

### Windows Quick Start (recommended)

```bat
:: Install dependencies (SDL2, FFmpeg, zstd)
setup_windows_deps.cmd

:: Configure + build (Ninja, Release, no OpenCV)
build_noopencv_check.cmd
```

Output:
```
build-noopencv\ascii-engine.exe
```

### CMake (generic)

**Without OpenCV:**
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF
cmake --build build --target ascii-engine -j$(nproc)
```

**With OpenCV:**
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=ON
cmake --build build --target ascii-engine -j$(nproc)
```

**Debug build (with AddressSanitizer):**
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

### Running Tests

```bash
cmake --build build --target test
# or
cd build && ctest --output-on-failure
```

---

## Usage

### Basic Examples

```bash
# Show help
ascii-engine --help

# Render a video in the terminal
ascii-engine ./media/clip.mp4

# Render a single image
ascii-engine ./images/photo.png --profile natural --cols 160 --rows 50

# Webcam (live)
ascii-engine webcam --fps 30 --color truecolor

# Export to encoded MP4 video
ascii-engine ./media/clip.mp4 -o output.mp4

# Export to text file
ascii-engine ./media/clip.mp4 -o output.txt

# Capture deterministic replay
ascii-engine ./media/clip.mp4 --replay session.areplay
```

### Full CLI Reference

| Flag | Description | Default | Range |
|------|-------------|---------|-------|
| `-h`, `--help` | Show help and exit | — | — |
| `-o`, `--output <FILE>` | Output file path (`.mp4` for video, `.txt` for text) | terminal | — |
| `--config <FILE>` | Config file path | platform default | — |
| `--replay <FILE>` | Write `.areplay` deterministic replay | — | — |
| `-f`, `--fps <N>` | Target framerate | `30` | 1–120 |
| `-c`, `--cols <N>` | Grid columns (0 = auto-detect from terminal) | `0` | 1–500 |
| `-r`, `--rows <N>` | Grid rows (0 = auto-detect from terminal) | `0` | 1–200 |
| `--char-set <NAME>` | Character set | `basic` | `basic`, `blocks`, `line-art` |
| `--profile <NAME>` | Content profile preset | — | `natural`, `anime`, `ui` |
| `--color <MODE>` | Color mode | `truecolor` | `none`, `16`, `256`, `truecolor`, `blockart` |
| `--edge-thresh <N>` | Edge detection threshold | `0.1` | 0.0–1.0 |
| `--blur <N>` | Pre-edge blur sigma | `1.0` | 0.1–10.0 |
| `--temporal <N>` | Temporal smoothing alpha (0 = no smoothing) | `0.3` | 0.0–1.0 |
| `--scale <MODE>` | Scaling mode | `fit` | `fit`, `fill`, `stretch` |
| `--font <PATH>` | TTF font file (auto-detects system monospace font if unset) | auto | — |
| `--no-audio` | Disable audio playback | off | — |
| `--no-hysteresis` | Disable Canny edge hysteresis | off | — |
| `--no-orientation` | Disable orientation-based glyph selection | off | — |
| `--simple-orientation` | Use simple 8-direction orientation mapping | off | — |
| `--debug <MODE>` | Debug visualization | — | `grayscale`, `edges`, `orientation` |
| `--profile-live` | Output per-frame profiling as JSONL to stderr | off | — |
| `--strict-memory` | Fail if memory budget is exceeded | off | — |

### Interactive Controls (during playback)

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `q` / `Esc` | Quit |
| `c` | Cycle color mode (`none` → `16` → `256` → `truecolor` → `blockart`) |
| `+` / `=` | Increase edge threshold |
| `-` | Decrease edge threshold |

---

## Configuration

### Config File

ASCII Engine reads a TOML config file from a platform-specific default location:

| Platform | Path |
|----------|------|
| Linux | `~/.config/ascii-engine/config.toml` |
| macOS | `~/Library/Application Support/ascii-engine/config.toml` |
| Windows | `%APPDATA%\ascii-engine\config.toml` |

Override with `--config <PATH>`.

### Precedence

1. Built-in defaults
2. Config file values
3. Content profile (`--profile`) overwrites
4. CLI flag overrides (highest priority)

### Example Config

```toml
# Grid
cols = 120
rows = 40
scale_mode = "fit"

# Character selection
char_set = "basic"
profile = "natural"

# Edge detection
[edge]
low_threshold = 0.05
high_threshold = 0.1
blur_sigma = 1.0
use_hysteresis = true
multi_scale = true
adaptive_mode = "hybrid"    # "tile", "global", or "hybrid"

# Temporal stabilization
[temporal]
alpha = 0.3
transition_penalty = 0.15
use_wavelet_flicker = true
wavelet_strength = 0.45
use_phase_correlation = true

# Character selector weights
[selector]
weight_brightness = 0.45
weight_orientation = 0.40
weight_contrast = 0.15
weight_frequency = 0.20
weight_texture = 0.15
enable_frequency_matching = true
enable_gabor_texture = true

# Color
[color]
mode = "truecolor"          # none, 16, 256, truecolor, blockart
quantization = "oklab"
dither_error_clamp = 0.12
use_blue_noise_halftone = false
halftone_strength = 0.24
use_bilateral_grid = false
```

### Content Profiles

| Profile | Best For | Key Tuning |
|---------|----------|------------|
| `natural` | Real-world video, photos | Balanced motion handling, temporal stability, halftone detail |
| `anime` | Animation, cartoons | Preserves line art and flat color regions, stronger flicker suppression |
| `ui` | Screenshots, text, UI | Crisp text/shapes, disables halftone noise by default |

Profiles apply their defaults first; explicit CLI flags always override profile values.

---

## Supported Input Formats

- **Video:** Any format supported by FFmpeg (MP4, AVI, MKV, MOV, WebM, etc.)
- **Images:** PNG, JPG/JPEG, BMP, GIF, TIFF, WebP (via stb_image)
- **Live:** `webcam` keyword for webcam capture

---

## Troubleshooting

### `SDL2 not found` at configure time (Windows)

Run the dependency setup script then rebuild:
```bat
setup_windows_deps.cmd
build_noopencv_check.cmd
```

### Image fails to open

- Ensure the file path is valid (use full path or correct relative path)
- Confirm extension is supported (`png`, `jpg`, `jpeg`, `bmp`, `gif`, `tiff`, `webp`)
- Rebuild to pick up latest fixes

### Output looks too busy / noisy

- Lower grid density: `--cols 80 --rows 24`
- Use a calmer profile: `--profile ui`
- Increase edge threshold: `--edge-thresh 0.15`
- Increase blur: `--blur 1.5`
- Use `--color truecolor` or `--color none`

### Temporal flicker in video

- Increase temporal smoothing: `--temporal 0.5`
- The `anime` profile has stronger flicker suppression by default

### Characters look wrong / mismatched

- Try a different font: `--font /path/to/monospace.ttf`
- Switch character set: `--char-set blocks` or `--char-set line-art`
- Disable orientation matching: `--no-orientation` or `--simple-orientation`

---

## License

See [LICENSE](LICENSE) for details.

# ASCII Engine

Status: Windows/MSVC with OpenCV and AVX2 disabled is the supported local baseline. Current validation and the separately dated hosted-CI/release evidence are tracked in `PROJECT.md` and `tests/baseline/README.md`.

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

1. Sparse block motion with hierarchical phase-correlation refinement
2. Portable scalar paths and SSE2 on supported CPUs; AVX2 remains opt-in
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

- Windows 10 version 1903 or later (verified locally on Windows 11)
- Visual Studio 2022 (Build Tools or Community) with C++ desktop workload
- CMake
- Ninja
- Git

MSVC executables use a process-local UTF-8 manifest so non-ASCII filenames work consistently across the filesystem, C runtime, and FFmpeg. This uses [Windows UTF-8 code-page support](https://learn.microsoft.com/en-us/windows/apps/design/globalizing/use-utf8-code-page); no system locale change is needed.

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

Video, still-image and replay outputs are finalized through temporary files. Failures before finalization preserve the corresponding existing destination. Each destination is finalized separately: if another output fails later, an already finalized output remains. Numbered text frames are written incrementally; a later failure can leave earlier completed frames. Input, output and replay destinations must be distinct, including generated numbered filenames. PNG is intentionally not an output target in the verified no-OpenCV Windows baseline. Available video containers also require an installed encoder; unsupported encoder/container combinations fail explicitly.

Bitmap exports blend antialiased glyphs in linear light. Ordinary ASCII uses source-colored glyph strokes on black, so its average brightness is limited by the selected glyphs' coverage. Use `--color blockart` for foreground/background quadrant reconstruction. Terminal appearance also depends on its configured font.

### Common Flags

- `--profile natural|anime|ui`
- `--char-set basic|traditional|blocks|line-art` (`traditional` uses the compact ` .:-=+*#%@` ramp)
- `--color none|16|256|truecolor|blockart`
- `--fps N --cols N --rows N`
- `--edge-thresh X --blur X --temporal X`
- `--no-contours --contour-thresh X`
- `--motion-solve-div N --motion-reuse N --motion-still-thresh X`
- `--phase-interval N --phase-scene-trigger X`
- `--scale fit|fill|stretch`
- `--font <PATH>` (an explicitly requested font must load successfully)
- `--no-audio`
- `--debug grayscale|edges|orientation` (terminal, file and replay output; honors `--color`)
- `--profile-live`
- `--strict-memory` (rejects requests whose estimated render memory exceeds 512 MiB)
- `--fast` (disables costly analysis features for speed-focused preview)

### Performance Output

At program exit the engine prints a summary to `stderr`:

- `[PERF]` total frames, wall time, effective FPS, processing FPS
- `[PERF_STAGES]` absolute stage times (pipeline, motion, select, render, encode, misc)
- `[PERF_STAGES_PCT]` stage percentages of processing time

The 2026-10-02 measurement used 120 changing 1920x1080 frames, `120x40`, truecolor, no audio, and offline text export on a Ryzen 7 7800X3D. Release MSVC/OpenMP, OpenCV off, AVX2 off; FPS values are medians of three independent runs, and memory is the maximum measured peak:

| Mode | OpenMP threads | Processing | Including startup/exit | Peak working set |
|---|---:|---:|---:|---:|
| Full quality | 12 | 27.40 FPS | 26.60 FPS | 113.51 MiB |
| `--fast --no-contours` | 4 | 56.34 FPS | 53.44 FPS | 93.95 MiB |

Both meet the measured host's targets of 24 FPS full quality and 30 FPS fast mode. The supplied real-video segment reaches 37.11 FPS full quality. Independent one-image startup/render/exit runs took 0.10-0.11 seconds. These are processing/export measurements; terminal painting, other hardware, codecs and grid sizes can change throughput. The complete workload matrix and limitations are in `tests/baseline/README.md`.

Set the measured thread count before launching (the best count depends on the host):

```powershell
$env:OMP_NUM_THREADS = '12'  # Use '4' for the measured fast mode.
.\build-noopencv\ascii-engine.exe input.mp4 --cols 120 --rows 40 --color truecolor --no-audio -o frames.txt
```

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

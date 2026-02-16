# ASCII Engine

ASCII Engine is a deterministic, non-ML C++20 renderer that converts video/image input into ANSI/ASCII output for terminal playback and export.

## What It Does

- Converts video/images/webcam frames into ASCII in real time.
- Supports `none`, `16`, `256`, `truecolor`, and `blockart` color modes.
- Uses edge-aware glyph selection with temporal stabilization.
- Supports replay capture (`.areplay`) for deterministic inspection.
- Exports to terminal, text (`.txt`), and encoded video output.

## Project Layout

```text
src/
  core/        pipeline, config, frame sources, motion, temporal, replay
  glyph/       font loading, glyph cache, char sets
  mapping/     glyph and color mapping
  render/      terminal/block/bitmap rendering + encoder + dithering
  terminal/    terminal capabilities/output
  audio/       audio playback/sync
  cli/         CLI parsing

tests/          test targets
vendor/         vendored single-header dependencies
assets/         project assets
```

## Requirements

### Windows

- Visual Studio 2022 (Build Tools or Community) with C++ desktop tools
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
- Optional OpenCV 4.x (if enabling `ASCII_USE_OPENCV=ON`)

## Build

### Windows quick start (recommended)

From repo root:

```bat
setup_windows_deps.cmd
build_noopencv_check.cmd
```

Output binary:

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

## Usage

Show help:

```powershell
.\build-noopencv\ascii-engine.exe --help
```

Run a video:

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4"
```

Run a single image:

```powershell
.\build-noopencv\ascii-engine.exe ".\images\shot.png" --profile ui --cols 120 --rows 40
```

Use webcam:

```powershell
.\build-noopencv\ascii-engine.exe webcam --fps 30
```

Export encoded video:

```powershell
.\build-noopencv\ascii-engine.exe ".\media\clip.mp4" -o out.mp4
```

Export text output:

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
- `--scale fit|fill|stretch`
- `--font <PATH>`
- `--no-audio`
- `--debug grayscale|edges|orientation`
- `--profile-live`
- `--strict-memory`

### Interactive Controls

- `Space`: pause/resume
- `q` or `Esc`: quit
- `c`: cycle color mode
- `+` / `-`: adjust edge threshold

## Config

Default config file location:

- Linux: `~/.config/ascii-engine/config.toml`
- macOS: `~/Library/Application Support/ascii-engine/config.toml`
- Windows: `%APPDATA%\ascii-engine\config.toml`

Precedence:

1. built-in defaults
2. config file
3. CLI overrides

Presets are documented in `PRESETS.md`.

## Troubleshooting

### `SDL2 not found` at configure time (Windows)

Run:

```bat
setup_windows_deps.cmd
```

Then rebuild:

```bat
build_noopencv_check.cmd
```

### Image fails to open

- Rebuild and run the latest executable.
- Use full path or a valid relative path.
- Confirm extension is supported (`png/jpg/jpeg/bmp/gif/tiff/webp`).

### Output looks too busy

- Lower grid density (`--cols`, `--rows`).
- Use `--profile ui` for screenshots and UI content.
- Increase `--edge-thresh` and/or `--blur`.
- Use `--color truecolor` or `--color none` for cleaner output.

## Notes

- Main build config: `CMakeLists.txt`
- Windows scripts: `setup_windows_deps.cmd`, `build_noopencv_check.cmd`
- Presets: `PRESETS.md`

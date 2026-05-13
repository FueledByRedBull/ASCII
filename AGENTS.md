# AGENTS.md - ASCII Engine Local Instructions

> Scope: repository-local instructions for `C:/Users/ancha/Documents/Projects/ASCII`.
> This file overrides global defaults only where it is more specific to this project.

## Project Snapshot

ASCII Engine is a deterministic, non-ML C++20 renderer for converting local images/video into terminal ASCII/ANSI output, text frames, encoded media, still images, and `.areplay` replay files.

The canonical project/development document is `PROJECT.md`.
The user-facing quickstart is `README.md`.
Do not recreate separate roadmap/spec/remediation/preset docs unless explicitly asked; consolidate project guidance into `PROJECT.md` and keep quickstart material in `README.md`.

## Current Baseline

The default supported baseline is:

- Windows/MSVC no-OpenCV build.
- `ASCII_USE_OPENCV=OFF`.
- `ASCII_ENABLE_AVX2=OFF`.
- `BUILD_TESTS=ON`.
- Project-local dependencies installed through `.tools/vcpkg`.

OpenCV, AVX2, webcam polish, audio polish, GPU compute paths, and richer packaging are optional/future work unless the user explicitly promotes them.

## Repository Layout

```text
src/core/      config, frame sources, pipeline, replay, temporal, motion, composition
src/glyph/     font loading, glyph cache, character sets
src/mapping/   glyph selection and color mapping
src/render/    terminal, block, bitmap, video rendering
src/terminal/  terminal capability and ANSI output
src/audio/     audio decode/playback
src/cli/       CLI parsing
tests/         unit, replay, renderer, CLI smoke, baseline notes
vendor/        vendored headers
images/        local sample media
```

## Build And Test Commands

### Windows Recommended

Use the project scripts for the normal Windows path:

```bat
setup_windows_deps.cmd
build_noopencv_check.cmd
```

`setup_windows_deps.cmd` may take several minutes because it bootstraps/installs vcpkg packages.

### MSVC CTest Validation

Use this after code changes that affect behavior, tests, CMake, CLI, replay, rendering, or docs that claim test status:

```powershell
cmd /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"" -arch=x64 -host_arch=x64 && cmake --build build-noopencv -j 4 && ctest --test-dir build-noopencv --output-on-failure"
```

Plain PowerShell may not have `cmake` or `ctest` on PATH. Prefer the Visual Studio developer environment command above on Windows.

### Generic CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF -DASCII_ENABLE_AVX2=OFF
cmake --build build --target ascii-engine -j
ctest --test-dir build --output-on-failure
```

### Optional Builds

OpenCV build:

```bash
cmake -S . -B build-opencv -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=ON
```

AVX2 build:

```bash
cmake -S . -B build-avx2 -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF -DASCII_ENABLE_AVX2=ON
```

Do not mark OpenCV build support complete unless OpenCV is actually installed and the build/test path has been verified.
Do not mark hosted CI complete unless CI has run successfully on the hosted provider.

## Dependency Rules

- Do not add, remove, or upgrade dependencies without asking first.
- Required baseline dependencies are SDL2, FFmpeg, and zstd.
- OpenCV is optional and off by default.
- Vendored headers live under `vendor/`; avoid replacing them casually.
- Do not commit generated vcpkg/build artifacts from `.tools/`, `build/`, `build-noopencv/`, `build-vs/`, or `output/`.

## CMake Rules

- Keep `ascii-engine-core` as the reusable production-code target.
- Keep `ascii-engine` as the CLI executable target that links `ascii-engine-core`.
- Tests should link `ascii-engine-core`; do not reintroduce repeated production `.cpp` lists in tests.
- Keep `ASCII_USE_OPENCV=OFF` and `ASCII_ENABLE_AVX2=OFF` as defaults unless the user explicitly changes the release baseline.
- Gate OpenCV-only tests or code paths behind `ASCII_USE_OPENCV`.
- Be careful editing `CMakeLists.txt`; validate with MSVC CTest afterward.

## Code Style And Architecture

- Use C++20.
- Match existing module boundaries before adding new abstractions.
- Keep `main.cpp` focused on CLI/bootstrap/runtime orchestration. Prefer reusable logic in `src/core`, `src/render`, `src/mapping`, or `src/cli` when it needs direct tests.
- Preserve deterministic behavior in core mapping/replay paths.
- Avoid randomness in glyph selection, color mapping, replay serialization, and tests.
- Public behavior changes must update `README.md`, `PROJECT.md`, CLI help, and tests when relevant.
- Comments should clarify non-obvious algorithmic logic only; avoid narration comments.

## Runtime Contracts

### Output Behavior

Keep this behavior aligned across implementation, CLI help, `README.md`, and `PROJECT.md`:

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

Known local limitation: `.png` output is intentionally rejected in the no-OpenCV MSVC/vcpkg baseline because the FFmpeg still-image path crashed during smoke testing.

### Contour Behavior

CPU contours are enabled by default and run before normal glyph selection. Active contours override non-`blockart` glyphs with ASCII line glyphs (`-`, `|`, `/`, `\`, `+`) while preserving the normal foreground color path. `--no-contours` disables the overlay, and `--contour-thresh` maps to contour cell occupancy.

### Replay Behavior

Replay commands are part of the v1 baseline:

```powershell
.\build-noopencv\ascii-engine.exe input.mp4 --replay run.areplay
.\build-noopencv\ascii-engine.exe --inspect-replay run.areplay
.\build-noopencv\ascii-engine.exe --play-replay run.areplay
.\build-noopencv\ascii-engine.exe --play-replay run.areplay -o replay.txt
```

Replay tests must cover round-trip behavior, delta frames, and deterministic bytes.

### CLI Flags

If a flag is advertised in `src/cli/args.cpp`, it must either work or be clearly documented as best-effort/deferred.
Current important flags include:

- `--replay`
- `--inspect-replay`
- `--play-replay`
- `--no-orientation`
- `--no-contours`
- `--contour-thresh`
- `--strict-memory`
- `--fast`
- `--profile natural|anime|ui`

## Testing Expectations

Run the narrowest reliable validation for the change:

- CLI/parser/config changes: relevant unit tests plus `MainHelpTest`.
- Replay changes: `ReplayTest` and replay CLI smoke tests.
- Terminal rendering changes: `TerminalRendererTest`.
- Output behavior changes: CLI smoke tests and unsupported output test.
- CMake/build/dependency changes: full MSVC no-OpenCV configure/build/test.
- Docs-only changes: no build required unless docs claim a command or test result changed.

Default CTest coverage currently includes:

- `ComprehensiveTest`
- `CriticalFixesTest`
- `ReplayTest`
- `TerminalRendererTest`
- `MainHelpTest`
- `UnsupportedOutputTest`
- image text/still/replay smoke tests
- replay inspect smoke test

Before closing meaningful code work, scan changed source/tests for accidental markers:

```powershell
rg -n "TODO|FIXME|stub|placeholder|not implemented|unimplemented|TODO\(remove\)" CMakeLists.txt src tests README.md PROJECT.md
```

Intentional terms inside documentation or known limitations are acceptable; call them out if relevant.

## Documentation Rules

- Keep `README.md` concise and user-facing: purpose, requirements, build, usage, output matrix, troubleshooting.
- Keep `PROJECT.md` canonical for development/spec/release/status details.
- Keep `tests/baseline/README.md` for concrete validation snapshots.
- Do not reintroduce `PLAN.md`, `ROADMAP.md`, `IMPLEMENTATION.md`, `IMPLEMENTATION_REMEDIATION_PLAN.md`, `COMPLETION_PLAN.md`, or `PRESETS.md` unless the user explicitly asks for separate documents.
- When deleting or consolidating docs, update all references in remaining docs.

## Git And Generated Files

- The worktree may be dirty. Do not revert user changes.
- Do not stage or commit unless explicitly asked.
- Ignore build outputs and generated smoke artifacts unless the user explicitly wants them tracked.
- Do not delete sample media under `images/` without explicit approval.
- It is okay to remove temporary files created by your own smoke tests under `output/`.

## Known Risks

- OpenCV support is unverified locally unless OpenCV is installed and tested.
- Hosted CI status cannot be claimed until CI actually runs.
- Audio is best-effort/deferred and should not be treated as a v1 release gate.
- Webcam support is v2/deferred for the no-OpenCV baseline.
- Terminal behavior can differ by platform; keep capability fallback explicit when changing rendering.
- PNG still-image output is not part of the verified no-OpenCV MSVC baseline.

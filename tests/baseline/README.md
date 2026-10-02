# Baseline Validation

## Current local snapshot: 2026-10-02

The Windows/MSVC no-OpenCV baseline passed **58/58 Release tests** (6.54 seconds) and **58/58 AddressSanitizer tests** (16.21 seconds). This is local validation of the final working-tree source, not a new hosted CI or packaged-release result.

- Host: AMD Ryzen 7 7800X3D, 16 logical CPUs, Windows 11.
- Toolchain: Visual Studio 2022 Build Tools, MSVC 19.44, C++20, OpenMP; dependencies under `.tools/vcpkg`.
- Release build: `build-noopencv`, `ASCII_USE_OPENCV=OFF`, `ASCII_ENABLE_AVX2=OFF`, `BUILD_TESTS=ON`.
- Tested CLI SHA-256: `A9F6E5F18FC9A3C86C004AB73B55E947E5DBA0C43EB1B8D6155B1D2887E5197C`.
- ASan compiled project C++ with `/fsanitize=address /Zi /O2`; no sanitizer diagnostic was found. Existing FFmpeg, SDL2, zstd and OpenMP DLLs were not instrumented. ASan does not establish freedom from data races.

### Throughput, memory and startup

Each row summarizes three independent sequential CLI processes, each exporting all 120 frames to text without playback pacing. Processing FPS includes decoding; external-wall FPS also includes startup and exit. Memory is the **maximum sampled process working set across the three runs**, not a median or allocation bound. Output hashes matched between repeats within each row.

Settings: `--cols 120 --rows 40 --color truecolor --no-audio`, built-in defaults with no user config, Consolas (`consola.ttf`), and offline `-o frames.txt`. Full quality retains contours and uses `OMP_NUM_THREADS=12`. The speed path adds `--fast --no-contours` and uses `OMP_NUM_THREADS=4`. These thread counts are measured settings for this host, not automatic defaults.

| Fixture | Mode / threads | Median processing FPS | Median external-wall FPS | Maximum MiB |
|---|---|---:|---:|---:|
| Changing synthetic | Full / 12 | 27.40 | 26.60 | 113.51 |
| Changing synthetic | Fast / 4 | 56.34 | 53.44 | 93.95 |
| Static synthetic | Full / 12 | 77.99 | 73.90 | 111.01 |
| Static synthetic | Fast / 4 | 146.31 | 134.07 | 93.44 |
| Scene cut | Full / 12 | 79.81 | 75.46 | 113.01 |
| Scene cut | Fast / 4 | 151.25 | 137.93 | 93.50 |
| Local real video | Full / 12 | 37.11 | 32.11 | 114.04 |
| Local real video | Fast / 4 | 66.52 | 62.53 | 93.80 |

Synthetic inputs are 1920x1080 at 30 FPS; the real fixture is the first 120 decoded frames of the local 1080x1920, 24 FPS clip. Input, font and executable hashes are retained. Static scenes and cuts exercise reuse/reset behavior; they do not replace the changing-content workload.

The **24 FPS processing target passes at 120x40 with full quality and 12 threads** on this corpus; all four fast-mode workloads also exceed their 30 FPS target. The 512 MiB working-set target passes. A single changing-content scaling run at **240x80** reached full/12 **7.90 processing / 7.73 wall FPS, 265.57 MiB** and fast/4 **16.23 / 15.58 FPS, 189.96 MiB**. Those scaling observations do not meet 24 FPS and do not establish repeat determinism.

Three image-to-text startup/process/exit runs at 120x40 took a median **0.103 seconds** (maximum 0.108 seconds), with maximum sampled working set **55.86 MiB**; all met the 2-second target. The accepted strict-memory reference run exported all 120 frames at 113.42 MiB. Oversized source/grid cases failed before replacing existing outputs. The strict estimate is a budget guard, not a hard bound on third-party process allocations.

### Correctness and reference coverage

- The final suite covers CLI/config validation, geometry/resource limits, decoding, linear-light color resampling and alpha blending, motion direction/scaling, edge/contour boundaries, temporal state, glyph selection, replay, terminal rendering and export failures. Unicode image/video paths produce text identical to equivalent ASCII paths.
- Full-quality changing and real 120-frame text exports **and replay bytes** matched at **1, 8 and 12 OpenMP threads**. This is corpus-specific determinism evidence, not a cross-compiler/platform floating-point guarantee.
- Independent descriptor references cover DCT/Gabor values. Source/cache orientation checks include 210 isolated font references and 12 multicell line cases with default/custom blur and scale settings. Linear-light patch/alpha known answers and frozen-reference comparisons support arithmetic corrections. Arbitrary neighboring texture context and general perceptual quality are not exhaustively measured.
- Colored ASCII retains mean foreground color on a black background. Sparse glyph coverage limits reconstructed brightness; whole-image radiometric brightness preservation is not claimed. Tone remapping was not introduced to improve benchmark numbers.
- Final format checks decoded all 18 supported image/video combinations: `.jpg`, `.jpeg`, `.bmp`, `.mp4`, `.m4v`, `.mov`, `.mkv`, `.avi`, `.gif`. Image exports contain one frame; video-to-still contains the first frame; video-to-video contains all eight fixture frames. Binary RGB/RGBA pipe inputs passed complete-frame checks and rejected partial final frames.

### Reproduction and retained evidence

From an x64 Visual Studio developer shell with existing local dependencies configured:

```powershell
cmake -S . -B build-noopencv -DCMAKE_BUILD_TYPE=Release -DASCII_USE_OPENCV=OFF -DASCII_ENABLE_AVX2=OFF -DBUILD_TESTS=ON
cmake --build build-noopencv -j 4
ctest --test-dir build-noopencv --output-on-failure

$env:OMP_NUM_THREADS = '12'
.\build-noopencv\ascii-engine.exe output/goal-audit/reference-1080p.mkv --cols 120 --rows 40 --color truecolor --no-audio -o output/reference-frames.txt
```

Use four threads and add `--fast --no-contours` for the measured speed path. Helpers and raw evidence remain local under `output/goal-audit/` and are intentionally not release artifacts:

| Evidence | Local path under `output/goal-audit/` |
|---|---|
| Release gate and source/binary hashes | `final-58-ctest.log`, `final-58-sources-{before,after}.json`, `final-58-binaries.json` |
| ASan gate, compile flags, tool/dependency hashes | `asan-final58/README.md` and proof files; `run-asan-gate.ps1` |
| Three-run timings, arguments, hashes and text digests | `final-benchmarks/{summary,results,environment}.json`; `run-benchmarks.ps1 -FullThreads 12 -FastThreads 4 -Repeats 3` |
| Fixture provenance and generation | `benchmark-fixtures/manifest.json`, `create-benchmark-fixtures.ps1` |
| Single-run 240x80 scaling | `final-scaling/{summary,results,environment}.json` |
| Startup, strict-memory, thread determinism, Unicode paths | `final-runtime/results.json`, `final-runtime/hashes-{before,after}.json`; `verify-final-runtime.ps1` |
| Decoded exports and binary-pipe contracts | `final-formats/results.json`, `final-raw-pipe/results.json`; `verify-formats.ps1`, `verify-raw-pipe.ps1` |
| Diagnostic quality/reference limits | `analysis-repro/quality-report.md`, `analysis-repair-handoff.md` |

### Remaining limits

OpenCV and AVX2 builds remain unverified. Webcam and audio playback polish remain deferred; audio regression checks use a dummy device. Live terminal/device behavior and encoder throughput are not represented by the text-export benchmark. PNG still output is intentionally rejected. WebM exports failed with the installed `av1_d3d12va` encoder (`Invalid argument`); this is an unavailable encoder path, not a successful export. Both failed formats left no destination file. No fresh hosted CI run or distributable ZIP was produced for this snapshot.

## Historical snapshot: 2026-07-16

The earlier clean `cmake-build-final-proof` configuration built 41 targets/objects without compiler warnings and passed 34/34 tests. Its packaged Windows x64 ZIP passed `--help` on this host and contained the executable, license, README, runtime notes, SDL2, FFmpeg and zstd DLLs. Historical ZIP SHA-256: `FAEAED14B1E9E17EA250096D5B643FCED718D903DEFDB84B7AA3F2A6902085D3`.

[Hosted CI run 29510258100](https://github.com/FueledByRedBull/ASCII/actions/runs/29510258100) was recorded as passing that 34-test baseline on Windows, Linux and macOS, followed by launching the packaged ZIP on a separate clean Windows runner. This retained July evidence does not validate the October working tree or package.

The July 60-frame 1920x1080/30 FPS, 120x40 text-export measurement was full quality **10.86 processing FPS / 136.43 MiB**, speed path **25.70 FPS / 106.70 MiB**, and one-image startup **0.32 seconds / 75.91 MiB**. It used a different workload length and recorded settings; this is a historical anchor, not a controlled before/after speedup comparison with the current matrix.

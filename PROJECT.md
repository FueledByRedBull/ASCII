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

Complete and verified for the v1 no-OpenCV, non-AVX2 baseline as of 2026-07-16:

- A clean Release configure/build in `cmake-build-final-proof` passed all 34 no-OpenCV, no-AVX2 CTest cases.
- Every Blocker, High, and Medium implementation finding below is fixed with focused regression coverage.
- Generated real-video tests cover decode count/FPS, numbered text frames, encoded video, first-frame stills, block-art media, and truncated-input failure.
- Replay coverage includes strict validation, random delta access, full Unicode, deterministic bytes, and deterministic golden text export.
- Requested text, replay, and encoded outputs use failure-aware finalization and do not report success for absent/incomplete files.
- A versioned Windows x64 ZIP built from the clean tree contains the executable, license, `README.md`, runtime notes, and required DLLs; its bundled `--help` smoke test passes both on the build host and after artifact transfer to a separate clean Windows runner.
- [Hosted CI run 29510258100](https://github.com/FueledByRedBull/ASCII/actions/runs/29510258100) passed on Windows, Linux, and macOS and completed the independent Windows package smoke test.

The v1 release evidence gates are satisfied. OpenCV and AVX2 remain optional follow-up baselines.

The reference workload is measured rather than assumed. On a Ryzen 7 7800X3D, full-quality processing reaches 10.86 FPS, while `--fast --no-contours` reaches 25.70 processing FPS. Both stay well below 512 MiB and startup is below 2 seconds. The full-quality path therefore retains a documented performance gap; the explicit speed path meets the throughput target.

## v1 Completion Audit (2026-07-16)

This audit combined source review, the existing Windows CTest suite, a targeted CLI failure reproduction, and Semgrep OSS scans. The existing build passed 13/13 tests. A requested text and replay export into a missing directory produced neither file but returned exit code `0`, confirming that the current happy-path tests are not sufficient release evidence.

Semgrep's `p/c`, `p/security-audit`, `p/secrets`, and `p/github-actions` rules reported no findings. The required 0xdea C/C++ rules produced 758 mostly lexical or audit-hint results; manual triage did not confirm a security defect from those hits. Its parser could not fully analyze the vendored headers plus two source locations, so the scan is supporting evidence rather than a memory-safety proof. Raw and merged local artifacts are under `static_analysis_semgrep_1/` and should not be committed as release artifacts.

Priority meanings:

- **Blocker:** violates a required or currently advertised v1 behavior, can lose requested output, or substantially corrupts rendering.
- **High:** visible correctness or control failure that needs a regression test before release.
- **Medium:** quality, performance, resilience, or maintainability work that should be resolved or explicitly accepted in the release notes.

Resolution status: V1-R01 through V1-R09 and V1-A01 through V1-A13 are implemented and covered by the 34-test no-OpenCV suite. The tables remain as the historical defect statement and acceptance criteria.

### Confirmed Runtime And Contract Defects

| ID | Priority | Area | Finding and required outcome |
|---|---|---|---|
| V1-R01 | Blocker | Config precedence | `Args` defaults are applied as if the user supplied them. A config file's `char_set`, `scale_mode`, hysteresis/orientation booleans, `no_audio`, `profile_live`, and `strict_memory` can be overwritten by absent CLI flags. Track option presence explicitly and test defaults < config < explicit CLI precedence. |
| V1-R02 | Blocker | Pause | The main loop calls `source->read(frame)` before checking `paused`; while paused it consumes and discards frames. Move frame acquisition after pause handling and test that source position does not advance. |
| V1-R03 | Blocker | Output errors | Text writers return `void` and silently ignore open/write failures. Requested replay open/write failures are warnings, and the process can return success without either requested file. Propagate errors, remove incomplete outputs where safe, and return non-zero. |
| V1-R04 | High | CLI validation | Missing values, unknown options, invalid enums/numbers, and rejected paths are silently ignored or replaced with defaults. Invalid output arguments can therefore fall back to terminal rendering. Parse into a validated result with one clear diagnostic per bad argument. |
| V1-R05 | High | Terminal controls | Changing color mode does not invalidate `TerminalRenderer`'s previous-cell cache or reset a block-art background. Unchanged cells retain the old mode/background. Force a full repaint and color reset when mode changes. |
| V1-R06 | Blocker | Replay | The reader does not reject unsupported versions or cap dimensions/compressed sizes, trusts frame offsets and flags, and can allocate from untrusted header values. Delta-frame random reads/seeks do not rebuild prior state, and writers truncate codepoints to 16 bits despite a 32-bit field. Define strict v1 validation, checked arithmetic, sequential/random-access semantics, corruption tests, and full Unicode round trips. |
| V1-R07 | High | Decode completion | `FrameSource::read` uses one `false` result for clean EOF and decode failure, so partial/truncated processing can be reported as success. Expose EOF versus error and fail requested exports on decode errors. |
| V1-R08 | Medium | Export pacing | The real-time sleep runs for terminal playback and offline text/video/still export. Multi-frame exports are unnecessarily capped at playback FPS. Pace only live playback unless an explicit real-time export mode is requested. |
| V1-R09 | Medium | Memory safety | Core image containers and several allocation calculations use unchecked signed dimension products. Replay and pipe/source dimensions also bypass the strict-memory estimate. Add checked size multiplication, source/replay caps, and graceful allocation failure handling. |

### Confirmed Algorithm And Implementation Defects

| ID | Priority | Area | Finding and required outcome |
|---|---|---|---|
| V1-A01 | Blocker | Glyph model | Zero-area glyphs such as space are dropped from `GlyphCache`; the remaining glyph bounding boxes are stretched to the full cell with nearest-neighbor sampling while advance and bearings are discarded. This removes a true blank candidate and distorts density/orientation statistics. Rasterize every glyph into a fixed baseline-aligned cell canvas, preserve space as an all-zero bitmap, and use filtered resampling only when needed. |
| V1-A02 | Blocker | Temporal state | One `initialized` flag is shared by luminance, edge, coherence, and glyph state. Because luminance initializes first, first-frame edge/coherence values are incorrectly blended against zero. Use independent initialization or initialize the complete cell state atomically. |
| V1-A03 | Blocker | Glyph hysteresis | Change decisions compare a current candidate against a stale score/loss saved on an earlier frame, not the old glyph's loss on the current cell. Same-glyph frames do not refresh the baseline. Bounded score `1.0` in simple/block modes can make later changes impossible. Recompute both keep/change costs on current data and test step changes, moving edges, and scene cuts. |
| V1-A04 | High | Motion/temporal integration | Candidate transition cost is computed from the same-index prior glyph, the accept/reject decision may use a motion-shifted glyph, and rejection emits the same-index glyph again. Use one motion-compensated reference consistently for selection, comparison, and fallback. |
| V1-A05 | High | Motion confidence | Dense-flow confidence is derived largely from motion magnitude, is ignored when cell flow is averaged, and `motion_reuse_confidence_decay` therefore has no observable effect. Define confidence from match quality/consistency, weight or gate temporal warping with it, and test the decay control. |
| V1-A06 | Blocker | Orientation | `cell_orientation` is a gradient normal, but simple/fast mode maps it directly to a line glyph, rotating horizontal/vertical edge glyphs by 90 degrees. In unified mode, `--no-orientation` is ignored because orientation loss is still computed. Correct the tangent mapping, honor the flag in the loss, and add polarity-invariant synthetic edge tests. |
| V1-A07 | High | Edge controls | With default multi-scale/hybrid detection, `--blur` does not affect the multi-scale path, `scale_variance_floor/ceil` are unused, `hybrid` is identical to `local`, and `--edge-thresh` does not set the detector's adaptive pixel threshold. Either wire each advertised control to a measurable behavior or remove/defer it. |
| V1-A08 | High | Runtime cache | A global sampled mean-difference threshold can reuse an entire prior pipeline result while a small object moves, freezing luminance, color, edges, and cell stats. Partial stats reuse can also pair current pixels with prior statistics. Replace this with tile/cell invalidation or restrict reuse to proven-identical frames; add localized-motion tests. |
| V1-A09 | High | ANSI dithering | Cells are always composed left-to-right, but odd rows distribute error as if they were traversed right-to-left. Most odd-row error is sent to already processed cells. Traverse odd rows in reverse or use one-direction diffusion and test exact error propagation. |
| V1-A10 | High | Block-art export | Block-art selection emits Unicode block codepoints independently of the configured glyph set, while bitmap/video rendering can only draw glyphs present in `GlyphCache`. With the default basic set, selected block shapes may render as background-only cells. Always cache renderer-required block glyphs and cover block-art still/video output. |
| V1-A11 | Medium | Glyph search | Unified selection estimates the brightness position as `count * luminance`, assuming glyph brightness is uniformly distributed, then searches only a local window. It can exclude the actual nearest glyph; low-adaptive edge search also skips every second candidate without a quality bound. Center searches using actual brightness lower-bound data and validate any pruning against exhaustive selection. |
| V1-A12 | Medium | Resampling | The no-OpenCV resize path uses corner-aligned bilinear samples without an area/low-pass downsampler; the OpenCV path uses `INTER_LINEAR` for reduction. Large source-to-cell reductions can alias into false edges and texture. Use center-aligned sampling and area/prefiltered downscaling, then compare against fixed synthetic patterns. |
| V1-A13 | Medium | Bilateral grid | `bilateral_spatial_bins` is validated, hashed, and configurable but never used; the grid always has one spatial bin per output cell. Implement the documented resolution control or remove it from the v1 config surface. |

### Algorithm Evidence Disposition

- **Terminal font model:** documented in `README.md`; known-font pixel checks use bitmap/video output because terminal applications control their own display font.
- **Orientation representation:** glyph orientation comparison is folded modulo pi, and normal/inverted horizontal, vertical, and diagonal fixtures produce the same line direction.
- **Scale selection:** the public two-scale endpoints are used directly, local variance controls are regression-tested at both extremes, and no unsupported claim of spatially optimal scale selection is made.
- **Contour override policy:** synthetic line and intersection fixtures verify the retained confidence/occupancy-gated contour override and tone-preserving foreground-color path.
- **Frame-rate dependence:** smoothing uses time-normalized alpha and is tested for equivalent one-second responses at 15, 24, 30, and 60 FPS; reuse and phase intervals scale from the 30 FPS reference.
- **Feature weights and presets:** profile application is deterministic and config/CLI precedence is tested. Profiles remain explicitly described as hand-tuned presets, not empirically ranked quality claims.

### Required Work To Finish v1

Complete in this order; later phases depend on the earlier contracts being stable.

**Compatibility-first rule:** finish v1 by correcting and integrating the features already implemented in the current pipeline. Preserve the existing documented modes, controls, output behaviors, determinism guarantees, and supported baseline unless a feature is proven unsafe or impossible to support. Do not simplify the reference algorithm by deleting existing stages, flags, or modes; establish a correct baseline for each stage, then repair or retune the advanced behavior against tests. Any proposed removal or deferral requires an explicit project decision and corresponding documentation before code is changed.

1. **Correctness blockers**
   - [x] Fix V1-R01, V1-R02, V1-R03, and V1-R06.
   - [x] Fix V1-A01, V1-A02, V1-A03, and V1-A06.
   - [x] Add focused regression tests for every blocker before changing tuning constants.
2. **Advertised behavior**
   - [x] Fix the remaining High findings while preserving the affected flags and modes and making their advertised behavior measurable in tests.
   - [x] Make all requested outputs atomic enough to avoid reporting success for absent or incomplete files.
   - [x] Keep implementation, CLI help, output matrix, `README.md`, and this document aligned.
3. **Algorithm validation**
   - [x] Add synthetic fixtures for blank/ramp cells, horizontal/vertical/diagonal edges, inverted polarity, intersections, localized motion, scene cuts, and color ramps.
   - [x] Compare glyph choices against exhaustive selection and verify temporal convergence rather than only no-crash behavior.
   - [x] Add deterministic golden text/replay outputs and small rendered-image diffs with documented tolerances.
4. **End-to-end media coverage**
   - [x] Add a tiny deterministic video fixture or generate one during tests.
   - [x] Verify decoded frame count, numbered text frames, encoded video frame count/duration, first-frame still behavior, and corrupt/truncated input failure.
   - [x] Test unwritable text/replay targets, partial encoder failures, malformed CLI input, and config precedence.
5. **Release proof**
   - [x] Run a clean Windows/MSVC no-OpenCV configure, build, and full CTest suite.
   - [x] Record the documented 1080p `120x40` performance/memory workload after correctness fixes; performance shortcuts must pass visual regressions.
   - [x] Get green hosted CI runs on Windows, Linux, and macOS and record the run links/date: [2026-07-16 run 29510258100](https://github.com/FueledByRedBull/ASCII/actions/runs/29510258100).
   - [x] Produce a minimal versioned release bundle with the executable, license, `README.md`, and dependency/runtime notes.

OpenCV, AVX2, webcam polish, audio polish, GPU compute, PNG output, and richer installers remain optional and are not v1 blockers unless they continue to be advertised as supported v1 behavior.

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

Measured on 2026-07-16 with a deterministic 60-frame 1920x1080/30 FPS MKV, Release MSVC/OpenMP, no OpenCV, and no AVX2, on an AMD Ryzen 7 7800X3D:

- Full-quality `120x40`, truecolor, no audio: 10.86 processing FPS, 136.43 MiB peak working set.
- `--fast --no-contours` with the same workload: 25.70 processing FPS, 106.70 MiB peak working set.
- One-image `120x40` startup/process/exit: 0.32 seconds, 75.91 MiB peak working set.

The memory and startup targets pass. The throughput target passes in the explicit speed path; full-quality mode remains below target and is a known performance limitation rather than an unmeasured claim.

`--strict-memory` enforces a conservative 512 MiB estimated-memory budget.

## Testing

Default no-OpenCV CTest coverage includes:

- algorithm/config/CLI unit coverage
- critical regression tests
- replay round-trip and determinism tests
- terminal renderer diff tests
- CLI smoke tests
- unsupported output validation
- generated real-video decode/export and truncated-input tests
- block-art still/video tests
- deterministic replay-to-text golden output

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
- [x] Output mode matrix happy paths are implemented and documented.
- [x] Replay write/read/inspect/play happy paths work sequentially.
- [x] Replay determinism test passes.
- [x] Terminal rendering tests cover glyph/fg/bg diffs.
- [x] CLI smoke tests pass locally.
- [x] Known limitations are documented.
- [x] All v1 Blocker findings in the completion audit are fixed with regressions.
- [x] All v1 High findings are fixed or removed from the advertised v1 surface.
- [x] Real video decode and every documented video output behavior have automated coverage.
- [x] Algorithm fixtures and golden outputs pass for space, edge direction, motion, temporal changes, and color.
- [x] Requested output failures return non-zero and do not claim success.
- [x] The documented performance and memory workload is measured after correctness fixes.
- [x] Hosted CI passes on supported platforms: [2026-07-16 run 29510258100](https://github.com/FueledByRedBull/ASCII/actions/runs/29510258100).
- [x] A minimal versioned release bundle is smoke-tested on a separate clean Windows runner in [run 29510258100](https://github.com/FueledByRedBull/ASCII/actions/runs/29510258100).

Optional follow-up, not a v1 release gate:

- [ ] OpenCV build configures and tests when OpenCV is installed.
- [ ] AVX2 build configures, tests, and matches deterministic reference behavior where promised.

## Known Limitations

- On Windows, `cmake` and `ctest` may require the Visual Studio developer environment.
- Verified still-image targets for the no-OpenCV MSVC/vcpkg build are `.jpg`, `.jpeg`, and `.bmp`.
- `.png` output is intentionally rejected in this baseline because the FFmpeg still-image path crashed in local smoke testing.
- Webcam support is v2/deferred for the no-OpenCV baseline.
- Audio is best-effort/deferred and not a v1 release gate.
- OpenCV-enabled builds need separate validation with OpenCV installed.
- Full-quality processing is below the 24 FPS reference target on the measured workload; `--fast --no-contours` meets it.
- Terminal output can select glyphs using a loaded font, but the terminal ultimately renders those codepoints with its own configured font. Known-font visual validation is therefore performed through bitmap/video outputs.
- Content profiles are deterministic hand-tuned presets, not empirically ranked quality claims.

## Future GPU Edge Downscaling

A compute-shader path is technically applicable, but not required for v1.

The proposed approach:

- Use tile-sized workgroups, ideally matching glyph/cell dimensions such as `8x8`.
- Build a group-shared histogram of edge direction/class values.
- Emit the dominant edge class only when a tile crosses an edge-density threshold.
- Keep a CPU reference path as the deterministic baseline.
- Document backend, supported platforms, determinism differences, fallback behavior, and visual parity tests.

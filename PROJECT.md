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

### Completed correctness, algorithm and performance goal (2026-10-02)

Ponytail full mode was used throughout. The goal was reordered into correctness
and algorithm repair first, followed by measured optimization and final verification.
Both phases are complete for the Windows/MSVC no-OpenCV, non-AVX2 baseline.
This is evidence for the tested contracts and inputs, not a claim that all possible
bugs or platform differences have been eliminated.

- [x] Read all project-owned source, tests, documentation, scripts and TODOs,
      plus all three vendored headers; retain the 80-file initial inventory.
- [x] Freeze the initial source, reproduce confirmed defects, repair their root
      causes, and add focused regression coverage.
- [x] Review algorithm choices and integration independently; retain useful
      features and reject optimizations without measured benefit.
- [x] Run the full Release and AddressSanitizer suites after the final code change.
- [x] Meet 24 FPS full-quality and 30 FPS fast-mode processing targets at 120x40
      on the measured host, using repeated changing/static/cut/real-video workloads.
- [x] Verify startup below 2 seconds, reference peak working set below 512 MiB,
      240x80 scaling, strict-memory rejection, and deterministic replay/text bytes.
- [x] Decode supported exports, verify malformed/truncated input and preservation
      failures, and exercise Unicode filenames across Windows input/output paths.

**Validation provenance:** the audited working tree was prepared on `main`,
with pre-existing user changes preserved in the frozen starting baseline.
Existing SDL2 2.32.10, FFmpeg 8.1.1 and zstd 1.5.7 dependencies were restored
with user authorization. No dependency was added or upgraded.

The unchanged starting source passed all 34 original tests, despite focused
reproductions finding 13 configuration/CLI failures, nine analysis failures and
21 mapping failures. Later reproductions exposed output replacement on failure,
Windows pipe/path errors and FFmpeg buffer-contract violations. The original
source, failures and subsequent proofs remain under `output/goal-audit/`;
passing final checks do not erase those historical failures.

**Main repairs and algorithm decisions**

- Configuration rejects malformed/nonfinite/out-of-range values, applies explicit
  flags consistently, and hashes rendering semantics into replay fingerprints.
  Debug views now reach every output path and honor the chosen color palette.
- Linear-light resampling, fit/fill geometry, color means, bitmap alpha blending,
  adaptive thresholds, gradient orientation and cache invalidation have independent
  references. Glyph descriptors use the same configured gradient processing as
  the source. Frequency matching remains because removing it reduced exact matches.
- Motion uses consistent forward displacement and per-axis full-resolution units,
  complete border coverage, bounded thread-local FFT caches and deterministic
  confidence. The invalid unused alternate dense solver was removed; supported
  sparse block matching and hierarchical phase correlation remain.
- Block art fits all 16 quadrant masks, caches every emitted primitive, and avoids
  freezing glyphs with mismatched colors. The ineffective optional spectral palette
  implementation was replaced with deterministic bounded OKLab clustering.
- Decoder/encoder conversion uses FFmpeg-owned aligned, padded frames. Guarded
  reproductions found old decoder tail writes; a focused ASan boundary harness
  independently caught the encoder's undersized plane arrays and missing padding.
  Source timing, format changes, clean EOF, binary stdin and geometry limits are tested.
- Replay validates memory/index bounds and cleans up failed or closed state.
  Writers preserve existing destinations when failure precedes finalization.
  Directory, direct, numbered and sequence input/output collisions are rejected.
  MSVC executables use a process-local UTF-8 manifest for Unicode paths; the new
  regression covers media, text, replay, config, font and wildcard-sequence paths.
- Empty geometry terminates promptly; font, palette, temporal and renderer state
  resets correctly; terminal output sanitizes invalid/control codepoints and restores
  changed console state. Audio ownership, synchronization and bounds are tested
  with SDL's dummy driver; actual speaker playback remains outside this gate.
- Equivalent optimizations remove full-image prefix passes and discarded copies,
  fuse edge classification, retain only needed DCT work, vectorize independent
  Gabor/Laplacian work and avoid redundant glyph scoring/temporary zero fills.
  Slower SIMD NMS and row-scheduling candidates were rejected, as was a rounding
  helper whose small measured gain did not justify added code.

**Final verification**

| Check | Result |
|---|---|
| MSVC Release configure/build/CTest | 58/58 pass; CTest 6.54 seconds |
| MSVC AddressSanitizer configure/build/CTest | 58/58 pass; CTest 16.21 seconds; no sanitizer reports |
| Full changing 1080p, 120x40, 12 threads | 27.40 processing / 26.60 external-wall FPS median; 113.51 MiB peak |
| Fast changing 1080p, 120x40, 4 threads | 56.34 processing / 53.44 external-wall FPS median; 93.95 MiB peak |
| Supplied real-video segment, full / fast | 37.11 / 66.52 processing FPS medians |
| One-image startup/render/exit, three processes | 0.1004-0.1075 seconds; maximum 55.86 MiB |
| 240x80 changing-video stress, one run per mode | Full 7.90 FPS / 265.57 MiB; fast 16.23 FPS / 189.96 MiB |
| Replay/text determinism | Exact bytes across 1/8/12 threads for both 120-frame video fixtures; repeated benchmark text also identical |
| Export matrix | All 18 supported image/video combinations decoded with expected geometry/frame count and clean EOF |
| Input and admission failures | Truncated RGB/RGBA pipes and over-budget strict-memory requests rejected; existing outputs preserved |

Measurements use a Ryzen 7 7800X3D, Windows 11, MSVC 19.44, Release/OpenMP,
truecolor, no audio, no OpenCV and no AVX2. Normal benchmark values are medians
of three independent 120-frame text-export processes; maximum measured memory
is reported, not averaged. Full mode uses 12 OpenMP threads and fast mode uses
4. The 8-thread intermediate checkpoint reached only 21.85 FPS full quality;
thread-count tuning is part of the measured configuration, not a default-setting
claim. The initial corrected diagnostic was 10.50 FPS. Static/cut/real-video
results and exact commands are in [the validation snapshot](tests/baseline/README.md).
The larger grid is a scaling observation, not a 24 FPS acceptance target.

Meaningful quality/reference coverage includes 210 exact glyph references,
12 directional grid fixtures, 8,064 bitwise DCT/Gabor cases, 42,192 moment-reference
cells, and independent edge/motion equivalence corpora. Bitmap blending agrees
with an independent sRGB reference within one byte; its tested RGB patch RMSE
improved from 0.4443 to 0.3764. Ordinary ASCII remains limited by glyph coverage;
block art is the foreground/background reconstruction mode. These finite corpora
support the chosen algorithms without claiming universal visual optimality.

Evidence: `output/goal-audit/final-58-*`, `asan-final58/`, `final-benchmarks/`,
`final-scaling/`, `final-runtime/`, `final-formats/` and `final-raw-pipe/`.
The final CLI SHA-256 is
`A9F6E5F18FC9A3C86C004AB73B55E947E5DBA0C43EB1B8D6155B1D2887E5197C`.
Source hashes stayed unchanged during each build/test gate, and executable/input
hashes stayed unchanged during the final runtime checks. All 46 ASan compile
commands carry `/fsanitize=address /Zi /O2`; prebuilt dependency DLLs are not
instrumented, and ASan does not establish race freedom. Full original sample
media decoding reached clean EOF at 1,874 MP4 frames and 52 GIF frames.

OpenCV, AVX2, other-host performance, fresh hosted CI and packaging remain
unverified by this goal. PNG output remains deliberately excluded. The installed
FFmpeg lacks a usable WebM encoder; the CLI fails explicitly without leaving a
partial destination. Strict-memory is a conservative admission estimate, not an
OS quota. Startup measurements use independent processes, not forced cold caches.
No source/build work remains for the defined baseline goal; optional future work
is listed below. Historical July evidence is retained separately.

### Historical v1 checkpoint (2026-07-16)

Complete and verified for the v1 no-OpenCV, non-AVX2 baseline as of 2026-07-16:

- A clean Release configure/build in `cmake-build-final-proof` passed all 34 no-OpenCV, no-AVX2 CTest cases.
- Every Blocker, High, and Medium implementation finding below is fixed with focused regression coverage.
- Generated real-video tests cover decode count/FPS, numbered text frames, encoded video, first-frame stills, block-art media, and truncated-input failure.
- Replay coverage includes strict validation, random delta access, full Unicode, deterministic bytes, and deterministic golden text export.
- Requested text, replay, and encoded outputs use failure-aware finalization and do not report success for absent/incomplete files.
- A versioned Windows x64 ZIP built from the clean tree contains the executable, license, `README.md`, runtime notes, and required DLLs; its bundled `--help` smoke test passes both on the build host and after artifact transfer to a separate clean Windows runner.
- [Hosted CI run 29510258100](https://github.com/FueledByRedBull/ASCII/actions/runs/29510258100) passed on Windows, Linux, and macOS and completed the independent Windows package smoke test.

The v1 release evidence gates are satisfied. OpenCV and AVX2 remain optional follow-up baselines.

At the July checkpoint, full-quality processing reached 10.86 FPS on a Ryzen 7 7800X3D, while `--fast --no-contours` reached 25.70 FPS. Both stayed below 512 MiB and startup was below 2 seconds. Full quality missed that snapshot's throughput target; the October results above supersede this performance limitation.

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

### Character Sets

- `traditional`: compact plain-ASCII luminance ramp (` .:-=+*#%@`) for the classic, less noisy look.
- `basic`: extended plain-ASCII vocabulary for finer tonal and structural matching.
- `line-art`: ASCII plus Unicode box-drawing glyphs for graphic sources.
- `blocks`: Unicode shade and block glyphs.

Renderer-only block primitives remain cached for block-art output but are excluded from normal glyph selection unless the selected set contains them. Contour overrides use `-`, `|`, `/`, `\\`, and `+`.

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

Measured baseline (2026-10-02):

- CPU: Ryzen 7 7800X3D; Windows 11/MSVC 19.44. Other hardware needs its own measurements.
- Build: Release, OpenMP enabled, OpenCV and AVX2 disabled.
- Workload: 120-frame changing/static/cut video at 1920x1080/30 FPS and a real
  portrait clip at 1080x1920/24 FPS; `120x40`, truecolor, audio disabled, offline
  text export; three independent runs per case.
- Targets: at least 24 processing FPS full quality and 30 FPS with
  `--fast --no-contours`, startup/render/exit below 2 seconds, and peak working
  set below 512 MiB on this host.
- Results: all targets pass with `OMP_NUM_THREADS=12` for full quality and `4`
  for fast mode. Changing-video medians are 27.40 and 56.34 FPS respectively;
  the full-quality reference peak is 113.51 MiB. Threads are an explicit setting,
  and terminal painting is outside the offline-export throughput measurement.

The complete current matrix, 240x80 stress result, commands and proof locations
are in [tests/baseline/README.md](tests/baseline/README.md). Its separately dated
July snapshot preserves the earlier performance miss and hosted release evidence.

`--strict-memory` rejects requests above a conservative 512 MiB estimate including
analysis buffers, cells, the optional bilateral grid, decode headroom and audio.
Strict-mode audio PCM is capped at 32 MiB; normal audio is capped at 256 MiB.
Replay readers independently cap estimated index/decode working storage at
256 MiB by default. These are allocation estimates, not an operating-system
memory quota: FFmpeg may allocate while probing streams discovered after open,
before their dimensions can be checked. Optional OpenCV allocations are unverified.

Rendering uses linear-light area/bilinear resampling and bitmap alpha blending.
ASCII color semantics are source-colored foreground strokes on black, preserving
the explicitly selected charset. Sparse glyphs have limited coverage, so this mode
does not promise whole-cell radiometric brightness reconstruction. Block art fits
foreground/background colors to all 16 quadrant masks. Its optional historical
`block_spectral_palette` setting now uses deterministic bounded OKLab clustering.
Frequency matching is retained: the font-reference corpus showed a material loss
of exact matches when it was disabled. Texture matching also remains available.

Configuration accepts the implemented reserved modes only: `input.mode = "file"`,
`output.mode = "terminal"`, and `color.quantization = "oklab"`. Output filenames
still determine export behavior. Unsupported values fail instead of being ignored.

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

## Historical V1 Release Checklist (2026-07-16)

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
- Performance targets pass on the measured Windows/MSVC host with the documented thread counts; other hosts and larger grids need separate measurements.
- Unicode Windows paths require the MSVC UTF-8 application manifest and Windows 10 version 1903 or later; local verification used Windows 11.
- The installed FFmpeg has no usable WebM encoder; `.webm` output fails clearly rather than producing a partial file.
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

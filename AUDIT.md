# ASCII Engine - Complete Implementation and Alignment Plan

## 1) Purpose

This plan defines a full, execution-ready implementation roadmap to bring the codebase into behavioral alignment with:

- `PLAN.md` (feature/spec intent)
- `IMPLEMENTATION.md` (architecture and pipeline intent)

It is focused on closing current integration gaps, correcting logic risks, and producing a stable, testable, cross-platform release.

---

## 2) Current State Summary

The repository has strong module coverage but partial runtime integration.

### What is already in place

- Core pipeline modules exist (`frame_source`, `pipeline`, `edge_detector`, `cell_stats`, `temporal`).
- Glyph model modules exist (`font_loader`, `glyph_cache`, `glyph_stats`, `char_sets`).
- Mapping/render modules exist (`char_selector`, `color_mapper`, `terminal_renderer`, `bitmap_renderer`, `video_encoder`).
- CLI and tests exist.

### Main gaps to close

- Runtime currently bypasses edge-aware glyph selection and mostly uses luminance ramp output.
- Temporal smoothing/hysteresis state exists but is not used to drive glyph transitions.
- Several correctness risks exist (font signature handling, sigma=0 blur path, ANSI16 bright color mapping).
- Scaling semantics and output semantics do not fully match fit/fill/stretch + output mode intent.
- Media and platform behavior is narrower than documented scope (audio, pipe/image sequence, Windows terminal path).

---

## 3) Success Criteria

The effort is complete when all are true:

1. Runtime glyph decisions follow documented edge-vs-fill logic with temporal hysteresis.
2. Pipeline behavior matches spec semantics for resize, edge extraction, cell stats, and smoothing.
3. Output modes are deterministic and explicit (terminal, text export, encoded video, single-image output).
4. Color behavior is correct for none/16/256/truecolor/block-art modes.
5. Cross-platform behavior is valid on Linux/macOS/Windows terminals.
6. Unit + integration + regression tests cover all critical paths and pass in CI.
7. Documentation reflects actual behavior (no spec drift).

---

## 4) Workstreams

1. Runtime logic integration and algorithm correctness
2. Pipeline/scaling correctness
3. Glyph modeling and selector quality
4. Renderer and color correctness
5. Input, output, and audio behavior
6. Platform and build portability
7. Performance and instrumentation
8. Testing, CI, and documentation sync

---

## 5) Detailed Phased Plan

## Phase 0 - Baseline, Safety, and Planning Artifacts

### Goals

- Establish reproducible baseline behavior before large edits.
- Add minimal safety checks to prevent known crash/invalid paths.

### Tasks

- Add a plan tracker section in this file with per-phase checkboxes.
- Capture baseline outputs for one video and one image input:
  - terminal screenshots
  - generated text output
  - encoded output sample
- Guard known invalid parameter paths:
  - prevent sigma <= 0 in blur path
  - normalize threshold config validation

### Files

- `src/core/edge_detector.cpp`
- `src/cli/args.cpp`
- `tests/test_comprehensive.cpp`

### Acceptance criteria

- No division-by-zero risk in blur path.
- Invalid params are clamped or rejected consistently.
- Baseline artifacts stored in `tests/baseline/` (or equivalent).

---

## Phase 1 - Core Runtime Integration (Highest Priority)

### Goals

- Make runtime faithfully execute planned selection pipeline.
- Replace fallback-only luminance mapping as the default decision path.

### Tasks

1. Wire `CharSelector` into frame loop:
   - use `CellStats` edge/fill decision
   - select with orientation mode based on CLI config
2. Wire `TemporalSmoother` for both:
   - smoothed input features (`luminance`, `edge_strength`)
   - glyph hysteresis (`should_change_glyph`, `update_glyph`)
3. Ensure per-cell index mapping is stable frame-to-frame.
4. Keep simple luminance ramp as explicit fallback mode only.

### Files

- `src/main.cpp`
- `src/mapping/char_selector.cpp`
- `src/core/temporal.cpp`
- `src/core/temporal.hpp`

### Acceptance criteria

- Main loop no longer hardcodes luminance-only glyph selection for default path.
- Glyph changes show reduced flicker under stable scenes.
- Temporal hysteresis is active and measurable.
- Existing tests pass; new integration tests added (see Phase 8).

---

## Phase 2 - Pipeline and Scaling Semantics

### Goals

- Align resize and grid semantics with documented fit/fill/stretch behavior.
- Guarantee deterministic mapping between terminal grid and processed image data.

### Tasks

1. Redesign resize path to output exactly terminal-grid-aligned sampling domain.
2. Implement proper fit/fill/stretch semantics:
   - fit: preserve aspect with letterbox
   - fill: preserve aspect with crop
   - stretch: direct grid deformation
3. Keep color/luminance buffers aligned after scaling/crop.
4. Make char aspect ratio handling explicit and unit tested.
5. Add optional gamma handling flag with clear default behavior.

### Files

- `src/core/pipeline.cpp`
- `src/core/pipeline.hpp`
- `src/core/types.hpp`
- `src/cli/args.cpp`
- `src/cli/args.hpp`

### Acceptance criteria

- Grid dimensions are stable and match renderer expectations.
- Fit/fill/stretch output is visually and numerically correct.
- Aspect-ratio tests cover portrait, landscape, and square inputs.

---

## Phase 3 - Glyph Model and Selector Fidelity

### Goals

- Improve correctness of glyph precomputation and edge glyph ranking.
- Ensure font loading is robust and standards-compliant.

### Tasks

1. Fix font header signature handling for endian-safe comparisons.
2. Expand font fallback strategy:
   - system font discovery
   - project fallback font in `assets/`
3. Improve edge glyph subset derivation:
   - contrast + orientation peak constraints
   - avoid ambiguous symbols in edge subset
4. Upgrade orientation matching:
   - use cell-local orientation histogram where available
   - keep simple orientation mode as explicit fallback

### Files

- `src/glyph/font_loader.cpp`
- `src/glyph/glyph_cache.cpp`
- `src/glyph/glyph_stats.hpp`
- `src/mapping/char_selector.cpp`
- `assets/` (font asset)

### Acceptance criteria

- Valid TTF/OTF files load correctly on all supported platforms.
- Edge glyph selection quality improves in diagonal/line-heavy scenes.
- Selector behavior is deterministic for same input/config.

---

## Phase 4 - Rendering and Color Correctness

### Goals

- Ensure ANSI and truecolor output is standards-correct.
- Implement real block-art behavior (fg/bg paired rendering).

### Tasks

1. Fix ANSI16 bright color mapping (normal vs bright ranges).
2. Add terminal capability-aware color fallback path.
3. Extend `TerminalRenderer` diff logic:
   - include background color differences
   - minimize redundant color escape writes
4. Implement block-art mode:
   - half/full block glyph strategy
   - fg/bg color pairing for sub-cell approximation
5. Keep UTF-8 capability fallback for limited terminals.

### Files

- `src/terminal/terminal.cpp`
- `src/terminal/terminal.hpp`
- `src/render/terminal_renderer.cpp`
- `src/render/terminal_renderer.hpp`
- `src/mapping/color_mapper.cpp`

### Acceptance criteria

- ANSI16 colors are visually correct in standard terminals.
- Block-art mode is materially different from truecolor text mode.
- Render output has lower escape-sequence overhead under frame differencing.

---

## Phase 5 - Input, Output, and Audio Completion

### Goals

- Align media handling with documented source/output modes.
- Make audio behavior valid for real video inputs.

### Tasks

1. Expand `FrameSource` implementations:
   - `ImageSequenceSource`
   - `PipeSource` (stdin/raw frame mode)
2. Improve webcam source parsing:
   - multi-digit indices
   - explicit URI parsing
3. Clarify output mode matrix and implement missing paths:
   - image -> txt
   - video -> txt sequence/log (optional mode flag)
   - image -> encoded frame output (if requested)
4. Replace WAV-only audio assumption with decode pipeline suitable for media files.
5. Improve A/V sync and drift correction strategy.

### Files

- `src/core/frame_source.hpp`
- `src/core/frame_source.cpp`
- `src/audio/audio_player.cpp`
- `src/audio/audio_player.hpp`
- `src/main.cpp`
- `src/cli/args.cpp`
- `src/cli/args.hpp`

### Acceptance criteria

- Non-WAV video inputs do not falsely report "audio open" success/failure due to unsupported assumptions.
- Input/output behavior is consistent with CLI help text.
- Pipe and image sequence modes are test-covered and documented.

---

## Phase 6 - Platform and Build Portability

### Goals

- Make cross-platform claim true in practice.

### Tasks

1. Introduce terminal platform abstraction:
   - POSIX backend (current behavior)
   - Windows console backend (WinAPI + VT mode enable)
2. Replace Linux-centric build assumptions:
   - conditional package discovery for Windows/macOS
   - compiler-specific warning flags
3. Verify dependency handling per platform:
   - OpenCV
   - SDL2
   - FFmpeg

### Files

- `src/terminal/terminal.*` (or split into platform-specific translation units)
- `CMakeLists.txt`
- `tests/CMakeLists.txt`

### Acceptance criteria

- Project builds and runs on Linux/macOS/Windows with documented commands.
- No POSIX-only includes in Windows code paths.

---

## Phase 7 - Performance and Instrumentation

### Goals

- Quantify and optimize stage cost with hard numbers.

### Tasks

1. Add stage timing metrics:
   - decode/read
   - grayscale/resize
   - blur/sobel/nms/hysteresis
   - cell stats
   - selection
   - render/encode
2. Add lightweight profiling output mode.
3. Review OpenMP coverage and data locality in hot loops.
4. Add adaptive degradation policy if FPS target is missed.

### Files

- `src/main.cpp`
- `src/core/pipeline.cpp`
- `src/core/edge_detector.cpp`
- `src/core/cell_stats.cpp`

### Acceptance criteria

- Performance logs are available and parseable.
- Typical terminal workloads maintain target FPS or degrade gracefully.

---

## Phase 8 - Tests, CI, and Documentation Sync

### Goals

- Guarantee behavior with regression-resistant test coverage.
- Keep docs faithful to implementation.

### Tasks

1. Extend test pyramid:
   - unit: algorithms/utilities
   - integration: frame -> cells -> renderer decisions
   - end-to-end smoke: sample media in/out
2. Add golden tests:
   - deterministic frame snapshots for known seeds/configs
3. Add sanitizer and warnings gates in CI:
   - ASan/UBSan where supported
   - warnings-as-errors for core targets
4. Update docs to match shipped behavior:
   - `IMPLEMENTATION.md`
   - `PLAN.md` (only where spec intent evolved)
   - CLI help examples

### Files

- `tests/test_comprehensive.cpp`
- `tests/test_critical_fixes.cpp`
- new integration test files under `tests/`
- CI config (if present)
- `IMPLEMENTATION.md`
- `PLAN.md`

### Acceptance criteria

- CI green on required platforms/toolchains.
- Docs and runtime flags are consistent.
- Regression suite detects previously observed misalignment bugs.

---

## 6) Milestone Schedule (Suggested)

Assuming one primary engineer with review support.

- Milestone A (Week 1): Phase 0 + Phase 1
- Milestone B (Week 2): Phase 2
- Milestone C (Week 3): Phase 3 + Phase 4
- Milestone D (Week 4): Phase 5
- Milestone E (Week 5): Phase 6 + Phase 7
- Milestone F (Week 6): Phase 8 + release hardening

If multiple engineers are available, run Phases 3/4/6 in parallel after Phase 1 is merged.

---

## 7) PR Breakdown Strategy

Keep PRs small and verifiable:

1. PR-1: Runtime selector + temporal hysteresis integration
2. PR-2: Scaling semantics + char aspect correctness
3. PR-3: Font validation and glyph model upgrades
4. PR-4: ANSI/block-art renderer correctness
5. PR-5: FrameSource/audio/output mode completion
6. PR-6: Cross-platform terminal/build work
7. PR-7: Performance instrumentation
8. PR-8: Tests/CI/docs sync

Each PR must include:

- behavior summary
- test additions/updates
- before/after evidence (images/logs where relevant)

---

## 8) Risk Register

1. Cross-platform terminal behavior divergence
   - Mitigation: platform abstraction + targeted integration tests.
2. Performance regressions from correctness fixes
   - Mitigation: stage timers and perf gates in PR review.
3. FFmpeg/SDL integration complexity for audio decode/sync
   - Mitigation: isolate decode pipeline and test with fixed fixtures.
4. Spec drift while implementing optional features
   - Mitigation: tag optional features explicitly and keep defaults stable.

---

## 9) Definition of Done (Release Gate)

- All Phase acceptance criteria are met.
- No critical/high severity audit findings remain open.
- Docs are updated and consistent with shipped behavior.
- Test suite includes algorithm, integration, and e2e smoke coverage.
- Release build runs on target platforms with validated sample media.

---

## 10) Tracking Checklist

- [x] Phase 0 complete
- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Phase 3 complete
- [x] Phase 4 complete
- [x] Phase 5 complete
- [x] Phase 6 complete
- [x] Phase 7 complete (performance instrumentation - optional)
- [x] Phase 8 complete (additional testing/docs)
- [x] Release gate passed



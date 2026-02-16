# ASCII Animation Engine - Implementation Plan (Execution Contract)

Version: 2.0
Status: Execution plan aligned with `PLAN.md`

This plan translates the feature specification into a staged implementation strategy with explicit acceptance gates.

---

## 1) Implementation Principles

This plan enforces the following principles:

1. `PLAN.md` is normative for runtime behavior.
2. Scope is phased (`MVP`, `Quality`, `Advanced`) with v1 freeze.
3. Performance and quality gates are release-blocking.
4. Deterministic replay is a first-class requirement.
5. Config is schema-driven and versioned.
6. Text ASCII and block-art are separate pipelines.
7. Extension points are explicit and testable.
8. Platform support is contractual, not aspirational.
9. Instrumentation is required in core loops.
10. Release policy and CI gates are mandatory.

---

## 2) Tech Stack

| Component | Primary Choice | Notes |
|-----------|----------------|-------|
| Language | C++20 | Core implementation |
| Build System | CMake | Multi-platform build orchestration |
| Decode/Timing | FFmpeg | Single ownership for media timing |
| Optional image ops | OpenCV | Utility transforms where beneficial |
| Audio | FFmpeg decode + SDL2 output | Clear decode/output split |
| Font Rendering | stb_truetype.h | Vendored single-header |
| Parallelism | OpenMP (optional) | Row/tile parallel loops |
| Tests | CTest + custom golden tests | Unit/integration/perf layers |

---

## 3) Module Structure

```text
ASCII/
|-- CMakeLists.txt
|-- src/
|   |-- main.cpp
|   |-- core/
|   |   |-- types.hpp
|   |   |-- config.hpp/.cpp                 # schema + validation + migration
|   |   |-- frame_source.hpp/.cpp           # video/image/webcam/pipe abstractions
|   |   |-- color_space.hpp/.cpp            # sRGB <-> linear, Lab/OKLab conversion
|   |   |-- pipeline.hpp/.cpp               # end-to-end frame processing orchestration
|   |   |-- edge_detector.hpp/.cpp          # Sobel, NMS, adaptive hysteresis, multi-scale
|   |   |-- cell_stats.hpp/.cpp             # integral-image and tensor-based cell stats
|   |   |-- motion.hpp/.cpp                 # optical flow / motion compensation
|   |   `-- temporal.hpp/.cpp               # smoothing + transition-cost optimization
|   |-- glyph/
|   |   |-- font_loader.hpp/.cpp            # font loading and validation
|   |   |-- glyph_cache.hpp/.cpp            # glyph raster cache and precompute
|   |   |-- glyph_stats.hpp                 # brightness/contrast/orientation features
|   |   `-- char_sets.hpp/.cpp              # set definitions + metadata
|   |-- mapping/
|   |   |-- char_selector.hpp/.cpp          # data loss + temporal transition objective
|   |   `-- color_mapper.hpp/.cpp           # perceptual quantization logic
|   |-- render/
|   |   |-- terminal_renderer.hpp/.cpp      # diff output for text ASCII
|   |   |-- block_renderer.hpp/.cpp         # dedicated block-art renderer
|   |   |-- bitmap_renderer.hpp/.cpp        # offscreen RGBA frame rendering
|   |   `-- video_encoder.hpp/.cpp          # encoded output path
|   |-- terminal/
|   |   |-- terminal.hpp                    # backend-agnostic terminal interface
|   |   |-- terminal_posix.cpp
|   |   `-- terminal_windows.cpp
|   |-- audio/
|   |   `-- audio_player.hpp/.cpp           # buffered playback + sync control
|   `-- cli/
|       `-- args.hpp/.cpp                   # CLI parse + schema mapping
|-- assets/
|   `-- fallback.ttf
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- golden/
|   `-- perf/
`-- vendor/
    `-- stb_truetype.h
```

---

## 4) Algorithm-to-Module Mapping

The 10 required algorithm upgrades map as follows:

1. Linear-light processing
- `core/color_space.*`, `core/pipeline.*`

2. Adaptive edge thresholds
- `core/edge_detector.*`

3. Structure tensor per cell
- `core/cell_stats.*`

4. Unified reconstruction loss
- `mapping/char_selector.*`

5. Multi-scale edge analysis
- `core/edge_detector.*`, `core/pipeline.*`

6. Temporal state-transition cost
- `core/temporal.*`, `mapping/char_selector.*`

7. Motion compensation
- `core/motion.*`, `core/temporal.*`

8. Perceptual color distance (Lab/OKLab)
- `core/color_space.*`, `mapping/color_mapper.*`

9. Error diffusion dithering
- `mapping/char_selector.*` (decision hints), `render/*renderer*.*`

10. Integral-image acceleration
- `core/cell_stats.*`

---

## 5) Core Data Contracts

### 5.1 Frame Domain

- Internal scalar processing uses float arrays in `[0, 1]` in linear-light.
- Render domain uses RGBA8.

### 5.2 Cell Feature Contract

Every cell exports:

- mean luminance
- luminance variance/stddev
- edge strength (mean + max)
- edge occupancy
- orientation histogram
- tensor coherence
- mean color (linear RGB + perceptual vector)

### 5.3 Selection Contract

For each candidate glyph `g` and cell `c`:

- `L_data(c, g) = w1 * brightness_error + w2 * orientation_hist_distance + w3 * contrast_error`
- Temporal decision adds transition cost from previous glyph state.

---

## 6) Pipeline Contract (Execution Order)

1. Decode frame.
2. Convert color to linear-light.
3. Apply scaling transform (`fit`/`fill`/`stretch`) with char aspect.
4. Build luminance plane.
5. Blur.
6. Multi-scale Sobel.
7. Magnitude/orientation.
8. NMS.
9. Adaptive hysteresis.
10. Integral-image prep.
11. Cell stats + structure tensor.
12. Glyph scoring and selection.
13. Temporal optimization and hysteresis.
14. Color mapping using perceptual quantization.
15. Render via selected renderer pipeline.

---

## 7) Scaling and Geometry Requirements

- Output grid is always exactly target `cols x rows`.
- `fit` must letterbox/pillarbox.
- `fill` must crop.
- `stretch` must warp without preserving source aspect.
- Geometry mapping must be deterministic and unit tested.

---

## 8) Renderer Separation

### 8.1 Text ASCII Renderer

- Primary glyph set output.
- Foreground color required.
- Background optional.

### 8.2 Block-Art Renderer

- Distinct code path and state.
- Uses block primitives and both fg/bg to approximate local detail.

Both renderers must support frame differencing and ANSI minimization.

---

## 9) Configuration and CLI

### 9.1 Schema

- Add versioned schema with defaults, validation, and migration behavior.
- CLI must map into schema cleanly.

### 9.2 Required config keys

- input/output mode
- grid and scaling
- cell geometry
- edge/adaptive controls
- temporal controls
- selector mode
- color mode
- deterministic replay path
- debug and profiling toggles

---

## 10) Determinism, Replay, and Observability

### 10.1 Determinism

- Same input + config + build must produce same cell decisions.

### 10.2 Replay artifact

- Persist frame index + chosen glyph/color grid + config hash.

### 10.3 Runtime instrumentation

Collect stage timings and counters for:

- decode
- preprocess
- edges
- stats
- selection
- temporal
- render
- encode

---

## 11) Platform Strategy

### 11.1 Terminal abstraction

- Common interface in `terminal.hpp`.
- POSIX and Windows backends isolated in separate translation units.

### 11.2 Build portability

- CMake must use platform-conditional dependency and compile flag handling.
- Avoid POSIX-only includes in shared headers.

### 11.3 Compatibility matrix

Track tested support for:

- UTF-8
- ANSI 16/256
- truecolor
- cursor controls

---

## 12) Performance Strategy

- Use contiguous row-major buffers.
- Use integral images for per-cell metrics.
- Parallelize safe loops (grayscale, blur, sobel, stats, selection).
- Keep terminal write path single-threaded.
- Add adaptive quality degradation policy when FPS gate is missed.

---

## 13) Test Strategy

### 13.1 Unit tests

- color transforms
- edge detector math
- structure tensor and stats
- selection loss
- temporal transitions

### 13.2 Integration tests

- frame -> cell -> glyph -> render buffer pipeline
- scaling mode correctness
- color quantization correctness

### 13.3 Golden tests

- deterministic frame snapshots against known fixtures

### 13.4 Perf regression tests

- enforce gate thresholds from `PLAN.md`

---

## 14) Delivery Phases

## Phase 0 - Foundations

- Add schema system, deterministic replay scaffolding, timing instrumentation.
- Acceptance: schema parse/validate tests pass; replay skeleton writes deterministic metadata.

## Phase 1 - MVP Core Pipeline

- Implement linear-light path, scaling contract, edge detector with adaptive thresholds, cell stats with integral images.
- Acceptance: unit tests pass; visual baseline generated.

## Phase 2 - Selection + Temporal

- Implement unified loss, temporal transition costs, hysteresis.
- Acceptance: flicker gate improves on static scenes; deterministic golden tests pass.

## Phase 3 - Color + Rendering

- Implement perceptual quantization and text renderer diff correctness.
- Acceptance: color correctness tests pass across modes.

## Phase 4 - Block-Art and Offline Output

- Implement dedicated block-art renderer and encoded output stabilization.
- Acceptance: separate renderer tests pass; no regression in text mode.

## Phase 5 - Platform Hardening

- Finalize Windows backend and compatibility matrix.
- Acceptance: matrix updated with tested environments.

## Phase 6 - Release Readiness

- Full CI, perf gates, docs sync, release checklist.
- Acceptance: all gates green; no critical open defects.

---

## 15) PR Strategy

Use small, verifiable PRs with strict scope:

1. schema + determinism
2. pipeline geometry + linear-light
3. adaptive edges + stats acceleration
4. selector objective + temporal transitions
5. perceptual color + renderer correctness
6. block-art pipeline
7. platform abstraction
8. perf + CI + docs sync

Each PR must include:

- behavior delta summary
- test evidence
- before/after perf or visual evidence when relevant

---

## 16) Release Gate Checklist

- [ ] Feature requirements from `PLAN.md` met for target tier.
- [ ] Determinism checks pass.
- [ ] Quality/perf gates pass.
- [ ] Compatibility matrix updated.
- [ ] Unit/integration/golden/perf tests green.
- [ ] Documentation and implementation are in sync.

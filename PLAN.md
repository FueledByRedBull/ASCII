# ASCII Animation Engine (Non-ML) - Normative Feature Specification

Version: 2.0
Status: Normative
Audience: Core contributors and reviewers

This document defines required behavior. If implementation conflicts with this document, this document wins unless explicitly revised.

***

## 1. Product Goal

Build a high-quality, deterministic ASCII animation engine in C++ using only classical (non-ML) image processing, suitable for real-time terminal playback and offline rendering.

***

## 2. Scope Tiers and v1 Freeze

### 2.1 MVP (v1 Required)

- Input: local video files and local image files.
- Output: terminal playback and text-frame export.
- Core algorithm: edge-aware glyph selection with temporal stability.
- Color: none, ANSI 16, ANSI 256, truecolor.
- Platforms: Linux, macOS, Windows.
- Deterministic replay support.

### 2.2 Quality Tier (v1.x)

- Webcam input.
- Video encoding output.
- Block-art rendering mode as a separate rendering pipeline.
- Advanced diagnostics and profiling overlays.

### 2.3 Advanced Tier (v2)

- Pipe/stdin raw frame mode.
- Motion-compensated temporal stabilization refinements.
- Additional glyph model packs and extension plugins.

### 2.4 v1 Frozen Out-of-Scope

- ML models, OCR, learned glyph ranking.
- Arbitrary remote streams/protocol stacks.
- Fully interactive editor-like UI.

***

## 3. Non-Goals

- Replacing full video player UX ecosystems.
- Supporting every terminal quirk with custom per-terminal hacks.
- Perfect photorealistic reconstruction.

***

## 4. Quality and Performance Gates (Release Blocking)

These are mandatory release gates for v1:

- FPS floor: >= 24 FPS at 120x40 grid on reference hardware.
- End-to-end frame latency: <= 42 ms at 24 FPS target.
- Flicker gate: <= 8 percent cell flips per frame on static camera scenes after warm-up.
- Memory gate: <= 512 MB resident memory under v1 default settings.
- Startup gate: <= 2 seconds to first rendered frame for local video input.

***

## 5. Determinism and Replay Contract

- Same input bytes + same config + same build profile must produce same per-frame cell decisions.
- Randomized behavior is forbidden in core mapping path.
- A deterministic replay artifact must be supported:
  - stores config hash
  - stores frame index
  - stores selected glyph and color per cell
- Determinism exceptions must be explicitly documented (for example, platform terminal color fallback differences).

***

## 6. Media and Input Handling

### 6.1 Required Inputs

- Video file source.
- Single image source.

### 6.2 Optional Inputs (non-v1)

- Webcam source.
- Image sequence source.
- Pipe/raw frame source.

### 6.3 FrameSource Interface

Each source must implement:

- `bool open(const std::string& uri)`
- `bool read(Frame& out)`
- `double fps() const`
- `Size frame_size() const`
- `void reset()`

### 6.4 Media Stack Requirement

Use one primary media stack for decode, timing, and audio sync ownership to reduce split responsibility and drift.

***

## 7. Terminal Geometry and Scaling Semantics

Let:

- `Wc`, `Hc` be terminal columns and rows.
- `char_aspect = char_height / char_width`.
- `Ws`, `Hs` be source image size.

### 7.1 Logical Target

- Logical output grid is exactly `Wc x Hc`.
- Cell-to-source mapping must be fully defined for each scaling mode.

### 7.2 Scaling Modes (Required Semantics)

- `fit`: preserve source aspect; letterbox/pillarbox if needed.
- `fill`: preserve source aspect; crop overflow to fill full grid.
- `stretch`: ignore source aspect; map directly to full grid.

### 7.3 Char Aspect Application

- Char aspect correction must be applied in transform math, not by ad-hoc visual tweaks.
- The transform path must be unit tested for portrait, landscape, and square inputs.

***

## 8. Core Algorithmic Pipeline

Processing order per frame:

1. Decode and normalize input frame.
2. Convert sRGB to linear-light.
3. Resize/crop according to scaling mode and char aspect.
4. Convert to luminance in linear-light.
5. Apply Gaussian blur (small sigma).
6. Compute multi-scale gradients (fine and coarse).
7. Run Sobel, magnitude, orientation.
8. Apply non-maximum suppression.
9. Apply adaptive hysteresis thresholding.
10. Aggregate per-cell statistics.
11. Score/select glyph per cell.
12. Apply temporal optimization.
13. Map color and render.
14. Convert linear-light color back to output color space.

***

## 9. Algorithmic Requirements (Must-Have)

All 10 requirements below are mandatory in the design baseline.

1. Linear-light processing
- Core luminance, edges, and blending decisions happen in linear-light space.

2. Adaptive edge thresholds
- Thresholds are dynamic (per-frame or per-tile percentile/Otsu style), not fixed constants only.

3. Structure tensor per cell
- Cell orientation and coherence derive from structure tensor, not only mean `Gx/Gy`.

4. Unified reconstruction loss for glyph choice
- Candidate glyphs scored with:
  - `L = w1 * brightness_error + w2 * orientation_hist_distance + w3 * contrast_error`
- Lower `L` is better.

5. Multi-scale edge analysis
- Use at least two scales so thin and broad structures are preserved.

6. Temporal state-transition cost
- Glyph updates use transition cost (for example Viterbi-style dynamic scoring), not only local threshold checks.

7. Motion compensation
- Temporal smoothing follows estimated local motion (classical optical flow), not fixed screen cells only.

8. Perceptual color distance
- Color quantization uses Lab/OKLab distance, not raw RGB euclidean distance.

9. Dithering for fill regions
- Apply light error diffusion in fill/background regions to reduce banding.

10. Integral-image acceleration
- Use integral images (or equivalent summed-area structures) for fast per-cell mean/variance style metrics.

***

## 10. Per-Cell Statistics Contract

For each cell, compute at minimum:

- Mean luminance.
- Luminance variance (or stddev).
- Edge strength (mean and max).
- Edge occupancy fraction.
- Orientation histogram.
- Structure tensor eigenvalue ratio (coherence).
- Mean color (in linear-light RGB and perceptual color space representation).

***

## 11. Glyph Modeling Contract

At startup or precompute stage, for each glyph:

- Brightness/density metric.
- Contrast metric.
- Orientation histogram.
- Optional edge suitability score.

Glyph sets:

- Basic ASCII ramp.
- Extended block ramp.
- Line-art set.

Each set must provide:

- Brightness ordering.
- Edge subset metadata.

***

## 12. Runtime Selection Contract

### 12.1 Edge vs Fill Partition

Compute edge dominance score using occupancy, strength, and coherence.

### 12.2 Selection Objective

Select glyph `g` minimizing:

- `L_data(cell, g)` from Section 9.4.

### 12.3 Temporal Objective

Select sequence minimizing:

- `L_total = L_data + lambda * L_transition(prev_glyph, new_glyph)`

### 12.4 Mode Support

- Simple orientation mode (fallback).
- Histogram matching mode (default).

***

## 13. Temporal Coherence and Anti-Flicker

Required per-cell state:

- Previous chosen glyph.
- Previous confidence/score.
- Previous luminance and edge features.
- Motion-compensated source lookup offset.

Required behavior:

- Exponential feature smoothing with configurable alpha.
- Glyph transition penalty to prevent jitter.
- Edge hysteresis with separate enter/exit thresholds.

***

## 14. Rendering Pipelines (Separate, Not Blended)

### 14.1 Text ASCII Pipeline

- Character-centric output.
- Foreground color required.
- Background color optional.

### 14.2 Block-Art Pipeline

- Block glyph primitives (`full`, `upper half`, `lower half`, etc.).
- Foreground and background color jointly optimized.
- Distinct code path from text ASCII mode.

### 14.3 Frame Differencing

- Compare full cell state (`glyph`, `fg`, `bg`).
- Emit minimal ANSI runs for changed regions.

***

## 15. Color Mapping Requirements

- Quantization targets: none, 16, 256, truecolor.
- Quantization distance metric: perceptual (Lab/OKLab).
- Capability fallback path must be explicit.
- Color mapping and glyph mapping are coupled but independently testable.

***

## 16. Configuration Model

A versioned config schema is required.

Minimum schema fields:

- input mode/source
- output mode/target
- grid size and scaling mode
- char aspect and cell size
- edge thresholds and adaptive mode
- temporal parameters
- glyph set and selector mode
- color mode
- debug/profiling toggles

Schema must include version key, defaults, validation ranges, and migration behavior.

***

## 17. CLI and UX Contract

CLI must expose, at minimum:

- input and output selection
- FPS target
- cols/rows
- char set
- color mode
- edge sensitivity and blur strength
- temporal controls
- scaling mode
- deterministic replay output

Interactive controls are optional for v1.

***

## 18. Extension Model

Extension points must be documented and stable:

- add new `FrameSource`
- add new glyph set
- add new color quantizer
- add new renderer backend

Core contracts must not require touching unrelated modules when adding an extension.

***

## 19. Platform Compatibility Matrix (Required)

The project must maintain a tested compatibility table for:

- Linux terminals (xterm, tmux, etc.)
- macOS terminals
- Windows terminals (Windows Terminal, modern ConHost with VT)

Matrix fields:

- ANSI support
- UTF-8 support
- truecolor support
- known caveats

***

## 20. Observability Requirements

Required runtime instrumentation:

- stage timing: decode, preprocess, edge, stats, select, render, encode
- frame time and rolling FPS
- dropped/late frame counters
- optional per-cell decision debug trace

***

## 21. Testing and CI Gates

Required test layers:

- Unit tests: math/algorithm primitives.
- Integration tests: frame -> cell decisions -> rendered buffers.
- Golden tests: deterministic frame snapshots.
- Performance regression tests: compare against baseline thresholds.

CI must fail on:

- correctness regressions
- determinism regressions
- performance gate regressions beyond configured tolerance

***

## 22. Release Policy

Before release tag:

- All quality gates in Section 4 pass.
- Determinism checks pass.
- Compatibility matrix updated.
- Documentation and implementation are in sync.
- No open critical defects in core pipeline.

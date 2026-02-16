# ASCII Engine v2.0 - Implementation Roadmap

## Overview

This roadmap implements the PLAN.md v2.0 specification based on decisions in QUESTIONS.md.

---

## Module Additions Required

### New Modules
| Module | Purpose |
|--------|---------|
| `core/config.*` | TOML schema, validation, precedence |
| `core/color_space.*` | sRGB EOTF/OETF, OKLab transforms |
| `core/motion.*` | Farneback optical flow hook |
| `render/block_renderer.*` | Dedicated block-art pipeline |
| `core/replay.*` | Binary .areplay with zstd |

### Modified Modules
| Module | Changes |
|--------|---------|
| `core/frame_source.*` | FFmpeg primary, OpenCV optional |
| `core/pipeline.*` | Linear-light path, multi-scale |
| `core/edge_detector.*` | Adaptive thresholds, 2-scale fusion |
| `core/cell_stats.*` | Structure tensor, integral images |
| `core/temporal.*` | Motion-aware offsets, transition cost |
| `mapping/char_selector.*` | Unified loss function |
| `mapping/color_mapper.*` | OKLab distance |
| `render/terminal_renderer.*` | Dithering integration |
| `terminal/terminal.*` | Platform abstraction complete |
| `cli/args.*` | Config file integration |

---

## Phased Implementation

## Phase 1 - Foundation (PR 1)

### Goals
- TOML config system
- FFmpeg decode ownership
- Determinism scaffolding

### Tasks
1. Create `core/config.*` with TOML parsing (toml++ header)
2. Implement config precedence: defaults < file < CLI
3. Platform-specific config paths
4. Switch frame_source to FFmpeg primary
5. Add OpenCV optional build flag
6. Create `core/replay.*` with zstd framing
7. Add config hash to replay metadata

### Acceptance
- [ ] Config file loads and validates
- [ ] CLI overrides config correctly
- [ ] FFmpeg decode works without OpenCV
- [ ] Replay file writes with config hash
- [ ] Build succeeds with `-DASCII_USE_OPENCV=OFF`

---

## Phase 2 - Color Space Pipeline (PR 2)

### Goals
- Linear-light processing
- OKLab perceptual space

### Tasks
1. Create `core/color_space.*`
2. Implement sRGB -> linear LUT (decode)
3. Implement linear -> sRGB LUT (encode)
4. Implement OKLab forward transform
5. Implement OKLab distance function
6. Add OKLab inverse utility
7. Update pipeline to process in linear-light
8. Update color_mapper to use OKLab distance

### Acceptance
- [ ] sRGB <-> linear roundtrip within tolerance
- [ ] OKLab distance matches reference
- [ ] Pipeline operates in linear-light domain
- [ ] Color quantization uses perceptual distance

---

## Phase 3 - Edge Detection Upgrade (PR 3)

### Goals
- Multi-scale edges
- Adaptive thresholds
- Integral images

### Tasks
1. Add integral image computation to `cell_stats.*`
2. Implement 2-scale Sobel (sigma 0.8, 1.6)
3. Add scale fusion logic (weighted magnitude, best orientation)
4. Implement global percentile threshold
5. Implement tile-local adaptive mode
6. Add hybrid mode with dark-scene fallback
7. Update cell_stats to use integral images for mean/variance

### Acceptance
- [ ] Integral images produce correct sums
- [ ] 2-scale fusion outputs expected values
- [ ] Adaptive thresholds respond to content
- [ ] Dark frames use fallback correctly

---

## Phase 4 - Structure Tensor & Unified Loss (PR 4)

### Goals
- Structure tensor per cell
- Unified reconstruction loss
- Transition cost optimization

### Tasks
1. Implement 2x2 structure tensor per cell in `cell_stats.*`
2. Add coherence computation (eigenvalue ratio)
3. Add dominant orientation from tensor
4. Create unified loss in `char_selector.*`:
   - `L = 0.45*brightness_err + 0.40*orientation_dist + 0.15*contrast_err`
5. Add user-configurable weights via config
6. Implement glyph transition cost
7. Update temporal to use transition cost

### Acceptance
- [ ] Structure tensor coherence in [0,1]
- [ ] Orientation matches gradient direction
- [ ] Unified loss ranking is deterministic
- [ ] Transition cost reduces flicker on test scene

---

## Phase 5 - Motion & Dithering (PR 5)

### Goals
- Motion compensation hook
- Floyd-Steinberg dithering

### Tasks
1. Create `core/motion.*`
2. Implement Farneback optical flow (deterministic params)
3. Add motion vector cap (6px)
4. Create motion-aware temporal offset hook
5. Implement serpentine Floyd-Steinberg in renderer
6. Add dithering mask (skip edge cells)
7. Clamp error propagation to +/-0.12

### Acceptance
- [ ] Flow field has expected magnitude/direction
- [ ] Motion vectors capped at 6px
- [ ] Dithering applied only in fill regions
- [ ] No visible shimmer on static content

---

## Phase 6 - Block-Art Renderer (PR 6)

### Goals
- Dedicated block-art pipeline
- FG/BG joint optimization

### Tasks
1. Create `render/block_renderer.*`
2. Implement block glyph selection (full, upper, lower, quadrants)
3. Implement fg/bg color pairing optimization
4. Add block-art specific frame differencing
5. Wire block-art mode in main.cpp
6. Update CLI to expose block-art mode

### Acceptance
- [ ] Block-art uses separate renderer path
- [ ] FG/BG jointly approximate cell color
- [ ] Frame differencing includes bg changes

---

## Phase 7 - Platform & CI (PR 7)

### Goals
- Complete platform abstraction
- GitHub Actions CI
- Compatibility matrix

### Tasks
1. Finalize `terminal_windows.cpp` with VT enablement
2. Add Windows 11 version check
3. Create GitHub Actions workflow
4. Build matrix: Linux/gcc, macOS/clang, Windows/msvc
5. Add perf gate job
6. Create compatibility matrix documentation
7. Test on required terminals

### Acceptance
- [ ] Windows Terminal works correctly
- [ ] CI passes on all platforms
- [ ] Compatibility matrix populated
- [ ] Perf gates enforced in CI

---

## Phase 8 - Testing & Release (PR 8)

### Goals
- Golden tests
- Perf regression
- Release artifacts

### Tasks
1. Add golden test fixtures (small, in-repo)
2. Create golden test runner
3. Add `--update-golden` workflow
4. Implement perf regression tests
5. Create release artifact builds
6. Update documentation
7. Run full compatibility matrix

### Acceptance
- [ ] Golden tests pass
- [ ] Perf gates pass on reference hardware
- [ ] Release artifacts build
- [ ] Docs match implementation

---

## Dependency Changes

### Add
- `toml++` (header-only) - TOML parsing
- `zstd` - Replay compression
- FFmpeg libs (avcodec, avformat, avutil, swscale, swresample)

### Make Optional
- OpenCV (image transforms only, not decode)

### Update Minimums
- FFmpeg >= 6.0
- SDL2 >= 2.26
- OpenCV >= 4.8 (optional)

---

## Build System Changes

```cmake
option(ASCII_USE_OPENCV "Enable OpenCV utility paths" OFF)

# Core deps (required)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED 
    libavcodec>=60
    libavformat>=60
    libavutil>=58
    libswscale>=7
    libswresample>=4
)
pkg_check_modules(SDL2 REQUIRED sdl2>=2.26)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Optional
if(ASCII_USE_OPENCV)
    find_package(OpenCV 4.8 REQUIRED)
endif()
```

---

## Quality Gates

| Gate | Threshold |
|------|-----------|
| FPS floor | >= 24 FPS at 120x40 |
| Frame latency | <= 42ms |
| Flicker | <= 8% cell flips |
| Memory | <= 512MB resident |
| Startup | <= 2s to first frame |

---

## Tracking

- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Phase 3 complete
- [x] Phase 4 complete
- [x] Phase 5 complete
- [x] Phase 6 complete
- [ ] Phase 7 complete
- [ ] Phase 8 complete
- [ ] All gates pass

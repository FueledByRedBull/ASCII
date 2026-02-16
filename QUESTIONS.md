# ASCII Engine v2.0 - Open Questions

This document captures all outstanding questions before implementing the PLAN.md v2.0 specification.

---

## Architecture Questions

### Q1. FFmpeg vs OpenCV
The spec says FFmpeg for decode/timing, but OpenCV is "optional utility." Currently OpenCV handles video decode, resize, and image loading. Do you want to:
- Replace all OpenCV decode with FFmpeg?
- Keep OpenCV for resize/image ops?
- What about build complexity (FFmpeg is harder to link than OpenCV)?

Answer:
- Use FFmpeg as the only media decode/timing owner (video + audio + timestamps).
- Keep OpenCV as optional utility for image transforms/prototyping only, not decode ownership.
- Build policy:
  - `-DASCII_USE_OPENCV=ON` enables OpenCV utility paths.
  - Core build must succeed with FFmpeg + SDL2 only.
  - No dual-clock decode paths in runtime.

### Q2. Config Schema
- What format? (JSON, TOML, YAML, custom)
- Should CLI args override config file, or vice versa?
- Where does the config file live? (`~/.ascii-engine/config.toml`? `--config path`?)

Answer:
- Use TOML (`config_version` required).
- Precedence: built-in defaults < config file < CLI flags.
- Config locations:
  - Linux: `~/.config/ascii-engine/config.toml`
  - macOS: `~/Library/Application Support/ascii-engine/config.toml`
  - Windows: `%APPDATA%\\ascii-engine\\config.toml`
- Always support `--config <path>` override.

### Q3. Replay Artifact Format
- Binary or text?
- Should it be human-readable for debugging?
- How large is acceptable for a 5-minute video at 30fps?

Answer:
- Use binary replay (`.areplay`) with zstd compression and delta framing.
- Include optional JSON sidecar for debug metadata only (config hash, dimensions, frame count).
- Size target for 5 min @ 30 fps:
  - Acceptable: <= 150 MB
  - Target: <= 75 MB on typical content

---

## Algorithm Questions

### Q4. Linear-Light
Do you want full gamma-correct pipeline (2.2 decode, process, 2.2 encode) or simplified linear approximation?

Answer:
- Use full sRGB EOTF/OETF conversions (not fixed 2.2 approximation) through core metrics.
- Implement LUT-based conversion for speed and determinism.

### Q5. Adaptive Thresholds
- Per-frame Otsu?
- Per-tile local?
- Both with configurable mode?
- What's the fallback for uniform/dark frames?

Answer:
- Support both global and tile-local adaptive modes.
- Default mode: global percentile + local clamp (hybrid).
- Fallback for low-variance frames: static minimum thresholds with dark-scene floor.

### Q6. Structure Tensor
- 2x2 tensor per cell?
- What eigenvalue ratio formula for coherence?
- Do you want anisotropy direction too?

Answer:
- Yes: 2x2 tensor per cell (`Jxx, Jxy, Jyy`).
- Coherence formula: `(lambda1 - lambda2) / (lambda1 + lambda2 + eps)`.
- Also expose dominant orientation: `theta = 0.5 * atan2(2*Jxy, Jxx - Jyy)`.

### Q7. Unified Loss Weights
`w1, w2, w3` for brightness/orientation/contrast - are these:
- Fixed constants?
- User-configurable via CLI/config?
- Auto-tuned per-frame?

Answer:
- Provide fixed defaults plus user-config overrides.
- Default weights: `w_brightness=0.45`, `w_orientation=0.40`, `w_contrast=0.15`.
- No per-frame auto-tune in v1 (keeps determinism simpler and testable).

### Q8. Multi-Scale Edges
- How many scales? (2, 3, 4?)
- What sigma values?
- Do you blend results or select best per-cell?

Answer:
- v1: 2 scales.
- Default sigmas: `0.8` and `1.6`.
- Fuse by weighted blend for magnitude; orientation taken from highest-confidence scale.
- v1.x can add optional third scale (`2.4`) for low-detail scenes.

### Q9. Motion Compensation
- What optical flow algorithm? (block matching, Lucas-Kanade, Farneback?)
- What's the max motion vector budget per frame?
- Is this truly v1 or v2?

Answer:
- Algorithm: dense Farneback (deterministic parameters), coarse-to-fine.
- Cap motion vectors at 6 px/frame in processed domain for stability.
- Scope split:
  - v1: minimal motion-aware temporal offset hook.
  - v2: full motion-compensated temporal optimization.

### Q10. Perceptual Color
- Lab or OKLab? (OKLab is newer/better but less tested)
- Do we need inverse transforms or just distance computation?

Answer:
- Use OKLab as default perceptual space.
- Implement forward transforms and distance in v1.
- Keep inverse utility available for future blend/reconstruction paths.

### Q11. Dithering
- Floyd-Steinberg? Ordered dither?
- Only in fill regions or edges too?
- What's the error budget?

Answer:
- Use serpentine Floyd-Steinberg.
- Apply only in fill/background regions; disable on edge-dominant cells.
- Clamp propagated error to +/-0.12 (normalized luminance) to avoid shimmer.

### Q12. Integral Images
- For which metrics? Mean, variance, both?
- What about edge strength aggregation?
- Memory budget for multi-channel integral images?

Answer:
- Use integral images for mean + variance (luma and luma^2), plus edge magnitude sums.
- Keep gradient orientation histogram accumulation separate (not integral-image based).
- Budget target for integral buffers: <= 128 MB worst-case resident.

---

## Scope & Prioritization Questions

### Q13. v1 vs v1.x Algorithm Line
Which of the 10 mandatory algorithms are truly blocking v1? Can we ship v1 with:

| Algorithm | Include in v1? |
|-----------|----------------|
| Linear-light processing | YES/NO |
| Adaptive thresholds | YES/NO |
| Structure tensor | YES/NO |
| Unified loss function | YES/NO |
| Multi-scale edges | YES/NO |
| Motion compensation | YES/NO |
| Perceptual color (Lab/OKLab) | YES/NO |
| Error diffusion dithering | YES/NO |
| Integral images | YES/NO |

Answer:

| Algorithm | Include in v1? |
|-----------|----------------|
| Linear-light processing | YES |
| Adaptive thresholds | YES |
| Structure tensor | YES |
| Unified loss function | YES |
| Multi-scale edges | YES |
| Motion compensation | YES (minimal hook) |
| Perceptual color (Lab/OKLab) | YES |
| Error diffusion dithering | YES |
| Integral images | YES |

Rule: v1 ships with all 10 in baseline form; v1.x/v2 deepen quality and performance.

### Q14. Block-Art Pipeline
- Is this truly separate code or just a different selector mode?
- Do you want two completely independent renderers or shared infrastructure with mode switch?

Answer:
- Treat block-art as a separate renderer pipeline (per spec), not only a selector toggle.
- Share common upstream buffers/stats infrastructure, but keep selection + render logic distinct.

### Q15. Webcam in v1.x
- What's the expected latency budget?
- What about webcam format/resolution auto-negotiation?
- Multi-camera support?

Answer:
- Latency budget: <= 120 ms glass-to-terminal at default grid.
- Auto-negotiation: select closest supported mode to requested fps/resolution, then downscale internally.
- Multi-camera: not required in v1.x; single-camera + explicit index required.

---

## Platform Questions

### Q16. Windows Terminal Support
- Windows Terminal, ConHost, both?
- What about older cmd.exe (no VT support)?
- What's the minimum Windows version? (10? 11?)

Answer:
- Support Windows Terminal and modern ConHost with VT enabled.
- Older non-VT `cmd.exe` is best-effort (degraded/no-color), not release-blocking.
- Minimum supported OS: Windows 11.

### Q17. Compatibility Matrix Granularity
How many terminals to test?
- Just major ones (xterm, tmux, Windows Terminal, iTerm2, Alacritty)?
- Broader coverage?

Answer:
- v1 release matrix: major terminals only.
- Required set:
  - Linux: xterm, tmux, Alacritty, Kitty
  - macOS: iTerm2, Terminal.app
  - Windows: Windows Terminal, modern ConHost
- Broader matrix can be community-validated post-v1.

### Q18. Audio Sync
- FFmpeg decode + SDL2 output: who owns the clock?
- What drift tolerance is acceptable?
- How to handle video-only inputs?

Answer:
- Realtime playback: audio clock is master.
- Drift policy:
  - Soft correction threshold: 40 ms
  - Hard seek threshold: 100 ms
- Video-only inputs: video clock owns pacing; no synthetic audio clock.

---

## Testing Questions

### Q19. Golden Test Fixtures
- Where do these live? In repo (size concern) or separate download?
- What's the baseline update policy?
- Who owns baseline regeneration?

Answer:
- Keep small core fixtures in repo; large fixture packs as versioned external download.
- Baseline updates only via explicit `--update-golden` workflow in dedicated PRs.
- Ownership: core maintainers (at least one approver required for golden updates).

### Q20. Performance Baseline
- What's "reference hardware" for the 24 FPS gate?
- CPU specs?
- RAM?
- SSD required?

Answer:
- Reference hardware (minimum):
  - 6-core modern x86 CPU (Zen 2 / Intel 10th-gen class or newer)
  - 16 GB RAM
  - SSD storage
- Gates in `PLAN.md` are measured on this baseline profile.

### Q21. Determinism Scope
- Floating-point order can vary across compilers/optimization.
- Is bit-exact required or just visual-exact?
- What about across platforms (x86 vs ARM)?

Answer:
- Same platform + same compiler profile: deterministic cell decisions required.
- Cross-platform: visual-equivalent with tolerance; bit-exact not required.
- Replay hash should include platform/compiler metadata.

---

## Build & CI Questions

### Q22. Dependency Version Policy
- Minimum FFmpeg version?
- Minimum OpenCV version if keeping it?
- How to handle ABI breaks?

Answer:
- Minimums:
  - FFmpeg >= 6.0
  - OpenCV >= 4.8 (optional path only)
  - SDL2 >= 2.26
- Pin and test exact versions in CI matrix; wrap external APIs behind thin adapters to absorb ABI changes.

### Q23. CI Platforms
- GitHub Actions?
- GitLab CI?
- Both?
- What about macOS/Windows runners (cost/availability)?

Answer:
- Primary CI: GitHub Actions (Linux, Windows, macOS).
- GitLab CI optional later if needed for enterprise mirroring.
- Keep default matrix lean; nightly jobs run heavier perf/golden suites.

### Q24. Release Artifacts
- Static binaries?
- Dynamic with bundled deps?
- Package managers (apt, brew, chocolatey, scoop)?

Answer:
- Windows/macOS: dynamic binaries with bundled runtime deps where legally allowed.
- Linux: portable tarball + distro-specific packages later.
- Package manager targets after v1: Homebrew, Scoop/Chocolatey.

---

## UX Questions

### Q25. Interactive Controls
The spec says optional for v1. Are you:
- Keeping the current keyboard controls (space, q, c, +/-)?
- Removing them for v1?

Answer:
- Keep current interactive controls for terminal mode.
- Treat controls as optional UX layer; must not affect deterministic offline outputs.

### Q26. Error Messages
- How verbose?
- Structured (JSON) for scripting or human-readable?
- Localization plans?

Answer:
- Default: concise human-readable errors.
- Add `--errors=json` for scripting/CI integration.
- No localization in v1.

### Q27. Profiling Output
- Per-frame timing to stderr?
- Summary at end?
- Both?
- What format? (CSV, JSON, plain text?)

Answer:
- Both:
  - Optional per-frame stream (`--profile-live`) as JSONL
  - End-of-run summary default in plain text
- Optional CSV export for spreadsheet analysis.

---

## Additional Questions

### Q28. Memory Budget
Spec says 512MB resident. Is this:
- Hard limit (abort if exceeded)?
- Soft target (graceful degradation)?
- Does it include frame buffers, glyph cache, or just working memory?

Answer:
- Treat 512 MB as a soft operational target by default.
- If exceeded: degrade quality tiers first (scales, buffers, cache aggressiveness), then warn.
- `--strict-memory` mode can enforce hard-fail for CI/perf gates.
- Budget includes total process resident memory (all buffers + caches).

### Q29. Startup Time (2s gate)
- Includes font loading?
- Includes video file open/parse?
- Includes first frame decode?

Answer:
- Yes to all three.
- Startup metric is from process start to first successfully rendered frame.

### Q30. Flicker Metric (8% gate)
- How is "cell flip" defined? Glyph only? Glyph+color?
- Over what time window? Rolling average?
- What's the warm-up period before measurement?

Answer:
- Release gate metric uses glyph flips (primary).
- Track secondary metric for glyph+major-color-bin flips for diagnostics.
- Measurement window: rolling 120 frames after 30-frame warm-up.

---

## Answers Template
Resolved in-place above; this file now serves as the answered decision record for implementation.

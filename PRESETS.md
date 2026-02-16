# Content Presets

Use the `--profile` flag (or set `profile = "..."` in config) to apply tuned defaults for different source types.

## Profiles

- `natural`
  - Balanced motion, temporal stability, and halftone detail for real-world video.
- `anime`
  - Preserves line art and flat regions with stronger temporal flicker suppression.
- `ui`
  - Prioritizes crisp text/shapes and disables halftone noise by default.

## CLI Usage

```bat
ascii-engine --profile natural input.mp4
ascii-engine --profile anime input.mp4
ascii-engine --profile ui input.mp4
```

## Config Usage

```toml
profile = "anime"  # natural | anime | ui
```

## Override Behavior

- Profile values are applied first.
- Explicit CLI flags still override profile-tuned values.

# Baseline Validation

Status: refreshed on 2026-07-16 for the Windows no-OpenCV MSVC baseline.

Environment:

- AMD Ryzen 7 7800X3D, Windows 11.
- Visual Studio 2022 Build Tools, MSVC 19.44, OpenMP enabled.
- Project-local vcpkg dependencies under `.tools/vcpkg`.
- Build directory: `build-noopencv`.
- Independent clean proof directory: `cmake-build-release-proof`.
- Build flags: `ASCII_USE_OPENCV=OFF`, `ASCII_ENABLE_AVX2=OFF`, `CMAKE_BUILD_TYPE=Release`.

Verified commands:

```powershell
.\setup_windows_deps.cmd
.\build_noopencv_check.cmd
cmd /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"" -arch=x64 -host_arch=x64 && cmake --build build-noopencv -j 4 && ctest --test-dir build-noopencv --output-on-failure"
```

Observed validation:

- A clean configure built 41 targets/objects without compiler warnings; CTest passed 34/34 tests.
- Coverage includes strict CLI/config behavior, temporal/motion/edge/glyph regressions, deterministic replay/golden text, generated real-video decode/export, block-art still/video, requested-output failures, and truncated decode failure.
- The versioned Windows x64 ZIP built from the clean tree passed a bundled `--help` smoke test on this host and contained the executable, license, README, runtime notes, SDL2, FFmpeg, and zstd DLLs.

Performance workload:

- Fixture: deterministic 60-frame 1920x1080 MKV at 30 FPS.
- CLI shape: `--cols 120 --rows 40 --color truecolor --no-audio`, offline text export.
- Full-quality result: 10.86 processing FPS, 136.43 MiB peak working set.
- Speed-path result with `--fast --no-contours`: 25.70 processing FPS, 106.70 MiB peak working set.
- One-image `120x40` startup/process/exit: 0.32 seconds, 75.91 MiB peak working set.
- Interpretation: memory and startup targets pass; the 24 FPS target passes only in the explicit speed path.

Known baseline limitations:

- OpenCV-enabled configuration is not part of this baseline.
- Hosted CI must still run before cross-platform status is considered proven.
- The release ZIP still needs a smoke test on a separate clean Windows machine.
- Full-quality processing remains below the 24 FPS reference target on this workload.
- Verified still-image CLI targets are `.jpg`, `.jpeg`, and `.bmp`; `.png` is intentionally rejected in the no-OpenCV MSVC baseline until the FFmpeg still encoder path is fixed.

# Baseline Validation

Status: captured on 2026-05-12 for the Windows no-OpenCV MSVC baseline.

Environment:

- Visual Studio 2022 Build Tools, MSVC 19.44.
- Project-local vcpkg dependencies under `.tools/vcpkg`.
- Build directory: `build-noopencv`.
- Build flags: `ASCII_USE_OPENCV=OFF`, `ASCII_ENABLE_AVX2=OFF`, `CMAKE_BUILD_TYPE=Release`.

Verified commands:

```powershell
.\setup_windows_deps.cmd
.\build_noopencv_check.cmd
cmd /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"" -arch=x64 -host_arch=x64 && cmake --build build-noopencv -j 4 && ctest --test-dir build-noopencv --output-on-failure"
```

Observed validation:

- `ascii-engine` configures and builds with the no-OpenCV default path.
- CTest passes the unit, replay, terminal-renderer, help, invalid-output, image text, image still, replay write, and replay inspect tests.
- CLI smoke output on `images/potsepis.png` at `16x8 --fast --no-audio` reports one processed frame and emits `[PERF]`, `[PERF_STAGES]`, and `[PERF_STAGES_PCT]` summaries.

Known baseline limitations:

- OpenCV-enabled configuration is not part of this baseline.
- Hosted CI must still run before cross-platform status is considered proven.
- Verified still-image CLI targets are `.jpg`, `.jpeg`, and `.bmp`; `.png` is intentionally rejected in the no-OpenCV MSVC baseline until the FFmpeg still encoder path is fixed.

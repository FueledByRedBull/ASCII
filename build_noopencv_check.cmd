@echo on
setlocal EnableDelayedExpansion

set "EXITCODE=0"
set "VSDEVCMD="
set "USE_VCPKG_TOOLCHAIN=0"
set "SCRIPT_DIR=%~dp0"
set "LOCAL_VCPKG=!SCRIPT_DIR!.tools\vcpkg"

if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
)

if "%VSDEVCMD%"=="" (
    echo [ERROR] Could not find VsDevCmd.bat for Visual Studio 2022.
    echo         Install "Desktop development with C++" and CMake/Ninja components.
    set "EXITCODE=1"
    goto :done
)

call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 (
    echo [ERROR] Failed to initialize Visual Studio Developer Command Prompt.
    set "EXITCODE=!errorlevel!"
    goto :done
)

if exist "!LOCAL_VCPKG!\scripts\buildsystems\vcpkg.cmake" (
    set "VCPKG_ROOT=!LOCAL_VCPKG!"
    set "USE_VCPKG_TOOLCHAIN=1"
    echo [INFO] Using local vcpkg toolchain: !VCPKG_ROOT!
) else if defined VCPKG_ROOT (
    if exist "!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake" (
        set "USE_VCPKG_TOOLCHAIN=1"
        echo [INFO] Using VCPKG_ROOT toolchain: !VCPKG_ROOT!
    )
)

if "!USE_VCPKG_TOOLCHAIN!"=="1" (
    cmake --fresh -S . -B build-noopencv -G "Ninja" ^
        -DASCII_USE_OPENCV=OFF ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_TOOLCHAIN_FILE="!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake" ^
        -DVCPKG_TARGET_TRIPLET=x64-windows
) else (
    cmake --fresh -S . -B build-noopencv -G "Ninja" -DASCII_USE_OPENCV=OFF -DCMAKE_BUILD_TYPE=Release
)
if errorlevel 1 (
    echo [ERROR] CMake configure failed.
    echo [HINT] Missing SDL2/FFmpeg/Zstd on Windows is common.
    echo [HINT] Run setup_windows_deps.cmd first to install deps into .tools\vcpkg.
    set "EXITCODE=!errorlevel!"
    goto :done
)

if exist "build-noopencv\ascii-engine.exe" (
    echo [INFO] Releasing old ascii-engine.exe lock if present...
    taskkill /F /IM ascii-engine.exe >nul 2>&1
    timeout /t 1 /nobreak >nul
)

cmake --build build-noopencv --target ascii-engine -j 4
if errorlevel 1 (
    echo [WARN] Build failed. Retrying once after lock cleanup...
    taskkill /F /IM ascii-engine.exe >nul 2>&1
    timeout /t 1 /nobreak >nul
    cmake --build build-noopencv --target ascii-engine -j 4
    if errorlevel 1 (
        echo [ERROR] Build failed.
        set "EXITCODE=!errorlevel!"
        goto :done
    )
)

echo [OK] Build completed successfully.

:done
echo.
echo Exit code: !EXITCODE!
pause
endlocal & exit /b %EXITCODE%

@echo on
setlocal EnableDelayedExpansion

set "EXITCODE=0"
set "SCRIPT_DIR=%~dp0"
set "VSDEVCMD="
set "PROJECT_VCPKG=!SCRIPT_DIR!.tools\vcpkg"

if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" (
    set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
)

if "!VSDEVCMD!"=="" (
    echo [ERROR] Could not find Visual Studio 2022 developer tools.
    set "EXITCODE=1"
    goto :done
)

call "!VSDEVCMD!" -arch=x64 -host_arch=x64
if errorlevel 1 (
    echo [ERROR] Failed to initialize Visual Studio developer prompt.
    set "EXITCODE=!errorlevel!"
    goto :done
)

if not exist "!PROJECT_VCPKG!\vcpkg.exe" (
    if not exist "!PROJECT_VCPKG!" (
        where git >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] git was not found in PATH.
            echo         Install Git for Windows, then rerun this script.
            set "EXITCODE=1"
            goto :done
        )
        git clone https://github.com/microsoft/vcpkg "!PROJECT_VCPKG!"
        if errorlevel 1 (
            echo [ERROR] Failed to clone vcpkg.
            set "EXITCODE=!errorlevel!"
            goto :done
        )
    )

    pushd "!PROJECT_VCPKG!"
    if not exist "bootstrap-vcpkg.bat" (
        echo [ERROR] bootstrap-vcpkg.bat not found in !PROJECT_VCPKG!
        set "EXITCODE=1"
        popd
        goto :done
    )

    call bootstrap-vcpkg.bat -disableMetrics
    if errorlevel 1 (
        echo [ERROR] vcpkg bootstrap failed.
        set "EXITCODE=!errorlevel!"
        popd
        goto :done
    )
    popd
)

pushd "!PROJECT_VCPKG!"
vcpkg.exe install sdl2:x64-windows ffmpeg:x64-windows zstd:x64-windows
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    set "EXITCODE=!errorlevel!"
    popd
    goto :done
)
popd

setx VCPKG_ROOT "!PROJECT_VCPKG!" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Could not persist VCPKG_ROOT with setx.
    echo [INFO] Use this in terminals before building:
    echo        set VCPKG_ROOT=!PROJECT_VCPKG!
) else (
    echo [OK] Persisted VCPKG_ROOT=!PROJECT_VCPKG!
)

echo [OK] Dependencies installed.
echo [INFO] Re-run build_noopencv_check.cmd

:done
echo.
echo Exit code: !EXITCODE!
pause
endlocal & exit /b %EXITCODE%

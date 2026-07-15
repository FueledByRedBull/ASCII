param(
    [string]$BuildDirectory = "build-noopencv",
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $repo $BuildDirectory
$executable = Join-Path $build "ascii-engine.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Release executable was not found: $executable"
}

$outputRoot = Join-Path $repo "output"
$bundleName = "ascii-engine-$Version-windows-x64"
$bundle = Join-Path $outputRoot $bundleName
New-Item -ItemType Directory -Path $bundle -Force | Out-Null

Copy-Item -LiteralPath $executable -Destination $bundle -Force
Copy-Item -LiteralPath (Join-Path $repo "LICENSE") -Destination $bundle -Force
Copy-Item -LiteralPath (Join-Path $repo "README.md") -Destination $bundle -Force

$runtimeLibraries = Get-ChildItem -LiteralPath $build -Filter "*.dll" -File
foreach ($library in $runtimeLibraries) {
    Copy-Item -LiteralPath $library.FullName -Destination $bundle -Force
}

$runtimeNotes = @"
ASCII Engine $Version - Windows x64 runtime notes

Baseline: Windows 10/11 x64, MSVC Release build, OpenCV OFF, AVX2 OFF.
Bundled runtime libraries: SDL2, FFmpeg, and zstd DLLs copied from the verified build tree.
The executable processes local files only and does not use machine-learning services.
PNG still-image output is intentionally unsupported in this baseline; use JPG, JPEG, or BMP.

Run: .\ascii-engine.exe --help
"@
Set-Content -LiteralPath (Join-Path $bundle "RUNTIME.txt") -Value $runtimeNotes -Encoding utf8

$smoke = Start-Process -FilePath (Join-Path $bundle "ascii-engine.exe") `
    -ArgumentList "--help" -WorkingDirectory $bundle -NoNewWindow -Wait -PassThru
if ($smoke.ExitCode -ne 0) {
    throw "Packaged executable smoke test failed with exit code $($smoke.ExitCode)"
}

$archive = Join-Path $outputRoot "$bundleName.zip"
Compress-Archive -LiteralPath $bundle -DestinationPath $archive -Force
Write-Output $archive

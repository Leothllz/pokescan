$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvRoot = Join-Path $ProjectRoot "pokescan-rocm312"
$SitePackages = Join-Path $VenvRoot "Lib\site-packages"

$core = Join-Path $SitePackages "_rocm_sdk_coregfx103X-all\bin"
$devel = Join-Path $SitePackages "_rocm_sdk_develgfx103X-all\bin"
$libs = Join-Path $SitePackages "_rocm_sdk_libraries_gfx103X_allgfx103X-all\bin"

if (!(Test-Path $VenvRoot)) {
    throw "ROCm venv not found: $VenvRoot"
}

foreach ($path in @($core, $devel, $libs)) {
    if (!(Test-Path $path)) {
        throw "ROCm path not found: $path"
    }
}

$env:PATH = "$core;$devel;$libs;$env:PATH"
Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:HSA_OVERRIDE_GFX_VERSION = "10.3.0"

Write-Host "ROCm environment ready for RX 6750 XT"
Write-Host "Python: $VenvRoot\Scripts\python.exe"

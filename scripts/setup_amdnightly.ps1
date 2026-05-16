$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvRoot = Join-Path $ProjectRoot "pokescan-amdnightly"
$SitePackages = Join-Path $VenvRoot "Lib\site-packages"

$core = Join-Path $SitePackages "_rocm_sdk_core"
$coreBin = Join-Path $core "bin"
$libsBin = Join-Path $SitePackages "_rocm_sdk_libraries_gfx103X_dgpu\bin"
$clangInclude = Join-Path $core "lib\llvm\lib\clang\23\include"

$msvcInclude = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\include"
$ucrtInclude = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt"

foreach ($path in @($VenvRoot, $coreBin, $libsBin, $clangInclude, $msvcInclude, $ucrtInclude)) {
    if (!(Test-Path $path)) {
        throw "Required path not found: $path"
    }
}

$includes = "$msvcInclude;$ucrtInclude;$clangInclude"
$env:PATH = "$coreBin;$libsBin;$env:PATH"
$env:INCLUDE = "$includes;$env:INCLUDE"
$env:CPLUS_INCLUDE_PATH = $includes
Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:HSA_OVERRIDE_GFX_VERSION = "10.3.0"

Write-Host "AMD nightly ROCm environment ready for RX 6750 XT"
Write-Host "Python: $VenvRoot\Scripts\python.exe"

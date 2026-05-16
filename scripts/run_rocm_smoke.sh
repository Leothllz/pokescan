#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-gfx1030}"
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"

timeout 35s .venv-wsl/bin/python -u scripts/rocm_smoke.py
code=$?
echo "exit_code=$code"

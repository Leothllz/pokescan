#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

mkdir -p /tmp/pokescan_hip_test

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-gfx1030}"
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"

echo "Compiling HIP kernel smoke test..."
hipcc --offload-arch=gfx1030 scripts/hip_kernel_smoke.cpp -o /tmp/pokescan_hip_test/hip_kernel_smoke

run_case() {
  name="$1"
  shift
  echo
  echo "== $name =="
  env "$@" timeout 25s /tmp/pokescan_hip_test/hip_kernel_smoke
  code=$?
  echo "exit_code=$code"
}

run_case "baseline"
run_case "sdma-disabled" HSA_ENABLE_SDMA=0
run_case "heap-tuned" GPU_MAX_HEAP_SIZE=100 GPU_SINGLE_ALLOC_PERCENT=100
run_case "sync-wait" AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1

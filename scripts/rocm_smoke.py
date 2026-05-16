#!/usr/bin/env python3
"""Small ROCm/PyTorch smoke test for WSL."""

import os


os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HCC_AMDGPU_TARGET", "gfx1030")
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")


def main() -> int:
    import torch

    print(f"torch={torch.__version__}", flush=True)
    print(f"hip={torch.version.hip}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        return 1

    print(f"device_count={torch.cuda.device_count()}", flush=True)
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    print("allocating_tensor", flush=True)
    x = torch.ones(1, device="cuda")
    print(f"tensor={x}", flush=True)
    torch.cuda.synchronize()
    print("ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

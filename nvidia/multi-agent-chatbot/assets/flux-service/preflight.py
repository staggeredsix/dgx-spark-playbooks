import sys
import torch

print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    try:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA device: {device_name}")
    except Exception as exc:
        print(f"Failed to get CUDA device name: {exc}", file=sys.stderr)
else:
    print(
        "CUDA not available at runtime. Ensure docker-compose has gpus: all and NVIDIA Container Toolkit is installed.",
        file=sys.stderr,
    )
    sys.exit(1)

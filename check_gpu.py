#!/usr/bin/env python3
"""
GPU Diagnostic Script for VisiTrack.

Checks CUDA availability, GPU hardware, driver versions, precision
capabilities, and runs a quick CUDA tensor test.

Usage:
    python check_gpu.py
"""

from __future__ import annotations

import os
import sys

# Fix Windows console encoding for Unicode box-drawing characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _header(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def check_pytorch() -> bool:
    """Check PyTorch installation and CUDA support."""
    _header("PyTorch Installation")
    try:
        import torch

        print(f"  PyTorch version:    {torch.__version__}")
        print(f"  CUDA available:     {torch.cuda.is_available()}")
        print(f"  CUDA runtime ver:   {torch.version.cuda or 'N/A'}")  # type: ignore[attr-defined]
        print(f"  cuDNN version:      {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print(f"  cuDNN enabled:      {torch.backends.cudnn.enabled}")
        return torch.cuda.is_available()
    except ImportError:
        print("  ❌ PyTorch is NOT installed.")
        print("  Install with CUDA:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_gpu_details() -> None:
    """Print detailed GPU information."""
    import torch

    _header("GPU Hardware")

    gpu_count = torch.cuda.device_count()
    print(f"  GPU count:          {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024 ** 3)
        cc = f"{props.major}.{props.minor}"
        print(f"\n  ── GPU {i} ──")
        print(f"    Name:             {torch.cuda.get_device_name(i)}")
        print(f"    Compute cap.:     {cc}")
        print(f"    Total memory:     {total_mem:.1f} GB")
        print(f"    Multi-processors: {props.multi_processor_count}")

        # Precision support
        fp16_ok = props.major >= 7
        bf16_ok = props.major >= 8
        tf32_ok = props.major >= 8
        print(f"    FP16 support:     {'✅ Yes' if fp16_ok else '❌ No (compute cap < 7.0)'}")
        print(f"    BF16 support:     {'✅ Yes' if bf16_ok else '❌ No (compute cap < 8.0)'}")
        print(f"    TF32 support:     {'✅ Yes' if tf32_ok else '❌ No (compute cap < 8.0)'}")

    current = torch.cuda.current_device()
    print(f"\n  Selected GPU:       {current} ({torch.cuda.get_device_name(current)})")


def check_memory() -> None:
    """Print current GPU memory usage."""
    import torch

    _header("GPU Memory")
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        print(f"  GPU {i}:")
        print(f"    Total:     {total:.2f} GB")
        print(f"    Allocated: {alloc:.4f} GB")
        print(f"    Reserved:  {reserved:.4f} GB")


def check_tensor_computation() -> None:
    """Run a small CUDA tensor calculation."""
    import torch

    _header("CUDA Tensor Test")

    device = torch.device("cuda:0")
    print(f"  Creating 1000×1000 random tensors on {device} …")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = a @ b
    torch.cuda.synchronize()
    print(f"  Matrix multiply result shape: {c.shape}")
    print(f"  Result sum (sanity check):    {c.sum().item():.2f}")
    print(f"  Result dtype:                 {c.dtype}")
    print("  ✅ CUDA tensor computation PASSED")

    # FP16 test
    print("\n  Testing FP16 matmul …")
    a16 = a.half()
    b16 = b.half()
    c16 = a16 @ b16
    torch.cuda.synchronize()
    print(f"  FP16 result dtype:            {c16.dtype}")
    print(f"  FP16 result sum:              {c16.sum().item():.2f}")
    print("  ✅ FP16 computation PASSED")

    # autocast test
    print("\n  Testing torch.autocast …")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        c_amp = a @ b
    torch.cuda.synchronize()
    print(f"  Autocast result dtype:        {c_amp.dtype}")
    print("  ✅ torch.autocast PASSED")

    del a, b, c, a16, b16, c16, c_amp
    torch.cuda.empty_cache()


def check_nvdec() -> None:
    """Check FFmpeg NVDEC availability."""
    import subprocess

    _header("NVDEC (Video Hardware Decoding)")
    try:
        result = subprocess.run(
            ["ffmpeg", "-decoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        h264 = "h264_cuvid" in result.stdout
        hevc = "hevc_cuvid" in result.stdout
        print(f"  FFmpeg found:       ✅")
        print(f"  h264_cuvid:         {'✅ Available' if h264 else '❌ Not found'}")
        print(f"  hevc_cuvid:         {'✅ Available' if hevc else '❌ Not found'}")
    except FileNotFoundError:
        print("  ❌ FFmpeg not found in PATH.")
    except subprocess.TimeoutExpired:
        print("  ⚠️ FFmpeg timed out.")


def check_dependencies() -> None:
    """Check availability of key dependencies."""
    _header("Dependencies")

    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("rfdetr", "RF-DETR"),
        ("insightface", "InsightFace"),
        ("onnxruntime", "ONNX Runtime"),
        ("torchreid", "TorchReID"),
    ]

    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  {display_name:20s} ✅ {version}")
        except ImportError:
            print(f"  {display_name:20s} ❌ Not installed")

    # Check onnxruntime GPU provider
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        cuda_prov = "CUDAExecutionProvider" in providers
        trt_prov = "TensorrtExecutionProvider" in providers
        print(f"\n  ONNX CUDA provider: {'✅' if cuda_prov else '❌'}")
        print(f"  ONNX TensorRT:      {'✅' if trt_prov else '❌'}")
    except ImportError:
        pass


def main() -> None:
    print("╔═══════════════════════════════════════════════════╗")
    print("║        VisiTrack — GPU Diagnostic Report         ║")
    print("╚═══════════════════════════════════════════════════╝")

    cuda_ok = check_pytorch()

    if cuda_ok:
        check_gpu_details()
        check_memory()
        check_tensor_computation()
    else:
        print("\n  ⚠️  CUDA is NOT available — skipping GPU tests.")
        print("  Install CUDA-enabled PyTorch:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    check_nvdec()
    check_dependencies()

    _header("Summary")
    if cuda_ok:
        import torch

        print(f"  ✅ CUDA is available — {torch.cuda.get_device_name(0)}")
        print(f"  ✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
        print("  ✅ Ready for GPU-accelerated inference")
    else:
        print("  ❌ CUDA is NOT available")
        print("  The system requires an NVIDIA GPU with CUDA support.")

    print()


if __name__ == "__main__":
    main()

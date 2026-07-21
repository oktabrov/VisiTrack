# VisiTrack

**GPU-accelerated real-time visitor tracking for retail stores.**

Person detection, face recognition, and person re-identification —
powered by **NVIDIA CUDA** with mandatory GPU acceleration.

## Architecture

| Component | Model | Device |
|---|---|---|
| Person Detection | RF-DETR (DINOv2 backbone) | CUDA |
| Face Detection | SCRFD / RetinaFace (InsightFace) | CUDA |
| Face Recognition | ArcFace (InsightFace) | CUDA |
| Person ReID | OSNet (torchreid) | CUDA |
| Video Decoding | FFmpeg NVDEC (h264_cuvid) | GPU (optional) |

## Prerequisites

- **NVIDIA GPU** with CUDA support (compute capability ≥ 7.0 for FP16)
- **NVIDIA Drivers** ≥ 525.x
- **CUDA Toolkit** 12.1+ (or matching your driver)
- **Python** 3.10, 3.11, or 3.12
- **FFmpeg** with NVDEC support (for GPU video decoding)

## Installation

### 1. Install CUDA-enabled PyTorch

> **⚠️ Do this FIRST** — do not install CPU-only PyTorch.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Verify GPU Access

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Both commands should return `True` and your GPU name. If not, check your
NVIDIA drivers and CUDA installation.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install TorchReID (from source)

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

### 5. Run GPU Diagnostics

```bash
python check_gpu.py
```

This prints a full report: CUDA status, GPU details, precision support,
NVDEC availability, and runs a test CUDA tensor calculation.

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings. Key variables:

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | *(required)* | RTSP stream URL |
| `DEVICE` | `cuda` | Device for AI inference |
| `CUDA_DEVICE_INDEX` | `0` | GPU to use (or set `CUDA_VISIBLE_DEVICES`) |
| `USE_HALF_PRECISION` | `true` | Enable FP16 mixed precision |
| `USE_GPU_DECODING` | `true` | Use NVDEC for video decoding |
| `ALLOW_CPU_FALLBACK` | `false` | Allow CPU if no CUDA (NOT recommended) |
| `DETECTION_BATCH_SIZE` | `1` | Batch size for RF-DETR |
| `FACE_BATCH_SIZE` | `16` | Batch size for face embeddings |
| `REID_BATCH_SIZE` | `16` | Batch size for ReID features |
| `ENABLE_TENSORRT` | `false` | Enable TensorRT optimization |
| `ENABLE_TORCH_COMPILE` | `false` | Enable torch.compile() |

See [.env.example](.env.example) for the full list.

## Usage

```bash
python -m store_visitor_system.main
```

### Expected Startup Output

```
═══════════════════════════════════════════════════
  VisiTrack — GPU Information
═══════════════════════════════════════════════════
  GPU:              NVIDIA GeForce RTX 3080
  CUDA Version:     12.1
  PyTorch CUDA:     12.1
  GPU Memory:       10.0 GB
  Device Index:     0
  Compute Cap.:     8.6
  FP16 Support:     True
  BF16 Support:     True
═══════════════════════════════════════════════════

═══════════════════════════════════════════════════
  VisiTrack — Device Summary
═══════════════════════════════════════════════════
  AI inference device:  cuda:0
  RF-DETR device:       cuda:0
  Face model device:    cuda:0
  ReID model device:    cuda:0
  Mixed precision:      FP16
  Video decoding:       NVIDIA NVDEC
═══════════════════════════════════════════════════
```

## GPU Memory Considerations

All three AI models remain loaded in GPU memory simultaneously:

| Model | Approximate VRAM (FP16) |
|---|---|
| RF-DETR Base | ~800 MB |
| InsightFace (buffalo_l) | ~600 MB |
| OSNet x1.0 | ~50 MB |
| **Total** | **~1.5 GB** |

Recommended: GPU with ≥ 4 GB VRAM. For 8+ GB GPUs, increase batch sizes
for better throughput.

### OOM Recovery

If the system encounters an out-of-memory error:
1. Batch sizes are automatically halved.
2. `torch.cuda.empty_cache()` is called (emergency only).
3. The inference worker restarts cleanly.
4. No confirmed events are lost.

## Optional: TensorRT Acceleration

When `ENABLE_TENSORRT=true`:

1. Install TensorRT: `pip install tensorrt`
2. Export models to ONNX format
3. TensorRT engines are cached after first compilation
4. Falls back to PyTorch CUDA if initialization fails

### Exporting RF-DETR to ONNX

```python
import torch
from rfdetr import RFDETRBase

model = RFDETRBase()
dummy = torch.randn(1, 3, 640, 640).cuda()
torch.onnx.export(model.model, dummy, "rfdetr.onnx", opset_version=17)
```

## Optional: torch.compile

When `ENABLE_TORCH_COMPILE=true`:

- `torch.compile()` is applied once after model loading.
- A warm-up inference is run to trigger compilation.
- Falls back to eager CUDA if compilation fails.
- Requires PyTorch ≥ 2.0.

## Performance Monitoring

The system logs performance metrics periodically (default: every 30s):

```
[PERF] capture=25.0fps | detect=12.3fps | processed=12.3fps | dropped=42 |
       det_lat=45.2ms | face_lat=12.1ms | reid_lat=8.3ms |
       pipeline=65.5ms | gpu_alloc=1.2GB | gpu_res=2.0GB | queue=1/2
```

## Project Structure

```
store_visitor_system/
├── __init__.py          # Package init
├── config.py            # Centralized configuration (env vars)
├── gpu.py               # CUDA management (THE single device source)
├── performance.py       # FPS, latency, GPU memory monitoring
├── video_decoder.py     # RTSP with NVDEC + CPU fallback
├── detector.py          # RF-DETR person detection
├── face_processor.py    # InsightFace face detection + embeddings
├── reid.py              # OSNet person re-identification
├── tracker.py           # IoU-based multi-object tracker
├── pipeline.py          # Main inference orchestrator
└── main.py              # Application entry point
check_gpu.py             # GPU diagnostic script
requirements.txt         # Dependencies
.env.example             # Configuration template
README.md                # This file
```

## License

Proprietary — internal use only.

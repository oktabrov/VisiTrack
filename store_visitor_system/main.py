"""
VisiTrack — Application Entry Point.

Startup sequence:
  1. Load configuration from .env / environment variables.
  2. Initialize GPUManager → validate CUDA → log GPU info.
  3. Load RF-DETR onto CUDA → warm up.
  4. Load face models onto CUDA → warm up.
  5. Load OSNet ReID onto CUDA → warm up.
  6. Log all model device placements.
  7. Initialize video decoder (NVDEC or CPU fallback).
  8. Start performance monitor.
  9. Start inference pipeline.
  10. Run until interrupted.
"""

from __future__ import annotations

import logging
import os
import sys

# Fix Windows console encoding for Unicode characters in banners
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from .config import Config
from .gpu import GPUManager
from .pipeline import InferencePipeline
from .video_decoder import NVDECDecoder


def _setup_logging() -> None:
    """Configure root logging with a clean format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)


def _log_device_summary(gpu: GPUManager, config: Config) -> None:
    """Print the final device-placement summary."""
    nvdec_status = "NVIDIA NVDEC" if (
        config.use_gpu_decoding and NVDECDecoder.is_available()
    ) else "CPU (OpenCV)"

    banner = (
        "\n"
        "═══════════════════════════════════════════════════\n"
        "  VisiTrack — Device Summary\n"
        "═══════════════════════════════════════════════════\n"
        f"  AI inference device:  {gpu.device}\n"
        f"  RF-DETR device:       {gpu.device}\n"
        f"  Face model device:    {gpu.device}\n"
        f"  ReID model device:    {gpu.device}\n"
        f"  Mixed precision:      {gpu.precision_label}\n"
        f"  Video decoding:       {nvdec_status}\n"
        "═══════════════════════════════════════════════════"
    )
    print(banner)


def main() -> None:
    """Main entry point for the store visitor system."""
    _setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting VisiTrack …")

    # ── Step 1: Configuration ────────────────────────────────────────
    try:
        config = Config()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    logger.info("\n%s", config.summary())

    # ── Step 2: GPU Initialization ───────────────────────────────────
    gpu = GPUManager(config)

    try:
        gpu.validate_cuda()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    gpu.select_device()
    gpu.log_gpu_info()

    # ── Step 3: Validate RTSP URL ────────────────────────────────────
    if not config.rtsp_url:
        logger.error(
            "No RTSP_URL configured. Set it in .env or environment:\n"
            "  RTSP_URL=rtsp://user:pass@camera-ip:554/stream"
        )
        sys.exit(1)

    # ── Step 4: Load models & start pipeline ─────────────────────────
    # Models are loaded inside InferencePipeline.__init__
    # (detector, face_processor, reid — all on CUDA with warm-up)
    try:
        pipeline = InferencePipeline(gpu, config)
    except Exception as exc:
        logger.error("Failed to initialize pipeline: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Step 5: Device summary ───────────────────────────────────────
    _log_device_summary(gpu, config)
    gpu.log_memory()

    # ── Step 6: Run ──────────────────────────────────────────────────
    logger.info("All systems ready. Starting real-time inference …")
    pipeline.run()

    logger.info("VisiTrack terminated.")


if __name__ == "__main__":
    main()

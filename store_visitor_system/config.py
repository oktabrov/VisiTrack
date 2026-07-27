"""
Centralized configuration for VisiTrack.

All settings are loaded from environment variables (with .env file support).
This is the single source of truth for every tunable parameter in the system.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional at import time


def _env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    val = os.environ.get(key, "").strip().lower()
    if val == "":
        return default
    return val in ("1", "true", "yes", "on")


def _env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer."""
    val = os.environ.get(key, "").strip()
    if val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float."""
    val = os.environ.get(key, "").strip()
    if val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_str(key: str, default: str = "") -> str:
    """Read an environment variable as a string."""
    return os.environ.get(key, default).strip()


@dataclass
class Config:
    """Immutable application configuration populated from environment variables."""

    # ── GPU / Device ─────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: _env_str("DEVICE", "cuda"))
    cuda_device_index: int = field(
        default_factory=lambda: _env_int("CUDA_DEVICE_INDEX", 0)
    )
    use_half_precision: bool = field(
        default_factory=lambda: _env_bool("USE_HALF_PRECISION", True)
    )
    use_gpu_decoding: bool = field(
        default_factory=lambda: _env_bool("USE_GPU_DECODING", True)
    )
    allow_cpu_fallback: bool = field(
        default_factory=lambda: _env_bool("ALLOW_CPU_FALLBACK", False)
    )
    gpu_memory_limit_mb: int = field(
        default_factory=lambda: _env_int("GPU_MEMORY_LIMIT_MB", 0)
    )

    # ── Batch Sizes ──────────────────────────────────────────────────────
    detection_batch_size: int = field(
        default_factory=lambda: _env_int("DETECTION_BATCH_SIZE", 1)
    )
    face_batch_size: int = field(
        default_factory=lambda: _env_int("FACE_BATCH_SIZE", 16)
    )
    reid_batch_size: int = field(
        default_factory=lambda: _env_int("REID_BATCH_SIZE", 16)
    )

    # ── Optional Optimizations ───────────────────────────────────────────
    enable_tensorrt: bool = field(
        default_factory=lambda: _env_bool("ENABLE_TENSORRT", False)
    )
    enable_torch_compile: bool = field(
        default_factory=lambda: _env_bool("ENABLE_TORCH_COMPILE", False)
    )

    # ── Video / RTSP ─────────────────────────────────────────────────────
    rtsp_url: str = field(default_factory=lambda: _env_str("RTSP_URL", ""))
    frame_skip: int = field(default_factory=lambda: _env_int("FRAME_SKIP", 0))
    queue_max_size: int = field(
        default_factory=lambda: _env_int("QUEUE_MAX_SIZE", 2)
    )
    reconnect_delay: float = field(
        default_factory=lambda: _env_float("RECONNECT_DELAY", 2.0)
    )
    max_reconnect_delay: float = field(
        default_factory=lambda: _env_float("MAX_RECONNECT_DELAY", 60.0)
    )

    # ── Detection / Recognition Thresholds ───────────────────────────────
    detection_confidence: float = field(
        default_factory=lambda: _env_float("DETECTION_CONFIDENCE", 0.35)
    )
    face_detection_threshold: float = field(
        default_factory=lambda: _env_float("FACE_DETECTION_THRESHOLD", 0.5)
    )
    reid_match_threshold: float = field(
        default_factory=lambda: _env_float("REID_MATCH_THRESHOLD", 0.6)
    )

    # ── Performance Monitoring ───────────────────────────────────────────
    perf_log_interval: int = field(
        default_factory=lambda: _env_int("PERF_LOG_INTERVAL", 30)
    )

    # ── Model Paths (optional overrides) ─────────────────────────────────
    rfdetr_weights: str = field(
        default_factory=lambda: _env_str("RFDETR_WEIGHTS", "")
    )
    face_model_name: str = field(
        default_factory=lambda: _env_str("FACE_MODEL_NAME", "buffalo_l")
    )
    reid_model_name: str = field(
        default_factory=lambda: _env_str("REID_MODEL_NAME", "osnet_x1_0")
    )

    # ── Tracker ──────────────────────────────────────────────────────────
    track_max_age: int = field(
        default_factory=lambda: _env_int("TRACK_MAX_AGE", 30)
    )
    track_min_hits: int = field(
        default_factory=lambda: _env_int("TRACK_MIN_HITS", 1)
    )
    track_iou_threshold: float = field(
        default_factory=lambda: _env_float("TRACK_IOU_THRESHOLD", 0.3)
    )

    # Database Settings
    use_postgres: bool = field(
        default_factory=lambda: _env_bool("USE_POSTGRES", True)
    )
    postgres_host: str = field(
        default_factory=lambda: _env_str("POSTGRES_HOST", "localhost")
    )
    postgres_port: int = field(
        default_factory=lambda: _env_int("POSTGRES_PORT", 5432)
    )
    postgres_db: str = field(
        default_factory=lambda: _env_str("POSTGRES_DB", "visitrack")
    )
    postgres_user: str = field(
        default_factory=lambda: _env_str("POSTGRES_USER", "postgres")
    )
    postgres_password: str = field(
        default_factory=lambda: _env_str("POSTGRES_PASSWORD", "postgres")
    )
    visit_cooldown_minutes: int = field(
        default_factory=lambda: _env_int("VISIT_COOLDOWN_MINUTES", 10)
    )

    # Web Dashboard Settings
    enable_web_server: bool = field(
        default_factory=lambda: _env_bool("ENABLE_WEB_SERVER", True)
    )
    web_server_port: int = field(
        default_factory=lambda: _env_int("WEB_SERVER_PORT", 8000)
    )

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.device not in ("cuda", "cpu"):
            raise ValueError(
                f"DEVICE must be 'cuda' or 'cpu', got '{self.device}'"
            )
        if self.cuda_device_index < 0:
            raise ValueError(
                f"CUDA_DEVICE_INDEX must be >= 0, got {self.cuda_device_index}"
            )
        if self.queue_max_size < 1:
            raise ValueError(
                f"QUEUE_MAX_SIZE must be >= 1, got {self.queue_max_size}"
            )
        if self.detection_batch_size < 1:
            raise ValueError(
                f"DETECTION_BATCH_SIZE must be >= 1, got {self.detection_batch_size}"
            )

    def summary(self) -> str:
        """Return a formatted summary of the current configuration."""
        lines = [
            "Configuration Summary",
            "─" * 50,
            f"  Device:               {self.device}",
            f"  CUDA Device Index:    {self.cuda_device_index}",
            f"  Half Precision:       {self.use_half_precision}",
            f"  GPU Decoding:         {self.use_gpu_decoding}",
            f"  CPU Fallback:         {self.allow_cpu_fallback}",
            f"  GPU Memory Limit:     {'unlimited' if self.gpu_memory_limit_mb == 0 else f'{self.gpu_memory_limit_mb} MB'}",
            f"  Detection Batch:      {self.detection_batch_size}",
            f"  Face Batch:           {self.face_batch_size}",
            f"  ReID Batch:           {self.reid_batch_size}",
            f"  TensorRT:             {self.enable_tensorrt}",
            f"  torch.compile:        {self.enable_torch_compile}",
            f"  RTSP URL:             {self.rtsp_url or '(not set)'}",
            f"  Frame Skip:           {self.frame_skip}",
            f"  Queue Max Size:       {self.queue_max_size}",
            f"  Detection Confidence: {self.detection_confidence}",
            f"  Face Threshold:       {self.face_detection_threshold}",
            f"  ReID Threshold:       {self.reid_match_threshold}",
            f"  Perf Log Interval:    {self.perf_log_interval}s",
        ]
        return "\n".join(lines)


DOOR_ROI_FILE = Path("door_roi.json")


def load_door_roi() -> dict:
    """Load the door ROI polygon points and enable status from door_roi.json."""
    if DOOR_ROI_FILE.exists():
        try:
            with open(DOOR_ROI_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"enabled": False, "points": []}


def save_door_roi(data: dict) -> None:
    """Save the door ROI polygon points and enable status to door_roi.json."""
    try:
        with open(DOOR_ROI_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

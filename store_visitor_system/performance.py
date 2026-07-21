"""
Performance monitoring for VisiTrack.

Thread-safe FPS counters, latency trackers, dropped-frame statistics,
and GPU memory stats with periodic logging.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, Generator, Optional

if TYPE_CHECKING:
    from .gpu import GPUManager

logger = logging.getLogger(__name__)

# Rolling window size for latency / FPS calculations
_WINDOW_SIZE = 120


@dataclass
class _Counter:
    """Thread-safe rolling-window counter for FPS calculation."""

    _timestamps: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_WINDOW_SIZE)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def tick(self) -> None:
        with self._lock:
            self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1] - self._timestamps[0]
            if span <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / span


@dataclass
class _LatencyTracker:
    """Thread-safe rolling-window latency tracker (milliseconds)."""

    _samples: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_WINDOW_SIZE)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, ms: float) -> None:
        with self._lock:
            self._samples.append(ms)

    @property
    def avg_ms(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)

    @property
    def max_ms(self) -> float:
        with self._lock:
            return max(self._samples) if self._samples else 0.0

    @property
    def min_ms(self) -> float:
        with self._lock:
            return min(self._samples) if self._samples else 0.0


class PerformanceMonitor:
    """Collects and periodically logs system performance metrics.

    Usage::

        perf = PerformanceMonitor(gpu_manager, log_interval=30)
        perf.start()

        # In capture thread:
        perf.capture_fps.tick()

        # In inference thread:
        with perf.measure_detection():
            results = detector.detect(frame)

        perf.stop()
    """

    def __init__(
        self,
        gpu_manager: Optional["GPUManager"] = None,
        log_interval: int = 30,
        queue_max_size: int = 2,
    ) -> None:
        self._gpu_manager = gpu_manager
        self._log_interval = log_interval
        self._queue_max_size = queue_max_size

        # FPS counters
        self.capture_fps = _Counter()
        self.detection_fps = _Counter()
        self.processed_fps = _Counter()

        # Latency trackers
        self.detection_latency = _LatencyTracker()
        self.face_latency = _LatencyTracker()
        self.reid_latency = _LatencyTracker()
        self.pipeline_latency = _LatencyTracker()

        # Frame statistics
        self._dropped_frames: int = 0
        self._total_frames: int = 0
        self._dropped_lock = threading.Lock()

        # Queue size (updated externally)
        self._queue_size: int = 0

        # Logging thread
        self._stop_event = threading.Event()
        self._log_thread: Optional[threading.Thread] = None

    # ── Frame tracking ───────────────────────────────────────────────────

    def record_dropped_frame(self) -> None:
        """Increment the dropped-frame counter."""
        with self._dropped_lock:
            self._dropped_frames += 1

    def record_total_frame(self) -> None:
        """Increment the total captured-frame counter."""
        with self._dropped_lock:
            self._total_frames += 1

    @property
    def dropped_frames(self) -> int:
        with self._dropped_lock:
            return self._dropped_frames

    @property
    def total_frames(self) -> int:
        with self._dropped_lock:
            return self._total_frames

    def update_queue_size(self, size: int) -> None:
        """Set the current frame-queue size (called from the queue manager)."""
        self._queue_size = size

    # ── Latency context managers ─────────────────────────────────────────

    @contextmanager
    def measure_detection(self) -> Generator[None, None, None]:
        """Measure detection inference latency in milliseconds."""
        t0 = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.detection_latency.record(elapsed)
        self.detection_fps.tick()

    @contextmanager
    def measure_face(self) -> Generator[None, None, None]:
        """Measure face detection + embedding latency in milliseconds."""
        t0 = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.face_latency.record(elapsed)

    @contextmanager
    def measure_reid(self) -> Generator[None, None, None]:
        """Measure ReID feature extraction latency in milliseconds."""
        t0 = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.reid_latency.record(elapsed)

    @contextmanager
    def measure_pipeline(self) -> Generator[None, None, None]:
        """Measure total pipeline latency for one frame."""
        t0 = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.pipeline_latency.record(elapsed)
        self.processed_fps.tick()

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, object]:
        """Return a dict of all current metrics."""
        gpu_mem = (
            self._gpu_manager.get_memory_stats() if self._gpu_manager else {}
        )
        return {
            "capture_fps": round(self.capture_fps.fps, 1),
            "detection_fps": round(self.detection_fps.fps, 1),
            "processed_fps": round(self.processed_fps.fps, 1),
            "dropped_frames": self.dropped_frames,
            "total_frames": self.total_frames,
            "det_latency_ms": round(self.detection_latency.avg_ms, 1),
            "face_latency_ms": round(self.face_latency.avg_ms, 1),
            "reid_latency_ms": round(self.reid_latency.avg_ms, 1),
            "pipeline_latency_ms": round(self.pipeline_latency.avg_ms, 1),
            "gpu_allocated_mb": round(gpu_mem.get("allocated_mb", 0), 1),
            "gpu_reserved_mb": round(gpu_mem.get("reserved_mb", 0), 1),
            "queue_size": self._queue_size,
            "queue_capacity": self._queue_max_size,
        }

    # ── Periodic logging ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background performance-logging thread."""
        if self._log_thread is not None:
            return
        self._stop_event.clear()
        self._log_thread = threading.Thread(
            target=self._log_loop, daemon=True, name="perf-monitor"
        )
        self._log_thread.start()
        logger.info(
            "Performance monitor started (logging every %ds).",
            self._log_interval,
        )

    def stop(self) -> None:
        """Stop the background logging thread."""
        self._stop_event.set()
        if self._log_thread is not None:
            self._log_thread.join(timeout=5)
            self._log_thread = None

    def _log_loop(self) -> None:
        """Internal loop that logs metrics at the configured interval."""
        while not self._stop_event.wait(timeout=self._log_interval):
            self._emit_log()
        # Final log on shutdown
        self._emit_log()

    def _emit_log(self) -> None:
        """Format and emit the performance log line."""
        s = self.snapshot()
        msg = (
            f"[PERF] capture={s['capture_fps']}fps | "
            f"detect={s['detection_fps']}fps | "
            f"processed={s['processed_fps']}fps | "
            f"dropped={s['dropped_frames']} | "
            f"det_lat={s['det_latency_ms']}ms | "
            f"face_lat={s['face_latency_ms']}ms | "
            f"reid_lat={s['reid_latency_ms']}ms | "
            f"pipeline={s['pipeline_latency_ms']}ms | "
            f"gpu_alloc={s['gpu_allocated_mb']:.1f}MB | "
            f"gpu_res={s['gpu_reserved_mb']:.1f}MB | "
            f"queue={s['queue_size']}/{s['queue_capacity']}"
        )
        logger.info(msg)

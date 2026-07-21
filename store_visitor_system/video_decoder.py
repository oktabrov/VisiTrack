"""
Video decoder with NVIDIA NVDEC GPU decoding and CPU fallback.

Provides two implementations:
  1. NVDECDecoder — GPU-based decoding via FFmpeg subprocess with h264_cuvid / hevc_cuvid
  2. CPUDecoder  — OpenCV-based CPU decoding (fallback)

Both are wrapped by ``VideoCapture`` which runs capture in a background thread
with a bounded queue, stale-frame dropping, and automatic reconnection.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)


# ── Abstract base ────────────────────────────────────────────────────────


class _DecoderBase(ABC):
    """Interface for a video decoder that produces raw BGR numpy frames."""

    @abstractmethod
    def open(self, url: str) -> bool:
        """Open the video source. Returns True on success."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one frame. Returns (success, frame_bgr | None)."""

    @abstractmethod
    def release(self) -> None:
        """Release resources."""

    @abstractmethod
    def is_opened(self) -> bool:
        """True if the source is currently open."""

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def backend_name(self) -> str: ...


# ── NVDEC GPU decoder ───────────────────────────────────────────────────


class NVDECDecoder(_DecoderBase):
    """GPU-accelerated video decoder using FFmpeg subprocess + NVDEC.

    Supports ``h264_cuvid`` and ``hevc_cuvid`` hardware decoders.
    Frames are decoded by the GPU, converted to raw BGR, and piped to Python.
    """

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._process: Optional[subprocess.Popen] = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0
        self._url: str = ""
        self._frame_size: int = 0

    # ── Static check ─────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        """Check if FFmpeg has NVDEC (cuvid) support."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-decoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            has_h264 = "h264_cuvid" in result.stdout
            has_hevc = "hevc_cuvid" in result.stdout
            if has_h264 or has_hevc:
                logger.info(
                    "NVDEC available — h264_cuvid: %s, hevc_cuvid: %s",
                    has_h264,
                    has_hevc,
                )
                return True
            logger.info("NVDEC decoders not found in FFmpeg build.")
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("FFmpeg not found — NVDEC unavailable.")
            return False

    # ── Interface implementation ─────────────────────────────────────

    def open(self, url: str) -> bool:
        self._url = url
        # First probe the stream to get resolution/fps
        if not self._probe_stream(url):
            return False
        # Start the decode process
        return self._start_ffmpeg(url)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._process is None or self._process.poll() is not None:
            return False, None
        try:
            raw = self._process.stdout.read(self._frame_size)  # type: ignore[union-attr]
            if len(raw) != self._frame_size:
                return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self._height, self._width, 3)
            )
            return True, frame
        except Exception as exc:
            logger.debug("NVDEC read error: %s", exc)
            return False, None

    def release(self) -> None:
        if self._process is not None:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass
            self._process = None

    def is_opened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def backend_name(self) -> str:
        return "NVIDIA NVDEC"

    # ── Private helpers ──────────────────────────────────────────────

    def _probe_stream(self, url: str) -> bool:
        """Use ffprobe to determine stream resolution and frame rate."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-rtsp_transport", "tcp",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,r_frame_rate",
                    "-of", "csv=p=0",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                self._width = int(parts[0])
                self._height = int(parts[1])
                # r_frame_rate is like "30/1"
                fps_parts = parts[2].split("/")
                if len(fps_parts) == 2 and int(fps_parts[1]) != 0:
                    self._fps = int(fps_parts[0]) / int(fps_parts[1])
                else:
                    self._fps = 25.0
                self._frame_size = self._width * self._height * 3
                logger.info(
                    "Stream probed: %dx%d @ %.1f fps", self._width, self._height, self._fps
                )
                return True
            logger.warning("Could not parse ffprobe output: %s", result.stdout)
            return False
        except Exception as exc:
            logger.warning("ffprobe failed: %s", exc)
            return False

    def _start_ffmpeg(self, url: str) -> bool:
        """Launch the FFmpeg decode subprocess with NVDEC."""
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-hwaccel_device", str(self._gpu_index),
            "-hwaccel_output_format", "cuda",
            "-c:v", "h264_cuvid",
            "-rtsp_transport", "tcp",
            "-i", url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",                      # no audio
            "-sn",                      # no subtitles
            "-v", "warning",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 4,
            )
            # Give it a moment to fail (e.g. bad codec)
            time.sleep(0.5)
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace")  # type: ignore[union-attr]
                logger.warning("FFmpeg NVDEC exited early: %s", stderr[:500])
                # Try HEVC fallback
                return self._start_ffmpeg_hevc(url)
            logger.info("NVDEC decoder started (h264_cuvid).")
            return True
        except FileNotFoundError:
            logger.warning("FFmpeg binary not found.")
            return False

    def _start_ffmpeg_hevc(self, url: str) -> bool:
        """Retry with hevc_cuvid for HEVC/H.265 streams."""
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-hwaccel_device", str(self._gpu_index),
            "-hwaccel_output_format", "cuda",
            "-c:v", "hevc_cuvid",
            "-rtsp_transport", "tcp",
            "-i", url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an", "-sn",
            "-v", "warning",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 4,
            )
            time.sleep(0.5)
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace")  # type: ignore[union-attr]
                logger.warning("FFmpeg NVDEC HEVC also failed: %s", stderr[:500])
                return False
            logger.info("NVDEC decoder started (hevc_cuvid).")
            return True
        except FileNotFoundError:
            return False


# ── CPU decoder (OpenCV fallback) ────────────────────────────────────────


class CPUDecoder(_DecoderBase):
    """Standard CPU-based video decoder using OpenCV + FFmpeg backend."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0

    def open(self, url: str) -> bool:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            logger.warning("OpenCV CPU decoder failed to open: %s", url)
            return False
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        logger.info(
            "CPU decoder opened: %dx%d @ %.1f fps",
            self._width, self._height, self._fps,
        )
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def backend_name(self) -> str:
        return "OpenCV CPU"


# ── VideoCapture wrapper ─────────────────────────────────────────────────


class VideoCapture:
    """High-level video capture with bounded-queue buffering and reconnection.

    Features:
      - Runs capture in a dedicated daemon thread.
      - Bounded queue (default maxsize=2) drops stale frames.
      - Automatic reconnection with exponential backoff.
      - Prefers NVDEC GPU decoding; falls back to CPU if unavailable.
    """

    def __init__(
        self,
        config: "Config",
        perf_monitor: Optional["PerformanceMonitor"] = None,
    ) -> None:
        self._config = config
        self._perf = perf_monitor
        self._url = config.rtsp_url
        self._decoder: Optional[_DecoderBase] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=config.queue_max_size
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend_name: str = "unknown"

    # ── Public API ───────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._decoder:
            return self._decoder.width, self._decoder.height
        return 0, 0

    @property
    def stream_fps(self) -> float:
        return self._decoder.fps if self._decoder else 0.0

    def start(self) -> None:
        """Initialize the decoder and start the capture thread."""
        if self._thread is not None:
            return

        self._decoder = self._create_decoder()
        # _create_decoder may return an already-opened NVDEC decoder,
        # or an un-opened CPUDecoder. Open only if not already opened.
        if self._decoder is None:
            raise RuntimeError(
                f"Failed to open video source: {self._url}. "
                "Check the RTSP URL and network connectivity."
            )
        if not self._decoder.is_opened():
            if not self._decoder.open(self._url):
                raise RuntimeError(
                    f"Failed to open video source: {self._url}. "
                    "Check the RTSP URL and network connectivity."
                )
        self._backend_name = self._decoder.backend_name
        logger.info(
            "Video capture started — backend: %s, resolution: %dx%d, fps: %.1f",
            self._backend_name,
            self._decoder.width,
            self._decoder.height,
            self._decoder.fps,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="video-capture"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and release resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        if self._decoder is not None:
            self._decoder.release()
            self._decoder = None
        logger.info("Video capture stopped.")

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Return the most recent frame, or ``None`` if the queue is empty.

        This method is called from the inference worker thread.
        """
        try:
            frame = self._queue.get(timeout=timeout)
            if self._perf:
                self._perf.update_queue_size(self._queue.qsize())
            return frame
        except queue.Empty:
            return None

    # ── Private ──────────────────────────────────────────────────────

    def _create_decoder(self) -> Optional[_DecoderBase]:
        """Select the best available decoder."""
        if self._config.use_gpu_decoding and NVDECDecoder.is_available():
            logger.info("Attempting NVDEC GPU video decoding.")
            decoder = NVDECDecoder(gpu_index=self._config.cuda_device_index)
            if decoder.open(self._url):
                return decoder
            logger.warning(
                "NVDEC initialization failed — falling back to CPU decoding."
            )
            decoder.release()

        logger.info("Using CPU video decoding (OpenCV).")
        return CPUDecoder()

    def _capture_loop(self) -> None:
        """Background loop: read frames and push to the bounded queue."""
        reconnect_delay = self._config.reconnect_delay

        while not self._stop_event.is_set():
            if self._decoder is None or not self._decoder.is_opened():
                self._reconnect(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 2, self._config.max_reconnect_delay
                )
                continue

            ret, frame = self._decoder.read()
            if not ret or frame is None:
                logger.warning("Frame read failed — attempting reconnect.")
                self._decoder.release()
                continue

            # Reset backoff on successful read
            reconnect_delay = self._config.reconnect_delay

            if self._perf:
                self._perf.capture_fps.tick()
                self._perf.record_total_frame()

            # Bounded queue: drop oldest if full
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    if self._perf:
                        self._perf.record_dropped_frame()
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                # Extremely rare race — drop this frame
                if self._perf:
                    self._perf.record_dropped_frame()

            if self._perf:
                self._perf.update_queue_size(self._queue.qsize())

    def _reconnect(self, delay: float) -> None:
        """Try to reconnect after a stream failure."""
        logger.info("Reconnecting in %.1f s …", delay)

        try:
            from .web_server import update_pipeline_status
            update_pipeline_status("CONNECTING")
        except Exception:
            pass

        self._stop_event.wait(timeout=delay)
        if self._stop_event.is_set():
            return

        if self._decoder is not None:
            self._decoder.release()

        self._decoder = self._create_decoder()
        if self._decoder is None:
            logger.error("Could not create a decoder during reconnection.")
            try:
                from .web_server import update_pipeline_status
                update_pipeline_status("ERROR")
            except Exception:
                pass
            return
        # _create_decoder opens NVDEC internally but not CPUDecoder
        if not self._decoder.is_opened():
            if not self._decoder.open(self._url):
                logger.error("Decoder failed to open during reconnection.")
                self._decoder = None
                try:
                    from .web_server import update_pipeline_status
                    update_pipeline_status("ERROR")
                except Exception:
                    pass
                return

        # Successfully reconnected
        try:
            from .web_server import update_pipeline_status
            update_pipeline_status(
                "STREAMING",
                resolution=f"{self._decoder.width}x{self._decoder.height}",
                backend=self._decoder.backend_name,
            )
        except Exception:
            pass

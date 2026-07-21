"""
Main inference pipeline orchestrator.

Connects video capture, person detection (RF-DETR), face processing,
person ReID, and tracking into a real-time processing loop.

Architecture:
  - Video capture runs in a dedicated thread (inside VideoCapture).
  - Inference runs in the main thread or a dedicated worker thread.
  - A bounded queue (maxsize=2) ensures we always process the newest frame.
  - Stale frames are dropped to maintain real-time processing.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from .config import Config
    from .detector import Detection
    from .gpu import GPUManager

from .detector import PersonDetector
from .face_processor import FaceProcessor
from .performance import PerformanceMonitor
from .reid import PersonReID
from .tracker import Tracker
from .video_decoder import VideoCapture

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Real-time inference pipeline for the store visitor system.

    Orchestrates:
      1. Frame acquisition from the bounded queue (drop stale frames).
      2. RF-DETR person detection.
      3. Person crop extraction.
      4. Face detection + embedding (batched).
      5. ReID feature extraction (batched).
      6. Multi-object tracking with feature association.
      7. Event logging (new person, re-identification).
      8. Performance metric recording.
    """

    def __init__(
        self,
        gpu_manager: "GPUManager",
        config: "Config",
        db_manager: Optional["DatabaseManager"] = None,
    ) -> None:
        self._gpu = gpu_manager
        self._config = config
        self._db = db_manager

        # Performance monitor
        self._perf = PerformanceMonitor(
            gpu_manager=gpu_manager,
            log_interval=config.perf_log_interval,
            queue_max_size=config.queue_max_size,
        )

        # AI models (loaded on CUDA)
        self._detector = PersonDetector(gpu_manager, config, self._perf)
        self._face_proc = FaceProcessor(gpu_manager, config, self._perf)
        self._reid = PersonReID(gpu_manager, config, self._perf)

        # Tracker
        self._tracker = Tracker(
            max_age=config.track_max_age,
            min_hits=config.track_min_hits,
            iou_threshold=config.track_iou_threshold,
        )

        # Video capture
        self._video = VideoCapture(config, self._perf)

        # Control
        self._stop_event = threading.Event()
        self._frame_counter: int = 0

    # ── Public API ───────────────────────────────────────────────────

    def run(self) -> None:
        """Start the pipeline and block until interrupted."""
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            self._video.start()
            self._perf.start()

            logger.info(
                "Pipeline started — processing RTSP stream: %s",
                self._config.rtsp_url,
            )
            logger.info(
                "Video backend: %s | Frame skip: %d | Queue: %d",
                self._video.backend_name,
                self._config.frame_skip,
                self._config.queue_max_size,
            )

            self._inference_loop()

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except Exception as exc:
            logger.error("Pipeline fatal error: %s", exc, exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully stop all components."""
        logger.info("Shutting down pipeline …")
        self._stop_event.set()
        self._perf.stop()
        self._video.stop()
        logger.info("Pipeline shutdown complete.")

    # ── Inference loop ───────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """Main loop: pull frames from the queue and process them."""
        while not self._stop_event.is_set():
            frame = self._video.get_frame(timeout=1.0)
            if frame is None:
                continue

            self._frame_counter += 1

            # Frame skipping
            if (
                self._config.frame_skip > 0
                and (self._frame_counter % (self._config.frame_skip + 1)) != 0
            ):
                continue

            try:
                with self._perf.measure_pipeline():
                    self._process_frame(frame)
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU OOM in inference loop! Attempting recovery …")
                self._gpu.emergency_cleanup()
                time.sleep(0.5)  # brief cooldown
            except Exception as exc:
                logger.error(
                    "Error processing frame %d: %s",
                    self._frame_counter,
                    exc,
                    exc_info=True,
                )

    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame through the full pipeline."""
        # Step 1: Person detection
        with self._perf.measure_detection():
            detections = self._detector.detect(frame)

        if not detections:
            self._tracker.update([])
            return

        # Step 2: Crop persons from the frame
        person_crops = self._extract_crops(frame, detections)

        # Step 3: Face detection + embedding (batched)
        face_embeddings_per_person: List[Optional[np.ndarray]] = []
        with self._perf.measure_face():
            face_results = self._face_proc.process_crops(person_crops)
            for results in face_results:
                if results and results[0].embedding is not None:
                    face_embeddings_per_person.append(results[0].embedding)
                else:
                    face_embeddings_per_person.append(None)

        # Step 4: ReID feature extraction (batched)
        reid_features: Optional[np.ndarray] = None
        if person_crops:
            with self._perf.measure_reid():
                reid_features = self._reid.extract_features(person_crops)

        # Step 5: Update tracker
        tracks = self._tracker.update(
            detections,
            reid_features=reid_features,
            face_embeddings=face_embeddings_per_person,
        )

        # Step 6: Log events
        self._log_events(tracks, detections)

    def _extract_crops(
        self, frame: np.ndarray, detections: List["Detection"]
    ) -> List[np.ndarray]:
        """Extract person crop images from the frame using detection bboxes."""
        h, w = frame.shape[:2]
        crops = []
        for det in detections:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(w, int(det.x2))
            y2 = min(h, int(det.y2))

            if x2 <= x1 or y2 <= y1:
                # Degenerate box — provide a minimal crop
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
                continue

            crop = frame[y1:y2, x1:x2].copy()
            crops.append(crop)
        return crops

    def _log_events(
        self,
        tracks: list,
        detections: List["Detection"],
    ) -> None:
        """Log notable tracking events and record visits in database."""
        for track in tracks:
            visitor_id = f"VISITOR-{track.track_id:03d}"

            if track.hits == 1:
                logger.info(
                    "🟢 New person detected — Track #%d (%s, conf=%.2f)",
                    track.track_id,
                    visitor_id,
                    track.confidence,
                )
                if self._db is not None:
                    self._db.record_visit(
                        visitor_id=visitor_id,
                        confidence=track.confidence,
                    )

            if track.is_confirmed and track.hits == track.age:
                # Just confirmed
                if track.face_embedding is not None:
                    logger.info(
                        "👤 Track #%d (%s) confirmed with face embedding.",
                        track.track_id,
                        visitor_id,
                    )

    # ── Signal handling ──────────────────────────────────────────────

    def _signal_handler(self, signum, frame) -> None:
        """Handle SIGINT / SIGTERM for graceful shutdown."""
        logger.info("Signal %d received — initiating shutdown.", signum)
        self._stop_event.set()

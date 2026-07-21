"""
RF-DETR person detector — runs entirely on CUDA.

The model is loaded once at startup, kept in GPU memory permanently,
and warmed up before processing any RTSP frames.  Inference uses
``torch.inference_mode`` + ``torch.autocast`` for mixed-precision FP16.

Only final detection results (boxes, scores) are transferred back to CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import Config
    from .gpu import GPUManager
    from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single person detection result (coordinates in pixel space)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0  # 0 = person in COCO

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)


class PersonDetector:
    """RF-DETR-based person detector with full CUDA acceleration.

    Usage::

        detector = PersonDetector(gpu_manager, config)
        detections = detector.detect(frame_bgr)
    """

    # COCO class ID for 'person'
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        gpu_manager: "GPUManager",
        config: "Config",
        perf_monitor: Optional["PerformanceMonitor"] = None,
    ) -> None:
        self._gpu = gpu_manager
        self._config = config
        self._perf = perf_monitor
        self._batch_size = config.detection_batch_size
        self._model = None
        self._model_loaded = False

        self._load_model()

    def _load_model(self) -> None:
        """Load RF-DETR onto the CUDA device.

        RF-DETR manages its own CUDA placement internally via its
        ``ModelContext``.  When ``torch.cuda.is_available()`` is True,
        RF-DETR automatically runs inference on the GPU.  We do NOT
        call ``move_to_device`` because ``ModelContext`` is not an
        ``nn.Module``.
        """
        try:
            from rfdetr import RFDETRBase

            logger.info("Loading RF-DETR Base model …")

            if self._config.rfdetr_weights:
                self._model = RFDETRBase(pretrain_weights=self._config.rfdetr_weights)
            else:
                self._model = RFDETRBase()

            # RF-DETR uses CUDA automatically when available.
            # Verify by checking internal nn.Module parameters if accessible.
            self._verify_rfdetr_device()

            # Enable FP16 Tensor Core optimization for ~2-8x speedup
            # on Ampere+ GPUs (Compute >= 8.0).
            self._optimize_for_fp16()

            self._model_loaded = True
            logger.info("RF-DETR loaded — CUDA auto-managed by RF-DETR internals.")

            # Warm-up
            self._warmup()

        except ImportError:
            logger.error(
                "rfdetr package not installed. "
                "Install with: pip install rfdetr"
            )
            raise
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM while loading RF-DETR. Free GPU memory or reduce model size.")
            self._gpu.emergency_cleanup()
            raise

    def _optimize_for_fp16(self) -> None:
        """Enable FP16 Tensor Core inference if the GPU supports it.

        RF-DETR provides ``optimize_for_inference(dtype=torch.float16)``
        which fuses layers and converts weights to FP16, unlocking
        hardware Tensor Cores on Ampere+ GPUs for a ~2-8× speedup.
        """
        if not self._gpu.is_cuda:
            return

        try:
            if not hasattr(self._model, "optimize_for_inference"):
                logger.info(
                    "RF-DETR version does not expose optimize_for_inference(); "
                    "skipping FP16 optimization."
                )
                return

            logger.info("Applying RF-DETR FP16 Tensor Core optimization …")
            self._model.optimize_for_inference(dtype=torch.float16)
            logger.info(
                "RF-DETR FP16 optimization applied — "
                "expect ~2-8× inference speedup on Ampere+ Tensor Cores."
            )
        except Exception as exc:
            logger.warning(
                "RF-DETR FP16 optimization failed (non-fatal, "
                "falling back to FP32): %s",
                exc,
            )

    def _verify_rfdetr_device(self) -> None:
        """Log which device RF-DETR's internal model lives on."""
        try:
            # Try to find the actual nn.Module inside ModelContext
            inner = self._model
            if hasattr(inner, "model"):
                inner = inner.model
            # Walk one more level if needed
            if hasattr(inner, "model"):
                inner = inner.model
            # Check first parameter's device
            if hasattr(inner, "parameters"):
                param = next(inner.parameters(), None)
                if param is not None:
                    logger.info("RF-DETR internal model device: %s", param.device)
                    return
            logger.info(
                "RF-DETR device: auto-managed (cannot inspect ModelContext internals)"
            )
        except Exception:
            logger.info(
                "RF-DETR device: auto-managed (cannot inspect ModelContext internals)"
            )

    def _warmup(self) -> None:
        """Run a dummy inference to compile CUDA kernels."""
        logger.info("RF-DETR warm-up inference …")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            with self._gpu.inference_context():
                _ = self._model.predict(dummy, threshold=0.9)  # type: ignore[union-attr]
            if self._gpu.is_cuda:
                torch.cuda.synchronize(self._gpu.device)
            logger.info("RF-DETR warm-up complete.")
        except Exception as exc:
            logger.warning("RF-DETR warm-up failed (non-fatal): %s", exc)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run person detection on a BGR frame.

        Args:
            frame: H×W×3 uint8 BGR image (numpy array from the video decoder).

        Returns:
            List of ``Detection`` objects for persons above the confidence
            threshold.  All GPU work is done here; results are on CPU.
        """
        if not self._model_loaded or self._model is None:
            logger.error("RF-DETR model not loaded — skipping detection.")
            return []

        try:
            with self._gpu.inference_context():
                results = self._model.predict(
                    frame, threshold=self._config.detection_confidence
                )

            return self._parse_results(results)

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM during RF-DETR inference!")
            self._gpu.emergency_cleanup()
            self._handle_oom()
            return []
        except Exception as exc:
            logger.error("RF-DETR inference error: %s", exc, exc_info=True)
            return []

    def _parse_results(self, results) -> List[Detection]:
        """Convert RF-DETR output to a list of Detection objects.

        RF-DETR returns a ``supervision.Detections`` object with:
          - .xyxy: (N, 4) array of bounding boxes
          - .confidence: (N,) array of confidence scores
          - .class_id: (N,) array of class IDs
        """
        detections: List[Detection] = []
        if results is None:
            return detections

        # Handle supervision.Detections format
        xyxy = getattr(results, "xyxy", None)
        confidence = getattr(results, "confidence", None)
        class_id = getattr(results, "class_id", None)

        if xyxy is None or len(xyxy) == 0:
            return detections

        # Convert to numpy if tensors
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()
        if isinstance(class_id, torch.Tensor):
            class_id = class_id.cpu().numpy()

        for i in range(len(xyxy)):
            cid = int(class_id[i]) if class_id is not None else 0
            if cid != self.PERSON_CLASS_ID:
                continue
            conf = float(confidence[i]) if confidence is not None else 1.0
            box = xyxy[i]
            detections.append(
                Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=conf,
                    class_id=cid,
                )
            )

        return detections

    def _handle_oom(self) -> None:
        """Attempt recovery from GPU OOM by reducing batch size."""
        if self._batch_size > 1:
            self._batch_size = max(1, self._batch_size // 2)
            logger.warning(
                "Reduced detection batch size to %d after OOM.", self._batch_size
            )

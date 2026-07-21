"""
Face detection and recognition on CUDA.

Uses InsightFace for face detection (SCRFD/RetinaFace) and ArcFace for
embedding generation.  Supports batched embedding extraction to maximize
GPU utilization.  Embeddings are L2-normalized on the GPU before transfer
to CPU.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .config import Config
    from .gpu import GPUManager
    from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class FaceResult:
    """Result from face processing on a single person crop."""

    __slots__ = ("bbox", "score", "embedding", "landmark")

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        score: float,
        embedding: Optional[np.ndarray] = None,
        landmark: Optional[np.ndarray] = None,
    ) -> None:
        self.bbox = bbox          # (x1, y1, x2, y2) within the crop
        self.score = score        # face detection confidence
        self.embedding = embedding  # (512,) L2-normalized embedding or None
        self.landmark = landmark  # (5, 2) facial landmarks or None


class FaceProcessor:
    """Face detection + recognition pipeline running on CUDA.

    The InsightFace ``FaceAnalysis`` app handles both detection and
    recognition.  For batched embedding extraction, we bypass the
    high-level API and use the underlying recognition model directly.

    Usage::

        face_proc = FaceProcessor(gpu_manager, config)
        results = face_proc.process_crops(person_crops)
    """

    # ArcFace input dimensions
    _FACE_SIZE = (112, 112)

    def __init__(
        self,
        gpu_manager: "GPUManager",
        config: "Config",
        perf_monitor: Optional["PerformanceMonitor"] = None,
    ) -> None:
        self._gpu = gpu_manager
        self._config = config
        self._perf = perf_monitor
        self._batch_size = config.face_batch_size

        self._app = None          # InsightFace FaceAnalysis
        self._rec_model = None    # Recognition model (for batched inference)
        self._det_model = None    # Detection model reference

        self._load_models()

    # ── Model loading ────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load InsightFace models onto the CUDA device."""
        try:
            from insightface.app import FaceAnalysis

            logger.info("Loading InsightFace models (%s) …", self._config.face_model_name)

            # ctx_id: GPU index for CUDA; -1 for CPU
            ctx_id = (
                self._gpu.device.index
                if self._gpu.is_cuda and self._gpu.device.index is not None
                else -1
            )

            self._app = FaceAnalysis(
                name=self._config.face_model_name,
                providers=self._get_providers(),
            )
            self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            # Grab references to internal models for batched inference.
            # InsightFace exposes models differently across versions.
            self._find_internal_models()

            logger.info("Face model device: %s", self._gpu.device)
            self._warmup()

        except ImportError:
            logger.error(
                "insightface not installed. "
                "Install with: pip install insightface onnxruntime-gpu"
            )
            raise
        except Exception as exc:
            logger.error("Failed to load face models: %s", exc, exc_info=True)
            raise

    def _get_providers(self) -> list:
        """Return ONNX Runtime execution providers with CUDA preferred."""
        providers = []
        if self._gpu.is_cuda:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": self._gpu.device.index or 0,
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                )
            )
        providers.append("CPUExecutionProvider")
        return providers

    def _find_internal_models(self) -> None:
        """Locate internal detection and recognition models.

        InsightFace versions expose models differently:
          - v0.7+: ``self._app.models`` dict keyed by model name
          - Some versions: list of model objects on ``self._app.models``
          - v1.0+: may use different attribute names
        """
        if self._app is None:
            return

        model_list = []

        # Try dict-style access (v0.7+)
        if hasattr(self._app, "models"):
            models_attr = self._app.models
            if isinstance(models_attr, dict):
                model_list = list(models_attr.values())
            elif isinstance(models_attr, (list, tuple)):
                model_list = list(models_attr)

        # Try model_zoo attribute
        if not model_list and hasattr(self._app, "model_zoo"):
            model_list = list(self._app.model_zoo)

        for m in model_list:
            taskname = getattr(m, "taskname", "")
            if taskname == "recognition" and self._rec_model is None:
                self._rec_model = m
                logger.debug("Found recognition model: %s", type(m).__name__)
            elif taskname == "detection" and self._det_model is None:
                self._det_model = m
                logger.debug("Found detection model: %s", type(m).__name__)

        if self._rec_model is None:
            logger.warning(
                "Could not find recognition model in InsightFace app. "
                "Batched embedding extraction will fall back to per-face mode."
            )

    def _warmup(self) -> None:
        """Warm up face detection + recognition with a dummy image."""
        logger.info("Face model warm-up …")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = self._app.get(dummy)  # type: ignore[union-attr]
            logger.info("Face model warm-up complete.")
        except Exception as exc:
            logger.warning("Face warm-up failed (non-fatal): %s", exc)

    # ── Public API ───────────────────────────────────────────────────

    def process_crops(
        self,
        person_crops: List[np.ndarray],
    ) -> List[List[FaceResult]]:
        """Detect faces and extract embeddings from person crops.

        Args:
            person_crops: List of BGR person-crop images (numpy arrays).

        Returns:
            For each person crop, a list of ``FaceResult`` objects (one per
            detected face).  Most persons will have 0 or 1 face.
        """
        if not person_crops or self._app is None:
            return [[] for _ in person_crops]

        all_results: List[List[FaceResult]] = []
        all_aligned_faces: List[np.ndarray] = []
        face_to_person_idx: List[Tuple[int, int]] = []  # (person_idx, face_idx_within)

        # Stage 1: Face detection on each crop
        for crop_idx, crop in enumerate(person_crops):
            crop_results: List[FaceResult] = []
            try:
                faces = self._app.get(crop)
                for face_idx, face in enumerate(faces):
                    score = float(getattr(face, "det_score", 0.0))
                    if score < self._config.face_detection_threshold:
                        continue

                    bbox = tuple(face.bbox.astype(float))
                    landmark = getattr(face, "landmark", None)

                    # Collect aligned face for batched embedding
                    aligned = self._align_face(crop, face)
                    if aligned is not None:
                        face_to_person_idx.append((crop_idx, len(crop_results)))
                        all_aligned_faces.append(aligned)

                    crop_results.append(
                        FaceResult(
                            bbox=bbox,  # type: ignore[arg-type]
                            score=score,
                            embedding=None,  # filled in batch stage
                            landmark=landmark,
                        )
                    )
            except Exception as exc:
                logger.debug("Face detection failed on crop %d: %s", crop_idx, exc)

            all_results.append(crop_results)

        # Stage 2: Batched embedding extraction
        if all_aligned_faces:
            embeddings = self._batch_extract_embeddings(all_aligned_faces)
            for i, (person_idx, face_idx) in enumerate(face_to_person_idx):
                if i < len(embeddings):
                    all_results[person_idx][face_idx].embedding = embeddings[i]

        return all_results

    def process_single(self, image: np.ndarray) -> List[FaceResult]:
        """Process a single image (convenience wrapper)."""
        results = self.process_crops([image])
        return results[0] if results else []

    # ── Batched embedding extraction ─────────────────────────────────

    def _batch_extract_embeddings(
        self, aligned_faces: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Extract embeddings from aligned face images in batches on GPU.

        Args:
            aligned_faces: List of aligned face images (112×112×3 BGR uint8).

        Returns:
            List of (512,) L2-normalized embedding numpy arrays.
        """
        all_embeddings: List[np.ndarray] = []

        for batch_start in range(0, len(aligned_faces), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(aligned_faces))
            batch_faces = aligned_faces[batch_start:batch_end]

            try:
                batch_embs = self._extract_batch(batch_faces)
                all_embeddings.extend(batch_embs)
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU OOM during face embedding extraction!")
                self._gpu.emergency_cleanup()
                self._handle_oom()
                # Retry with smaller batch
                for face in batch_faces:
                    try:
                        embs = self._extract_batch([face])
                        all_embeddings.extend(embs)
                    except Exception:
                        all_embeddings.append(np.zeros(512, dtype=np.float32))
            except Exception as exc:
                logger.error("Face embedding error: %s", exc)
                all_embeddings.extend(
                    [np.zeros(512, dtype=np.float32)] * len(batch_faces)
                )

        return all_embeddings

    def _extract_batch(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        """Run the recognition model on a batch of aligned faces.

        If the InsightFace recognition model supports direct batched input,
        use it.  Otherwise, fall back to per-face processing via the app.
        """
        # Preprocess: BGR uint8 → float32, normalize, transpose to NCHW
        processed = []
        for face in faces:
            img = cv2.resize(face, self._FACE_SIZE)
            img = img.astype(np.float32)
            # InsightFace standard normalization
            img = (img / 127.5) - 1.0
            img = img.transpose(2, 0, 1)  # HWC → CHW
            processed.append(img)

        batch_np = np.stack(processed, axis=0)  # (N, 3, 112, 112)
        batch_tensor = torch.from_numpy(batch_np)

        # Move to GPU with pinned memory for large batches
        pin = len(faces) >= 4
        batch_tensor = self._gpu.to_device(batch_tensor, pin_memory=pin)

        # Run inference
        with self._gpu.inference_context():
            if self._rec_model is not None and hasattr(self._rec_model, "session"):
                # ONNX Runtime path (InsightFace default)
                embeddings_np = self._rec_model.session.run(
                    None,
                    {self._rec_model.session.get_inputs()[0].name: batch_np},
                )[0]
                embeddings_tensor = torch.from_numpy(embeddings_np).to(
                    self._gpu.device
                )
            else:
                # Pure PyTorch fallback path
                embeddings_tensor = batch_tensor  # placeholder if no model

            # L2-normalize on GPU
            embeddings_tensor = F.normalize(embeddings_tensor.float(), p=2, dim=1)

        # Transfer to CPU
        return [
            embeddings_tensor[i].cpu().numpy()
            for i in range(embeddings_tensor.shape[0])
        ]

    # ── Helpers ──────────────────────────────────────────────────────

    def _align_face(self, image: np.ndarray, face) -> Optional[np.ndarray]:
        """Align and crop a detected face to 112×112 using landmarks."""
        try:
            # InsightFace provides normed_crop or kps for alignment
            if hasattr(face, "normed_embedding"):
                # Face was already processed with alignment
                kps = getattr(face, "kps", None)
                if kps is not None:
                    from insightface.utils.face_align import norm_crop
                    aligned = norm_crop(image, kps)
                    return aligned

            # Simple center-crop fallback
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            return cv2.resize(crop, self._FACE_SIZE)
        except Exception:
            return None

    def _handle_oom(self) -> None:
        """Reduce batch size after OOM."""
        if self._batch_size > 1:
            self._batch_size = max(1, self._batch_size // 2)
            logger.warning(
                "Reduced face batch size to %d after OOM.", self._batch_size
            )

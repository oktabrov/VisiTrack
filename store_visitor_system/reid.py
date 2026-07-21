"""
OSNet person re-identification on CUDA.

Loads the OSNet model once at startup, keeps it in GPU memory for the
full application lifetime, and processes person crops in configurable
batches with FP16 mixed precision.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

if TYPE_CHECKING:
    from .config import Config
    from .gpu import GPUManager
    from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)

# Standard ReID preprocessing (ImageNet normalization, 256×128 input)
_REID_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

_REID_INPUT_SIZE = (256, 128)  # (height, width)


class PersonReID:
    """OSNet-based person re-identification with CUDA acceleration.

    Usage::

        reid = PersonReID(gpu_manager, config)
        features = reid.extract_features(person_crops)
        # features: (N, feature_dim) numpy array, L2-normalized
    """

    def __init__(
        self,
        gpu_manager: "GPUManager",
        config: "Config",
        perf_monitor: Optional["PerformanceMonitor"] = None,
    ) -> None:
        self._gpu = gpu_manager
        self._config = config
        self._perf = perf_monitor
        self._batch_size = config.reid_batch_size
        self._model = None
        self._feature_dim: int = 512

        self._load_model()

    # ── Model loading ────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load OSNet onto the CUDA device."""
        try:
            from torchreid import models as torchreid_models

            model_name = self._config.reid_model_name
            logger.info("Loading ReID model: %s …", model_name)

            self._model = torchreid_models.build_model(
                name=model_name,
                num_classes=1000,  # pretrained on ImageNet/Market classes
                pretrained=True,
            )

            # Move to device, set eval, optionally compile
            self._model = self._gpu.move_to_device(
                self._model, model_name=f"ReID ({model_name})"
            )

            # Determine feature dimensionality
            self._probe_feature_dim()

            logger.info(
                "ReID model device: %s | feature_dim: %d",
                self._gpu.device,
                self._feature_dim,
            )

            # Warm-up
            self._warmup()

        except ImportError:
            logger.error(
                "torchreid not installed. Install with:\n"
                "  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            )
            raise
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM loading ReID model.")
            self._gpu.emergency_cleanup()
            raise

    def _probe_feature_dim(self) -> None:
        """Run a dummy forward pass to discover the output feature dimension."""
        dummy = torch.randn(1, 3, *_REID_INPUT_SIZE, device=self._gpu.device)
        with self._gpu.inference_context():
            try:
                out = self._model(dummy)  # type: ignore[misc]
                if isinstance(out, torch.Tensor):
                    self._feature_dim = out.shape[-1]
                elif isinstance(out, (list, tuple)):
                    self._feature_dim = out[0].shape[-1]
            except Exception:
                self._feature_dim = 512  # safe default

    def _warmup(self) -> None:
        """Warm-up inference to compile CUDA kernels."""
        logger.info("ReID warm-up inference …")
        dummy = torch.randn(
            min(4, self._batch_size), 3, *_REID_INPUT_SIZE,
            device=self._gpu.device,
        )
        with self._gpu.inference_context():
            try:
                _ = self._model(dummy)  # type: ignore[misc]
                if self._gpu.is_cuda:
                    torch.cuda.synchronize(self._gpu.device)
                logger.info("ReID warm-up complete.")
            except Exception as exc:
                logger.warning("ReID warm-up failed (non-fatal): %s", exc)

    # ── Public API ───────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vectors."""
        return self._feature_dim

    def extract_features(
        self, person_crops: List[np.ndarray]
    ) -> np.ndarray:
        """Extract L2-normalized ReID features from person crops.

        Args:
            person_crops: List of BGR person-crop images (numpy uint8).

        Returns:
            ``(N, feature_dim)`` numpy array of L2-normalized feature vectors.
        """
        if not person_crops or self._model is None:
            return np.empty((0, self._feature_dim), dtype=np.float32)

        all_features: List[torch.Tensor] = []

        for batch_start in range(0, len(person_crops), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(person_crops))
            batch_crops = person_crops[batch_start:batch_end]

            try:
                features = self._extract_batch(batch_crops)
                all_features.append(features)
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU OOM during ReID feature extraction!")
                self._gpu.emergency_cleanup()
                self._handle_oom()
                # Retry one-by-one
                for crop in batch_crops:
                    try:
                        feat = self._extract_batch([crop])
                        all_features.append(feat)
                    except Exception:
                        all_features.append(
                            torch.zeros(1, self._feature_dim)
                        )
            except Exception as exc:
                logger.error("ReID extraction error: %s", exc)
                all_features.append(
                    torch.zeros(len(batch_crops), self._feature_dim)
                )

        if not all_features:
            return np.empty((0, self._feature_dim), dtype=np.float32)

        return torch.cat(all_features, dim=0).numpy()

    # ── Private ──────────────────────────────────────────────────────

    def _extract_batch(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Process a single batch of crops through the ReID model.

        Returns:
            ``(N, feature_dim)`` CPU tensor of L2-normalized features.
        """
        # Preprocess crops → tensor
        tensors = []
        for crop in crops:
            img = cv2.resize(crop, (_REID_INPUT_SIZE[1], _REID_INPUT_SIZE[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = _REID_TRANSFORM(img)  # (3, H, W) float32, normalized
            tensors.append(t)

        batch = torch.stack(tensors, dim=0)  # (N, 3, 256, 128)

        # Pin memory for larger batches, then transfer non-blocking
        batch = self._gpu.to_device(
            batch, pin_memory=(len(crops) >= 4), non_blocking=True
        )

        with self._gpu.inference_context():
            output = self._model(batch)  # type: ignore[misc]
            # Some torchreid models return a tuple (features, class_logits)
            if isinstance(output, (list, tuple)):
                features = output[0]
            else:
                features = output

            # Ensure 2D
            if features.dim() > 2:
                features = features.view(features.size(0), -1)

            # L2-normalize on GPU
            features = F.normalize(features.float(), p=2, dim=1)

        return features.cpu()

    def _handle_oom(self) -> None:
        """Reduce batch size after OOM."""
        if self._batch_size > 1:
            self._batch_size = max(1, self._batch_size // 2)
            logger.warning(
                "Reduced ReID batch size to %d after OOM.", self._batch_size
            )

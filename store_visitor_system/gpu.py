"""
GPU management module — the single source of truth for CUDA device selection.

Every other module imports the device from here. No independent device selection
is allowed anywhere else in the codebase.

Responsibilities:
  - CUDA availability validation
  - GPU selection (respects CUDA_VISIBLE_DEVICES and CUDA_DEVICE_INDEX)
  - Mixed-precision configuration
  - GPU information logging
  - Model-to-device helpers
  - GPU memory monitoring
  - Reusable inference context manager
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

# Fix Windows console encoding for Unicode banner output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import torch.nn as nn

from .config import Config

logger = logging.getLogger(__name__)


class GPUManager:
    """Centralized GPU / device manager.

    Create exactly one instance at application startup and pass it to every
    module that needs to place tensors or models on a device.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._device: Optional[torch.device] = None
        self._autocast_dtype: Optional[torch.dtype] = None
        self._gpu_name: str = "N/A"
        self._cuda_version: str = "N/A"
        self._pytorch_cuda_version: str = "N/A"
        self._total_memory_mb: float = 0.0
        self._compute_capability: tuple[int, int] = (0, 0)

    # ── Public properties ────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Return the selected torch device. Must call select_device() first."""
        if self._device is None:
            raise RuntimeError(
                "GPUManager.select_device() has not been called yet."
            )
        return self._device

    @property
    def autocast_dtype(self) -> torch.dtype:
        """Return the dtype to use with torch.autocast."""
        if self._autocast_dtype is None:
            self._autocast_dtype = self._determine_autocast_dtype()
        return self._autocast_dtype

    @property
    def is_cuda(self) -> bool:
        """Return True if the selected device is a CUDA device."""
        return self._device is not None and self._device.type == "cuda"

    # ── Startup sequence ─────────────────────────────────────────────────

    def validate_cuda(self) -> None:
        """Check CUDA availability.

        Raises ``RuntimeError`` when CUDA is not available and
        ``ALLOW_CPU_FALLBACK`` is ``False``.
        """
        if torch.cuda.is_available():
            logger.info("CUDA is available.")
            return

        if self._config.allow_cpu_fallback:
            logger.warning(
                "CUDA is NOT available. ALLOW_CPU_FALLBACK=true → "
                "falling back to CPU. AI inference will be very slow."
            )
        else:
            raise RuntimeError(
                "CUDA is NOT available and ALLOW_CPU_FALLBACK is disabled.\n"
                "Install a CUDA-enabled PyTorch build:\n"
                "  pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu121\n"
                "Or set ALLOW_CPU_FALLBACK=true to run on CPU (not recommended)."
            )

    def select_device(self) -> torch.device:
        """Select and return the target ``torch.device``.

        Respects ``CUDA_VISIBLE_DEVICES`` (handled by PyTorch/CUDA driver)
        and ``CUDA_DEVICE_INDEX`` from the configuration.
        """
        if torch.cuda.is_available() and self._config.device == "cuda":
            idx = self._config.cuda_device_index
            device_count = torch.cuda.device_count()
            if idx >= device_count:
                logger.warning(
                    "CUDA_DEVICE_INDEX=%d but only %d device(s) visible. "
                    "Falling back to cuda:0.",
                    idx,
                    device_count,
                )
                idx = 0
            self._device = torch.device(f"cuda:{idx}")
            torch.cuda.set_device(self._device)
            self._populate_gpu_info()
        elif self._config.allow_cpu_fallback:
            self._device = torch.device("cpu")
            logger.warning("Running on CPU — AI inference will be very slow.")
        else:
            raise RuntimeError(
                "Cannot select a CUDA device and CPU fallback is disabled."
            )

        # Apply GPU memory limit if configured
        if (
            self.is_cuda
            and self._config.gpu_memory_limit_mb > 0
        ):
            limit_bytes = self._config.gpu_memory_limit_mb * 1024 * 1024
            torch.cuda.set_per_process_memory_fraction(
                limit_bytes / (self._total_memory_mb * 1024 * 1024),
                self._device,
            )
            logger.info(
                "GPU memory limited to %d MB.", self._config.gpu_memory_limit_mb
            )

        return self._device

    def log_gpu_info(self) -> None:
        """Print a formatted banner with GPU details at startup."""
        if not self.is_cuda:
            logger.info("Running on CPU — no GPU information to display.")
            return

        banner = (
            "\n"
            "═══════════════════════════════════════════════════\n"
            "  VisiTrack — GPU Information\n"
            "═══════════════════════════════════════════════════\n"
            f"  GPU:              {self._gpu_name}\n"
            f"  CUDA Version:     {self._cuda_version}\n"
            f"  PyTorch CUDA:     {self._pytorch_cuda_version}\n"
            f"  GPU Memory:       {self._total_memory_mb / 1024:.1f} GB\n"
            f"  Device Index:     {self._device.index}\n"  # type: ignore[union-attr]
            f"  Compute Cap.:     {self._compute_capability[0]}.{self._compute_capability[1]}\n"
            f"  FP16 Support:     {self.supports_fp16}\n"
            f"  BF16 Support:     {self.supports_bf16}\n"
            "═══════════════════════════════════════════════════"
        )
        # Use print so it always appears even at WARNING log level.
        print(banner)
        logger.info("Selected device: %s", self._device)

    # ── Model helpers ────────────────────────────────────────────────────

    def move_to_device(
        self,
        model: nn.Module,
        *,
        compile_model: Optional[bool] = None,
        model_name: str = "model",
    ) -> nn.Module:
        """Move *model* to the selected device, set eval mode, optionally compile.

        Args:
            model: The PyTorch model to move.
            compile_model: Override for torch.compile. If ``None``, uses config.
            model_name: Human-readable name for logging.

        Returns:
            The model on the target device, in eval mode.
        """
        model = model.to(self._device)
        model.eval()
        logger.info("%s moved to %s", model_name, self._device)

        # Verify placement
        self._verify_model_device(model, model_name)

        # Optional torch.compile
        should_compile = (
            compile_model
            if compile_model is not None
            else self._config.enable_torch_compile
        )
        if should_compile and self.is_cuda:
            model = self._try_torch_compile(model, model_name)

        return model

    def warmup_model(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        model_name: str = "model",
    ) -> None:
        """Run a single warm-up inference to trigger CUDA kernel compilation.

        Args:
            model: The model to warm up.
            input_shape: Shape of the dummy input tensor (including batch dim).
            model_name: Human-readable name for logging.
        """
        logger.info("Warming up %s with input shape %s …", model_name, input_shape)
        dummy = torch.randn(*input_shape, device=self._device)
        with self.inference_context():
            try:
                _ = model(dummy)
            except Exception as exc:
                logger.warning(
                    "Warm-up for %s failed (non-fatal): %s", model_name, exc
                )
        # Free the dummy tensor immediately
        del dummy
        if self.is_cuda:
            torch.cuda.synchronize(self._device)
        logger.info("%s warm-up complete.", model_name)

    # ── Inference context ────────────────────────────────────────────────

    @contextmanager
    def inference_context(self) -> Generator[None, None, None]:
        """Context manager that combines ``torch.inference_mode`` and
        ``torch.autocast`` for safe, optimized inference.

        Usage::

            with gpu_manager.inference_context():
                outputs = model(inputs)
        """
        with torch.inference_mode():
            if self.is_cuda and self._config.use_half_precision:
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.autocast_dtype,
                    enabled=True,
                ):
                    yield
            else:
                yield

    # ── Memory monitoring ────────────────────────────────────────────────

    def get_memory_stats(self) -> Dict[str, float]:
        """Return current GPU memory statistics in MB.

        Returns an empty dict when running on CPU.
        """
        if not self.is_cuda:
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated(self._device) / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved(self._device) / (1024 ** 2),
            "max_allocated_mb": torch.cuda.max_memory_allocated(self._device)
            / (1024 ** 2),
        }

    def log_memory(self) -> None:
        """Log current GPU memory usage."""
        stats = self.get_memory_stats()
        if stats:
            logger.info(
                "GPU memory — allocated: %.1f MB | reserved: %.1f MB | peak: %.1f MB",
                stats["allocated_mb"],
                stats["reserved_mb"],
                stats["max_allocated_mb"],
            )

    def emergency_cleanup(self) -> None:
        """Release cached GPU memory. Only call during OOM recovery or model reload."""
        if self.is_cuda:
            torch.cuda.empty_cache()
            logger.warning("torch.cuda.empty_cache() called (emergency cleanup).")

    # ── Precision helpers ────────────────────────────────────────────────

    @property
    def supports_fp16(self) -> bool:
        """True if the GPU supports efficient FP16 (compute capability >= 7.0)."""
        return self._compute_capability >= (7, 0)

    @property
    def supports_bf16(self) -> bool:
        """True if the GPU supports BF16 (compute capability >= 8.0, Ampere+)."""
        return self._compute_capability >= (8, 0)

    @property
    def precision_label(self) -> str:
        """Human-readable label for the active mixed-precision mode."""
        if not self._config.use_half_precision:
            return "FP32"
        dt = self.autocast_dtype
        if dt == torch.bfloat16:
            return "BF16"
        if dt == torch.float16:
            return "FP16"
        return "FP32"

    # ── Tensor helpers ───────────────────────────────────────────────────

    def to_device(
        self,
        tensor: torch.Tensor,
        *,
        non_blocking: bool = True,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        """Move a tensor to the selected device.

        When *pin_memory* is ``True`` and the tensor lives on CPU, it is first
        pinned before the (non-blocking) transfer — useful for large batches.
        """
        if pin_memory and tensor.device.type == "cpu" and self.is_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(self._device, non_blocking=non_blocking)

    # ── Private helpers ──────────────────────────────────────────────────

    def _populate_gpu_info(self) -> None:
        """Fill GPU metadata fields."""
        assert self._device is not None and self._device.type == "cuda"
        idx = self._device.index or 0
        self._gpu_name = torch.cuda.get_device_name(idx)
        self._cuda_version = getattr(torch.version, "cuda", "N/A") or "N/A"
        self._pytorch_cuda_version = self._cuda_version  # same source in PyTorch
        props = torch.cuda.get_device_properties(idx)
        self._total_memory_mb = props.total_memory / (1024 ** 2)
        self._compute_capability = (props.major, props.minor)

    def _determine_autocast_dtype(self) -> torch.dtype:
        """Pick the best autocast dtype for the current GPU."""
        if not self._config.use_half_precision:
            return torch.float32

        # Check if user explicitly requested BF16 via env
        explicit = os.environ.get("AUTOCAST_DTYPE", "").strip().lower()
        if explicit == "bf16" and self.supports_bf16:
            return torch.bfloat16

        # Default: FP16 when supported, else FP32
        if self.supports_fp16:
            return torch.float16

        logger.warning(
            "GPU compute capability %s does not efficiently support FP16. "
            "Falling back to FP32.",
            self._compute_capability,
        )
        return torch.float32

    def _verify_model_device(self, model: nn.Module, name: str) -> None:
        """Log a warning if any model parameter is NOT on the expected device."""
        for param_name, param in model.named_parameters():
            if param.device != self._device:
                logger.error(
                    "DEVICE MISMATCH: %s parameter '%s' is on %s, expected %s",
                    name,
                    param_name,
                    param.device,
                    self._device,
                )
                return
        logger.debug("All parameters of %s verified on %s.", name, self._device)

    def _try_torch_compile(self, model: nn.Module, name: str) -> nn.Module:
        """Attempt ``torch.compile()`` on the model, falling back silently."""
        try:
            compiled = torch.compile(model)  # type: ignore[attr-defined]
            logger.info("torch.compile() applied to %s.", name)
            return compiled  # type: ignore[return-value]
        except Exception as exc:
            logger.warning(
                "torch.compile() failed for %s (non-fatal): %s. "
                "Continuing with eager-mode CUDA.",
                name,
                exc,
            )
            return model


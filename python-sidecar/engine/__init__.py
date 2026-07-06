"""Engine layer: pluggable inference/training backends.

Use ``make_engine(name, ...)`` to construct the right driver for the platform.
"""
from __future__ import annotations

import platform

from .base import (
    EngineCapabilities,
    EngineDriver,
    EngineError,
    TrainConfig,
    TrainResult,
)

__all__ = [
    "EngineDriver", "EngineCapabilities", "EngineError",
    "TrainConfig", "TrainResult", "make_engine", "auto_engine_name",
]


def auto_engine_name() -> str:
    """Pick a sensible default backend for the current machine."""
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mlx"
    return "llamacpp"


def make_engine(name: str, models_dir: str, adapters_dir: str, **kw) -> EngineDriver:
    """Factory. ``name`` in {'mlx','llamacpp','fake','auto'}."""
    if name == "auto":
        name = auto_engine_name()
    if name == "mlx":
        from .mlx_driver import MLXDriver
        return MLXDriver(models_dir, adapters_dir, **kw)
    if name == "llamacpp":
        from .llamacpp_driver import LlamaCppDriver
        return LlamaCppDriver(models_dir, adapters_dir, **kw)
    if name == "fake":
        from .fake_driver import FakeDriver
        return FakeDriver(models_dir, adapters_dir, **kw)
    raise EngineError(f"unknown engine: {name!r}")

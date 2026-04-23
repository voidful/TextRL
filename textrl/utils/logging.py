"""Logging helpers. TRL already wires ``report_to`` to WandB/TensorBoard/MLflow/etc."""
from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: str | int = "INFO") -> None:
    """Configure a sensible default logger for library use."""
    lvl = level if isinstance(level, int) else logging.getLevelName(level)
    logging.basicConfig(
        level=lvl,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def suppress_warnings() -> None:
    """Silence noisy warnings from transformers/accelerate unless TEXTRL_VERBOSE=1."""
    if os.environ.get("TEXTRL_VERBOSE"):
        return
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

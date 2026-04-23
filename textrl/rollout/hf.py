"""Default (no-op) rollout backend.

TRL handles generation internally via ``model.generate``. This wrapper exists
for symmetry with the vLLM backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HFRollout:
    """Placeholder marker for the default (transformers) rollout."""

    backend: str = "hf"

    def apply(self, config: Any) -> Any:
        """No-op; TRL's default generator is used."""
        return config

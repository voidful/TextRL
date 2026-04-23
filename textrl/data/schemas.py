"""Dataset schema validators for each algorithm family."""
from __future__ import annotations

from typing import Any

from datasets import Dataset


def _columns(ds: Dataset) -> set[str]:
    return set(ds.column_names)


def _require(ds: Dataset, required: set[str], *, name: str) -> None:
    missing = required - _columns(ds)
    if missing:
        raise ValueError(
            f"Dataset for {name} is missing required columns: {sorted(missing)}. "
            f"Got: {sorted(_columns(ds))}"
        )


def validate_prompt_dataset(ds: Any) -> Dataset:
    """Prompt-only dataset for online algorithms.

    Required: ``prompt`` (str) OR ``messages`` (list of chat-turn dicts).
    Extra columns pass through to the reward function as keyword arguments.
    """
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(ds).__name__}")
    cols = _columns(ds)
    if "prompt" not in cols and "messages" not in cols:
        raise ValueError(
            f"Prompt dataset must have 'prompt' or 'messages' column. Got: {sorted(cols)}"
        )
    return ds


def validate_preference_dataset(ds: Any) -> Dataset:
    """Preference dataset for DPO/IPO/ORPO/CPO/SimPO.

    Required: ``prompt``, ``chosen``, ``rejected``.
    """
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(ds).__name__}")
    _require(ds, {"prompt", "chosen", "rejected"}, name="preference")
    return ds


def validate_kto_dataset(ds: Any) -> Dataset:
    """KTO/BCO dataset - binary feedback.

    Required: ``prompt``, ``completion``, ``label`` (bool).
    """
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(ds).__name__}")
    _require(ds, {"prompt", "completion", "label"}, name="kto")
    return ds


def validate_reward_model_dataset(ds: Any) -> Dataset:
    """Reward-model training dataset.

    Required: ``chosen``, ``rejected`` (conversational or flat text).
    """
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(ds).__name__}")
    _require(ds, {"chosen", "rejected"}, name="reward_model")
    return ds

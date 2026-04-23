"""Convenience dataset loaders."""
from __future__ import annotations

from typing import Any, Iterable

from datasets import Dataset, load_dataset


def from_list(items: Iterable[Any]) -> Dataset:
    """Build a ``Dataset`` from a list of ``dict``s or plain strings.

    Plain strings are wrapped as ``{"prompt": s}``.
    """
    rows: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            rows.append({"prompt": item})
        elif isinstance(item, dict):
            rows.append(item)
        else:
            raise TypeError(f"Unsupported item type: {type(item).__name__}")
    if not rows:
        raise ValueError("from_list needs at least one item")
    return Dataset.from_list(rows)


def from_jsonl(path: str, split: str = "train") -> Dataset:
    """Load a JSONL file as a ``Dataset``."""
    ds = load_dataset("json", data_files=path, split=split)
    assert isinstance(ds, Dataset)
    return ds


def from_hub(name: str, split: str = "train", **kwargs: Any) -> Dataset:
    """Load a dataset from the HuggingFace Hub."""
    ds = load_dataset(name, split=split, **kwargs)
    assert isinstance(ds, Dataset)
    return ds

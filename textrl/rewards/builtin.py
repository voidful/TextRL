"""Built-in reward functions - all ``RewardFn``-compatible callables."""
from __future__ import annotations

from typing import Any

from textrl.rewards.base import reward_fn


@reward_fn
def length_penalty(
    prompts: list[str],
    completions: list[str],
    target_length: int = 64,
    scale: float = 0.01,
    **_: Any,
) -> list[float]:
    """Negative absolute deviation from ``target_length`` (in characters), scaled."""
    return [-scale * abs(len(c) - target_length) for c in completions]


@reward_fn
def repetition_penalty(
    prompts: list[str],
    completions: list[str],
    n: int = 3,
    scale: float = 1.0,
    **_: Any,
) -> list[float]:
    """Penalty for n-gram repetition. Returns negative = more repetition."""
    scores: list[float] = []
    for c in completions:
        tokens = c.split()
        if len(tokens) < n:
            scores.append(0.0)
            continue
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        unique_ratio = len(set(ngrams)) / max(1, len(ngrams))
        scores.append(scale * (unique_ratio - 1.0))
    return scores

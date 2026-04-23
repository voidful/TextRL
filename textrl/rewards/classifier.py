"""Wrap a HuggingFace ``pipeline`` as a reward function."""
from __future__ import annotations

from typing import Any, Callable

from textrl.rewards.base import BaseReward


class ClassifierReward(BaseReward):
    """Use a HuggingFace classification pipeline as a reward source.

    Example::

        from transformers import pipeline
        from textrl.rewards import ClassifierReward

        sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        reward = ClassifierReward(sentiment, target_label="LABEL_2")  # positive
    """

    def __init__(
        self,
        pipeline: Callable,
        target_label: str | None = None,
        score_fn: Callable[[list[dict]], float] | None = None,
        combine_prompt: bool = True,
        scale: float = 1.0,
        name: str = "classifier_reward",
    ) -> None:
        self._pipe = pipeline
        self._target_label = target_label
        self._score_fn = score_fn
        self._combine_prompt = combine_prompt
        self._scale = scale
        self.name = name

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        **_: Any,
    ) -> list[float]:
        texts = (
            [p + c for p, c in zip(prompts, completions)]
            if self._combine_prompt
            else list(completions)
        )
        raw = self._pipe(texts, top_k=None) if self._supports_top_k() else self._pipe(texts)
        out: list[float] = []
        for item in raw:
            out.append(self._scale * self._extract(item))
        return out

    def _supports_top_k(self) -> bool:
        try:
            import inspect
            sig = inspect.signature(self._pipe)
            return "top_k" in sig.parameters
        except (TypeError, ValueError):
            return False

    def _extract(self, item: Any) -> float:
        if self._score_fn is not None:
            return float(self._score_fn(item))
        if isinstance(item, list):
            if self._target_label is None:
                return max(float(d.get("score", 0.0)) for d in item)
            for d in item:
                if d.get("label") == self._target_label:
                    return float(d["score"])
            return 0.0
        if isinstance(item, dict):
            return float(item.get("score", 0.0))
        return float(item)

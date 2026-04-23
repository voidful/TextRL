"""Reward function abstractions.

TRL's online trainers expect ``reward(prompts, completions, **columns) -> list[float]``.
We provide:
  * ``RewardFn`` - the protocol
  * ``BaseReward`` - ABC for stateful rewards
  * ``@reward_fn`` - decorator that turns a plain function into a ``RewardFn``
  * ``compose(*fns, weights=...)`` - weighted sum of reward functions
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Protocol, runtime_checkable


@runtime_checkable
class RewardFn(Protocol):
    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **columns: Any,
    ) -> list[float]:
        ...


class BaseReward(ABC):
    """ABC for stateful rewards (e.g. a loaded classifier)."""

    name: str = "reward"

    @abstractmethod
    def score(
        self,
        prompts: list[str],
        completions: list[str],
        **columns: Any,
    ) -> list[float]:
        ...

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **columns: Any,
    ) -> list[float]:
        return self.score(prompts, completions, **columns)


class _FnWrap(BaseReward):
    def __init__(self, fn: Callable[..., list[float]], name: str | None = None) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "reward")

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        **columns: Any,
    ) -> list[float]:
        out = self._fn(prompts, completions, **columns)
        return [float(x) for x in out]


def reward_fn(fn: Callable[..., list[float]] | None = None, *, name: str | None = None):
    """Decorator wrapping a plain function as a ``RewardFn``.

    Usage::

        @reward_fn
        def my_reward(prompts, completions, **cols):
            return [...]
    """
    if fn is None:
        return lambda f: _FnWrap(f, name=name)
    return _FnWrap(fn, name=name)


def to_reward_fn(obj: Any) -> RewardFn:
    """Coerce anything callable-like into a ``RewardFn``."""
    if isinstance(obj, BaseReward):
        return obj
    if callable(obj):
        return _FnWrap(obj)
    raise TypeError(f"Cannot coerce {type(obj).__name__} to RewardFn")


class _Compose(BaseReward):
    def __init__(self, fns: Iterable[RewardFn], weights: Iterable[float] | None):
        self._fns = [to_reward_fn(f) for f in fns]
        if weights is None:
            self._weights = [1.0] * len(self._fns)
        else:
            self._weights = [float(w) for w in weights]
            if len(self._weights) != len(self._fns):
                raise ValueError("weights length must match number of functions")
        self.name = "+".join(getattr(f, "name", "r") for f in self._fns)

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        **columns: Any,
    ) -> list[float]:
        n = len(completions)
        total = [0.0] * n
        for fn, w in zip(self._fns, self._weights):
            scores = fn(prompts, completions, **columns)
            if len(scores) != n:
                raise ValueError(
                    f"{getattr(fn, 'name', fn)} returned {len(scores)} scores, expected {n}"
                )
            for i, s in enumerate(scores):
                total[i] += w * float(s)
        return total


def compose(*fns: RewardFn, weights: Iterable[float] | None = None) -> RewardFn:
    """Weighted sum of reward functions."""
    return _Compose(fns, weights)

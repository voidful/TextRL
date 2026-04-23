"""Online RL trainers: GRPO, RLOO, REINFORCE++.

Dispatches to the correct TRL trainer based on ``config.algo``. PPO and OnlineDPO
were removed in TRL 0.29+; use ``algo='rloo'`` or ``algo='grpo'`` instead.
"""
from __future__ import annotations

from typing import Any

from textrl.config import TextRLConfig
from textrl.data.schemas import validate_prompt_dataset
from textrl.rewards.base import BaseReward, RewardFn, to_reward_fn


class OnlineTrainer:
    """Unified front door for on-policy / online algorithms.

    Parameters
    ----------
    model : PreTrainedModel
        Policy model (optionally PEFT-wrapped).
    ref_model : PreTrainedModel | None
        Reference model for KL. Required for RLOO/REINFORCE++ unless PEFT is used.
        Unused by GRPO.
    tokenizer : PreTrainedTokenizerBase
    reward : RewardFn | list[RewardFn] | PreTrainedModel | None
        - GRPO: callable ``RewardFn`` or list of them (list -> weighted sum).
        - RLOO / REINFORCE++: a reward *model* (``nn.Module``). For callable
          rewards, use ``algo='grpo'``.
    train_dataset, eval_dataset : ``datasets.Dataset`` with a ``prompt`` column.
    config : TextRLConfig
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: TextRLConfig,
        *,
        ref_model: Any = None,
        reward: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
    ) -> None:
        if train_dataset is not None:
            train_dataset = validate_prompt_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = validate_prompt_dataset(eval_dataset)

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.reward = reward
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self._trainer = self._build()

    def _build(self):
        algo = self.config.algo
        args = self.config.to_trl_config()

        if algo == "grpo":
            return self._build_grpo(args)
        if algo in {"rloo", "reinforce_pp"}:
            return self._build_rloo(args)
        raise ValueError(f"OnlineTrainer does not handle algo={algo!r}")

    def _build_grpo(self, args: Any):
        from trl import GRPOTrainer

        reward_funcs = self._coerce_reward_funcs()
        return GRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

    def _build_rloo(self, args: Any):
        from trl import RLOOTrainer

        reward_model = self._require_reward_model("RLOO/REINFORCE++")
        return RLOOTrainer(
            config=args,
            processing_class=self.tokenizer,
            policy=self.model,
            ref_policy=self.ref_model,
            reward_model=reward_model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

    def _coerce_reward_funcs(self) -> list[RewardFn]:
        if self.reward is None:
            raise ValueError("GRPO requires reward=... (callable or list of callables)")
        if isinstance(self.reward, (list, tuple)):
            return [_wrap_for_grpo(to_reward_fn(r)) for r in self.reward]
        return [_wrap_for_grpo(to_reward_fn(self.reward))]

    def _require_reward_model(self, algo_name: str):
        if self.reward is None:
            raise ValueError(f"{algo_name} requires reward=<reward_model>")
        if not _looks_like_module(self.reward):
            raise TypeError(
                f"{algo_name} expects a reward *model* (nn.Module). Got "
                f"{type(self.reward).__name__}. For callable rewards, use algo='grpo'. "
                "To train a reward model, see textrl.trainers.RewardModelTrainer."
            )
        return self.reward

    def train(self, **kwargs: Any) -> Any:
        return self._trainer.train(**kwargs)

    def save_model(self, output_dir: str | None = None) -> None:
        self._trainer.save_model(output_dir)

    def push_to_hub(self, **kwargs: Any) -> Any:
        return self._trainer.push_to_hub(**kwargs)

    @property
    def trl_trainer(self):
        """Escape hatch: underlying TRL trainer for advanced use."""
        return self._trainer


def _looks_like_module(obj: Any) -> bool:
    import torch.nn as nn
    return isinstance(obj, nn.Module)


def _wrap_for_grpo(fn: BaseReward | RewardFn):
    """TRL's GRPOTrainer calls ``reward_func(completions, prompts=..., **cols)``.

    We normalize the argument order so user rewards can use ``(prompts, completions, ...)``.
    """
    import functools

    @functools.wraps(fn if callable(fn) else lambda: None)
    def _wrapped(completions, prompts=None, **columns):
        if prompts is None:
            prompts = columns.pop("prompts", None)
        if prompts is None:
            prompts = [""] * len(completions)
        return fn(prompts=list(prompts), completions=list(completions), **columns)

    _wrapped.__name__ = getattr(fn, "name", getattr(fn, "__name__", "reward"))
    return _wrapped

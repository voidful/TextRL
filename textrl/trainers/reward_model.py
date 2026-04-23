"""Reward-model training (wraps ``trl.RewardTrainer``).

A trained reward model can then be plugged into PPO/RLOO/OnlineDPO via
``OnlineTrainer(reward=reward_model, ...)``.
"""
from __future__ import annotations

from typing import Any

from textrl.config import TextRLConfig
from textrl.data.schemas import validate_reward_model_dataset


class RewardModelTrainer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: TextRLConfig,
        *,
        train_dataset: Any = None,
        eval_dataset: Any = None,
    ) -> None:
        if config.algo != "reward_model":
            raise ValueError("RewardModelTrainer requires config.algo == 'reward_model'")
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = (
            validate_reward_model_dataset(train_dataset) if train_dataset is not None else None
        )
        self.eval_dataset = (
            validate_reward_model_dataset(eval_dataset) if eval_dataset is not None else None
        )
        self._trainer = self._build()

    def _build(self):
        from trl import RewardTrainer

        args = self.config.to_trl_config()
        return RewardTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

    def train(self, **kwargs: Any) -> Any:
        return self._trainer.train(**kwargs)

    def save_model(self, output_dir: str | None = None) -> None:
        self._trainer.save_model(output_dir)

    @property
    def trl_trainer(self):
        return self._trainer

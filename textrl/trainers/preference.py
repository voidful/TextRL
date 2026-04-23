"""Offline preference trainers.

TRL 0.29+ unified most pairwise preference losses under ``DPOTrainer`` via
``DPOConfig(loss_type=...)``. Binary-label methods stay in ``KTOTrainer``.

Supported algorithms:
  * Pairwise (``DPOTrainer``): ``dpo`` (sigmoid), ``ipo``, ``hinge``, ``apo_zero``,
    ``apo_down``, ``bco_pair``, ``nca_pair``, ``robust``, ``aot``, ``aot_unpaired``,
    ``discopop``, ``sppo_hard``, ``exo_pair``, ``sft_on_dpo``.
  * Binary (``KTOTrainer``): ``kto``.
"""
from __future__ import annotations

from typing import Any

from textrl.config import (
    PREFERENCE_BINARY_ALGOS,
    PREFERENCE_PAIRWISE_ALGOS,
    TextRLConfig,
)
from textrl.data.schemas import validate_kto_dataset, validate_preference_dataset


class PreferenceTrainer:
    """Unified front door for preference-based / offline algorithms."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: TextRLConfig,
        *,
        ref_model: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.train_dataset = self._validate(train_dataset) if train_dataset is not None else None
        self.eval_dataset = self._validate(eval_dataset) if eval_dataset is not None else None
        self._trainer = self._build()

    def _validate(self, ds: Any) -> Any:
        if self.config.algo in PREFERENCE_PAIRWISE_ALGOS:
            return validate_preference_dataset(ds)
        if self.config.algo in PREFERENCE_BINARY_ALGOS:
            return validate_kto_dataset(ds)
        raise ValueError(f"Unsupported algo: {self.config.algo}")

    def _build(self):
        algo = self.config.algo
        args = self.config.to_trl_config()

        if algo in PREFERENCE_PAIRWISE_ALGOS:
            from trl import DPOTrainer
            return DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
            )
        if algo in PREFERENCE_BINARY_ALGOS:
            from trl import KTOTrainer
            return KTOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
            )
        raise ValueError(f"PreferenceTrainer does not handle algo={algo!r}")

    def train(self, **kwargs: Any) -> Any:
        return self._trainer.train(**kwargs)

    def save_model(self, output_dir: str | None = None) -> None:
        self._trainer.save_model(output_dir)

    def push_to_hub(self, **kwargs: Any) -> Any:
        return self._trainer.push_to_hub(**kwargs)

    @property
    def trl_trainer(self):
        return self._trainer

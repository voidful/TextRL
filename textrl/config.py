"""TextRL configuration.

Delegates to the correct TRL ``*Config`` class at runtime. TRL 0.29+ consolidated
many preference losses into ``DPOConfig(loss_type=...)``; this module maps
friendly algorithm names onto the right TRL class and loss type.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ONLINE_ALGOS = {"grpo", "rloo", "reinforce_pp"}
PREFERENCE_PAIRWISE_ALGOS = {
    "dpo", "ipo", "hinge", "apo_zero", "apo_down",
    "bco_pair", "nca_pair", "robust", "aot", "aot_unpaired",
    "discopop", "sppo_hard", "exo_pair", "sft_on_dpo",
}
PREFERENCE_BINARY_ALGOS = {"kto"}
PREFERENCE_ALGOS = PREFERENCE_PAIRWISE_ALGOS | PREFERENCE_BINARY_ALGOS
ALL_ALGOS = ONLINE_ALGOS | PREFERENCE_ALGOS | {"reward_model"}

REMOVED_ALGOS: dict[str, str] = {
    "ppo": "PPO was removed from TRL 0.29+. Use algo='rloo' (preferred) or algo='grpo'.",
    "online_dpo": "OnlineDPO was removed from TRL 0.29+. Use algo='grpo' or algo='rloo' with a reward model.",
    "orpo": "ORPOTrainer was removed in TRL 0.29+. Use algo='dpo' (sigmoid) or algo='apo_zero' as close substitutes.",
    "cpo": "CPOTrainer was removed in TRL 0.29+. Use algo='dpo' (sigmoid) - DPO now subsumes CPO.",
    "simpo": "SimPO was removed in TRL 0.29+. Use algo='dpo' with a suitable reference-free loss.",
    "bco": "BCOTrainer was removed in TRL 0.29+. Use algo='bco_pair' (paired BCO via DPOTrainer) or algo='kto'.",
}

DPO_LOSS_TYPES = {
    "dpo": "sigmoid",
    "ipo": "ipo",
    "hinge": "hinge",
    "apo_zero": "apo_zero",
    "apo_down": "apo_down",
    "bco_pair": "bco_pair",
    "nca_pair": "nca_pair",
    "robust": "robust",
    "aot": "aot",
    "aot_unpaired": "aot_unpaired",
    "discopop": "discopop",
    "sppo_hard": "sppo_hard",
    "exo_pair": "exo_pair",
    "sft_on_dpo": "sft",
}

AlgoName = Literal[
    "grpo", "rloo", "reinforce_pp",
    "dpo", "ipo", "hinge", "apo_zero", "apo_down",
    "bco_pair", "nca_pair", "robust", "aot", "aot_unpaired",
    "discopop", "sppo_hard", "exo_pair", "sft_on_dpo",
    "kto",
    "reward_model",
]


@dataclass
class TextRLConfig:
    """User-facing training config that delegates to the right TRL config."""

    algo: AlgoName
    output_dir: str

    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    report_to: list[str] | str = "none"

    beta: float | None = None
    loss_type: str | None = None  # override the default mapped from ``algo``
    num_generations: int | None = None

    peft: dict[str, Any] | None = None
    quantization: Literal["4bit", "8bit"] | None = None
    rollout: dict[str, Any] | None = None
    distributed: dict[str, Any] | None = None
    reward_weights: list[float] | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.algo in REMOVED_ALGOS:
            raise ValueError(REMOVED_ALGOS[self.algo])
        if self.algo not in ALL_ALGOS:
            raise ValueError(
                f"Unknown algo {self.algo!r}. Valid: {sorted(ALL_ALGOS)}"
            )

    def to_trl_config(self) -> Any:
        kwargs = self._base_training_kwargs()
        kwargs.update(self.extra)

        algo = self.algo
        if algo == "grpo":
            from trl import GRPOConfig
            kwargs = self._merge({
                "num_generations": self.num_generations,
                "beta": self.beta,
                "reward_weights": self.reward_weights,
            }, kwargs)
            return GRPOConfig(**self._filter(GRPOConfig, kwargs))
        if algo in {"rloo", "reinforce_pp"}:
            from trl import RLOOConfig
            if algo == "reinforce_pp":
                kwargs.setdefault("rloo_k", 1)
            kwargs = self._merge({"beta": self.beta}, kwargs)
            return RLOOConfig(**self._filter(RLOOConfig, kwargs))
        if algo in PREFERENCE_PAIRWISE_ALGOS:
            from trl import DPOConfig
            loss_type = self.loss_type or DPO_LOSS_TYPES[algo]
            kwargs = self._merge({"beta": self.beta, "loss_type": loss_type}, kwargs)
            return DPOConfig(**self._filter(DPOConfig, kwargs))
        if algo == "kto":
            from trl import KTOConfig
            kwargs = self._merge({"beta": self.beta}, kwargs)
            return KTOConfig(**self._filter(KTOConfig, kwargs))
        if algo == "reward_model":
            from trl import RewardConfig
            return RewardConfig(**self._filter(RewardConfig, kwargs))
        raise AssertionError(f"unreachable: {algo}")

    def _base_training_kwargs(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "seed": self.seed,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "report_to": self.report_to,
        }

    @staticmethod
    def _merge(extras: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for k, v in extras.items():
            if v is not None:
                out.setdefault(k, v)
        return out

    @staticmethod
    def _filter(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        import dataclasses as _dc
        if not _dc.is_dataclass(cls):
            return kwargs
        valid = {f.name for f in _dc.fields(cls)}
        return {k: v for k, v in kwargs.items() if k in valid}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextRLConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = {k: v for k, v in data.items() if k not in known}
        clean = {k: v for k, v in data.items() if k in known}
        if "extra" in clean and isinstance(clean["extra"], dict):
            clean["extra"].update(extra)
        else:
            clean["extra"] = extra
        return cls(**clean)

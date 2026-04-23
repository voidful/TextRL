"""TextRL - reinforcement learning for text generation, built on HuggingFace TRL."""

__version__ = "1.0.0"

from textrl.config import TextRLConfig
from textrl.models import load_model
from textrl.rewards import (
    BaseReward,
    ClassifierReward,
    RewardFn,
    compose,
    length_penalty,
    repetition_penalty,
    reward_fn,
    to_reward_fn,
)
from textrl.trainers import OnlineTrainer, PreferenceTrainer, RewardModelTrainer

__all__ = [
    "__version__",
    "TextRLConfig",
    "load_model",
    "OnlineTrainer",
    "PreferenceTrainer",
    "RewardModelTrainer",
    "RewardFn",
    "BaseReward",
    "ClassifierReward",
    "reward_fn",
    "to_reward_fn",
    "compose",
    "length_penalty",
    "repetition_penalty",
]

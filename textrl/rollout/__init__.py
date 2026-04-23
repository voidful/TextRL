"""Rollout backends.

By default, TRL trainers use HuggingFace ``model.generate``. vLLM is available
for faster sampling during GRPO rollouts (requires ``pip install 'textrl[vllm]'``).
"""
from textrl.rollout.hf import HFRollout

__all__ = ["HFRollout"]

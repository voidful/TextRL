# TextRL: Reinforcement Learning for Text Generation

<p align="center">
    <a href="https://pypi.org/project/textrl/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/textrl">
    </a>
    <a href="https://github.com/voidful/textrl">
        <img alt="Last Commit" src="https://img.shields.io/github/last-commit/voidful/textrl">
    </a>
</p>

TextRL is a thin, opinionated layer on top of [HuggingFace TRL](https://github.com/huggingface/trl) that makes modern text-generation RL ergonomic: one dataclass for configuration, one trainer class per algorithm family, callable reward functions, and first-class PEFT / accelerate / vLLM support.

> **v1.0 breaking change.** The legacy PFRL/gym API (`TextRLEnv`, `TextRLActor`, `train_agent_with_evaluation`) is gone. See [docs/migration.md](docs/migration.md).

## Supported algorithms

| Family | Algorithms | TRL trainer |
|---|---|---|
| **Online** | GRPO, RLOO, REINFORCE++ | `GRPOTrainer`, `RLOOTrainer` |
| **Preference (pairwise)** | DPO, IPO, Hinge, APO (zero/down), BCO-pair, NCA-pair, Robust-DPO, AOT, DiscoPOP, SPPO-hard, EXO-pair | `DPOTrainer` (unified `loss_type`) |
| **Preference (binary)** | KTO | `KTOTrainer` |
| **Reward model** | Pairwise reward training | `RewardTrainer` |

Removed in TRL 0.29+ and therefore not supported: PPO, OnlineDPO, ORPO, CPO, SimPO, BCO (binary). TextRL raises with a migration hint if you ask for them.

## Install

```bash
pip install textrl                         # core
pip install 'textrl[quant]'                # + bitsandbytes (QLoRA)
pip install 'textrl[vllm]'                 # + vLLM rollout
pip install 'textrl[quant,vllm,rewards]'   # kitchen sink
```

## Quickstart

### GRPO with a callable reward

```python
from textrl import OnlineTrainer, TextRLConfig, load_model, reward_fn
from textrl.data import from_list

@reward_fn
def length_reward(prompts, completions, **_):
    return [-abs(len(c) - 64) / 64 for c in completions]

model, tok, _ = load_model("Qwen/Qwen2.5-0.5B", peft={"type": "lora", "r": 16})

cfg = TextRLConfig(
    algo="grpo",
    output_dir="out/grpo",
    num_generations=8,
    beta=0.04,
    learning_rate=5e-6,
    bf16=True,
)

trainer = OnlineTrainer(
    model=model,
    tokenizer=tok,
    reward=length_reward,
    train_dataset=from_list(["Write a short poem.", "Explain gradient descent."] * 32),
    config=cfg,
)
trainer.train()
```

### DPO with a preference dataset

```python
from textrl import PreferenceTrainer, TextRLConfig, load_model
from textrl.data import from_hub

model, tok, ref = load_model("meta-llama/Llama-3.2-1B", peft={"type": "lora", "r": 16}, quantization="4bit")

cfg = TextRLConfig(algo="dpo", output_dir="out/dpo", beta=0.1, bf16=True)

trainer = PreferenceTrainer(
    model=model,
    ref_model=ref,
    tokenizer=tok,
    train_dataset=from_hub("trl-lib/ultrafeedback_binarized"),
    config=cfg,
)
trainer.train()
```

### KTO with binary feedback

```python
from textrl import PreferenceTrainer, TextRLConfig, load_model

cfg = TextRLConfig(algo="kto", output_dir="out/kto", beta=0.1, bf16=True)
model, tok, ref = load_model("Qwen/Qwen2.5-0.5B")
trainer = PreferenceTrainer(
    model=model, ref_model=ref, tokenizer=tok,
    train_dataset=my_kto_dataset,   # needs prompt/completion/label
    config=cfg,
)
trainer.train()
```

### RLOO with a trained reward model

```python
from textrl import OnlineTrainer, RewardModelTrainer, TextRLConfig, load_model

rm_cfg = TextRLConfig(algo="reward_model", output_dir="out/rm", bf16=True)
rm_model, tok, _ = load_model("distilbert/distilbert-base-uncased", load_ref=False)
RewardModelTrainer(model=rm_model, tokenizer=tok, train_dataset=rm_ds, config=rm_cfg).train()

model, tok, ref = load_model("Qwen/Qwen2.5-0.5B")
cfg = TextRLConfig(algo="rloo", output_dir="out/rloo", bf16=True)
OnlineTrainer(model=model, ref_model=ref, tokenizer=tok,
              reward=rm_model, train_dataset=prompts, config=cfg).train()
```

## Reward functions

Rewards are plain callables with the signature TRL expects:

```python
def reward(prompts: list[str], completions: list[str], **columns) -> list[float]: ...
```

Decorate with `@reward_fn` (coerces into a `RewardFn` protocol object), or subclass `BaseReward` for stateful rewards (e.g. a loaded classifier). Compose multiple rewards with `compose(*fns, weights=...)`:

```python
from textrl.rewards import compose, length_penalty, reward_fn

@reward_fn
def semantic_match(prompts, completions, **_):
    return [...]

reward = compose(semantic_match, length_penalty, weights=[1.0, 0.1])
```

`ClassifierReward` wraps any HuggingFace `pipeline`:

```python
from transformers import pipeline
from textrl.rewards import ClassifierReward

sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
reward = ClassifierReward(sentiment, target_label="LABEL_2")  # positive
```

## Data formats

| Mode | Required columns | Used by |
|---|---|---|
| Prompt-only | `prompt` (or `messages`) | GRPO, RLOO, REINFORCE++ |
| Pairwise preference | `prompt`, `chosen`, `rejected` | DPO, IPO, Hinge, APO, BCO-pair, etc. |
| Binary feedback | `prompt`, `completion`, `label: bool` | KTO |
| Reward model | `chosen`, `rejected` | `RewardModelTrainer` |

Use `textrl.data.from_list`, `from_jsonl`, or `from_hub` to construct datasets, or pass any `datasets.Dataset` directly.

## Model loading

`load_model` returns `(policy, tokenizer, ref_model_or_None)`:

```python
from textrl import load_model

model, tok, ref = load_model(
    "meta-llama/Llama-3.2-1B",
    peft={"type": "lora", "r": 16, "alpha": 32, "target_modules": "all-linear"},
    quantization="4bit",          # nf4 QLoRA
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    load_ref=True,                 # False for GRPO/RLOO to save memory
)
```

When `peft` is set, `ref_model` is `None` — TRL disables adapters for the reference forward pass.

## Distributed training

Launch via `accelerate`. TextRL adds no scaffolding of its own:

```bash
accelerate launch -m textrl.cli train --config configs/grpo.yaml
```

`TextRLConfig.distributed={"strategy": "deepspeed", "zero_stage": 3}` is forwarded to TRL via the `extra` field.

## vLLM rollout (GRPO only)

```python
cfg = TextRLConfig(
    algo="grpo", output_dir="out",
    extra={"use_vllm": True, "vllm_gpu_memory_utilization": 0.6},
)
```

Or use the helper `textrl.rollout.vllm.vllm_config(...)` to build the extras dict.

## CLI

| Command | Purpose |
|---|---|
| `textrl-train --config cfg.yaml` | YAML-driven training |
| `textrl-merge --adapter DIR --output DIR` | Merge a PEFT adapter into a standalone HF checkpoint |
| `textrl-eval --model PATH --dataset SPEC --reward module:fn` | Rollout + reward stats (no training) |
| `textrl-dump` | Deprecated alias for `textrl-merge` |

Example YAML:

```yaml
algo: grpo
output_dir: out/grpo
learning_rate: 5e-6
num_train_epochs: 1
num_generations: 8
beta: 0.04
bf16: true

model:
  name: Qwen/Qwen2.5-0.5B

dataset:
  hub: trl-lib/tldr
  split: train[:1%]

reward: my_rewards:length_reward
```

## Development

```bash
pip install -e '.[dev,quant,rewards]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/unit
pytest -m smoke tests/smoke   # needs a small model to be downloadable
```

## License

Apache 2.0.

# Migrating from TextRL 0.x to 1.0

TextRL 1.0 is a clean break. The PFRL-based gym API is gone; everything now runs on top of [HuggingFace TRL](https://github.com/huggingface/trl). If you're on 0.2.x, read this document end-to-end before upgrading — names, signatures, and the training loop have all changed.

## What was removed

| 0.x symbol | Status | Replacement |
|---|---|---|
| `TextRLEnv` | Removed | Write a callable reward (see below) |
| `TextRLActor` | Removed | `OnlineTrainer` with `algo="grpo"` |
| `train_agent_with_evaluation` | Removed | `trainer.train()` |
| `textrl.dump` | Removed | `textrl-merge` CLI / `textrl.utils.merge.merge_adapter` |
| `pfrl`, `gym` deps | Dropped | TRL + `accelerate` |

## Algorithm renames

Some algorithms were removed upstream in TRL 0.29+ and are therefore not supported here either. TextRL raises `ValueError` with a migration hint if you try to use them:

| Old algo | Use instead |
|---|---|
| `ppo` | `rloo` (or `grpo` if you don't have a reward model) |
| `online_dpo` | `grpo` with a preference-modeling reward |
| `orpo`, `cpo`, `simpo` | `dpo` (TRL unified these under `DPOTrainer` with `loss_type=...`) |
| `bco` (binary) | `bco_pair` (pairwise) |

## Rewriting a reward

**Before (0.x):** subclass `TextRLEnv`, implement per-token `get_reward`:

```python
class MyEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):
        if not finish:
            return 0.0
        text = self.tokenizer.decode(predicted_list[0])
        return score(text)
```

**After (1.0):** a plain callable that receives full sequences in batch:

```python
from textrl import reward_fn

@reward_fn
def my_reward(prompts, completions, **columns):
    return [score(c) for c in completions]
```

TRL batches rollouts and calls the reward once per batch on finished sequences — there is no per-token hook anymore. If you had logic that depended on partial sequences, move it into `logits_processors` on the generation side or into a learned reward model.

## Rewriting the training loop

**Before (0.x):**

```python
env = MyEnv(model, tokenizer, observation_list)
actor = TextRLActor(env, model, tokenizer)
agent = actor.agent_ppo(update_interval=50, minibatch_size=2000, epochs=20)
train_agent_with_evaluation(agent, env, steps=1000, ...)
```

**After (1.0):**

```python
from textrl import OnlineTrainer, TextRLConfig, load_model
from textrl.data import from_list

model, tok, _ = load_model("Qwen/Qwen2.5-0.5B", peft={"type": "lora", "r": 16})

cfg = TextRLConfig(
    algo="grpo",
    output_dir="out",
    num_generations=8,
    beta=0.04,
    learning_rate=5e-6,
    bf16=True,
)

trainer = OnlineTrainer(
    model=model, tokenizer=tok,
    reward=my_reward,
    train_dataset=from_list(my_prompts),
    config=cfg,
)
trainer.train()
```

PPO is gone from TRL core; `algo="grpo"` is the closest drop-in. If you need a learned reward model in the loop, train one with `RewardModelTrainer` and pass it to `OnlineTrainer(algo="rloo", reward=rm)`.

## Checkpoint dump → merge

**Before:**

```bash
textrl-dump --model base_model --rl rl_ckpt --dump merged
```

**After:**

```bash
textrl-merge --adapter rl_ckpt --output merged
```

`textrl-dump` still exists as a deprecated alias; it prints a warning and delegates to `textrl-merge`.

## Things that are new

- **PEFT / QLoRA** via `load_model(peft=..., quantization="4bit")`.
- **DPO and friends** through `PreferenceTrainer` with `algo="dpo" | "ipo" | "kto" | ...`.
- **vLLM rollout** for GRPO via `TextRLConfig(extra={"use_vllm": True, ...})`.
- **Distributed training** via `accelerate launch -m textrl.cli train --config cfg.yaml`.
- **YAML-driven runs** via `textrl-train --config cfg.yaml`.

## If you're stuck on 0.x

Pin the old version:

```bash
pip install 'textrl<1.0'
```

The 0.x series will not receive further releases.

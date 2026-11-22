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
train_agent_with_evaluation(
    agent, env, steps=1000, eval_n_steps=None, eval_n_episodes=5,
    eval_interval=100, outdir="out",
)
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

### The historical sentiment example

The [0.x sentiment notebook](https://github.com/voidful/TextRL/blob/dc8568e4ce61c4ac07e9d3996257410489f2ec59/example/2022-12-10-textrl-elon-musk.ipynb) uses a dictionary for each example. The historical fix ([#28](https://github.com/voidful/TextRL/pull/28)) reads its prompt with `input_item["input"]`, not `input_item[0]`. The notebook is not part of 1.0. To score both parts of a plain-text example under the 1.0 API, combine the prompt and completion explicitly:

```python
@reward_fn
def my_reward(prompts, completions, **columns):
    return [score(prompt + completion) for prompt, completion in zip(prompts, completions)]
```

### Legacy FLAN-T5 layer unfreezing

There is no `unfreeze_layer_from_past` setting in 1.0. In the 0.x FLAN-T5 path, setting it to `1` made the environment pass the final decoder hidden state to the policy, which then ran the last decoder block again. The extracted block was called with only a hidden state, rather than with the encoder context and decoder position bias passed by the normal T5 decoder stack. This changes inference even before training. To avoid that path on 0.x, keep the setting at `0` on both the actor and the environment. This does not guarantee nonempty generations.

For a new 1.0 online-RL run, start from a causal-LM model and configure LoRA via `load_model(..., peft={"type": "lora", "r": 16})`. The adapter works through the model's normal forward path; it is not equivalent to unfreezing the last FLAN-T5 block. See the [GRPO quickstart](../README.md#grpo-with-a-callable-reward).

## Checkpoint dump → merge

**Before:**

```bash
textrl-dump --model base_model --rl rl_ckpt --dumpdir merged
```

**After:**

```bash
textrl-merge --adapter rl_ckpt --output merged
```

In 1.0, the deprecated alias uses `--dump` instead of the old `--dumpdir` and delegates to `textrl-merge`. It has the same format requirements as `textrl-merge`: supply a PEFT adapter directory (with `adapter_config.json` and adapter weights). It does not convert old PFRL checkpoints. A full Hugging Face checkpoint can be loaded directly without merging.

### Why legacy `actor.predict` and exported inference differ

In 0.x, `actor.predict` ran the PFRL environment token by token, recalculating observations from the prompt and growing output. The old `dump` script rebuilt a default actor, loaded its checkpoint and saved the base Hugging Face model with the actor's language-model head. It did not export the environment, action selection, or any separately configured decoder layers. A direct `model.generate()` call can therefore be faster without producing the same answers. In particular, the old default dump was not designed to export a policy trained with `unfreeze_layer_from_past > 0`.

To compare a 1.0 PEFT model before and after merging, start with the same base checkpoint, tokenizer, prompt formatting, and numeric precision. Use greedy decoding (`do_sample=False`, `num_beams=1`), then match token limits and stopping tokens. Compare generated token IDs or next-token logits on a few prompts before comparing task scores. Test the adapter-equipped model and the merged model, not the unadapted base model.

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

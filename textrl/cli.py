"""CLI entry points: ``textrl-train``, ``textrl-merge``, ``textrl-eval``, ``textrl-dump`` (deprecated)."""
from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path
from typing import Any

from textrl.config import ONLINE_ALGOS, PREFERENCE_ALGOS, TextRLConfig


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml
    text = Path(path).read_text()
    return yaml.safe_load(text)


def _import_symbol(ref: str) -> Any:
    """``module.path:attr`` -> attr."""
    if ":" not in ref:
        raise ValueError(f"Expected 'module.path:attr', got {ref!r}")
    mod, attr = ref.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def _resolve_dataset(spec: str | dict | Any) -> Any:
    """Resolve a dataset spec from YAML.

    Supported forms:
      * ``module:attr`` string - imports and returns the object.
      * ``{"hub": "name", "split": "train"}`` - calls ``datasets.load_dataset``.
      * ``{"jsonl": "path"}`` - loads a jsonl file.
      * Already a ``Dataset`` - returned as-is.
    """
    if isinstance(spec, str):
        return _import_symbol(spec)
    if isinstance(spec, dict):
        if "hub" in spec:
            from textrl.data import from_hub
            return from_hub(spec["hub"], split=spec.get("split", "train"))
        if "jsonl" in spec:
            from textrl.data import from_jsonl
            return from_jsonl(spec["jsonl"], split=spec.get("split", "train"))
        if "list" in spec:
            from textrl.data import from_list
            return from_list(spec["list"])
    return spec


def train_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("textrl-train")
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args(argv)

    raw = _load_yaml(args.config)
    model_spec = raw.pop("model", {})
    dataset_spec = raw.pop("dataset", None)
    reward_spec = raw.pop("reward", None)
    ref_dataset_spec = raw.pop("eval_dataset", None)

    cfg = TextRLConfig.from_dict(raw)

    from textrl.config import PREFERENCE_PAIRWISE_ALGOS
    from textrl.models import load_model
    needs_ref = (
        cfg.peft is None
        and cfg.algo in (PREFERENCE_PAIRWISE_ALGOS | {"kto", "rloo", "reinforce_pp"})
    )
    model, tokenizer, ref_model = load_model(
        model_spec.pop("name"),
        peft=cfg.peft,
        quantization=cfg.quantization,
        load_ref=needs_ref,
        **model_spec,
    )

    train_dataset = _resolve_dataset(dataset_spec) if dataset_spec else None
    eval_dataset = _resolve_dataset(ref_dataset_spec) if ref_dataset_spec else None

    reward = None
    if reward_spec is not None:
        reward = (
            [_import_symbol(r) for r in reward_spec]
            if isinstance(reward_spec, list)
            else _import_symbol(reward_spec)
        )

    if cfg.algo in ONLINE_ALGOS:
        from textrl.trainers.online import OnlineTrainer
        trainer = OnlineTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            reward=reward,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=cfg,
        )
    elif cfg.algo in PREFERENCE_ALGOS:
        from textrl.trainers.preference import PreferenceTrainer
        trainer = PreferenceTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=cfg,
        )
    elif cfg.algo == "reward_model":
        from textrl.trainers.reward_model import RewardModelTrainer
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=cfg,
        )
    else:
        raise ValueError(f"Unknown algo {cfg.algo}")

    trainer.train()
    return 0


def merge_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("textrl-merge")
    ap.add_argument("--adapter", required=True, help="Trained adapter directory")
    ap.add_argument("--output", required=True, help="Output merged-model directory")
    ap.add_argument("--base-model", default=None, help="Override base model path")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args(argv)

    from textrl.utils.merge import merge_adapter
    out = merge_adapter(
        adapter_dir=args.adapter,
        output_dir=args.output,
        base_model=args.base_model,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Merged checkpoint written to {out}")
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("textrl-eval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="module:attr | hub:name | jsonl:path")
    ap.add_argument("--reward", required=True, help="module:attr")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()

    ds = _resolve_dataset(_parse_dataset_spec(args.dataset))
    reward = _import_symbol(args.reward)

    prompts = list(ds["prompt"])
    completions: list[str] = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        completions.extend(tok.batch_decode(gen, skip_special_tokens=True))

    scores = reward(prompts, completions)
    mean = sum(scores) / len(scores)
    print(f"samples={len(prompts)} mean_reward={mean:.4f} min={min(scores):.4f} max={max(scores):.4f}")
    return 0


def _parse_dataset_spec(s: str) -> str | dict[str, str]:
    if s.startswith("hub:"):
        return {"hub": s.split(":", 1)[1]}
    if s.startswith("jsonl:"):
        return {"jsonl": s.split(":", 1)[1]}
    return s  # module:attr


def _deprecated_dump(argv: list[str] | None = None) -> int:
    warnings.warn(
        "'textrl-dump' is deprecated; use 'textrl-merge' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ap = argparse.ArgumentParser("textrl-dump (deprecated)")
    ap.add_argument("--model", dest="base_model", help="Base model (pre-RL)", default=None)
    ap.add_argument("--rl", dest="adapter", required=True, help="RL-trained adapter/checkpoint dir")
    ap.add_argument("--dump", dest="output", required=True, help="Output directory")
    args = ap.parse_args(argv)
    return merge_main(
        ["--adapter", args.adapter, "--output", args.output]
        + (["--base-model", args.base_model] if args.base_model else [])
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(train_main())

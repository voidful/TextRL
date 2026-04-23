"""Model + tokenizer + reference-model loading with PEFT/quantization support."""
from __future__ import annotations

import copy
from typing import Any, Literal, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


LoadedTriple = Tuple[PreTrainedModel, PreTrainedTokenizerBase, PreTrainedModel | None]


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype | str:
    if isinstance(dtype, str) and dtype != "auto":
        return getattr(torch, dtype)
    return dtype


def _is_seq2seq(name_or_path: str, trust_remote_code: bool) -> bool:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    return getattr(cfg, "is_encoder_decoder", False)


def _build_quant_config(quantization: str | None):
    if quantization is None:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "quantization requires bitsandbytes. Install with: pip install 'textrl[quant]'"
        ) from e
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unknown quantization {quantization!r}")


def _apply_peft(model: PreTrainedModel, peft: dict[str, Any], quantized: bool) -> PreTrainedModel:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    peft = dict(peft)
    kind = peft.pop("type", "lora").lower()
    if kind != "lora":
        raise ValueError(f"Only peft type='lora' is supported, got {kind!r}")

    if quantized:
        model = prepare_model_for_kbit_training(model)

    task_type = peft.pop("task_type", None)
    if task_type is None:
        task_type = (
            TaskType.SEQ_2_SEQ_LM if getattr(model.config, "is_encoder_decoder", False)
            else TaskType.CAUSAL_LM
        )

    lora_cfg = LoraConfig(
        r=peft.pop("r", 16),
        lora_alpha=peft.pop("alpha", peft.pop("lora_alpha", 32)),
        lora_dropout=peft.pop("dropout", peft.pop("lora_dropout", 0.05)),
        bias=peft.pop("bias", "none"),
        target_modules=peft.pop("target_modules", "all-linear"),
        task_type=task_type,
        **peft,
    )
    return get_peft_model(model, lora_cfg)


def load_model(
    name_or_path: str,
    peft: dict[str, Any] | None = None,
    quantization: Literal["4bit", "8bit"] | None = None,
    torch_dtype: torch.dtype | str = "auto",
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
    load_ref: bool = True,
    device_map: str | dict | None = "auto",
) -> LoadedTriple:
    """Load a model + tokenizer, optionally wrapped in PEFT/quantization.

    Returns ``(policy_model, tokenizer, ref_model_or_None)``.

    The reference model is:
      * ``None`` when ``peft`` is set — TRL disables adapters for the ref pass.
      * ``None`` when ``load_ref=False`` (GRPO/RLOO/ORPO/SimPO/CPO don't need it).
      * Otherwise, a frozen copy of the policy.
    """
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    is_seq2seq = _is_seq2seq(name_or_path, trust_remote_code)
    ModelCls = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM

    quant_cfg = _build_quant_config(quantization)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(torch_dtype),
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = ModelCls.from_pretrained(name_or_path, **load_kwargs)

    if peft is not None:
        model = _apply_peft(model, peft, quantized=quant_cfg is not None)

    ref_model: PreTrainedModel | None = None
    if load_ref and peft is None:
        if quant_cfg is not None:
            ref_model = ModelCls.from_pretrained(name_or_path, **load_kwargs)
        else:
            try:
                from trl import create_reference_model
                ref_model = create_reference_model(model)
            except Exception:
                ref_model = copy.deepcopy(model)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

    return model, tokenizer, ref_model

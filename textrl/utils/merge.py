"""Merge a PEFT adapter back into the base model and save as a standard HF checkpoint."""
from __future__ import annotations

from pathlib import Path


def merge_adapter(
    adapter_dir: str,
    output_dir: str,
    base_model: str | None = None,
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> str:
    """Load a PEFT adapter, merge it into the base model, save the result.

    Args:
        adapter_dir: directory produced by a TextRL trainer (contains ``adapter_config.json``).
        output_dir: where to write the merged HF checkpoint.
        base_model: override the base model path; if ``None``, read from ``adapter_config.json``.

    Returns the ``output_dir`` path.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    if base_model is None:
        import json
        cfg_path = adapter_path / "adapter_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Expected adapter_config.json in {adapter_dir}; pass base_model= explicitly."
            )
        base_model = json.loads(cfg_path.read_text())["base_model_name_or_path"]

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    ModelCls = (
        AutoModelForSeq2SeqLM
        if getattr(config, "is_encoder_decoder", False)
        else AutoModelForCausalLM
    )

    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) and torch_dtype != "auto" else torch_dtype
    base = ModelCls.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    return output_dir

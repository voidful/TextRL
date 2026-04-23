"""DPO quickstart: offline preference tuning on UltraFeedback.

Run:
    pip install 'textrl[quant]'
    python example/dpo_quickstart.py
"""
from __future__ import annotations

from textrl import PreferenceTrainer, TextRLConfig, load_model
from textrl.data import from_hub


def main() -> None:
    model, tok, ref = load_model(
        "Qwen/Qwen2.5-0.5B",
        peft={"type": "lora", "r": 16, "alpha": 32, "target_modules": "all-linear"},
    )

    cfg = TextRLConfig(
        algo="dpo",
        output_dir="out/dpo-quickstart",
        beta=0.1,
        learning_rate=5e-7,
        per_device_train_batch_size=2,
        max_steps=50,
        bf16=True,
        logging_steps=5,
    )

    train_ds = from_hub("trl-lib/ultrafeedback_binarized", split="train[:256]")

    trainer = PreferenceTrainer(
        model=model,
        ref_model=ref,
        tokenizer=tok,
        train_dataset=train_ds,
        config=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()

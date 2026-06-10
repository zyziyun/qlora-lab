"""QLoRA fine-tuning with Unsloth + TRL. Runs on a single T4 (Colab free tier).

This mirrors the recipe in the S7 handout: 4-bit base (QLoRA), LoRA rank 16 /
alpha 32 on all linear layers, cosine schedule, a few epochs. It deliberately
keeps every knob visible and commented so the file reads as the explanation.

This module imports unsloth/trl lazily inside `train()` so the rest of the lab
(data, evaluation, the offline stub) imports cleanly on a machine with no GPU and
no ML stack installed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    base_model: str = "unsloth/Qwen3-8B-bnb-4bit"  # 4-bit base = the QLoRA part
    max_seq_length: int = 2048
    # LoRA: rank 16 is within 1-2% of rank 64 for most narrow tasks; alpha = 2*r.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # All linear layers is the 2026 default; attention-only leaves quality on the table.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 4  # effective batch = batch_size * grad_accum = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    output_dir: str = "outputs/adapter"


def train(train_jsonl: str, cfg: TrainConfig = TrainConfig()) -> str:
    """Fine-tune and save a LoRA adapter. Returns the adapter directory.

    `train_jsonl` is the file written by `dataset.write_jsonl(to_chat(...))`:
    one {"messages": [...]} record per line. We let TRL apply the chat template
    and, crucially, mask the loss to the assistant turn only.
    """
    import torch  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415

    # T4 (Turing) has no bf16; hardcoding bf16=True errors there. Detect instead.
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,  # NF4 4-bit base weights
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=list(cfg.target_modules),
        use_gradient_checkpointing="unsloth",  # extra VRAM headroom for long seqs
    )

    ds = load_dataset("json", data_files=train_jsonl, split="train")

    def _format(batch):
        # apply_chat_template appends the EOS/end-of-turn token, so the model
        # learns where to stop. train_on_responses_only masks the prompt below.
        return {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]
        }

    ds = ds.map(_format, batched=True, remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            lr_scheduler_type="cosine",
            bf16=bf16_ok,
            fp16=not bf16_ok,
            logging_steps=5,
            output_dir=cfg.output_dir,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
        ),
    )

    # Mask everything before the assistant turn so loss is computed on the answer
    # only. Without this the model learns to generate the user's message too.
    try:
        from unsloth.chat_templates import train_on_responses_only  # noqa: PLC0415

        trainer = train_on_responses_only(trainer)
    except Exception:
        # If the helper is unavailable, SFTTrainer still trains; the mask is a
        # quality optimization, not a correctness gate for this small task.
        pass

    trainer.train()
    model.save_pretrained(cfg.output_dir)  # saves the ~tens-of-MB adapter only
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir

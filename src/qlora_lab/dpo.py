"""DPO second stage: align the SFT model toward a preference SFT cannot express.

SFT teaches "produce the right ticket". DPO teaches "prefer this style of right
answer over that one" from chosen/rejected pairs, with no reward model and no RL
loop, just a classification loss over the pairs (the 2026 production default the
S7 handout describes).

The preference we encode here is concrete and measurable: a compact, prose-free
JSON object (chosen) over a chatty, fenced, pretty-printed one (rejected). Both
are *correct* tickets, so this is purely about output discipline, exactly what
SFT alone leaves on the table. After DPO you should see output tokens drop and
zero preambles, verifiable in evaluation.

Imports are lazy inside functions so the module loads on a CPU-only box.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .dataset import SYSTEM_PROMPT


def make_preference_pairs(examples: list[dict]) -> list[dict]:
    """Turn labeled examples into {prompt, chosen, rejected} for DPOTrainer.

    chosen   = the compact gold JSON (what we want).
    rejected = the same facts wrapped in a chatty preamble and a pretty-printed
               fenced block (the slop a model drifts toward without pressure).
    Both decode to the same Ticket, so the only thing DPO can learn is the style.
    """
    pairs = []
    for e in examples:
        compact = json.dumps(e["ticket"], separators=(",", ":"))
        pretty = json.dumps(e["ticket"], indent=2)
        rejected = f"Sure! Here is the extracted ticket:\n```json\n{pretty}\n```"
        pairs.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": e["message"]},
                ],
                "chosen": compact,
                "rejected": rejected,
            }
        )
    return pairs


@dataclass
class DPOConfig:
    sft_adapter: str = "outputs/adapter"  # the SFT adapter to refine
    output_dir: str = "outputs/adapter-dpo"
    beta: float = 0.1  # KL strength; higher keeps it closer to the SFT policy
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 5e-6  # an order below SFT; DPO is a gentle nudge
    max_seq_length: int = 2048


def train_dpo(pref_jsonl: str, base_model: str, cfg: DPOConfig = DPOConfig()) -> str:
    """Refine an SFT adapter with DPO on preference pairs. Returns the adapter dir.

    `pref_jsonl` is one {prompt, chosen, rejected} record per line. We load the
    4-bit base, attach the existing SFT adapter as the starting policy, and let
    TRL's DPOTrainer continue training it. The reference model is the same policy
    frozen, handled internally by TRL.
    """
    import torch  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from trl import DPOConfig as TRLDPOConfig  # noqa: PLC0415
    from trl import DPOTrainer  # noqa: PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.sft_adapter,  # loads base + SFT adapter as the policy
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    # keep training the LoRA params; the adapter is already attached
    FastLanguageModel.for_training(model)

    ds = load_dataset("json", data_files=pref_jsonl, split="train")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL uses the frozen base policy as reference
        args=TRLDPOConfig(
            beta=cfg.beta,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type="cosine",
            bf16=bf16_ok,
            fp16=not bf16_ok,
            logging_steps=5,
            output_dir=cfg.output_dir,
            max_length=cfg.max_seq_length,
        ),
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir

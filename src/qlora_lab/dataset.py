"""Turn labeled examples into SFT chat data, split it, and decontaminate.

This is the step the S7 handout calls the most important and the easiest to get
wrong. Three engineering points live here:

1. Chat format. We build {"messages": [system, user, assistant]} triples. The
   assistant turn is the gold JSON. Unsloth/TRL apply the loss mask so only the
   assistant tokens are trained on, which is why we never put the answer in the
   user turn.

2. EOS. TRL's chat template appends the end-of-turn token, so the model learns
   where to stop. We do not hand-roll this; we rely on the tokenizer's template.

3. Decontamination. The test split must never leak into train. We split first,
   then drop any train example whose message is near-identical to a test message
   (exact match here; swap in embedding similarity for real corpora).
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import SCHEMA_HINT

SYSTEM_PROMPT = (
    "You extract a structured support ticket from a customer message. " + SCHEMA_HINT
)


def to_chat(example: dict) -> dict:
    """One labeled example -> one SFT chat record.

    The assistant content is compact JSON (no spaces) so the model learns a tight
    output, which also trims output tokens at inference time.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["message"]},
            {"role": "assistant", "content": json.dumps(example["ticket"], separators=(",", ":"))},
        ]
    }


def split(examples: list[dict], n_test: int, n_val: int) -> dict[str, list[dict]]:
    """Deterministic tail split: the last n_test go to test, the next n_val to val.

    The input is already shuffled by the seeded generator, so a tail slice is a
    fine random split and it stays stable across runs.
    """
    test = examples[-n_test:]
    val = examples[-(n_test + n_val) : -n_test]
    train = examples[: -(n_test + n_val)]
    return {"train": train, "val": val, "test": test}


def decontaminate(train: list[dict], test: list[dict]) -> tuple[list[dict], int]:
    """Drop train examples whose message exactly matches any test message.

    Returns (clean_train, n_removed). For production corpora replace the exact-set
    check with an embedding-similarity threshold, but the contract is the same:
    nothing the model trains on may appear in what it is graded on.
    """
    test_msgs = {e["message"] for e in test}
    clean = [e for e in train if e["message"] not in test_msgs]
    return clean, len(train) - len(clean)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

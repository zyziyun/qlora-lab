"""Generate the SFT dataset and eval splits. No API key, no GPU.

    python scripts/make_data.py --n 800

Writes data/train.jsonl (chat format for TRL), data/val.jsonl, and
data/test.jsonl (labeled {message, ticket} for evaluation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qlora_lab import dataset as ds  # noqa: E402
from qlora_lab import synth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="total examples to generate")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    examples = synth.gen(args.n, seed=args.seed)
    parts = ds.split(examples, n_test=args.n_test, n_val=args.n_val)
    train, removed = ds.decontaminate(parts["train"], parts["test"])

    out = Path(args.out)
    ds.write_jsonl([ds.to_chat(e) for e in train], out / "train.jsonl")
    ds.write_jsonl([ds.to_chat(e) for e in parts["val"]], out / "val.jsonl")
    ds.write_jsonl(parts["test"], out / "test.jsonl")  # labeled, not chat-formatted

    print(f"train={len(train)} (decontam removed {removed})  val={len(parts['val'])}  test={len(parts['test'])}")
    print(f"wrote {out}/train.jsonl, {out}/val.jsonl, {out}/test.jsonl")


if __name__ == "__main__":
    main()

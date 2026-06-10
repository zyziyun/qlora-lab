"""One prediction interface for every model we compare.

The base model, the fine-tuned adapter served by vLLM, and a frontier API are all
OpenAI-compatible chat endpoints, so a single `extract()` works for all three.
That is the whole trick of a fair comparison: same prompt, same parser, only the
`model` and `base_url` differ.

For running the harness with no key and no GPU, `OfflineStub` returns canned
output so you can see evaluation execute end to end before you spend anything.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .dataset import SYSTEM_PROMPT


@dataclass
class Prediction:
    raw: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


def extract(
    client, model: str, message: str, temperature: float = 0.0, extra_body: dict | None = None
) -> Prediction:
    """Run one extraction against an OpenAI-compatible `client`.

    `client` is an `openai.OpenAI(...)` pointed at any base_url: api.openai.com for
    a frontier baseline, or http://localhost:8000/v1 for a local vLLM serving the
    base model or a LoRA adapter (selected by `model`).

    `extra_body` passes provider-specific options through. The one that matters
    here: vLLM-served Qwen3 has thinking mode on by default, which emits a long
    <think> block before the JSON. Disable it with
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}.
    """
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        extra_body=extra_body,
    )
    dt = time.perf_counter() - t0
    usage = resp.usage
    return Prediction(
        raw=resp.choices[0].message.content or "",
        latency_s=dt,
        prompt_tokens=getattr(usage, "prompt_tokens", 0),
        completion_tokens=getattr(usage, "completion_tokens", 0),
    )


class OfflineStub:
    """A keyless, GPU-free stand-in so the eval harness runs anywhere.

    It echoes a plausible-but-imperfect ticket from the gold label, deliberately
    corrupting a fraction of outputs so schema-validity is not a trivial 100%.
    Use it only to smoke-test the pipeline, never as a real result.
    """

    def __init__(self, gold_by_message: dict[str, dict], break_rate: float = 0.2, seed: int = 0):
        import random as _r

        self._gold = gold_by_message
        self._rng = _r.Random(seed)
        self._break_rate = break_rate

    def chat_completions_create(self, message: str) -> Prediction:
        import json

        gold = self._gold.get(message, {})
        clean = json.dumps(gold, separators=(",", ":"))
        if self._rng.random() < self._break_rate:
            # Simulate the ways a weak base model fails: prose-only (no JSON),
            # truncated JSON, or an out-of-enum value. Some of these the parser
            # recovers, some it cannot, so validity lands realistically below 1.0.
            mode = self._rng.choice(["prose", "truncated", "bad_enum", "fenced"])
            if mode == "prose":
                raw = "This looks like a delivery issue. I'd mark it high priority."
            elif mode == "truncated":
                raw = clean[: max(8, len(clean) // 2)]  # cut off mid-object
            elif mode == "bad_enum":
                raw = clean.replace('"priority":"high"', '"priority":"urgent"').replace(
                    '"priority":"medium"', '"priority":"urgent"'
                )
            else:  # fenced but valid -> parser should still recover this one
                raw = "Sure! Here is the ticket:\n```json\n" + clean + "\n```"
        else:
            raw = clean
        return Prediction(raw=raw, latency_s=0.01, prompt_tokens=120, completion_tokens=40)

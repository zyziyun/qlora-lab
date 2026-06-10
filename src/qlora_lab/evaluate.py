"""Score a model on the held-out test set: validity, accuracy, cost, latency.

These are the four numbers the S7 handout says you must put on the table to
defend "migrated to a self-hosted small model and cut per-query cost". The
comparison is always base vs fine-tuned on the *same* test set.

- schema_validity: fraction of outputs that parse into a `Ticket`. This is the
  headline; prompting a small base model fails here far more than people expect.
- field_accuracy: of the valid ones, how often each enum field matches gold.
- cost / latency: from token usage, so you can show the economics, not just quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Ticket, parse_ticket


@dataclass
class Report:
    n: int
    schema_validity: float
    field_accuracy: dict[str, float]
    exact_match: float
    mean_latency_s: float
    mean_completion_tokens: float
    cost_per_1k_usd: float
    failures: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        fa = ", ".join(f"{k}={v:.2f}" for k, v in self.field_accuracy.items())
        return (
            f"n={self.n}  schema_validity={self.schema_validity:.3f}  "
            f"exact_match={self.exact_match:.3f}  [{fa}]  "
            f"lat={self.mean_latency_s*1000:.0f}ms  "
            f"out_tok={self.mean_completion_tokens:.0f}  "
            f"${self.cost_per_1k_usd:.3f}/1k"
        )


# Cost per token, USD. Defaults are illustrative frontier-ish numbers; for a
# self-hosted model pass your measured $/token from the serving cost notebook.
def _cost_per_1k(prompt_tok: float, completion_tok: float, in_price: float, out_price: float) -> float:
    return 1000 * (prompt_tok * in_price + completion_tok * out_price)


_FIELDS = ("order_id", "issue", "sentiment", "priority")


def evaluate(predictions: list, test: list[dict], in_price: float, out_price: float) -> Report:
    """Score aligned lists of `Prediction` and gold examples.

    `predictions[i]` is the model output for `test[i]` (a {"message","ticket"}).
    """
    assert len(predictions) == len(test), "predictions and test must align"
    n = len(test)
    valid = 0
    exact = 0
    field_hits = {f: 0 for f in _FIELDS}
    field_den = {f: 0 for f in _FIELDS}
    lat = 0.0
    out_tok = 0.0
    in_tok = 0.0
    failures: list[dict] = []

    for pred, ex in zip(predictions, test):
        lat += pred.latency_s
        out_tok += pred.completion_tokens
        in_tok += pred.prompt_tokens
        gold = Ticket(**ex["ticket"])
        ticket, reason = parse_ticket(pred.raw)
        if ticket is None:
            failures.append({"message": ex["message"], "reason": reason, "raw": pred.raw[:200]})
            continue
        valid += 1
        if ticket == gold:
            exact += 1
        for f in _FIELDS:
            field_den[f] += 1
            if getattr(ticket, f) == getattr(gold, f):
                field_hits[f] += 1

    return Report(
        n=n,
        schema_validity=valid / n,
        field_accuracy={f: (field_hits[f] / field_den[f] if field_den[f] else 0.0) for f in _FIELDS},
        exact_match=exact / n,
        mean_latency_s=lat / n,
        mean_completion_tokens=out_tok / n,
        cost_per_1k_usd=_cost_per_1k(in_tok / n, out_tok / n, in_price, out_price),
        failures=failures,
    )


def compare(base: Report, tuned: Report) -> str:
    """A one-glance base-vs-tuned table, the artifact you bring to the interview."""
    rows = [
        ("schema_validity", base.schema_validity, tuned.schema_validity, "{:.3f}"),
        ("exact_match", base.exact_match, tuned.exact_match, "{:.3f}"),
        ("mean_latency_ms", base.mean_latency_s * 1000, tuned.mean_latency_s * 1000, "{:.0f}"),
        ("cost_per_1k_usd", base.cost_per_1k_usd, tuned.cost_per_1k_usd, "{:.3f}"),
    ]
    lines = [f"{'metric':<18} {'base':>10} {'tuned':>10} {'delta':>10}", "-" * 50]
    for name, b, t, fmt in rows:
        delta = t - b
        lines.append(f"{name:<18} {fmt.format(b):>10} {fmt.format(t):>10} {fmt.format(delta):>10}")
    if base.cost_per_1k_usd > 0:
        saved = 100 * (1 - tuned.cost_per_1k_usd / base.cost_per_1k_usd)
        lines.append("-" * 50)
        lines.append(f"cost reduction: {saved:.0f}%")
    return "\n".join(lines)

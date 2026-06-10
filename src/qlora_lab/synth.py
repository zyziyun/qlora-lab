"""Deterministic synthetic data generator, no API key, no GPU.

Real fine-tuning data comes from de-identified production logs or a strong model
generating then a human spot-checking. We cannot ship either in a public lab, so
we synthesize messages from templates with a seeded RNG. The point is that the
*pipeline* is real: the same `gen()` output flows through formatting, training,
and evaluation unchanged. Swap this module for your own loader and nothing else
changes.

Every example carries its gold `Ticket`, so this generator doubles as the
labeler. Determinism (a fixed seed) is what lets the eval set stay stable across
runs, which is the rule the S7 handout hammers on: the eval set is the game.
"""
from __future__ import annotations

import random

from .schema import Issue, Priority, Sentiment, Ticket

# Each issue maps to a few message templates and the sentiment/priority that a
# human labeler would assign. `{oid}` is filled with an order id when present.
_TEMPLATES: dict[Issue, list[tuple[str, Sentiment, Priority]]] = {
    Issue.delivery_delay: [
        ("Order {oid} was supposed to arrive three days ago and still nothing. This is unacceptable.", Sentiment.negative, Priority.high),
        ("Hi, just checking on order {oid}, the tracking has not updated since Monday.", Sentiment.neutral, Priority.medium),
        ("My package {oid} is a little late but no rush, just wanted to confirm it is coming.", Sentiment.neutral, Priority.low),
    ],
    Issue.damaged_item: [
        ("The blender from order {oid} arrived shattered. I want a replacement now.", Sentiment.negative, Priority.high),
        ("Item in {oid} had a small dent on the box but the product seems fine.", Sentiment.neutral, Priority.low),
    ],
    Issue.wrong_item: [
        ("I ordered a blue jacket on {oid} and got a red one. Please fix this.", Sentiment.negative, Priority.medium),
        ("Order {oid} came with the wrong size, can I swap it?", Sentiment.neutral, Priority.medium),
    ],
    Issue.billing_error: [
        ("I was charged twice for order {oid}. Refund the duplicate immediately.", Sentiment.negative, Priority.high),
        ("There is a $4 discrepancy on the invoice for {oid}, can you check?", Sentiment.neutral, Priority.low),
    ],
    Issue.refund_request: [
        ("I changed my mind about order {oid} and would like a full refund please.", Sentiment.neutral, Priority.medium),
        ("Cancel {oid} and refund me. I found it cheaper elsewhere.", Sentiment.negative, Priority.medium),
    ],
    Issue.account_access: [
        ("I cannot log in to my account, the password reset email never arrives. Very frustrating.", Sentiment.negative, Priority.high),
        ("How do I change the email on my account?", Sentiment.neutral, Priority.low),
    ],
    Issue.other: [
        ("Do you ship to Canada? Thinking about placing an order.", Sentiment.positive, Priority.low),
        ("Just wanted to say the support last week was great, thank you!", Sentiment.positive, Priority.low),
    ],
}

_SUMMARIES: dict[Issue, str] = {
    Issue.delivery_delay: "Customer reports a delayed delivery.",
    Issue.damaged_item: "Customer received a damaged item.",
    Issue.wrong_item: "Customer received the wrong item.",
    Issue.billing_error: "Customer reports a billing error.",
    Issue.refund_request: "Customer requests a refund.",
    Issue.account_access: "Customer has an account access problem.",
    Issue.other: "General inquiry or feedback.",
}


# Label-preserving variation axes: a greeting and a closing do not change the
# issue/sentiment/priority, but they make every message textually distinct, which
# keeps the train set from being a handful of duplicates and makes
# decontamination meaningful rather than catastrophic.
_GREETINGS = ["", "Hi, ", "Hello, ", "Hey team, ", "Good morning, ", "Quick one, "]
_CLOSINGS = ["", " Thanks.", " Please advise.", " Appreciate the help.", " Let me know.", " Looking forward to your reply."]


def _order_id(rng: random.Random) -> str:
    return str(rng.randint(10000, 99999))


def _vary(rng: random.Random, core: str) -> str:
    g = rng.choice(_GREETINGS)
    c = rng.choice(_CLOSINGS)
    body = core if not g else g + core[0].lower() + core[1:]
    return body + c


def gen(n: int, seed: int) -> list[dict]:
    """Generate `n` labeled examples.

    Each example is {"message": str, "ticket": dict}. ~15% of messages omit an
    order id (order_id=None) so the model has to learn the null case, not just
    parrot a number out of the text. A greeting/closing are added to vary surface
    form without touching the label.
    """
    rng = random.Random(seed)
    issues = list(_TEMPLATES.keys())
    out: list[dict] = []
    for _ in range(n):
        issue = rng.choice(issues)
        template, sentiment, priority = rng.choice(_TEMPLATES[issue])
        has_oid = "{oid}" in template and rng.random() > 0.15
        oid = _order_id(rng) if has_oid else None
        core = template.format(oid=oid) if oid else template.replace("order {oid}", "my order").replace("{oid}", "my order")
        message = _vary(rng, core)
        ticket = Ticket(
            order_id=oid,
            issue=issue,
            sentiment=sentiment,
            priority=priority,
            summary=_SUMMARIES[issue],
        )
        out.append({"message": message, "ticket": ticket.model_dump(mode="json")})
    return out

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

# Detailed summaries (opt-in, gen(detailed_summary=True)): include the order id so
# the summary carries the specific fact, not a canned sentence. Used to test the
# dual of "the model learns your data's shape" - feed it detail, it emits detail.
_DETAILED: dict[Issue, str] = {
    Issue.delivery_delay: "Order {oid} is delayed and has not arrived.",
    Issue.damaged_item: "Order {oid} arrived damaged.",
    Issue.wrong_item: "Order {oid} contained the wrong item.",
    Issue.billing_error: "Order {oid} was billed incorrectly.",
    Issue.refund_request: "Customer requests a refund for order {oid}.",
    Issue.account_access: "Customer cannot access their account.",
    Issue.other: "General inquiry or feedback.",
}

# Out-of-distribution styles the default templates never produce: slang and formal
# register, each with a distractor number sitting next to the real order id (the
# pressure that made a 1.7B blend "#88321 ... 2 weeks" into "28321"). Two uses:
# gen(diverse=True) mixes these into TRAINING to test whether data diversity fixes
# the OOD gap; gen_ood() builds a held-out TEST set from a separate wording list so
# the two never share exact strings.
_DIVERSE_TEMPLATES: dict[Issue, list[tuple[str, Sentiment, Priority]]] = {
    Issue.delivery_delay: [
        ("ugh i ordered like 2 weeks ago?? #{oid} and STILL nothing, so done with this", Sentiment.negative, Priority.high),
        ("I am writing to inquire about order {oid}, placed on the 3rd, which remains undelivered after 9 days.", Sentiment.neutral, Priority.medium),
    ],
    Issue.damaged_item: [
        ("box #{oid} showed up looking like it survived 3 wars, totally crushed, want a new one", Sentiment.negative, Priority.high),
        ("Regarding shipment {oid}, the unit arrived with damage to 2 of its corners.", Sentiment.neutral, Priority.medium),
    ],
    Issue.wrong_item: [
        ("lol ordered 1 thing on {oid} got something completely different, fix pls", Sentiment.negative, Priority.medium),
        ("Order {oid} was fulfilled with an incorrect item, model 4 instead of model 7.", Sentiment.neutral, Priority.medium),
    ],
    Issue.billing_error: [
        ("yall charged me 2x on {oid}, thats like 80 bucks, refund the extra NOW", Sentiment.negative, Priority.high),
        ("I observed a duplicate charge of 2 units against invoice {oid}; please reverse it.", Sentiment.neutral, Priority.medium),
    ],
    Issue.refund_request: [
        ("changed my mind on {oid}, found it 15% cheaper elsewhere, want my money back", Sentiment.neutral, Priority.medium),
        ("I would like to request a refund for order {oid}, purchased 4 days ago.", Sentiment.neutral, Priority.medium),
    ],
    Issue.account_access: [
        ("cant get into my acct, reset email never comes, tried like 5 times, ridiculous", Sentiment.negative, Priority.high),
        ("I am unable to access my account; the verification step fails repeatedly.", Sentiment.neutral, Priority.medium),
    ],
    Issue.other: [
        ("yo do u ship to 3 countries in the EU or just 1", Sentiment.positive, Priority.low),
        ("I wanted to commend your team for resolving my last 2 inquiries so quickly.", Sentiment.positive, Priority.low),
    ],
}

# Held-out OOD test wordings, same registers as _DIVERSE_TEMPLATES but distinct
# strings so a model trained on diverse data has not memorized these.
_OOD_TEMPLATES: dict[Issue, list[tuple[str, Sentiment, Priority]]] = {
    Issue.delivery_delay: [
        ("ordered 2 weeks back, #{oid}, nothing has moved, im furious", Sentiment.negative, Priority.high),
        ("Could you advise on order {oid}? It was due 5 days ago and has not shipped.", Sentiment.neutral, Priority.medium),
    ],
    Issue.damaged_item: [
        ("#{oid} arrived smashed, like 4 cracks across it, need a replacement", Sentiment.negative, Priority.high),
    ],
    Issue.wrong_item: [
        ("got the wrong thing on {oid} again, this is the 3rd time, sort it out", Sentiment.negative, Priority.medium),
    ],
    Issue.billing_error: [
        ("double billed on {oid}, 2 charges of 40 dollars, reverse one", Sentiment.negative, Priority.high),
    ],
    Issue.refund_request: [
        ("Please process a refund for {oid}; I returned it 6 days ago.", Sentiment.neutral, Priority.medium),
    ],
    Issue.account_access: [
        ("locked out of my account, reset link dead, this is the 2nd day", Sentiment.negative, Priority.high),
    ],
    Issue.other: [
        ("do you offer gift wrapping on orders over 2 items", Sentiment.positive, Priority.low),
    ],
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


def _fill(template: str, oid: str | None) -> str:
    if oid:
        return template.format(oid=oid)
    return template.replace("order {oid}", "my order").replace("#{oid}", "my order").replace("{oid}", "my order")


def gen(n: int, seed: int, diverse: bool = False, detailed_summary: bool = False) -> list[dict]:
    """Generate `n` labeled examples.

    Each example is {"message": str, "ticket": dict}. ~15% of messages omit an
    order id (order_id=None) so the model has to learn the null case, not just
    parrot a number out of the text. A greeting/closing are added to vary surface
    form without touching the label.

    `diverse=True` mixes in slang/formal/distractor templates (experiment: does
    data diversity fix the OOD gap?). `detailed_summary=True` swaps canned
    summaries for ones that include the order id (experiment: feed detail, get
    detail). Both default to False so the canonical eval set stays byte-identical.
    """
    rng = random.Random(seed)
    issues = list(_TEMPLATES.keys())
    summaries = _DETAILED if detailed_summary else _SUMMARIES
    out: list[dict] = []
    for _ in range(n):
        issue = rng.choice(issues)
        pool = _TEMPLATES[issue] + (_DIVERSE_TEMPLATES[issue] if diverse else [])
        template, sentiment, priority = rng.choice(pool)
        has_oid = "{oid}" in template and rng.random() > 0.15
        oid = _order_id(rng) if has_oid else None
        core = _fill(template, oid)
        message = _vary(rng, core)
        ticket = Ticket(
            order_id=oid,
            issue=issue,
            sentiment=sentiment,
            priority=priority,
            summary=_fill(summaries[issue], oid) if detailed_summary else summaries[issue],
        )
        out.append({"message": message, "ticket": ticket.model_dump(mode="json")})
    return out


def gen_ood(seed: int = 99, detailed_summary: bool = False) -> list[dict]:
    """A held-out out-of-distribution test set: every _OOD_TEMPLATES wording once.

    These styles (slang, formal, distractor numbers) never appear in default
    training data, so scoring a default-trained model here measures generalization,
    and scoring a `diverse=True`-trained model here measures whether more diverse
    data closed the gap. No greeting/closing wrapping; the styles are already
    varied, and a stable set keeps the OOD number comparable across runs.
    """
    rng = random.Random(seed)
    summaries = _DETAILED if detailed_summary else _SUMMARIES
    out: list[dict] = []
    for issue, templates in _OOD_TEMPLATES.items():
        for template, sentiment, priority in templates:
            oid = _order_id(rng) if "{oid}" in template else None
            ticket = Ticket(
                order_id=oid,
                issue=issue,
                sentiment=sentiment,
                priority=priority,
                summary=_fill(summaries[issue], oid) if detailed_summary else summaries[issue],
            )
            out.append({"message": _fill(template, oid), "ticket": ticket.model_dump(mode="json")})
    return out

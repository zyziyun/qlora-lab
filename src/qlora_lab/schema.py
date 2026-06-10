"""The task contract: a support message in, a strict JSON ticket out.

Everything in this lab is organized around this one schema. The whole point of
fine-tuning a small model here is to make it emit JSON that validates against
`Ticket` as reliably as a frontier model does, but at a fraction of the cost.

Keep this module dependency-light: it is imported by data generation, training
data formatting, evaluation, and serving alike.
"""
from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, ValidationError, field_validator


class Issue(str, Enum):
    delivery_delay = "delivery_delay"
    damaged_item = "damaged_item"
    wrong_item = "wrong_item"
    billing_error = "billing_error"
    refund_request = "refund_request"
    account_access = "account_access"
    other = "other"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Ticket(BaseModel):
    """One support ticket extracted from a customer message.

    `order_id` is optional because not every message mentions one. The four
    enum fields are what we actually score field-accuracy on in evaluation.
    """

    order_id: str | None = None
    issue: Issue
    sentiment: Sentiment
    priority: Priority
    summary: str

    @field_validator("summary")
    @classmethod
    def _summary_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("summary must be non-empty")
        return v.strip()


# The single source of truth for the JSON shape, injected into prompts so the
# base model and the fine-tuned model are asked for exactly the same thing.
SCHEMA_HINT = (
    "Return a single JSON object with keys: "
    'order_id (string or null), '
    f"issue (one of {[e.value for e in Issue]}), "
    f"sentiment (one of {[e.value for e in Sentiment]}), "
    f"priority (one of {[e.value for e in Priority]}), "
    "summary (string). Output JSON only, no prose."
)


def parse_ticket(raw: str) -> tuple[Ticket | None, str | None]:
    """Parse model output into a Ticket.

    Returns (ticket, None) on success or (None, reason) on failure. We tolerate
    a model that wraps JSON in ```json fences or adds a trailing sentence, since
    that is exactly the kind of slop fine-tuning is meant to remove: we measure
    how often it happens, we do not pretend it never does.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") : text.rfind("}") + 1] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None, "no_json_object"
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        return None, f"invalid_json: {e.msg}"
    try:
        return Ticket(**obj), None
    except ValidationError as e:
        return None, f"schema_violation: {e.errors()[0]['msg']}"

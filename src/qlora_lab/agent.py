"""Plug the fine-tuned model into an agent as a specialized sub-model.

The S7 punchline: fine-tuning is not a standalone trophy, it is a cheap, fast
component inside a larger system. Here the tuned extractor is one tool the agent
calls on its high-frequency narrow step, while a general model handles routing
and conversation. This is the "self-hosted small model for a high-volume step"
pattern straight from the VortexNet resume bullet.
"""
from __future__ import annotations

from collections import Counter

from .predict import extract
from .schema import Ticket, parse_ticket


def order_id_in_message(ticket: Ticket, message: str) -> bool:
    """Deterministic guardrail: a claimed order_id must appear in the message.

    This catches the failure that matters most. A small model on out-of-
    distribution text can hallucinate an order id by blending nearby digits
    (observed: a 1.7B turned "#88321 ... 2 weeks" into "28321"). A wrong id is
    worse than a missing one because it points downstream actions at the wrong
    order. A null id is fine, there was nothing to verify.
    """
    if ticket.order_id is None:
        return True
    return ticket.order_id in message


def extract_ticket_tool(client, model: str, message: str, guard: bool = True) -> dict:
    """The tool the agent exposes: message -> validated ticket dict.

    Runs on the cheap self-hosted tuned model. Two gates, both returning a
    structured error instead of raising so the agent can retry or escalate:
    the schema parse, and (when `guard`) the order_id substring check.
    """
    pred = extract(client, model, message)
    ticket, reason = parse_ticket(pred.raw)
    if ticket is None:
        return {"ok": False, "error": reason, "tokens": pred.completion_tokens}
    if guard and not order_id_in_message(ticket, message):
        return {"ok": False, "error": "order_id_not_in_message", "tokens": pred.completion_tokens}
    return {"ok": True, "ticket": ticket.model_dump(mode="json"), "tokens": pred.completion_tokens}


def route_with_fallback(
    message: str, cheap_client, cheap_model: str, strong_client, strong_model: str, guard: bool = True
) -> dict:
    """Cheap-first routing: try the tuned small model, fall back to a strong one.

    This is the economics in code: the tuned model absorbs the bulk of traffic at
    a fraction of the cost, and only the cases it cannot validate (bad schema or a
    failed order_id guard) escalate to the expensive model. Measure the escalation
    rate; that, times the price gap, is your real savings.
    """
    result = extract_ticket_tool(cheap_client, cheap_model, message, guard=guard)
    if result["ok"]:
        result["served_by"] = "tuned_small"
        return result
    escalation_reason = result["error"]
    pred = extract(strong_client, strong_model, message)
    ticket, reason = parse_ticket(pred.raw)
    return {
        "ok": ticket is not None,
        "ticket": ticket.model_dump(mode="json") if ticket else None,
        "error": None if ticket else reason,
        "served_by": "strong_fallback",
        "escalation_reason": escalation_reason,
    }


def measure_routing(messages, cheap_client, cheap_model, strong_client, strong_model, guard=True, cost_ratio=20.0):
    """Run a batch through the router and report the economics.

    `cost_ratio` is the strong model's per-query price divided by the cheap one's.
    The naive cost is "send everything to the strong model" = N * cost_ratio. The
    routed cost is "cheap for all, strong for the escalations". The saving is the
    headline number you defend in an interview.
    """
    results = [route_with_fallback(m, cheap_client, cheap_model, strong_client, strong_model, guard) for m in messages]
    n = len(results)
    served = Counter(r["served_by"] for r in results)
    escalated = served["strong_fallback"]
    reasons = Counter(r["escalation_reason"] for r in results if r["served_by"] == "strong_fallback")
    routed_cost = (n * 1.0) + (escalated * cost_ratio)  # one cheap call each, plus a strong call per escalation
    naive_cost = n * cost_ratio
    return {
        "n": n,
        "served_cheap": served["tuned_small"],
        "escalated": escalated,
        "escalation_rate": escalated / n if n else 0.0,
        "escalation_reasons": dict(reasons),
        "cost_vs_all_strong": routed_cost / naive_cost if naive_cost else 1.0,
        "saving_vs_all_strong": 1 - (routed_cost / naive_cost) if naive_cost else 0.0,
    }


# A minimal tool schema, the shape you would register with any agent framework.
TICKET_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "extract_ticket",
        "description": "Extract a structured support ticket from a customer message.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
}

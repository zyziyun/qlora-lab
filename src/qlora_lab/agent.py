"""Plug the fine-tuned model into an agent as a specialized sub-model.

The S7 punchline: fine-tuning is not a standalone trophy, it is a cheap, fast
component inside a larger system. Here the tuned extractor is one tool the agent
calls on its high-frequency narrow step, while a general model handles routing
and conversation. This is the "self-hosted small model for a high-volume step"
pattern straight from the VortexNet resume bullet.
"""
from __future__ import annotations

from .predict import extract
from .schema import Ticket, parse_ticket


def extract_ticket_tool(client, model: str, message: str) -> dict:
    """The tool the agent exposes: message -> validated ticket dict.

    Runs on the cheap self-hosted tuned model. On a schema failure it returns a
    structured error rather than raising, so the agent can retry or fall back to
    a stronger model, the same guardrail pattern as the S3/S4 handouts.
    """
    pred = extract(client, model, message)
    ticket, reason = parse_ticket(pred.raw)
    if ticket is None:
        return {"ok": False, "error": reason, "tokens": pred.completion_tokens}
    return {"ok": True, "ticket": ticket.model_dump(mode="json"), "tokens": pred.completion_tokens}


def route_with_fallback(message: str, cheap_client, cheap_model: str, strong_client, strong_model: str) -> dict:
    """Cheap-first routing: try the tuned small model, fall back to a strong one.

    This is the economics in code: the tuned model absorbs the bulk of traffic at
    a fraction of the cost, and only the cases it cannot validate escalate to the
    expensive model. Measure the escalation rate; that, times the price gap, is
    your real savings.
    """
    result = extract_ticket_tool(cheap_client, cheap_model, message)
    if result["ok"]:
        result["served_by"] = "tuned_small"
        return result
    pred = extract(strong_client, strong_model, message)
    ticket, reason = parse_ticket(pred.raw)
    return {
        "ok": ticket is not None,
        "ticket": ticket.model_dump(mode="json") if ticket else None,
        "error": None if ticket else reason,
        "served_by": "strong_fallback",
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

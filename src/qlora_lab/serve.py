"""Two ways to serve the result, and when to pick each.

The S7 handout's trade-off, made runnable:

- merge_adapter(): fold LoRA back into the base for one self-contained model.
  Simplest to ship, zero adapter-routing latency, but one full model per task.

- vLLM LoRA serving: load the base once and hot-swap many adapters by request.
  One GPU + N small adapter files serves N task variants. This is the ML-infra
  answer and the reason adapters beat full fine-tunes operationally.

Both expose an OpenAI-compatible endpoint, so `predict.extract()` and the whole
eval harness work against either without change.
"""
from __future__ import annotations


def merge_adapter(adapter_dir: str, out_dir: str, base_model: str) -> str:
    """Merge a LoRA adapter into its base and save a standalone model."""
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    return out_dir


# Start vLLM with LoRA hot-swapping enabled (run as a shell command, not Python):
#
#   vllm serve <BASE_MODEL> \
#       --enable-lora \
#       --lora-modules ticket=outputs/adapter \
#       --max-loras 4 --max-lora-rank 16
#
# Then the served model name "ticket" selects the adapter:
VLLM_SERVE_HINT = (
    "vllm serve {base} --enable-lora "
    "--lora-modules ticket={adapter} --max-loras 4 --max-lora-rank 16"
)


def openai_client(base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
    """An OpenAI client pointed at a local vLLM server.

    Pass model="ticket" to hit the adapter, or model=<BASE_MODEL> to hit the raw
    base on the same server, which is exactly the base-vs-tuned A/B you evaluate.
    """
    from openai import OpenAI  # noqa: PLC0415

    return OpenAI(base_url=base_url, api_key=api_key)

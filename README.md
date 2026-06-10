# qlora-lab

A hands-on, production-grade QLoRA fine-tuning lab built around one narrow, high-frequency task: turning a free-text customer-support message into a strict JSON ticket. Nine notebooks walk from *should you even fine-tune* through data prep, QLoRA training on a free Colab T4, vLLM adapter serving, a base-vs-fine-tuned comparison, and plugging the tuned model into an agent as a cheap sub-model. A trained 1.7B adapter ships with the repo, so you can serve and A/B it without training anything first.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

This is the runnable companion to the S7 handout. The point is not "I fine-tuned a model" but "I migrated a high-volume narrow task off a frontier API onto a self-hosted small model, and proved with the same eval set that it is both cheaper and more reliable."

---

## What you measure (and where the number is real)

The whole lab funnels to one base-vs-tuned table on a held-out test set:

| Metric | What it tells you |
|---|---|
| `schema_validity` | fraction of outputs that parse into a valid `Ticket`. The headline. |
| `exact_match` | of valid outputs, how many match gold on every field. |
| `cost_per_1k_usd` | from token usage and your `$/token`. The economics. |
| `mean_latency_ms` | per-request latency. |

**Offline (no key, no GPU)** — notebook `00` runs the entire harness with a deterministic stub standing in for a weak base and a strong tuned model, so you see the pipeline produce numbers before spending anything:

```
metric                   base      tuned      delta
--------------------------------------------------
schema_validity         0.690      0.970      0.280
exact_match             0.690      0.970      0.280
cost_per_1k_usd         0.700      0.014     -0.686
--------------------------------------------------
cost reduction: 98%
```

> Those are stub numbers, deliberately illustrative. The real numbers are below.

---

## Headline result (measured, not claimed)

From a real run of `notebooks/colab_t4_run.ipynb` on a Colab L4, 2026-06-10. Same 100-example held-out test set throughout; QLoRA r16/alpha32, all linear layers, 3 epochs (loss plateaus by step ~45 of 216 — 1 epoch would do).

**Finding 1 — modern instruct bases already nail JSON formatting; what degrades at small scale is judgment.** Both bases scored `schema_validity = 1.000` prompted. The convention-dependent fields are where they fail:

| field accuracy (prompted base) | Qwen3-8B | Qwen3-1.7B |
|---|---:|---:|
| order_id | 1.00 | 0.98 |
| issue | 1.00 | 0.95 |
| sentiment | 0.88 | **0.71** |
| priority | 0.58 | **0.50** |

`priority = 0.50` is a coin flip: the base model has its own opinion of what "high" means, not ours. That labeling convention is exactly what the fine-tune teaches.

**Finding 2 — after fine-tuning, the 1.7B matches the 8B**, every field at 1.00 on both. So you serve the model 4.7× smaller:

| metric | 1.7B base | 1.7B tuned | Δ |
|---|---:|---:|---:|
| priority accuracy | 0.50 | **1.00** | +0.50 |
| sentiment accuracy | 0.71 | **1.00** | +0.29 |
| output tokens / query | 56 | **32** | −42% |
| mean latency (L4) | 2515 ms | **2067 ms** | −18% |
| cost / 1k queries (same GPU) | $0.018 | $0.013 | −26% |

Trainable parameters on the 1.7B: 17.4M of 1.74B, **1.00% trained** — the LoRA promise, verbatim from the training log.

**Finding 3 — the adapter survives the serving transition.** Served by vLLM (`notebooks/colab_serve_vllm.ipynb`) with the adapter hot-swapped onto the **fp16** base it was never trained on (QLoRA trains against the 4-bit base), the same picture reproduced on a 30-example A/B: base sentiment 0.67 / priority 0.47, tuned all 1.00, tuned latency 563ms vs base 767ms on an H100. One server, two model names, one ~67MB adapter file — and the tuned model's compact no-space JSON output is the training data's `separators=(",", ":")` choice, visible byte for byte.

**Caveats, before you quote this anywhere:** the data is synthetic and the test set is drawn from the same template family as training, so these are in-distribution numbers. On a 5-message out-of-distribution spot check (slang, formal register, invoice-vs-order ambiguity, multi-issue messages) both adapters stayed 5/5 schema-valid, but they are not equal off-distribution: the 8B adapter got every field right, while the 1.7B miscopied an order id (`#88321` → `"28321"`, blending a nearby digit), put an invoice number into `order_id` where the 8B correctly said null, and drifted on priority. So *finding 2 holds in-distribution; OOD robustness still scales with base size*. If you serve the small one, add the cheap deterministic guardrail: an extracted `order_id` must be a substring of the message, else null it or escalate (see `agent.py`'s fallback routing). `exact_match = 0.000` for the bases is an artifact — gold summaries are canned template strings the base model can't guess; read `field_accuracy` for the fair base score. The −26% cost figure is base-vs-tuned on the *same* GPU; the order-of-magnitude story is frontier-API-vs-self-hosted, measured in notebook `01`.

---

## Quickstart

```bash
# 1. Core deps (CPU only) — data, schema, evaluation, agent wiring
pip install -r requirements.txt

# 2. Generate the dataset (no API key, no GPU)
python scripts/make_data.py --n 800
#   -> data/train.jsonl (chat format), data/val.jsonl, data/test.jsonl (labeled)

# 3. Build the notebooks
python scripts/build_notebooks.py

# 4. Open notebooks/00_quickstart.ipynb — runs end to end offline in ~5 min
```

Training and serving need a GPU. Two self-contained Colab paths, both verified end to end:

- **Train + compare**: `notebooks/colab_t4_run.ipynb` — pick a T4/L4 runtime, Run all. Clones the repo, measures the raw base, trains the QLoRA adapter, prints the base-vs-tuned table, downloads the adapter. ~30-50 min on T4, ~20 min on L4.
- **Serve + hot-swap A/B** (no training needed): `notebooks/colab_serve_vllm.ipynb` — the bundled `outputs/adapter-1.7b` is hot-swapped onto the fp16 base by one vLLM server; the client picks base or adapter per request by model name. Includes the CUDA-wheel-matching install cells, because `ImportError: libcudart.so.13` is the #1 self-hosting pitfall and you should meet it here, not in production.

---

## Reading path

Each module is short enough to read in one sitting. Map them to the S7 handout:

| Read | File | Handout section |
|---|---|---|
| 1 | `notebooks/00_quickstart.ipynb` | feel the whole loop offline |
| 2 | `src/qlora_lab/schema.py` | the task contract: `Ticket` + `parse_ticket` |
| 3 | `notebooks/01_decide_baseline.ipynb` | 一 decision tree, measure the baseline first |
| 4 | `src/qlora_lab/synth.py`, `dataset.py` | 二 SFT data: chat format, loss mask, decontamination |
| 5 | `src/qlora_lab/train.py` + `notebooks/03` | 三 QLoRA with Unsloth + TRL, r16/alpha32, all-linear |
| 6 | `src/qlora_lab/serve.py` + `notebooks/04` | 四 merge vs vLLM adapter hot-swap |
| 7 | `src/qlora_lab/evaluate.py` + `notebooks/05` | 五 base vs fine-tuned: validity, cost, latency |
| 8 | `src/qlora_lab/agent.py` + `notebooks/06` | 六 use it as an agent sub-model with fallback routing |
| 9 | `notebooks/colab_t4_run.ipynb` | the whole train + compare loop, one Run all |
| 10 | `notebooks/colab_serve_vllm.ipynb` | vLLM hot-swap serving, base-vs-adapter A/B |

## How to use it

- **First pass**: run notebook `00` offline, then `01` against a real model to get an honest baseline.
- **Second pass**: open `colab_t4_run.ipynb` on a T4/L4, Run all, and read the real base-vs-tuned table off your own run.
- **Third pass**: open `colab_serve_vllm.ipynb` and serve the adapter with vLLM — same numbers, production-shaped stack.
- **Make it yours**: replace `synth.gen` with a loader for your own labeled data; nothing downstream changes. Swap the task schema in `schema.py` and you have a fine-tuning lab for any structured-extraction problem.

**Experiments worth running next** (each one produced a finding above):

| Experiment | How | What it tells you |
|---|---|---|
| OOD spot check | feed hand-written off-template messages to the tuned model | whether it learned the task or the templates |
| Specialization gate | query the tuned model *without* the extraction system prompt | LoRA adds a prompt-gated behavior, base ability intact |
| 1 epoch vs 3 | `TrainConfig(epochs=1)` | loss plateaus by step ~45; watch curves, not epoch counts |
| Rank ablation | r8 vs r16 vs r64 | is r8 really within 1-2% on your task |
| Multi-base benchmark | swap `base_model`, keep data fixed | pick the Pareto point, not the biggest model |
| Frontier baseline | notebook `01` with an API key | the order-of-magnitude cost story vs self-hosting |

## Layout

```
src/qlora_lab/
  schema.py     Ticket schema + tolerant parser (the contract)
  synth.py      deterministic labeled-data generator (no API)
  dataset.py    SFT chat formatting, split, decontamination
  predict.py    one OpenAI-compatible extract() + offline stub
  evaluate.py   schema validity / field accuracy / cost / latency + compare()
  train.py      QLoRA via Unsloth + TRL
  serve.py      adapter merge, vLLM LoRA serving
  agent.py      tuned model as a sub-model, cheap-first routing with fallback
scripts/
  make_data.py        generate data/*.jsonl
  build_notebooks.py  emit notebooks/ (deterministic cell ids, rebuilds are byte-identical)
notebooks/            00-06 concepts + colab_t4_run (train) + colab_serve_vllm (serve)
outputs/
  adapter-1.7b/       trained QLoRA adapter, ships with the repo (67MB)
```

## Honest scope

This is a teaching reference, not a benchmark paper. The data is synthetic so the lab runs anywhere; the model code is real and runs unmodified on a T4. The numbers that matter are the ones *you* produce in notebooks `01` and `05` — reproduce them, do not copy a claim.

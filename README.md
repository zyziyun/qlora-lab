# qlora-lab

A hands-on, production-grade QLoRA fine-tuning lab built around one narrow, high-frequency task: turning a free-text customer-support message into a strict JSON ticket. Seven notebooks walk from *should you even fine-tune* through data prep, QLoRA training on a free Colab T4, vLLM adapter serving, a base-vs-fine-tuned comparison, and plugging the tuned model into an agent as a cheap sub-model.

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

> Those are stub numbers, deliberately illustrative. The **real** numbers come from notebooks `01` (a true prompting baseline) and `05` (your actual QLoRA adapter served by vLLM). For reference, the literature point this task tends to land on: a small base model prompts at roughly 70% schema validity and a QLoRA fine-tune pushes it past 95%, with self-hosting an order of magnitude cheaper per token than a frontier API. Reproduce it, do not quote it.

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

Training (notebook `03`) and serving (`04`) need a GPU. The fastest path is **`notebooks/colab_t4_run.ipynb`**: open it in Colab, pick a T4 runtime, Run all — it clones the repo, measures the raw base, trains the QLoRA adapter, prints the base-vs-tuned table, and downloads the adapter, in roughly 30-50 minutes.

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

## How to use it

- **First pass**: run notebook `00` offline, then `01` against a real model to get an honest baseline.
- **Second pass**: on a T4, run `02` → `03` → `04` → `05` and read the real base-vs-tuned table off your own run.
- **Make it yours**: replace `synth.gen` with a loader for your own labeled data; nothing downstream changes. Swap the task schema in `schema.py` and you have a fine-tuning lab for any structured-extraction problem.

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
  build_notebooks.py  emit notebooks/
```

## Honest scope

This is a teaching reference, not a benchmark paper. The data is synthetic so the lab runs anywhere; the model code is real and runs unmodified on a T4. The numbers that matter are the ones *you* produce in notebooks `01` and `05` — reproduce them, do not copy a claim.

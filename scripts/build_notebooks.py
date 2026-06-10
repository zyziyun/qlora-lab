"""Builds the lab notebooks. Idempotent - re-run anytime to refresh.

    python scripts/build_notebooks.py

Mirrors the fin-rag-lab convention: notebooks are generated from this source so
the prose and code stay in one reviewable place.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).parent.parent / "notebooks"


def build(cells, path: Path):
    nb = nbf.v4.new_notebook()
    # deterministic cell ids: nbformat randomizes them per call, which would
    # dirty every notebook on every rebuild
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i}"
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"  built {path.name}")


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


PATH_SETUP = code(
    "import sys; sys.path.insert(0, '../src')\n"
    "from qlora_lab import schema, synth, dataset, predict, evaluate, train, serve, agent"
)


# =====================================================================
# 00 - Quickstart: feel the whole pipeline offline (no key, no GPU)
# =====================================================================
def build_00():
    cells = [
        md(
            "# 00 - Quickstart: the whole loop in 5 minutes, offline\n\n"
            "**Goal**: see decide -> data -> evaluate -> compare run end to end with no API\n"
            "key and no GPU, using a deterministic stub model. Once the shape is clear, the\n"
            "later notebooks swap the stub for a real base model and a real QLoRA fine-tune.\n\n"
            "**The task**: turn a free-text support message into a strict JSON `Ticket`.\n"
            "This is the canonical 'high-frequency narrow task' you migrate off a frontier\n"
            "API onto a cheap self-hosted small model."
        ),
        PATH_SETUP,
        md("### 1. The contract\nEverything is organized around one schema."),
        code(
            "print(schema.SCHEMA_HINT)\n"
            "t, err = schema.parse_ticket('```json\\n{\"order_id\":\"12345\",\"issue\":\"delivery_delay\","
            "\"sentiment\":\"negative\",\"priority\":\"high\",\"summary\":\"late\"}\\n```')\n"
            "print(t, '|', err)"
        ),
        md("### 2. Generate labeled data (deterministic, no API)"),
        code(
            "examples = synth.gen(800, seed=7)\n"
            "parts = dataset.split(examples, n_test=100, n_val=100)\n"
            "train, removed = dataset.decontaminate(parts['train'], parts['test'])\n"
            "print('train', len(train), 'val', len(parts['val']), 'test', len(parts['test']), 'decontam removed', removed)\n"
            "print(examples[0]['message'])\nprint(examples[0]['ticket'])"
        ),
        md(
            "### 3. Score two models on the same test set\n"
            "We stand in a weak 'base' (fails often) and a strong 'tuned' (rarely fails)\n"
            "with the offline stub. Note: instantiate each stub **once** and reuse it."
        ),
        code(
            "test = parts['test']\n"
            "gold = {e['message']: e['ticket'] for e in test}\n"
            "base_stub = predict.OfflineStub(gold, break_rate=0.45, seed=1)\n"
            "tuned_stub = predict.OfflineStub(gold, break_rate=0.05, seed=2)\n"
            "base_preds = [base_stub.chat_completions_create(e['message']) for e in test]\n"
            "tuned_preds = [tuned_stub.chat_completions_create(e['message']) for e in test]\n"
            "rb = evaluate.evaluate(base_preds, test, in_price=2.5e-6, out_price=10e-6)\n"
            "rt = evaluate.evaluate(tuned_preds, test, in_price=0.05e-6, out_price=0.20e-6)\n"
            "print(evaluate.compare(rb, rt))"
        ),
        md(
            "The `schema_validity` gap and the cost gap are the two numbers this whole\n"
            "lab exists to produce for real. Next notebooks replace the stub with actual\n"
            "models. **The eval set is the game** - everything else serves these numbers."
        ),
    ]
    build(cells, NB / "00_quickstart.ipynb")


# =====================================================================
# 01 - Decide + measure the baseline
# =====================================================================
def build_01():
    cells = [
        md(
            "# 01 - Decide, then measure the baseline\n\n"
            "Most tasks should **not** be fine-tuned. Before training anything, prove the\n"
            "baseline is not good enough. This notebook builds the eval set and measures a\n"
            "prompting baseline against a real model.\n\n"
            "**Decision tree** (from the S7 handout):\n"
            "1. Build a 100+ eval set, measure the base model with a good prompt.\n"
            "2. Try prompt + few-shot. If it closes 80% of the gap, stop - no fine-tune.\n"
            "3. If the failure is factual, try RAG. If that closes it, stop.\n"
            "4. Only now fine-tune, and re-measure on the **same** eval set.\n"
            "5. If fine-tuning does not gain 10%+, your fine-tune is wrong, not the model."
        ),
        PATH_SETUP,
        md(
            "### Point the client at any OpenAI-compatible model\n"
            "Use a frontier API as the 'can it be done at all' ceiling, or a small base\n"
            "model served locally by vLLM as the honest baseline you will try to beat."
        ),
        code(
            "import os\n"
            "from openai import OpenAI\n"
            "# frontier baseline:\n"
            "client = OpenAI()  # reads OPENAI_API_KEY\n"
            "MODEL = 'gpt-4o-mini'\n"
            "# local small base instead:\n"
            "# client = serve.openai_client('http://localhost:8000/v1')\n"
            "# MODEL = 'unsloth/Qwen3-8B-bnb-4bit'"
        ),
        code(
            "test = dataset.read_jsonl('../data/test.jsonl')  # from scripts/make_data.py\n"
            "preds = [predict.extract(client, MODEL, e['message']) for e in test]\n"
            "rep = evaluate.evaluate(preds, test, in_price=0.15e-6, out_price=0.60e-6)\n"
            "print(rep.summary())\n"
            "print('first failures:', [f['reason'] for f in rep.failures[:5]])"
        ),
        md(
            "Save this baseline. The fine-tune in notebook 03 has to beat it on the same\n"
            "100 test examples or it is not worth shipping."
        ),
    ]
    build(cells, NB / "01_decide_baseline.ipynb")


# =====================================================================
# 02 - Data preparation
# =====================================================================
def build_02():
    cells = [
        md(
            "# 02 - Data: the step that decides everything\n\n"
            "Three engineering points the handout calls the easy-to-get-wrong ones:\n\n"
            "1. **Chat format** - the gold JSON goes in the *assistant* turn, never the user\n"
            "   turn, so the loss mask trains on the answer only.\n"
            "2. **EOS** - the tokenizer chat template appends the stop token; we do not\n"
            "   hand-roll it.\n"
            "3. **Decontamination** - split first, then drop any train item that matches a\n"
            "   test item. Nothing trained on may appear in what it is graded on."
        ),
        PATH_SETUP,
        code(
            "examples = synth.gen(800, seed=7)\n"
            "parts = dataset.split(examples, n_test=100, n_val=100)\n"
            "train, removed = dataset.decontaminate(parts['train'], parts['test'])\n"
            "print('decontam removed', removed, 'leaving', len(train), 'train')"
        ),
        md("### What one SFT record looks like\nSystem + user + assistant; assistant is the gold JSON."),
        code(
            "rec = dataset.to_chat(train[0])\n"
            "import json; print(json.dumps(rec, indent=2)[:600])"
        ),
        md("### Write the files TRL will read"),
        code(
            "dataset.write_jsonl([dataset.to_chat(e) for e in train], '../data/train.jsonl')\n"
            "dataset.write_jsonl([dataset.to_chat(e) for e in parts['val']], '../data/val.jsonl')\n"
            "dataset.write_jsonl(parts['test'], '../data/test.jsonl')\n"
            "print('wrote train/val/test')"
        ),
        md(
            "For real data, replace `synth.gen` with your loader: de-identified production\n"
            "logs, or a strong model generating then a human spot-checking. Everything\n"
            "downstream is unchanged."
        ),
    ]
    build(cells, NB / "02_data.ipynb")


# =====================================================================
# 03 - QLoRA training
# =====================================================================
def build_03():
    cells = [
        md(
            "# 03 - QLoRA fine-tuning (Colab T4)\n\n"
            "> Runtime -> Change runtime type -> T4 GPU. Then install the GPU stack:\n"
            "> `pip install unsloth trl peft transformers datasets accelerate bitsandbytes`\n\n"
            "QLoRA = a 4-bit (NF4) base + a small LoRA adapter trained on top. An 8B model\n"
            "fine-tunes in ~8-12GB, which fits a free T4. We train rank 16 / alpha 32 on all\n"
            "linear layers, the 2026 default."
        ),
        PATH_SETUP,
        md("### Configure and train\nEvery knob is commented in `src/qlora_lab/train.py`."),
        code(
            "cfg = train.TrainConfig(\n"
            "    base_model='unsloth/Qwen3-8B-bnb-4bit',\n"
            "    lora_r=16, lora_alpha=32, epochs=3, learning_rate=2e-4,\n"
            "    output_dir='outputs/adapter',\n"
            ")\n"
            "adapter_dir = train.train('../data/train.jsonl', cfg)\n"
            "print('adapter saved to', adapter_dir)  # ~tens of MB, not the whole model"
        ),
        md(
            "### Benchmark multiple bases\n"
            "The resume line says *benchmarking Qwen3, LLaMA3, Gemma, GPT-OSS*. That is\n"
            "literally running this cell with different `base_model` values on the same data\n"
            "and picking the Pareto-best on quality and cost - not defaulting to the biggest."
        ),
        code(
            "# for base in ['unsloth/Qwen3-8B-bnb-4bit', 'unsloth/llama-3.1-8b-bnb-4bit',\n"
            "#              'unsloth/gemma-2-9b-bnb-4bit']:\n"
            "#     train.train('../data/train.jsonl', train.TrainConfig(base_model=base,\n"
            "#                 output_dir=f'outputs/{base.split(\"/\")[-1]}'))"
        ),
    ]
    build(cells, NB / "03_train_qlora.ipynb")


# =====================================================================
# 04 - Serving: merge vs adapter
# =====================================================================
def build_04():
    cells = [
        md(
            "# 04 - Serve it: merge vs hot-swappable adapter\n\n"
            "Two options, and the trade-off is the ML-infra interview answer:\n\n"
            "- **merge**: fold LoRA into the base -> one standalone model. Simplest, zero\n"
            "  routing latency, but one full model per task.\n"
            "- **vLLM LoRA**: load the base once, hot-swap many adapters by request name.\n"
            "  One GPU + N small files serves N task variants."
        ),
        PATH_SETUP,
        md("### Option A: merge"),
        code(
            "# merged = serve.merge_adapter('outputs/adapter', 'outputs/merged',\n"
            "#                              base_model='unsloth/Qwen3-8B-bnb-4bit')\n"
            "# print('merged model at', merged)"
        ),
        md("### Option B: vLLM adapter serving (run in a terminal)"),
        code(
            "print(serve.VLLM_SERVE_HINT.format(\n"
            "    base='unsloth/Qwen3-8B-bnb-4bit', adapter='outputs/adapter'))\n"
            "# Then: client = serve.openai_client(); MODEL = 'ticket'  # selects the adapter"
        ),
        md(
            "Because both expose an OpenAI-compatible endpoint, the same `predict.extract`\n"
            "and the evaluation harness in notebook 05 work against either, unchanged."
        ),
    ]
    build(cells, NB / "04_serve.ipynb")


# =====================================================================
# 05 - Base vs fine-tuned, the artifact
# =====================================================================
def build_05():
    cells = [
        md(
            "# 05 - Base vs fine-tuned: prove it, do not claim it\n\n"
            "Run the **same** test set through the base model and the served adapter. The\n"
            "four numbers - schema validity, exact match, latency, cost - are what you put\n"
            "on the table to defend 'migrated to a self-hosted small model and cut cost'."
        ),
        PATH_SETUP,
        code(
            "test = dataset.read_jsonl('../data/test.jsonl')\n"
            "base_client = serve.openai_client(); BASE = 'unsloth/Qwen3-8B-bnb-4bit'\n"
            "tuned_client = serve.openai_client(); TUNED = 'ticket'  # adapter name in vLLM\n"
            "base_preds = [predict.extract(base_client, BASE, e['message']) for e in test]\n"
            "tuned_preds = [predict.extract(tuned_client, TUNED, e['message']) for e in test]"
        ),
        code(
            "# self-hosted $/token is roughly the same for base and tuned (same GPU), so the\n"
            "# cost win here is mostly fewer retries + shorter, valid outputs. The big cost\n"
            "# story vs a *frontier* baseline comes from notebook 01.\n"
            "rb = evaluate.evaluate(base_preds, test, in_price=0.05e-6, out_price=0.20e-6)\n"
            "rt = evaluate.evaluate(tuned_preds, test, in_price=0.05e-6, out_price=0.20e-6)\n"
            "print('BASE :', rb.summary())\nprint('TUNED:', rt.summary())\n"
            "print(); print(evaluate.compare(rb, rt))"
        ),
        md(
            "Save this table. If tuned does not clear base by 10%+ on schema validity, the\n"
            "fine-tune is the problem, not the model - revisit data and masking in 02."
        ),
    ]
    build(cells, NB / "05_eval_compare.ipynb")


# =====================================================================
# 06 - Integrate into an agent as a sub-model
# =====================================================================
def build_06():
    cells = [
        md(
            "# 06 - Use the tuned model as an agent sub-model\n\n"
            "Fine-tuning is not a trophy, it is a cheap component. Here the tuned extractor\n"
            "is one tool the agent calls on its high-frequency step, with a fallback to a\n"
            "strong model when the cheap one cannot produce valid JSON. That escalation\n"
            "rate, times the price gap, is your real, defensible saving."
        ),
        PATH_SETUP,
        code(
            "tuned_client = serve.openai_client(); TUNED = 'ticket'\n"
            "from openai import OpenAI\n"
            "strong_client = OpenAI(); STRONG = 'gpt-4o-mini'\n"
            "msg = 'Hey team, order 55012 arrived smashed and I am furious. Please advise.'\n"
            "print(agent.extract_ticket_tool(tuned_client, TUNED, msg))"
        ),
        md("### Cheap-first routing with fallback"),
        code(
            "test = dataset.read_jsonl('../data/test.jsonl')[:50]\n"
            "results = [agent.route_with_fallback(e['message'], tuned_client, TUNED,\n"
            "                                     strong_client, STRONG) for e in test]\n"
            "from collections import Counter\n"
            "print(Counter(r['served_by'] for r in results))\n"
            "# e.g. {'tuned_small': 47, 'strong_fallback': 3} -> 94% served cheap"
        ),
        md(
            "That Counter is the slide: most traffic served by the cheap self-hosted model,\n"
            "a small tail escalated. This is the VortexNet bullet made real - a high-volume\n"
            "step migrated off the frontier API, with numbers to defend it."
        ),
    ]
    build(cells, NB / "06_integrate_agent.ipynb")


# =====================================================================
# colab_t4_run - one self-contained notebook: clone, train, compare, download
# =====================================================================
def build_colab():
    cells = [
        md(
            "# qlora-lab on a free Colab T4: train + compare in one run\n\n"
            "**Before running**: Runtime -> Change runtime type -> **T4 GPU**.\n\n"
            "This notebook is self-contained: it clones the repo, generates data, measures\n"
            "the raw base model, runs the QLoRA fine-tune, re-measures, and prints the\n"
            "base-vs-tuned table. Total wall time on a T4 is roughly 30-50 minutes, most of\n"
            "it the ~3-epoch training run.\n\n"
            "It evaluates with Unsloth inference directly instead of a vLLM server, because\n"
            "running a vLLM server inside a T4 Colab is fragile. The numbers are the same\n"
            "kind: schema validity, exact match, latency, tokens. vLLM serving (notebook 04)\n"
            "is for when you deploy."
        ),
        code(
            "# 1. GPU stack. unsloth pulls trl/peft/transformers/datasets/bitsandbytes.\n"
            "%pip install -q unsloth"
        ),
        code(
            "# 2. Code + data\n"
            "!git clone https://github.com/zyziyun/qlora-lab.git\n"
            "%cd qlora-lab\n"
            "!python scripts/make_data.py --n 800\n"
            "import sys; sys.path.insert(0, 'src')"
        ),
        code(
            "# 3. Eval helper: run any (model, tokenizer) over the test set and build\n"
            "#    Prediction objects so the repo's evaluate harness works unchanged.\n"
            "import time, torch\n"
            "from qlora_lab import dataset as qds, evaluate as qev\n"
            "from qlora_lab.dataset import SYSTEM_PROMPT\n"
            "from qlora_lab.predict import Prediction\n"
            "\n"
            "test = qds.read_jsonl('data/test.jsonl')\n"
            "\n"
            "def run_eval(model, tokenizer, test, max_new_tokens=96):\n"
            "    preds = []\n"
            "    for e in test:\n"
            "        msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},\n"
            "                {'role': 'user', 'content': e['message']}]\n"
            "        # enable_thinking=False matters for Qwen3: thinking mode would emit a\n"
            "        # long <think> block before the JSON and wreck latency and token counts.\n"
            "        inputs = tokenizer.apply_chat_template(\n"
            "            msgs, add_generation_prompt=True, return_tensors='pt',\n"
            "            enable_thinking=False).to(model.device)\n"
            "        t0 = time.perf_counter()\n"
            "        out = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,\n"
            "                             do_sample=False, pad_token_id=tokenizer.eos_token_id)\n"
            "        dt = time.perf_counter() - t0\n"
            "        gen = out[0][inputs.shape[1]:]\n"
            "        preds.append(Prediction(\n"
            "            raw=tokenizer.decode(gen, skip_special_tokens=True),\n"
            "            latency_s=dt,\n"
            "            prompt_tokens=int(inputs.shape[1]),\n"
            "            completion_tokens=int(gen.shape[0])))\n"
            "    return preds"
        ),
        code(
            "# 4. Baseline: the raw 4-bit base, prompted. ~5 min for 100 examples.\n"
            "from unsloth import FastLanguageModel\n"
            "BASE = 'unsloth/Qwen3-8B-bnb-4bit'\n"
            "model, tokenizer = FastLanguageModel.from_pretrained(\n"
            "    BASE, max_seq_length=2048, load_in_4bit=True)\n"
            "FastLanguageModel.for_inference(model)\n"
            "base_preds = run_eval(model, tokenizer, test)\n"
            "rb = qev.evaluate(base_preds, test, in_price=0.05e-6, out_price=0.20e-6)\n"
            "print('BASE :', rb.summary())\n"
            "print('sample failures:', [f['reason'] for f in rb.failures[:5]])\n"
            "# free the GPU before training loads its own copy\n"
            "del model, tokenizer\n"
            "import gc; gc.collect(); torch.cuda.empty_cache()"
        ),
        code(
            "# 5. QLoRA fine-tune. ~20-40 min on a T4 for 573 examples x 3 epochs.\n"
            "from qlora_lab import train as qtrain\n"
            "cfg = qtrain.TrainConfig(base_model=BASE, output_dir='outputs/adapter')\n"
            "adapter_dir = qtrain.train('data/train.jsonl', cfg)\n"
            "print('adapter saved to', adapter_dir)\n"
            "import gc; gc.collect(); torch.cuda.empty_cache()"
        ),
        code(
            "# 6. Re-measure with the adapter and print the table that matters.\n"
            "model, tokenizer = FastLanguageModel.from_pretrained(\n"
            "    'outputs/adapter', max_seq_length=2048, load_in_4bit=True)\n"
            "FastLanguageModel.for_inference(model)\n"
            "tuned_preds = run_eval(model, tokenizer, test)\n"
            "rt = qev.evaluate(tuned_preds, test, in_price=0.05e-6, out_price=0.20e-6)\n"
            "print('TUNED:', rt.summary())\n"
            "print()\n"
            "print(qev.compare(rb, rt))"
        ),
        md(
            "If tuned does not clear base by 10%+ on schema validity, suspect the fine-tune,\n"
            "not the model: check the loss mask and data formatting in notebook 02.\n\n"
            "To benchmark more bases (the resume's *benchmarking Qwen3, LLaMA3, Gemma*),\n"
            "rerun cells 4-6 with `BASE` set to e.g. `unsloth/llama-3.1-8b-bnb-4bit` or\n"
            "`unsloth/gemma-2-9b-bnb-4bit` and keep the per-base tables."
        ),
        code(
            "# 7. Take the adapter home (~tens of MB) for vLLM serving (notebook 04).\n"
            "!zip -qr adapter.zip outputs/adapter\n"
            "from google.colab import files\n"
            "files.download('adapter.zip')"
        ),
    ]
    build(cells, NB / "colab_t4_run.ipynb")


# =====================================================================
# colab_serve_vllm - serve the adapter with vLLM on Colab, base-vs-adapter A/B
# =====================================================================
def build_colab_serve():
    cells = [
        md(
            "# Serve the adapter with vLLM on Colab: hot-swap A/B in one run\n\n"
            "**Before running**: Runtime -> Change runtime type -> **T4 or L4 GPU**.\n"
            "You also need `adapters.zip` (downloaded from `colab_t4_run.ipynb`) on your\n"
            "local machine.\n\n"
            "This demonstrates the serving half of the lab: one vLLM server loads the\n"
            "fp16 base **once** and exposes the LoRA adapter as a second model name\n"
            "(`ticket`). The client picks base or adapter per request by model name -\n"
            "that is adapter hot-swapping, the '1 GPU + N small files serves N variants'\n"
            "story. QLoRA trains on a 4-bit base but serves on the fp16 base; that is\n"
            "standard practice and the quality delta is negligible."
        ),
        code(
            "# 1. Check what CUDA your driver supports BEFORE installing vllm.\n"
            "#    Look at 'CUDA Version: X.Y' in the top-right of the output.\n"
            "!nvidia-smi | head -4"
        ),
        md(
            "### Install vLLM - match the wheel to your CUDA\n"
            "The #1 self-hosting pitfall is a wheel built for a different CUDA than the\n"
            "environment (symptom: `ImportError: libcudart.so.13: cannot open shared\n"
            "object file`). Pick ONE cell below based on the nvidia-smi output:"
        ),
        code(
            "# If CUDA Version showed 13.x: plain install, force torch to match.\n"
            "%pip install -q --force-reinstall vllm\n"
            "# Then: Runtime -> Restart session (files survive), rerun from cell 3."
        ),
        code(
            "# If CUDA Version showed 12.x: install the cu128-matched build instead.\n"
            "# %pip install -q uv\n"
            "# !uv pip install --system -q vllm --torch-backend=cu128\n"
            "# Fallback if uv cannot resolve: %pip install -q 'vllm==0.10.2'"
        ),
        code(
            "# 3. Code + data (same seed -> byte-identical test set as training time)\n"
            "!git clone https://github.com/zyziyun/qlora-lab.git\n"
            "%cd /content/qlora-lab\n"
            "!python scripts/make_data.py --n 800\n"
            "import sys; sys.path.insert(0, 'src')"
        ),
        md(
            "### 4. The adapter is already here\n"
            "The 1.7B adapter ships with the repo (`outputs/adapter-1.7b`, 67MB), so the\n"
            "clone in step 3 brought it along - nothing to upload. The 8B adapter exceeds\n"
            "GitHub's 100MB file limit and is not bundled; only if you want to A/B it too,\n"
            "drag your `adapters.zip` into the Files pane and run the optional cell below."
        ),
        code(
            "!ls outputs/adapter-1.7b/\n"
            "# Optional, 8B adapter from a local adapters.zip:\n"
            "# !unzip -q -o /content/adapters.zip -d /content/qlora-lab"
        ),
        code(
            "# 5. Start the vLLM server in the background.\n"
            "#    --dtype half is REQUIRED on T4 (no bf16 on Turing); harmless on L4.\n"
            "#    NOTE: Restart session kills this process - rerun this cell after any restart.\n"
            "import subprocess, time, requests\n"
            "\n"
            "server = subprocess.Popen([\n"
            "    'vllm', 'serve', 'Qwen/Qwen3-1.7B',\n"
            "    '--dtype', 'half',\n"
            "    '--enable-lora',\n"
            "    '--lora-modules', 'ticket=outputs/adapter-1.7b',\n"
            "    '--max-lora-rank', '16',\n"
            "    '--max-model-len', '4096',\n"
            "    '--gpu-memory-utilization', '0.85',\n"
            "    '--port', '8000',\n"
            "], stdout=open('vllm.log', 'w'), stderr=subprocess.STDOUT)\n"
            "\n"
            "print('first boot downloads ~3.4GB weights: expect 5-8 min on T4, less after caching')\n"
            "for i in range(150):\n"
            "    try:\n"
            "        if requests.get('http://localhost:8000/v1/models', timeout=2).ok:\n"
            "            print('server ready'); break\n"
            "    except Exception:\n"
            "        pass\n"
            "    time.sleep(5)\n"
            "else:\n"
            "    print('not ready - debug with: !tail -50 vllm.log')"
        ),
        code(
            "# 6. The hot-swap proof: one server, two model ids (base + adapter).\n"
            "!curl -s localhost:8000/v1/models | python3 -m json.tool | grep '\"id\"'"
        ),
        code(
            "# 7. A/B on the same server - the only difference is the model name string.\n"
            "from qlora_lab import serve, predict, dataset as ds, evaluate as ev\n"
            "\n"
            "client = serve.openai_client()\n"
            "NO_THINK = {'chat_template_kwargs': {'enable_thinking': False}}  # Qwen3: no <think> block\n"
            "\n"
            "test = ds.read_jsonl('data/test.jsonl')[:30]\n"
            "\n"
            "base_preds  = [predict.extract(client, 'Qwen/Qwen3-1.7B', e['message'], extra_body=NO_THINK) for e in test]\n"
            "tuned_preds = [predict.extract(client, 'ticket',          e['message'], extra_body=NO_THINK) for e in test]\n"
            "\n"
            "rb = ev.evaluate(base_preds,  test, in_price=0.05e-6, out_price=0.20e-6)\n"
            "rt = ev.evaluate(tuned_preds, test, in_price=0.05e-6, out_price=0.20e-6)\n"
            "print('BASE :', rb.summary())\n"
            "print('TUNED:', rt.summary())\n"
            "print(); print(ev.compare(rb, rt))\n"
            "print('base failures:', [f['reason'][:30] for f in rb.failures[:5]])"
        ),
        code(
            "# 8. One visible example - base vs adapter on the same message.\n"
            "msg = 'Hey team, order 55012 arrived smashed and I am furious. Please advise.'\n"
            "for m in ['Qwen/Qwen3-1.7B', 'ticket']:\n"
            "    p = predict.extract(client, m, msg, extra_body=NO_THINK)\n"
            "    print(f'{m:>18}: {p.raw[:110]}')"
        ),
        md(
            "Expected: tuned wins the judgment fields (priority/sentiment) with ~32 output\n"
            "tokens vs the base's longer output, and latency beats single-request HF\n"
            "generate thanks to vLLM's continuous batching + PagedAttention. If you serve\n"
            "the 1.7B in production, pair it with the deterministic guardrail from the\n"
            "README (extracted order_id must be a substring of the message) and escalate\n"
            "failures via `agent.route_with_fallback`."
        ),
    ]
    build(cells, NB / "colab_serve_vllm.ipynb")


# =====================================================================
# colab_experiments - the experiment menu, each section a finding
# =====================================================================
def build_experiments():
    cells = [
        md(
            "# Experiment menu: each section produces one finding\n\n"
            "> Runtime -> T4 or L4 GPU. Sections are independent — run the ones you want.\n"
            "> Training sections take 10-40 min each on a T4; inference sections are quick.\n\n"
            "Every experiment here maps to a claim you could put in a resume bullet or\n"
            "defend in an interview. Run it, read the number, keep the table."
        ),
        code("%pip install -q unsloth"),
        code(
            "!git clone https://github.com/zyziyun/qlora-lab.git\n"
            "%cd /content/qlora-lab\n"
            "!python scripts/make_data.py --n 800\n"
            "import sys; sys.path.insert(0, 'src')"
        ),
        code(
            "# Shared helpers: load an adapter (or base), eval it on a dataset, free VRAM.\n"
            "import time, torch, gc\n"
            "from unsloth import FastLanguageModel\n"
            "from qlora_lab import synth, train, dpo, agent, evaluate as qev, dataset as qds\n"
            "from qlora_lab.dataset import SYSTEM_PROMPT\n"
            "from qlora_lab.predict import Prediction\n"
            "from qlora_lab.schema import Ticket, parse_ticket\n"
            "\n"
            "test = qds.read_jsonl('data/test.jsonl')\n"
            "ood  = synth.gen_ood()           # held-out out-of-distribution set\n"
            "\n"
            "def run_eval(model, tokenizer, data, max_new_tokens=96):\n"
            "    preds = []\n"
            "    for e in data:\n"
            "        msgs = [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':e['message']}]\n"
            "        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,\n"
            "              return_tensors='pt', enable_thinking=False).to(model.device)\n"
            "        t0=time.perf_counter(); out=model.generate(input_ids=ids, max_new_tokens=max_new_tokens,\n"
            "              do_sample=False, pad_token_id=tokenizer.eos_token_id); dt=time.perf_counter()-t0\n"
            "        g=out[0][ids.shape[1]:]\n"
            "        preds.append(Prediction(raw=tokenizer.decode(g, skip_special_tokens=True),\n"
            "                     latency_s=dt, prompt_tokens=int(ids.shape[1]), completion_tokens=int(g.shape[0])))\n"
            "    return preds\n"
            "\n"
            "def free(*objs):\n"
            "    for o in objs:\n"
            "        del o\n"
            "    gc.collect(); torch.cuda.empty_cache()\n"
            "\n"
            "def eval_adapter(path, data, **kw):\n"
            "    m, t = FastLanguageModel.from_pretrained(path, max_seq_length=2048, load_in_4bit=True)\n"
            "    FastLanguageModel.for_inference(m)\n"
            "    rep = qev.evaluate(run_eval(m, t, data, **kw), data, in_price=0.05e-6, out_price=0.20e-6)\n"
            "    free(m, t); return rep"
        ),
        md(
            "## A. OOD gap, and whether data diversity closes it (experiment #3)\n"
            "The shipped 1.7B adapter aces the in-distribution test but slips on unseen\n"
            "styles. Retrain with `diverse=True` data and re-check the same OOD set."
        ),
        code(
            "# in-distribution vs OOD for the shipped adapter\n"
            "print('1.7B adapter, in-dist:', eval_adapter('outputs/adapter-1.7b', test).summary())\n"
            "print('1.7B adapter, OOD    :', eval_adapter('outputs/adapter-1.7b', ood).summary())"
        ),
        code(
            "# train a diverse-data adapter and re-check OOD (~10 min on T4)\n"
            "div_examples = synth.gen(800, seed=7, diverse=True)\n"
            "parts = qds.split(div_examples, n_test=100, n_val=100)\n"
            "tr, _ = qds.decontaminate(parts['train'], parts['test'])\n"
            "qds.write_jsonl([qds.to_chat(e) for e in tr], 'data/train_diverse.jsonl')\n"
            "train.train('data/train_diverse.jsonl', train.TrainConfig(\n"
            "    base_model='unsloth/Qwen3-1.7B-bnb-4bit', output_dir='outputs/adapter-1.7b-diverse'))\n"
            "print('diverse adapter, OOD :', eval_adapter('outputs/adapter-1.7b-diverse', ood).summary())\n"
            "# Closed the gap -> diversity beats a bigger model. Still short -> 1.7B capacity ceiling."
        ),
        md(
            "## B. Guardrail: catch hallucinated order ids, measure escalation (experiment #2)\n"
            "A deterministic check (order_id must be a substring of the message) catches the\n"
            "dangerous failure the OOD set exposes, and routes only those to a strong model."
        ),
        code(
            "m, t = FastLanguageModel.from_pretrained('outputs/adapter-1.7b', max_seq_length=2048, load_in_4bit=True)\n"
            "FastLanguageModel.for_inference(m)\n"
            "caught = 0\n"
            "for e in ood:\n"
            "    p = run_eval(m, t, [e])[0]\n"
            "    tk, _ = parse_ticket(p.raw)\n"
            "    if tk and not agent.order_id_in_message(tk, e['message']):\n"
            "        caught += 1\n"
            "        print('GUARD CAUGHT:', repr(e['message'][:60]), '-> claimed', tk.order_id)\n"
            "print(f'\\nwould escalate {caught}/{len(ood)} = {caught/len(ood):.0%} to the strong model')\n"
            "print('at a 20x price gap, routing cost vs all-strong =', round((len(ood)+caught*20)/(len(ood)*20), 2))\n"
            "free(m, t)"
        ),
        md(
            "## C. Data scaling curve (experiment #4)\n"
            "How many labeled examples does this narrow task actually need? Train on\n"
            "increasing slices, eval each on the same test set."
        ),
        code(
            "import json\n"
            "full = qds.read_jsonl('data/train.jsonl')\n"
            "rows = []\n"
            "for k in [50, 150, 300, len(full)]:\n"
            "    qds.write_jsonl(full[:k], f'data/train_{k}.jsonl')\n"
            "    train.train(f'data/train_{k}.jsonl', train.TrainConfig(\n"
            "        base_model='unsloth/Qwen3-1.7B-bnb-4bit', epochs=1, output_dir=f'outputs/scale-{k}'))\n"
            "    r = eval_adapter(f'outputs/scale-{k}', test)\n"
            "    rows.append((k, r.schema_validity, r.field_accuracy['priority'], r.exact_match))\n"
            "    print(f'n={k:>4}  validity={r.schema_validity:.3f}  priority={r.field_accuracy[\"priority\"]:.3f}  exact={r.exact_match:.3f}')\n"
            "print('\\nwhere the curve flattens is how much data you actually needed')"
        ),
        md(
            "## D. Rank ablation (experiment #6)\n"
            "Is r8 really within 1-2% of r64 on your task? Verify, do not assume."
        ),
        code(
            "for r in [8, 16, 64]:\n"
            "    train.train('data/train.jsonl', train.TrainConfig(\n"
            "        base_model='unsloth/Qwen3-1.7B-bnb-4bit', lora_r=r, lora_alpha=2*r, epochs=1,\n"
            "        output_dir=f'outputs/rank-{r}'))\n"
            "    rep = eval_adapter(f'outputs/rank-{r}', test)\n"
            "    print(f'r={r:>2}  validity={rep.schema_validity:.3f}  exact={rep.exact_match:.3f}  priority={rep.field_accuracy[\"priority\"]:.3f}')"
        ),
        md(
            "## E. Multi-base benchmark (experiment #7)\n"
            "The resume's *benchmarking Qwen3, LLaMA3, Gemma* — same data, swap the base,\n"
            "pick the Pareto point. (Each base downloads a few GB the first time.)"
        ),
        code(
            "for base in ['unsloth/Qwen3-1.7B-bnb-4bit', 'unsloth/Llama-3.2-3B-bnb-4bit']:\n"
            "    tag = base.split('/')[-1]\n"
            "    train.train('data/train.jsonl', train.TrainConfig(base_model=base, epochs=1, output_dir=f'outputs/{tag}'))\n"
            "    rep = eval_adapter(f'outputs/{tag}', test)\n"
            "    print(f'{tag:>26}  validity={rep.schema_validity:.3f}  exact={rep.exact_match:.3f}  out_tok={rep.mean_completion_tokens:.0f}')"
        ),
        md(
            "## F. Detailed summaries (experiment #5)\n"
            "The dual of 'the model learns your data's shape': feed summaries that carry the\n"
            "order id and key fact, and the tuned model emits detailed summaries too."
        ),
        code(
            "det = synth.gen(800, seed=7, detailed_summary=True)\n"
            "p = qds.split(det, n_test=100, n_val=100); tr,_ = qds.decontaminate(p['train'], p['test'])\n"
            "qds.write_jsonl([qds.to_chat(e) for e in tr], 'data/train_detailed.jsonl')\n"
            "train.train('data/train_detailed.jsonl', train.TrainConfig(\n"
            "    base_model='unsloth/Qwen3-1.7B-bnb-4bit', epochs=1, output_dir='outputs/adapter-detailed'))\n"
            "m,t = FastLanguageModel.from_pretrained('outputs/adapter-detailed', max_seq_length=2048, load_in_4bit=True)\n"
            "FastLanguageModel.for_inference(m)\n"
            "print(run_eval(m, t, [p['test'][0]])[0].raw)\n"
            "free(m, t)"
        ),
        md(
            "## G. DPO second stage (experiment #9)\n"
            "SFT makes it correct; DPO makes it prefer the compact prose-free style over a\n"
            "chatty fenced one. Watch output tokens drop with no loss of validity."
        ),
        code(
            "# preference pairs: chosen = compact JSON, rejected = chatty fenced JSON\n"
            "import json\n"
            "pairs = dpo.make_preference_pairs(qds.read_jsonl('data/test.jsonl')[:200] if False else synth.gen(400, seed=11))\n"
            "with open('data/prefs.jsonl','w') as f:\n"
            "    for r in pairs: f.write(json.dumps(r)+'\\n')\n"
            "dpo.train_dpo('data/prefs.jsonl', base_model='unsloth/Qwen3-1.7B-bnb-4bit',\n"
            "    cfg=dpo.DPOConfig(sft_adapter='outputs/adapter-1.7b', output_dir='outputs/adapter-dpo'))\n"
            "print('SFT  :', eval_adapter('outputs/adapter-1.7b', test).summary())\n"
            "print('DPO  :', eval_adapter('outputs/adapter-dpo', test).summary())\n"
            "# expect similar validity, fewer output tokens, zero preambles"
        ),
        md(
            "## H. vLLM concurrency (experiment #8)\n"
            "Lives in `colab_serve_vllm.ipynb` (needs a running server). The snippet: send N\n"
            "requests through a ThreadPoolExecutor and compare throughput to serial — the\n"
            "gap is vLLM's continuous batching + PagedAttention, the S4 goodput story made real.\n"
            "```python\n"
            "from concurrent.futures import ThreadPoolExecutor\n"
            "with ThreadPoolExecutor(max_workers=30) as ex:\n"
            "    out = list(ex.map(lambda e: predict.extract(client,'ticket',e['message'],extra_body=NO_THINK), test))\n"
            "```"
        ),
    ]
    build(cells, NB / "colab_experiments.ipynb")


if __name__ == "__main__":
    build_00()
    build_01()
    build_02()
    build_03()
    build_04()
    build_05()
    build_06()
    build_colab()
    build_colab_serve()
    build_experiments()
    print("done")

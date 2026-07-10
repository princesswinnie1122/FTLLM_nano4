# LLM Fine-Tuning Wizard for NANO4

Train ("fine-tune") a language model on NANO4's GPUs by answering a
few questions!

> 繁體中文版：[README.zh-TW.md](README.zh-TW.md)

## Setup

```bash
git clone https://github.com/princesswinnie1122/FTLLM_nano4.git
cd FTLLM_nano4 && bash setup.sh
```

## Try it now

```bash
/work/$USER/hpc-llm-finetune/venv/bin/python finetune_wizard.py
```

Answer the questions like this for a fast test run:

```
Language              -> English
Project/account ID    -> your project ID (ask your PI, e.g. mst114031)
Dataset file path      -> examples/sample_dataset.json
Base model              -> Qwen2.5 0.5B Instruct
Fine-tuning method      -> LoRA
Use estimated GPUs?     -> Yes
Expected run time       -> Quick test (<=4 hours)
Use default settings?   -> Yes
Job name                -> press Enter
Submit for real?         -> Yes
```

Then check on it:

```bash
squeue -u $USER
tail -f /work/$USER/hpc-llm-finetune/jobs/<job>/logs/*.out   # Ctrl+C to stop watching
```

A "loss" number will print during training — it should go down over time.

## Using your own data

Save a JSON file with your question/answer pairs:

```json
[
  {"instruction": "What is the capital of Taiwan?", "input": "", "output": "Taipei."}
]
```

Give the wizard the full path when it asks. More examples generally
help — 50–100 good ones is a reasonable starting point. (Multi-turn
conversations are also supported — "ShareGPT" style; the wizard will
tell you if your file's format isn't recognized.)

### From the MMLU Dataset web system

If your data lives in the [MMLU 資料集系統 web app](http://103.124.75.139/), you don't need to
hand-write the JSON. Open the **題目資料庫 (Question Database)** tab,
click the download menu on a dataset, and choose **匯出訓練格式
(Alpaca)**. It downloads `<dataset>_alpaca.json` already in the format
above. `scp` that file to nano4 and give its path to the wizard:

```bash
scp <dataset>_alpaca.json <user>@nano4.nchc.org.tw:/work/<user>/
```

## Choosing a method

- **LoRA** — default, fast, low memory. Use this unless you have a reason not to.
- **QLoRA** — for the largest models / least GPU memory.
- **Full fine-tune** — best quality, but only practical for small models.

You don't need to work out GPU counts yourself — the wizard estimates
them and lets you confirm or override.

## Where results go

```
/work/$USER/hpc-llm-finetune/jobs/<date-time>_<job-name>/output/
```

## Troubleshooting

- **"Project X is not allowed to use NANO4"** — your account isn't authorized for this cluster; ask your PI to check.
- **Dataset rejected** — make sure it's valid JSON with `instruction`/`output` fields (or `conversations` for ShareGPT style).
- **Rejected during the dry-run check** — the wizard prints the scheduler's real error (e.g. time limit too long). Nothing is submitted until this passes; fix your answers and rerun.
- **Job stuck "PENDING"** — the cluster is busy; it'll start once GPUs free up.

## Project files

- `finetune_wizard.py` — the wizard
- `setup.sh` — one-time installer
- `examples/sample_dataset.json` — example dataset used above
- `lib/` — cluster policy, model list, dataset validation, config generation (see comments in each file)

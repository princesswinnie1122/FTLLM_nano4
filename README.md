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

## Where it runs: SLURM or local

The wizard picks a backend automatically — **SLURM** when `sbatch` is
available, otherwise a **local** run — and both share the same
`train_config.yaml`, so the training itself is identical. Force one with
`--backend slurm` or `--backend local`.

- **SLURM** — writes an sbatch script and queues the job. Needs a project
  account (if the site requires one) and a partition/time limit.
- **Local** — runs `llamafactory-cli train` right where the wizard is
  running, no queue, no account. Use it on a GPU workstation without SLURM,
  on Colab, or inside an interactive compute-node session (`salloc`) where
  you already hold the GPUs. It writes a `run_local.sh` you can re-run,
  streams output live, and tees it to `logs/run_local.log`. GPU memory and
  count are read from `nvidia-smi` so the estimate fits the actual card.

## Running on another cluster

Cluster facts live in a YAML site profile under `profiles/`. The default is
`profiles/nano4.yaml`; point the wizard at another with `--site`:

```bash
python finetune_wizard.py --site profiles/generic_slurm.yaml
```

A profile sets `work_root`, `partitions`, `gpu_flag` (`gpus-per-node` or
`gres`), `account_required`, `gpu_mem_gb`, `max_cpus_per_gpu`,
`max_mem_per_gpu`, and `module_loads`. Supporting a new cluster is a new
profile, not a code change.

## Where results go

```
<work_root>/jobs/<date-time>_<job-name>/output/
```

`work_root` comes from the site profile (`/work/$USER/hpc-llm-finetune` on
NANO4).

## Troubleshooting

- **"Project X is not allowed to use NANO4"** — your account isn't authorized for this cluster; ask your PI to check.
- **Dataset rejected** — make sure it's valid JSON with `instruction`/`output` fields (or `conversations` for ShareGPT style).
- **Rejected during the dry-run check** — the wizard prints the scheduler's real error (e.g. time limit too long). Nothing is submitted until this passes; fix your answers and rerun.
- **Job stuck "PENDING"** — the cluster is busy; it'll start once GPUs free up.

## Project files

- `finetune_wizard.py` — the wizard
- `setup.sh` — one-time installer
- `profiles/` — site profiles (`nano4.yaml`, `generic_slurm.yaml`)
- `examples/sample_dataset.json` — example dataset used above
- `lib/` — site profile loader, GPU detection, model list, dataset validation, config generation (see comments in each file)

## License

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright (c) 2026 Winnie Sheng. Developed at the National Center for
High-Performance Computing (NCHC), Taiwan.

This license covers this tool only. The base models you fine-tune (Llama,
Qwen, …) each carry their own separate licenses — e.g. the Llama Community
License and the Qwen License — which govern your use of the weights and any
models you train or distribute. Review the license of whichever model you choose.

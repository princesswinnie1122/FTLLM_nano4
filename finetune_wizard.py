#!/usr/bin/env python3
"""
Interactive terminal wizard for submitting LLM fine-tuning jobs on the
NANO4 (nano4.nchc.org.tw) GPU cluster. No SLURM or command-line
experience required -- answer a few questions and the wizard writes
a validated sbatch script + LLaMA-Factory config and submits it for you.

Run it after `bash setup.sh` has installed the fine-tuning environment:

    /work/$USER/hpc-llm-finetune/venv/bin/python finetune_wizard.py
"""

import datetime
import getpass
import os
import subprocess
import sys

import questionary
from questionary import Choice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import cluster_policy, config_gen, dataset_utils
from lib.i18n import translator
from lib.models import METHODS, MODELS, estimate_gpus, label, short_label

USER = getpass.getuser()
WORK_ROOT = f"/work/{USER}/hpc-llm-finetune"
VENV_PYTHON = f"{WORK_ROOT}/venv/bin/python"
HF_HOME = f"{WORK_ROOT}/hf_cache"
JOBS_ROOT = f"{WORK_ROOT}/jobs"

STYLE = questionary.Style([
    ("qmark", "fg:#00b894 bold"),
    ("question", "bold"),
    ("answer", "fg:#00b894 bold"),
    ("pointer", "fg:#00b894 bold"),
    ("highlighted", "fg:#00b894 bold"),
])


def ask_language() -> str:
    choice = questionary.select(
        "Language / 語言選擇:",
        choices=[Choice(title="English", value="en"), Choice(title="繁體中文", value="zh")],
        style=STYLE,
    ).ask()
    if choice is None:
        sys.exit(1)
    return choice


def fail(t, msg: str):
    print(f"\n✗ {msg}\n")
    sys.exit(1)


def check_environment(t):
    if not os.path.exists(VENV_PYTHON):
        fail(t, t("env_missing", work_root=WORK_ROOT))


def ask_account(t) -> str:
    print(f"\n{t('account_intro')}\n")
    account = questionary.text(t("account_prompt"), style=STYLE).ask()
    if not account:
        fail(t, t("account_required"))
    return account.strip()


def ask_dataset(t) -> tuple:
    while True:
        path = questionary.path(t("dataset_prompt"), style=STYLE).ask()
        if not path:
            fail(t, t("dataset_required"))
        path = os.path.abspath(os.path.expanduser(path))
        try:
            fmt, n = dataset_utils.load_and_detect(path)
        except dataset_utils.DatasetError as e:
            print(f"  ✗ {e}")
            if not questionary.confirm(t("dataset_retry"), default=True, style=STYLE).ask():
                fail(t, t("dataset_none"))
            continue
        print(t("dataset_detected", fmt=fmt, n=n))
        return path, fmt


def ask_model(t, lang: str) -> dict:
    choice = questionary.select(
        t("model_prompt"),
        choices=[Choice(title=label(m, lang), value=m["key"]) for m in MODELS],
        style=STYLE,
    ).ask()
    return next(m for m in MODELS if m["key"] == choice)


def ask_method(t, lang: str) -> dict:
    choice = questionary.select(
        t("method_prompt"),
        choices=[Choice(title=label(m, lang), value=m["key"]) for m in METHODS],
        style=STYLE,
    ).ask()
    return next(m for m in METHODS if m["key"] == choice)


def ask_gpus(t, lang: str, model: dict, method: dict) -> int:
    estimate = estimate_gpus(model["params_billion"], method["gb_per_billion"], cluster_policy.GPU_MEM_GB)
    print(t(
        "gpu_estimate", n=estimate, gpu_model=cluster_policy.GPU_MODEL,
        model=short_label(model, lang), method=short_label(method, lang),
    ))
    use_estimate = questionary.confirm(
        t("gpu_use_estimate", n=estimate), default=True, style=STYLE
    ).ask()
    if use_estimate:
        return estimate
    gpus = questionary.text(
        t("gpu_custom_prompt"), default=str(estimate), style=STYLE,
        validate=lambda v: v.isdigit() and 1 <= int(v) <= cluster_policy.MAX_GPUS_PER_NODE,
    ).ask()
    return int(gpus)


def ask_partition_and_time(t) -> tuple:
    choice = questionary.select(
        t("time_choice_prompt"),
        choices=[
            Choice(title=t("time_choice_dev"), value="dev"),
            Choice(title=t("time_choice_8gpus"), value="8gpus"),
        ],
        style=STYLE,
    ).ask()
    max_time = cluster_policy.PARTITIONS[choice]["max_time"]
    time_limit = questionary.text(
        t("time_limit_prompt", max_time=max_time), default=max_time, style=STYLE,
    ).ask()
    return choice, time_limit


def ask_training_params(t, method: dict) -> dict:
    advanced = questionary.confirm(t("defaults_confirm"), default=True, style=STYLE).ask()
    default_lr = "1.0e-5" if method["finetuning_type"] == "full" else "1.0e-4"
    if advanced:
        return {"epochs": 3.0, "learning_rate": default_lr, "batch_size": 2, "cutoff_len": 1024}
    epochs = questionary.text(t("epochs_prompt"), default="3.0", style=STYLE).ask()
    lr = questionary.text(t("lr_prompt"), default=default_lr, style=STYLE).ask()
    batch = questionary.text(t("batch_prompt"), default="2", style=STYLE).ask()
    cutoff = questionary.text(t("cutoff_prompt"), default="1024", style=STYLE).ask()
    return {
        "epochs": float(epochs), "learning_rate": lr,
        "batch_size": int(batch), "cutoff_len": int(cutoff),
    }


def main():
    lang = ask_language()
    t = translator(lang)

    print("=" * 70)
    print(f"  {t('banner')}")
    print("=" * 70)
    check_environment(t)

    account = ask_account(t)
    dataset_path, dataset_fmt = ask_dataset(t)
    model = ask_model(t, lang)
    method = ask_method(t, lang)
    gpus = ask_gpus(t, lang, model, method)
    partition, time_limit = ask_partition_and_time(t)
    params = ask_training_params(t, method)

    default_job_name = f"{model['key']}-{method['key']}"
    job_name = questionary.text(t("job_name_prompt"), default=default_job_name, style=STYLE).ask()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = os.path.join(JOBS_ROOT, f"{ts}_{job_name}")
    os.makedirs(job_dir, exist_ok=True)

    cpus_per_task = min(cluster_policy.MAX_CPUS_PER_GPU, 8) * gpus
    mem_gb = min(cluster_policy.MAX_MEM_GB_PER_GPU - 20, 180) * gpus

    data_dir = config_gen.write_dataset_registration(
        job_dir, dataset_path, dataset_fmt, os.path.splitext(dataset_path)[1]
    )
    train_config_path, output_dir = config_gen.write_train_config(
        job_dir, data_dir, model, method, gpus,
        params["epochs"], params["learning_rate"], params["batch_size"], params["cutoff_len"],
        HF_HOME,
    )
    sbatch_path = config_gen.write_sbatch_script(
        job_dir, job_name, account, partition, gpus, cpus_per_task, mem_gb,
        time_limit, VENV_PYTHON, train_config_path, HF_HOME,
    )

    print(t("generated_files", job_dir=job_dir))
    print(f"  - {train_config_path}")
    print(f"  - {sbatch_path}")

    print(t("validating"))
    result = subprocess.run(
        ["sbatch", "--test-only", sbatch_path], capture_output=True, text=True
    )
    output = (result.stdout + result.stderr).strip()
    print(f"  {output}")
    if result.returncode != 0:
        fail(t, t("dry_run_fail"))

    print(t("output_location", output_dir=output_dir))
    submit = questionary.confirm(t("submit_confirm"), default=True, style=STYLE).ask()
    if not submit:
        print(t("not_submitted", sbatch_path=sbatch_path))
        return

    result = subprocess.run(["sbatch", sbatch_path], capture_output=True, text=True)
    if result.returncode != 0:
        fail(t, t("submission_failed", msg=result.stdout + result.stderr))
    print(t("submitted", msg=result.stdout.strip()))
    job_id = result.stdout.strip().split()[-1]
    print(t("check_status", user=USER))
    print(t("watch_logs", log_path=f"{job_dir}/logs/{job_name}-{job_id}.out"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled. / 已取消。")
        sys.exit(1)

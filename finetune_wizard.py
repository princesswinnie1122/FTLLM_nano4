#!/usr/bin/env python3
"""
Interactive terminal wizard for submitting LLM fine-tuning jobs. No SLURM
or command-line experience required -- answer a few questions and the
wizard writes a validated LLaMA-Factory config plus either an sbatch
script (SLURM clusters) or a local runner, and launches it for you.

Cluster facts come from a YAML site profile (profiles/*.yaml); pass a
different one with --site. The backend is chosen automatically: SLURM when
sbatch is available, otherwise a direct local run. Force it with --backend.

Run it after `bash setup.sh` has installed the fine-tuning environment:

    python finetune_wizard.py
    python finetune_wizard.py --site profiles/generic_slurm.yaml
    python finetune_wizard.py --backend local
"""

import argparse
import datetime
import getpass
import os
import shutil
import subprocess
import sys

import questionary
from questionary import Choice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import config_gen, dataset_utils, gpu_detect, site_profile
from lib.i18n import translator
from lib.models import METHODS, MODELS, estimate_gpus, label, short_label

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SITE = os.path.join(HERE, "profiles", "nano4.yaml")
USER = getpass.getuser()

STYLE = questionary.Style([
    ("qmark", "fg:#00b894 bold"),
    ("question", "bold"),
    ("answer", "fg:#00b894 bold"),
    ("pointer", "fg:#00b894 bold"),
    ("highlighted", "fg:#00b894 bold"),
])


def parse_args():
    p = argparse.ArgumentParser(description="LLM fine-tuning wizard")
    p.add_argument("--site", default=DEFAULT_SITE, help="path to a site profile YAML")
    p.add_argument("--backend", choices=["slurm", "local"], default=None,
                   help="override backend (default: slurm if sbatch is present, else local)")
    return p.parse_args()


def resolve_cli(work_root: str) -> str:
    venv_cli = os.path.join(work_root, "venv", "bin", "llamafactory-cli")
    if os.path.exists(venv_cli):
        return venv_cli
    return shutil.which("llamafactory-cli") or "llamafactory-cli"


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


def check_environment(t, work_root: str, cli: str):
    if not (os.path.isabs(cli) and os.path.exists(cli)) and shutil.which(cli) is None:
        fail(t, t("env_missing", work_root=work_root))


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


def ask_gpus(t, lang: str, model: dict, method: dict, profile, gpu_mem_gb: int) -> int:
    estimate = estimate_gpus(
        model["params_billion"], method["gb_per_billion"],
        gpu_mem_gb, profile.max_gpus_per_node,
    )
    print(t(
        "gpu_estimate", n=estimate, gpu_model=profile.gpu_model,
        model=short_label(model, lang), method=short_label(method, lang),
    ))
    if questionary.confirm(t("gpu_use_estimate", n=estimate), default=True, style=STYLE).ask():
        return estimate
    gpus = questionary.text(
        t("gpu_custom_prompt"), default=str(estimate), style=STYLE,
        validate=lambda v: v.isdigit() and 1 <= int(v) <= profile.max_gpus_per_node,
    ).ask()
    return int(gpus)


def ask_partition_and_time(t, profile) -> tuple:
    names = list(profile.partitions.keys())
    if len(names) == 1:
        partition = names[0]
    else:
        partition = questionary.select(
            t("time_choice_prompt"),
            choices=[
                Choice(title=p.get("description", name), value=name)
                for name, p in profile.partitions.items()
            ],
            default=profile.default_partition,
            style=STYLE,
        ).ask()
    max_time = profile.partitions[partition]["max_time"]
    time_limit = questionary.text(
        t("time_limit_prompt", max_time=max_time), default=max_time, style=STYLE,
    ).ask()
    return partition, time_limit


def ask_training_params(t, method: dict) -> dict:
    use_defaults = questionary.confirm(t("defaults_confirm"), default=True, style=STYLE).ask()
    default_lr = "1.0e-5" if method["finetuning_type"] == "full" else "1.0e-4"
    if use_defaults:
        return {"epochs": 3.0, "learning_rate": default_lr, "batch_size": 2, "cutoff_len": 1024}
    epochs = questionary.text(t("epochs_prompt"), default="3.0", style=STYLE).ask()
    lr = questionary.text(t("lr_prompt"), default=default_lr, style=STYLE).ask()
    batch = questionary.text(t("batch_prompt"), default="2", style=STYLE).ask()
    cutoff = questionary.text(t("cutoff_prompt"), default="1024", style=STYLE).ask()
    return {
        "epochs": float(epochs), "learning_rate": lr,
        "batch_size": int(batch), "cutoff_len": int(cutoff),
    }


def run_local(t, job_dir, gpus, cli, train_config_path, hf_home, output_dir):
    script_path, log_path = config_gen.write_local_script(
        job_dir, gpus, cli, train_config_path, hf_home
    )
    print(t("local_generated", job_dir=job_dir, config=train_config_path, script=script_path))
    print(t("output_location", output_dir=output_dir))

    if not questionary.confirm(t("local_run_confirm"), default=True, style=STYLE).ask():
        print(t("local_not_run", script=script_path))
        return

    print(t("local_starting"))
    result = subprocess.run(["bash", script_path])
    if result.returncode != 0:
        fail(t, t("local_failed", code=result.returncode, log_path=log_path))
    print(t("local_done", output_dir=output_dir, log_path=log_path))


def main():
    args = parse_args()
    lang = ask_language()
    t = translator(lang)

    profile = site_profile.load_profile(args.site)
    backend = args.backend or ("slurm" if shutil.which("sbatch") else "local")

    hf_home = os.path.join(profile.work_root, "hf_cache")
    jobs_root = os.path.join(profile.work_root, "jobs")
    cli = resolve_cli(profile.work_root)

    print("=" * 70)
    print(f"  {t('banner')}")
    print(f"  {t('mode_line', site=os.path.basename(args.site), backend=backend)}")
    print("=" * 70)
    check_environment(t, profile.work_root, cli)
    os.makedirs(jobs_root, exist_ok=True)

    if backend == "local":
        print(t("local_intro"))

    account = ask_account(t) if (backend == "slurm" and profile.account_required) else None
    dataset_path, dataset_fmt = ask_dataset(t)
    model = ask_model(t, lang)
    method = ask_method(t, lang)

    gpu_mem_gb = gpu_detect.detect_gpu_mem_gb(profile.gpu_mem_gb)
    gpus = ask_gpus(t, lang, model, method, profile, gpu_mem_gb)
    if backend == "local":
        detected = gpu_detect.detect_gpu_count()
        if detected and gpus > detected:
            gpus = detected
            print(t("local_gpu_cap", n=gpus))

    partition = time_limit = None
    if backend == "slurm":
        partition, time_limit = ask_partition_and_time(t, profile)
    params = ask_training_params(t, method)

    default_job_name = f"{model['key']}-{method['key']}"
    job_name = questionary.text(t("job_name_prompt"), default=default_job_name, style=STYLE).ask()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = os.path.join(jobs_root, f"{ts}_{job_name}")
    os.makedirs(job_dir, exist_ok=True)

    data_dir = config_gen.write_dataset_registration(
        job_dir, dataset_path, dataset_fmt, os.path.splitext(dataset_path)[1]
    )
    train_config_path, output_dir = config_gen.write_train_config(
        job_dir, data_dir, model, method, gpus,
        params["epochs"], params["learning_rate"], params["batch_size"], params["cutoff_len"],
        hf_home,
    )

    if backend == "local":
        run_local(t, job_dir, gpus, cli, train_config_path, hf_home, output_dir)
        return

    cpus_per_task = min(profile.max_cpus_per_gpu, 8) * gpus
    mem_gb = min(profile.max_mem_per_gpu - 20, 180) * gpus
    sbatch_path = config_gen.write_sbatch_script(
        job_dir, job_name, account, partition, gpus, cpus_per_task, mem_gb,
        time_limit, cli, train_config_path, hf_home,
        profile.gpu_flag, profile.module_loads,
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
    if not questionary.confirm(t("submit_confirm"), default=True, style=STYLE).ask():
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

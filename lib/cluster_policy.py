"""
Cluster facts for NANO4 (nano4.nchc.org.tw), verified empirically via
`sbatch --test-only`, `scontrol show partition`, and a real smoke-test job
on 2026-07-01. These are not published anywhere -- they were reverse
engineered from the job_submit/lua plugin's error messages. Re-verify
if submissions start failing in new ways.

Key facts:
- --account is mandatory; there is no usable default account.
- GPU partitions REQUIRE --gpus-per-node (1-8); there is no CPU-only
  partition available to general project accounts (the ngs* CPU
  partitions are restricted to a specific course account).
- CPU cap: 12 cores per requested GPU (enforced by job_submit.lua).
- Mem cap: ~200GB per requested GPU (auto-capped with a warning by
  job_submit.lua if exceeded).
- Partition names are an advisory sizing convention, not a hard cap
  enforced by Slurm -- what IS strictly enforced is each partition's
  wall-time limit. The 256gpus partition additionally requires special
  account authorization beyond the others.
"""

MAX_CPUS_PER_GPU = 12
MAX_MEM_GB_PER_GPU = 200
MAX_GPUS_PER_NODE = 8
GPU_MODEL = "H200"
GPU_MEM_GB = 140  # usable headroom per H200 (143771 MiB nominal)

# Partitions this wizard offers for a single-node fine-tuning job.
# (16gpus/32gpus/64gpus/256gpus exist but are for multi-node jobs or
# need extra authorization -- out of scope for this single-node wizard.)
PARTITIONS = {
    "dev": {
        "max_time": "04:00:00",
        "description": "Quick test / debug run (max 4 hours)",
    },
    "8gpus": {
        "max_time": "2-00:00:00",
        "description": "Standard training run (max 2 days)",
    },
}

DEFAULT_PARTITION = "8gpus"

HF_CACHE_ENV = "HF_HOME"

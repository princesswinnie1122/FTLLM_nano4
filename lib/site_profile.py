"""Load a cluster's facts from a YAML site profile (profiles/*.yaml), so a
new cluster is supported by adding a profile with no code change. See
profiles/nano4.yaml for the full set of fields.
"""

import getpass
import os
from dataclasses import dataclass

import yaml

GPU_FLAGS = ("gpus-per-node", "gres")


@dataclass
class Profile:
    name: str
    work_root: str
    account_required: bool
    gpu_flag: str
    gpu_model: str
    gpu_mem_gb: int
    max_cpus_per_gpu: int
    max_mem_per_gpu: int
    max_gpus_per_node: int
    module_loads: list
    partitions: dict
    default_partition: str


def load_profile(path: str) -> Profile:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    gpu_flag = data.get("gpu_flag", "gpus-per-node")
    if gpu_flag not in GPU_FLAGS:
        raise ValueError(f"gpu_flag must be one of {GPU_FLAGS}, got {gpu_flag!r}")

    partitions = data["partitions"]
    work_root = os.path.expanduser(
        str(data["work_root"]).replace("{user}", getpass.getuser())
    )

    return Profile(
        name=data.get("name", "cluster"),
        work_root=work_root,
        account_required=bool(data.get("account_required", False)),
        gpu_flag=gpu_flag,
        gpu_model=data.get("gpu_model", "GPU"),
        gpu_mem_gb=int(data["gpu_mem_gb"]),
        max_cpus_per_gpu=int(data.get("max_cpus_per_gpu", 8)),
        max_mem_per_gpu=int(data.get("max_mem_per_gpu", 64)),
        max_gpus_per_node=int(data.get("max_gpus_per_node", 8)),
        module_loads=list(data.get("module_loads") or []),
        partitions=partitions,
        default_partition=data.get("default_partition") or next(iter(partitions)),
    )

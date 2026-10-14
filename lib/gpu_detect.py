"""Detect GPU memory and count from nvidia-smi, falling back to the
site profile's declared values when nvidia-smi is unavailable."""

import shutil
import subprocess

MEM_HEADROOM = 0.9


def _query_memory_mib():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        return [int(float(line)) for line in out.stdout.splitlines() if line.strip()]
    except ValueError:
        return None


def detect_gpu_count() -> int:
    """Number of visible GPUs, or 0 if none / nvidia-smi is unavailable."""
    mibs = _query_memory_mib()
    return len(mibs) if mibs else 0


def detect_gpu_mem_gb(fallback: int) -> int:
    """Usable per-GPU memory in GB, measured across the smallest GPU with a
    headroom margin; falls back to the profile value when undetectable."""
    mibs = _query_memory_mib()
    if not mibs:
        return fallback
    usable = int(min(mibs) / 1024 * MEM_HEADROOM)
    return usable or fallback

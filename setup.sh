#!/bin/bash
# One-time setup for the NANO4 LLM fine-tuning wizard.
# Installs LLaMA-Factory + PyTorch into a venv under /work/$USER (NOT
# /home, which has a much smaller quota). Safe to rerun.
set -euo pipefail

WORK_ROOT="/work/${USER}/hpc-llm-finetune"
VENV="${WORK_ROOT}/venv"

echo "Installing fine-tuning environment to: ${WORK_ROOT}"
mkdir -p "${WORK_ROOT}" "${WORK_ROOT}/hf_cache" "${WORK_ROOT}/jobs"

if [ ! -d "${VENV}" ]; then
    python3.9 -m venv "${VENV}"
fi

"${VENV}/bin/pip" install -q --upgrade pip setuptools wheel
"${VENV}/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124
"${VENV}/bin/pip" install -q "llamafactory[metrics]" questionary huggingface_hub[cli]

echo ""
echo "Done. Verify with:"
echo "  ${VENV}/bin/llamafactory-cli version"
echo ""
echo "Then run the wizard with:"
echo "  ${VENV}/bin/python $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/finetune_wizard.py"

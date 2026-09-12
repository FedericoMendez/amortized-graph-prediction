#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --gpus=0
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
PYTHON_EXEC="${SLURM_SUBMIT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_EXEC}" ]]; then
  echo "Missing project Python environment: ${PYTHON_EXEC}" >&2
  exit 1
fi
echo "[environment] python=${PYTHON_EXEC}"
"${PYTHON_EXEC}" --version

srun "${PYTHON_EXEC}" -m any2graph_v2.preprocess_fingerprints \
  --atom-limit 32 --workers 32

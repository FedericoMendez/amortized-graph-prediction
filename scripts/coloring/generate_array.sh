#!/bin/bash
#SBATCH --job-name=coloring-generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=CPU
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

CAPS=(8 16 32 64)
CAP="${CAPS[${SLURM_ARRAY_TASK_ID:?Missing array task ID}]}"
IMAGE_RESOLUTION=$((4 * CAP))
COLORING_SEED="${COLORING_SEED:-0}"
OUTPUT_DIR="src/data/coloring/image_max${CAP}"

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
PYTHON_EXEC="${SLURM_SUBMIT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_EXEC}" ]]; then
  echo "Missing uv environment: ${PYTHON_EXEC}" >&2
  echo "Run 'uv sync --frozen' from the repository root first." >&2
  exit 1
fi
echo "[environment] python=${PYTHON_EXEC}"
"${PYTHON_EXEC}" --version

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ $# -eq 0 && -f "${OUTPUT_DIR}/manifest.json" ]]; then
  if "${PYTHON_EXEC}" -c '
import json
import sys

manifest = json.load(open(sys.argv[1]))
expected = {
    "format_version": 2,
    "input_modality": "rgb_image",
    "seed": int(sys.argv[2]),
    "min_nodes": 5,
    "max_nodes": int(sys.argv[3]),
    "image_resolution": int(sys.argv[4]),
}
valid = all(manifest.get(name) == value for name, value in expected.items())
valid = valid and all(
    manifest["splits"][split]["samples"] == count
    for split, count in (("train", 150000), ("val", 10000), ("test", 10000))
)
raise SystemExit(0 if valid else 1)
' "${OUTPUT_DIR}/manifest.json" "${COLORING_SEED}" "${CAP}" "${IMAGE_RESOLUTION}"; then
    echo "[coloring] reusing compatible dataset at ${OUTPUT_DIR}"
    exit 0
  fi
  echo "Existing dataset is incompatible with the default generation grid: ${OUTPUT_DIR}" >&2
  echo "Move it aside explicitly before regenerating." >&2
  exit 1
fi

echo "[coloring] generating max_nodes=${CAP} resolution=${IMAGE_RESOLUTION} seed=${COLORING_SEED} at ${OUTPUT_DIR}"
srun "${PYTHON_EXEC}" -m any2graph_v2.generate_coloring \
  --output-dir "${OUTPUT_DIR}" \
  --min-nodes 5 \
  --max-nodes "${CAP}" \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --train-samples 150000 \
  --val-samples 10000 \
  --test-samples 10000 \
  --seed "${COLORING_SEED}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  "$@"

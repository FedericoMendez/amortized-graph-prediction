#!/bin/bash
#SBATCH --job-name=coloring-size-generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=CPU
#SBATCH --array=0-5
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

CAPS=(10 20 30 40 50 60)
TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing array task ID}"
if (( TASK_ID < 0 || TASK_ID >= ${#CAPS[@]} )); then
  echo "Invalid array task ID: ${TASK_ID}" >&2
  exit 1
fi
CAP="${CAPS[${TASK_ID}]}"
IMAGE_RESOLUTION=$((4 * CAP))
COLORING_SEED="${COLORING_SEED:-0}"
OUTPUT_DIR="src/data/coloring/image_max${CAP}"
SAMPLES_PER_MAX_NODE=20000
TOTAL_SAMPLES=$((SAMPLES_PER_MAX_NODE * CAP))
# Preserve the existing 15:1:1 train/validation/test proportions. Assign the
# integer-division remainder to training so the three splits sum exactly to the
# requested capacity-dependent total.
VAL_SAMPLES=$((TOTAL_SAMPLES / 17))
TEST_SAMPLES="${VAL_SAMPLES}"
TRAIN_SAMPLES=$((TOTAL_SAMPLES - VAL_SAMPLES - TEST_SAMPLES))

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

if [[ -f "${OUTPUT_DIR}/manifest.json" ]]; then
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
expected_splits = {
    "train": int(sys.argv[5]),
    "val": int(sys.argv[6]),
    "test": int(sys.argv[7]),
}
valid = all(manifest.get(name) == value for name, value in expected.items())
valid = valid and all(
    manifest["splits"][split]["samples"] == count
    for split, count in expected_splits.items()
)
raise SystemExit(0 if valid else 1)
' "${OUTPUT_DIR}/manifest.json" "${COLORING_SEED}" "${CAP}" \
      "${IMAGE_RESOLUTION}" "${TRAIN_SAMPLES}" "${VAL_SAMPLES}" "${TEST_SAMPLES}"; then
    echo "[coloring] reusing compatible dataset at ${OUTPUT_DIR}"
    exit 0
  fi
  echo "Existing dataset is incompatible with the size-epsilon sweep: ${OUTPUT_DIR}" >&2
  echo "Move it aside explicitly before regenerating." >&2
  exit 1
fi

echo "[coloring] generating max_nodes=${CAP} resolution=${IMAGE_RESOLUTION} total_samples=${TOTAL_SAMPLES} train=${TRAIN_SAMPLES} val=${VAL_SAMPLES} test=${TEST_SAMPLES} seed=${COLORING_SEED} at ${OUTPUT_DIR}"
srun "${PYTHON_EXEC}" -m any2graph_v2.generate_coloring \
  --output-dir "${OUTPUT_DIR}" \
  --min-nodes 5 \
  --max-nodes "${CAP}" \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --val-samples "${VAL_SAMPLES}" \
  --test-samples "${TEST_SAMPLES}" \
  --seed "${COLORING_SEED}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  "$@"

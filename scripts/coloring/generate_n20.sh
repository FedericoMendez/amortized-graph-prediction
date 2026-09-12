#!/bin/bash
#SBATCH --job-name=coloring-n20-generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=CPU
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

CAP=20
IMAGE_RESOLUTION=$((4 * CAP))
COLORING_SEED="${COLORING_SEED:-0}"
OUTPUT_DIR="src/data/coloring/image_max${CAP}"
SAMPLES_PER_MAX_NODE=20000
TOTAL_SAMPLES=$((SAMPLES_PER_MAX_NODE * CAP))
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
    for split, count in (
        ("train", int(sys.argv[5])),
        ("val", int(sys.argv[6])),
        ("test", int(sys.argv[7])),
    )
)
raise SystemExit(0 if valid else 1)
' "${OUTPUT_DIR}/manifest.json" "${COLORING_SEED}" "${CAP}" \
      "${IMAGE_RESOLUTION}" "${TRAIN_SAMPLES}" "${VAL_SAMPLES}" "${TEST_SAMPLES}"; then
    echo "[coloring] reusing compatible N20 dataset at ${OUTPUT_DIR}"
    exit 0
  fi
  echo "Existing N20 dataset is incompatible with this experiment: ${OUTPUT_DIR}" >&2
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

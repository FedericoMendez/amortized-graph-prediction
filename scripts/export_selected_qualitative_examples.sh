#!/bin/bash
#SBATCH --job-name=qualitative-best
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=V100
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
PYTHON_EXEC="${SLURM_SUBMIT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_EXEC}" ]]; then
  echo "Missing uv environment: ${PYTHON_EXEC}" >&2
  echo "Run 'uv sync --frozen' from the repository root first." >&2
  exit 1
fi
echo "[environment] python=${PYTHON_EXEC}"
"${PYTHON_EXEC}" --version

mkdir -p logs paper_figures/selected_best_models
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
WANDB_PROJECT="${WANDB_PROJECT:-any2graph-v2}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-selected-best-model-qualitative}"
WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ENTITY_ARGS=(--wandb-entity "${WANDB_ENTITY}")
fi

RUN_NAMES=(
  "ms2graph-sinkhorn-iterations20-epsilon3e-5-seed0-a943676"
  "fp2graph-pubchem32-n16-learned-inner20-seed0-952048"
  "coloring-n20-segformer-learned-eps7.5e-6-latent256-b512-cosine-120ep-seed0-958022"
)

for RUN_NAME in "${RUN_NAMES[@]}"; do
  CHECKPOINT_DIR="artifacts/${RUN_NAME}/checkpoints"
  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
    exit 1
  fi
  shopt -s nullglob
  BEST_CHECKPOINTS=("${CHECKPOINT_DIR}"/best-*.ckpt)
  shopt -u nullglob
  if (( ${#BEST_CHECKPOINTS[@]} != 1 )); then
    echo "Expected exactly one best checkpoint in ${CHECKPOINT_DIR}; found ${#BEST_CHECKPOINTS[@]}." >&2
    exit 1
  fi
  BEST_CHECKPOINT="${BEST_CHECKPOINTS[0]}"
  OUTPUT_DIR="paper_figures/selected_best_models/${RUN_NAME}"
  echo "[qualitative] run=${RUN_NAME} checkpoint=${BEST_CHECKPOINT}"
  srun "${PYTHON_EXEC}" -m any2graph_v2.qualitative_examples_cli \
    --checkpoint "${BEST_CHECKPOINT}" \
    --selection distinct_targets \
    --num-examples 4 \
    --device cuda \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "${RUN_NAME}-distinct-targets" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}-qualitative-distinct" \
    --wandb-group "${WANDB_GROUP}" \
    "${WANDB_ENTITY_ARGS[@]}" \
    "$@"
done

#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

set -euo pipefail

NANO_REPO_ROOT=/mnt/weka/home/duc.pham/nano-vllm-uno
NANO_PYTHON=/mnt/weka/home/duc.pham/.venvs/xllm/bin/python
NANO_MODEL=/mnt/weka/shrd/k2m/duc.pham/qwen3
NANO_GATED_LORA_PATH=/mnt/weka/shrd/k2m/duc.pham/uno-model
NANO_DATA_ROOT=/mnt/weka/home/duc.pham/data/source
NANO_RESULTS_ROOT=/mnt/weka/shrd/k2m/duc.pham/benchmark-results/pull-suite
NANO_RUN_NAME=uno-qwen3-8b-bs16-maxseq64
NANO_GPUS_PER_NODE=8
NANO_WORKERS=$((SLURM_NNODES * NANO_GPUS_PER_NODE))
NANO_PULL_COORDINATOR_HOST="$(
  scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p'
)"
NANO_PULL_COORDINATOR_PORT="${NANO_PULL_COORDINATOR_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
NANO_PULL_AUTH_KEY="${NANO_PULL_AUTH_KEY:-${SLURM_JOB_ID}}"

export NANO_PULL_COORDINATOR_HOST
export NANO_PULL_COORDINATOR_PORT
export NANO_PULL_AUTH_KEY
export NANO_MODEL NANO_GATED_LORA_PATH NANO_DATA_ROOT NANO_RESULTS_ROOT
export NANO_RUN_NAME

command=(
  "${NANO_PYTHON}"
  -m nano_vllm_uno.pull_suite
)

cd "${NANO_REPO_ROOT}"
printf 'Launching %d tp=1 workers across %d node(s):' \
  "${NANO_WORKERS}" "${SLURM_NNODES}"
printf ' %q' "${command[@]}"
printf '\n'

exec srun \
  --unbuffered \
  --overlap \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${NANO_WORKERS}" \
  --ntasks-per-node="${NANO_GPUS_PER_NODE}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --kill-on-bad-exit=1 \
  "${command[@]}"

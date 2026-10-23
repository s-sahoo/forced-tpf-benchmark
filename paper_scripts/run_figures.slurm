#!/usr/bin/env bash
#SBATCH --job-name=paper-figures
#SBATCH --partition=main
#SBATCH --account=arch
#SBATCH --qos=arch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j-%N.out
#SBATCH --error=slurm-%x-%j-%N.err

set -euo pipefail

if (( $# == 0 )); then
  echo "usage: sbatch $0 FIGURE [FIGURE ...]" >&2
  echo "figures: figure1 figure3_t0 figure3_t1 figure4" >&2
  exit 2
fi

repo=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
repo=$(cd -- "${repo}" && pwd)
run_root=${PAPER_RUN_ROOT:-${repo}/runs/paper-${SLURM_JOB_ID}}
figures=("$@")
runner_args=()
if (( SLURM_NTASKS > 1 )); then
  case ${PAPER_SCHEDULER:-balanced} in
    balanced) runner_args+=(--balanced-repetitions) ;;
    paired) runner_args+=(--paired-repetitions) ;;
    *) echo "unknown PAPER_SCHEDULER: ${PAPER_SCHEDULER}" >&2; exit 2 ;;
  esac
fi

srun "${repo}/scripts/run_figures.sh" \
  --figures "${figures[@]}" \
  --out-root "${run_root}" \
  --gpus 0,1,2,3,4,5,6,7 \
  "${runner_args[@]}"

"${repo}/scripts/run_figures.sh" \
  --figures "${figures[@]}" \
  --out-root "${run_root}" \
  --aggregate-only

echo "Paper results: ${run_root}"

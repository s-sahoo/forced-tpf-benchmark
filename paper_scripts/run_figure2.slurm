#!/usr/bin/env bash
#SBATCH --job-name=figure2
#SBATCH --partition=main
#SBATCH --account=arch
#SBATCH --qos=arch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
repo=${SLURM_SUBMIT_DIR:?submit from the repository root}
out_root=${FIGURE2_RUN_ROOT:-${repo}/runs/figure2-${SLURM_JOB_ID}}
srun "${repo}/scripts/reproduce_figure2.sh" "${out_root}"
echo "Figure 2 results: ${out_root}"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/doctor.sh [--metadata | --imports | --gpu]

Validate installed package versions and reject hidden user/global Python
paths. --imports loads each vendored engine without a GPU. --gpu additionally
runs a minimal CUDA tensor check; run that mode inside a one-GPU allocation.
EOF
}

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=metadata
while (( $# )); do
  case $1 in
    --metadata) mode=metadata ;;
    --imports) mode=imports ;;
    --gpu) mode=gpu ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

runtime_root=${FORCED_TPF_ENV_ROOT:-${repo}/.venvs/forced-tpf}
benchmark_env=${BENCHMARK_ENV:-${runtime_root}/benchmark.env}
if [[ ! -f ${benchmark_env} ]]; then
  echo "missing runtime environment: ${benchmark_env}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${benchmark_env}"

extra=()
if [[ ${mode} == imports ]]; then
  extra+=(--imports)
elif [[ ${mode} == gpu ]]; then
  extra+=(--gpu)
fi

clean_python() {
  local python=$1
  shift
  env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "${python}" -I "${repo}/scripts/doctor_runtime.py" "$@" "${extra[@]}"
}

clean_python "${PLOT_PYTHON}" --environment tools
clean_python "${NANO_PYTHON}" --environment spec \
  --source "${repo}/engines/nano-vllm-uno" \
  --source "${repo}/engines/sglang-eagle3-dflash-timing/python"
clean_python "${SGLANG_PYTHON}" --environment nemotron \
  --source "${repo}/engines/sglang-nemotrondiff-lora-timing/python"
clean_python "${VLLM_DIFFGEMMA_PYTHON}" --environment vllm

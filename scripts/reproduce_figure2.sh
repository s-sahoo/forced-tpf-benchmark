#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out_root=${1:?usage: scripts/reproduce_figure2.sh OUT_ROOT [OPTIONS]}
shift
runtime_root=${FORCED_TPF_ENV_ROOT:-${repo}/.venvs/forced-tpf}
local_env=${runtime_root}/benchmark.env
shared_k2m_env=/mnt/weka/shrd/k2m/duc.pham/forced-tpf-runtime/benchmark.env
if [[ -n ${BENCHMARK_ENV:-} ]]; then
  benchmark_env=${BENCHMARK_ENV}
elif [[ -f ${local_env} ]]; then
  benchmark_env=${local_env}
elif [[ -f ${shared_k2m_env} ]]; then
  benchmark_env=${shared_k2m_env}
else
  benchmark_env=${local_env}
fi

if [[ ! -f ${benchmark_env} ]]; then
  echo "missing benchmark environment: ${benchmark_env}" >&2
  echo "run scripts/install.sh or set BENCHMARK_ENV" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${benchmark_env}"
cd "${repo}"
if [[ " $* " == *" --help "* || " $* " == *" -h "* ]]; then
  exec "${SWEEP_PYTHON}" -m benchmark.figure2 --out-root "${out_root}" "$@"
fi
"${SWEEP_PYTHON}" -m benchmark.figure2 --out-root "${out_root}" "$@"

"${PLOT_PYTHON}" paper/figure2/plot.py \
  --data "${out_root}/figure2.csv" \
  --output-dir "${out_root}"

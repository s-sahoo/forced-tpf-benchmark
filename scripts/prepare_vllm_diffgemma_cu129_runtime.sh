#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 VENV_ROOT" >&2
  exit 2
fi

venv_root=$1
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_root=${repo}/engines/vllm-diffgemma-timing/vllm
python=${venv_root}/bin/python

if [[ ! -x "${python}" ]]; then
  echo "Python executable not found at ${python}" >&2
  exit 1
fi
site_packages=$("${python}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
site_root=${site_packages}/vllm

if [[ ! -d "${site_root}" ]]; then
  echo "vLLM package not found at ${site_root}" >&2
  exit 1
fi

# These files differ from the CUDA-12.9 wheel only by the forced-TPF and
# per-sequence diffusion accounting changes. Keep the wheel's native binaries.
files=(
  config/diffusion.py
  outputs.py
  v1/request.py
  v1/core/sched/scheduler.py
  v1/engine/output_processor.py
  v1/engine/__init__.py
  model_executor/models/diffusion_gemma.py
)

for file in "${files[@]}"; do
  install -m 0644 "${source_root}/${file}" "${site_root}/${file}"
done

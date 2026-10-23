#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/install.sh [OPTIONS]

Install the binary-only runtimes used by paper Figures 1–4. This command does
not require a GPU, CUDA toolkit, Slurm allocation, or model checkpoints.

Options:
  --root DIR          Runtime root (default: .venvs/forced-tpf)
  --wheelhouse DIR    Install offline from a prepared runtime bundle
  --uv PATH           Existing uv executable
  --dry-run           Print installation commands without executing them
  -h, --help          Show this help

The runtime contains isolated tools, spec, Nemotron, and vLLM Python
environments. Their shared uv cache prevents duplicate downloads.
EOF
}

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
env_root=${FORCED_TPF_ENV_ROOT:-${repo}/.venvs/forced-tpf}
wheelhouse=
uv_bin=${UV:-}
dry_run=${DRY_RUN:-0}

while (( $# )); do
  case $1 in
    --root)
      env_root=${2:?--root requires a directory}
      shift 2
      ;;
    --wheelhouse)
      wheelhouse=${2:?--wheelhouse requires a directory}
      shift 2
      ;;
    --uv)
      uv_bin=${2:?--uv requires an executable}
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ ${dry_run} != 1 ]]; then
    "$@"
  fi
}

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "the paper CUDA wheels require Linux x86_64" >&2
  exit 1
fi

glibc_version=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
if [[ -z ${glibc_version} ]] || ! printf '2.31\n%s\n' "${glibc_version}" | sort -V -C; then
  echo "the paper wheels require glibc 2.31 or newer (found ${glibc_version:-unknown})" >&2
  exit 1
fi

# Do not let a currently activated environment leak into uv's interpreter
# selection or the generated runtimes.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

env_root=$(realpath -m -- "${env_root}")
run mkdir -p "${env_root}"

if [[ -n ${wheelhouse} ]]; then
  if [[ ! -d ${wheelhouse} ]]; then
    echo "wheelhouse is not a directory: ${wheelhouse}" >&2
    exit 1
  fi
  wheelhouse=$(realpath -e -- "${wheelhouse}")
  if [[ ${wheelhouse} == *' '* ]]; then
    echo "wheelhouse path cannot contain spaces: ${wheelhouse}" >&2
    exit 1
  fi
fi

if [[ -z ${uv_bin} ]]; then
  if [[ -x ${env_root}/bootstrap/bin/uv ]]; then
    uv_bin=${env_root}/bootstrap/bin/uv
  elif [[ -n ${wheelhouse} && -x ${wheelhouse}/bootstrap/uv ]]; then
    uv_bin=${wheelhouse}/bootstrap/uv
  elif [[ ${dry_run} == 1 ]]; then
    uv_bin=${env_root}/bootstrap/bin/uv
    echo "+ bootstrap uv 0.11.19 into ${env_root}/bootstrap/bin"
  else
    if ! command -v curl >/dev/null 2>&1; then
      echo "curl is required to bootstrap uv" >&2
      exit 1
    fi
    temporary=$(mktemp -d)
    trap 'rm -rf -- "${temporary}"' EXIT
    curl -LsSf \
      https://astral.sh/uv/0.11.19/install.sh \
      -o "${temporary}/install-uv.sh"
    run env UV_INSTALL_DIR="${env_root}/bootstrap/bin" \
      sh "${temporary}/install-uv.sh" --no-modify-path
    uv_bin=${env_root}/bootstrap/bin/uv
  fi
fi

if [[ ${dry_run} != 1 && ! -x ${uv_bin} ]]; then
  echo "uv executable not found: ${uv_bin}" >&2
  exit 1
fi
if [[ ${dry_run} != 1 && $("${uv_bin}" --version) != "uv 0.11.19"* ]]; then
  echo "the runtime lock requires uv 0.11.19 (found $("${uv_bin}" --version))" >&2
  exit 1
fi

export UV_CACHE_DIR=${env_root}/cache
export UV_PYTHON_INSTALL_DIR=${env_root}/python
export UV_LINK_MODE=hardlink
export UV_NO_BUILD=1

python_version=3.12.13
locks=${repo}/requirements/runtime
envs=${env_root}/envs
if [[ -n ${wheelhouse} ]]; then
  run "${uv_bin}" python install --no-bin --offline \
    --mirror "file://${wheelhouse}/python" "${python_version}"
else
  run "${uv_bin}" python install --no-bin "${python_version}"
fi

create_env() {
  local name=$1
  local path=${envs}/${name}
  if [[ ! -x ${path}/bin/python ]]; then
    run "${uv_bin}" venv --managed-python --python "${python_version}" "${path}"
  fi
}

render_offline_lock() {
  local source=$1
  local target=$2
  awk '
    /^torchao @ / { print "torchao==0.9.0"; next }
    /^torchcodec @ .*torchcodec-0\.8\.0/ { print "torchcodec==0.8.0"; next }
    /^sglang-kernel @ .*v0\.4\.3/ { print "sglang-kernel==0.4.3+cu129"; next }
    /^sglang-kernel @ .*v0\.4\.5/ { print "sglang-kernel==0.4.5+cu129"; next }
    /^sgl-deep-gemm @ / { print "sgl-deep-gemm==0.1.2+cu129"; next }
    /^vllm @ / { print "vllm==0.26.1rc1.dev353+g7794b1e08.cu129"; next }
    { print }
  ' "${source}" > "${target}"
}

sync_env() {
  local name=$1
  local torch_backend=$2
  local path=${envs}/${name}
  local lock=${locks}/${name}.lock.txt
  local install_lock=${lock}
  local args=(
    pip sync
    --python "${path}/bin/python"
    --no-build
    --link-mode hardlink
  )

  create_env "${name}"
  if [[ -n ${wheelhouse} ]]; then
    run mkdir -p "${env_root}/rendered-locks"
    install_lock=${env_root}/rendered-locks/${name}.lock.txt
    if [[ ${dry_run} == 1 ]]; then
      echo "+ render offline lock ${install_lock}"
    else
      render_offline_lock "${lock}" "${install_lock}"
    fi
    args+=(--offline --no-index --find-links "${wheelhouse}")
  elif [[ -n ${torch_backend} ]]; then
    args+=(--torch-backend "${torch_backend}")
  fi
  run "${uv_bin}" "${args[@]}" "${install_lock}"
}

sync_env tools ""
# The validated Nemotron wheel is the PyPI CUDA-12.8 build. Its separately
# pinned SGLang kernel remains CUDA 12.9; do not redirect Torch to cu129.
sync_env nemotron ""
sync_env spec cu129
sync_env vllm cu129

vllm_dir=${envs}/vllm
if [[ ${dry_run} == 1 ]]; then
  run "${repo}/scripts/prepare_vllm_diffgemma_cu129_runtime.sh" "${vllm_dir}"
else
  "${repo}/scripts/prepare_vllm_diffgemma_cu129_runtime.sh" "${vllm_dir}"
fi

if [[ ${dry_run} == 1 ]]; then
  echo "DRY RUN: would write ${env_root}/benchmark.env"
  exit 0
fi

env_file=${env_root}/benchmark.env
{
  printf 'export FORCED_TPF_INSTALL_ENV_ROOT=%q\n' "${env_root}"
  printf 'export PLOT_PYTHON=%q\n' "${envs}/tools/bin/python"
  printf 'export SWEEP_PYTHON=%q\n' "${envs}/tools/bin/python"
  printf 'export NANO_PYTHON=%q\n' "${envs}/spec/bin/python"
  printf 'export NANO_FA3_PROVIDER=%q\n' sgl_kernel
  printf 'export SGLANG_SPEC_PYTHON=%q\n' "${envs}/spec/bin/python"
  printf 'export SGLANG_PYTHON=%q\n' "${envs}/nemotron/bin/python"
  printf 'export VLLM_DIFFGEMMA_PYTHON=%q\n' "${envs}/vllm/bin/python"
  printf 'export NANO_VLLM_ROOT=%q\n' "${repo}/engines/nano-vllm-uno"
  printf 'if [[ -z ${HF_HOME:-} ]]; then export HF_HOME=%q; fi\n' "${env_root}/huggingface"
  printf 'export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}\n'
  printf 'asset_env=${FORCED_TPF_ASSET_ENV:-}\n'
  printf 'if [[ -z ${asset_env} ]]; then asset_env=%q; fi\n' "${env_root}/assets.env"
  printf 'if [[ -f ${asset_env} ]]; then source "${asset_env}"; fi\n'
  printf 'unset asset_env\n'
} > "${env_file}"
install -m 0644 "${repo}/configs/paper-benchmark.env.example" \
  "${env_root}/assets.env.example"

BENCHMARK_ENV=${env_file} "${repo}/scripts/doctor.sh" --metadata

echo
echo "Installed binary-only runtimes under: ${env_root}"
echo "Runtime environment: ${env_file}"
echo "Asset template: ${env_root}/assets.env.example"
echo "No GPU, model checkpoint, experiment, or plot was used."
echo "After configuring assets, validate on a GPU with:"
echo "  srun --gres=gpu:1 ${repo}/scripts/doctor.sh --gpu"

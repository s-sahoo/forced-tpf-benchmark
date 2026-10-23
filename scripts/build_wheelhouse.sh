#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/build_wheelhouse.sh OUTPUT_DIR [--uv PATH]

Download uv, managed Python, and every pinned package wheel for a fully offline
Linux x86_64 install. This command needs network access but no GPU or CUDA
toolkit. Copy OUTPUT_DIR and the repository to the target machine, then run:

  scripts/install.sh --wheelhouse OUTPUT_DIR
EOF
}

if (( $# == 0 )); then
  usage >&2
  exit 2
fi
if [[ $1 == -h || $1 == --help ]]; then
  usage
  exit 0
fi

output=$1
shift
uv_bin=${UV:-}
while (( $# )); do
  case $1 in
    --uv)
      uv_bin=${2:?--uv requires an executable}
      shift 2
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

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo "wheelhouse creation requires Linux x86_64" >&2
  exit 1
fi
if [[ -z ${uv_bin} ]]; then
  uv_bin=$(command -v uv || true)
fi
if [[ -z ${uv_bin} || ! -x ${uv_bin} ]]; then
  echo "uv is required; install it or pass --uv PATH" >&2
  exit 1
fi
if [[ $("${uv_bin}" --version) != "uv 0.11.19"* ]]; then
  echo "wheelhouse creation requires uv 0.11.19 (found $("${uv_bin}" --version))" >&2
  exit 1
fi

unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=$(realpath -m -- "${output}")
if [[ ${output} == *' '* ]]; then
  echo "output path cannot contain spaces: ${output}" >&2
  exit 1
fi
mkdir -p "${output}"

mkdir -p "${output}/bootstrap"
install -m 0755 "${uv_bin}" "${output}/bootstrap/uv"

python_release=20260602
python_archive=cpython-3.12.13+${python_release}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
python_dir=${output}/python/${python_release}
mkdir -p "${python_dir}"
if [[ ! -f ${python_dir}/${python_archive} ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download managed Python" >&2
    exit 1
  fi
  curl -fL \
    "https://github.com/astral-sh/python-build-standalone/releases/download/${python_release}/${python_archive/+/%2B}" \
    -o "${python_dir}/${python_archive}"
fi

temporary=$(mktemp -d)
trap 'rm -rf -- "${temporary}"' EXIT
export UV_CACHE_DIR=${UV_CACHE_DIR:-${temporary}/uv-cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-${temporary}/python}
"${uv_bin}" python install --no-bin --mirror "file://${output}/python" 3.12.13
"${uv_bin}" venv --seed --managed-python --python 3.12.13 "${temporary}/download"
pip=${temporary}/download/bin/pip

download() {
  local name=$1
  shift
  echo "Downloading ${name} wheels"
  "${pip}" download \
    --dest "${output}" \
    --no-deps \
    --only-binary=:all: \
    "$@" \
    --requirement "${repo}/requirements/runtime/${name}.lock.txt"
}

download tools
download nemotron
for name in spec vllm; do
  download "${name}" --extra-index-url https://download.pytorch.org/whl/cu129
done

echo "Wheelhouse ready: ${output}"

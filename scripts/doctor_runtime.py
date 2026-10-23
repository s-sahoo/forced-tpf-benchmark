#!/usr/bin/env python3
"""Validate one isolated paper runtime without loading model checkpoints."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import sys


EXPECTED = {
    "tools": {"matplotlib": "3.11.0", "numpy": "2.5.0"},
    "nemotron": {
        "sglang": "0.5.9",
        "torch": "2.9.1",
        "transformers": "5.11.0",
        "flashinfer-python": "0.6.12",
        "sglang-kernel": "0.4.3+cu129",
    },
    "spec": {
        "sglang": "0.5.9",
        "torch": "2.11.0+cu129",
        "transformers": "5.11.0",
        "flashinfer-python": "0.6.12",
        "sglang-kernel": "0.4.5+cu129",
    },
    "vllm": {
        "vllm": "0.26.1rc1.dev353+g7794b1e08.cu129",
        "torch": "2.13.0+cu129",
        "transformers": "5.14.1",
        "flashinfer-python": "0.6.15.post1",
    },
}


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_paths() -> None:
    allowed = (Path(sys.prefix), Path(sys.base_prefix))
    foreign = []
    for value in sys.path:
        if not value or "site-packages" not in value:
            continue
        path = Path(value)
        if not any(within(path, root) for root in allowed):
            foreign.append(str(path))
    if foreign:
        raise RuntimeError(f"foreign site-packages on sys.path: {foreign}")


def validate_metadata(environment: str) -> None:
    for package, expected in EXPECTED[environment].items():
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(
                f"{environment}: {package}={actual}, expected {expected}"
            )
        distribution = metadata.distribution(package)
        root = Path(distribution.locate_file(""))
        if not within(root, Path(sys.prefix)):
            raise RuntimeError(
                f"{environment}: {package} loaded metadata from {root}"
            )


def validate_imports(environment: str, sources: list[Path], gpu: bool) -> None:
    for source in reversed(sources):
        sys.path.insert(0, str(source.resolve()))

    if environment == "tools":
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        return

    import torch

    if environment == "spec":
        import flashinfer  # noqa: F401
        import nano_vllm_uno
        import sgl_kernel  # noqa: F401
        import sglang

        expected = {"nano_vllm_uno": sources[0], "sglang": sources[1]}
        modules = {"nano_vllm_uno": nano_vllm_uno, "sglang": sglang}
    elif environment == "nemotron":
        import flashinfer  # noqa: F401
        import sglang
        from sglang.srt.dllm.algorithm.linear_spec import LinearSpec  # noqa: F401

        expected = {"sglang": sources[0]}
        modules = {"sglang": sglang}
    elif environment == "vllm":
        import flashinfer  # noqa: F401
        import vllm  # noqa: F401

        expected = {}
        modules = {}
    else:  # pragma: no cover
        raise ValueError(environment)

    for name, root in expected.items():
        origin = Path(modules[name].__file__)
        if not within(origin, root):
            raise RuntimeError(f"{name} imported from {origin}, expected {root}")

    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_name = torch.cuda.get_device_name()
        if "H200" not in device_name:
            raise RuntimeError(
                f"paper measurements require an H200, got {device_name}"
            )
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            raise RuntimeError(
                f"paper runtimes require an SM90-or-newer GPU, got SM{capability[0]}{capability[1]}"
            )
        value = torch.ones(1, device="cuda")
        if value.item() != 1:
            raise RuntimeError("CUDA tensor smoke test failed")
        torch.cuda.synchronize()
        print(
            f"{environment}: {device_name}, "
            f"CUDA SM{capability[0]}{capability[1]} OK"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=EXPECTED, required=True)
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--imports", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    os.environ.pop("PYTHONPATH", None)
    validate_paths()
    validate_metadata(args.environment)
    if args.imports or args.gpu:
        validate_imports(args.environment, args.source, args.gpu)
    print(f"{args.environment}: metadata OK")


if __name__ == "__main__":
    main()

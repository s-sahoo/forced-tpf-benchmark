# Binary runtime locks

These exact locks reproduce the four validated Python environments used by
the paper workflow:

- `tools`: planner, tests, aggregation, and plots.
- `spec`: Nano, SGLang AR, EAGLE-3, and DFlash.
- `nemotron`: Nemotron linear-spec LoRA.
- `vllm`: DiffusionGemma on vLLM.

`scripts/install.sh` synchronizes them with `uv pip sync --no-build`, so a
missing binary wheel is a hard failure instead of an accidental local CUDA
compilation. Direct URLs pin the custom SGLang and vLLM CUDA 12.9 wheels.

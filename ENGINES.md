# Engine provenance

The engine snapshots are ordinary versioned directories. Use path-scoped Git
history to inspect the imported baseline and the timing/forced-TPF delta.

| Engine | Path | Upstream baseline | Paper use |
|---|---|---|---|
| Nano-vLLM-Uno | `engines/nano-vllm-uno` | `d08f6fcb329ad7dfc1e5ab75f23125f5ba16a7a0` | Figures 1, 2, 3, 4 |
| SGLang EAGLE-3/DFlash | `engines/sglang-eagle3-dflash-timing` | `3d3ba4f7468764a056b211c6fd6e2fdf4b9ba030` | Figures 1, 3 |
| SGLang Nemotron LinearSpec-LoRA | `engines/sglang-nemotrondiff-lora-timing` | `ce4286ed0aa1a242e79f36c493f31140217feff9` | Figure 4 |
| vLLM DiffusionGemma | `engines/vllm-diffgemma-timing` | `811622c410c2f1baf2fa7056ca19753264b95815` | Figure 4 |

Each directory contains `SOURCE_PROVENANCE.json` where available. Local
changes provide deterministic forced TPF, per-sequence output counters, timing
transport, and the compatibility fixes needed by its paper runner.
The detailed Figure 2 hooks are opt-in and add no work unless their timing
environment variable is set.

DiffusionGemma runs a pinned prebuilt vLLM wheel
(`0.26.1rc1.dev353+g7794b1e08.cu129`) with the repository's Python delta
overlaid by `scripts/prepare_vllm_diffgemma_cu129_runtime.sh`; vLLM is not
compiled locally.

Example:

```bash
git log -- engines/sglang-eagle3-dflash-timing
git diff 3d3ba4f7468764a056b211c6fd6e2fdf4b9ba030 -- \
  engines/sglang-eagle3-dflash-timing
```

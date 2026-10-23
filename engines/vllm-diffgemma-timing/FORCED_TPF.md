# DiffusionGemma forced-TPF instrumentation

This directory is an ordinary vendored snapshot of `vllm-project/vllm`; it is
not a nested Git repository. The exact upstream revision is recorded in
`SOURCE_PROVENANCE.json`.

## Launch configuration

Enable the paper calibration target through vLLM's native diffusion config:

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
  --diffusion-config '{"forced_tpf": 16.8}'
```

`forced_tpf` is benchmark-only. Native stable-and-confident stopping is used
when it is omitted.

For a full 256-token canvas and target 16.8 TPF, the planner assigns a shared
15- or 16-denoising-forward budget to every sequence that begins a canvas in
the same engine step. The assignment is deterministic and balances cumulative
token slots against forward slots, including when cohort sizes change. The
normal 48-step temperature schedule, sampling, entropy, renoising, and
self-conditioning work are still executed. The subsequent causal commit
forward remains unchanged and is not counted as a denoising forward.

## Per-sequence output

Offline `CompletionOutput` objects carry these cumulative fields:

- `dllm_generated_tokens`
- `dllm_decode_forward_passes`
- `dllm_decode_tpf`

OpenAI chat and completion responses expose the same fields under each
choice's `meta_info`. Streaming content and terminal choices carry the latest
cumulative values. Ordinary non-diffusion choices leave `meta_info` unset.

For paper runs, use output lengths divisible by the 256-token canvas length.
With 16,384 output tokens the deterministic schedule uses 975 denoising
forwards, giving an attained TPF of `16384 / 975 = 16.80410256`.

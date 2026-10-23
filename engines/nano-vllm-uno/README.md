# nano-vllm-uno

`nano-vllm-uno` is a compact GPU inference engine built around a unified
two-pass decoder. A forward block size of `1` performs autoregressive decoding;
larger blocks perform parallel draft and causal verification passes.

The runtime includes paged KV-cache management, tensor and data parallelism,
CUDA graphs, fused verification, and optional draft-gated LoRA.

## Installation

Install a CUDA-compatible PyTorch build for the target platform, then install
the package:

```bash
pip install .
```

For development, use:

```bash
pip install -e .
```

FlashAttention 3 is the default backend and requires a separate installation.
The wheel used in development targets Hopper GPUs, CUDA 12.8, and PyTorch 2.8:

```bash
uv pip install \
  'https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.05.20-072445f/flash_attn_3-3.0.0%2B20260519.cu128torch280cxx11abitrue.8a8b2f-cp39-abi3-linux_x86_64.whl'
```

For a different CUDA or PyTorch combination, build FlashAttention 3 from the
[official repository](https://github.com/Dao-AILab/flash-attention):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

Verify that the FA3 interface is importable:

```bash
python -c 'from flash_attn_interface import flash_attn_with_kvcache; print("FA3 ready")'
```

Select the backend explicitly when constructing the engine, for example
`LLM(..., attention_backend="fa2")`, or pass `--attention-backend fa2` to the
HTTP server. The default is `fa3`; a missing requested backend raises an import
error instead of silently changing implementations. FlashAttention 2 is
installed by `pip install .`. Tree verification requires FA3 or FA4.

FA4 is available through SGLang's compatibility wrapper and requires a
FlashAttention-4/CuTeDSL installation. Use
`LLM(..., attention_backend="fa4")`; its required 128-token KV page size is
selected automatically. FA4 tree verification is supported; FA2 tree
verification is not.

## Quantized checkpoints

The runtime loads serialized 128x128 block-FP8 decoder linears
(`quant_method=fp8`, dynamic activations) without dequantizing their weights.
They execute through SGLang's DeepGEMM kernel and return BF16 outputs. A
compatible SGLang/`sgl-kernel` installation with DeepGEMM support is required.

Runtime gated LoRA is supported over quantized base weights. Serialized base
weights remain FP8, while LoRA A/B use the model activation dtype and their
residual is accumulated into the kernel's BF16 output. Use an unmerged base
checkpoint when the adapter is not already part of the quantized weights.

## Basic usage

This single-GPU example follows the BBQ AIME25 setup, with a shorter context
and output limit for a quick run:

```python
from transformers import AutoTokenizer

from nano_vllm_uno import LLM, SamplingParams

MODEL = "/path/to/bbq-base-checkpoint"
GATED_LORA = "/path/to/bbq-diffusion-lora"
BLOCK_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=True,
    trust_remote_code=True,
)
problem = (
    r"Find the sum of all integer bases $b>9$ for which $17_b$ is a "
    r"divisor of $97_b.$"
)
prompt = tokenizer.apply_chat_template(
    [{
        "role": "user",
        "content": (
            "Please reason step by step and put your final answer in "
            "\\boxed{}.\n\n" + problem
        ),
    }],
    tokenize=False,
    add_generation_prompt=True,
    reasoning_effort="high",
)

llm = LLM(
    model=MODEL,
    tokenizer_path=MODEL,
    tensor_parallel_size=1,
    data_parallel_size=1,
    max_model_len=8192,
    max_num_batched_tokens=8192,
    max_num_seqs=1,
    gpu_memory_utilization=0.85,
    attention_backend="fa3",
    max_diffusion_block_size=BLOCK_SIZE,
    cuda_graph_block_sizes=[1, BLOCK_SIZE],
    cuda_graph_batch_sizes=[1],
    gated_lora_path=GATED_LORA,
)

params = SamplingParams(
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    max_tokens=4096,
    stop_token_ids=[250019],
    mask_token_id=151669,
    noise_mode="random_uniform",
    diffusion_block_size=BLOCK_SIZE,
)

try:
    outputs = llm.generate([prompt], params)
    print(outputs[0]["text"])
finally:
    llm.exit()
```

All prompts in one `generate` call share one `SamplingParams` object. With
`data_parallel_size > 1`, persistent replicas shard the prompt batch across
the visible GPUs and restore the original result order.

## Two-pass decoding

For a diffusion block of length `L`, each cycle performs:

1. Draft forward: `[seed, noise_1, ..., noise_(L-1)]`.
2. Verify forward: `[clean, spec_1, ..., spec_(L-1)]`.
3. Left-to-right acceptance or correction and scheduler post-processing.

```python
llm = LLM(
    model="/path/to/base/checkpoint",
    max_diffusion_block_size=8,
    gated_lora_path="/path/to/adapter",
)

params = SamplingParams(
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    max_tokens=4096,
    diffusion_block_size=8,
    mask_token_id=151669,
    noise_mode="random_uniform",
)

outputs = llm.generate(prompts, params)
```

`mask_token_id` must match the diffusion model's training setup. Available
noise modes are `random_uniform`, `deterministic_uniform`, and `mask`.

Supplying `gated_lora_path` applies the adapter only to draft noise rows.
Prefill, seed, verification, and autoregressive rows continue to use the base
model. Omitting the path disables LoRA.

### Opt-in tree verification

Set a fixed target query size to replace the linear verifier with a
best-first prefix tree:

```python
llm = LLM(
    model="/path/to/base/checkpoint",
    max_diffusion_block_size=8,
    tree_verify_size=16,
    tree_candidate_top_k=16,
)
```

The request `diffusion_block_size` must not exceed the engine's
`max_diffusion_block_size`. By default, every width from 1 through that maximum
is captured as a CUDA graph. `tree_verify_size` is independent of the runtime
draft width; leaving it as `None` preserves the existing linear path. FA3
handles the shared prefix, and one Triton kernel computes the ancestor-only
tree suffix and exact merge in place. Other backends fall back to dense masked
SDPA as a correctness reference. Fixed tree shapes are captured in their own
CUDA graphs.

Tree construction uses BF16 top-k, GPU log normalization, and a fixed-budget
best-first kernel without a host synchronization. Fixed tree, walk, and KV
compaction buffers are reused across cycles; the Python heap implementation
remains the CPU correctness reference. The query budget is fixed for the whole
batch, and dynamic CaDD-style budgets are not implemented.
## Non-streaming API server

The API server keeps one `LLMEngine` alive behind a request queue. Requests
that arrive during generation join the scheduler between engine steps:

```bash
python -m nano_vllm_uno.server \
  --model /path/to/checkpoint \
  --served-model-name nano-vllm-uno \
  --max-tokens 4096 \
  --temperature 0 \
  --host 0.0.0.0 \
  --port 1919
```

It provides `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health`.
The request contract accepts `model` and exactly one of `messages` or `prompt`:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nano-vllm-uno","prompt":"Hello"}'
```

Sampling, stopping, diffusion, and output-length settings are fixed when the
server starts. Request-level sampling fields, streaming, and multiple choices
are not part of this initial API contract.

## Cross-node pull evaluation

The offline suite runner uses one shared pull queue across all allocated GPUs.
It always runs one `tp=1` engine per GPU, saves each completion immediately to
the matching benchmark directory, and resumes by skipping generation IDs that
are already present. `NANO_DATA_ROOT` must contain the prepared JSONL files
from the release benchmark suite.

Submit any number of eight-GPU nodes with one Slurm call:

```bash
NANO_MODEL=/shared/model \
NANO_GATED_LORA_PATH=/shared/adapter \
NANO_DATA_ROOT=/shared/benchmarks \
NANO_RESULTS_ROOT=/shared/results \
NANO_RUN_NAME=uno-eval \
sbatch --nodes=4 scripts/run_pull_suite_slurm.sh
```

The hard-coded suite contains the 14 release benchmarks, with the number of
generations per problem specified independently for each benchmark. All
requests use 32,768 maximum output tokens, 64 active sequences per GPU,
diffusion width 16, and CUDA-graph block width 16. CUDA-graph batch sizes use
the engine's default ladder.

## Data-parallel AIME25 example

[`examples/run_aime25_dp.py`](examples/run_aime25_dp.py) reads AIME25-style
JSONL records and runs one shared generation call across independent GPU
replicas:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/run_aime25_dp.py \
  --model /path/to/checkpoint \
  --data /path/to/aime25.jsonl \
  --data-parallel-size 4 \
  --output aime25-generations.jsonl
```

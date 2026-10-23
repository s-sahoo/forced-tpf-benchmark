#!/usr/bin/env bash

set -euo pipefail

: "${MODEL:?Set MODEL to the base model directory}"
: "${GATED_LORA_PATH:?Set GATED_LORA_PATH to the adapter directory}"
: "${DATA:?Set DATA to the MATH500 JSONL file}"
: "${OUTPUT:?Set OUTPUT to the generation JSONL file}"

python examples/run_aime25_dp.py \
  --model "${MODEL}" \
  --tokenizer-path "${MODEL}" \
  --gated-lora-path "${GATED_LORA_PATH}" \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --max-num-seqs 4 \
  --max-num-batched-tokens 262144 \
  --gpu-memory-utilization 0.90 \
  --max-tokens 131072 \
  --num-samples 10 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 50 \
  --diffusion-block-size 16 \
  --noise-mode random_uniform

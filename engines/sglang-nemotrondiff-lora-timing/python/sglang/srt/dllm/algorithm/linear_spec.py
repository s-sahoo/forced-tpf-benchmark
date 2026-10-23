"""
LinearSpec algorithm for SGLang DLLM — diffusion model as speculator.

Implements the linear_spec_generate approach from Nemotron-Labs-Diffusion v2:
each block needs only 2 forward passes (1 bidirectional draft + 1 causal verify)
instead of FastDiffuser's iterative denoising.

Algorithm per block:
  1. DRAFT (bidirectional): forward [seed, mask, ..., mask] → draft tokens
  2. VERIFY (causal): forward [seed, draft_1, ..., draft_N] → AR tokens
  3. ACCEPT: c = consecutive shifted matches (draft[i] == ar[i-1]), output seed + c tokens
  4. FREE rejected KV positions

Key insight: in the diffusion model,
  - Bidirectional (draft) logits[i] predicts the token at position i
  - Causal (verify) logits[i] predicts the token at position i+1

So draft[i] and ar[i-1] both predict position i → comparison is shifted by 1.
The seed at position 0 provides the correct AR token (from prefix/prev block),
and ar[c] becomes the seed for the next block.

Usage:
  dllm_algorithm: LinearSpec
  dllm_algorithm_config:
    causal_context: true
    forced_tpf: null  # benchmark-only; e.g. 5.0
    forced_tpf_seed: 0
    stats_file: null
"""

import json as _json
import logging
import os
import random
import time
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.lora_utils import load_peft_lora_deltas
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _forced_emit_count(
    target_tpf: Fraction,
    max_emit: int,
    completed_forwards: int,
    emitted_tokens: int,
    rng: random.Random,
) -> int:
    """Stochastically round the cumulative target for this two-forward cycle."""
    target_total = target_tpf * (completed_forwards + 2)
    lower_total = target_total.numerator // target_total.denominator
    remainder = target_total.numerator % target_total.denominator
    desired_total = lower_total
    if remainder and rng.random() < remainder / target_total.denominator:
        desired_total += 1
    return min(max(desired_total - emitted_tokens, 1), max_emit)


class LinearSpec(DllmAlgorithm):
    """
    LinearSpec: diffusion model as speculator for speculative decoding.

    Each block is processed with exactly 2 forward passes:
      - Draft pass (bidirectional attention): generates candidate tokens
      - Verify pass (causal attention): produces AR-quality tokens
    The seed at position 0 is always correct (from prefix or prev block).
    Acceptance: longest consecutive prefix where draft[i] == ar[i-1] for i>=1.
    Output includes the seed, giving c+1 tokens total.
    """

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        cfg = config.algorithm_config
        self.causal_context: bool = config.causal_context
        self._seed_tokens: Dict[str, int] = {}

        # Benchmark-only acceptance forcing. LinearSpec always performs two
        # full model forwards per cycle, so a target TPF of x requires an
        # average of 2*x committed tokens.
        forced_tpf = cfg.get("forced_tpf")
        self._forced_tpf: Optional[Fraction] = (
            Fraction(str(forced_tpf)) if forced_tpf is not None else None
        )
        self._forced_num_forwards: Dict[str, int] = {}
        self._forced_num_tokens: Dict[str, int] = {}
        self._forced_tpf_rng: Optional[random.Random] = None
        if self._forced_tpf is not None:
            if self._forced_tpf <= 0:
                raise ValueError("LinearSpec forced_tpf must be positive")
            forced_tokens_per_cycle = 2 * self._forced_tpf
            if forced_tokens_per_cycle < 1:
                raise ValueError(
                    "LinearSpec forced_tpf must be at least 0.5 so every "
                    "two-forward cycle commits a token"
                )
            max_forced_tokens = (
                forced_tokens_per_cycle.numerator
                + forced_tokens_per_cycle.denominator
                - 1
            ) // forced_tokens_per_cycle.denominator
            if max_forced_tokens > self.block_size:
                raise ValueError(
                    "LinearSpec forced_tpf requires up to "
                    f"{max_forced_tokens} committed tokens per cycle, but "
                    f"block_size is only {self.block_size}"
                )
            self._forced_tpf_rng = random.Random(
                int(cfg.get("forced_tpf_seed", 0))
            )

        # Stats
        self._stats_file: Optional[str] = cfg.get("stats_file", None)
        self._stats_forward_passes: int = 0
        # Per-request decode forwards from the most recent run. The scheduler
        # consumes this immediately and accumulates it on each request.
        self._last_decode_forward_passes: List[int] = []

        # LoRA: pre-compute fused deltas, bake into CUDA graphs or swap at runtime.
        # lora_mode: "draft_only" (default) — LoRA on draft pass only.
        #            "both"       — LoRA on both draft and verify passes (matches HF).
        #
        # _lora_deltas stores (param, delta_bf16, module) triples.
        # The module ref is needed for FP8 mode to access weight_scale.
        self._lora_path: Optional[str] = cfg.get("lora_path", None)
        self._lora_mode: str = cfg.get("lora_mode", "draft_only")
        self._lora_deltas: Optional[
            List[Tuple[torch.nn.Parameter, torch.Tensor, torch.nn.Module]]
        ] = None
        self._dual_weights: Optional[
            List[Tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]]
        ] = None
        self._graphs_baked: bool = False

        # Profiling accumulators (wall-clock with cuda sync, zero-overhead when disabled)
        self._profile: bool = cfg.get("profile", False)
        self._prof_n: int = 0
        self._prof_scheduler: float = 0.0  # between blocks (scheduler overhead)
        self._prof_pre_draft: float = 0.0  # setup inside run() before draft fwd
        self._prof_draft_fwd: float = 0.0
        self._prof_between_fwd: float = 0.0
        self._prof_verify_fwd: float = 0.0
        self._prof_accept: float = 0.0
        self._prof_total: float = 0.0
        self._prof_last_return: float = 0.0  # time.perf_counter at last return

        # Opt-in, low-overhead decode timing. CUDA events are resolved only
        # after the request cohort finishes, never in the decode hot path.
        self._decode_timing_enabled = bool(
            os.environ.get("SGLANG_DECODE_TIMING_PATH")
        )
        self._decode_timing_step = 0
        self._decode_forward_events: List[dict] = []

        logger.info(
            "LinearSpec: block_size=%d  causal_context=%s  lora=%s  "
            "lora_mode=%s  forced_tpf=%s",
            self.block_size,
            self.causal_context,
            self._lora_path or "none",
            self._lora_mode,
            float(self._forced_tpf) if self._forced_tpf is not None else "off",
        )

    def setup(self, model_runner: ModelRunner) -> None:
        """Called once at startup (before any requests) to load LoRA and
        optionally bake dual weights into CUDA graphs."""
        if self._lora_path is None:
            return
        self._load_lora_deltas(model_runner)
        if self._lora_deltas and model_runner.graph_runner is not None:
            self._bake_dual_weights_into_graphs(model_runner)

    def _bake_dual_weights_into_graphs(self, model_runner: ModelRunner) -> None:
        """Build draft weight copies from LoRA deltas, set CUDA graph hooks,
        and trigger deferred capture.

        After this, draft CUDA graphs read from fused (base+LoRA) memory and
        verify graphs read from the original param.data — zero per-block overhead.
        Only one extra copy per LoRA-targeted param (draft_copy), not two.

        FP8 mode: weights are FP8-quantized and transposed.  We dequantize to
        BF16, add the LoRA delta, and re-quantize.  Both weight AND weight_scale
        parameters get dual entries so the GEMM stays correct.
        """
        dual: List[Tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []

        # Detect FP8: check if the first LoRA-targeted weight is FP8
        first_param = self._lora_deltas[0][0]
        is_fp8 = first_param.data.dtype in (
            torch.float8_e4m3fn,
            getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
        )

        if is_fp8:
            from sglang.srt.layers.quantization.fp8_kernel import (
                per_token_group_quant_fp8,
            )

            logger.info("LinearSpec: FP8 mode — dequant+requant for draft weights")

        for param, delta, module in self._lora_deltas:
            base_ref = param.data  # reuse original tensor, no clone

            if is_fp8:
                # --- FP8 path: dequant → add delta → requant ---
                scale_param = module.weight_scale
                # Stored layout: weight (K, N) FP8, scale (groups, N) float
                # Original layout: weight (N, K), scale (N, groups)
                qw_orig = base_ref.t().float()  # (N, K) float32
                sc_orig = scale_param.data.t().float()  # (N, groups) float32
                base_bf16 = (qw_orig * sc_orig).to(torch.bfloat16)  # dequant

                draft_bf16 = base_bf16 + delta.to(base_bf16.device, torch.bfloat16)

                K = draft_bf16.shape[-1]
                draft_fp8, draft_sc = per_token_group_quant_fp8(draft_bf16, K)
                # Store in same layout as original:
                # weight: qweight.t() — column-major (K, N) view (no .contiguous()!)
                # scale: weight_scale.t().contiguous() — row-major (groups, N)
                draft_fp8_stored = draft_fp8.t()
                draft_sc_stored = draft_sc.t().contiguous()

                dual.append((param, base_ref, draft_fp8_stored))
                dual.append((scale_param, scale_param.data, draft_sc_stored))
            else:
                draft_copy = base_ref + delta
                dual.append((param, base_ref, draft_copy))

        self._dual_weights = dual

        total_mb = sum(d.numel() * d.element_size() for _, _, d in dual) / 1e6
        logger.info(
            "LinearSpec dual weights built from LoRA: %d params, %.1f MB extra",
            len(dual),
            total_mb,
        )

        def set_lora():
            for p, _, d in self._dual_weights:
                p.data = d

        def set_base():
            for p, b, _ in self._dual_weights:
                p.data = b

        gr = model_runner.graph_runner
        gr._dllm_pre_draft_hook = set_lora
        # "both" mode: verify graph also bakes in LoRA weights.
        # "draft_only": verify graph uses base weights.
        gr._dllm_pre_verify_hook = set_lora if self._lora_mode == "both" else set_base

        if model_runner.server_args.defer_cuda_graph_capture:
            logger.info(
                "LinearSpec: capturing CUDA graphs with baked-in LoRA weights..."
            )
            gr.init_capture()
            self._graphs_baked = True
            # Free deltas — no longer needed once graphs are baked
            self._lora_deltas = None
            logger.info("LinearSpec: CUDA graph capture done (LoRA baked)")

    def _load_lora_deltas(self, model_runner: ModelRunner) -> None:
        """Load PEFT LoRA checkpoint and compute fused weight deltas (once).

        Skipped if no LoRA configured, deltas already loaded, or the LoRA
        has been baked into CUDA graphs (deltas are freed post-bake and
        re-loading from disk every block would be wasteful).
        """
        if (
            self._lora_deltas is not None
            or self._lora_path is None
            or self._graphs_baked
        ):
            return
        deltas = load_peft_lora_deltas(self._lora_path, model_runner.model)
        self._lora_deltas = deltas
        total_mb = sum(d.numel() * d.element_size() for _, d, _ in deltas) / 1e6
        logger.info(
            "LinearSpec LoRA loaded: %d delta tensors, %.1f MB", len(deltas), total_mb
        )

    def release_request_state(self, rid: str) -> None:
        """Drop seed and forced-accounting state when a request finishes."""
        self._seed_tokens.pop(rid, None)
        self._forced_num_forwards.pop(rid, None)
        self._forced_num_tokens.pop(rid, None)

    def reset_decode_timing(self) -> None:
        self._decode_timing_step = 0
        self._decode_forward_events.clear()

    def get_decode_timing(self) -> dict:
        if not self._decode_timing_enabled:
            return {"enabled": False, "forward_records": []}
        torch.cuda.synchronize()
        records = []
        for raw in self._decode_forward_events:
            record = {
                key: value
                for key, value in raw.items()
                if key not in ("start", "end")
            }
            record["gpu_ms"] = raw["start"].elapsed_time(raw["end"])
            records.append(record)
        return {"enabled": True, "forward_records": records}

    def _timed_forward(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        role: str,
    ):
        timing_start = None
        if self._decode_timing_enabled:
            timing_start = torch.cuda.Event(enable_timing=True)
            timing_start.record()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        if timing_start is not None:
            timing_end = torch.cuda.Event(enable_timing=True)
            timing_end.record()
            self._decode_forward_events.append(
                {
                    "start": timing_start,
                    "end": timing_end,
                    "step_index": self._decode_timing_step,
                    "role": role,
                    "batch_size": forward_batch.batch_size,
                    "forward_width": self.block_size,
                    "cuda_graph_hit": bool(out.can_run_graph),
                    "lm_head_in_graph": bool(out.can_run_graph),
                }
            )
        return out

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if self._profile:
            _t_entry = time.perf_counter()

        batch_size = forward_batch.batch_size
        bs = self.block_size

        # ----------------------------------------------------------------
        # Fast path: no masks → single forward (STAGING_PREFILL phase)
        # ----------------------------------------------------------------
        mask_index_all = forward_batch.input_ids == self.mask_id
        if not mask_index_all.any():
            if forward_batch.input_ids.numel() == 0:
                empty_logits = LogitsProcessorOutput(
                    next_token_logits=None, full_logits=None
                )
                return empty_logits, [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

            # Extract seed token from the prefill's last-position logits.
            # In causal mode, logits[-1] predicts the first generated token.
            logits_output = out.logits_output
            if logits_output.next_token_logits is not None:
                rids = forward_batch.rids
                seed_logits = logits_output.next_token_logits  # [batch, V]
                seed_logits_no_mask = seed_logits.clone()
                seed_logits_no_mask[:, self.mask_id] = -np.inf
                seeds = torch.argmax(seed_logits_no_mask, dim=-1)

                for b_idx in range(len(rids)):
                    rid = rids[b_idx]
                    self._seed_tokens[rid] = int(seeds[b_idx].item())
                    if self._forced_tpf is not None:
                        self._forced_num_forwards[rid] = 0
                        self._forced_num_tokens[rid] = 0

            return logits_output, [], out.can_run_graph

        expected_tokens = batch_size * bs
        actual_tokens = forward_batch.input_ids.numel()
        if actual_tokens != expected_tokens:
            raise ValueError(
                "LinearSpec decode requires one complete block per request: "
                f"expected {batch_size} * {bs} = {expected_tokens} tokens, "
                f"got {actual_tokens}"
            )

        if self._decode_timing_enabled:
            self._decode_timing_step += 1

        # ----------------------------------------------------------------
        # Per-request: figure out gen_start (where masks begin in block)
        # ----------------------------------------------------------------
        start_list: List[int] = []
        for b in range(batch_size):
            b_start = b * bs
            b_end = b_start + bs
            n_masks = int(
                (forward_batch.input_ids[b_start:b_end] == self.mask_id).sum()
            )
            start_list.append(bs - n_masks)

        # ----------------------------------------------------------------
        # Seed token handling: inject at first mask position
        # ----------------------------------------------------------------
        rids = forward_batch.rids[:batch_size]
        has_seed = [False] * batch_size
        for b in range(batch_size):
            rid = rids[b]
            if rid in self._seed_tokens:
                seed = self._seed_tokens[rid]
                block_start = b * bs + start_list[b]
                if block_start < forward_batch.input_ids.numel():
                    forward_batch.input_ids[block_start] = seed
                    has_seed[b] = True

        # ----------------------------------------------------------------
        # DRAFT pass (bidirectional attention) — with optional LoRA
        # ----------------------------------------------------------------
        # Lazy init for non-baked path (when setup() wasn't called).
        self._load_lora_deltas(model_runner)

        if self._profile:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
            if self._prof_last_return > 0:
                self._prof_scheduler += _t_entry - self._prof_last_return
            self._prof_pre_draft += _t0 - _t_entry

        # When _graphs_baked, CUDA graphs already have the correct weights —
        # no runtime delta swap needed.
        need_swap = self._lora_deltas is not None and not self._graphs_baked
        if need_swap:
            for param, delta, _mod in self._lora_deltas:
                param.data.add_(delta)
        out = self._timed_forward(model_runner, forward_batch, "draft")
        # draft_only: remove LoRA after draft so verify uses base weights.
        # both: keep LoRA active for verify (removed after verify below).
        if need_swap and self._lora_mode == "draft_only":
            for param, delta, _mod in self._lora_deltas:
                param.data.sub_(delta)
        self._stats_forward_passes += 1
        draft_logits = out.logits_output.full_logits  # [B*bs, V]

        if self._profile:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

        # Get draft tokens via argmax (exclude mask_id, in-place)
        draft_logits[:, self.mask_id] = -1e9
        draft_all = torch.argmax(draft_logits, dim=-1)  # [B*bs]

        # Replace mask positions with draft tokens (vectorized)
        mask_positions = forward_batch.input_ids == self.mask_id
        forward_batch.input_ids[mask_positions] = draft_all[mask_positions]

        if self._profile:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()

        # ----------------------------------------------------------------
        # VERIFY pass (causal attention)
        # ----------------------------------------------------------------
        forward_batch.dllm_causal_kv_update = True
        out = self._timed_forward(model_runner, forward_batch, "verify")
        forward_batch.dllm_causal_kv_update = False
        # both mode: remove LoRA after verify (restore base weights).
        if need_swap and self._lora_mode == "both":
            for param, delta, _mod in self._lora_deltas:
                param.data.sub_(delta)
        self._stats_forward_passes += 1
        self._last_decode_forward_passes = [2] * batch_size
        verify_logits = out.logits_output.full_logits  # [B*bs, V]
        logits_output = out.logits_output
        can_run_graph = out.can_run_graph

        # Get AR tokens via argmax (exclude mask_id, in-place)
        verify_logits[:, self.mask_id] = -1e9
        ar_all = torch.argmax(verify_logits, dim=-1)  # [B*bs]

        if self._profile:
            torch.cuda.synchronize()
            _t3 = time.perf_counter()

        # ----------------------------------------------------------------
        # ACCEPT: shifted comparison — draft[i] vs ar[i-1]
        # Both predict position i (draft from bidirectional, ar from causal shift).
        # Output: [seed, ar[0], ar[1], ..., ar[c-1]] = c+1 tokens
        # ----------------------------------------------------------------
        next_token_ids_list: List[torch.Tensor] = []
        accepted_counts: List[int] = []
        natural_accepted_counts: List[int] = []

        for b in range(batch_size):
            b_start = b * bs
            gen_start = start_list[b]
            gen_len = bs - gen_start

            # Vectorized shifted consecutive match: draft[i] vs ar[i-1]
            if gen_len > 1:
                offset = b_start + gen_start
                draft_block = forward_batch.input_ids[offset + 1 : offset + gen_len]
                ar_prev = ar_all[offset : offset + gen_len - 1]
                matches = draft_block == ar_prev
                # cumprod of bools: 1,1,...,1,0,0,...  sum = count of leading matches
                c = int(matches.cumprod(0).sum().item())
            else:
                c = 0

            # Natural acceptance is still computed in forced mode so the
            # benchmark performs the same verification work and can report
            # how the real algorithm would have behaved.
            natural_accepted = c + 1
            natural_accepted_counts.append(natural_accepted)

            seed_pos = b_start + gen_start
            rid = rids[b]
            if self._forced_tpf is not None:
                completed_forwards = self._forced_num_forwards.get(rid, 0)
                emitted_tokens = self._forced_num_tokens.get(rid, 0)
                assert self._forced_tpf_rng is not None
                forced_tokens = _forced_emit_count(
                    self._forced_tpf,
                    gen_len,
                    completed_forwards,
                    emitted_tokens,
                    self._forced_tpf_rng,
                )
                accepted = forced_tokens

                # Commit the verified input prefix. These are exactly the
                # positions whose KV the verify forward just materialized;
                # using arbitrary correction tokens past a natural mismatch
                # would make the retained KV inconsistent with token_ids.
                output_tokens = forward_batch.input_ids[
                    seed_pos : seed_pos + accepted
                ]
            else:
                accepted = natural_accepted
                # Build output: [seed, ar[0], ar[1], ..., ar[c-1]]. At every
                # accepted position ar equals the corresponding draft input.
                seed_tensor = forward_batch.input_ids[seed_pos : seed_pos + 1]
                if c > 0:
                    ar_slice = ar_all[seed_pos : seed_pos + c]
                    output_tokens = torch.cat([seed_tensor, ar_slice])
                else:
                    output_tokens = seed_tensor

            if self._forced_tpf is not None:
                self._forced_num_forwards[rid] = completed_forwards + 2
                self._forced_num_tokens[rid] = emitted_tokens + accepted

            next_token_ids_list.append(output_tokens)
            accepted_counts.append(accepted)

            # The causal logit at the last retained input predicts the next
            # seed, for either natural or forced acceptance.
            next_seed_pos = seed_pos + accepted - 1
            if next_seed_pos < b_start + bs:
                self._seed_tokens[rid] = int(ar_all[next_seed_pos].item())
            else:
                self._seed_tokens[rid] = int(ar_all[b_start + bs - 1].item())

        # ----------------------------------------------------------------
        # Stats
        # ----------------------------------------------------------------
        total_accepted = sum(accepted_counts)
        avg_accepted = total_accepted / max(batch_size, 1)

        if self._stats_file:
            with open(self._stats_file, "a") as _sf:
                for b in range(batch_size):
                    tokens = int(next_token_ids_list[b].shape[0])
                    gen_len = bs - start_list[b]
                    _sf.write(
                        _json.dumps(
                            {
                                "forward_passes": 2,
                                "tokens": tokens,
                                "natural_tokens": natural_accepted_counts[b],
                                "forced_tpf": (
                                    float(self._forced_tpf)
                                    if self._forced_tpf is not None
                                    else None
                                ),
                                "block_gen_positions": gen_len,
                                "acceptance_rate": (
                                    tokens / gen_len if gen_len > 0 else 0
                                ),
                            }
                        )
                        + "\n"
                    )

        logger.debug(
            "LinearSpec block done: fp=%d  avg_accepted=%.1f/%d  total_tokens=%d",
            self._stats_forward_passes,
            avg_accepted,
            bs,
            total_accepted,
        )

        if self._profile:
            _t4 = time.perf_counter()
            self._prof_draft_fwd += _t1 - _t0
            self._prof_between_fwd += _t2 - _t1
            self._prof_verify_fwd += _t3 - _t2
            self._prof_accept += _t4 - _t3
            self._prof_total += _t4 - _t0
            self._prof_n += 1
            if self._prof_n % 500 == 0:
                n = self._prof_n
                sched = self._prof_scheduler / max(n - 1, 1) * 1000
                total_wall = (self._prof_total + self._prof_scheduler) / n * 1000
                fwd_time = (self._prof_draft_fwd + self._prof_verify_fwd) / n * 1000
                logger.info(
                    "LinearSpec PROFILE (%d blocks, per-block avg): "
                    "scheduler=%.2fms  pre_draft=%.2fms  draft_fwd=%.2fms  "
                    "between_fwd=%.2fms  verify_fwd=%.2fms  accept=%.2fms | "
                    "total_wall=%.2fms  fwd_only=%.2fms  overhead=%.1f%%",
                    n,
                    sched,
                    self._prof_pre_draft / n * 1000,
                    self._prof_draft_fwd / n * 1000,
                    self._prof_between_fwd / n * 1000,
                    self._prof_verify_fwd / n * 1000,
                    self._prof_accept / n * 1000,
                    total_wall,
                    fwd_time,
                    (1 - fwd_time / total_wall) * 100 if total_wall > 0 else 0,
                )
            self._prof_last_return = time.perf_counter()

        return logits_output, next_token_ids_list, can_run_graph


Algorithm = LinearSpec

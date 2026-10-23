# Official forced-TPF calibration

The machine-readable source of truth is
`configs/forced_tpf_calibration.json`. Every algorithm entry stores both its
engine-ready TPF and the exact runtime parameters that produced that
calibration. It intentionally omits per-dataset acceptance lengths, reported
averages, and conversion metadata.
Run `python configs/validate_forced_tpf_calibration.py` after editing it.

Before entering an Uno value, manually divide its reported acceptance length
by two because Nano Uno executes two target forwards per decoding step.

## Qwen3-8B

Paper gated-LoRA checkpoint:

```text
/mnt/weka/shrd/k2m/duc.pham/checkpoints/Qwen3-8B-Uno-LoRA
```

The current paper figures use system-optimal Linear-4 and per-user-optimal
Tree-60. The catalog also records the Linear-8 and Tree-32 variants for future
figures. DFlash uses thinking-enabled decoding.

| Temperature | Objective | Method | Configuration | Engine forced TPF |
|---:|---|---|---|---:|
| 0 | System | EAGLE-3 | `B=4` | **2.27** |
| 0 | System | DFlash | `B=4` | **2.18** |
| 0 | System | Uno | Linear `N=4` | **1.975** |
| 0 | Per-user | EAGLE-3 | Tree `B=60` | **3.80** |
| 0 | Per-user | DFlash | `B=16` | **3.09** |
| 0 | Per-user | Uno | Tree `V=60, N=16, K=32` | **3.47** |
| 1 | System | EAGLE-3 | `B=4` | **2.06** |
| 1 | System | DFlash | `B=4` | **2.07** |
| 1 | System | Uno | Linear `N=4` | **1.835** |
| 1 | Per-user | EAGLE-3 | Tree `B=60` | **3.39** |
| 1 | Per-user | DFlash | `B=16` | **2.72** |
| 1 | Per-user | Uno | Tree `V=60, N=16, K=32` | **2.93** |

Additional B8/Tree-32 calibrations use the displayed macro averages across all
14 benchmarks. Uno's engine values are half of its reported acceptance length.

| Temperature | Method | Configuration | Reported average | Engine forced TPF |
|---:|---|---|---:|---:|
| 0 | EAGLE-3 | `B=8` | 2.53 | **2.53** |
| 0 | DFlash | `B=8` | 2.64 | **2.64** |
| 0 | Uno | Linear `N=8` | 5.01 | **2.505** |
| 0 | Uno | Tree `N=16, K=32, V=32` | 6.61 | **3.305** |
| 1 | EAGLE-3 | `B=8` | 2.20 | **2.20** |
| 1 | DFlash | `B=8` | 2.41 | **2.41** |
| 1 | Uno | Linear `N=8` | 4.31 | **2.155** |
| 1 | Uno | Tree `N=16, K=32, V=32` | 5.63 | **2.815** |

Temperature 1 uses top-p 0.95 and top-k 50. Earlier calibration artifacts are
available from the pre-cleanup archive branch.

## BBQ

Paper base-model checkpoint:

```text
/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-8b-mid4_v2/checkpoint_0010000
```

Paper gated-LoRA checkpoint:

```text
/mnt/weka/shrd/k2m/lingjie.chen/diffusion_train_runs/xllm-bbq-b8-tv1-alpha1-tpshard-from-bwd845-64k-8n-1979292-20260731T105758Z/eval_peft_qk_verified/checkpoint_0000620
```

BBQ uses the 131K-context, temperature-1 calibration:

| Configuration | Average / engine forced TPF |
|---|---:|
| Linear `B=4` | 1.8285 |
| Linear `B=8` | 2.0990 |
| Linear `B=16` | 2.1780 |
| Tree `L8/Q16/K16` | 2.4090 |
| Tree `L16/Q32/K32` | 2.6200 |
| Tree `L16/Q64/K64` | 2.7418 |

## Additional diffusion baselines

Missing source-table entries are excluded from each model's average.

| Model | Reported datasets | Average / engine forced TPF |
|---|---:|---:|
| Nemotron Diffusion-14B, LinearSpec LoRA | 14 | **5.807142857** |
| DiffusionGemma-26B | 15 | **16.8** |

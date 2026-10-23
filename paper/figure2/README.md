# Figure 2: diffusion-cycle latency

Figure 2 decomposes one natural-acceptance UNO draft-and-verify cycle into
draft-forward, verify-forward, and exposed non-forward overhead for
Nano-vLLM-UNO at Qwen3-8B, C=1, B=4, 1K input, and 5K output.

The GPU workflow runs three repetitions and reports the median.
Detailed tracing is opt-in and is not enabled by the ordinary paper runners.

Run and plot into an ignored directory:

```bash
scripts/reproduce_figure2.sh runs/figure2 --gpus 0,1,2
```

After reviewing the candidate table and plot, promote it with:

```bash
"$PLOT_PYTHON" paper/promote_run.py runs/figure2 --figures figure2
```

This updates `paper/data/figure2.csv`, its checksum, and this directory's plot
artifacts.

Regenerate the PNG, PDF, and SVG with:

```bash
make figures
```

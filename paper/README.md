# Paper artifacts

This directory contains Figures 1–4. Figures 1, 3, and 4 are generated from
the forced-TPF suite; Figure 2 uses a dedicated detailed-timing workflow.
Plot scripts never require ignored raw-run folders.

```bash
make figures
```

The command validates data checksums and forced-TPF calibration, then rewrites
each figure's PNG, PDF, SVG, and Pareto-frontier CSV files. The reviewed input
tables are under `paper/data/`; GPU workflows are documented in
`BENCHMARKING.md`.

Candidate GPU runs are kept outside `paper/`. After review,
`paper/promote_run.py` promotes selected normalized tables and regenerates
only their official artifacts.

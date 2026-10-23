# Canonical paper data

The CSV files in this directory are the exact reviewed inputs to Figures 1–4.
They are small and versioned so plots do not depend on private checkpoints or
ignored cluster runs. Figure 2 comes from a paired detailed-timing experiment;
the other tables come from the forced-TPF suite.

The tables contain normalized measurements only; raw run paths stay in the
user-selected run directory. `SHA256SUMS` freezes the reviewed inputs and
`paper/validate_data.py` checks their integrity and schemas.

Figures 1, 3, and 4 use the same normalized columns and retain every valid
candidate. Their plot scripts derive the Pareto frontiers automatically.
Figure 2 has a separate latency-breakdown schema.

For the forced-TPF suite, throughput and measured TPF are medians of three
repetitions. `peak_gpu_memory_mib` is the maximum device-level memory sample
across those repetitions, so it includes child processes and speculative
draft-model state rather than only PyTorch allocator-owned tensors.

Use `paper/promote_run.py` after reviewing a candidate run. It replaces only
the selected tables, refreshes `SHA256SUMS`, and regenerates their official
plots.

# Experiment D

Experiment D learns annotator clusters from a fixed offline dataset using 1-D annotator signatures.
The default configuration uses three seeds: `0, 1, 2`.

Run from the workspace root:

```bash
.venv/bin/python3 -m exp_d.run
```

Outputs:

```text
outputs/exp_d/learned_clusters_results.pkl
outputs/exp_d/learned_clusters.png
```

Run the online/offline hard/soft replacement flow and plots:

```bash
.venv/bin/python3 plotting/plot_panel_b.py
```

This reuses the three-seed caches when present and writes:

```text
outputs/exp_d/panel_b_hard_online.png
outputs/exp_d/panel_b_hard_offline.png
outputs/exp_d/panel_b_soft_online.png
outputs/exp_d/panel_b_all_k_sweep.png
```

Run the harder-world stress test from the workspace root:

```bash
.venv/bin/python3 stress_test_d.py
```

Stress-test outputs:

```text
outputs/exp_d/stress_results.pkl
outputs/exp_d/stress_comparison.png
```

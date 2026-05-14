# Experiment D

Experiment D learns annotator clusters from a fixed offline dataset using 1-D annotator signatures.

Run from the workspace root:

```bash
.venv/bin/python3 -m exp_d.run
```

Outputs:

```text
outputs/exp_d/learned_clusters_results.pkl
outputs/exp_d/learned_clusters.png
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

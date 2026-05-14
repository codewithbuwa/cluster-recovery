# Experiment C

Experiment C runs the four ablations from `CPO_new.pdf`.

Shared setup code lives in `scr/`; this folder contains only the Experiment-C-specific ablation orchestration, plotting, summary, and CLI.

Run from the workspace root:

```bash
.venv/bin/python3 -m exp_c.run
```

Outputs:

```text
outputs/exp_c/ablations_results.pkl
outputs/exp_c/ablations.png
```


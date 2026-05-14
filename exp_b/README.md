# Experiment B

Experiment B runs the `pi_A` sweep from `CPO_new.pdf`.

Shared setup code lives in `scr/`; this folder contains only the Experiment-B-specific sweep, plotting, summary, and CLI.

Run from the workspace root:

```bash
.venv/bin/python3 -m exp_b.run
```

Outputs:

```text
outputs/exp_b/pi_sweep_results.pkl
outputs/exp_b/pi_sweep.png
```


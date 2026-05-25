# Experiment B

Experiment B sweeps the annotator-mixture parameter $\pi_A$ to test when CPO has the largest advantage over pooled KTO. The sweep uses the same asymmetric preference thresholds as Experiment A:

```text
tau_A = 0.25, tau_B = 0.75, eps_A = 0.05, eps_B = 0.05
```

and evaluates:

$$
\Delta(\pi_A) = \mathbb{E}[q]_{\mathrm{CPO}} - \mathbb{E}[q]_{\mathrm{KTO}}.
$$

The expected shape is an inverted U: little advantage when the annotator mix is balanced, a larger advantage when one group is moderately dominant, and a smaller advantage again when the minority group becomes too rare.

Default sweep values:

```text
pi_A in {0.50, 0.70, 0.85, 0.95, 0.99}
seeds = 0..49
```

## Run the experiment

Run from the workspace root:

```bash
.venv/bin/python3 -m exp_b.run
```

Outputs:

```text
outputs/exp_b/pi_sweep_results.pkl
outputs/exp_b/pi_sweep.png
```

Use `--skip-plot` to save the pickle without writing the PNG:

```bash
.venv/bin/python3 -m exp_b.run --skip-plot
```

## Render from the pickle

Regenerate the figure from the saved pickle without rerunning training:

```bash
.venv/bin/python3 -m exp_b.pi_sweep
```

This reads `outputs/exp_b/pi_sweep_results.pkl` and writes `outputs/exp_b/pi_sweep.png`.

## Implementation References

- Sweep orchestration: [`experiment.py`](experiment.py), `run_experiment_b`.
- Sweep values and seeds: [`config.py`](config.py), `ExperimentBConfig`.
- Shared training loop: [`../scr/training.py`](../scr/training.py), `train_method`.
- Plotting: [`plotting.py`](plotting.py), `plot_experiment_b`.
- Summary criteria: [`summary.py`](summary.py), `summarize`.
- Render-only helper: [`pi_sweep.py`](pi_sweep.py).

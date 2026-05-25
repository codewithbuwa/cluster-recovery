# Experiment A

Experiment A is the main CPO-vs-KTO mechanism diagnostic. It tests whether cluster-specific CPO keeps learning from the minority annotator group after pooled KTO has effectively stopped updating. The experiment uses the asymmetric world setting:

```text
pi_A = 0.9, tau_A = 0.25, tau_B = 0.75, eps_A = 0.05, eps_B = 0.05
```

The run compares three policies:

- `kto`: pooled KTO reference, with no cluster-specific reference.
- `cpo`: CPO with true annotator clusters.
- `oracle_bob_only`: KTO trained only on Bob-cluster samples, used as a sanity check for the minority-cluster target.

The diagnostic plot shows final quality and gradient-weight traces by cluster. The key mechanism is that KTO's gradient weights collapse for both clusters, while CPO preserves useful weight on Bob-cluster samples.

## Run the experiment

Run from the workspace root:

```bash
.venv/bin/python3 -m exp_a.run
```

Outputs:

```text
outputs/exp_a/grad_weight_results.pkl
outputs/exp_a/grad_weight_diagnostic.png
```

Use `--skip-plot` to save the pickle without writing the PNG:

```bash
.venv/bin/python3 -m exp_a.run --skip-plot
```

## Render from the pickle

Regenerate the figure from the saved pickle without rerunning training:

```bash
.venv/bin/python3 -m exp_a.grad_weight
```

This reads `outputs/exp_a/grad_weight_results.pkl` and writes `outputs/exp_a/grad_weight_diagnostic.png`.

## Implementation References

- Experiment orchestration: [`experiment.py`](experiment.py), `run_experiment_a`.
- Configuration: [`config.py`](config.py), `ExperimentAConfig`.
- Training loop and gradient weights: [`../scr/training.py`](../scr/training.py), `train_method` and `_loss_grad`.
- Plotting: [`plotting.py`](plotting.py), `plot_experiment_a`.
- Summary criteria: [`summary.py`](summary.py), `summarize`.
- Render-only helper: [`grad_weight.py`](grad_weight.py).

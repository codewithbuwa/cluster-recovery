# Experiment C

Experiment C runs four ablations that test which parts of the CPO setup are necessary for the observed gain over KTO. All panels reuse the shared synthetic world and training loop, then vary the reference definition, heterogeneity source, inverse-temperature parameter, or cluster assignment.

The four ablations are:

- `C1 reference variant`: compares the default undesirable-reference variant `r|U` against alternatives `r` and `KL`.
- `C2 noise-only heterogeneity`: removes threshold heterogeneity and leaves mostly noise heterogeneity, where CPO should no longer gain much over KTO.
- `C3 beta sweep`: varies $\beta \in \{0.3, 1.0, 3.0, 10.0\}$ to test sensitivity to the sigmoid sharpness in the loss.
- `C4 misspecified clusters`: replaces true CPO clusters with random clusters; CPO should collapse back toward KTO.

An interactive C3 value-function drawer is available in [`value_function_drawer/`](value_function_drawer/). Open its `index.html` directly in a browser to see how increasing $\beta$ sharpens the curve around zero margin.

The main reported quantity is the final quality gap:

$$
\Delta = \mathbb{E}[q]_{\mathrm{CPO}} - \mathbb{E}[q]_{\mathrm{KTO}}.
$$

## Run the experiment

Run from the workspace root:

```bash
python3 exp_c/run.py
```

Outputs:

```text
outputs/exp_c/ablations_results.pkl
outputs/exp_c/ablations.png
```

Use `--skip-plot` to save the pickle without writing the PNG:

```bash
python3 exp_c/run.py --skip-plot
```

## Render from the pickle

Regenerate the figure from the saved pickle without rerunning training:

```bash
.venv/bin/python3 -m exp_c.ablations
```

This reads `outputs/exp_c/ablations_results.pkl` and writes `outputs/exp_c/ablations.png`.

## Implementation References

- Ablation orchestration: [`experiment.py`](experiment.py), `run_experiment_c`.
- Ablation settings: [`config.py`](config.py), `ExperimentCConfig`.
- Reference variants: [`experiment.py`](experiment.py), `REFERENCE_VARIANTS`; implementation in [`../scr/reference.py`](../scr/reference.py).
- Training loop and $\beta$: [`../scr/training.py`](../scr/training.py), `train_method` and `_loss_grad`.
- Random-cluster ablation: [`../scr/training.py`](../scr/training.py), `cluster_mode="random"`.
- Plotting: [`plotting.py`](plotting.py), `plot_experiment_c`.
- Summary criteria: [`summary.py`](summary.py), `summarize`.
- Render-only helper: [`ablations.py`](ablations.py).

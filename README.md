# CPO

This repository contains the code and figures for the cluster-referenced preference optimization experiments from `CPO_new.pdf`.

The shared training code also supports the mixed unary + pairwise regime described in
`documentation/CPO_part2.pdf`. That extension reuses the unary CPO/KTO flow and adds a
true DPO-style pairwise stream with label-equivalent budget accounting.

The shared implementation lives in `scr/`, and the experiment-specific code lives in:

- `exp_a/` - main CPO vs KTO mechanism diagnostic
- `exp_b/` - `pi_A` sweep
- `exp_c/` - ablations
- `exp_d/` - cluster recovery and streaming clustering diagnostics
- `exp_e/` - mixed unary + pairwise follow-up experiments

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Run the experiments from the repository root:

```bash
python3 exp_a/run.py
python3 exp_b/run.py
python3 exp_c/run.py
python3 exp_d/run.py
python3 exp_e/run.py
```

Each run recomputes the experiment and saves a pickle cache plus PNG figures under `outputs/`.

For render-only helpers, use:

```bash
.venv/bin/python3 -m exp_a.grad_weight
.venv/bin/python3 -m exp_b.pi_sweep
.venv/bin/python3 -m exp_c.ablations
.venv/bin/python3 -m exp_d.cluster_recovery
```

## Mixed Unary + Pairwise Training

The mixed-regime support is controlled through `scr.config.TrainConfig` and
`scr.training.train_method`.

Relevant configuration fields:

- `alpha` - loss mixing weight. `0.0` is unary-only; `1.0` is pairwise-only DPO.
- `pair_fraction` - fraction of label-equivalent effort spent on pairwise samples.
- `total_effort` - total label-equivalent budget per training step. If unset, the existing
  `batch_size` behavior is preserved for unary-only experiments.
- `pair_noise` - probability of flipping a pairwise winner/loser label.

Pairwise labels are sampled by `SyntheticWorld.sample_pair_batch`, which returns
`(x, y_winner, y_loser)` from the latent quality table. Unary labels remain the only source used
to update the CPO/KTO reference points `z` or `z_k`.

For the `CPO_part2.pdf` Experiment 1 budget sweep, use
`scr.training.valid_budget_sweep_cell` to skip method-budget combinations whose required data
stream is empty.

Example modes:

```python
from scr.config import TrainConfig
from scr.training import valid_budget_sweep_cell

# KTO or CPO, unary-only. This preserves the original A-D behavior.
unary_only = TrainConfig(alpha=0.0, pair_fraction=0.0)

# Mixed CPO with 256 label-equivalent samples per step:
# 128 unary labels and 64 pairwise labels.
mixed_cpo = TrainConfig(alpha=0.5, pair_fraction=0.5, total_effort=256)

# Pure DPO with all effort spent on pairwise labels.
dpo = TrainConfig(alpha=1.0, pair_fraction=1.0, total_effort=256)

assert valid_budget_sweep_cell("cpo", mixed_cpo)
```

## Outputs

The default outputs are:

- `outputs/exp_a/grad_weight_results.pkl`
- `outputs/exp_a/grad_weight_diagnostic.png`
- `outputs/exp_b/pi_sweep_results.pkl`
- `outputs/exp_b/pi_sweep.png`
- `outputs/exp_c/ablations_results.pkl`
- `outputs/exp_c/ablations.png`
- `outputs/exp_d/cluster_recovery_results.pkl`
- `outputs/exp_d/learned_clusters.png`
- `outputs/exp_d/panel_b.png`
- `outputs/exp_d/panel_b_hard_online.png`
- `outputs/exp_d/panel_b_hard_offline.png`
- `outputs/exp_d/panel_b_soft_online.png`
- `outputs/exp_d/panel_b_all_k_sweep.png`
- `outputs/exp_e/budget_sweep_results.pkl`
- `outputs/exp_e/budget_sweep.png`
- `outputs/exp_e/budget_sweep_deltas.png`
- `outputs/exp_e/alpha_sweep_results.pkl`
- `outputs/exp_e/alpha_sweep.png`
- `outputs/exp_e/alpha_pair_sweep_results.pkl`
- `outputs/exp_e/alpha_pair_sweep.png`
- `outputs/exp_e/pia_sweep_results.pkl`
- `outputs/exp_e/pia_sweep.png`
- `outputs/exp_e/ref_ablation_results.pkl`
- `outputs/exp_e/ref_ablation.png`

## Experiment Notes

Each experiment has its own README with the setup, outputs, and implementation references:

- [Experiment A](exp_a/README.md)
- [Experiment B](exp_b/README.md)
- [Experiment C](exp_c/README.md)
- [Experiment D](exp_d/README.md)
- [Experiment E](exp_e/README.md)

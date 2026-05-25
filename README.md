# CPO

This repository contains the code and figures for the cluster-referenced preference optimization experiments from `CPO_new.pdf`.

The shared implementation lives in `scr/`, and the experiment-specific code lives in:

- `exp_a/` - main CPO vs KTO mechanism diagnostic
- `exp_b/` - `pi_A` sweep
- `exp_c/` - ablations
- `exp_d/` - cluster recovery and streaming clustering diagnostics

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Run the experiments from the repository root:

```bash
.venv/bin/python3 -m exp_a.run
.venv/bin/python3 -m exp_b.run
.venv/bin/python3 -m exp_c.run
.venv/bin/python3 -m exp_d.run
```

Each experiment saves a pickle cache and a PNG figure under `outputs/`.

For render-only helpers, use:

```bash
.venv/bin/python3 -m exp_a.grad_weight
.venv/bin/python3 -m exp_b.pi_sweep
.venv/bin/python3 -m exp_c.ablations
.venv/bin/python3 -m exp_d.cluster_recovery
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

## Experiment Notes

Each experiment has its own README with the setup, outputs, and implementation references:

- [Experiment A](exp_a/README.md)
- [Experiment B](exp_b/README.md)
- [Experiment C](exp_c/README.md)
- [Experiment D](exp_d/README.md)


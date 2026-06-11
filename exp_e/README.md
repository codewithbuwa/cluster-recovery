# Experiment E

Experiment E implements the mixed unary + pairwise follow-up experiments from
`documentation/CPO_part2.pdf`.

The budget sweep uses:

```text
f in {0, 0.125, 0.25, 0.5, 0.75, 1.0}
```

N/A filtering skips only configurations whose required data stream is empty:

```text
f=0:     KTO, CPO
0<f<1:   KTO, CPO, mixed CPO, DPO
f=1:     DPO
```

The budget sweep writes both the absolute-quality figure and an additional delta figure:

```text
outputs/exp_e/budget_sweep.png
outputs/exp_e/budget_sweep_deltas.png
```

Run individual panels with:

```bash
python3 exp_e/run.py --budget-sweep
python3 exp_e/run.py --alpha-sweep
python3 exp_e/run.py --alpha-pair-sweep
python3 exp_e/run.py --pia-sweep
python3 exp_e/run.py --ref-ablation
```

If no panel flag is passed, all Experiment E panels run. Use `--skip-plot` to write only caches
and text summaries.

## Follow-up C: reference and mixing ablation

The primary ablation uses the same nominal design in all four cells:

```text
N_unary=128
N_pair=64
alpha in {0, 0.5}
reference in {global, per-cluster}
```

When `alpha=0`, training does not consume pair batches because the pairwise loss is disabled; the
declared data budget is nevertheless unchanged across the primary cells. A secondary mixed
control repeats the global/per-cluster comparison at `N_pair=8` and `alpha=0.5`.

```text
outputs/exp_e/ref_ablation.png
outputs/exp_e/ref_ablation_pair8.png
outputs/exp_e/ref_ablation_summary.txt
outputs/exp_e/ref_ablation_results.pkl
```

## Follow-up B: optimal alpha by pair budget

The alpha-pair sweep holds the unary stream fixed at `N_unary=128` labels per step, varies
`N_pair` over `{4, 16, 64, 256, 1024}`, and evaluates alpha over
`{0, 0.1, 0.25, 0.5, 0.75, 0.9, 1}`. This sweep intentionally does not hold total labeller
effort constant.

Unary and pair losses are each normalized by their own sample count. Increasing `N_pair` therefore
reduces pair-gradient variance rather than mechanically increasing its loss scale, so movement in
alpha-star measures how much weight the optimizer assigns to increasingly precise pairwise evidence.

The summary reports the empirical

```text
alpha*(N_pair) = argmax_alpha mean_seed final E[q]
```

and checks whether the selected alpha-star values are nondecreasing. The left figure panel shows
one `E[q]`-versus-alpha curve per pair budget. The right panel shows alpha-star against `N_pair`
on a logarithmic x-axis:

```text
outputs/exp_e/alpha_pair_sweep.png
outputs/exp_e/alpha_pair_sweep_summary.txt
outputs/exp_e/alpha_pair_sweep_results.pkl
```

Each cached `TrainResult` includes `reference_values`, the reference-point trajectory used during
training. Rows correspond to optimization steps and columns correspond to reference clusters:

```text
KTO:                 (steps, 1), pooled z
CPO / mixed CPO:     (steps, 2), z_A and z_B
DPO:                 (steps, 1), unchanged zeros because DPO has no unary reference update
```

The value in each row is the `z_k` value used at the start of that training step. Existing pickle
caches must be regenerated to include these trajectories.

The cache also stores diagnostics without assigning prompts or pairs to annotator clusters:

```text
expected_quality_per_prompt:          (evals, n_prompts)
expected_desirability_by_cluster:     (evals, n_annotator_clusters)
n_unary_per_cluster_seen:             (steps, n_annotator_clusters)
n_pair_seen:                          (steps,)
```

`expected_desirability_by_cluster` evaluates the same policy and prompts under each cluster's
threshold and label-noise model. Pairwise observations have no annotator-cluster identity, so only
their cumulative total is recorded.

# Experiment E

Experiment E implements the mixed unary + pairwise follow-up experiments from
`documentation/CPO_part2.pdf`.

The budget sweep uses:

```text
f in {0, 0.125, 0.25, 0.5, 0.75, 1.0}
seeds = {0, 1}
```

N/A filtering skips only configurations whose required data stream is empty. This leaves nineteen
valid method-budget cells, so the budget sweep runs thirty-eight training jobs:

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
python3 exp_e/run.py --pia-sweep
python3 exp_e/run.py --ref-ablation
```

If no panel flag is passed, all Experiment E panels run. Use `--skip-plot` to write only caches
and text summaries.

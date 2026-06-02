# Experiment E

Experiment E implements the mixed unary + pairwise follow-up experiments from
`documentation/CPO_part2.pdf`.

Run individual panels with:

```bash
python3 exp_e/run.py --budget-sweep
python3 exp_e/run.py --alpha-sweep
python3 exp_e/run.py --pia-sweep
python3 exp_e/run.py --ref-ablation
```

If no panel flag is passed, all Experiment E panels run. Use `--skip-plot` to write only caches
and text summaries.

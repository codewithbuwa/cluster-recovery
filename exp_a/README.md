# Experiment A

Experiment A is the main CPO-vs-KTO comparison with the gradient-weight diagnostic from `CPO_new.pdf`.

Shared setup code lives in the workspace-level `scr/` package:

```text
scr/config.py
scr/world.py
scr/policy.py
scr/training.py
```

Experiment-A-specific code lives here:

```text
exp_a/config.py
exp_a/experiment.py
exp_a/plotting.py
exp_a/summary.py
exp_a/run.py
```

Run from the repository root:

```bash
.venv/bin/python3 -m exp_a.run
```

Or run the script path directly:

```bash
.venv/bin/python3 exp_a/run.py
```

Outputs are written by default to:

```text
outputs/experiment_a.png
outputs/experiment_a_results.pkl
outputs/experiment_a_summary.txt
```

Use a separate output directory without touching the current accepted output:

```bash
.venv/bin/python3 -m exp_a.run --output-dir outputs_exp_a_check
```

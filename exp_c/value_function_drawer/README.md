# C3 Value Function Drawer

This folder contains a self-contained interactive HTML drawer for Experiment C3.

Open `index.html` in a browser to explore how the C3 beta sweep changes:

- `V_beta(m) = sigmoid(beta * m)`
- `log(1 + exp(-beta * m))`
- `V_beta(m) * (1 - V_beta(m))`

The C3 presets match `ExperimentCConfig.beta_values`: `0.3`, `1.0`, `3.0`, and `10.0`.

No build step or package install is required.

# Experiment D

Experiment D tests whether CPO can recover useful annotator clusters instead of relying on random or oracle cluster assignments. It builds 1-D annotator signatures from preference data, clusters annotators with hard or soft methods, and compares the resulting CPO policies against KTO, random CPO clusters, and oracle CPO clusters.

The default configuration uses three seeds: `0, 1, 2`. The run produces one bundled pickle, `outputs/exp_d/cluster_recovery_results.pkl`, which is the only cache needed to regenerate every Experiment D figure.

The bundle contains three payloads:

- `learned_clusters`: offline learned-cluster experiment with KTO, random, learned, oracle, and K-sweep results.
- `online_flows`: streaming hard/soft refit diagnostics used for `panel_b.png`.
- `online_variant_panel`: online/offline hard/soft replacement results used for the replacement panels and all-method K-sweep.

## Clustering Setup

The offline learned-cluster baseline first samples a fixed preference dataset, computes each annotator's 1-D signature from their observed desirable-label rate, and clusters those signatures before policy training. This tests whether a simple learned partition can recover the latent annotator groups well enough for CPO to match the oracle cluster assignment.

The online streaming variants update annotator signatures during training instead of fitting clusters once up front. As new labeled preferences arrive, each annotator's running desirable-label rate is updated. At fixed refit steps, the current signatures are reclustered and the CPO reference assignment is refreshed. This creates a stricter setting: clustering must recover useful groups from partial data while the policy is already learning.

Hard online clustering uses 1-D K-means over the current streaming signatures. Each annotator is assigned to exactly one cluster, and CPO uses that discrete cluster id for its reference point. The implementation keeps cluster labels stable across refits by matching new centers to previous centers, so a cluster id does not arbitrarily flip between refit steps.

Soft online clustering uses a 1-D Gaussian mixture model over the same streaming signatures. Instead of assigning each annotator to a single cluster, it maintains responsibilities over clusters. Training can then use fractional cluster membership, and recovery is evaluated with soft NMI and soft purity. This is useful when annotator groups are not cleanly separated early in the stream.

Hard offline clustering is included as a comparison point for the online variants. It uses the same hard K-means style assignment, but fits on the fixed offline signatures rather than progressively updating from the stream. The replacement panels compare hard online, hard offline, and soft online against the same KTO, random, and oracle baselines.

## Some Definitions

Let annotator $a$ provide preference labels $d_i \in \{0, 1\}$, where $d_i = 1$ means the sampled response was labeled desirable. The 1-D annotator signature is the empirical desirable rate:

$$
s_a = \frac{\sum_{i:\mathrm{annotator}(i)=a} d_i}{n_a}
$$

For online streaming, the signature is updated from the prefix of labels observed so far:

$$
s_a(t) = \frac{D_a(t)}{N_a(t)}
$$

where $D_a(t)$ is the number of desirable labels from annotator $a$ seen by step $t$, and $N_a(t)$ is the number of labels from annotator $a$ seen by step $t$. At refit steps, the current vector $s(t)$ is reclustered.

Hard clustering solves a 1-D K-means objective:

$$
\min_{c,\mu}\sum_a \left(s_a - \mu_{c_a}\right)^2
$$

where each annotator gets one cluster id $c_a \in \{1,\ldots,K\}$. CPO then uses $c_a$ to select the annotator-specific reference value for each preference update.

Soft clustering fits a 1-D Gaussian mixture:

$$
p(s_a) = \sum_k \pi_k \mathcal{N}\left(s_a \mid \mu_k, \sigma_k^2\right)
$$

and uses posterior responsibilities:

$$
r_{ak} = p(k \mid s_a)
$$

Instead of a single hard cluster id, the annotator has fractional membership over clusters. The soft online flow evaluates recovery using the soft contingency induced by $r_{ak}$ rather than first forcing a hard argmax assignment.

Cluster recovery is measured with NMI and purity. For hard labels, the contingency table is:

$$
C_{kz} = \left|\{a : c_a = k \ \mathrm{and}\ z_a = z\}\right|
$$

For soft labels, the corresponding soft contingency is:

$$
C_{kz} = \sum_{a:z_a=z} r_{ak}
$$

Purity measures how much cluster mass lands on the best matching true group:

$$
\mathrm{purity} = \frac{1}{A}\sum_k \max_z C_{kz}
$$

NMI normalizes mutual information between predicted clusters and true latent annotator groups, so high NMI and high purity indicate that the learned signatures recover the ground-truth annotator structure.

Implementation references:

- Signature $s_a$: `compute_signatures` in [`clustering.py`](clustering.py).
- Streaming signature $s_a(t)$: `StreamingSignatures` in [`clustering_online.py`](clustering_online.py).
- Hard K-means assignment $c_a$: `kmeans_1d` in [`clustering.py`](clustering.py) and `OnlineHardClusterer` in [`clustering_online.py`](clustering_online.py).
- Soft GMM responsibilities $r_{ak}$: `gmm_1d_soft` in [`clustering_soft.py`](clustering_soft.py) and `OnlineSoftClusterer` in [`clustering_online.py`](clustering_online.py).
- Hard and soft CPO references: `_HardReference` and `_SoftReference` in [`../scr/training_online.py`](../scr/training_online.py).
- NMI and purity: `normalized_mutual_information`, `purity`, `soft_normalized_mutual_information`, and `soft_purity` in [`clustering.py`](clustering.py) and [`clustering_soft.py`](clustering_soft.py).

## Run the experiment

Run from the workspace root to build the single bundled cache and render every Experiment D figure:

```bash
.venv/bin/python3 -m exp_d.run
```

The bundled pickle is the only Exp D cache needed for the figures. It contains the learned-cluster experiment, the online hard/soft flow, and the online/offline variant panel payloads.

Outputs:

```text
outputs/exp_d/cluster_recovery_results.pkl
outputs/exp_d/learned_clusters.png
outputs/exp_d/panel_b.png
outputs/exp_d/panel_b_hard_online.png
outputs/exp_d/panel_b_hard_offline.png
outputs/exp_d/panel_b_soft_online.png
outputs/exp_d/panel_b_all_k_sweep.png
```

Figure meanings:

- `learned_clusters.png`: main learned-cluster comparison, cluster recovery scores, and learned-vs-random K-sweep.
- `panel_b.png`: streaming hard vs soft clustering recovery over refit steps, with offline K-means shown as a reference line.
- `panel_b_hard_online.png`: replaces learned clustering with hard online clustering in the main comparison.
- `panel_b_hard_offline.png`: replaces learned clustering with hard offline clustering.
- `panel_b_soft_online.png`: replaces learned clustering with soft online clustering.
- `panel_b_all_k_sweep.png`: compares random, learned, hard-online, hard-offline, and soft-online clustering across K.

Use `--skip-plot` to save or reuse the bundled pickle without writing PNGs:

```bash
.venv/bin/python3 -m exp_d.run --skip-plot
```

Use `--recompute` to ignore an existing bundle and rerun the experiment:

```bash
.venv/bin/python3 -m exp_d.run --recompute
```

## Render from the bundle

Regenerate all Experiment D plots from the existing bundled pickle without rerunning training:

```bash
.venv/bin/python3 -m exp_d.cluster_recovery
```

This reads:

```text
outputs/exp_d/cluster_recovery_results.pkl
```

and writes:

```text
outputs/exp_d/learned_clusters.png
outputs/exp_d/panel_b.png
outputs/exp_d/panel_b_hard_online.png
outputs/exp_d/panel_b_hard_offline.png
outputs/exp_d/panel_b_soft_online.png
outputs/exp_d/panel_b_all_k_sweep.png
```

import numpy as np


def _values(runs: list | None) -> np.ndarray | None:
    if runs is None:
        return None
    return np.asarray([run.expected_quality[-1] for run in runs])


def _mean_std(runs: list | None) -> str:
    values = _values(runs)
    if values is None:
        return "N/A"
    return f"{values.mean():.4f} +/- {values.std():.4f}"


def summarize_budget_sweep(payload: dict[str, object]) -> str:
    lines = ["Experiment E budget sweep", ""]
    for pair_fraction in payload["pair_fraction_values"]:
        row = payload["results"][pair_fraction]
        lines.append(f"f={pair_fraction:.2f}")
        for method in ("kto", "cpo", "mixed_cpo", "dpo"):
            lines.append(f"    {method}: {_mean_std(row[method])}")
    return "\n".join(lines) + "\n"


def summarize_alpha_sweep(payload: dict[str, object]) -> str:
    lines = ["Experiment E alpha sweep", ""]
    best_alpha = None
    best_value = -np.inf
    for alpha in payload["alpha_values"]:
        values = _values(payload["results"][alpha])
        mean_value = float(values.mean())
        if mean_value > best_value:
            best_alpha = alpha
            best_value = mean_value
        lines.append(f"alpha={alpha:.2f}: {_mean_std(payload['results'][alpha])}")
    lines.extend(["", f"Best alpha: {best_alpha:.2f} with E[q]={best_value:.4f}"])
    return "\n".join(lines) + "\n"


def summarize_alpha_pair_sweep(payload: dict[str, object]) -> str:
    lines = [
        "Experiment E alpha* by pair budget",
        "",
        f"Fixed unary labels per step: {payload['fixed_n_unary']}",
        "",
    ]
    best_alphas = []
    for n_pair in payload["pair_budget_values"]:
        best_alpha = None
        best_value = -np.inf
        lines.append(f"N_pair={n_pair}")
        for alpha in payload["alpha_values"]:
            runs = payload["results"][n_pair][alpha]
            lines.append(f"    alpha={alpha:.2f}: {_mean_std(runs)}")
            values = _values(runs)
            if values is not None and values.mean() > best_value:
                best_alpha = alpha
                best_value = float(values.mean())
        best_alphas.append(best_alpha)
        lines.append(f"    alpha*={best_alpha:.2f}, E[q]={best_value:.4f}")

    comparable_alphas = np.asarray([alpha for alpha in best_alphas if alpha is not None])
    nondecreasing = bool(np.all(np.diff(comparable_alphas) >= 0.0))
    lines.extend(
        [
            "",
            "alpha* curve: "
            + ", ".join(
                f"N_pair={n_pair}: {alpha:.2f}"
                for n_pair, alpha in zip(payload["pair_budget_values"], best_alphas)
            ),
            f"Nondecreasing alpha*: {nondecreasing}",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_pia_sweep(payload: dict[str, object]) -> str:
    lines = ["Experiment E pi_A sweep", ""]
    for pi_a in payload["pi_a_values"]:
        row = payload["results"][pi_a]
        lines.append(f"pi_A={pi_a:.2f}")
        kto = _values(row["kto"])
        for method in ("kto", "cpo", "mixed_cpo", "dpo"):
            values = _values(row[method])
            delta = "" if method == "kto" else f", delta_vs_kto={values.mean() - kto.mean():.4f}"
            lines.append(f"    {method}: {values.mean():.4f} +/- {values.std():.4f}{delta}")
    return "\n".join(lines) + "\n"


def summarize_ref_ablation(payload: dict[str, object]) -> str:
    results = payload["results"]
    secondary = payload["secondary_results"]
    cluster_gain_unary = _values(results["cluster_alpha0"]).mean() - _values(results["global_alpha0"]).mean()
    cluster_gain_mixed = _values(results["cluster_alpha05"]).mean() - _values(results["global_alpha05"]).mean()
    cluster_gain_mixed_pair8 = (
        _values(secondary["cluster_alpha05"]).mean()
        - _values(secondary["global_alpha05"]).mean()
    )
    mixing_gain_global = _values(results["global_alpha05"]).mean() - _values(results["global_alpha0"]).mean()
    mixing_gain_cluster = _values(results["cluster_alpha05"]).mean() - _values(results["cluster_alpha0"]).mean()
    lines = [
        "Experiment E reference/mixing ablation",
        "",
        f"global z, alpha=0:      {_mean_std(results['global_alpha0'])}",
        f"cluster z_k, alpha=0:   {_mean_std(results['cluster_alpha0'])}",
        f"global z, alpha=0.5:    {_mean_std(results['global_alpha05'])}",
        f"cluster z_k, alpha=0.5: {_mean_std(results['cluster_alpha05'])}",
        "",
        f"Cluster gain, unary:  {cluster_gain_unary:.4f}",
        f"Cluster gain, mixed:  {cluster_gain_mixed:.4f}",
        f"Mixing gain, global:  {mixing_gain_global:.4f}",
        f"Mixing gain, cluster: {mixing_gain_cluster:.4f}",
        "",
        f"Secondary check, N_pair={payload['secondary_n_pair']}, alpha=0.5",
        f"global z:             {_mean_std(secondary['global_alpha05'])}",
        f"cluster z_k:          {_mean_std(secondary['cluster_alpha05'])}",
        f"Cluster gain, mixed:  {cluster_gain_mixed_pair8:.4f}",
        "",
        "Primary nominal design: "
        f"N_unary={payload['nominal_primary_counts']['n_unary']}, "
        f"N_pair={payload['nominal_primary_counts']['n_pair']}. "
        "At alpha=0, the zero-weight pair batch is not sampled.",
    ]
    return "\n".join(lines) + "\n"


def summarize(payload: dict[str, object]) -> str:
    name = payload["name"]
    if name == "budget_sweep":
        return summarize_budget_sweep(payload)
    if name == "alpha_sweep":
        return summarize_alpha_sweep(payload)
    if name == "alpha_pair_sweep":
        return summarize_alpha_pair_sweep(payload)
    if name == "pia_sweep":
        return summarize_pia_sweep(payload)
    if name == "ref_ablation":
        return summarize_ref_ablation(payload)
    raise ValueError(f"unknown Experiment E payload: {name}")

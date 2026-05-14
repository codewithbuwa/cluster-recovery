import numpy as np


def method_values(results: dict[str, list]) -> dict[str, np.ndarray]:
    return {
        method: np.asarray([result.expected_quality[-1] for result in runs])
        for method, runs in results.items()
    }


def gap(results: dict[str, list]) -> float:
    values = method_values(results)
    return float(values["cpo"].mean() - values["kto"].mean())


def summarize(payload: dict[str, object]) -> str:
    results = payload["results"]
    c1_gaps = {variant: gap(variant_results) for variant, variant_results in results["c1"].items()}
    c2_gap = gap(results["c2"])
    c3_gaps = {beta: gap(beta_results) for beta, beta_results in results["c3"].items()}
    c4_gap = gap(results["c4"])
    c4_values = method_values(results["c4"])
    c4_tolerance = max(c4_values["kto"].std(), c4_values["cpo"].std()) + 0.03

    c1_default_ok = c1_gaps["r|U"] >= 0.15
    c1_collapse_ok = c1_gaps["r"] < 0.03 and c1_gaps["KL"] < 0.03
    c2_ok = c2_gap < 0.03
    c3_ok = c3_gaps[10.0] > c3_gaps[1.0] > c3_gaps[0.3]
    c4_ok = abs(c4_gap) < c4_tolerance

    lines = [
        "Success criteria",
        "",
        "[1] C1 reference variant",
        "    Requirement A: r|U reference gives CPO - KTO >= 0.15",
        f"    Values:        r|U gap={c1_gaps['r|U']:.4f}",
        f"    Result:        {'PASS' if c1_default_ok else 'FAIL'}",
        "",
        "    Requirement B: r and KL references collapse with gaps < 0.03",
        f"    Values:        r gap={c1_gaps['r']:.4f}, KL gap={c1_gaps['KL']:.4f}",
        f"    Result:        {'PASS' if c1_collapse_ok else 'FAIL'}",
        "",
        "[2] C2 noise-only heterogeneity",
        "    Requirement: CPO - KTO gap < 0.03",
        f"    Values:      gap={c2_gap:.4f}",
        f"    Result:      {'PASS' if c2_ok else 'FAIL'}",
        "",
        "[3] C3 beta sweep",
        "    Requirement: delta(beta=10) > delta(beta=1) > delta(beta=0.3)",
        (
            "    Values:      "
            f"delta(0.3)={c3_gaps[0.3]:.4f}, "
            f"delta(1)={c3_gaps[1.0]:.4f}, "
            f"delta(10)={c3_gaps[10.0]:.4f}"
        ),
        f"    Result:      {'PASS' if c3_ok else 'FAIL'}",
        "",
        "[4] C4 misspecified clusters",
        "    Requirement: random-cluster CPO and KTO bars overlap within tolerance",
        f"    Values:      gap={c4_gap:.4f}, tolerance={c4_tolerance:.4f}",
        f"    Result:      {'PASS' if c4_ok else 'FAIL'}",
        "",
        "Overall",
        f"    C1 default criterion:  {'PASS' if c1_default_ok else 'FAIL'}",
        f"    C1 collapse criterion: {'PASS' if c1_collapse_ok else 'FAIL'}",
        f"    C2 criterion:          {'PASS' if c2_ok else 'FAIL'}",
        f"    C3 criterion:          {'PASS' if c3_ok else 'FAIL'}",
        f"    C4 criterion:          {'PASS' if c4_ok else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"

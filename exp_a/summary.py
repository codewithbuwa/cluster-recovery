import numpy as np


def summarize(payload: dict[str, object]) -> str:
    results = payload["results"]
    diagnostics = payload["diagnostics"]

    finals = {
        name: np.asarray([r.expected_quality[-1] for r in method_results])
        for name, method_results in results.items()
    }
    cpo_gap = float(finals["cpo"].mean() - finals["kto"].mean())
    oracle_gap = float(finals["cpo"].mean() - finals["oracle_bob_only"].mean())

    kto_tail = np.nanmean(diagnostics["kto_grad_weight_by_cluster"][:, 300:, :], axis=(0, 1))
    cpo_tail = np.nanmean(diagnostics["cpo_grad_weight_by_cluster"][:, 300:, :], axis=(0, 1))

    kto_mechanism_ok = bool(kto_tail[0] < 0.02 and kto_tail[1] < 0.02)
    cpo_mechanism_ok = bool(cpo_tail[1] > 0.02)
    quality_ok = cpo_gap >= 0.15
    oracle_ok = oracle_gap > 0.0
    mechanism_ok = bool(kto_mechanism_ok and cpo_mechanism_ok)

    lines = [
        "Success criteria",
        "",
        "[1] Quality gap",
        "    Requirement: E[q]_CPO - E[q]_KTO >= 0.15",
        (
            "    Values:      "
            f"{finals['cpo'].mean():.4f} - {finals['kto'].mean():.4f} = {cpo_gap:.4f}"
        ),
        f"    Result:      {'PASS' if quality_ok else 'FAIL'}",
        "",
        "[2] Oracle beat",
        "    Requirement: E[q]_CPO > E[q]_Oracle-Bob-only",
        (
            "    Values:      "
            f"{finals['cpo'].mean():.4f} > {finals['oracle_bob_only'].mean():.4f}"
        ),
        f"    Result:      {'PASS' if oracle_ok else 'FAIL'}",
        "",
        "[3] Mechanism visible",
        "    Requirement A: KTO Alice and Bob gradient weights < 0.02 near the end",
        f"    Values:        KTO [Alice, Bob] = [{kto_tail[0]:.4f}, {kto_tail[1]:.4f}]",
        f"    Result:        {'PASS' if kto_mechanism_ok else 'FAIL'}",
        "",
        "    Requirement B: CPO Bob gradient weight > 0.02 over the last 25%",
        f"    Values:        CPO [Alice, Bob] = [{cpo_tail[0]:.4f}, {cpo_tail[1]:.4f}]",
        f"    Result:        {'PASS' if cpo_mechanism_ok else 'FAIL'}",
        "",
        "Overall",
        f"    Quality criterion:   {'PASS' if quality_ok else 'FAIL'}",
        f"    Oracle criterion:    {'PASS' if oracle_ok else 'FAIL'}",
        f"    Mechanism criterion: {'PASS' if mechanism_ok else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"

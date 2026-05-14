import numpy as np


def final_values(payload: dict[str, object]) -> dict[float, dict[str, np.ndarray]]:
    results = payload["results"]
    values = {}
    for pi_a, method_results in results.items():
        values[pi_a] = {
            method: np.asarray([result.expected_quality[-1] for result in runs])
            for method, runs in method_results.items()
        }
    return values


def summarize(payload: dict[str, object]) -> str:
    values = final_values(payload)
    pi_a_values = list(payload["pi_a_values"])
    deltas = {
        pi_a: float(values[pi_a]["cpo"].mean() - values[pi_a]["kto"].mean())
        for pi_a in pi_a_values
    }
    peak_pi_a = max(deltas, key=deltas.get)
    max_delta = deltas[peak_pi_a]

    inverted_u_ok = peak_pi_a in {0.85, 0.95}
    magnitude_ok = max_delta >= 0.08
    endpoint_ok = deltas[0.5] < 0.04 and deltas[0.99] < 0.04

    values_by_pi = []
    for pi_a in pi_a_values:
        kto = values[pi_a]["kto"]
        cpo = values[pi_a]["cpo"]
        values_by_pi.append(
            f"pi_A={pi_a:.2f}: KTO={kto.mean():.4f}, CPO={cpo.mean():.4f}, delta={deltas[pi_a]:.4f}"
        )

    lines = [
        "Success criteria",
        "",
        "[1] Inverted-U shape",
        "    Requirement: max delta occurs at pi_A in {0.85, 0.95}",
        f"    Values:      peak pi_A={peak_pi_a:.2f}, max delta={max_delta:.4f}",
        f"    Result:      {'PASS' if inverted_u_ok else 'FAIL'}",
        "",
        "[2] Magnitude",
        "    Requirement: max_pi_A delta(pi_A) >= 0.08",
        f"    Values:      max delta={max_delta:.4f}",
        f"    Result:      {'PASS' if magnitude_ok else 'FAIL'}",
        "",
        "[3] Endpoints",
        "    Requirement: delta(0.50) < 0.04 and delta(0.99) < 0.04",
        f"    Values:      delta(0.50)={deltas[0.5]:.4f}, delta(0.99)={deltas[0.99]:.4f}",
        f"    Result:      {'PASS' if endpoint_ok else 'FAIL'}",
        "",
        "Overall",
        f"    Inverted-U criterion: {'PASS' if inverted_u_ok else 'FAIL'}",
        f"    Magnitude criterion:  {'PASS' if magnitude_ok else 'FAIL'}",
        f"    Endpoint criterion:   {'PASS' if endpoint_ok else 'FAIL'}",
        "",
        "Final values",
        *[f"    {line}" for line in values_by_pi],
    ]
    return "\n".join(lines) + "\n"

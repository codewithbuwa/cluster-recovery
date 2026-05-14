import numpy as np


def final_values(results: dict[str, list]) -> dict[str, np.ndarray]:
    return {
        method: np.asarray([result.expected_quality[-1] for result in runs])
        for method, runs in results.items()
    }


def summarize(payload: dict[str, object]) -> str:
    values = final_values(payload["results"])
    kto = values["kto"].mean()
    random = values["cpo_random"].mean()
    learned = values["cpo_learned"].mean()
    oracle = values["cpo_oracle"].mean()
    clustering = payload["clustering"]
    ksweep = payload["ksweep"]

    learned_oracle_gap = abs(learned - oracle)
    learned_random_gap = learned - random
    random_kto_gap = abs(random - kto)
    learned_nmi = float(clustering["learned"]["nmi"].mean())
    learned_purity = float(clustering["learned"]["purity"].mean())
    k2_value = np.asarray([run.expected_quality[-1] for run in ksweep[2]["learned"]]).mean()
    overcluster_values = {
        k: np.asarray([run.expected_quality[-1] for run in ksweep[k]["learned"]]).mean()
        for k in (3, 5, 10)
    }
    k1_learned = np.asarray([run.expected_quality[-1] for run in ksweep[1]["learned"]]).mean()
    k1_random = np.asarray([run.expected_quality[-1] for run in ksweep[1]["random"]]).mean()

    learned_oracle_ok = learned_oracle_gap < 0.02
    learned_random_ok = learned_random_gap >= 0.05
    random_kto_ok = random_kto_gap < 0.03
    clustering_ok = learned_nmi >= 0.9 and learned_purity >= 0.95
    overcluster_ok = all(abs(value - k2_value) < 0.02 for value in overcluster_values.values())
    k1_ok = abs(k1_learned - kto) < 0.01 and abs(k1_random - kto) < 0.01

    lines = [
        "Success criteria",
        "",
        "[1] Learned near oracle",
        "    Requirement: abs(E[q]_learned - E[q]_oracle) < 0.02",
        f"    Values:      abs({learned:.4f} - {oracle:.4f}) = {learned_oracle_gap:.4f}",
        f"    Result:      {'PASS' if learned_oracle_ok else 'FAIL'}",
        "",
        "[2] Learned beats random",
        "    Requirement: E[q]_learned - E[q]_random >= 0.05",
        f"    Values:      {learned:.4f} - {random:.4f} = {learned_random_gap:.4f}",
        f"    Result:      {'PASS' if learned_random_ok else 'FAIL'}",
        "",
        "[3] Random near KTO",
        "    Requirement: abs(E[q]_random - E[q]_KTO) < 0.03",
        f"    Values:      abs({random:.4f} - {kto:.4f}) = {random_kto_gap:.4f}",
        f"    Result:      {'PASS' if random_kto_ok else 'FAIL'}",
        "",
        "[4] Clustering recovery",
        "    Requirement: learned NMI >= 0.9 and purity >= 0.95",
        f"    Values:      NMI={learned_nmi:.4f}, purity={learned_purity:.4f}",
        f"    Result:      {'PASS' if clustering_ok else 'FAIL'}",
        "",
        "[5] Graceful over-clustering",
        "    Requirement: K in {3, 5, 10} stays within 0.02 of K=2 learned",
        (
            "    Values:      "
            + ", ".join(f"K={k}: {value:.4f}" for k, value in overcluster_values.items())
            + f", K=2: {k2_value:.4f}"
        ),
        f"    Result:      {'PASS' if overcluster_ok else 'FAIL'}",
        "",
        "[6] K=1 collapse",
        "    Requirement: learned and random K=1 match KTO",
        f"    Values:      learned={k1_learned:.4f}, random={k1_random:.4f}, KTO={kto:.4f}",
        f"    Result:      {'PASS' if k1_ok else 'FAIL'}",
        "",
        "Overall",
        f"    Learned-oracle criterion: {'PASS' if learned_oracle_ok else 'FAIL'}",
        f"    Learned-random criterion: {'PASS' if learned_random_ok else 'FAIL'}",
        f"    Random-KTO criterion:     {'PASS' if random_kto_ok else 'FAIL'}",
        f"    Clustering criterion:     {'PASS' if clustering_ok else 'FAIL'}",
        f"    Over-cluster criterion:   {'PASS' if overcluster_ok else 'FAIL'}",
        f"    K=1 criterion:            {'PASS' if k1_ok else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"


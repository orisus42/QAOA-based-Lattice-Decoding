"""Secondary analysis: smaller native caps versus larger caps on the same groups.

This is explicitly post-protocol analysis, not the EXP026 primary test.
Each cap has separately fitted global angles, so it compares complete policies.
"""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import shutil
import numpy as np
from threadpoolctl import threadpool_limits


def main(run):
    cfg = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    rows = json.loads((run / "per_instance.json").read_text())
    assert json.loads((run / "manifest.json").read_text())["status"] == "completed"
    n, reps = len(rows), cfg["bootstrap_replicates"]
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    weights = rng.multinomial(n, np.full(n, 1/n), size=reps)
    good = np.array([r["babai_cost"] == r["optimum"] for r in rows], dtype=float)
    fail = np.flatnonzero(good == 0)
    shots = cfg["primary_budget"]
    def hit(p):
        return -np.expm1(shots * np.log1p(-np.clip(p, 0, 1-1e-16)))
    point, boot = {}, {}
    for method in cfg["methods"]:
        ideal = np.array([r["methods"][method]["ideal"][str(shots)]["hybrid_success"] for r in rows])
        noisy = good.copy()
        draws = weights @ good / n
        for i in fail:
            row = rows[i]
            with np.load(run / "noise" / f'{row["id"]}_{method}.npz') as f:
                p = f["gate_idle_readout"][:, row["optimum"]].copy()
            t = len(p)
            w = rng.multinomial(t, np.full(t, 1/t), size=reps)
            draws += weights[:, i] * hit(w @ p / t) / n
            noisy[i] = hit(p.mean())
        point[method] = {"ideal": float(ideal.mean()), "noise": float(noisy.mean())}
        boot[method] = {"ideal": weights @ ideal / n, "noise": draws}
        np.testing.assert_allclose(noisy.mean(),
            summary["methods"][method]["channels"]["gate_idle_readout"][str(shots)]["hybrid_success"], atol=1e-12)

    def estimate(value, draws):
        return {"difference": float(value), "group_plus_mc_ci95": np.quantile(draws, [.025, .975]).tolist()}

    result = {"recorded_at": datetime.now().astimezone().isoformat(),
        "analysis_status": "Secondary post-protocol analysis; not a new confirmatory test.",
        "shots": shots, "count": n, "noise_penalties": {}, "lower_minus_higher_cap": {},
        "limits": [
            "Caps use separately trained global angles; this does not isolate gate deletion as a causal mechanism.",
            "CZ counts are near matched across families, not durations or calibrated error exposure.",
            "Intervals are descriptive and not adjusted for multiple comparisons.",
            "Native-cap matching jointly profiles both families; this is a benchmark procedure, not a cheap standalone deployment rule.",
            "Saved calibration noise is an approximation, not hardware evidence."
        ]}
    for method in cfg["methods"]:
        result["noise_penalties"][method] = estimate(
            point[method]["noise"] - point[method]["ideal"],
            boot[method]["noise"] - boot[method]["ideal"])
    for family in ["cached", "incumbent"]:
        lower, higher = f"{family}_cap128", f"{family}_cap160"
        result["lower_minus_higher_cap"][family] = {
            channel: estimate(point[lower][channel] - point[higher][channel],
                              boot[lower][channel] - boot[higher][channel])
            for channel in ["ideal", "noise"]}
        result["lower_minus_higher_cap"][family]["change_in_noise_penalty"] = estimate(
            (point[lower]["noise"] - point[lower]["ideal"]) - (point[higher]["noise"] - point[higher]["ideal"]),
            (boot[lower]["noise"] - boot[lower]["ideal"]) - (boot[higher]["noise"] - boot[higher]["ideal"]))
    primary = boot["cached_cap160"]["noise"] - boot["incumbent_cap160"]["noise"]
    np.testing.assert_allclose(np.quantile(primary, [.025, .975]),
        summary["paired"]["cap160"]["gate_idle_readout"][str(shots)]["group_plus_mc_ci95"], atol=1e-12)
    out = run / ("budget_analysis_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z"))
    out.mkdir()
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(__file__, out / "analysis_script.py")
    (out / "manifest.json").write_text(json.dumps({
        "input_manifest_sha256": sha256((run / "manifest.json").read_bytes()).hexdigest(),
        "outputs": {p.name: sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("ANALYSIS:", out)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main(Path(sys.argv[1]).resolve())

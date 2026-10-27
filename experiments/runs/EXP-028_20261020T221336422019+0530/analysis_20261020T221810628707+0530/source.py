"""Analyze immutable measured counts; distinguish fixed-cohort and population claims."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import numpy as np
from scipy.stats import beta, norm
ROOT = Path(__file__).resolve().parents[1]


def binomial_interval(k, n, alpha=.05):
    return [float(beta.ppf(alpha/2, k, n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha/2, k+1, n-k)) if k < n else 1.]


def wilson(k, n):
    p = k/n; z = norm.ppf(.975); d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(center-half), float(center+half)]


def main():
    run = Path((ROOT/"project/current_hardware_run.txt").read_text())
    assert (run/"hardware_complete.json").exists()
    out = run/("analysis_"+datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z"));out.mkdir()
    rows = json.loads((run/"rows.json").read_text())
    order = json.loads((run/"pub_order.json").read_text())
    protocol = json.loads((run/"frozen_protocol.json").read_text())
    result = []
    for i, row in enumerate(rows):
        stem = row["id"]+"_"+row["method"]
        data = np.load(run/"circuits"/(stem+".npz"))
        blocks = []
        for j, pub in enumerate(order):
            if pub["row_index"] == i:
                a = np.load(run/"counts"/f"pub_{j:03d}_ordered.npy")
                counts = json.loads((run/"counts"/f"pub_{j:03d}.json").read_text())["counts"]
                assert sum(counts.values()) == len(a) == pub["shots"]
                unique, count = np.unique(a, return_counts=True)
                assert {int(k,2):v for k,v in counts.items()} == dict(zip(unique.tolist(),count.tolist()))
                blocks.append(a)
        assert len(blocks) == 2
        values = np.concatenate(blocks); costs = data["costs"][values]
        assert len(values) == 8192
        r = dict(row); r["shots"] = len(values)
        r["best_observed_cost"] = int(min(costs))
        r["retained_babai_cost"] = int(min(row["babai_cost"], min(costs)))
        r["mean_sampled_cost"] = float(np.mean(costs))
        r["cost_histogram"] = {int(c):int(n) for c,n in zip(*np.unique(costs,return_counts=True))}
        for metric, mask in [("improve",costs<row["babai_cost"]),("at_most_babai",costs<=row["babai_cost"]),("optimum",costs==row["optimum_cost"])]:
            k = int(mask.sum())
            r[metric] = {"count":k,"probability":k/len(costs),"wilson_ci95":wilson(k,len(costs)),
                         "clopper_pearson_ci95":binomial_interval(k,len(costs))}
        r["block_improvement_counts"] = [int((data["costs"][b]<row["babai_cost"]).sum()) for b in blocks]
        result.append(r)
    lookup = {(r["id"],r["method"]):r for r in result}
    failure_ids = [r["id"] for r in result if r["stratum"]=="failure" and r["method"]=="full_fixed"]
    rng = np.random.default_rng(20261020286)
    pairs = {}
    for method, baseline in [("weak75_frozen","full_fixed"),("weak75_retrained","full_fixed"),("cached24","incumbent24")]:
        a = [lookup[(ident,method)] for ident in failure_ids]
        b = [lookup[(ident,baseline)] for ident in failure_ids]
        pa = np.array([r["improve"]["probability"] for r in a]);pb=np.array([r["improve"]["probability"] for r in b])
        boots = (rng.binomial(8192,pa,(20000,6))/8192-rng.binomial(8192,pb,(20000,6))/8192).mean(axis=1)
        # Simultaneous exact binomial bounds remain meaningful at zero counts.
        ca=np.array([binomial_interval(r["improve"]["count"],8192,.05/12) for r in a])
        cb=np.array([binomial_interval(r["improve"]["count"],8192,.05/12) for r in b])
        pairs[method+"_minus_"+baseline]={"mean_probability_difference":float(np.mean(pa-pb)),
           "fixed_cohort_shot_bootstrap_ci95":np.quantile(boots,[.025,.975]).tolist(),
           "conservative_simultaneous_exact_ci95":[float(np.mean(ca[:,0]-cb[:,1])),float(np.mean(ca[:,1]-cb[:,0]))],
           "case_differences":(pa-pb).tolist(),"zero_count_cells":int(sum(r["improve"]["count"]==0 for r in a+b)),
           "note":"Shot uncertainty conditional on six fixed cases; no population CI. Bootstrap can underrepresent uncertainty at zero counts; exact simultaneous bounds supplied."}
    metrics=json.loads((run/"job_metrics.json").read_text())
    usage=metrics.get("usage",{})
    summary={"experiment_id":"EXP-028","hardware":json.loads((run/"hardware_complete.json").read_text()),
        "qpu_usage":usage,"primary":pairs["weak75_frozen_minus_full_fixed"],"comparisons":pairs,
        "failure_case_ids":failure_ids,"all_case_count":8,"failure_case_count":6,"methods":{},
        "limitations":protocol["limitations"]+["Per-shot/binomial intervals assume independent stationary shots; block counts permit a limited drift diagnostic."]}
    for m in protocol["methods"]:
        rr=[r for r in result if r["method"]==m];ff=[r for r in rr if r["stratum"]=="failure"]
        summary["methods"][m]={"mean_native_cz":float(np.mean([r["native_cz"] for r in rr])),
           "failure_mean_p_improve":float(np.mean([r["improve"]["probability"] for r in ff])),
           "failure_improvement_counts":sum(r["improve"]["count"] for r in ff),
           "failure_optimum_counts":sum(r["optimum"]["count"] for r in ff),
           "all_observed_optimum_with_babai":sum(r["retained_babai_cost"]==r["optimum_cost"] for r in rr),
           "all_observed_improved_over_babai":sum(r["retained_babai_cost"]<r["babai_cost"] for r in rr)}
    for name,value in [("per_circuit.json",result),("summary.json",summary)]:
        (out/name).write_text(json.dumps(value,indent=2)+"\n")
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    (out/"manifest.json").write_text(json.dumps({"at":datetime.now().astimezone().isoformat(),
        "input_counts_sha256":{p.name:sha256(p.read_bytes()).hexdigest() for p in (run/"counts").iterdir()},
        "output_sha256":{p.name:sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}},indent=2))
    (run/"latest_analysis_path.txt").write_text(str(out))
    print(json.dumps(summary,indent=2))
    print("ANALYSIS",out)


if __name__=="__main__":main()

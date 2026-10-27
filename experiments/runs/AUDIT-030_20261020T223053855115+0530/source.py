"""Independent integer-cost and submission-manifest audit for the two QPU runs."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from qiskit import qpy
from lattice_qaoa.reference import babai


def main():
    runs=[Path((ROOT/"project"/n).read_text()) for n in ["current_hardware_run.txt","repeat_hardware_run.txt"]]
    source=json.loads((ROOT/"experiments/runs/EXP-007_20261020T044846904050+0530/instances.json").read_text())
    out=ROOT/"experiments/runs"/("AUDIT-030_"+datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z"));out.mkdir()
    hashes=0;shots=0;rows_checked=0;usage=0;reconstructed=[]
    for run in runs:
        manifest=json.loads((run/"manifest.json").read_text())
        for name,digest in manifest["frozen_files_sha256"].items():
            assert sha256((run/name).read_bytes()).hexdigest()==digest;hashes+=1
        rows=json.loads((run/"rows.json").read_text());order=json.loads((run/"pub_order.json").read_text())
        snap=json.loads((run/"snapshot.json").read_text())
        analysis=Path((run/"latest_analysis_path.txt").read_text())
        for i,row in enumerate(rows):
            x=source[row["source_index"]];R=np.array(x["reduced_basis"],dtype=np.int64)
            y=np.r_[x["public_target"],np.zeros(4,dtype=np.int64)]
            center=babai(R,y)
            stem=row["id"]+"_"+row["method"];data=np.load(run/"circuits"/(stem+".npz"))
            with (run/row["qpy"]).open("rb") as f:qc=qpy.load(f)[0]
            measured={qc.find_bit(inst.clbits[0]).index:qc.find_bit(inst.qubits[0]).index
                      for inst in qc.data if inst.operation.name=="measure"}
            assert measured=={j:snap["physical_qubits"][int(p)] for j,p in enumerate(data["final_permutation"])}
            values=np.concatenate([np.load(run/"counts"/f"pub_{j:03d}_ordered.npy") for j,p in enumerate(order) if p["row_index"]==i])
            bits=(values.astype(np.int64)[:,None]>>np.arange(16))&1
            z=center-2+bits[:,0::2]+2*bits[:,1::2]
            residual=z@R.T-y
            costs=np.sum(residual*residual,axis=1)
            np.testing.assert_array_equal(costs,data["costs"][values])
            assert np.min(costs)>=row["optimum_cost"]
            reconstructed.append({"run":run.name,"id":row["id"],"method":row["method"],
                                  "shots":len(values),"improve":int(np.sum(costs<row["babai_cost"])),
                                  "optimum":int(np.sum(costs==row["optimum_cost"]))})
            rows_checked+=1;shots+=len(values)
        usage+=json.loads((run/"job_metrics.json").read_text())["usage"]["quantum_seconds"]
    assert usage==280 and shots==983040 and rows_checked==80
    first=runs[0];pooled=Path((first/"latest_analysis_path.txt").read_text())
    analyzed=json.loads((pooled/"per_circuit.json").read_text())
    for row in analyzed:
        rr=[r for r in reconstructed if r["id"]==row["id"] and r["method"]==row["method"]]
        assert sum(r["improve"] for r in rr)==row["improve"]["count"]
        assert sum(r["optimum"] for r in rr)==row["optimum"]["count"]
    result={"at":datetime.now().astimezone().isoformat(),"status":"passed","frozen_file_hashes_checked":hashes,
        "submitted_circuits_checked":rows_checked,"shots_independently_rescored":shots,
        "measurement_maps_checked":rows_checked,"qpu_seconds":usage,"account_remaining_seconds":320,
        "checks":"Reconstructed original integer lattice residual costs directly from sampled bits, basis and public target. Checked physical-to-logical measurement maps, frozen manifests, pooled count totals and actual job usage.",
        "scope":"Does not prove quantum advantage, statistical generalization, or physical correctness of IBM calibration models.",
        "source_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    (out/"reconstructed_counts.json").write_text(json.dumps(reconstructed,indent=2))
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    (first/"latest_audit_path.txt").write_text(str(out))
    print(json.dumps(result,indent=2));print(out)


if __name__=="__main__":main()

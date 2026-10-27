"""Seaborn comparison of EXP-003; every render receives a new folder."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lattice_qaoa.sampling import repeated_success


def main():
    run = Path(sys.argv[1]).resolve()
    s = json.loads((run/"summary.json").read_text())
    rows = json.loads((run/"per_instance.json").read_text())
    assert s["experiment_id"] == "EXP-003"
    revision, out = 1, run/"figures"
    while out.exists():
        revision += 1
        out = run/f"figures-v{revision}"
    out.mkdir()
    shutil.copy2(__file__, out/"plot_source.py")
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    plt.rcParams.update({"svg.fonttype": "none", "text.color": "#173042", "axes.labelcolor": "#173042",
                         "axes.edgecolor": "#d7dede", "grid.color": "#e8eded", "grid.linewidth": .6})
    colors = {"qaoa_tuned": "#087f8c", "qaoa_fixed": "#ce6a43", "uniform_box": "#94a2ad", "cbd_secret": "#7963ad"}
    names = {"qaoa_tuned": "QAOA · tuned per instance", "qaoa_fixed": "QAOA · fixed trained angles",
             "uniform_box": "Uniform box sampling", "cbd_secret": "Small-secret classical sampling"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("#fbfaf7")
    fig.subplots_adjust(left=.07, right=.97, top=.69, bottom=.24, wspace=.66)
    fig.text(.07, .94, "MODULE-LWE / IDEAL SIMULATION", color="#087f8c", fontsize=11, weight="bold")
    fig.text(.07, .865, "Testing the role of quantum sampling", fontsize=25, weight="bold")
    fig.text(.07, .805, "64 toy examples  /  n=2, k=2, q=17  /  16 qubits  /  one QAOA layer  /  the same decoding objective", fontsize=11, color="#536877")
    grid = np.unique(np.rint(np.geomspace(1, 1024, 100)).astype(int))
    for method in ["cbd_secret", "qaoa_tuned", "qaoa_fixed", "uniform_box"]:
        p = np.array([r["probabilities"][method] for r in rows])
        y = [100*repeated_success(p, int(n)).mean() for n in grid]
        sns.lineplot(x=grid, y=y, color=colors[method], label=names[method], linewidth=2.4, ax=axes[0])
        actual = s["methods"][method]["by_budget"]
        axes[0].scatter([r["shots_or_samples"] for r in actual], [100*r["observed_solved_fraction"] for r in actual],
                        color=colors[method], s=28, edgecolor="white", linewidth=.7, zorder=4)
    axes[0].axhline(s["babai"]["solved_fraction"]*100, color="#173042", linewidth=1.2, linestyle=(0, (3, 3)))
    axes[0].text(1.2, s["babai"]["solved_fraction"]*100-5, "Babai alone: 53/64", fontsize=10, color="#173042")
    axes[0].set(xscale="log", xlim=(.9, 1200), ylim=(0, 104), xlabel="Samples per example (log scale)", ylabel="Examples solved (%)")
    axes[0].set_xticks([1, 10, 100, 1024], labels=["1", "10", "100", "1,024"])
    axes[0].set_title("A   Exact predictions and sampled outcomes", loc="left", fontsize=12, weight="bold", pad=17)
    axes[0].legend(loc="upper left", bbox_to_anchor=(-.02, -.21), ncol=2, frameon=False, fontsize=9)
    entries = [
        ("Babai", s["babai"]["solved_count"], None, "#173042"),
        ("Babai + 1-coordinate moves", s["classical_refinement"]["refine_1"]["solved_count"], None, "#526b78"),
        ("Babai + 1/2-coordinate moves", s["classical_refinement"]["refine_2"]["solved_count"], None, "#526b78"),
    ]
    for method in ["cbd_secret", "qaoa_tuned", "qaoa_fixed", "uniform_box"]:
        r = s["methods"][method]["by_budget"][-1]
        entries.append((names[method], r["observed_solved_count"], r["expected_solved_fraction"]*100, colors[method]))
    for y, (name, count, expected, color) in enumerate(entries):
        value = count/64*100
        axes[1].hlines(y, 0, value, color=color, linewidth=3, alpha=.28)
        axes[1].scatter([value], [y], color=color, s=65, zorder=3)
        if expected is not None:
            axes[1].scatter([expected], [y], facecolor="white", edgecolor=color, marker="D", s=30, linewidth=1.1, zorder=4)
        axes[1].text(max(value, expected or 0)+3, y, f"{count}/64", va="center", color=color, fontsize=10, weight="bold")
    axes[1].set_yticks(range(len(entries)), labels=[x[0] for x in entries])
    axes[1].invert_yaxis()
    axes[1].set(xlim=(-1, 115), xlabel="Examples solved (%)")
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_title("B   At 1,024 samples, or a classical procedure", loc="left", fontsize=12, weight="bold", pad=17)
    axes[1].grid(axis="y", visible=False)
    axes[0].grid(axis="x", visible=False)
    for ax in axes:
        sns.despine(ax=ax)
    fig.text(.07, .08, "Lines / open diamonds: exact expected success. Filled markers: observed success in one seeded run. All 64 binary boxes contain a global optimum.", fontsize=9, color="#536877")
    fig.text(.07, .035, "Equal sample counts do not include QAOA parameter preparation or imply equal runtime. Classical refinement has its own recorded evaluation budget. No hardware used.", fontsize=9, color="#536877")
    for extension in ["png", "svg"]:
        fig.savefig(out/f"module_lwe_comparison.{extension}", dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    manifest_path = run/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in out.iterdir():
        manifest["output_sha256"][path.relative_to(run).as_posix()] = sha256(path.read_bytes()).hexdigest()
    manifest["latest_figure"] = (out/"module_lwe_comparison.png").relative_to(run).as_posix()
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(out/"module_lwe_comparison.png")


if __name__ == "__main__":
    main()

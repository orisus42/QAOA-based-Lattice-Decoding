"""Reproducible Seaborn figure for an EXP-002 run. Creates new figure files only."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lattice_qaoa.sampling import repeated_success, solved_count_interval


def main():
    run = Path(sys.argv[1]).resolve()
    summary = json.loads((run/"summary.json").read_text())
    records = json.loads((run/"per_instance.json").read_text())
    assert summary["experiment_id"] == "EXP-002"
    figure_dir = run/"figures"
    figure_dir.mkdir(exist_ok=False)
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    plt.rcParams.update({"svg.fonttype": "none", "axes.titleweight": "bold", "axes.labelcolor": "#173042",
                         "text.color": "#173042", "xtick.color": "#526572", "ytick.color": "#526572",
                         "axes.edgecolor": "#d5dedf", "grid.color": "#e8eded", "grid.linewidth": .7})
    colors = {"learned": "#087f8c", "fixed": "#d35e42", "uniform": "#8b99a4"}
    labels = {"learned": "QAOA · learned selection", "fixed": "QAOA · fixed angles", "uniform": "Uniform classical sampling"}
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("#fbfaf7")
    fig.subplots_adjust(left=.075, right=.975, bottom=.23, top=.76, wspace=.30)
    fig.text(.075, .94, "QAOA UNDER IDEAL CONDITIONS", color="#087f8c", fontsize=11, weight="bold")
    fig.text(.075, .865, "From individual measurements to solved problems", fontsize=23, weight="bold")
    fig.text(.075, .805, "200 fixed toy lattice instances  /  6 or 8 qubits  /  one QAOA layer  /  no hardware noise", fontsize=11, color="#526572")
    grid = np.unique(np.rint(np.geomspace(1, 1024, 110)).astype(int))
    for method in ["learned", "fixed", "uniform"]:
        p = np.array([r["probabilities"][method] for r in records])
        curve = [100*repeated_success(p, int(n)).mean() for n in grid]
        sns.lineplot(x=grid, y=curve, color=colors[method], linewidth=2.5, label=labels[method], ax=axes[0])
        if method == "learned":
            intervals = np.array([solved_count_interval(repeated_success(p, int(n))) for n in grid])/len(records)*100
            axes[0].fill_between(grid, intervals[:, 0], intervals[:, 1], color=colors[method], alpha=.10, linewidth=0)
        if method in ["fixed", "learned"]:
            budgets = summary["methods"][method]["by_shot_budget"]
            axes[0].scatter([r["shots"] for r in budgets], [100*r["observed_solved_fraction"] for r in budgets],
                            s=33, color=colors[method], edgecolor="white", linewidth=.9, zorder=4)
        sns.ecdfplot(x=p*100, stat="percent", color=colors[method], linewidth=2.2, ax=axes[1])
    axes[0].axhline(summary["babai"]["solved_fraction"]*100, color="#173042", linestyle=(0, (3, 3)), linewidth=1.3)
    axes[0].text(1.2, 92.5, "Babai: 193/200 = 96.5%", fontsize=10, color="#173042")
    axes[0].set(xscale="log", xlim=(.9, 1200), ylim=(0, 103), xlabel="Measurements per problem (log scale)", ylabel="Problems solved at least once (%)")
    axes[0].set_xticks([1, 10, 100, 1024], labels=["1", "10", "100", "1,024"])
    axes[0].set_title("A   Repetition changes the success rate", loc="left", pad=14, fontsize=12)
    axes[0].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e4e8e8", fontsize=9)
    axes[1].set(xlabel="Chance of an optimum in one shot (%)", ylabel="Cumulative share of examples (%)", ylim=(0, 103))
    axes[1].set_xlim(left=0)
    axes[1].set_title("B   Success probability varies by example", loc="left", pad=14, fontsize=12)
    for ax in axes:
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)
    fig.text(.075, .125, "Lines: exact ideal probabilities. Dots: one simulated measurement batch. Teal band: central 95% measurement range on this fixed cohort.", fontsize=9, color="#526572")
    fig.text(.075, .075, "Success means an optimal lattice point, not secret recovery. Classical enumeration of each 64/256-candidate box solves all 200 instances.", fontsize=9, color="#526572")
    png, svg = figure_dir/"ideal_success.png", figure_dir/"ideal_success.svg"
    fig.savefig(png, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)
    manifest_path = run/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in [png, svg]:
        manifest["output_sha256"][path.relative_to(run).as_posix()] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(png)


if __name__ == "__main__":
    main()

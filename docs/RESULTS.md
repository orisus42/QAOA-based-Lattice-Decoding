# Results and evidence map

## Main comparisons

| Study | Cohort | Main finding | Saved evidence |
|---|---|---|---|
| EXP-007 | 512 fresh Module-LWE matrix groups | Frozen aggressive-pruning variants evaluated ideally, with finite-shot and reference checks | [Summary](../experiments/runs/EXP-007_20261020T044846904050+0530/summary.json) |
| EXP-008 | Same 512 groups | IBM Fez routing/calibrated noise: expected 1,024-shot hybrid optimum success 87.14% full, 88.42% unchanged-angle 75% pruning, 88.65% retrained 75% pruning | [Summary with uncertainty](../experiments/runs/EXP-008_20261020T050837222393+0530/summary.json) |
| EXP-010 | 75 full-size key-generation vectors, plus tiny sensitivity families | Checks ML-KEM algebra/provenance; does not decode the secure-size examples | [Summary](../experiments/runs/EXP-010_20261020T060129580284+0530/summary.json) |
| EXP-023 | 128 fresh groups; 32 timed | Same masks/paths/probabilities; 14.0177x cached selector speedup | [Summary](../experiments/runs/EXP-023_20261020T180743053982+0530/summary.json) |
| EXP-026 | 128 fresh groups | Near-matched native-budget comparison does not establish cached superiority over inexpensive incumbent-informed pruning | [Summary](../experiments/runs/EXP-026_20261020T190619077356+0530/summary.json) |
| EXP-028 / EXP-029 | Eight selected cases, five methods | Two IBM Marrakesh jobs, 983,040 measured shots, 280 QPU seconds | [Complete results including classical controls](../results/complete_results.json) |

## Baselines and denominators

In the 512-case noise study, Babai attains the reference optimum in 441/512 cases (86.13%). Classical one-coordinate and two-coordinate refinement attain 471/512 and 497/512, respectively. These differ from QAOA's *expected* retained-incumbent success at a finite sample budget.

On hardware, the headline 7/8 and 4/8 are *observed* retained-incumbent outcomes on a selected cohort. Each method used 24,576 shots per case. The six difficult cases therefore provide 147,456 shots per method for improving-event counts, while all eight cases and all five methods contribute 983,040 shots overall.

See the [CSV](../results/poster_result_facts.csv) for the complete hardware table. The incumbent-informed method attains 8/8, as does classical two-coordinate refinement; the cached method is not the overall winner.

## Matched native-budget comparison

At the prespecified cap-160 / 1,024-sample comparison in EXP-026, cached minus incumbent-informed expected hybrid success is **−0.7422 percentage points**, paired group-plus-Monte-Carlo 95% interval **[−1.6268, +0.0832]**. Native circuit costs are matched to within two CZ gates. The noisy evaluation covers the 28 Babai failures plus four sentinels, while retained-Babai aggregate metrics include all 128 groups.

The interval does not establish a quality advantage for either method. Its point estimate favors the cheaper control. A lower native-gate cap reduces modeled noise damage, but separately fitted angles prevent interpreting that comparison as a pure causal effect of gate deletion. [Analysis](../experiments/runs/EXP-026_20261020T190619077356+0530/analysis_20261020T193655731512+0530/)

## Hardware source files

- [Initial run, EXP-028](../experiments/runs/EXP-028_20261020T221336422019+0530/): 327,680 shots; 94 QPU seconds; job `dbbpkvk24mrc73ci7mjg`.
- [Precision extension, EXP-029](../experiments/runs/EXP-029_20261020T221914491556+0530/): 655,360 shots; 186 QPU seconds; job `dbbpn7424mrc73ci7os0`.
- [Pooled final analysis](../experiments/runs/EXP-028_20261020T221336422019+0530/pooled_analysis_20261020T223051213990+0530/): all five methods, exact intervals, block counts, classical references, and limitations.
- [Historical independent audit](../experiments/runs/AUDIT-030_20261020T223053855115+0530/summary.json): original frozen hashes, 80 measurement maps, and independent integer rescoring.
- [Release audit](../scripts/verify_hardware_release.py): repeats count/cost/map checks locally and distinguishes redacted export metadata from original frozen bytes.

Hardware jobs cannot be expected to reproduce the same bitstrings when rerun. The published raw shots make the *recorded analysis* reproducible without new cloud use.

## Negative findings and diagnostics

The catalogue retains earlier unsuccessful, paused, resumed, and diagnostic runs. In particular, the centered prototype did not establish improvement, 90% pruning did not reliably preserve quality, exact caching did not establish superiority to cheap pruning, and failed noise-fusion attempts are retained. See the [experiment index](EXPERIMENTS.md) and [limitations](LIMITATIONS.md).

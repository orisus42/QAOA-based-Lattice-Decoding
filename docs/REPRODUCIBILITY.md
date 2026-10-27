# Reproducing and auditing the research

## Three levels of reproduction

1. **Inspect the evidence:** open the [results](../results/) and [figure gallery](../figures/README.md). No Python or cloud account is required.
2. **Verify the release and recorded hardware analysis:** run the checks below. These read the archived data locally and do not use a QPU.
3. **Rerun an experiment:** use the corresponding source/configuration and run dependencies. A rerun creates a new timestamped run. Long simulations and new hardware jobs are separate from verification.

## Environment

The recorded environment is Python 3.14.3 with the exact versions in [requirements-research.txt](../requirements-research.txt). Per-run manifests preserve earlier snapshots. Use a virtual environment and run from the repository root:

```bash
python -m pip install -r requirements-research.txt
python -m pytest -q
python examples/small_decoding_demo.py
```

The original platform was Windows. NumPy/SciPy, BLAS, and Qiskit differences can affect timings and floating-point tie decisions; use pinned versions for comparison. The release checks have been exercised locally; this is not a claim of validation on every OS. If a sandbox restricts its default temporary directory, first create `local_outputs/`, then use `python -m pytest -q --basetemp=local_outputs/pytest-check`.

## Verify saved evidence offline

```bash
python scripts/verify_release.py
python scripts/verify_hardware_release.py
```

`verify_release.py` uses only the standard library and checks every listed file against [SHA256SUMS](../reproducibility/SHA256SUMS). `.gitattributes` preserves archive bytes across Git checkouts.

`verify_hardware_release.py` requires NumPy, SymPy, and Qiskit. It checks count histograms against ordered shots, independently reconstructs integer lattice residual costs from public matrices and measured bits, checks QPY measurement permutations, compares the pooled results, and totals the measured QPU usage. No IBM login is used. The original manifest had 299 frozen-file hashes; three account/service-selection metadata files are intentionally omitted. Their historical digests are accounted for separately, not reported as content-verified.

To recompute the statistical analysis into a **new** timestamped directory:

```bash
python scripts/analyze_hardware.py --pooled
```

This uses the two portable pointer files in `project/`. It updates an analysis pointer but never overwrites a historical analysis folder. The release hashes for any intentionally updated pointer will then differ; keep an untouched checkout for integrity verification. The new `verify_hardware_release.py` is the recommended read-only audit; historical `audit_hardware.py` assumes the unredacted workspace and will not accept omitted account metadata.

## Regenerate the scientific figures

The archived PNG/PDF/SVG files are immediately usable. Every figure has a corresponding `*_data.json` describing the plotted values and their interpretation.

To regenerate the original base figures into a new timestamped folder and then the hardware figures:

```bash
python scripts/build_poster_assets.py
python scripts/build_poster_assets.py hardware
```

These commands use saved summaries, hardware circuit arrays and recorded shots, and may take several minutes for the angle landscapes. They write under `deliverables/`, which is ignored by Git, and maintain a local `project/current_poster_assets.txt` pointer. They make no cloud calls. The base program relies on the recorded EXP-008, EXP-023, and EXP-026 paths included in this release.

To compile the circuit diagram, use `circuits/main.tex` or `circuits/expanded.tex` with a LaTeX installation containing TikZ/Quantikz. This does not simulate or execute the quantum circuit.

## Rerun selected local studies

| Purpose | Command | Notes |
|---|---|---|
| Generate 64 small structured instances and exact references | `python experiments/run_module_lwe_baseline.py` | EXP-001; no cloud |
| Ideal QAOA baseline | `python experiments/run_module_lwe_qaoa.py` | Reads recorded baseline/config inputs |
| Initial phase-sparsification study | `python experiments/run_sparsification.py` | Development study, not a new holdout by itself |
| Full-size equation provenance and tiny parameter sensitivity | `python experiments/run_mlkem_provenance.py` | Uses pinned vendored code and public vectors; no secure-size decoding |

For other studies, use [EXPERIMENTS.md](EXPERIMENTS.md) and the corresponding JSON configuration. Several research scripts intentionally reference a particular earlier timestamped run. Review those dependencies before launching a long rerun; do not silently substitute a newly generated cohort or claim it is the same experiment. Large simulator intermediates must be regenerated before running an audit that consumes them.

## Archive boundaries

**Included:** source and tests, experiment configurations and seeds, all historical run summaries and available text result tables, frozen source snapshots, failures and diagnostic reports, saved calibrations, all measured hardware count JSON and ordered NPY samples, hardware QPY circuits and circuit NPZ arrays, figure assets and figure data, and pinned cryptographic reference sources.

**Omitted from Git:** derived non-hardware simulator QPY/NPZ/NPY/joblib files, Python caches and local environments, duplicate ZIP bundles, local compiler binaries, account-usage snapshots and service-selection metadata, conversation/question logs, personal learning notes, institution template/logo assets, and unverified final poster/abstract drafts. Original workspace files were preserved. Omitted scientific intermediates have sizes, reasons, and original hashes in [omitted_intermediates.json](../reproducibility/omitted_intermediates.json). The archive therefore supports complete hardware reanalysis and result inspection, but is not a byte-for-byte export of every simulator cache.

The full-size ML-KEM vectors and toy generated secrets are public/experimental reference data, not account credentials. Reference answers are kept out of solver inputs.

## Provenance and privacy transformations

[export_provenance.json](../reproducibility/export_provenance.json) maps each exported research artifact to its original relative path, original SHA-256, released SHA-256, byte count, and whether local-path metadata changed. Numeric arrays and measured count values are preserved. Historical source manifests are retained as historical records rather than rewritten to conceal these transformations.

The [release validation record](../reproducibility/validation.json) documents what was checked for publication. Checks of file integrity and computation do not establish algorithmic novelty or independent scientific review.

## Optional new hardware execution

The archived runner is `experiments/run_hardware.py`, with explicit `prepare`, `submit`, and `collect` actions. Preparation queries live backends and needs an independently configured IBM Quantum account; submission consumes that account's allowance. The historical ten-minute allocation and 320 seconds remaining were observations from October 2026, not a promise about a reader's account or a current quota.

No quick-start, test, verification, or figure-generation command submits a job. Do not reuse stale calibrations or assume a recorded shot count guarantees a particular QPU duration. Keep credentials in the provider's normal account store, never in this repository.

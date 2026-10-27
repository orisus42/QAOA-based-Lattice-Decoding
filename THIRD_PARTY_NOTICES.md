# Third-party sources and attribution

## Educational kyber-py

The source under `research/reference_sources/kyber_py_897923667b2f/` comes from [GiacomoPope/kyber-py](https://github.com/GiacomoPope/kyber-py), commit `897923667b2fa80afcf910fc8c1825dddf54a97d`. Its complete upstream [LICENSE](research/reference_sources/kyber_py_897923667b2f/LICENSE) (MIT and Apache 2.0 texts) and source manifest are preserved. Copyright notices belong to the upstream authors. The vendored implementation is used for provenance checks and is not presented as this project's implementation of ML-KEM.

## NIST ACVP vectors

The public ML-KEM key-generation test vectors under `research/reference_sources/nist_acvp_975de31eb83d/` come from the National Institute of Standards and Technology's [ACVP-Server](https://github.com/usnistgov/ACVP-Server), commit `975de31eb83d87039ec88934fdc47d8c312b892d`. The [source manifest](research/reference_sources/nist_acvp_975de31eb83d/source_manifest.json) preserves the exact URL and checksum. The [upstream license notice](research/reference_sources/nist_acvp_975de31eb83d/NOTICE.txt) is included in full. These published test-vector secrets are test data, not live credentials.

## Libraries, services, and prior methods

Qiskit, Qiskit Aer, Qiskit IBM Runtime, NumPy, SciPy, SymPy, Matplotlib, Seaborn, and other dependencies remain under their respective upstream licenses. They are installed as dependencies rather than relicensed here. IBM Quantum supplied cloud access to the recorded hardware jobs; this does not imply IBM endorsement of the interpretations.

Babai decoding, phase sparsification, and the analytic p=1 expectation have prior literature; see [references](docs/REFERENCES.md). Original research files are distinct from the vendored third-party material. The author has not yet selected a redistribution license for the original project.

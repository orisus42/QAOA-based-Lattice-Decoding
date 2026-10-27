# References

These are sources of background, methods, and provenance, not claims that this project reproduces their full scope.

1. L. Babai. **On Lovasz' lattice reduction and the nearest lattice point problem.** *Combinatorica* 6, 1–13 (1986). [DOI](https://doi.org/10.1007/BF02579403). Classical nearest-plane baseline.
2. B. Priestley and P. Wallden. **A Practically Scalable Approach to the Closest Vector Problem for Sieving via QAOA with Fixed Angles.** *Quantum Science and Technology* 11, 025018 (2026). [Preprint](https://arxiv.org/abs/2503.08403) · [DOI](https://doi.org/10.1088/2058-9565/ae4cc6). A QAOA/CVP motivation anchor; their constrained sieving setting differs from these toy Module-LWE-derived instances.
3. X. Liu, R. Shaydulin, and I. Safro. **Quantum Approximate Optimization Algorithm with Sparsified Phase Operator.** *IEEE QCE*, 133–141 (2022). [Preprint](https://arxiv.org/abs/2205.00118). Prior phase-sparsification work. Our magnitude heuristic is not their guaranteed ground-state-preserving construction.
4. A. Ozaeta, W. van Dam, and P. L. McMahon. **Expectation Values from the Single-Layer Quantum Approximate Optimization Algorithm on Ising Problems.** [Preprint](https://arxiv.org/abs/2012.03421). Prior analytic p=1 expectation method used by the implementation.
5. NIST. **FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard** (2024). [Official standard](https://doi.org/10.6028/NIST.FIPS.203). Cryptographic provenance, not a claim of attacking standardized parameters.
6. Giacomo Pope and contributors. **kyber-py.** [Pinned source](https://github.com/GiacomoPope/kyber-py/tree/897923667b2fa80afcf910fc8c1825dddf54a97d). Educational reference implementation used in the provenance check.
7. NIST. **ACVP-Server ML-KEM key-generation test vectors.** [Pinned source](https://github.com/usnistgov/ACVP-Server/tree/975de31eb83d87039ec88934fdc47d8c312b892d). Public reference examples and source hashes are included locally.

The [historical literature notes](../research/literature.json) and [later related-work review](../research/literature_review_Q045.json) preserve the exploratory survey. They are not a systematic review or proof of novelty. See [third-party notices](../THIRD_PARTY_NOTICES.md) for redistribution attribution.

# Method and definitions

## 1. Structured instances

Generate a public polynomial matrix A, a small secret s, and error e in the ring R_q = Z_q[x]/(x^n+1), then compute t = A s + e (mod q). The main family has n=2, k=2, q=17, eta=2, with independent centered-binomial coefficients for s and e. The generator preserves negacyclic polynomial multiplication; it is not a generic random q-ary matrix relabelled as Module-LWE.

Expand polynomial multiplication into a signed coefficient matrix A_exp. For d=nk, use the column-basis embedding

```text
B = [ q I_d   A_exp ]       y = [ t ]
    [   0      I_d  ]           [ 0 ]

C(w,s) = ||q w + A_exp s - t||² + ||s||².
```

This Euclidean objective favors short errors and short secrets. It is a surrogate, not the exact maximum-likelihood objective for centered-binomial noise. Closest-vector success, planted-secret recovery, and search-box coverage are separate measurements. A noisy instance is not automatically certified bounded-distance decoding (BDD).

## 2. Classical preprocessing and bounded encoding

LLL produces a reduced basis R=B U with an integer unimodular transform U. Babai nearest-plane supplies coefficients z_B. Two bits per coefficient encode

```text
z_i = z_B,i - 2 + x_(2i) + 2 x_(2i+1),    x in {0,1}^16.
```

The neighborhood is {-2,-1,0,+1} relative to each Babai coefficient; it is asymmetric and does not cover the entire lattice. Reference optima and planted secrets are available only to the evaluation code. The encoder takes the public basis and target, not hidden answers. Exact enumeration is feasible here because there are only 65,536 bitstrings.

## 3. QAOA and pruning

The encoded cost is converted to a normalized Ising Hamiltonian H = sum h_i Z_i + sum J_ij Z_i Z_j plus a constant. At p=1:

```text
|psi(gamma,beta)> = exp(-i beta sum X_i) exp(-i gamma H_S) |+>^16.
```

H_S keeps the local fields and selected pair interactions. Magnitude pruning deletes the smallest |J_ij| values with deterministic tie-breaking. Frozen-angle and separately retrained-angle variants are distinguished. QAOA generates candidate bitstrings; final candidates are always evaluated with the **original complete lattice objective**, and the hybrid keeps Babai if no sampled candidate improves it.

## 4. Objective-guided selection and exact caching

At fixed selection angles, backward greedy deletion chooses an edge removal using the analytic expectation of the original full objective. At each requested interaction budget K, a magnitude-pruned mask at the same budget is evaluated as a fallback.

The cached implementation precomputes and reuses trigonometric factors. Near-tied candidate decisions and fallback comparisons are checked with the original evaluator. It aims to preserve the original mask, path, and probability distribution, not to create a different quantum algorithm. EXP-023 found identical masks/paths and a maximum independently recomputed probability difference of zero on 128 fresh groups.

The cheap incumbent-informed selector is a separate baseline. Its quality and selection cost are retained in the public results; no cached-versus-cheap superiority is established.

## 5. Metrics and evaluation

- **Native CZ count:** gate count after backend-specific routing/transpilation; different from the number of logical ZZ interactions.
- **Single-shot success:** probability of sampling a reference optimum.
- **Expected N-shot success:** average of `1-(1-p_i)^N` across instances; do not insert the average p into this nonlinear formula.
- **Hybrid success:** additionally retains a correct Babai incumbent, so Babai-optimal cases have success one without quantum improvement.
- **Observed hardware success:** whether the finite measured samples plus Babai attained an optimum on a particular selected instance.
- **Strict improvement:** lower original cost than Babai; this need not be an optimum.
- **Selection time:** selector-only CPU time. It excludes shared preprocessing, training, compilation, queueing, and QPU use.

Development, training, and fresh grouped evaluations are distinguished in each run's configuration. The 512-case cohort was fresh for its original frozen candidates; later hardware selection from that cohort is a conditional replication, not a new population holdout.

## ML-KEM provenance

[EXP-010](../experiments/runs/EXP-010_20261020T060129580284+0530/) checks 25 public NIST key-generation vectors for each of the three ML-KEM parameter sets, using pinned educational reference code. It also studies tiny parameter variants, including q=3329. Matching a modulus or verifying full-size equations does not establish secure-size decoding capability. No secure ML-KEM key is recovered by these experiments.

Implementation entry points: [instances.py](../src/lattice_qaoa/instances.py), [reference.py](../src/lattice_qaoa/reference.py), [encoding.py](../src/lattice_qaoa/encoding.py), [qaoa.py](../src/lattice_qaoa/qaoa.py), [cached_pruning.py](../src/lattice_qaoa/cached_pruning.py).

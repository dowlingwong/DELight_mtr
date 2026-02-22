# EMPCA optimization profile (reusable)

This summarizes the changes between `empca_TCY.py` (original) and `empca_TCY_optimized.py`, and why the results stay equivalent.

## What changed
1) **Diagonal weights are handled as a vector**
- **Before:** build a dense diagonal matrix `W = diag(w)` and do `W @ X`.
- **After:** keep `w` as a 1D vector and use element‑wise weighting.

2) **Vectorized coefficient solve**
- **Before:** per‑event `lstsq` inside a Python loop.
- **After:** form a small `n_comp × n_comp` system once per iteration and solve in batch.

3) **Vectorized χ² calculation**
- **Before:** `residual @ W @ residual^H` (slow, and includes cross‑event terms).
- **After:** `sum(|residual|^2 * w)` per event (correct for diagonal weights).

4) **Compatible weight inputs**
- Optimized code accepts **vector weights**, **dense diagonal matrices**, or **sparse diagonals**.

## Why it’s equivalent
- When weights are diagonal, `W @ x == w * x` exactly.
- The EM‑PCA update equations use `W` only through multiplication by data or templates.
- Replacing dense `W` with vector `w` gives **identical math** but much faster execution.

## Practical impact
- **Speed:** orders‑of‑magnitude faster (no giant matrix multiplications; no per‑event loops).
- **Memory:** no `n_var × n_var` matrix stored.
- **Model size:** small, because it no longer pickles `solver.weights` or `solver.data`.

## Key equivalence checks (from verification notebook)
- Subspace overlap singular values = `[1, 1, 1, 1]`
- Weighted reconstruction error old/new = identical
- Legacy χ² recomputed old/new = identical

## Notes
- Reported χ² in the old code can differ because it included cross‑event terms via dense `W`.
- For correct diagonal weighting, use the optimized χ² definition.

## GPU Implementation (`empca_TCY_gpu.py`)
A GPU-accelerated version is available in `reusable/empca_TCY_gpu.py`.

### Requirements
- **PyTorch** with CUDA support.

### Performance
- **Speedup:** ~40x faster than the optimized CPU version for typical workloads (e.g., 20k observations, 5k variables).
- **Usage:** Drop-in replacement for the CPU version. The API is identical, but the internal solver moves data to the GPU.
- **Verification:** Results are numerically equivalent to the CPU version (within float32 precision limits).

### usage
```python
from reusable.empca_TCY_gpu import EMPCA_GPU
empca = EMPCA_GPU(n_comp=5)
empca.fit(data, weights)
```

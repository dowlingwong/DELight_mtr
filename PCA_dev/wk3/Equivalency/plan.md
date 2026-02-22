# Two Notebooks Plan (Sum-Channel Only): Equivalence Tests + Noise Ladder

This document defines **two Jupyter notebooks** (Markdown-only) for a **sum-channel** study.  
Each notebook is structured to produce **paper-grade evidence**: explicit assumptions, analytic equivalence statements, and quantitative validation plots/tables.

---

## Notebook A — OF ⇔ EMPCA (k = 1)

### Title
**Equivalence of Optimal Filtering and EMPCA in the Rank-1 Gaussian Model**

### Goal
Show that **Optimal Filter (OF)** amplitude estimation is **equivalent** to **EMPCA** coefficient estimation when EMPCA is constrained to **k=1**, and both use the **same noise weighting** (PSD/covariance).

This notebook should produce:
1) a concise analytic statement of equivalence + assumptions  
2) empirical amplitude agreement (OF vs EMPCA(k=1)) on held-out test data  
3) residual/χ² consistency checks  
4) a controlled “failure mode” demonstration when assumptions are violated

---

### A1. Setup & Inputs (sum-channel)
**Data:**
- `X_train`, `X_test`: sum-channel traces, shape `(N, T)`
- (optional) true injected amplitude/energy `E_true` for simulation validation

**Noise:**
- `J(f)`: one-sided PSD (shape `(T//2 + 1,)`) measured from MMC calibration segments  
- Optionally also create synthetic PSDs:
  - White noise PSD matched to MMC RMS
  - Pink noise PSD (∝ 1/f) matched to MMC RMS

**OF:**
- OF template `s` (time domain)
- OF implementation that uses `J(f)` (frequency-domain OF)

**EMPCA:**
- EMPCA implementation that can learn a basis from `X_train`
- Set rank **k = 1** only

**Preprocessing policy (must record):**
- baseline subtraction window
- alignment policy (fixed alignment or shift-corrected); keep consistent across methods
- train/test split method
- PSD estimation protocol (noise-only segments; independent of pulses)

---

### A2. Conditions for Equivalence (put this in the notebook as a Methods statement)
Assume the rank-1 Gaussian noise model:
\[
x = a\,s + n,\qquad n \sim \mathcal{N}(0,\Sigma)
\]
Equivalence requires:

- **(E1) Rank-1 signal model**: pulses differ only by amplitude (and possibly a known shift handled consistently).
- **(E2) Same noise weighting**: both methods use the same \(\Sigma^{-1}\) (or the same PSD \(J(f)\)).
- **(E3) EMPCA uses k=1** and learns a basis \(u\) proportional to the OF template \(s\) (up to scale/sign).
- **(E4) Amplitude definition matches ML coefficient** (GLS projection under \(\Sigma^{-1}\)).

Under these conditions, both reduce to the same ML/GLS estimator:
\[
\hat a \;=\; \frac{s^\top\Sigma^{-1}x}{s^\top\Sigma^{-1}s}
\]

---

### A3. Implementation Plan
#### A3.1 OF amplitude
Compute \(\hat a^{OF}\) for each `x` using template `s` and PSD `J(f)`.

#### A3.2 EMPCA(k=1) training
Train EMPCA on `X_train` with **k=1**, obtaining a single basis vector \(u\).

#### A3.3 EMPCA amplitude (GLS coefficient)
For each test trace `x`, compute the coefficient:
\[
\hat a^{EMPCA} \;=\; (u^\top\Sigma^{-1}u)^{-1}\,u^\top\Sigma^{-1}x
\]
If your EMPCA code returns coefficients in a different normalization, calibrate a single global scale factor on training set and apply it to test.

---

### A4. Required Empirical Tests (paper-grade)
#### Test 1 — Template alignment (u vs s)
Compute cosine similarity (time-domain) and optionally whitened similarity:
\[
\rho = \frac{|u^\top s|}{\|u\|\|s\|}
\]
For colored noise, also compute \(\rho_w\) with whitened vectors.

**Deliverable:** report \(\rho\) (and optionally \(\rho_w\)).

---

#### Test 2 — Amplitude agreement (main equivalence result)
For test set, compare \(\hat a^{OF}\) vs \(\hat a^{EMPCA}\):

**Report:**
- regression fit \(\hat a^{EMPCA} \approx \alpha \hat a^{OF} + \beta\)
- Pearson correlation
- median relative error

**Deliverable:** scatter plot + fit parameters.

---

#### Test 3 — Weighted residual / χ² consistency
Compute residuals using the same reconstructed model:
- \(r^{OF} = x - \hat a^{OF} s\)
- \(r^{EMPCA} = x - \hat a^{EMPCA} u\)

Compute weighted residual energy (or χ² proxy):
\[
\chi^2 \propto r^\top \Sigma^{-1} r
\]
Compare distributions.

**Deliverable:** histogram/ECDF of χ² (OF vs EMPCA).

---

### A5. Failure Mode (must include at least one)
Provide one controlled demonstration where equivalence breaks:

- **PSD mismatch:** use wrong PSD for one method (e.g., white instead of MMC)  
- **shape variability:** inject rise/decay jitter (rank>1 structure)  
- **misalignment:** add random shifts without correcting them

**Deliverable:** show amplitude agreement degrades (scatter widens; errors increase).

---

### A6. Noise Ladder (optional but recommended)
Repeat Tests 1–3 under:
1) noise-free (optional)
2) white noise matched RMS
3) pink noise matched RMS
4) MMC calibration noise

**Goal:** show equivalence is robust under consistent weighting, and identify breakdowns.

---

### A7. Notebook Outputs (to save)
- `u_empca_k1.npy` (basis)
- `a_of.npy`, `a_empca.npy`
- metrics JSON: `{rho, alpha, beta, corr, med_rel_err, chi2_stats}`
- plots (PDF/PNG)

---

---

## Notebook B — EMPCA ⇔ Linear Autoencoder (Noise-Aware, Sum-Channel)

### Title
**Equivalence of EMPCA and a Noise-Aware Linear Autoencoder via Whitening**

### Goal
Show that EMPCA is equivalent to a **linear autoencoder** trained to minimize the **noise-weighted reconstruction error**, implemented cleanly by training in **whitened space**.

This notebook should produce:
1) a clear equivalence statement: EMPCA ⇔ PCA in whitened space ⇔ linear AE in whitened space  
2) subspace equivalence (principal angles)  
3) weighted reconstruction error agreement on held-out data  
4) ablations: wrong PSD / rank sweep / training size sweep

---

### B1. Equivalence Statement (Methods)
Define weighted reconstruction error:
\[
\|x-\hat x\|_{\Sigma^{-1}}^2 = (x-\hat x)^\top\Sigma^{-1}(x-\hat x)
\]
Let \(L\) satisfy \(L^\top L = \Sigma^{-1}\). Define whitened data \(y = Lx\). Then:
\[
\|x-\hat x\|_{\Sigma^{-1}}^2 = \|y-\hat y\|_2^2
\]
Therefore:
- EMPCA in \(x\)-space is equivalent to ordinary PCA in whitened \(y\)-space.
- A **linear autoencoder** trained on \(y\) with MSE learns the same rank-k subspace (up to rotation).

---

### B2. Inputs (sum-channel)
- `X_train`, `X_test`: traces `(N, T)`
- PSD `J(f)` from MMC calibration noise
- rank `k` (e.g. k ∈ {1,2,4,8})
- EMPCA implementation returning basis `U_empca` (T×k)

---

### B3. Whitening Definition (frequency-domain; recommended)
Using PSD \(J(f)\), define whitening operator \(L\) implicitly:
- \(X(f) \leftarrow \mathrm{rFFT}(x)\)
- \(Y(f) = X(f) / \sqrt{J(f)}\)
- \(y = \mathrm{irFFT}(Y)\)

This implements \(y = Lx\) for diagonal Σ in frequency domain.

---

### B4. Linear Autoencoder Model (rigorous, PCA-like)
Train on whitened traces \(y\) with a **tied-weight linear AE**:

- Encoder: \(z = W^\top y\)
- Decoder: \(\hat y = W z = WW^\top y\)

Where:
- \(W\in\mathbb{R}^{T\times k}\)
- optionally enforce \(W^\top W = I\) (orthonormal columns) via QR retraction each step

Loss:
\[
\min_W \;\sum_i \|y_i - WW^\top y_i\|_2^2
\]
This is a direct, defensible linear representation learner.

---

### B5. What to Compare (EMPCA vs AE)
#### Step 1 — Train EMPCA (rank k)
Obtain `U_empca` from `X_train`.

#### Step 2 — Map EMPCA basis to whitened space
Compute `U_w = L U_empca`, implemented by whitening each column of `U_empca` the same way as traces.

#### Step 3 — Train AE on whitened train data
Obtain `W` (decoder subspace) from the saved model weights.

---

### B6. Required Tests (paper-grade)
#### Test 1 — Subspace equivalence (principal angles)
Compare span(`U_w`) vs span(`W`) using principal angles \(\theta_1,\dots,\theta_k\).

**Deliverable:** table/plot of principal angles (near 0 indicates equivalence up to rotation).

---

#### Test 2 — Weighted reconstruction error agreement
Compute on test set:
- EMPCA reconstruction: \(\hat x_{empca}\)
- AE reconstruction in x-space: \(\hat x_{ae} = L^{-1}\hat y\) (or compare errors in whitened space directly)

Compute weighted error:
\[
E = \|x-\hat x\|_{\Sigma^{-1}}^2
\]
Compare distributions and summary stats.

**Deliverable:** ECDF/hist of errors + mean/median/p90.

---

#### Test 3 — Robustness / Ablations (minimum set)
- **Wrong PSD whitening:** use \(\tilde{J}(f)\), show angles increase, errors increase.
- **Rank sweep:** k ∈ {1,2,4,8}, show error decreases/saturates.
- **Training size sweep:** show subspace stability and error stability.

**Deliverable:** 1–2 concise plots.

---

### B7. Recommended Noise Ladder for AE/EMPCA comparison
Repeat tests under:
- white noise (Σ ∝ I) → AE ≈ PCA ≈ EMPCA
- pink noise (colored) → whitening becomes crucial
- MMC calibration noise → realistic case + robustness

---

### B8. Notebook Outputs (to save)
- `U_empca.npy`, `U_w.npy`
- AE checkpoint `best.pt` + config JSON
- principal angles report JSON
- reconstruction error report JSON
- plots (PDF/PNG)

---

---

## Notes for Agent Implementation (strict requirements)
- Sum-channel only: all inputs must be (N,T).
- Whitening must use the **same FFT conventions** for traces and basis vectors.
- All results must be computed on **held-out test set**.
- PSD must be estimated from noise-only calibration segments independent of the pulse test set.
- Prefer saving all intermediate artifacts to disk for reproducibility.

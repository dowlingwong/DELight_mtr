# Section 7 Restructure Plan (for LaTeX integration)

## 7. EMPCA as a Noise-Aware Linear Autoencoder (Two Equivalent Formulations)

**Goal:** Present equivalence in both:
1. whitened complex space, and
2. native (no-prewhiten) PSD-weighted space with metric-aware normalization.

This should replace the current single-path Section 7 narrative.

---

## 7.1 Setup, assumptions, and notation

Keep this concise and strict.

- Data in frequency domain: \(X \in \mathbb{C}^{N \times d}\), row = event.
- PSD-derived diagonal weighting matrix \(D_w = \mathrm{diag}(w)\), with one-sided OF convention.
- Rank \(k\) linear subspace model.
- Matched preprocessing and train/test split across all compared methods.
- Equivalence is up to gauge (phase/sign/rotation for bases).

**Assumptions for rigor:**
- linear models only,
- Gaussian noise likelihood (weighted quadratic objective),
- matched weighting and rank \(k\).

---

## 7.2 Whitened complex formulation (existing derivation, tightened)

Use current derivation with explicit projector form:

\[
\tilde X = X D_w^{1/2}
\]

Tied linear AE in whitened space:

\[
Z = \tilde X W,\qquad \hat{\tilde X} = ZW^H = \tilde X W W^H,\qquad W^H W = I.
\]

Objective:

\[
\min_{W^H W=I}\ \|\tilde X - \tilde X W W^H\|_F^2.
\]

State that minimizer spans the top-\(k\) right singular subspace of \(\tilde X\), hence equivalent to complex PCA / EMPCA in whitened space.

---

## 7.3 Native weighted formulation (new subsection)

Add a second, explicit proof path with **no prewhitening**.

Define native weighted objective:

\[
\min_W \sum_i \|x_i - \hat x_i\|_{D_w}^2
\]
with metric-aware orthonormality:
\[
W^H D_w W = I.
\]

Use weighted projector and weighted inner product \(\langle a,b\rangle_{D_w}=a^H D_w b\).

Show change-of-variables relation:
\[
\tilde W = D_w^{1/2} W,\qquad W = D_w^{-1/2}\tilde W
\]
(with safe handling of zero-weight bins in implementation).

Conclude: native weighted tied linear AE is equivalent to whitened tied linear AE and weighted EMPCA.

---

## 7.4 Bridge theorem (new compact theorem/proposition)

Add one proposition:

> Under the assumptions in 7.1, the following rank-\(k\) optimization problems are equivalent (same optimum subspace up to unitary/gauge transform):
> 1. weighted EMPCA in native space,
> 2. tied linear AE in whitened complex space,
> 3. tied linear AE in native weighted space with \(W^H D_w W=I\).

Proof sketch: objective-preserving variable transform via \(D_w^{1/2}\), and equivalent rank-\(k\) projector constraints.

---

## 7.5 Numerical verification (split current 7.3 into two tables)

Replace one mixed table with two subtables.

### Table 2A — Whitened formulation equivalence (EMPCA vs exact complex AE)

From `section_A_whitened`:
- Principal-angle cosine: **0.9999999832**
- Principal angle (deg): **0.0105078°**
- Residual KS statistic: **0.00137678**
- Residual KS p-value: **1.0**
- Relative mean residual difference: **8.19 × 10⁻⁷**

### Table 2B — Native weighted formulation equivalence

From `section_B_native_weighted`:

**EMPCA vs exact weighted baseline**
- Principal-angle cosine: **0.9999999999999993**
- Principal angle (deg): **2.09 × 10⁻⁶**
- Residual KS statistic: **0.000458926**
- Residual KS p-value: **1.0**
- Relative mean residual difference: **2.36 × 10⁻¹⁶**

**Iterative weighted AE vs exact weighted baseline**
- Principal-angle cosine: **0.9999999999999994**
- Principal angle (deg): **1.91 × 10⁻⁶**
- Residual KS statistic: **0.000458926**
- Residual KS p-value: **1.0**
- Relative mean residual difference: **0.0**

Add interpretation sentence:
- Section A shows very strong practical equivalence.
- Section B reaches near machine-precision agreement, strongly validating the no-prewhiten weighted formulation.

---

## 7.6 Scope and limits (keep current 7.4, with one addition)

Retain current bullets:
- linear subspace models,
- Gaussian likelihood,
- matched weighting/preprocessing/rank.

Add:
- Real-feature rank-\(k\) comparisons may differ due to representation mismatch; this does not contradict complex weighted equivalence.

---

## Editorial notes for agent

- Keep Section 8 (“Numerical performance of EMPCA”) separate from equivalence logic.
- Move long diagnostic discussion (real-feature mismatch) to 7.6 or appendix.
- Ensure symbols \(D_w\), weighted inner product, and gauge equivalence are defined once and reused.
- If theorem environment exists, place 7.4 in `Proposition` + short proof.

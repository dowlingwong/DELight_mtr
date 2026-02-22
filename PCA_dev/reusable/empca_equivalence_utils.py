import numpy as np

import reusable.empca_TCY_optimized as empca_mod


def baseline_correct(X, pretrigger=4000):
    X = np.asarray(X, dtype=np.float64)
    if pretrigger <= 0:
        return X
    return X - np.mean(X[:, :pretrigger], axis=1, keepdims=True)


def build_of_one_sided_weights(J_psd, trace_len):
    J = np.asarray(J_psd, dtype=np.float64)
    n_rfft = trace_len // 2 + 1
    if J.shape[0] != n_rfft:
        raise ValueError(f"PSD length {J.shape[0]} does not match rfft bins {n_rfft}")

    w = np.zeros_like(J)
    w[0] = 0.0
    if trace_len % 2 == 0:
        w[1:-1] = 2.0 / J[1:-1]
        w[-1] = 1.0 / J[-1]
    else:
        w[1:] = 2.0 / J[1:]
    return w


def weighted_inner(a, b, w):
    return np.sum(np.conj(a) * b * w)


def weighted_cosine(a, b, w):
    num = np.abs(weighted_inner(a, b, w))
    den = np.sqrt(np.real(weighted_inner(a, a, w))) * np.sqrt(np.real(weighted_inner(b, b, w)))
    return float(num / den)


def phase_align_basis(u, s_ref, w):
    overlap = weighted_inner(u, s_ref, w)
    phase = np.angle(overlap)
    u_aligned = u * np.exp(-1j * phase)
    if np.real(weighted_inner(u_aligned, s_ref, w)) < 0:
        u_aligned = -u_aligned
    return u_aligned


def project_gls(X_f, basis_f, w, return_complex=False):
    X_f = np.asarray(X_f)
    basis_f = np.asarray(basis_f)
    w = np.asarray(w)

    if basis_f.ndim == 1:
        den = weighted_inner(basis_f, basis_f, w)
        num = np.sum(np.conj(basis_f)[None, :] * X_f * w[None, :], axis=1)
        coeff = num / den
        return coeff if return_complex else np.real(coeff)

    if basis_f.ndim == 2:
        gram = (basis_f * w[None, :]) @ basis_f.conj().T
        rhs = (X_f * w[None, :]) @ basis_f.conj().T
        coeff = np.linalg.solve(gram, rhs.T).T
        return coeff if return_complex else np.real(coeff)

    raise ValueError(f"basis_f must be 1D or 2D, got ndim={basis_f.ndim}")


def weighted_residual_energy(X_f, basis_f, coeff, w):
    X_f = np.asarray(X_f)
    basis_f = np.asarray(basis_f)
    coeff = np.asarray(coeff)
    w = np.asarray(w)

    if basis_f.ndim == 1:
        residual = X_f - coeff[:, None] * basis_f[None, :]
    elif basis_f.ndim == 2:
        residual = X_f - coeff @ basis_f
    else:
        raise ValueError(f"basis_f must be 1D or 2D, got ndim={basis_f.ndim}")

    return np.real(np.sum((np.abs(residual) ** 2) * w[None, :], axis=1))


def fit_empca_no_smoothing(X_f, w, n_comp=1, n_iter=50, patience=8, mode="fast"):
    solver = empca_mod.empca_solver(n_comp, np.asarray(X_f), np.asarray(w))
    chi2s = []
    best = np.inf
    stale = 0

    for _ in range(n_iter):
        solver.eigvec = empca_mod.orthonormalize(solver.solve_eigvec(mode=mode))
        solver.coeff = solver.solve_coeff()
        chi2 = solver.chi2()
        chi2s.append(chi2)

        if chi2 + 1e-12 < best:
            best = chi2
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    return solver.eigvec.copy(), solver.coeff.copy(), np.array(chi2s, dtype=np.float64)


def rfft_to_weighted_real_features(X_f, w):
    X_f = np.asarray(X_f)
    w = np.asarray(w, dtype=np.float64)

    squeeze = False
    if X_f.ndim == 1:
        X_f = X_f[None, :]
        squeeze = True

    if X_f.shape[1] != w.shape[0]:
        raise ValueError(f"X_f bins {X_f.shape[1]} != weight bins {w.shape[0]}")

    X_w = X_f * np.sqrt(w)[None, :]
    feat = np.concatenate(
        [
            X_w.real[:, :1],
            X_w.real[:, 1:],
            X_w.imag[:, 1:],
        ],
        axis=1,
    )
    return feat[0] if squeeze else feat

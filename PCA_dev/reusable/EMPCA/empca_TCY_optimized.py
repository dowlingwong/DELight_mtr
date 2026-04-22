# EMPCA by To Chin Yu (optimized diagonal-weight implementation)
#
# Key changes vs. original:
# - Supports diagonal weights efficiently: pass a 1D weight vector, a dense diagonal matrix, or a sparse diagonal matrix.
# - Avoids materializing huge (n_var x n_var) dense diagonal matrices during training.
# - Vectorizes coefficient solve, eigvec solve (fast mode), and chi2 for diagonal weights.

import numpy as np
import scipy
from scipy.signal import savgol_filter
from tqdm import tqdm

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


def polar(x):
    return np.absolute(x), np.angle(x)


def rect(r, theta):
    return r * np.exp(1j * theta)


def ti_rfft(x, truncate=None, axis=-1):
    if x.ndim == 1:
        x = x.reshape([1, -1])
    end_ind = None if truncate is None else -truncate
    r, theta = polar(np.fft.rfft(x, axis=axis)[:, :end_ind])
    theta_shifted = np.roll(theta, -1, axis=axis)
    # keep last component unchanged for inverse transform
    if theta.ndim == 1:
        theta_shifted[-1] = 0
    else:
        theta_shifted[:, -1] = 0
    theta = theta - theta_shifted
    return np.squeeze(rect(r, theta))


def ti_irfft(x, padding=0, axis=-1):
    if x.ndim == 1:
        x = x.reshape([1, -1])

    r, theta = polar(np.concatenate((x, np.zeros([x.shape[0], padding], dtype=complex)), axis=-1))
    theta = np.flip(np.cumsum(np.flip(theta, axis=axis), axis=axis), axis=axis)
    return np.squeeze(np.fft.irfft(rect(r, theta), axis=axis))


def orthonormalize(A):
    n_comp, _ = A.shape
    for i in range(n_comp):
        nrm = np.linalg.norm(A[i])
        if nrm == 0:
            continue
        A[i] /= nrm

    for i in range(1, n_comp):
        for j in range(0, i):
            A[i] -= np.dot(np.conjugate(A[j]), A[i]) * A[j]
            nrm = np.linalg.norm(A[i])
            if nrm == 0:
                continue
            A[i] /= nrm
    return A


def random_orthonormals(n_comp, n_var, seed=1):
    if seed is not None:
        np.random.seed(seed)
    return orthonormalize(np.random.normal(size=(n_comp, n_var)))


def smooth(A, window=15, polyord=3, deriv=1):
    # Smooth real/imag independently (same as original behavior)
    return savgol_filter(np.real(A), window, polyord, deriv) + 1j * savgol_filter(
        np.imag(A), window, polyord, deriv
    )


def _extract_diag_weights(weights, n_var=None):
    """
    Return a 1D weight vector w (length n_var) if weights are diagonal-like.
    Supports:
      - 1D ndarray/list
      - dense 2D diagonal ndarray
      - scipy.sparse diagonal matrices (if scipy.sparse available)
    Returns:
      (w, is_diag) where w is np.ndarray(float/complex) or None if not diagonal.
    """
    if weights is None:
        return None, False

    # 1D vector
    if isinstance(weights, (list, tuple, np.ndarray)) and np.asarray(weights).ndim == 1:
        w = np.asarray(weights)
        if n_var is not None and w.shape[0] != n_var:
            raise ValueError(f"Weight vector length {w.shape[0]} != n_var {n_var}")
        return w, True

    # scipy sparse
    if sp is not None and sp.issparse(weights):
        # Works for diagonal/any sparse; we assume caller means diagonal weights
        w = np.asarray(weights.diagonal())
        if n_var is not None and w.shape[0] != n_var:
            raise ValueError(f"Sparse weight diagonal length {w.shape[0]} != n_var {n_var}")
        return w, True

    # dense matrix
    W = np.asarray(weights)
    if W.ndim == 2:
        if n_var is not None and W.shape != (n_var, n_var):
            raise ValueError(f"Weight matrix shape {W.shape} != ({n_var},{n_var})")
        # check diagonal (cheap-ish)
        off = W.copy()
        np.fill_diagonal(off, 0)
        if np.all(off == 0):
            return np.diag(W), True

    return None, False


class empca_solver:
    def __init__(self, n_comp, data, weights):
        self.n_comp = int(n_comp)
        self.set_data(data)
        self.set_weights(weights)
        self.eigvec = random_orthonormals(self.n_comp, self.n_var)
        self.coeff = self.solve_coeff()

    def set_data(self, data):
        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2D (n_obs,n_var); got shape {self.data.shape}")
        self.n_obs, self.n_var = self.data.shape

    def set_weights(self, weights):
        self.weights = weights
        self.w_vec, self._weights_are_diag = _extract_diag_weights(weights, n_var=self.n_var)

    def chi2(self):
        residual = self.data - self.coeff @ self.eigvec
        if self._weights_are_diag:
            # mean over observations of sum_k w_k * |r_ik|^2
            w = self.w_vec
            # ensure broadcast works for complex
            return float(np.mean(np.sum((np.abs(residual) ** 2) * w[None, :], axis=1)))
        else:
            # fallback to original quadratic form
            return np.absolute(
                np.mean(np.sum(residual @ self.weights @ (np.conjugate(residual).T), axis=-1))
            )

    def solve_coeff(self, data=None):
        if data is None:
            data = self.data
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"data must be 2D; got shape {data.shape}")

        Phi = self.eigvec.T  # (n_var, n_comp)

        if self._weights_are_diag:
            w = np.asarray(self.w_vec)
            # Build small system A c = b for all observations at once:
            # A = Phi^H diag(w) Phi  (n_comp,n_comp)
            # B = X diag(w) Phi*     (n_obs,n_comp)
            A = Phi.conj().T @ (Phi * w[:, None])
            B = (data * w[None, :]) @ Phi.conj()

            # Solve A^T C^T = B^T -> C = (A^{-1} B^T)^T
            # Use Cholesky when possible; fallback to solve/lstsq.
            try:
                # A should be Hermitian positive-definite if w>0 and Phi full rank
                c_factor = scipy.linalg.cho_factor(A, lower=False, check_finite=False)
                C = scipy.linalg.cho_solve(c_factor, B.T, check_finite=False).T
            except Exception:
                try:
                    C = scipy.linalg.solve(A, B.T, check_finite=False).T
                except Exception:
                    C = scipy.linalg.lstsq(A, B.T, lapack_driver="gelsy", check_finite=False)[0].T
            return C

        # fallback (original approach) for non-diagonal weights
        WPd = np.conjugate(self.weights @ Phi).T
        return np.stack(
            [
                scipy.linalg.lstsq(WPd @ Phi, WPd @ X, lapack_driver="gelsy", check_finite=False)[0]
                for X in data
            ]
        )

    def solve_eigvec(self, data=None, mode="fast"):
        if mode.lower() == "fast":
            return self.solve_eigvec_fast(data)
        elif mode.lower() == "full":
            return self.solve_eigvec_full(data)
        else:
            raise ValueError("Mode should be 'fast' or 'full'.")

    def solve_eigvec_fast(self, data=None):
        if data is None:
            data = self.data
        data = np.asarray(data)

        # Diagonal weights allow a very efficient update.
        # Even more: for the specific "fast" normal equations in the original code,
        # the diagonal weights cancel, so the update is simply:
        #   v_i = (X^T c_i*) / sum|c_i|^2
        CC = np.sum(np.square(np.absolute(self.coeff)), axis=0)  # (n_comp,)
        # avoid division by zero: re-init any dead components
        dead = CC <= 0
        CC_safe = CC.copy()
        CC_safe[dead] = 1.0

        V = (data.T @ np.conjugate(self.coeff)) / CC_safe[None, :]  # (n_var, n_comp)
        eig = V.T  # (n_comp, n_var)

        if np.any(dead):
            eig[dead] = random_orthonormals(int(np.sum(dead)), self.n_var, seed=None)

        self.eigvec = eig
        return orthonormalize(self.eigvec)

    def solve_eigvec_full(self, data=None):
        # Original full solve retained for compatibility with non-diagonal weights.
        if data is None:
            data = self.data
        data = np.asarray(data)

        BigW = np.zeros([self.n_comp * self.n_var, self.n_comp * self.n_var], dtype=complex)
        for n in range(self.n_comp):
            for m in range(self.n_comp):
                BigW[n * self.n_var : (n + 1) * self.n_var, m * self.n_var : (m + 1) * self.n_var] = (
                    np.dot(np.conjugate(self.coeff[:, n]), self.coeff[:, m]) * self.weights
                )

        WXC = (self.weights @ (data.T) @ np.conjugate(self.coeff)).reshape([-1, 1], order="F")

        self.eigvec = scipy.linalg.lstsq(BigW, WXC, lapack_driver="gelsy", check_finite=False)[0]
        self.eigvec = self.eigvec.reshape([self.n_comp, -1])
        return orthonormalize(self.eigvec)


class EMPCA:
    def __init__(self, n_comp=5):
        self.n_comp = int(n_comp)
        self.solver = None

    def fit(
        self,
        X,
        weights,
        n_iter=50,
        window=15,
        polyord=3,
        deriv=0,
        patience=5,
        mode="fast",
        verbose=False,
    ):
        _patience = patience
        chi2s = []

        if self.solver is None:
            self.solver = empca_solver(self.n_comp, X, weights)
        else:
            self.solver.set_data(X)
            self.solver.set_weights(weights)

        for _ in tqdm(range(n_iter)):
            self.solver.eigvec = smooth(
                self.solver.solve_eigvec(mode=mode), window=window, polyord=polyord, deriv=deriv
            )
            self.solver.coeff = self.solver.solve_coeff()
            chi2 = self.solver.chi2()
            if verbose:
                print(f"chi2= {chi2}")

            # Original stopping logic preserved (stop after patience consecutive increases)
            if len(chi2s) > 0 and chi2 > chi2s[-1]:
                if patience <= 0:
                    break
                else:
                    patience -= 1

            chi2s.append(chi2)

        self.eigvec = self.solver.eigvec
        self.coeff = self.solver.coeff
        return chi2s

    def project(self, X):
        if self.solver is None:
            raise Exception("Solver has not been initialized. Please run fit() first.")
        return self.solver.solve_coeff(X)

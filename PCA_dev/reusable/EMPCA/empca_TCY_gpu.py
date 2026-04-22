# EMPCA by To Chin Yu (GPU optimized implementation using PyTorch)
#
# Key changes vs. optimized CPU version:
# - Uses PyTorch for GPU acceleration.
# - Weights are handled as tensors on the GPU.
# - Calculations (coefficients, eigenvectors, chi2) are performed on the device.
# - Smoothing is performed on CPU (numpy/scipy) for simplicity as it's not the bottleneck.

import numpy as np
import scipy.signal
from tqdm import tqdm
import torch

class empca_solver_gpu:
    def __init__(self, n_comp, data, weights, device=None):
        self.n_comp = int(n_comp)
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.set_data(data)
        self.set_weights(weights)
        
        # Initialize eigenvectors randomly on GPU
        # n_var should be set by set_data
        self.eigvec = self.random_orthonormals(self.n_comp, self.n_var)
        
        # Solve for initial coefficients
        self.coeff = self.solve_coeff()

    def set_data(self, data):
        # Expect data to be numpy array or torch tensor
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).to(self.device)
        elif isinstance(data, torch.Tensor):
            self.data = data.to(self.device)
        else:
            raise ValueError("Data must be numpy array or torch tensor")
            
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2D (n_obs,n_var); got shape {self.data.shape}")
            
        # Ensure correct type (float32 or float64 depending on precision needs, defaulting to float32 for speed)
        # Trying to maintain input precision or default to float32
        if self.data.dtype == torch.float64:
             pass # keep double
        else:
             self.data = self.data.float() # ensure float32 at least
             
        self.n_obs, self.n_var = self.data.shape

    def set_weights(self, weights):
        # Handle weights similar to optimized CPU version but convert to tensor
        w, is_diag = self._extract_diag_weights_gpu(weights, self.n_var)
        self.weights = w
        self._weights_are_diag = is_diag
        
        if not self._weights_are_diag:
             # If not diagonal, move full matrix to GPU
             if isinstance(weights, np.ndarray):
                 self.weights = torch.from_numpy(weights).to(self.device, dtype=self.data.dtype)
             elif isinstance(weights, torch.Tensor):
                 self.weights = weights.to(self.device, dtype=self.data.dtype)

    def _extract_diag_weights_gpu(self, weights, n_var):
        """
        Extract diagonal weights and move to GPU.
        """
        if weights is None:
            return None, False
            
        # 1D vector (numpy)
        if isinstance(weights, (list, tuple, np.ndarray)) and np.asarray(weights).ndim == 1:
            w = torch.from_numpy(np.asarray(weights)).to(self.device, dtype=self.data.dtype)
            return w, True
            
        # 1D vector (torch)
        if isinstance(weights, torch.Tensor) and weights.ndim == 1:
            return weights.to(self.device, dtype=self.data.dtype), True

        # Handle other cases (sparse, dense diagonal) - implementing basic check
        # Assuming for high perf usage, user provides 1D array. 
        # If dense 2D provided:
        if isinstance(weights, (np.ndarray, torch.Tensor)):
             w_np = np.asarray(weights) if isinstance(weights, np.ndarray) else weights.cpu().numpy()
             if w_np.ndim == 2:
                 # Check if diagonal
                 off_diag = w_np.copy()
                 np.fill_diagonal(off_diag, 0)
                 if np.all(off_diag == 0):
                      return torch.from_numpy(np.diag(w_np)).to(self.device, dtype=self.data.dtype), True
        
        return weights, False # Not diagonal or not handled as diagonal

    def random_orthonormals(self, n_comp, n_var, seed=1):
        if seed is not None:
             torch.manual_seed(seed)
        
        # Generate random normal on device
        A = torch.randn(n_comp, n_var, device=self.device, dtype=self.data.dtype)
        return self.orthonormalize(A)

    def orthonormalize(self, A):
        # Gram-Schmidt on GPU
        n_comp, _ = A.shape
        for i in range(n_comp):
            nrm = torch.norm(A[i])
            if nrm > 0:
                A[i] /= nrm
        
        for i in range(1, n_comp):
            # Orthogonalize against previous
            # Vectorized projection: A[i] = A[i] - sum_j (dot(A[j], A[i]) * A[j])
            # But doing iterative for stability as per original code
            for j in range(0, i):
                dot_prod = torch.dot(torch.conj(A[j]), A[i]) # conj handle complex if needed?
                # For real data: dot
                A[i] -= dot_prod * A[j]
                
                nrm = torch.norm(A[i])
                if nrm > 0:
                    A[i] /= nrm
        return A

    def chi2(self):
        # Calculate residual on GPU: data - coeff @ eigvec
        residual = self.data - self.coeff @ self.eigvec
        
        if self._weights_are_diag:
            # Vectorized weighted sum squares
            # w is 1D tensor
            # sum( |resid|^2 * w )
            # Broadcast w: (1, n_var)
            w = self.weights
            val = torch.sum((torch.abs(residual)**2) * w.unsqueeze(0), dim=1)
            return torch.mean(val).item()
        else:
            # Fallback for full weight matrix
            # residual @ weights @ residual.T
            # This is huge and likely OOMs on GPU for large data, but implementing for completeness
            # Compute diagonal terms only? No, chi2 is scalar mean.
            # term = (residual @ self.weights) * residual.conj() -> sum axis 1
            # But weights is (n_var, n_var).
            # (N_obs, N_var) @ (N_var, N_var) -> (N_obs, N_var)
            
            # Warn user?
            rw = residual @ self.weights
            val = torch.sum(rw * torch.conj(residual), dim=1)
             # Take real part (should be real)
            return torch.mean(torch.abs(val)).item()

    def solve_coeff(self, data=None):
        if data is None:
            data = self.data
        else:
            # Ensure data is on device
             if isinstance(data, np.ndarray):
                 data = torch.from_numpy(data).to(self.device, dtype=self.data.dtype)
             elif isinstance(data, torch.Tensor):
                 data = data.to(self.device)

        Phi = self.eigvec.T # (n_var, n_comp)

        if self._weights_are_diag:
            w = self.weights
            # A = Phi^H * w * Phi
            # A is (n_comp, n_comp) - efficient
            # Phi is (n_var, n_comp), w is (n_var)
            # Phi * w[:, None] scales rows of Phi
            
            # (n_comp, n_var) @ (n_var, n_comp)
            A = torch.matmul(Phi.conj().T, Phi * w.unsqueeze(1))
            
            # B = X * w * Phi*
            # X is (n_obs, n_var)
            # data * w[None, :] scales columns of data
            # (n_obs, n_var) @ (n_var, n_comp) -> (n_obs, n_comp)
            B = torch.matmul(data * w.unsqueeze(0), Phi.conj())
            
            # Solve A C^T = B^T  => C = (A^-1 B^T)^T = B (A^-1)^T
            # PyTorch linalg.solve(A, B) solves AX = B.
            # We want C such that C A = B (roughly).
            # Actually from derived math: A C^T = B.T (for each obs). 
            # Code says: A c = b.
            # So C_rows @ A = B_rows.
            # C = B @ inv(A). 
            
            # Check dimensions:
            # A is (n_comp, n_comp)
            # B is (n_obs, n_comp)
            # C is (n_obs, n_comp)
            # C @ A_transposed? Symmetric A?
            # Original code: A c = b -> c = solve(A, b).
            # Here b is B.T columns. 
            # solve(A, B.T) -> returns (n_comp, n_obs)
            # Transpose result -> (n_obs, n_comp)
            
            try:
                # Use cholesky if positive definite
                C_T = torch.linalg.solve(A, B.T)
                return C_T.T
            except RuntimeError:
                 # Fallback to lstsq
                C_T = torch.linalg.lstsq(A, B.T).solution
                return C_T.T
                
        else:
            # Non-diagonal weights - slow path
            # WPd = (Weights @ Phi).conj().T
            WPd = torch.matmul(self.weights, Phi).conj().T # (n_comp, n_var)
            
            # Loop over data is slow in Python, but torch.linalg.lstsq supports batching?
            # A = WPd @ Phi # (n_comp, n_comp)
            # RHS = WPd @ data.T # (n_comp, n_obs)
            
            A = torch.matmul(WPd, Phi)
            RHS = torch.matmul(WPd, data.T)
            
            C_T = torch.linalg.lstsq(A, RHS).solution
            return C_T.T


    def solve_eigvec(self, data=None, mode='fast'):
        if mode == 'fast':
             return self.solve_coeff_eigvec_fast(data)
        # Full mode not prioritized for GPU, fallback to standard or implement if needed
        # Assuming fast mode for now as per efficiency goals
        return self.solve_coeff_eigvec_fast(data)

    def solve_coeff_eigvec_fast(self, data=None):
        if data is None:
            data = self.data
        
        # CC = sum(|coeff|^2)
        CC = torch.sum(torch.abs(self.coeff)**2, dim=0) # (n_comp,)
        
        dead = CC <= 1e-10
        CC_safe = CC.clone()
        CC_safe[dead] = 1.0
        
        # V = (X.T @ coeff*) / CC
        # (n_var, n_obs) @ (n_obs, n_comp) -> (n_var, n_comp)
        
        V = torch.matmul(data.T, torch.conj(self.coeff)) / CC_safe.unsqueeze(0)
        eig = V.T # (n_comp, n_var)
        
        # Re-init dead components
        if torch.any(dead):
             # Generate on CPU or GPU? GPU.
             n_dead = torch.sum(dead).item()
             if n_dead > 0:
                 eig[dead] = self.random_orthonormals(int(n_dead), self.n_var, seed=None)
        
        self.eigvec = eig
        return self.orthonormalize(self.eigvec)

def smooth_cpu(A, window=15, polyord=3, deriv=1):
    # Perform smoothing on CPU using scipy
    # A is torch tensor on GPU or CPU
    is_tensor = isinstance(A, torch.Tensor)
    if is_tensor:
         A_np = A.cpu().numpy()
         device = A.device
         original_dtype = A.dtype
    else:
         A_np = A
    
    # Apply savgol on real and imag parts
    res_real = scipy.signal.savgol_filter(np.real(A_np), window, polyord, deriv)
    res_imag = scipy.signal.savgol_filter(np.imag(A_np), window, polyord, deriv)
    
    res = res_real + 1j * res_imag
    
    # Return as tensor
    if is_tensor:
         # If the result is complex but original was real, we have two choices:
         # 1. Cast back to real (discarding imag, which might be 0 anyway)
         # 2. Return complex (changing dtype)
         # The original code works with complex numbers potentially.
         # If we started with float32, we probably want to stay float32 if imag is negligible.
         
         # However, if imag is mathematically significant, valid EMPCA might need it.
         # Let's check if we strictly need to preserve complex.
         # If original dtype was real, and res has 0 imaginary, we can cast back.
         
         if not np.iscomplexobj(A_np) and not torch.is_complex(torch.tensor(0, dtype=original_dtype)):
             # Original was real.
             # If result has negligible imaginary part, cast to real to supress warning.
             if np.allclose(res.imag, 0, atol=1e-7):
                 return torch.from_numpy(res.real).to(device, dtype=original_dtype)
             else:
                 # It became complex? Promote to complex tensor.
                 # E.g. float32 -> complex64
                 if original_dtype == torch.float32:
                     target_dtype = torch.complex64
                 else:
                     target_dtype = torch.complex128
                 return torch.from_numpy(res).to(device, dtype=target_dtype)
         
         # If original was complex, just returning res (which is complex) is fine.
         # But we should respect precision (32 vs 64)
         return torch.from_numpy(res).to(device) # Let torch infer or use safe cast?

    return res

class EMPCA_GPU:
    def __init__(self, n_comp=5, device=None):
        self.n_comp = int(n_comp)
        self.solver = None
        self.device = device

    def fit(self, X, weights, n_iter=50, window=15, polyord=3, deriv=0, patience=5, mode='fast', verbose=False):
        _patience = patience
        chi2s = []
        
        if self.solver is None:
            self.solver = empca_solver_gpu(self.n_comp, X, weights, device=self.device)
        else:
            self.solver.set_data(X)
            self.solver.set_weights(weights)
            
        for _ in tqdm(range(n_iter)):
            # Warning: smoothing happens on CPU, involves transfer
            eig_new = self.solver.solve_eigvec(mode=mode)
            
            # Smooth
            self.solver.eigvec = smooth_cpu(eig_new, window=window, polyord=polyord, deriv=deriv)
            
            # Solve coeff
            self.solver.coeff = self.solver.solve_coeff()
            
            chi2 = self.solver.chi2()
            if verbose:
                print(f'chi2= {chi2}')
            
            if len(chi2s) > 0 and chi2 > chi2s[-1]:
                if patience <= 0:
                    break
                else:
                    patience -= 1
            chi2s.append(chi2)
            
        # Store results in CPU numpy format for compatibility
        self.eigvec = self.solver.eigvec.cpu().numpy()
        self.coeff = self.solver.coeff.cpu().numpy()
        return chi2s

    def project(self, X):
        if self.solver is None:
             raise Exception('Solver has not been initialized.')
        
        # Project new data using GPU solver
        # Return numpy array
        coeffs = self.solver.solve_coeff(X)
        return coeffs.cpu().numpy()

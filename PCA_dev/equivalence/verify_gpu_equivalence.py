
import numpy as np
import sys
import os
import torch

# Add path to import modules
sys.path.append(os.getcwd())

from reusable.empca_TCY_optimized import EMPCA as EMPCA_CPU
from reusable.empca_TCY_gpu import EMPCA_GPU

def verify():
    print("Verifying GPU EMPCA equivalence...")
    
    # 1. Generate synthetic data
    n_obs = 1000
    n_var = 500
    n_comp = 5
    
    np.random.seed(42)
    # Random structured data: A * B
    A = np.random.randn(n_obs, n_comp)
    B = np.random.randn(n_comp, n_var)
    data = np.dot(A, B) + 0.1 * np.random.randn(n_obs, n_var) # Add noise
    
    # Random weights (diagonal)
    weights = np.random.rand(n_var) + 0.1
    
    # 2. Run CPU EMPCA
    print("Running CPU EMPCA...")
    empca_cpu = EMPCA_CPU(n_comp=n_comp)
    # Set seed for reproducibility inside? The classes use random init.
    # To compare exactly, we should ideally inject the same initial eigenvectors.
    # But EMPCA converges to same subspace usually. Let's try to set seed or init manually.
    
    # Initial random guess
    init_eigvec = np.random.randn(n_comp, n_var)
    
    # To force same start, we might need to hack or rely on robust convergence.
    # Let's rely on convergence first.
    chi2_cpu = empca_cpu.fit(data, weights, n_iter=20, verbose=False, patience=0)
    
    # 3. Run GPU EMPCA
    print("Running GPU EMPCA...")
    empca_gpu = EMPCA_GPU(n_comp=n_comp)
    
    # Manually seed torch to match?
    # It's hard to match random generation exactly between numpy and torch distributions unless we pass the values.
    # We can pass the same initial eigenvectors if we modify the code or just set them after init?
    # empca_gpu.solver.eigvec = torch.from_numpy(init_eigvec).cuda() ...
    
    # For now, let's just see if they converge to similar Chi2 and reconstruction error.
    chi2_gpu = empca_gpu.fit(data, weights, n_iter=20, verbose=False, patience=0) # wait, GPU code I wrote has CPU calls, should work
    
    print(f"Final Chi2 CPU: {chi2_cpu[-1]:.6f}")
    # Chi2 GPU returns a list of values? Yes.
    # But wait, my GPU code returns chi2s.
    # The return from fit might fail if I didn't verify return type.
    # Assuming it returns list of floats.
    print(f"Final Chi2 GPU: {chi2_gpu[-1]:.6f}")
    
    # Verify reconstruction error
    # CPU
    coeff_cpu = empca_cpu.project(data)
    recon_cpu = coeff_cpu @ empca_cpu.eigvec
    resid_cpu = data - recon_cpu
    w_cpu = weights
    err_cpu = np.mean(np.sum(resid_cpu**2 * w_cpu, axis=1))
    
    # GPU
    # project returns numpy
    coeff_gpu = empca_gpu.project(data)
    # eigvec back to cpu
    eigvec_gpu_np = empca_gpu.eigvec
    recon_gpu = coeff_gpu @ eigvec_gpu_np
    resid_gpu = data - recon_gpu
    err_gpu = np.mean(np.sum(resid_gpu**2 * w_cpu, axis=1))
    
    print(f"Reconstruction Error CPU: {err_cpu:.6f}")
    print(f"Reconstruction Error GPU: {err_gpu:.6f}")
    
    diff = abs(err_cpu - err_gpu)
    print(f"Difference: {diff:.6e}")
    
    if diff < 1e-4:
        print("SUCCESS: GPU implementation matches CPU performance.")
    else:
        print("WARNING: Divergence detected.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        verify()
    else:
        print("Skipping verification (No CUDA)")


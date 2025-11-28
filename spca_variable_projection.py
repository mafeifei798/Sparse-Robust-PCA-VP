"""
Sparse Principal Component Analysis via Variable Projection.
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def soft_threshold(X, threshold):
    """Soft thresholding operator - proximal operator for L1 regularization"""
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0.0)


def update_A_procrustes(X, B):
    """
    Update A: solve orthogonal Procrustes problem
    A = argmin_A ||X - XBA^T||_F^2 s.t. A^T A = I
    
    Solution: A = UV^T, where X^T XB = UΣV^T (SVD)
    """
    Z = X.T @ X @ B  # X^T X B
    U, _, Vt = linalg.svd(Z, full_matrices=False)
    A = U @ Vt
    return A


def compute_nonstationarity(X, A, B, gamma, alpha):
    """
    Compute nonstationarity criterion T(A,B)
    Here we use the norm of the proximal gradient mapping as a simplified version
    """
    # Compute gradient ∇v(B) = X^T(XBA^T - X)A
    XB = X @ B
    residual = XB @ A.T - X
    gradient = X.T @ (residual @ A)
    
    # Proximal gradient mapping
    B_prox = soft_threshold(B - gamma * gradient, gamma * alpha)
    mapping = (B - B_prox) / gamma
    
    return linalg.norm(mapping, 'fro')


def spca_vp(X, r, alpha=0.1, max_iter=100, eps=1e-6, gamma=None, adaptive_alpha=False):
    """
    Variable-projection proximal gradient method for SPCA
    
    Parameters:
    -----------
    X : ndarray, shape (m, n)
        Input data matrix (centered)
    r : int 
        Number of principal components
    alpha : float
        L1 regularization parameter
    max_iter : int
        Maximum number of iterations  
    eps : float
        Convergence tolerance
    gamma : float, optional
        Step size, default is 1/||X||₂²
        
    Returns:
    --------
    A : ndarray, shape (n, r)
        Orthogonal matrix, A^T A = I
    B : ndarray, shape (n, r) 
        Sparse loadings matrix
    objectives : list
        Sequence of objective function values
    criteria : list
        Sequence of nonstationarity criteria
    """
    
    m, n = X.shape
    
    # Set step size γ = 1/||X||₂² (Lipschitz constant)
    if gamma is None:
        sigma_max = linalg.norm(X, ord=2)  # Maximum singular value
        gamma = 1.0 / (sigma_max**2) if sigma_max > 0 else 1.0
    
    # Initialize: use left singular vectors V of X
    _, _, Vt = linalg.svd(X, full_matrices=False)
    V = Vt.T
    A = V[:, :r].copy()  # A₀ = V
    B = V[:, :r].copy()  # B₀ = V
    
    print(f"VP-SPCA - Variable Projection")
    print(f"Data: {m}×{n}, Components: {r}, α={alpha}, γ={gamma:.6f}")
    print("-" * 60)
    
    objectives = []
    criteria = []
    
    # Main loop
    for k in range(max_iter):
        
        # Step 2a: Update B - proximal gradient step
        # Compute gradient: ∇v(B) = X^T(XBA^T - X)A
        XB = X @ B
        residual = XB @ A.T - X
        gradient = X.T @ (residual @ A)
        
        # Proximal gradient update: Bₖ₊₁ ← prox_γr(Bₖ - γ∇v(Bₖ)) with optional adaptive alpha
        if adaptive_alpha:
            # Mild, stable adaptation: scale with normalized residual magnitude
            res = residual  # XBA^T - X from above
            res_norm = linalg.norm(res, 'fro') / (linalg.norm(X, 'fro') + 1e-12)
            # Clamp within [0.5, 1.0] to avoid over-shrink or explosion
            scale = float(np.clip(res_norm, 0.5, 1.0))
            alpha_eff = alpha * scale
        else:
            alpha_eff = alpha
        B = soft_threshold(B - gamma * gradient, gamma * alpha_eff)
        
        # Step 2b: Update A - orthogonal Procrustes solution
        # Aₖ₊₁ ← argmin_A ||X - XBₖ₊₁A^T||_F^2 s.t. A^T A = I
        A = update_A_procrustes(X, B)
        
        # Compute objective function
        residual = X - X @ B @ A.T
        obj = 0.5 * linalg.norm(residual, 'fro')**2 + alpha_eff * np.sum(np.abs(B))
        objectives.append(obj)
        
        # Compute nonstationarity criterion T(A,B)
        criterion = compute_nonstationarity(X, A, B, gamma, alpha)
        criteria.append(criterion)
        
        # Print progress
        if k % 100 == 0 or k < 10:
            print(f"Iter {k:4d}: Obj = {obj:.6e}, T(A,B) = {criterion:.6e}")
        
        # Convergence check
        if criterion <= eps:
            print(f"✓ Converged at iteration {k}: T(A,B) = {criterion:.6e} ≤ {eps}")
            break
    
    else:
        print(f"✗ Reached max iterations {max_iter}")
    
    # Final statistics
    sparsity = (np.abs(B) < 1e-10).mean()
    ortho_err = linalg.norm(A.T @ A - np.eye(r), 'fro')
    recon_err = linalg.norm(X - X @ B @ A.T, 'fro') / linalg.norm(X, 'fro')
    
    print("-" * 60)
    print(f"Final Results:")
    print(f"  Sparsity ratio: {sparsity:.3f}")
    print(f"  Orthogonality error ||A^T A - I||_F: {ortho_err:.3e}")
    print(f"  Relative reconstruction error: {recon_err:.4f}")
    
    return A, B, objectives, criteria


def demo():
    """Demo for SPCA via variable projection"""
    
    # Generate synthetic data
    np.random.seed(42)
    m, n, r_true = 100, 30, 2
    
    # Create low-rank + sparse structure
    U_true = linalg.orth(np.random.randn(m, r_true))
    V_true = linalg.orth(np.random.randn(n, r_true))
    S_true = np.diag([8, 4])  # Singular values
    
    X_clean = U_true @ S_true @ V_true.T
    X = X_clean + 0.1 * np.random.randn(m, n)  # Add noise
    X = X - X.mean(axis=0, keepdims=True)  # Center the data
    
    print("=== VP-SPCA Demo ===")
    print(f"Data: {m}×{n}, True rank: {r_true}")
    print(f"||X||_F = {linalg.norm(X, 'fro'):.2f}")
    print()
    
    # Run VP-SPCA
    r = 2
    alpha = 10
    A, B, objectives, criteria = spca_vp(
        X, r=r, alpha=alpha, max_iter=500, eps=1e-8
    )
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Objective function
    axes[0,0].semilogy(objectives)
    axes[0,0].set_title('Objective Function')
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].grid(True)
    
    # Nonstationarity criterion
    axes[0,1].semilogy(criteria)
    axes[0,1].set_title('Nonstationarity Criterion T(A,B)')
    axes[0,1].set_xlabel('Iteration')
    axes[0,1].grid(True)
    
    # Sparse loadings matrix B
    im1 = axes[1,0].imshow(np.abs(B), cmap='viridis', aspect='auto')
    axes[1,0].set_title('Sparse Loadings |B|')
    axes[1,0].set_xlabel('Components')
    axes[1,0].set_ylabel('Features')
    plt.colorbar(im1, ax=axes[1,0])
    
    # Orthogonal matrix A
    im2 = axes[1,1].imshow(np.abs(A), cmap='plasma', aspect='auto')
    axes[1,1].set_title('Orthogonal Matrix |A|')
    axes[1,1].set_xlabel('Components')
    axes[1,1].set_ylabel('Features')
    plt.colorbar(im2, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('spca_vp_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return A, B, objectives, criteria


if __name__ == "__main__":
    A, B, objectives, criteria = demo()

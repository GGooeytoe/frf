import numpy as np
import scipy

def compute_overall_frf(possible_Y, indices, f):
    """
    Computes the overall frequency response vector u using Frequency Based 
    Substructuring (FBS) and generic Cholesky factorization.
    
    Parameters:
    possible_Y : list of np.ndarray
        List of possible unit cell receptance matrices (each of shape n x n).
    indices : list of int
        List specifying which index from possible_Y to use for each of the N unit cells.
    f : np.ndarray
        Excitation vector of shape (N * n,) or (N * n, 1).
        
    Returns:
    u : np.ndarray
        Overall frequency response vector of the same shape as f.
    """
    f_shape_orig = f.shape
    f_flat = f.ravel()
    
    N = len(indices)
    n = possible_Y[0].shape[0]
    
    if len(f_flat) != N * n:
        raise ValueError(f"Excitation vector f must have size N*n ({N * n}).")

    # 1. Assemble the block matrices and vectors
    Y = [possible_Y[idx] for idx in indices]
    f_blocks = f_flat.reshape(N, n)
    
    # Precompute Y * f blocks
    Yf = np.zeros((N, n))
    for i in range(N):
        Yf[i] = Y[i] @ f_blocks[i]
        
    # Edge case: Single unit cell (no interfaces)
    if N == 1:
        return Yf.reshape(f_shape_orig)
    
    # 2. compute g=BYf
    m = N - 1  # number of interfaces/blocks in the tridiagonal system
    g = -np.concatenate(np.diff(Yf,axis=0))

    # 3. solve for lam = (B*Y*B.T)^-1 * B*Y*f
    B=np.block([[np.zeros((n,n)) for j in range(i)]+[np.eye(n),-np.eye(n)]+[np.zeros((n,n)) for j in range(i+2,m)] for i in range(m)])
    Ymat=scipy.linalg.block_diag(*Y)
    BYBT=B@Ymat@B.T
    lam=np.linalg.solve(BYBT,g)

    # 4. compute output Y@f-Y@B.T@lam
    u=Yf.reshape(f_shape_orig)-Ymat@B.T@lam
    return u


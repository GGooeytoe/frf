'''
code is originally from Gemini 3 Pro on 23 March 2026. I provided it with the text of section 3.1 of our proposal, describing the frequency based substructuring and the use of block Thomas' algorithm.
Then prompted:
write a python function using numpy that takes a list of possible values for the unit cell receptance matrices Yi, a list of indices specifying which values appear, and an excitation vector f, and returns the overall frequency response vector u 
The full text of the prompt is at the bottom of this file and includes Latex math notation. Note that the subscripts in the latex were sloppy (used n in BYB^T where it really should be N-1), but Gemini seems to have gotten it right anyway.
'''
import numpy as np

def compute_overall_frf(possible_Y, indices, f):
    """
    Computes the overall frequency response vector u using Frequency Based 
    Substructuring (FBS) and the block Thomas algorithm.
    
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

    # 2. Setup Block Tridiagonal System A * lam = g where A = B*Y*B.T and g = B*Y*f
    m = N - 1  # number of interfaces/blocks in the tridiagonal system
    
    # D: Main diagonal blocks, U: Off-diagonal blocks (upper and lower are symmetric)
    D = [Y[i] + Y[i+1] for i in range(m)]
    U = [-Y[i+1] for i in range(m)] # U[m-1] is unused but kept for index alignment
    g = [Yf[i] - Yf[i+1] for i in range(m)]

    # 3. Block Thomas Algorithm to solve for lam = (B*Y*B.T)^-1 * B*Y*f
    C_prime = [None] * (m - 1)
    g_prime = [None] * m

    # --- Forward Sweep ---
    if m > 1:
        C_prime[0] = np.linalg.solve(D[0], U[0])
    g_prime[0] = np.linalg.solve(D[0], g[0])

    for i in range(1, m):
        # The lower diagonal A_i matches U_i-1 due to symmetry
        matrix_to_invert = D[i] - U[i-1] @ C_prime[i-1]
        
        if i < m - 1:
            C_prime[i] = np.linalg.solve(matrix_to_invert, U[i])
            
        g_prime[i] = np.linalg.solve(matrix_to_invert, g[i] - U[i-1] @ g_prime[i-1])

    # --- Backward Sweep ---
    lam = [None] * m
    lam[m-1] = g_prime[m-1]

    for i in range(m - 2, -1, -1):
        lam[i] = g_prime[i] - C_prime[i] @ lam[i+1]

    # 4. Compute final output u = Yf - Y * B.T * lam
    BT_lam = np.zeros((N, n))
    BT_lam[0] = lam[0]
    for i in range(1, N - 1):
        BT_lam[i] = -lam[i-1] + lam[i]
    BT_lam[N-1] = -lam[N-2]

    u_blocks = np.zeros((N, n))
    for i in range(N):
        u_blocks[i] = Yf[i] - Y[i] @ BT_lam[i]

    return u_blocks.reshape(f_shape_orig)

'''
Full Gemini 3 Pro prompt:
An essential need is thus to rapidly predict the frequency response function of a metamaterial isolator when its unit cells are in a particular configuration. Prior work \cite{chavan_bistable_2025} has shown that the overall FRF can be predicted by employing Frequency Based Substructuring (FBS), a mathematical technique that assembles the overall FRF from the FRFs of substructures. Here, it is natural to take the unit cells as the substructures. Consider the Lagrange multiplier FBS formulation as presented in de Klerk 2008 \cite{de_klerk_general_2008}:
\begin{equation}
u=Yf-YB^\top(BYB^\top)^{-1}BYf \label{eq:deKlerkFRF}
\end{equation}
Where $u\in\mathbb{R}^{Nn}$ is the output, $Y\in\mathbb{R}^{Nn\times{}Nn}$ is the block diagonal matrix containing the receptance matrices $Y_i\in{}\mathbb{R}^{n\times{}n}$ of each substructure in isolation, $f\in\mathbb{R}^{Nn}$ is the excitation, and $B\in{}\mathbb{R}^{Nn\times{}Nn}$ is the boolean matrix defining the interfaces between the structures. $n$ is the number of input-output degrees of freedom of each unit cell, and $N$ is the number of unit cells.
In our case, the control input switches individual blocks $Y_i$ between one of two blocks while $B$ is constant. $Y$ can thus be cheaply assembled on the fly. The majority of the computational cost in computing $u$ would be expected to lie in inverting the $Nn\times{}Nn$ matrix $BYB^\top$. However, the isolator designs of interest in this work consist of unit cells arranged in series, placing banded structure on $B$: 
\begin{equation}
B=\begin{bmatrix}
I& -I &      0 &0 &\ldots& 0\\
0&  I &     -I &0 &\ldots& 0\\
\vdots&\vdots &\vdots&\vdots&&\vdots
\end{bmatrix}
\end{equation}
$BYB^\top$ is thus symmetric block tridiagonal:
\begin{equation}
    BYB^\top=\begin{bmatrix}
    Y_1+Y_2 &    -Y_2 &         &      &             &         \\
       -Y_2 & Y_2+Y_3 &    -Y_3 &      &             &           \\
            &    -Y_3 & Y_3+Y_4 & -Y_4 &             &           \\
            &         &   -Y_4  &\ddots& \ddots      &      \\
            &         &         &\ddots& Y_{n-1}+Y_n & -Y_n\\
            &         &         &      & -Y_n        & Y_n+Y_{n+1}
    \end{bmatrix}
\end{equation}
with the elements of row $i$ equal to $-Y_i$, $Y_i+Y_{i+1}$, and $-Y_{i+1}$. The inverse of each of these blocks can be precomputed (each $Y_i$ takes one of two values), then the solution of a linear system defined by the matrix as a whole can be computed in time linear in $N$ (the number of unit cells) using the block Thomas’ algorithm

write a python function using numpy that takes a list of possible values for the unit cell receptance matrices Yi, a list of indices specifying which values appear, and an excitation vector f, and returns the overall frequency response vector u
'''
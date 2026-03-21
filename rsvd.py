import numpy as np

def rsvd(
    A,
    k,
    p=10,
    n_iter=0,
    seed=None,
):
    """
    Randomized SVD (Halko et al.) using a Gaussian sketch.

    Parameters
    ----------
    A : (m, n) array_like
        Input matrix.
    k : int
        Target rank k.
    p : int, default=10
        Oversampling parameter p, so sketch size is l = k + p.
    n_iter : int, default=1
        Number of power iterations.
    seed : int, or None
        Random seed / generator.

    Returns
    -------
    U : (m, k) ndarray
    s : (k,) ndarray
    Vt : (k, n) ndarray
        Approximate rank-k SVD: A ≈ U @ np.diag(s) @ Vt
    """
    A = np.asarray(A)
    m, n = A.shape
    l = k + p

    if k <= 0:
        raise ValueError("k must be positive")
    if k > min(m, n):
        raise ValueError("k cannot exceed min(m, n)")
    if p < 0:
        raise ValueError("p must be non-negative")
    if l > n:
        raise ValueError("k + p cannot exceed n")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")

    # RNG handling
    rng = np.random.default_rng(seed)

    # Step 1: Gaussian sketch Omega ~ N(0,1)^(n x l)
    Omega = rng.standard_normal(size=(n, l))

    # Step 2: Form sample matrix Y = A Omega
    Y = A @ Omega

    # Step 3: Optional power iterations
    # Y <- (A A^T)^q A Omega
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(Y, mode="reduced")
        Y = A.T @ Y
        Y , _ = np.linalg.qr(Y, mode="reduced")
        Y = A @ Y

    # Step 4: Orthonormal basis Q for range(Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # Step 5: Compress A to small matrix B = Q^T A
    B = Q.T @ A

    # Step 6: Exact SVD on the small matrix
    U_tilde, Sigma, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 7: Lift left singular vectors back
    U = Q @ U_tilde

    # Truncate to target rank
    U = U[:, :k]
    Sigma = Sigma[:k]
    Vt = Vt[:k, :]

    return U, Sigma, Vt
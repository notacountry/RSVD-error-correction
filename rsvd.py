import numpy as np
from s_transform import S_transform, S_inverse

DOMAIN = 0.95
GRANULARITY = 500

def rsvd(
    A,
    k,
    p=10,
    seed=None,
    correction=False,
):
    """
    RSVD using a Gaussian sketch and optional correction.

    Parameters
    ----------
    A : (m, n) array_like
        Input matrix.
    k : int
        Target rank k.
    p : int, default=10
        Oversampling parameter p, so sketch size is l = k + p.
    seed : int, or None
        Random seed / generator.
    correction : bool, default=False
        If True, apply the S-transform singular value correction.

    Returns
    -------
    U : (m, k) ndarray
    Sigma : (k,) ndarray
    Vt : (k, n) ndarray
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

    rng = np.random.default_rng(seed)
    
    Omega = rng.standard_normal(size=(n, l))
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ A
    U_tilde, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    # Truncate
    U     = U[:, :k]
    Sigma = Sigma[:k]
    Vt    = Vt[:k, :]

    if not correction:
        return U, Sigma, Vt

    c = n / l

    # Use the (l x l) matrix Y^T Y instead of (m x m) Y Y^T;
    # they share the same nonzero eigenvalues.
    eigs_nonzero = np.linalg.eigvalsh(Y.T @ Y) / l

    # Pad eigenvalues of Y^T Y with m - l zeros
    # to represent the full m-dimensional ESM.
    eigs_Y = np.concatenate([eigs_nonzero, np.zeros(max(0, m - l))])

    # The psi_Y-transform maps the negative real axis onto (bound, 0), so
    # the S-transform is only well-defined for w in that interval.
    # Stay DOMAIN% inside the boundary to avoid numerical issues at the edges.
    bound = -np.sum(eigs_nonzero > 0.0) / m
    w = np.linspace(bound * DOMAIN, bound * (1 - DOMAIN), GRANULARITY)

    S_Y = S_transform(eigs_Y, w)
    S_A = S_Y * (1.0 + c * w)
    sigma_corr  = np.sqrt(S_inverse(w, S_A, k))

    # The deconvolution should always reduce singular values.
    # Reject singular values that were not reduced.
    n_rec = len(sigma_corr)
    accept = sigma_corr <= Sigma[:n_rec]
    Sigma[:n_rec][accept] = sigma_corr[accept]

    return U, Sigma, Vt

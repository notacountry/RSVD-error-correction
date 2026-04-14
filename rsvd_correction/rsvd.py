import numpy as np

from rsvd_correction.free_probability import correct_singular_values


def _rsvd_sketch(A, k, p, seed):
    """
    Execute the randomized sketch and return all intermediate results.

    Parameters
    ----------
    A : (m, n) ndarray  (already validated and converted)
    k : int
    p : int
    seed : int or None

    Returns
    -------
    Y : (m, l) ndarray  — sketch matrix A @ Omega
    m, n, l : int
    U : (m, k) ndarray
    Sigma : (k,) ndarray  — plain (uncorrected) singular values
    Vt : (k, n) ndarray
    """
    m, n = A.shape
    l = k + p
    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal(size=(n, l))
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ A
    U_tilde, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return Y, m, n, l, U[:, :k], Sigma[:k], Vt[:k, :]


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
    if k <= 0:
        raise ValueError("k must be positive")
    if p < 0:
        raise ValueError("p must be non-negative")

    A = np.asarray(A)
    m, n = A.shape
    if k > min(m, n):
        raise ValueError("k cannot exceed min(m, n)")

    l = k + p
    if l > n:
        raise ValueError("k + p cannot exceed n")

    Y, m, n, l, U, Sigma, Vt = _rsvd_sketch(A, k, p, seed)
    if correction:
        Sigma = correct_singular_values(Y, m, n, l, k, Sigma)

    return U, Sigma, Vt

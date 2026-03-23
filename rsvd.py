import numpy as np
from s_transform import S_transform, S_inverse


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
    eigs_nonzero = np.linalg.eigvalsh((1.0 / l) * (Y.T @ Y))

    # Pad eigenvalues of Y^T Y with m - l zeros
    # to represent the full m-dimensional ESM.
    eigs_Y = np.concatenate([eigs_nonzero, np.zeros(max(0, m - l))])

    # w-grid in (-1, 0). Use many more points than k for a good AAA fit.
    # The psi_Y-transform maps the negative real axis onto (-n_nz/m, 0), where
    # n_nz is the number of strictly positive eigenvalues in eigs_Y.  For w
    # below -n_nz/m there is no solution z < 0 to ψ_Y(z) = w, so the Newton
    # iteration in S_transform diverges and produces NaN/inf.
    #   • Full-rank sketch (signal+noise): n_nz = l, bound = -l/m = -1/c.
    #   • Low-rank A (rank r < l):         n_nz = r, bound = -r/m  (tighter).
    ev_thresh = eigs_nonzero.max() * 1e-8 if eigs_nonzero.max() > 0 else 0.0
    n_nz = int(np.sum(eigs_nonzero > ev_thresh))
    # Use a 5% relative inset so the margin scales with the domain width.
    # A fixed +1e-3 offset can equal or exceed the domain width (-n_nz/m) for
    # very low rank (e.g. rank-1, n=1000 gives domain width 0.001 = margin).
    w_lo = max(-(n_nz / m) * 0.95, -1.0 + 1e-3)
    w = np.linspace(w_lo, -1e-3, 500)

    # S^Y(w) from the empirical spectral measure of (1/l) Y Y^T.
    S_Y = S_transform(eigs_Y, w)

    # Deconvolve Marchenko-Pastur: S^MP(w) = 1/(1+cw), so S^A = S^Y * S^MP^{-1}.
    # Standard identity: S^A(w) = S^Y(w) * (1 + cw).
    S_A = S_Y * (1.0 + c * w)

    # Recover corrected eigenvalues of A A^T, then convert to singular values.
    lambda_corr = S_inverse(w, S_A, k)
    sigma_corr  = np.sqrt(np.maximum(lambda_corr, 0.0))

    # Replace as many singular values as were successfully recovered,
    # but only where the correction is downward (sigma_corr <= sigma_plain).
    # A valid MP deconvolution should always reduce singular values: RSVD
    # inflates them due to the noise bulk, so any corrected value larger than
    # the corresponding RSVD value signals a failed deconvolution and is
    # discarded in favour of the original RSVD estimate.
    n_rec = len(sigma_corr)
    accept = sigma_corr <= Sigma[:n_rec]
    Sigma[:n_rec][accept] = sigma_corr[accept]

    return U, Sigma, Vt

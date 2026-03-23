import numpy as np
from s_transform import S_transform, S_inverse


def rsvd(
    A,
    k,
    p=10,
    n_iter=0,
    seed=None,
    correction=False,
):
    """
    RSVD using a Gaussian sketch, with optional correction.

    Parameters
    ----------
    A : (m, n) array_like
        Input matrix.
    k : int
        Target rank k.
    p : int, default=10
        Oversampling parameter p, so sketch size is l = k + p.
    n_iter : int, default=0
        Number of power iterations.
    seed : int, or None
        Random seed / generator.
    correction : bool, default=False
        If True, apply the S-transform singular value correction.

        Note: the theory assumes n_iter = 0.  Power iterations alter the
        spectral distribution, so the correction is applied to the original
        sketch Y = A @ Omega regardless of n_iter.

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
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")

    # RNG handling
    rng = np.random.default_rng(seed)

    # Step 1: Gaussian sketch Omega ~ N(0,1)^(n x l)
    Omega = rng.standard_normal(size=(n, l))

    # Step 2: Form sample matrix Y = A Omega
    Y = A @ Omega

    # Save the original sketch before power iterations: the S-transform
    # correction theory is derived for Y = A Omega, not the power-iterated form.
    Y_sketch = Y if n_iter == 0 else Y.copy()

    # Step 3: Optional power iterations
    # Y <- (A A^T)^q A Omega
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(Y, mode="reduced")
        Y = A.T @ Y
        Y, _ = np.linalg.qr(Y, mode="reduced")
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
    U     = U[:, :k]
    Sigma = Sigma[:k]
    Vt    = Vt[:k, :]

    if not correction:
        return U, Sigma, Vt

    # ------------------------------------------------------------------
    # S-transform singular value correction
    # ------------------------------------------------------------------
    # Marchenko-Pastur ratio: Omega is (n x l), so W_l = (1/l) Omega Omega^T
    # is (n x n) with limiting ratio c = n / l.
    c = n / l

    # Eigenvalues of (1/l) Y_sketch Y_sketch^T.
    # Use the (l x l) matrix Y_sketch^T Y_sketch to avoid forming the
    # potentially large (m x m) outer product; they share the same nonzero
    # eigenvalues.  Pad with m - l zeros to represent the full m-dimensional
    # empirical spectral measure.
    eigs_nonzero = np.linalg.eigvalsh((1.0 / l) * (Y_sketch.T @ Y_sketch))
    eigs_Y = np.concatenate([eigs_nonzero, np.zeros(max(0, m - l))])

    # w-grid in (-1, 0) — use many more points than k for a good AAA fit.
    # The ψ_Y-transform maps the negative real axis onto (-n_nz/m, 0), where
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

    # Deconvolve Marchenko-Pastur: S^MP(w) = 1/(1+cw), so S^A = S^Y / S^MP.
    # NOTE: S_transform returns S_code = (1+w)/w * z (not the standard S_std =
    # (1+w)/(w*z)).  In this convention S_code = S_std * z^2, so the standard
    # identity S_A_std = S_Y_std * (1+cw) becomes S_code_A = S_code_Y / (1+cw).
    S_A = S_Y.real / (1.0 + c * w)

    # Recover corrected eigenvalues of A A^T, then convert to singular values.
    lambda_corr = S_inverse(w, S_A, k)
    sigma_corr  = np.sqrt(np.maximum(lambda_corr, 0.0))

    # Replace as many singular values as were successfully recovered.
    Sigma[:len(sigma_corr)] = sigma_corr

    return U, Sigma, Vt

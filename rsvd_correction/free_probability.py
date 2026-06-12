"""
Free probability tools for RSVD eigenvalue correction.

Assumptions
-----------
1. Eigenvalues are non-negative.
2. w lies in (-1, 0).
"""
import numpy as np
from scipy.linalg import eig as scipy_eig
from scipy.interpolate import CubicSpline


_DOMAIN            = 0.95
_GRANULARITY       = 500
_Z_GRID_SIZE       = 2000
_AAA_TOL           = 1e-13
_IMAG_TOL          = 1e-6
_EPSILON           = 1e-10
_REG_EPS           = 1e-30

def stieltjes_transform(z, eigenvalues):
    """
    Empirical Stieltjes transform.

    Parameters
    ----------
    z : array_like, shape (M,)
    eigenvalues : array_like, shape (n,)

    Returns
    -------
    ndarray, shape (M,)
    """
    return np.mean(1.0 / (z[..., None] - eigenvalues), axis=-1)


def S_transform(eigenvalues, w_vals):
    """
    Compute S-transform of the ESM of a matrix with given eigenvalues.

    Parameters
    ----------
    eigenvalues : array_like of real, shape (n,)
        Eigenvalues defining the ESM.
    w_vals : array_like of real
        Points where S(w) is evaluated.

    Returns
    -------
    S_vals : ndarray of float, shape (len(w_vals),)
        S-transform values at w_vals.
    """
    if len(w_vals) == 0:
        return np.array([], dtype=float)

    eigs_pos = eigenvalues[eigenvalues > 0]
    n_total, n_pos = len(eigenvalues), len(eigs_pos)
    if n_pos == 0:
        return np.full(len(w_vals), np.nan, dtype=float)

    # Log-spaced z-grid on the negative real axis.
    # Dense near z = 0; sparse near boundary (psi_Y ~ 0, S_Y ~ const).
    z_grid = -np.geomspace(eigs_pos.mean() * 1e-4, eigs_pos.max() * 1e4, _Z_GRID_SIZE)

    alpha   = (n_total - n_pos) / n_total
    w_param = (alpha - 1.0) + (n_pos / n_total) * z_grid * stieltjes_transform(z_grid, eigs_pos)

    # Ensure w in (-1, 0)
    valid = w_param < -_EPSILON
    w_param, z_grid = w_param[valid], z_grid[valid]

    if len(w_param) < 2:
        return np.full(len(w_vals), np.nan, dtype=float)

    # Find chi_Y s.t. chi_Y(w) = z, then S_Y(w) = (1 + w) / w * chi_Y(w).
    # chi_Y(w) = z is smooth and monotone on the entire domain, unlike S.
    chi = CubicSpline(w_param, z_grid, extrapolate=True)
    return (1.0 + w_vals) / w_vals * chi(w_vals)


def psi_inverse(w, S_w):
    """
    Given S-transform values S = (1+w)/w * chi(w)
    Compute chi(w) = psi^{-1}(w) and G(chi(w)).

    Parameters
    ----------
    w : array_like, shape (m,)
        Points in (-1, 0).
    S_w : array_like, shape (m,)
        S-transform values at w.

    Returns
    -------
    z_vals : ndarray, shape (m,)
        chi(w) = psi^{-1}(w)
    G_vals : ndarray, shape (m,)
        Stieltjes transform G(chi(w)).
    """
    if w.ndim != 1 or S_w.ndim != 1:
        raise ValueError("w and S_w must be 1D arrays.")
    if len(w) != len(S_w):
        raise ValueError("w and S_w must have the same length.")

    z = (w / (1.0 + w)) * S_w
    G = (1.0 + w) / z
    return z, G


def _aaa(z, F, tol=_AAA_TOL, mmax=100):
    """
    AAA rational approximation (Nakatsukasa, Sète, Trefethen 2018).

    Builds a barycentric rational approximant r(z) = N(z)/D(z) by greedily
    selecting support points and computing barycentric weights via SVD of the
    Loewner matrix. The barycentric form is backward-stable by construction,
    avoiding the Vandermonde ill-conditioning of monomial-basis approaches.

    Parameters
    ----------
    z : array_like, shape (M,)
        Sample points (real).
    F : array_like, shape (M,)
        Function values at z (real).
    tol : float
        Convergence tolerance on the relative sup-norm residual.
    mmax : int
        Maximum number of support points (degree bound).

    Returns
    -------
    zj : ndarray
        Support points chosen greedily.
    fj : ndarray
        Function values at support points.
    wj : ndarray
        Barycentric weights.
    """
    M = len(z)
    F_scale = np.max(np.abs(F))
    n_iters = min(mmax, M - 1)

    # Cauchy matrix; fill one column per iteration.
    C = np.empty((M, n_iters), dtype=float)

    mask = np.ones(M, dtype=bool)
    zj_list, fj_list = [], []
    R = np.full(M, np.mean(F))
    wj = None

    for col in range(n_iters):
        J_arr = np.where(mask)[0]
        err = np.abs(F[J_arr] - R[J_arr])

        if err.max() <= tol * F_scale:
            break

        j_star = J_arr[np.argmax(err)]
        zj_list.append(z[j_star])
        fj_list.append(F[j_star])
        mask[j_star] = False

        with np.errstate(divide="ignore"):
            C[:, col] = 1.0 / (z - zj_list[-1])

        fj = np.array(fj_list)
        J_arr = np.where(mask)[0]
        C_view = C[:, :col + 1]

        # Loewner matrix: L[i,k] = (F[i] - fj[k]) / (z[i] - zj[k])
        #                        = (F[i] - fj[k]) * C[i,k]
        L = (F[J_arr, None] - fj[None, :]) * C_view[J_arr, :]

        # Weights = right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(L, full_matrices=False)
        wj = Vh[-1, :]

        # Update rational approximant at non-support points
        N = C_view[J_arr, :] @ (wj * fj)
        D = C_view[J_arr, :] @ wj
        safe = np.abs(D) > 0
        R[J_arr[safe]] = N[safe] / D[safe]

    return np.array(zj_list), np.array(fj_list), wj


def _aaa_poles_residues(zj, fj, wj):
    """
    Extract poles and residues from an AAA barycentric approximant.

    Poles are found via the companion-matrix generalized eigenvalue problem
    from Nakatsukasa, Sète & Trefethen (2018).  The denominator of the
    barycentric form is D(z) = Sum_k w_k/(z - z_k).  Its zeros (the poles of
    r) are the finite eigenvalues of the (m+1)*(m+1) pencil (E, B):

        E[0, 1:]  = wj
        E[k, 0]   = 1       for k = 1..m
        E[k, k]   = zj[k-1] for k = 1..m
        B          = diag(0, 1, ..., 1)

    Eliminating the bottom m rows gives Sum_k w_k/(lambda - z_k) = 0, which is
    exactly D(lambda) = 0.

    Residues are computed via the barycentric formulas for N(z) and D'(z),
    which are numerically stable.

    Parameters
    ----------
    zj, fj, wj : ndarray
        Support points, values, and weights from _aaa.

    Returns
    -------
    poles : ndarray of complex
    residues : ndarray of complex
    """
    m = len(zj)
    if m == 0:
        return np.array([]), np.array([])
    if wj is None or not np.isfinite(wj).all():
        return np.array([]), np.array([])

    # Build the (m+1)*(m+1) companion pencil (E, B).
    E = np.zeros((m + 1, m + 1), dtype=complex)
    B = np.zeros((m + 1, m + 1), dtype=complex)

    E[0, 1:]   = wj           # barycentric weights in first row
    E[1:, 0]   = 1.0          # ones in first column (rows 1..m)
    E[1:, 1:]  = np.diag(zj)  # support points on diagonal (rows 1..m)
    B[1:, 1:]  = np.eye(m)    # lower-right block is identity; B[0,0] = 0

    # Generalized eigenvalues of E v = lambda B v.
    # B is singular (B[0,0]=0), so some eigenvalues are infinite.
    # The finite ones satisfying D(lambda)=0 are the poles of r.
    try:
        evals = scipy_eig(E, B, right=False)
    except Exception:
        return np.array([]), np.array([])

    # Retain finite eigenvalues only.
    finite_mask = np.isfinite(evals.real) & np.isfinite(evals.imag)
    poles = evals[finite_mask]

    # Discard eigenvalues that coincide with a support point (spurious roots
    # of the companion pencil that are not true poles of D).
    dist_to_support = np.min(np.abs(poles[:, None] - zj[None, :]), axis=1)
    scale = np.abs(poles) + np.abs(zj).mean() + _REG_EPS
    poles = poles[dist_to_support > _EPSILON * scale]

    if len(poles) == 0:
        return np.array([]), np.array([])

    # Residues: res_i = N(pole_i) / D'(pole_i)
    #   N(z)  = sum_k w_k f_k / (z - z_k)
    #   D'(z) = -sum_k w_k / (z - z_k)^2
    #
    # Vectorised over all poles simultaneously: d[i, k] = poles[i] - zj[k]
    d = poles[:, None] - zj[None, :]                    # (n_poles, m)
    near = np.min(np.abs(d), axis=1) < _EPSILON         # mask near-coincident poles
    N_vals  =  (wj * fj / d).sum(axis=1)
    Dp_vals = -(wj / d**2).sum(axis=1)
    residues = np.where(near, np.nan + 0j, N_vals / Dp_vals)

    return poles, residues


def eigenvalues_from_greens_function(z, G, k, tol=_AAA_TOL, imag_tol=_IMAG_TOL):
    """
    Recover k eigenvalues of mu^A from (z, G) data on the negative real axis.

    Uses the AAA algorithm (Nakatsukasa, Sète, Trefethen 2018) to build a
    barycentric rational approximant to G^A, then extracts poles and enforces
    the Stieltjes constraints:
      - poles must be real and positive  (support of mu^A lies in (0, inf))
      - residues must be real and positive  (mu^A is a positive measure)

    Poles failing either constraint are discarded as spurious.

    Parameters
    ----------
    z : array_like, shape (M,)
        Negative real z-values (from psi_inverse). More points give a better
        fit; M >> k is recommended.
    G : array_like, shape (M,)
        Corresponding G^A(z)-values.
    k : int
        Number of eigenvalues to return (largest k retained after filtering).
    tol : float
        AAA convergence tolerance on the relative sup-norm residual.
    imag_tol : float
        Relative imaginary tolerance for filtering spurious complex poles.
        Poles with |Im(pole)| / (|Re(pole)| + _REG_EPS) > imag_tol are discarded.
        Tighter values risk dropping nearly-real poles; looser values risk
        accepting complex noise. Default: 1e-6.

    Returns
    -------
    eigenvalues : ndarray, shape (<= k,)
        Corrected eigenvalues, sorted descending. Fewer than k may be returned
        if fewer than k poles survive Stieltjes filtering.
    """
    zj, fj, wj = _aaa(z, G, tol=tol)
    if wj is None or len(zj) == 0:
        raise RuntimeError("AAA produced no support points.")

    poles, residues = _aaa_poles_residues(zj, fj, wj)

    # Enforce physical Stieltjes constraints only: poles must be real and
    # positive, residues must be real and positive.
    valid = (
        np.isfinite(residues)
        & (np.abs(poles.imag)    < imag_tol * (np.abs(poles.real)    + _REG_EPS))
        & (np.abs(residues.imag) < imag_tol * (np.abs(residues.real) + _REG_EPS))
        & (poles.real    > 0)
        & (residues.real > 0)
    )

    candidates = np.sort(poles[valid].real)[::-1]
    return candidates[:k]


def S_inverse(w, S_w, k, imag_tol=_IMAG_TOL):
    """
    Recover k corrected eigenvalues from S-transform values.

    Parameters
    ----------
    w : array_like, shape (m,)
        Points in (-1, 0)
    S_w : array_like, shape (m,)
        S-transform values at w.
    k : int
        Number of eigenvalues to recover.
    imag_tol : float
        Passed through to eigenvalues_from_greens_function. See that
        function's docstring.

    Returns
    -------
    eigenvalues : ndarray, shape (<= k,)
        Corrected eigenvalues, sorted descending. Fewer than k may be returned
        if fewer than k poles survive Stieltjes filtering.
    """
    z, G = psi_inverse(w, S_w)
    finite = np.isfinite(z) & np.isfinite(G)

    return eigenvalues_from_greens_function(z[finite], G[finite], k, imag_tol=imag_tol)


def _detect_spike_count(Sigma):
    """
    Detect the number of spike components via the largest relative gap in
    the RSVD singular values.

    Parameters
    ----------
    Sigma : (k,) ndarray
        RSVD singular values, sorted descending.

    Returns
    -------
    n_spikes : int
        Number of leading singular values identified as spike components.
        Returns 0 if no significant gap is found.
    """
    Sigma_s = np.sort(np.abs(Sigma))[::-1]
    if len(Sigma_s) < 2:
        return 0
    ratios = Sigma_s[:-1] / (Sigma_s[1:] + _EPSILON)
    gap_idx = int(np.argmax(ratios))
    if ratios[gap_idx] < 1.5:
        return 0
    return gap_idx + 1


def _stransform_deconvolve(eigs_input, m, c, k):
    """
    Run S-transform deconvolution on a set of sketch eigenvalues.

    Parameters
    ----------
    eigs_input : (p,) ndarray
        Sketch eigenvalues to use (p <= l; may be a bulk-only subset).
    m : int
        Number of rows of the original matrix A (total ESM dimension).
    c : float
        Aspect ratio n / l.
    k : int
        Number of corrected eigenvalues to return.

    Returns
    -------
    eigs_corr : ndarray, length <= k
        Corrected eigenvalues of A A^T, sorted descending.
    """
    n_pos  = int(np.sum(eigs_input > 0))
    n_pad  = max(0, m - len(eigs_input))
    eigs_Y = np.concatenate([eigs_input, np.zeros(n_pad)])
    bound  = -n_pos / m
    w      = np.linspace(bound * _DOMAIN, bound * (1 - _DOMAIN), _GRANULARITY)
    S_Y    = S_transform(eigs_Y, w)
    S_A    = S_Y * (1.0 + c * w)
    return S_inverse(w, S_A, k)


def sketch_spectral_info(Y, m, n, l, k, Sigma=None, bbp=False):
    """
    Expose the intermediate spectral quantities computed during correction.

    Parameters
    ----------
    Y : (m, l) ndarray
        Sketch matrix A @ Omega.
    m, n, l, k : int
        Same as in correct_singular_values.
    Sigma : (k,) ndarray or None
        Plain RSVD singular values.  Required when bbp=True; ignored otherwise.
    bbp : bool
        If True, apply spike protection: detect the spectral gap in Sigma and
        replace the S-transform estimates for the leading spike components
        with the squared RSVD singular values.  Default False.

    Returns
    -------
    eigs_sketch : (l,) ndarray
        Eigenvalues of (1/l) Y^T Y, sorted ascending.
    eigs_corrected : (<= k,) ndarray
        Corrected eigenvalues of A A^T, sorted descending.
    c : float
        Aspect ratio n / l.
    """
    c           = n / l
    eigs_sketch = np.linalg.eigvalsh(Y.T @ Y) / l
    eigs_corr   = _stransform_deconvolve(eigs_sketch, m, c, k)

    if bbp:
        Sigma_in = np.zeros(k) if Sigma is None else Sigma
        eigs_corr = _apply_bbp(eigs_corr, k, Sigma_in)

    return eigs_sketch, eigs_corr, c


def _apply_bbp(eigs_corrected, k, Sigma):
    """
    Replace S-transform estimates for spike components with squared RSVD values.

    Detects the spectral gap in the RSVD singular values Sigma and overwrites
    the leading n_spikes entries of eigs_corrected with Sigma[:n_spikes]**2.
    The bulk components are returned unchanged.

    Parameters
    ----------
    eigs_corrected : ndarray
        Eigenvalues from S-transform deconvolution, sorted descending.
    k : int
        Number of eigenvalues to return.
    Sigma : (k,) ndarray
        Plain RSVD singular values (before correction), sorted descending.

    Returns
    -------
    eigs_out : ndarray, length <= k
    """
    n_spikes = _detect_spike_count(Sigma)
    if n_spikes == 0 or len(eigs_corrected) == 0:
        return eigs_corrected

    Sigma_sorted = np.sort(np.abs(Sigma))[::-1]
    spike_eigs   = Sigma_sorted[:n_spikes] ** 2
    bulk_out     = np.sort(eigs_corrected)[::-1][n_spikes:]
    return np.concatenate([spike_eigs, bulk_out])[:k]


def correct_singular_values(Y, m, n, l, k, Sigma, bbp=False):
    """
    Apply Marchenko-Pastur S-transform deconvolution to correct RSVD singular
    values, with optional spike protection for bilevel-type spectra.

    Parameters
    ----------
    Y : (m, l) ndarray
        Sketch matrix A @ Omega from the RSVD step.
    m : int
        Number of rows of the original matrix A.
    n : int
        Number of columns of the original matrix A.
    l : int
        Sketch size (= k + p).
    k : int
        Target rank; number of singular values to correct.
    Sigma : (k,) ndarray
        RSVD singular values.
    bbp : bool
        If True, detect spike components via the spectral gap in Sigma and
        replace the S-transform estimates for those components with the
        squared RSVD singular values.  The S-transform deconvolution still
        runs on the full sketch spectrum; only the output slots corresponding
        to spike components are overwritten.  Default False.

    Returns
    -------
    Sigma_out : (k,) ndarray
        Corrected singular values.
    """
    c           = n / l
    eigs_sketch = np.linalg.eigvalsh(Y.T @ Y) / l
    eigs_corr   = _stransform_deconvolve(eigs_sketch, m, c, k)

    if bbp:
        eigs_corr = _apply_bbp(eigs_corr, k, Sigma)

    sigma_corr          = np.sqrt(eigs_corr)
    Sigma_out           = np.zeros_like(Sigma)
    Sigma_out[:len(sigma_corr)] = sigma_corr
    return Sigma_out
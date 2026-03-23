"""
S-transform computation for RSVD eigenvalue correction via free multiplicative
deconvolution.

Assumptions
-----------
1. Eigenvalues are non-negative.
2. The input is a probability measure.
3. w lies in (-1, 0).
"""
import warnings

import numpy as np
from scipy.linalg import eig as scipy_eig
from scipy.interpolate import CubicSpline


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
    if len(n_pos) == 0:
        return np.full(len(w_vals), np.nan, dtype=float)

    # Log-spaced z-grid on the negative real axis.
    # Dense near z = 0; sparse near boundary (psi_Y ~ 0, S_Y ~ const).
    z_grid = -np.geomspace(eigs_pos.mean() * 1e-4, eigs_pos.max() * 1e4, 2000)

    alpha   = (n_total - n_pos) / n_total
    w_param = (alpha - 1.0) + (n_pos / n_total) * z_grid * stieltjes_transform(z_grid, eigs_pos)

    # Ensure w in (-1, 0)
    valid = w_param < -1e-12
    w_param, z_grid = w_param[valid], z_grid[valid]

    if len(w_param) < 2:
        return np.full(len(w_vals), np.nan, dtype=float)

    # Find chi_Y s.t. chi_Y(w) = z, then S_Y(w) = (1 + w) / w * chi_Y(w).
    # chi_Y(w) = z is smooth and monotone on the entire domain, unlike S.
    chi_Y  = CubicSpline(w_param, z_grid, extrapolate=True)
    return (1.0 + w_vals) / w_vals * chi_Y(w_vals)


# ---------------------------------------------------------------------------
# psi_inverse
# ---------------------------------------------------------------------------

def psi_inverse(w_vals, S_vals):
    """
    Given S-transform values, compute chi(w) = psi^{-1}(w) and G(chi(w)).

    Parameters
    ----------
    w_vals : array_like, shape (m,)
        Points in (-1, 0).
    S_vals : array_like, shape (m,)
        S-transform values at w_vals.

    Returns
    -------
    z_vals : ndarray, shape (m,)
        chi(w) = psi^{-1}(w)
    G_vals : ndarray, shape (m,)
        G(chi(w))
    """
    w_vals = np.asarray(w_vals, dtype=float)
    S_vals = np.asarray(S_vals, dtype=float)

    if w_vals.ndim != 1 or S_vals.ndim != 1:
        raise ValueError("w_vals and S_vals must be 1D arrays.")
    if len(w_vals) != len(S_vals):
        raise ValueError("w_vals and S_vals must have the same length.")

    z_vals = (w_vals / (1.0 + w_vals)) * S_vals
    G_vals = (1.0 + w_vals) / z_vals
    return z_vals, G_vals


# ---------------------------------------------------------------------------
# AAA rational approximation
# ---------------------------------------------------------------------------

def _aaa(z_vals, G_vals, tol=1e-13, mmax=100):
    """
    AAA rational approximation (Nakatsukasa, Sète, Trefethen 2018).

    Builds a barycentric rational approximant r(z) = N(z)/D(z) by greedily
    selecting support points and computing barycentric weights via SVD of the
    Loewner matrix. The barycentric form is backward-stable by construction,
    avoiding the Vandermonde ill-conditioning of monomial-basis approaches.

    Parameters
    ----------
    z_vals : array_like, shape (M,)
        Sample points (real).
    G_vals : array_like, shape (M,)
        Function values at z_vals (real).
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
    z = np.asarray(z_vals, dtype=float)
    F = np.asarray(G_vals, dtype=float)
    M = len(z)
    F_scale = np.max(np.abs(F)) or 1.0

    # Pre-allocate the full Cauchy matrix; fill one column per iteration.
    n_iters = min(mmax, M - 1)
    C = np.empty((M, n_iters), dtype=float)

    mask = np.ones(M, dtype=bool)   # True = non-support point
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
    barycentric form is D(z) = Σ_k w_k/(z - z_k).  Its zeros (the poles of
    r) are the finite eigenvalues of the (m+1)×(m+1) pencil (E, B):

        E[0, 1:]  = wj
        E[k, 0]   = 1       for k = 1..m
        E[k, k]   = zj[k-1] for k = 1..m
        B          = diag(0, 1, ..., 1)

    Eliminating the bottom m rows gives Σ_k w_k/(λ - z_k) = 0, which is
    exactly D(λ) = 0.  This replaces the polyfromroots → polydiv → polyroots
    chain, which overflows in monomial basis for m ≳ 15 support points.

    Residues are computed via the barycentric formulas for N(z) and D′(z),
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

    # Build the (m+1) x (m+1) companion pencil (E, B).
    E = np.zeros((m + 1, m + 1), dtype=complex)
    B = np.zeros((m + 1, m + 1), dtype=complex)

    E[0, 1:]   = wj           # barycentric weights in first row
    E[1:, 0]   = 1.0          # ones in first column (rows 1..m)
    E[1:, 1:]  = np.diag(zj)  # support points on diagonal (rows 1..m)
    B[1:, 1:]  = np.eye(m)    # lower-right block is identity; B[0,0] = 0

    # Generalized eigenvalues of E v = λ B v.
    # B is singular (B[0,0]=0), so some eigenvalues are infinite.
    # The finite ones satisfying D(λ)=0 are the poles of r.
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
    scale = np.abs(poles) + np.abs(zj).mean() + 1e-30
    poles = poles[dist_to_support > 1e-10 * scale]

    if len(poles) == 0:
        return np.array([]), np.array([])

    # Residues: res_i = N(pole_i) / D'(pole_i)
    #   N(z)  = sum_k w_k f_k / (z - z_k)
    #   D'(z) = -sum_k w_k / (z - z_k)^2
    #
    # Vectorised over all poles simultaneously: d[i, k] = poles[i] - zj[k]
    d = poles[:, None] - zj[None, :]                    # (n_poles, m)
    near = np.min(np.abs(d), axis=1) < 1e-14           # mask near-coincident poles
    N_vals  =  (wj * fj / d).sum(axis=1)
    Dp_vals = -(wj / d**2).sum(axis=1)
    residues = np.where(near, np.nan + 0j, N_vals / Dp_vals)

    return poles, residues


# ---------------------------------------------------------------------------
# Eigenvalue recovery from Green's function data
# ---------------------------------------------------------------------------

def eigenvalues_from_G(z_vals, G_vals, k, tol=1e-13, imag_tol=1e-6):
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
    z_vals : array_like, shape (M,)
        Negative real z-values (from psi_inverse). More points give a better
        fit; M >> k is recommended.
    G_vals : array_like, shape (M,)
        Corresponding G^A(z)-values.
    k : int
        Number of eigenvalues to return (largest k retained after filtering).
    tol : float
        AAA convergence tolerance on the relative sup-norm residual.
    imag_tol : float
        Relative imaginary tolerance for filtering spurious complex poles.
        Poles with |Im(pole)| / (|Re(pole)| + 1e-30) > imag_tol are discarded.
        Tighter values risk dropping nearly-real poles; looser values risk
        accepting complex noise. Default: 1e-6.

    Returns
    -------
    eigenvalues : ndarray, shape (<= k,)
        Corrected eigenvalues, sorted descending. Fewer than k may be returned
        if fewer than k poles survive Stieltjes filtering.
    """
    z = np.asarray(z_vals, dtype=float)
    G = np.asarray(G_vals, dtype=float)

    zj, fj, wj = _aaa(z, G, tol=tol)
    if wj is None or len(zj) == 0:
        raise RuntimeError("AAA produced no support points.")

    poles, residues = _aaa_poles_residues(zj, fj, wj)

    # Enforce real positive poles & residues.
    # Also discard poles at lambda > max|z|: such poles contribute only
    # residue / (z - lambda) ~ residue / lambda to G on the negative real
    # axis, which is negligible when lambda >> max|z|.  Any pole found there
    # is a spurious artefact of the rational approximation and cannot be
    # resolved from data on [-max|z|, 0).
    max_z = np.max(np.abs(z_vals))
    valid = (
        np.isfinite(residues)
        & (np.abs(poles.imag)    < imag_tol * (np.abs(poles.real)    + 1e-30))
        & (np.abs(residues.imag) < imag_tol * (np.abs(residues.real) + 1e-30))
        & (poles.real    > 0)
        & (poles.real    < max_z)
        & (residues.real > 0)
    )

    # Second pass: remove poles whose residue is negligibly small relative to
    # the other valid poles.  A genuine eigenvalue of A contributes residue
    # ~ 1/N to m_A(z); a spurious AAA pole can have an arbitrarily small
    # residue that still passes the positivity check.  If any valid pole has a
    # residue less than 10% of the mean valid residue, it is too small to be a
    # real eigenvalue and is discarded.
    if valid.any():
        mean_res = residues[valid].real.mean()
        valid &= (residues.real >= 0.1 * mean_res)

    candidates = np.sort(poles[valid].real)[::-1]
    return candidates[:k]


# ---------------------------------------------------------------------------
# S_inverse
# ---------------------------------------------------------------------------

def S_inverse(w_vals, S_vals, k, imag_tol=1e-6):
    """
    Recover k corrected eigenvalues from S-transform values.
    For best results, pass a fine, uniform w-grid in (-1, 0) with
    many more than 2*k points.

    Parameters
    ----------
    w_vals : array_like, shape (m,)
        Points in (-1, 0)
    S_vals : array_like, shape (m,)
        S-transform values at w_vals.
    k : int
        Number of eigenvalues to recover.
    imag_tol : float
        Passed through to eigenvalues_from_G. See that function's docstring.

    Returns
    -------
    eigenvalues : ndarray, shape (<= k,)
        Corrected eigenvalues, sorted descending. Fewer than k may be returned
        if fewer than k poles survive Stieltjes filtering in eigenvalues_from_G.
    """
    z_vals, G_vals = psi_inverse(w_vals, S_vals)
    finite = (
        np.isfinite(z_vals) & np.isfinite(G_vals)
        & (z_vals < 0) & (G_vals < 0)
    )
    eigenvalues = eigenvalues_from_G(z_vals[finite], G_vals[finite], k, imag_tol=imag_tol)
    if len(eigenvalues) < k:
        warnings.warn(
            f"Only {len(eigenvalues)} of {k} requested eigenvalues were recovered.",
            RuntimeWarning,
            stacklevel=2,
        )

    return eigenvalues

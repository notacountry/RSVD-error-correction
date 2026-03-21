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
import numpy.polynomial.polynomial as nppoly


def stieltjes_transform(z, eigenvalues):
    """
    Empirical Stieltjes transform.

    Parameters
    ----------
    z : array_like of complex, shape (M,)
    eigenvalues : array_like of real, shape (n,)

    Returns
    -------
    ndarray of complex, shape (M,)
    """
    z = np.asarray(z, dtype=complex)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return np.mean(1.0 / (z[..., None] - eigenvalues), axis=-1)


def S_transform(eigenvalues, w_vals, tol=1e-10, max_iter=100):
    """
    Compute S-transform of the ESM of a matrix with given eigenvalues.

    Parameters
    ----------
    eigenvalues : array_like of real, shape (n,)
        Eigenvalues defining the empirical spectral measure.
    w_vals : array_like of complex
        Points where S(w) is evaluated.
    tol : float
        Convergence tolerance on the Newton step size.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    S_vals : ndarray of complex, shape (len(w_vals),)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    w_vals = np.asarray(w_vals, dtype=complex)

    if len(w_vals) == 0:
        return np.array([], dtype=complex)

    mean_ev = np.mean(eigenvalues)
    z = mean_ev / w_vals

    for _ in range(max_iter):
        Gz   = stieltjes_transform(z, eigenvalues)
        d    = 1.0 / (z[..., None] - eigenvalues)
        dGz  = -np.mean(d**2, axis=-1)
        fz   = z * Gz - 1.0 - w_vals
        dfz  = Gz + z * dGz
        step = np.where(np.abs(dfz) > 1e-14, fz / dfz, 0.0 + 0j)
        z   -= step
        if not (np.abs(step) >= tol).any():
            break

    return (1.0 + w_vals) / w_vals * z


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

    mask = np.ones(M, dtype=bool)   # True = non-support point
    zj_list, fj_list = [], []
    C_cols = []                     # columns of the Cauchy matrix
    R = np.full(M, np.mean(F))
    wj = None

    for _ in range(min(mmax, M - 1)):
        J_arr = np.where(mask)[0]
        err = np.abs(F[J_arr] - R[J_arr])

        if err.max() <= tol * F_scale:
            break

        j_star = J_arr[np.argmax(err)]
        zj_list.append(z[j_star])
        fj_list.append(F[j_star])
        mask[j_star] = False

        C_cols.append(1.0 / (z - zj_list[-1]))
        C = np.column_stack(C_cols)

        fj = np.array(fj_list)
        J_arr = np.where(mask)[0]

        # Loewner matrix: L[i,k] = (F[i] - fj[k]) / (z[i] - zj[k])
        #                        = (F[i] - fj[k]) * C[i,k]
        L = (F[J_arr, None] - fj[None, :]) * C[J_arr, :]

        # Weights = right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(L, full_matrices=False)
        wj = Vh[-1, :]

        # Update rational approximant at non-support points
        N = C[J_arr, :] @ (wj * fj)
        D = C[J_arr, :] @ wj
        safe = np.abs(D) > 0
        R[J_arr[safe]] = N[safe] / D[safe]

    return np.array(zj_list), np.array(fj_list), wj


def _aaa_poles_residues(zj, fj, wj):
    """
    Extract poles and residues from an AAA barycentric approximant.

    Poles are roots of the denominator polynomial
        p(z) = sum_k w_k * prod_{j != k} (z - z_j),
    computed via polynomial arithmetic and nppoly.polyroots.

    Residues are computed as N(pole) / D'(pole) using the barycentric
    representations of the numerator N and derivative D'.

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

    # Denominator polynomial p(z) = sum_k w_k * prod_{j!=k}(z - z_j)
    # q(z) = prod_k (z - z_k), degree m, ascending coefficients
    q = nppoly.polyfromroots(zj)
    p = np.zeros(m, dtype=complex)   # degree m-1, ascending
    for k in range(m):
        quot, _ = nppoly.polydiv(q, [-zj[k], 1.0])
        p += wj[k] * quot

    poles = nppoly.polyroots(p) if m > 1 else np.array([], dtype=complex)

    # Residues: res_i = N(pole_i) / D'(pole_i)
    #   N(z)  = sum_k w_k f_k / (z - z_k)
    #   D'(z) = -sum_k w_k / (z - z_k)^2
    residues = np.empty(len(poles), dtype=complex)
    for i, pole in enumerate(poles):
        d = pole - zj
        if np.min(np.abs(d)) < 1e-14:
            residues[i] = np.nan
            continue
        N_val  =  np.sum(wj * fj / d)
        Dp_val = -np.sum(wj / d**2)
        residues[i] = N_val / Dp_val

    return poles, residues


def eigenvalues_from_G(z_vals, G_vals, k, tol=1e-13):
    """
    Recover k eigenvalues of mu^A from (z, G) data on the negative real axis.

    Uses the AAA algorithm (Nakatsukasa, Sète, Trefethen 2018) to build a
    barycentric rational approximant to G^A, then extracts poles and enforces
    the Stieltjes constraints:
      - poles must be real and positive  (support of mu^A lies in (0, inf))
      - residues must be real and positive  (mu^A is a positive measure)

    Poles failing either constraint are discarded as spurious. This can happen
    when mu^A has a nontrivial continuous component (the k-atom model is then
    only approximate), or when AAA overfits with too many support points.

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

    # Enforce real positive poles & residues
    imag_tol = 1e-6
    valid = (
        np.isfinite(residues)
        & (np.abs(poles.imag)    < imag_tol * (np.abs(poles.real)    + 1e-30))
        & (np.abs(residues.imag) < imag_tol * (np.abs(residues.real) + 1e-30))
        & (poles.real    > 0)
        & (residues.real > 0)
    )

    candidates = np.sort(poles[valid].real)[::-1]
    return candidates[:k]


def S_inverse(w_vals, S_vals, k):
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

    Returns
    -------
    eigenvalues : ndarray, shape (<= k,)
        Corrected eigenvalues, sorted descending. Fewer than k may be returned
        if fewer than k poles survive Stieltjes filtering in eigenvalues_from_G.
    """
    z_vals, G_vals = psi_inverse(w_vals, S_vals)
    eigenvalues = eigenvalues_from_G(z_vals, G_vals, k)
    if len(eigenvalues) < k:
        warnings.warn(
            f"Only {len(eigenvalues)} of {k} requested eigenvalues were recovered. "
            "Consider using a finer w-grid or check that the input measure is discrete.",
            RuntimeWarning,
            stacklevel=2,
        )

    return eigenvalues

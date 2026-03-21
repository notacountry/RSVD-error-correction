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
from scipy.linalg import eig as scipy_eig


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

    m = len(eigenvalues)

    # Separate nonzero and zero eigenvalues.
    # Zero eigenvalues contribute a 1/z pole to G(z), making ψ′(z) → 0 at
    # z = 0.  Near the left ψ-domain boundary w → −n_nz/m, the correct
    # solution z_★ → 0⁻ and Newton's convergence degenerates (step → ∞).
    #
    # Mathematical fix: use the identity
    #   ψ_Y(z) = (n_nz/m) · ψ_nz(z)
    # where ψ_nz is the ψ-function of the NONZERO probability sub-measure
    # (n_nz eigenvalues, normalised to sum to 1).  Solving ψ_Y(z) = w is
    # then equivalent to ψ_nz(z) = α·w where α = m/n_nz.  With no 1/z
    # contribution, ψ_nz′ is bounded away from zero and Newton is
    # well-conditioned throughout w ∈ (−1, 0).
    ev_max = eigenvalues.max()
    nz_mask = eigenvalues > (ev_max * 1e-8 if ev_max > 0 else 0.0)
    eigs_nz = eigenvalues[nz_mask]
    n_nz = len(eigs_nz)

    if n_nz == 0:
        return np.full_like(w_vals, np.nan)

    alpha   = m / n_nz            # rescaling factor: w′ = α·w ∈ (−1, 0)
    w_prime = alpha * w_vals      # domain of the nonzero probability measure

    # Initial guess: z = mean_nz / w′ is always negative for w′ < 0.
    mean_nz = eigs_nz.mean()
    z = mean_nz / w_prime

    # Newton on ψ_nz(z) − w′ = 0 in the t = log(−z) parametrisation.
    #
    # Standard Newton in z-space has f(z) = ψ_nz(z) − w′ and
    # f′(z) = ψ_nz′(z) which is tiny when |z| >> max(λᵢ), causing the
    # step f/f′ to overshoot past z = 0 on the first iteration.
    #
    # In t-space (z = −exp(t)):
    #   f_t′ = z · ψ_nz′(z) = z · (Gz + z·dGz)
    #        = z · dfz   (always positive since z < 0 and dfz < 0)
    # This is O(1) rather than O(1/z²), so Newton is well-conditioned
    # throughout the domain and can never cross z = 0.
    # Work in real arithmetic: z < 0 always, so t = log(−z) is real.
    # Complex drift would let t.real grow unboundedly → exp overflow.
    t = np.log(-z.real)   # real array: t = log(−z), so z = −exp(t)

    for _ in range(max_iter):
        z    = -np.exp(t)
        Gz   = stieltjes_transform(z, eigs_nz).real
        d    = 1.0 / (z[..., None] - eigs_nz)
        dGz  = -np.mean(d**2, axis=-1).real
        fz   = (z * Gz - 1.0 - w_prime.real)
        dfz  = Gz + z * dGz           # ψ_nz′(z) < 0
        # t-space derivative: f_t′ = z · dfz  (> 0 since z < 0 and dfz < 0)
        dft  = z * dfz
        safe = np.abs(dft) > 1e-14
        step = np.where(safe, fz / np.where(safe, dft, 1.0), 0.0)
        # Clamp step to at most 3 in log-space (|z| changes by ≤ e^3 ≈ 20x per
        # iteration) and keep t bounded to prevent exp overflow.
        t   -= np.clip(step, -3.0, 3.0)
        t    = np.clip(t, -600.0, 600.0)
        if not (np.abs(step) >= tol).any():
            break

    z = -np.exp(t)

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

        with np.errstate(divide="ignore"):
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
    # psi_inverse can produce inf/NaN near the degenerate boundary of the MP
    # distribution (z → 0 when S_A → 0).  Discard those points so AAA receives
    # only well-behaved data.
    # Valid Stieltjes data requires z < 0 and G < 0 (both become positive when
    # Newton converges to the wrong branch near the ψ-domain left boundary,
    # where ψ′(z)→0 makes Newton ill-conditioned and causes it to overshoot to z>0).
    finite = (
        np.isfinite(z_vals) & np.isfinite(G_vals)
        & (z_vals < 0) & (G_vals < 0)
    )
    eigenvalues = eigenvalues_from_G(z_vals[finite], G_vals[finite], k)
    if len(eigenvalues) < k:
        warnings.warn(
            f"Only {len(eigenvalues)} of {k} requested eigenvalues were recovered. "
            "Consider using a finer w-grid or check that the input measure is discrete.",
            RuntimeWarning,
            stacklevel=2,
        )

    return eigenvalues

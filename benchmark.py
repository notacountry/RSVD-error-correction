"""
Benchmark utilities for RSVD eigenvalue correction.

Separates evaluation logic from test matrix generation so that test_matrices.py
remains a pure data-generation module.
"""
import numpy as np

from rsvd import rsvd


def run_test(name, A, sigma_true, k, p, seed=None, verbose=True):
    """
    Run plain and corrected RSVD on A and return a comparison result.

    Parameters
    ----------
    name : str
        Label for this test case.
    A : ndarray
        Input matrix.
    sigma_true : ndarray, shape (k,)
        Ground-truth singular values.
    k : int
        Target rank passed to rsvd.
    p : int
        Oversampling parameter passed to rsvd.
    seed : int or None
        Random seed passed to rsvd.
    verbose : bool, default True
        If True, print the comparison table.

    Returns
    -------
    result : dict with keys:
        "name"      : str   — the test label
        "rsvd_rmse" : float — RMSE of plain RSVD against sigma_true
        "corr_rmse" : float — RMSE of corrected RSVD against sigma_true
    """
    _, sigma_rsvd, _ = rsvd(A, k=k, p=p, seed=seed)
    _, sigma_corr, _ = rsvd(A, k=k, p=p, seed=seed, correction=True)

    rsvd_rmse = np.sqrt(np.mean((sigma_rsvd - sigma_true) ** 2))
    corr_rmse = np.sqrt(np.mean((sigma_corr - sigma_true) ** 2))

    if verbose:
        print(f"=== {name} ===")
        print(f"RSVD RMSE:      {rsvd_rmse}")
        print(f"Corrected RMSE: {corr_rmse}")

    return {"name": name, "rsvd_rmse": rsvd_rmse, "corr_rmse": corr_rmse}

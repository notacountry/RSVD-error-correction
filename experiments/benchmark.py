"""
Benchmark utilities for RSVD eigenvalue correction.

Separates evaluation logic from matrix generation so that matrix_generators.py
remains a pure data-generation module.
"""
import numpy as np

from rsvd_correction.rsvd import rsvd


def run_benchmark(name, A, sigma_true, k, p, seed=None):
    """
    Run plain and corrected RSVD on A and return a comparison result.

    Parameters
    ----------
    name : str
        Label for this benchmark case.
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

    Returns
    -------
    result : dict with keys:
        "name"      : str   — the benchmark label
        "rsvd_rmse" : float — RMSE of plain RSVD against sigma_true
        "corr_rmse" : float — RMSE of corrected RSVD against sigma_true
    """
    sigma_rsvd, sigma_corr = rsvd_pair(A, k=k, p=p, seed=seed)

    rsvd_rmse = np.sqrt(np.mean((sigma_rsvd - sigma_true) ** 2))
    corr_rmse = np.sqrt(np.mean((sigma_corr - sigma_true) ** 2))

    return {"name": name, "rsvd_rmse": rsvd_rmse, "corr_rmse": corr_rmse}


def rsvd_pair(A, k, p, seed):
    """
    Run plain and corrected RSVD.

    Returns
    -------
    s_plain, s_corr : ndarray
    """
    _, s_plain, _ = rsvd(A, k=k, p=p, seed=seed)
    _, s_corr,  _ = rsvd(A, k=k, p=p, seed=seed, correction=True)
    return s_plain, s_corr


def harmonic_signal(k, amplitude=10.0):
    """Return amplitude / i for i = 1, ..., k."""
    return amplitude / np.arange(1, k + 1)


def print_results(result):
    """Print a formatted summary of a run_benchmark result."""
    print(f"=== {result['name']} ===")
    print(f"RSVD RMSE:      {result['rsvd_rmse']}")
    print(f"Corrected RMSE: {result['corr_rmse']}")

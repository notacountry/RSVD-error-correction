"""
Benchmark utilities for RSVD eigenvalue correction.
"""
import numpy as np

from rsvd_correction.matrix_generators import SignalPlusNoise
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


def k_from_c(N, c, p):
    """Target rank K such that sketch aspect ratio c = N / (K + p)."""
    return round(N / c) - p


def make_spiked_gen(K, noise_level):
    """
    Return (SignalPlusNoise generator, sigma_signal) for a harmonic signal of rank K.

    Parameters
    ----------
    K : int
        Rank of the signal component.
    noise_level : float
        Noise scale passed to SignalPlusNoise.

    Returns
    -------
    gen : SignalPlusNoise
    sigma_signal : ndarray, shape (K,)
    """
    sigma = harmonic_signal(K)
    return SignalPlusNoise(sigma_signal=sigma, noise_level=noise_level), sigma


def _trial_worker(args):
    gen, N, K, p, seed = args
    A, _ = gen(n=N, k=K, seed=seed)
    return rsvd_pair(A, k=K, p=p, seed=seed)


def run_trials(gen, N, K, p, n_seeds, n_jobs=None):
    """
    Run rsvd_pair for each seed in range(n_seeds).

    Parameters
    ----------
    gen : MatrixGenerator
        Matrix generator instance.
    N : int
        Matrix dimension.
    K : int
        Target rank.
    p : int
        Oversampling parameter.
    n_seeds : int
        Number of independent trials (seeds 0 .. n_seeds-1).
    n_jobs : int or None
        Number of parallel workers. None uses all available CPU cores. 1 runs sequentially.

    Returns
    -------
    list of (s_plain, s_corr) ndarrays, one per seed.
    """
    args = [(gen, N, K, p, seed) for seed in range(n_seeds)]
    if n_jobs == 1:
        return [_trial_worker(a) for a in args]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_trial_worker, args))


def print_results(result):
    """Print a formatted summary of a run_benchmark result."""
    print(f"{result['name']}")
    print(f"RSVD RMSE:      {result['rsvd_rmse']}")
    print(f"Corrected RMSE: {result['corr_rmse']}")

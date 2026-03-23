"""
Paired one-sided t-test: does corrected RSVD reduce per-trial MSE?
H0: E[D] = 0; H1: E[D] < 0
"""
import warnings

import numpy as np
from scipy import stats

from rsvd_correction.rsvd import rsvd
from rsvd_correction.matrix_generators import SignalPlusNoise


def run_hypothesis_test(configs, n_trials=100, p=10, alpha=0.05):
    """
    Run a paired one-sided t-test for each (N, c, noise) configuration.

    Parameters
    ----------
    configs : list of (N, c_target, noise_level) tuples
    n_trials : int
        Number of paired trials per configuration.
    p : int
        RSVD oversampling parameter.
    alpha : float
        Significance level.
    """
    header = (
        f"{'N':>6} {'K':>5} {'c':>5} {'noise':>6} | "
        f"{'D_bar':>10} {'s_D':>10} {'t':>8} {'p':>8} | "
        f"{'CI (-inf, hi)':>20} | {'reject?':>7}"
    )
    print(header)
    print("-" * len(header))

    for N, c_target, noise in configs:
        K = round(N / c_target) - p
        if K < 1:
            continue
        actual_c     = N / (K + p)
        sigma_signal = np.array([10.0 / (i + 1) for i in range(K)])
        gen = SignalPlusNoise(sigma_signal=sigma_signal, noise_level=noise)

        D = []
        for seed in range(n_trials):
            A, _ = gen(n=N, k=K, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, s_plain, _ = rsvd(A, k=K, p=p, seed=seed)
                    _, s_corr,  _ = rsvd(A, k=K, p=p, seed=seed, correction=True)
                except Exception:
                    continue

            mse_rsvd = np.mean((s_plain - sigma_signal) ** 2)
            mse_corr = np.mean((s_corr  - sigma_signal) ** 2)
            D.append(mse_corr - mse_rsvd)

        D = np.array(D)
        n = len(D)
        if n < 2:
            continue

        D_bar = np.mean(D)
        s_D   = np.std(D, ddof=1)
        t_obs = D_bar / (s_D / np.sqrt(n))
        p_val = stats.t.cdf(t_obs, df=n - 1)

        t_crit = stats.t.ppf(1 - alpha, df=n - 1)
        ci_hi  = D_bar + t_crit * s_D / np.sqrt(n)

        reject = "YES" if p_val < alpha else "no"

        print(
            f"{N:>6} {K:>5} {actual_c:>5.2f} {noise:>6.1f} | "
            f"{D_bar:>10.4f} {s_D:>10.4f} {t_obs:>8.3f} {p_val:>8.4f} | "
            f"(-inf, {ci_hi:>8.4f}) | {reject:>7}"
        )

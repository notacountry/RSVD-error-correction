"""
Paired one-sided t-test: does corrected RSVD reduce per-trial MSE?
H0: E[D] = 0; H1: E[D] < 0
"""
import numpy as np
import pandas as pd
from scipy import stats

from experiments.benchmark import k_from_c, make_spiked_gen, run_trials


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

    Returns
    -------
    pd.DataFrame with columns: N, K, c, noise, D_bar, s_D, t, p, CI_hi, reject
    """
    rows = []

    for N, c_target, noise in configs:
        K = k_from_c(N, c_target, p)
        if K < 1:
            continue
        actual_c = N / (K + p)
        gen, sigma_signal = make_spiked_gen(K, noise)

        D = np.array([
            np.mean((s_corr - sigma_signal) ** 2) - np.mean((s_plain - sigma_signal) ** 2)
            for s_plain, s_corr in run_trials(gen, N, K, p, n_trials)
        ])

        n = len(D)
        if n < 2:
            continue

        D_bar = np.mean(D)
        s_D   = np.std(D, ddof=1)
        t_obs = D_bar / (s_D / np.sqrt(n))
        p_val = stats.t.cdf(t_obs, df=n - 1)

        t_crit = stats.t.ppf(1 - alpha, df=n - 1)
        ci_hi  = D_bar + t_crit * s_D / np.sqrt(n)

        rows.append({
            "N":      N,
            "K":      K,
            "c":      round(actual_c, 2),
            "noise":  noise,
            "D_bar":  D_bar,
            "s_D":    s_D,
            "t":      t_obs,
            "p":      p_val,
            "CI_hi":  ci_hi,
            "reject": p_val < alpha,
        })

    return pd.DataFrame(rows)

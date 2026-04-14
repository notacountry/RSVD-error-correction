"""
Sweep N and c = n/l over the spiked signal-plus-noise model, comparing
plain RSVD with the corrected RSVD.

For each (N, c) pair we derive K = round(N/c) - P and measure the error
on the top k_signal singular values over N_SEEDS random seeds.
Ratio < 1 means the correction helps relative to plain RSVD.
"""
from itertools import product

import numpy as np
import pandas as pd

from experiments.benchmark import k_from_c, make_spiked_gen, run_trials


P            = 10
N_SEEDS      = 5

NOISE_LEVELS = [0.1, 0.5, 1.0, 2.0, 5.0]
N_VALS       = [100, 300, 600, 1000]
C_TARGETS    = [2, 3, 5, 8, 12]


def run_parameter_sweep():
    """
    Run the parameter sweep over noise levels, matrix sizes, and sketch ratios.

    Returns
    -------
    pd.DataFrame with columns: noise, N, K, c, RSVD, Corrected, ratio
    """
    rows = []

    for noise, N, c in product(NOISE_LEVELS, N_VALS, C_TARGETS):
        K = k_from_c(N, c, P)
        if K < 1 or K + P > N:
            continue

        actual_c          = N / (K + P)
        gen, sigma_signal = make_spiked_gen(K, noise)

        trials = run_trials(gen, N, K, P, N_SEEDS)
        if not trials:
            continue

        rsvd_errs = [np.linalg.norm(s_plain - sigma_signal) for s_plain, _ in trials]
        corr_errs = [np.linalg.norm(s_corr  - sigma_signal) for _, s_corr  in trials]

        mean_r    = np.mean(rsvd_errs)
        mean_corr = np.mean(corr_errs)

        rows.append({
            "noise":     noise,
            "N":         N,
            "K":         K,
            "c":         round(actual_c, 2),
            "RSVD":      mean_r,
            "Corrected": mean_corr,
            "ratio":     mean_corr / mean_r,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run_parameter_sweep().to_string(index=False))

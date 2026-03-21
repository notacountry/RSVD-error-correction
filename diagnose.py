"""
Sweep N and c = n/l over the spiked (signal+noise) model to find where
the S-transform correction outperforms plain RSVD.

For each (N, c) pair we derive K = round(N/c) - P and measure the error
on the top k_signal singular values over N_SEEDS random seeds.
Ratio < 1 means the correction helps.
"""
import warnings

import numpy as np

from rsvd import rsvd
from test_matrices import signal_plus_noise

NOISE_LEVEL  = 1.0
P            = 10
N_SEEDS      = 10

N_VALS    = [200, 500, 1000, 2000, 5000]
C_TARGETS = [2, 3, 4, 6, 8, 10, 15, 20]

header = f"{'N':>6} {'K':>6} {'c':>6} | {'RSVD_err':>10} {'Corr_err':>10} {'ratio':>8}"
print(header)
print("-" * len(header))

for N in N_VALS:
    for c in C_TARGETS:
        K = round(N / c) - P
        if K < 1 or K + P > N:
            continue

        actual_c = N / (K + P)
        sigma_signal = np.array([10.0 / (i + 1) for i in range(K)])

        rsvd_errs, corr_errs = [], []
        for seed in range(N_SEEDS):
            A, _ = signal_plus_noise(
                n=N, k=K,
                sigma_signal=sigma_signal,
                noise_level=NOISE_LEVEL,
                seed=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    _, s_rsvd, _ = rsvd(A, k=K, p=P, seed=seed)
                    _, s_corr, _ = rsvd(A, k=K, p=P, seed=seed, correction=True)
                except Exception:
                    continue

            rsvd_errs.append(np.linalg.norm(s_rsvd - sigma_signal))
            corr_errs.append(np.linalg.norm(s_corr - sigma_signal))

        if not rsvd_errs:
            continue

        mean_r = np.mean(rsvd_errs)
        mean_c = np.mean(corr_errs)
        ratio  = mean_c / mean_r
        marker = " <-- BETTER" if ratio < 1.0 else ""

        print(f"{N:>6} {K:>6} {actual_c:>6.2f} | {mean_r:>10.4f} {mean_c:>10.4f} {ratio:>8.3f}{marker}")

    print()

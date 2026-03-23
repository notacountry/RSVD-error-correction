"""
Sweep N and c = n/l over the spiked signal-plus-noise model, comparing
plain RSVD with the corrected RSVD.

For each (N, c) pair we derive K = round(N/c) - P and measure the error
on the top k_signal singular values over N_SEEDS random seeds.
Ratio < 1 means the correction helps relative to plain RSVD.
"""
from itertools import product

import numpy as np

from rsvd_correction.matrix_generators import SignalPlusNoise
from benchmark import harmonic_signal, rsvd_pair


P            = 10
N_SEEDS      = 5

NOISE_LEVELS = [0.1, 0.5, 1.0, 2.0, 5.0]
N_VALS       = [100, 300, 600, 1000]
C_TARGETS    = [2, 3, 5, 8, 12]


def run_parameter_sweep():
    header = (
        f"{'noise':>6} {'N':>6} {'K':>5} {'c':>5} | "
        f"{'RSVD':>9} {'Corrected':>9} | "
        f"{'ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    for noise, N, c in product(NOISE_LEVELS, N_VALS, C_TARGETS):
        K = round(N / c) - P
        if K < 1 or K + P > N:
            continue

        actual_c     = N / (K + P)
        sigma_signal = harmonic_signal(K)
        gen = SignalPlusNoise(sigma_signal=sigma_signal, noise_level=noise)

        rsvd_errs, corr_errs = [], []

        for seed in range(N_SEEDS):
            A, _ = gen(n=N, k=K, seed=seed)
            s_plain, s_corr = rsvd_pair(A, k=K, p=P, seed=seed)

            rsvd_errs.append(np.linalg.norm(s_plain - sigma_signal))
            corr_errs.append(np.linalg.norm(s_corr  - sigma_signal))

        if not rsvd_errs:
            continue

        mean_r    = np.mean(rsvd_errs)
        mean_corr = np.mean(corr_errs)
        ratio     = mean_corr / mean_r
        flag      = " <" if ratio < 1.0 else "  "

        print(
            f"{noise:>6.1f} {N:>6} {K:>5} {actual_c:>5.2f} | "
            f"{mean_r:>9.4f} {mean_corr:>9.4f} | "
            f"{ratio:>6.3f}{flag}"
        )


if __name__ == "__main__":
    run_parameter_sweep()
"""Shared experiment constants used across all notebooks and scripts."""
from experiments.benchmark import harmonic_signal

N           = 5000
K           = 555
P           = 20
SIGMA       = harmonic_signal(K)
SEED        = 42

ALPHA       = 1.0   # PolynomialDecay exponent
BETA        = 0.5   # ExponentialDecay rate
NOISE_LEVEL = 1.0   # SignalPlusNoise noise level

N_TRIALS    = 100

HT_CONFIGS = [
    # best
    (600,  2, 2.0),
    (600,  3, 2.0),
    (600,  5, 2.0),
    (1000, 2, 5.0),
    (1000, 3, 5.0),
    (1000, 5, 5.0),
    # mid
    (600,  3, 1.0),
    (1000, 5, 1.0),
    (1000, 8, 2.0),
    # worst
    (600,  2, 0.5),
    (600,  5, 0.5),
    (600,  8, 2.0),
    (600, 12, 5.0),
    (1000, 12, 5.0),
]

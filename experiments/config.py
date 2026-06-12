"""Shared experiment constants used across all notebooks and scripts."""
from experiments.benchmark import harmonic_signal

N           = 5000
K           = 555
P           = 20
SIGMA       = harmonic_signal(K)
SEED        = 42

SIGNAL_RANK = 50    # BilevelNoise: number of high-tier signal components
SIGNAL_HIGH = 5.0   # BilevelNoise: high-tier singular value
SIGNAL_LOW  = 1.0   # BilevelNoise: low-tier singular value

ALPHA       = 1.5   # PowerLawNoise decay exponent
BETA        = 0.5   # ExponentialNoise decay rate
NOISE_LEVEL = 0.5   # noise level for all SignalPlusNoise-style generators

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

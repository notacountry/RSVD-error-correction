"""
Test matrix generators for RSVD eigenvalue correction experiments.
"""
from abc import ABC, abstractmethod

import numpy as np


class MatrixGenerator(ABC):
    """Callable that produces a random matrix and its true top-k singular values."""

    name: str

    @abstractmethod
    def __call__(
        self, n: int, k: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        n : int
            Matrix dimension.
        k : int
            Number of top singular values to return.
        seed : int or None

        Returns
        -------
        A : (n, n) ndarray
        sigma_true : (k,) ndarray
            Top-k exact singular values.
        """


def _random_orthonormal(n, k, rng):
    """Return a (n, k) matrix with orthonormal columns."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return Q


def _random_svd_matrix(sigma, n, rng):
    """Return U @ diag(sigma) @ V.T with random orthonormal U, V."""
    U = _random_orthonormal(n, len(sigma), rng)
    V = _random_orthonormal(n, len(sigma), rng)
    return U @ np.diag(sigma) @ V.T


class ExactLowRank(MatrixGenerator):
    """
    A = U diag(sigma) Vt, exactly rank k.

    Parameters
    ----------
    sigma : array_like of shape (k,)
        Singular values.
    """

    name = "Exact low-rank"

    def __init__(self, sigma):
        self.sigma = np.asarray(sigma, dtype=float)

    def __call__(self, n: int, k: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        return _random_svd_matrix(self.sigma, n, rng), np.sort(self.sigma)[::-1][:k]


class SignalPlusNoise(MatrixGenerator):
    """
    A = U diag(sigma_signal) Vt + (noise_level / sqrt(n)) * G,
    G ~ N(0,1)^{n x n}.

    Parameters
    ----------
    sigma_signal : array_like of shape (k,)
        Signal singular values.
    noise_level : float
        Scales the Gaussian noise matrix.
    """

    name = "Signal plus noise"

    def __init__(self, sigma_signal, noise_level: float = 1.0):
        self.sigma_signal = np.asarray(sigma_signal, dtype=float)
        self.noise_level = noise_level

    def __call__(self, n: int, k: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        signal = _random_svd_matrix(self.sigma_signal, n, rng)
        noise = (self.noise_level / np.sqrt(n)) * rng.standard_normal((n, n))
        return signal + noise, np.sort(self.sigma_signal)[::-1][:k]


class BilevelNoise(MatrixGenerator):
    """
    Two-tier signal + Gaussian noise.
    sigma_j = signal_high for j <= signal_rank, signal_low otherwise.

    Controls the spectral gap (signal_high / signal_low); small gaps are the
    hard case for subspace separation. See Romanov (arXiv:2305.17435).

    Parameters
    ----------
    signal_rank : int
        Number of high-tier signal components.
    signal_high : float
        Singular value for the high tier.
    signal_low : float
        Singular value for the low tier.
    noise_level : float
        Scales the Gaussian noise matrix.
    """

    name = "Bilevel noise"

    def __init__(self, signal_rank: int, signal_high: float, signal_low: float, noise_level: float = 1.0):
        self.signal_rank = signal_rank
        self.signal_high = signal_high
        self.signal_low = signal_low
        self.noise_level = noise_level

    def __call__(self, n: int, k: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        sigma_signal = np.where(np.arange(1, n + 1) <= self.signal_rank, self.signal_high, self.signal_low)
        signal = _random_svd_matrix(sigma_signal, n, rng)
        noise = (self.noise_level / np.sqrt(n)) * rng.standard_normal((n, n))
        return signal + noise, np.sort(sigma_signal)[::-1][:k]


class PowerLawNoise(MatrixGenerator):
    """
    Power-law signal sigma_j = j^{-alpha} + Gaussian noise.

    alpha=1 (slow decay, hard case) and alpha=4 (fast decay, easy) bracket the
    range of interest. See Halko, Martinsson, Tropp (2011) Section 7.

    Parameters
    ----------
    alpha : float
        Decay exponent. Larger values give faster decay.
    noise_level : float
        Scales the Gaussian noise matrix.
    """

    name = "Power-law noise"

    def __init__(self, alpha: float = 1.0, noise_level: float = 1.0):
        self.alpha = alpha
        self.noise_level = noise_level

    def __call__(self, n: int, k: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        sigma_signal = 1.0 / np.arange(1, n + 1) ** self.alpha
        signal = _random_svd_matrix(sigma_signal, n, rng)
        noise = (self.noise_level / np.sqrt(n)) * rng.standard_normal((n, n))
        return signal + noise, sigma_signal[:k]


class ExponentialNoise(MatrixGenerator):
    """
    Exponential signal sigma_j = exp(-beta * (j-1)) + Gaussian noise.

    The easy baseline: signal decays fast so rank-k approximation is nearly
    optimal without power iteration. See Halko, Martinsson, Tropp (2011).

    Parameters
    ----------
    beta : float
        Decay rate. Larger values give faster decay.
    noise_level : float
        Scales the Gaussian noise matrix.
    """

    name = "Exponential noise"

    def __init__(self, beta: float = 0.5, noise_level: float = 1.0):
        self.beta = beta
        self.noise_level = noise_level

    def __call__(self, n: int, k: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        sigma_signal = np.exp(-self.beta * np.arange(n))
        signal = _random_svd_matrix(sigma_signal, n, rng)
        noise = (self.noise_level / np.sqrt(n)) * rng.standard_normal((n, n))
        return signal + noise, sigma_signal[:k]

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
        return _random_svd_matrix(self.sigma, n, rng), np.sort(self.sigma)[::-1]


class DiagonalKnownSpectrum(MatrixGenerator):
    """
    A = diag(sigma_1, ..., sigma_k, 0, ..., 0).

    Parameters
    ----------
    sigma : array_like of shape (k,)
        Singular values.
    """

    name = "Diagonal known spectrum"

    def __init__(self, sigma):
        self.sigma = np.asarray(sigma, dtype=float)

    def __call__(self, n: int, k: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = np.zeros(n)
        d[:k] = np.sort(self.sigma)[::-1]
        return np.diag(d), d[:k]


class PolynomialDecay(MatrixGenerator):
    """
    Singular values sigma_i = i^{-alpha}, i = 1, ..., n.

    Parameters
    ----------
    alpha : float
        Decay exponent.  alpha=1 gives sigma_i = 1/i.
    """

    name = "Polynomial decay"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, n: int, k: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        sigma = 1.0 / np.arange(1, n + 1) ** self.alpha
        return _random_svd_matrix(sigma, n, rng), sigma[:k]


class ExponentialDecay(MatrixGenerator):
    """
    Singular values sigma_i = exp(-beta * i), i = 0, ..., n-1.

    Parameters
    ----------
    beta : float
        Decay rate.
    """

    name = "Exponential decay"

    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def __call__(self, n: int, k: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        sigma = np.exp(-self.beta * np.arange(n))
        return _random_svd_matrix(sigma, n, rng), sigma[:k]


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
        return signal + noise, np.sort(self.sigma_signal)[::-1]

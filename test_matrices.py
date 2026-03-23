"""
Test matrices for RSVD eigenvalue correction.
"""
import numpy as np


def _random_orthonormal(n, k, rng):
    """Return a (n, k) matrix with orthonormal columns."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return Q


def exact_low_rank(n, k, sigma, seed=None):
    """
    A = U diag(sigma) Vt, exactly rank k.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Rank.
    sigma : array_like of shape (k,)
        Singular values.
    seed : int or None

    Returns
    -------
    A : (n, n) ndarray
    sigma_true : (k,) ndarray
        Top-k exact singular values.
    """
    rng = np.random.default_rng(seed)
    sigma = np.asarray(sigma, dtype=float)
    U = _random_orthonormal(n, k, rng)
    V = _random_orthonormal(n, k, rng)
    A = U @ np.diag(sigma) @ V.T
    return A, np.sort(sigma)[::-1]


def diagonal_known_spectrum(n, k, sigma, seed=None):
    """
    A = diag(sigma_1, ..., sigma_k, 0, ..., 0).

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Number of nonzero singular values.
    sigma : array_like of shape (k,)
        Singular values.
    seed : int or None
        Unused; present for a consistent interface.

    Returns
    -------
    A : (n, n) ndarray
    sigma_true : (k,) ndarray
        Top-k exact singular values.
    """
    sigma = np.asarray(sigma, dtype=float)
    d = np.zeros(n)
    d[:k] = np.sort(sigma)[::-1]
    A = np.diag(d)
    return A, d[:k]


def polynomial_decay(n, k, alpha=1.0, seed=None):
    """
    Singular values sigma_i = i^{-alpha}, i = 1, ..., n.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Number of singular values to treat as signal (top k returned).
    alpha : float
        Decay exponent.  alpha=1 gives sigma_i = 1/i.
    seed : int or None

    Returns
    -------
    A : (n, n) ndarray
    sigma_true : (k,) ndarray
        Top-k exact singular values.
    """
    rng = np.random.default_rng(seed)
    sigma = 1.0 / np.arange(1, n + 1) ** alpha
    U = _random_orthonormal(n, n, rng)
    V = _random_orthonormal(n, n, rng)
    A = U @ np.diag(sigma) @ V.T
    return A, sigma[:k]


def exponential_decay(n, k, beta=0.5, seed=None):
    """
    Singular values sigma_i = exp(-beta * i), i = 0, ..., n-1.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Number of singular values to treat as signal (top k returned).
    beta : float
        Decay rate.
    seed : int or None

    Returns
    -------
    A : (n, n) ndarray
    sigma_true : (k,) ndarray
        Top-k exact singular values.
    """
    rng = np.random.default_rng(seed)
    sigma = np.exp(-beta * np.arange(n))
    U = _random_orthonormal(n, n, rng)
    V = _random_orthonormal(n, n, rng)
    A = U @ np.diag(sigma) @ V.T
    return A, sigma[:k]
 

def signal_plus_noise(n, k, sigma_signal, noise_level=1.0, seed=None):
    """
    A = U diag(sigma_signal) Vt + (noise_level / sqrt(n)) * G,
    G ~ N(0,1)^{n x n}. 
    
    Parameters
    ----------
    n : int
        Matrix dimension (square).
    k : int
        Rank of the signal component.
    sigma_signal : array_like of shape (k,)
        Signal singular values.
    noise_level : float
        Scales the Gaussian noise matrix.
    seed : int or None

    Returns
    -------
    A : (n, n) ndarray
    sigma_true : (k,) ndarray
        Top-k exact singular values.
    """
    rng = np.random.default_rng(seed)
    sigma_signal = np.asarray(sigma_signal, dtype=float)
    U = _random_orthonormal(n, k, rng)
    V = _random_orthonormal(n, k, rng)
    signal = U @ np.diag(sigma_signal) @ V.T
    noise = (noise_level / np.sqrt(n)) * rng.standard_normal((n, n))
    A = signal + noise
    return A, np.sort(sigma_signal)[::-1]

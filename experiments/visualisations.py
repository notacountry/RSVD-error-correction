"""
Visualisation helpers for RSVD eigenvalue correction experiments.
"""
import numpy as np
import matplotlib.pyplot as plt

from rsvd_correction.free_probability import sketch_spectral_info
from rsvd_correction.rsvd import _rsvd_sketch, rsvd


def generate_results(gen, n, k, p, seed):
    """
    Sample one matrix from gen and run plain + corrected RSVD (with and without BBP).

    Returns
    -------
    dict with keys:
        sigma_true      : (k,) true singular values
        Sigma_rsvd      : (k,) uncorrected RSVD singular values
        Sigma_corr      : (k,) corrected RSVD singular values (S-transform)
        Sigma_corr_bbp  : (k,) corrected RSVD singular values (S-transform + BBP spike protection)
        eigs_corr       : corrected eigenvalues of AA^T from S-transform deconvolution
    """
    A, sigma_true = gen(n=n, k=k, seed=seed)
    Y, m, n_cols, l, _, Sigma_rsvd, _ = _rsvd_sketch(A, k, p, seed)
    _, Sigma_corr, _ = rsvd(A, k, p=p, seed=seed, correction=True, bbp=False)
    _, Sigma_corr_bbp, _ = rsvd(A, k, p=p, seed=seed, correction=True, bbp=True)
    _, eigs_corr, _ = sketch_spectral_info(Y, m, n_cols, l, k, Sigma=Sigma_rsvd, bbp=False)
    return dict(
        sigma_true=sigma_true,
        Sigma_rsvd=Sigma_rsvd,
        Sigma_corr=Sigma_corr,
        Sigma_corr_bbp=Sigma_corr_bbp,
        eigs_corr=eigs_corr
    )


def plot_generator(name, results):
    """
    1×2 figure for one generator: eigenvalue spectrum (left) and singular value
    estimates vs true (right), comparing uncorrected, S-transform corrected,
    and BBP-corrected estimates.
    """
    sigma_true = results["sigma_true"]
    Sigma_rsvd = results["Sigma_rsvd"]
    Sigma_corr = results["Sigma_corr"]
    Sigma_corr_bbp = results["Sigma_corr_bbp"]
    eigs_corr  = results["eigs_corr"]
    k = len(sigma_true)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(name)

    _plot_eigenvalue_spectrum(ax1, sigma_true, eigs_corr, Sigma_rsvd)
    _plot_singular_values(ax2, sigma_true, Sigma_corr, Sigma_corr_bbp, Sigma_rsvd, k)

    plt.tight_layout()
    plt.show()


def _plot_eigenvalue_spectrum(ax, sigma_true, eigs_corr, Sigma_rsvd):
    all_eigs = np.concatenate([sigma_true ** 2, eigs_corr, Sigma_rsvd ** 2])
    bins      = np.linspace(0, all_eigs.max() * 1.05, 30)

    counts_true, _ = np.histogram(sigma_true ** 2, bins=bins, density=True)
    counts_corr, _ = np.histogram(eigs_corr,       bins=bins, density=True)
    counts_rsvd, _ = np.histogram(Sigma_rsvd ** 2, bins=bins, density=True)

    centres   = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    bar_width = bin_width * 0.25

    ax.bar(centres - bar_width, counts_true, width=bar_width, color="green",  label=r"True $AA^\top$ spectrum")
    ax.bar(centres,             counts_corr, width=bar_width, color="orange", label="Corrected RSVD spectrum")
    ax.bar(centres + bar_width, counts_rsvd, width=bar_width, color="red",    label="Uncorrected RSVD spectrum")

    ax.set_xlabel(r"Eigenvalue $\lambda$")
    ax.set_ylabel("Density")
    ax.set_title(r"Spectrum of $\frac{1}{\ell}YY^\top$")
    ax.legend()


def _plot_singular_values(ax, sigma_true, Sigma_corr, Sigma_corr_bbp, Sigma_rsvd, k):
    indices = np.arange(1, k + 1)
    ax.plot(indices, sigma_true,      color="green",  lw=1.5, label=r"True $\sigma_i$")
    ax.plot(indices, Sigma_corr,      color="orange", lw=1, label="Corrected (S-transform)")
    ax.plot(indices, Sigma_corr_bbp,  color="blue",   lw=1, label="Corrected (S-transform + BBP)")
    ax.plot(indices, Sigma_rsvd,      color="red",    lw=1, linestyle="--", label="Uncorrected RSVD")
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_title("Singular value estimates vs. true")
    ax.legend()
    ax.grid(True, alpha=0.3)

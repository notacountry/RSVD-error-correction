# Error Correction in Randomized SVD
Below is a brief overview of my research.

## Why this question

The truncated singular value decomposition (TSVD) gives the optimal rank $k$ approximation under unitarily invariant norms [Mirsky, 1960]. However, its computational cost
$$O(mn\min{m,n})$$
is prohibitive for large matrices.

Randomized SVD (RSVD) reduces this to
$$O(mn\ell)$$
while maintaining accuracy [Halko et al., 2010].

However, RSVD introduces **nonlinear distortion in singular values**. In particular, below a threshold, recovery becomes unreliable [Romanov, 2023].

This work uses **free probability theory** [Voiculescu, 1992; Rao, 2006] to correct these distortions.

---

## Randomized Singular Value Decomposition

Given $A \in \mathbb{R}^{N \times N}$, target rank $k$:

1. Choose $\ell = k + p$
2. Draw Gaussian sketch:
$$\Omega \in \mathbb{R}^{N \times \ell}, \quad \Omega_{ij}\sim \mathcal{N}(0,1)$$
3. Compute:
$$Y = A\Omega$$
1. QR decomposition and projection:
$$Y = QR, \quad B = Q^\top A$$
2. SVD:
$$B = \widetilde U \Sigma V^\top$$
1. Reconstruct:
$$U = Q \widetilde U, \quad A \approx U_k \Sigma_k V_k^\top$$

---

## Motivation

Write:
$$Y = [A\omega_1, \dots, A\omega_\ell], \quad \omega_j \sim \mathcal{N}(0, I)$$

Then:
$$\mathbb{E}(y_j) = 0, \quad \mathrm{Cov}(y_j) = AA^\top$$

Thus:
$$y_j \sim \mathcal{N}(0, AA^\top)$$

and:
$$\frac{1}{\ell}YY^\top = \frac{1}{\ell} \sum y_j y_j^\top$$

is a **sample covariance matrix**.

This connects RSVD to **random matrix theory**, specifically the **Marchenko–Pastur law**.

---

## Methods

### Corrected RSVD via Free Probability

Define:

* $\mu_N$: ESM of $W_N = \frac{1}{\ell}\Omega\Omega^\top$
* $\mu_N^Y$: ESM of $\frac{1}{\ell}YY^\top$
* $\mu^A$: ESM of $AA^\top$

Using rotational invariance:
$$\frac{1}{\ell}YY^\top \stackrel{d}{=} (AA^\top)^{1/2} W_N (AA^\top)^{1/2}$$

---

### Free Multiplicative Convolution

$$\mu_N^Y \approx \mu^A \boxtimes \mu$$

Thus:
$$\mu^A \approx \mu_N^Y \boxtimes^{-1} \mu$$

---

### S-transform

$$S^Y(z) = S^A(z) S^{\mathrm{MP}}(z)$$

For Marchenko–Pastur:
$$S^{\mathrm{MP}}(z) = \frac{1}{1 + cz}$$

Hence:
$$S^A(z) = S^Y(z)(1 + cz)$$

After inversion:
$$\sigma_i^{\mathrm{corr}} = \sqrt{\lambda_i^{\mathrm{corr}}}$$

Final approximation:
$$A \approx U_k \Sigma_k^{\mathrm{corr}} V_k^\top$$

---

## Hypothesis Testing

Define:
$$\mathrm{MSE}_j^m = \frac{1}{k} \sum_{i=1}^k (\hat\sigma_{i,j}^{(m)} - \sigma_i)^2$$

Difference:
$$D_j = \mathrm{MSE}_j^{\mathrm{corr}} - \mathrm{MSE}_j^{\mathrm{RSVD}}$$

Test:
$$H_0: \mathbb{E}[D] \ge 0, \quad H_1: \mathbb{E}[D] < 0$$

Statistic:
$$t = \frac{\overline D}{\sqrt{s_D^2 / n}} \sim t_{n-1}$$

Reject if:
$$p < 0.05$$

---

## Spiked Signal + Noise Model

$$A = U \operatorname{diag}(\sigma_1,\dots,\sigma_k) V^\top + \frac{\sigma_{\text{noise}}}{\sqrt{N}} G$$

* $G_{ij} \sim \mathcal{N}(0,1)$
* Produces structured spectral noise

Observation:

* Correction **fails on noiseless data**
* Works well when **noise induces spectral bias**

---

## Results

| N    | K   | c   | noise | mean D  | s_D   | t      | p      | CI_hi  | reject |
| ---- | --- | --- | ----- | ------- | ----- | ------ | ------ | ------ | ------ |
| 300  | 130 | 2.0 | 2.0   | -0.7336 | 0.111 | -66.0  | <1e-3  | -0.715 | Yes    |
| 300  | 40  | 5.0 | 2.0   | -0.9205 | 0.382 | -24.1  | <1e-3  | -0.857 | Yes    |
| 600  | 280 | 2.0 | 2.0   | -0.3790 | 0.055 | -68.6  | <1e-3  | -0.370 | Yes    |
| 600  | 180 | 3.0 | 2.0   | -0.1229 | 1.249 | -0.98  | 0.164  | 0.085  | No     |
| 600  | 100 | 5.0 | 2.0   | -0.4882 | 0.207 | -23.6  | <1e-3  | -0.454 | Yes    |
| 1000 | 480 | 2.0 | 5.0   | -2.1143 | 0.108 | -196.5 | <1e-3  | -2.097 | Yes    |
| 1000 | 180 | 5.0 | 5.0   | -0.0958 | 0.395 | -2.43  | 0.0085 | -0.030 | Yes    |

**Conclusion:**
Correction improves top-$k$ singular value recovery under noisy regimes.

---

## Appendix: S-transform

$$\psi(z) = z m(z) - 1, \quad m(z) = \int \frac{1}{z - x} d\mu(x)$$

$$S(w) = \frac{1+w}{w} \psi^{-1}(w)$$

Key property:
$$S^{\mu \boxtimes \nu}(w) = S^\mu(w) S^\nu(w)$$
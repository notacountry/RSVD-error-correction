Below is a draft paper copy-pasted from LaTeX.

# Error correction in the randomized singular value decomposition of large matrices

## Abstract

The truncated singular value decomposition (TSVD) provides the best rank-$k$ approximation of a matrix under unitarily invariant norms.\citep{mirsky1960} Despite its optimality, SVD is computationally prohibitive, with cost
$O(mn\min{m,n})$ for an $m\times n$ matrix. Randomized SVD (RSVD) makes use of random matrix sketching to decrease compute cost to that of $O(mn\ell)$, where $\ell$ is the target dimension plus some oversampling, whilst preserving accuracy.\citep{halko2010}

However, RSVD introduces a nonlinear distortion of individual singular values. Romanov (2023)\citep{romanov2023} finds that, below a certain threshold, RSVD cannot reliably recover singular values. Existing analyses quantify errors in both singular values and operator norm, but do not explicitly correct them.

At the same time, free probability provides a non-commutative analogue of classical probability, in which independence is replaced by the notion of freeness.\citep{voiculescu1992} Free probability has subsequently been used to describe the singular value distribution of random matrices.\citep{rao2006} In this work, we use free probability to correct errors in the top-$k$ singular values produced by RSVD.

---

## Randomized Singular Value Decomposition

Given $A\in\mathbb{R}^{N\times N}$, and a target rank $k$, our goal is to efficiently compute a good rank-$k$ approximation of $A$ using the RSVD algorithm.\citep{halko2010}

We first define $\ell\coloneq k+p$ as the number of random test vectors used to probe $A$, where $p$ is the oversampling parameter. We now create a Gaussian random sketch matrix, $\Omega$, and calculate $A\Omega$:

$$
\Omega \in \mathbb{R}^{N \times \ell}, \quad \Omega_{ij} \sim \mathcal{N}(0,1), \quad Y = A\Omega
$$

$Y$ approximately captures the column space of the rank-$k$ truncation of $A$. We then perform a $QR$-decomposition, project $A$ onto the subspace spanned by $Q$, and compute the usual SVD:

$$
Y=QR, \quad B=Q^\top A, \quad B=\tilde{U}\Sigma V^\top
$$

We then compute $U=Q\tilde{U}$ and truncate to rank-$k$, finding that
$A\approx U_k\Sigma_k V_k^\top$.

---

## Motivation

We focus on $Y$ and the random matrix $\Omega$.

$$
\Omega=[\omega_1, \cdots, \omega_\ell], \quad \omega_j\sim\mathcal{N}(0, I_N)\ \forall j\in[1, \cdots, \ell]
$$

$$
Y=A\Omega=[A\omega_1, \cdots, A\omega_\ell], \quad y_j\coloneq A\omega_j
$$

We then compute the mean and covariance of the $y_j$'s. Noting that the $\omega_j$'s are i.i.d.:

$$
\mathbb{E}(y_j)=A\mathbb{E}(\omega_j)=0
$$

$$
\text{Cov}(y_j)=\mathbb{E}(y_jy_j^\top)=A\text{Cov}(\omega_j)A^\top=AA^\top
$$

So $y_j\sim\mathcal{N}(0, AA^\top)$, and
$\frac{1}{\ell}YY^\top=\frac{1}{\ell}\sum_{j=1}^\ell y_jy_j^\top$
is the sample covariance matrix for $AA^\top$. Similarly, we have that $\frac{1}{\ell}YY^\top$ is unbiased, since
$\mathbb{E}(\frac{1}{\ell}YY^\top)=AA^\top$.

Sample covariance matrices are oft-referenced objects in random matrix theory and free probability, since their spectra are governed by the Marchenko--Pastur theorem (see below). This motivates analysis of the RSVD sketch's spectrum via free probability.

---

## Methods

### Corrected Randomized Singular Value Decomposition

**Theorem (Marchenko--Pastur).**

Let $\Omega\in\mathbb{R}^{N\times\ell}$, with $\Omega_{ij}\sim\mathcal{N}(0,1)$, and define

$$
W_N\coloneq \frac{1}{\ell}\Omega\Omega^\top.
$$

Let $\lambda_1,\ldots,\lambda_N$ be the eigenvalues of $W_N$, and define the empirical spectral measure (ESM)

$$
\mu_N(A)\coloneq \frac{1}{N}|{\lambda_j\in A}|,\qquad A\subset\mathbb{R}.
$$

Assume $N,\ell\to\infty$ with $\frac{N}{\ell}\to c\in(0,+\infty)$. Then $\mu_N\stackrel{d}{\to}\mu$ weakly, where

$$
\mu(A)=
\left(1-\frac{1}{c}\right)\mathbf{1}*{{0\in A}}\mathbf{1}*{{c>1}}
+\nu(A)
$$

and $\nu$ has density

$$
d\nu(x)=\frac{1}{2\pi cx}\sqrt{(\lambda_+-x)(x-\lambda_-)},\mathbf{1}*{x\in[\lambda*-,\lambda_+]},dx, \quad \lambda_\pm=\bigl(1\pm\sqrt{c}\bigr)^2.
$$

Note that $\mu$ is known as the Marchenko--Pastur distribution.

---

Although the theorem is asymptotic, we apply it as a finite approximation.

Let $\mu_N$ denote the ESM of $W_N$, $\mu^Y_N$ denote the ESM of $\frac{1}{\ell}YY^\top$, and $\mu^A$ denote the ESM of $AA^\top$.

Crucially, $\mu^Y_N$ and $(AA^\top)^{1/2}W_N(AA^\top)^{1/2}$ have the same empirical spectral distribution. Using SVD $A = U\Sigma V^\top$:

$$
\frac{1}{\ell}YY^\top = A W_N A^\top = U\Sigma (V^\top W_N V)\Sigma U^\top
$$

Using orthogonal invariance of $W_N$:

$$
OW_N O^\top \stackrel{d}{=} W_N
$$

we obtain

$$
\frac{1}{\ell}YY^\top \stackrel{d}{=} \Sigma W_N \Sigma \stackrel{d}{=} (AA^\top)^{1/2} W_N (AA^\top)^{1/2}
$$

---

**Definition (Free multiplicative convolution).**

Let $\mu, \nu$ be probability measures on $[0,+\infty)$. If $X, Y$ are freely independent random variables with laws $\mu, \nu$, then

$$
\mu\boxtimes\nu
$$

is the law of $X^{1/2}YX^{1/2}$.

---

We obtain the approximation

$$
\mu_N^Y\approx\mu^A\boxtimes\mu.
$$

Thus we recover $\mu^A$ via

$$
\mu^A \approx \mu_N^Y \boxtimes^{-1} \mu.
$$

Using the $S$-transform:

$$
S^Y(z) = S^A(z)S^{\mathrm{MP}}(z)
$$

$$
S^{\mathrm{MP}}(z) = \frac{1}{1+cz}
$$

$$
S^A(z) = S^Y(z)(1+cz)
$$

Inverting yields corrected eigenvalues $\lambda_i^{\mathrm{corr}}$, giving

$$
\sigma_i^\mathrm{corr} = \sqrt{\lambda_i^\mathrm{corr}}.
$$

Final reconstruction:

$$
A \approx U_k \Sigma_k^\mathrm{corr} V_k^\top.
$$

---

### Hypothesis Testing

Define

$$
\mathrm{MSE}*j^{m} = \frac{1}{k}\sum*{i=1}^k(\widehat\sigma_{i,j}^{m}-\sigma_i)^2
$$

$$
D_j = \mathrm{MSE}_j^\mathrm{corr}-\mathrm{MSE}_j^\mathrm{RSVD}
$$

$$
\overline D = \frac{1}{n}\sum D_j, \quad s_D^2 = \frac{1}{n-1}\sum(D_j-\overline D)^2
$$

Hypothesis:

$$
H_0:\mathbb{E}[D]\ge0,\qquad H_1:\mathbb{E}[D]<0
$$

Test statistic:

$$
t = \frac{\overline D}{\sqrt{s_D^2/n}}\sim t_{n-1}
$$

p-value:

$$
p=\mathbb{P}(T_{n-1}\le t)
$$

Confidence interval:

$$
(-\infty, \overline D+\frac{s_D}{\sqrt{n}}t_{n-1,1-\alpha})
$$

---

### Spiked signal-plus-noise model

$$
A = U \text{diag}(\sigma_1, \ldots, \sigma_k) V^\top + \frac{\sigma_\text{noise}}{\sqrt{N}}G
$$

where $G_{ij}\sim\mathcal{N}(0,1)$.

---

## Results

The results of our investigation are shown below.

| N    | K   | c   | $\sigma_\text{noise}$ | $\overline D$ | $s_D$  | $t$      | $p$    | CI$_\mathrm{hi}$ | reject? |
| ---- | --- | --- | --------------------- | ------------- | ------ | -------- | ------ | ---------------- | ------- |
| 300  | 130 | 2.0 | 2.0                   | -0.7336       | 0.1112 | -65.997  | <1e-3  | -0.7152          | Yes     |
| 300  | 40  | 5.0 | 2.0                   | -0.9205       | 0.3822 | -24.083  | <1e-3  | -0.8571          | Yes     |
| 600  | 280 | 2.0 | 2.0                   | -0.3790       | 0.0552 | -68.622  | <1e-3  | -0.3698          | Yes     |
| 600  | 180 | 3.0 | 2.0                   | -0.1229       | 1.2488 | -0.984   | 0.1638 | 0.0845           | No      |
| 600  | 100 | 5.0 | 2.0                   | -0.4882       | 0.2072 | -23.557  | <1e-3  | -0.4538          | Yes     |
| 1000 | 480 | 2.0 | 5.0                   | -2.1143       | 0.1076 | -196.523 | <1e-3  | -2.0965          | Yes     |
| 1000 | 180 | 5.0 | 5.0                   | -0.0958       | 0.3950 | -2.426   | 0.0085 | -0.0302          | Yes     |

---

## Critical use of AI tools

* We used OpenAI's Prism\citep{prism} to format and reword parts of the LaTeX source.
* We used GitHub Copilot\citep{copilot} to aid in programming.

---

## Appendix: The S-transform

$$
\psi(z) = z m(z)-1, \qquad m(z)=\int \frac{1}{z-x},d\mu(x)
$$

$$
S(w)=\frac{1+w}{w},\psi^{-1}(w)
$$

$$
S^{\mu \boxtimes \nu}(w) = S^\mu(w)S^\nu(w)
$$

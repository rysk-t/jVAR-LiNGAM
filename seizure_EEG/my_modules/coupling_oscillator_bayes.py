# -*- coding: utf-8 -*-
"""Sequential Bayesian estimation of the phase-coupled oscillator model
(Yokoyama & Kitajo, NeuroImage 2022, Eq. 6), output-compatible fast implementation.

Original implementation: H. Yokoyama (2025). This version keeps the interface and the outputs
of the original ``CouplingOscillator`` (agreement ~1e-13 relative, see
seizure_EEG/tests/test_coupling_oscillator_bayes.py) and is about 100x faster.

Why the result does not change
------------------------------
* The prior covariance starts as the identity and the design matrix is block diagonal
  (oscillator i only uses its own coefficient block), with isotropic observation noise.
  The posterior covariance therefore stays block diagonal, and the update splits into N
  independent problems of size d = 2*P*N + 1 instead of one problem of size N*d.
* There is no process noise, so B consecutive one-sample updates equal one B-sample update.
  Blocks of B samples are processed with BLAS-3 operations, and every per-sample output
  (one-step log-likelihood, filtered prediction, omega history) is recovered from the
  Cholesky factor of the block's predictive covariance:
      S = s2*I + F K F^T = L L^T,  z = L^{-1}(y - F mu),  G = K F^T L^{-T}
      per-sample predictive variance = L_tt^2,  standardised innovation = z_t
      posterior mean after sample t = mu + sum_{s<=t} G[:, s] z_s,   K <- K - G G^T
  ``block_size=1`` is the plain per-sample update.
* The sum over samples of the one-step log-likelihoods equals the log marginal likelihood
  of the batch model, so ``log_evidence_grid`` scores a whole (P, precision) grid from one
  set of Gram matrices.

Only ``T == 1`` (the value used by every script) is supported.
"""

import numpy as np
from scipy.linalg import block_diag, solve_triangular

TWO_PI = 2 * np.pi
LOG_2PI = np.log(2 * np.pi)


def phase_velocity(x, h):
    """(theta[t] - theta[t-1]) / h with the 2-sample np.unwrap of the original implementation.

    x : (Nt, N) wrapped phases. Returns (Nt-1, N); bit-identical to the original.
    """
    x0, x1 = x[:-1], x[1:]
    dd = x1 - x0
    ddmod = np.mod(dd + np.pi, TWO_PI) - np.pi
    ddmod[(ddmod == -np.pi) & (dd > 0)] = np.pi
    pc = np.where(np.abs(dd) < np.pi, 0.0, ddmod - dd)
    return ((x1 + pc) - x0) / h


def fourier_features(th, P):
    """Regressors of the original ``make_fourier_features`` for several time points.

    th : (B, N) phases at the feature times (theta[t-1] for target t).
    Returns (B, N, d), d = 2*P*N + 1, columns [cos p=1 | sin p=1 | ... | cos P | sin P | 1].
    """
    B, N = th.shape
    pd = np.mod(th[:, None, :] - th[:, :, None], TWO_PI)   # [b, i, j] = theta_j - theta_i
    eye = np.eye(N)
    blocks = []
    for p in range(1, P + 1):
        blocks.append(np.cos(p * pd) - eye)
        blocks.append(np.sin(p * pd))
    blocks.append(np.ones((B, N, 1)))
    return np.concatenate(blocks, axis=2)


class CouplingOscillator:
    """Same constructor and ``fit_model()`` as the original class (T == 1).

    Attributes set by ``fit_model()``:
      omega (Nt-1, N), coeff_cos (N, N, P), coeff_sin (N, N, P), y_hat (Nt-1, N),
      loglike (Nt-1,), P, T, Nosc   -- same meaning and shape as the original
      mu (N, d), K_blocks (N, d, d)  -- final posterior mean / covariance per oscillator
      pred_var (Nt-1, N)            -- one-step predictive variance (diagonal of the original S)
      Kb0                            -- full block-diagonal covariance, built on access
    The per-sample ``beta`` and ``S`` histories of the original are not stored.
    """

    def __init__(self, x, P, T, h, prec_param, block_size=128):
        if T != 1:
            raise NotImplementedError("only T == 1 is supported (all scripts use Tstep = 1)")
        self.x = x
        self.P = int(P)
        self.T = T
        self.h = h
        self.prec_param = prec_param
        self.block_size = int(block_size)

    def fit_model(self):
        x = np.asarray(self.x, dtype=float)
        P, h, B = self.P, self.h, self.block_size
        Nt, N = x.shape
        d = 2 * P * N + 1
        s2 = 1.0 / self.prec_param
        n = Nt - 1

        y = phase_velocity(x, h)
        mu = np.zeros((N, d))
        K = np.tile(np.eye(d), (N, 1, 1))
        loglike = np.zeros(n)
        y_hat = np.empty((n, N))
        omega = np.empty((n, N))
        pred_var = np.empty((n, N))

        for t0 in range(0, n, B):
            t1 = min(t0 + B, n)
            F_all = fourier_features(x[t0:t1], P)          # features from theta[t-1]
            yb = y[t0:t1]
            for i in range(N):
                F = F_all[:, i, :]                          # (b, d)
                r = yb[:, i] - F @ mu[i]
                KF = K[i] @ F.T                             # (d, b)
                S = F @ KF
                S[np.diag_indices_from(S)] += s2
                L = np.linalg.cholesky(S)
                z = solve_triangular(L, r, lower=True)
                G = solve_triangular(L, KF.T, lower=True).T  # K F^T L^{-T}, (d, b)
                ldiag = np.diag(L)
                loglike[t0:t1] += -0.5 * LOG_2PI - np.log(ldiag) - 0.5 * z ** 2
                pred_var[t0:t1, i] = ldiag ** 2
                Mc = mu[i][:, None] + np.cumsum(G * z, axis=1)   # posterior mean after each sample
                y_hat[t0:t1, i] = np.einsum("bd,db->b", F, Mc)
                omega[t0:t1, i] = Mc[-1]
                mu[i] = Mc[:, -1]
                Ki = K[i] - G @ G.T
                K[i] = 0.5 * (Ki + Ki.T)

        idx = np.arange(N)
        self.coeff_cos = np.stack([mu[:, 2 * N * k + idx] for k in range(P)], axis=2)
        self.coeff_sin = np.stack([mu[:, 2 * N * k + N + idx] for k in range(P)], axis=2)
        self.omega = omega
        self.y_hat = y_hat
        self.loglike = loglike
        self.pred_var = pred_var
        self.mu = mu
        self.K_blocks = K
        self.Nosc = N
        return self

    @property
    def Kb0(self):
        return block_diag(*self.K_blocks)


def log_evidence_grid(theta, h, P_list, prec_param_list, chunk=2048):
    """Scores of ``CouplingOscillator`` for every (P, prec_param) cell, without fitting each cell.

    theta : (Nt, N) wrapped phases (the array passed to ``CouplingOscillator``).
    Returns a dict of arrays shaped (len(P_list), len(prec_param_list)):
      loglike_sum   -- model.loglike.sum()  (= log marginal likelihood)
      loglike_mean  -- np.mean(model.loglike)
      BIC           -- k*log(Nt-1) - 2*loglike_sum with k = N*(2*P*N + 1)
    """
    x = np.asarray(theta, dtype=float)
    Nt, N = x.shape
    n = Nt - 1
    P_list = np.asarray(P_list, dtype=int)
    prec = np.asarray(prec_param_list, dtype=float)
    Pmax = int(P_list.max())
    dmax = 2 * Pmax * N + 1
    y = phase_velocity(x, h)

    Gm = np.zeros((N, dmax, dmax))
    bv = np.zeros((N, dmax))
    for t0 in range(0, n, chunk):
        t1 = min(t0 + chunk, n)
        F = fourier_features(x[t0:t1], Pmax).transpose(1, 0, 2)   # (N, b, dmax)
        Gm += np.matmul(F.transpose(0, 2, 1), F)
        bv += np.einsum("nbd,bn->nd", F, y[t0:t1])
    yy = (y ** 2).sum(axis=0)

    s2 = 1.0 / prec
    ll = np.zeros((len(P_list), len(prec)))
    for a, P in enumerate(P_list):
        cols = np.r_[0:2 * N * P, dmax - 1]                         # columns of the order-P model
        d = len(cols)
        for i in range(N):
            lam, Q = np.linalg.eigh(Gm[i][np.ix_(cols, cols)])
            c2 = (Q.T @ bv[i][cols]) ** 2
            den = s2[:, None] + lam[None, :]                       # (n_prec, d)
            logdet = np.log(den).sum(axis=1) + (n - d) * np.log(s2)
            quad = (yy[i] - (c2[None, :] / den).sum(axis=1)) / s2
            ll[a] += -0.5 * (n * LOG_2PI + logdet + quad)
    k = (N * (N * 2 * P_list + 1))[:, None]
    return dict(P_list=P_list, prec_param_list=prec, loglike_sum=ll, loglike_mean=ll / n,
                BIC=k * np.log(n) - 2 * ll)

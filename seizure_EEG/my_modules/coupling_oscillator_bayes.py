# -*- coding: utf-8 -*-
"""Sequential Bayesian estimation of the phase-coupled oscillator model
(Yokoyama & Kitajo, NeuroImage 2022, Eq. 6), output-compatible fast implementation.

Original implementation: H. Yokoyama (2025). This version keeps the interface and the outputs
of the original ``CouplingOscillator`` (agreement ~1e-12 relative on float64 input, see
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
  streamed QR factorisation per oscillator.

Input dtype
-----------
Phases are converted to float64 before any computation. Compatibility with the original was
verified for float64 input. The original class computed part of the features (phase
differences, unwrap) in the input dtype, so float32 input gives small differences from it
(up to 5e-4 in the per-sample log-likelihood on a 300-sample EEG segment; 2e-11 when the same
float32 values are passed to the original as float64).

Only ``T == 1`` (the value used by every script) is supported.
"""

import numbers
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg import block_diag, qr as scipy_qr, solve_triangular

TWO_PI = 2 * np.pi
LOG_2PI = np.log(2 * np.pi)
_MAX_FEATURE_ELEMENTS = 2 ** 25      # cap on (N * samples * d) per pass in log_evidence_grid
_PARALLEL_MIN_K = 64                 # threads in log_evidence_grid only from this matrix size on
_MAX_AUTO_WORKERS = 8                # no gain beyond 8 threads measured on chb15_06 (15 channels)


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Integral, np.integer)) \
            or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _positive_float(name, value):
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return v


def _as_phase_array(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"phases must be a 2-D array (Nt >= 2, N >= 1), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("phases contain non-finite values")
    return x


def phase_velocity(x, h):
    """(theta[t] - theta[t-1]) / h with the 2-sample np.unwrap of the original implementation.

    x : (Nt, N) wrapped phases. Returns (Nt-1, N); bit-identical to the original for float64.
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
    ``y_hat`` is the prediction after updating with the same sample (as in the original),
    not an out-of-sample prediction.
    """

    def __init__(self, x, P, T, h, prec_param, block_size=128):
        if T != 1:
            raise NotImplementedError("only T == 1 is supported (all scripts use Tstep = 1)")
        self.x = x
        self.P = _positive_int("P", P)
        self.T = T
        self.h = _positive_float("h", h)
        self.prec_param = _positive_float("prec_param", prec_param)
        self.block_size = _positive_int("block_size", block_size)

    def fit_model(self):
        x = _as_phase_array(self.x)
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


def log_evidence_grid(theta, h, P_list, prec_param_list, chunk=2048, n_jobs=None):
    """Scores of ``CouplingOscillator`` for every (P, prec_param) cell, without fitting each cell.

    theta : (Nt, N) wrapped phases (the array passed to ``CouplingOscillator``).
    chunk : maximum number of samples per block of the pass over the data (lowered
            automatically so that one block of features stays below about 256 MB).
    n_jobs : threads for the per-oscillator QR updates and SVDs. None (default): up to 8
             threads when the per-oscillator matrices have at least 64 columns, otherwise
             serial; 1: serial. BLAS settings are never changed. The threaded path is fastest
             when the caller limits BLAS to one thread, e.g.
             ``with threadpoolctl.threadpool_limits(1): log_evidence_grid(...)``
             (chb15_06, 15 channels, 15 cells: 0.09 s with 1 BLAS thread, 0.21 s with 24).
    Returns a dict of arrays shaped (len(P_list), len(prec_param_list)):
      loglike_sum   -- model.loglike.sum(), i.e. the log marginal likelihood (log evidence)
      loglike_mean  -- np.mean(model.loglike)
      BIC           -- k*log(Nt-1) - 2*loglike_sum with k = N*(2*P*N + 1).
                       This reproduces the score of the authors' grid search. It differs from
                       the usual BIC in two ways: loglike_sum is the log marginal likelihood,
                       which already penalises model size, not the maximised log-likelihood;
                       and k includes the N*2*P self-coupling columns, which are always zero.

    Method. For each oscillator the identically zero self-coupling columns are dropped and the
    other columns are ordered [1 | cos p=1 | sin p=1 | ... | cos Pmax | sin Pmax], so the order-P
    model uses the leading k = 1 + 2P(N-1) columns. One pass over the data computes the R factor
    of the QR decomposition of [F | y]. For each P, the SVD R[:k, :k] = W diag(sv) V^T then gives,
    for every noise variance s2 = 1/prec_param,
        log|s2 I_n + F F^T|           = sum log(sv^2 + s2) + (n - k) log s2
        y^T (s2 I_n + F F^T)^{-1} y   = ||y_perp||^2 / s2 + sum (W^T c)^2 / (sv^2 + s2)
    with c = R[:k, -1] and ||y_perp||^2 = sum R[k:, -1]^2 (the least-squares residual).
    No term is a difference of large numbers, so the result does not lose digits when the
    residual is much smaller than y, and each extra precision value costs O(k).
    """
    chunk = _positive_int("chunk", chunk)
    h = _positive_float("h", h)
    x = _as_phase_array(theta)
    P_list = np.atleast_1d(np.asarray(P_list))
    if P_list.size == 0:
        raise ValueError("P_list is empty")
    P_list = np.array([_positive_int("P", P) for P in P_list.tolist()])
    prec = np.atleast_1d(np.asarray(prec_param_list, dtype=float))
    if prec.size == 0:
        raise ValueError("prec_param_list is empty")
    prec = np.array([_positive_float("prec_param", p) for p in prec])

    Nt, N = x.shape
    n = Nt - 1
    Pmax = int(P_list.max())
    dmax = 2 * Pmax * N + 1
    kmax = 1 + 2 * Pmax * (N - 1)

    # per oscillator: columns of fourier_features without the self-coupling ones, P-nested order
    idx = np.empty((N, kmax), dtype=int)
    for i in range(N):
        others = np.array([j for j in range(N) if j != i], dtype=int)
        cols = [dmax - 1]
        for p in range(Pmax):
            cols.extend(2 * N * p + others)
            cols.extend(2 * N * p + N + others)
        idx[i] = cols

    # one pass: R factor of [F | y] per oscillator (streamed QR)
    step = max(1, min(chunk, _MAX_FEATURE_ELEMENTS // (N * dmax)))
    y = phase_velocity(x, h)
    if n_jobs is None:
        workers = min(N, _MAX_AUTO_WORKERS, os.cpu_count() or 1)             if (N >= 2 and kmax >= _PARALLEL_MIN_K) else 1
    else:
        workers = min(_positive_int("n_jobs", n_jobs), N)

    def blocks():
        for t0 in range(0, n, step):
            t1 = min(t0 + step, n)
            F = fourier_features(x[t0:t1], Pmax).transpose(1, 0, 2)      # (N, b, dmax)
            Fi = np.take_along_axis(F, idx[:, None, :], axis=2)          # (N, b, kmax)
            yield np.concatenate([Fi, y[t0:t1].T[:, :, None]], axis=2)   # (N, b, kmax + 1)

    kP = [1 + 2 * int(P) * (N - 1) for P in P_list]
    results = []                                     # ((a, i), singular values, (W^T c)^2)
    if workers == 1:
        R = np.zeros((N, kmax + 1, kmax + 1))
        for A in blocks():
            R = np.linalg.qr(np.concatenate([R, A], axis=1), mode="r")
        for a, k in enumerate(kP):
            W, sv, _ = np.linalg.svd(R[:, :k, :k])
            u2 = np.einsum("nkj,nk->nj", W, R[:, :k, -1]) ** 2
            results += [((a, i), sv[i], u2[i]) for i in range(N)]
    else:
        def svd_task(t):
            a, i = t
            k = kP[a]
            W, sv, _ = np.linalg.svd(R[i][:k, :k])
            return t, sv, (W.T @ R[i][:k, -1]) ** 2

        with ThreadPoolExecutor(max_workers=workers) as ex:
            R = [np.zeros((kmax + 1, kmax + 1)) for _ in range(N)]
            for A in blocks():
                R = list(ex.map(
                    lambda i: scipy_qr(np.concatenate([R[i], A[i]], axis=0), mode="r",
                                       check_finite=False)[0][:kmax + 1], range(N)))
            tasks = sorted(((a, i) for a in range(len(kP)) for i in range(N)), key=lambda t: -kP[t[0]])
            results = list(ex.map(svd_task, tasks))      # largest SVDs first

    s2 = 1.0 / prec
    ll = np.zeros((len(P_list), len(prec)))
    for (a, i), sv, u2 in results:
        k = kP[a]
        tail2 = (R[i][k:, -1] ** 2).sum()
        den = sv[None, :] ** 2 + s2[:, None]                              # (n_prec, k)
        logdet = np.log(den).sum(axis=1) + (n - k) * np.log(s2)
        quad = tail2 / s2 + (u2[None, :] / den).sum(axis=1)
        ll[a] += -0.5 * (n * LOG_2PI + logdet + quad)
    kb = (N * (N * 2 * P_list + 1))[:, None]
    return dict(P_list=P_list, prec_param_list=prec, loglike_sum=ll, loglike_mean=ll / n,
                BIC=kb * np.log(n) - 2 * ll)

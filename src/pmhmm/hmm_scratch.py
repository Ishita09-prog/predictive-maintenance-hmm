"""Gaussian HMM implemented from scratch in NumPy (Unit 4 algorithms).

Implements, without any HMM library:
  * scaled Forward algorithm        (Rabiner / Bishop 13.2.4 scaling factors c_t)
  * scaled Backward algorithm
  * Forward-Backward smoothing      (gamma_t(i) = P(z_t = i | x_1..x_T))
  * Viterbi algorithm (log domain)  (most probable hidden-state path)
  * Baum-Welch (EM) maximum-likelihood estimation, multiple sequences
  * left-to-right (degradation) topology constraint (free start distribution over non-final states)
  * log-likelihood, AIC and BIC

Emissions are Gaussian with diagonal covariance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

LOG_EPS = 1e-300


def _log_gauss_diag(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """log N(x_t | mu_k, diag(var_k)) for every t, k.  Returns (T, K)."""
    T, D = X.shape
    K = means.shape[0]
    out = np.empty((T, K))
    for k in range(K):
        diff = X - means[k]
        out[:, k] = -0.5 * (
            D * np.log(2.0 * np.pi) + np.sum(np.log(covars[k])) + np.sum(diff * diff / covars[k], axis=1)
        )
    return out


@dataclass
class GaussianHMM:
    n_states: int
    left_to_right: bool = True
    n_iter: int = 300
    tol: float = 1e-8
    min_covar: float = 1e-3
    startprob_: np.ndarray = field(default=None, repr=False)
    transmat_: np.ndarray = field(default=None, repr=False)
    means_: np.ndarray = field(default=None, repr=False)
    covars_: np.ndarray = field(default=None, repr=False)
    loglik_history_: List[float] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ init
    def init_from_time_segments(self, seqs: Sequence[np.ndarray]) -> "GaussianHMM":
        """Unsupervised initialisation that uses ONLY the position in time.

        Each sequence is cut into K equal chunks; chunk k initialises state k.
        (No health labels / RUL values are used anywhere in training.)
        """
        K = self.n_states
        D = seqs[0].shape[1]
        buckets = [[] for _ in range(K)]
        for X in seqs:
            for k, chunk in enumerate(np.array_split(X, K)):
                buckets[k].append(chunk)
        self.means_ = np.vstack([np.vstack(b).mean(0) for b in buckets])
        self.covars_ = np.vstack([np.maximum(np.vstack(b).var(0), self.min_covar) for b in buckets])
        mean_len = float(np.mean([len(s) for s in seqs]))
        stay = 1.0 - K / mean_len
        A = np.zeros((K, K))
        for i in range(K):
            if i == K - 1:
                A[i, i] = 1.0
            else:
                A[i, i], A[i, i + 1] = stay, 1.0 - stay
        if not self.left_to_right:
            A = 0.9 * A + 0.1 / K
        self.transmat_ = A
        pi = np.zeros(K)
        if self.left_to_right:
            pi[:-1] = 1.0 / (K - 1)       # engines may start with different initial wear: any non-final state
        else:
            pi[:] = 1.0 / K
        self.startprob_ = pi
        return self

    # ------------------------------------------------- forward / backward
    def _scaled_forward(self, logB: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Scaled forward pass.

        Returns alpha_hat (T,K) = P(z_t | x_1..t)  (the *filtering* distribution),
        scaling factors c (T,) and the sequence log-likelihood.
        """
        T, K = logB.shape
        m = logB.max(axis=1)                       # per-row max for numerical safety
        B = np.exp(logB - m[:, None])
        alpha = np.empty((T, K))
        c = np.empty(T)
        a = self.startprob_ * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.transmat_) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        loglik = float(np.log(c).sum() + m.sum())
        return alpha, c, loglik

    def _scaled_backward(self, logB: np.ndarray, c: np.ndarray) -> np.ndarray:
        T, K = logB.shape
        m = logB.max(axis=1)
        B = np.exp(logB - m[:, None])
        beta = np.empty((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.transmat_ @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    # ---------------------------------------------------------- public API
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Filtering posteriors P(z_t | x_1..t) and log-likelihood (causal)."""
        logB = _log_gauss_diag(X, self.means_, self.covars_)
        alpha, _, ll = self._scaled_forward(logB)
        return alpha, ll

    def score(self, X: np.ndarray) -> float:
        return self.forward(X)[1]

    def forward_backward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Smoothed posteriors gamma (T,K), pairwise xi (T-1,K,K), loglik."""
        logB = _log_gauss_diag(X, self.means_, self.covars_)
        alpha, c, ll = self._scaled_forward(logB)
        beta = self._scaled_backward(logB, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        m = logB.max(axis=1)
        B = np.exp(logB - m[:, None])
        T, K = alpha.shape
        xi = np.empty((T - 1, K, K))
        for t in range(T - 1):
            xi[t] = alpha[t][:, None] * self.transmat_ * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
        return gamma, xi, ll

    def smooth(self, X: np.ndarray) -> np.ndarray:
        return self.forward_backward(X)[0]

    def viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Most probable state path (log domain) and its log-probability."""
        logB = _log_gauss_diag(X, self.means_, self.covars_)
        logA = np.log(self.transmat_ + LOG_EPS)
        logpi = np.log(self.startprob_ + LOG_EPS)
        T, K = logB.shape
        delta = np.empty((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = logpi + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA           # (from, to)
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + logB[t]
        path = np.empty(T, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(delta[-1].max())

    # ------------------------------------------------------------ training
    def fit(self, seqs: Sequence[np.ndarray], verbose: bool = False) -> "GaussianHMM":
        if self.means_ is None:
            self.init_from_time_segments(seqs)
        K, D = self.means_.shape
        self.loglik_history_ = []
        prev = -np.inf
        for it in range(self.n_iter):
            pi_acc = np.zeros(K)
            xi_acc = np.zeros((K, K))
            g_sum = np.zeros(K)
            gx = np.zeros((K, D))
            gxx = np.zeros((K, D))
            total_ll = 0.0
            for X in seqs:
                gamma, xi, ll = self.forward_backward(X)
                total_ll += ll
                pi_acc += gamma[0]
                xi_acc += xi.sum(axis=0)
                g_sum += gamma.sum(axis=0)
                gx += gamma.T @ X
                gxx += gamma.T @ (X * X)
            self.loglik_history_.append(total_ll)
            if verbose:
                print(f"iter {it:3d}  loglik {total_ll:14.3f}")
            # ---- M step
            self.startprob_ = pi_acc / pi_acc.sum()
            self.transmat_ = xi_acc / np.maximum(xi_acc.sum(axis=1, keepdims=True), LOG_EPS)
            # absorbing last state when left-to-right (no outgoing xi mass)
            if self.left_to_right:
                self.transmat_[-1] = 0.0
                self.transmat_[-1, -1] = 1.0
            self.means_ = gx / g_sum[:, None]
            var = gxx / g_sum[:, None] - self.means_ ** 2
            self.covars_ = np.maximum(var, self.min_covar)
            if abs(total_ll - prev) < self.tol * max(1.0, abs(total_ll)):
                break
            prev = total_ll
        return self

    # ----------------------------------------------------- model selection
    def n_free_params(self) -> int:
        K, D = self.means_.shape
        if self.left_to_right:
            trans = K - 1                 # one free prob per non-absorbing row
            start = K - 2                 # start distribution over the K-1 non-final states
        else:
            trans = K * (K - 1)
            start = K - 1
        return trans + start + 2 * K * D  # means + diagonal variances

    def aic_bic(self, seqs: Sequence[np.ndarray]) -> Tuple[float, float]:
        ll = sum(self.score(X) for X in seqs)
        n = sum(len(X) for X in seqs)
        p = self.n_free_params()
        return -2 * ll + 2 * p, -2 * ll + p * np.log(n)

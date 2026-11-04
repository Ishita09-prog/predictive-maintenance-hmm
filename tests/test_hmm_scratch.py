import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from pmhmm.hmm_scratch import GaussianHMM, _log_gauss_diag  # noqa: E402


def tiny_model(left_to_right=False, K=3, D=2, seed=0):
    rng = np.random.default_rng(seed)
    m = GaussianHMM(K, left_to_right=left_to_right)
    A = rng.random((K, K)) + 0.1
    if left_to_right:
        A = np.triu(A)
        A[-1] = 0
        A[-1, -1] = 1
    m.transmat_ = A / A.sum(1, keepdims=True)
    pi = rng.random(K) + 0.1
    m.startprob_ = pi / pi.sum()
    m.means_ = rng.normal(size=(K, D)) * 2
    m.covars_ = rng.random((K, D)) + 0.3
    return m


def brute_force(m, X):
    """Enumerate every state path: exact likelihood and exact best path."""
    K, T = m.n_states, len(X)
    logB = _log_gauss_diag(X, m.means_, m.covars_)
    best, best_p, total = -np.inf, None, 0.0
    for path in itertools.product(range(K), repeat=T):
        lp = np.log(m.startprob_[path[0]]) + logB[0, path[0]]
        for t in range(1, T):
            lp += np.log(m.transmat_[path[t - 1], path[t]]) + logB[t, path[t]]
        total += np.exp(lp)
        if lp > best:
            best, best_p = lp, path
    return np.log(total), np.array(best_p), best


def test_forward_matches_brute_force_likelihood():
    m = tiny_model()
    X = np.random.default_rng(1).normal(size=(6, 2))
    ll_bf, _, _ = brute_force(m, X)
    assert m.score(X) == pytest.approx(ll_bf, abs=1e-9)


def test_viterbi_matches_brute_force():
    m = tiny_model(seed=3)
    X = np.random.default_rng(2).normal(size=(6, 2))
    _, path_bf, lp_bf = brute_force(m, X)
    path, lp = m.viterbi(X)
    assert (path == path_bf).all()
    assert lp == pytest.approx(lp_bf, abs=1e-9)


def test_forward_backward_posteriors_are_distributions_and_consistent():
    m = tiny_model(seed=4)
    X = np.random.default_rng(5).normal(size=(30, 2))
    gamma, xi, ll = m.forward_backward(X)
    assert np.allclose(gamma.sum(1), 1)
    assert np.allclose(xi.sum(axis=(1, 2)), 1)
    assert np.allclose(xi.sum(axis=2), gamma[:-1])      # marginalising xi gives gamma
    assert ll == pytest.approx(m.score(X))


def test_last_filtered_equals_last_smoothed():
    m = tiny_model(seed=6)
    X = np.random.default_rng(7).normal(size=(25, 2))
    assert np.allclose(m.forward(X)[0][-1], m.smooth(X)[-1])


def test_scaling_prevents_underflow_on_long_sequences():
    m = tiny_model(seed=8)
    X = np.random.default_rng(9).normal(size=(5000, 2)) * 3
    assert np.isfinite(m.score(X))


def test_em_loglik_is_monotone_and_left_to_right_structure_kept():
    true = tiny_model(left_to_right=True, K=3, seed=10)
    rng = np.random.default_rng(11)
    seqs = []
    for _ in range(15):
        z, X = 0, []
        for _ in range(60):
            X.append(true.means_[z] + rng.normal(size=2) * np.sqrt(true.covars_[z]))
            z = rng.choice(3, p=true.transmat_[z])
        seqs.append(np.array(X))
    m = GaussianHMM(3, left_to_right=True, n_iter=50).fit(seqs)
    h = np.array(m.loglik_history_)
    assert (np.diff(h) > -1e-6).all()
    assert np.allclose(np.tril(m.transmat_, -1), 0)     # never moves backwards
    assert m.transmat_[-1, -1] == 1.0


def test_matches_hmmlearn_reference_with_identical_parameters():
    hl = pytest.importorskip("hmmlearn.hmm")
    m = tiny_model(seed=12, K=4, D=3)
    X = np.random.default_rng(13).normal(size=(200, 3))
    ref = hl.GaussianHMM(4, covariance_type="diag", init_params="")
    ref.startprob_, ref.transmat_, ref.means_, ref.covars_ = m.startprob_, m.transmat_, m.means_, m.covars_
    assert m.score(X) == pytest.approx(ref.score(X), abs=1e-8)
    assert (m.viterbi(X)[0] == ref.predict(X)).all()
    assert np.allclose(m.smooth(X), ref.predict_proba(X), atol=1e-8)

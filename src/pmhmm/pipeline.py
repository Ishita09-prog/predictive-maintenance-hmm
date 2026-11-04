"""End-to-end helpers shared by the notebook, the training script and the Streamlit app."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .data import load_fd001
from .evaluate import evaluate_alerts, tune_threshold
from .features import FeaturePipeline, split_engines, to_sequences
from .hmm_scratch import GaussianHMM

SEED = 42
N_COMPONENTS = 3
N_STATES = 6
STATE_NAMES = {0: "Healthy", 1: "Early wear", 2: "Mild", 3: "Degrading", 4: "Severe", 5: "Critical"}
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
HMM_GRID = np.round(np.concatenate([np.arange(0.05, 0.95, 0.05), [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]]), 4)


@dataclass
class Prepared:
    train: pd.DataFrame
    test: pd.DataFrame
    fit_units: List[int]
    val_units: List[int]
    fp: FeaturePipeline
    seq_train: Dict[int, np.ndarray]     # all 100 train engines (features)
    seq_test: Dict[int, np.ndarray]
    rul_train: Dict[int, np.ndarray]
    rul_test: Dict[int, np.ndarray]

    @property
    def fit_seqs(self):
        return [self.seq_train[u] for u in self.fit_units]


def _rul_dict(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    return {int(u): g.sort_values("cycle")["rul"].to_numpy() for u, g in df.groupby("unit")}


def prepare(n_components: int = N_COMPONENTS, seed: int = SEED) -> Prepared:
    train, test = load_fd001()
    fit_units, val_units = split_engines(train["unit"].unique(), val_frac=0.3, seed=seed)
    fp = FeaturePipeline(n_components).fit(train[train["unit"].isin(fit_units)])   # fit engines only
    return Prepared(
        train=train, test=test, fit_units=fit_units, val_units=val_units, fp=fp,
        seq_train=to_sequences(fp.transform(train), n_components),
        seq_test=to_sequences(fp.transform(test), n_components),
        rul_train=_rul_dict(train), rul_test=_rul_dict(test),
    )


def fit_hmm(prep: Prepared, n_states: int = N_STATES) -> GaussianHMM:
    return GaussianHMM(n_states, left_to_right=True).fit(prep.fit_seqs)


def hmm_fault_scores(model: GaussianHMM, seqs: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Causal score = filtered probability of the last (Critical) state."""
    return {u: model.forward(X)[0][:, -1] for u, X in seqs.items()}


def gmm_baseline(prep: Prepared, n_states: int = N_STATES):
    """Non-sequential baseline: diagonal GMM, same features, same alert rule."""
    gmm = GaussianMixture(n_states, covariance_type="diag", random_state=SEED, n_init=5).fit(np.vstack(prep.fit_seqs))
    order = np.argsort(gmm.means_[:, 0])          # order components by PC1 (degradation axis)
    worst = order[-1]
    return gmm, (lambda seqs: {u: gmm.predict_proba(X)[:, worst] for u, X in seqs.items()})


def pc1_baseline(prep: Prepared):
    """Simplest baseline: threshold on the first principal component (health indicator)."""
    allpc1 = np.concatenate([s[:, 0] for s in prep.fit_seqs])
    grid = np.quantile(allpc1, np.linspace(0.50, 0.995, 60))
    return grid, (lambda seqs: {u: X[:, 0] for u, X in seqs.items()})


def run_all(prep: Prepared, model: GaussianHMM) -> Dict[str, dict]:
    """Tune thresholds on validation engines, then evaluate val + test once."""
    val_seqs = {u: prep.seq_train[u] for u in prep.val_units}
    val_rul = {u: prep.rul_train[u] for u in prep.val_units}
    out: Dict[str, dict] = {}
    methods = {}
    methods["HMM (from scratch)"] = (hmm_fault_scores(model, val_seqs), hmm_fault_scores(model, prep.seq_test), HMM_GRID)
    _, gmm_fn = gmm_baseline(prep)
    methods["GMM (no temporal model)"] = (gmm_fn(val_seqs), gmm_fn(prep.seq_test), HMM_GRID)
    grid, pc_fn = pc1_baseline(prep)
    methods["PC1 threshold"] = (pc_fn(val_seqs), pc_fn(prep.seq_test), grid)
    for name, (sv, st, g) in methods.items():
        thr = tune_threshold(sv, val_rul, g)
        out[name] = {
            "threshold": thr,
            "val": evaluate_alerts(sv, val_rul, thr),
            "test": evaluate_alerts(st, prep.rul_test, thr),
        }
    return out


def save_artifacts(prep: Prepared, model: GaussianHMM, results: dict, directory: Path = ARTIFACT_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "hmm_params.npz", startprob=model.startprob_, transmat=model.transmat_,
             means=model.means_, covars=model.covars_)
    joblib.dump(prep.fp, directory / "feature_pipeline.joblib")
    (directory / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (directory / "split.json").write_text(json.dumps({"fit_units": prep.fit_units, "val_units": prep.val_units}))


def load_model(directory: Path = ARTIFACT_DIR) -> GaussianHMM:
    p = np.load(directory / "hmm_params.npz")
    m = GaussianHMM(len(p["startprob"]))
    m.startprob_, m.transmat_, m.means_, m.covars_ = p["startprob"], p["transmat"], p["means"], p["covars"]
    return m

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from pmhmm.data import FAULT_RUL, health_band, load_fd001  # noqa: E402
from pmhmm.evaluate import first_alert  # noqa: E402
from pmhmm.features import FeaturePipeline, causal_window_features, split_engines  # noqa: E402


def test_dataset_shape_and_rul_labels():
    train, test = load_fd001()
    assert train["unit"].nunique() == 100 and test["unit"].nunique() == 100
    assert (train.groupby("unit")["rul"].min() == 0).all()          # run to failure
    assert (test.groupby("unit")["rul"].min() >= 1).all()            # truncated before failure
    assert health_band(np.array([200, 80, FAULT_RUL])).tolist() == [0, 1, 2]


def test_window_features_are_causal():
    train, _ = load_fd001()
    u = train[train["unit"] == 1].copy()
    base = causal_window_features(u)
    changed = u.copy()
    sens = [c for c in u.columns if c.startswith("s")]
    changed.loc[changed["cycle"] > 100, sens] += 1000.0             # corrupt the future
    out = causal_window_features(changed)
    past = base["cycle"] <= 100
    assert np.allclose(base.loc[past, sens].to_numpy(), out.loc[past, sens].to_numpy())


def test_engine_split_is_disjoint_and_pipeline_fit_on_fit_engines_only():
    train, test = load_fd001()
    fit_u, val_u = split_engines(train["unit"].unique())
    assert set(fit_u).isdisjoint(val_u) and len(fit_u) + len(val_u) == 100
    fp = FeaturePipeline(3).fit(train[train["unit"].isin(fit_u)])
    n_fit = (train["unit"].isin(fit_u)).sum()
    assert fp.scaler_.n_samples_seen_ == n_fit                       # scaler never saw val/test rows


def test_alert_needs_persistence_and_is_causal():
    s = np.array([0, 1, 0, 1, 1, 1, 0.0])
    assert first_alert(s, 0.5, persist=3) == 5
    assert first_alert(np.zeros(5), 0.5) == -1

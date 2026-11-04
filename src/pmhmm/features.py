"""Cleaning, causal window features and sequence construction.

Leakage rules enforced here
---------------------------
* Window features are *trailing* (causal): the value at cycle t only uses cycles <= t.
* Scaler and PCA are fitted on the fit-engines only, then applied unchanged to
  validation / test engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sensors that are constant (or carry no information) in FD001 are removed.
CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSORS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in CONSTANT_SENSORS]  # 14 sensors
WINDOW = 10


def find_constant_sensors(df: pd.DataFrame, tol: float = 1e-8) -> List[str]:
    """Data-driven check used in the notebook to justify CONSTANT_SENSORS."""
    sens = [c for c in df.columns if c.startswith("s")]
    return [c for c in sens if df[c].std() < tol or df[c].nunique() <= 2]


def causal_window_features(df: pd.DataFrame, window: int = WINDOW, sensors=SENSORS) -> pd.DataFrame:
    """Trailing rolling mean of each sensor, computed per engine (no future data)."""
    out = df.copy()
    rolled = (
        out.groupby("unit")[sensors]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out[sensors] = rolled.loc[out.index, sensors]
    return out


@dataclass
class FeaturePipeline:
    """scale -> PCA, fitted on training engines only."""
    n_components: int = 4
    window: int = WINDOW

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        sm = causal_window_features(df, self.window)
        self.scaler_ = StandardScaler().fit(sm[SENSORS])
        self.pca_ = PCA(n_components=self.n_components, random_state=0).fit(self.scaler_.transform(sm[SENSORS]))
        # Orient PC1 so that it increases with degradation (sign fixed using fit data only:
        # compare PC1 at the last vs first third of every engine, no RUL values needed).
        z = self.pca_.transform(self.scaler_.transform(sm[SENSORS]))
        pos = sm.groupby("unit").cumcount() / sm.groupby("unit")["cycle"].transform("count")
        early, late = z[pos.values < 0.2, 0].mean(), z[pos.values > 0.8, 0].mean()
        self.sign_ = np.ones(self.n_components)
        if late < early:
            self.sign_[0] = -1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        sm = causal_window_features(df, self.window)
        z = self.pca_.transform(self.scaler_.transform(sm[SENSORS])) * self.sign_
        out = pd.DataFrame(z, columns=[f"pc{i+1}" for i in range(self.n_components)], index=df.index)
        out.insert(0, "unit", df["unit"].values)
        out.insert(1, "cycle", df["cycle"].values)
        if "rul" in df:
            out["rul"] = df["rul"].values
        return out


def to_sequences(feat: pd.DataFrame, n_components: int) -> Dict[int, np.ndarray]:
    """Sequence construction: {engine id -> (T, D) array in cycle order}."""
    cols = [f"pc{i+1}" for i in range(n_components)]
    return {int(u): g.sort_values("cycle")[cols].to_numpy() for u, g in feat.groupby("unit")}


def split_engines(units, val_frac: float = 0.3, seed: int = 42):
    """Engine-level split (each engine is an independent run-to-failure trajectory)."""
    units = np.array(sorted(units))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = int(round(len(units) * val_frac))
    return sorted(units[n_val:].tolist()), sorted(units[:n_val].tolist())

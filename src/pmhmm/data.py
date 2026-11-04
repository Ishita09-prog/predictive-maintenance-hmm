"""Loading and labelling the NASA C-MAPSS turbofan data (sub-set FD001)."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MIRROR = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"
FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

COLS = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]

# Ground-truth health bands used ONLY for evaluation (never for training).
# RUL > HEALTHY_RUL -> healthy, FAULT_RUL < RUL <= HEALTHY_RUL -> degrading, RUL <= FAULT_RUL -> fault.
HEALTHY_RUL = 100
FAULT_RUL = 30


def download(raw_dir: Path = RAW_DIR) -> None:
    """Download FD001 files if missing (public mirror of the NASA PCoE repository)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = raw_dir / f
        if not dst.exists() or dst.stat().st_size == 0:
            print(f"downloading {f} ...")
            urllib.request.urlretrieve(f"{MIRROR}/{f}", dst)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def load_fd001(raw_dir: Path = RAW_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) data frames with a per-cycle ground-truth `rul` column.

    * train: engines run to failure -> rul = max_cycle - cycle
    * test : engines are truncated; RUL_FD001.txt gives the remaining life after the
             last observed cycle -> rul = (last_cycle + rul_final) - cycle
    """
    download(raw_dir)
    train = _read(raw_dir / "train_FD001.txt")
    test = _read(raw_dir / "test_FD001.txt")
    rul_final = pd.read_csv(raw_dir / "RUL_FD001.txt", header=None, names=["rul_final"])["rul_final"].to_numpy()

    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    last = test.groupby("unit")["cycle"].transform("max")
    final_map = {u: r for u, r in zip(sorted(test["unit"].unique()), rul_final)}
    test["rul"] = last + test["unit"].map(final_map) - test["cycle"]
    return train, test


def health_band(rul: np.ndarray) -> np.ndarray:
    """0 = healthy, 1 = degrading, 2 = fault (evaluation labels only)."""
    rul = np.asarray(rul)
    return np.where(rul <= FAULT_RUL, 2, np.where(rul <= HEALTHY_RUL, 1, 0))

"""Alert logic and evaluation metrics (fault recall, precision, F1, log-likelihood, lead time)."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .data import FAULT_RUL

PERSIST = 3  # consecutive cycles a score must stay above the threshold to raise an alert


def first_alert(score: np.ndarray, thr: float, persist: int = PERSIST) -> int:
    """Index of the cycle at which an alert is raised (causal), or -1 if never.

    The alert fires at the end of the first run of `persist` consecutive cycles with
    score >= thr, i.e. it only uses information available at that cycle.
    """
    run = 0
    for t, s in enumerate(score):
        run = run + 1 if s >= thr else 0
        if run >= persist:
            return t
    return -1


def latched_prediction(n: int, alert_idx: int) -> np.ndarray:
    """Binary 'fault predicted' flag per cycle: 0 before the alert, 1 from the alert on."""
    pred = np.zeros(n, dtype=int)
    if alert_idx >= 0:
        pred[alert_idx:] = 1
    return pred


def evaluate_alerts(scores: Dict[int, np.ndarray], rul: Dict[int, np.ndarray], thr: float,
                    persist: int = PERSIST, fault_rul: int = FAULT_RUL) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    leads, alerted, in_fault_zone, detected_in_zone = [], 0, 0, 0
    early = 0
    for u, s in scores.items():
        r = rul[u]
        a = first_alert(s, thr, persist)
        pred = latched_prediction(len(s), a)
        truth = (r <= fault_rul).astype(int)
        tp += int(((pred == 1) & (truth == 1)).sum())
        fp += int(((pred == 1) & (truth == 0)).sum())
        fn += int(((pred == 0) & (truth == 1)).sum())
        tn += int(((pred == 0) & (truth == 0)).sum())
        if a >= 0:
            alerted += 1
            leads.append(float(r[a]))            # true RUL at the moment of the alert = lead time
            if r[a] > fault_rul + 70:            # alert more than 100 cycles before failure
                early += 1
        if truth.any():
            in_fault_zone += 1
            detected_in_zone += int(a >= 0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {
        "precision": prec, "fault_recall": rec, "f1": f1, "false_positive_rate": fpr,
        "engine_recall": detected_in_zone / max(in_fault_zone, 1),
        "engines_in_fault_zone": in_fault_zone,
        "engines_alerted": alerted, "n_engines": len(scores),
        "mean_lead_time": float(np.mean(leads)) if leads else float("nan"),
        "median_lead_time": float(np.median(leads)) if leads else float("nan"),
        "early_alerts_gt100": early,
    }


def tune_threshold(scores: Dict[int, np.ndarray], rul: Dict[int, np.ndarray],
                   grid: Iterable[float], persist: int = PERSIST) -> float:
    """Pick the threshold with the best cycle-level fault F1 on VALIDATION engines only."""
    best_thr, best_f1 = None, -1.0
    for thr in grid:
        f1 = evaluate_alerts(scores, rul, thr, persist)["f1"]
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return float(best_thr)

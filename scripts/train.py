"""Reproduce all numbers: download data, fit the HMM, tune on validation, evaluate on test, save artifacts.

    python scripts/train.py
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from pmhmm.pipeline import ARTIFACT_DIR, fit_hmm, prepare, run_all, save_artifacts  # noqa: E402


def main() -> None:
    prep = prepare()
    model = fit_hmm(prep)
    results = run_all(prep, model)
    save_artifacts(prep, model, results)
    print(f"EM iterations: {len(model.loglik_history_)}   final train log-lik: {model.loglik_history_[-1]:.1f}")
    for name, r in results.items():
        t = r["test"]
        print(f"{name:26s} thr={r['threshold']:.3f}  TEST  F1={t['f1']:.3f}  recall={t['fault_recall']:.3f}  "
              f"precision={t['precision']:.3f}  engine-recall={t['engine_recall']:.2f}  mean lead={t['mean_lead_time']:.1f}")
    print(f"artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()

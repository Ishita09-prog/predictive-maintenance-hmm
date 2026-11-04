# Intelligent Predictive Maintenance and Machine Failure Monitoring System (HMM)

**Unit 4 – Sequential Data · Industry-Oriented Project 1** · SRM Institute of Science and Technology, Kattankulatham

Detect latent machine-health conditions and warn before failure using a **Hidden Markov Model** whose
**Forward / Backward / Forward-Backward / Viterbi / Baum-Welch** algorithms are implemented from scratch in NumPy
and verified against `hmmlearn`.

* **Dataset:** NASA C-MAPSS **FD001** (100 train engines run to failure, 100 test engines truncated, true RUL given)
* **Model:** left-to-right Gaussian HMM (6 states, *Healthy → … → Critical*) on causal-smoothed, PCA-reduced sensor sequences
* **Output:** Streamlit machine-health dashboard with state probabilities, health trend and alerts

> ⚠️ **Disclosures.** C-MAPSS is **simulated** data (physics-based simulator), no extra synthetic data was added.
> There are **no access restrictions** (public dataset). This is an **educational prototype, not a production maintenance system.**

## Deliverables (project-brief checklist)

| Required | Where |
|---|---|
| Reproducible Jupyter Notebook | [`notebooks/predictive_maintenance_hmm.ipynb`](notebooks/predictive_maintenance_hmm.ipynb) (executed, outputs saved; fixed seed 42; data auto-downloaded) |
| Concise technical report | [`reports/Technical_Report.docx`](reports/Technical_Report.docx) |
| Presentation | [`reports/Presentation.pptx`](reports/Presentation.pptx) |
| GitHub repository with README | this repository |
| Working demonstration | Streamlit dashboard: `streamlit run app/streamlit_app.py` |

**Workflow (Dataset → preprocessing → sequence construction → model formulation → algorithm implementation → inference → evaluation → visualization → interpretation)** is the section order of the notebook (sections 1–10).

**Academic requirements:** own implementation of Unit 4 algorithms (`src/pmhmm/hmm_scratch.py`) compared with a reference implementation (`hmmlearn`) *and* baselines (GMM, PC1 threshold); chronological/causal features and engine-disjoint splits with no leakage (see below); disclosures above.

## Quick start

```bash
pip install -r requirements.txt
python scripts/train.py                       # downloads data, trains, tunes on validation, evaluates test, saves artifacts/
jupyter nbconvert --to notebook --execute --inplace notebooks/predictive_maintenance_hmm.ipynb   # or open it in Jupyter
streamlit run app/streamlit_app.py            # dashboard
pytest -q tests                               # 11 unit tests
```

## Results (official FD001 test set, 100 engines; thresholds and K chosen on 30 validation engines only)

Fault = RUL ≤ 30 cycles; alert latched once raised; lead time = true RUL at first alert.

| Method | Fault recall | Precision | F1 | Engine recall | Mean lead (cycles) |
|---|---|---|---|---|---|
| **HMM (from scratch)** | 0.485 | 0.774 | 0.596 | 0.60 | 27.1 |
| GMM (no temporal model) | 0.334 | 1.000 | 0.501 | 0.32 | 25.2 |
| PC1 threshold | 0.672 | 0.556 | 0.608 | 0.72 | 39.2 |

Held-out HMM log-likelihood per cycle: validation −2.712, test −2.200. The from-scratch HMM reproduces `hmmlearn` exactly
(same parameters: log-likelihood difference < 1e-9, identical Viterbi paths; independent training: log-likelihood −38529.85 vs −38529.85, 100 % Viterbi agreement).

**Honest reading:** the HMM beats the non-temporal GMM, but its F1 is *not* better than a tuned PC1 threshold (it is more precise, less sensitive). Its value is the probabilistic state ladder (uncertainty, dwell times, explanations). Only 25 of 100 test engines end inside the fault zone, so engine-level numbers rest on small counts.

## Leakage control
* Features are **trailing/causal** (unit-tested by corrupting future cycles).
* Engines are independent run-to-failure trajectories → **engine-level** 70/30 fit/validation split; the official test set is evaluated once.
* Scaler + PCA fitted on fit engines only; thresholds and K tuned on validation only.
* Alerts use the **forward (filtering)** posterior only (no future data). RUL labels are used for evaluation only.

## Repository layout
```
src/pmhmm/hmm_scratch.py   scaled Forward/Backward, Forward-Backward, Viterbi, Baum-Welch, AIC/BIC  (own implementation)
src/pmhmm/data.py          download + RUL labelling         src/pmhmm/features.py   causal windows, scaler+PCA, sequences, split
src/pmhmm/evaluate.py      alert rule + metrics             src/pmhmm/pipeline.py   shared pipeline, baselines
scripts/train.py           reproduce everything             app/streamlit_app.py    dashboard
tests/                     11 unit tests (brute-force checks, hmmlearn equivalence, causality, split)
notebooks/ · reports/ · docs/figures/ · artifacts/
```

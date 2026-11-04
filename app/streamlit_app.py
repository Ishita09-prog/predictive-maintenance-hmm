"""Machine-health monitoring dashboard (educational prototype).  Run:  streamlit run app/streamlit_app.py"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from pmhmm import pipeline as P  # noqa: E402
from pmhmm.data import FAULT_RUL, load_fd001  # noqa: E402
from pmhmm.evaluate import PERSIST, first_alert  # noqa: E402
from pmhmm.features import SENSORS, causal_window_features  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance – HMM", layout="wide", page_icon="🛠️")
COLORS = ["#2e7d32", "#7cb342", "#fbc02d", "#fb8c00", "#e64a19", "#b71c1c"]


@st.cache_resource(show_spinner="Loading data and model …")
def load_all():
    if not (P.ARTIFACT_DIR / "hmm_params.npz").exists():
        st.error("Model artifacts missing – run `python scripts/train.py` first.")
        st.stop()
    prep = P.prepare()
    model = P.load_model()
    results = json.loads((P.ARTIFACT_DIR / "results.json").read_text())
    return prep, model, results


prep, model, results = load_all()
K = model.n_states
names = P.STATE_NAMES
hmm_res = results["HMM (from scratch)"]

st.title("🛠️ Intelligent Predictive Maintenance – Machine-Health Dashboard")
st.warning("**Educational prototype** on *simulated* NASA C-MAPSS FD001 turbofan data – not a production maintenance or safety system.")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Controls")
split = st.sidebar.radio("Engine set", ["Test engines (held out)", "Validation engines (run to failure)"])
if split.startswith("Test"):
    seqs, ruls, df = prep.seq_test, prep.rul_test, prep.test
    units = sorted(seqs)
else:
    seqs, ruls, df = prep.seq_train, prep.rul_train, prep.train
    units = sorted(prep.val_units)
unit = st.sidebar.selectbox("Engine", units, index=min(67, len(units) - 1) if split.startswith("Test") else 0)
thr = st.sidebar.slider("Alert threshold on P(Critical)", 0.01, 0.999, float(hmm_res["threshold"]), 0.01,
                        help="Default = value tuned on validation engines.")
persist = st.sidebar.slider("Consecutive cycles above threshold", 1, 6, PERSIST)
X, rul = seqs[unit], ruls[unit]
T = len(X)
cur = st.sidebar.slider("Replay: cycles revealed", 5, T, T, help="Moves the 'present' – only data up to this cycle is used for alerts.")
show_truth = st.sidebar.checkbox("Show ground-truth RUL (evaluation only)", value=True)

# --------------------------------------------------------------- inference
filt, ll = model.forward(X[:cur])                  # causal: only revealed cycles
smooth = model.smooth(X[:cur])
vit, _ = model.viterbi(X[:cur])
p_crit = filt[:, -1]
a = first_alert(p_crit, thr, persist)
state_now = int(filt[-1].argmax())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Engine / cycle", f"{unit} / {cur}")
c2.metric("Most likely health state", names[state_now])
c3.metric("P(Critical | past)", f"{p_crit[-1]:.3f}")
c4.metric("Alert", "🚨 RAISED" if a >= 0 else "✅ none", f"at cycle {a + 1}" if a >= 0 else None, delta_color="off")
if show_truth:
    c5.metric("True RUL now", int(rul[cur - 1]), f"alert lead {int(rul[a])} cycles" if a >= 0 else None, delta_color="off")
else:
    c5.metric("Observed cycles", cur)

tab1, tab2, tab3, tab4 = st.tabs(["Engine monitor", "Fleet overview", "Model & metrics", "About"])

with tab1:
    x = np.arange(1, cur + 1)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=("State probabilities (forward / filtering – uses past only)",
                                        "Health indicator (PC1) and decoded Viterbi state",
                                        "Probability of Critical state vs alert threshold"))
    for k in range(K):
        fig.add_trace(go.Scatter(x=x, y=filt[:, k], stackgroup="s", name=names[k], line=dict(width=0.5, color=COLORS[k])), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=X[:cur, 0], name="PC1", line=dict(color="#1565c0")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=model.means_[vit, 0], name="Viterbi state mean", line=dict(color="#6a1b9a", shape="hv", dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=p_crit, name="P(Critical)", line=dict(color="#b71c1c")), row=3, col=1)
    fig.add_hline(y=thr, line_dash="dot", line_color="gray", row=3, col=1)
    if show_truth:
        fig.add_trace(go.Scatter(x=x, y=np.clip(rul[:cur], 0, 150) / 150, name="RUL/150 (truth)", line=dict(color="green", dash="dash")), row=3, col=1)
        fault_start = np.where(rul[:cur] <= FAULT_RUL)[0]
        if len(fault_start):
            fig.add_vrect(x0=fault_start[0] + 1, x1=cur, fillcolor="red", opacity=0.07, line_width=0, annotation_text="true fault zone (RUL ≤ 30)", annotation_position="bottom right")
    if a >= 0:
        fig.add_vline(x=a + 1, line_color="red", line_dash="dash", annotation_text="ALERT", annotation_position="top left")
    fig.update_layout(height=760, legend=dict(orientation="h", y=-0.08), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensor trends (causal 10-cycle mean)")
    chosen = st.multiselect("Sensors", SENSORS, default=["s2", "s4", "s11", "s15"])
    e = df[df["unit"] == unit].sort_values("cycle").head(cur)
    sm = causal_window_features(e)
    if chosen:
        z = (sm[chosen] - sm[chosen].iloc[0])
        st.line_chart(z.set_index(sm["cycle"]), height=240)
        st.caption("Each sensor shown as change from its first value, so drift is comparable across sensors.")

with tab2:
    st.subheader("Fleet status at the end of each engine's observation")
    rows = []
    for u in units:
        pc = model.forward(seqs[u])[0][:, -1]
        ai = first_alert(pc, thr, persist)
        r = {"engine": u, "cycles": len(pc), "P(Critical) now": round(float(pc[-1]), 3),
             "state now": names[int(model.forward(seqs[u])[0][-1].argmax())], "alert": "🚨" if ai >= 0 else "",
             "alert cycle": ai + 1 if ai >= 0 else None}
        if show_truth:
            r["true RUL at end"] = int(ruls[u][-1])
            r["lead time (cycles)"] = int(ruls[u][ai]) if ai >= 0 else None
        rows.append(r)
    fleet = pd.DataFrame(rows)
    st.dataframe(fleet.sort_values("P(Critical) now", ascending=False), use_container_width=True, height=420, hide_index=True)
    s1, s2 = st.columns(2)
    s1.metric("Engines alerted", f"{int((fleet['alert'] == '🚨').sum())} / {len(fleet)}")
    if show_truth and fleet["lead time (cycles)"].notna().any():
        s2.metric("Mean lead time at alert (cycles)", f"{fleet['lead time (cycles)'].mean():.1f}")
    cnt = fleet["state now"].value_counts().reindex([names[i] for i in range(K)]).fillna(0)
    bar = go.Figure(go.Bar(x=list(cnt.index), y=cnt.values, marker_color=COLORS))
    bar.update_layout(height=280, title="Engines by current most-likely state", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(bar, use_container_width=True)

with tab3:
    st.subheader("Official test-set results (thresholds tuned on validation engines only)")
    t = pd.DataFrame({m: v["test"] for m, v in results.items()}).T[
        ["fault_recall", "precision", "f1", "false_positive_rate", "engine_recall", "mean_lead_time"]].round(3)
    st.dataframe(t, use_container_width=True)
    st.subheader(f"Learned HMM ({K} states, left-to-right)")
    A = pd.DataFrame(model.transmat_, index=[names[i] for i in range(K)], columns=[names[i] for i in range(K)]).round(4)
    st.dataframe(A, use_container_width=True)
    st.caption("Metrics: fault = RUL ≤ 30 cycles; alert latched once raised; lead time = true RUL at first alert. "
               "See the notebook for validation results, baselines and the reference-implementation check.")

with tab4:
    st.markdown("""
**Method.** Sensor readings → causal 10-cycle smoothing → standardise → PCA (3 comps) → Gaussian **HMM trained with Baum-Welch (own NumPy implementation)**.
Alerts use the **forward (filtering)** probability of the *Critical* state, so they only use information available at that cycle.
Forward-Backward smoothing and Viterbi decoding are available for retrospective explanation.

**Data.** NASA C-MAPSS FD001 – *simulated* turbofan run-to-failure data (public, no access restriction). Ground-truth RUL is used only to evaluate, never to train.

**Disclaimer.** Educational prototype; not a production system.
""")

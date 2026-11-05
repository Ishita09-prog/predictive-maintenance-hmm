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
from pmhmm.data import FAULT_RUL  # noqa: E402
from pmhmm.evaluate import PERSIST, first_alert  # noqa: E402
from pmhmm.features import SENSORS, causal_window_features  # noqa: E402

st.set_page_config(page_title="Machine Health", layout="wide", initial_sidebar_state="expanded")

# Apple system colours: green → yellow → orange → red
COLORS = ["#30d158", "#a4d62c", "#ffd60a", "#ff9f0a", "#ff6a3d", "#ff453a"]
BLUE, PURPLE, TEXT, MUTED = "#0a84ff", "#bf5af2", "#f5f5f7", "rgba(245,245,247,.62)"

CSS = """
<style>
html, body, [class*="css"], .stApp { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Segoe UI", sans-serif !important; font-weight: 400; }
html, body { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
.stApp {
  background:
    radial-gradient(1100px 700px at 8% -10%, rgba(10,132,255,.38), transparent 60%),
    radial-gradient(900px 650px at 96% 2%, rgba(191,90,242,.32), transparent 60%),
    radial-gradient(1000px 700px at 55% 112%, rgba(48,209,88,.16), transparent 60%),
    radial-gradient(700px 500px at 0% 90%, rgba(255,159,10,.12), transparent 60%),
    #0b0d17;
  background-attachment: fixed;
}
#MainMenu, footer, [data-testid="stToolbar"], .stAppDeployButton, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1320px; }

/* sidebar = macOS translucent sidebar */
section[data-testid="stSidebar"] { background: rgba(255,255,255,.055); backdrop-filter: blur(40px) saturate(180%); -webkit-backdrop-filter: blur(40px) saturate(180%); border-right: 1px solid rgba(255,255,255,.10); }
section[data-testid="stSidebar"] > div { background: transparent; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: rgba(245,245,247,.5); font-weight: 600; }
div[data-baseweb="select"] > div, div[data-baseweb="select"] > div > div { background: rgba(255,255,255,.09) !important; background-color: rgba(255,255,255,.09) !important; border: 1px solid rgba(255,255,255,.14) !important; border-radius: 12px !important; }
div[data-baseweb="tag"] { background: rgba(10,132,255,.35) !important; border-radius: 8px !important; }
label, .stMarkdown p { color: rgba(245,245,247,.85); }
[data-testid="stWidgetLabel"] p { font-size: .82rem; color: rgba(245,245,247,.7); }
[data-testid="stSlider"] [role="slider"] { box-shadow: 0 0 0 4px rgba(10,132,255,.25); }

/* window chrome */
.chrome { display:flex; align-items:center; justify-content:space-between; padding: 10px 18px; margin-bottom: 18px;
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); border-radius: 18px;
  backdrop-filter: blur(30px) saturate(170%); -webkit-backdrop-filter: blur(30px) saturate(170%);
  box-shadow: 0 8px 32px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.16); }
.lights { display:flex; gap:8px; width:150px; }
.lights i { width:12px; height:12px; border-radius:50%; display:block; box-shadow: inset 0 0 0 .5px rgba(0,0,0,.25); }
.chrome .title { font-weight:600; font-size:.95rem; letter-spacing:.01em; color:#f5f5f7; text-align:center; }
.chrome .title small { display:block; font-weight:400; font-size:.7rem; color:rgba(245,245,247,.55); letter-spacing:.04em; }
.pill { white-space:nowrap; display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:999px; font-size:.72rem; font-weight:600;
  background: rgba(255,159,10,.16); color:#ffb340; border:1px solid rgba(255,159,10,.35); }
.chrome .right { width:150px; display:flex; justify-content:flex-end; }

/* iOS-style widgets */
.glass { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); border-radius: 26px; padding: 18px 20px;
  backdrop-filter: blur(26px) saturate(170%); -webkit-backdrop-filter: blur(26px) saturate(170%);
  box-shadow: 0 12px 40px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.18); height: 168px; position: relative; overflow: hidden; }
.glass .lbl { font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color: rgba(245,245,247,.55); font-weight:600; }
.glass .big { font-size: 2.2rem; font-weight: 600; letter-spacing:-.02em; line-height:1.1; margin-top:10px; color:#fff; }
.glass .sub { font-size:.82rem; color: rgba(245,245,247,.62); margin-top:6px; }
.glass .dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:9px; vertical-align:middle; }
.glow-red { box-shadow: 0 12px 40px rgba(0,0,0,.38), 0 0 60px rgba(255,69,58,.35), inset 0 1px 0 rgba(255,255,255,.18); border-color: rgba(255,69,58,.5); }
.glow-green { box-shadow: 0 12px 40px rgba(0,0,0,.38), 0 0 50px rgba(48,209,88,.20), inset 0 1px 0 rgba(255,255,255,.18); }
.ring-wrap { display:flex; align-items:center; gap:16px; height:100%; }
.ring-wrap .txt { flex:1; }

/* charts as glass cards */
div[data-testid="stPlotlyChart"] { background: rgba(255,255,255,.075); border: 1px solid rgba(255,255,255,.16); border-radius: 24px; padding: 10px 12px;
  backdrop-filter: blur(22px) saturate(160%); -webkit-backdrop-filter: blur(22px) saturate(160%); box-shadow: 0 12px 40px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.14); margin-bottom: 14px; }

/* segmented-control tabs (macOS) */
[role="tablist"] { background: rgba(255,255,255,.09) !important; border: 1px solid rgba(255,255,255,.14) !important; border-radius: 13px !important; padding: 4px !important; gap: 2px !important; width: fit-content !important; margin: 22px 0 16px !important; backdrop-filter: blur(20px); box-shadow: inset 0 1px 0 rgba(255,255,255,.12); }
[role="tablist"] [role="tab"], div[data-testid="stTab"] { background: transparent !important; border-radius: 10px !important; height: auto !important; padding: 7px 26px !important; color: rgba(245,245,247,.62) !important; border: none !important; cursor: pointer; }
div[data-testid="stTab"] p { color: inherit !important; font-size: .9rem; font-weight: 500; margin: 0; }
div[data-testid="stTab"][aria-selected="true"] { background: rgba(255,255,255,.22) !important; color: #fff !important; box-shadow: 0 2px 12px rgba(0,0,0,.3), inset 0 1px 0 rgba(255,255,255,.28); }
div[data-testid="stTab"]:hover { color:#fff !important; }
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"] { background: rgba(255,255,255,.09) !important; border: 1px solid rgba(255,255,255,.16) !important; border-radius: 12px !important; backdrop-filter: blur(16px); }
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input { color:#f5f5f7 !important; }

/* table */
.tbl-wrap { background: rgba(255,255,255,.075); border: 1px solid rgba(255,255,255,.16); border-radius: 24px; padding: 8px 10px; max-height: 470px; overflow:auto;
  backdrop-filter: blur(22px); box-shadow: 0 12px 40px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.14); }
table.g { width:100%; border-collapse: collapse; font-size:.86rem; }
table.g th { position: sticky; top:0; text-align:left; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color: rgba(245,245,247,.5); font-weight:600; padding:12px 12px; background: rgba(20,24,43,.85); backdrop-filter: blur(10px); }
table.g td { padding: 10px 12px; border-top: 1px solid rgba(255,255,255,.07); color: rgba(245,245,247,.92); }
table.g tr:hover td { background: rgba(255,255,255,.05); }
.chip { display:inline-block; padding: 3px 11px; border-radius: 999px; font-size: .74rem; font-weight: 600; }
.foot { text-align:center; color: rgba(245,245,247,.38); font-size:.74rem; margin-top: 26px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def style_fig(fig, height, margin=None):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="-apple-system, SF Pro Text, Inter, sans-serif", color=TEXT, size=12),
                      margin=margin or dict(l=14, r=14, t=44, b=10), hoverlabel=dict(bgcolor="rgba(30,34,56,.95)", font_color="#fff", bordercolor="rgba(255,255,255,.2)"),
                      legend=dict(orientation="h", y=-0.12, font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.1)", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor="rgba(255,255,255,.1)", tickfont=dict(color=MUTED))
    fig.update_annotations(font=dict(color="rgba(245,245,247,.85)", size=13))
    return fig


def html(s: str):
    st.markdown("".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def ring(p, color):
    r, c = 42, 2 * np.pi * 42
    return (f'<svg width="104" height="104" viewBox="0 0 104 104"><defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="{color}"/><stop offset="100%" stop-color="{hex_rgba(color, .55)}"/></linearGradient></defs>'
            f'<circle cx="52" cy="52" r="{r}" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="11"/>'
            f'<circle cx="52" cy="52" r="{r}" fill="none" stroke="url(#rg)" stroke-width="11" stroke-linecap="round" '
            f'stroke-dasharray="{c * max(p, .015):.1f} {c:.1f}" transform="rotate(-90 52 52)"/>'
            f'<text x="52" y="58" text-anchor="middle" fill="#fff" font-size="20" font-weight="700" font-family="-apple-system, Inter, sans-serif">{p * 100:.0f}%</text></svg>')


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

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### Data")
split = st.sidebar.radio("Engine set", ["Test engines (held out)", "Validation engines (run to failure)"], label_visibility="collapsed")
if split.startswith("Test"):
    seqs, ruls, df = prep.seq_test, prep.rul_test, prep.test
    units = sorted(seqs)
else:
    seqs, ruls, df = prep.seq_train, prep.rul_train, prep.train
    units = sorted(prep.val_units)
unit = st.sidebar.selectbox("Engine", units, index=min(67, len(units) - 1) if split.startswith("Test") else 0)
st.sidebar.markdown("### Alert rule")
thr = st.sidebar.slider("Threshold on P(Critical)", 0.01, 0.999, float(hmm_res["threshold"]), 0.01, help="Default = value tuned on validation engines.")
persist = st.sidebar.slider("Consecutive cycles", 1, 6, PERSIST)
X, rul = seqs[unit], ruls[unit]
T = len(X)
st.sidebar.markdown("### Live replay")
cur = st.sidebar.slider("Cycles revealed", 5, T, T, help="Moves the 'present': only data up to this cycle is used for alerts.")
show_truth = st.sidebar.toggle("Show ground-truth RUL", value=True, help="Evaluation only – never used by the model.")

# --------------------------------------------------------------- inference
filt, ll = model.forward(X[:cur])
smooth = model.smooth(X[:cur])
vit, _ = model.viterbi(X[:cur])
p_crit = filt[:, -1]
a = first_alert(p_crit, thr, persist)
state_now = int(filt[-1].argmax())
col_now = COLORS[state_now]

# ------------------------------------------------------------------ chrome
html(f"""
<div class="chrome">
  <div class="lights"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i></div>
  <div class="title">Machine Health<small>Turbofan engine {unit} · cycle {cur} of {T} · {K}-state HMM</small></div>
  <div class="right"><span class="pill">Simulated data</span></div>
</div>
""")

# ------------------------------------------------------------------ widgets
w1, w2, w3, w4 = st.columns(4, gap="medium")
with w1:
    html(f'<div class="glass"><div class="lbl">Failure risk</div><div class="ring-wrap">{ring(float(p_crit[-1]), "#ff453a" if p_crit[-1] > .5 else col_now)}'
         f'<div class="txt"><div class="sub" style="margin:0">Probability</div><div class="sub" style="color:#fff;font-weight:600;font-size:.95rem">{p_crit[-1]:.3f}</div></div></div></div>')
with w2:
    top2 = np.argsort(filt[-1])[::-1][:2]
    html(f'<div class="glass"><div class="lbl">Health state</div><div class="big"><span class="dot" style="background:{col_now};box-shadow:0 0 14px {col_now}"></span>{names[state_now]}</div>'
         f'<div class="sub">{filt[-1][top2[0]] * 100:.0f}% confidence · state {state_now + 1} of {K}</div></div>')
with w3:
    if a >= 0:
        lead = f"{int(rul[a])} cycles early" if show_truth else f"raised at cycle {a + 1}"
        html(f'<div class="glass glow-red"><div class="lbl">Alert</div><div class="big" style="color:#ff6961">Raised</div><div class="sub">Cycle {a + 1} · {lead}</div></div>')
    else:
        html('<div class="glass glow-green"><div class="lbl">Alert</div><div class="big" style="color:#30d158">All clear</div><div class="sub">No alert in the revealed cycles</div></div>')
with w4:
    if show_truth:
        html(f'<div class="glass"><div class="lbl">True remaining life</div><div class="big">{int(rul[cur - 1])}<span style="font-size:1rem;font-weight:500;color:{MUTED}"> cycles</span></div>'
             f'<div class="sub">{"Inside fault zone (≤ 30)" if rul[cur - 1] <= FAULT_RUL else "Outside fault zone"}</div></div>')
    else:
        html(f'<div class="glass"><div class="lbl">Observed</div><div class="big">{cur}<span style="font-size:1rem;font-weight:500;color:{MUTED}"> cycles</span></div><div class="sub">Ground truth hidden</div></div>')

tab1, tab2, tab3, tab4 = st.tabs(["Monitor", "Fleet", "Model", "About"])

with tab1:
    x = np.arange(1, cur + 1)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=("Health-state probabilities · forward / filtering (past data only)",
                                        "Health indicator (PC1) and decoded Viterbi state", "P(Critical) against the alert threshold"))
    for k in range(K):
        fig.add_trace(go.Scatter(x=x, y=filt[:, k], stackgroup="s", name=names[k], line=dict(width=0.6, color=COLORS[k]), fillcolor=hex_rgba(COLORS[k], .78)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=X[:cur, 0], name="PC1", line=dict(color=BLUE, width=2.4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=model.means_[vit, 0], name="Viterbi state", line=dict(color=PURPLE, shape="hv", dash="dot", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=p_crit, name="P(Critical)", fill="tozeroy", fillcolor="rgba(255,69,58,.18)", line=dict(color="#ff453a", width=2.4)), row=3, col=1)
    fig.add_hline(y=thr, line_dash="dot", line_color="rgba(255,255,255,.45)", row=3, col=1)
    if show_truth:
        fig.add_trace(go.Scatter(x=x, y=np.clip(rul[:cur], 0, 150) / 150, name="RUL / 150 (truth)", line=dict(color="#30d158", dash="dash", width=1.8)), row=3, col=1)
        fz = np.where(rul[:cur] <= FAULT_RUL)[0]
        if len(fz):
            fig.add_vrect(x0=fz[0] + 1, x1=cur, fillcolor="rgba(255,69,58,.10)", line_width=0)
    if a >= 0:
        fig.add_vline(x=a + 1, line_color="#ff453a", line_dash="dash", line_width=1.5)
        fig.add_annotation(x=a + 1, y=1.0, yref="y3", text="ALERT", showarrow=False, xanchor="right", font=dict(color="#ff6961", size=11))
    style_fig(fig, 790)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    chosen = st.multiselect("Sensors", SENSORS, default=["s2", "s4", "s11", "s15"])
    e = df[df["unit"] == unit].sort_values("cycle").head(cur)
    sm = causal_window_features(e)
    if chosen:
        f2 = go.Figure()
        pal = [BLUE, PURPLE, "#30d158", "#ff9f0a", "#ff453a", "#64d2ff", "#ffd60a", "#ff375f"]
        for i, c in enumerate(chosen):
            f2.add_trace(go.Scatter(x=sm["cycle"], y=sm[c] - sm[c].iloc[0], name=c, line=dict(width=2, color=pal[i % len(pal)])))
        f2.update_layout(title=dict(text="Sensor drift (change from first value, 10-cycle causal mean)", font=dict(size=13, color="rgba(245,245,247,.85)")))
        style_fig(f2, 330)
        st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})

with tab2:
    rows = []
    for u in units:
        fa = model.forward(seqs[u])[0]
        pc = fa[:, -1]
        ai = first_alert(pc, thr, persist)
        rows.append(dict(engine=u, cycles=len(pc), p=float(pc[-1]), state=int(fa[-1].argmax()), alert=ai, rul_end=int(ruls[u][-1]),
                         lead=int(ruls[u][ai]) if ai >= 0 else None))
    fleet = pd.DataFrame(rows)
    n_alert = int((fleet.alert >= 0).sum())
    f1, f2c, f3 = st.columns(3, gap="medium")
    f1.markdown(f'<div class="glass" style="height:120px"><div class="lbl">Engines monitored</div><div class="big">{len(fleet)}</div></div>', unsafe_allow_html=True)
    f2c.markdown(f'<div class="glass {"glow-red" if n_alert else ""}" style="height:120px"><div class="lbl">Engines alerted</div><div class="big">{n_alert}</div></div>', unsafe_allow_html=True)
    ml = fleet.lead.dropna().mean() if show_truth and fleet.lead.notna().any() else None
    f3.markdown(f'<div class="glass" style="height:120px"><div class="lbl">Mean warning</div><div class="big">{"–" if ml is None else f"{ml:.1f}"}<span style="font-size:1rem;font-weight:500;color:{MUTED}"> cycles</span></div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    body = ""
    for r in fleet.sort_values("p", ascending=False).itertuples():
        c = COLORS[r.state]
        al = f'<span class="chip" style="background:rgba(255,69,58,.18);color:#ff6961">Alert · cycle {r.alert + 1}</span>' if r.alert >= 0 else '<span style="color:rgba(245,245,247,.35)">—</span>'
        extra = f"<td>{r.rul_end}</td><td>{'' if pd.isna(r.lead) else int(r.lead)}</td>" if show_truth else ""
        body += (f'<tr><td><b>{r.engine}</b></td><td>{r.cycles}</td><td>{r.p:.3f}</td>'
                 f'<td><span class="chip" style="background:{hex_rgba(c, .2)};color:{c}">{names[r.state]}</span></td><td>{al}</td>{extra}</tr>')
    head = "<th>Engine</th><th>Cycles</th><th>P(Critical)</th><th>State</th><th>Alert</th>" + ("<th>True RUL at end</th><th>Lead time</th>" if show_truth else "")
    html(f'<div class="tbl-wrap"><table class="g"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    cnt = fleet.state.value_counts().reindex(range(K)).fillna(0)
    fb = go.Figure(go.Bar(x=[names[i] for i in range(K)], y=cnt.values, marker=dict(color=COLORS, cornerradius=8)))
    fb.update_layout(title=dict(text="Engines by current most-likely state", font=dict(size=13, color="rgba(245,245,247,.85)")))
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.plotly_chart(style_fig(fb, 300), use_container_width=True, config={"displayModeBar": False})

with tab3:
    st.markdown("##### Official test-set results · thresholds tuned on validation engines only")
    body = ""
    for m, v in results.items():
        t = v["test"]
        hl = ' style="background:rgba(10,132,255,.12)"' if m.startswith("HMM") else ""
        body += (f'<tr{hl}><td><b>{m}</b></td><td>{t["fault_recall"]:.3f}</td><td>{t["precision"]:.3f}</td><td>{t["f1"]:.3f}</td>'
                 f'<td>{t["false_positive_rate"]:.3f}</td><td>{t["engine_recall"]:.2f}</td><td>{t["mean_lead_time"]:.1f}</td></tr>')
    html('<div class="tbl-wrap"><table class="g"><thead><tr><th>Method</th><th>Fault recall</th><th>Precision</th><th>F1</th><th>False-pos. rate</th><th>Engine recall</th><th>Mean lead (cycles)</th></tr></thead>'
         f'<tbody>{body}</tbody></table></div>')
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(f"##### Learned transition matrix · left-to-right, {K} states")
    labs = [names[i] for i in range(K)]
    hm = go.Figure(go.Heatmap(z=model.transmat_, x=labs, y=labs, colorscale=[[0, "rgba(255,255,255,0.03)"], [1, "#0a84ff"]], zmin=0, zmax=1,
                              text=np.round(model.transmat_, 3), texttemplate="%{text}", textfont=dict(size=12, color="#fff"), showscale=False, xgap=3, ygap=3))
    hm.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(hm, 420, dict(l=14, r=14, t=10, b=10)), use_container_width=True, config={"displayModeBar": False})
    st.caption("Fault = RUL ≤ 30 cycles; alert latched once raised; lead time = true RUL at the first alert. See the notebook for validation results, baselines and the reference-implementation check.")

with tab4:
    st.markdown("""
##### How it works
**Method.** Sensor readings → causal 10-cycle smoothing → standardise → PCA (3 components) → Gaussian **HMM trained with Baum-Welch (own NumPy implementation)**.
Alerts use the **forward (filtering)** probability of the *Critical* state, so they only use information available at that cycle. Forward-Backward smoothing and Viterbi decoding are for retrospective explanation.

**Data.** NASA C-MAPSS FD001 – *simulated* turbofan run-to-failure data (public, no access restriction). Ground-truth RUL is used only to evaluate, never to train.

**Disclaimer.** Educational prototype – not a production maintenance or safety system.
""")

html('<div class="foot">Predictive Maintenance · Hidden Markov Model · NASA C-MAPSS FD001 · SRM Institute of Science and Technology</div>')

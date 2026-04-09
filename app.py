"""
BreatheEasy – Indoor Air Quality Dashboard
Two jobs:
  1. Room Forecast  – pick room + time, see CO₂ now and where it's headed
  2. Infra Compare  – which infrastructure type has the worst air quality
"""

import io, json, os, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="BreatheEasy", page_icon="🌬️", layout="wide")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
HF_BASE  = "https://huggingface.co/datasets/mufliha/iaq-prediction/resolve/main"
HF_CACHE = Path(".hf_cache"); HF_CACHE.mkdir(exist_ok=True)

CO2_WARN     = 1000   # ppm
CO2_CRITICAL = 2000   # ppm

HORIZONS = {
    "11min": ("11 min", pd.Timedelta(minutes=11)),
    "33min": ("33 min", pd.Timedelta(minutes=33)),
    "1h":    ("1 h",    pd.Timedelta(hours=1)),
    "3h":    ("3 h",    pd.Timedelta(hours=3)),
    "6h":    ("6 h",    pd.Timedelta(hours=6)),
}

SECONDARY = {
    "Measured PM2.5": ("PM2.5",       "ug/m3"),
    "Measured PM10":  ("PM10",        "ug/m3"),
    "Measured T":     ("Temperature", "C"),
    "Measured RH":    ("Humidity",    "%"),
}

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.banner-safe     {background:#ecfdf3;border-left:5px solid #22c55e;color:#166534;padding:12px 16px;border-radius:10px;margin:8px 0;}
.banner-warn     {background:#fff7ed;border-left:5px solid #f59e0b;color:#9a6700;padding:12px 16px;border-radius:10px;margin:8px 0;}
.banner-critical {background:#fef2f2;border-left:5px solid #ef4444;color:#991b1b;padding:12px 16px;border-radius:10px;margin:8px 0;}

.hz-card {border-radius:12px;padding:14px 8px;text-align:center;border:1.5px solid #e5e7eb;margin-bottom:4px;}
.hz-label {font-size:.75rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;}
.hz-pred  {font-size:1.5rem;font-weight:700;margin:4px 0 2px 0;}
.hz-status{font-size:.78rem;font-weight:600;margin-bottom:8px;}
.hz-actual{font-size:.8rem;color:#374151;border-top:1px solid #e5e7eb;padding-top:6px;margin-top:4px;}
.hz-err   {font-size:.75rem;color:#9ca3af;}
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def fetch(filename):
    cache = HF_CACHE / filename
    if cache.exists():
        return io.BytesIO(cache.read_bytes())
    try:
        r = requests.get(f"{HF_BASE}/{filename}", timeout=180)
        r.raise_for_status()
        cache.write_bytes(r.content)
        return io.BytesIO(r.content)
    except Exception:
        return None

def load_local_or_hf(filename):
    for p in [Path(filename), Path("processed") / filename]:
        if p.exists():
            return io.BytesIO(p.read_bytes())
    return fetch(filename)

@st.cache_data(show_spinner="Loading data...")
def load_all():
    full, test = None, None
    for name in ["full_featured.parquet"]:
        b = load_local_or_hf(name)
        if b:
            full = pd.read_parquet(b)
            break
    # test.parquet (17.8 MB) has the _future_ columns we need for validation.
    # test_display.parquet (1.29 MB) is a trimmed version that drops them — load it only as fallback.
    for name in ["test.parquet", "test_display.parquet"]:
        b = load_local_or_hf(name)
        if b:
            test = pd.read_parquet(b)
            # Verify it actually has future columns; if not, keep trying
            future_cols = [c for c in test.columns if "_future_" in c]
            if future_cols:
                break
            # Has no future cols — keep as fallback but continue looking
            test_fallback = test
            test = None
    # If nothing had future cols, use the fallback
    if test is None:
        test = test_fallback if 'test_fallback' in dir() else None

    for df in [full, test]:
        if df is not None and "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

    scaler_p, feat_m = None, None
    b = load_local_or_hf("scaler_params.json")
    if b:
        scaler_p = json.loads(b.read())
    b = load_local_or_hf("feature_list.json")
    if b:
        feat_m = json.loads(b.read())

    return full, test, scaler_p, feat_m

@st.cache_resource(show_spinner=False)
def load_model(horizon):
    fname = f"LightGBM_Measured_CO2_{horizon}.pkl"
    for p in [Path("models") / fname, Path(fname)]:
        if p.exists():
            return pickle.load(open(p, "rb"))
    b = load_local_or_hf(fname)
    return pickle.load(b) if b else None


# ── HELPERS ───────────────────────────────────────────────────────────────────
def build_scaler(p):
    if not p:
        return None
    s = MinMaxScaler()
    s.data_min_      = np.array(p["data_min_"])
    s.data_max_      = np.array(p["data_max_"])
    s.data_range_    = s.data_max_ - s.data_min_
    s.scale_         = np.where(s.data_range_ != 0, 1 / s.data_range_, 0)
    s.min_           = -s.scale_ * s.data_min_
    s.feature_range  = (0, 1)
    s.n_features_in_ = len(s.data_min_)
    return s

def co2_status(v):
    if v >= CO2_CRITICAL:
        return "critical", "Critical",  "#991b1b", "#fef2f2", "#ef4444"
    if v >= CO2_WARN:
        return "warn",     "Elevated",  "#9a6700", "#fff7ed", "#f59e0b"
    return                 "safe",      "Safe",    "#166534", "#ecfdf3", "#22c55e"

def do_predict(row, horizon, scaler, feature_cols):
    model = load_model(horizon)
    if model is None or scaler is None or not feature_cols:
        return None
    X = np.array([[float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
                   for c in feature_cols]])
    try:
        return float(np.ravel(model.predict(scaler.transform(X)))[0])
    except Exception:
        return None

def nearest(df, ts):
    return df.loc[(df["datetime"] - ts).abs().idxmin()]

def banner(cls, html):
    st.markdown(f'<div class="banner-{cls}">{html}</div>', unsafe_allow_html=True)


# ── BOOT ──────────────────────────────────────────────────────────────────────
try:
    df_full, df_test, scaler_params, feature_meta = load_all()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

scaler       = build_scaler(scaler_params)
feature_cols = (feature_meta or {}).get("feature_cols", [])
ref_df       = df_full if df_full is not None else df_test
test_df      = df_test if df_test is not None else ref_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 - ROOM FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## BreatheEasy")
st.caption("Indoor air quality monitoring and AI-powered CO2 forecasting for school administrators.")
st.divider()

st.markdown("### Select a Classroom")
f1, f2, f3, f4, f5 = st.columns([1.4, 0.8, 0.8, 0.9, 0.9])

infra   = f1.selectbox("Infrastructure type", sorted(ref_df["Classroom Type"].dropna().unique()))
sub_ref = ref_df[ref_df["Classroom Type"] == infra]
school  = f2.selectbox("School", sorted(sub_ref["School No"].dropna().astype(int).unique()))
sub_ref = sub_ref[sub_ref["School No"] == school]
room    = f3.selectbox("Room",   sorted(sub_ref["Room No"].dropna().astype(int).unique()))

# Use test_df for predictions - it has _future_ actual columns
room_test = test_df[
    (test_df["Classroom Type"] == infra) &
    (test_df["School No"]      == school) &
    (test_df["Room No"]        == room)
].copy() if test_df is not None else pd.DataFrame()

active_df = room_test if len(room_test) > 0 else sub_ref[sub_ref["Room No"] == room].copy()

min_d     = active_df["datetime"].min().date()
max_d     = active_df["datetime"].max().date()
default_d = max(min_d, min(max_d, pd.Timestamp("2023-11-06").date()))
sel_date  = f4.date_input("Date", value=default_d, min_value=min_d, max_value=max_d)
sel_time  = f5.time_input("Time", value=pd.Timestamp("07:45").time())

selected_ts = pd.Timestamp(f"{sel_date} {sel_time}")
cur_row     = nearest(active_df, selected_ts)
actual_ts   = pd.to_datetime(cur_row["datetime"])

st.caption(
    f"Selected: **{selected_ts.strftime('%Y-%m-%d %H:%M')}** — "
    f"nearest recorded reading: **{actual_ts.strftime('%Y-%m-%d %H:%M')}**"
)

# Show which future columns are actually present so user can diagnose missing actuals
_future_cols = [c for c in active_df.columns if "_future_" in c]
if not _future_cols:
    st.warning(
        "**No future columns found** in the loaded dataset — actual future CO2 values cannot be shown. "
        "This usually means `test_display.parquet` loaded instead of `test.parquet`. "
        "The 17.8 MB `test.parquet` on HuggingFace contains the `_future_` columns needed for validation."
    )
else:
    st.caption(f"Validation data available")

# ── Current snapshot ──────────────────────────────────────────────────────────
co2_now = float(cur_row["Measured CO2"]) if "Measured CO2" in cur_row.index and pd.notna(cur_row["Measured CO2"]) else np.nan

if not np.isnan(co2_now):
    cls, lbl, *_ = co2_status(co2_now)
    msgs = {
        "safe":     f"CO2 is <b>{co2_now:.0f} ppm</b> - air quality is good.",
        "warn":     f"CO2 is <b>{co2_now:.0f} ppm</b> - above the 1,000 ppm threshold. Consider increasing ventilation.",
        "critical": f"CO2 is <b>{co2_now:.0f} ppm</b> - critically high. Immediate action required.",
    }
    banner(cls, msgs[cls])

m_cols = st.columns(5)
m_cols[0].metric("CO2 (ppm)", f"{co2_now:.0f}" if not np.isnan(co2_now) else "-")
for col, (target, (label, unit)) in zip(m_cols[1:], SECONDARY.items()):
    v = float(cur_row[target]) if target in cur_row.index and pd.notna(cur_row[target]) else np.nan
    col.metric(f"{label} ({unit})", f"{v:.1f}" if not np.isnan(v) else "-")

st.divider()

# ── Forecast ──────────────────────────────────────────────────────────────────
st.markdown("### CO2 Forecast from Selected Moment")
st.markdown(
    "LightGBM predicts CO2 at **11 min, 33 min, 1 h, 3 h and 6 h** ahead. "
    "The **actual recorded CO2** at each future time is shown so you can see how accurate the model is."
)

# Run all predictions
results = []
for hz_key, (hz_label, hz_delta) in HORIZONS.items():
    future_ts  = actual_ts + hz_delta
    pred       = do_predict(cur_row, hz_key, scaler, feature_cols)

    # The actual CO2 value recorded at that future time
    actual_col = f"Measured CO2_future_{hz_key}"
    actual_val = (
        float(cur_row[actual_col])
        if actual_col in cur_row.index and pd.notna(cur_row[actual_col])
        else np.nan
    )

    results.append({
        "hz_key":    hz_key,
        "label":     hz_label,
        "future_ts": future_ts,
        "pred":      pred,
        "actual":    actual_val,
    })

# Risk summary
risk_cls, risk_msg = "safe", "Forecast: CO2 expected to remain within safe limits."
for r in results:
    if r["pred"] is None:
        continue
    cls, *_ = co2_status(r["pred"])
    if cls == "critical":
        risk_cls = "critical"
        risk_msg = f"Forecast: CO2 predicted to reach <b>critical levels by {r['label']}</b>. Open windows and increase ventilation now."
        break
    if cls == "warn" and risk_cls == "safe":
        risk_cls = "warn"
        risk_msg = f"Forecast: CO2 predicted to become <b>elevated within {r['label']}</b>. Plan to ventilate soon."

banner(risk_cls, f"<b>{risk_cls.capitalize()}.</b> {risk_msg}")

# Chart
history = active_df[active_df["datetime"] <= actual_ts].tail(20)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history["datetime"], y=history["Measured CO2"],
    mode="lines+markers", name="Actual history",
    line=dict(width=3, color="#335CFF"), marker=dict(size=5),
))

fig.add_trace(go.Scatter(
    x=[actual_ts], y=[co2_now],
    mode="markers", name="Current reading",
    marker=dict(size=14, color="#111827", symbol="circle"),
))

valid_pred = [(r["future_ts"], r["pred"]) for r in results if r["pred"] is not None]
if valid_pred:
    fig.add_trace(go.Scatter(
        x=[actual_ts] + [t for t, _ in valid_pred],
        y=[co2_now]   + [v for _, v in valid_pred],
        mode="lines+markers", name="Model prediction",
        line=dict(width=3, color="#7C3AED", dash="dot"),
        marker=dict(size=10, symbol="diamond"),
    ))

valid_actual = [(r["future_ts"], r["actual"]) for r in results if not np.isnan(r["actual"])]
if valid_actual:
    fig.add_trace(go.Scatter(
        x=[actual_ts] + [t for t, _ in valid_actual],
        y=[co2_now]   + [v for _, v in valid_actual],
        mode="lines+markers", name="Actual CO2 (recorded)",
        line=dict(width=2.5, color="#10B981", dash="dash"),
        marker=dict(size=8, symbol="circle-open", line=dict(width=2, color="#10B981")),
    ))

fig.add_hline(y=CO2_WARN,     line_dash="dash", line_color="#F59E0B",
              annotation_text="Elevated (1,000 ppm)", annotation_position="top left")
fig.add_hline(y=CO2_CRITICAL, line_dash="dash", line_color="#EF4444",
              annotation_text="Critical (2,000 ppm)", annotation_position="top left")

if valid_pred:
    fig.add_vrect(
        x0=actual_ts, x1=results[-1]["future_ts"],
        fillcolor="rgba(124,58,237,0.05)", line_width=0,
        annotation_text="Forecast window", annotation_position="top left",
    )

fig.update_layout(
    title="CO2 - Actual history, model predictions, and actual future values",
    height=420, margin=dict(l=8, r=8, t=48, b=8),
    yaxis_title="CO2 (ppm)", xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(gridcolor="#f3f4f6"), xaxis=dict(gridcolor="#f3f4f6"),
)
st.plotly_chart(fig, width=True)

# Horizon cards - predicted vs actual side by side
st.markdown("**Predicted vs Actual CO2 at each forecast horizon:**")
hz_cols = st.columns(5)

for col, r in zip(hz_cols, results):
    pred   = r["pred"]
    actual = r["actual"]

    if pred is not None:
        p_cls, p_lbl, p_txt, p_bg, p_bdr = co2_status(pred)
        pred_str = f"{pred:.0f} ppm"
    else:
        p_cls, p_lbl, p_txt, p_bg, p_bdr = "safe", "-", "#374151", "#f9fafb", "#e5e7eb"
        pred_str = "-"

    if not np.isnan(actual):
        a_cls, a_lbl, a_txt, *_ = co2_status(actual)
        icon = "✅" if a_cls == "safe" else "⚠️" if a_cls == "warn" else "🔴"
        err_html = (
            f'<span class="hz-err">Error: {abs(pred - actual):.0f} ppm</span>'
            if pred is not None else ""
        )
        actual_html = (
            f'<div class="hz-actual">'
            f'Actual: <b style="color:{a_txt};">{actual:.0f} ppm</b> {icon}'
            f'<br>{err_html}</div>'
        )
    else:
        actual_html = '<div class="hz-actual" style="color:#9ca3af;">No actual data</div>'

    pred_icon = "✅" if p_cls == "safe" else "⚠️" if p_cls == "warn" else "🔴"

    col.markdown(f"""
    <div class="hz-card" style="background:{p_bg};border-color:{p_bdr};">
      <div class="hz-label">{r["label"]} ahead</div>
      <div class="hz-pred" style="color:{p_txt};">{pred_str}</div>
      <div class="hz-status" style="color:{p_txt};">{pred_icon} {p_lbl}</div>
      {actual_html}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 - INFRASTRUCTURE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Infrastructure Type Comparison")
st.markdown(
    "Which type of classroom has the worst air quality across all schools? "
    "**Warning rate** = percentage of readings where CO2 exceeded 1,000 ppm."
)

cmp = ref_df.dropna(subset=["Measured CO2"]).copy()

infra_stats = (
    cmp.groupby("Classroom Type")["Measured CO2"]
    .agg(
        Mean      ="mean",
        Peak      ="max",
        warn_rate =lambda x: (x >= CO2_WARN).mean() * 100,
        crit_rate =lambda x: (x >= CO2_CRITICAL).mean() * 100,
        Readings  ="count",
    )
    .reset_index()
    .sort_values("warn_rate", ascending=False)
    .rename(columns={
        "Classroom Type": "Type",
        "warn_rate":      "Warning rate (%)",
        "crit_rate":      "Critical rate (%)",
    })
)

worst = infra_stats.iloc[0]
best  = infra_stats.iloc[-1]
w_cls = "critical" if worst["Warning rate (%)"] > 30 else "warn" if worst["Warning rate (%)"] > 10 else "safe"

banner(w_cls,
    f"<b>{worst['Type']}</b> has the highest CO2 warning rate: "
    f"<b>{worst['Warning rate (%)']:.1f}%</b> of readings exceed 1,000 ppm "
    f"(mean {worst['Mean']:.0f} ppm, peak {worst['Peak']:.0f} ppm). "
    f"<b>{best['Type']}</b> performs best at {best['Warning rate (%)']:.1f}%."
)

bar_colors = [
    "#DC2626" if v > 30 else "#F59E0B" if v > 10 else "#22c55e"
    for v in infra_stats["Warning rate (%)"]
]

left, right = st.columns(2)

with left:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=infra_stats["Type"],
        y=infra_stats["Warning rate (%)"],
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in infra_stats["Warning rate (%)"]],
        textposition="outside",
        customdata=np.stack([
            infra_stats["Mean"].round(0),
            infra_stats["Peak"].round(0),
            infra_stats["Critical rate (%)"].round(1),
            infra_stats["Readings"],
        ], axis=-1),
        hovertemplate=(
            "<b>%{x}</b><br>Warning rate: %{y:.1f}%<br>"
            "Mean CO2: %{customdata[0]:.0f} ppm<br>"
            "Peak: %{customdata[1]:.0f} ppm<br>"
            "Critical rate: %{customdata[2]:.1f}%<br>"
            "Readings: %{customdata[3]:,}<extra></extra>"
        ),
    ))
    fig2.update_layout(
        title="CO2 Warning Rate by Infrastructure Type",
        height=380, margin=dict(l=8, r=8, t=48, b=8),
        yaxis_title="% readings above 1,000 ppm",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#f3f4f6"),
    )
    st.plotly_chart(fig2, width=True)

with right:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=infra_stats["Type"], y=infra_stats["Mean"],
        name="Mean CO2", marker_color="#335CFF", opacity=0.85,
        text=[f"{v:.0f}" for v in infra_stats["Mean"]], textposition="outside",
    ))
    fig3.add_trace(go.Bar(
        x=infra_stats["Type"], y=infra_stats["Peak"],
        name="Peak CO2", marker_color="#DC2626", opacity=0.6,
        text=[f"{v:.0f}" for v in infra_stats["Peak"]], textposition="outside",
    ))
    fig3.add_hline(y=CO2_WARN, line_dash="dash", line_color="#F59E0B",
                   annotation_text="1,000 ppm threshold", annotation_position="top right")
    fig3.update_layout(
        title="Mean & Peak CO2 by Infrastructure Type (ppm)",
        height=380, barmode="group", margin=dict(l=8, r=8, t=48, b=8),
        yaxis_title="CO2 (ppm)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#f3f4f6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, width=True)

# with st.expander("Full comparison table"):
#     display = infra_stats.copy()
#     display["Mean"] = display["Mean"].round(1)
#     display["Peak"] = display["Peak"].round(1)
#     st.dataframe(display, width=True, hide_index=True)

st.divider()
st.caption(
    "Thresholds: CO2 elevated >= 1,000 ppm | critical >= 2,000 ppm | "
    "LightGBM model | IAQ sensor data from Malaysian primary schools."
)

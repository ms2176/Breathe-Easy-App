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
    "11min": ("11 min",  pd.Timedelta(minutes=11)),
    "33min": ("33 min",  pd.Timedelta(minutes=33)),
    "1h":    ("1 hour",  pd.Timedelta(hours=1)),
    "3h":    ("3 hours", pd.Timedelta(hours=3)),
    "6h":    ("6 hours", pd.Timedelta(hours=6)),
}

SECONDARY = {
    "Measured PM2.5": ("PM2.5",       "µg/m³"),
    "Measured PM10":  ("PM10",        "µg/m³"),
    "Measured T":     ("Temperature", "°C"),
    "Measured RH":    ("Humidity",    "%"),
}

# ── ACTION ADVICE per horizon status ─────────────────────────────────────────
HORIZON_ADVICE = {
    "11min": {
        "safe":     "No action needed in the next 11 minutes.",
        "warn":     "Air quality will become elevated in 11 minutes — open a window now.",
        "critical": "Air quality will be critical in 11 minutes — open windows and doors immediately.",
    },
    "33min": {
        "safe":     "No action needed in the next 33 minutes.",
        "warn":     "Consider opening windows in the next 30 minutes.",
        "critical": "Open windows now — CO₂ is forecast to reach dangerous levels within 33 minutes.",
    },
    "1h": {
        "safe":     "Air quality expected to remain safe for the next hour.",
        "warn":     "Plan to ventilate within the next hour.",
        "critical": "Ventilation required within the next hour — consider rescheduling the class.",
    },
    "3h": {
        "safe":     "Safe conditions expected for the next 3 hours.",
        "warn":     "Air quality may deteriorate this session — monitor closely.",
        "critical": "Conditions forecast to become critical this session. Plan ventilation breaks.",
    },
    "6h": {
        "safe":     "Safe for the rest of the school day.",
        "warn":     "Air quality concerns expected later today — prepare ventilation plan.",
        "critical": "Poor air quality expected for most of today. Prioritise this classroom.",
    },
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
.hz-advice{font-size:.72rem;color:#374151;font-style:italic;margin-top:5px;line-height:1.4;}

.context-box {background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:12px 16px;margin-bottom:12px;color:#0c4a6e;font-size:.88rem;line-height:1.6;}
.infra-row {display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f3f4f6;}
.infra-name {flex:1;font-size:.9rem;font-weight:500;color:#111827;}
.infra-bar-wrap {flex:2;background:#f3f4f6;border-radius:6px;height:10px;overflow:hidden;}
.infra-bar {height:10px;border-radius:6px;}
.infra-stat {font-size:.82rem;color:#374151;min-width:80px;text-align:right;}
.infra-verdict {font-size:.78rem;font-weight:600;min-width:70px;text-align:right;}
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
    for name in ["test.parquet", "test_display.parquet"]:
        b = load_local_or_hf(name)
        if b:
            test = pd.read_parquet(b)
            future_cols = [c for c in test.columns if "_future_" in c]
            if future_cols:
                break
            test_fallback = test
            test = None
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
    return                 "safe",      "Good",    "#166534", "#ecfdf3", "#22c55e"

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
    if df.empty or "datetime" not in df.columns or df["datetime"].dropna().empty:
        return None
    idx = (df["datetime"] - ts).abs().idxmin()
    return df.loc[idx]

def banner(cls, html):
    st.markdown(f'<div class="banner-{cls}">{html}</div>', unsafe_allow_html=True)

def worst_horizon_status(results):
    """Return the worst status across all predicted horizons."""
    for r in results:
        if r["pred"] is not None and r["pred"] >= CO2_CRITICAL:
            return "critical"
    for r in results:
        if r["pred"] is not None and r["pred"] >= CO2_WARN:
            return "warn"
    return "safe"


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
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🌬️ BreatheEasy")
st.caption("Indoor air quality monitoring and AI-powered CO₂ forecasting for South African school administrators.")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – ROOM FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### Is my classroom safe right now — and in the next few hours?")

# ── Plain-language context for the administrator ──────────────────────────────
st.markdown(
    '<div class="context-box">'
    '🏫 <b>How to use this section:</b> Select a classroom below, then pick a date and time. '
    'The dashboard will show the current CO₂ level and predict whether air quality will become a '
    'problem in the next 11 minutes, 33 minutes, 1 hour, 3 hours, or 6 hours ahead. '
    'CO₂ above <b>1,000 ppm</b> causes reduced concentration. Above <b>2,000 ppm</b> requires immediate action. '
    'Use these forecasts to decide when to open windows or take ventilation breaks.'
    '</div>',
    unsafe_allow_html=True
)

f1, f2, f3, f4, f5 = st.columns([1.4, 0.8, 0.8, 0.9, 0.9])

infra_options = sorted(ref_df["Classroom Type"].dropna().unique())
if not infra_options:
    st.error("No infrastructure types are available in the dataset.")
    st.stop()

infra = f1.selectbox("Infrastructure type", infra_options)
sub_ref = ref_df[ref_df["Classroom Type"] == infra].copy()

school_options = sorted(sub_ref["School No"].dropna().astype(int).unique())
if not school_options:
    st.warning("No schools are available for the selected infrastructure type.")
    st.stop()

school = f2.selectbox("School", school_options)
sub_ref = sub_ref[sub_ref["School No"] == school].copy()

room_options = sorted(sub_ref["Room No"].dropna().astype(int).unique())
if not room_options:
    st.warning("No rooms are available for the selected school.")
    st.stop()

room = f3.selectbox("Room", room_options)

room_test = test_df[
    (test_df["Classroom Type"] == infra) &
    (test_df["School No"] == school) &
    (test_df["Room No"] == room)
].copy() if test_df is not None else pd.DataFrame()

room_ref = sub_ref[sub_ref["Room No"] == room].copy()
active_df = room_test if not room_test.empty else room_ref

if active_df.empty or "datetime" not in active_df.columns:
    st.warning(
        "No data is available for this classroom combination. "
        "Please choose a different infrastructure type, school, or room."
    )
    st.stop()

active_df = active_df.dropna(subset=["datetime"]).copy()

if active_df.empty:
    st.warning(
        "This classroom combination exists, but it has no valid timestamps. "
        "Please choose a different room."
    )
    st.stop()

min_d = active_df["datetime"].min().date()
max_d = active_df["datetime"].max().date()
default_d = max(min_d, min(max_d, pd.Timestamp("2023-11-06").date()))

sel_date = f4.date_input("Date", value=default_d, min_value=min_d, max_value=max_d)
sel_time = f5.time_input("Time", value=pd.Timestamp("07:45").time())

selected_ts = pd.Timestamp(f"{sel_date} {sel_time}")
cur_row = nearest(active_df, selected_ts)

if cur_row is None:
    st.warning("No valid reading is available for the selected date and time.")
    st.stop()

actual_ts = pd.to_datetime(cur_row["datetime"])

st.caption(
    f"Showing data for: **{infra}** · School {school} · Room {room} · "
    f"Nearest recorded reading to your selection: **{actual_ts.strftime('%d %b %Y, %H:%M')}**"
)

# ── Current snapshot ──────────────────────────────────────────────────────────
co2_now = float(cur_row["Measured CO2"]) if "Measured CO2" in cur_row.index and pd.notna(cur_row["Measured CO2"]) else np.nan

if not np.isnan(co2_now):
    cls, lbl, *_ = co2_status(co2_now)
    msgs = {
        "safe":     f"✅ <b>Air quality is good.</b> CO₂ is currently <b>{co2_now:.0f} ppm</b> — well within the safe limit of 1,000 ppm.",
        "warn":     f"⚠️ <b>Air quality is elevated.</b> CO₂ is <b>{co2_now:.0f} ppm</b> — above the 1,000 ppm threshold. Consider increasing ventilation.",
        "critical": f"🔴 <b>Immediate action required.</b> CO₂ is <b>{co2_now:.0f} ppm</b> — critically high. Open windows and doors now.",
    }
    banner(cls, msgs[cls])

st.markdown("**Current readings in this classroom:**")
m_cols = st.columns(5)
m_cols[0].metric("CO₂ (ppm)", f"{co2_now:.0f}" if not np.isnan(co2_now) else "—", help="Safe below 1,000 ppm. Above 2,000 ppm is critical.")
for col, (target, (label, unit)) in zip(m_cols[1:], SECONDARY.items()):
    v = float(cur_row[target]) if target in cur_row.index and pd.notna(cur_row[target]) else np.nan
    col.metric(f"{label} ({unit})", f"{v:.1f}" if not np.isnan(v) else "—")

st.divider()

# ── Forecast ──────────────────────────────────────────────────────────────────
st.markdown("### What will the air quality be like in the next few hours?")

# Run all predictions
results = []
for hz_key, (hz_label, hz_delta) in HORIZONS.items():
    future_ts  = actual_ts + hz_delta
    pred       = do_predict(cur_row, hz_key, scaler, feature_cols)
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

# Overall risk summary — plain language for administrator
worst_cls = worst_horizon_status(results)
risk_msgs = {
    "safe":     "✅ <b>Good news.</b> Air quality is forecast to remain safe for the next 6 hours. No ventilation action is needed right now.",
    "warn":     "⚠️ <b>Heads up.</b> CO₂ levels are forecast to become elevated in the coming hours. See the timeline below to plan ventilation.",
    "critical": "🔴 <b>Action required.</b> CO₂ is forecast to reach critical levels. Open windows now and check the timeline below for when conditions are worst.",
}
banner(worst_cls, risk_msgs[worst_cls])

# Chart
history = active_df[active_df["datetime"] <= actual_ts].tail(20)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history["datetime"], y=history["Measured CO2"],
    mode="lines+markers", name="Past CO₂ readings",
    line=dict(width=3, color="#335CFF"), marker=dict(size=5),
))
fig.add_trace(go.Scatter(
    x=[actual_ts], y=[co2_now],
    mode="markers", name="Now",
    marker=dict(size=14, color="#111827", symbol="circle"),
))

valid_pred = [(r["future_ts"], r["pred"]) for r in results if r["pred"] is not None]
if valid_pred:
    fig.add_trace(go.Scatter(
        x=[actual_ts] + [t for t, _ in valid_pred],
        y=[co2_now]   + [v for _, v in valid_pred],
        mode="lines+markers", name="AI forecast",
        line=dict(width=3, color="#7C3AED", dash="dot"),
        marker=dict(size=10, symbol="diamond"),
    ))

valid_actual = [(r["future_ts"], r["actual"]) for r in results if not np.isnan(r["actual"])]
if valid_actual:
    fig.add_trace(go.Scatter(
        x=[actual_ts] + [t for t, _ in valid_actual],
        y=[co2_now]   + [v for _, v in valid_actual],
        mode="lines+markers", name="What actually happened",
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
    title="CO₂ levels: past readings, AI forecast, and what actually happened (for validation)",
    height=420, margin=dict(l=8, r=8, t=48, b=8),
    yaxis_title="CO₂ (ppm)", xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(gridcolor="#f3f4f6"), xaxis=dict(gridcolor="#f3f4f6"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Horizon cards ─────────────────────────────────────────────────────────────
st.markdown("**Should I act now? — Forecast at each time horizon:**")
hz_cols = st.columns(5)

for col, r in zip(hz_cols, results):
    pred   = r["pred"]
    actual = r["actual"]

    if pred is not None:
        p_cls, p_lbl, p_txt, p_bg, p_bdr = co2_status(pred)
        pred_str = f"{pred:.0f} ppm"
        advice   = HORIZON_ADVICE[r["hz_key"]][p_cls]
    else:
        p_cls, p_lbl, p_txt, p_bg, p_bdr = "safe", "—", "#374151", "#f9fafb", "#e5e7eb"
        pred_str = "—"
        advice   = ""

    if not np.isnan(actual):
        a_cls, a_lbl, a_txt, *_ = co2_status(actual)
        icon = "✅" if a_cls == "safe" else "⚠️" if a_cls == "warn" else "🔴"
        err_html = (
            f'<span class="hz-err">Model was off by {abs(pred - actual):.0f} ppm</span>'
            if pred is not None else ""
        )
        actual_html = (
            f'<div class="hz-actual">'
            f'Recorded: <b style="color:{a_txt};">{actual:.0f} ppm</b> {icon}'
            f'<br>{err_html}</div>'
        )
    else:
        actual_html = ""

    pred_icon = "✅" if p_cls == "safe" else "⚠️" if p_cls == "warn" else "🔴"

    col.markdown(f"""
    <div class="hz-card" style="background:{p_bg};border-color:{p_bdr};">
      <div class="hz-label">In {r["label"]}</div>
      <div class="hz-pred" style="color:{p_txt};">{pred_str}</div>
      <div class="hz-status" style="color:{p_txt};">{pred_icon} {p_lbl}</div>
      <div class="hz-advice">{advice}</div>
      {actual_html}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – INFRASTRUCTURE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Which type of classroom has the worst air quality?")

st.markdown(
    '<div class="context-box">'
    '📊 <b>How to use this section:</b> This compares air quality across all six classroom types in the dataset. '
    'The <b>warning rate</b> shows what percentage of the time CO₂ was above the safe limit of 1,000 ppm. '
    'A higher warning rate means students in that classroom type are more frequently exposed to poor air quality. '
    'Use this to prioritise which building types need ventilation improvements most urgently.'
    '</div>',
    unsafe_allow_html=True
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

# Plain-language verdict using actual numbers
banner(
    w_cls,
    f"🏫 <b>{worst['Type']}</b> needs the most attention: CO₂ exceeded the safe limit "
    f"<b>{worst['Warning rate (%)']:.1f}% of the time</b> "
    f"(average {worst['Mean']:.0f} ppm, peak {worst['Peak']:.0f} ppm). "
    f"<b>{best['Type']}</b> performed best, with unsafe CO₂ in only {best['Warning rate (%)']:.1f}% of readings."
)

left, right = st.columns(2)

with left:
    # Add a plain verdict column
    def infra_verdict(warn_rate):
        if warn_rate > 30:
            return "🔴 High concern"
        elif warn_rate > 10:
            return "⚠️ Moderate concern"
        elif warn_rate > 5:
            return "🟡 Low concern"
        return "✅ Generally safe"

    bar_colors = [
        "#DC2626" if v > 30 else "#F59E0B" if v > 10 else "#84cc16" if v > 5 else "#22c55e"
        for v in infra_stats["Warning rate (%)"]
    ]

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
            [infra_verdict(v) for v in infra_stats["Warning rate (%)"]],
        ], axis=-1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Unsafe air quality: %{y:.1f}% of the time<br>"
            "Average CO₂: %{customdata[0]:.0f} ppm<br>"
            "Peak CO₂: %{customdata[1]:.0f} ppm<br>"
            "Critically high: %{customdata[2]:.1f}% of the time<br>"
            "Total readings: %{customdata[3]:,}<br>"
            "Assessment: %{customdata[4]}<extra></extra>"
        ),
    ))
    fig2.update_layout(
        title="How often was CO₂ above the safe limit? (by classroom type)",
        height=380, margin=dict(l=8, r=8, t=48, b=8),
        yaxis_title="% of time CO₂ exceeded 1,000 ppm",
        xaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#f3f4f6"),
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=infra_stats["Type"], y=infra_stats["Mean"],
        name="Average CO₂", marker_color="#335CFF", opacity=0.85,
        text=[f"{v:.0f}" for v in infra_stats["Mean"]], textposition="outside",
    ))
    fig3.add_trace(go.Bar(
        x=infra_stats["Type"], y=infra_stats["Peak"],
        name="Highest recorded CO₂", marker_color="#DC2626", opacity=0.6,
        text=[f"{v:.0f}" for v in infra_stats["Peak"]], textposition="outside",
    ))
    fig3.add_hline(y=CO2_WARN, line_dash="dash", line_color="#F59E0B",
                   annotation_text="Safe limit: 1,000 ppm", annotation_position="top right")
    fig3.update_layout(
        title="Average and highest CO₂ recorded (by classroom type)",
        height=380, barmode="group", margin=dict(l=8, r=8, t=48, b=8),
        yaxis_title="CO₂ (ppm)",
        xaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#f3f4f6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Summary verdict table ──────────────────────────────────────────────────────
st.markdown("**Summary: Air quality assessment by classroom type**")
verdict_cols = st.columns(len(infra_stats))
for col, (_, row) in zip(verdict_cols, infra_stats.iterrows()):
    v = infra_verdict(row["Warning rate (%)"])
    cls_bg = (
        "#fef2f2" if row["Warning rate (%)"] > 30
        else "#fff7ed" if row["Warning rate (%)"] > 10
        else "#f0fdf4"
    )
    cls_txt = (
        "#991b1b" if row["Warning rate (%)"] > 30
        else "#9a6700" if row["Warning rate (%)"] > 10
        else "#166534"
    )
    col.markdown(
        f"<div style='background:{cls_bg};border-radius:10px;padding:10px 8px;text-align:center;'>"
        f"<div style='font-size:.75rem;font-weight:600;color:#6b7280;margin-bottom:4px;'>{row['Type']}</div>"
        f"<div style='font-size:1.1rem;font-weight:700;color:{cls_txt};'>{row['Warning rate (%)']:.1f}%</div>"
        f"<div style='font-size:.72rem;color:{cls_txt};margin-top:3px;'>{v}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

st.divider()
st.caption(
    "CO₂ safe limit: 1,000 ppm (WHO guideline) · Critical: 2,000 ppm · "
    "Forecasts powered by LightGBM trained on 372,084 sensor readings from South African primary schools."
)
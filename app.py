import os, json, pickle, warnings, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import requests

warnings.filterwarnings("ignore")

# ─── HUGGINGFACE CONFIG ───────────────────────────────────────────────────────
# Your repo is public — no token needed.
# All files live at the ROOT of the repo (no subfolders).
HF_REPO  = "mufliha/iaq-prediction"
HF_BASE  = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
HF_CACHE = ".hf_cache"   # downloaded files are cached here so we don't re-download

# Every .pkl file in your repo
HF_MODEL_FILES = [
    "LightGBM_Measured_CO2_11min.pkl",
    "LightGBM_Measured_CO2_33min.pkl",
    "LightGBM_Measured_CO2_1h.pkl",
    "LightGBM_Measured_CO2_3h.pkl",
    "LightGBM_Measured_CO2_6h.pkl",
    "LightGBM_Measured_PM1.0_11min.pkl",
    "LightGBM_Measured_PM1.0_33min.pkl",
    "LightGBM_Measured_PM1.0_1h.pkl",
    "LightGBM_Measured_PM1.0_3h.pkl",
    "LightGBM_Measured_PM1.0_6h.pkl",
    "LightGBM_Measured_PM2.5_11min.pkl",
    "LightGBM_Measured_PM2.5_33min.pkl",
    "LightGBM_Measured_PM2.5_1h.pkl",
    "LightGBM_Measured_PM2.5_3h.pkl",
    "LightGBM_Measured_PM2.5_6h.pkl",
    "LightGBM_Measured_PM10_11min.pkl",
    "LightGBM_Measured_PM10_1h.pkl",
    "LightGBM_Measured_PM10_3h.pkl",
    "LightGBM_Measured_PM10_6h.pkl",
]

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreatheEasy",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── STYLING ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.alert-ok       { background:#d4edda; border-left:5px solid #28a745; padding:14px 18px; border-radius:6px; color:#155724; margin:8px 0; }
.alert-warn     { background:#fff3cd; border-left:5px solid #ffc107; padding:14px 18px; border-radius:6px; color:#856404; margin:8px 0; }
.alert-critical { background:#f8d7da; border-left:5px solid #dc3545; padding:14px 18px; border-radius:6px; color:#721c24; margin:8px 0; }
.task-header    { background:#f0f4ff; border-radius:10px; padding:12px 18px; margin-bottom:12px; border-left:4px solid #4361ee; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "Measured CO2":   {"unit": "ppm",   "warn": 1000, "danger": 1500, "label": "CO₂",        "color": "#4361ee"},
    "Measured PM1.0": {"unit": "μg/m³", "warn": 10,   "danger": 25,   "label": "PM₁.₀",      "color": "#e76f51"},
    "Measured PM2.5": {"unit": "μg/m³", "warn": 15,   "danger": 35,   "label": "PM₂.₅",      "color": "#f72585"},
    "Measured PM10":  {"unit": "μg/m³", "warn": 45,   "danger": 100,  "label": "PM₁₀",       "color": "#7209b7"},
    "Measured T":     {"unit": "°C",    "warn": 28,   "danger": 35,   "label": "Temperature","color": "#f4a261"},
    "Measured RH":    {"unit": "%",     "warn": 70,   "danger": 85,   "label": "Humidity",   "color": "#2a9d8f"},
}

# Pollutants that have forecast models
POLLUTANTS = ["Measured CO2", "Measured PM1.0", "Measured PM2.5", "Measured PM10"]

# All horizons — short-term (11min, 33min) AND long-term (1h, 3h, 6h)
HORIZON_KEYS  = ["11min", "33min", "1h", "3h", "6h"]
HORIZON_LABEL = {
    "11min": "11 minutes  (short)",
    "33min": "33 minutes  (short)",
    "1h":    "1 hour      (medium)",
    "3h":    "3 hours     (long)",
    "6h":    "6 hours     (long)",
}
HORIZON_FACTOR = {"11min": 1.02, "33min": 1.04, "1h": 1.08, "3h": 1.13, "6h": 1.10}

INFRA_TIPS = {
    "Container No Insulation":   "Container rooms lose ventilation benefit quickly — act earlier.",
    "Container With Insulation": "Insulated containers trap CO₂; open both top vents if available.",
    "Mobile/Prefab":             "Mobile classrooms have high humidity risk — ensure cross-ventilation.",
    "Brick First Floor":         "Ground-level rooms benefit from door ventilation.",
    "Brick Second Floor":        "Open windows on opposite walls for cross-ventilation.",
    "Brick Single Story":        "Roof ventilation is most effective if available.",
}

STATUS_ICON  = {"ok": "🟢", "warn": "🟡", "critical": "🔴"}

# ─── FILE HELPERS ─────────────────────────────────────────────────────────────
os.makedirs(HF_CACHE, exist_ok=True)

def _fetch_bytes(filename: str):
    """Return file as BytesIO — from local cache if available, else download."""
    cache_path = os.path.join(HF_CACHE, filename)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return io.BytesIO(f.read())
    try:
        url = f"{HF_BASE}/{filename}"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(r.content)
        return io.BytesIO(r.content)
    except Exception:
        return None

def _fetch_json(filename: str):
    b = _fetch_bytes(filename)
    return json.loads(b.read().decode()) if b else None

# ─── DATA & MODEL LOADING ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading data from HuggingFace…")
def load_data():
    try:
        b_test = _fetch_bytes("test_display.parquet")
        b_full = _fetch_bytes("full_featured.parquet")
        if b_test is None or b_full is None:
            return None, None, None, None
        df_test   = pd.read_parquet(b_test)
        df_full   = pd.read_parquet(b_full)
        scaler_p  = _fetch_json("scaler_params.json")
        feat_meta = _fetch_json("feature_list.json")
        for df in [df_test, df_full]:
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
        return df_test, df_full, scaler_p, feat_meta
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return None, None, None, None

@st.cache_resource(show_spinner="🤖 Loading forecast models…")
def load_models():
    models = {}
    for fname in HF_MODEL_FILES:
        key = fname.replace(".pkl", "")
        b = _fetch_bytes(fname)
        if b:
            try:
                models[key] = pickle.load(b)
            except Exception:
                pass
    return models

def build_scaler(params):
    s = MinMaxScaler()
    s.data_min_      = np.array(params["data_min_"])
    s.data_max_      = np.array(params["data_max_"])
    s.scale_         = np.where(s.data_max_ - s.data_min_ != 0,
                                1.0 / (s.data_max_ - s.data_min_), 0.0)
    s.data_range_    = s.data_max_ - s.data_min_
    s.feature_range  = (0, 1)
    s.n_features_in_ = len(s.data_min_)
    s.min_           = -s.scale_ * s.data_min_
    return s

# ─── PREDICTION ───────────────────────────────────────────────────────────────
def predict(models, scaler, feature_cols, row, target, horizon):
    """
    Model key format in your pkl files: LightGBM_Measured_CO2_1h
    Dots in target names (PM1.0, PM2.5) are kept as-is in filenames,
    but we need to match exactly — so we do NOT strip dots here.
    """
    safe = target.replace(" ", "_")          # e.g. "Measured_CO2", "Measured_PM1.0"
    mkey = f"LightGBM_{safe}_{horizon}"      # e.g. "LightGBM_Measured_CO2_1h"
    if mkey not in models:
        return None, None
    try:
        vals = row[feature_cols].values.astype(float).reshape(1, -1)
        x = scaler.transform(vals)
        return float(models[mkey].predict(x)[0]), "LightGBM"
    except Exception:
        return None, None

# ─── STATUS ───────────────────────────────────────────────────────────────────
def get_status(val, target):
    t = THRESHOLDS.get(target, {})
    if val >= t.get("danger", 9e9): return "critical"
    if val >= t.get("warn",   9e9): return "warn"
    return "ok"

# ─── DEMO DATA ────────────────────────────────────────────────────────────────
def demo_snapshot(seed=0):
    np.random.seed(seed % 9999)
    return {
        "Measured CO2":   round(800  + np.random.normal(0, 80)),
        "Measured PM1.0": round(7    + np.random.normal(0, 2),  1),
        "Measured PM2.5": round(10   + np.random.normal(0, 3),  1),
        "Measured PM10":  round(18   + np.random.normal(0, 5),  1),
        "Measured T":     round(22   + np.random.normal(0, 2),  1),
        "Measured RH":    round(60   + np.random.normal(0, 8),  1),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("## 🌬️ BreatheEasy — IAQ Monitor")
    st.caption("School Indoor Air Quality · Facility Management Dashboard")

    df_test, df_full, scaler_params, feat_meta = load_data()
    models = load_models()
    using_real   = df_test is not None
    scaler       = build_scaler(scaler_params) if (using_real and scaler_params) else None
    feature_cols = feat_meta.get("feature_cols", []) if feat_meta else []

    if not using_real:
        st.info(
            "📂 **Demo mode** — could not load from HuggingFace. "
            "Check that `mufliha/iaq-prediction` is public and your connection is working. "
            "Showing simulated values.",
            icon="ℹ️"
        )
    else:
        st.success(f"✅ Data loaded · {len(models)} forecast models ready")

    tab1, tab2 = st.tabs(["📍 Check IAQ Levels", "📊 Classroom Overview"])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        

        # Filters
        if using_real:
            infra_opts = sorted(df_test["Classroom Type"].dropna().unique())
        else:
            infra_opts = ["Brick First Floor", "Brick Second Floor", "Brick Single Story",
                          "Container No Insulation", "Container With Insulation", "Mobile/Prefab"]

        fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
        with fc1:
            infra = st.selectbox("Infrastructure type", infra_opts, key="t1_infra")

        if using_real:
            school_opts = sorted(df_test[df_test["Classroom Type"]==infra]["School No"].dropna().unique().astype(int))
        else:
            school_opts = [1, 2, 3]
        with fc2:
            school = st.selectbox("School", school_opts, key="t1_school")

        if using_real:
            room_opts = sorted(df_test[
                (df_test["Classroom Type"]==infra) &
                (df_test["School No"]==school)
            ]["Room No"].dropna().unique().astype(int))
        else:
            room_opts = [1, 2, 3]
        with fc3:
            room = st.selectbox("Room", room_opts, key="t1_room")

        dev_col = None
        if using_real:
            dev_col = next((c for c in df_test.columns if "device" in c.lower()), None)
        if dev_col:
            dev_opts = sorted(df_test[
                (df_test["Classroom Type"]==infra) &
                (df_test["School No"]==school) &
                (df_test["Room No"]==room)
            ][dev_col].dropna().unique().astype(int))
        else:
            dev_opts = [22, 23, 24]
        with fc4:
            device = st.selectbox("Device", dev_opts, key="t1_device")

        # Date / time
        dc1, dc2 = st.columns(2)
        if using_real:
            mask = (
                (df_test["Classroom Type"]==infra) &
                (df_test["School No"]==school) &
                (df_test["Room No"]==room)
            )
            if dev_col:
                mask &= df_test[dev_col]==device
            sub  = df_test[mask].sort_values("datetime")
            dmin = sub["datetime"].min().date()
            dmax = sub["datetime"].max().date()
            default_date = pd.Timestamp("2023-11-06").date()
            default_date = max(dmin, min(dmax, default_date))
        else:
            dmin, dmax = pd.Timestamp("2023-10-01").date(), pd.Timestamp("2023-11-30").date()
            default_date = pd.Timestamp("2023-11-06").date()

        with dc1:
            sel_date = st.date_input("Date", value=default_date, min_value=dmin, max_value=dmax, key="t1_date")
        with dc2:
            sel_time = st.time_input("Time", value=pd.Timestamp("07:45").time(), key="t1_time")

        sel_ts = pd.Timestamp(f"{sel_date} {sel_time}")
        st.divider()

        # Resolve row
        if using_real:
            full_mask = (
                (df_full["Classroom Type"]==infra) &
                (df_full["School No"]==school) &
                (df_full["Room No"]==room)
            )
            if dev_col:
                full_mask &= df_full[dev_col]==device
            room_df = df_full[full_mask].sort_values("datetime").reset_index(drop=True)
            if room_df.empty:
                st.error("No data for this selection. Try a different school / room / device.")
                return
            cur_idx    = (room_df["datetime"] - sel_ts).abs().idxmin()
            cur_row    = room_df.loc[cur_idx]
            history_df = room_df[room_df["datetime"] <= sel_ts].tail(12).copy()
        else:
            seed = abs(hash(f"{infra}{school}{room}")) % 9999
            snap = demo_snapshot(seed)
            cur_row = pd.Series(snap)
            times = [sel_ts - pd.Timedelta(minutes=11*(11-i)) for i in range(12)]
            history_df = pd.DataFrame({"datetime": times})
            for k, v in snap.items():
                np.random.seed(seed + abs(hash(k)) % 999)
                history_df[k] = [round(v * (0.85 + np.random.random()*0.3), 1) for _ in range(12)]

        # Current conditions
        st.markdown(f"#### 📍 {infra} · School {school} · Room {room} · Device {device}")
        st.caption(f"Snapshot: {sel_ts.strftime('%A %d %b %Y, %H:%M')}")

        metric_cols = st.columns(6)
        for i, tgt in enumerate(list(THRESHOLDS.keys())):
            val = float(cur_row.get(tgt, np.nan))
            cfg = THRESHOLDS[tgt]
            if np.isnan(val):
                metric_cols[i].metric(cfg["label"], "N/A")
                continue
            s = get_status(val, tgt)
            metric_cols[i].metric(
                label=f"{STATUS_ICON[s]} {cfg['label']}",
                value=f"{val:.1f} {cfg['unit']}",
            )

        # Overall banner
        poll_vals = [(t, float(cur_row.get(t, np.nan))) for t in POLLUTANTS]
        poll_vals = [(t, v) for t, v in poll_vals if not np.isnan(v)]
        if poll_vals:
            overall = max((get_status(v, t) for t, v in poll_vals),
                          key=lambda x: {"ok":0,"warn":1,"critical":2}[x])
            msgs = {
                "ok":       "✅ <b>Air quality is Good.</b> All pollutants within safe limits.",
                "warn":     "⚠️ <b>Elevated pollution detected.</b> Consider opening windows.",
                "critical": "🔴 <b>Critical air quality!</b> Immediate ventilation required — consider relocating students.",
            }
            st.markdown(f'<div class="alert-{overall}">{msgs[overall]}</div>', unsafe_allow_html=True)

        st.divider()

        # History chart + forecast table
        left, right = st.columns([3, 2])

        with left:
            poll_choice = st.selectbox(
                "Pollutant to plot", POLLUTANTS,
                format_func=lambda x: THRESHOLDS[x]["label"], key="t1_poll"
            )
            cfg = THRESHOLDS[poll_choice]
            hist_times = history_df["datetime"].tolist() if "datetime" in history_df else list(range(len(history_df)))
            hist_vals  = history_df[poll_choice].tolist() if poll_choice in history_df.columns else []

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_times, y=hist_vals,
                mode="lines+markers",
                line=dict(color=cfg["color"], width=2.5),
                marker=dict(size=6),
            ))
            fig.add_hline(y=cfg["warn"],   line_dash="dash", line_color="#ffc107",
                          annotation_text=f"⚠️ Warn {cfg['warn']} {cfg['unit']}", annotation_position="top right")
            fig.add_hline(y=cfg["danger"], line_dash="dash", line_color="#dc3545",
                          annotation_text=f"🔴 Critical {cfg['danger']} {cfg['unit']}", annotation_position="top right")
            fig.update_layout(
                title=f"{cfg['label']} — Last ~2 hours (11-min intervals)",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                yaxis_title=cfg["unit"], showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### 🔮 Forecast")
            cur_val = float(cur_row.get(poll_choice, np.nan))
            fc_rows = []

            for hz in HORIZON_KEYS:
                if using_real and scaler and feature_cols:
                    pval, mname = predict(models, scaler, feature_cols, cur_row, poll_choice, hz)
                else:
                    np.random.seed(abs(hash(hz)) % 999)
                    pval  = round(cur_val * HORIZON_FACTOR[hz] + np.random.normal(0, 15), 1)
                    mname = "Demo"

                label = HORIZON_LABEL[hz]
                if pval is None:
                    fc_rows.append({"Horizon": label, f"{cfg['label']} ({cfg['unit']})": "—", "Δ": "—", "Status": "—"})
                    continue

                delta = round(pval - cur_val, 1)
                s = get_status(pval, poll_choice)
                fc_rows.append({
                    "Horizon":                          label,
                    f"{cfg['label']} ({cfg['unit']})": round(pval, 1),
                    "Δ":                                f"+{delta}" if delta >= 0 else str(delta),
                    "Status":                           STATUS_ICON[s],
                })

            st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)

            # Alert for worst forecast
            numeric_preds = [r[f"{cfg['label']} ({cfg['unit']})"] for r in fc_rows
                             if isinstance(r[f"{cfg['label']} ({cfg['unit']})"], (int, float))]
            if numeric_preds:
                worst = max(numeric_preds)
                ws    = get_status(worst, poll_choice)
                tip   = INFRA_TIPS.get(infra, "")
                if ws == "critical":
                    st.markdown(f'<div class="alert-critical">🔴 <b>Critical forecast!</b> {cfg["label"]} may reach <b>{worst} {cfg["unit"]}</b>. Open all windows and doors. {tip}</div>', unsafe_allow_html=True)
                elif ws == "warn":
                    st.markdown(f'<div class="alert-warn">⚠️ <b>Elevated forecast.</b> {cfg["label"]} forecast to reach <b>{worst} {cfg["unit"]}</b>. Pre-ventilate now. {tip}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-ok">✅ <b>Forecast looks good.</b> {cfg["label"]} expected to stay within safe levels. {tip}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2
    # ═══════════════════════════════════════════════════════════════════
    with tab2:
        

        compare_poll = st.selectbox(
            "Pollutant to compare", POLLUTANTS,
            format_func=lambda x: THRESHOLDS[x]["label"], key="t2_poll"
        )
        cfg2 = THRESHOLDS[compare_poll]

        if using_real:
            grp = (
                df_test.groupby("Classroom Type")[compare_poll]
                .apply(lambda x: (x >= cfg2["warn"]).mean() * 100)
                .reset_index().rename(columns={compare_poll: "Warning Rate (%)"})
            )
            crit = (
                df_test.groupby("Classroom Type")[compare_poll]
                .apply(lambda x: (x >= cfg2["danger"]).mean() * 100)
                .reset_index().rename(columns={compare_poll: "Critical Rate (%)"})
            )
            means = (
                df_test.groupby("Classroom Type")[compare_poll].mean()
                .reset_index().rename(columns={compare_poll: f"Mean {cfg2['label']}"})
            )
            grp = grp.merge(crit, on="Classroom Type").merge(means, on="Classroom Type")
            grp = grp.sort_values("Warning Rate (%)", ascending=True)
        else:
            names = ["Brick First Floor", "Brick Second Floor", "Brick Single Story",
                     "Container No Insulation", "Container With Insulation", "Mobile/Prefab"]
            np.random.seed(42)
            grp = pd.DataFrame({
                "Classroom Type":       names,
                "Warning Rate (%)":     np.random.uniform(5, 65, len(names)).round(1),
                "Critical Rate (%)":    np.random.uniform(1, 20, len(names)).round(1),
                f"Mean {cfg2['label']}": np.random.uniform(600, 1400, len(names)).round(1),
            }).sort_values("Warning Rate (%)", ascending=True)

        # Bar chart
        st.markdown(f"#### ⚠️ {cfg2['label']} Warning Rate by Classroom Type")
        bar_colors = [
            "#28a745" if v < 20 else "#ffc107" if v < 50 else "#dc3545"
            for v in grp["Warning Rate (%)"]
        ]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=grp["Classroom Type"], x=grp["Warning Rate (%)"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in grp["Warning Rate (%)"]],
            textposition="outside",
        ))
        fig2.update_layout(
            height=340, margin=dict(l=10, r=80, t=20, b=20),
            xaxis_title=f"% of readings ≥ warn threshold ({cfg2['warn']} {cfg2['unit']})",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        worst_row = grp.sort_values("Warning Rate (%)", ascending=False).iloc[0]
        ws_pct    = worst_row["Warning Rate (%)"]
        ws_cls    = "alert-critical" if ws_pct >= 50 else "alert-warn"
        st.markdown(
            f'<div class="{ws_cls}">📊 <b>{worst_row["Classroom Type"]}</b> has the highest '
            f'{cfg2["label"]} warning rate: <b>{ws_pct:.1f}%</b> of readings above '
            f'{cfg2["warn"]} {cfg2["unit"]}.</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # Summary table
        st.markdown("#### 📋 Summary Table")
        disp = grp.sort_values("Warning Rate (%)", ascending=False).copy()
        mean_col = f"Mean {cfg2['label']}"
        disp[mean_col]             = disp[mean_col].round(1).astype(str) + f" {cfg2['unit']}"
        disp["Warning Rate (%)"]   = disp["Warning Rate (%)"].round(1).astype(str) + "%"
        disp["Critical Rate (%)"]  = disp["Critical Rate (%)"].round(1).astype(str) + "%"
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Box plot (real data only)
        if using_real:
            st.divider()
            st.markdown(f"#### 📦 {cfg2['label']} Distribution by Classroom Type")
            fig3 = go.Figure()
            for infra_name in sorted(df_test["Classroom Type"].dropna().unique()):
                vals = df_test[df_test["Classroom Type"]==infra_name][compare_poll].dropna()
                fig3.add_trace(go.Box(y=vals, name=infra_name, boxpoints="outliers",
                                      marker_color=cfg2["color"]))
            fig3.add_hline(y=cfg2["warn"],   line_dash="dash", line_color="#ffc107",
                           annotation_text=f"Warn {cfg2['warn']}", annotation_position="top right")
            fig3.add_hline(y=cfg2["danger"], line_dash="dash", line_color="#dc3545",
                           annotation_text=f"Critical {cfg2['danger']}", annotation_position="top right")
            fig3.update_layout(
                height=380, margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title=cfg2["unit"], showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # Footer
    st.divider()
    mode = f"Live inference · {len(models)} models" if using_real else "Demo mode"
    st.caption(f"🌬️ BreatheEasy · {mode} · Source: {HF_REPO} · Thresholds: WHO / ASHRAE")


if __name__ == "__main__":
    main()

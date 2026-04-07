"""
BreatheEasy — IAQ Forecast Dashboard
Streamlit app for school facilities managers.

SETUP (run once):
    pip install streamlit pandas numpy plotly lightgbm xgboost scikit-learn pyarrow

RUN:
    streamlit run breatheeasy_app.py

REQUIRED FILES (copy from Colab outputs):
    processed/test_display.parquet     <- test set raw values
    processed/full_featured.parquet    <- full dataset with features
    processed/scaler_params.json       <- MinMax scaler parameters
    processed/feature_list.json        <- feature column names
    models/LightGBM_Measured_CO2_1h.pkl
    models/LightGBM_Measured_CO2_3h.pkl
    models/LightGBM_Measured_CO2_6h.pkl
    models/LightGBM_Measured_PM2.5_1h.pkl
    models/LightGBM_Measured_PM2.5_3h.pkl
    models/LightGBM_Measured_PM10_1h.pkl
    (LSTM .keras files optional — falls back to LightGBM if not found)
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreatheEasy — IAQ Monitor",
    page_icon="🌬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
  .stMetric { background: var(--background-secondary-color); border-radius: 8px; padding: 0.75rem 1rem; }
  .alert-critical { background: #FCEBEB; border-left: 4px solid #E24B4A; padding: 0.75rem 1rem; border-radius: 4px; color: #A32D2D; }
  .alert-warn     { background: #FAEEDA; border-left: 4px solid #BA7517; padding: 0.75rem 1rem; border-radius: 4px; color: #854F0B; }
  .alert-ok       { background: #EAF3DE; border-left: 4px solid #3B6D11; padding: 0.75rem 1rem; border-radius: 4px; color: #3B6D11; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
THRESHOLDS = {
    'Measured CO2':   {'unit': 'ppm',    'warn': 1000, 'danger': 1500, 'label': 'CO₂'},
    'Measured PM2.5': {'unit': 'μg/m³',  'warn': 15,   'danger': 35,   'label': 'PM₂.₅'},
    'Measured PM10':  {'unit': 'μg/m³',  'warn': 45,   'danger': 100,  'label': 'PM₁₀'},
    'Measured T':     {'unit': '°C',     'warn': 28,   'danger': 35,   'label': 'Temperature'},
    'Measured RH':    {'unit': '%',      'warn': 70,   'danger': 85,   'label': 'Humidity'},
}

INFRA_ORDER = [
    'Brick First Floor', 'Brick Second Floor', 'Brick Single Story',
    'Container No Insulation', 'Container With Insulation', 'Mobile/Prefab'
]

HORIZONS = {'1h': 6, '3h': 16, '6h': 33}
HORIZON_LABELS = {'11min': '11 min', '33min': '33 min', '1h': '1 hour', '3h': '3 hours', '6h': '6 hours'}

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df_test = pd.read_parquet('iaq_pipeline_outputs/processed/test_display.parquet')
        df_full = pd.read_parquet('iaq_pipeline_outputs/processed/full_featured.parquet')
        with open('iaq_pipeline_outputs/processed/scaler_params.json') as f:
            scaler_params = json.load(f)
        with open('iaq_pipeline_outputs/processed/feature_list.json') as f:
            feature_meta = json.load(f)
        return df_test, df_full, scaler_params, feature_meta
    except FileNotFoundError as e:
        return None, None, None, None

@st.cache_resources
def load_models():
    models = {}
    model_dir = 'iaq_pipeline_outputs/models'
    if not os.path.exists(model_dir):
        return models
    for fname in os.listdir(model_dir):
        if fname.endswith('.pkl'):
            key = fname.replace('.pkl','')
            try:
                with open(os.path.join(model_dir, fname), 'rb') as f:
                    models[key] = pickle.load(f)
            except Exception:
                pass
    # Try loading LSTM models
    try:
        import tensorflow as tf
        for fname in os.listdir(model_dir):
            if fname.endswith('.keras'):
                key = fname.replace('.keras','').replace('lstm_','LSTM_')
                try:
                    models[key] = tf.keras.models.load_model(os.path.join(model_dir, fname))
                except Exception:
                    pass
    except ImportError:
        pass
    return models

def build_scaler(scaler_params):
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array(scaler_params['data_min_'])
    scaler.data_max_ = np.array(scaler_params['data_max_'])
    scaler.scale_ = np.where(
        scaler.data_max_ - scaler.data_min_ != 0,
        1.0 / (scaler.data_max_ - scaler.data_min_),
        0.0
    )
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.feature_range = (0, 1)
    scaler.n_features_in_ = len(scaler.data_min_)
    scaler.min_ = -scaler.scale_ * scaler.data_min_
    return scaler

# ── PREDICTION HELPERS ────────────────────────────────────────────────────────
def get_prediction(models, scaler, feature_cols, row_features, target, horizon):
    """Get forecast for a single row. Falls back gracefully if model not found."""
    target_safe = target.replace(' ', '_').replace('.', '')
    
    # Try LSTM for longer horizons
    if horizon in ['3h', '6h']:
        lstm_key = f'LSTM_{target_safe}_{horizon}'
        if lstm_key in models:
            try:
                import tensorflow as tf
                x_scaled = scaler.transform(row_features.reshape(1, -1))
                # For LSTM we'd need a sequence — for single-row inference, use the row as the sequence
                x_seq = np.repeat(x_scaled, 6, axis=0).reshape(1, 6, -1).astype(np.float32)
                pred = models[lstm_key].predict(x_seq, verbose=0).flatten()[0]
                return float(pred), 'LSTM'
            except Exception:
                pass
    
    # Tree model fallback
    for mname in ['LightGBM', 'XGBoost', 'RandomForest']:
        mkey = f'{mname}_{target_safe}_{horizon}'
        if mkey in models:
            try:
                x_scaled = scaler.transform(row_features.reshape(1, -1))
                pred = models[mkey].predict(x_scaled)[0]
                return float(pred), mname
            except Exception:
                pass
    
    return None, None

def get_alert_level(value, target):
    t = THRESHOLDS.get(target, {})
    if value >= t.get('danger', 9999):
        return 'critical'
    elif value >= t.get('warn', 9999):
        return 'warn'
    return 'ok'

# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("## 🌬 BreatheEasy")
    st.markdown("*Indoor air quality monitoring and forecast dashboard*")
    st.divider()

    # Load data
    df_test, df_full, scaler_params, feature_meta = load_data()
    models = load_models()

    using_real_data = df_test is not None

    if not using_real_data:
        st.warning("Model files not found — running in **demo mode** with simulated data. "
                   "Copy `processed/` and `models/` from your Colab outputs to this directory.")

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Classroom selection")

        if using_real_data:
            infra_types = sorted(df_test['Classroom Type'].dropna().unique())
        else:
            infra_types = INFRA_ORDER

        infra = st.selectbox("Infrastructure type", infra_types)

        if using_real_data:
            schools = sorted(df_test[df_test['Classroom Type']==infra]['School No'].dropna().unique().astype(int))
        else:
            schools = [3, 5, 7, 9]
        school = st.selectbox("School", schools)

        if using_real_data:
            rooms = sorted(df_test[
                (df_test['Classroom Type']==infra) &
                (df_test['School No']==school)
            ]['Room No'].dropna().unique().astype(int))
        else:
            rooms = [1, 2, 3]
        room = st.selectbox("Room", rooms)

        st.divider()
        st.markdown("### Forecast settings")

        if using_real_data:
            dev_df = df_test[
                (df_test['Classroom Type']==infra) &
                (df_test['School No']==school) &
                (df_test['Room No']==room)
            ].sort_values('datetime')
            date_min = dev_df['datetime'].min().date()
            date_max = dev_df['datetime'].max().date()
            sel_date = st.date_input("Date", value=date_min, min_value=date_min, max_value=date_max)
            sel_time = st.time_input("Time", value=pd.Timestamp("09:30").time())
        else:
            sel_date = st.date_input("Date", value=pd.Timestamp("2023-10-25").date())
            sel_time = st.time_input("Time", value=pd.Timestamp("09:30").time())

        primary_target = st.selectbox("Primary pollutant", ['Measured CO2', 'Measured PM2.5', 'Measured PM10'],
                                      format_func=lambda x: THRESHOLDS[x]['label'])
        show_all = st.checkbox("Show all pollutants", value=True)
        show_actual = st.checkbox("Reveal actual values (test set)", value=False)

        st.divider()
        run_btn = st.button("▶  Run forecast", use_container_width=True, type="primary")

    # ── MAIN PANEL ────────────────────────────────────────────────────────────
    if not run_btn and not st.session_state.get('ran'):
        st.info("Select a classroom and click **Run forecast** to generate predictions.")
        return

    st.session_state['ran'] = True
    sel_ts = pd.Timestamp(f"{sel_date} {sel_time}")

    # Get or simulate data
    if using_real_data:
        dev_df_full = df_full[
            (df_full['Classroom Type']==infra) &
            (df_full['School No']==school) &
            (df_full['Room No']==room)
        ].sort_values('datetime').reset_index(drop=True)

        # Find closest row to selected timestamp
        time_diffs = (dev_df_full['datetime'] - sel_ts).abs()
        closest_idx = time_diffs.idxmin()
        current_row = dev_df_full.loc[closest_idx]

        # History: last 12 readings before this point
        history = dev_df_full[dev_df_full['datetime'] <= sel_ts].tail(12)

        feature_cols = feature_meta['feature_cols']
        scaler = build_scaler(scaler_params)
        row_features = current_row[feature_cols].values.astype(np.float64)

    else:
        # Demo mode — simulated values
        np.random.seed(42 + hash(f"{infra}{school}{room}") % 1000)
        h = sel_ts.hour
        school_factor = max(0, 1 - abs(h - 10) / 4) if 7 <= h <= 14 else 0
        base_co2 = 430 + 450 * school_factor + np.random.normal(0, 30)
        current_row = pd.Series({
            'Measured CO2': round(base_co2),
            'Measured PM2.5': round(5 + 8 * school_factor + np.random.normal(0, 2), 1),
            'Measured PM10':  round(8 + 12 * school_factor + np.random.normal(0, 3), 1),
            'Measured T':     round(18 + 4 * school_factor + np.random.normal(0, 1), 1),
            'Measured RH':    round(58 + 10 * school_factor + np.random.normal(0, 5), 1),
        })
        # Simulate history
        times = [sel_ts - pd.Timedelta(minutes=11*(11-i)) for i in range(12)]
        history = pd.DataFrame({'datetime': times})
        for col in ['Measured CO2', 'Measured PM2.5', 'Measured PM10', 'Measured T', 'Measured RH']:
            base = float(current_row[col])
            history[col] = [round(base * (0.85 + np.random.random()*0.3)) for _ in range(12)]
        row_features = None
        feature_cols = None
        scaler = None

    # ── CURRENT CONDITIONS ────────────────────────────────────────────────────
    st.markdown(f"#### Current conditions — {infra}, School {school}, Room {room}")
    st.caption(f"Timestamp: {sel_ts.strftime('%A %d %b %Y, %H:%M')}")

    cols = st.columns(5)
    for i, (tgt, cfg) in enumerate(list(THRESHOLDS.items())[:5]):
        val = float(current_row.get(tgt, np.nan))
        if np.isnan(val):
            cols[i].metric(cfg['label'], "N/A")
            continue
        level = get_alert_level(val, tgt)
        delta_color = "inverse" if level == 'ok' else "normal"
        cols[i].metric(
            cfg['label'],
            f"{val:.1f} {cfg['unit']}",
            delta=f"↑ Warn: {cfg['warn']}" if level != 'ok' else "Normal range",
            delta_color=delta_color
        )

    st.divider()

    # ── HISTORY CHART ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**Recent history** (last ~2 hours, 11-min intervals)")
        t_cfg = THRESHOLDS[primary_target]

        fig_hist = go.Figure()
        hist_times = history['datetime'].tolist() if 'datetime' in history.columns else list(range(len(history)))
        hist_vals  = history[primary_target].tolist() if primary_target in history.columns else [0]*len(history)

        fig_hist.add_trace(go.Scatter(
            x=hist_times, y=hist_vals,
            mode='lines+markers',
            line=dict(color='#185FA5', width=2),
            marker=dict(size=5),
            name=t_cfg['label']
        ))
        fig_hist.add_hline(y=t_cfg['warn'],   line_dash='dash', line_color='#BA7517', opacity=0.8,
                           annotation_text=f"Warn {t_cfg['warn']}", annotation_position='top right')
        fig_hist.add_hline(y=t_cfg['danger'], line_dash='dash', line_color='#E24B4A', opacity=0.8,
                           annotation_text=f"Critical {t_cfg['danger']}", annotation_position='top right')
        fig_hist.update_layout(
            height=260, margin=dict(l=10,r=10,t=10,b=10),
            yaxis_title=t_cfg['unit'], showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── FORECAST TABLE ────────────────────────────────────────────────────────
    with col_right:
        st.markdown("**Multi-horizon forecast**")
        all_horizons = ['11min', '33min', '1h', '3h', '6h']

        fc_rows = []
        cur_val = float(current_row.get(primary_target, np.nan))

        for h_label in all_horizons:
            if using_real_data and row_features is not None:
                pred_val, used_model = get_prediction(models, scaler, feature_cols, row_features, primary_target, h_label)
            else:
                # Demo simulation
                h_idx = all_horizons.index(h_label)
                trend = 1 + 0.06 * h_idx * (1 if sel_ts.hour < 12 else -0.5)
                pred_val = round(cur_val * trend + np.random.normal(0, 15))
                used_model = 'LightGBM' if h_idx < 3 else 'LSTM'

            if pred_val is None:
                continue

            delta = round(pred_val - cur_val)
            level = get_alert_level(pred_val, primary_target)
            icon = '🔴' if level == 'critical' else '🟡' if level == 'warn' else '🟢'
            fc_rows.append({
                'Horizon': HORIZON_LABELS[h_label],
                f'{THRESHOLDS[primary_target]["label"]} ({THRESHOLDS[primary_target]["unit"]})': round(pred_val),
                'Δ from now': f'+{delta}' if delta >= 0 else str(delta),
                'Status': icon,
                'Model': used_model or '—',
            })

        if fc_rows:
            fc_df = pd.DataFrame(fc_rows)
            st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # ── ALERT BOX ─────────────────────────────────────────────────────────────
    if fc_rows:
        max_pred = max(r[list(r.keys())[1]] for r in fc_rows)
        t_cfg = THRESHOLDS[primary_target]
        infra_tips = {
            'Container No Insulation': 'Container rooms lose ventilation benefit quickly — act earlier than brick classrooms.',
            'Container With Insulation': 'Insulated containers trap heat and CO₂; open both top vents if available.',
            'Mobile/Prefab': 'Mobile classrooms have high humidity risk — ensure cross-ventilation.',
            'Brick First Floor': 'Ground-level brick rooms benefit from door ventilation.',
            'Brick Second Floor': 'Upper-floor brick rooms: open windows on opposite walls for cross-ventilation.',
            'Brick Single Story': 'Single-story brick: roof ventilation is most effective if available.',
        }
        infra_tip = infra_tips.get(infra, '')

        if max_pred >= t_cfg['danger']:
            msg = (f"**Critical alert:** {t_cfg['label']} forecast to reach **{max_pred} {t_cfg['unit']}**. "
                   f"Open all windows and doors immediately. Consider relocating students. {infra_tip}")
            st.markdown(f'<div class="alert-critical">{msg}</div>', unsafe_allow_html=True)
        elif max_pred >= t_cfg['warn']:
            msg = (f"**Elevated {t_cfg['label']}:** Forecast to reach **{max_pred} {t_cfg['unit']}** "
                   f"within the next few hours. Open windows now to pre-ventilate. {infra_tip}")
            st.markdown(f'<div class="alert-warn">{msg}</div>', unsafe_allow_html=True)
        else:
            msg = f"**Air quality normal.** {t_cfg['label']} forecast to remain within safe levels across all horizons."
            st.markdown(f'<div class="alert-ok">{msg}</div>', unsafe_allow_html=True)

    st.divider()

    # ── REVEAL ACTUAL ─────────────────────────────────────────────────────────
    if show_actual and using_real_data:
        st.markdown("**Actual test-set values** (held-out ground truth)")
        future_cols_avail = [c for c in dev_df_full.columns if f'{primary_target}_future_' in c]
        if future_cols_avail and closest_idx < len(dev_df_full):
            actual_row = dev_df_full.loc[closest_idx]
            actual_data = []
            for col in future_cols_avail:
                h_label = col.split('_future_')[-1]
                actual_data.append({
                    'Horizon': HORIZON_LABELS.get(h_label, h_label),
                    f'Actual {THRESHOLDS[primary_target]["label"]}': round(float(actual_row[col])) if not pd.isna(actual_row[col]) else '—'
                })
            if actual_data:
                st.dataframe(pd.DataFrame(actual_data), use_container_width=True, hide_index=True)
    elif show_actual and not using_real_data:
        st.info("Actual values only available when running with real exported data.")

    # ── ALL POLLUTANTS ────────────────────────────────────────────────────────
    if show_all:
        st.divider()
        st.markdown("**All pollutants — current snapshot**")
        poll_cols = st.columns(5)
        for i, (tgt, cfg) in enumerate(list(THRESHOLDS.items())[:5]):
            val = float(current_row.get(tgt, np.nan))
            if np.isnan(val):
                poll_cols[i].metric(cfg['label'], "N/A")
                continue
            level = get_alert_level(val, tgt)
            status = '🔴 Critical' if level=='critical' else '🟡 Elevated' if level=='warn' else '🟢 Good'
            poll_cols[i].metric(cfg['label'], f"{val:.1f} {cfg['unit']}", status)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.divider()
    n_models = len(models)
    data_note = "Live model inference" if using_real_data else "Demo mode — simulated data"
    st.caption(
        f"BreatheEasy | {data_note} | {n_models} models loaded | "
        f"Test set: Oct–Nov 2023 | Thresholds: WHO/ASHRAE guidelines"
    )


if __name__ == '__main__':
    main()

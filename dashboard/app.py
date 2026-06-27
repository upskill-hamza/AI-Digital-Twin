import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))          # project root on path
from src.digital_twin import run_digital_twin, apply_scenario          # now src is a package

st.set_page_config(layout="wide", page_title="AI Climate Twin – India")

# ---- Load everything once (cached) ----
@st.cache_resource
def load_all():
    from src.download_data import download_all
    download_all()
    
    base = os.path.join(os.path.dirname(__file__), '..')
    model = tf.keras.models.load_model(os.path.join(base, 'models', 'convlstm_best.keras'))
    with open(os.path.join(base, 'data', 'norm_params.json')) as f:
        norm_params = json.load(f)
    coords = np.load(os.path.join(base, 'data', 'pilot_coords.npz'))
    test = np.load(os.path.join(base, 'data', 'test_data.npz'))
    dates = np.load(os.path.join(base, 'data', 'test_start_dates.npy'), allow_pickle=True)
    return model, norm_params, coords, test, dates

model, norm_params, coords, test, test_dates = load_all()
lat, lon = coords['lat'], coords['lon']
X_test = test['X']

# ---- Helpers ----
def denorm(arr):
    rain = arr[...,0] * (norm_params['rain']['vmax'] - norm_params['rain']['vmin']) + norm_params['rain']['vmin']
    tmax = arr[...,1] * (norm_params['tmax']['vmax'] - norm_params['tmax']['vmin']) + norm_params['tmax']['vmin']
    tmin = arr[...,2] * (norm_params['tmin']['vmax'] - norm_params['tmin']['vmin']) + norm_params['tmin']['vmin']
    return rain, tmax, tmin

def make_heatmap(data_2d, lon, lat, title, cmap='Blues', vmin=0, vmax=50, width=600, height=450):
    fig = go.Figure(go.Heatmap(
        z=data_2d, x=lon, y=lat,
        colorscale=cmap, zmin=vmin, zmax=vmax,
        hovertemplate='Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<br>Value: %{z:.1f}<extra></extra>'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Longitude', yaxis_title='Latitude',
        margin=dict(l=20, r=20, t=40, b=20),
        width=width, height=height
    )
    return fig

def compute_region_stats(rain, tmax, day_idx):
    r = rain[day_idx]
    t = tmax[day_idx]
    return {
        'rain_min': np.min(r), 'rain_max': np.max(r), 'rain_mean': np.mean(r),
        'tmax_min': np.min(t), 'tmax_max': np.max(t), 'tmax_mean': np.mean(t)
    }

# ---- Session state ----
if 'base_rain' not in st.session_state:
    st.session_state.base_rain = None
    st.session_state.base_tmax = None
    st.session_state.scen_rain = None
    st.session_state.scen_tmax = None
    st.session_state.total_days = 0
    st.session_state.scenario_name = "Baseline (No Change)"
    st.session_state.overlay_cache = {}

# ---- UI ----
st.title("🌍 AI‑Powered Digital Twin – India")
st.markdown("Pilot region: Tamil Nadu & Kerala | Climate what‑if simulation using IMD data & AI")

# Sidebar
st.sidebar.header("1️⃣ Scenario")
scenario_preset = st.sidebar.radio(
    "Select a scenario:",
    ["Baseline (No Change)", "🔥 Heatwave (+2°C)", "🌧️ Heavy Rain (+50%)", "🏜️ Drought (–50% Rain)", "⚙️ Custom"],
    index=0
)
if scenario_preset == "Baseline (No Change)":
    temp_offset = 0.0
    rain_scale  = 1.0
elif scenario_preset == "🔥 Heatwave (+2°C)":
    temp_offset = 2.0
    rain_scale  = 1.0
elif scenario_preset == "🌧️ Heavy Rain (+50%)":
    temp_offset = 0.0
    rain_scale  = 1.5
elif scenario_preset == "🏜️ Drought (–50% Rain)":
    temp_offset = 0.0
    rain_scale  = 0.5
else:
    temp_offset = st.sidebar.slider("Temperature change (°C)", -5.0, 5.0, 0.0, 0.5)
    rain_scale  = st.sidebar.slider("Rainfall multiplier", 0.0, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ Forecast Duration")
forecast_days = st.sidebar.selectbox("How many days ahead?", [3, 6, 9, 12, 15, 21], index=1)
n_steps = forecast_days // 3

st.sidebar.markdown("---")
st.sidebar.header("3️⃣ Start Date")
date_options = [str(d)[:10] for d in test_dates]
default_idx = len(test_dates) - 1
start_date_str = st.sidebar.selectbox("Forecast start date", options=date_options, index=default_idx)
start_idx = date_options.index(start_date_str)

run_btn = st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True)

# ---- Run simulation ----
if run_btn:
    with st.spinner("Running digital twin… (2‑3 seconds)"):
        init_state = X_test[start_idx:start_idx+1].copy()
        baseline_init = X_test[start_idx:start_idx+1].copy()

        if scenario_preset != "Baseline (No Change)":
            init_state = apply_scenario(init_state, temp_offset, rain_scale, norm_params)

        base_pred = run_digital_twin(model, baseline_init, n_steps)
        scen_pred = run_digital_twin(model, init_state, n_steps)

    base_daily = base_pred.reshape(-1, *base_pred.shape[2:])
    scen_daily = scen_pred.reshape(-1, *scen_pred.shape[2:])
    base_rain, base_tmax, _ = denorm(base_daily)
    scen_rain, scen_tmax, _ = denorm(scen_daily)

    total_days = base_rain.shape[0]
    st.session_state.base_rain = base_rain
    st.session_state.base_tmax = base_tmax
    st.session_state.scen_rain = scen_rain
    st.session_state.scen_tmax = scen_tmax
    st.session_state.total_days = total_days
    st.session_state.scenario_name = scenario_preset
    st.session_state.start_date_str = start_date_str
    st.session_state.overlay_cache = {}

# ---- Display maps & graphs ----
if st.session_state.base_rain is not None:
    total_days = st.session_state.total_days
    day = st.slider("📆 Day of forecast", 1, total_days, 1, key='day_slider')

    cache = st.session_state.overlay_cache
    def get_figure(key, create_fn):
        if key in cache: return cache[key]
        fig = create_fn()
        cache[key] = fig
        return fig

    st.markdown("## AI Forecast (Pilot Region) – Day‑wise Maps")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌧️ Rainfall")
        def rain_fig():
            return make_heatmap(st.session_state.scen_rain[day-1], lon, lat, f"Rainfall – Day {day}", 'Blues', 0, 50)
        st.plotly_chart(get_figure(f'rain_{day}', rain_fig), use_container_width=True)

        if st.session_state.scenario_name != "Baseline (No Change)":
            diff_rain = st.session_state.scen_rain[day-1] - st.session_state.base_rain[day-1]
            fig_diff = make_heatmap(diff_rain, lon, lat, "Rainfall Difference", 'RdBu', -20, 20)
            st.plotly_chart(fig_diff, use_container_width=True, key=f'diff_rain_{day}')

    with col2:
        st.markdown("### 🌡️ Max Temperature")
        def tmax_fig():
            return make_heatmap(st.session_state.scen_tmax[day-1], lon, lat, f"Max Temp – Day {day}", 'Reds', 20, 45)
        st.plotly_chart(get_figure(f'tmax_{day}', tmax_fig), use_container_width=True)

        if st.session_state.scenario_name != "Baseline (No Change)":
            diff_tmax = st.session_state.scen_tmax[day-1] - st.session_state.base_tmax[day-1]
            fig_diff_t = make_heatmap(diff_tmax, lon, lat, "Temperature Difference", 'RdBu', -5, 5)
            st.plotly_chart(fig_diff_t, use_container_width=True, key=f'diff_tmax_{day}')

    # Stats expander
    with st.expander("📊 Stats for Selected Day"):
        base_stats = compute_region_stats(st.session_state.base_rain, st.session_state.base_tmax, day-1)
        scen_stats = compute_region_stats(st.session_state.scen_rain, st.session_state.scen_tmax, day-1)
        stats_data = {
            'Rain Min (mm)': [base_stats['rain_min'], scen_stats['rain_min']],
            'Rain Max (mm)': [base_stats['rain_max'], scen_stats['rain_max']],
            'Rain Mean (mm)': [base_stats['rain_mean'], scen_stats['rain_mean']],
            'Tmax Min (°C)': [base_stats['tmax_min'], scen_stats['tmax_min']],
            'Tmax Max (°C)': [base_stats['tmax_max'], scen_stats['tmax_max']],
            'Tmax Mean (°C)': [base_stats['tmax_mean'], scen_stats['tmax_mean']]
        }
        stats_df = pd.DataFrame(stats_data, index=['Baseline', 'Scenario'])
        st.dataframe(stats_df.style.format("{:.2f}"))

    # Time series
    st.markdown("## 📈 Region‑Averaged Forecast Over Time")
    base_rain_mean = np.mean(st.session_state.base_rain, axis=(1,2))
    scen_rain_mean = np.mean(st.session_state.scen_rain, axis=(1,2))
    base_tmax_mean = np.mean(st.session_state.base_tmax, axis=(1,2))
    scen_tmax_mean = np.mean(st.session_state.scen_tmax, axis=(1,2))
    days_arr = np.arange(1, total_days+1)

    fig_ts_rain = go.Figure()
    fig_ts_rain.add_trace(go.Scatter(x=days_arr, y=base_rain_mean, mode='lines+markers', name='Baseline'))
    if st.session_state.scenario_name != "Baseline (No Change)":
        fig_ts_rain.add_trace(go.Scatter(x=days_arr, y=scen_rain_mean, mode='lines+markers', name='Scenario'))
    fig_ts_rain.update_layout(title="Average Rainfall (mm/day)", xaxis_title="Day", yaxis_title="mm")
    st.plotly_chart(fig_ts_rain, use_container_width=True)

    fig_ts_tmax = go.Figure()
    fig_ts_tmax.add_trace(go.Scatter(x=days_arr, y=base_tmax_mean, mode='lines+markers', name='Baseline'))
    if st.session_state.scenario_name != "Baseline (No Change)":
        fig_ts_tmax.add_trace(go.Scatter(x=days_arr, y=scen_tmax_mean, mode='lines+markers', name='Scenario'))
    fig_ts_tmax.update_layout(title="Average Max Temperature (°C)", xaxis_title="Day", yaxis_title="°C")
    st.plotly_chart(fig_ts_tmax, use_container_width=True)

    # Cumulative rainfall bar
    base_cum = np.sum(st.session_state.base_rain)
    scen_cum = np.sum(st.session_state.scen_rain)
    fig_bar = go.Figure(data=[go.Bar(name='Baseline', x=['Total Rainfall (mm)'], y=[base_cum], marker_color='blue')])
    if st.session_state.scenario_name != "Baseline (No Change)":
        fig_bar.add_trace(go.Bar(name='Scenario', x=['Total Rainfall (mm)'], y=[scen_cum], marker_color='orange'))
    fig_bar.update_layout(title="Total Rainfall Over Forecast Period")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Temperature histogram
    st.markdown("## 🌡️ Temperature Distribution (All Grid Cells, Day 1)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=st.session_state.base_tmax[0].flatten(), name='Baseline', opacity=0.6, nbinsx=20))
    if st.session_state.scenario_name != "Baseline (No Change)":
        fig_hist.add_trace(go.Histogram(x=st.session_state.scen_tmax[0].flatten(), name='Scenario', opacity=0.6, nbinsx=20))
    fig_hist.update_layout(barmode='overlay', title="Day 1 Max Temperature Distribution", xaxis_title="°C")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.success(f"Showing day {day} of {total_days} — simulation started on {st.session_state.start_date_str}")
else:
    st.info("👈 Choose a scenario, length, and start date, then click **Run Simulation**.")
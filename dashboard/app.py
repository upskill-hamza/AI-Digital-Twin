import streamlit as st
import numpy as np
import tensorflow as tf
import json
import os
import sys
from datetime import datetime
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
from PIL import Image
import base64
import xarray as xr
from scipy.ndimage import zoom

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from digital_twin import run_digital_twin, apply_scenario

st.set_page_config(layout="wide", page_title="AI Climate Twin – India")

# ---- Load everything once (cached) ----
@st.cache_resource
def load_all():
    base = os.path.join(os.path.dirname(__file__), '..')
    model = tf.keras.models.load_model(os.path.join(base, 'models', 'convlstm_best.keras'))
    with open(os.path.join(base, 'data', 'norm_params.json')) as f:
        norm_params = json.load(f)
    coords = np.load(os.path.join(base, 'data', 'pilot_coords.npz'))
    test = np.load(os.path.join(base, 'data', 'test_data.npz'))
    dates = np.load(os.path.join(base, 'data', 'test_start_dates.npy'), allow_pickle=True)
    ds_full = xr.open_dataset(os.path.join(base, 'data', 'climate_combined.nc'))
    return model, norm_params, coords, test, dates, ds_full

model, norm_params, coords, test, test_dates, ds_full = load_all()
lat, lon = coords['lat'], coords['lon']
X_test = test['X']

# ---- Helper functions ----
def denorm(arr):
    rain = arr[...,0] * (norm_params['rain']['vmax'] - norm_params['rain']['vmin']) + norm_params['rain']['vmin']
    tmax = arr[...,1] * (norm_params['tmax']['vmax'] - norm_params['tmax']['vmin']) + norm_params['tmax']['vmin']
    tmin = arr[...,2] * (norm_params['tmin']['vmax'] - norm_params['tmin']['vmin']) + norm_params['tmin']['vmin']
    return rain, tmax, tmin

def array_to_data_uri(values, lon, lat, cmap, vmin, vmax, opacity=0.6, scale_factor=2):
    """Smooth overlay – moderate upscale for speed."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.cm.get_cmap(cmap)
    h, w = values.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    values_smooth = zoom(values, (new_h / h, new_w / w), order=1)
    lat_smooth = np.linspace(lat.min(), lat.max(), new_h)
    lon_smooth = np.linspace(lon.min(), lon.max(), new_w)
    coloured = colormap(norm(values_smooth))
    coloured[:, :, 3] = opacity
    img = Image.fromarray((coloured * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    data_uri = f'data:image/png;base64,{img_base64}'
    bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
    return data_uri, bounds

def make_map(data_uri, bounds, center_lat=20.0, center_lon=78.0, zoom_start=4):
    """Clean full‑India map with state boundaries + climate overlay."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True
    )
    folium.TileLayer(
        tiles='cartodbpositron_nolabels',
        name='Light Background',
        control=False,
        show=True,
        opacity=0.3
    ).add_to(m)

    geojson_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'geojson', 'india_states.geojson')
    if os.path.exists(geojson_path):
        folium.GeoJson(
            geojson_path,
            name='India States',
            style_function=lambda x: {
                'fillColor': 'none',
                'color': 'black',
                'weight': 1.5,
                'fillOpacity': 0
            }
        ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=data_uri,
        bounds=bounds,
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=10,
    ).add_to(m)
    return m

def get_observed_overlay(ds, date, variable, cmap, vmin, vmax, opacity=0.7):
    """Observed data overlay – no upscaling (fast)."""
    da = ds[variable].sel(time=date, method='nearest')
    lat_full = da.lat.values
    lon_full = da.lon.values
    values = da.values
    return array_to_data_uri(values, lon_full, lat_full, cmap, vmin, vmax, opacity, scale_factor=1)

# ---- Session state initialisation ----
if 'base_rain' not in st.session_state:
    st.session_state.base_rain = None
    st.session_state.base_tmax = None
    st.session_state.scen_rain = None
    st.session_state.scen_tmax = None
    st.session_state.total_days = 0
    st.session_state.scenario_name = "Baseline (No Change)"
    st.session_state.overlays = {}   # precomputed URIs for each day

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
# ---- Run simulation (fast, no pre‑computation) ----
if run_btn:
    with st.spinner("Running digital twin… (2‑3 seconds)"):
        init_state = X_test[start_idx:start_idx+1].copy()
        baseline_init = X_test[start_idx:start_idx+1].copy()

        if scenario_preset != "Baseline (No Change)":
            init_state = apply_scenario(init_state, temp_offset, rain_scale, norm_params)

        base_pred = run_digital_twin(model, baseline_init, n_steps)
        scen_pred = run_digital_twin(model, init_state, n_steps)

    # Denormalise
    base_daily = base_pred.reshape(-1, *base_pred.shape[2:])
    scen_daily = scen_pred.reshape(-1, *scen_pred.shape[2:])
    base_rain, base_tmax, _ = denorm(base_daily)
    scen_rain, scen_tmax, _ = denorm(scen_daily)

    total_days = base_rain.shape[0]

    # Store raw arrays in session state
    st.session_state.base_rain = base_rain
    st.session_state.base_tmax = base_tmax
    st.session_state.scen_rain = scen_rain
    st.session_state.scen_tmax = scen_tmax
    st.session_state.total_days = total_days
    st.session_state.scenario_name = scenario_preset
    st.session_state.start_date_str = start_date_str

    # Clear any old overlay cache
    st.session_state.overlay_cache = {}

# ---- Display maps (lazy overlay generation) ----
if st.session_state.base_rain is not None:
    total_days = st.session_state.total_days
    day = st.slider("📆 Day of forecast", 1, total_days, 1, key='day_slider')

    # Initialise cache if not present
    if 'overlay_cache' not in st.session_state:
        st.session_state.overlay_cache = {}

    cache = st.session_state.overlay_cache

    # Helper to get or create overlay for a given key
    def get_overlay(key, data_fn):
        if key in cache:
            return cache[key]
        uri, _ = data_fn()
        cache[key] = uri
        return uri

    # Bounds for pilot region
    bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]

    st.markdown("## AI Forecast (Pilot Region)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌧️ Rainfall Forecast")
        # Lazy generate rain overlay
        def rain_data_fn():
            return array_to_data_uri(st.session_state.scen_rain[day-1], lon, lat, 'Blues', 0, 50, 0.7, scale_factor=2)
        rain_uri = get_overlay(f'rain_{day}', rain_data_fn)
        rain_map = make_map(rain_uri, bounds, 20.0, 78.0, 4)
        st_folium(rain_map, width=700, height=500, key=f'rain_map_{day}')

        if st.session_state.scenario_name != "Baseline (No Change)":
            st.markdown("#### Rainfall Change (Scenario − Baseline)")
            def diff_rain_fn():
                diff = st.session_state.scen_rain[day-1] - st.session_state.base_rain[day-1]
                return array_to_data_uri(diff, lon, lat, 'RdBu', -20, 20, 0.7, 2)
            diff_uri = get_overlay(f'diff_rain_{day}', diff_rain_fn)
            diff_map = make_map(diff_uri, bounds, 20.0, 78.0, 4)
            st_folium(diff_map, width=700, height=500, key=f'rain_diff_{day}')

    with col2:
        st.markdown("### 🌡️ Temperature Forecast")
        def tmax_data_fn():
            return array_to_data_uri(st.session_state.scen_tmax[day-1], lon, lat, 'Reds', 20, 45, 0.7, scale_factor=2)
        tmax_uri = get_overlay(f'tmax_{day}', tmax_data_fn)
        tmax_map = make_map(tmax_uri, bounds, 20.0, 78.0, 4)
        st_folium(tmax_map, width=700, height=500, key=f'tmax_map_{day}')

        if st.session_state.scenario_name != "Baseline (No Change)":
            st.markdown("#### Temperature Change (Scenario − Baseline)")
            def diff_tmax_fn():
                diff = st.session_state.scen_tmax[day-1] - st.session_state.base_tmax[day-1]
                return array_to_data_uri(diff, lon, lat, 'RdBu', -5, 5, 0.7, 2)
            diff_turi = get_overlay(f'diff_tmax_{day}', diff_tmax_fn)
            diff_map_t = make_map(diff_turi, bounds, 20.0, 78.0, 4)
            st_folium(diff_map_t, width=700, height=500, key=f'tmax_diff_{day}')

    st.success(f"Showing day {day} of {total_days} — simulation started on {st.session_state.start_date_str}")

    # ---- Observed Data Toggle ----
    st.markdown("---")
    show_obs = st.checkbox("📡 Show Observed IMD Data for Start Date (All India)")
    if show_obs:
        start_dt = np.datetime64(st.session_state.start_date_str)
        st.markdown("## Observed Data (All India)")
        # Observed overlays are also cached separately
        if 'obs_rain_uri' not in st.session_state:
            st.session_state.obs_rain_uri, obs_bounds = get_observed_overlay(ds_full, start_dt, 'rain', 'Blues', 0, 50, 0.7)
            st.session_state.obs_bounds = obs_bounds
            st.session_state.obs_tmax_uri, obs_bounds_t = get_observed_overlay(ds_full, start_dt, 'tmax', 'Reds', 20, 45, 0.7)
            st.session_state.obs_bounds_t = obs_bounds_t
        else:
            obs_bounds = st.session_state.obs_bounds
            obs_bounds_t = st.session_state.obs_bounds_t

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Observed Rainfall")
            obs_rain_map = make_map(st.session_state.obs_rain_uri, obs_bounds, 20.0, 78.0, 4)
            st_folium(obs_rain_map, width=700, height=500, key='obs_rain')
        with col2:
            st.markdown("### Observed Max Temperature")
            obs_tmax_map = make_map(st.session_state.obs_tmax_uri, obs_bounds_t, 20.0, 78.0, 4)
            st_folium(obs_tmax_map, width=700, height=500, key='obs_tmax')
else:
    st.info("👈 Choose a scenario, length, and start date, then click **Run Simulation**.")
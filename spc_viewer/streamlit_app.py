# streamlit_app.py
# -------------------------------------------------------------
# Interactive SPC viewer for your Auto‑Caliper capture logs.
# 
# • Reads the latest CSV in C:\\CaliperCapture\\Logs (or latest per Order ID)
# • Individuals (I) chart + Moving Range (MR) chart
# • Sigma estimated from MR(2); WECO Rule 1 highlighting (beyond limits)
# • Optional filters: profile, reel; adjustable sigma limits, window size
# • Auto‑refresh by polling file mtime (no GitHub, no database)
# 
# How to run (once Python + Streamlit are installed):
#   pip install streamlit plotly pandas
#   streamlit run streamlit_app.py
# -------------------------------------------------------------

import os
import time
import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Local SPC Viewer", layout="wide")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Data Source")
logs_dir = st.sidebar.text_input("Logs folder", r"C:\\CaliperCapture\\Logs")
order_id = st.sidebar.text_input("Order ID (optional)", "").strip().upper()
pattern = f"{order_id}_*.csv" if order_id else "*.csv"
pick_latest = st.sidebar.checkbox("Always use most recent file", True)
refresh_s = st.sidebar.slider("Auto-refresh (seconds)", 0, 60, 10, 1)

st.sidebar.header("Filters & Chart")
profile_filter = st.sidebar.text_input("Profile filter (optional)", "").strip()
reel_filter = st.sidebar.text_input("Reel filter (optional)", "").strip().upper()
sigma_mult = st.sidebar.slider("Sigma limits (±)", 2.0, 4.0, 3.0, 0.5)
window_pts = st.sidebar.number_input("Show last N points (0 = all)", 0, 50000, 0, 100)
show_table = st.sidebar.checkbox("Show data table", False)

# ---------------- File discovery helpers ----------------
def list_csvs(folder: str, patt: str) -> list[str]:
    files = glob.glob(os.path.join(folder, patt))
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=os.path.getmtime, reverse=True)
    return files

files = list_csvs(logs_dir, pattern)
if not files:
    st.info("No CSVs found yet. Waiting for data…\n\nFolder: %s\nPattern: %s" % (logs_dir, pattern))
    st.stop()

file_path = files[0] if pick_latest else st.sidebar.selectbox("Choose file", files, index=0)
file_path = Path(file_path)

# ---------------- Robust CSV loading with cache ----------------
@st.cache_data
def load_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    # retry a few times in case another process is writing
    last_err: Optional[Exception] = None
    for _ in range(5):
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    if last_err:
        raise last_err
    return pd.read_csv(path)

try:
    mtime = os.path.getmtime(file_path)
except Exception:
    mtime = time.time()

df = load_csv_cached(str(file_path), mtime)

# ---------------- Basic hygiene & optional filters ----------------
# Expected columns from your capture app: timestamp, operator, part_id, order_id, reel, profile, DIM n, DIM n_result …
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    # fabricate an index timestamp if missing
    df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="T")

# Optional quick-filters based on profile/reel text inputs
if profile_filter:
    mask = df.get("profile", "").astype(str).str.contains(profile_filter, case=False, na=False)
    df = df[mask]
if reel_filter:
    if "reel" in df.columns:
        df = df[df["reel"].astype(str).str.upper() == reel_filter]

# Choose which numeric series to chart: by default take the first DIM* column that is numeric
value_cols = [c for c in df.columns if c.startswith("DIM ")]
num_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().any()]
if not num_cols:
    # fall back to the last numeric column in the frame
    fallback = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not fallback:
        st.error("No numeric measurement columns found (e.g., 'DIM 1').")
        st.stop()
    series_col = fallback[-1]
else:
    series_col = num_cols[0]

st.subheader(f"File: {file_path.name}")
st.caption(f"Last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
st.write(f"**Charting column:** `{series_col}`  · Rows: {len(df)}")

# Trim to last N points if requested
if window_pts and window_pts > 0:
    df = df.tail(int(window_pts)).copy()

# Ensure numeric
x = pd.to_numeric(df[series_col], errors="coerce").astype(float)

# ---------------- SPC calculations (I & MR) ----------------
def individuals_limits(series: pd.Series, sigma_mult: float) -> Tuple[float, float, float, float]:
    # MR(2) method
    mr = series.diff().abs()
    d2 = 1.128  # for subgroup size = 2
    sigma_hat = mr.mean() / d2 if mr.notna().any() else np.nan
    center = series.mean()
    ucl = center + sigma_mult * sigma_hat
    lcl = center - sigma_mult * sigma_hat
    return center, lcl, ucl, sigma_hat

center, lcl, ucl, sigma_hat = individuals_limits(x, sigma_mult)

# MR chart limits (n=2): D3=0, D4=3.267
mr = x.diff().abs()
Rbar = mr.mean()
D3, D4 = 0.0, 3.267
MR_UCL = D4 * Rbar
MR_LCL = max(D3 * Rbar, 0.0)

# Rule 1: beyond control limits
viol_mask = (x < lcl) | (x > ucl)

# ---------------- Plotly: Individuals chart ----------------
fig_i = go.Figure()
fig_i.add_trace(go.Scatter(
    x=df["timestamp"], y=x,
    mode="lines+markers", name="Value",
    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Value=%{y:.5f}<extra></extra>",
))
fig_i.add_hline(y=center, line_dash="dash", annotation_text=f"Center {center:.5f}")
fig_i.add_hline(y=ucl, line_dash="dot", annotation_text=f"UCL {ucl:.5f}")
fig_i.add_hline(y=lcl, line_dash="dot", annotation_text=f"LCL {lcl:.5f}")

if viol_mask.any():
    v = df.loc[viol_mask]
    fig_i.add_trace(go.Scatter(
        x=v["timestamp"], y=v[series_col],
        mode="markers", name="Violations",
        marker=dict(size=10, symbol="x"),
        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Out-of-control=%{y:.5f}<extra></extra>",
    ))

fig_i.update_layout(
    title=f"Individuals (I) · μ≈{center:.5f} · σ̂≈{(sigma_hat if pd.notna(sigma_hat) else 0):.5f}",
    xaxis_title="timestamp", yaxis_title=series_col,
    height=420, hovermode="x unified", legend_title="Series",
)

# ---------------- Plotly: Moving Range chart ----------------
fig_mr = go.Figure()
fig_mr.add_trace(go.Scatter(
    x=df["timestamp"], y=mr,
    mode="lines+markers", name="MR(2)",
    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>MR=%{y:.5f}<extra></extra>",
))
fig_mr.add_hline(y=Rbar, line_dash="dash", annotation_text=f"Ȓ {Rbar:.5f}")
fig_mr.add_hline(y=MR_UCL, line_dash="dot", annotation_text=f"UCL {MR_UCL:.5f}")
fig_mr.add_hline(y=MR_LCL, line_dash="dot", annotation_text=f"LCL {MR_LCL:.5f}")
fig_mr.update_layout(
    title="Moving Range (MR)", xaxis_title="timestamp", yaxis_title="|ΔX|",
    height=320, hovermode="x unified", legend_title="Series",
)

# ---------------- Layout ----------------
st.plotly_chart(fig_i, use_container_width=True)
st.plotly_chart(fig_mr, use_container_width=True)

# ---------------- Optional table & download ----------------
if show_table:
    st.subheader("Data preview")
    st.dataframe(df, use_container_width=True, height=360)

st.download_button(
    label="Download plotted data (CSV)",
    data=df.to_csv(index=False).encode(),
    file_name=f"spc_view_{file_path.stem}.csv",
)

# ---------------- Auto-refresh ----------------
# If refresh_s > 0, ping rerun periodically; cache bust happens when mtime changes.
if refresh_s > 0:
    st.autorefresh(interval=refresh_s * 1000, key="spc-autorefresh")

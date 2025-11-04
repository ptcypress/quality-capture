# streamlit_app.py
# -------------------------------------------------------------
# Local SPC viewer for Auto-Caliper capture logs
# - Polls a local folder for newest CSV/XLSX (or newest for a given Order ID)
# - Individuals (I) chart + Moving Range (MR) chart
# - Sigma estimated from MR(2); Rule-1 highlighting (beyond limits)
# - Debug expander shows exactly what files were found/matched
# -------------------------------------------------------------

import os
import time
import glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Local SPC Viewer", layout="wide")

# ---------------- Defaults ----------------
# Prefer repo-relative logs: ../app_capture/Logs
REPO_LOGS = (Path(__file__).resolve().parents[1] / "app_capture" / "Logs")
FALLBACK_LOGS = Path(r"C:\CaliperCapture\Logs")
DEFAULT_LOGS = str(REPO_LOGS if REPO_LOGS.exists() else FALLBACK_LOGS)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Data Source")
logs_dir = st.sidebar.text_input("Logs folder", DEFAULT_LOGS)
order_id = st.sidebar.text_input("Order ID (optional)", "").strip().upper()

# Pattern is lenient; we still enforce ORDERID_ prefix later if order_id provided
pattern = f"{order_id}_*" if order_id else "*"
pick_latest = st.sidebar.checkbox("Always use most recent file", True)
refresh_s = st.sidebar.slider("Auto-refresh (seconds)", 0, 60, 10, 1)

st.sidebar.header("Filters & Chart")
profile_filter = st.sidebar.text_input("Profile filter (optional)", "").strip()
reel_filter = st.sidebar.text_input("Reel filter (optional)", "").strip().upper()
sigma_mult = st.sidebar.slider("Sigma limits (±)", 2.0, 4.0, 3.0, 0.5)
window_pts = st.sidebar.number_input("Show last N points (0 = all)", 0, 50000, 0, 100)
show_table = st.sidebar.checkbox("Show data table", False)

# ---------------- File discovery helpers (more forgiving + debug) ----------------
def list_files_debug(folder: str, patt: str) -> Dict:
    """
    Return diagnostic info including:
      - folder existence
      - all files in folder
      - matched candidates (case-insensitive, csv/xlsx only)
    """
    import traceback
    info = {
        "folder": folder,
        "pattern": patt,
        "exists": False,
        "files_all": [],
        "files_match": [],
        "error": None,
    }
    try:
        info["exists"] = os.path.isdir(folder)
        if info["exists"]:
            # List all files (non-recursive)
            all_files = [str(p) for p in Path(folder).glob("*") if Path(p).is_file()]
            info["files_all"] = sorted(all_files, key=os.path.getmtime, reverse=True)

            patt_lower = patt.lower()
            candidates: List[str] = []
            for f in all_files:
                name = os.path.basename(f)
                name_lower = name.lower()
                # Loose match: substring or glob
                if patt_lower.replace("*", "") in name_lower or glob.fnmatch.fnmatch(name_lower, patt_lower):
                    candidates.append(f)

            # Accept only CSV/XLSX
            info["files_match"] = [
                f for f in candidates if os.path.splitext(f)[1].lower() in (".csv", ".xlsx")
            ]
        return info
    except Exception as e:
        info["error"] = f"{e}\n{traceback.format_exc()}"
        return info

debug = list_files_debug(logs_dir, pattern)

with st.expander("🔎 Debug: folder scan", expanded=False):
    st.write(debug)

# Choose files (prefer CSV first, then XLSX)
files = debug["files_match"] if debug["exists"] else []
# If user typed an Order ID, strongly filter to ORDERID_* prefix
if order_id:
    files = [f for f in files if os.path.basename(f).upper().startswith(order_id + "_")]

# Prefer CSVs first
files = sorted(files, key=os.path.getmtime, reverse=True)
files_csv = [f for f in files if f.lower().endswith(".csv")]
files_xlsx = [f for f in files if f.lower().endswith(".xlsx")]
files = files_csv + files_xlsx

if not files:
    st.error(
        "No matching files found.\n\n"
        "Tips:\n"
        "• Clear the Order ID box to see all files\n"
        "• Make sure the file extension is .csv or .xlsx\n"
        "• Confirm the path in the debug panel above"
    )
    st.stop()

file_path = Path(files[0]) if pick_latest else Path(st.sidebar.selectbox("Choose file", files, index=0))

# ---------------- Robust loader for CSV/XLSX with cache ----------------
@st.cache_data
def load_any(path: str, mtime: float) -> pd.DataFrame:
    """
    Cache is keyed by file path + mtime. Retries a few times in case another process
    is writing the file. Supports .csv and .xlsx.
    """
    ext = os.path.splitext(path)[1].lower()
    last_err: Optional[Exception] = None
    for _ in range(5):
        try:
            if ext == ".xlsx":
                return pd.read_excel(path)
            return pd.read_csv(path)
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    if last_err:
        raise last_err
    # final attempt without catching
    if ext == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)

try:
    mtime = os.path.getmtime(file_path)
except Exception:
    mtime = time.time()

df = load_any(str(file_path), mtime)

# ---------------- Basic hygiene & optional filters ----------------
# Expected columns: timestamp, operator, part_id, order_id, reel, profile, DIM n, DIM n_result …
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    # fabricate timestamps if missing
    df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="T")

# Optional text filters
if profile_filter:
    if "profile" in df.columns:
        df = df[df["profile"].astype(str).str.contains(profile_filter, case=False, na=False)]
if reel_filter:
    if "reel" in df.columns:
        df = df[df["reel"].astype(str).str.upper() == reel_filter]

# Pick a numeric DIM column by default
value_cols = [c for c in df.columns if c.upper().startswith("DIM ")]
num_cols = []
for c in value_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    if s.notna().any():
        num_cols.append(c)

if not num_cols:
    # Fall back to any numeric column
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
    title="Moving Range (MR)",
    xaxis_title="timestamp", yaxis_title="|ΔX|",
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
# Works on old and new Streamlit versions
if refresh_s > 0:
    auto = getattr(st, "autorefresh", None)
    if callable(auto):
        st.autorefresh(interval=int(refresh_s * 1000), key="spc-autorefresh")
    else:
        import time
        time.sleep(refresh_s)
        st.experimental_rerun()


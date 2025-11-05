# streamlit_app.py — multi-DIM + Raw/Actual selector
# -------------------------------------------------------------
# • Local/GitHub modes (from previous version)
# • Select multiple DIMs for the Individuals chart
# • Toggle: Actual (corrected) vs Raw (caliper)
# • MR chart shows when exactly ONE DIM is selected
# • Works with .csv/.xlsx; adds debug expander
# -------------------------------------------------------------

import os, time, glob, io
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional libs
try:
    import requests
except Exception:
    requests = None
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
except Exception:
    _st_autorefresh = None

st.set_page_config(page_title="SPC Viewer", layout="wide")

# ---------------- Sidebar: Source & common controls ----------------
st.sidebar.header("Source")
source = st.sidebar.radio("Where are the logs?", ["Local folder", "GitHub repo"], index=0)

REPO_LOGS = (Path(__file__).resolve().parents[1] / "app_capture" / "Logs")
FALLBACK_LOGS = Path(r"C:\CaliperCapture\Logs")
DEFAULT_LOGS = str(REPO_LOGS if REPO_LOGS.exists() else FALLBACK_LOGS)

order_id = st.sidebar.text_input("Order ID (optional)", "").strip().upper()
profile_filter = st.sidebar.text_input("Profile filter (optional)", "").strip()
reel_filter = st.sidebar.text_input("Reel filter (optional)", "").strip().upper()

st.sidebar.header("Chart options")
value_mode = st.sidebar.radio("Values to plot", ["Actual (corrected)", "Raw (caliper)"], index=0)
sigma_mult = st.sidebar.slider("Sigma limits (±)", 2.0, 4.0, 3.0, 0.5)
window_pts = st.sidebar.number_input("Show last N points (0 = all)", 0, 50000, 0, 100)
show_table = st.sidebar.checkbox("Show data table", False)
refresh_s = st.sidebar.slider("Auto-refresh (seconds)", 0, 120, 10, 1)

# ---------------- Helpers ----------------
def is_numeric_series(s: pd.Series) -> bool:
    try:
        return pd.to_numeric(s, errors="coerce").notna().any()
    except Exception:
        return False

def individuals_limits(series: pd.Series, sigma_mult: float) -> Tuple[float, float, float, float]:
    mr = series.diff().abs()
    d2 = 1.128
    sigma_hat = mr.mean() / d2 if mr.notna().any() else np.nan
    center = series.mean()
    ucl = center + sigma_mult * sigma_hat
    lcl = center - sigma_mult * sigma_hat
    return center, lcl, ucl, sigma_hat

# ---------------- Local mode helpers ----------------
def list_files_debug_local(folder: str, patt: str) -> Dict:
    info = {"mode": "local", "folder": folder, "pattern": patt, "exists": False,
            "files_all": [], "files_match": [], "error": None}
    try:
        info["exists"] = os.path.isdir(folder)
        if info["exists"]:
            all_files = [str(p) for p in Path(folder).glob("*") if Path(p).is_file()]
            info["files_all"] = sorted(all_files, key=os.path.getmtime, reverse=True)
            patt_lower = patt.lower()
            candidates = []
            for f in all_files:
                name = os.path.basename(f)
                name_lower = name.lower()
                if patt_lower.replace("*", "") in name_lower or glob.fnmatch.fnmatch(name_lower, patt_lower):
                    candidates.append(f)
            info["files_match"] = [f for f in candidates if os.path.splitext(f)[1].lower() in (".csv", ".xlsx")]
    except Exception as e:
        info["error"] = str(e)
    return info

@st.cache_data
def load_local_file(path: str, mtime: float) -> pd.DataFrame:
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
    return pd.read_excel(path) if ext == ".xlsx" else pd.read_csv(path)

# ---------------- GitHub mode helpers ----------------
def gh_list_files(owner: str, repo: str, path: str, ref: str = "main") -> List[Dict]:
    if requests is None:
        raise RuntimeError("requests not installed. pip install requests")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, params={"ref": ref}, timeout=15)
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list):
        return []
    return [
        {"name": it["name"], "path": it["path"], "download_url": it.get("download_url", "")}
        for it in items
        if it.get("type") == "file" and str(it.get("name", "")).lower().endswith((".csv", ".xlsx"))
    ]

@st.cache_data
def gh_download_any(download_url: str, ext: str) -> pd.DataFrame:
    r = requests.get(download_url, timeout=20)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    return pd.read_excel(buf) if ext == ".xlsx" else pd.read_csv(buf)

# ---------------- Load data (Local or GitHub) ----------------
if source == "Local folder":
    logs_dir = st.sidebar.text_input("Logs folder", DEFAULT_LOGS)
    pattern = f"{order_id}_*" if order_id else "*"
    debug = list_files_debug_local(logs_dir, pattern)
    with st.expander("🔎 Debug (local folder)", expanded=False):
        st.write(debug)

    files = debug["files_match"] if debug["exists"] else []
    if order_id:
        files = [f for f in files if os.path.basename(f).upper().startswith(order_id + "_")]

    files = sorted(files, key=os.path.getmtime, reverse=True)
    files_csv = [f for f in files if f.lower().endswith(".csv")]
    files_xlsx = [f for f in files if f.lower().endswith(".xlsx")]
    files = files_csv + files_xlsx

    if not files:
        st.error("No matching files found. Clear Order ID to see all; check debug panel.")
        st.stop()

    pick_latest = st.sidebar.checkbox("Always use most recent file", True)
    file_path = Path(files[0]) if pick_latest else Path(st.sidebar.selectbox("Choose file", files, index=0))

    try:
        mtime = os.path.getmtime(file_path)
    except Exception:
        mtime = time.time()

    df = load_local_file(str(file_path), mtime)
    file_label = file_path.name
    last_modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))

else:
    st.sidebar.write("Pull from a **public GitHub repo**")
    owner = st.sidebar.text_input("GitHub owner/org", "your-org-or-user").strip()
    repo  = st.sidebar.text_input("Repo", "caliper-quality-suite").strip()
    folder_path = st.sidebar.text_input("Folder path in repo", "app_capture/Logs").strip()
    branch = st.sidebar.text_input("Branch/Ref", "main").strip()
    if not (owner and repo and folder_path):
        st.info("Enter owner, repo, and folder path.")
        st.stop()
    try:
        listing = gh_list_files(owner, repo, folder_path, ref=branch)
    except Exception as e:
        st.error(f"GitHub listing failed:\n{e}")
        st.stop()
    files = [it for it in listing if (order_id == "" or it["name"].upper().startswith(order_id + "_"))]
    if not files:
        st.error("No matching CSV/XLSX in the GitHub folder (check Order ID and path).")
        st.stop()
    files_csv = [it for it in files if it["name"].lower().endswith(".csv")]
    files_xlsx = [it for it in files if it["name"].lower().endswith(".xlsx")]
    files = sorted(files_csv + files_xlsx, key=lambda it: it["name"], reverse=True)
    pick_latest = st.sidebar.checkbox("Always use most recent file", True)
    chosen = files[0] if pick_latest else st.sidebar.selectbox("Choose file", files, index=0, format_func=lambda it: it["name"])
    ext = ".xlsx" if chosen["name"].lower().endswith(".xlsx") else ".csv"
    df = gh_download_any(chosen["download_url"], ext)
    file_label = f"{owner}/{repo}/{chosen['path']}"
    last_modified = "(GitHub fetch)"

# ---------------- Clean & filter ----------------
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="T")

if profile_filter and "profile" in df.columns:
    df = df[df["profile"].astype(str).str.contains(profile_filter, case=False, na=False)]
if reel_filter and "reel" in df.columns:
    df = df[df["reel"].astype(str).str.upper() == reel_filter]

# Identify DIMs present (Actual columns are "DIM n"; Raw columns are "_raw_DIM n")
dim_actual = [c for c in df.columns if c.upper().startswith("DIM ")]
# derive a clean list of DIM names like "DIM 1","DIM 2", keeping only those where either Actual or Raw exists as numeric
dims_available: List[str] = []
for c in sorted(dim_actual, key=lambda s: (int(''.join(ch for ch in s if ch.isdigit()) or 1), s)):
    dims_available.append(c)
# If there are DIMs that only exist as raw (edge case), include them too
for c in df.columns:
    if c.startswith("_raw_DIM "):
        base = c.replace("_raw_", "")
        if base not in dims_available:
            dims_available.append(base)

# UI: pick DIMs (multi-select)
if not dims_available:
    st.error("No DIM columns found.")
    st.stop()
dim_picks = st.sidebar.multiselect("Select DIMs to plot", dims_available, default=dims_available[:1])

# Short-circuit if none picked
if not dim_picks:
    st.info("Select at least one DIM to plot from the sidebar.")
    st.stop()

st.subheader(f"File: {file_label}")
st.caption(f"Last modified: {last_modified}")
st.write(f"**Values:** {value_mode}  · **Rows:** {len(df)}  · **DIMs:** {', '.join(dim_picks)}")

# Window
if window_pts and window_pts > 0:
    df = df.tail(int(window_pts)).copy()

# Helper to fetch the series by mode
def get_series_for_dim(frame: pd.DataFrame, dim_name: str, mode: str) -> pd.Series:
    if mode.startswith("Actual"):
        # corrected lives in the DIM column itself
        col = dim_name
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").astype(float)
        # fallback: if only raw exists, warn
        raw_col = "_raw_" + dim_name
        if raw_col in frame.columns:
            st.warning(f"{dim_name}: Actual column missing; using RAW as fallback")
            return pd.to_numeric(frame[raw_col], errors="coerce").astype(float)
        return pd.Series(dtype=float)
    else:
        # Raw (caliper)
        raw_col = "_raw_" + dim_name
        if raw_col in frame.columns:
            return pd.to_numeric(frame[raw_col], errors="coerce").astype(float)
        # fallback: if raw missing, use actual
        if dim_name in frame.columns:
            st.warning(f"{dim_name}: RAW column missing; using ACTUAL as fallback")
            return pd.to_numeric(frame[dim_name], errors="coerce").astype(float)
        return pd.Series(dtype=float)

# ---------------- Individuals chart (multi-series) ----------------
fig_i = go.Figure()
centers, lcls, ucls, sigmas = {}, {}, {}, {}

for dim in dim_picks:
    y = get_series_for_dim(df, dim, value_mode)
    fig_i.add_trace(go.Scatter(
        x=df["timestamp"], y=y, mode="lines+markers", name=dim,
        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>"+dim+"=%{y:.5f}<extra></extra>",
    ))
    # compute per-dim limits for legend display (we won't draw per-dim lines to avoid clutter)
    c, lo, hi, s_hat = individuals_limits(y, sigma_mult)
    centers[dim], lcls[dim], ucls[dim], sigmas[dim] = c, lo, hi, s_hat

# If exactly one DIM, draw its control lines; if many, show an average center line for context
if len(dim_picks) == 1:
    d = dim_picks[0]
    fig_i.add_hline(y=centers[d], line_dash="dash", annotation_text=f"{d} Center {centers[d]:.5f}")
    fig_i.add_hline(y=ucls[d], line_dash="dot",  annotation_text=f"{d} UCL {ucls[d]:.5f}")
    fig_i.add_hline(y=lcls[d], line_dash="dot",  annotation_text=f"{d} LCL {lcls[d]:.5f}")
    viol_mask = (get_series_for_dim(df, d, value_mode) < lcls[d]) | (get_series_for_dim(df, d, value_mode) > ucls[d])
    if viol_mask.any():
        v = df.loc[viol_mask]
        fig_i.add_trace(go.Scatter(
            x=v["timestamp"], y=get_series_for_dim(df, d, value_mode)[viol_mask],
            mode="markers", name=f"{d} Violations",
            marker=dict(size=10, symbol="x"),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>"+d+" out-of-control=%{y:.5f}<extra></extra>",
        ))
    title_suffix = f"{d} · μ≈{centers[d]:.5f} · σ̂≈{(sigmas[d] if pd.notna(sigmas[d]) else 0):.5f}"
else:
    # multi-dim: draw a grand mean line (of all plotted series means) for context
    means = [centers[d] for d in dim_picks if pd.notna(centers[d])]
    if means:
        gm = float(np.mean(means))
        fig_i.add_hline(y=gm, line_dash="dash", annotation_text=f"Grand mean {gm:.5f}")
    title_suffix = f"{len(dim_picks)} DIMs"

fig_i.update_layout(
    title=f"Individuals (I) — {value_mode} — {title_suffix}",
    xaxis_title="timestamp", yaxis_title="Value",
    height=460, hovermode="x unified", legend_title="Series",
)

st.plotly_chart(fig_i, use_container_width=True)

# ---------------- MR chart (single-dim only) ----------------
if len(dim_picks) == 1:
    d = dim_picks[0]
    y = get_series_for_dim(df, d, value_mode)
    mr = y.diff().abs()
    Rbar = mr.mean()
    D3, D4 = 0.0, 3.267
    MR_UCL = D4 * Rbar
    MR_LCL = max(D3 * Rbar, 0.0)

    fig_mr = go.Figure()
    fig_mr.add_trace(go.Scatter(
        x=df["timestamp"], y=mr, mode="lines+markers", name=f"{d} MR(2)",
        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>MR=%{y:.5f}<extra></extra>",
    ))
    fig_mr.add_hline(y=Rbar,   line_dash="dash", annotation_text=f"{d} Ȓ {Rbar:.5f}")
    fig_mr.add_hline(y=MR_UCL, line_dash="dot",  annotation_text=f"{d} UCL {MR_UCL:.5f}")
    fig_mr.add_hline(y=MR_LCL, line_dash="dot",  annotation_text=f"{d} LCL {MR_LCL:.5f}")
    fig_mr.update_layout(
        title=f"Moving Range (MR) — {d}",
        xaxis_title="timestamp", yaxis_title="|ΔX|",
        height=320, hovermode="x unified", legend_title="Series",
    )
    st.plotly_chart(fig_mr, use_container_width=True)
else:
    st.caption("Select a single DIM to show its MR chart.")

# ---------------- Optional table & download ----------------
if show_table:
    st.subheader("Data preview")
    st.dataframe(df, use_container_width=True, height=360)

st.download_button(
    label="Download plotted data (CSV)",
    data=df.to_csv(index=False).encode(),
    file_name="spc_view_export.csv",
)

# ---------------- Safe auto-refresh ----------------
if refresh_s > 0:
    if _st_autorefresh is not None:
        _st_autorefresh(interval=int(refresh_s * 1000), key="spc-autorefresh")
    else:
        auto = getattr(st, "autorefresh", None)
        if callable(auto):
            auto(interval=int(refresh_s * 1000), key="spc-autorefresh")
        # older Streamlit: no-op; user can click Rerun

# streamlit_app.py
# -------------------------------------------------------------
# SPC viewer for Auto-Caliper capture logs
# • Source = Local folder (PC)  OR  GitHub repo path (Streamlit Cloud)
# • Finds newest CSV/XLSX (or newest for Order ID)
# • I-Chart + MR Chart with Rule-1 highlights
# • Safe auto-refresh (streamlit_autorefresh if available)
# • Debug expander shows what files were found
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
    # Preferred on Streamlit Cloud (JS-side timer)
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
except Exception:
    _st_autorefresh = None

st.set_page_config(page_title="SPC Viewer", layout="wide")

# ---------------- UI: Source selection ----------------
st.sidebar.header("Source")
source = st.sidebar.radio("Where are the logs?", ["Local folder", "GitHub repo"], index=0)

# Defaults
REPO_LOGS = (Path(__file__).resolve().parents[1] / "app_capture" / "Logs")
FALLBACK_LOGS = Path(r"C:\CaliperCapture\Logs")
DEFAULT_LOGS = str(REPO_LOGS if REPO_LOGS.exists() else FALLBACK_LOGS)

# Common filters
order_id = st.sidebar.text_input("Order ID (optional)", "").strip().upper()
profile_filter = st.sidebar.text_input("Profile filter (optional)", "").strip()
reel_filter = st.sidebar.text_input("Reel filter (optional)", "").strip().upper()

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

# ---------------- Local mode ----------------
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

# ---------------- GitHub mode ----------------
def gh_list_files(owner: str, repo: str, path: str, ref: str = "main") -> List[Dict]:
    """
    Returns list of items with fields: name, path, download_url (for public repos).
    """
    if requests is None:
        raise RuntimeError("requests is not installed. pip install requests")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if it.get("type") == "file" and str(it.get("name", "")).lower().endswith((".csv", ".xlsx")):
            out.append({"name": it["name"], "path": it["path"], "download_url": it.get("download_url", "")})
    return out

@st.cache_data
def gh_download_csv(download_url: str) -> pd.DataFrame:
    r = requests.get(download_url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data
def gh_download_xlsx(download_url: str) -> pd.DataFrame:
    r = requests.get(download_url, timeout=20)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content))

# ---------------- Source selection UI ----------------
if source == "Local folder":
    logs_dir = st.sidebar.text_input("Logs folder", DEFAULT_LOGS)
    # build pattern; be lenient, enforce prefix later
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
        st.error("No matching files found. Check the debug panel and clear the Order ID filter to see all files.")
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
    st.sidebar.write("Pull logs from a **public GitHub repo** path (CSV/XLSX).")
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

    # sort newest by name if names contain timestamps; we can't rely on mtime via API
    files = [it for it in listing if (order_id == "" or it["name"].upper().startswith(order_id + "_"))]
    if not files:
        st.error("No matching CSV/XLSX in the GitHub folder (check Order ID filter and path).")
        st.stop()

    # prefer CSV over XLSX and then sort descending by name (ORDERID_YYYYMMDD_HHMMSS)
    files_csv = [it for it in files if it["name"].lower().endswith(".csv")]
    files_xlsx = [it for it in files if it["name"].lower().endswith(".xlsx")]
    files = files_csv + files_xlsx
    files = sorted(files, key=lambda it: it["name"], reverse=True)

    pick_latest = st.sidebar.checkbox("Always use most recent file", True)
    chosen = files[0] if pick_latest else st.sidebar.selectbox("Choose file", files, index=0, format_func=lambda it: it["name"])

    if chosen["name"].lower().endswith(".xlsx"):
        df = gh_download_xlsx(chosen["download_url"])
    else:
        df = gh_download_csv(chosen["download_url"])

    file_label = f"{owner}/{repo}/{chosen['path']}"
    last_modified = "(GitHub fetch)"

# ---------------- Clean + Filters ----------------
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="T")

if profile_filter and "profile" in df.columns:
    df = df[df["profile"].astype(str).str.contains(profile_filter, case=False, na=False)]
if reel_filter and "reel" in df.columns:
    df = df[df["reel"].astype(str).str.upper() == reel_filter]

# Pick a numeric DIM column by default
dim_cols = [c for c in df.columns if c.upper().startswith("DIM ")]
num_dim_cols = [c for c in dim_cols if is_numeric_series(df[c])]
if num_dim_cols:
    series_col = num_dim_cols[0]
else:
    numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
    if not numeric_cols:
        st.error("No numeric measurement columns found (e.g., 'DIM 1').")
        st.stop()
    series_col = numeric_cols[-1]

st.subheader(f"File: {file_label}")
st.caption(f"Last modified: {last_modified}")
st.write(f"**Charting column:** `{series_col}`  · Rows: {len(df)}")

# Window
if window_pts and window_pts > 0:
    df = df.tail(int(window_pts)).copy()

x = pd.to_numeric(df[series_col], errors="coerce").astype(float)
mr = x.diff().abs()

center, lcl, ucl, sigma_hat = individuals_limits(x, sigma_mult)
Rbar = mr.mean()
D3, D4 = 0.0, 3.267
MR_UCL = D4 * Rbar
MR_LCL = max(D3 * Rbar, 0.0)

viol_mask = (x < lcl) | (x > ucl)

# ---------------- Plot: Individuals ----------------
fig_i = go.Figure()
fig_i.add_trace(go.Scatter(
    x=df["timestamp"], y=x, mode="lines+markers", name="Value",
    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Value=%{y:.5f}<extra></extra>",
))
fig_i.add_hline(y=center, line_dash="dash", annotation_text=f"Center {center:.5f}")
fig_i.add_hline(y=ucl, line_dash="dot",  annotation_text=f"UCL {ucl:.5f}")
fig_i.add_hline(y=lcl, line_dash="dot",  annotation_text=f"LCL {lcl:.5f}")

if viol_mask.any():
    v = df.loc[viol_mask]
    fig_i.add_trace(go.Scatter(
        x=v["timestamp"], y=v[series_col], mode="markers", name="Violations",
        marker=dict(size=10, symbol="x"),
        hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Out-of-control=%{y:.5f}<extra></extra>",
    ))

fig_i.update_layout(
    title=f"Individuals (I) · μ≈{center:.5f} · σ̂≈{(sigma_hat if pd.notna(sigma_hat) else 0):.5f}",
    xaxis_title="timestamp", yaxis_title=series_col,
    height=420, hovermode="x unified", legend_title="Series",
)

# ---------------- Plot: MR ----------------
fig_mr = go.Figure()
fig_mr.add_trace(go.Scatter(
    x=df["timestamp"], y=mr, mode="lines+markers", name="MR(2)",
    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>MR=%{y:.5f}<extra></extra>",
))
fig_mr.add_hline(y=Rbar,    line_dash="dash", annotation_text=f"Ȓ {Rbar:.5f}")
fig_mr.add_hline(y=MR_UCL,  line_dash="dot",  annotation_text=f"UCL {MR_UCL:.5f}")
fig_mr.add_hline(y=MR_LCL,  line_dash="dot",  annotation_text=f"LCL {MR_LCL:.5f}")
fig_mr.update_layout(
    title="Moving Range (MR)", xaxis_title="timestamp", yaxis_title="|ΔX|",
    height=320, hovermode="x unified", legend_title="Series",
)

st.plotly_chart(fig_i, use_container_width=True)
st.plotly_chart(fig_mr, use_container_width=True)

# ---------------- Optional table & download ----------------
if show_table:
    st.subheader("Data preview")
    st.dataframe(df, use_container_width=True, height=360)

st.download_button(
    label="Download plotted data (CSV)",
    data=df.to_csv(index=False).encode(),
    file_name=f"spc_view_export.csv",
)

# ---------------- Safe auto-refresh (no experimental_rerun) ----------------
if refresh_s > 0:
    if _st_autorefresh is not None:
        _st_autorefresh(interval=int(refresh_s * 1000), key="spc-autorefresh")
    else:
        auto = getattr(st, "autorefresh", None)
        if callable(auto):
            auto(interval=int(refresh_s * 1000), key="spc-autorefresh")
        # else: do nothing (older Streamlit). User can click Rerun.

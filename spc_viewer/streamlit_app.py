# streamlit_app.py
# -------------------------------------------------------------
# SPC viewer for Auto-Caliper capture logs
# • Source: Local folder  OR  GitHub repo (recursive)
# • Select multiple DIMs; choose Actual (corrected) vs Raw (caliper)
# • X-axis toggle: Sequence (even spacing) or Timestamp; optional reset per Reel
# • I-Chart + MR (MR shown when exactly one DIM is selected)
# • NEW: Draw USL/LSL from YAML config (per DIM) when plotting Actual values
# • Auto-locate YAML by part/profile name (e.g., SP-001-25.yml / .yaml)
# -------------------------------------------------------------

import io
import os
import re
import time
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path, PurePosixPath
from typing import Optional, Tuple, List, Dict

# Optional libs (only needed for GitHub mode / Cloud refresh)
try:
    import requests
except Exception:
    requests = None

try:
    import yaml
except Exception:
    yaml = None

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
except Exception:
    _st_autorefresh = None

st.set_page_config(page_title="SPC Viewer", layout="wide")

# ---------------- Sidebar: Source & controls ----------------
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

st.sidebar.header("X-Axis")
x_mode = st.sidebar.radio("Plot against", ["Sequence", "Timestamp"], index=0)
reset_seq_per_reel = st.sidebar.checkbox("Reset sequence per Reel", False)

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

def infer_part_label(df: pd.DataFrame, file_label: str) -> str:
    # 1) Try standard columns
    for col in ("part_id", "profile"):
        if col in df.columns:
            vals = df[col].dropna().astype(str)
            if not vals.empty:
                v = vals.iloc[0]
                m = re.search(r"SP-\d{3}-\d{2}", v, flags=re.IGNORECASE)
                if m:
                    return m.group(0).upper()
                if v.upper().startswith("SP-"):
                    return v.upper()
    # 2) Scan first few rows for SP-###-##
    try:
        sample = " ".join(map(str, df.head(10).fillna("").values.flatten()))
        m = re.search(r"SP-\d{3}-\d{2}", sample, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
    except Exception:
        pass
    # 3) Look in the path/label
    try:
        parts = list(Path(file_label).parts) + list(PurePosixPath(file_label).parts)
        for seg in reversed(parts):
            m = re.search(r"SP-\d{3}-\d{2}", seg, flags=re.IGNORECASE)
            if m:
                return m.group(0).upper()
    except Exception:
        pass
    return ""

# ---------------- Local mode ----------------
def list_files_debug_local(folder: str, patt: str) -> Dict:
    info = {"mode": "local", "folder": folder, "pattern": patt, "exists": False,
            "files_all": [], "files_match": [], "error": None}
    try:
        info["exists"] = os.path.isdir(folder)
        if info["exists"]:
            # RECURSIVE: include subfolders
            all_files = [str(p) for p in Path(folder).rglob("*") if p.is_file()]
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

# ---------------- GitHub mode (recursive) ----------------
def gh_list_files_recursive(owner: str, repo: str, base_path: str, ref: str = "main"):
    """
    Recursively list CSV/XLSX files under base_path (public repos) using Git Trees API.
    Returns items: {"name", "path", "download_url"} (raw URL).
    """
    if requests is None:
        raise RuntimeError("requests not installed. pip install requests")
    base_path = base_path.strip("/")
    api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}"
    r = requests.get(api, params={"recursive": "1"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    tree = data.get("tree", [])
    if not isinstance(tree, list):
        return []
    results = []
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path", "")
        if not path.lower().startswith(base_path.lower() + "/"):
            continue
        if not path.lower().endswith((".csv", ".xlsx")):
            continue
        name = path.split("/")[-1]
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        results.append({"name": name, "path": path, "download_url": raw_url})
    results.sort(key=lambda it: it["name"], reverse=True)
    return results

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
    st.sidebar.write("Pull from a **public GitHub repo** (recursive under folder path)")
    owner = st.sidebar.text_input("GitHub owner/org", "your-org-or-user").strip()
    repo  = st.sidebar.text_input("Repo", "caliper-logs").strip()
    folder_path = st.sidebar.text_input("Folder path in repo", "app_capture/Logs").strip()
    branch = st.sidebar.text_input("Branch/Ref", "main").strip()

    if not (owner and repo and folder_path):
        st.info("Enter owner, repo, and folder path.")
        st.stop()

    try:
        listing = gh_list_files_recursive(owner, repo, folder_path, ref=branch)
    except Exception as e:
        st.error(f"GitHub listing failed:\n{e}")
        st.stop()

    with st.expander("🔎 Debug (GitHub files)", expanded=False):
        st.write("Base path:", folder_path)
        st.write("Total matches:", len(listing))
        st.write(pd.DataFrame(listing[:10]))

    files = [it for it in listing if (order_id == "" or it["name"].upper().startswith(order_id + "_"))]
    if not files:
        st.error("No matching CSV/XLSX in the GitHub folder (check Order ID and path).")
        st.stop()

    files_csv = [it for it in files if it["name"].lower().endswith(".csv")]
    files_xlsx = [it for it in files if it["name"].lower().endswith(".xlsx")]
    files = files_csv + files_xlsx  # already sorted by name desc

    pick_latest = st.sidebar.checkbox("Always use most recent file", True)
    chosen = files[0] if pick_latest else st.sidebar.selectbox("Choose file", files, index=0, format_func=lambda it: it["name"])
    ext = ".xlsx" if chosen["name"].lower().endswith(".xlsx") else ".csv"
    df = gh_download_any(chosen["download_url"], ext)
    file_label = f"{owner}/{repo}/{chosen['path']}"
    last_modified = "(GitHub fetch)"

# ---------------- Clean, filters, sequence ----------------
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="T")

if profile_filter and "profile" in df.columns:
    df = df[df["profile"].astype(str).str.contains(profile_filter, case=False, na=False)]
if reel_filter and "reel" in df.columns:
    df = df[df["reel"].astype(str).str.upper() == reel_filter]

# Build sequence index (1..N). Optionally reset per Reel.
if reset_seq_per_reel and "reel" in df.columns:
    df = df.copy()
    df["seq"] = (df.groupby(df["reel"].astype(str), dropna=False).cumcount() + 1)
else:
    df = df.copy()
    df["seq"] = range(1, len(df) + 1)

x_field = "seq" if x_mode == "Sequence" else "timestamp"
x_label = "sequence" if x_field == "seq" else "timestamp"

# Infer part label for chart titles
part_name = infer_part_label(df, file_label)
chart_title_prefix = f"{part_name} — " if part_name else ""

# ---------------- Specs (LSL/USL) ----------------
st.sidebar.header("Specs (USL/LSL)")

if source == "Local folder":
    default_cfg_dir = str((Path(__file__).resolve().parents[1] / "app_capture" / "configs")) \
                      if (Path(__file__).resolve().parents[1] / "app_capture" / "configs").exists() \
                      else r"C:\CaliperCapture\configs"
    cfg_dir_local = st.sidebar.text_input("Config folder (local)", default_cfg_dir)
    cfg_repo_dir = None
else:
    cfg_dir_local = None
    cfg_repo_dir = st.sidebar.text_input("Config folder in repo", "app_capture/configs").strip()
    cfg_owner = owner
    cfg_repo = repo
    cfg_ref = branch

@st.cache_data
def load_limits_from_yaml_local(cfg_dir: str, part: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out = {}
    if not yaml:
        return out
    if not part:
        return out
    base = Path(cfg_dir)
    # try both .yml and .yaml
    for ext in (".yml", ".yaml"):
        p = base / f"{part}{ext}"
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                feats = data.get("features", [])
                for f in feats:
                    name = str(f.get("name", "")).strip()
                    if name:
                        out[name.upper()] = (f.get("lsl", None), f.get("usl", None))
            except Exception:
                pass
            break
    return out

@st.cache_data
def load_limits_from_yaml_github(owner: str, repo: str, cfg_folder: str, part: str, ref: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out = {}
    if not (requests and yaml and part):
        return out
    # construct raw URL: https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{cfg_folder}/{part}.yml
    for ext in (".yml", ".yaml"):
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{cfg_folder.strip('/')}/{part}{ext}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            try:
                data = yaml.safe_load(r.text)
                feats = data.get("features", [])
                for f in feats:
                    name = str(f.get("name", "")).strip()
                    if name:
                        out[name.upper()] = (f.get("lsl", None), f.get("usl", None))
            except Exception:
                pass
            break
    return out

# Build limits map { "DIM 1": (lsl, usl), ... } (case-insensitive by key)
limits_by_dim: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
if part_name:
    if source == "Local folder" and cfg_dir_local:
        limits_by_dim = load_limits_from_yaml_local(cfg_dir_local, part_name)
    elif source == "GitHub repo" and cfg_repo_dir:
        limits_by_dim = load_limits_from_yaml_github(cfg_owner, cfg_repo, cfg_repo_dir, part_name, cfg_ref)

# ---------------- DIM selection ----------------
# Actual columns are "DIM n"; Raw columns are "_raw_DIM n"
dim_actual = [c for c in df.columns if c.upper().startswith("DIM ")]
dims_available: List[str] = []
for c in sorted(dim_actual, key=lambda s: (int(''.join(ch for ch in s if ch.isdigit()) or 1), s)):
    dims_available.append(c)
for c in df.columns:
    if c.startswith("_raw_DIM "):
        base = c.replace("_raw_", "")
        if base not in dims_available:
            dims_available.append(base)

if not dims_available:
    st.error("No DIM columns found.")
    st.stop()

dim_picks = st.sidebar.multiselect("Select DIMs to plot", dims_available, default=dims_available[:1])
if not dim_picks:
    st.info("Select at least one DIM to plot from the sidebar.")
    st.stop()

# Window
if window_pts and window_pts > 0:
    df = df.tail(int(window_pts)).copy()

def get_series_for_dim(frame: pd.DataFrame, dim_name: str, mode: str) -> pd.Series:
    if mode.startswith("Actual"):
        col = dim_name
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").astype(float)
        raw_col = "_raw_" + dim_name
        if raw_col in frame.columns:
            st.warning(f"{dim_name}: Actual column missing; using RAW as fallback")
            return pd.to_numeric(frame[raw_col], errors="coerce").astype(float)
        return pd.Series(dtype=float)
    else:
        raw_col = "_raw_" + dim_name
        if raw_col in frame.columns:
            return pd.to_numeric(frame[raw_col], errors="coerce").astype(float)
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
        x=df["seq"] if x_mode == "Sequence" else df["timestamp"],
        y=y, mode="lines+markers", name=dim,
        hovertemplate=(
            "%{x}<br>"+dim+"=%{y:.5f}<extra></extra>"
            if x_mode == "Sequence"
            else "%{x|%Y-%m-%d %H:%M:%S}<br>"+dim+"=%{y:.5f}<extra></extra>"
        ),
    ))
    c, lo, hi, s_hat = individuals_limits(y, sigma_mult)
    centers[dim], lcls[dim], ucls[dim], sigmas[dim] = c, lo, hi, s_hat

# Draw per-DIM center/UCL/LCL or grand mean
if len(dim_picks) == 1:
    d = dim_picks[0]
    fig_i.add_hline(y=centers[d], line_dash="dash", annotation_text=f"{d} Center {centers[d]:.5f}")
    fig_i.add_hline(y=ucls[d], line_dash="dot",  annotation_text=f"{d} UCL {ucls[d]:.5f}")
    fig_i.add_hline(y=lcls[d], line_dash="dot",  annotation_text=f"{d} LCL {lcls[d]:.5f}")
    y_d = get_series_for_dim(df, d, value_mode)
    viol_mask = (y_d < lcls[d]) | (y_d > ucls[d])
    if viol_mask.any():
        v = df.loc[viol_mask]
        fig_i.add_trace(go.Scatter(
            x=v["seq"] if x_mode == "Sequence" else v["timestamp"],
            y=y_d[viol_mask], mode="markers", name=f"{d} Violations",
            marker=dict(size=10, symbol="x"),
            hovertemplate=(
                "%{x}<br>"+d+" out-of-control=%{y:.5f}<extra></extra>"
                if x_mode == "Sequence"
                else "%{x|%Y-%m-%d %H:%M:%S}<br>"+d+" out-of-control=%{y:.5f}<extra></extra>"
            ),
        ))
    title_suffix = f"{d} · μ≈{centers[d]:.5f} · σ̂≈{(sigmas[d] if pd.notna(sigmas[d]) else 0):.5f}"
else:
    means = [centers[d] for d in dim_picks if pd.notna(centers[d])]
    if means:
        gm = float(np.mean(means))
        fig_i.add_hline(y=gm, line_dash="dash", annotation_text=f"Grand mean {gm:.5f}")
    title_suffix = f"{len(dim_picks)} DIMs"

# ---- NEW: Spec lines (USL/LSL) when in Actual mode ----
if value_mode.startswith("Actual"):
    spec_lines_drawn = False
    for dim in dim_picks:
        limits = limits_by_dim.get(_norm_dim_key(dim))
        if limits:
            lsl, usl = limits
            if lsl is not None:
                fig_i.add_hline(y=float(lsl), line_dash="solid", line_color="gray",
                                annotation_text=f"{dim} LSL {float(lsl):.5f}")
                spec_lines_drawn = True
            if usl is not None:
                fig_i.add_hline(y=float(usl), line_dash="solid", line_color="gray",
                                annotation_text=f"{dim} USL {float(usl):.5f}")
                spec_lines_drawn = True
    if not spec_lines_drawn:
        st.caption("Specs: no matching LSL/USL found in YAML for selected DIMs.")
else:
    st.caption("Specs hidden: switch to **Actual (corrected)** to view USL/LSL.")
    
fig_i.update_layout(
    title=f"{(infer_part_label(df, file_label) + ' — ') if infer_part_label(df, file_label) else ''}Individuals (I) — {value_mode} — {title_suffix}",
    xaxis_title=("sequence" if x_mode == "Sequence" else "timestamp"),
    yaxis_title="Value",
    height=460, hovermode="x unified", legend_title="Series",
)

st.plotly_chart(fig_i, use_container_width=True)

# ---------------- MR chart (single-DIM only) ----------------
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
        x=df["seq"] if x_mode == "Sequence" else df["timestamp"],
        y=mr, mode="lines+markers", name=f"{d} MR(2)",
        hovertemplate=(
            "%{x}<br>MR=%{y:.5f}<extra></extra>"
            if x_mode == "Sequence"
            else "%{x|%Y-%m-%d %H:%M:%S}<br>MR=%{y:.5f}<extra></extra>"
        ),
    ))
    fig_mr.add_hline(y=Rbar,   line_dash="dash", annotation_text=f"{d} Ȓ {Rbar:.5f}")
    fig_mr.add_hline(y=MR_UCL, line_dash="dot",  annotation_text=f"{d} UCL {MR_UCL:.5f}")
    fig_mr.add_hline(y=MR_LCL, line_dash="dot",  annotation_text=f"{d} LCL {MR_LCL:.5f}")
    fig_mr.update_layout(
        title=f"{infer_part_label(df, file_label)+' — ' if infer_part_label(df, file_label) else ''}Moving Range (MR) — {d}",
        xaxis_title=("sequence" if x_mode == "Sequence" else "timestamp"),
        yaxis_title="|ΔX|",
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
        # On older Streamlit: no-op; click Rerun manually.

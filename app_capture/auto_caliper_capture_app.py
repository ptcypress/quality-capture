"""
Auto-Caliper Capture — Profile/Operator-Pro Edition
---------------------------------------------------
• Three (or more) product profiles via ./configs/*.yml
• Operator Mode (simple) / Admin Mode (advanced)
• Remembers last used profile, output dir, port, baud, log mode (settings.json)
• Config validation + duplicate name checks
• Virtual zero (zero_at + zero_mode) with live OK/LOW/HIGH
• Shows Dim, Caliper target (raw), Raw, Corrected, Result, LSL, USL
• CORRECTED values are saved to CSV/XLSX
• Per-Order naming: ORDERID_YYYYMMDD_HHMMSS
• Uppercases only operator/order/reel/part inputs
"""

from __future__ import annotations
import sys, json, csv, threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    print("Tkinter is required:", e); sys.exit(1)

# Optional deps (app still runs in wedge mode without serial/pandas/yaml)
try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None; list_ports = None

try:
    import yaml
except Exception:
    yaml = None

try:
    import pandas as pd
except Exception:
    pd = None


APP_DIR = Path(__file__).resolve().parent
LOGS_DIR = APP_DIR / "Logs"
CONFIGS_DIR = APP_DIR / "configs"
SETTINGS_FILE = APP_DIR / "settings.json"


@dataclass
class Feature:
    name: str
    lsl: Optional[float]
    usl: Optional[float]
    zero_at: float = 0.0             # pin size to subtract
    zero_mode: str = "diameter"      # 'diameter' | 'radius'
    nominal: Optional[float] = None  # corrected target (optional)
    caliper_target: Optional[float] = None  # raw target (optional)

    def judge(self, x: float) -> str:
        if self.lsl is not None and x < self.lsl: return "LOW"
        if self.usl is not None and x > self.usl: return "HIGH"
        return "OK"

    def _pin_adjust(self) -> float:
        try:
            z = float(self.zero_at or 0.0)
        except Exception:
            z = 0.0
        if (self.zero_mode or "diameter").lower().startswith("rad"):
            z = z / 2.0
        return z

    def correct(self, raw: float) -> float:
        return raw - self._pin_adjust()

    def caliper_target_value(self) -> Optional[float]:
        if self.caliper_target is not None:
            return self.caliper_target
        if self.nominal is not None:
            return self.nominal + self._pin_adjust()
        return None


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_settings(d: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


def discover_profiles() -> List[Path]:
    CONFIGS_DIR.mkdir(exist_ok=True)
    roots = list(CONFIGS_DIR.glob("*.yml")) + list(CONFIGS_DIR.glob("*.yaml"))
    # include root-level default if present
    root_default = APP_DIR / "caliper_config.yml"
    if root_default.exists():
        roots = [root_default] + roots
    return roots


def validate_config(data: dict) -> tuple[List[Feature], List[str]]:
    errs: List[str] = []
    feats: List[Feature] = []
    seen_names = set()

    if not isinstance(data, dict) or "features" not in data or not isinstance(data["features"], list):
        return [], ["Config must have a 'features' list."]

    for idx, item in enumerate(data["features"], start=1):
        if not isinstance(item, dict):
            errs.append(f"Feature #{idx} must be a mapping/object."); continue
        name = (item.get("name") or f"DIM {idx}").strip()
        if not name:
            errs.append(f"Feature #{idx} has empty name."); continue
        if name in seen_names:
            errs.append(f"Duplicate feature name: '{name}'.")
        seen_names.add(name)

        # numeric fields
        def _num(k):
            v = item.get(k, None)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                errs.append(f"{name}: '{k}' must be numeric.")
                return None

        lsl = _num("lsl")
        usl = _num("usl")
        zero_at = float(item.get("zero_at") or item.get("offset") or 0.0)
        zero_mode = (item.get("zero_mode") or "diameter").lower()
        if zero_mode not in ("diameter", "radius"):
            errs.append(f"{name}: zero_mode must be 'diameter' or 'radius'.")

        nominal = _num("nominal")
        ct = _num("caliper_target")

        feats.append(Feature(name=name.upper(), lsl=lsl, usl=usl,
                             zero_at=zero_at, zero_mode=zero_mode,
                             nominal=nominal, caliper_target=ct))
    return feats, errs


class SerialReader(threading.Thread):
    def __init__(self, port: str, baud: int, cb_line):
        super().__init__(daemon=True)
        self.port_name = port; self.baud = baud; self.cb_line = cb_line
        self.stop_flag = False

    def run(self):
        try:
            ser = serial.Serial(self.port_name, self.baud, timeout=0.1)
        except Exception as e:
            self.cb_line(f"__ERROR__:{e}"); return
        buf = bytearray()
        while not self.stop_flag:
            try:
                b = ser.read(1)
                if not b: continue
                if b in (b"\n", b"\r"):
                    if buf:
                        line = buf.decode(errors="ignore").strip(); buf.clear()
                        if line: self.cb_line(line)
                else:
                    buf.extend(b)
            except Exception as e:
                self.cb_line(f"__ERROR__:{e}"); break
        try: ser.close()
        except Exception: pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Auto-Caliper Capture — Profiles")
        self.geometry("1180x720")

        LOGS_DIR.mkdir(exist_ok=True)

        # Tk vars
        self.settings = load_settings()
        self.output_dir = tk.StringVar(value=self.settings.get("output_dir", str(LOGS_DIR)))
        self.session_name = tk.StringVar(value=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
        self.operator = tk.StringVar(value="")
        self.part_id = tk.StringVar()
        self.baud = tk.IntVar(value=int(self.settings.get("baud", 9600)))
        self.port = tk.StringVar(value=self.settings.get("port", "(keyboard-wedge)"))
        self.status = tk.StringVar(value="Select a product profile and start.")
        self.auto_commit = tk.BooleanVar(value=bool(self.settings.get("auto_commit", True)))
        self.log_mode = tk.StringVar(value=self.settings.get("log_mode", "order"))  # session|order|append_file
        self.order_id = tk.StringVar()
        self.reel_id = tk.StringVar()
        self.append_path = tk.StringVar(value=self.settings.get("append_path", ""))
        self.admin_mode = tk.BooleanVar(value=False)

        # Profiles
        self.profiles = discover_profiles()
        self.profile_map: Dict[str, Path] = {p.name: p for p in self.profiles}
        last_profile_name = self.settings.get("profile_name", "")
        default_profile = last_profile_name if last_profile_name in self.profile_map else (self.profiles[0].name if self.profiles else "")
        self.profile_name = tk.StringVar(value=default_profile)

        # Capture state
        self.serial_thread: Optional[SerialReader] = None
        self.features: List[Feature] = []
        self.active_idx = 0
        self.current_row: Dict[str, float | str] = {}
        self.armed = False
        self._arm_win: Optional[tk.Toplevel] = None

        # Per-order filename base
        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._order_file_base: Optional[str] = None

        self._build_ui()
        self._install_uppercase_hooks()
        self._bind_admin_hotkey()

        self._load_selected_profile()
        self._refresh_ports()

    # UI
    def _build_ui(self):
        # Top: Profile + Admin toggle
        top = ttk.Frame(self); top.pack(fill=tk.X, padx=12, pady=(10,6))
        ttk.Label(top, text="Product Profile:").pack(side=tk.LEFT)
        self.profile_combo = ttk.Combobox(top, textvariable=self.profile_name, width=40, values=[p.name for p in self.profiles])
        self.profile_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load", command=self._load_selected_profile).pack(side=tk.LEFT, padx=(2,10))
        ttk.Button(top, text="Reload Config", command=self._reload_profile).pack(side=tk.LEFT)
        ttk.Checkbutton(top, text="Admin Mode", variable=self.admin_mode, command=self._toggle_admin).pack(side=tk.RIGHT)

        # Advanced/Admin bar
        adv = ttk.LabelFrame(self, text="Admin / Advanced"); adv.pack(fill=tk.X, padx=12, pady=6)
        self._admin_frame = adv
        ttk.Label(adv, text="Output dir:").pack(side=tk.LEFT)
        ttk.Entry(adv, textvariable=self.output_dir, width=36).pack(side=tk.LEFT, padx=4)
        ttk.Button(adv, text="…", command=self._browse_out).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(adv, text="Port:").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(adv, textvariable=self.port, width=24)
        self.port_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(adv, text="Refresh", command=self._refresh_ports).pack(side=tk.LEFT)
        ttk.Label(adv, text="Baud:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Entry(adv, textvariable=self.baud, width=8).pack(side=tk.LEFT)
        ttk.Radiobutton(adv, text="Per session", variable=self.log_mode, value="session").pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(adv, text="Per order", variable=self.log_mode, value="order").pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(adv, text="Append file", variable=self.log_mode, value="append_file").pack(side=tk.LEFT, padx=8)
        ttk.Entry(adv, textvariable=self.append_path, width=36).pack(side=tk.LEFT, padx=4)
        ttk.Button(adv, text="Browse", command=self._browse_append_file).pack(side=tk.LEFT)

        # Operator panel
        op = ttk.LabelFrame(self, text="Operator"); op.pack(fill=tk.X, padx=12, pady=6)
        ttk.Label(op, text="Operator:").pack(side=tk.LEFT)
        ttk.Entry(op, textvariable=self.operator, width=16).pack(side=tk.LEFT, padx=4)
        ttk.Label(op, text="Order ID:").pack(side=tk.LEFT)
        ttk.Entry(op, textvariable=self.order_id, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Label(op, text="Reel:").pack(side=tk.LEFT)
        ttk.Entry(op, textvariable=self.reel_id, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(op, text="Part ID:").pack(side=tk.LEFT)
        ttk.Entry(op, textvariable=self.part_id, width=20).pack(side=tk.LEFT, padx=4)
        ttk.Button(op, text="New Order", command=self._new_order).pack(side=tk.LEFT, padx=(12,2))

        # Device control
        dev = ttk.LabelFrame(self, text="Device"); dev.pack(fill=tk.X, padx=12, pady=6)
        ttk.Label(dev, text="Mode:").pack(side=tk.LEFT)
        ttk.Label(dev, textvariable=self.port).pack(side=tk.LEFT, padx=6)
        ttk.Button(dev, text="Start", command=self.start_capture).pack(side=tk.LEFT, padx=8)
        ttk.Button(dev, text="Stop", command=self.stop_capture).pack(side=tk.LEFT)
        ttk.Checkbutton(dev, text="Auto-commit when all features captured", variable=self.auto_commit).pack(side=tk.LEFT, padx=10)

        # Table
        mid = ttk.Frame(self); mid.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.tree = ttk.Treeview(
            mid,
            columns=("Dim", "Caliper target", "Raw", "Corrected", "Result", "LSL", "USL"),
            show="headings",
        )
        for col, w in (("Dim",120), ("Caliper target",130), ("Raw",130), ("Corrected",150), ("Result",100), ("LSL",80), ("USL",80)):
            self.tree.heading(col, text=col); self.tree.column(col, width=w, anchor=tk.CENTER)
        self.tree.tag_configure("OK",    background="#eaffea")
        self.tree.tag_configure("LOW",   background="#ffecec")
        self.tree.tag_configure("HIGH",  background="#ffecec")
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Bottom
        bottom = ttk.Frame(self); bottom.pack(fill=tk.X, padx=12, pady=8)
        ttk.Button(bottom, text="Arm read (Enter)", command=self._arm).pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom, text="Commit row", command=self._commit_row).pack(side=tk.LEFT, padx=6)
        self.status_lbl = ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W)
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X)

        # Start in Operator Mode (hide admin panel)
        self._toggle_admin()

        # Bind Enter to arm
        self.bind("<Return>", self._global_return)

    def _bind_admin_hotkey(self):
        def _toggle(_e=None): self.admin_mode.set(not self.admin_mode.get()); self._toggle_admin()
        self.bind("<Control-Shift-A>", _toggle)

    def _toggle_admin(self):
        if self.admin_mode.get():
            self._admin_frame.pack(fill=tk.X, padx=12, pady=6)
        else:
            self._admin_frame.pack_forget()

    def _install_uppercase_hooks(self):
        for var in (self.operator, self.part_id, self.order_id, self.reel_id):
            def _mk(v=var):
                def cb(*_):
                    try:
                        s = v.get()
                        if s != s.upper(): v.set(s.upper())
                    except Exception: pass
                return cb
            var.trace_add("write", _mk())

    # Profiles
    def _load_selected_profile(self):
        name = self.profile_name.get().strip()
        p = self.profile_map.get(name)
        if not p:
            # Nothing selected—show minimal default
            self.features = [Feature("DIM 1", None, None)]
            self._rebuild_table()
            self.status.set("Select a product profile (configs/*.yml).")
            return
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) if yaml else {}
        except Exception as e:
            messagebox.showerror("Config error", f"Couldn't read {p.name}:\n{e}")
            return
        feats, errs = validate_config(data)
        if errs:
            messagebox.showerror("Config validation", "Errors in config:\n- " + "\n- ".join(errs))
            return
        self.features = feats
        self._rebuild_table()
        self.status.set(f"Loaded profile: {p.name} ({len(self.features)} features).")
        # remember in settings
        self.settings["profile_name"] = name; save_settings(self.settings)

    def _reload_profile(self):
        self._load_selected_profile()

    # Ports/Output
    def _refresh_ports(self):
        vals = ["(keyboard-wedge)"]
        if list_ports:
            vals += [p.device for p in list_ports.comports()]
        if self.port.get() not in vals:
            self.port.set("(keyboard-wedge)")
        if hasattr(self, "port_combo"):
            self.port_combo["values"] = vals

    def _browse_out(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_dir.set(d)
            self.settings["output_dir"] = d; save_settings(self.settings)

    def _browse_append_file(self):
        path = filedialog.askopenfilename(title="Select CSV to append", filetypes=[("CSV","*.csv"), ("All","*.*")])
        if path:
            self.append_path.set(path)
            self.settings["append_path"] = path; save_settings(self.settings)

    # Table
    def _rebuild_table(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for f in self.features:
            tgt = f.caliper_target_value()
            tgt_str = f"{tgt:.5f}" if tgt is not None else ""
            self.tree.insert("", tk.END, iid=f.name, values=(f.name, tgt_str, "", "", "", f.lsl, f.usl))

        # reset state for a new row
        self.active_idx = 0
        self.current_row.clear()

    # Capture
    def start_capture(self):
        self.settings["baud"] = self.baud.get()
        self.settings["port"] = self.port.get()
        self.settings["auto_commit"] = self.auto_commit.get()
        self.settings["log_mode"] = self.log_mode.get()
        save_settings(self.settings)

        if self.port.get() == "(keyboard-wedge)":
            self.status.set("Keyboard-wedge mode. Click Arm, then press DATA.")
            return
        if serial is None:
            messagebox.showerror("Missing dependency", "pyserial required for serial capture."); return
        self.serial_thread = SerialReader(self.port.get(), self.baud.get(), self._on_serial_line)
        self.serial_thread.start()
        self.status.set(f"Listening on {self.port.get()} @ {self.baud.get()} baud…")

    def stop_capture(self):
        if self.serial_thread:
            self.serial_thread.stop_flag = True
            self.serial_thread = None
        self.status.set("Stopped.")

    def _global_return(self, _e):
        if not self.armed: self._arm()

    def _arm(self):
        self.armed = True
        if self.port.get() == "(keyboard-wedge)":
            if self._arm_win is None or not tk.Toplevel.winfo_exists(self._arm_win):
                self._arm_win = tk.Toplevel(self); self._arm_win.title("Send measurement…")
                self._arm_win.grab_set()
                ttk.Label(self._arm_win, text="Press DATA on caliper (or type) then Enter").pack(padx=10, pady=(10,4))
                self._arm_var = tk.StringVar()
                ent = ttk.Entry(self._arm_win, textvariable=self._arm_var, width=26)
                ent.pack(padx=10, pady=(0,10)); ent.focus_set()
                ent.bind("<Return>", lambda e: self._accept_wedge())
                ttk.Button(self._arm_win, text="Cancel", command=self._cancel_wedge).pack(pady=(0,10))
            else:
                self._arm_win.lift()
            self.status.set("Armed (wedge): waiting for reading…")
        else:
            self.status.set("Armed: waiting for serial reading…")

    def _cancel_wedge(self):
        self.armed = False
        if self._arm_win and tk.Toplevel.winfo_exists(self._arm_win):
            self._arm_win.destroy()
        self._arm_win = None; self.status.set("Cancelled.")

    def _accept_wedge(self):
        val = self._arm_var.get().strip() if hasattr(self, "_arm_var") else ""
        try: x = float(val)
        except ValueError:
            messagebox.showerror("Parse error", f"Couldn't parse '{val}' as a number."); return
        if self._arm_win and tk.Toplevel.winfo_exists(self._arm_win):
            self._arm_win.destroy()
        self._arm_win = None; self._handle_measurement(x)

    def _on_serial_line(self, line: str):
        if line.startswith("__ERROR__:"):
            self.after(0, lambda: self.status.set(line)); return
        cleaned = line.strip().replace("in","").replace("mm","").strip()
        try: x = float(cleaned)
        except ValueError: return
        self.after(0, lambda: self._handle_measurement(x))

    def _handle_measurement(self, value: float):
        if not self.features or not self.armed: return
        f = self.features[self.active_idx]
        corrected = f.correct(value)
        result = f.judge(corrected)
        tgt = f.caliper_target_value(); tgt_str = f"{tgt:.5f}" if tgt is not None else ""

        self.tree.set(f.name, "Caliper target", tgt_str)
        self.tree.set(f.name, "Raw", f"{value:.5f}")
        self.tree.set(f.name, "Corrected", f"{corrected:.5f}")
        self.tree.set(f.name, "Result", result)
        self.tree.item(f.name, tags=(result,))

        self.current_row[f.name] = corrected
        self.current_row["_raw_"+f.name] = value
        self.current_row["_result_"+f.name] = result

        self.active_idx += 1
        self.armed = False

        if self.active_idx >= len(self.features):
            if self.auto_commit.get():
                self.status.set("All features captured — auto-committing row…"); self._commit_row()
            else:
                self.active_idx = len(self.features)-1
                self.status.set("All features measured. Click Commit row.")
        else:
            nxt = self.features[self.active_idx].name
            self.status.set(f"Captured {f.name}. Arm next: {nxt}.")

    # Order & save
    def _new_order(self):
        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._order_file_base = None
        self.status.set("New order session initialized.")

    def _commit_row(self):
        pid = (self.part_id.get() or "").strip().upper()
        if not pid:
            messagebox.showwarning("Missing Part ID", "Enter or scan a Part ID."); return
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "operator": (self.operator.get() or "").strip().upper(),
            "part_id": pid,
            "order_id": (self.order_id.get() or "").strip().upper(),
            "reel": (self.reel_id.get() or "").strip().upper(),
            "profile": self.profile_name.get(),
        }
        for f in self.features:
            row[f.name] = self.current_row.get(f.name, "")
            row[f"{f.name}_result"] = self.current_row.get("_result_"+f.name, "")
        self._save_row(row)
        self.current_row.clear(); self.active_idx = 0; self._rebuild_table()
        self.status.set("Row saved. Ready for next part.")

    def _save_row(self, row: dict):
        # destination
        try:
            base_dir = Path(self.output_dir.get()).expanduser()
        except Exception:
            base_dir = LOGS_DIR
        base_dir.mkdir(parents=True, exist_ok=True)

        mode = self.log_mode.get()
        if mode == "append_file":
            if not self.append_path.get().strip():
                messagebox.showwarning("No file selected", "Choose a CSV file to append or switch logging mode."); return
            out_csv = Path(self.append_path.get()); out_base = out_csv.with_suffix("")
        elif mode == "order":
            oid = (row.get("order_id","") or "").strip().upper()
            if not oid:
                messagebox.showwarning("Missing Order ID", "Enter an Order ID for 'Per Order' logging mode."); return
            if not self._order_file_base:
                self._order_file_base = f"{oid}_{self.session_ts}"
            out_base = base_dir / self._order_file_base; out_csv = out_base.with_suffix(".csv")
        else:
            out_base = base_dir / f"{self.session_name.get()}"; out_csv = out_base.with_suffix(".csv")

        header = list(row.keys())

        if out_csv.exists():
            try:
                with out_csv.open("r", encoding="utf-8", newline="") as rf:
                    first = rf.readline().strip()
                if first:
                    existing = [h.strip() for h in first.split(",")]
                    if set(existing) != set(header):
                        messagebox.showerror("Header mismatch", "Existing CSV columns differ from current config."); return
                    header = existing
            except Exception:
                pass

        def write_to(p: Path):
            exists = p.exists()
            with p.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if not exists: w.writeheader()
                w.writerow({k: row.get(k,"") for k in header})

        try:
            write_to(out_csv)
        except PermissionError:
            try:
                fb_dir = Path.home() / "Documents" / "CaliperLogs"; fb_dir.mkdir(parents=True, exist_ok=True)
                fb_csv = fb_dir / (out_csv.name.replace(".csv","_fallback.csv")); write_to(fb_csv)
                self.output_dir.set(str(fb_dir)); self.status.set(f"Output locked at {out_csv}. Wrote fallback: {fb_csv}")
                out_base = fb_csv.with_suffix(""); out_csv = fb_csv
            except Exception as e:
                messagebox.showerror("Save error", f"Couldn't write log file.\n{e}"); return
        except Exception as e:
            messagebox.showerror("Save error", f"Unhandled error writing CSV:\n{e}"); return

        if pd is not None:
            try:
                xlsx = out_base.with_suffix(".xlsx")
                if xlsx.exists():
                    old = pd.read_excel(xlsx); df = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
                else:
                    df = pd.DataFrame([row])
                df.to_excel(xlsx, index=False)
            except Exception:
                pass

        if mode in ("session","order"):
            self.session_name.set(out_base.name)

    # end App


if __name__ == "__main__":
    # Ensure default dirs; seed a default profile if none exist
    LOGS_DIR.mkdir(exist_ok=True); CONFIGS_DIR.mkdir(exist_ok=True)
    default_cfg = APP_DIR / "caliper_config.yml"
    if not default_cfg.exists():
        default_cfg.write_text(
            "features:\n"
            "  - { name: DIM 1, lsl: 0.560, usl: 0.620, zero_at: 0.1095, zero_mode: diameter, nominal: 0.5800, caliper_target: 0.6895 }\n"
            "  - { name: DIM 2, lsl: 4.005, usl: 4.035, zero_at: 0.219,  zero_mode: diameter }\n"
            "  - { name: DIM 3, lsl: 0.380, usl: 0.440, zero_at: 0.1095, zero_mode: diameter }\n"
            "  - { name: DIM 4, lsl: 4.945, usl: 5.095, zero_at: 0.0,    zero_mode: diameter }\n",
            encoding="utf-8"
        )
    app = App()
    app.mainloop()

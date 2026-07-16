#!/usr/bin/env python3
#
# windblade.py:   Mission Planner and Launcher GUI
# AUTHOR:         DANIEL DESAI
# UPDATED:        2026-07-16
# VERSION:        0.1.4

"""
Single-file entry point.  Run and a browser window opens with the
mission planner GUI.  Fill in the form and Launch — the script
passes weather (METARs) and cruise parameters directly to test_flight.py
as CLI args, reads subsystems/propulsion/rotor_config.csv for the rotor
fleet, injects overrides into test_card.json after planning, then runs
the sim. 

Usage
-----
    python3 windblade.py                    # open GUI on http://localhost:5780
    python3 windblade.py --port 8080        # alternate port
    python3 windblade.py --no-browser       # server only, open URL manually
    python3 windblade.py --preview-command  # GUI opens; Launch prints command only

Rotor fleet is defined in subsystems/propulsion/rotor_config.csv.
Edit that file and reload the Rotor Config tab to see changes.

Exit codes mirror test_flight.py:
    0   all checks passed
    1   one or more test_executive checks failed
    2   build failed
    3   sim failed / no CSV produced
    4   flight planning failed
   10   config payload invalid
   11   backup/restore error
   12   file write error
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import webbrowser
from datetime import datetime, UTC
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ── repo layout ───────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
PLANNING   = ROOT / "planning"
CONTROLS   = ROOT / "controls"
ROTOR_CSV  = ROOT / "subsystems" / "propulsion" / "rotor_config.csv"

# ── NVG terminal palette ──────────────────────────────────────────────────────
_ESC = "\033["
NC   = f"{_ESC}0m";  NW = f"{_ESC}96m"; GA = f"{_ESC}92m"
YL   = f"{_ESC}93m"; RD = f"{_ESC}91m"; BL = f"{_ESC}96m"
DIM  = f"{_ESC}2m";  BOLD = f"{_ESC}1m"
_BAR = "─" * 60

def _ts():        return datetime.now().strftime("%H:%M:%S")
def _hdr(msg):    print(f"\n{DIM}{_BAR}{NC}\n{NW}{BOLD}  {msg}{NC}\n{DIM}{_BAR}{NC}\n")
def info(msg):    print(f"{BL}[{_ts()}  INFO ]{NC}  {msg}")
def ok(msg):      print(f"{GA}[{_ts()}  PASS ]{NC}  {msg}")
def caution(msg): print(f"{YL}[{_ts()}  CAUT ]{NC}  {msg}")
def warn(msg):    print(f"{RD}[{_ts()}  FAIL ]{NC}  {msg}")

# ── rotor CSV reader ──────────────────────────────────────────────────────────
def read_rotor_csv() -> list[dict]:
    if not ROTOR_CSV.exists():
        return []
    rows = []
    with open(ROTOR_CSV, newline="", encoding="utf-8") as f:
        for row in csv_mod.DictReader(
                (l for l in f if not l.strip().startswith("#"))):
            try:
                rows.append({
                    "rotor_id":          int(row["rotor_id"].strip()),
                    "R_m":               float(row["R_m"].strip()),
                    "n_blades":          int(row["n_blades"].strip()),
                    "chord_m":           float(row["chord_m"].strip()),
                    "twist_root_deg":    float(row["twist_root_deg"].strip()),
                    "twist_tip_deg":     float(row["twist_tip_deg"].strip()),
                    "pitch_offset_deg":  float(row["pitch_offset_deg"].strip()),
                    "P_max_kW":          float(row["P_max_kW"].strip()),
                    "rpm_hover":         float(row["rpm_hover"].strip()),
                    "powerplant":   (row.get("powerplant") or "electric").strip(),
                    "notes":             (row.get("notes") or "").strip(),
                })
            except (KeyError, ValueError):
                continue
    return rows

def _patch_test_card(card_path: Path, overrides: list) -> None:
    if not overrides:
        return
    card = json.loads(card_path.read_text())
    card.setdefault("rotor_fleet", {})["overrides"] = overrides
    card_path.write_text(json.dumps(card, indent=2))

# Wind group in a METAR: dddss[Ggmax]KT|MPS|KMH  (ddd or VRB, ss = sustained speed)
_METAR_WIND_RE = re.compile(r'\b(?:\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?(KT|MPS|KMH)\b')

def _metar_wind_speed_ms(metar: str) -> float:
    """Sustained wind speed in m/s parsed from a raw METAR string, or 0.0 if none found."""
    if not metar:
        return 0.0
    m = _METAR_WIND_RE.search(metar.upper())
    if not m:
        return 0.0
    speed, unit = float(m.group(1)), m.group(2)
    if unit == "KT":  return speed * 0.514444
    if unit == "MPS": return speed
    if unit == "KMH": return speed / 3.6
    return 0.0

def _build_argv(sim: dict, out_dir: Path,
                dep_metar: str = "", arr_metar: str = "",
                cru: dict = {}) -> list[str]:
    mode = sim.get("mode", "auto")
    sf   = sim.get("speed_factor", None)
    gui  = sim.get("gui", False)
    argv = [sys.executable, str(ROOT / "test_flight.py"), f"--{mode}"]
    if mode == "auto" and sf is not None and not gui:
        argv += ["--speed", str(sf)]
    if gui and mode == "auto":
        argv += ["--gui"]
    if sim.get("terrain"):  argv += ["--terrain"]
    if sim.get("no_build"): argv += ["--no-build"]
    if sim.get("no_plan"):  argv += ["--no-plan"]
    if sim.get("db"):       argv += ["--db"]
    argv += ["--out", str(out_dir)]
    # Pass weather and cruise directly — no planning/ files needed
    if dep_metar: argv += ["--dep-metar", dep_metar]
    if arr_metar: argv += ["--arr-metar", arr_metar]
    if cru.get("speed_kmh"):            argv += ["--speed-kmh",      str(cru["speed_kmh"])]
    if cru.get("altitude_ft"):          argv += ["--alt-ft",          str(cru["altitude_ft"])]
    if cru.get("hover_alt_m"):          argv += ["--hover-m",            str(cru["hover_alt_m"])]
    if cru.get("back_trans_speed_ms"):  argv += ["--bt-speed-ms",        str(cru["back_trans_speed_ms"])]
    if cru.get("nacelle_tilt_deg"):     argv += ["--nacelle-tilt-deg",   str(cru["nacelle_tilt_deg"])]
    if sim.get("turb"):
        turb_intensity = round(0.1 * _metar_wind_speed_ms(dep_metar), 3)
        if turb_intensity > 0:
            argv += ["--turb-intensity", str(turb_intensity)]
    return argv



# ═════════════════════════════════════════════════════════════════════════════
#  Results plotting  (ported from plot_test_flight.py — kept inline so
#  windblade.py stays a single-file entry point; see module docstring)
# ═════════════════════════════════════════════════════════════════════════════

# Default columns rendered to the Last Results tab after every run —
# overridable per-run from the Parameters tab (Plots section).
DEFAULT_PLOT_COLUMNS = ["altitude_msl_ft", "speed_kmh"]

# Friendly axis labels for common telemetry columns; anything else falls
# back to a title-cased version of the raw column name.
_PLOT_LABELS = {
    "altitude_msl_ft": "Altitude (ft MSL)",
    "speed_kmh":        "Speed (km/h)",
    "power_kw":         "Power (kW)",
    "soc_pct":          "State of Charge (%)",
    "gz":               "Vertical load factor (gz)",
}

_PLOT_PHASE_COLORS = {
    "hover": "#02CCFE", "transition": "#fb9f3a", "fw_climb": "#de3163",
    "dash": "#2C75FF", "fw_descent": "#9c179e", "back_transition": "#fb9f3a",
    "descent": "#02CCFE", "landed": "#179236"
}
_PLOT_THEME = dict(
    bg=(0.06, 0.06, 0.10), panel=(0.10, 0.10, 0.16),
    grid=(1.0, 1.0, 1.0, 0.07), tick=(0.55, 0.55, 0.65),
    label=(0.85, 0.85, 0.95), title=(0.95, 0.95, 1.00),
    line_fallback="#02CCFE",
)

def _plot_label(col: str) -> str:
    return _PLOT_LABELS.get(col) or (col.replace("_", " ").strip().title() or col)

def _load_flight_csv(csv_path: Path):
    import pandas as pd
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if "phase" in df.columns:
        df["phase"] = (df["phase"].astype(str).str.strip().str.lower()
                        .str.replace(r"^autoland:", "", regex=True))
    return df

def _make_flight_plot(df, y_col: str, y_label: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    th, x_col = _PLOT_THEME, "timestamp_s"
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=th["bg"])
    ax.set_facecolor(th["panel"])

    if "phase" in df.columns:
        present = [ph for ph in _PLOT_PHASE_COLORS if (df["phase"] == ph).any()]
        for ph in present:
            mask = df["phase"] == ph
            ax.plot(df.loc[mask, x_col], df.loc[mask, y_col],
                    color=_PLOT_PHASE_COLORS.get(ph, "#ffffff"), lw=2.2)
        color_groups: dict[str, list[str]] = {}
        for ph in present:
            color_groups.setdefault(_PLOT_PHASE_COLORS.get(ph, "#ffffff"), []).append(ph)
        handles = [mlines.Line2D([], [], color=c, linewidth=4, label=" & ".join(phs))
                   for c, phs in color_groups.items()]
        if handles:
            ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.10,
                       facecolor=th["panel"], edgecolor=(0.5, 0.5, 0.5, 0.3),
                       labelcolor=th["label"], fontsize=9)
    else:
        ax.plot(df[x_col], df[y_col], color=th["line_fallback"], lw=2.2)

    ax.set_title(y_label, color=th["title"], fontsize=13)
    ax.set_xlabel("time (s)", color=th["label"], fontsize=11)
    ax.set_ylabel(y_label, color=th["label"], fontsize=11)
    ax.tick_params(colors=th["tick"], labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, color=th["grid"], linewidth=0.8, linestyle=(0, (1, 3)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)

def _generate_plots(df, out_dir: Path, columns: list[str]) -> dict[str, str]:
    """Render PNGs for the requested columns into out_dir.
    Returns {column: filename} for whichever columns were present."""
    if "timestamp_s" not in df.columns:
        return {}
    plots = {}
    for col in columns:
        if not col or col not in df.columns:
            continue
        fname = f"plot_{col}.png"
        try:
            _make_flight_plot(df, col, _plot_label(col), out_dir / fname)
            plots[col] = fname
        except Exception as e:
            warn(f"plot generation failed for {col}: {e}")

    return plots

def _compute_flight_metrics(df) -> dict:
    """Summary stats parsed straight from the same results CSV used for the
    plots: flight time, range from origin, energy used, arrival SoC."""
    m: dict = {}
    if "timestamp_s" in df.columns and len(df):
        m["flight_time_s"] = round(float(df["timestamp_s"].max()), 1)
    if "x_m" in df.columns and "y_m" in df.columns and len(df):
        x_m = abs(float(df["x_m"].iloc[-1]))
        y_m = abs(float(df["y_m"].iloc[-1]))
        m["distance_km"] = round(((x_m ** 2 + y_m ** 2) ** 0.5) / 1000.0, 2)
    if "power_kw" in df.columns and "timestamp_s" in df.columns and len(df) > 1:
        import numpy as np
        t = df["timestamp_s"].to_numpy(dtype=float)
        p = df["power_kw"].to_numpy(dtype=float)
        energy_kj = float(np.sum(np.diff(t) * (p[:-1] + p[1:]) / 2.0))  # kW·s = kJ
        m["energy_mj"] = round(energy_kj / 1000.0, 2)
    if "soc_pct" in df.columns and len(df):
        m["arrival_soc_pct"] = round(float(df["soc_pct"].iloc[-1]), 1)
    return m

def _process_results_csv(csv_path: Path, out_dir: Path,
                          columns: list[str]) -> tuple[dict, dict]:
    """Load csv_path once; render plots for `columns` and compute summary
    metrics from the same DataFrame. Returns (plots, metrics)."""
    try:
        df = _load_flight_csv(csv_path)
    except Exception as e:
        warn(f"could not read {csv_path.name} for plotting: {e}")
        return {}, {}
    return _generate_plots(df, out_dir, columns), _compute_flight_metrics(df)


# ═════════════════════════════════════════════════════════════════════════════
#  Embedded HTML GUI
# ═════════════════════════════════════════════════════════════════════════════
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🚁 WINDBLADE</title>
<link href="https://fonts.googleapis.com/css2?family=B612+Mono:wght@400;700&family=B612:wght@400;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  /* NVG palette — matches glass_cockpit.jl TH */
  --bg:#080908;--panel:#0c0e0c;--hi:#101310;
  --stroke:#243520;--stroke-hi:#405a3a;
  --nw:#b8ffd0;--ga:#00e040;--yl:#e8c000;--rd:#ff2800;--bl:#47a6f2;
  --dim:#618a6b;--faint:#2e4432;
  --mono:'B612 Mono',monospace;--sans:'B612',sans-serif;
}
html,body{background:var(--bg);color:var(--nw);font-family:var(--mono);font-size:26px;min-height:100vh}
.shell{display:grid;grid-template-columns:320px 1fr;min-height:100vh}
.sidebar{background:var(--panel);border-right:1px solid var(--stroke-hi);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}
.main{display:flex;flex-direction:column}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:18px 36px;background:var(--hi);position:sticky;top:0;z-index:10}
.topbar-title{font-family:var(--sans);font-size:20px;font-weight:700;letter-spacing:.06em;color:var(--dim)}
.content{flex:1;padding:36px 44px;overflow-y:auto}
.statusbar{font-size:20px;color:var(--dim);padding:10px 36px;border-top:1px solid var(--stroke);background:var(--panel);position:sticky;bottom:0}
.sb-logo{padding:20px 22px 16px}
.sb-logo svg{display:block}
.sb-sec{font-size:16px;letter-spacing:.14em;color:var(--faint);padding:18px 22px 6px;text-transform:uppercase}
.nav-item{display:flex;align-items:center;gap:14px;padding:16px 22px;cursor:pointer;font-size:22px;color:var(--dim);border-left:3px solid transparent;transition:all .12s}
.nav-item:hover{background:var(--hi);color:var(--nw)}
.nav-item.active{background:var(--hi);color:var(--ga);border-left-color:var(--ga)}
.nav-item svg{width:22px;height:22px;flex-shrink:0;opacity:.7}
.nav-item.active svg{opacity:1}
.panel{display:none}.panel.active{display:block}
.sec{font-size:18px;letter-spacing:.14em;color:var(--faint);text-transform:uppercase;margin:28px 0 10px;padding-bottom:5px;border-bottom:1px solid var(--stroke)}
.sec:first-child{margin-top:0}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:18px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:18px}
.field{display:flex;flex-direction:column;gap:6px}
.field label{font-size:18px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase}
.field input,.field select,.field textarea{background:var(--panel);border:1px solid var(--stroke-hi);border-radius:2px;padding:10px 14px;font-size:22px;font-family:var(--mono);color:var(--nw);width:100%;transition:border-color .12s}
.field input,.field select{height:52px}
.field textarea{resize:vertical;line-height:1.5}
.field input:focus,.field select:focus,.field textarea:focus{outline:none;border-color:var(--ga)}
.field select option{background:var(--panel)}
.toggle-row{display:flex;align-items:center;justify-content:space-between;padding:12px 0;border-bottom:1px solid var(--stroke)}
.toggle-row:last-child{border-bottom:none}
.tl{font-size:22px;color:var(--nw)}.ts{font-size:18px;color:var(--dim);margin-top:2px}
.switch{width:54px;height:28px;border-radius:14px;background:var(--stroke-hi);position:relative;cursor:pointer;transition:.15s;border:1px solid var(--stroke-hi);flex-shrink:0}
.switch.on{background:var(--ga);border-color:var(--ga)}
.switch::after{content:'';position:absolute;width:20px;height:20px;border-radius:50%;background:var(--bg);top:3px;left:3px;transition:.15s}
.switch.on::after{left:29px}
.data-table{width:100%;border-collapse:collapse;font-size:20px;margin-bottom:20px}
.data-table th{text-align:left;font-size:17px;letter-spacing:.07em;color:var(--dim);text-transform:uppercase;padding:0 12px 8px;border-bottom:1px solid var(--stroke-hi)}
.data-table td{padding:9px 12px;border-bottom:1px solid var(--stroke);color:var(--nw)}
.data-table tr:last-child td{border-bottom:none}
.data-table tr:hover td{background:var(--hi)}
.c-ok{color:var(--ga)}.c-warn{color:var(--yl)}.c-fail{color:var(--rd)}.c-dim{color:var(--dim)}
.callout{border-left:3px solid var(--bl);padding:14px 18px;background:var(--panel);font-size:20px;color:var(--dim);margin-bottom:20px;line-height:1.7}
.callout.ok{border-color:var(--ga)}.callout.warn{border-color:var(--yl)}
.callout code{color:var(--nw);font-family:var(--mono)}
.cmd-box{background:var(--panel);border:1px solid var(--stroke-hi);border-radius:2px;padding:16px 20px;font-size:20px;line-height:1.9;margin-bottom:20px;word-break:break-all;color:var(--ga)}
.cmd-box .dim{color:var(--faint)}.cmd-box .arg{color:var(--yl)}.cmd-box .flag{color:var(--bl)}
.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.metric{background:var(--panel);border:1px solid var(--stroke);border-radius:2px;padding:16px 18px}
.metric .val{font-size:34px;font-weight:700;font-family:var(--sans);color:var(--nw)}
.metric .lbl{font-size:16px;letter-spacing:.08em;color:var(--faint);text-transform:uppercase;margin-top:4px}
.metric .sub{font-size:18px;margin-top:3px}
.actions{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap}
.btn{padding:12px 22px;border-radius:2px;font-size:22px;font-family:var(--mono);cursor:pointer;border:1px solid var(--stroke-hi);background:transparent;color:var(--nw);transition:all .12s;display:flex;align-items:center;gap:10px}
.btn:hover{background:var(--hi)}.btn:disabled{opacity:.35;cursor:default}
.btn.primary{border-color:var(--ga);color:var(--ga)}.btn.primary:hover{background:var(--faint)}
#launch-log{background:var(--panel);border:1px solid var(--stroke-hi);border-radius:2px;padding:16px 20px;font-size:20px;font-family:var(--mono);line-height:1.8;min-height:200px;max-height:50vh;overflow-y:auto;white-space:pre-wrap;color:var(--dim)}
#launch-log .ga{color:var(--ga)}#launch-log .yl{color:var(--yl)}
#launch-log .rd{color:var(--rd)}#launch-log .nw{color:var(--nw)}
.chk-row{display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--stroke)}
.chk-row:last-child{border-bottom:none}
.badge{font-size:18px;padding:3px 10px;border-radius:1px;border:1px solid;font-family:var(--mono)}
.badge.ok{border-color:var(--ga);color:var(--ga)}.badge.warn{border-color:var(--yl);color:var(--yl)}
.badge.info{border-color:var(--bl);color:var(--bl)}
.path-note{font-size:18px;color:var(--faint);margin-top:8px;font-family:var(--mono)}
</style>
</head>
<body>
<div class="shell">
<div class="sidebar">
  <div class="sb-logo">
    <svg viewBox="0 0 220 44" width="220" height="44" xmlns="http://www.w3.org/2000/svg">
      <text x="6" y="34"
        font-family="B612,sans-serif" font-size="28" font-weight="700" font-style="italic"
        fill="#47a6f2" letter-spacing="2"
        transform="skewX(-8)">WINDBLADE</text>
    </svg>
  </div>
  <div class="sb-sec">Mission</div>
  <div class="nav-item active" onclick="nav('route',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12h18M3 6l9-3 9 3M3 18l9 3 9-3"/></svg>Route &amp; weather
  </div>
  <div class="nav-item" onclick="nav('flight',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2l2 7h7l-5.5 4 2 7L12 16l-5.5 4 2-7L3 9h7z"/></svg>Parameters
  </div>
  <div class="sb-sec">Propulsion</div>
  <div class="nav-item" onclick="nav('rotors',this);loadRotors()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M2 12h4M18 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>Rotor config
  </div>
  <div class="sb-sec">Run</div>
  <div class="nav-item" onclick="nav('launch',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="5,3 19,12 5,21"/></svg>Launch
  </div>
  <div class="nav-item" onclick="nav('results',this)">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>Last results
  </div>
</div>

<div class="main">
  <div class="topbar">
    <span class="topbar-title">Mission Planner</span>
    <div style="display:flex;gap:8px;align-items:center">
      <span class="badge info" id="rotor-badge">loading rotors...</span>
    </div>
  </div>
  <div class="content">

    <!-- Route & Weather -->
    <div class="panel active" id="panel-route">
      <div class="sec">Departure — METAR</div>
      <div class="field">
        <label>METAR_DEP</label>
        <textarea id="dep-metar" rows="3" oninput="sync()">KAXX 151155Z 00000KT 10SM CLR M01/M10 A3018 RMK AO2 T10141096</textarea>
      </div>
      <div class="sec">Arrival — METAR</div>
      <div class="field">
        <label>METAR_ARR</label>
        <textarea id="arr-metar" rows="3" oninput="sync()">KSAF 151153Z 24005KT 10SM CLR 13/M09 A3005 RMK AO2 T01281094</textarea>
      </div>
      <div class="callout" style="margin-top:20px">Paste real METARs here — passed directly to <code>test_flight.py</code> via <code>--dep-metar</code> / <code>--arr-metar</code>. No planning/ files are written. The ICAO code is read from the first token.</div>
    </div>

    <!-- Flight params -->
    <div class="panel" id="panel-flight">
      <div class="sec">Flight</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:20px;margin-bottom:18px">
        <div class="field"><label>Cruise spd (km/h)</label><input id="speed" value="296" oninput="sync()"></div>
        <div class="field"><label>Cruise alt (ft MSL)</label><input id="alt" value="11000" oninput="sync()"></div>
        <div class="field"><label>Hover alt (m AGL)</label><input id="hover" value="30" oninput="sync()"></div>
        <div class="field"><label>Trans2 spd (m/s)</label><input id="bt-speed" value="50" oninput="sync()" title="fw_descent decelerates to this speed before pitching up"></div>
        <div class="field"><label>Nacelle tilt (deg)</label><input id="nacelle-tilt" value="70" min="45" max="90" oninput="sync()"></div>
      </div>
      <div class="sec">Execution</div>
      <div class="row2">
        <div class="field"><label>Speed factor</label>
          <select id="sfactor" onchange="sync()">
            <option value="1" selected>1x — realtime</option>
            <option value="6">6x</option>
            <option value="12">12x</option>
            <option value="60">60x</option>
            <option value="360">360x</option>
          </select>
        </div>
        <div class="field"><label>Mode</label>
          <select id="mode" onchange="sync()">
            <option value="auto" selected>--auto</option>
            <option value="manual">--manual (HOTAS)</option>
          </select>
        </div>
      </div>

      <div class="sec">Plots</div>
      <div class="row2">
        <div class="field"><label>Plot column 1</label><input id="plot-col1" value="altitude_msl_ft" oninput="sync()" title="CSV column name to render as a PNG in the Last Results tab"></div>
        <div class="field"><label>Plot column 2</label><input id="plot-col2" value="speed_kmh" oninput="sync()" title="CSV column name to render as a PNG in the Last Results tab"></div>
      </div>

      <div class="sec">Options</div>
      <div class="toggle-row">
        <div><div class="tl">Show glass cockpit</div><div class="ts">Auto mode only; not compatible with speed factor</div></div>
        <div class="switch" id="sw-gui" onclick="tog(this,'gui')"></div>
      </div>
      <div class="toggle-row">
        <div><div class="tl">Dryden turbulence</div><div class="ts">Approximated as 0.1 x reference wind speed</div></div>
        <div class="switch" id="sw-turb" onclick="tog(this,'turb')"></div>
      </div>
      <div class="toggle-row">
        <div><div class="tl">Download SRTM terrain</div><div class="ts">Force terrain profile refresh for this route</div></div>
        <div class="switch" id="sw-terrain" onclick="tog(this,'terrain')"></div>
      </div>
      <div class="toggle-row">
        <div><div class="tl">Skip rebuild (.so)</div><div class="ts">Reuse existing autopilot.so / autoland.so</div></div>
        <div class="switch" id="sw-nobuild" onclick="tog(this,'nobuild')"></div>
      </div>
      <div class="toggle-row">
        <div><div class="tl">Skip planning</div><div class="ts">Reuse existing test_card.json</div></div>
        <div class="switch" id="sw-noplan" onclick="tog(this,'noplan')"></div>
      </div>
      <div class="toggle-row">
        <div><div class="tl">Export SQLite DB</div><div class="ts">Write dash_results_&lt;ts&gt;.db alongside the CSV (test_parameters + telemetry tables)</div></div>
        <div class="switch" id="sw-db" onclick="tog(this,'db')"></div>
      </div>
    </div>

    <!-- Rotor config -->
    <div class="panel" id="panel-rotors">
      <div class="sec">Fleet — <span id="rotor-csv-path" style="color:var(--faint)"></span></div>
      <div id="rotor-fleet-error" style="display:none" class="callout" style="border-color:var(--rd);color:var(--rd)"></div>
      <table class="data-table" id="rotor-table">
        <thead><tr><th>#</th><th>Position</th><th>R (m)</th><th>Blades</th><th>Chord (m)</th><th>Twist root</th><th>P max (kW)</th><th>RPM hover</th><th>Propulsion</th><th>Notes</th></tr></thead>
        <tbody id="rotor-tbody"><tr><td colspan="10" class="c-dim" style="padding:12px 8px">Loading...</td></tr></tbody>
      </table>
      <p class="path-note">Edit <code>subsystems/propulsion/rotor_config.csv</code> and click the Rotor Config tab again to reload.</p>
      <div id="rotor-disks" style="margin-top:24px"></div>
    </div>

    <!-- Launch -->
    <div class="panel" id="panel-launch">
      <div class="sec">Command</div>
      <div class="cmd-box" id="cmd-box"></div>
      <div class="sec">Preflight</div>
      <div id="checklist"></div>
      <div class="actions" style="margin-top:20px">
        <button class="btn primary" id="btn-launch" onclick="doLaunch()">&#9654; Launch simulation</button>
      </div>
      <div class="sec" style="margin-top:24px">Terminal output</div>
      <div id="launch-log">Waiting for launch...</div>
    </div>

    <!-- Results -->
    <div class="panel" id="panel-results">
      <div class="metric-row">
        <div class="metric"><div class="val" id="rv-time">—</div><div class="lbl">Flight time</div></div>
        <div class="metric"><div class="val" id="rv-dist">—</div><div class="lbl">Distance flown</div></div>
        <div class="metric"><div class="val" id="rv-energy">—</div><div class="lbl">Energy</div></div>
        <div class="metric"><div class="val" id="rv-soc">—</div><div class="lbl">Arrival SoC</div></div>
      </div>
      <div class="callout">Drop a <code>dash_results_*.csv</code> to analyse, or results populate automatically after a run.</div>
      <div class="callout" id="db-status" style="display:none;border-color:var(--ga)">&#10003; SQLite DB exported alongside CSV</div>
      <div id="auto-plots"></div>
      <div style="border:1px dashed var(--stroke-hi);border-radius:2px;padding:28px;text-align:center;cursor:pointer;background:var(--panel);margin-bottom:16px" onclick="document.getElementById('res-file').click()">
        <input type="file" id="res-file" accept=".csv" style="display:none" onchange="loadResults(this.files[0])">
        <p style="font-size:22px;color:var(--dim)">Drop results CSV or click to browse</p>
      </div>
      <div id="results-detail" style="display:none">
        <div class="sec">Phase timeline</div>
        <table class="data-table"><thead><tr><th>Phase</th><th>Last row</th><th>Rows</th></tr></thead><tbody id="phase-tbody"></tbody></table>
      </div>
    </div>

  </div>
  <div class="statusbar" id="statusbar">planner ready</div>
</div>
</div>

<script>
var sw={terrain:false,nobuild:false,noplan:false,gui:false,db:false,turb:false};
var running=false;
var rotorData=[];

var POSITIONS=['fwd-port','fwd-stbd','mid-port','mid-stbd','aft-port','aft-stbd'];
var S4={R_m:1.45,n_blades:5,chord_m:0.096,twist_root_deg:16.0,twist_tip_deg:6.0,pitch_offset_deg:4.4,P_max_kW:280,rpm_hover:1260};

function nav(id,el){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
  if(el)el.classList.add('active');
  if(id==='launch')renderChecklist();
  if(id==='results')loadAutoPlots();
}

function tog(el,key){
  sw[key]=!sw[key];
  el.classList.toggle('on',sw[key]);
  sync();
}

function v(id){return document.getElementById(id).value.trim();}

function buildCmd(){
  var mode=v('mode'),sf=v('sfactor');
  var parts=['python3 test_flight.py','--'+mode];
  if(mode==='auto'&&sf!=='1'&&!sw.gui)parts.push('--speed '+sf);
  if(sw.gui&&mode==='auto')parts.push('--gui');
  if(sw.terrain)parts.push('--terrain');
  if(sw.nobuild)parts.push('--no-build');
  if(sw.noplan)parts.push('--no-plan');
  if(sw.db)parts.push('--db');
  
  return parts.join(' ');
}

function sync(){
  var raw=buildCmd();
  var html=raw.replace('python3 ','<span class="dim">python3 </span>')
              .replace(/--[\w-]+/g,s=>'<span class="flag">'+s+'</span>')
              .replace(/\b(\d{5,})\b/g,s=>'<span class="arg">'+s+'</span>');
  var box=document.getElementById('cmd-box');
  if(box)box.innerHTML=html;
}

function loadRotors(){
  fetch('/rotors')
    .then(r=>r.json())
    .then(data=>{
      rotorData=data.rotors||[];
      var n=rotorData.length;
      var err=data.error||null;
      // Show/hide fleet error banner
      var errEl=document.getElementById('rotor-fleet-error');
      if(err){
        errEl.textContent=err;
        errEl.style.display='block';
        errEl.style.borderColor='var(--rd)';
        errEl.style.color='var(--rd)';
      } else {
        errEl.style.display='none';
      }
      renderRotorTable(rotorData);
      renderRotorDisks(rotorData);
      document.getElementById('rotor-csv-path').textContent=data.path||'';
      if(err){
        document.getElementById('rotor-badge').textContent=err;
        document.getElementById('rotor-badge').className='badge warn';
      } else if(n===0){
        document.getElementById('rotor-badge').textContent='csv not found';
        document.getElementById('rotor-badge').className='badge warn';
      } else {
        document.getElementById('rotor-badge').textContent=n+' rotor'+(n!==1?'s':'')+' loaded';
        document.getElementById('rotor-badge').className='badge ok';
      }
    })
    .catch(()=>{
      document.getElementById('rotor-badge').textContent='csv not found';
      document.getElementById('rotor-badge').className='badge warn';
    });
}

function propCls(t){
  if(t==='electric')return 'c-ok';
  if(t==='turboshaft')return 'c-warn';
  if(t==='turbine_electric')return 'c-dim';
  return '';
}

function renderRotorTable(rotors){
  var tbody=document.getElementById('rotor-tbody');
  if(!rotors.length){
    tbody.innerHTML='<tr><td colspan="10" class="c-warn" style="padding:12px 8px">rotor_config.csv not found or empty</td></tr>';
    return;
  }
  tbody.innerHTML=rotors.map((r,i)=>{
    var pos=POSITIONS[i]||'pos-'+i;
    var rc=Math.abs(r.R_m-S4.R_m)>0.001?'c-warn':'';
    var pc=Math.abs(r.P_max_kW-S4.P_max_kW)>0.1?'c-warn':'';
    var rpc=Math.abs(r.rpm_hover-S4.rpm_hover)>1?'c-warn':'';
    var pt=r.powerplant||'electric';
    return '<tr>'
      +'<td class="c-dim">'+r.rotor_id+'</td>'
      +'<td>'+pos+'</td>'
      +'<td class="'+rc+'">'+r.R_m.toFixed(3)+'</td>'
      +'<td>'+r.n_blades+'</td>'
      +'<td>'+r.chord_m.toFixed(3)+'</td>'
      +'<td>'+r.twist_root_deg.toFixed(1)+'&deg;</td>'
      +'<td class="'+pc+'">'+r.P_max_kW.toFixed(0)+'</td>'
      +'<td class="'+rpc+'">'+r.rpm_hover.toFixed(0)+'</td>'
      +'<td class="'+propCls(pt)+'">'+pt+'</td>'
      +'<td class="c-dim">'+r.notes+'</td>'
      +'</tr>';
  }).join('');
}

// ── Inline rotor disk SVGs (outline style, NVG palette) ──────────────
// All tiles share the same pixel footprint (tileSize). The SVG coordinate
// space is anchored to the fleet's largest rotor (R_max), so each disk
// renders proportionally smaller if its R_m is below the fleet maximum.
function buildDiskSVG(r, tileSize, R_max){
  var LABEL_H = 22;    // px reserved at bottom for the R label (outside disk area)
  var PAD     = 6;     // px margin above and on sides
  var S = tileSize || 160;
  var drawH = S - LABEL_H;          // vertical space available for the disk
  // Disk centre sits in the middle of the draw area, with top/side padding
  var cx = S / 2;
  var cy = PAD + (drawH - PAD) / 2;
  // Maximum disk radius: largest rotor just touches the margins
  var maxDiskPx = Math.min(S/2 - PAD, (drawH - PAD) / 2) * 0.96;
  var R_m = r.R_m || 1.45;
  var R = maxDiskPx * (R_m / R_max);   // proportional disk radius
  var hubR = R * 0.18;
  var nb = r.n_blades || 6;
  var chordRoot = r.chord_m || 0.096;
  var chordTip = chordRoot * 0.55;
  var scale = R / R_m;
  var cRpx = Math.min(chordRoot * scale * 3.5, R * 0.28);
  var cTpx = Math.min(chordTip  * scale * 3.5, R * 0.16);
  var pt = r.powerplant || 'electric';
  var diskColor = pt==='electric' ? '#00e040' : pt==='turboshaft' ? '#e8c000' : '#47a6f2';
  var strokeW = 1.2;

  var blades = '';
  for(var b = 0; b < nb; b++){
    var ang = 2*Math.PI*b/nb - Math.PI/2;
    var ca = Math.cos(ang), sa = Math.sin(ang);
    function pt2(r_px, c_px, side){
      var offset = side===1 ? c_px/4 : -3*c_px/4;
      return [(cx + r_px*ca - offset*sa), (cy + r_px*sa + offset*ca)];
    }
    var p0 = pt2(hubR, cRpx,  1);   // root LE
    var p1 = pt2(R,    cTpx,  1);   // tip LE
    var p2 = pt2(R,    cTpx, -1);   // tip TE
    var p3 = pt2(hubR, cRpx, -1);   // root TE
    var hw = Math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / 2;
    var d = 'M'+p0[0].toFixed(1)+','+p0[1].toFixed(1)
          + ' L'+p1[0].toFixed(1)+','+p1[1].toFixed(1)
          + ' A'+hw.toFixed(1)+','+hw.toFixed(1)+' 0 0,0 '+p2[0].toFixed(1)+','+p2[1].toFixed(1)
          + ' L'+p3[0].toFixed(1)+','+p3[1].toFixed(1)+' Z';
    blades += '<path d="'+d+'" fill="none" stroke="'+diskColor+'" stroke-width="'+strokeW+'" stroke-linejoin="round"/>';
  }

  // Disk outline — dashed circle at actual R (not maxDiskPx)
  var disk = '<circle cx="'+cx+'" cy="'+cy+'" r="'+R.toFixed(1)+'" fill="none" stroke="#2e4432" stroke-width="0.8" stroke-dasharray="3,3"/>';
  // R_max reference ring (faint) so the relative scale is visually legible
  var refRing = R < maxDiskPx
    ? '<circle cx="'+cx+'" cy="'+cy+'" r="'+maxDiskPx.toFixed(1)+'" fill="none" stroke="#1a2a1a" stroke-width="0.5" stroke-dasharray="1,4"/>'
    : '';
  var hub    = '<circle cx="'+cx+'" cy="'+cy+'" r="'+hubR.toFixed(1)+'" fill="#0c0e0c" stroke="'+diskColor+'" stroke-width="1.0"/>';
  var rLabel = R_m.toFixed(3)+' m';
  var label  = '<text x="'+cx+'" y="'+(S-6)+'" text-anchor="middle" font-family="B612 Mono,monospace" font-size="11" fill="#618a6b">R'+r.rotor_id+' · '+rLabel+'</text>';

  return '<svg xmlns="http://www.w3.org/2000/svg" width="'+S+'" height="'+S+'" viewBox="0 0 '+S+' '+S+'" style="background:#0c0e0c;border:1px solid #243520;border-radius:2px">'
    + refRing + disk + blades + hub + label + '</svg>';
}

function renderRotorDisks(rotors){
  var el = document.getElementById('rotor-disks');
  if(!rotors.length){ el.innerHTML=''; return; }
  var n = rotors.length;
  // Fleet-wide max radius — sets the common scale reference
  var R_max = Math.max.apply(null, rotors.map(function(r){ return r.R_m || 1.45; }));
  // ≤6 rotors: single row; >6: wrap at 4 columns
  var cols = n<=6 ? n : 4;
  var tileSize = n<=6 ? Math.min(170, Math.floor(900/n)) : 150;
  var html = '<div style="display:grid;grid-template-columns:repeat('+cols+','+tileSize+'px);gap:12px;margin-top:12px">';
  rotors.forEach(function(r){ html += buildDiskSVG(r, tileSize, R_max); });
  html += '</div>';
  el.innerHTML = html;
}

function renderChecklist(){
  var depIcao=v('dep-metar').split(' ')[0]||'?';
  var arrIcao=v('arr-metar').split(' ')[0]||'?';
  var rows=[
    ['DEP',    v('dep-metar').substring(0,65), 'var(--ga)'],
    ['ARR',    v('arr-metar').substring(0,65), 'var(--ga)'],
    ['CRUISE', v('speed')+' km/h  /  '+v('alt')+' ft MSL  /  hover '+v('hover')+' m', 'var(--ga)'],
    ['MODE',   v('mode').toUpperCase()+'  x'+v('sfactor'), 'var(--ga)'],
    ['ROTORS', rotorData.length+' rotors loaded from rotor_config.csv', rotorData.length?'var(--ga)':'var(--yl)'],

  ];
  document.getElementById('checklist').innerHTML=rows.map(([l,d,c])=>
    '<div class="chk-row"><span style="color:'+c+';font-size:22px;min-width:140px">'+l+'</span>'
    +'<span style="font-size:20px;color:var(--dim)">'+d+'</span></div>'
  ).join('');
  sync();
}

function logLine(html){var el=document.getElementById('launch-log');el.innerHTML+=html+'\n';el.scrollTop=el.scrollHeight;}
function logClear(){
  document.getElementById('launch-log').innerHTML='';
  var el=document.getElementById('db-status');if(el)el.style.display='none';
}

function getConfig(){
  return{
    dep_metar: v('dep-metar'),
    arr_metar: v('arr-metar'),
    cruise:{speed_kmh:parseFloat(v('speed'))||296,altitude_ft:parseFloat(v('alt'))||11000,hover_alt_m:parseFloat(v('hover'))||30,back_trans_speed_ms:parseFloat(v('bt-speed'))||50,nacelle_tilt_deg:Math.min(90,Math.max(45,parseFloat(v('nacelle-tilt'))||65))},
    sim:{mode:v('mode'),speed_factor:parseInt(v('sfactor'))||1,
         terrain:sw.terrain,no_build:sw.nobuild,no_plan:sw.noplan,gui:sw.gui,db:sw.db,turb:sw.turb},
    plot_columns:[v('plot-col1')||'altitude_msl_ft', v('plot-col2')||'speed_kmh'],

  };
}

function doLaunch(){_launch();}

function _launch(){
  if(running)return;
  running=true;
  document.getElementById('btn-launch').disabled=true;
  document.getElementById('statusbar').textContent='simulation running...';
  logClear();
  logLine('<span class="nw">[ launch_sim ]  submitting config...</span>');
  fetch('/launch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(getConfig())})
    .then(r=>r.json())
    .then(d=>{
      if(d.status==='started'){
        logLine('<span class="ga">[ launch_sim ]  started</span>');
        pollLog(0);
      } else {
        logLine('<span class="rd">error: '+(d.message||'unknown')+'</span>');
        resetLaunch();
      }
    })
    .catch(e=>{logLine('<span class="rd">fetch error: '+e+'</span>');resetLaunch();});
}

function pollLog(offset){
  fetch('/log?offset='+offset)
    .then(r=>r.json())
    .then(d=>{
      (d.lines||[]).forEach(l=>{
        var cls=l.includes('PASS')||l.includes('done')?'ga':l.includes('CAUT')||l.includes('warn')?'yl':l.includes('FAIL')||l.includes('error')?'rd':'nw';
        logLine('<span class="'+cls+'">'+l.replace(/</g,'&lt;')+'</span>');
        if(l.includes('SQLite DB:')){
          var el=document.getElementById('db-status');
          if(el){el.style.display='block';}
        }
      });
      if(d.done){
        var rc=d.exit_code;
        logLine(rc===0?'\n<span class="ga">&#9552;&#9552;&#9552;  complete  rc=0  &#9552;&#9552;&#9552;</span>':'\n<span class="rd">&#9552;&#9552;&#9552;  exited  rc='+rc+'  &#9552;&#9552;&#9552;</span>');
        document.getElementById('statusbar').textContent='last run: rc='+rc;
        resetLaunch();
        if(rc<=1){nav('results',document.querySelectorAll('.nav-item')[4]);}
      } else {
        setTimeout(()=>pollLog(d.next_offset),800);
      }
    })
    .catch(()=>setTimeout(()=>pollLog(offset),1500));
}

function resetLaunch(){running=false;document.getElementById('btn-launch').disabled=false;}

function fmtMinSec(totalSeconds){
  var s=Math.round(totalSeconds), mm=Math.floor(s/60), ss=s%60;
  return mm+'m '+ss+'s';
}

function loadAutoPlots(){
  fetch('/plots').then(r=>r.json()).then(d=>{
    var m=d.metrics||{};
    document.getElementById('rv-time').textContent   = m.flight_time_s!=null ? fmtMinSec(m.flight_time_s) : '—';
    document.getElementById('rv-dist').textContent    = m.distance_km!=null ? m.distance_km.toFixed(2)+' km' : '—';
    document.getElementById('rv-energy').textContent  = m.energy_mj!=null ? m.energy_mj.toFixed(2)+' MJ' : '—';
    document.getElementById('rv-soc').textContent     = m.arrival_soc_pct!=null ? m.arrival_soc_pct.toFixed(1)+'%' : '—';

    var el=document.getElementById('auto-plots');
    if(!el)return;
    if(!d.plots||!d.plots.length){el.innerHTML='';return;}
    el.innerHTML='<div class="sec">Flight plots</div>'
      +'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">'
      +d.plots.map(function(p){
        var src=p.url+'?t='+Date.now();
        return '<div><a href="'+p.url+'" download title="Click to save PNG">'
          +'<img src="'+src+'" alt="'+p.col+'" style="width:100%;display:block;border:1px solid var(--stroke-hi);border-radius:2px"></a>'
          +'<div style="font-size:16px;color:var(--dim);margin-top:4px;text-align:center">'+p.col+' &middot; click to save PNG</div></div>';
      }).join('')
      +'</div>';
  }).catch(function(){});
}

function loadResults(f){
  if(!f)return;
  var r=new FileReader();
  r.onload=function(e){
    var lines=e.target.result.trim().split('\n');
    var hdr=lines[0].split(',').map(s=>s.trim().toLowerCase());
    var rows=lines.slice(1).map(l=>l.split(',').map(s=>s.trim()));
    var ci=k=>hdr.indexOf(k);

    var phases={};
    var rowNum=0;
    rows.forEach(r=>{
      rowNum++;
      var ph=ci('phase')>=0?r[ci('phase')]:'';
      if(ph){
        if(!phases[ph])phases[ph]={first:rowNum,last:rowNum,n:0};
        phases[ph].last=rowNum;
        phases[ph].n++;
      }
    });

    document.getElementById('phase-tbody').innerHTML=Object.entries(phases).map(([ph,v])=>
      '<tr><td>'+ph+'</td><td>'+v.last+'</td><td>'+v.n+'</td></tr>').join('');
    document.getElementById('results-detail').style.display='block';
  };r.readAsText(f);
}

loadRotors();
sync();
</script>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
#  Sim pipeline
# ═════════════════════════════════════════════════════════════════════════════

_sim_log:  list[str] = []
_sim_done: bool      = False
_sim_rc:   int       = -1
_sim_lock  = threading.Lock()

_last_out_dir: Path | None = None
_last_plots:   dict[str, str] = {}   # {column: filename}, relative to _last_out_dir
_last_metrics: dict = {}             # flight_time_s / distance_km / energy_mj / arrival_soc_pct

def _run_sim(cfg: dict) -> None:
    global _sim_done, _sim_rc

    def _log(msg: str) -> None:
        with _sim_lock:
            _sim_log.append(msg)
        print(msg)

    dep_metar = cfg.get("dep_metar", "").strip()
    arr_metar = cfg.get("arr_metar", "").strip()
    cru       = cfg.get("cruise", {})
    sim       = cfg.get("sim",    {})
    plot_cols = [c.strip() for c in (cfg.get("plot_columns") or []) if c and c.strip()]
    if not plot_cols:
        plot_cols = list(DEFAULT_PLOT_COLUMNS)
    out_dir   = (ROOT / f"results_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}").resolve()

    ovrs = read_rotor_csv()

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        argv = _build_argv(sim, out_dir, dep_metar, arr_metar, cru)
        _log(f"[INFO]  DEP: {dep_metar[:70]}")
        _log(f"[INFO]  ARR: {arr_metar[:70]}")
        if sim.get("turb"):
            wind_ms = _metar_wind_speed_ms(dep_metar)
            _log(f"[INFO]  Dryden turbulence: dep wind {wind_ms:.1f} m/s -> "
                 f"intensity {round(0.1 * wind_ms, 3)} m/s")
        _log(f"[INFO]  Invoking: {' '.join(argv[:5])}...\n")

        proc = subprocess.Popen(argv, cwd=str(ROOT),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            _log(line.rstrip())
        proc.wait()
        sim_rc = proc.returncode

        # Patch rotor overrides into the card test_flight.py just generated
        card_path = PLANNING / "test_card.json"
        if ovrs and card_path.exists():
            _log(f"[INFO]  Patching test_card.json with {len(ovrs)} rotor override(s)")
            _patch_test_card(card_path, ovrs)

        _log(f"\n[{'PASS' if sim_rc == 0 else 'FAIL'}]  test_flight.py exited rc={sim_rc}")

        plots, metrics = {}, {}
        if sim_rc <= 1:
            csvs = sorted(out_dir.glob("dash_results_*.csv"))
            if csvs:
                plots, metrics = _process_results_csv(csvs[-1], out_dir, plot_cols)
                if plots:
                    _log(f"[INFO]  Saved plot(s): {', '.join(plots.values())}")
            else:
                _log("[CAUT]  no dash_results_*.csv found — skipping plots")

    except Exception as e:
        _log(f"[FAIL]  {e}")
        sim_rc = 3
        plots, metrics = {}, {}

    global _last_out_dir, _last_plots, _last_metrics
    with _sim_lock:
        _sim_done = True; _sim_rc = sim_rc
        _last_out_dir = out_dir
        _last_plots   = plots
        _last_metrics = metrics


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def _send(self, code, ctype, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _HTML.encode())

        elif path == "/rotors":
            rows = read_rotor_csv()
            n = len(rows)
            error = None
            if n > 0 and not (2 <= n <= 8):
                error = f"Invalid rotor count: {n}. Fleet must have 2–8 rotors."
            payload = json.dumps({"rotors": rows, "path": str(ROTOR_CSV), "error": error}).encode()
            self._send(200, "application/json", payload)

        elif path == "/card":
            card_path = PLANNING / "test_card.json"
            if card_path.exists():
                payload = card_path.read_bytes()
                self._send(200, "application/json", payload)
            else:
                self._send(404, "application/json", b"{}")

        elif path == "/log":
            from urllib.parse import parse_qs
            qs     = parse_qs(urlparse(self.path).query)
            offset = int(qs.get("offset", ["0"])[0])
            with _sim_lock:
                lines = _sim_log[offset:]
                done  = _sim_done
                rc    = _sim_rc
            payload = json.dumps({
                "lines": lines, "next_offset": offset + len(lines),
                "done": done, "exit_code": rc,
            }).encode()
            self._send(200, "application/json", payload)

        elif path == "/plots":
            with _sim_lock:
                out_dir = _last_out_dir
                plots   = dict(_last_plots)
                metrics = dict(_last_metrics)
            payload = json.dumps({
                "out_dir": str(out_dir) if out_dir else None,
                "plots": [{"col": c, "url": f"/plot/{f}"} for c, f in plots.items()],
                "metrics": metrics,
            }).encode()
            self._send(200, "application/json", payload)

        elif path.startswith("/plot/"):
            fname = path[len("/plot/"):]
            with _sim_lock:
                out_dir = _last_out_dir
                allowed = set(_last_plots.values())
            fpath = out_dir / fname if out_dir else None
            if fname in allowed and fpath and fpath.exists():
                self._send(200, "image/png", fpath.read_bytes())
            else:
                self._send(404, "text/plain", b"not found")

        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if urlparse(self.path).path == "/launch":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                cfg = json.loads(body)
            except json.JSONDecodeError as e:
                self._send(400, "application/json",
                           json.dumps({"status":"error","message":str(e)}).encode())
                return

            global _sim_log, _sim_done, _sim_rc
            with _sim_lock:
                _sim_log  = []
                _sim_done = False
                _sim_rc   = -1

            threading.Thread(target=_run_sim, args=(cfg,), daemon=True).start()
            self._send(200, "application/json",
                       json.dumps({"status":"started"}).encode())
        else:
            self._send(404, "text/plain", b"not found")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port",            type=int, default=5780)
    p.add_argument("--no-browser",      action="store_true")
    p.add_argument("--preview-command", action="store_true",
                   help="GUI opens normally; Launch prints command only, nothing runs")
    args = p.parse_args()

    url = f"http://localhost:{args.port}"
    _hdr("eVTOL  ·  Mission Planner")
    info(f"Serving GUI at  {GA}{url}{NC}")
    info(f"Repo root:      {ROOT}")
    info(f"Rotor CSV:      {ROTOR_CSV}")
    if args.preview_command:
        caution("--preview-command mode active")
    info(f"Press  {YL}Ctrl+C{NC}  to stop\n")

    server = HTTPServer(("localhost", args.port), _Handler)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        caution("Shutting down")
    return 0

if __name__ == "__main__":
    sys.exit(main())

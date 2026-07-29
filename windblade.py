#!/usr/bin/env python3
#
# windblade.py:   Mission Planner, Flight Engine, and Launcher GUI
# AUTHOR:         DANIEL DESAI
# UPDATED:        2026-07-28
# VERSION:        0.2.1
#
# Single standalone file: planning, build, simulate, and analyse all run
# in-process here — `python3 windblade.py --auto ...` / `--manual ...` runs
# the full pipeline headlessly, no browser. See run_pipeline() / RunSpec.
#
# The Rotor Config tab is editable in the GUI. Edits are SESSION-ONLY: they
# never touch subsystems/propulsion/rotor_config.csv on disk. An edit updates
# the in-memory fleet used for the next run (and, if a test_card.json already
# exists, patches its rotor_fleet block immediately so the Launch preview and
# any --no-plan run reflect it). "Reset to CSV" drops the session override and
# goes back to reading the file. Restarting the server also drops it — nothing
# session-only ever survives a restart, by design.

"""
Single-file entry point.  Run and a browser window opens with the
mission planner GUI.

Usage
-----
    python3 windblade.py                    # open GUI on http://localhost:5780
    python3 windblade.py --port 8080        # alternate port
    python3 windblade.py --no-browser       # server only, open URL manually
    python3 windblade.py --auto ...         # headless pipeline run, no GUI
    python3 windblade.py --manual ...       # headless HOTAS run, no GUI

Rotor fleet is defined in subsystems/propulsion/rotor_config.csv. Edit that
file for a permanent change, or use the Rotor Config tab for a session-only
change that affects runs only until the server restarts or you hit Reset.

Exit codes (headless / pipeline):
    0   all checks passed
    1   one or more checks failed
    2   build failed
    3   sim failed / no CSV produced
    4   flight planning failed
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import datetime
import gzip
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from datetime import UTC
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np

# ── repo layout ───────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
PLANNING   = ROOT / "planning"
CONTROLS   = ROOT / "controls"
ROTOR_CSV  = ROOT / "subsystems" / "propulsion" / "rotor_config.csv"
TERRAIN_JL = ROOT / "world" / "terrain.jl"

# ── Physical constants ────────────────────────────────────────────────────────
R_DRY_AIR    = 287.058
R_EARTH      = 6_371_000.0
STD_PRESSURE = 101_325.0
STD_TEMP_K   = 288.15
LAPSE_RATE   = 0.0065
G            = 9.80665
INHG_TO_PA   = 3386.389
MPS_PER_KT   = 0.514444
FT_TO_M      = 0.3048

SRTM_MIRRORS = [
    "https://opentopography.s3.sdsc.edu/raster/SRTM_GL3/SRTM_GL3_srtm/{tile}.hgt",
    "https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/{tile}.hgt.zip",
]

# ── Live weather ──────────────────────────────────────────────────────────────
# The browser cannot hit aviationweather.gov directly (no CORS headers), so
# GET /metar proxies it server-side.  Timeouts are deliberately short: the
# handler thread is blocked while they run and the Launch button waits on it.
METAR_HOST     = "aviationweather.gov"
METAR_URL      = "https://aviationweather.gov/api/data/metar?ids={icao}&format=raw"
METAR_UA       = "windblade.py/0.2.0 (eVTOL mission planner; single-user desktop tool)"
METAR_TIMEOUT  = 5.0      # s — HTTP fetch
METAR_TTL_S    = 600.0    # s — in-memory cache lifetime (~10 min, one obs cycle)
NET_PROBE_TIMEOUT = 2.0   # s — TCP connect for the connectivity probe
NET_PROBE_TTL_S   = 30.0  # s — how long a probe result is reused

# Bundled reference observations for the default route.  These are the two
# literal strings that used to be the METAR textarea defaults; they survive the
# textarea→ICAO change as fallback constants.  Used only when a live fetch for
# these two stations fails or comes back empty — for any other airport the
# fallback is the ISA lapse-rate METAR from synthetic_metar().
BUNDLED_METAR = {
    "KAXX": "KAXX 151155Z 00000KT 10SM CLR M01/M10 A3018 RMK AO2 T10141096",
    "KSAF": "KSAF 151153Z 24005KT 10SM CLR 13/M09 A3005 RMK AO2 T01281094",
}

# ══════════════════════════════════════════════════════════════════════════════
#  Logging — one seam, five helpers. Every log line in the engine, the build
#  step, and the sim runner goes through these, so a GUI run can tee its
#  output into the browser log just by pointing _LOG_SINK at a list. See
#  _run_sim for where that happens.
# ══════════════════════════════════════════════════════════════════════════════
_ESC = "\033["
NC   = f"{_ESC}0m";  NW = f"{_ESC}96m"; GA = f"{_ESC}92m"
YL   = f"{_ESC}93m"; RD = f"{_ESC}91m"; BL = f"{_ESC}96m"
DIM  = f"{_ESC}2m";  BOLD = f"{_ESC}1m"
_BAR = "─" * 60

_LOG_SINK: list | None = None      # non-None only while a GUI run is active
_LOG_LOCK = threading.Lock()

def _ts(): return datetime.datetime.now().strftime("%H:%M:%S")

def _emit(line: str) -> None:
    print(line)
    with _LOG_LOCK:
        if _LOG_SINK is not None:
            _LOG_SINK.append(line)

def info(msg):    _emit(f"{BL}[{_ts()}  INFO ]{NC}  {msg}")
def success(msg): _emit(f"{GA}[{_ts()}  PASS ]{NC}  {msg}")
def warn(msg):    _emit(f"{YL}[{_ts()}  CAUT ]{NC}  {msg}")
def fail(msg):    _emit(f"{RD}[{_ts()}  FAIL ]{NC}  {msg}")
def header(msg):  _emit(f"\n{DIM}{_BAR}{NC}\n{NW}{BOLD}  {msg}{NC}\n{DIM}{_BAR}{NC}\n")
def plain(msg):   _emit(msg)   # for output that has no PASS/CAUT/FAIL severity
                                # (compiler dumps, the analysis report, the summary)


# ══════════════════════════════════════════════════════════════════════════════
#  Rotor fleet — one shared parser, one shared baseline. Every reader of
#  rotor_config.csv (the GUI table, the test-card generator, the SQLite
#  export) goes through _read_rotor_rows_from_csv() and _ROTOR_BASELINE below,
#  so the "differs from baseline" highlighting and the override-suppression
#  logic always agree on what "default" means.
# ══════════════════════════════════════════════════════════════════════════════

# Baseline rotor spec. Rows that match these exactly are not emitted as
# test_card.json overrides (keeps the card clean for the all-default case),
# and are what the GUI table diffs against to highlight non-default cells.
_ROTOR_BASELINE = {
    "R_m": 1.45, "n_blades": 5, "chord_m": 0.096,
    "twist_root_deg": 16.0, "twist_tip_deg": 6.0,
    "pitch_offset_deg": 4.4, "P_max_kW": 236.0, "rpm_hover": 1284.0,
}
_ROTOR_BASELINE_POWERPLANT = "electric"
POWERPLANTS    = ["electric", "turbine_electric", "turboshaft"]

_TRUE_STRS  = {"true", "t", "yes", "y", "1"}
_FALSE_STRS = {"false", "f", "no", "n", "0"}

def _csv_bool(raw, default: bool = True) -> bool:
    """lift / thrust columns. Anything unrecognised (including blank) keeps the
    default so an older CSV without these columns reads as an all-tiltrotor
    fleet, which is what it was."""
    s = (raw or "").strip().lower()
    if s in _TRUE_STRS:
        return True
    if s in _FALSE_STRS:
        return False
    return default

def rotor_mode(lift: bool, thrust: bool) -> str:
    """TILT | LIFT | THRUST | OFF — see the mode table in rotor_mixer.jl."""
    if lift and thrust:
        return "TILT"
    if lift:
        return "LIFT"
    if thrust:
        return "THRUST"
    return "OFF"

def _read_rotor_rows_from_csv() -> list[dict]:
    """Parse subsystems/propulsion/rotor_config.csv. This is the ONLY CSV
    reader in the file — both the GUI table and the test-card generator go
    through it (or through the session override, see _effective_rotor_rows)."""
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
                    "lift":              _csv_bool(row.get("lift")),
                    "thrust":            _csv_bool(row.get("thrust")),
                    "notes":             (row.get("notes") or "").strip(),
                })
                rows[-1]["mode"] = rotor_mode(rows[-1]["lift"], rows[-1]["thrust"])
            except (KeyError, ValueError):
                continue
    return rows

def _rotor_fleet_overrides(rows: list[dict] | None = None) -> dict:
    """Build the rotor_fleet block for test_card.json: a per-rotor override
    entry for any rotor whose geometry/powerplant/mode differs from the
    baseline. Rows that are all-default produce no entry.

    rows=None reads the CSV directly — this is the CLI/headless path (no
    session concept applies there). The GUI path always passes explicit rows
    (the session override if one is active, otherwise a CSV read it already
    did), so this function never re-reads disk out from under a session edit.
    """
    if rows is None:
        rows = _read_rotor_rows_from_csv()
    overrides = []
    for r in rows:
        entry = {"rotor_id": r["rotor_id"]}
        changed = False
        for field_name, default in _ROTOR_BASELINE.items():
            val = r[field_name]
            if isinstance(default, int):
                val = int(round(val))
            if abs(val - default) > 1e-9:
                changed = True
            entry[field_name] = val
        powerplant = r.get("powerplant") or _ROTOR_BASELINE_POWERPLANT
        entry["powerplant"] = powerplant
        if powerplant != _ROTOR_BASELINE_POWERPLANT:
            changed = True
        lift, thrust = bool(r.get("lift", True)), bool(r.get("thrust", True))
        entry["lift"], entry["thrust"] = lift, thrust
        if not (lift and thrust):
            changed = True
        entry["notes"] = r.get("notes", "")
        if changed:
            overrides.append(entry)
    return {"overrides": overrides}

# ── Session-only rotor override (never written to rotor_config.csv) ──────────
_rotor_session_rows: list[dict] | None = None
_rotor_session_lock = threading.Lock()

def _effective_rotor_rows() -> tuple[list[dict], str]:
    """Rows to use for the *next* run: the session override if one is active,
    else a fresh CSV read. Returns (rows, source) where source is
    'session' or 'csv', so callers/log lines can say which was used."""
    with _rotor_session_lock:
        session = _rotor_session_rows
    if session is not None:
        return list(session), "session"
    return _read_rotor_rows_from_csv(), "csv"

def _validate_rotor_rows(rows_in: list) -> tuple[bool, list[dict], list[dict], list[str]]:
    """Validate a posted rotor fleet. Returns (ok, normalized_rows, errors,
    warnings). Never touches disk. Rejects the whole payload on any error —
    no partial application.

    The fleet size and the set of rotor_ids are fixed to whatever
    rotor_config.csv defines — this only edits the values of the rotors
    already in that file. Adding or removing a rotor changes the aircraft
    configuration and has to happen in the CSV directly.
    """
    errors:   list[dict] = []
    warnings: list[str]  = []
    n = len(rows_in)

    csv_rows = _read_rotor_rows_from_csv()
    expected_n = len(csv_rows) or n   # if the CSV is missing/empty, fall back
    expected_ids = {r["rotor_id"] for r in csv_rows} if csv_rows else None

    if n != expected_n:
        errors.append({"row": None, "field": "fleet", "message":
                        f"Fleet must have exactly {expected_n} rotor(s), matching "
                        f"rotor_config.csv — add or remove a rotor by editing the "
                        f"CSV directly, not from this run."})
        return False, [], errors, warnings
    if not (2 <= n <= 8):
        errors.append({"row": None, "field": "fleet", "message":
                        f"Invalid rotor count: {n}. Fleet must have 2–8 rotors."})
        return False, [], errors, warnings

    def _num(row, i, field_name, lo, hi):
        raw = row.get(field_name)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append({"row": i, "field": field_name, "message": "not a number"})
            return None
        if not (lo <= v <= hi):
            errors.append({"row": i, "field": field_name,
                            "message": f"{v} outside [{lo}, {hi}]"})
            return None
        return v

    normalized = []
    seen_ids: set[int] = set()
    for i, row in enumerate(rows_in):
        try:
            rid = int(row.get("rotor_id"))
        except (TypeError, ValueError):
            errors.append({"row": i, "field": "rotor_id", "message": "not an integer"})
            continue
        if rid in seen_ids:
            errors.append({"row": i, "field": "rotor_id", "message": f"duplicate id {rid}"})
        seen_ids.add(rid)

        R_m              = _num(row, i, "R_m",              0.1, 6.0)
        n_blades_raw     = row.get("n_blades")
        chord_m          = _num(row, i, "chord_m",           0.01, 1.0)
        twist_root_deg   = _num(row, i, "twist_root_deg",   -30.0, 60.0)
        twist_tip_deg    = _num(row, i, "twist_tip_deg",    -30.0, 60.0)
        pitch_offset_deg = _num(row, i, "pitch_offset_deg", -20.0, 20.0)
        P_max_kW         = _num(row, i, "P_max_kW",          1.0, 5000.0)
        rpm_hover        = _num(row, i, "rpm_hover",         100.0, 6000.0)

        try:
            n_blades = int(n_blades_raw)
            if not (2 <= n_blades <= 12):
                errors.append({"row": i, "field": "n_blades", "message": "outside [2, 12]"})
        except (TypeError, ValueError):
            errors.append({"row": i, "field": "n_blades", "message": "not an integer"})
            n_blades = None

        if R_m is not None and chord_m is not None and chord_m >= R_m:
            errors.append({"row": i, "field": "chord_m",
                            "message": "chord_m must be less than R_m"})

        powerplant = (row.get("powerplant") or "electric").strip()
        if powerplant not in POWERPLANTS:
            errors.append({"row": i, "field": "powerplant",
                            "message": f"must be one of {POWERPLANTS}"})

        lift   = bool(row.get("lift", True))
        thrust = bool(row.get("thrust", True))
        notes  = re.sub(r"[,\n\r]", " ", str(row.get("notes", ""))).strip()

        if None in (R_m, chord_m, twist_root_deg, twist_tip_deg, pitch_offset_deg,
                    P_max_kW, rpm_hover, n_blades):
            continue  # this row already carries at least one error above

        entry = {
            "rotor_id": rid, "R_m": R_m, "n_blades": n_blades, "chord_m": chord_m,
            "twist_root_deg": twist_root_deg, "twist_tip_deg": twist_tip_deg,
            "pitch_offset_deg": pitch_offset_deg, "P_max_kW": P_max_kW,
            "rpm_hover": rpm_hover, "powerplant": powerplant,
            "lift": lift, "thrust": thrust, "notes": notes,
        }
        entry["mode"] = rotor_mode(lift, thrust)
        normalized.append(entry)

        # advisory: tip Mach
        tip_speed = (rpm_hover / 60.0) * 2 * math.pi * R_m
        if tip_speed > 0.75 * 340.0:
            warnings.append(f"rotor {rid}: tip speed {tip_speed:.0f} m/s — above M0.75")

    if not errors and not any(r["lift"] for r in normalized):
        warnings.append("no lift-capable rotor in the fleet — hover will not converge")

    if not errors and expected_ids is not None:
        got_ids = {r["rotor_id"] for r in normalized}
        if got_ids != expected_ids:
            errors.append({"row": None, "field": "rotor_id", "message":
                            f"rotor IDs must match rotor_config.csv exactly "
                            f"(expected {sorted(expected_ids)}, got {sorted(got_ids)}) — "
                            f"renumbering, adding, or removing a rotor isn't allowed here."})

    return (len(errors) == 0), normalized, errors, warnings


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

# Wind group in a METAR: dddss[Ggmax]KT|MPS|KMH  (ddd or VRB, ss = sustained speed)
_METAR_WIND_RE = re.compile(r'\b(?:\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?(KT|MPS|KMH)\b')

# fly.jl runs a terrain-module self-test on every launch (fixed routes like
# KAXX-KSAF, KDEN-KCOS, checking predefined profiles against known deltas) —
# it's not about the mission being flown, so it's filtered out of both the
# console and the browser log rather than streamed through plain().
_TERRAIN_SELFTEST_INFO_RE = re.compile(r'^\[\s*Info:\s*Terrain:', re.IGNORECASE)
_TERRAIN_SELFTEST_CHECK_RE = re.compile(r'^\s*\[(?:PASS|FAIL)\]\s.*\bgot=.*\bexpected=', re.IGNORECASE)

def _is_terrain_selftest_line(line: str) -> bool:
    return bool(_TERRAIN_SELFTEST_INFO_RE.match(line)
                or _TERRAIN_SELFTEST_CHECK_RE.match(line))


# ══════════════════════════════════════════════════════════════════════════════
#  Flight engine — plan → build → simulate → analyse
# ══════════════════════════════════════════════════════════════════════════════

# ── planning/dash.py  →  DashConfig ───────────────────────────────────────────

@dataclass
class DashConfig:
    speed_kmh:              float           = 300.0
    altitude_ft:            float           = 11500.0
    hover_alt_m:            float           = 30.0
    back_trans_speed_ms:    float           = 50.0
    nacelle_tilt_deg:       float           = 65.0
    heading_deg:            Optional[float] = None
    turbulence_intensity_ms: float          = 0.0   # Dryden σ (m/s); 0=off
    terrain:     bool            = False

    @classmethod
    def load(cls) -> "DashConfig":
        import importlib.util
        path = PLANNING / "dash.py"
        if not path.exists():
            warn("planning/dash.py not found — using defaults (300 km/h, 11500 ft)")
            return cls()
        spec = importlib.util.spec_from_file_location("dash", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return cls(
            speed_kmh   = float(getattr(mod, "SPEED_KMH",   300.0)),
            altitude_ft = float(getattr(mod, "ALTITUDE_FT", 11500.0)),
            hover_alt_m = float(getattr(mod, "HOVER_ALT_M", 30.0)),
            heading_deg          = getattr(mod, "HEADING_DEG", None),
            terrain              = bool(getattr(mod,  "TERRAIN",              False)),
            back_trans_speed_ms  = float(getattr(mod, "BACK_TRANS_SPEED_MS", 50.0)),
            nacelle_tilt_deg     = float(getattr(mod, "NACELLE_TILT_DEG",   65.0)),
        )


# ── Airport database ──────────────────────────────────────────────────────────

@dataclass
class Airport:
    icao:   str
    lat:    float
    lon:    float
    elev_m: float
    name:   str = ""


def load_airports() -> dict[str, Airport]:
    """CSV format (header required): icao,lat_deg,lon_deg,elev_m[,name]"""
    path = PLANNING / "airports.csv"
    out: dict[str, Airport] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            try:
                icao = row["icao"].strip().upper()
                out[icao] = Airport(icao=icao,
                                    lat=float(row["lat_deg"]),
                                    lon=float(row["lon_deg"]),
                                    elev_m=float(row["elev_m"]),
                                    name=(row.get("name") or "").strip())
            except (KeyError, ValueError):
                continue
    return out


def get_airport(icao: str, airports: dict[str, Airport],
                interactive: bool = True) -> Airport:
    """Return Airport for icao; prompt user if not in database and
    interactive=True. interactive=False (always the case for GUI/server runs —
    there is no TTY to prompt on) raises instead of blocking forever."""
    if icao in airports:
        return airports[icao]
    if not interactive:
        raise ValueError(
            f"{icao} not in planning/airports.csv — add it there before "
            f"running from the GUI (interactive entry is disabled for server runs)")
    info(f"{icao} not in airports.csv")
    return Airport(
        icao=icao,
        lat   = float(input("  Latitude  (°N, negative for S): ")),
        lon   = float(input("  Longitude (°E, negative for W): ")),
        elev_m= float(input("  Elevation (m MSL):               ")),
    )


# ── METAR parser ──────────────────────────────────────────────────────────────

@dataclass
class MetarData:
    icao:          str
    temp_c:        float
    dewpoint_c:    float
    altimeter_pa:  float
    wind_from_deg: float
    wind_speed_ms: float
    wind_gust_ms:  float = 0.0
    raw:           str   = ""


_METAR_REPORT_TYPE_RE = re.compile(r'^(?:METAR|SPECI)\s+', re.IGNORECASE)


def strip_report_type(raw: str) -> str:
    """Drop a leading METAR/SPECI report-type token. aviationweather.gov's
    raw text sometimes carries it (SPECI marks an unscheduled special
    report) — everything downstream, here and in the GUI's manual-entry
    ICAO check, expects the station identifier to be the first token."""
    raw = (raw or "").strip()
    prev = None
    while prev != raw:               # tolerate a stray double prefix
        prev = raw
        raw = _METAR_REPORT_TYPE_RE.sub("", raw, count=1).strip()
    return raw


def parse_metar(raw: str) -> MetarData:
    raw   = strip_report_type(raw)
    parts = raw.split()
    if not parts:
        raise ValueError("Empty METAR")
    icao  = parts[0].upper()

    temp_c = dew_c = 0.0
    m = re.search(r'\b(M?\d{2})/(M?\d{2})\b', raw)
    if m:
        def dt(s): return -float(s[1:]) if s.startswith('M') else float(s)
        temp_c, dew_c = dt(m.group(1)), dt(m.group(2))
    t = re.search(r'\bT([01]\d{3})([01]\d{3})\b', raw)
    if t:
        def dtg(s): return (-1 if s[0]=='1' else 1)*int(s[1:])/10.0
        temp_c, dew_c = dtg(t.group(1)), dtg(t.group(2))

    alt_pa = STD_PRESSURE
    a = re.search(r'\bA(\d{4})\b', raw)
    if a: alt_pa = int(a.group(1))/100.0 * INHG_TO_PA
    q = re.search(r'\bQ(\d{4})\b', raw)
    if q: alt_pa = float(q.group(1))*100.0

    wf = ws = wg = 0.0
    w = re.search(r'\b(VRB|\d{3})(\d{2,3})(G(\d{2,3}))?(KT|MPS|KMH)\b', raw)
    if w:
        if w.group(1) != 'VRB': wf = float(w.group(1))
        spd  = float(w.group(2)); gst = float(w.group(4) or 0)
        u = w.group(5)
        if   u=='KT':  ws,wg = spd*MPS_PER_KT, gst*MPS_PER_KT
        elif u=='MPS': ws,wg = spd, gst
        elif u=='KMH': ws,wg = spd/3.6, gst/3.6

    return MetarData(icao=icao, temp_c=temp_c, dewpoint_c=dew_c,
                     altimeter_pa=alt_pa, wind_from_deg=wf,
                     wind_speed_ms=ws, wind_gust_ms=wg, raw=raw)


def read_metar(path: Path) -> MetarData:
    return parse_metar(path.read_text(encoding="utf-8", errors="replace").strip())


# ══════════════════════════════════════════════════════════════════════════════
#  Weather & terrain auto-resolution
#
#  The Route & Weather tab takes ICAO codes only.  Everything below turns a
#  dep/arr ICAO pair into the raw METAR text the rest of the pipeline already
#  expects — dep_metar/arr_metar stay raw strings on the wire into /launch →
#  plan_flight → generate_test_card, so nothing downstream changes.
#
#  Four inputs drive the decision table: internet, both airports in
#  airports.csv, METAR text available for both ends (live, cached, manually
#  entered, or bundled), and a predefined terrain pair in terrain.jl.
# ══════════════════════════════════════════════════════════════════════════════

ICAO_RE = re.compile(r"^[A-Z0-9]{4}$")

# Actions — the six the decision table can produce.
ACT_FLY         = "fly"                  # everything resolved, go
ACT_SRTM        = "srtm_then_fly"        # no predefined profile: download SRTM first
ACT_SYNTHETIC   = "synthetic_metar"      # no obs: ISA lapse-rate METAR from field elevation
ACT_MANUAL      = "manual_metar"         # offline: require manual METAR entry
ACT_MANUAL_FLAT = "manual_metar_flat"    # offline, no predefined pair: manual METAR + flat world
ACT_NO_FLY      = "do_not_fly"

# Key: (internet, both_airports_in_db, metar_available, predefined_pair).
#
# Seven rows are specified.  The remaining nine (marked "default") were never
# signed off on, so they take the most restrictive of the six actions rather
# than a guessed-at graceful degradation.  Revisit them deliberately — do not
# silently smarten them up.
DECISION_TABLE: dict[tuple[bool, bool, bool, bool], str] = {
    (True,  True,  True,  True ): ACT_FLY,
    (True,  True,  True,  False): ACT_SRTM,
    (True,  True,  False, True ): ACT_SYNTHETIC,
    (True,  True,  False, False): ACT_NO_FLY,       # default
    (True,  False, True,  True ): ACT_NO_FLY,
    (True,  False, True,  False): ACT_NO_FLY,       # default
    (True,  False, False, True ): ACT_NO_FLY,       # default
    (True,  False, False, False): ACT_NO_FLY,       # default
    # Offline: "METAR available" can only be true because someone typed it in,
    # so every offline row that can fly at all requires manual entry first.
    (False, True,  True,  True ): ACT_MANUAL,
    (False, True,  True,  False): ACT_MANUAL_FLAT,
    (False, True,  False, True ): ACT_MANUAL,
    (False, True,  False, False): ACT_MANUAL_FLAT,
    (False, False, True,  True ): ACT_NO_FLY,       # default
    (False, False, True,  False): ACT_NO_FLY,       # default
    (False, False, False, True ): ACT_NO_FLY,       # default
    (False, False, False, False): ACT_NO_FLY,
}


# ── Internet connectivity probe ───────────────────────────────────────────────

_net_lock  = threading.Lock()
_net_state: tuple[float, bool] | None = None   # (monotonic_at, reachable)


def has_internet(force: bool = False) -> bool:
    """TCP-connect to the METAR host — DNS + handshake only, no request body.
    Cached for NET_PROBE_TTL_S; force=True re-checks (every launch attempt
    does, rather than trusting a probe from minutes ago)."""
    global _net_state
    now = time.monotonic()
    if not force:
        with _net_lock:
            if _net_state and (now - _net_state[0]) < NET_PROBE_TTL_S:
                return _net_state[1]
    try:
        socket.create_connection((METAR_HOST, 443),
                                 timeout=NET_PROBE_TIMEOUT).close()
        ok = True
    except OSError:
        ok = False
    with _net_lock:
        _net_state = (now, ok)
    return ok


# ── METAR fetch (server-side proxy for GET /metar) ────────────────────────────

@dataclass
class MetarFetch:
    icao:    str
    raw:     str   = ""
    status:  str   = "error"   # live | cache | bundled | empty | error | invalid
    age_s:   Optional[float] = None
    message: str   = ""

    @property
    def usable(self) -> bool:
        return bool(self.raw)


_metar_cache: dict[str, tuple[float, str]] = {}   # icao -> (unix_fetched_at, raw)
_metar_lock  = threading.Lock()                   # same pattern as _sim_lock


def fetch_metar(icao: str, allow_network: bool = True) -> MetarFetch:
    """Resolve one station's observation.  Cache hit → cache; live fetch →
    live; station reporting nothing → empty (a distinct, non-fatal case);
    network trouble → error, falling back to a stale cache entry if there is
    one.  Never raises: every caller degrades instead of blocking Launch."""
    icao = (icao or "").strip().upper()
    if not ICAO_RE.match(icao):
        return MetarFetch(icao=icao, status="invalid",
                          message="ICAO must be 4 letters/digits")

    now = time.time()
    with _metar_lock:
        hit = _metar_cache.get(icao)
    if hit and (now - hit[0]) < METAR_TTL_S:
        return MetarFetch(icao=icao, raw=hit[1], status="cache",
                          age_s=now - hit[0])

    def stale(msg: str) -> MetarFetch:
        if hit:
            return MetarFetch(icao=icao, raw=hit[1], status="cache",
                              age_s=now - hit[0], message=f"{msg}; using cached obs")
        return MetarFetch(icao=icao, status="error", message=msg)

    if not allow_network:
        return stale("offline")

    req = urllib.request.Request(METAR_URL.format(icao=icao),
                                 headers={"User-Agent": METAR_UA})
    try:
        with urllib.request.urlopen(req, timeout=METAR_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as e:                       # URLError, timeout, HTTPError…
        return stale(f"fetch failed: {e}")

    raw = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    raw = strip_report_type(raw)
    if not raw:
        # Station exists but has no current observation — not an error, and
        # not something to cache.
        return MetarFetch(icao=icao, status="empty",
                          message="station reported no current observation")

    with _metar_lock:
        _metar_cache[icao] = (now, raw)
    return MetarFetch(icao=icao, raw=raw, status="live", age_s=0.0)


# ── Predefined terrain pairs (read from terrain.jl) ───────────────────────────

# Matches the dict entries in terrain.jl's PREDEFINED_PROFILES ("KDEP-KARR" =>).
# Parsed out of the source rather than duplicated here so the two cannot drift.
_PREDEF_KEY_RE = re.compile(r'"([A-Z0-9]{4})-([A-Z0-9]{4})"\s*=>')
_predef_cache: tuple[float, frozenset[str]] | None = None
_predef_lock  = threading.Lock()


def predefined_pairs() -> frozenset[str]:
    """Route keys with a hardcoded profile in terrain.jl.  Empty set if the
    file is missing — that reads as "no predefined pair", the conservative
    side of the table."""
    global _predef_cache
    try:
        mtime = TERRAIN_JL.stat().st_mtime
    except OSError:
        return frozenset()
    with _predef_lock:
        if _predef_cache and _predef_cache[0] == mtime:
            return _predef_cache[1]
    try:
        src = TERRAIN_JL.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return frozenset()
    # Drop line comments first — terrain.jl documents the key format with a
    # literal "KDEP-KARR" => ... example that is not a real route.
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    keys = frozenset(f"{a}-{b}" for a, b in _PREDEF_KEY_RE.findall(code))
    with _predef_lock:
        _predef_cache = (mtime, keys)
    return keys


def has_predefined_pair(dep_icao: str, arr_icao: str) -> bool:
    """True if terrain.jl can serve this route from PREDEFINED_PROFILES.
    Either direction counts — load_terrain reverses a stored profile."""
    keys = predefined_pairs()
    dep, arr = dep_icao.upper(), arr_icao.upper()
    return f"{dep}-{arr}" in keys or f"{arr}-{dep}" in keys


# ── Synthetic METAR (ISA standard day at field elevation) ─────────────────────

def _metar_temp_group(temp_c: float) -> str:
    v = int(round(temp_c))
    return f"M{abs(v):02d}" if v < 0 else f"{v:02d}"


def synthetic_metar(ap: Airport) -> str:
    """ISA standard day at ap.elev_m, emitted as raw METAR text so it rides the
    same wire as a real observation (plan_flight parses raw).  Temperature
    comes off the standard lapse rate; QNH is 1013 hPa by definition of ISA —
    the field-elevation correction happens downstream in station_pressure().
    Calm wind.  Dewpoint is cosmetic: nothing downstream reads it."""
    temp_c  = (STD_TEMP_K - LAPSE_RATE * ap.elev_m) - 273.15
    qnh_hpa = int(round(STD_PRESSURE / 100.0))
    stamp   = datetime.datetime.now(UTC).strftime("%d%H%MZ")
    return (f"{ap.icao} {stamp} 00000KT 9999 "
            f"{_metar_temp_group(temp_c)}/{_metar_temp_group(temp_c - 5.0)} "
            f"Q{qnh_hpa:04d} RMK SYNTHETIC ISA")


# ── Route resolution ──────────────────────────────────────────────────────────

@dataclass
class RouteResolution:
    ok:            bool
    action:        str
    message:       str
    dep_metar:     str  = ""
    arr_metar:     str  = ""
    force_terrain: Optional[bool] = None   # True: fetch SRTM · False: flat world
                                           # None: leave the GUI toggle alone
    need_manual:   list[str] = field(default_factory=list)   # ["dep", "arr"]
    suggest:       dict = field(default_factory=dict)        # prefill for manual entry
    log:           list[str] = field(default_factory=list)


def _resolve_one(icao: str, manual: str, internet: bool) -> tuple[str, str]:
    """(raw_metar, source) for one end.  Manual text wins, then live/cached,
    then the bundled reference for the default route.  '' if nothing."""
    manual = (manual or "").strip()
    if manual:
        return manual, "manual"
    f = fetch_metar(icao, allow_network=internet)
    if f.usable:
        return f.raw, f.status
    bundled = BUNDLED_METAR.get(icao)
    if bundled:
        return bundled, "bundled"
    return "", f.status


def resolve_route(dep_icao: str, arr_icao: str,
                  dep_manual: str = "", arr_manual: str = "",
                  force_net_probe: bool = False) -> RouteResolution:
    """Evaluate the decision table for one dep→arr pair and return either the
    raw METAR pair to fly with, or a refusal explaining which input failed."""
    dep_icao = (dep_icao or "").strip().upper()
    arr_icao = (arr_icao or "").strip().upper()
    dep_manual = strip_report_type(dep_manual)
    arr_manual = strip_report_type(arr_manual)

    bad = [f"{lbl}={c or '(blank)'}"
           for lbl, c in (("departure", dep_icao), ("arrival", arr_icao))
           if not ICAO_RE.match(c)]
    if bad:
        return RouteResolution(
            ok=False, action=ACT_NO_FLY,
            message=f"Invalid ICAO code: {', '.join(bad)} — "
                    f"expected four letters or digits.")

    # plan_flight takes the airport from the METAR's first token, so a manual
    # paste that disagrees with the ICAO field would silently fly a different
    # route than the one on screen.
    for label, icao, manual in (("Departure", dep_icao, dep_manual),
                                ("Arrival",   arr_icao, arr_manual)):
        first = (manual or "").strip().split()
        if first and first[0].upper() != icao:
            return RouteResolution(
                ok=False, action=ACT_NO_FLY,
                message=f"{label} METAR is for {first[0].upper()}, not {icao} — "
                        f"fix the ICAO field or the pasted text.")

    airports = load_airports()
    missing  = [c for c in (dep_icao, arr_icao) if c not in airports]
    in_db    = not missing

    internet   = has_internet(force=force_net_probe)
    dep_raw, dep_src = _resolve_one(dep_icao, dep_manual, internet)
    arr_raw, arr_src = _resolve_one(arr_icao, arr_manual, internet)
    metar_ok   = bool(dep_raw and arr_raw)
    predefined = has_predefined_pair(dep_icao, arr_icao)

    action = DECISION_TABLE[(internet, in_db, metar_ok, predefined)]

    log = [
        f"Weather/terrain resolution — internet={'yes' if internet else 'no'}  "
        f"airports in DB={'yes' if in_db else 'no'}  "
        f"METAR available={'yes' if metar_ok else 'no'}  "
        f"predefined terrain pair={'yes' if predefined else 'no'}  "
        f"→ {action}",
        f"DEP {dep_icao}: {dep_src}    ARR {arr_icao}: {arr_src}",
    ]
    state = (f"(internet={'yes' if internet else 'no'}, "
             f"airports in DB={'yes' if in_db else 'no'}, "
             f"METAR={'yes' if metar_ok else 'no'}, "
             f"predefined terrain={'yes' if predefined else 'no'})")

    if action == ACT_NO_FLY:
        why = (f"{' and '.join(missing)} not in planning/airports.csv"
               if missing else
               "no weather available and no predefined terrain profile for this route"
               if internet else
               "offline with no usable inputs")
        return RouteResolution(ok=False, action=action, log=log,
                               message=f"Do not fly — {why} {state}.")

    if action in (ACT_MANUAL, ACT_MANUAL_FLAT):
        flat = action == ACT_MANUAL_FLAT
        need = [w for w, m in (("dep", dep_manual), ("arr", arr_manual))
                if not (m or "").strip()]
        if need:
            # Offline, so anything we "have" was not fetched live — the user
            # confirms it by hand.  Bundled reference text is offered as a
            # prefill, never assumed.
            suggest = {w: BUNDLED_METAR[c]
                       for w, c in (("dep", dep_icao), ("arr", arr_icao))
                       if c in BUNDLED_METAR and w in need}
            return RouteResolution(
                ok=False, action=action, need_manual=need, suggest=suggest, log=log,
                message=("No internet — enter the METAR for "
                         f"{' and '.join(need)} manually before launching"
                         + (" (terrain falls back to the flat world model: no "
                            "predefined profile for this route)" if flat else "")
                         + f" {state}."))
        log.append("Offline: flying on manually entered METARs"
                   + (" with flat-world terrain" if flat else ""))
        return RouteResolution(ok=True, action=action, log=log,
                               dep_metar=dep_manual.strip(),
                               arr_metar=arr_manual.strip(),
                               force_terrain=False if flat else None,
                               message="Manual METARs accepted.")

    if action == ACT_SYNTHETIC:
        for icao, raw, which in ((dep_icao, dep_raw, "dep"), (arr_icao, arr_raw, "arr")):
            if not raw:
                gen = synthetic_metar(airports[icao])
                log.append(f"{which.upper()} {icao}: no observation — "
                           f"synthetic ISA METAR from {airports[icao].elev_m:.0f} m field "
                           f"elevation: {gen}")
                if which == "dep": dep_raw = gen
                else:              arr_raw = gen
        return RouteResolution(ok=True, action=action, log=log,
                               dep_metar=dep_raw, arr_metar=arr_raw,
                               message="Flying on synthetic ISA weather.")

    if action == ACT_SRTM:
        log.append("No predefined terrain profile for this route — "
                   "SRTM download enabled for this run")
        return RouteResolution(ok=True, action=action, log=log,
                               dep_metar=dep_raw, arr_metar=arr_raw,
                               force_terrain=True,
                               message="SRTM terrain will be built before the run.")

    return RouteResolution(ok=True, action=ACT_FLY, log=log,
                           dep_metar=dep_raw, arr_metar=arr_raw,
                           message="Route resolved.")


# ── Geodesy helpers ────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    la1,lo1 = math.radians(lat1), math.radians(lon1)
    la2,lo2 = math.radians(lat2), math.radians(lon2)
    a = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*R_EARTH*math.asin(math.sqrt(min(a,1.0)))


def initial_bearing(lat1, lon1, lat2, lon2) -> float:
    la1,lo1 = math.radians(lat1), math.radians(lon1)
    la2,lo2 = math.radians(lat2), math.radians(lon2)
    y = math.sin(lo2-lo1)*math.cos(la2)
    x = math.cos(la1)*math.sin(la2)-math.sin(la1)*math.cos(la2)*math.cos(lo2-lo1)
    return math.degrees(math.atan2(y,x)) % 360


def station_pressure(qnh_pa: float, elev_m: float) -> float:
    return qnh_pa*(1-LAPSE_RATE*elev_m/STD_TEMP_K)**(G/(R_DRY_AIR*LAPSE_RATE))


def density_altitude_ft(pressure_pa: float, temp_c: float) -> float:
    rho    = pressure_pa/(R_DRY_AIR*(temp_c+273.15))
    rho_sl = STD_PRESSURE/(R_DRY_AIR*STD_TEMP_K)
    return 145366*(1-(rho/rho_sl)**0.2349)


# ── Test card  →  planning/test_card.json ─────────────────────────────────────

def generate_test_card(dep: Airport, arr: Airport,
                       dep_wx: MetarData, arr_wx: MetarData,
                       cfg: DashConfig,
                       rotor_rows: list[dict] | None = None) -> dict:
    dist_m  = haversine(dep.lat, dep.lon, arr.lat, arr.lon)
    brg     = initial_bearing(dep.lat, dep.lon, arr.lat, arr.lon)
    brg_rad = math.radians(brg)

    x_m = round(dist_m * math.cos(brg_rad), 0)
    y_m = round(dist_m * math.sin(brg_rad), 0)
    z_m = round(arr.elev_m - dep.elev_m, 0)

    # Systematic offset pre-correction: waypoint is set 280 m upstream in
    # fly.jl (make_ap_config). Shift nominal target by the same amount.
    x_m -= round(280.0 * math.cos(brg_rad), 1)
    y_m -= round(280.0 * math.sin(brg_rad), 1)

    dash_alt_agl = cfg.altitude_ft * FT_TO_M - dep.elev_m
    initial_hdg  = cfg.heading_deg if cfg.heading_deg is not None else round(brg, 1)

    dep_psta = station_pressure(dep_wx.altimeter_pa, dep.elev_m)
    arr_psta = station_pressure(arr_wx.altimeter_pa, arr.elev_m)
    dep_da   = density_altitude_ft(dep_psta, dep_wx.temp_c)
    arr_da   = density_altitude_ft(arr_psta, arr_wx.temp_c)

    return {
        "_comment":   (f"AIRCRAFT — {dep.icao} → {arr.icao}  "
                       f"{dist_m/1000:.1f} km  {brg:.0f}°  "
                       f"Generated by windblade.py"),
        "_version":   "0.2.0",
        "_generated": {
            "dep_metar":           dep_wx.raw,
            "arr_metar":           arr_wx.raw,
            "distance_km":         round(dist_m/1000, 1),
            "initial_bearing_deg": round(brg, 1),
            "dep_density_alt_ft":  round(dep_da),
            "arr_density_alt_ft":  round(arr_da),
        },
        "preflight":  {"hold_s": 5.0, "ramp_s": 2.0},
        "hover":      {"alt_m": cfg.hover_alt_m, "climb_rate_ms": 3.0},
        "transition": {"duration_s": 10.0, "thrust_comp": 0.5},
        "fixed_wing": {
            "dash_speed_kmh":     cfg.speed_kmh,
            "dash_altitude_m":    round(dash_alt_agl, 0),
            "climb_rate_fw_ms":   5.0,
            "descent_rate_fw_ms": 4.0,
            "nacelle_tilt_deg":   max(45.0, min(90.0, cfg.nacelle_tilt_deg)),
        },
        "landing": {
            "pitch_up_deg":       35.0, "pitch_up_rate_s":    4.0,
            "pitch_hold_s":       10.0, "pitch_down_s":       4.0,
            "tilt_s":             12.0, "thrust_comp":        0.6,
            "descent_rate_ms":     1.5,
            "back_trans_entry_ms": cfg.back_trans_speed_ms,
        },
        "airport": {
            "icao":                dep.icao,
            "alt_m":               dep.elev_m,
            "ambient_temp_c":      dep_wx.temp_c,
            "ambient_pressure_pa": round(dep_wx.altimeter_pa, 0),
            "wind_from_deg":       dep_wx.wind_from_deg,
            "wind_speed_ms":       round(dep_wx.wind_speed_ms, 2),
        },
        "destination": {
            "icao":                arr.icao,
            "alt_m":               arr.elev_m,
            "ambient_temp_c":      arr_wx.temp_c,
            "ambient_pressure_pa": round(arr_wx.altimeter_pa, 0),
            "wind_from_deg":       arr_wx.wind_from_deg,
            "wind_speed_ms":       round(arr_wx.wind_speed_ms, 2),
        },
        "navigation": {
            "return_to_base":      False,
            "initial_heading_deg": initial_hdg,
            "target":              {"x_m": x_m, "y_m": y_m, "z_m": z_m},
        },
        "rotor_fleet": _rotor_fleet_overrides(rotor_rows),
        "turbulence_intensity_ms": cfg.turbulence_intensity_ms,
    }


# ── SRTM terrain profile  →  planning/terrain_profile.json ────────────────────

def _tile_name(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "W" if lon < 0 else "E"
    return f"{ns}{int(math.floor(lat)):02d}{ew}{int(math.floor(abs(lon))):03d}"


def _download_tile(tile: str, cache_dir: Path) -> Optional[Path]:
    hgt = cache_dir / f"{tile}.hgt"
    if hgt.exists():
        info(f"Terrain: cached {tile}.hgt"); return hgt
    cache_dir.mkdir(parents=True, exist_ok=True)
    for tmpl in SRTM_MIRRORS:
        url = tmpl.format(tile=tile)
        info(f"Terrain: downloading {url}")
        tmp = cache_dir / f"{tile}.tmp"
        try:
            try:
                urllib.request.urlopen(
                    urllib.request.Request(url, method="HEAD"), timeout=10)
            except urllib.error.HTTPError as he:
                if he.code == 404: continue
            urllib.request.urlretrieve(url, tmp)
            if url.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(tmp) as zf:
                    names = [n for n in zf.namelist() if n.endswith(".hgt")]
                    zf.extract(names[0], cache_dir)
                    (cache_dir/names[0]).rename(hgt)
                tmp.unlink(missing_ok=True)
            elif url.endswith(".gz"):
                with gzip.open(tmp,"rb") as gz, open(hgt,"wb") as out_f: out_f.write(gz.read())
                tmp.unlink(missing_ok=True)
            else:
                tmp.rename(hgt)
            info(f"Terrain: {hgt} ({hgt.stat().st_size//1024} KB)")
            return hgt
        except Exception as e:
            warn(f"Terrain: mirror failed — {e}")
            if tmp.exists(): tmp.unlink(missing_ok=True)
    warn(f"Terrain: could not download {tile}.hgt — place manually in {cache_dir}")
    return None


def _read_hgt(path: Path):
    data   = path.read_bytes()
    n      = int(math.sqrt(len(data) // 2))
    grid   = np.frombuffer(data, dtype=">i2").reshape(n, n).astype(np.float64)
    grid   = np.ascontiguousarray(grid)
    grid[grid == -32768] = np.nan
    stem   = path.stem
    lat_sw = float(stem[1:3]) * (-1 if stem[0] == "S" else 1)
    lon_sw = float(stem[4:7]) * (-1 if stem[3] == "W" else 1)
    return grid, lat_sw, lon_sw


def _sample(grids, lat, lon):
    t = _tile_name(lat, lon)
    if t not in grids:
        return np.nan
    grid, lat_sw, lon_sw = grids[t]
    n    = grid.shape[0]
    step = 1.0 / (n - 1)
    rf = (lat_sw + 1 - lat) / step
    cf = (lon - lon_sw) / step
    r0 = int(np.clip(rf, 0, n - 2))
    c0 = int(np.clip(cf, 0, n - 2))
    fr, fc = rf - r0, cf - c0
    def g(r, c):
        v = grid[r, c]
        return 0.0 if np.isnan(v) else float(v)
    return (g(r0,   c0  ) * (1-fr) * (1-fc)
          + g(r0,   c0+1) * (1-fr) * fc
          + g(r0+1, c0  ) * fr     * (1-fc)
          + g(r0+1, c0+1) * fr     * fc)


def build_terrain_profile(dep: Airport, arr: Airport,
                           brg_deg: float, n_pts: int = 200,
                           cache_dir: Optional[Path] = None) -> Optional[dict]:
    if cache_dir is None: cache_dir = Path.home()/".cache"/"srtm"
    needed = {_tile_name(dep.lat+i/(n_pts-1)*(arr.lat-dep.lat),
                          dep.lon+i/(n_pts-1)*(arr.lon-dep.lon))
              for i in range(n_pts)}
    info(f"Terrain: tiles needed: {sorted(needed)}")
    grids = {}
    for tile in sorted(needed):
        p = _download_tile(tile, cache_dir)
        if p:
            try: grids[tile] = _read_hgt(p)
            except Exception as e: warn(f"Terrain: read {p.name} — {e}")
    missing = needed - set(grids)
    if missing: warn(f"Terrain: {len(missing)} unavailable {sorted(missing)} — linear fallback")
    if not grids: return None
    dist_m = haversine(dep.lat, dep.lon, arr.lat, arr.lon)
    fs      = np.linspace(0.0, 1.0, n_pts)
    lats    = dep.lat + fs * (arr.lat - dep.lat)
    lons    = dep.lon + fs * (arr.lon - dep.lon)
    xs      = (fs * dist_m).round(1).tolist()
    fallback = dep.elev_m + fs * (arr.elev_m - dep.elev_m)
    raw_elv  = np.array([_sample(grids, la, lo) for la, lo in zip(lats, lons)])
    zs_arr   = np.where(np.isnan(raw_elv), fallback, raw_elv).round(1)
    zs       = zs_arr.tolist()
    info(f"Terrain: {n_pts} pts  elev {zs_arr.min():.0f}–{zs_arr.max():.0f} m")
    return {"x_m": xs, "elev_m": zs, "origin_elev_m": dep.elev_m,
            "source": "SRTM GL3", "dep": dep.icao, "arr": arr.icao, "n_points": n_pts}


# ── Stage 1: flight planning ──────────────────────────────────────────────────

def plan_flight(cfg: DashConfig, terrain_flag: bool,
               dep_metar: Optional[str] = None,
               arr_metar: Optional[str] = None,
               rotor_rows: list[dict] | None = None,
               interactive: bool = True) -> Path:
    dep_wx = parse_metar(dep_metar) if dep_metar else read_metar(PLANNING / "METAR_DEP")
    arr_wx = parse_metar(arr_metar) if arr_metar else read_metar(PLANNING / "METAR_ARR")

    info(f"DEP: {dep_wx.icao}  {dep_wx.temp_c:.1f}°C  "
         f"{dep_wx.altimeter_pa/100:.0f} hPa  "
         f"wind {dep_wx.wind_speed_ms:.1f} m/s from {dep_wx.wind_from_deg:.0f}°")
    info(f"ARR: {arr_wx.icao}  {arr_wx.temp_c:.1f}°C  "
         f"{arr_wx.altimeter_pa/100:.0f} hPa  "
         f"wind {arr_wx.wind_speed_ms:.1f} m/s from {arr_wx.wind_from_deg:.0f}°")

    airports = load_airports()
    dep = get_airport(dep_wx.icao, airports, interactive=interactive)
    arr = get_airport(arr_wx.icao, airports, interactive=interactive)

    info(f"{dep.icao}: {dep.lat:.4f}°N  {dep.lon:.4f}°E  {dep.elev_m:.0f} m MSL")
    info(f"{arr.icao}: {arr.lat:.4f}°N  {arr.lon:.4f}°E  {arr.elev_m:.0f} m MSL")

    card = generate_test_card(dep, arr, dep_wx, arr_wx, cfg, rotor_rows=rotor_rows)
    g    = card["_generated"]
    z_m  = card["navigation"]["target"]["z_m"]
    info(f"Route: {dep.icao} → {arr.icao}  {g['distance_km']} km  {g['initial_bearing_deg']}°")
    info(f"Cruise: {cfg.altitude_ft:.0f} ft MSL  "
         f"({card['fixed_wing']['dash_altitude_m']:.0f} m AGL)")
    info(f"Elevation offset: {z_m:+.0f} m  "
         f"({'arrival lower' if z_m < 0 else 'arrival higher'})")
    info(f"Density altitude — dep: {g['dep_density_alt_ft']:.0f} ft  "
         f"arr: {g['arr_density_alt_ft']:.0f} ft")

    card_path = PLANNING / "test_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    success(f"test_card.json → {card_path}")

    if terrain_flag or cfg.terrain:
        info("Building terrain profile...")
        profile = build_terrain_profile(dep, arr, brg_deg=g["initial_bearing_deg"])
        if profile:
            prof = PLANNING / "terrain_profile.json"
            prof.write_text(json.dumps(profile, separators=(",", ":")))
            success(f"terrain_profile.json ({profile['n_points']} pts)")
        else:
            warn("Terrain unavailable — predefined profile or flat_model will be used")

    return card_path


# ── Stage 2: build ─────────────────────────────────────────────────────────────

def _compile_so(src: Path, out: Path, extra_flags: list[str] = []) -> bool:
    """Compile a single C++ source to a shared library. Returns True on success."""
    cmd = ["g++", "-O3", "-std=c++17", "-fPIC", "-shared",
           *extra_flags, "-o", str(out), str(src)]
    if not getattr(_compile_so, "_version_printed", False):
        rv = subprocess.run(["g++", "--version"], capture_output=True, text=True)
        info(f"g++ version: {rv.stdout.splitlines()[0] if rv.stdout else rv.stderr.strip()}")
        _compile_so._version_printed = True
    info(f"cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    info(f"exit: {r.returncode}  stdout={repr(r.stdout[:200])}  stderr={repr(r.stderr[:200])}")
    if r.returncode != 0:
        fail(f"Compilation failed ({src.name}) [exit {r.returncode}]:")
        if r.stdout.strip(): plain(r.stdout)
        if r.stderr.strip(): plain(r.stderr)
        return False
    if r.stderr.strip():
        warn(f"Compiler warnings ({src.name}):")
        plain(r.stderr)
    if not out.exists():
        fail(f"Compilation appeared to succeed (exit 0) but {out.name} was not created.")
        if r.stderr.strip(): plain(r.stderr)
        return False
    if out.stat().st_size < 1024:
        fail(f"{out.name} is suspiciously small ({out.stat().st_size} bytes) — "
             f"likely an empty library. Check that all source files were compiled.")
        return False
    return True


def build_autopilot() -> str:
    """Compile autopilot.cpp to a versioned shared library (autopilot_<ts>.so)
    so Julia dlopens fresh on every run — no process restart needed."""
    src_ap = CONTROLS / "autopilot.cpp"
    if not src_ap.exists():
        raise FileNotFoundError(f"autopilot.cpp not found at {src_ap}")

    ver = str(int(time.time()))

    so_ap = CONTROLS / f"autopilot_{ver}.so"
    info(f"Compiling autopilot_{ver}.so")
    if not _compile_so(src_ap, so_ap):
        raise RuntimeError("autopilot build failed")
    (CONTROLS / "autopilot.version").write_text(ver)
    success(f"autopilot_{ver}.so")
    for old in sorted(CONTROLS.glob("autopilot_*.so"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[1:]:
        old.unlink(); info(f"Pruned {old.name}")

    hc = CONTROLS / "hotas.c"
    if hc.exists():
        rh = subprocess.run(["gcc", "-O2", "-o", str(CONTROLS/"hotas"), str(hc)],
                            capture_output=True, text=True)
        success("hotas") if rh.returncode == 0 else warn("hotas build failed")

    return ver


# ── Stage 3: simulate ──────────────────────────────────────────────────────────

_active_proc: subprocess.Popen | None = None
_active_proc_lock = threading.Lock()
_stop_requested = threading.Event()

def _set_active_proc(p: subprocess.Popen | None) -> None:
    global _active_proc
    with _active_proc_lock:
        _active_proc = p

def request_stop() -> bool:
    """Called from the /stop HTTP handler. Sends SIGINT to the running fly.jl
    child (if any) and flags run_simulation to treat that as a normal manual
    termination rather than a failure. Returns True if a process was signalled."""
    _stop_requested.set()
    with _active_proc_lock:
        p = _active_proc
    if p is None:
        return False
    try:
        p.send_signal(signal.SIGINT)
        return True
    except Exception as e:
        warn(f"stop: could not signal child process: {e}")
        return False


def run_simulation(gui: bool, manual: bool,
                   speed: Optional[float], out_dir: Path) -> Path:
    """Launch fly.jl and return the absolute path of the CSV it produced.

    Streams the child process line-by-line through plain() rather than
    inheriting stdio, so fly.jl's output reaches the browser log during a
    GUI run.
    """
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"dash_results_{ts}.csv"
    csv_path = out_dir / csv_name

    env = os.environ.copy()
    env["FLYSIM_CSV_PATH"] = str(csv_path)
    if speed is not None:
        env["FLYSIM_SPEED"] = str(speed)
        if gui:
            warn("--speed with --gui: GUI rendering will throttle the sim; "
                 "drop --gui for accurate speed runs")

    no_gui = (not gui) or (speed is not None and not gui)

    threads = env.get("JULIA_NUM_THREADS", "auto")
    cmd = ["julia", f"--threads={threads}", str(ROOT / "fly.jl")]
    if no_gui:    cmd.append("--no-gui")
    if manual:    cmd.append("--manual")

    info(f"Command: {' '.join(cmd)}")
    info(f"CSV target: {csv_path}")

    _stop_requested.clear()
    t0  = time.time()
    proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    _set_active_proc(proc)
    try:
        for line in proc.stdout:
            if _is_terrain_selftest_line(line):
                continue
            plain(line.rstrip())
        ret = proc.wait()
    finally:
        _set_active_proc(None)
    elapsed = time.time() - t0

    if ret != 0:
        if _stop_requested.is_set():
            warn(f"Simulation stopped by user (rc={ret}) after {elapsed:.0f}s "
                 f"— attempting to salvage CSV")
        else:
            raise RuntimeError(f"fly.jl exited with error (code {ret})")
    else:
        success(f"Simulation complete in {elapsed:.0f}s")

    if csv_path.exists():
        rows = sum(1 for _ in csv_path.open()) - 1
        success(f"CSV: {csv_path}  ({rows} rows)")
        return csv_path

    candidates = sorted(
        list(out_dir.glob("dash_results_*.csv")) +
        ([] if out_dir == ROOT else list(ROOT.glob("dash_results_*.csv"))),
        key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise FileNotFoundError(
            f"No dash_results CSV found in {out_dir} or {ROOT}. "
            "Check fly.jl output or set FLYSIM_CSV_PATH explicitly.")

    found = candidates[0]
    warn(f"Expected {csv_name} but found {found.name} — using that")
    rows = sum(1 for _ in found.open()) - 1
    success(f"CSV: {found}  ({rows} rows)")
    return found


# ── Stage 4: analysis ──────────────────────────────────────────────────────────

SOC_MIN          = 20.0    # % minimum arrival SoC
CRUISE_SPEED_TOL = 10.0    # km/h
CRUISE_ALT_TOL   = 150.0   # ft
LAND_ACCURACY_M  = 50.0    # m
RPM_ASYM_LIMIT   = 0.05    # fraction
GZ_NORMAL_LIMIT  = 1.5     # g
GZ_EMERGENCY_LIM = 2.5     # g

RPM_COLS = [f"rpm_r{i}" for i in range(1, 7)]

def _steady(df, phase, tail_frac=0.20):
    rows = df[df["phase"] == phase]
    if rows.empty: return rows
    n = len(rows)
    if phase == "dash" and n >= 20:
        lo = int(n * 0.25); hi = int(n * 0.75)
        return rows.iloc[lo:hi]
    return rows.iloc[-max(1, int(n * tail_frac)):]

def _gear_events(df):
    gc = df["gear_contact"].astype(float)
    return df[(gc==1) & (gc.shift(1, fill_value=0)==0)]

def _check_phase_sequence(df):
    REQUIRED = ["landed", "hover", "transition", "fw_climb", "dash", "descent"]
    OPTIONAL = ["fw_descent", "back_transition"]
    seen    = df["phase"].drop_duplicates().tolist()
    missing_req = [p for p in REQUIRED if p not in seen]
    if missing_req:
        return False, f"Missing required phases: {', '.join(missing_req)}"
    CANONICAL = ["landed", "hover", "transition", "fw_climb", "dash",
                 "fw_descent", "back_transition", "descent"]
    present   = [p for p in CANONICAL if p in seen]
    first_idx = {p: seen.index(p) for p in present}
    ordered   = sorted(present, key=lambda p: first_idx[p])
    if ordered != present:
        bad = next((p for p, o in zip(present, ordered) if p != o), "?")
        return False, f"Phase '{bad}' out of order. Observed: {' → '.join(ordered)}"
    optional_found = [p for p in OPTIONAL if p in seen]
    note = f" (optional: {', '.join(optional_found)})" if optional_found else " (fw_descent/back_transition not observed — fast decel)"
    return True, f"All {len(REQUIRED)} required phases present in order{note}"

def _check_cruise_speed(df, tc):
    target = float(tc.get("fixed_wing",{}).get("dash_speed_kmh", 320.0))
    steady = _steady(df, "dash")
    if steady.empty: return False, f"No 'dash' phase rows", None, CRUISE_SPEED_TOL
    spd = steady["speed_kmh"].to_numpy()
    max_err = float(np.abs(spd-target).max()); rms_err = float(np.sqrt(((spd-target)**2).mean()))
    return max_err<=CRUISE_SPEED_TOL, (f"mean {spd.mean():.1f} km/h | target {target:.0f} | "
        f"max err {max_err:.1f} | RMS {rms_err:.1f} km/h (limit ±{CRUISE_SPEED_TOL:.0f})"), max_err, CRUISE_SPEED_TOL

def _check_cruise_alt(df, tc):
    dash_m  = float(tc.get("fixed_wing",{}).get("dash_altitude_m", 951.0))
    orig_ft = float(tc.get("airport",{}).get("alt_m", 0.0))*3.28084
    target  = orig_ft + dash_m*3.28084
    steady  = _steady(df, "dash")
    if steady.empty: return False, "No 'dash' phase rows", None, CRUISE_ALT_TOL
    alts = steady["altitude_msl_ft"].to_numpy()
    max_err = float(np.abs(alts-target).max()); rms_err = float(np.sqrt(((alts-target)**2).mean()))
    return max_err<=CRUISE_ALT_TOL, (f"mean {alts.mean():.0f} ft | target {target:.0f} ft | "
        f"max err {max_err:.0f} | RMS {rms_err:.0f} ft (limit ±{CRUISE_ALT_TOL:.0f})"), max_err, CRUISE_ALT_TOL

def _check_soc(df):
    ev = _gear_events(df)
    soc = ev["soc_pct"].iloc[0] if not ev.empty else df["soc_pct"].iloc[-1]
    src = f"t={ev['timestamp_s'].iloc[0]:.1f}s" if not ev.empty else "last row"
    return soc>=SOC_MIN, f"SoC {soc:.2f}% at {src} | minimum {SOC_MIN:.0f}%", soc, SOC_MIN

def _check_landing(df, tc):
    nav   = tc.get("navigation",{}).get("target",{})
    x_tgt, y_tgt = float(nav.get("x_m",0)), float(nav.get("y_m",0))
    ev    = _gear_events(df)
    td    = ev.iloc[0] if not ev.empty else df.iloc[-1]
    src   = f"t={td['timestamp_s']:.1f}s" if not ev.empty else "last row"
    err   = float(np.hypot(td["x_m"]-x_tgt, td["y_m"]-y_tgt))
    return err<=LAND_ACCURACY_M, (f"touchdown ({td['x_m']:.0f}, {td['y_m']:.0f}) m at {src} | "
        f"target ({x_tgt:.0f}, {y_tgt:.0f}) m | offset {err:.1f} m (limit {LAND_ACCURACY_M:.0f})"), err, LAND_ACCURACY_M

def _check_rpm(df):
    steady = _steady(df, "dash")
    if steady.empty: return False, "No 'dash' phase rows", None, RPM_ASYM_LIMIT
    rpm    = steady[RPM_COLS].to_numpy(dtype=float)
    active = rpm[rpm.max(axis=1)>10]
    if len(active)==0: return False, "All RPM near zero in dash", None, RPM_ASYM_LIMIT
    means  = active.mean(axis=1, keepdims=True)
    imb    = np.abs(active-means)/np.where(means>0,means,1)
    max_imb= imb.max(); worst = int(imb.max(axis=0).argmax())+1
    return max_imb<=RPM_ASYM_LIMIT, (f"max imbalance {max_imb*100:.2f}% (rotor {worst}) | "
        f"mean {imb.mean()*100:.2f}% | limit {RPM_ASYM_LIMIT*100:.0f}%"), max_imb, RPM_ASYM_LIMIT

def _check_gz(df):
    ev = _gear_events(df)
    if ev.empty: return False, "No gear_contact transition found", None, GZ_NORMAL_LIMIT
    idx0  = ev.index[0]
    peak  = float(df.loc[idx0:idx0+3,"gz"].abs().max())
    if peak <= GZ_NORMAL_LIMIT:    status = f"≤ normal limit {GZ_NORMAL_LIMIT}g ✓"
    elif peak <= GZ_EMERGENCY_LIM: status = f"exceeds normal {GZ_NORMAL_LIMIT}g but within emergency {GZ_EMERGENCY_LIM}g ⚠"
    else:                          status = f"EXCEEDS emergency limit {GZ_EMERGENCY_LIM}g ✗"
    return peak<=GZ_NORMAL_LIMIT, f"peak gz {peak:.3f}g at t={df.loc[idx0,'timestamp_s']:.2f}s — {status}", peak, GZ_NORMAL_LIMIT

def run_analysis(csv_path: Path, card_path: Path, out_dir: Path) -> int:
    try:
        import pandas as pd
    except ImportError:
        warn("pandas not installed — skipping analysis. pip install pandas numpy")
        return 0

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["phase"] = df["phase"].str.strip().str.lower().str.replace(r"^autoland:", "", regex=True)
    tc = json.loads(card_path.read_text()) if card_path.exists() else {}

    checks = [
        ("Phase sequence",   _check_phase_sequence(df)),
        ("Cruise speed",     _check_cruise_speed(df, tc)),
        ("Cruise altitude",  _check_cruise_alt(df, tc)),
        ("Arrival SoC",      _check_soc(df)),
        ("Landing accuracy", _check_landing(df, tc)),
        ("Rotor RPM symmetry", _check_rpm(df)),
        ("Touchdown gz",     _check_gz(df)),
    ]

    n_pass = sum(1 for _,(ok,*_) in checks if ok)
    n_fail = len(checks) - n_pass
    verdict = "PASS ✅" if n_fail == 0 else "FAIL ❌"

    dep  = tc.get("airport", {}).get("icao", "?")
    arr  = tc.get("destination", {}).get("icao", "?")
    dist = tc.get("_generated", {}).get("distance_km", "")
    mission = f"{dep} → {arr}" + (f"  ({dist:.1f} km)" if dist else "")

    plain(f"\n{'─'*60}\n  {mission}\n  {verdict}  ({n_pass}/{len(checks)} checks passed)\n{'─'*60}")
    for name, (ok, detail, *_rest) in checks:
        plain(f"  {'✅' if ok else '❌'} {name}\n       {detail}")
    plain("")

    if n_fail:
        fail(f"{n_fail} check(s) failed")
    return 0 if n_fail == 0 else 1


# ── SQLite export ──────────────────────────────────────────────────────────────

def _flatten_json(obj: dict, prefix: str = "", sep: str = "__") -> dict:
    out: dict = {}
    for k, v in obj.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_json(v, full_key, sep))
        elif isinstance(v, list):
            out[full_key] = json.dumps(v)
        else:
            out[full_key] = v
    return out


def export_sqlite(csv_path: Path, card_path: Path, out_dir: Path,
                   rotor_rows: list[dict] | None = None) -> Path:
    """Write a SQLite database alongside the CSV: test_parameters, rotor_config,
    telemetry.

    rotor_config is built from `rotor_rows` — the fleet actually used to plan
    this run (session override or CSV, whichever is active) — rather than
    re-reading rotor_config.csv from disk, so it always matches the card for
    the run being exported, even when a session override is in effect.
    """
    import sqlite3
    import pandas as pd

    db_name = csv_path.stem + ".db"
    db_path = out_dir / db_name
    db_path.unlink(missing_ok=True)

    info(f"Exporting SQLite: {db_path.name}")

    def _safe(s: str) -> str:
        return re.sub(r"[^\w]", "_", s)

    with sqlite3.connect(db_path) as con:

        if card_path.exists():
            try:
                card  = json.loads(card_path.read_text())
                flat  = _flatten_json(card)
                params_df = pd.DataFrame(
                    [{_safe(k): (json.dumps(v) if isinstance(v, (list, dict)) else v)
                      for k, v in flat.items()}]
                )
                params_df.to_sql("test_parameters", con, index=False, if_exists="replace")
                success(f"  test_parameters: {len(params_df.columns)} columns")
            except Exception as e:
                warn(f"  test_parameters skipped: {e}")
        else:
            warn("  test_card.json not found — test_parameters table will be empty")
            pd.DataFrame([{"note": "test_card.json not found"}]).to_sql(
                "test_parameters", con, index=False, if_exists="replace")

        try:
            rows = rotor_rows if rotor_rows is not None else _read_rotor_rows_from_csv()
            rotor_df = pd.DataFrame(rows)
            rotor_df.columns = [_safe(c) for c in rotor_df.columns]
            rotor_df.to_sql("rotor_config", con, index=False, if_exists="replace")
            source_note = "session override" if rotor_rows is not None else "rotor_config.csv"
            success(f"  rotor_config: {len(rotor_df)} rotors × {len(rotor_df.columns)} columns ({source_note})")
        except Exception as e:
            warn(f"  rotor_config skipped: {e}")

        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df.columns = [_safe(c.strip()) for c in df.columns]
            df.to_sql("telemetry", con, index=False, if_exists="replace")
            success(f"  telemetry: {len(df)} rows × {len(df.columns)} columns")
        except Exception as e:
            warn(f"  telemetry table failed: {e}")
            raise

    success(f"SQLite DB: {db_path}")
    return db_path


# ══════════════════════════════════════════════════════════════════════════════
#  RunSpec / run_pipeline — the single entry point used by:
#    - windblade.py --auto/--manual (headless, no GUI, no browser)
#    - the GUI's Launch button (_run_sim, via the HTTP server)
#  Never calls sys.exit()/os._exit() — always returns an int exit code, so the
#  server thread that calls it doesn't die with the process.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunSpec:
    mode:                str            = "auto"     # "auto" | "manual"
    speed:               Optional[float] = None
    gui:                 bool           = False
    terrain:             bool           = False
    no_build:            bool           = False
    no_plan:             bool           = False
    db:                  bool           = False
    out_dir:             Path           = field(default_factory=lambda: ROOT)
    csv:                 Optional[Path] = None
    dep_metar:           Optional[str]  = None
    arr_metar:           Optional[str]  = None
    speed_kmh:           Optional[float] = None
    altitude_ft:         Optional[float] = None
    hover_alt_m:         Optional[float] = None
    back_trans_speed_ms: Optional[float] = None
    nacelle_tilt_deg:    Optional[float] = None
    turb_intensity_ms:   Optional[float] = None   # explicit override (CLI --turb-intensity)
    auto_turb:           bool           = False   # GUI toggle: derive from dep METAR wind
    interactive:         bool           = True    # False ⇒ get_airport never calls input()
    rotor_rows:          Optional[list] = None    # None ⇒ read rotor_config.csv


def run_pipeline(spec: RunSpec) -> int:
    """plan → build → simulate → analyse. Returns the exit code. Never exits
    the process — safe to call from a daemon thread."""
    out_dir = Path(spec.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = DashConfig.load()
    except Exception as e:
        fail(str(e)); return 4

    if spec.speed_kmh           is not None: cfg.speed_kmh               = spec.speed_kmh
    if spec.altitude_ft         is not None: cfg.altitude_ft             = spec.altitude_ft
    if spec.hover_alt_m         is not None: cfg.hover_alt_m             = spec.hover_alt_m
    if spec.back_trans_speed_ms is not None: cfg.back_trans_speed_ms     = spec.back_trans_speed_ms
    if spec.nacelle_tilt_deg    is not None: cfg.nacelle_tilt_deg        = spec.nacelle_tilt_deg
    if spec.turb_intensity_ms   is not None:
        cfg.turbulence_intensity_ms = spec.turb_intensity_ms
    elif spec.auto_turb and spec.dep_metar:
        wind_ms = _metar_wind_speed_ms(spec.dep_metar)
        turb = round(0.1 * wind_ms, 3)
        if turb > 0:
            cfg.turbulence_intensity_ms = turb
            info(f"Dryden turbulence: dep wind {wind_ms:.1f} m/s -> intensity {turb} m/s")

    # Resolve rotor rows once, up front, so planning and the post-run SQLite
    # export agree on exactly what fleet was used for this run.
    rotor_rows = spec.rotor_rows
    rotor_source = "session override" if rotor_rows is not None else "rotor_config.csv"
    if rotor_rows is None:
        rotor_rows = _read_rotor_rows_from_csv()
    if spec.rotor_rows is not None:
        info(f"Rotor fleet: {len(rotor_rows)} rotor(s) from session override "
             f"(rotor_config.csv on disk is unchanged)")

    card_path = PLANNING / "test_card.json"
    csv_path: Optional[Path] = Path(spec.csv).resolve() if spec.csv else None
    exit_code = 0
    dep_lbl = arr_lbl = "?"

    # ── 1: Plan ───────────────────────────────────────────────────────
    header("Flight planning")
    if not spec.no_plan and not spec.csv:
        try:
            card_path = plan_flight(cfg, spec.terrain,
                                     dep_metar=spec.dep_metar,
                                     arr_metar=spec.arr_metar,
                                     rotor_rows=rotor_rows,
                                     interactive=spec.interactive)
        except Exception as e:
            if card_path.exists():
                warn(f"Planning failed ({e}) — using existing test_card.json")
            else:
                fail(f"Planning failed: {e}"); return 4
    elif spec.csv:
        info("csv mode: skipping planning")
    else:
        if not card_path.exists():
            fail("test_card.json not found — run without no_plan first"); return 4
        info(f"no_plan: using {card_path}")
        # Still respect a session rotor override on a no-plan run: patch the
        # existing card's rotor_fleet block in place so the sim sees it, same
        # as the immediate patch POST /rotors already applies — this covers
        # the case where the override was set *after* the card was written by
        # some earlier run and the user now launches with "skip planning".
        if spec.rotor_rows is not None:
            try:
                d = json.loads(card_path.read_text())
                d["rotor_fleet"] = _rotor_fleet_overrides(rotor_rows)
                card_path.write_text(json.dumps(d, indent=2))
                info("Patched existing test_card.json with session rotor override")
            except Exception as e:
                warn(f"Could not patch test_card.json with rotor override: {e}")

    if card_path.exists():
        try:
            d = json.loads(card_path.read_text())
            dep_lbl = d.get("airport",     {}).get("icao", "?")
            arr_lbl = d.get("destination", {}).get("icao", "?")
        except Exception:
            pass
    success(f"{dep_lbl} → {arr_lbl} | {cfg.speed_kmh:.0f} km/h | {cfg.altitude_ft:.0f} ft "
            f"| rotors: {rotor_source}")

    # ── 2: Build ──────────────────────────────────────────────────────
    if not spec.csv:
        header("Building")
        if not spec.no_build:
            try:
                build_autopilot()
            except Exception as e:
                fail(str(e)); return 2
        else:
            def _ver_label(version_file: Path, glob_name: str) -> str:
                vf = version_file
                if vf.exists():
                    return f"{glob_name}_{vf.read_text().strip()}.so"
                existing = sorted(vf.parent.glob(f"{glob_name}_*.so"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
                return existing[0].name if existing else f"{glob_name}.so (no version file)"
            info(f"no_build: {_ver_label(CONTROLS/'autopilot.version', 'autopilot')}")

    # ── 3: Simulate ───────────────────────────────────────────────────
    manual = (spec.mode == "manual")
    gui    = manual or spec.gui
    if not spec.csv:
        header(f"{'Auto' if spec.mode == 'auto' else 'Manual'} flight: {dep_lbl} → {arr_lbl}")
        if manual:
            info("Cockpit launching. Fly the aircraft. Stop/Ctrl+C ends and generates a report.")
        try:
            csv_path = run_simulation(gui, manual, spec.speed, out_dir)
        except KeyboardInterrupt:
            # CLI Ctrl+C path (windblade.py --auto/--manual run from a real terminal).
            warn("Ctrl+C received — stopping simulation")
            candidates = sorted(out_dir.glob("dash_results_*.csv"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                candidates = sorted(ROOT.glob("dash_results_*.csv"),
                                    key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                csv_path = candidates[0]
                success(f"Using CSV: {csv_path}")
            else:
                fail("No CSV found after Ctrl+C — simulation may not have started")
                return 3
        except Exception as e:
            fail(str(e)); return 3
    else:
        if not csv_path.exists():
            fail(f"csv: file not found: {csv_path}"); return 3
        info(f"csv mode: {csv_path}")

    # ── 4: Analyse ────────────────────────────────────────────────────
    header("Analysis")
    if csv_path:
        try:
            exit_code = run_analysis(csv_path, card_path, out_dir)
        except Exception as e:
            warn(f"Analysis error: {e}")

        if spec.db:
            try:
                export_sqlite(csv_path, card_path, out_dir, rotor_rows=rotor_rows)
            except Exception as e:
                warn(f"SQLite export failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    header("Summary")
    mode_label = "AUTO" if spec.mode == "auto" else "MANUAL"
    plain(f"  {'Mode:':<14} {mode_label}")
    plain(f"  {'Route:':<14} {dep_lbl} → {arr_lbl}")
    plain(f"  {'Speed:':<14} {cfg.speed_kmh:.0f} km/h")
    plain(f"  {'Altitude:':<14} {cfg.altitude_ft:.0f} ft MSL")
    plain(f"  {'Rotors:':<14} {rotor_source}")
    if csv_path:
        plain(f"  {'CSV:':<14} {csv_path}")
    if spec.db and csv_path:
        db_candidate = out_dir / (csv_path.stem + ".db")
        if db_candidate.exists():
            plain(f"  {'SQLite DB:':<14} {db_candidate}")
    if spec.speed:
        plain(f"  {'Sim speed:':<14} {spec.speed}×")
    plain(f"  {'Exit code:':<14} {exit_code}\n")
    return exit_code


# ── CLI parser for windblade.py's own headless invocation. --auto/--manual
#    are optional here — omitting both just means "open the GUI instead".
#    See main().  ────────────────────────────────────────────────────────────

def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--auto",   action="store_true",
                      help="Autonomous flight: plan → build → simulate → analyse")
    mode.add_argument("--manual", action="store_true",
                      help="Manual HOTAS flight with cockpit. Ctrl+C stops the sim.")
    p.add_argument("--speed",    type=float, default=None, metavar="X")
    p.add_argument("--gui",      action="store_true")
    p.add_argument("--terrain",  action="store_true")
    p.add_argument("--no-plan",  action="store_true")
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--out",      default=str(ROOT), metavar="DIR")
    p.add_argument("--csv",      default=None, metavar="PATH")
    p.add_argument("--db",       action="store_true")
    p.add_argument("--dep-metar", default=None, metavar="METAR")
    p.add_argument("--arr-metar", default=None, metavar="METAR")
    p.add_argument("--speed-kmh", type=float, default=None, metavar="KMH")
    p.add_argument("--alt-ft",    type=float, default=None, metavar="FT")
    p.add_argument("--hover-m",   type=float, default=None, metavar="M")
    p.add_argument("--turb-intensity", type=float, default=None, metavar="MS")
    p.add_argument("--bt-speed-ms",    type=float, default=None, metavar="MS")
    p.add_argument("--nacelle-tilt-deg", type=float, default=None, metavar="DEG")


def spec_from_args(args: argparse.Namespace) -> RunSpec:
    if args.speed is not None and args.manual:
        fail("--speed is not valid with --manual (manual mode always runs at realtime)")
        sys.exit(1)
    if args.speed is not None and args.gui:
        fail("--speed and --gui are mutually exclusive (GUI throttles the sim)")
        sys.exit(1)
    return RunSpec(
        mode="manual" if args.manual else "auto",
        speed=args.speed, gui=args.gui, terrain=args.terrain,
        no_build=args.no_build, no_plan=args.no_plan, db=args.db,
        out_dir=Path(args.out), csv=Path(args.csv) if args.csv else None,
        dep_metar=args.dep_metar, arr_metar=args.arr_metar,
        speed_kmh=args.speed_kmh, altitude_ft=args.alt_ft, hover_alt_m=args.hover_m,
        back_trans_speed_ms=args.bt_speed_ms, nacelle_tilt_deg=args.nacelle_tilt_deg,
        turb_intensity_ms=args.turb_intensity,
        interactive=True,       # CLI has a real TTY — input() fallback stays available
        rotor_rows=None,        # CLI always reads rotor_config.csv; no session concept
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Results plotting — kept inline so windblade.py stays a single-file
#  entry point
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_PLOT_COLUMNS = ["altitude_msl_ft", "speed_kmh"]

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
    m: dict = {}
    if "timestamp_s" in df.columns and len(df):
        m["flight_time_s"] = round(float(df["timestamp_s"].max()), 1)
    if "x_m" in df.columns and "y_m" in df.columns and len(df):
        x_m = abs(float(df["x_m"].iloc[-1]))
        y_m = abs(float(df["y_m"].iloc[-1]))
        m["distance_km"] = round(((x_m ** 2 + y_m ** 2) ** 0.5) / 1000.0, 2)
    if "power_kw" in df.columns and "timestamp_s" in df.columns and len(df) > 1:
        t = df["timestamp_s"].to_numpy(dtype=float)
        p = df["power_kw"].to_numpy(dtype=float)
        energy_kj = float(np.sum(np.diff(t) * (p[:-1] + p[1:]) / 2.0))
        m["energy_mj"] = round(energy_kj / 1000.0, 2)
    if "soc_pct" in df.columns and len(df):
        m["arrival_soc_pct"] = round(float(df["soc_pct"].iloc[-1]), 1)
    return m

def _process_results_csv(csv_path: Path, out_dir: Path,
                          columns: list[str]) -> tuple[dict, dict]:
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
.data-table{width:100%;border-collapse:collapse;font-size:18px;margin-bottom:20px}
.data-table th{text-align:left;font-size:15px;letter-spacing:.06em;color:var(--dim);text-transform:uppercase;padding:0 8px 8px;border-bottom:1px solid var(--stroke-hi)}
.data-table td{padding:6px 8px;border-bottom:1px solid var(--stroke);color:var(--nw)}
.data-table tr:last-child td{border-bottom:none}
.data-table tr:hover td{background:var(--hi)}
.data-table input,.data-table select{background:var(--panel);border:1px solid var(--stroke-hi);border-radius:2px;padding:4px 6px;font-size:16px;font-family:var(--mono);color:var(--nw);width:100%;height:34px}
.data-table input[type=number]{-moz-appearance:textfield;appearance:textfield}
.data-table input[type=number]::-webkit-outer-spin-button,
.data-table input[type=number]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.data-table input.err{border-color:var(--rd);color:var(--rd)}
.data-table input[type=checkbox]{width:auto;height:auto;accent-color:var(--ga)}
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
.btn.danger{border-color:var(--rd);color:var(--rd)}.btn.danger:hover{background:#2a0000}
.btn.small{padding:6px 12px;font-size:16px}
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
      <div class="sec">Route</div>
      <div class="row2">
        <div class="field"><label>Departure ICAO</label>
          <input id="dep-icao" value="KAXX" maxlength="4" placeholder="KAXX"
                 oninput="icaoChanged('dep')" onblur="fetchWx('dep')"></div>
        <div class="field"><label>Arrival ICAO</label>
          <input id="arr-icao" value="KSAF" maxlength="4" placeholder="KSAF"
                 oninput="icaoChanged('arr')" onblur="fetchWx('arr')"></div>
      </div>
      <div class="ts" id="route-info" style="margin-top:-8px;margin-bottom:18px"></div>

      <div class="sec">Weather</div>
      <div class="row2">
        <div class="field"><label>METAR_DEP <span id="dep-age" class="c-dim"></span></label>
          <textarea id="dep-wx" rows="3" readonly></textarea></div>
        <div class="field"><label>METAR_ARR <span id="arr-age" class="c-dim"></span></label>
          <textarea id="arr-wx" rows="3" readonly></textarea></div>
      </div>

      <div class="callout">Weather and terrain resolve automatically at launch. METARs sourced from aviationweather.gov. Terrain for the route from a predefined profile in <code>terrain.jl</code>, else a SRTM download, otherwise a flat-world model. If a station is reporting nothing, a nominal METAR is constructed from its elevation. Both airports must exist in <code>planning/airports.csv</code>.</div>

      <div class="toggle-row" style="cursor:pointer" onclick="toggleManual()">
        <div><div class="tl">Paste METARs manually instead</div>
          <div class="ts">Required with no internet; also the fallback if the fetch endpoint is down</div></div>
        <div class="switch" id="sw-manual"></div>
      </div>
      <div id="manual-wrap" style="display:none;margin-top:18px">
        <div class="row2">
          <div class="field"><label>METAR_DEP — manual</label>
            <textarea id="dep-metar" rows="3" oninput="sync()"></textarea></div>
          <div class="field"><label>METAR_ARR — manual</label>
            <textarea id="arr-metar" rows="3" oninput="sync()"></textarea></div>
        </div>
        <div class="callout warn">Manual text overrides the fetched observation. The first token must match the ICAO above.</div>
      </div>
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
        <div><div class="tl">Download SRTM terrain</div><div class="ts">Force terrain profile refresh — enabled automatically when the route has no predefined profile</div></div>
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
        <div><div class="tl">Export SQLite DB</div><div class="ts">Write dash_results_&lt;ts&gt;.db alongside the CSV (test_parameters + rotor_config + telemetry tables)</div></div>
        <div class="switch" id="sw-db" onclick="tog(this,'db')"></div>
      </div>
    </div>

    <!-- Rotor config -->
    <div class="panel" id="panel-rotors">
      <div class="sec">Source — <span id="rotor-csv-path" style="color:var(--faint)"></span></div>
      <div id="rotor-fleet-error" style="display:none" class="callout" style="border-color:var(--rd);color:var(--rd)"></div>
      <div id="rotor-warnings"></div>
      <table class="data-table" id="rotor-table">
        <thead><tr><th>#</th><th>Position</th><th>R (m)</th><th>Blades</th><th>Chord (m)</th><th>Twist root</th><th>Twist tip</th><th>Pitch offset</th><th>P max (kW)</th><th>RPM hover</th><th>Propulsion</th><th>Lift</th><th>Thrust</th><th>Notes</th></tr></thead>
        <tbody id="rotor-tbody"><tr><td colspan="14" class="c-dim" style="padding:12px 8px">Loading...</td></tr></tbody>
      </table>
      <div class="actions">
        <button class="btn primary" onclick="updateRotors()">&#8635; Update (this run only)</button>
        <button class="btn danger small" onclick="resetRotors()">Reset to CSV</button>
      </div>
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
        <button class="btn danger" id="btn-stop" onclick="doStop()" disabled>&#9632; Stop</button>
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
var rotorBaseline={};
var rotorPowerplants=["electric","turbine_electric","turboshaft"];
var rotorLocked=false;
var rotorSource='csv';

var POSITIONS=['fwd-port','fwd-stbd','mid-port','mid-stbd','aft-port','aft-stbd'];

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

/* ── Weather resolution (client half) ───────────────────────────────────────
   The ICAO inputs are the only weather input; GET /metar proxies
   aviationweather.gov server-side and the resolved text is read-only. The
   manual textareas below are the offline / endpoint-down path — the server
   resolves the full decision table again at launch either way. */
var wxTimer={dep:null,arr:null};
var manualOpen=false;
var airportData={};

function loadAirports(){
  fetch('/airports').then(r=>r.json()).then(function(d){airportData=d;updateRouteInfo();}).catch(function(){});
}

function haversineKm(lat1,lon1,lat2,lon2){
  var R=6371.0,toRad=function(d){return d*Math.PI/180;};
  var dLat=toRad(lat2-lat1),dLon=toRad(lon2-lon1);
  var a=Math.sin(dLat/2)*Math.sin(dLat/2)
      +Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dLon/2)*Math.sin(dLon/2);
  return 2*R*Math.asin(Math.sqrt(Math.min(1,a)));
}

function updateRouteInfo(){
  var el=document.getElementById('route-info');
  if(!el)return;
  var dep=v('dep-icao').toUpperCase(),arr=v('arr-icao').toUpperCase();
  var da=airportData[dep],aa=airportData[arr];
  var depLbl=dep?(da?(da.name||dep):dep+' — not in airports.csv'):'?';
  var arrLbl=arr?(aa?(aa.name||arr):arr+' — not in airports.csv'):'?';
  var dist=(da&&aa)?' — '+haversineKm(da.lat,da.lon,aa.lat,aa.lon).toFixed(1)+' km':'';
  el.textContent=depLbl+'  →  '+arrLbl+dist;
}

function icaoChanged(which){
  var el=document.getElementById(which+'-icao');
  el.value=el.value.toUpperCase();
  updateRouteInfo();
  clearTimeout(wxTimer[which]);
  wxTimer[which]=setTimeout(function(){fetchWx(which);},600);
}

function fetchWx(which){
  clearTimeout(wxTimer[which]);
  var icao=v(which+'-icao').toUpperCase();
  var box=document.getElementById(which+'-wx'),age=document.getElementById(which+'-age');
  if(!/^[A-Z0-9]{4}$/.test(icao)){box.value='';age.textContent='— invalid ICAO';age.className='c-warn';return;}
  age.textContent='— fetching...';age.className='c-dim';
  fetch('/metar?icao='+encodeURIComponent(icao))
    .then(r=>r.json())
    .then(d=>{
      box.value=d.raw||'';
      if(d.raw){
        var mins=(d.age_s!=null)?', '+Math.round(d.age_s/60)+' min old':'';
        age.textContent='— '+d.status+mins;
        age.className=(d.status==='bundled')?'c-warn':'c-ok';
      }else{
        age.textContent='— '+(d.message||d.status);
        age.className='c-warn';
      }
    })
    .catch(function(){box.value='';age.textContent='— /metar unreachable';age.className='c-warn';});
}

function toggleManual(open){
  manualOpen=(open===undefined)?!manualOpen:!!open;
  document.getElementById('manual-wrap').style.display=manualOpen?'block':'none';
  document.getElementById('sw-manual').classList.toggle('on',manualOpen);
}

/* Launch refused for want of a hand-entered METAR: open the manual block and
   prefill whichever ends have a bundled reference. Nothing is submitted — the
   user confirms the text and launches again. */
function openManual(need,suggest){
  toggleManual(true);
  (need||[]).forEach(function(w){
    var el=document.getElementById(w+'-metar');
    if(el&&!el.value.trim()&&suggest&&suggest[w])el.value=suggest[w];
  });
}

function buildCmd(){
  var mode=v('mode'),sf=v('sfactor');
  var parts=['python3 windblade.py','--'+mode];
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
      rotorBaseline=data.baseline||{};
      rotorPowerplants=data.powerplants||rotorPowerplants;
      rotorLocked=!!data.locked;
      rotorSource=data.source||'csv';
      var n=rotorData.length;
      var err=data.error||null;
      var errEl=document.getElementById('rotor-fleet-error');
      if(err){
        errEl.textContent=err;
        errEl.style.display='block';
      } else {
        errEl.style.display='none';
      }
      renderRotorTable();
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
        document.getElementById('rotor-badge').className='badge '+(rotorSource==='session'?'warn':'ok');
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

function fleetModeSummary(){
  if(!rotorData.length)return '';
  var c={TILT:0,LIFT:0,THRUST:0,OFF:0};
  rotorData.forEach(function(r){ c[r.mode||'TILT']=(c[r.mode||'TILT']||0)+1; });
  var parts=[];
  ['TILT','LIFT','THRUST','OFF'].forEach(function(k){ if(c[k])parts.push(c[k]+' '+k); });
  return parts.length?'  ('+parts.join(', ')+')':'';
}

function computeMode(lift,thrust){
  if(lift&&thrust)return 'TILT';
  if(lift)return 'LIFT';
  if(thrust)return 'THRUST';
  return 'OFF';
}

function diffCls(field,val){
  var base=rotorBaseline[field];
  if(base===undefined)return '';
  return Math.abs(parseFloat(val)-base)>0.001 ? 'c-warn' : '';
}

// Rebuild the whole editable table from rotorData. Each input writes straight
// back into rotorData[i][field] on change, so Update always posts current
// on-screen state, not a stale snapshot from load time.
function renderRotorTable(){
  var tbody=document.getElementById('rotor-tbody');
  var rotors=rotorData;
  if(!rotors.length){
    tbody.innerHTML='<tr><td colspan="14" class="c-warn" style="padding:12px 8px">no rotors loaded</td></tr>';
    return;
  }
  var opts=rotorPowerplants.map(function(p){return '<option value="'+p+'">'+p+'</option>';}).join('');
  tbody.innerHTML=rotors.map(function(r,i){
    var pos=POSITIONS[i]||'pos-'+i;
    function numInput(field,step){
      return '<input type="number" step="'+(step||'any')+'" value="'+r[field]+'" class="'+diffCls(field,r[field])+'" '
        +'onchange="onRotorField('+i+',\''+field+'\',this.value,false)" '+(rotorLocked?'disabled':'')+'>';
    }
    var ppOpts=rotorPowerplants.map(function(p){
      return '<option value="'+p+'"'+(p===r.powerplant?' selected':'')+'>'+p+'</option>';
    }).join('');
    return '<tr data-i="'+i+'">'
      +'<td class="c-dim">'+numInput('rotor_id')+'</td>'
      +'<td class="c-dim">'+pos+'</td>'
      +'<td>'+numInput('R_m','0.001')+'</td>'
      +'<td>'+numInput('n_blades','1')+'</td>'
      +'<td>'+numInput('chord_m','0.001')+'</td>'
      +'<td>'+numInput('twist_root_deg','0.1')+'</td>'
      +'<td>'+numInput('twist_tip_deg','0.1')+'</td>'
      +'<td>'+numInput('pitch_offset_deg','0.1')+'</td>'
      +'<td>'+numInput('P_max_kW','1')+'</td>'
      +'<td>'+numInput('rpm_hover','1')+'</td>'
      +'<td><select onchange="onRotorField('+i+',\'powerplant\',this.value,false)" '+(rotorLocked?'disabled':'')+'>'+ppOpts+'</select></td>'
      +'<td><input type="checkbox" '+(r.lift?'checked':'')+' onchange="onRotorField('+i+',\'lift\',this.checked,true)" '+(rotorLocked?'disabled':'')+'></td>'
      +'<td><input type="checkbox" '+(r.thrust?'checked':'')+' onchange="onRotorField('+i+',\'thrust\',this.checked,true)" '+(rotorLocked?'disabled':'')+'></td>'
      +'<td><input type="text" value="'+(r.notes||'').replace(/"/g,'&quot;')+'" '
        +'onchange="onRotorField('+i+',\'notes\',this.value,false)" '+(rotorLocked?'disabled':'')+'></td>'
      +'</tr>';
  }).join('');
}

function onRotorField(i,field,val,isBool){
  if(isBool){
    rotorData[i][field]=!!val;
  } else if(field==='notes'||field==='powerplant'){
    rotorData[i][field]=val;
  } else {
    rotorData[i][field]=parseFloat(val);
  }
  if(field==='lift'||field==='thrust'){
    rotorData[i].mode=computeMode(rotorData[i].lift,rotorData[i].thrust);
  }
  renderRotorTable();   // re-render so diff-highlighting and mode stay current
}

function updateRotors(){
  var errEl=document.getElementById('rotor-fleet-error');
  var warnEl=document.getElementById('rotor-warnings');
  fetch('/rotors',{method:'POST',headers:{'Content-Type':'application/json'},
                  body:JSON.stringify({rotors:rotorData})})
    .then(function(r){ return r.json().then(function(d){ return {ok:r.ok,d:d}; }); })
    .then(function(res){
      if(!res.ok){
        var errs=(res.d.errors||[]).map(function(e){
          return 'row '+(e.row!=null?e.row+1:'?')+' · '+e.field+': '+e.message;
        });
        errEl.innerHTML=errs.length?errs.join('<br>'):(res.d.message||'update rejected');
        errEl.style.display='block';
        return;
      }
      errEl.style.display='none';
      rotorData=res.d.rotors||rotorData;
      rotorSource='session';
      warnEl.innerHTML=(res.d.warnings||[]).map(function(w){
        return '<div class="callout warn">'+w+'</div>';
      }).join('');
      renderRotorTable();
      renderRotorDisks(rotorData);
      var badge=document.getElementById('rotor-badge');
      badge.textContent=rotorData.length+' rotor'+(rotorData.length!==1?'s':'')+' loaded'
        +(res.d.card_patched?' · card patched':' · no test_card.json yet');
      badge.className='badge warn';
      document.getElementById('statusbar').innerHTML='Rotor fleet updated<br>rotor_config.csv unchanged';
      renderChecklist();
    })
    .catch(function(e){
      errEl.textContent='update failed: '+e;
      errEl.style.display='block';
    });
}

function resetRotors(){
  fetch('/rotors',{method:'POST',headers:{'Content-Type':'application/json'},
                  body:JSON.stringify({reset:true})})
    .then(function(r){return r.json();})
    .then(function(d){
      document.getElementById('rotor-fleet-error').style.display='none';
      document.getElementById('rotor-warnings').innerHTML='';
      loadRotors();
      document.getElementById('statusbar').textContent='rotor fleet reverted to rotor_config.csv';
    });
}

function buildDiskSVG(r, tileSize, R_max){
  var LABEL_H = 22;
  var PAD     = 6;
  var S = tileSize || 160;
  var drawH = S - LABEL_H;
  var cx = S / 2;
  var cy = PAD + (drawH - PAD) / 2;
  var maxDiskPx = Math.min(S/2 - PAD, (drawH - PAD) / 2) * 0.96;
  var R_m = r.R_m || 1.45;
  var R = maxDiskPx * (R_m / R_max);
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
    var p0 = pt2(hubR, cRpx,  1);
    var p1 = pt2(R,    cTpx,  1);
    var p2 = pt2(R,    cTpx, -1);
    var p3 = pt2(hubR, cRpx, -1);
    var hw = Math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / 2;
    var d = 'M'+p0[0].toFixed(1)+','+p0[1].toFixed(1)
          + ' L'+p1[0].toFixed(1)+','+p1[1].toFixed(1)
          + ' A'+hw.toFixed(1)+','+hw.toFixed(1)+' 0 0,0 '+p2[0].toFixed(1)+','+p2[1].toFixed(1)
          + ' L'+p3[0].toFixed(1)+','+p3[1].toFixed(1)+' Z';
    blades += '<path d="'+d+'" fill="none" stroke="'+diskColor+'" stroke-width="'+strokeW+'" stroke-linejoin="round"/>';
  }

  var disk = '<circle cx="'+cx+'" cy="'+cy+'" r="'+R.toFixed(1)+'" fill="none" stroke="#2e4432" stroke-width="0.8" stroke-dasharray="3,3"/>';
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
  var R_max = Math.max.apply(null, rotors.map(function(r){ return r.R_m || 1.45; }));
  var cols = n<=6 ? n : 4;
  var tileSize = n<=6 ? Math.min(170, Math.floor(900/n)) : 150;
  var html = '<div style="display:grid;grid-template-columns:repeat('+cols+','+tileSize+'px);gap:12px;margin-top:12px">';
  rotors.forEach(function(r){ html += buildDiskSVG(r, tileSize, R_max); });
  html += '</div>';
  el.innerHTML = html;
}

function wxText(which){
  return v(which+'-metar')||v(which+'-wx')||(v(which+'-icao')||'?')+' (resolved at launch)';
}

function renderChecklist(){
  var rows=[
    ['DEP',    wxText('dep').substring(0,60), 'var(--ga)'],
    ['ARR',    wxText('arr').substring(0,60), 'var(--ga)'],
    ['CRUISE', v('speed')+' km/h  /  '+v('alt')+' ft MSL  /  hover '+v('hover')+' m', 'var(--ga)'],
    ['MODE',   v('mode').toUpperCase()+'  x'+v('sfactor'), 'var(--ga)'],
    ['ROTORS', rotorData.length+' rotors'+fleetModeSummary()+' — '+
               (rotorSource==='session'?'session override':'rotor_config.csv'),
               rotorData.length?'var(--ga)':'var(--yl)'],
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
    dep_icao:  v('dep-icao').toUpperCase(),
    arr_icao:  v('arr-icao').toUpperCase(),
    dep_metar: v('dep-metar'),   /* manual override, empty unless pasted */
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
  document.getElementById('btn-stop').disabled=false;
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
        logLine('<span class="rd">'+(d.message||'error: unknown')+'</span>');
        if(d.need_manual&&d.need_manual.length){
          openManual(d.need_manual,d.suggest);
          logLine('<span class="yl">[ launch_sim ]  manual METAR entry opened on Route &amp; weather</span>');
        }
        document.getElementById('statusbar').textContent='launch refused — '+(d.action||'error');
        resetLaunch();
      }
    })
    .catch(e=>{logLine('<span class="rd">fetch error: '+e+'</span>');resetLaunch();});
}

function doStop(){
  document.getElementById('btn-stop').disabled=true;
  logLine('<span class="yl">[ launch_sim ]  stop requested...</span>');
  fetch('/stop',{method:'POST'}).catch(function(){});
}

function pollLog(offset){
  fetch('/log?offset='+offset)
    .then(r=>r.json())
    .then(d=>{
      (d.lines||[]).forEach(l=>{
        var cls=/\[.*PASS.*\]/.test(l)?'ga':/\[.*CAUT.*\]/.test(l)?'yl':/\[.*FAIL.*\]/.test(l)?'rd':'nw';
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

function resetLaunch(){
  running=false;
  document.getElementById('btn-launch').disabled=false;
  document.getElementById('btn-stop').disabled=true;
}

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
loadAirports();
fetchWx('dep');
fetchWx('arr');
sync();
</script>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
#  Sim pipeline wrapper — bridges the HTTP server to run_pipeline()
# ═════════════════════════════════════════════════════════════════════════════

_sim_log:     list[str] = []
_sim_done:    bool      = False
_sim_rc:      int       = -1
_sim_running: bool      = False
_sim_lock     = threading.Lock()

_last_out_dir: Path | None = None
_last_plots:   dict[str, str] = {}
_last_metrics: dict = {}


def _is_running() -> bool:
    with _sim_lock:
        return _sim_running


def _run_sim(cfg: dict) -> None:
    global _sim_done, _sim_rc, _sim_running

    with _sim_lock:
        _sim_running = True

    def install_sink():
        global _LOG_SINK
        with _LOG_LOCK:
            _LOG_SINK = _sim_log

    def clear_sink():
        global _LOG_SINK
        with _LOG_LOCK:
            _LOG_SINK = None

    install_sink()
    try:
        # Resolution happened in the /launch handler, before the sink existed —
        # replay its lines here so they land in the browser log.
        for line in cfg.get("_wx_log") or []:
            info(line)

        dep_metar = cfg.get("dep_metar", "").strip()
        arr_metar = cfg.get("arr_metar", "").strip()
        cru       = cfg.get("cruise", {})
        sim       = cfg.get("sim",    {})
        plot_cols = [c.strip() for c in (cfg.get("plot_columns") or []) if c and c.strip()]
        if not plot_cols:
            plot_cols = list(DEFAULT_PLOT_COLUMNS)
        out_dir   = (ROOT / f"results_{datetime.datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}").resolve()

        rotor_rows, rotor_source = _effective_rotor_rows()
        if rotor_source == "session":
            info(f"Rotor fleet: using session override ({len(rotor_rows)} rotors, "
                 f"rotor_config.csv unchanged)")

        speed_factor = sim.get("speed_factor")
        speed = None
        if sim.get("mode", "auto") == "auto" and speed_factor is not None and not sim.get("gui"):
            speed = float(speed_factor) if float(speed_factor) != 1 else None

        spec = RunSpec(
            mode=sim.get("mode", "auto"),
            speed=speed,
            gui=bool(sim.get("gui")),
            terrain=bool(sim.get("terrain")),
            no_build=bool(sim.get("no_build")),
            no_plan=bool(sim.get("no_plan")),
            db=bool(sim.get("db")),
            out_dir=out_dir,
            csv=None,
            dep_metar=dep_metar or None,
            arr_metar=arr_metar or None,
            speed_kmh=cru.get("speed_kmh"),
            altitude_ft=cru.get("altitude_ft"),
            hover_alt_m=cru.get("hover_alt_m"),
            back_trans_speed_ms=cru.get("back_trans_speed_ms"),
            nacelle_tilt_deg=cru.get("nacelle_tilt_deg"),
            auto_turb=bool(sim.get("turb")),
            interactive=False,           # no TTY in a daemon thread — never block on input()
            rotor_rows=rotor_rows,        # session override, or a CSV read we already did
        )

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            sim_rc = run_pipeline(spec)
        except Exception as e:
            fail(str(e))
            sim_rc = 3

        plots, metrics = {}, {}
        if sim_rc <= 1:
            csvs = sorted(out_dir.glob("dash_results_*.csv"))
            if csvs:
                plots, metrics = _process_results_csv(csvs[-1], out_dir, plot_cols)
                if plots:
                    info(f"Saved plot(s): {', '.join(plots.values())}")
            else:
                warn("no dash_results_*.csv found — skipping plots")

    finally:
        clear_sink()

    global _last_out_dir, _last_plots, _last_metrics
    with _sim_lock:
        _sim_done = True; _sim_rc = sim_rc
        _sim_running = False
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

    def _send_json(self, code, obj) -> None:
        self._send(code, "application/json", json.dumps(obj).encode())

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _HTML.encode())

        elif path == "/rotors":
            rows, source = _effective_rotor_rows()
            n = len(rows)
            error = None
            if n > 0 and not (2 <= n <= 8):
                error = f"Invalid rotor count: {n}. Fleet must have 2–8 rotors."
            self._send_json(200, {
                "rotors": rows, "path": str(ROTOR_CSV), "error": error,
                "baseline": _ROTOR_BASELINE, "powerplants": POWERPLANTS,
                "source": source, "locked": _is_running(),
            })

        elif path == "/airports":
            airports = load_airports()
            self._send_json(200, {
                icao: {"name": a.name, "lat": a.lat, "lon": a.lon, "elev_m": a.elev_m}
                for icao, a in airports.items()
            })

        elif path == "/metar":
            from urllib.parse import parse_qs
            qs   = parse_qs(urlparse(self.path).query)
            icao = (qs.get("icao", [""])[0] or "").strip().upper()
            f    = fetch_metar(icao, allow_network=has_internet())
            if not f.usable and icao in BUNDLED_METAR:
                f = MetarFetch(icao=icao, raw=BUNDLED_METAR[icao], status="bundled",
                               message="bundled reference observation")
            self._send_json(200, {
                "icao": f.icao, "raw": f.raw, "status": f.status,
                "age_s": f.age_s, "message": f.message,
            })

        elif path == "/card":
            card_path = PLANNING / "test_card.json"
            if card_path.exists():
                self._send(200, "application/json", card_path.read_bytes())
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
            self._send_json(200, {
                "lines": lines, "next_offset": offset + len(lines),
                "done": done, "exit_code": rc,
            })

        elif path == "/plots":
            with _sim_lock:
                out_dir = _last_out_dir
                plots   = dict(_last_plots)
                metrics = dict(_last_metrics)
            self._send_json(200, {
                "out_dir": str(out_dir) if out_dir else None,
                "plots": [{"col": c, "url": f"/plot/{f}"} for c, f in plots.items()],
                "metrics": metrics,
            })

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
        path = urlparse(self.path).path

        if path == "/launch":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                cfg = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json(400, {"status": "error", "message": str(e)})
                return

            # ICAO-driven client: resolve weather and terrain against the
            # decision table before anything starts.  A refusal comes back on
            # this response — the run never begins — except on a no-plan run,
            # where the existing test_card.json is what flies and the
            # resolution is advisory only.
            if cfg.get("dep_icao") or cfg.get("arr_icao"):
                res = resolve_route(cfg.get("dep_icao", ""), cfg.get("arr_icao", ""),
                                    cfg.get("dep_metar", "") or "",
                                    cfg.get("arr_metar", "") or "",
                                    force_net_probe=True)
                advisory = bool((cfg.get("sim") or {}).get("no_plan"))
                if not res.ok and not advisory:
                    self._send_json(200, {
                        "status": "error", "message": res.message,
                        "action": res.action, "need_manual": res.need_manual,
                        "suggest": res.suggest,
                    })
                    return
                cfg["_wx_log"] = res.log + ([] if res.ok else
                                            [f"no_plan: {res.message} — flying the "
                                             f"existing test card anyway"])
                if res.dep_metar: cfg["dep_metar"] = res.dep_metar
                if res.arr_metar: cfg["arr_metar"] = res.arr_metar
                if res.force_terrain is not None:
                    cfg.setdefault("sim", {})["terrain"] = res.force_terrain

            global _sim_log, _sim_done, _sim_rc
            with _sim_lock:
                _sim_log  = []
                _sim_done = False
                _sim_rc   = -1

            threading.Thread(target=_run_sim, args=(cfg,), daemon=True).start()
            self._send_json(200, {"status": "started"})

        elif path == "/stop":
            signalled = request_stop()
            self._send_json(200, {"status": "stopping" if signalled else "not_running"})

        elif path == "/rotors":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json(400, {"code": 10, "errors": [{"message": str(e)}]})
                return

            global _rotor_session_rows

            if payload.get("reset"):
                with _rotor_session_lock:
                    _rotor_session_rows = None
                rows = _read_rotor_rows_from_csv()
                self._send_json(200, {"rotors": rows, "source": "csv",
                                       "message": "reverted to rotor_config.csv"})
                return

            if _is_running():
                self._send_json(409, {"code": 9, "message": "a run is in progress"})
                return

            rows_in = payload.get("rotors", [])
            ok, normalized, errors, warnings = _validate_rotor_rows(rows_in)
            if not ok:
                self._send_json(400, {"code": 10, "errors": errors})
                return

            with _rotor_session_lock:
                _rotor_session_rows = normalized

            # Patch the *current* test_card.json immediately, if one exists,
            # so the Launch preview and any --no-plan run reflect the edit
            # right away. This never touches rotor_config.csv — the CSV on
            # disk is untouched by design; only this run (and any run before
            # a Reset / server restart) sees the override.
            card_path = PLANNING / "test_card.json"
            card_patched = False
            if card_path.exists():
                try:
                    card = json.loads(card_path.read_text())
                    card["rotor_fleet"] = _rotor_fleet_overrides(normalized)
                    card_path.write_text(json.dumps(card, indent=2))
                    card_patched = True
                except Exception as e:
                    warn(f"could not patch test_card.json with rotor override: {e}")

            self._send_json(200, {
                "rotors": normalized, "source": "session",
                "card_patched": card_patched, "warnings": warnings,
            })

        else:
            self._send(404, "text/plain", b"not found")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port",            type=int, default=5780)
    p.add_argument("--no-browser",      action="store_true")
    _add_pipeline_args(p)
    args = p.parse_args()

    # Headless pipeline run: `python3 windblade.py --auto ...` / `--manual ...`.
    # --auto/--manual are optional here — omitting both just opens the GUI.
    if args.auto or args.manual:
        spec = spec_from_args(args)
        return run_pipeline(spec)

    url = f"http://localhost:{args.port}"
    header("Mission Planner")
    info(f"Serving GUI at  {GA}{url}{NC}")
    info(f"Repo root:      {ROOT}")
    info(f"Rotor CSV:      {ROTOR_CSV}")
    info(f"Press  {YL}Ctrl+C{NC}  to stop\n")

    # Threaded: /metar and /launch both block on network I/O now, and a
    # single-threaded server would stall log polling while they run.
    server = ThreadingHTTPServer(("localhost", args.port), _Handler)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        warn("Shutting down")
    return 0

if __name__ == "__main__":
    sys.exit(main())

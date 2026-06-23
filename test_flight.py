#!/usr/bin/env python3
#
# <test_flight.py>:     Unified flight test runner>
# Author:               DANIEL DESAI>
# Updated:              2026-06-22
# Version:              0.1.3

"""
Usage:
    python3 test_flight.py --auto                    # full run (reads planning/ files)
    python3 test_flight.py --auto --gui              # autonomous with cockpit window
    python3 test_flight.py --manual                  # HOTAS manual mode
    python3 test_flight.py --no-plan                 # skip planning, reuse test_card.json
    python3 test_flight.py --no-build                # skip recompile
    python3 test_flight.py --terrain                 # force SRTM terrain download
    python3 test_flight.py --speed 3.0               # sim speed multiplier
    python3 test_flight.py --out results/r1          # custom output directory
    python3 test_flight.py --csv /path/to/file.csv   # analyse existing CSV

    # Pass weather and cruise directly (bypasses planning/ files entirely):
    python3 test_flight.py --auto \\
        --dep-metar "KAXX 151155Z 00000KT 10SM CLR M01/M10 A3018 RMK AO2 T10141096" \\
        --arr-metar "KSAF 151153Z 24005KT 10SM CLR 13/M09 A3005 RMK AO2 T01281094" \\
        --speed-kmh 300 --alt-ft 11500 --hover-m 30 --bt-speed-ms 50

planning/dash.py schema (used when --dep-metar/--arr-metar not supplied):
    SPEED_KMH    = 300.0       # cruise speed (km/h)
    ALTITUDE_FT  = 11500.0     # cruise altitude MSL (ft)
    HOVER_ALT_M  = 30.0        # hover altitude AGL at destination (m)
    HEADING_DEG  = None        # heading override (None = compute from route)
    TERRAIN      = False       # download SRTM terrain profile on next run

Exit codes:
    0   all checks passed (or test_executive not present)
    1   one or more test_executive checks failed
    2   build failed
    3   sim failed / no CSV produced
    4   flight planning failed
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import datetime
import gzip
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ── Repo layout ───────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.resolve()
PLANNING = ROOT / "planning"
CONTROLS = ROOT / "controls"

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

# ── Colour helpers ────────────────────────────────────────────────────────────
CYAN  = "\033[0;36m";  GREEN  = "\033[0;32m"
YELLOW= "\033[1;33m";  RED    = "\033[0;31m"
BOLD  = "\033[1m";     NC     = "\033[0m"

def info(msg):    print(f"{CYAN}[test_flight]{NC} {msg}")
def success(msg): print(f"{GREEN}[test_flight] ✓{NC} {msg}")
def warn(msg):    print(f"{YELLOW}[test_flight] ⚠{NC} {msg}")
def fail(msg):    print(f"{RED}[test_flight] ✗{NC} {msg}")
def header(msg):
    bar = "═" * 44
    print(f"\n{BOLD}{bar}{NC}\n{BOLD}  {msg}{NC}\n{BOLD}{bar}{NC}\n")


# ══════════════════════════════════════════════════════════════════════════════
# planning/dash.py  →  DashConfig
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Airport database — extensible CSV lookup with interactive fallback
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Airport:
    icao:   str
    lat:    float
    lon:    float
    elev_m: float


def load_airports() -> dict[str, Airport]:
    """
    Load airport database from planning/airports.csv.

    CSV format (header required):
        icao,lat_deg,lon_deg,elev_m

    Add new destinations here.  Unknown airports trigger interactive entry.
    """
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
                                    elev_m=float(row["elev_m"]))
            except (KeyError, ValueError):
                continue
    return out


def get_airport(icao: str, airports: dict[str, Airport]) -> Airport:
    """Return Airport for icao; prompt user if not in database."""
    if icao in airports:
        return airports[icao]
    info(f"{icao} not in airports.csv")
    return Airport(
        icao=icao,
        lat   = float(input("  Latitude  (°N, negative for S): ")),
        lon   = float(input("  Longitude (°E, negative for W): ")),
        elev_m= float(input("  Elevation (m MSL):               ")),
    )


# ══════════════════════════════════════════════════════════════════════════════
# METAR parser
# ══════════════════════════════════════════════════════════════════════════════

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


def parse_metar(raw: str) -> MetarData:
    raw   = raw.strip()
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
    """Read METAR from file (one line)."""
    return parse_metar(path.read_text(encoding="utf-8", errors="replace").strip())


# ══════════════════════════════════════════════════════════════════════════════
# Geodesy helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Test card  →  planning/test_card.json
# ══════════════════════════════════════════════════════════════════════════════

_ROTOR_CSV = ROOT / "subsystems" / "propulsion" / "rotor_config.csv"

# S4-class baseline — rows that match these values exactly are not emitted
# as overrides (keeps test_card.json clean for the all-default case).
_S4_DEFAULTS = {
    "R_m": 1.45, "n_blades": 5, "chord_m": 0.096,
    "twist_root_deg": 16.0, "twist_tip_deg": 6.0,
    "pitch_offset_deg": 4.4, "P_max_kW": 236, "rpm_hover": 1284,
}

_S4_POWERPLANT = "electric"  # baseline for override-suppression logic

def _rotor_fleet_overrides() -> dict:
    """Read rotor_config.csv and return a rotor_fleet dict with per-rotor
    overrides for any rotor whose geometry or powerplant differs from S4
    defaults.  Rows that are all-default produce no entry (compact JSON for
    stock builds).

    powerplant is a string field ("electric" | "turboshaft" |
    "turbine-electric") and is always written into the entry when it differs
    from the S4 baseline ("electric"), so rotor_system.jl can branch on it to
    construct the correct powerplant struct.
    """
    overrides = []
    if not _ROTOR_CSV.exists():
        return {"overrides": overrides}
    try:
        import csv as _csv
        with open(_ROTOR_CSV, newline="") as fh:
            reader = _csv.DictReader(
                (line for line in fh if not line.startswith("#")),
                skipinitialspace=True)
            for row in reader:
                rotor_id = int(row["rotor_id"])
                entry = {"rotor_id": rotor_id}
                changed = False
                for field, default in _S4_DEFAULTS.items():
                    raw = row.get(field)
                    if raw is None:
                        continue
                    if isinstance(default, int):
                        val = int(round(float(raw)))
                    else:
                        val = float(raw)
                    if abs(val - default) > 1e-9:
                        changed = True
                    entry[field] = val
                # powerplant — string field, compared separately
                powerplant = row.get("powerplant", "").strip() or _S4_POWERPLANT
                entry["powerplant"] = powerplant
                if powerplant != _S4_POWERPLANT:
                    changed = True
                # Always include notes if present
                entry["notes"] = row.get("notes", "")
                if changed:
                    overrides.append(entry)
    except Exception as e:
        import warnings
        warnings.warn(f"rotor_config.csv parse error — using S4 defaults: {e}")
    return {"overrides": overrides}


def generate_test_card(dep: Airport, arr: Airport,
                       dep_wx: MetarData, arr_wx: MetarData,
                       cfg: DashConfig) -> dict:
    dist_m  = haversine(dep.lat, dep.lon, arr.lat, arr.lon)
    brg     = initial_bearing(dep.lat, dep.lon, arr.lat, arr.lon)
    brg_rad = math.radians(brg)

    # Flat-Earth ENU decomposition (valid for < ~500 km legs).
    x_m = round(dist_m * math.cos(brg_rad), 0)
    y_m = round(dist_m * math.sin(brg_rad), 0)
    z_m = round(arr.elev_m - dep.elev_m, 0)

    # Systematic offset pre-correction: waypoint is set 280 m
    # upstream in fly.jl (make_ap_config).  Shift nominal target by the same
    # amount so test_card.json reflects the true landing-pad centre.
    x_m -= round(280.0 * math.cos(brg_rad), 1)
    y_m -= round(280.0 * math.sin(brg_rad), 1)

    dash_alt_agl = cfg.altitude_ft * FT_TO_M - dep.elev_m
    initial_hdg  = cfg.heading_deg if cfg.heading_deg is not None else round(brg, 1)

    dep_psta = station_pressure(dep_wx.altimeter_pa, dep.elev_m)
    arr_psta = station_pressure(arr_wx.altimeter_pa, arr.elev_m)
    dep_da   = density_altitude_ft(dep_psta, dep_wx.temp_c)
    arr_da   = density_altitude_ft(arr_psta, arr_wx.temp_c)

    return {
        "_comment":   (f"eVTOL Tiltrotor — {dep.icao} → {arr.icao}  "
                       f"{dist_m/1000:.1f} km  {brg:.0f}°  "
                       f"Generated by test_flight.py"),
        "_version":   "1.3.0",
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
        "rotor_fleet": _rotor_fleet_overrides(),
        "turbulence_intensity_ms": cfg.turbulence_intensity_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SRTM terrain profile  →  planning/terrain_profile.json
# ══════════════════════════════════════════════════════════════════════════════

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
    # np.frombuffer replaces an O(n²) Python loop (1201×1201 = 1.44M iters for SRTM3).
    # >i2 = big-endian int16, matching the HGT binary spec.
    data   = path.read_bytes()
    n      = int(math.sqrt(len(data) // 2))
    grid   = np.frombuffer(data, dtype=">i2").reshape(n, n).astype(np.float64)
    grid   = np.ascontiguousarray(grid)  # ensure writeable copy for nan assignment
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
    # Bilinear interpolation; NaN cells treated as zero (sea/void fill)
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
    # _sample is scalar; vectorize over the 200 waypoints
    fallback = dep.elev_m + fs * (arr.elev_m - dep.elev_m)
    raw_elv  = np.array([_sample(grids, la, lo) for la, lo in zip(lats, lons)])
    zs_arr   = np.where(np.isnan(raw_elv), fallback, raw_elv).round(1)
    zs       = zs_arr.tolist()
    info(f"Terrain: {n_pts} pts  elev {zs_arr.min():.0f}–{zs_arr.max():.0f} m")
    return {"x_m": xs, "elev_m": zs, "origin_elev_m": dep.elev_m,
            "source": "SRTM GL3", "dep": dep.icao, "arr": arr.icao, "n_points": n_pts}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Flight planning
# ══════════════════════════════════════════════════════════════════════════════

def plan_flight(cfg: DashConfig, terrain_flag: bool,
               dep_metar: Optional[str] = None,
               arr_metar: Optional[str] = None) -> Path:
    dep_wx = parse_metar(dep_metar) if dep_metar else read_metar(PLANNING / "METAR_DEP")
    arr_wx = parse_metar(arr_metar) if arr_metar else read_metar(PLANNING / "METAR_ARR")

    info(f"DEP: {dep_wx.icao}  {dep_wx.temp_c:.1f}°C  "
         f"{dep_wx.altimeter_pa/100:.0f} hPa  "
         f"wind {dep_wx.wind_speed_ms:.1f} m/s from {dep_wx.wind_from_deg:.0f}°")
    info(f"ARR: {arr_wx.icao}  {arr_wx.temp_c:.1f}°C  "
         f"{arr_wx.altimeter_pa/100:.0f} hPa  "
         f"wind {arr_wx.wind_speed_ms:.1f} m/s from {arr_wx.wind_from_deg:.0f}°")

    airports = load_airports()
    dep = get_airport(dep_wx.icao, airports)
    arr = get_airport(arr_wx.icao, airports)

    info(f"{dep.icao}: {dep.lat:.4f}°N  {dep.lon:.4f}°E  {dep.elev_m:.0f} m MSL")
    info(f"{arr.icao}: {arr.lat:.4f}°N  {arr.lon:.4f}°E  {arr.elev_m:.0f} m MSL")

    card = generate_test_card(dep, arr, dep_wx, arr_wx, cfg)
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


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Build
# ══════════════════════════════════════════════════════════════════════════════

def _compile_so(src: Path, out: Path, extra_flags: list[str] = []) -> bool:
    """Compile a single C++ source to a shared library. Returns True on success."""
    cmd = ["g++", "-O3", "-std=c++17", "-fPIC", "-shared",
           *extra_flags, "-o", str(out), str(src)]
    # On first call, print g++ identity for diagnostics
    if not getattr(_compile_so, "_version_printed", False):
        rv = subprocess.run(["g++", "--version"], capture_output=True, text=True)
        info(f"g++ version: {rv.stdout.splitlines()[0] if rv.stdout else rv.stderr.strip()}")
        _compile_so._version_printed = True
    info(f"cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    info(f"exit: {r.returncode}  stdout={repr(r.stdout[:200])}  stderr={repr(r.stderr[:200])}")
    if r.returncode != 0:
        fail(f"Compilation failed ({src.name}) [exit {r.returncode}]:")
        if r.stdout.strip(): print(r.stdout)
        if r.stderr.strip(): print(r.stderr)
        return False
    if r.stderr.strip():
        warn(f"Compiler warnings ({src.name}):")
        print(r.stderr)
    # Independently verify the output file was actually produced and is non-trivial.
    # Some g++ wrappers exit 0 even on failure; this catches that case.
    if not out.exists():
        fail(f"Compilation appeared to succeed (exit 0) but {out.name} was not created.")
        if r.stderr.strip(): print(r.stderr)
        return False
    if out.stat().st_size < 1024:
        fail(f"{out.name} is suspiciously small ({out.stat().st_size} bytes) — "
             f"likely an empty library. Check that all source files were compiled.")
        return False
    return True


def build_autopilot() -> str:
    """
    Compile autopilot.cpp to a versioned shared library.

    Versioned filenames (autopilot_<unix_ts>.so) force Julia to dlopen fresh
    on every run — no process restart is needed even when C++ source changes
    mid-session.  Old versions are pruned to keep controls/ tidy.
    """
    src_ap = CONTROLS / "autopilot.cpp"
    if not src_ap.exists():
        raise FileNotFoundError(f"autopilot.cpp not found at {src_ap}")

    ver = str(int(time.time()))

    # ── autopilot.so ──────────────────────────────────────────────────
    so_ap = CONTROLS / f"autopilot_{ver}.so"
    info(f"Compiling autopilot_{ver}.so")
    if not _compile_so(src_ap, so_ap):
        raise RuntimeError("autopilot build failed")
    (CONTROLS / "autopilot.version").write_text(ver)
    success(f"autopilot_{ver}.so")
    for old in sorted(CONTROLS.glob("autopilot_*.so"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[1:]:
        old.unlink(); info(f"Pruned {old.name}")

    # ── hotas helper (optional) ───────────────────────────────────────
    hc = CONTROLS / "hotas.c"
    if hc.exists():
        rh = subprocess.run(["gcc", "-O2", "-o", str(CONTROLS/"hotas"), str(hc)],
                            capture_output=True, text=True)
        success("hotas") if rh.returncode == 0 else warn("hotas build failed")

    return ver



# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Simulate
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(gui: bool, manual: bool,
                   speed: Optional[float], out_dir: Path) -> Path:
    """
    Launch fly.jl and return the absolute path of the CSV it produced.

    The CSV filename is *chosen here* with a wall-clock timestamp and passed
    to Julia via FLYSIM_CSV_PATH.  This guarantees that the subsequent
    analysis stage always reads the data from *this* run, never from a
    previous one.  A fallback glob is used only if Julia wrote a slightly
    different name (e.g. sub-second clock skew).
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

    # Speed runs are always no-gui unless explicitly requested.
    no_gui = (not gui) or (speed is not None and not gui)

    threads = env.get("JULIA_NUM_THREADS", "auto")
    cmd = ["julia", f"--threads={threads}", str(ROOT / "fly.jl")]
    if no_gui:    cmd.append("--no-gui")
    if manual:    cmd.append("--manual")

    info(f"Command: {' '.join(cmd)}")
    info(f"CSV target: {csv_path}")
    print()

    t0  = time.time()
    ret = subprocess.run(cmd, env=env).returncode
    elapsed = time.time() - t0

    if ret != 0:
        raise RuntimeError(f"fly.jl exited with error (code {ret})")

    success(f"Simulation complete in {elapsed:.0f}s")

    # Exact match preferred; fall back to most-recent CSV in out_dir / ROOT.
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


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Analysis (test checks + HTML report)
# Replaces the former subprocess calls to plot_results.py and test_executive.py,
# both of which are now defunct. Plotting itself now lives in plot_test_flight.py
# — run it separately against the resulting CSV.
# ══════════════════════════════════════════════════════════════════════════════

# ── Check limits (edit here) ──────────────────────────────────────────────────
SOC_MIN          = 20.0    # % minimum arrival SoC
CRUISE_SPEED_TOL = 10.0    # km/h
CRUISE_ALT_TOL   = 150.0   # ft
LAND_ACCURACY_M  = 50.0    # m
RPM_ASYM_LIMIT   = 0.05    # fraction
GZ_NORMAL_LIMIT  = 1.5     # g
GZ_EMERGENCY_LIM = 2.5     # g

RPM_COLS = [f"rpm_r{i}" for i in range(1, 7)]

# ── Check helpers ─────────────────────────────────────────────────────────────

def _steady(df, phase, tail_frac=0.20):
    rows = df[df["phase"] == phase]
    if rows.empty: return rows
    n = len(rows)
    if phase == "dash" and n >= 20:
        # Use the middle 50% — avoids the speed ramp-up at the start and
        # the deceleration onset at the end where the aircraft begins slowing
        # toward the descent arm range while still labelled "dash".
        lo = int(n * 0.25); hi = int(n * 0.75)
        return rows.iloc[lo:hi]
    return rows.iloc[-max(1, int(n * tail_frac)):]

def _gear_events(df):
    gc = df["gear_contact"].astype(float)
    return df[(gc==1) & (gc.shift(1, fill_value=0)==0)]

def _check_phase_sequence(df):
    # Phases that must always appear in a complete auto flight
    REQUIRED = ["landed", "hover", "transition", "fw_climb", "dash", "descent"]
    # Phases that appear only when the approach is slow enough for the
    # autopilot to dwell in them (may be skipped if decel is very fast)
    OPTIONAL = ["fw_descent", "back_transition"]
    seen    = df["phase"].drop_duplicates().tolist()
    missing_req = [p for p in REQUIRED if p not in seen]
    if missing_req:
        return False, f"Missing required phases: {', '.join(missing_req)}"
    # Check ordering of present phases against canonical order
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
    import numpy as np
    target = float(tc.get("fixed_wing",{}).get("dash_speed_kmh", 320.0))
    steady = _steady(df, "dash")
    if steady.empty: return False, f"No 'dash' phase rows", None, CRUISE_SPEED_TOL
    spd = steady["speed_kmh"].to_numpy()
    max_err = float(np.abs(spd-target).max()); rms_err = float(np.sqrt(((spd-target)**2).mean()))
    return max_err<=CRUISE_SPEED_TOL, (f"mean {spd.mean():.1f} km/h | target {target:.0f} | "
        f"max err {max_err:.1f} | RMS {rms_err:.1f} km/h (limit ±{CRUISE_SPEED_TOL:.0f})"), max_err, CRUISE_SPEED_TOL

def _check_cruise_alt(df, tc):
    import numpy as np
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
    import numpy as np
    nav   = tc.get("navigation",{}).get("target",{})
    x_tgt, y_tgt = float(nav.get("x_m",0)), float(nav.get("y_m",0))
    ev    = _gear_events(df)
    td    = ev.iloc[0] if not ev.empty else df.iloc[-1]
    src   = f"t={td['timestamp_s']:.1f}s" if not ev.empty else "last row"
    err   = float(np.hypot(td["x_m"]-x_tgt, td["y_m"]-y_tgt))
    return err<=LAND_ACCURACY_M, (f"touchdown ({td['x_m']:.0f}, {td['y_m']:.0f}) m at {src} | "
        f"target ({x_tgt:.0f}, {y_tgt:.0f}) m | offset {err:.1f} m (limit {LAND_ACCURACY_M:.0f})"), err, LAND_ACCURACY_M

def _check_rpm(df):
    import numpy as np
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

def _energy_table(df) -> str:
    try:
        import numpy as np
        _trapz = getattr(np,"trapezoid",None) or np.trapz
        rows = []
        for phase, grp in df.groupby("phase", sort=False):
            grp = grp.sort_values("timestamp_s")
            t   = grp["timestamp_s"].to_numpy(); p = grp["power_kw"].to_numpy()
            rows.append((phase, round(t[-1]-t[0],1), round(float(_trapz(p,t)),1),
                         round(float(p.mean()),1), round(float(p.max()),1)))
        order = {p:i for i,p in enumerate(df["phase"].drop_duplicates())}
        rows.sort(key=lambda r: order.get(r[0],99))
        lines = ["| Phase | Duration (s) | Energy (kJ) | Mean (kW) | Peak (kW) |",
                 "|-------|-------------|------------|----------|----------|"]
        tot_dur = sum(r[1] for r in rows); tot_e = sum(r[2] for r in rows)
        for ph,dur,e,mn,pk in rows:
            lines.append(f"| {ph} | {dur} | {e:,.1f} | {mn} | {pk} |")
        lines.append(f"| **TOTAL** | {tot_dur} | **{tot_e:,.1f}** | "
                     f"{round(tot_e/max(tot_dur,1),1)} | {max(r[4] for r in rows)} |")
        return "\n".join(lines)
    except Exception:
        return "_Energy data unavailable._"

def _mission_params_md(tc: dict) -> str:
    gen=tc.get("_generated",{}); dep=tc.get("airport",{}); arr=tc.get("destination",{})
    fw=tc.get("fixed_wing",{}); hov=tc.get("hover",{})
    def kt(ms): return ms*1.94384
    cruise_ft=(dep.get("alt_m",0)+fw.get("dash_altitude_m",0))*3.28084
    lines=[
        "### Route","| | |","|---|---|",
        f"| Departure | **{dep.get('icao','?')}** |",
        f"| Arrival | **{arr.get('icao','?')}** |",
        f"| Distance | {gen.get('distance_km','?')} km |",
        f"| Initial bearing | {gen.get('initial_bearing_deg','?')}° |",
        f"| Elevation change | {arr.get('alt_m',0)-dep.get('alt_m',0):+.0f} m |",
        "","### Planned Profile","| | |","|---|---|",
        f"| Hover altitude | {hov.get('alt_m','?')} m AGL |",
        f"| Cruise speed | {fw.get('dash_speed_kmh','?')} km/h |",
        f"| Cruise altitude | {fw.get('dash_altitude_m','?')} m AGL / {cruise_ft:,.0f} ft MSL |",
        "","### Departure","| | |","|---|---|",
        f"| ICAO | {dep.get('icao','?')} |",
        f"| METAR | `{gen.get('dep_metar','—')}` |",
        f"| Temperature | {dep.get('ambient_temp_c','?')} °C |",
        f"| Density altitude | {gen.get('dep_density_alt_ft','?'):,} ft |",
        f"| Wind | {dep.get('wind_from_deg',0):.0f}° / {kt(dep.get('wind_speed_ms',0)):.0f} kt |",
        "","### Arrival","| | |","|---|---|",
        f"| ICAO | {arr.get('icao','?')} |",
        f"| METAR | `{gen.get('arr_metar','—')}` |",
        f"| Temperature | {arr.get('ambient_temp_c','?')} °C |",
        f"| Density altitude | {gen.get('arr_density_alt_ft','?'):,} ft |",
        f"| Wind | {arr.get('wind_from_deg',0):.0f}° / {kt(arr.get('wind_speed_ms',0)):.0f} kt |",
    ]
    return "\n".join(lines)

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

    # ── Checks ────────────────────────────────────────────────────────
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

    # Console summary
    print(f"\n{'─'*60}\n  {verdict}  ({n_pass}/{len(checks)} checks passed)\n{'─'*60}")
    for name, result in checks:
        ok = result[0]; detail = result[1]
        print(f"  {'✅' if ok else '❌'} {name}\n       {detail}")
    print()

    # ── Markdown report ────────────────────────────────────────────────
    dep = tc.get("airport",{}).get("icao","?")
    arr = tc.get("destination",{}).get("icao","?")
    dist = tc.get("_generated",{}).get("distance_km","")
    mission = f"{dep} → {arr}" + (f"  ({dist:.1f} km)" if dist else "")

    md_lines = [
        "# eVTOL Flight Test Report\n",
        f"| | |","|---|---|",
        f"| **CSV** | `{csv_path}` |",
        f"| **Mission** | {mission} |",
        f"| **Overall** | **{verdict}** — {n_pass}/{len(checks)} checks passed |\n",
        "## Mission Parameters\n", _mission_params_md(tc),
        "\n## Check Results\n",
        "| # | Check | Result | Detail |","|---|-------|:------:|--------|",
    ]
    for i, (name, result) in enumerate(checks, 1):
        ok = result[0]; detail = result[1]
        md_lines.append(f"| {i} | {name} | {'✅' if ok else '❌'} | {detail} |")

    # Phase timeline
    grp = (df.groupby("phase", sort=False)["timestamp_s"]
             .agg(t_start="min", t_end="max", n_rows="count").reset_index())
    grp["_ord"] = grp["phase"].map({p:i for i,p in enumerate(df["phase"].drop_duplicates())})
    grp = grp.sort_values("_ord")
    grp["duration_s"] = (grp["t_end"]-grp["t_start"]).round(1)
    md_lines += ["\n## Phase Timeline\n",
                 "| Phase | t_start | t_end | Duration (s) | Rows |",
                 "|-------|---------|-------|-------------|------|"]
    for _,r in grp.iterrows():
        md_lines.append(f"| {r.phase} | {r.t_start:.1f} | {r.t_end:.1f} | {r.duration_s} | {r.n_rows} |")
    md_lines += ["\n## Energy Consumption by Phase\n", _energy_table(df),
                 "\n---\n_Generated by test_flight.py_"]

    md_path = out_dir / "test_report.md"
    md_path.write_text("\n".join(md_lines)+"\n")
    success(f"Report: {md_path}")

    # ── HTML report ────────────────────────────────────────────────────
    colour  = "#1e8449" if n_fail==0 else "#c0392b"
    verd_w  = "PASS" if n_fail==0 else "FAIL"
    check_rows = ""
    for i,(name,result) in enumerate(checks,1):
        ok=result[0]; detail=result[1]
        bg="#eafaf1" if ok else "#fdf0ef"; sym="✅" if ok else "❌"
        check_rows += (f"<tr style='background:{bg}'><td>{i}</td><td>{name}</td>"
                       f"<td style='text-align:center;font-size:1.2em'>{sym}</td>"
                       f"<td style='font-family:monospace;font-size:.9em'>{detail}</td></tr>")
    phase_rows = "".join(
        f"<tr><td>{r.phase}</td><td>{r.t_start:.1f}</td><td>{r.t_end:.1f}</td>"
        f"<td>{r.duration_s}</td><td>{r.n_rows}</td></tr>"
        for _,r in grp.iterrows())
    html_path = out_dir / "report.html"
    html_path.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Flight Test Report — {mission}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 1em;color:#222}}
h1{{border-bottom:2px solid #ddd;padding-bottom:.4em}}h2{{margin-top:2em;color:#444}}
table{{border-collapse:collapse;width:100%;margin:.5em 0}}
th,td{{border:1px solid #ccc;padding:7px 12px;text-align:left;vertical-align:top}}
th{{background:#f4f4f4;font-weight:600}}.verdict{{font-size:1.6em;font-weight:700;color:{colour}}}
code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}</style></head><body>
<h1>eVTOL Flight Test Report</h1>
<table>
  <tr><td style="font-weight:600">CSV</td><td><code>{csv_path}</code></td></tr>
  <tr><td style="font-weight:600">Mission</td><td>{mission}</td></tr>
  <tr><td style="font-weight:600">Overall</td>
      <td><span class="verdict">{verd_w}</span> &nbsp;— {n_pass}/{len(checks)} checks passed</td></tr>
</table>
<h2>Check Results</h2>
<table><thead><tr><th>#</th><th>Check</th><th>Result</th><th>Detail</th></tr></thead>
<tbody>{check_rows}</tbody></table>
<h2>Phase Timeline</h2>
<table><thead><tr><th>Phase</th><th>t start</th><th>t end</th><th>Duration (s)</th><th>Rows</th></tr></thead>
<tbody>{phase_rows}</tbody></table>
<p style="font-size:.8em;color:#888;margin-top:3em">Generated by test_flight.py</p>
</body></html>""")
    success(f"Report: {html_path}")

    if n_fail:
        fail(f"{n_fail} check(s) failed")
    return 0 if n_fail==0 else 1


# ══════════════════════════════════════════════════════════════════════════════
# SQLite export
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_json(obj: dict, prefix: str = "", sep: str = "__") -> dict:
    """
    Recursively flatten a nested dict into a single-level dict.

    Keys are joined with `sep`, e.g.
        {"fixed_wing": {"dash_speed_kmh": 300}} → {"fixed_wing__dash_speed_kmh": 300}

    Lists are JSON-encoded so they fit in a single TEXT cell.
    """
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


def export_sqlite(csv_path: Path, card_path: Path, out_dir: Path) -> Path:
    """
    Write a SQLite database alongside the CSV:

        dash_results_<timestamp>.db
            ├── test_parameters  — 1 row, one column per (flattened) test-card field
            ├── rotor_config     — one row per rotor from rotor_config.csv
            └── telemetry        — all rows/columns from the CSV as a DataFrame

    Column names are sanitised to valid SQL identifiers (non-word chars → '_').
    List values in the test card (e.g. rotor_fleet.overrides) are JSON-encoded
    into a single TEXT cell so nothing is silently dropped.
    Returns the .db path.
    """
    import sqlite3
    import pandas as pd

    db_name = csv_path.stem + ".db"          # e.g. dash_results_20260604_143201.db
    db_path = out_dir / db_name
    db_path.unlink(missing_ok=True)          # start fresh each export

    info(f"Exporting SQLite: {db_path.name}")

    def _safe(s: str) -> str:
        return re.sub(r"[^\w]", "_", s)

    with sqlite3.connect(db_path) as con:

        # ── Table 1: test_parameters ──────────────────────────────────────────
        if card_path.exists():
            try:
                card  = json.loads(card_path.read_text())
                flat  = _flatten_json(card)
                # Build a single-row DataFrame; sanitise column names
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

        # ── Table 2: rotor_config ─────────────────────────────────────────────
        rotor_csv = ROOT / "subsystems" / "propulsion" / "rotor_config.csv"
        try:
            rotor_df = pd.read_csv(rotor_csv, skipinitialspace=True,
                                   comment="#")
            rotor_df.columns = [_safe(c.strip()) for c in rotor_df.columns]
            rotor_df.to_sql("rotor_config", con, index=False, if_exists="replace")
            success(f"  rotor_config: {len(rotor_df)} rotors × {len(rotor_df.columns)} columns")
        except FileNotFoundError:
            warn(f"  rotor_config skipped: {rotor_csv} not found")
        except Exception as e:
            warn(f"  rotor_config skipped: {e}")

        # ── Table 3: telemetry ────────────────────────────────────────────────
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
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Primary demo modes (required, mutually exclusive) ─────────────
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--auto",   action="store_true",
                      help="Autonomous flight: plan → build → simulate → HTML report")
    mode.add_argument("--manual", action="store_true",
                      help="Manual HOTAS flight with cockpit. "
                           "Ctrl+C stops the sim; analysis runs automatically.")

    # ── Auto-mode options ─────────────────────────────────────────────
    p.add_argument("--speed",    type=float, default=None, metavar="X",
                   help="(--auto) Speed multiplier, implies no cockpit. "
                        "Any value; flat-out above ~10×.")

    # ── Shared options ────────────────────────────────────────────────
    p.add_argument("--gui",      action="store_true",
                   help="Show cockpit window in --auto mode. "
                        "Not compatible with --speed (GUI throttles the sim).")
    p.add_argument("--terrain",  action="store_true",
                   help="Force SRTM terrain profile download for this route.")
    p.add_argument("--no-plan",  action="store_true",
                   help="Reuse existing test_card.json; skip flight planning.")
    p.add_argument("--no-build", action="store_true",
                   help="Reuse existing .so files; skip C++ compilation.")
    p.add_argument("--out",      default=str(ROOT), metavar="DIR",
                   help="Output directory for CSV and HTML report.")
    p.add_argument("--csv",      default=None, metavar="PATH",
                   help="Skip simulation; analyse an existing CSV directly.")
    p.add_argument("--db",       action="store_true",
                   help="Export results as a SQLite .db file in addition to CSV. "
                        "Creates dash_results_<timestamp>.db with two tables: "
                        "test_parameters (one row from test_card.json) and "
                        "telemetry (all CSV rows).")

    # ── Direct weather / cruise input (bypasses planning/ files) ─────────
    p.add_argument("--dep-metar", default=None, metavar="METAR",
                   help="Raw DEP METAR string. If supplied, METAR_DEP file is not read.")
    p.add_argument("--arr-metar", default=None, metavar="METAR",
                   help="Raw ARR METAR string. If supplied, METAR_ARR file is not read.")
    p.add_argument("--speed-kmh", type=float, default=None, metavar="KMH",
                   help="Cruise speed (km/h). Overrides dash.py SPEED_KMH.")
    p.add_argument("--alt-ft",    type=float, default=None, metavar="FT",
                   help="Cruise altitude ft MSL. Overrides dash.py ALTITUDE_FT.")
    p.add_argument("--hover-m",   type=float, default=None, metavar="M",
                   help="Hover altitude AGL at destination (m). Overrides dash.py HOVER_ALT_M.")
    p.add_argument("--turb-intensity", type=float, default=None, metavar="MS",
                   help="Dryden turbulence intensity σ (m/s). 0=off, 1.5=light, 3.0=moderate, 6.0=severe.")
    p.add_argument("--bt-speed-ms",    type=float, default=None, metavar="MS",
                   help="Back-transition entry speed (m/s). fw_descent decelerates to this before pitch-up. Default 50.")
    p.add_argument("--nacelle-tilt-deg", type=float, default=None, metavar="DEG",
                   help="Cruise nacelle tilt angle (deg). Min 45, max 90, default 65.")
    args = p.parse_args()

    if args.speed is not None and args.manual:
        fail("--speed is not valid with --manual (manual mode always runs at realtime)")
        sys.exit(1)
    if args.speed is not None and args.gui:
        fail("--speed and --gui are mutually exclusive (GUI throttles the sim)")
        sys.exit(1)

    # Derive internal flags from the two clean modes
    gui    = args.manual or args.gui   # cockpit on for manual or --auto --gui
    manual = args.manual
    speed  = args.speed           # None = realtime

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = DashConfig.load()
    except Exception as e:
        fail(str(e)); sys.exit(4)

    # Apply CLI overrides to DashConfig (only if explicitly passed)
    if args.speed_kmh       is not None: cfg.speed_kmh               = args.speed_kmh
    if args.alt_ft          is not None: cfg.altitude_ft             = args.alt_ft
    if args.hover_m         is not None: cfg.hover_alt_m             = args.hover_m
    if args.turb_intensity  is not None: cfg.turbulence_intensity_ms = args.turb_intensity
    if args.bt_speed_ms       is not None: cfg.back_trans_speed_ms     = args.bt_speed_ms
    if args.nacelle_tilt_deg  is not None: cfg.nacelle_tilt_deg        = args.nacelle_tilt_deg

    card_path = PLANNING / "test_card.json"
    csv_path: Optional[Path] = Path(args.csv).resolve() if args.csv else None
    exit_code = 0
    dep_lbl = arr_lbl = "?"

    # ── 1: Plan ───────────────────────────────────────────────────────
    header("Flight planning")
    if not args.no_plan and not args.csv:
        try:
            card_path = plan_flight(cfg, args.terrain,
                                       dep_metar=args.dep_metar,
                                       arr_metar=args.arr_metar)
        except Exception as e:
            if card_path.exists():
                warn(f"Planning failed ({e}) — using existing test_card.json")
            else:
                fail(f"Planning failed: {e}"); sys.exit(4)
    elif args.csv:
        info("--csv mode: skipping planning")
    else:
        if not card_path.exists():
            fail("test_card.json not found — run without --no-plan first"); sys.exit(4)
        info(f"--no-plan: using {card_path}")

    if card_path.exists():
        try:
            d = json.loads(card_path.read_text())
            dep_lbl = d.get("airport",     {}).get("icao", "?")
            arr_lbl = d.get("destination", {}).get("icao", "?")
        except Exception:
            pass
    success(f"{dep_lbl} → {arr_lbl} | {cfg.speed_kmh:.0f} km/h | {cfg.altitude_ft:.0f} ft")

    # ── 2: Build ──────────────────────────────────────────────────────
    if not args.csv:
        header("Building")
        if not args.no_build:
            try:
                build_autopilot()   # flight controller 
            except Exception as e:
                fail(str(e)); sys.exit(2)
        else:
            def _ver_label(version_file: Path, glob_name: str) -> str:
                vf = version_file
                if vf.exists():
                    return f"{glob_name}_{vf.read_text().strip()}.so"
                existing = sorted(vf.parent.glob(f"{glob_name}_*.so"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
                return existing[0].name if existing else f"{glob_name}.so (no version file)"
            # info(f"--no-build: blades/fly.so labels omitted — C++ solver disabled")
            info(f"--no-build: {_ver_label(CONTROLS/'autopilot.version', 'autopilot')}")

    # ── 3: Simulate ───────────────────────────────────────────────────
    if not args.csv:
        header(f"{'Auto' if args.auto else 'Manual'} flight: {dep_lbl} → {arr_lbl}")
        if args.manual:
            info("Cockpit launching. Fly the aircraft. Press Ctrl+C to end and generate report.")
        try:
            csv_path = run_simulation(gui, manual, speed, out_dir)
        except KeyboardInterrupt:
            # Manual mode: Ctrl+C is normal termination — find the CSV and analyse
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
                sys.exit(3)
        except Exception as e:
            fail(str(e)); sys.exit(3)
    else:
        if not csv_path.exists():
            fail(f"--csv: file not found: {csv_path}"); sys.exit(3)
        info(f"--csv mode: {csv_path}")

    # ── 4: Analyse ────────────────────────────────────────────────────
    header("Analysis")
    if csv_path:
        try:
            exit_code = run_analysis(csv_path, card_path, out_dir)
        except Exception as e:
            warn(f"Analysis error: {e}")

        if args.db:
            try:
                export_sqlite(csv_path, card_path, out_dir)
            except Exception as e:
                warn(f"SQLite export failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    header("Summary")
    mode_label = "AUTO" if args.auto else "MANUAL"
    print(f"  {'Mode:':<14} {mode_label}")
    print(f"  {'Route:':<14} {dep_lbl} → {arr_lbl}")
    print(f"  {'Speed:':<14} {cfg.speed_kmh:.0f} km/h")
    print(f"  {'Altitude:':<14} {cfg.altitude_ft:.0f} ft MSL")
    if csv_path:
        print(f"  {'CSV:':<14} {csv_path}")
    if args.db and csv_path:
        db_candidate = out_dir / (csv_path.stem + ".db")
        if db_candidate.exists():
            print(f"  {'SQLite DB:':<14} {db_candidate}")
    rep_html = out_dir / "report.html"
    rep_md   = out_dir / "test_report.md"
    if rep_html.exists():
        print(f"  {'Report:':<14} {rep_html}")
    elif rep_md.exists():
        print(f"  {'Report:':<14} {rep_md}")
    if args.speed:
        print(f"  {'Sim speed:':<14} {args.speed}×")
    print(f"  {'Exit code:':<14} {exit_code}\n")
    os._exit(exit_code)


if __name__ == "__main__":
    main()

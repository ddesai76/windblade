# fly.jl:         Advanced Air Mobility Tiltrotor Simulation
# AUTHOR:         DANIEL DESAI
# UPDATED:        2026-06-19
# VERSION:        0.1.1
#
# Single entry point: loads subsystems, runs physics model, streams to glass_cockpit.jl
#
#
# State vector (22 states):
#   1  vx          forward speed (m/s)
#   2  alt         altitude AGL (m)
#   3  tilt        rotor tilt angle (rad)   0=hover, π/2=cruise
#   4  dtilt       tilt rate (rad/s)
#   5  pitch       pitch angle (rad)        positive=nose up
#   6  dpitch      pitch rate (rad/s)       — driven by ωy (state 17)
#   7  roll        roll angle (rad)
#   8  droll       roll rate (rad/s)        — driven by ωx (state 16)
#   9  yaw         yaw angle (rad)
#   10 dyaw        yaw rate (rad/s)         — driven by ωz (state 18)
#   11 thrust_lag  actual rotor thrust (N) — first-order spool lag (aggregate)
#   12 soc         battery state of charge (0–1)
#   13 τ           mission clock (s); negative = pre-hover
#   14 x           ground-track position, forward from origin (m)
#   15 y           ground-track position, rightward from origin (m)
#   16 ωx          body roll rate  (rad/s) — Euler equation state
#   17 ωy          body pitch rate (rad/s) — Euler equation state
#   18 ωz          body yaw rate   (rad/s) — Euler equation state
#   19 terrain_agl terrain elevation AGL at departure datum (m) — held 0.0
#   20 turb_u      Dryden longitudinal gust velocity (m/s)
#   21 turb_v      Dryden lateral gust velocity (m/s)
#   22 turb_w      Dryden vertical gust velocity (m/s)
#
# Note: states 19–22 added with Dryden turbulence feature.
# The C++ autopilot interface still receives only states 1–18 + terrain AGL
# at position 19 via u_f64 = vcat(Float64.(u[1:18]), _agl_t) — see saving
# callback. Do not reorder or insert states between 1–18.
#
# Controls flow:
#   AUTO:   saving callback calls compute_controls() → autopilot.so via @ccall,
#           caches result in AP_CTRL_CACHE[]. build_ode reads cache every step.
#           ForwardDiff Jacobian never touches C code.
#   MANUAL: hotas subprocess reads /dev/input/js0 at 50 Hz via pipe →
#           HotasState atomics → hotas_to_control_output() inline in build_ode.
#   Both paths produce ControlOutput consumed identically by build_ode.
#
# Navigation:
#   planning/navigation.jl    — NavTarget, nav_guidance, NavMapState
#   planning/mission_planner.jl — TC, TIMINGS, phase_label, tau_derivative
#   planning/test_card.json   — single source of truth for all mission params
#   Dash to descent triggered by range-to-waypoint, not by fixed timer.
#   nav_guidance() injects a bearing-error yaw command into build_ode
#   during fixed-wing cruise to steer the aircraft toward the waypoint.
#
# Wrench reconstruction (per-rotor RPM/kW for CSV + cockpit):
#   Saving callback reads AP_CTRL_CACHE[] directly — no sched_* calls.
#
# Usage:
#   julia --threads auto fly.jl                  # autopilot + cockpit
#   julia --threads auto fly.jl --no-gui         # CSV only
#   julia --threads auto fly.jl --manual         # HOTAS input
#   FLYSIM_SPEED=3.0 julia --threads auto fly.jl # 3× realtime
#   FLYSIM_CSV=myrun.csv julia --threads auto fly.jl
#
# Build autopilot shared library:
#   g++ -O2 -std=c++17 -fPIC -shared -o controls/autopilot.so controls/autopilot.cpp
#
# Install deps:
#   julia -e 'using Pkg; Pkg.add(["OrdinaryDiffEq","DiffEqCallbacks",
#             "GLMakie","Observables","CSV","DataFrames","JSON"])'
#

using OrdinaryDiffEq
using SciMLBase: terminate!
using DiffEqCallbacks
using CSV, DataFrames, Observables, Printf, Dates
using JSON
import Libdl

# ── Parse arguments ───────────────────────────────────────────────────
const SHOW_GUI  = !("--no-gui" in ARGS)
const MANUAL    = "--manual"  in ARGS
const OUT_CSV   = let
    _ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    get(Base.ENV, "FLYSIM_CSV_PATH", "dash_results_$(_ts).csv")
end
const RT_FACTOR = parse(Float64, get(Base.ENV, "FLYSIM_SPEED", "1.0"))

# ── Subsystems ────────────────────────────────────────────────────────
# Load order:
#   atmosphere   — rho(), atm_wind()
#   rotor_system — FLEET, RP
#   battery_model, airframe, landing_gear — physics subsystems
#   actuators    — ControllerParams/CP, actuator_accel (no dependencies)
#   navigation   — NavTarget, nav_guidance, NavMapState  (_MiniJSON via JSON.jl)
#   mission_planner — TC, TIMINGS, DESCENT_INITIATION_M, phase_label, tau_derivative
#   rotor_mixer  — needs TC, TIMINGS, CP, FLEET all finalised
const _SUBSYS     = joinpath(@__DIR__, "subsystems")
const _CONTROLS   = joinpath(@__DIR__, "controls")
const _PROPULSION = joinpath(@__DIR__, "subsystems", "propulsion")
const _PLANNING   = joinpath(@__DIR__, "planning")
const _WORLD = joinpath(@__DIR__, "world")
include(joinpath(_WORLD,   "atmosphere.jl"))
include(joinpath(_WORLD,   "terrain.jl"))
using .Terrain
include(joinpath(_PROPULSION, "rotor_system.jl"))
include(joinpath(_SUBSYS,     "battery_model.jl"))
include(joinpath(_SUBSYS,     "airframe.jl"))
include(joinpath(_SUBSYS,     "landing_gear.jl"))
include(joinpath(_SUBSYS,     "actuators.jl"))
include(joinpath(_PLANNING,   "navigation.jl"))
include(joinpath(_PLANNING,   "mission_planner.jl"))

# ── Apply airport / environment from TC → ATM ─────────────────────────
# Done here so ATM is guaranteed to exist before rotor_mixer.jl loads.
ATM.airport_icao     = TC.airport_icao
ATM.airport_alt_m    = TC.airport_alt_m
ATM.ambient_temp_c   = TC.ambient_temp_c
ATM.ambient_pressure = TC.ambient_pressure_pa
ATM.wind.u           = TC.wind_u_ms
ATM.wind.v           = TC.wind_v_ms

compute_da_correction()                    # writes RP.da_correction + RP.hover_thrust_N
include(joinpath(_PROPULSION, "rotor_mixer.jl"))   # must follow ATM setup

# ── Dryden turbulence init (DRYDEN) ──────────────────────────────────
# Reads turbulence_intensity_ms from test_card.json.
# Default 0.0 keeps all existing runs deterministic.
ATM.wind.turbulence_intensity = Float64(TC.turbulence_intensity_ms)
init_dryden_rng!(ATM, 1010)

# ── Navigation init ───────────────────────────────────────────────────
# Reads navigation block from planning/test_card.json, validates target,
# creates the NavMapState ring buffer for the moving-map display.
const NAV, NAV_MAP = nav_init()

# ── Terrain model ─────────────────────────────────────────────────────
# Priority: predefined profile → terrain_profile.json → flat_model.
# Predefined profiles (KAXX-KSAF, KDEN-KCOS, etc.) are in world/terrain.jl
# and require no downloads or files. For unknown routes, flat_model is used
# unless flight_plan.py --terrain has generated terrain_profile.json.
const TERRAIN = let
    prof_path = joinpath(_PLANNING, "terrain_profile.json")
    # Route bearing (rad): needed by terrain_alt to project (x,y) → along-track.
    # atan(y, x) in ENU where x=North, y=East gives bearing from North.
    _route_bearing_rad = atan(Float64(nav_wy(NAV)), Float64(nav_wx(NAV)))
    t = load_terrain(TC.airport_icao, TC.dest_icao,
                     Float64(TC.airport_alt_m), Float64(TC.dest_alt_m),
                     Float64(hypot(nav_wx(NAV), nav_wy(NAV))),
                     prof_path;
                     bearing_rad = _route_bearing_rad)
    # If load_terrain fell back to flat_model AND a JSON profile exists,
    # load the JSON here (load_terrain avoids a JSON dependency).
    if occursin("flat_model", t.source) && isfile(prof_path)
        try
            prof = JSON.parsefile(prof_path)
            @info "Terrain: loaded SRTM profile from $prof_path ($(length(prof["x_m"])) pts)"
            TerrainModel(Float64.(prof["x_m"]), Float64.(prof["elev_m"]),
                         Float64(TC.airport_alt_m),
                         Float64[], Float64[], Matrix{Float64}(undef,0,0),
                         "terrain_profile.json (SRTM)", _route_bearing_rad)
        catch e
            @warn "Terrain: failed to parse $prof_path — $e. Using flat_model."
            t
        end
    else
        t
    end
end
terrain_selftest()

# ── Callback-side state ───────────────────────────────────────────────
const _alt_prev_cb   = Ref(0.0)
const _last_phase_cb = Ref("")   # hysteresis: suppress rapid phase toggling
const _last_phase_t  = Ref(0.0)  # sim-time of last phase change
const PHASE_DWELL_S  = 10.0      # min seconds between phase label changes
const _t_cb_ref    = Ref(0.0)

# ── Cockpit ───────────────────────────────────────────────────────────
if SHOW_GUI
    include(joinpath(@__DIR__, "glass_cockpit.jl"))
end

# ── Derived constants ─────────────────────────────────────────────────
const TARGET_MS = TC.dash_speed_kmh / 3.6
const WEIGHT_N  = AP.mass_kg * 9.81
@show ATM.airport_alt_m ATM.ambient_temp_c ATM.ambient_pressure  # ← temporary
@show rho(0.0)  
preflight_da_warning(WEIGHT_N)             # @warn if T/W < 1.05 at departure DA

# ── Inertia tensor (body frame, kg·m²) ────────────────────────────────
const Ixx = 3500.0
const Iyy = 4800.0
const Izz = 7200.0

# ── Unit helpers ──────────────────────────────────────────────────────
const M_TO_FT   = 3.28084
agl_m_to_msl_ft(alt_agl_m) = (alt_agl_m + ATM.airport_alt_m) * M_TO_FT

# ── Rotor fleet labels ────────────────────────────────────────────────
const _ROTOR_LABELS = [FLEET.units[i].label for i in 1:6]
# Per-rotor powerplant tag ("electric" | "turbine_electric") for cockpit
# fuel-gauge gating and CSV export — see fleet_powerplant_labels() in
# rotor_system.jl. Static for the run (FLEET's motor mix doesn't change
# mid-flight), so computed once here rather than per saving-callback step.
const _ROTOR_POWERPLANT_LABELS = fleet_powerplant_labels(FLEET)

# ══════════════════════════════════════════════════════════════════════
#  ControlOutput — shared interface between autopilot.so and HOTAS path
# ══════════════════════════════════════════════════════════════════════
# Must match C struct layout in autopilot.cpp exactly (field order, types).
struct ControlOutput
    thrust_cmd :: Cdouble   # Newtons
    roll_cmd   :: Cdouble   # radians, target roll angle
    pitch_cmd  :: Cdouble   # radians, target pitch angle
    yaw_cmd    :: Cdouble   # [-1, 1], scaled by MAX_YAW_RATE below
    tilt_mode  :: Cint      # 0 = hover nacelles, 1 = cruise nacelles
    autopilot  :: Cint      # 0/1 as Cint — C++ bool(1)+pad(3) → 4 bytes
    brakes     :: Cint      # 0/1 wheel brakes
    _pad       :: Cint      # C++ struct alignment padding (sizeof=48)
end

# ══════════════════════════════════════════════════════════════════════
#  APConfig — mission params passed to autopilot.so each call
# ══════════════════════════════════════════════════════════════════════
# Must match typedef struct APConfig in autopilot.cpp exactly.
# t_dash_end removed — dash→fw_descent is range-triggered via wp_x/wp_y.
struct APConfig
    mass_kg              :: Cdouble
    hover_thrust_N       :: Cdouble
    weight_N             :: Cdouble
    gear_cg_m            :: Cdouble
    preflight_hold_s     :: Cdouble
    t_trans_end          :: Cdouble
    t_fw_climb           :: Cdouble
    t_fw_desc            :: Cdouble   # worst-case sentinel (not the trigger)
    t_pitch_down         :: Cdouble
    wp_x                 :: Cdouble   # active waypoint x (m forward)  — RTB=0
    wp_y                 :: Cdouble   # active waypoint y (m rightward) — RTB=0
    descent_initiation_m :: Cdouble   # range trigger, from mission_planner.jl
    hover_alt_m          :: Cdouble
    target_hover_alt_m   :: Cdouble   # hover_alt_m + z_m: descent target in ODE AGL frame
    dash_altitude_m      :: Cdouble
    dash_speed_ms        :: Cdouble
    soc_min              :: Cdouble
    pitch_limit_rad      :: Cdouble
    roll_limit_rad       :: Cdouble
    land_tilt_s          :: Cdouble   # TC.land_tilt_s
    min_dash_s           :: Cdouble   # computed from target range vs descent_initiation_m
end

function make_ap_config()::APConfig
    APConfig(
        Float64(AP.mass_kg),
        Float64(RP.hover_thrust_N),
        WEIGHT_N,
        Float64(GEAR.cg_to_ground_m),
        Float64(TC.preflight_hold_s),
        Float64(TIMINGS.T_TRANS_END),
        Float64(TIMINGS.T_FW_CLIMB),
        Float64(TIMINGS.T_FW_DESC),
        Float64(TIMINGS.T_PITCH_DOWN),
        Float64(nav_wx(NAV)) - 280.0*cos(atan(Float64(nav_wy(NAV)),Float64(nav_wx(NAV)))),  # wp_x
        Float64(nav_wy(NAV)) - 280.0*sin(atan(Float64(nav_wy(NAV)),Float64(nav_wx(NAV)))),  # wp_y
        Float64(DESCENT_INITIATION_M),    # descent_initiation_m
        Float64(TC.hover_alt_m),
        Float64(TC.target_hover_alt_m),   # destination hover ref in ODE AGL frame
        Float64(TC.dash_altitude_m),
        Float64(TC.dash_speed_kmh) / 3.6,
        0.10,                             # soc_min
        deg2rad(60.0),                    # pitch_limit_rad
        deg2rad(45.0),                    # roll_limit_rad
        Float64(TC.land_tilt_s),          # land_tilt_s
        Float64(let                        # min_dash_s — scaled to mission range
            target_range = hypot(nav_wx(NAV), nav_wy(NAV))
            avail_m      = max(target_range - DESCENT_INITIATION_M, 0.0)
            clamp(avail_m / (TC.dash_speed_kmh / 3.6), 5.0, 20.0)
        end),
    )
end

# ══════════════════════════════════════════════════════════════════════
#  Autopilot shared library
# ══════════════════════════════════════════════════════════════════════
# Versioned .so: test_flight.py writes controls/autopilot.version on each
# build. Julia dlopen()s a unique path per build — no restart needed.
# Falls back to autopilot.so for manual compile workflows.
const _AUTOPILOT_SO = let
    ver_path = joinpath(@__DIR__, "controls", "autopilot.version")
    if isfile(ver_path)
        ver = strip(read(ver_path, String))
        joinpath(@__DIR__, "controls", "autopilot_$(ver).so")
    else
        joinpath(@__DIR__, "controls", "autopilot.so")
    end
end

if !MANUAL
    isfile(_AUTOPILOT_SO) || error(
        "$(_AUTOPILOT_SO) not found.\n" *
        "Run: python3 test_flight.py --no-run\n" *
        "or:  g++ -O2 -std=c++17 -fPIC -shared " *
        "-o controls/autopilot.so controls/autopilot.cpp")
    Libdl.dlopen(_AUTOPILOT_SO, Libdl.RTLD_NOW)
end



@assert isbitstype(ControlOutput)
@assert isbitstype(APConfig)
# Layout check: 4×double(32) + Cint(4) + Cint(4) + Cint(4) + Cint_pad(4) = 48 bytes.
if !MANUAL
    @assert sizeof(ControlOutput) == 48 """
ControlOutput size mismatch: Julia=$(sizeof(ControlOutput)) expected=48.
Check Bool/_Bool padding vs C struct layout in autopilot.cpp.
"""
end

const AP_CFG     = make_ap_config()
const AP_CFG_REF = Ref(AP_CFG)

# Reset any persistent state in autopilot.so and mission planner (latching flags)
if !MANUAL
    @ccall _AUTOPILOT_SO.reset_autopilot_state()::Cvoid
end
reset_mission_state()

function compute_controls(state::Vector{Float64},
                          cfg_ref::Ref{APConfig})::ControlOutput
    out = Ref{ControlOutput}()
    GC.@preserve out cfg_ref begin
        @ccall _AUTOPILOT_SO.compute_controls(
            state         :: Ptr{Cdouble},
            length(state) :: Cint,
            cfg_ref       :: Ptr{APConfig},
            out           :: Ptr{ControlOutput})::Cvoid
    end
    return out[]
end

# ══════════════════════════════════════════════════════════════════════
#  AP handoff flag
# ══════════════════════════════════════════════════════════════════════
const AP_HANDOFF      = Threads.Atomic{Bool}(false)
const _thrust_prev_cb = Ref(0.0)
const _DESCENT_ARMED  = Ref(false)
const _LANDED         = Ref(false)
const AUTOLAND_RNG_M  = 1500.0

# ══════════════════════════════════════════════════════════════════════
#  HOTAS — shared state populated by the hotas reader subprocess
# ══════════════════════════════════════════════════════════════════════
Base.@kwdef mutable struct HotasState
    roll_cmd    :: Threads.Atomic{Float64} = Threads.Atomic{Float64}(0.0)
    pitch_cmd   :: Threads.Atomic{Float64} = Threads.Atomic{Float64}(0.0)
    thrust_frac :: Threads.Atomic{Float64} = Threads.Atomic{Float64}(0.0)
    yaw_cmd     :: Threads.Atomic{Float64} = Threads.Atomic{Float64}(0.0)
    tilt_frac   :: Threads.Atomic{Float64} = Threads.Atomic{Float64}(0.0)
    trim        :: Threads.Atomic{Float64} = Threads.Atomic{Float64}(0.0)
    brakes      :: Threads.Atomic{Bool}    = Threads.Atomic{Bool}(false)
    connected   :: Threads.Atomic{Bool}    = Threads.Atomic{Bool}(false)
end
const HOTAS = HotasState()

const MAX_YAW_RATE = deg2rad(20.0)

function hotas_to_control_output()::ControlOutput
    # Thrust ceiling: fleet_thrust_available sums BEM per rotor using each
    # unit's actual blade_geom — super rotors get their full capability.
    # Evaluated at departure elevation (conservative; safe for demo).
    _manual_thrust_max = fleet_thrust_available(0.0)
    ControlOutput(
        HOTAS.thrust_frac[] * 2.5 * _manual_thrust_max,
        clamp(HOTAS.roll_cmd[]  * deg2rad(20.0), -0.35, 0.35),
        clamp((HOTAS.pitch_cmd[] + HOTAS.trim[] * 0.5) * deg2rad(25.0), -0.52, 0.52),
        HOTAS.yaw_cmd[],
        HOTAS.tilt_frac[] >= 0.5 ? Cint(1) : Cint(0),
        Cint(0),    # autopilot = 0 (HOTAS in command)
        HOTAS.brakes[] ? Cint(1) : Cint(0),
        Cint(0),    # _pad
    )
end

function start_hotas(device::String="/dev/input/js0")
    bin = joinpath(@__DIR__, "controls", "hotas")
    isfile(bin)  || (@warn "controls/hotas not found — build: gcc -O2 -o controls/hotas controls/hotas.c"; return nothing)
    ispath(device) || (@warn "HOTAS device $device not found"; return nothing)

    proc = try
        open(pipeline(`$bin $device`, stderr=stderr), "r")
    catch e
        @warn "hotas launch failed: $e"
        return nothing
    end

    function parse_line(line)
        p = split(strip(line))
        length(p) < 5 && return
        Threads.atomic_xchg!(HOTAS.roll_cmd,    parse(Float64, p[1]))
        Threads.atomic_xchg!(HOTAS.pitch_cmd,   parse(Float64, p[2]))
        Threads.atomic_xchg!(HOTAS.thrust_frac, parse(Float64, p[3]))
        Threads.atomic_xchg!(HOTAS.yaw_cmd,     parse(Float64, p[4]))
        Threads.atomic_xchg!(HOTAS.tilt_frac,   parse(Float64, p[5]))
        length(p) >= 6 && Threads.atomic_xchg!(HOTAS.trim,   parse(Float64, p[6]))
        length(p) >= 7 && Threads.atomic_xchg!(HOTAS.brakes, parse(Int, p[7]) != 0)
        if length(p) >= 8 && parse(Int, p[8]) != 0 && !_DESCENT_ARMED[]
            _DESCENT_ARMED[] = true
            @warn "[AP] Descent armed via HOTAS btn 2"
        end
    end

    first_line = readline(proc)
    if isempty(first_line)
        @warn "hotas produced no output — check permissions: sudo usermod -aG input \$USER"
        return nothing
    end
    parse_line(first_line)
    Threads.atomic_xchg!(HOTAS.connected, true)
    println("HOTAS first sample: $first_line")

    return Threads.@spawn try
        for line in eachline(proc); parse_line(line); end
        @warn "hotas process ended"
        Threads.atomic_xchg!(HOTAS.connected, false)
    catch e
        @warn "hotas reader error: $e"
        Threads.atomic_xchg!(HOTAS.connected, false)
    end
end

# ══════════════════════════════════════════════════════════════════════
#  Cached autopilot output
# ══════════════════════════════════════════════════════════════════════
# Updated by saving callback (~10 Hz, plain Float64, outside ForwardDiff).
# Read by build_ode every step — treated as a constant by the Jacobian.
const AP_CTRL_CACHE = Ref(ControlOutput(
    WEIGHT_N,   # thrust_cmd — safe initial hover weight
    0.0, 0.0, 0.0,
    Cint(0),    # tilt_mode — hover
    Cint(1),    # autopilot = 1
    Cint(0),    # brakes
    Cint(0),    # _pad
))

# BEM fleet power cache — updated by saving callback (~10 Hz) from
# allocate_wrench_vx, which is geometry-aware (FLEET radius/chord).
# Read by build_ode for SoC integration so dsoc reflects actual rotor
# geometry rather than the geometry-blind rotor_power_kw() estimate.
# One-step lag is negligible for SoC — same pattern as AP_CTRL_CACHE.
const BEM_POWER_KW_CACHE = Ref(0.0)

# Vertical velocity from the previous ODE step (m/s, positive = climbing).
# Written by build_ode after dalt is computed; read at the top of the next
# step for VRS gating. One-step lag is negligible — VRS onset is ~seconds.
const _VZ_CACHE = Ref(0.0)

# ══════════════════════════════════════════════════════════════════════
#  ODE
# ══════════════════════════════════════════════════════════════════════
function build_ode(du, u, p, t)
    vx, alt, tilt, dtilt, pitch, dpitch, roll, droll, yaw, dyaw,
        thrust_lag, soc, τ, x, y, ωx, ωy, ωz = u

    dt::Float64 = max(p[], 0.001)
    T = TIMINGS

    # ── Ground effect ─────────────────────────────────────────────────
    ge_r = rotor_ge(alt)
    ge_l = fw_ge_lift(alt)
    ge_d = fw_ge_drag(alt)

    # ── Airspeed ──────────────────────────────────────────────────────
    wu, wv, ww = atm_wind(ATM, alt)
    vx_air = vx - wu

    # ── Dryden turbulence (DRYDEN) ────────────────────────────────────
    # States u[end-2:end] are the three gust velocity components (m/s).
    # Noise is held constant between saving callback resamples (every
    # DRYDEN_DT_S) — safe for ForwardDiff because dryden_noise is not
    # mutated inside the ODE; only the saving callback writes to it.
    if ATM.wind.turbulence_intensity > 0.0
        dtu, dtv, dtw = atm_turbulence_deriv(
            u[end-2], u[end-1], u[end],
            abs(vx_air), Float64(alt), ATM, ATM.dryden_noise)
        du[end-2] = dtu; du[end-1] = dtv; du[end] = dtw
        vx_air -= u[end-2]   # subtract longitudinal gust from airspeed
    else
        du[end-2] = du[end-1] = du[end] = 0.0
    end

    # ── Control output ────────────────────────────────────────────────
    ctrl = MANUAL ? hotas_to_control_output() : AP_CTRL_CACHE[]

    thrust_cmd_ap   = ctrl.thrust_cmd
    pitch_cmd       = ctrl.pitch_cmd
    roll_cmd        = ctrl.roll_cmd
    yaw_rate_target = ctrl.yaw_cmd * MAX_YAW_RATE

    # ── Tilt actuator ─────────────────────────────────────────────────
    # Tilt command: full cruise angle when tilt_mode=1, hover when 0.
    # The actuator bandwidth (tilt_wn=1.2 rad/s) limits how fast tilt
    # actually moves — no need to velocity-schedule the command itself.
    # The old vx_air-based schedule created a deadlock: tilt needs to
    # move to generate forward thrust, but tilt waited for forward speed.
    tilt_cmd = ctrl.tilt_mode == 1 ? deg2rad(65.0) : 0.0
    ddtilt = actuator_accel(tilt, dtilt, tilt_cmd, CP.tilt_wn, CP.tilt_zeta)

    # ── Thrust (preflight ramp AUTO only) ─────────────────────────────
    # Ramp is keyed to ODE time t (not τ) because τ is frozen at
    # -preflight_hold_s until hover altitude is reached. Using τ meant
    # the ramp never fired when min(σ_alt, σ_tau) kept τ from advancing.
    ramp_t_start = TC.preflight_hold_s - TC.preflight_ramp_s
    ramp_frac    = clamp((t - ramp_t_start) / max(TC.preflight_ramp_s, 0.1),
                          zero(t), one(t))
    thrust_held  = WEIGHT_N + ramp_frac * (thrust_cmd_ap - WEIGHT_N)
    # Apply ramp throughout the entire hover climb (τ < 0), not just near the ground.
    # The narrow alt threshold caused the ramp to cut out at 1.22m while the AP
    # cache still held a stale low-thrust value, stalling the climb.
    thrust_cmd   = ifelse(!MANUAL && τ < 0.0,
                          thrust_held, thrust_cmd_ap)
    # Negative thrust_cmd = reverse rotors for braking (fw_descent)
    _T_max = rotor_thrust_available(alt)
    thrust_cmd = clamp(thrust_cmd, -0.5 * _T_max, _T_max)
    thrust_cmd, _vrs_f = vrs_gated_thrust(thrust_cmd, _VZ_CACHE[], alt)
    dthrust    = thrust_derivative(thrust_lag, thrust_cmd)
    thrust_act = thrust_lag

    # Debug: cruise force balance (manual only)
    if MANUAL && mod(t, 5.0) < 0.02 && tilt > deg2rad(30.0)
        wl = wing_lift(max(vx_air, 0.0), pitch, tilt, alt)
        rv = thrust_act * cos(tilt)
        @printf("CRUISE: vx=%.1f tilt=%.1f° thrust=%.0f lift=%.0f vert=%.0f W=%.0f Fz=%.0f\n",
                vx_air, rad2deg(tilt), thrust_act, wl, rv, WEIGHT_N, rv+wl-WEIGHT_N)
    end

    # ── Forces ────────────────────────────────────────────────────────
    f_body = fuselage_drag(vx_air, alt)

    Fx = if MANUAL
        fwd = thrust_act * sin(tilt) +
              thrust_act * sin(clamp(-pitch, 0.0, 0.35)) * cos(tilt)
        fwd - wing_drag(max(vx_air, 0.1), pitch, tilt, alt) * ge_d - f_body
    elseif τ < 0.0
        -f_body
    elseif τ < T.T_TRANS_END
        fwd = min(thrust_act * sin(tilt),
                  wing_drag(max(vx_air, 0.1), pitch, tilt, alt) +
                  AP.mass_kg * (TARGET_MS / TC.trans_duration_s) * 1.5)
        fwd - wing_drag(max(vx_air, 0.1), pitch, tilt, alt) - f_body
    else
        # fw_climb, dash, fw_descent, back_transition, descent:
        # In cruise (tilt=π/2): sin(tilt)≈1 → full forward thrust.
        # In hover (tilt=0):    sin(tilt)=0 → rotor thrust is vertical.
        #   Forward motion comes from body pitch tilting the thrust vector.
        #   Include pitch contribution: Fx += thrust * sin(-pitch) * cos(tilt)
        #   Negative pitch = nose down = forward thrust.
        #   Clamped to nose-down only (no reverse thrust from nose-up pitch).
        let pitch_fwd = clamp(-pitch, -0.35, 0.35)
            # thrust_act < 0 means reverse rotors — braking force regardless of tilt
            thrust_fwd = thrust_act * sin(tilt) +
                         thrust_act * sin(pitch_fwd) * cos(tilt)
            thrust_fwd - wing_drag(max(vx_air, 0.1), pitch, tilt, alt) * ge_d - f_body
        end
    end

    Fz = if MANUAL
        thrust_act * cos(tilt) * ge_r +
            wing_lift(max(vx_air, 0.0), pitch, tilt, alt) * ge_l - WEIGHT_N
    elseif τ < 0.0
        # Hover climb: use thrust directly without ground effect multiplier.
        # ge_r at 1.22m creates a false equilibrium where the aircraft floats
        # in ground effect rather than climbing to hover_alt_m.
        # ctrl_hover already commands the correct thrust to reach hover_alt_m;
        # applying ge_r on top creates a locally-stable equilibrium below it.
        thrust_act - WEIGHT_N
    elseif τ < T.T_TRANS_END
        thrust_act * cos(tilt) + wing_lift(vx_air, pitch, tilt, alt) * ge_l - WEIGHT_N
    elseif τ < T.T_FW_CLIMB
        # fw_climb: altitude P-controller drives to dash_altitude_m
        clamp((TC.dash_altitude_m - alt) * CP.alt_wn^2 * AP.mass_kg,
              -WEIGHT_N, WEIGHT_N)
    elseif !_DESCENT_ARMED[]
        # dash: altitude P-controller holds dash_altitude_m.
        (TC.dash_altitude_m - alt) * CP.alt_wn^2 * AP.mass_kg * 0.1
    else
        # fw_descent, back_transition, descent:
        # Ground effect must use height above ACTUAL terrain, not the ODE
        # departure datum. When alt_ode≈0 but the aircraft is over KSAF
        # (619m below KAXX), ge_r(alt_ode) = 1.085 — falsely boosting
        # thrust and creating a hover equilibrium at alt_ode=0.86m that
        # prevents descent to KSAF elevation.
        _ge_terrain  = Float64(alt) - terrain_alt(TERRAIN, Float64(x), Float64(y))
        ge_r_terrain = rotor_ge(max(_ge_terrain, 0.0))
        thrust_act * cos(tilt) * ge_r_terrain +
            wing_lift(max(vx_air, 0.0), pitch, tilt, alt) * ge_l - WEIGHT_N
    end

    Fz_gust = AP.mass_kg * ww

    # ── Landing gear ──────────────────────────────────────────────────
    # terrain_alt(TERRAIN, x) returns elevation delta from origin (m).
    # Subtracting it from alt_ode gives AGL above local terrain.
    # The _DESCENT_ARMED gate is preserved — gear spring only active
    # post-transition; during departure alt_ode IS the AGL.
    alt_gear = _DESCENT_ARMED[] && τ >= TIMINGS.T_TRANS_END ?
               Float64(alt) - terrain_alt(TERRAIN, Float64(x), Float64(y)) :
               Float64(alt)
    Fz_gear  = contact_spring(GEAR, alt_gear)
    Fx_gear  = ifelse(contact_active(GEAR, alt_gear), ground_friction(GEAR, alt_gear, vx), zero(vx))
    Fx_brake = ifelse(contact_active(GEAR, alt_gear) && ctrl.brakes != 0,
                      -sign(vx) * WEIGHT_N * 0.6, zero(vx))

    # ── Kinematics ────────────────────────────────────────────────────
    dvx  = (Fx + Fx_gear + Fx_brake) / AP.mass_kg
    dalt = (Fz + Fz_gust + Fz_gear) / AP.mass_kg
    _VZ_CACHE[] = Float64(dalt)   # cache for VRS gating on next step

    # ── Yaw disturbance (AUTO only — asymmetric drag proxy) ───────────
    yaw_dist = MANUAL ? 0.0 : 0.0003 * Fx / AP.mass_kg

    # ── Navigation heading guidance (AUTO, while above back-transition speed) ─
    # Nav guidance active at all speeds post-transition.
    # Removed vx > 35 m/s gate — aircraft needs yaw-toward-target
    # during back-transition and descent to avoid landing short.
    if !MANUAL && τ >= TIMINGS.T_TRANS_END && !_DESCENT_ARMED[]
        _ng = nav_guidance(NAV, x, y, yaw)
        yaw_rate_target = _ng.delta_yaw_rad * MAX_YAW_RATE
    end

    # ── Rotor moments ─────────────────────────────────────────────────
    ct        = cos(tilt)
    ct_eff    = MANUAL ? 1.0 : ct
    M_x_rotor = ALLOC.roll_moment_scale  * 3.0 * (roll_cmd        - roll)  * ct_eff
    _pitch_cmd_s = clamp(pitch_cmd, -0.524, 0.524)
    M_y_rotor = ALLOC.pitch_moment_scale * (3.0 * (_pitch_cmd_s - pitch) - 2.0 * ωy) * ct_eff
    M_z_rotor = ALLOC.yaw_moment_scale   * 3.0 * (yaw_rate_target - ωz)    * ct_eff

    # ── Aerodynamic moments ───────────────────────────────────────────
    My_aero = wing_pitch_moment(vx_air, pitch, tilt, alt)
    Mx_aero = dihedral_roll_moment(roll, vx_air, alt)
    My_damp = pitch_damping_moment(ωy, vx_air, tilt, alt)
    Mz_damp = yaw_damping_moment(ωz, vx_air, tilt, alt)
    Mz_aero = yaw_dist * Izz
    Mx_damp = -ALLOC.roll_moment_scale * 0.3 * ωx * ct_eff

    M_x = M_x_rotor + Mx_aero + Mx_damp
    M_y = M_y_rotor + My_aero + My_damp
    M_z = M_z_rotor + Mz_aero + Mz_damp

    # ── Euler equations ───────────────────────────────────────────────
    dωx = (M_x - (Iyy - Izz) * ωy * ωz) / Ixx
    dωy = (M_y - (Izz - Ixx) * ωz * ωx) / Iyy
    dωz = (M_z - (Ixx - Iyy) * ωx * ωy) / Izz

    # ── Battery ───────────────────────────────────────────────────────
    # BEM_POWER_KW_CACHE is updated by the saving callback (~10 Hz) from
    # allocate_wrench_vx — geometry-aware, sees actual FLEET radius/chord.
    # rotor_power_kw() is geometry-blind and would ignore rotor config changes.
    power_kw = BEM_POWER_KW_CACHE[]
    dsoc     = soc_derivative(soc, power_kw)

    # ── Mission clock ─────────────────────────────────────────────────
    dτ = MANUAL ? zero(τ) : tau_derivative(τ, alt, TC.hover_alt_m)

    # ── Ground track ──────────────────────────────────────────────────
    dx = vx * cos(yaw)
    dy = vx * sin(yaw)

    du[1]=dvx;      du[2]=dalt
    du[3]=dtilt;    du[4]=ddtilt
    du[5]=ωy;       du[6]=dωy
    du[7]=ωx;       du[8]=dωx
    du[9]=ωz;       du[10]=dωz
    du[11]=dthrust; du[12]=dsoc;  du[13]=dτ
    du[14]=dx;      du[15]=dy
    du[16]=dωx;     du[17]=dωy;   du[18]=dωz
end

# ══════════════════════════════════════════════════════════════════════
#  SHARED STATE (GUI ↔ solver)
# ══════════════════════════════════════════════════════════════════════
# fleet_fuel_state() here reads FLEET's tanks before any burn has happened,
# so this is the as-loaded fuel mass (FuelTank's initial_L, not necessarily
# full capacity) — the correct starting point for the cockpit fuel gauge.
# (Kept as a single tuple const, not `const a, b = ...` destructuring —
# `const` does not support multi-target tuple assignment in Julia.)
const _INIT_FUEL = fleet_fuel_state()   # (fuel_kg, fuel_capacity_kg)
const _cockpit_state = SHOW_GUI ?
    CockpitState(n_rotors=6, labels=_ROTOR_LABELS,
                 powerplants=collect(_ROTOR_POWERPLANT_LABELS),
                 fuel_kg=_INIT_FUEL[1], fuel_capacity_kg=_INIT_FUEL[2]) : nothing
const _state_obs     = SHOW_GUI ? Observable(_cockpit_state) : nothing
const _csv_io = Ref{Union{IOStream,Nothing}}(nothing)
const _HIST_MAX      = 600

# ══════════════════════════════════════════════════════════════════════
#  CSV WRITER
# ══════════════════════════════════════════════════════════════════════
function start_csv_writer(path::String)
    f = open(path, "w")
    println(f,
        "timestamp_s,tau_s,phase,speed_kmh,altitude_msl_ft,power_kw,vrs_factor," *
        "tilt_deg,soc_pct,voltage_v,batt_temp_c," *
        "x_m,y_m,alt_agl_m,alt_agl_terrain_m,omega_x_rads,omega_y_rads,omega_z_rads," *
        "gx,gy,gz," *
        "turb_u_ms,turb_v_ms,turb_w_ms," *
        "gear_contact,strut_load_n," *
        "rpm_r1,rpm_r2,rpm_r3,rpm_r4,rpm_r5,rpm_r6," *
        "kw_r1,kw_r2,kw_r3,kw_r4,kw_r5,kw_r6," *
        "q0,q1,q2,q3," *
        "powerplant_r1,powerplant_r2,powerplant_r3,powerplant_r4,powerplant_r5,powerplant_r6," *
        "fuel_kg,fuel_capacity_kg")
    _csv_io[] = f
end

function write_csv_row(row)
    f = _csv_io[]
    f === nothing && return
    @printf(f, "%.2f,%.2f,%s,%.2f,%.1f,%.1f,%.3f,%.1f,%.2f,%.1f,%.1f,%.1f,%.1f,%.2f,%.1f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%d,%.1f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.6f,%.6f,%.6f,%.6f,%s,%s,%s,%s,%s,%s,%.3f,%.3f\n",
        row.t, row.tau, row.phase, row.speed, row.alt_msl_ft, row.power, row.vrs_factor,
        row.tilt,
        row.soc, row.voltage, row.batt_temp,
        row.x_m, row.y_m, row.alt_agl_m, row.alt_agl_terrain_m,
        row.omega_x, row.omega_y, row.omega_z,
        row.gx, row.gy, row.gz,
        row.turb_u, row.turb_v, row.turb_w,
        row.gear_contact, row.strut_load_n,
        row.rpm[1], row.rpm[2], row.rpm[3],
        row.rpm[4], row.rpm[5], row.rpm[6],
        row.kw[1],  row.kw[2],  row.kw[3],
        row.kw[4],  row.kw[5],  row.kw[6],
        row.q0, row.q1, row.q2, row.q3,
        row.powerplant[1], row.powerplant[2], row.powerplant[3],
        row.powerplant[4], row.powerplant[5], row.powerplant[6],
        row.fuel_kg, row.fuel_capacity_kg)
    flush(f)
end

function close_csv_writer()
    f = _csv_io[]
    if f !== nothing
        close(f)
        _csv_io[] = nothing
    end
end

# ══════════════════════════════════════════════════════════════════════
#  SAVING CALLBACK
# ══════════════════════════════════════════════════════════════════════
function make_saving_cb(rt_factor::Float64, dt_ref::Ref{Float64})
    t_wall_start   = Ref(time())
    t_sim_start    = Ref(0.0)
    first_call     = Ref(true)
    t_prev         = Ref(0.0)
    t_last_print   = Ref(-10.0)
    t_last_dryden  = Ref(-1.0)   # DRYDEN: tracks last noise resample time

    SavingCallback(
        (u, t, integrator) -> begin
            τ = u[13]

            # ── Dryden noise resample (DRYDEN) ────────────────────────
            # Saving callback fires at 10 Hz; resample noise at DRYDEN_DT_S
            # intervals. Kept here to avoid PeriodicCallback tstop conflicts.
            if ATM.wind.turbulence_intensity > 0.0 &&
               t - t_last_dryden[] >= DRYDEN_DT_S
                ATM.dryden_noise .= randn(ATM.dryden_rng, 3)
                t_last_dryden[] = t
            end

            if !first_call[]
                dt_ref[] = max(t - t_prev[], 1e-4)
            end
            t_prev[] = t

            if first_call[]
                t_wall_start[] = time()
                t_sim_start[]  = t
                first_call[]   = false
            else
                wait_s = (t - t_sim_start[]) / rt_factor - (time() - t_wall_start[])
                wait_s > 0.001 && sleep(wait_s)
            end

            vx=u[1]; alt=u[2]; tilt=u[3]; pitch=u[5]
            roll=u[7]; yaw=u[9]; soc=u[12]
            x_m=u[14]; y_m=u[15]
            ωx_f=Float64(u[16]); ωy_f=Float64(u[17]); ωz_f=Float64(u[18])

            cb_dt          = max(t - _t_cb_ref[], 1e-4)
            _t_cb_ref[]    = t
            alt_f_prev     = _alt_prev_cb[]
            _alt_prev_cb[] = Float64(alt)

            # ── Autopilot cache update ─────────────────────────────────
            if !MANUAL
                # Arm descent by range — autopilot.cpp's ctrl_descent takes
                # over internally once armed; no separate autoland.so needed.
                if !_DESCENT_ARMED[] && Float64(τ) >= TIMINGS.T_TRANS_END
                    _rng_now = sqrt((Float64(x_m)-Float64(AP_CFG.wp_x))^2 +
                                   (Float64(y_m)-Float64(AP_CFG.wp_y))^2)
                    if _rng_now < AUTOLAND_RNG_M
                        _DESCENT_ARMED[] = true
                        @warn "[AP] Descent armed at rng=$(round(_rng_now,digits=0))m  t=$(round(t,digits=1))s"
                    end
                end
                # Always pass terrain AGL as s[19] so ctrl_descent FLARE
                # uses real terrain height regardless of airport elevation.
                # Slice to the original 18 core states before appending so
                # the C++ interface position of s[19] is stable even when
                # the ODE state vector is extended (e.g. Dryden states 20–22).
                _agl_t = Float64(alt) - terrain_alt(TERRAIN, Float64(x_m), Float64(y_m))
                u_f64  = vcat(Float64.(u[1:18]), _agl_t)
                new_ctrl = compute_controls(u_f64, AP_CFG_REF)
                AP_CTRL_CACHE[] = new_ctrl
                if new_ctrl.autopilot == 0 && !AP_HANDOFF[]
                    Threads.atomic_xchg!(AP_HANDOFF, true)
                    @warn "Autopilot handoff at t=$(round(t,digits=1))s — terminating."
                    terminate!(integrator)
                    return nothing
                end
            end

            # ── Landing gear (full model) ─────────────────────────────
            _cb_use_dest = _DESCENT_ARMED[] && Float64(τ) >= TIMINGS.T_TRANS_END
            _alt_gear    = _cb_use_dest ? Float64(alt) - terrain_alt(TERRAIN, Float64(x_m), Float64(y_m)) : Float64(alt)
            _alt_g_prev  = _cb_use_dest ? alt_f_prev   - terrain_alt(TERRAIN, Float64(x_m), Float64(y_m)) : alt_f_prev
            Fz_g, Fx_g, gear_on, strut_n =
                contact_full(GEAR, _alt_gear, _alt_g_prev, cb_dt, Float64(vx))
            if gear_on && _DESCENT_ARMED[] && !_LANDED[]
                _LANDED[] = true
                @info "[AP] ✅ LANDED at t=$(round(t,digits=1))s"
                # Terminate the solver — gear contact is the authoritative
                # landed signal. autopilot.cpp also sets autopilot=false at
                # touchdown, but gear_on is more reliable since it doesn't
                # depend on TOUCHDOWN_ALT_M matching cg_to_ground_m exactly.
                if !AP_HANDOFF[]
                    Threads.atomic_xchg!(AP_HANDOFF, true)
                    terminate!(integrator)
                    return nothing
                end
            end

            speed_kmh  = vx * 3.6
            alt_msl_ft = agl_m_to_msl_ft(alt)
            _raw_phase = if MANUAL
                HOTAS.tilt_frac[] >= 0.5 ? "manual-cruise" : "manual-hover"
            elseif gear_on
                "landed"
            elseif Float64(τ) < 0.0
                Float64(alt) > 2.0 ? "hover" : "landed"
            elseif Float64(τ) < TIMINGS.T_TRANS_END
                "transition"
            elseif !_DESCENT_ARMED[]
                Float64(tilt) > deg2rad(30.0) &&
                    Float64(alt) < TC.dash_altitude_m * 0.95 ? "fw_climb" : "dash"
            else
                if Float64(tilt) > deg2rad(30.0); "fw_descent"
                elseif abs(Float64(vx)) > 8.0;   "back_transition"
                else;                             "descent"
                end
            end
            _phase_base = if _raw_phase != _last_phase_cb[] &&
                       t - _last_phase_t[] >= PHASE_DWELL_S
                _last_phase_cb[] = _raw_phase
                _last_phase_t[]  = t
                _raw_phase
            else
                _last_phase_cb[]
            end
            phase = _DESCENT_ARMED[] ? "AUTOLAND:" * _phase_base : _phase_base

            if t - t_last_print[] >= 10.0
                t_last_print[] = t
                _agl_ft = (Float64(alt) - terrain_alt(TERRAIN, Float64(x_m), Float64(y_m))) * 3.28084
                _rng_pr = sqrt((Float64(x_m)-Float64(AP_CFG.wp_x))^2+(Float64(y_m)-Float64(AP_CFG.wp_y))^2)
                @printf("[fly] t=%6.0fs  %-26s  spd=%5.1f km/h  MSL=%6.0f ft  AGL=%6.0f ft  rng=%6.0fm  SoC=%4.1f%%\n",
                    t, phase, speed_kmh, alt_msl_ft, _agl_ft, _rng_pr, Float64(soc)*100)
            end
            # ── Per-rotor quantities ──────────────────────────────────
            tilt_f = Float64(tilt)
            vx_f   = Float64(vx)
            ctrl_cb = MANUAL ? hotas_to_control_output() : AP_CTRL_CACHE[]

            # Zero all rotor output once on the ground — rotors are off
            landed = (phase == "landed") || (gear_on && τ > TIMINGS.T_TRANS_END)
            if landed
                rpm_each  = ntuple(_ -> 0.0, 6)
                kw_each   = ntuple(_ -> 0.0, 6)
                power_kw  = 0.0
            else
                T_w, Mx_w, My_w, Mz_w = build_wrench(u,
                    Float64(ctrl_cb.thrust_cmd),
                    Float64(ctrl_cb.pitch_cmd),
                    Float64(ctrl_cb.roll_cmd),
                    0.0, tilt_f)
                rpm_each, kw_each = allocate_wrench_vx(T_w, Mx_w, My_w, Mz_w,
                                        FLEET, tilt_f, vx_f, Float64(alt), ALLOC)
                power_kw  = sum(kw_each)
            end
            BEM_POWER_KW_CACHE[] = power_kw

            # ── Fuel burn (turbine/turbine-electric rotors only) ───────
            # No-op for all-electric fleets — fleet_fuel_burn! routes every
            # ElectricMotor rotor's kw to battery_kw_total (unused below;
            # battery draw is still computed from power_kw via
            # battery_current() as before — see TODO in fleet_fuel_burn!'s
            # docstring re: wiring battery_kw_total through that path).
            # fleet_fuel_state reads back FLEET's tank(s) immediately after,
            # so fuel_kg/fuel_capacity_kg below always reflect this step's
            # burn, not last step's.
            _battery_kw_routed, _fuel_kg_burned_step = fleet_fuel_burn!(kw_each, Float64(alt), cb_dt)
            fuel_kg, fuel_capacity_kg = fleet_fuel_state()

            current_a = battery_current(power_kw)
            voltage_v = terminal_voltage(soc, current_a)
            batt_temp = steady_state_temp(power_kw)

            # ── G-forces (body frame, load factor in g) ──────────────
            # gz: vertical load factor felt by occupants.
            #     = (rotor vertical + wing lift + gear reaction) / weight
            #     = 1.0 in steady level flight (thrust + lift = weight)
            #     > 1.0 at touchdown (gear reaction adds to thrust)
            #     Uses thrust_lag (ODE state 11) not ctrl_cb.thrust_cmd so
            #     the actual rotor thrust is used, not the cached command.
            # gx: longitudinal — net forward force / weight
            # gy: lateral — centripetal + gravity component from roll
            _g       = 9.80665
            _thr_act = Float64(u[11])   # thrust_lag — actual rotor thrust
            _ct      = cos(Float64(tilt))
            _st      = sin(Float64(tilt))
            _pitch_f = Float64(pitch)
            _roll_f  = Float64(roll)
            _wt      = AP.mass_kg * _g

            gz = if gear_on
                # On ground: load factor from actual damped gear force.
                # strut_n from contact_full uses finite-diff vz over the
                # callback interval — physically correct at 10 Hz since
                # the touchdown is resolved over multiple callback steps
                # once the aircraft is on the gear spring.
                strut_n / _wt
            else
                # Airborne: (rotor vertical + wing lift) / weight = 1.0 level
                (_thr_act * _ct +
                 wing_lift(max(vx_f, 0.0), _pitch_f, Float64(tilt), Float64(alt))) / _wt
            end

            gx = (_thr_act * _st - fuselage_drag(vx_f, Float64(alt))) / _wt

            gy = (vx_f * Float64(u[10]) + _g * sin(_roll_f)) / _g


            _cr=cos(Float64(roll)/2); _sr=sin(Float64(roll)/2)
            _cp=cos(Float64(pitch)/2); _sp=sin(Float64(pitch)/2)
            _cy=cos(Float64(yaw)/2);  _sy=sin(Float64(yaw)/2)
            _cb_vz = (Float64(alt) - alt_f_prev) / cb_dt
            write_csv_row((
                t=t, tau=τ, phase=phase, speed=speed_kmh, alt_msl_ft=alt_msl_ft,
                power=power_kw,
                vrs_factor=_cb_vz < 0.0 ?
                    vrs_factor(Float64(u[11]) / 6.0, _cb_vz,
                               rho(Float64(alt)), π * RP.radius_m^2) : 1.0,
                tilt=rad2deg(tilt),
                soc=soc*100, voltage=voltage_v, batt_temp=batt_temp,
                q0=_cr*_cp*_cy + _sr*_sp*_sy,
                q1=_sr*_cp*_cy - _cr*_sp*_sy,
                q2=_cr*_sp*_cy + _sr*_cp*_sy,
                q3=_cr*_cp*_sy - _sr*_sp*_cy,
                x_m=Float64(x_m), y_m=Float64(y_m), alt_agl_m=Float64(alt),
                alt_agl_terrain_m=Float64(alt) - terrain_alt(TERRAIN, Float64(x_m), Float64(y_m)),
                omega_x=ωx_f, omega_y=ωy_f, omega_z=ωz_f,
                gx=gx, gy=gy, gz=gz,
                turb_u=Float64(u[20]), turb_v=Float64(u[21]), turb_w=Float64(u[22]),
                gear_contact=gear_on, strut_load_n=strut_n,
                rpm=rpm_each .* (60.0 / 2π), kw=kw_each,
                powerplant=_ROTOR_POWERPLANT_LABELS,
                fuel_kg=fuel_kg, fuel_capacity_kg=fuel_capacity_kg))

            if SHOW_GUI
                s = _cockpit_state
                s.vals[IDX.t]        = t
                s.vals[IDX.tau]      = τ
                s.vals[IDX.speed]    = speed_kmh
                s.vals[IDX.alt]      = alt_msl_ft
                s.vals[IDX.power]    = power_kw
                s.vals[IDX.tilt]     = rad2deg(tilt)
                s.vals[IDX.pitch]    = rad2deg(pitch)
                s.vals[IDX.roll]     = rad2deg(roll)
                s.vals[IDX.yaw]      = rad2deg(yaw)
                s.vals[IDX.soc]      = soc * 100.0
                s.vals[IDX.voltage]  = voltage_v
                s.vals[IDX.batt_temp]= batt_temp
                s.vals[IDX.x_m]      = Float64(x_m)
                s.vals[IDX.y_m]      = Float64(y_m)
                s.vals[IDX.omega_x]  = ωx_f
                s.vals[IDX.omega_y]  = ωy_f
                s.vals[IDX.omega_z]  = ωz_f
                s.vals[IDX.alt_agl_m]= Float64(alt)
                s.gear_contact       = gear_on
                s.strut_load_n       = strut_n
                s.brakes_on          = (ctrl_cb.brakes != 0)
                s.vals[IDX.gx]       = gx
                s.vals[IDX.gy]       = gy
                s.vals[IDX.gz]       = gz
                s.phase              = phase
                s.fuel_kg            = fuel_kg
                for i in 1:6
                    s.rotor_rpm[i] = rpm_each[i] * (60.0 / 2π)
                    s.rotor_kw[i]  = kw_each[i]
                end
                push!(s.history_power, power_kw)
                length(s.history_power) > _HIST_MAX && popfirst!(s.history_power)
                _state_obs[].vals[1] = t
            end

            # ── Navigation track update ───────────────────────────────
            nav_push!(NAV_MAP, u, t, phase)

            # ── Ground termination guard ──────────────────────────────
            _cb_use_dest2 = _DESCENT_ARMED[] && Float64(τ) > TIMINGS.T_TRANS_END
            _alt_term     = _cb_use_dest2 ? Float64(alt) - terrain_alt(TERRAIN, Float64(x_m), Float64(y_m)) : Float64(alt)
            if gear_on && τ > TIMINGS.T_TRANS_END && abs(Float64(vx)) < 2.0 &&
               _alt_term < GEAR.cg_to_ground_m + 2.0
                terminate!(integrator)
            end

            return nothing
        end,
        SavedValues(Float64, Nothing),
        saveat=0.1)
end

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
function main()
    println("="^60)
    println("eVTOL Tiltrotor — fly.jl v0.2.0")
    @printf("Mode: %s | GUI: %s | Speed: %.1fx\n",
            MANUAL ? "MANUAL (HOTAS)" : "AUTO (autopilot.so)",
            SHOW_GUI ? "ON" : "OFF", RT_FACTOR)

    # Mission profile
    if NAV.return_to_base
        @printf("Mission: %.0f km/h | %.0f ft AGL | RTB → [0, 0]\n",
                TC.dash_speed_kmh, TC.dash_altitude_m * M_TO_FT)
    else
        @printf("Mission: %.0f km/h | %.0f ft AGL | FLY-TO [%.0f, %.0f] m  Range %.0f m  Brg %.1f°  Arm %.0f m\n",
                TC.dash_speed_kmh, TC.dash_altitude_m * M_TO_FT,
                NAV.x_m, NAV.y_m, hypot(NAV.x_m, NAV.y_m),
                nav_bearing(0.0, 0.0, NAV.x_m, NAV.y_m),
                DESCENT_INITIATION_M)
    end

    # Airport / environment
    @printf("Departure: %s  %.0f ft MSL | %.1f°C | Wind %.1f kt from %03.0f°\n",
            ATM.airport_icao, ATM.airport_alt_m * M_TO_FT,
            ATM.ambient_temp_c,
            TC.wind_speed_ms * 1.94384,
            TC.wind_from_deg)
    if TC.dest_icao != ""
        @printf("Arrival:   %s  %.0f ft MSL | %.1f°C\n",
                TC.dest_icao, TC.dest_alt_m * M_TO_FT, TC.dest_temp_c)
    end

    # Rotor fleet — suppress detail when all rotors are stock defaults
    all_stock = all(FLEET.units[i].label == "R$i" for i in 1:6)
    @printf("Rotors: %d × %.2fm | Hover thrust %.0f N | Spool τ %.2fs | Battery %.0f kWh\n",
            n_rotors(FLEET), FLEET.units[1].radius_m, RP.hover_thrust_N,
            TAU_SPOOL, BP.capacity_kwh)
    if !all_stock
        for i in 1:6
            u = FLEET.units[i]
            u.label != "R$i" &&
                @printf("  [%s] R=%.2fm  kT=%.4f  η=%.2f  spin=%+d  ← PROTOTYPE\n",
                        u.label, u.radius_m, u.kT, u.eta_rotor, u.spin_dir)
        end
    end
    println("="^60)

    start_csv_writer(OUT_CSV)
    println("Streaming to $OUT_CSV")

    fig = nothing
    if SHOW_GUI
        fig = launch_cockpit(_state_obs; nav_map=NAV_MAP)
        println("Cockpit window open.")
    end

    MANUAL && Threads.nthreads() == 1 &&
        @warn "Single thread — HOTAS may not update during solve. Use: julia --threads auto"

    MANUAL && begin
        start_hotas()
        if HOTAS.connected[]
            println("HOTAS ready:")
            println("  Stick: roll/pitch | Twist: yaw | Throttle: thrust")
            println("  Rocker: left=HOVER right=CRUISE | Btn 4/5=trim↑↓ | Btn 7=brakes")
        end
    end

    # ── Initial conditions ─────────────────────────────────────────────
    u0_base = [0.0,                      # 1  vx
          GEAR.cg_to_ground_m,      # 2  alt AGL
          0.0,                      # 3  tilt
          0.0,                      # 4  dtilt
          0.0,                      # 5  pitch
          0.0,                      # 6  dpitch
          0.0,                      # 7  roll
          0.0,                      # 8  droll
          deg2rad(TC.initial_heading_deg), # 9  yaw
          0.0,                      # 10 dyaw
          MANUAL ? 0.0 : WEIGHT_N,  # 11 thrust_lag
          BP.soc_init,              # 12 soc
          -TC.preflight_hold_s,     # 13 τ
          0.0,                      # 14 x
          0.0,                      # 15 y
          0.0,                      # 16 ωx
          0.0,                      # 17 ωy
          0.0]                      # 18 ωz
    # State 19: terrain AGL at departure (always 0.0 at origin by definition).
    # States 20–22: Dryden longitudinal/lateral/vertical gust components (m/s).
    terrain_agl_0 = 0.0   # departure point is the terrain datum
    u0 = vcat(u0_base, [terrain_agl_0], zeros(3))   # DRYDEN

    dt_ref = Ref(0.05)

    # Range-based tspan: cruise time from actual mission distance dominates.
    # Add generous margin for descent + destination elevation delta.
    # For lower destinations (z_m < 0), descent runs longer —
    # KAXX→KSAF needs ~290s extra to descend the additional 619m to KSAF.
    _mission_range  = hypot(nav_wx(NAV), nav_wy(NAV))
    _cruise_s       = _mission_range / (TC.dash_speed_kmh / 3.6)
    # Back-transition now translates toward target — add time for that.
    # DIM / 18 m/s = time to cover descent_initiation range at max vx_target.
    # abs(z_m) / 3.5 = time to descend elevation delta at nominal rate.
    _bt_trans_s     = Float64(DESCENT_INITIATION_M) / 18.0
    _elev_delta_s   = max(-Float64(NAV.z_m), 0.0) / 0.5   # autoland ~0.6 m/s descent rate
    _tspan_upper    = TC.preflight_hold_s +
                      TC.hover_alt_m / TC.climb_rate_ms +
                      TC.trans_duration_s +
                      (TC.dash_altitude_m - TC.hover_alt_m) / TC.climb_rate_fw_ms +
                      _cruise_s +
                      TIMINGS.fw_desc_duration + _bt_trans_s +
                      _elev_delta_s + 300.0   # 5-min margin
    tspan  = (0.0, _tspan_upper)
    prob   = ODEProblem(build_ode, u0, tspan, dt_ref)

    saving_cb = make_saving_cb(RT_FACTOR, dt_ref)

    # ── Dryden noise resampler (DRYDEN) ───────────────────────────────
    # Resampling is handled inside make_saving_cb to avoid PeriodicCallback
    # tstop conflicts with terminate!(). No separate callback needed.
    dryden_cb = nothing  # noise resampled inside saving callback every DRYDEN_DT_S

    landed_cb = ContinuousCallback(
        (u, t, integrator) -> begin
            use_dest = _DESCENT_ARMED[] && u[13] > TIMINGS.T_TRANS_END
            alt_g    = use_dest ? Float64(u[2]) - terrain_alt(TERRAIN, Float64(u[14]), Float64(u[15])) : Float64(u[2])
            u[13] > TIMINGS.T_TRANS_END ? alt_g - (GEAR.cg_to_ground_m + 0.3) : 1.0
        end,
        integrator -> begin
            println("  Landed at t=$(round(integrator.t, digits=1))s")
            u   = integrator.u
            t_l = integrator.t
            vx_l  = Float64(u[1]);  alt_l   = Float64(u[2])
            tilt_l= Float64(u[3]);  pitch_l = Float64(u[5])
            roll_l= Float64(u[7]);  yaw_l   = Float64(u[9])
            soc_l = Float64(u[12]); τ_l     = Float64(u[13])
            x_l   = Float64(u[14]); y_l     = Float64(u[15])
            ωx_l  = Float64(u[16]); ωy_l    = Float64(u[17]); ωz_l = Float64(u[18])
            _, _, _, strut_l = contact_full(GEAR, alt_l, _alt_prev_cb[], 0.1, vx_l)
            _cr_l=cos(Float64(roll_l)/2); _sr_l=sin(Float64(roll_l)/2)
            _cp_l=cos(Float64(pitch_l)/2); _sp_l=sin(Float64(pitch_l)/2)
            _cy_l=cos(Float64(yaw_l)/2);  _sy_l=sin(Float64(yaw_l)/2)
            _fuel_kg_l, _fuel_cap_l = fleet_fuel_state()
            write_csv_row((
                t=t_l, tau=τ_l, phase="landed", speed=vx_l*3.6,
                alt_msl_ft=agl_m_to_msl_ft(alt_l), power=0.0, vrs_factor=1.0,
                tilt=rad2deg(tilt_l),
                soc=soc_l*100, voltage=terminal_voltage(soc_l, 0.0),
                batt_temp=steady_state_temp(0.0),
                q0=_cr_l*_cp_l*_cy_l + _sr_l*_sp_l*_sy_l,
                q1=_sr_l*_cp_l*_cy_l - _cr_l*_sp_l*_sy_l,
                q2=_cr_l*_sp_l*_cy_l + _sr_l*_cp_l*_sy_l,
                q3=_cr_l*_cp_l*_sy_l - _sr_l*_sp_l*_cy_l,
                x_m=x_l, y_m=y_l, alt_agl_m=alt_l, alt_agl_terrain_m=0.0,
                omega_x=ωx_l, omega_y=ωy_l, omega_z=ωz_l,
                gx=0.0, gy=0.0, gz=strut_l/(AP.mass_kg*9.80665),
                turb_u=Float64(u[20]), turb_v=Float64(u[21]), turb_w=Float64(u[22]),
                gear_contact=true, strut_load_n=strut_l,
                rpm=ntuple(_->0.0, 6), kw=ntuple(_->0.0, 6),
                powerplant=_ROTOR_POWERPLANT_LABELS,
                fuel_kg=_fuel_kg_l, fuel_capacity_kg=_fuel_cap_l))
            if SHOW_GUI
                s = _cockpit_state
                s.gear_contact = true; s.strut_load_n = strut_l
                s.phase = "landed"; s.fuel_kg = _fuel_kg_l
                _state_obs[].vals[1] = t_l
            end
            terminate!(integrator)
        end, nothing)

    println("Solving...")
    @printf("  tspan=(0, %.0f)  cruise est %.0fs  u0: alt=%.2f thrust_lag=%.0f τ=%.2f\n",
            tspan[2], _cruise_s, u0[2], u0[11], u0[13])
    MANUAL && @printf("  HOTAS: thrust=%.3f tilt=%d pitch=%.3f roll=%.3f yaw=%.3f\n",
                      HOTAS.thrust_frac[], HOTAS.tilt_frac[]>=0.5 ? 1 : 0,
                      HOTAS.pitch_cmd[], HOTAS.roll_cmd[], HOTAS.yaw_cmd[])

    solver_task = Threads.@spawn solve(prob,
                      MANUAL ? Tsit5() : Rodas5P(autodiff=AutoFiniteDiff()),
                      abstol=1e-4, reltol=1e-4,
                      dtmax = MANUAL ? 0.02 : 0.5,
                      callback=CallbackSet(saving_cb, landed_cb))

    while !istaskdone(solver_task)
        SHOW_GUI && notify(_state_obs)
        sleep(0.05)
    end

    sol = fetch(solver_task)

    if AP_HANDOFF[]
        @printf("Solve ended on AP handoff at t=%.1f s\n", sol.t[end])
    else
        @printf("Done — %d steps, t_end=%.1f s\n", length(sol.t), sol.t[end])
    end

    close_csv_writer()

    if SHOW_GUI && fig !== nothing
        println("Simulation complete. Close cockpit window to exit.")
        # Retrieve the screen that was created when display(fig) was called.
        # Notify the observable on each tick so the cockpit stays live
        # (phase frozen at landing, instruments visible) until the user
        # closes the window.
        try
            scr = GLMakie.getscreen(fig.scene)
            while scr !== nothing && isopen(scr)
                notify(_state_obs)
                sleep(0.1)
            end
        catch
            # Fallback: sleep briefly then exit cleanly
            sleep(2.0)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

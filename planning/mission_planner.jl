# mission_planner.jl:   Mission profile loader and phase scheduler
# AUTHOR:               DANIEL DESAI
# UPDATED:              2026-05-10
# VERSION:              0.1.0
#
#
# Public interface
#   TC            — NamedTuple with all mission parameters (replaces TestCard struct)
#   TIMINGS       — NamedTuple of phase boundary τ-values (T_DASH_END removed)
#   phase_label() — τ × position → String phase name
#   tau_derivative() — sigmoid activation for mission clock
#
# Read test_card.json and return a NamedTuple with the same field names
# as the old TestCard struct so fly.jl references (TC.hover_alt_m etc.)
# compile without modification.

function load_mission(json_path::String)
    isfile(json_path) || error("mission_planner: $json_path not found")
    raw = JSON.parsefile(json_path)

    pre = get(raw, "preflight",   Dict{String,Any}())
    hov = get(raw, "hover",       Dict{String,Any}())
    trn = get(raw, "transition",  Dict{String,Any}())
    fw  = get(raw, "fixed_wing",  Dict{String,Any}())
    lnd = get(raw, "landing",     Dict{String,Any}())
    apt = get(raw, "airport",     Dict{String,Any}())
    dst = get(raw, "destination", Dict{String,Any}())

    g(d, k, def) = Float64(get(d, k, def))
    gs(d, k, def) = String(get(d, k, def))

    return (
        # Preflight
        preflight_hold_s     = g(pre, "hold_s",              5.0),
        preflight_ramp_s     = g(pre, "ramp_s",              2.0),

        # Hover
        hover_alt_m          = g(hov, "alt_m",              30.0),
        climb_rate_ms        = g(hov, "climb_rate_ms",       3.0),

        # Transition
        trans_duration_s     = g(trn, "duration_s",         10.0),
        trans_thrust_comp    = g(trn, "thrust_comp",         0.5),

        # Fixed-wing  (dash_duration_s intentionally absent)
        dash_speed_kmh       = g(fw,  "dash_speed_kmh",    210.0),
        dash_altitude_m      = g(fw,  "dash_altitude_m",   300.0),
        climb_rate_fw_ms     = g(fw,  "climb_rate_fw_ms",    5.0),
        descent_rate_fw_ms   = g(fw,  "descent_rate_fw_ms",  3.0),

        # Landing
        land_pitch_up_deg    = g(lnd, "pitch_up_deg",       35.0),
        land_pitch_up_rate_s = g(lnd, "pitch_up_rate_s",    4.0),
        land_pitch_hold_s    = g(lnd, "pitch_hold_s",       10.0),
        land_pitch_down_s    = g(lnd, "pitch_down_s",        4.0),
        land_tilt_s          = g(lnd, "tilt_s",             12.0),
        land_thrust_comp     = g(lnd, "thrust_comp",         0.6),
        land_descent_rate_ms = g(lnd, "descent_rate_ms",    1.5),

        # Departure airport / environment
        airport_icao         = gs(apt, "icao",             "KDEN"),
        airport_alt_m        = g(apt,  "alt_m",           1656.0),
        ambient_temp_c       = g(apt,  "ambient_temp_c",    15.0),
        ambient_pressure_pa  = g(apt,  "ambient_pressure_pa", 101325.0),
        wind_from_deg        = g(apt,  "wind_from_deg",      0.0),
        wind_speed_ms        = g(apt,  "wind_speed_ms",      0.0),
        wind_u_ms            = -g(apt, "wind_speed_ms", 0.0) * cos(deg2rad(g(apt, "wind_from_deg", 0.0))),
        wind_v_ms            = -g(apt, "wind_speed_ms", 0.0) * sin(deg2rad(g(apt, "wind_from_deg", 0.0))),

        # Destination airport conditions (used during descent/landing phases)
        dest_icao            = gs(dst, "icao",             ""),
        dest_alt_m           = g(dst,  "alt_m",             0.0),
        dest_temp_c          = g(dst,  "ambient_temp_c",    15.0),
        dest_pressure_pa     = g(dst,  "ambient_pressure_pa", 101325.0),
        dest_wind_from_deg   = g(dst,  "wind_from_deg",      0.0),
        dest_wind_speed_ms   = g(dst,  "wind_speed_ms",      0.0),

        # Navigation
        initial_heading_deg  = Float64(get(get(raw, "navigation", Dict()), "initial_heading_deg", 0.0)),
        # target_hover_alt_m: hover altitude AGL in ODE frame at the destination.
        # = hover_alt_m + z_m (z_m negative if destination is lower than takeoff).
        # Used by APConfig so the descent controller targets the correct elevation.
        target_hover_alt_m   = Float64(get(hov, "alt_m", 30.0)) +
                               Float64(get(get(get(raw, "navigation", Dict()),
                                              "target", Dict()), "z_m", 0.0)),

        # Turbulence
        turbulence_intensity_ms = g(raw, "turbulence_intensity_ms", 0.0),
    )
end


#  Descent-initiation threshold
#     descent_initiation_range(tc) → Float64 (metres)
#
# Horizontal range at which the autopilot transitions dash → fw_descent.
# Mirrors the computation in autopilot.cpp get_phase() so Julia-side
# phase_label() agrees with the C++ phase enum.
#
# Derivation: the back-transition takes (pitch_up + pitch_hold + pitch_down
# + tilt) seconds while speed bleeds from dash_speed to zero.  Using half
# the dash speed as the mean gives a braking distance of:
#
#     0.5 × dash_speed_ms × back_trans_duration_s
#
# A 20 % margin is added so descent begins slightly before the aircraft is
# geometrically over the waypoint, giving the back-transition room to
# complete over the pad.

# descent_initiation_range(tc) → Float64 (metres)
#
# Horizontal range FROM THE TARGET at which descent is triggered.
# This is purely the braking distance needed to stop from fw_descent
# entry speed (~25 m/s) to hover — the back-transition deceleration leg.
# The fw_descent leg distance is not included because fw_descent begins
# wherever the aircraft happens to be when the trigger fires; only the
# back-transition needs to complete over the pad.
#
# The trigger fires when range_to_target < descent_initiation_range,
# so the aircraft arrives at the target with speed already reduced to
# back-transition entry speed.

function descent_initiation_range(tc)::Float64
    dash_ms      = tc.dash_speed_kmh / 3.6
    bt_entry_ms  = 25.0     # fw_descent decelerates to this before back-transition

    # fw_descent leg: decelerate from dash_speed to bt_entry_ms.
    # ctrl_fw_descent cuts thrust to ~0.35×weight; estimate average decel ~0.7 m/s².
    fw_desc_s    = (dash_ms - bt_entry_ms) / 0.7
    fw_desc_dist = 0.5 * (dash_ms + bt_entry_ms) * fw_desc_s

    # back-transition leg: bt_entry_ms → 0 at half-speed average
    back_trans_s = tc.land_pitch_up_rate_s +
                   tc.land_pitch_hold_s    +
                   tc.land_pitch_down_s    +
                   tc.land_tilt_s
    bt_dist = 0.5 * bt_entry_ms * back_trans_s

    return (fw_desc_dist + bt_dist) * 1.30   # 30% margin
end

# make_timings(tc) → NamedTuple
#
# Compute phase-boundary τ-values from the test card.
# T_DASH_END is no longer a fixed time — descent is range-triggered.
# T_FW_DESC is kept as a sentinel: it is set to a value that can never
# be reached by τ alone, so the τ-based branch in legacy code paths is
# never taken.  The actual dash→descent transition is owned by get_phase()
# in autopilot.cpp (range check) and phase_label() below (same check).

function make_timings(tc)
    T_TRANS_END  = tc.trans_duration_s
    T_FW_CLIMB   = T_TRANS_END + (tc.dash_altitude_m - tc.hover_alt_m) /
                                  tc.climb_rate_fw_ms

    # T_FW_DESC / T_PITCH_* are offsets from when the descent trigger
    # fires (call that t_desc_actual).  We compute their *durations*
    # here; fly.jl / autopilot.cpp add them to the actual trigger time.
    fw_desc_duration  = (tc.dash_altitude_m - tc.hover_alt_m) /
                         tc.descent_rate_fw_ms
    back_trans_duration = tc.land_pitch_up_rate_s +
                          tc.land_pitch_hold_s    +
                          tc.land_pitch_down_s

    # tspan_upper: safe upper bound for the ODE tspan.
    # Based on actual mission geometry: preflight + hover climb + transition +
    # fw_climb + cruise (range / speed) + descent sequence + generous margin.
    # NAV may not be defined yet at module load time, so use a generous
    # cruise proxy: max(2 × fw_climb_duration, 1800s) as the cruise ceiling.
    # fly.jl overrides this with the actual range once NAV is available.
    fw_climb_duration  = (tc.dash_altitude_m - tc.hover_alt_m) / tc.climb_rate_fw_ms
    fw_desc_duration   = (tc.dash_altitude_m - tc.hover_alt_m) / tc.descent_rate_fw_ms
    back_trans_s_total = tc.land_pitch_up_rate_s + tc.land_pitch_hold_s +
                         tc.land_pitch_down_s + tc.land_tilt_s
    worst_cruise_s     = max(2.0 * fw_climb_duration, 1800.0)
    tspan_upper_base   = tc.preflight_hold_s + tc.hover_alt_m / tc.climb_rate_ms +
                         tc.trans_duration_s + fw_climb_duration + worst_cruise_s +
                         fw_desc_duration + back_trans_s_total + 300.0

    T_FW_DESC_WORST    = T_FW_CLIMB + worst_cruise_s + fw_desc_duration
    T_PITCH_UP_WORST   = T_FW_DESC_WORST + tc.land_pitch_up_rate_s
    T_PITCH_HOLD_WORST = T_PITCH_UP_WORST  + tc.land_pitch_hold_s
    T_PITCH_DOWN_WORST = T_PITCH_HOLD_WORST + tc.land_pitch_down_s

    return (
        T_TRANS_END  = T_TRANS_END,
        T_FW_CLIMB   = T_FW_CLIMB,
        fw_desc_duration     = fw_desc_duration,
        back_trans_duration  = back_trans_s_total,
        T_FW_DESC    = T_FW_DESC_WORST,
        T_PITCH_UP   = T_PITCH_UP_WORST,
        T_PITCH_HOLD = T_PITCH_HOLD_WORST,
        T_PITCH_DOWN = T_PITCH_DOWN_WORST,
        T_TILT_START = T_FW_DESC_WORST + tc.land_pitch_up_rate_s,
        tspan_upper  = tspan_upper_base,   # overridden by fly.jl with range-based value
        land_tilt_s       = tc.land_tilt_s,
        hover_alt         = tc.hover_alt_m,
        dash_alt          = tc.dash_altitude_m,
        descent_rate      = tc.descent_rate_fw_ms,
        land_descent_rate = tc.land_descent_rate_ms,
    )
end


#  phase_label — range-triggered dash→fw_descent
#     phase_label(τ, t, x, y, nav; alt) → String
#
# Mirrors the get_phase() logic in autopilot.cpp exactly.
#
# Arguments
# ─────────
# - `τ`   — mission clock (ODE state 13)
# - `t`   — TIMINGS NamedTuple (defaults to module-level TIMINGS)
# - `x`   — ground-track x (ODE state 14, metres forward)
# - `y`   — ground-track y (ODE state 15, metres rightward)
# - `nav` — NavTarget (defaults to module-level NAV)
# - `alt` — altitude AGL (m); keyword, defaults to Inf (airborne)
#
# The dash→fw_descent transition fires when horizontal range to the active
# waypoint falls below `DESCENT_INITIATION_M` — the same threshold
# computed in autopilot.cpp from the back-transition geometry.
#
# For CSV playback (no position available), pass `x=nothing` to fall back
# to the old τ-based T_FW_DESC sentinel, which will show \"dash\" until the
# worst-case τ is reached — conservative but harmless for post-analysis.

# Latching flag — set when descent is initiated, never reset within a run.
# Prevents reversion to dash if the aircraft overshoots the waypoint.
const _DESCENT_ARMED = Ref(false)

function phase_label(τ::Real,
                     t   = TIMINGS,
                     x::Union{Real,Nothing} = nothing,
                     y::Union{Real,Nothing} = nothing,
                     nav = isdefined(Main, :NAV) ? Main.NAV : nothing;
                     alt::Float64 = Inf,
                     vx::Float64  = 0.0)

    if τ < 0
        gear_alt = isdefined(Main, :GEAR) ? GEAR.cg_to_ground_m : 0.0
        return alt <= gear_alt + 0.05 ? "landed" : "hover"
    end

    τ < t.T_TRANS_END  && return "transition"
    τ < t.T_FW_CLIMB   && return "fw_climb"

    # Minimum dash time — enough to reach cruise speed and level off,
    # but capped to the time available before the descent trigger would
    # fire at the target range. For short missions the hardcoded 20s
    # would overshoot the waypoint before descent even arms.
    dash_speed_ms = Float64(TC.dash_speed_kmh) / 3.6
    if nav !== nothing
        wx = nav_wx(nav);  wy = nav_wy(nav)
        target_range = hypot(wx, wy)   # range from origin
        avail_dash_m = max(target_range - DESCENT_INITIATION_M, 0.0)
        min_dash_s   = clamp(avail_dash_m / dash_speed_ms, 5.0, 20.0)
    else
        min_dash_s = 20.0
    end
    τ < t.T_FW_CLIMB + min_dash_s && return "dash"

    # Latching range trigger — once armed, never reverts to dash
    if !_DESCENT_ARMED[]
        if x !== nothing && y !== nothing && nav !== nothing
            wx  = nav_wx(nav);  wy = nav_wy(nav)
            rng = hypot(Float64(x) - wx, Float64(y) - wy)
            if rng < DESCENT_INITIATION_M
                _DESCENT_ARMED[] = true
            end
        elseif τ >= t.T_FW_DESC
            _DESCENT_ARMED[] = true
        end
    end
    !_DESCENT_ARMED[] && return "dash"

    # Sub-phases after descent armed — mirrors autopilot.cpp
    vx > 25.0                                && return "fw_descent"
    alt > Float64(TC.hover_alt_m) + 5.0     && return "back_transition"
    return "descent"
end

# ── fw_descent label: needed by fly.jl saving callback independently ──
# phase_label returns "fw_descent" only when:
#   range < DESCENT_INITIATION_M  AND  τ < T_PITCH_DOWN + land_tilt_s
# The back_transition / descent split is still τ-based (unchanged).
# We patch the function to handle the fw_descent window:
# (Handled inside the in_dash branch above by negation — when in_dash
# is false we fall through to the τ-based back_transition / descent
# branches, which correctly cover fw_descent implicitly via T_PITCH_DOWN.)
# NOTE: the window between range-trigger and T_PITCH_DOWN is fw_descent.
# This is already correct: once in_dash=false, τ < T_PITCH_DOWN+land_tilt_s
# catches both fw_descent and back_transition.  To distinguish them we
# need the τ offset from when descent was initiated.  autopilot.cpp
# handles this via its own internal state; here we use T_FW_DESC sentinel.
# For phase_label correctness the distinction is: the saving callback in
# fly.jl should pass actual x/y/nav so autopilot.cpp and phase_label agree.


function tau_derivative(τ, alt, hover_alt_m)
    # τ advances when the aircraft reaches hover_alt_m.
    # The preflight hold is enforced by the t-based thrust ramp in fly.jl
    # (thrust < weight until t=ramp_t_start), so the aircraft cannot reach
    # hover_alt_m before the hold expires — no need to gate on τ itself.
    # Using min(σ_alt, σ_tau) froze τ at -5 because σ_tau(-5)≈0 kept
    # the derivative near zero even after the aircraft reached hover altitude.
    σ_alt = 1.0 / (1.0 + exp(-10.0 * (alt - hover_alt_m)))
    return σ_alt
end


#  Module-level constants 

const _PLAN_DIR  = @__DIR__
const _CARD_PATH = joinpath(_PLAN_DIR, "test_card.json")

const TC      = load_mission(_CARD_PATH)
const TIMINGS = make_timings(TC)
const DESCENT_INITIATION_M = descent_initiation_range(TC)

# reset_mission_state() — call at the start of each run to clear
# latching flags that persist across Julia session re-runs.
function reset_mission_state()
    _DESCENT_ARMED[] = false
end
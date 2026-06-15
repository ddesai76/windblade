# actuators.jl:    Actuator model and controller tuning parameters
# AUTHOR:          DANIEL DESAI
# UPDATED:         2026-05-10
# VERSION:         0.1.0
#
#
# Provides:
#   ControllerParams (CP)  — actuator bandwidths + PID gain constants
#   actuator_accel         — 2nd-order spring-damper actuator model
#
# PIDState and sched_* functions are intentionally NOT here.
# PIDState is a transitional stub pending full wrench reconstruction
# from AP_CTRL_CACHE; it lives in flight_controller.jl until that work
# is complete, at which point flight_controller.jl is deleted entirely.
# sched_* will be folded into autopilot.cpp as C##++ phase controllers
# mature and wrench reconstruction switches to the cached ControlOutput.
#
# Load order: no dependencies — include before flight_controller.jl.


# ── Controller Parameters ─────────────────────────────────────────────
Base.@kwdef struct ControllerParams
    # Actuator bandwidths
    tilt_wn   :: Float64 = 1.2    # Tilt  natural frequency (rad/s)
    tilt_zeta :: Float64 = 0.8    # Tilt  damping ratio
    pitch_wn  :: Float64 = 2.5    # Pitch natural frequency (rad/s)
    pitch_zeta:: Float64 = 0.85
    roll_wn   :: Float64 = 2.0    # Roll  natural frequency (rad/s)
    roll_zeta :: Float64 = 0.85
    yaw_wn    :: Float64 = 1.0    # Yaw   natural frequency (rad/s)
    yaw_zeta  :: Float64 = 0.8
    alt_wn    :: Float64 = 0.4    # Altitude hold bandwidth (rad/s)

    # ── PID gains — speed (vx) channel ───────────────────────────────
    # Active in DASH phase ONLY (T_FW_CLIMB ≤ τ < T_DASH_END).
    pid_spd_kp    :: Float64 = 600.0
    pid_spd_ki    :: Float64 =  80.0
    pid_spd_kd    :: Float64 =  40.0
    pid_spd_ilim  :: Float64 = 3000.0
    pid_spd_outlim:: Float64 = 8000.0

    # ── PID gains — altitude (alt) channel ───────────────────────────
    pid_alt_kp   :: Float64 = 0.012
    pid_alt_ki   :: Float64 = 0.002
    pid_alt_kd   :: Float64 = 0.008
    pid_alt_ilim :: Float64 = 0.15

    # ── PID gains — crosswind (roll) channel ─────────────────────────
    pid_roll_kp   :: Float64 = 0.008
    pid_roll_ki   :: Float64 = 0.001
    pid_roll_ilim :: Float64 = 0.20

    # ── PID gains — hover velocity hold ──────────────────────────────
    pid_hover_vx_kp   :: Float64 = 0.012
    pid_hover_vx_ki   :: Float64 = 0.003
    pid_hover_vx_ilim :: Float64 = 0.12
    pid_hover_vx_lim  :: Float64 = 0.15

    # ── Climb thrust budget ───────────────────────────────────────────
    climb_thrust_frac :: Float64 = 0.60
end

const CP = ControllerParams()

# ── Actuator Model ─────────────────────────────────────────────────────
"""
    actuator_accel(x, dx, cmd, wn, zeta) → ddx

Second-order spring-damper actuator tracking `cmd`.
ForwardDiff-safe — no conditionals, no type coercions.

  ddx = wn² · (cmd − x) − 2·ζ·wn · dx
"""
function actuator_accel(x, dx, cmd, wn::Float64, zeta::Float64)
    return wn^2 * (cmd - x) - 2.0 * zeta * wn * dx
end
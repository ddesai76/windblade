# airframe.jl:     Airframe and aerodynamics subsystem
# AUTHOR:          DANIEL DESAI
# UPDATED:         2026-06-17
# VERSION:         0.1.1
#
#
# Coordinate convention:
#   x — forward (positive = forward)
#   z — vertical (positive = up)
#   Angles: pitch positive = nose up, tilt 0 = hover (rotors up),
#           tilt π/2 = cruise (rotors forward)
#
#
# Depends on: atmosphere.jl (rho), rotor_system.jl (rotor_ge)
# =====================================================================

Base.@kwdef struct AirframeParams
    mass_kg             :: Float64 = 2177.0   # MTOW (kg)
    wing_area_m2        :: Float64 = 15.3     # Reference wing area (m²)
    CD0                 :: Float64 = 0.025    # Zero-lift drag coefficient
    k_induced           :: Float64 = 0.040    # Oswald induced drag factor
    CL_alpha            :: Float64 = 5.5      # Lift curve slope (rad⁻¹)
    alpha_0             :: Float64 = -0.03    # Zero-lift AoA (rad)
    CL_max              :: Float64 =  1.60    # Stall CL
    CL_min              :: Float64 = -0.80    # Negative stall CL
    trans_power_deficit :: Float64 = 0.06     # Thrust deficit during transition
    wingspan_m          :: Float64 = 11.0     # Wingspan (m)
    rotor_radius_m      :: Float64 = 1.0      # Rotor radius (m)
    k_dihedral          :: Float64 = 0.08     # Dihedral roll restoring coefficient
    # ── Body / fuselage drag (hover wind response) ────────────────── [FIX-6]
    CD_body             :: Float64 = 0.7      # Bluff-body drag coefficient (fuselage)
    A_front_m2          :: Float64 = 5.0      # Frontal area (m²) — fuselage + nacelles
    # ── Pitch moment (6-DOF My_aero) ─────────────────────────────── [FIX-7]
    mean_chord_m        :: Float64 = 1.39     # S/b = 15.3/11.0 (m)
    CM0                 :: Float64 = 0.02     # Pitching moment at zero AoA (slight nose-up)
    CM_alpha            :: Float64 = -0.80    # Pitch stability derivative (rad⁻¹, negative = stable)
    # ── Rotary damping derivatives (6-DOF) ───────────────────────── [FIX-8]
    Cmq                 :: Float64 = -12.0    # Pitch damping derivative (rad⁻¹, must be negative)
    Cnr                 :: Float64 = -0.15    # Yaw damping derivative   (rad⁻¹, must be negative)
    # ── Conversion corridor (VCON) ────────────────────────────────── [FIX-9]
    vcon_hi_ias_kmh     :: Float64 = 165.0    # Structural upper bound at sea level (km/h IAS)
                                               # Vcon_lo is computed from wing lift — no constant needed.
end

const AP = AirframeParams()

const G        = 9.80665

"""
    weight_N(ap::AirframeParams = AP) → N

Aircraft weight in Newtons for the given AirframeParams instance.
Use this instead of `WEIGHT_N` anywhere mass can vary at runtime (e.g.
mission-variable fuel loading constructing a fresh `AirframeParams(mass_kg=...)`
at launch) — `WEIGHT_N` below is a module-load-time snapshot of the
*default* `AP.mass_kg` and does not update if a different `ap` is used.

NOTE (2026-06-17): the mission-variable construction described above is
not yet implemented anywhere in this codebase — mission_planner.jl has
no mass_kg/fuel field, and nothing currently builds a non-default
AirframeParams. AP.mass_kg is therefore always 2177.0 (the struct
default) regardless of mission. This function and the WEIGHT_N fix
below are forward-looking: correct now, and correct once that wiring
is added, but not yet exercised by any live mission-mass value. See
the turbine-electric roadmap, Section 06 task list and AC-13.
"""
weight_N(ap::AirframeParams = AP) = ap.mass_kg * G

# Retained for backward compatibility with any existing caller that
# references WEIGHT_N directly. New code, and any caller that needs to
# reflect a non-default AirframeParams (e.g. mission-variable fuel load),
# should call weight_N(ap) instead — see that function's docstring.
const WEIGHT_N = weight_N(AP)

# ── Body / Fuselage Drag ──────────────────────────────────────────────

"""
    fuselage_drag(vx_air, alt, ap) → N
Bluff-body drag on the fuselage + nacelles. Acts in all flight phases,
but is only significant in hover where it is the *only* drag force
(wing drag is negligible at near-zero airspeed).

Uses vx_air * |vx_air| to preserve sign: positive vx_air (headwind)
produces negative Fx (drag opposes motion). This form is also smooth
at vx_air = 0, which matters for ForwardDiff.
"""
function fuselage_drag(vx_air, alt, ap::AirframeParams=AP)
    ρ = rho(alt)
    return 0.5 * ρ * ap.CD_body * ap.A_front_m2 * vx_air * abs(vx_air)
end

# ── Angle of Attack ───────────────────────────────────────────────────

"""
    effective_alpha(pitch, tilt) → rad
Effective wing angle of attack, blended across transition.
"""
function effective_alpha(pitch::Real, tilt::Real)
    return pitch * (1.0 - 0.5 * sin(tilt))
end

"""
    cl_from_alpha(alpha, ap) → dimensionless
Linear lift curve with stall clamp.
"""
function cl_from_alpha(alpha::Real, ap::AirframeParams=AP)
    CL = ap.CL_alpha * (alpha - ap.alpha_0)
    return clamp(CL, ap.CL_min, ap.CL_max)
end

# ── Rotor-Wing Interference ───────────────────────────────────────────

"""
    rotor_wash_factor(tilt) → (0, 1]
Rotor downwash penalty on wing lift. Maximum (25%) at hover, zero at cruise.
"""
function rotor_wash_factor(tilt::Real)
    return 1.0 - 0.25 * cos(tilt)^2
end

# ── Wing Aerodynamics ─────────────────────────────────────────────────

"""
    wing_lift(vx, pitch, tilt, alt, ap) → N
Wing lift using AoA-dependent CL(α) with rotor downwash correction.
"""
function wing_lift(vx, pitch, tilt, alt, ap::AirframeParams=AP)
    ρ     = rho(alt)
    q     = 0.5 * ρ * max(vx, zero(vx))^2
    alpha = effective_alpha(pitch, tilt)
    CL    = cl_from_alpha(alpha, ap)
    wash  = rotor_wash_factor(tilt)
    return q * ap.wing_area_m2 * CL * wash
end

"""
    wing_drag(vx, pitch, tilt, alt, ap) → N
Total wing drag: zero-lift + induced drag using actual CL(α)².
"""
function wing_drag(vx, pitch, tilt, alt, ap::AirframeParams=AP)
    ρ     = rho(alt)
    q     = 0.5 * ρ * max(vx, 0.1)^2
    alpha = effective_alpha(pitch, tilt)
    CL    = cl_from_alpha(alpha, ap)
    CD    = ap.CD0 + ap.k_induced * CL^2
    return q * ap.wing_area_m2 * CD
end

"""
    back_transition_drag(vx, pitch_rad, tilt, alt, ap) → N
Increased drag during back-transition from pitching manoeuvre.
"""
function back_transition_drag(vx, pitch_rad, tilt, alt, ap::AirframeParams=AP)
    ρ      = rho(alt)
    q      = 0.5 * ρ * max(vx, 0.1)^2
    ge_d   = fw_ge_drag(alt, ap)
    alpha  = effective_alpha(pitch_rad, tilt)
    CL     = cl_from_alpha(alpha, ap)
    CD_eff = ap.CD0 + ap.k_induced * CL^2 + 1.2 * sin(pitch_rad)^2
    return q * ap.wing_area_m2 * CD_eff * ge_d
end

# ── Wing Ground Effect ────────────────────────────────────────────────

function fw_ge_lift(alt, ap::AirframeParams=AP)
    return 1.0 + 0.10 * exp(-2.0 * max(alt, zero(alt)) / ap.wingspan_m)
end

function fw_ge_drag(alt, ap::AirframeParams=AP)
    return 1.0 / fw_ge_lift(alt, ap)
end

# ── Dihedral Roll Moment ──────────────────────────────────────────────

"""
    dihedral_roll_moment(roll, vx, alt, ap) → N·m
Aerodynamic restoring moment opposing roll.
Wire into fly.jl: du[8] += dihedral_roll_moment(...) / Ixx
"""
function dihedral_roll_moment(roll, vx, alt, ap::AirframeParams=AP)
    ρ = rho(alt)
    q = 0.5 * ρ * max(vx, zero(vx))^2
    return -ap.k_dihedral * roll * q * ap.wing_area_m2 * ap.wingspan_m
end

# ── Aerodynamic Pitching Moment (6-DOF) ──────────────────────────────

"""
    wing_pitch_moment(vx, pitch, tilt, alt, ap) → N·m

Aerodynamic pitching moment about the CG. Positive = nose-up.

Uses a neutral-point stability model:
  CM(α) = CM0 + CM_alpha · α
  My_aero = q · S · c̄ · CM(α)

where q is dynamic pressure, S is wing area, c̄ is mean aerodynamic
chord, and α is effective AoA from effective_alpha().

CM_alpha < 0 gives pitch stability (positive AoA → nose-down restoring
moment). CM0 > 0 gives a slight nose-up trim at zero AoA typical of
cambered wings.

Blended by (1 - cos²(tilt)/2) so the wing loses pitching authority
smoothly in hover when dynamic pressure is negligible and the rotors
dominate pitch control. At cruise (tilt=π/2) the factor is 1.0; at
hover (tilt=0) it is 0.5, consistent with reduced wing effectiveness.

ForwardDiff-safe: no Float64 coercions on state variables.
"""
function wing_pitch_moment(vx, pitch, tilt, alt, ap::AirframeParams=AP)
    ρ     = rho(alt)
    q     = 0.5 * ρ * max(vx, zero(vx))^2
    alpha = effective_alpha(pitch, tilt)
    # Clamp alpha to the linear-lift range before computing CM.
    # Without this, a pitch angle past ±90° (which should never happen but
    # did in the tumbling case) feeds a huge unwrapped alpha into CM_alpha,
    # producing a destabilising moment that drives further tumbling.
    alpha_clamped = clamp(alpha, -0.5, 0.5)   # ≈ ±28°, well past stall
    CM    = ap.CM0 + ap.CM_alpha * alpha_clamped
    blend = 1.0 - 0.5 * cos(tilt)^2   # 0.5 at hover → 1.0 at cruise
    return q * ap.wing_area_m2 * ap.mean_chord_m * CM * blend
end

# ── Rotary Damping Moments (6-DOF) ───────────────────────────────────

"""
    pitch_damping_moment(ωy, vx, tilt, alt, ap) → N·m

Aerodynamic pitch damping. Opposes pitch rate ωy (body pitch rate, rad/s).
Scales with dynamic pressure so it is negligible in hover and fully active
in cruise — exactly complementing the rotor moment authority which fades
as cos(tilt) → 0.

  My_damp = q · S · c̄ · Cmq · (ωy · c̄ / (2 · vx))

Cmq is the pitch damping derivative (dimensionless, strongly negative for
stable aircraft). A value of -12 gives ~770 N·m at cruise ωy = 0.17 rad/s,
sufficient to halve the short-period oscillation amplitude per half-cycle.

ForwardDiff-safe: no Float64 coercions on state-derived arguments.
"""
function pitch_damping_moment(ωy, vx, tilt, alt, ap::AirframeParams=AP)
    v = max(vx, zero(vx))
    v_safe = max(v, 0.5)            # avoid division by zero in hover
    ρ = rho(alt)
    q = 0.5 * ρ * v^2
    q̂ = ωy * ap.mean_chord_m / (2.0 * v_safe)   # non-dimensional pitch rate
    blend = 1.0 - 0.5 * cos(tilt)^2              # matches wing_pitch_moment blend
    return q * ap.wing_area_m2 * ap.mean_chord_m * ap.Cmq * q̂ * blend
end

"""
    yaw_damping_moment(ωz, vx, tilt, alt, ap) → N·m

Aerodynamic yaw damping. Opposes yaw rate ωz (body yaw rate, rad/s).
Prevents ωz accumulation in cruise where rotor yaw authority is zero
(M_z_rotor = 0 when cos(tilt) = 0).

  Mz_damp = q · S · b · Cnr · (ωz · b / (2 · vx))

Cnr is the yaw damping derivative (dimensionless, negative for stable aircraft).
A value of -0.15 gives ~620 N·m at cruise ωz = 0.22 rad/s — enough to
oppose the accumulated yaw drift within a few seconds of it developing.

ForwardDiff-safe.
"""
function yaw_damping_moment(ωz, vx, tilt, alt, ap::AirframeParams=AP)
    v = max(vx, zero(vx))
    v_safe = max(v, 0.5)
    ρ = rho(alt)
    q = 0.5 * ρ * v^2
    r̂ = ωz * ap.wingspan_m / (2.0 * v_safe)     # non-dimensional yaw rate
    blend = 1.0 - 0.5 * cos(tilt)^2              # full at cruise, half at hover
    return q * ap.wing_area_m2 * ap.wingspan_m * ap.Cnr * r̂ * blend
end

# ── Conversion Corridor (VCON) ────────────────────────────────────────

"""
    vcon_limits(tilt_rad, alt_agl_m; ap, weight_n) →
        (lo_kmh::Float64, hi_kmh::Float64)

Computes the conversion corridor boundaries as functions of tilt angle
and density altitude. Both values are in km/h IAS (indicated airspeed —
i.e. referenced to sea-level density ρ₀ = 1.225 kg/m³).

Lower bound — Vcon_lo (wing-lift limited)
──────────────────────────────────────────
During tilt-forward the rotor vertical thrust component reduces as
cos(tilt). The wing must cover the deficit. The minimum safe IAS is the
speed at which wing lift at a nominal cruise angle of attack (α ≈ 2°)
equals the required wing load share:

    L_wing = ½·ρ·V_TAS²·S·CL ≥ W·(1 - cos(tilt))
    → V_TAS = √(2·W·(1-cos(tilt)) / (ρ·S·CL))

Converted to IAS: V_IAS = V_TAS · √(ρ/ρ₀)

This cancels ρ in the numerator and denominator, giving an IAS that is
independent of density altitude — which is correct: IAS is what the
airspeed indicator reads, and the lift equation in terms of IAS is
ρ-invariant. The altitude dependence enters only through the rotor share
term, which is tilt-only.

At tilt = 0 (hover) the wing share is zero → Vcon_lo = 0.
At tilt = π/2 (cruise) the wing must carry 100% of weight — but the
transition is complete by then, so Vcon_lo peaks at mid-transition.
The function clamps the output to [0, 140] km/h for sanity.

Upper bound — Vcon_hi (structural / rotor-authority limited)
────────────────────────────────────────────────────────────
At cruise (tilt = π/2) the rotors are in edgewise flow. Advancing blade
IAS = V + ωR, retreating = V − ωR. The oscillating load scales with V².
There is a structural limit on nacelle tilt as a function of airspeed.

Simple model: the full structural limit applies in hover (tilt = 0) and
degrades proportionally with tilt toward a minimum of about 60% at
tilt = π/2:

    Vcon_hi = vcon_hi_ias_kmh · max(cos(tilt/2)^0.5, 0.6)

At hover (tilt = 0):  factor = 1.0  → full limit (165 km/h default)
At mid-trans (45°):   factor = 0.92 → 151 km/h
At cruise (90°):      factor = 0.84 → 139 km/h  (but corridor inactive)

This ensures Vcon_hi stays above Vcon_lo across the transition range,
with at least 20 km/h margin enforced by clamping.

Note: both boundaries are IAS, so the cockpit display and comparison
against indicated airspeed (speed_kmh from the ODE's vx · 3.6, which
is TAS) is slightly inconsistent at high altitude. For an S4-class
operating below 400 m AGL the difference is < 5% and ignorable. A
future refinement could convert v[IDX.speed] to IAS using
`v[IDX.speed] * sqrt(rho(alt) / 1.225)` before passing to draw_vcon!.
"""
function vcon_limits(tilt_rad::Float64, alt_agl_m::Float64;
                     ap::AirframeParams = AP,
                     weight_n::Float64  = weight_N(ap)) ::
         NamedTuple{(:lo_kmh, :hi_kmh), Tuple{Float64, Float64}}

    # ── Lower bound: wing-lift share at this tilt angle ───────────────
    cos_t      = cos(tilt_rad)
    wing_share = max(1.0 - cos_t, 0.0) * weight_n   # N the wing must carry

    # CL at nominal cruise AoA (≈ 2°), accounting for rotor wash at this tilt
    alpha_cruise = 0.035 - ap.alpha_0               # ≈ 2° above zero-lift
    CL_cruise    = clamp(ap.CL_alpha * alpha_cruise, 0.1, ap.CL_max)
    wash         = rotor_wash_factor(tilt_rad)
    CL_eff       = CL_cruise * wash

    # IAS (m/s) from ½·ρ₀·V_IAS²·S·CL = wing_share  (ρ₀ = 1.225 kg/m³)
    rho_sl   = 1.225
    q_needed = wing_share / max(ap.wing_area_m2 * CL_eff, 0.1)
    V_lo_ms  = sqrt(2.0 * q_needed / rho_sl)
    V_lo_kmh = V_lo_ms * 3.6

    # ── Upper bound: structural / authority limit ─────────────────────
    # Degrades with tilt as edgewise blade loads increase.
    tilt_factor = max(sqrt(cos(tilt_rad / 2.0)), 0.60)
    V_hi_kmh    = ap.vcon_hi_ias_kmh * tilt_factor

    # Enforce minimum margin so the corridor never inverts
    V_lo_kmh = clamp(V_lo_kmh, 0.0, 140.0)
    V_hi_kmh = max(V_hi_kmh, V_lo_kmh + 20.0)

    return (lo_kmh = V_lo_kmh, hi_kmh = V_hi_kmh)
end

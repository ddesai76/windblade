# rotor_system.jl:    Powerplant top-level model
# AUTHOR:             DANIEL DESAI
# UPDATED:            2026-05-10
# VERSION:            0.1.0
#
# Motor swap patterns (all produce a valid RotorParams / RotorFleet):
# ──────────────────────────────────────────────────────────────────────
#   All-turboshaft:
#     RP.motor = TurboshaftEngine(P_sl_W=1_200_000.0)
#
#   All-hybrid:
#     RP.motor = HybridTurbineElectric(turbine=TurboshaftEngine(P_sl_W=400_000.0),
#                                       motor=ElectricMotor(P_max_W=250_000.0),
#                                       E_batt_J=3.6e8)
#
#   Mixed fleet — aft pair turboshaft:
#     ts = TurboshaftEngine(P_sl_W=600_000.0)
#     FLEET_TS = RotorFleet(Dict(5 => RotorUnit(_default_unit(5); motor=ts),
#                                6 => RotorUnit(_default_unit(6); motor=ts)))
#
# Depends on: blades.jl, atmosphere.jl


include("blades.jl")
using .Blades

# ── Single Rotor Unit ──────────────────────────────────────────────────
"""
    RotorUnit

Physical parameters for one rotor.  Two new fields vs Priority 2:
  blade_geom :: BladeGeometry  — geometry passed to blade_coefficients()
  motor      :: AbstractMotor  — power backend (electric / turboshaft / hybrid)

All other fields retained for backwards compatibility with rotor_mixer.jl and
the legacy kT/kQ momentum path.

Joby S4 spin convention (alternating, looking from above):
    R1(fwd-L, CCW)  R2(fwd-R, CW)
    R3(mid-L, CW)   R4(mid-R, CCW)
    R5(aft-L, CCW)  R6(aft-R, CW)
Opposite pairs cancel torque at equal RPM.
"""
Base.@kwdef struct RotorUnit
    id           :: Int            = 1
    label        :: String         = "R1"
    radius_m     :: Float64        = BG.R      # default = BEM geometry radius
    kT           :: Float64        = 0.3592    # legacy scalar (used if use_bem=false)
    kQ           :: Float64        = 0.0473    # legacy scalar
    omega_nom    :: Float64        = 130.9     # rad/s ≈ 1250 RPM (= 2π×1250/60)
    inertia      :: Float64        = 6.0
    eta_rotor    :: Float64        = 0.75
    k_induced    :: Float64        = 1.15
    c_profile    :: Float64        = 0.008
    arm_x_m      :: Float64        = 0.0
    arm_y_m      :: Float64        = 0.0
    arm_z_m      :: Float64        = 0.0
    spin_dir     :: Int            = 1
    blade_geom   :: BladeGeometry  = BG           # BEM blade geometry
    motor        :: AbstractMotor  = ElectricMotor(P_max_W=280_000.0)  # power backend
end

# ── Default unit constructors (S4-class geometry) ──────────────────────
function _default_unit(i::Int) :: RotorUnit
    arms = (
        ( 3.0,  4.5,  0.3,  1),   # R1 fwd-L  CCW
        ( 3.0, -4.5,  0.3, -1),   # R2 fwd-R  CW
        ( 0.0,  5.5,  0.1, -1),   # R3 mid-L  CW
        ( 0.0, -5.5,  0.1,  1),   # R4 mid-R  CCW
        (-3.0,  4.5,  0.3,  1),   # R5 aft-L  CCW
        (-3.0, -4.5,  0.3, -1),   # R6 aft-R  CW
    )
    x, y, z, spin = arms[i]
    RotorUnit(
        id         = i,
        label      = "R$i",
        radius_m   = BG.R,
        omega_nom  = 130.9,   # 1250 RPM
        arm_x_m    = x,
        arm_y_m    = y,
        arm_z_m    = z,
        spin_dir   = spin,
        blade_geom = BG,
        motor      = ElectricMotor(P_max_W=280_000.0),
    )
end

# ── Rotor Fleet ────────────────────────────────────────────────────────
"""
    RotorFleet

Six-rotor fleet with per-unit override capability.

Motor swap example — aft pair turboshaft:
```julia
ts = TurboshaftEngine(P_sl_W=600_000.0)
FLEET_TS = RotorFleet(Dict(5 => RotorUnit(_default_unit(5); motor=ts),
                           6 => RotorUnit(_default_unit(6); motor=ts)))
```
Blade geometry override example — prototype rotor on positions 1–2:
```julia
bg_proto = BladeGeometry(R=1.65, chord=0.13, pitch_offset_deg=4.0)
FLEET_P = RotorFleet(Dict(1 => RotorUnit(_default_unit(1); blade_geom=bg_proto),
                          2 => RotorUnit(_default_unit(2); blade_geom=bg_proto)))
```
"""
struct RotorFleet
    units :: NTuple{6, RotorUnit}
end

function RotorFleet(overrides::Dict{Int,RotorUnit} = Dict{Int,RotorUnit}())
    RotorFleet(ntuple(i -> get(overrides, i, _default_unit(i)), 6))
end

n_rotors(fleet::RotorFleet)    = 6
total_area(fleet::RotorFleet)  = sum(π * u.radius_m^2 for u in fleet.units)
mean_radius(fleet::RotorFleet) = sum(u.radius_m for u in fleet.units) / 6.0

# ── Legacy scalar interface (RP) ───────────────────────────────────────
"""
    RotorParams

Aggregate scalar rotor parameters.  New fields vs Priority 2:
  motor        :: AbstractMotor — active power backend (default: ElectricMotor)
  use_bem      :: Bool          — true = BEM path; false = legacy kT/momentum
  bem_rpm_ref  :: Float64       — RPM reference for BEM back-calculation
                                  (set by compute_da_correction)

ODE calls rotor_power_kw(vx, thrust, alt, rp) which dispatches on use_bem.
"""
Base.@kwdef mutable struct RotorParams
    n                  :: Int            = 6
    radius_m           :: Float64        = BG.R
    hover_thrust_N_sl  :: Float64        = 26_055.0   # BEM-derived: 6 × 4342 N at SL ISA
    da_correction      :: Float64        = 1.0
    hover_thrust_N     :: Float64        = 26_055.0   # = sl × correction
    max_thrust_N       :: Float64        = 28_820.0   # ≈ 1.107 × hover (existing ratio)
    kT                 :: Float64        = 0.3592     # legacy only (BEM path ignores this)
    omega0             :: Float64        = 130.9      # rad/s ≈ 1250 RPM
    inertia            :: Float64        = 6.0
    eta_rotor          :: Float64        = 0.75
    k_induced          :: Float64        = 1.15
    c_profile          :: Float64        = 0.008
    motor              :: AbstractMotor  = ElectricMotor(P_max_W=280_000.0)  # 280 kW — covers KAXX (9000 ft) T/W≥1.10
    use_bem            :: Bool           = true
    bem_rpm_ref        :: Float64        = 1250.0     # nominal RPM for BEM back-calc
end

# ── Rotor fleet: S4 defaults, with optional overrides from test_card.json ──
# test_flight.py / windblade.py write rotor geometry into test_card.json under
# "rotor_fleet" → "overrides". Edit rotor_config.csv and re-run to customise.
const FLEET = let
    _card_path = joinpath(dirname(dirname(@__DIR__)), "planning", "test_card.json")
    _ovrs = try
        get(get(Main.JSON.parsefile(_card_path), "rotor_fleet", Dict()), "overrides", [])
    catch
        []
    end
    if isempty(_ovrs)
        RotorFleet()
    else
        _od = Dict{Int,RotorUnit}()
        for entry in _ovrs
            id = Int(entry["rotor_id"])
            u  = _default_unit(id)
            bg = u.blade_geom
            new_bg = BladeGeometry(
                R                = get(entry, "R_m",               bg.R),
                n_blades         = get(entry, "n_blades",          bg.n_blades),
                chord            = get(entry, "chord_m",           bg.chord),
                twist_root_deg   = get(entry, "twist_root_deg",    bg.twist_root_deg),
                twist_tip_deg    = get(entry, "twist_tip_deg",     bg.twist_tip_deg),
                pitch_offset_deg = get(entry, "pitch_offset_deg",  bg.pitch_offset_deg),
            )
            _P_max_W = get(entry, "P_max_kW", u.motor.P_max_W / 1000.0) * 1000.0
            _omega   = get(entry, "rpm_hover", u.omega_nom * 60.0 / (2π)) * 2π / 60.0
            _od[id] = RotorUnit(
                id         = u.id,
                label      = u.label,
                radius_m   = new_bg.R,
                kT         = u.kT,
                kQ         = u.kQ,
                omega_nom  = _omega,
                inertia    = u.inertia,
                eta_rotor  = u.eta_rotor,
                k_induced  = u.k_induced,
                c_profile  = u.c_profile,
                arm_x_m    = u.arm_x_m,
                arm_y_m    = u.arm_y_m,
                arm_z_m    = u.arm_z_m,
                spin_dir   = u.spin_dir,
                blade_geom = new_bg,
                motor      = ElectricMotor(P_max_W=_P_max_W),
            )
        end
        @info "Custom rotor geometry: $(length(_od)) rotor(s) overridden"
        for (id, u) in sort(collect(_od), by=first)
            @printf("  R%d: R=%.3fm  chord=%.3fm  n_blades=%d  rpm_nom=%.0f  P_max=%.0f kW
",
                    id, u.radius_m, u.blade_geom.chord, u.blade_geom.n_blades,
                    u.omega_nom * 60.0 / (2π), u.motor.P_max_W / 1000.0)
        end
        RotorFleet(_od)
    end
end
const RP    = RotorParams()

rotor_area(rp::RotorParams) = rp.n * π * rp.radius_m^2
tau_spool(rp::RotorParams)  = rp.inertia / (rp.kT * rp.omega0)

const ROTOR_AREA = rotor_area(RP)
const TAU_SPOOL  = tau_spool(RP)

# ── Ground Effect ──────────────────────────────────────────────────────
"""
    rotor_ge(alt_m, radius_m) → multiplier ≥ 1

Exponential ground effect; avoids Cheeseman-Bennett singularity at z < R/4.
At alt=0: +15% lift.
"""
function rotor_ge(alt_m::Real, radius_m::Float64 = RP.radius_m)
    return 1.0 + 0.15 * exp(-max(alt_m, 0.0) / radius_m)
end

# ── Thrust Dynamics ────────────────────────────────────────────────────
"""
    thrust_derivative(thrust_act, thrust_cmd, rp) → dT/dt (N/s)

First-order lag: τ_spool = I_rotor / (kT · ω₀).
Operates on aggregate total thrust (all 6 rotors).
"""
function thrust_derivative(thrust_act, thrust_cmd, rp::RotorParams = RP)
    return (thrust_cmd - thrust_act) / tau_spool(rp)  # negative = reverse thrust
end

# ── BEM Thrust (per rotor, saving callback) ────────────────────────────
"""
    bem_thrust_fleet(rpms, vx_body, tilt_rad, alt, fleet, rp) → Float64 (N total)

Per-rotor BEM thrust summed over the fleet.  For use in saving callback only
(pure Float64 — never called inside the ODE).

Each rotor receives vx_axial = vx_body × cos(tilt_rad).
Falls back to legacy kT formula if rp.use_bem = false.
"""
function bem_thrust_fleet(rpms::NTuple{6,Float64}, vx_body::Float64,
                          tilt_rad::Float64, alt::Float64,
                          fleet::RotorFleet = FLEET,
                          rp::RotorParams   = RP) :: Float64
    ρ        = rho(alt)
    vx_axial = vx_body * cos(tilt_rad)
    T_total  = 0.0

    for i in 1:6
        u   = fleet.units[i]
        rpm = rpms[i]
        if rp.use_bem
            bc = blade_coefficients(rpm, vx_axial, ρ; bg = u.blade_geom)
            T_total += bc.thrust_N
        else
            ω = rpm * 2π / 60.0
            T_total += u.kT * ρ * ω^2 * u.radius_m^4
        end
    end
    return T_total
end

# ── BEM Thrust Aggregate (ODE-compatible) ─────────────────────────────
"""
    bem_thrust_aggregate(thrust_cmd, vx, alt, rp) → (thrust_N, power_kW)

ODE-compatible BEM call.  Back-calculates the per-rotor RPM that would
produce thrust_cmd/6 per rotor at the current vx/alt, then returns both the
BEM thrust and the associated shaft power.

RPM back-calculation:  T = kT_prop · ρ · n² · D⁴
  → n = √(T / (kT_prop · ρ · D⁴))
Uses bem_rpm_ref × √(T_demanded / T_ref) as a linearised starting point,
then applies a single Newton correction via kT_kQ(J=0).

Compatible with ForwardDiff Dual numbers: kT_kQ and blade_coefficients are
pure arithmetic — no branching on type.
"""
function bem_thrust_aggregate(thrust_cmd, vx, alt, rp::RotorParams = RP)
    T_per = thrust_cmd / 6.0  # negative = reverse; BEM handles magnitude
    ρ     = rho(alt)
    D     = 2.0 * BG.R

    # RPM back-calculation via momentum theory: T = kT·ρ·n²·D⁴ → n = √(T/(kT·ρ·D⁴))
    # Use bem_rpm_ref as an upper bound — this is the DA-corrected nominal RPM
    # (set by compute_da_correction as 1250/√(ρ_field/ρ_sl)), which correctly
    # scales RPM up at high-density-altitude airports like KAXX (9000 ft).
    kT0, _ = kT_kQ(0.0, ρ)
    kT0    = max(kT0, 1e-6)
    reverse = T_per < 0.0
    n_est  = sqrt(abs(T_per) / (kT0 * ρ * D^4 + 1e-9))
    rpm    = min(n_est * 60.0, rp.bem_rpm_ref)

    bc       = blade_coefficients(rpm, vx, ρ)
    # Reverse: thrust opposes forward direction, motor absorbs power (regen)
    sign     = reverse ? -1.0 : 1.0
    thrust_N = bc.thrust_N * 6.0 * sign
    power_kW = bc.power_W  * 6.0 / 1000.0 * (reverse ? -0.7 : 1.0)  # 70% regen efficiency
    return (thrust_N, power_kW)
end

# ── VRS-Gated Thrust ──────────────────────────────────────────────────
"""
    vrs_gated_thrust(thrust_cmd, vz, alt, rp) → (effective_N, vrs_factor)

Applies VRS thrust reduction when the vehicle is descending (vz < 0).

Arguments:
  thrust_cmd — demanded thrust from autopilot (N, aggregate all 6 rotors)
  vz         — body-frame vertical velocity (m/s, positive = climbing)
  alt        — AGL altitude (m)
  rp         — RotorParams

Returns:
  effective_N — thrust after VRS gating
  vrs_f       — VRS factor ∈ (0, 1]  (1.0 = no penalty)

Note: VRS penalty is only applied during descent (vz < 0) to avoid
false thrust loss during normal cruise/climb.
"""
function vrs_gated_thrust(thrust_cmd, vz, alt, rp::RotorParams = RP)
    if thrust_cmd < 0.0
        return (thrust_cmd, 1.0)  # reverse thrust: no VRS, pass through
    end
    T_cmd = thrust_cmd

    if vz >= 0.0
        return (T_cmd, 1.0)
    end

    ρ         = rho(alt)
    T_per     = rp.hover_thrust_N_sl / 6.0
    A_per     = π * rp.radius_m^2
    vrs_f     = vrs_factor(Float64(vz), T_per, Float64(ρ), A_per)
    return (T_cmd * vrs_f, vrs_f)
end

# ── Motor Shaft Power Limit ────────────────────────────────────────────
"""
    motor_shaft_limit(rp, rpm, alt) → Float64 (W, aggregate ceiling)

Returns the total shaft power ceiling across all 6 rotors given the current
RPM and altitude. Uses the scalar motor in RP (uniform fleet).

For per-rotor limits with a mixed FLEET, call motor_power_available_W on
each fleet.units[i].motor directly.
"""
function motor_shaft_limit(rp::RotorParams, rpm::Float64, alt::Float64)::Float64
    return 6.0 * motor_power_available_W(rp.motor, rpm, alt)
end

# Overload for full fleet (per-unit motors)
function motor_shaft_limit(fleet::RotorFleet, rpm::Float64, alt::Float64)::Float64
    total = 0.0
    for u in fleet.units
        total += motor_power_available_W(u.motor, rpm, alt)
    end
    return total
end

# ── Per-Rotor RPM (legacy, saving callback) ────────────────────────────
"""
    rotor_rpm_each(T_total, alt, fleet) → NTuple{6, Float64} (RPM)

Back-calculates per-rotor speed from aggregate thrust (legacy kT formula).
Equal thrust-sharing assumed.  Saving callback only.
"""
function rotor_rpm_each(T_total::Float64, alt::Float64,
                        fleet::RotorFleet = FLEET) :: NTuple{6, Float64}
    reverse = T_total < 0.0
    T_per = abs(T_total) / 6.0
    ρ     = rho(alt)
    ntuple(6) do i
        u = fleet.units[i]
        ω = sqrt(max(T_per, 0.0) / (u.kT * ρ * u.radius_m^4 + 1e-9))
        rpm = ω * 60.0 / (2π)
        reverse ? -rpm : rpm  # negative RPM indicates reverse
    end
end

# ── Per-Rotor BEM Power (saving callback) ─────────────────────────────
"""
    rotor_kw_each_bem(T_total, vx, tilt_rad, alt, fleet) → NTuple{6, Float64} (kW)

Per-rotor BEM power via blade_coefficients.  Back-calculates RPM from
equal-share thrust demand, then evaluates the BEM model.
Saving callback only (pure Float64).
"""
function rotor_kw_each_bem(T_total::Float64, vx::Float64, tilt_rad::Float64,
                            alt::Float64, fleet::RotorFleet = FLEET) :: NTuple{6, Float64}
    T_per    = max(T_total, 0.0) / 6.0
    ρ        = rho(alt)
    vx_axial = vx * cos(tilt_rad)
    D        = 2.0 * BG.R

    ntuple(6) do i
        u       = fleet.units[i]
        bg      = u.blade_geom
        kT0, _  = kT_kQ(0.0, ρ; bg = bg)
        kT0     = max(kT0, 1e-6)
        n_est   = sqrt(T_per / (kT0 * ρ * D^4 + 1e-9))
        rpm     = n_est * 60.0
        bc      = blade_coefficients(rpm, vx_axial, ρ; bg = bg)
        bc.power_W / 1000.0
    end
end

# ── Per-Rotor Power (legacy momentum, saving callback) ─────────────────
"""
    rotor_kw_each(T_total, vx, alt, fleet) → NTuple{6, Float64} (kW)

Legacy momentum-theory per-rotor power.  Retained for backwards compat.
"""
function rotor_kw_each(T_total::Float64, vx::Float64, alt::Float64,
                       fleet::RotorFleet = FLEET) :: NTuple{6, Float64}
    T_per = max(T_total, 0.0) / 6.0
    v     = max(vx, 0.0)
    ρ     = rho(alt)
    ntuple(6) do i
        u     = fleet.units[i]
        A_i   = π * u.radius_m^2
        v_ih  = sqrt(T_per / (2.0 * ρ * A_i + 1e-6))
        v_i   = v_ih^2 / sqrt(v_ih^2 + (v / 2.0)^2 + 1e-6)
        P_ind = u.k_induced * T_per * v_i / u.eta_rotor
        P_prf = u.c_profile * ρ * A_i * (u.omega_nom * u.radius_m)^3
        (P_ind + P_prf) / 1000.0
    end
end

# ── Aggregate Power (ODE-compatible) ──────────────────────────────────
"""
    rotor_power_kw(vx, thrust, alt, rp) → kW

Fleet aggregate shaft power.  Dispatches on rp.use_bem:

  use_bem=true  → calls bem_thrust_aggregate and returns BEM-derived power.
  use_bem=false → legacy three-component momentum-theory model.

Compatible with ForwardDiff Dual numbers in both paths.
"""
function rotor_power_kw(vx, thrust, alt, rp::RotorParams = RP)
    if rp.use_bem
        _, P_kw = bem_thrust_aggregate(thrust, vx, alt, rp)
        return P_kw
    else
        # Legacy momentum-theory path (retained as fallback)
        v  = max(vx, 0.0)
        T  = max(thrust, 0.0)
        ρ  = rho(alt)
        A  = rotor_area(rp)
        vi_h = sqrt(T / (2.0 * ρ * A + 1e-6))
        vi   = vi_h^2 / sqrt(vi_h^2 + (v / 2.0)^2 + 1e-6)
        P_i  = rp.k_induced * T * vi / rp.eta_rotor
        P_pr = rp.c_profile * ρ * A * (rp.omega0 * rp.radius_m)^3
        return (P_i + P_pr) / 1000.0
    end
end

# ── VRS Hazard Metric (diagnostic, saving callback) ───────────────────
"""
    vrs_risk(vd_ms, vx_ms, T_total, alt, rp) → Float64 ∈ [0, 1]

Wolkovitch VRS hazard index.  Diagnostic metric for the cockpit warning;
does not reduce thrust (vrs_gated_thrust does that).
"""
function vrs_risk(vd_ms::Float64, vx_ms::Float64,
                  T_total::Float64, alt::Float64,
                  rp::RotorParams = RP) :: Float64
    A    = rotor_area(rp)
    ρ    = rho(alt)
    v_ih = sqrt(max(T_total, 1.0) / (2.0 * ρ * A + 1e-6))

    descent_r = max(vd_ms, 0.0) / (v_ih + 1e-6)
    lateral_r = abs(vx_ms)      / (v_ih + 1e-6)

    hazard_d   = clamp((descent_r - 0.15) / 0.50, 0.0, 1.0) *
                 clamp((2.0 - descent_r)  / 1.50, 0.0, 1.0)
    hazard_lat = clamp(1.0 - lateral_r, 0.0, 1.0)
    return hazard_d * hazard_lat
end

# ── Density-Altitude Thrust Correction ────────────────────────────────
const RHO_SL = 1.225

"""
    compute_da_correction() → Float64

Compute ρ_field / ρ_sl and write RP.da_correction, RP.hover_thrust_N,
and RP.bem_rpm_ref.

Call after ATM has been set from test_card.json, before any ODE build.
"""
function compute_da_correction()::Float64
    rho_field  = rho(0.0)
    correction = rho_field / RHO_SL

    RP.da_correction = correction

    # DA-corrected RPM: T ∝ ρ·n² at fixed kT → n ∝ 1/√ρ to maintain thrust
    RP.bem_rpm_ref = 1250.0 / sqrt(max(correction, 0.1))

    # BEM hover thrust at the DA-corrected RPM and field density.
    # This is what the motor actually delivers — RPM scales up to compensate
    # for lower ρ, so hover_thrust_N reflects real capability, not ρ-scaled SL.
    bc = blade_coefficients(RP.bem_rpm_ref, 0.0, rho_field)
    RP.hover_thrust_N_sl = bc.thrust_N * 6.0   # BEM value supersedes geometric default
    RP.hover_thrust_N    = bc.thrust_N * 6.0   # effective at field — RPM already corrected

    @info "DA thrust correction: ρ=$(round(rho_field,digits=4)) kg/m³  " *
          "factor=$(round(correction,digits=4))  " *
          "BEM rpm_ref=$(round(RP.bem_rpm_ref,digits=0)) RPM  " *
          "hover_thrust=$(round(RP.hover_thrust_N,digits=0)) N"
    return correction
end

"""
    rotor_thrust_available(alt_agl) → Float64 (N)

Maximum aggregate thrust at altitude. Called per ODE step to clamp thrust_cmd.
"""
function rotor_thrust_available(alt_agl::Real)::Float64
    ρ      = rho(alt_agl)
    rpm    = 1250.0 / sqrt(max(ρ / RHO_SL, 0.1))   # DA-corrected RPM at this altitude
    bc     = blade_coefficients(rpm, 0.0, ρ)
    return bc.thrust_N * 6.0
end

"""
    fleet_thrust_available(alt_agl, fleet) → Float64 (N)

Per-rotor BEM thrust summed over FLEET using each rotor's actual blade_geom.
DA-corrected RPM computed individually per rotor radius.
Use in place of rotor_thrust_available when FLEET has geometry overrides.
"""
function fleet_thrust_available(alt_agl::Real, fleet::RotorFleet = FLEET)::Float64
    ρ   = rho(alt_agl)
    tot = 0.0
    for u in fleet.units
        rpm = 1250.0 / sqrt(max(ρ / RHO_SL, 0.1))
        bc  = blade_coefficients(rpm, 0.0, ρ; bg = u.blade_geom)
        tot += bc.thrust_N
    end
    return tot
end

"""
    preflight_da_warning(weight_N)

Log a warning if T/W at departure is marginal.
"""
function preflight_da_warning(weight_N::Float64)
    tw = RP.hover_thrust_N / weight_N
    if tw < 1.05
        @warn "Low T/W at departure: $(round(tw,digits=3)). " *
              "Hover margin $(round((tw-1)*100,digits=1))%. " *
              "Consider cooler window or reduced payload."
    elseif tw < 1.10
        @info "Reduced hover margin: T/W=$(round(tw,digits=3))  " *
              "($(round((tw-1)*100,digits=1))% above hover)"
    end
end
# blades.jl:       Blade element momentum (BEM) implementation
# AUTHOR:          DANIEL DESAI
# UPDATED:         2026-05-10
# VERSION:         0.1.0
#
# Implements:
#
#   §1  BladeGeometry — struct with defaults
#   §2  Airfoil polars — NACA 0012-like (replace with XFOIL table for production)
#   §3  local_twist() — linear root→tip schedule with collective offset
#   §4  kT_kQ()       — BEM iteration, 30 stations, under-relaxed inflow
#   §5  blade_coefficients() — primary interface; maps (rpm, vx_axial, rho) → forces
#   §6  vrs_factor()  — Peters–HaQuang Gaussian dip model
#   §7  autorotation_rpm() — equilibrium RPM search
#   §8  Motor backends — AbstractMotor, ElectricMotor, TurboshaftEngine,
#                        HybridTurbineElectric, build_motor()
#   §9  blades_selftest() — embedded validation (8 tests)
#
# Tilt-rotor advance-ratio convention
# ─────────────────────────────────────
# blade_coefficients() receives  vx_axial = vx_body * cos(tilt_rad).
#   At cruise (tilt = π/2): vx_axial ≈ 0 → J ≈ 0 → hover polar → correct thrust.
#   At hover  (tilt = 0):   vx_axial = vx_body → correct transition physics.
# Fixed-pitch blades go to zero thrust when J rises above ~0.15 — this is
# correct: the tiltrotor keeps J small by tilting the disk into the flow.
#
# Calibration targets (SL ISA, 1250 RPM, 3-blade, R=1.524 m):
#   J=0.00  → T ≈ 4 342 N/rotor, T_total ≈ 26 055 N  (T/W=1.22 at MTOW=2177 kg) ✓
#   J=0.10  → CT_heli = kT_prop × 4/π² ∈ [0.008, 0.015]                          ✓
#   VRS onset ≈ 5 m/s descent (μz/v_h = 0.32), peak loss 30% at μz/v_h = 1.1     ✓
#
# References
# ──────────
#   Leishman, "Principles of Helicopter Aerodynamics", 2nd ed., Ch 3–4
#   Peters & HaQuang (1988) – dynamic inflow / VRS
#   Gagg & Ferrar (1934)    – altitude power lapse for turboshafts


module Blades

export BladeGeometry, BG,
       blade_coefficients, kT_kQ, vrs_factor, autorotation_rpm,
       AbstractMotor, ElectricMotor, TurboshaftEngine, HybridTurbineElectric,
       build_motor, motor_power_available_W, motor_shaft_power_W,
       blades_selftest

# ─────────────────────────────────────────────────────────────────────────────
# §1  Blade Geometry
# ─────────────────────────────────────────────────────────────────────────────
"""
    BladeGeometry

Physical blade parameters for one rotor.  All angles are stored as degrees in
the struct for readability; converted to radians inside BEM functions.

Defaults are calibrated to a Joby S4-class rotor (see module header for targets).
"""
struct BladeGeometry
    R               ::Float64   # tip radius (m)
    n_blades        ::Int       # number of blades
    chord           ::Float64   # mean chord (m)
    twist_root_deg  ::Float64   # geometric pitch at root (°)
    twist_tip_deg   ::Float64   # geometric pitch at tip  (°)
    pitch_offset_deg::Float64   # collective pitch offset β₀ (°)
    solidity        ::Float64   # σ = N·c/(π·R), auto-computed
end

# Keyword constructor: auto-computes solidity.
function BladeGeometry(;
        R               = 1.524,
        n_blades        = 3,
        chord           = 0.12,
        twist_root_deg  = 16.0,
        twist_tip_deg   =  6.0,
        pitch_offset_deg=  4.4)
    σ = n_blades * chord / (π * R)
    BladeGeometry(R, n_blades, chord, twist_root_deg, twist_tip_deg, pitch_offset_deg, σ)
end

""" Module-level default BladeGeometry (Joby S4-class). """
const BG = BladeGeometry()

# ─────────────────────────────────────────────────────────────────────────────
#  Airfoil Polars
# ─────────────────────────────────────────────────────────────────────────────
# NACA 0012-like thin-airfoil approximation.
# For production: replace with bilinear interpolation from an XFOIL/BEMT table:
#   Cl_table[Re_idx, α_idx],  Cd_table[Re_idx, α_idx]

"""
    airfoil_Cl(alpha_rad) → Float64

2π lift slope, linear below stall (15°), tapering post-stall.
"""
@inline function airfoil_Cl(alpha_rad::Float64)::Float64
    a0          = 2π
    alpha_stall = 0.2618  # deg2rad(15)
    abs_α       = abs(alpha_rad)
    if abs_α < alpha_stall
        return a0 * alpha_rad
    else
        overshoot = (abs_α - alpha_stall) / alpha_stall
        return a0 * alpha_stall * sign(alpha_rad) * (1.0 - 0.3 * overshoot)
    end
end

"""
    airfoil_Cd(alpha_rad) → Float64

Profile drag: Cd₀ = 0.012 (NACA 0012 at Re ≈ 1×10⁶) + quadratic induced term.
"""
@inline function airfoil_Cd(alpha_rad::Float64)::Float64
    return 0.012 + 0.020 * alpha_rad^2
end

# ─────────────────────────────────────────────────────────────────────────────
#  Local Twist Schedule
# ─────────────────────────────────────────────────────────────────────────────
"""
    local_twist(r_frac, bg) → Float64 (rad)

Total blade pitch angle at normalised radius r_frac ∈ (0, 1]:
    θ(r) = twist_root + r·(twist_tip − twist_root) + pitch_offset
All terms in the struct are degrees; result returned in radians.
"""
@inline function local_twist(r_frac::Float64, bg::BladeGeometry)::Float64
    return (π / 180.0) * (bg.twist_root_deg
                          + r_frac * (bg.twist_tip_deg - bg.twist_root_deg)
                          + bg.pitch_offset_deg)
end

# ─────────────────────────────────────────────────────────────────────────────
#   BEM Iteration
# ─────────────────────────────────────────────────────────────────────────────
"""
    kT_kQ(J, rho; bg, n_stations, tol, max_iter) → (kT, kQ)

Blade-Element Momentum iteration.  Returns non-dimensional thrust and torque
coefficients in the propeller convention:

    T = kT · ρ · n² · D⁴    (N, per rotor,  n in rev/s, D = 2R)
    Q = kQ · ρ · n² · D⁵    (N·m, per rotor)

Helicopter CT convention: CT = kT_prop × 4/π²

Algorithm (Leishman §3.4)
─────────────────────────
1. Guess inflow ratio λ.
   Hover (J≈0): λ₀ = 0.05.  Forward flight: λ₀ = J/π.
2. For each radial station i:
     r = (i−½)/N,  θ = local_twist(r)
     Vt = π·r,     Va = J·π + λ,     ϕ = atan(Va, Vt)
     α  = θ − ϕ
     dkT = ½·σ·(Cl·cosϕ − Cd·sinϕ)·r²·dr
     dkQ = ½·σ·(Cl·sinϕ + Cd·cosϕ)·r³·dr
3. Integrate → kT_new, kQ_new; scale by π².
4. Momentum update: λ_new = kT_new / (2·√(J²+λ²))
5. Under-relax: λ ← 0.4·λ + 0.6·λ_new.  Repeat until |Δλ| < tol.

Under-relaxation at 0.4/0.6 gives robust convergence near blade stall
without oscillation.  Convergence in ~15 iter (hover) to ~80 iter (stall).
"""
function kT_kQ(J::Float64, rho::Float64;
               bg::BladeGeometry = BG,
               n_stations::Int   = 30,
               tol::Float64      = 1e-7,
               max_iter::Int     = 150) :: Tuple{Float64,Float64}

    # Suppress unused rho warning — rho needed by callers for dimensional forces,
    # passed through here so the function signature is consistent.
    _ = rho

    λ   = J < 0.01 ? 0.05 : J / π
    dr  = 1.0 / n_stations
    kT_out = 0.0
    kQ_out = 0.0

    for _ in 1:max_iter
        kT_new = 0.0
        kQ_new = 0.0

        for i in 1:n_stations
            r  = (i - 0.5) * dr
            θ  = local_twist(r, bg)

            Vt = π * r
            Va = J * π + λ
            ϕ  = atan(Va, Vt)
            α  = θ - ϕ

            Cl = airfoil_Cl(α)
            Cd = airfoil_Cd(α)

            kT_new += 0.5 * bg.solidity * (Cl * cos(ϕ) - Cd * sin(ϕ)) * r^2 * dr
            kQ_new += 0.5 * bg.solidity * (Cl * sin(ϕ) + Cd * cos(ϕ)) * r^3 * dr
        end

        kT_new *= π^2
        kQ_new *= π^2

        denom = 2.0 * sqrt(J^2 + λ^2)
        λ_new = denom > 1e-8 ? kT_new / denom : sqrt(max(kT_new / 2.0, 0.0))

        kT_out = max(kT_new, 0.0)
        kQ_out = max(kQ_new, 0.0)

        abs(λ_new - λ) < tol && return (kT_out, kQ_out)

        λ = 0.4 * λ + 0.6 * λ_new
    end

    return (kT_out, kQ_out)     # last iterate on non-convergence (better than zeros)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Primary Interface
# ─────────────────────────────────────────────────────────────────────────────
"""
    blade_coefficients(rpm, vx_axial, rho; bg) → NamedTuple

Primary call site consumed by rotor_system.jl.

Arguments
─────────
  rpm       — rotor speed (rev/min)
  vx_axial  — axial inflow velocity into the rotor disk (m/s).
               Tiltrotor callers: pass  vx_body * cos(tilt_rad).
               Negative values are clamped to zero (no reverse inflow model).
  rho       — local air density (kg/m³); supply rho_alt from compute_da_correction().
  bg        — BladeGeometry override (default: module BG).

Returns NamedTuple
──────────────────
  thrust_N   (N)     per-rotor thrust (≥ 0)
  torque_Nm  (N·m)   per-rotor shaft torque
  power_W    (W)     per-rotor shaft power  (= Q · Ω)
  kT         (–)     propeller thrust coefficient
  kQ         (–)     propeller torque coefficient
  J          (–)     advance ratio V/(n·D)
"""
function blade_coefficients(rpm::Float64, vx_axial::Float64, rho::Float64;
                             bg::BladeGeometry = BG)
    n = rpm / 60.0
    D = 2.0 * bg.R

    J = n > 0.5 ? max(vx_axial, 0.0) / (n * D) : 0.0

    kT, kQ = kT_kQ(J, rho; bg = bg)

    T = kT * rho * n^2 * D^4
    Q = kQ * rho * n^2 * D^5
    P = Q * 2π * n

    return (thrust_N  = T,
            torque_Nm = Q,
            power_W   = P,
            kT        = kT,
            kQ        = kQ,
            J         = J)
end

# ─────────────────────────────────────────────────────────────────────────────
#  VRS Factor
# ─────────────────────────────────────────────────────────────────────────────
"""
    vrs_factor(vz, hover_thrust_N, rho, disk_area) → Float64 ∈ (0, 1]

Thrust multiplier for Vortex Ring State.

VRS occurs when the blade wake recirculates back through the disk during
axial descent. Modelled as a Gaussian thrust dip (Peters–HaQuang metric):

    peak loss = 30%  at  μz/v_h = 1.1
    window:  0.3 ≤ μz/v_h ≤ 2.5

Arguments
─────────
  vz             — body-frame vertical velocity (m/s, positive = climbing).
                   VRS activates only when vz < 0 (descending).
  hover_thrust_N — per-rotor hover thrust (N), used to compute v_h.
  rho            — local air density (kg/m³).
  disk_area      — per-rotor disk area = π·R² (m²).

Returns 1.0 outside the VRS window (no penalty).
Callers should gate this on vz < 0 as well (vrs_gated_thrust in rotor_system.jl).

Note: full Peters–HaQuang dynamic inflow requires ODE states in fly.jl.
This quasi-static approximation is suitable for mission-level simulation.
"""
function vrs_factor(vz::Float64, hover_thrust_N::Float64,
                    rho::Float64, disk_area::Float64)::Float64
    v_h   = sqrt(max(hover_thrust_N, 1.0) / (2.0 * rho * disk_area + 1e-9))
    μz    = -vz                                    # descent rate (positive = downward)
    ratio = v_h > 0.1 ? μz / v_h : 0.0

    if ratio < 0.3 || ratio > 2.5
        return 1.0
    end

    loss = 0.30 * exp(-((ratio - 1.1)^2) / (2.0 * 0.45^2))
    return 1.0 - loss
end

# ─────────────────────────────────────────────────────────────────────────────
#  Autorotation RPM
# ─────────────────────────────────────────────────────────────────────────────
"""
    autorotation_rpm(vz, rho; bg, target_kT) → Float64 (RPM)

Estimate equilibrium autorotation RPM.

At autorotation, net rotor torque = 0: energy extracted from the descending
airstream equals profile drag.  This model iterates n (rev/s) until BEM kT
matches `target_kT` at the current J.

Arguments
─────────
  vz         — descent rate (m/s, positive = descending into wind)
  rho        — air density (kg/m³)
  target_kT  — required kT to sustain autorotation (default 0.010)

Returns RPM, clamped to [600, 3600].
"""
function autorotation_rpm(vz::Float64, rho::Float64;
                          bg::BladeGeometry = BG,
                          target_kT::Float64 = 0.010)::Float64
    D       = 2.0 * bg.R
    n_guess = 15.0          # initial guess ≈ 900 RPM

    for _ in 1:50
        J       = vz / max(n_guess * D, 1e-6)
        kT, _   = kT_kQ(J, rho; bg = bg)
        err     = kT - target_kT
        n_guess = clamp(n_guess - err * 8.0, 10.0, 60.0)
    end

    return clamp(n_guess * 60.0, 600.0, 3600.0)
end

# ─────────────────────────────────────────────────────────────────────────────
#   Motor Backends
# ─────────────────────────────────────────────────────────────────────────────
# All backends share AbstractMotor.  rotor_system.jl selects the active backend
# via RP.motor (scalar, all rotors) or per-unit via FLEET.units[i].motor.
#
# Mandatory interface (implement for every new backend):
#   motor_power_available_W(m, rpm, alt_m)         → Float64  (ceiling W)
#   motor_shaft_power_W(m, rpm, torque_Nm, alt_m)  → Float64  (actual W)
#
# Optional (implement for state-tracking backends):
#   update_motor_thermal!(m, P_W, dt_s, T_amb_C)   — electric winding heat
#   burn_fuel!(m, P_W, dt_s)                        — turboshaft fuel
#   update_batt_soc!(m, P_W, dt_s, alt_m)           — hybrid battery SoC

"""Supertype for all motor/engine backends."""
abstract type AbstractMotor end

const _RHO_SL = 1.225   # ISA SL density used internally

#   Electric Motor ───────────────────────────────────────────────────────
"""
    ElectricMotor

Permanent-magnet synchronous motor (PMSM) with:
  - Peak shaft power ceiling P_max_W (default 200 kW per rotor)
  - Altitude de-rating above cool_alt_m (cooling air density drop)
  - Simple winding thermal model for thermal de-rating

At hover SL ISA: P_demand ≈ 141 kW/rotor → motor loaded at 71%.  Margin for
transients (VRS recovery, gusty hover) ≈ 29%.

Motor swap example:
    FLEET_HI = RotorFleet(Dict(
        1 => RotorUnit(_default_unit(1); motor = ElectricMotor(P_max_W=250_000.0)),
        2 => RotorUnit(_default_unit(2); motor = ElectricMotor(P_max_W=250_000.0))))
"""
Base.@kwdef mutable struct ElectricMotor <: AbstractMotor
    P_max_W        ::Float64 = 200_000.0  # peak shaft power (W)
    eta_peak       ::Float64 = 0.96       # peak motor efficiency
    eta_rpm_peak   ::Float64 = 1250.0     # RPM at peak efficiency
    thermal_mass   ::Float64 = 2.0        # winding heat capacity (kJ/K)
    tau_thermal_s  ::Float64 = 120.0      # winding cooling time constant (s)
    T_cont_C       ::Float64 = 100.0      # continuous winding temperature (°C)
    T_max_C        ::Float64 = 160.0      # peak winding temperature (°C)
    T_winding_C    ::Float64 = 25.0       # current winding temperature (°C, mutable)
    cool_alt_m     ::Float64 = 3000.0     # altitude above which cooling degrades
end

"""
    motor_power_available_W(m::ElectricMotor, rpm, alt_m) → Float64

P_avail = P_max × η_alt × η_thermal.

Altitude de-rate:  linear 1.0 → 0.80 from cool_alt_m to cool_alt_m+3000 m.
Thermal de-rate:   linear 1.0 → 0.50 from T_cont_C to T_max_C.
"""
function motor_power_available_W(m::ElectricMotor, rpm::Float64, alt_m::Float64)::Float64
    alt_excess  = max(alt_m - m.cool_alt_m, 0.0)
    η_alt       = max(1.0 - 0.20 * alt_excess / 3000.0, 0.80)

    T_margin    = m.T_max_C - m.T_cont_C
    η_thermal   = T_margin > 0.0 ?
                  clamp(1.0 - (m.T_winding_C - m.T_cont_C) / T_margin, 0.5, 1.0) :
                  1.0

    return m.P_max_W * η_alt * η_thermal
end

"""
    motor_shaft_power_W(m::ElectricMotor, rpm, torque_Nm, alt_m) → Float64

min(P_demand, P_available).  Does not mutate thermal state.
"""
function motor_shaft_power_W(m::ElectricMotor, rpm::Float64,
                              torque_Nm::Float64, alt_m::Float64)::Float64
    P_demand = max(torque_Nm * 2π * rpm / 60.0, 0.0)
    return min(P_demand, motor_power_available_W(m, rpm, alt_m))
end

"""
    update_motor_thermal!(m::ElectricMotor, P_shaft_W, dt_s, T_ambient_C=20.0)

Advance winding temperature by one saving-callback time step.
Do NOT call inside the ODE integrator.
"""
function update_motor_thermal!(m::ElectricMotor, P_shaft_W::Float64,
                                dt_s::Float64, T_ambient_C::Float64 = 20.0)
    P_loss       = P_shaft_W * (1.0 - m.eta_peak) / m.eta_peak
    dT_heat      = (P_loss / 1000.0) / m.thermal_mass * dt_s
    dT_cool      = (m.T_winding_C - T_ambient_C) / m.tau_thermal_s * dt_s
    m.T_winding_C = m.T_winding_C + dT_heat - dT_cool
    return nothing
end

#   Turboshaft Engine ────────────────────────────────────────────────────
"""
    TurboshaftEngine

Free-turbine turboshaft with Gagg–Ferrar altitude lapse and SFC fuel burn.

    P(alt) = P_sl × (ρ_alt / ρ_sl)^n_lapse × η_gearbox

Lapse exponent n_lapse:
  1.132 — Gagg–Ferrar turboprop (default, standard for S4-class)
  1.0   — pure density-proportional (conservative)
  0.7   — turbofan-like (better altitude performance)

SFC = 7.78e-8 kg/(W·s) ≡ 0.28 kg/(kW·h) — typical medium turboshaft.

Turboshaft fleet swap example:
    ts = TurboshaftEngine(P_sl_W=1_200_000.0)
    RP.motor = ts     # scalar: all rotors backed by single turboshaft gas path
"""
Base.@kwdef mutable struct TurboshaftEngine <: AbstractMotor
    P_sl_W         ::Float64 = 1_000_000.0  # sea-level shaft power (W)
    n_lapse        ::Float64 = 1.132        # Gagg–Ferrar lapse exponent
    SFC_kg_per_Ws  ::Float64 = 7.78e-8      # specific fuel consumption (kg/(W·s))
    tau_spool_s    ::Float64 = 2.0          # spool-up time constant (s)
    eta_gearbox    ::Float64 = 0.97         # gearbox efficiency
    fuel_kg        ::Float64 = 200.0        # remaining fuel (kg, mutable)
end

"""
    motor_power_available_W(e::TurboshaftEngine, rpm, alt_m) → Float64

Gagg–Ferrar lapse.  Uses exponential atmosphere (accurate to <2% below 12 km).
"""
function motor_power_available_W(e::TurboshaftEngine, rpm::Float64, alt_m::Float64)::Float64
    ρ     = _RHO_SL * exp(-alt_m / 8500.0)
    lapse = (ρ / _RHO_SL)^e.n_lapse
    return e.P_sl_W * lapse * e.eta_gearbox
end

"""
    motor_shaft_power_W(e::TurboshaftEngine, rpm, torque_Nm, alt_m) → Float64
"""
function motor_shaft_power_W(e::TurboshaftEngine, rpm::Float64,
                              torque_Nm::Float64, alt_m::Float64)::Float64
    P_demand = max(torque_Nm * 2π * rpm / 60.0, 0.0)
    return min(P_demand, motor_power_available_W(e, rpm, alt_m))
end

"""
    burn_fuel!(e::TurboshaftEngine, P_shaft_W, dt_s) → Float64 (kg burned)

Decrement fuel load.  Call from saving callback only.
"""
function burn_fuel!(e::TurboshaftEngine, P_shaft_W::Float64, dt_s::Float64)::Float64
    burned   = e.SFC_kg_per_Ws * P_shaft_W * dt_s
    e.fuel_kg = max(e.fuel_kg - burned, 0.0)
    return burned
end

#   Hybrid Turbine-Electric ──────────────────────────────────────────────
"""
    HybridTurbineElectric

Series-hybrid: turbine → generator → battery bus → electric motor → shaft.

Power budget:
    P_available = min(P_motor_ceiling, (P_gen + P_batt_discharge) × η_inverter)

Battery:
    SoC-limited discharge at up to P_batt_max_W.
    Regenerative charging during descent when P_rotor_demand < P_gen.
    SoC ∈ [SoC_min, 1.0] (hard reserve floor prevents damage).

Example — aft pair turboshaft, fore/mid pairs hybrid:
    ts = TurboshaftEngine(P_sl_W=600_000.0)
    hy = HybridTurbineElectric(
             turbine=TurboshaftEngine(P_sl_W=400_000.0),
             motor=ElectricMotor(P_max_W=250_000.0),
             E_batt_J=3.6e8)
    FLEET_MIXED = RotorFleet(Dict(5 => RotorUnit(_default_unit(5); motor=ts),
                                  6 => RotorUnit(_default_unit(6); motor=ts)))
"""
Base.@kwdef mutable struct HybridTurbineElectric <: AbstractMotor
    turbine        ::TurboshaftEngine = TurboshaftEngine()
    motor          ::ElectricMotor   = ElectricMotor()
    E_batt_J       ::Float64 = 3.6e8         # battery capacity (J), default 100 kWh
    P_batt_max_W   ::Float64 = 200_000.0     # max discharge rate (W)
    P_regen_max_W  ::Float64 = 100_000.0     # max regen charge rate (W)
    SoC            ::Float64 = 0.80          # state of charge ∈ [0,1] (mutable)
    eta_generator  ::Float64 = 0.94          # turbine → electrical
    eta_inverter   ::Float64 = 0.97          # DC bus → motor
    SoC_min        ::Float64 = 0.05          # hard reserve floor
end

"""
    motor_power_available_W(h::HybridTurbineElectric, rpm, alt_m) → Float64
"""
function motor_power_available_W(h::HybridTurbineElectric, rpm::Float64, alt_m::Float64)::Float64
    P_gen        = motor_power_available_W(h.turbine, rpm, alt_m) * h.eta_generator
    batt_avail   = (h.SoC - h.SoC_min) * h.E_batt_J > 0.0 ? h.P_batt_max_W : 0.0
    P_motor_ceil = motor_power_available_W(h.motor, rpm, alt_m)
    return min(P_motor_ceil, (P_gen + batt_avail) * h.eta_inverter)
end

"""
    motor_shaft_power_W(h::HybridTurbineElectric, rpm, torque_Nm, alt_m) → Float64
"""
function motor_shaft_power_W(h::HybridTurbineElectric, rpm::Float64,
                              torque_Nm::Float64, alt_m::Float64)::Float64
    P_demand = max(torque_Nm * 2π * rpm / 60.0, 0.0)
    return min(P_demand, motor_power_available_W(h, rpm, alt_m))
end

"""
    update_batt_soc!(h::HybridTurbineElectric, P_rotor_W, dt_s, alt_m)

Advance battery SoC.  Call from saving callback only.
  P_rotor_W > P_gen → battery discharges.
  P_rotor_W < P_gen → turbine surplus charges battery (regen).
"""
function update_batt_soc!(h::HybridTurbineElectric, P_rotor_W::Float64,
                           dt_s::Float64, alt_m::Float64)
    P_gen  = motor_power_available_W(h.turbine, 1250.0, alt_m) * h.eta_generator
    P_net  = P_rotor_W / h.eta_inverter - P_gen   # positive = discharge

    ΔE = if P_net > 0.0
        -min(P_net, h.P_batt_max_W)  * dt_s
    else
        min(-P_net, h.P_regen_max_W) * dt_s
    end

    h.SoC = clamp(h.SoC + ΔE / h.E_batt_J, h.SoC_min, 1.0)
    return nothing
end

#   Motor Factory ────────────────────────────────────────────────────────
"""
    build_motor(type::Symbol; kwargs...) → AbstractMotor

Generic motor factory.  Unknown kwargs are silently ignored (safe to pass a
uniform config dict regardless of which backend is active).

Types
─────
  :electric   → ElectricMotor
  :turboshaft → TurboshaftEngine
  :hybrid     → HybridTurbineElectric
               (pass `turbine=` and `motor=` as keyword args)

Examples
────────
    build_motor(:electric)
    build_motor(:turboshaft; P_sl_W=1_200_000.0)
    build_motor(:hybrid;
                turbine = TurboshaftEngine(P_sl_W=400_000.0),
                motor   = ElectricMotor(P_max_W=250_000.0),
                E_batt_J = 3.6e8)
"""
function build_motor(type::Symbol; kwargs...)::AbstractMotor
    kw = Dict(kwargs)
    if type === :electric
        valid = (:P_max_W, :eta_peak, :eta_rpm_peak, :thermal_mass,
                 :tau_thermal_s, :T_cont_C, :T_max_C, :T_winding_C, :cool_alt_m)
        return ElectricMotor(; filter(p -> p.first ∈ valid, kw)...)
    elseif type === :turboshaft
        valid = (:P_sl_W, :n_lapse, :SFC_kg_per_Ws, :tau_spool_s, :eta_gearbox, :fuel_kg)
        return TurboshaftEngine(; filter(p -> p.first ∈ valid, kw)...)
    elseif type === :hybrid
        turbine = get(kw, :turbine, TurboshaftEngine())
        motor   = get(kw, :motor,   ElectricMotor())
        valid   = (:E_batt_J, :P_batt_max_W, :P_regen_max_W,
                   :SoC, :eta_generator, :eta_inverter, :SoC_min)
        return HybridTurbineElectric(; turbine, motor, filter(p -> p.first ∈ valid, kw)...)
    else
        error("build_motor: unknown type :$(type). Use :electric, :turboshaft, or :hybrid.")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#   Self-Test
# ─────────────────────────────────────────────────────────────────────────────
"""
    blades_selftest() → Bool

Runs 8 embedded validation checks covering geometry calibration, VRS, and
motor backends.  Returns true if all pass; prints PASS/FAIL for each.

No external dependencies — safe to call at module load or in CI.

Expected at SL ISA (ρ = 1.225 kg/m³), R = 1.524 m, 1250 RPM:
  T1  kT_prop(J=0) ∈ (0.07, 0.14)            — coefficient in physical range
  T2  T/rotor(J=0) ∈ (3800, 4900) N           — matches 4342 N hover target
  T3  CT_helicopter(J=0.10) ∈ (0.008, 0.015)  — mild forward-flight criterion
  T4  vrs_factor(vz=0) = 1.0                  — no VRS in hover
  T5  vrs_factor(vz=-8 m/s) < 0.95            — VRS active at moderate descent
  T6  ElectricMotor P_avail > 150 kW at SL    — covers 141 kW hover demand
  T7  TurboshaftEngine P_avail > 500 kW at SL — covers full 6-rotor fleet
  T8  HybridTurbineElectric P_avail > 150 kW  — hybrid covers per-rotor demand
"""
function blades_selftest()::Bool
    all_pass = true
    function check(tag, cond, got, expected)
        s = cond ? "PASS" : "FAIL"
        println("  [$s] $tag: got=$(got)  expected=$(expected)")
        cond || (all_pass = false)
    end

    println("blades_selftest():")
    ρ   = 1.225
    R   = BG.R
    A   = π * R^2

    # T1 — hover kT in physical range
    kT0, _ = kT_kQ(0.0, ρ)
    check("T1 kT_prop(J=0)", 0.07 < kT0 < 0.14, round(kT0, digits=4), "(0.07, 0.14)")

    # T2 — hover thrust matches 4342 N target
    bc  = blade_coefficients(1250.0, 0.0, ρ)
    T_r = bc.thrust_N
    check("T2 T/rotor(1250RPM,J=0)", 3800.0 < T_r < 4900.0, round(T_r), "(3800, 4900) N")

    # T3 — CT criterion at J=0.10
    kT10, _ = kT_kQ(0.10, ρ)
    CT10    = kT10 * 4.0 / π^2
    check("T3 CT_heli(J=0.10)", 0.008 < CT10 < 0.015, round(CT10, digits=4), "(0.008, 0.015)")

    # T4 — no VRS in hover
    f0 = vrs_factor(0.0, T_r, ρ, A)
    check("T4 vrs_factor(vz=0)", f0 == 1.0, f0, "1.0")

    # T5 — VRS active at 8 m/s descent
    f8 = vrs_factor(-8.0, T_r, ρ, A)
    check("T5 vrs_factor(vz=-8)", f8 < 0.95, round(f8, digits=3), "< 0.95")

    # T6 — ElectricMotor ceiling > hover demand
    em  = ElectricMotor()
    Pel = motor_power_available_W(em, 1250.0, 0.0)
    check("T6 ElectricMotor P(SL)", Pel > 150_000.0, round(Pel/1000, digits=1), "> 150 kW")

    # T7 — TurboshaftEngine ceiling > full fleet demand
    ts  = TurboshaftEngine()
    Pts = motor_power_available_W(ts, 1250.0, 0.0)
    check("T7 TurboshaftEngine P(SL)", Pts > 500_000.0, round(Pts/1000, digits=1), "> 500 kW")

    # T8 — Hybrid ceiling per rotor
    hy  = HybridTurbineElectric(
              turbine = TurboshaftEngine(P_sl_W=400_000.0),
              motor   = ElectricMotor(P_max_W=250_000.0))
    Phy = motor_power_available_W(hy, 1250.0, 0.0)
    check("T8 Hybrid P(SL)", Phy > 150_000.0, round(Phy/1000, digits=1), "> 150 kW")

    println(all_pass ? "\nAll tests PASSED ✓" : "\nSome tests FAILED ✗")
    return all_pass
end

end # module Blades

#   Swap examples (not executed at include time) ───────────────────────────────
#=
# ──────────────────────────────────────────────────────────────
# All-electric fleet (default):
#   Uses module BG geometry + ElectricMotor(200 kW) per rotor.
# ──────────────────────────────────────────────────────────────
#   RP.motor = ElectricMotor()           # already the default

# ──────────────────────────────────────────────────────────────
# All-turboshaft fleet:
# ──────────────────────────────────────────────────────────────
#   RP.motor = TurboshaftEngine(P_sl_W=1_200_000.0)

# ──────────────────────────────────────────────────────────────
# All-hybrid fleet:
# ──────────────────────────────────────────────────────────────
#   RP.motor = HybridTurbineElectric(
#                  turbine = TurboshaftEngine(P_sl_W=400_000.0),
#                  motor   = ElectricMotor(P_max_W=250_000.0),
#                  E_batt_J = 3.6e8)

# ──────────────────────────────────────────────────────────────
# Mixed fleet — aft pair turboshaft, all others electric:
# ──────────────────────────────────────────────────────────────
#   ts = TurboshaftEngine(P_sl_W=600_000.0)
#   FLEET_MIXED = RotorFleet(Dict(
#       5 => RotorUnit(_default_unit(5); motor = ts),
#       6 => RotorUnit(_default_unit(6); motor = ts)))

# ──────────────────────────────────────────────────────────────
# Custom blade geometry (larger radius prototype):
# ──────────────────────────────────────────────────────────────
#   bg_proto = BladeGeometry(R=1.65, chord=0.13, pitch_offset_deg=4.0)
#   bc = blade_coefficients(1200.0, 0.0, 1.225; bg=bg_proto)

# ──────────────────────────────────────────────────────────────
# Run self-test:
# ──────────────────────────────────────────────────────────────
#   blades_selftest()
=#
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
#   §8  blades_selftest() — embedded validation (5 tests; motor backend tests
#                            moved to powerplant_selftest() in powerplant.jl)
#
# Motor/engine backends (AbstractMotor, ElectricMotor, TurboshaftEngine,
# HybridTurbineElectric, build_motor) have moved to powerplant.jl (2026-06-17).
# This file is BEM aerodynamics only and does not own powerplant types.
#
# Tilt-rotor advance-ratio convention
# ─────────────────────────────────────
# blade_coefficients() receives  vx_axial = vx_body * cos(tilt_rad).
#   At cruise (tilt = π/2): vx_axial ≈ 0 → J ≈ 0 → hover polar → correct thrust.
#   At hover  (tilt = 0):   vx_axial = vx_body → correct transition physics.
# Fixed-pitch blades go to zero thrust when J rises above ~0.15 — this is
# correct: the tiltrotor keeps J small by tilting the disk into the flow.
#
# Calibration targets (SL ISA, 1250 RPM, 5-blade, R=1.45 m — matches the
# calibrated stock rotor in rotor_config.csv rows 3-6):
#   J=0.00  → T ≈ 4 483 N/rotor, T_total ≈ 26 897 N  (T/W=1.26 at MTOW=2177 kg) ✓
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
# Defaults match the calibrated stock rotor (rotor_config.csv rows 3-6),
# not the original S4-class placeholder (R=1.524, n_blades=3, chord=0.12)
# this constructor shipped with. That placeholder is stale: any caller of
# blade_coefficients/kT_kQ that omits the `bg` keyword (see rotor_system.jl
# lines ~358, 569, 588) silently falls back to this default, so it must
# track whatever the real stock geometry is, not an old design point.
function BladeGeometry(;
        R               = 1.45,
        n_blades        = 5,
        chord           = 0.096,
        twist_root_deg  = 16.0,
        twist_tip_deg   =  6.0,
        pitch_offset_deg=  4.4)
    σ = n_blades * chord / (π * R)
    BladeGeometry(R, n_blades, chord, twist_root_deg, twist_tip_deg, pitch_offset_deg, σ)
end

""" Module-level default BladeGeometry (stock rotor, matches rotor_config.csv rows 3-6). """
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
#   Self-Test
# ─────────────────────────────────────────────────────────────────────────────
"""
    blades_selftest() → Bool

Runs 5 embedded validation checks covering geometry calibration and VRS.
Returns true if all pass; prints PASS/FAIL for each. Motor backend tests
(formerly T6–T8) have moved to powerplant_selftest() in powerplant.jl.

No external dependencies — safe to call at module load or in CI.

Expected at SL ISA (ρ = 1.225 kg/m³), R = 1.45 m, 5-blade, 1250 RPM:
  T1  kT_prop(J=0) ∈ (0.07, 0.14)            — coefficient in physical range
  T2  T/rotor(J=0) ∈ (3800, 4900) N           — matches 4483 N hover target
  T3  CT_helicopter(J=0.10) ∈ (0.008, 0.015)  — mild forward-flight criterion
  T4  vrs_factor(vz=0) = 1.0                  — no VRS in hover
  T5  vrs_factor(vz=-8 m/s) < 0.95            — VRS active at moderate descent
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

    # T2 — hover thrust matches 4483 N target
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

    println(all_pass ? "\nAll tests PASSED ✓" : "\nSome tests FAILED ✗")
    return all_pass
end

end # module Blades

#   Swap examples (not executed at include time) ───────────────────────────────
#=
# ──────────────────────────────────────────────────────────────
# Motor/engine swap examples (ElectricMotor, TurboshaftEngine,
# HybridTurbineElectric) have moved to powerplant.jl — see the
# swap-examples block at the bottom of that file.
# ──────────────────────────────────────────────────────────────

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

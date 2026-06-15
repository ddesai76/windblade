# landing_gear.jl:    Landing gear and contact physics
# AUTHOR:             DANIEL DESAI
# UPDATED:            2026-05-10
# VERSION:            0.1.0
#
# Three-point compliant strut model
# Provides normal contact force, longitudinal friction, and ground-state
# flags for the ODE, saving callback, and cockpit.
#
# Architecture
# ─────────────
# Layer:   subsystems/ — pure physics, no mission knowledge, no PIDs.
# Depends: atmosphere.jl (rho, for tyre friction scaling — optional)
#          airframe.jl   (AP.mass_kg, WEIGHT_N — for defaults only)
#
# ODE integration strategy
# ─────────────────────────
# The strut spring force (Fz_spring) is ForwardDiff-safe and enters
# build_ode directly, replacing the hard floor `dalt = 0` guard.
#
# The strut damping force requires dalt (= du[2]), which cannot be read
# while it is being written. Damping is therefore split:
#   • In build_ode: spring term only → drives the ODE trajectory.
#   • In saving callback: full spring + velocity damping via finite
#     difference of consecutive alt values → used for strut_load_n
#     (cockpit, CSV) only. Pure Float64, no Dual constraint.
#
# This matches the pattern already used for rotor RPM back-calculation.
#
# Geometry
# ─────────
# alt (u[2]) is the aircraft CG height AGL in metres.
# The gear extends `cg_to_ground_m` below the CG when unloaded.
# Contact begins when alt < cg_to_ground_m (penetration δ > 0).
#
# Three-point layout (nose + two main):
#   Nose gear  : x_n =  +3.2 m forward of CG, on centreline
#   Main-L     : x_m =  -1.1 m aft  of CG, y = +2.1 m
#   Main-R     : x_m =  -1.1 m aft  of CG, y = -2.1 m
# These produce a realistic pitch-down moment at nose contact and a
# roll-stabilising base. The strut force is split 30/35/35 (nose/L/R)
# for load reporting; the aggregate Fz is what matters for the ODE.
#
# Friction model
# ──────────────
# Longitudinal friction uses a smooth Coulomb approximation:
#   Fx_friction = -Fz_normal · µ · tanh(vx / v_blend)
# tanh blends from static (near-zero vx) to kinetic without a
# discontinuity — important for ForwardDiff and solver step control.
# v_blend = 0.3 m/s is the crossover speed (~1 km/h).
#
# Brake input (0–1) scales µ between µ_roll and µ_brake.
# At brake=0 (free-roll) friction is rolling resistance only.
# At brake=1 the tyre is at the braking limit.
#
# ForwardDiff safety
# ──────────────────
# contact_spring() uses only elementary arithmetic and clamp/max on
# ODE state variables — safe with AutoFiniteDiff or ForwardDiff.
# contact_full() is Float64-only (saving callback / postprocess).
# Never call contact_full() from build_ode.
#
# Integration into fly.jl
# ────────────────────────
# See "fly.jl integration" at the bottom of this file.
#
# Depends on: airframe.jl (AP.mass_kg, G, WEIGHT_N)


# ── Parameters ────────────────────────────────────────────────────────

"""
    LandingGear

Physical parameters for the three-point gear system.
All defaults are sized for the Joby S4-class (2177 kg MTOW).

Fields
──────
- `cg_to_ground_m` : distance from CG to unloaded tyre contact (m).
                     At rest on flat ground, alt (u[2]) = cg_to_ground_m.
                     Penetration δ = cg_to_ground_m - alt.
- `k_strut`        : combined strut spring rate, all three legs (N/m).
                     Sized so MTOW produces ~0.10 m static deflection:
                     k = WEIGHT_N / 0.10 ≈ 214 000 N/m → round to 220 000.
- `c_strut`        : combined strut damping (N·s/m).
                     Critical damping ratio ζ ≈ c / (2√(m·k)) ≈ 0.5.
                     c = 2 · 0.5 · √(2177 · 220000) ≈ 14 000 N·s/m.
- `mu_roll`        : rolling friction coefficient (free-roll).
- `mu_brake`       : peak braking friction coefficient (full brake pedal).
- `mu_static`      : static friction limit (prevents drift at rest).
- `v_blend_ms`     : tanh crossover speed (m/s). Below this the tyre
                     behaves statically; above it kinetically.
- `nose_frac`      : fraction of Fz carried by the nose leg (for load
                     reporting only; ODE uses aggregate Fz).
"""
Base.@kwdef struct LandingGear
    cg_to_ground_m :: Float64 = 0.72     # CG height above ground when parked (m)
    k_strut        :: Float64 = 220_000.0 # spring rate, all legs combined (N/m)
    c_strut        :: Float64 =  25_000.0 # damping, all legs combined (N·s/m)
    # Increased from 14000: old value gave gz=6.6g at 4 m/s sink;
    # 25000 targets gz<3g at 2 m/s sink (OVERHEAD arrest should limit entry speed)
    mu_roll        :: Float64 =    0.020  # free-rolling tyre friction
    mu_brake       :: Float64 =    0.450  # braking tyre friction (dry tarmac)
    mu_static      :: Float64 =    0.600  # static friction limit
    v_blend_ms     :: Float64 =    0.30   # Coulomb crossover speed (m/s)
    nose_frac      :: Float64 =    0.30   # nose leg share of total Fz (display)
end

# Default instance — override in test_card.jl if needed
const GEAR = LandingGear()

# ── ODE-safe contact (spring only, no damping) ────────────────────────

"""
    contact_spring(gear, alt) → Fz_spring

Spring component of the strut normal force. ForwardDiff-safe.

Returns the upward contact force (N) due to strut compression alone.
Zero when airborne (alt ≥ cg_to_ground_m).

Use this in build_ode to add to Fz before computing dalt.
"""
function contact_spring(gear::LandingGear, alt)
    δ = gear.cg_to_ground_m - alt          # penetration depth (m); + when on ground
    return gear.k_strut * max(δ, zero(δ))  # one-sided: no tension
end

"""
    contact_active(gear, alt) → Bool

Returns true when the gear is in contact with the ground.
Pure altitude comparison — safe for use anywhere.
"""
contact_active(gear::LandingGear, alt::Real) =
    alt < gear.cg_to_ground_m

# ── Full contact model (Float64 only — saving callback / postprocess) ──

"""
    contact_full(gear, alt, alt_prev, dt_cb, vx, brake) →
        (Fz_normal, Fx_friction, gear_contact, strut_load_n)

Full strut model including velocity damping. Float64 only — do NOT call
from build_ode (Dual number constraint).

Arguments
─────────
- `alt`       : current CG altitude AGL (m), from u[2]
- `alt_prev`  : altitude at previous callback step (m), for vz estimate
- `dt_cb`     : callback timestep (s)
- `vx`        : forward speed (m/s), from u[1]
- `brake`     : brake command [0, 1]; 0 = free-roll, 1 = full brake

Returns
───────
- `Fz_normal`   : total upward contact force (N); 0 when airborne
- `Fx_friction` : longitudinal friction force (N); negative opposes +vx
- `gear_contact`: true when any strut is loaded
- `strut_load_n`: total strut compressive load (N) for cockpit display

Notes
─────
Vertical speed vz is estimated as (alt - alt_prev) / dt_cb — a
backward finite difference. This is sufficient at callback rate (~10 Hz)
and avoids storing vz as an ODE state.
"""
function contact_full(gear::LandingGear,
                      alt::Float64, alt_prev::Float64, dt_cb::Float64,
                      vx::Float64,  brake::Float64 = 0.0) ::
         Tuple{Float64, Float64, Bool, Float64}

    δ = gear.cg_to_ground_m - alt
    if δ <= 0.0
        return (0.0, 0.0, false, 0.0)
    end

    # Descent rate (positive = sinking). Guard against dt_cb = 0.
    vz = (alt_prev - alt) / max(dt_cb, 1e-4)   # sinking positive

    # Spring + damper: damping opposes vz when compressing (vz > 0)
    Fz_spring = gear.k_strut * δ
    Fz_damp   = gear.c_strut * max(vz, 0.0)     # compression damping only
    Fz_normal = max(Fz_spring + Fz_damp, 0.0)   # strut cannot pull

    # Effective friction coefficient — blend roll ↔ brake
    brake_c  = clamp(brake, 0.0, 1.0)
    mu_eff   = gear.mu_roll + brake_c * (gear.mu_brake - gear.mu_roll)

    # Coulomb friction with tanh crossover (smooth, no discontinuity)
    Fx_friction = -Fz_normal * mu_eff * tanh(vx / gear.v_blend_ms)

    strut_load = Fz_normal   # could split by nose_frac for per-leg display later

    return (Fz_normal, Fx_friction, true, strut_load)
end

# ── Convenience: spring-only Fx (for ODE ground roll) ────────────────

"""
    ground_friction(gear, alt, vx, brake) → Fx_friction

Friction force for use in build_ode. Uses the spring-only Fz estimate
(no damping) so it is ForwardDiff-safe. Adequate for the ODE trajectory;
contact_full gives the more accurate damped value in the callback.
"""
function ground_friction(gear::LandingGear, alt, vx, brake::Float64 = 0.0)
    Fz = contact_spring(gear, alt)
    mu_eff = gear.mu_roll + clamp(brake, 0.0, 1.0) *
             (gear.mu_brake - gear.mu_roll)
    return -Fz * mu_eff * tanh(vx / gear.v_blend_ms)
end

# ── Self-test ─────────────────────────────────────────────────────────

"""
    gear_selftest(; gear)

Sanity checks for the landing gear model. Run from the REPL:

    julia> include("subsystems/landing_gear.jl")
    julia> gear_selftest()
"""
function gear_selftest(; gear::LandingGear = GEAR)
    println("\n=== Landing Gear Self-Test ===\n")
    pass = Bool[]

    # Reference values (must match airframe.jl AP)
    mass_kg  = 2177.0
    G        = 9.80665
    weight_n = mass_kg * G   # ≈ 21 357 N

    # ── Test 1: no contact when airborne ─────────────────────────────
    Fz, Fx, contact, load = contact_full(gear, gear.cg_to_ground_m + 0.1,
                                          gear.cg_to_ground_m + 0.1, 0.1, 0.0)
    ok1 = !contact && Fz == 0.0 && Fx == 0.0
    push!(pass, ok1)
    println("Test 1 — Airborne (alt = cg + 0.1 m):")
    println("  Fz=$(Fz) N  Fx=$(Fx) N  contact=$(contact)  $(ok1 ? "✓ PASS" : "✗ FAIL")")
    println()

    # ── Test 2: static equilibrium — Fz ≈ WEIGHT_N ───────────────────
    # At rest: δ = k_strut·δ_static = WEIGHT_N → δ_static = WEIGHT_N/k_strut
    δ_static = weight_n / gear.k_strut
    alt_static = gear.cg_to_ground_m - δ_static
    Fz2, _, contact2, _ = contact_full(gear, alt_static, alt_static, 0.1, 0.0)
    ok2 = contact2 && abs(Fz2 - weight_n) < 1.0
    push!(pass, ok2)
    @printf("Test 2 — Static equilibrium (δ=%.3f m, alt=%.4f m):\n",
            δ_static, alt_static)
    @printf("  Fz=%.1f N  WEIGHT_N=%.1f N  err=%.2f N  %s\n",
            Fz2, weight_n, abs(Fz2-weight_n), ok2 ? "✓ PASS" : "✗ FAIL")
    println()

    # ── Test 3: damping opposes sinking, not rising ───────────────────
    # Sinking at 1 m/s: alt falls → alt_prev > alt → vz > 0 → extra Fz
    alt_now  = alt_static
    alt_sink = alt_now - 0.01   # sank 1 cm in 0.01 s → vz = 1 m/s
    Fz_sink, _, _, _ = contact_full(gear, alt_sink, alt_now, 0.01, 0.0)
    # Rising at 1 m/s: alt rises → alt_prev < alt → vz < 0 → no extra Fz
    alt_rise = alt_now + 0.01
    Fz_rise, _, _, _ = contact_full(gear, alt_rise, alt_now, 0.01, 0.0)
    ok3 = Fz_sink > Fz_rise   # sinking adds damping force, rising does not
    push!(pass, ok3)
    @printf("Test 3 — Damping asymmetry:\n")
    @printf("  Fz_sink=%.0f N  Fz_rise=%.0f N  sink>rise: %s\n",
            Fz_sink, Fz_rise, ok3 ? "✓ PASS" : "✗ FAIL")
    println()

    # ── Test 4: free-roll friction direction ──────────────────────────
    # Moving forward (+vx) → friction opposes → Fx < 0
    _, Fx_fwd, _, _ = contact_full(gear, alt_static, alt_static, 0.1, 5.0)
    # Moving backward (-vx) → friction opposes → Fx > 0
    _, Fx_rev, _, _ = contact_full(gear, alt_static, alt_static, 0.1, -5.0)
    ok4 = Fx_fwd < 0.0 && Fx_rev > 0.0
    push!(pass, ok4)
    @printf("Test 4 — Friction direction (free-roll, vx=±5 m/s):\n")
    @printf("  Fx(+vx)=%.1f N  Fx(-vx)=%.1f N  %s\n",
            Fx_fwd, Fx_rev, ok4 ? "✓ PASS" : "✗ FAIL")
    println()

    # ── Test 5: brake increases friction force ────────────────────────
    _, Fx_roll,  _, _ = contact_full(gear, alt_static, alt_static, 0.1, 5.0, 0.0)
    _, Fx_brake, _, _ = contact_full(gear, alt_static, alt_static, 0.1, 5.0, 1.0)
    ok5 = abs(Fx_brake) > abs(Fx_roll)
    push!(pass, ok5)
    @printf("Test 5 — Brake increases friction:\n")
    @printf("  Fx(roll)=%.1f N  Fx(full brake)=%.1f N  %s\n",
            Fx_roll, Fx_brake, ok5 ? "✓ PASS" : "✗ FAIL")
    println()

    # ── Test 6: contact_spring matches contact_full (no damping, vz=0) ─
    Fz_spring = contact_spring(gear, alt_static)
    Fz_full, _, _, _ = contact_full(gear, alt_static, alt_static, 0.1, 0.0)
    ok6 = abs(Fz_spring - Fz_full) < 1.0   # should be identical when vz=0
    push!(pass, ok6)
    @printf("Test 6 — Spring-only vs full (vz=0):\n")
    @printf("  contact_spring=%.1f N  contact_full=%.1f N  %s\n",
            Fz_spring, Fz_full, ok6 ? "✓ PASS" : "✗ FAIL")
    println()

    println("=== $(count(pass))/$(length(pass)) tests passed ===\n")
    return all(pass)
end



# powerplant.jl:    Propulsor model (AbstractMotor hierarchy)
# AUTHOR:           DANIEL DESAI
# UPDATED:          2026-06-17
# VERSION:          0.1.0
#
#
# Depends on: fuel.jl (fuel chemistry + tank capacity — see that file's
# header for the fuel.jl / powerplant.jl boundary). TurboshaftEngine holds
# a FuelTank rather than bare fuel_kg/SFC fields, so adjusting energy
# density or tank capacity never requires touching this file.
#
# Implements:
#
#   §1  AbstractMotor          — supertype, mandatory interface contract
#   §2  ElectricMotor          — PMSM with altitude + thermal de-rating
#   §3  TurboshaftEngine       — turbogenerator core, Gagg-Ferrar lapse, fuel burn
#   §4  HybridTurbineElectric  — series hybrid (turbine+battery+motor), Mode 2/3 only
#   §5  build_motor()          — generic factory
#   §6  powerplant_selftest()  — embedded validation
#
# Design note — no gearbox:
#   The WINDBLADE turbogenerator (R1/R2 prototype) has NO mechanical gearbox
#   anywhere in the power path. The turbine drives an integrated PM generator
#   directly on its own shaft (no step-down to rotor speed); the rotor is
#   driven by a separate motor with its own inverter. TurboshaftEngine here
#   therefore has no eta_gearbox field — that field existed in the original
#   blades.jl draft and implied a conventional helicopter-style mechanical
#   output shaft, which does not describe this architecture. If a future
#   config genuinely has a mechanical reduction gearbox, add the loss term
#   back as an explicit eta_shaft field on that specific motor instance
#   rather than reintroducing it as a blanket default.
#
# Design note — altitude lapse vs fuel exhaustion (read before touching
# motor_power_available_W or burn_fuel!):
#   The Gagg-Ferrar altitude lapse on TurboshaftEngine is large enough to
#   matter operationally. At 3,500 m (KAXX-class density altitude) it
#   reduces available shaft power to ~62.7% of the sea-level rating — for
#   the 746 kW design point, that is a ceiling around 468 kW, not 746 kW.
#   This is an air-density effect, completely independent of how much fuel
#   remains in the tank. The two failure modes look similar from a rotor's
#   perspective (it isn't getting the power it asked for) but have
#   different causes and different correct responses:
#     - Fuel exhausted (tank.mass_kg == 0): the turbine cannot run at all.
#     - Altitude lapse (tank.mass_kg > 0, but ceiling < demand): the
#       turbine is running fine, it just physically cannot produce more
#       shaft power at this air density.
#   rotor_system.jl's fleet_fuel_burn! checks tank.mass_kg directly for the
#   first case (full fallback to battery) and separately computes a
#   shortfall_kw = max(demand - ceiling, 0) for the second case (partial
#   top-up from battery, turbine still supplies what it can). Do not
#   conflate these by using motor_power_available_W's return value as a
#   fuel-exhaustion proxy — it was never designed to signal that, and an
#   earlier draft of fleet_fuel_burn! had exactly this bug (caught in
#   validation, fixed 2026-06-17).
#
# Mode 1 (thru-air hybrid, Phase 1 target) usage:
#   R1/R2 use TurboshaftEngine directly as their `motor` field — fully
#   independent of the aircraft battery, exactly per Mode 1 spec. Do NOT
#   wrap them in HybridTurbineElectric for Phase 1; that type couples to a
#   battery and models Mode 2 (charging)/Mode 3 (all-electric) territory.
#
#   tank = FuelTank(45.0, 24.4)   # 45 L capacity, standard-mission load
#   ts   = TurboshaftEngine(tank = tank)
#   FLEET_TE = RotorFleet(Dict(1 => RotorUnit(_default_unit(1); motor=ts),
#                              2 => RotorUnit(_default_unit(2); motor=ts)))
#
# Mandatory interface (implement for every new backend):
#   motor_power_available_W(m, rpm, alt_m)         → Float64  (ceiling W)
#   motor_shaft_power_W(m, rpm, torque_Nm, alt_m)  → Float64  (actual W)
#
# Optional (implement for state-tracking backends):
#   update_motor_thermal!(m, P_W, dt_s, T_amb_C)   — electric winding heat
#   burn_fuel!(m, P_W, dt_s)                        — turboshaft fuel
#   update_batt_soc!(m, P_W, dt_s, alt_m)           — hybrid battery SoC

module Powerplant

include("fuel.jl")
using .Fuel

export AbstractMotor, ElectricMotor, TurboshaftEngine, HybridTurbineElectric,
       build_motor, motor_power_available_W, motor_shaft_power_W,
       update_motor_thermal!, burn_fuel!, update_batt_soc!,
       powerplant_selftest,
       FuelProperties, JET_A, SAF, FuelTank,
       fuel_mass_for_energy, fuel_energy_for_mass,
       refuel!, fuel_fraction, fuel_capacity_L, fuel_mass_L

# ─────────────────────────────────────────────────────────────────────────────
# §1  AbstractMotor
# ─────────────────────────────────────────────────────────────────────────────
"""Supertype for all motor/engine backends."""
abstract type AbstractMotor end

const _RHO_SL = 1.225   # ISA SL density used internally

# ─────────────────────────────────────────────────────────────────────────────
# §2  Electric Motor
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# §3  Turboshaft Engine (turbogenerator core — no mechanical gearbox)
# ─────────────────────────────────────────────────────────────────────────────
"""
    TurboshaftEngine

Gas turbine power core with Gagg–Ferrar altitude lapse. Despite the name
(retained for interface compatibility with existing callers), this models
the WINDBLADE turbogenerator: a turbine driving an integrated generator on
its own shaft, with no mechanical output gearbox. The motor at the rotor
end is a separate ElectricMotor with its own inverter — RPM matching
between turbine and rotor is handled electrically, not mechanically.
See module header for the no-gearbox rationale.

    P(alt) = P_sl × (ρ_alt / ρ_sl)^n_lapse

Lapse exponent n_lapse:
  1.132 — Gagg–Ferrar turboprop (default, standard for S4-class)
  1.0   — pure density-proportional (conservative)
  0.7   — turbofan-like (better altitude performance)

Fuel chemistry lives in `tank::FuelTank` (fuel.jl) — NOT on this struct.
Adjusting energy density, fuel type, or tank capacity is a fuel.jl change
only; this struct only needs `eta_thermal` (how much of the fuel's energy
becomes shaft power). SFC is therefore DERIVED, not a free constant:

    SFC (kg/J) = 1 / (eta_thermal × energy_density_J_per_kg)

Defaults: eta_thermal = 1/3 exactly reproduces the Rev 0.3 design-point
SFC of 0.25 kg/(kW·h) at Jet-A's 43.2 MJ/kg LHV. P_sl_W = 746 kW (1000 hp)
is the Rev 0.3 shaft power design point.

Phase 1 (Mode 1) usage — R1/R2 prototype, 45 L shared tank:
    tank = FuelTank(45.0, 24.4)   # capacity_L, initial_L — see fuel.jl
    ts   = TurboshaftEngine(tank = tank)
    FLEET_TE = RotorFleet(Dict(1 => RotorUnit(_default_unit(1); motor=ts),
                               2 => RotorUnit(_default_unit(2); motor=ts)))
Both rotors share `ts`, and `ts.tank` is the same FuelTank instance, so
fuel depletes from one shared 45 L pool regardless of which rotor's
dispatch calls burn_fuel!.
"""
Base.@kwdef mutable struct TurboshaftEngine <: AbstractMotor
    P_sl_W      ::Float64  = 746_000.0          # sea-level shaft power (W) — 1000 hp design point
    n_lapse     ::Float64  = 1.132              # Gagg–Ferrar lapse exponent
    eta_thermal ::Float64  = 1.0/3.0            # shaft energy / fuel energy — gives 0.25 kg/(kWh) at Jet-A LHV
    tau_spool_s ::Float64  = 2.0                # spool-up time constant (s)
    tank        ::FuelTank = FuelTank(45.0, 24.4)  # 45 L capacity, standard-mission default load
end

"""
    motor_power_available_W(e::TurboshaftEngine, rpm, alt_m) → Float64

Gagg–Ferrar lapse.  Uses exponential atmosphere (accurate to <2% below 12 km).
No gearbox efficiency term — see module header. Does NOT check fuel
remaining — see burn_fuel! and rotor_system.jl's fleet_fuel_burn! for why
fuel exhaustion is checked on tank.mass_kg directly, not via this ceiling.
"""
function motor_power_available_W(e::TurboshaftEngine, rpm::Float64, alt_m::Float64)::Float64
    ρ     = _RHO_SL * exp(-alt_m / 8500.0)
    lapse = (ρ / _RHO_SL)^e.n_lapse
    return e.P_sl_W * lapse
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

Decrement e.tank's fuel load. Converts shaft energy demand to fuel energy
demand via eta_thermal, then asks fuel.jl how many kg that requires at
this tank's chemistry. Call from saving callback only. Clamps at zero —
never goes negative (enforced by FuelTank semantics in fuel.jl). Caller
(rotor_system.jl's fleet_fuel_burn!) is responsible for falling back to
battery draw, or zero thrust, once tank.mass_kg reaches zero — check
e.tank.mass_kg directly, not this function's return value, since a return
of 0.0 here could also mean P_shaft_W was 0.0.
"""
function burn_fuel!(e::TurboshaftEngine, P_shaft_W::Float64, dt_s::Float64)::Float64
    e.tank.mass_kg <= 0.0 && return 0.0
    shaft_energy_J = max(P_shaft_W, 0.0) * dt_s
    fuel_energy_J  = shaft_energy_J / e.eta_thermal
    demanded_kg    = fuel_mass_for_energy(e.tank, fuel_energy_J)
    burned         = min(demanded_kg, e.tank.mass_kg)
    e.tank.mass_kg -= burned
    return burned
end

# ─────────────────────────────────────────────────────────────────────────────
# §4  Hybrid Turbine-Electric (Mode 2 / Mode 3 — NOT used in Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
"""
    HybridTurbineElectric

Series-hybrid: turbine → generator → battery bus → electric motor → shaft.
Couples a turbine to the AIRCRAFT battery via SoC tracking. This is the
correct model for Mode 2 (charging — turbine exports to shared battery
bus) and Mode 3 (all-electric — battery-only, turbine idle), but it is
explicitly NOT used for the Phase 1 / Mode 1 (thru-air hybrid) prototype,
where R1/R2 must be electrically independent of the aircraft battery.

Power budget:
    P_available = min(P_motor_ceiling, (P_gen + P_batt_discharge) × η_inverter)

Battery:
    SoC-limited discharge at up to P_batt_max_W.
    Regenerative charging during descent when P_rotor_demand < P_gen.
    SoC ∈ [SoC_min, 1.0] (hard reserve floor prevents damage).

Deferred to Phase 4/5 (roadmap). Retained here, tested, and ready —
do not delete.
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

# ─────────────────────────────────────────────────────────────────────────────
# §5  Motor Factory
# ─────────────────────────────────────────────────────────────────────────────
"""
    build_motor(type::Symbol; kwargs...) → AbstractMotor

Generic motor factory.  Unknown kwargs are silently ignored (safe to pass a
uniform config dict regardless of which backend is active).

Types
─────
  :electric   → ElectricMotor
  :turboshaft → TurboshaftEngine   (Phase 1 / Mode 1 — turbogenerator core)
  :hybrid     → HybridTurbineElectric   (Mode 2/3 — NOT Phase 1)
               (pass `turbine=` and `motor=` as keyword args)

Examples
────────
    build_motor(:electric)
    build_motor(:turboshaft)                          # 746 kW design-point defaults
    build_motor(:turboshaft; P_sl_W=600_000.0)         # custom rating
    build_motor(:turboshaft; tank=FuelTank(90.0, 60.0)) # custom tank (fuel.jl)
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
        valid = (:P_sl_W, :n_lapse, :eta_thermal, :tau_spool_s, :tank)
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
# §6  Self-Test
# ─────────────────────────────────────────────────────────────────────────────
"""
    powerplant_selftest() → Bool

Embedded validation for the motor backend hierarchy. Returns true if all
pass; prints PASS/FAIL for each. No external dependencies — safe to call
at module load or in CI.

Expected:
  T1  ElectricMotor P_avail > 150 kW at SL          — covers 141 kW hover demand
  T2  TurboshaftEngine P_avail ≈ 746 kW at SL        — matches Rev 0.3 design point
  T3  TurboshaftEngine SFC matches 0.25 kg/(kW·h)    — derived from eta_thermal, fuel.jl arithmetic
  T4  TurboshaftEngine fuel_kg clamps at zero        — never goes negative
  T5  TurboshaftEngine altitude lapse reduces power  — Gagg-Ferrar sanity at 3500m
  T6  HybridTurbineElectric P_avail > 150 kW         — Mode 2/3 path still functional
"""
function powerplant_selftest()::Bool
    all_pass = true
    function check(tag, cond, got, expected)
        s = cond ? "PASS" : "FAIL"
        println("  [$s] $tag: got=$(got)  expected=$(expected)")
        cond || (all_pass = false)
    end

    println("powerplant_selftest():")

    # T1 — ElectricMotor ceiling > hover demand
    em  = ElectricMotor()
    Pel = motor_power_available_W(em, 1250.0, 0.0)
    check("T1 ElectricMotor P(SL)", Pel > 150_000.0, round(Pel/1000, digits=1), "> 150 kW")

    # T2 — TurboshaftEngine matches 746 kW design point at SL
    ts  = TurboshaftEngine()
    Pts = motor_power_available_W(ts, 1250.0, 0.0)
    check("T2 TurboshaftEngine P(SL)", abs(Pts - 746_000.0) < 1.0,
          round(Pts/1000, digits=1), "746.0 kW")

    # T3 — SFC arithmetic: burning at 746 kW for 1 hour should consume
    # 746 kW * 0.25 kg/(kW.h) = 186.5 kg, via eta_thermal -> fuel.jl, not a
    # free SFC constant. Plenty of fuel in this tank for the check.
    ts2 = TurboshaftEngine(tank = FuelTank(2000.0, 1000.0))
    burned_1h = burn_fuel!(ts2, 746_000.0, 3600.0)
    check("T3 SFC 1h burn at 746kW", abs(burned_1h - 186.5) < 0.5,
          round(burned_1h, digits=2), "186.5 kg")

    # T4 — fuel clamps at zero, never negative
    ts3 = TurboshaftEngine(tank = FuelTank(45.0, 0.01))
    burned_clamp = burn_fuel!(ts3, 746_000.0, 100.0)  # demand far exceeds tiny tank
    check("T4 fuel clamp at zero", ts3.tank.mass_kg == 0.0,
          ts3.tank.mass_kg, "0.0 (no negative)")

    # T5 — altitude lapse reduces available power at 3500m vs SL
    P_sl3500   = motor_power_available_W(ts, 1250.0, 0.0)
    P_alt3500  = motor_power_available_W(ts, 1250.0, 3500.0)
    check("T5 altitude lapse at 3500m", P_alt3500 < P_sl3500,
          round(P_alt3500/1000, digits=1), "< $(round(P_sl3500/1000, digits=1)) kW")

    # T6 — Hybrid ceiling per rotor (Mode 2/3 path still functional)
    hy  = HybridTurbineElectric(
              turbine = TurboshaftEngine(P_sl_W=400_000.0),
              motor   = ElectricMotor(P_max_W=250_000.0))
    Phy = motor_power_available_W(hy, 1250.0, 0.0)
    check("T6 Hybrid P(SL)", Phy > 150_000.0, round(Phy/1000, digits=1), "> 150 kW")

    println(all_pass ? "\nAll tests PASSED ✓" : "\nSome tests FAILED ✗")
    return all_pass
end

end # module Powerplant

#   Swap examples (not executed at include time) ───────────────────────────────
#=
# ──────────────────────────────────────────────────────────────
# All-electric fleet (default):
# ──────────────────────────────────────────────────────────────
#   RP.motor = ElectricMotor()           # already the default

# ──────────────────────────────────────────────────────────────
# R1/R2 turbogenerator prototype (Mode 1, Phase 1 target):
# Tank capacity/initial load are mission-variable — see
# mission_planner.jl fuel_required_L(). 45 L / 24.4 kg are the
# standard-mission defaults if FuelTank() is called with no args.
# ──────────────────────────────────────────────────────────────
#   tank = FuelTank(45.0, 24.4)   # capacity_L, initial_L (fuel.jl)
#   ts   = TurboshaftEngine(tank = tank)   # 746 kW, eta=1/3 design point
#   FLEET_TE = RotorFleet(Dict(
#       1 => RotorUnit(_default_unit(1); motor = ts),
#       2 => RotorUnit(_default_unit(2); motor = ts)))

# ──────────────────────────────────────────────────────────────
# Same prototype on 100% SAF instead of Jet-A — drop-in, no other
# changes needed anywhere in powerplant.jl or rotor_system.jl:
# ──────────────────────────────────────────────────────────────
#   tank_saf = FuelTank(45.0, 24.4; properties = SAF)
#   ts_saf   = TurboshaftEngine(tank = tank_saf)

# ──────────────────────────────────────────────────────────────
# All-6 turbogenerator expansion (Phase 6, future) — note each
# pair would want its own FuelTank instance (or one large shared
# tank across all 6, per fuel_tank_id in rotor_system.jl):
# ──────────────────────────────────────────────────────────────
#   tank6 = FuelTank(135.0, 90.0)   # 3x capacity for 6 units
#   ts6   = TurboshaftEngine(tank = tank6)
#   FLEET_TE6 = RotorFleet(Dict(i => RotorUnit(_default_unit(i); motor = ts6) for i in 1:6))

# ──────────────────────────────────────────────────────────────
# Mode 2/3 — hybrid with shared aircraft battery (deferred):
# ──────────────────────────────────────────────────────────────
#   hy = HybridTurbineElectric(
#            turbine = TurboshaftEngine(tank = FuelTank(45.0, 24.4)),
#            motor   = ElectricMotor(P_max_W=250_000.0),
#            E_batt_J = 3.6e8)
#   FLEET_HYBRID = RotorFleet(Dict(
#       1 => RotorUnit(_default_unit(1); motor = hy),
#       2 => RotorUnit(_default_unit(2); motor = hy)))

# ──────────────────────────────────────────────────────────────
# Run self-test:
# ──────────────────────────────────────────────────────────────
#   powerplant_selftest()
=#
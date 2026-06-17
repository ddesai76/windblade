# fuel.jl:         Fuel chemistry and tank capacity model
# AUTHOR:           DANIEL DESAI
# UPDATED:          2026-06-17
# VERSION:          0.1.0
#
# Owns fuel PROPERTIES (energy density, density) and tank CAPACITY
# bookkeeping. Does NOT own engine thermodynamics — thermal efficiency,
# altitude lapse, and shaft power ceilings live in powerplant.jl on
# TurboshaftEngine. This separation means changing fuel chemistry (energy
# density, blend ratio, a future SAF/ethanol mix) never requires touching
# the engine model, and changing engine efficiency never requires touching
# fuel properties.
#
# Boundary, concretely:
#   fuel.jl       — "how much energy is in this much fuel, and how much
#                    fuel is left in the tank"
#   powerplant.jl — "how much of that energy becomes shaft power, and how
#                    does that ceiling change with altitude"
#
# burn_fuel!() in powerplant.jl calls fuel_mass_for_energy() here to convert
# a shaft-energy demand into a fuel mass, given the engine's own
# eta_thermal. fuel.jl has no knowledge of turbines, rotors, or shaft power
# — it only converts between energy, mass, and volume.
#
# Future fuel chemistry (deferred, not Phase 1):
#   SAF blends, ethanol/Jet-A blends — see WINDBLADE turbine-electric
#   roadmap §07 for the energy density / volume tradeoff tables already
#   worked out for SAF, LNG, LH2, ammonia, and ethanol blends. When one of
#   those is implemented, it becomes a new FuelProperties entry below;
#   nothing in powerplant.jl changes.

module Fuel

export FuelProperties, JET_A, SAF,
       FuelTank, fuel_mass_for_energy, fuel_energy_for_mass,
       refuel!, fuel_fraction, fuel_capacity_L, fuel_mass_L

# ─────────────────────────────────────────────────────────────────────────────
# §1  Fuel Properties
# ─────────────────────────────────────────────────────────────────────────────
"""
    FuelProperties

Chemistry constants for one fuel type. Adjust these to retune energy
density or density without touching any engine code.
"""
struct FuelProperties
    name             ::String
    energy_density_J_per_kg ::Float64   # lower heating value
    density_kg_per_L ::Float64          # at 15°C
end

"""Jet-A, lower heating value 43.2 MJ/kg, density 0.800 kg/L at 15°C."""
const JET_A = FuelProperties("jet_a", 43.2e6, 0.800)

"""100% SAF (ASTM D7566), LHV 42.8 MJ/kg, density 0.790 kg/L. Drop-in —
same tank, same engine, no powerplant.jl changes needed to use this."""
const SAF = FuelProperties("saf", 42.8e6, 0.790)

# ─────────────────────────────────────────────────────────────────────────────
# §2  Fuel Tank
# ─────────────────────────────────────────────────────────────────────────────
"""
    FuelTank

Capacity and current fuel mass for one tank. `properties` selects the
fuel chemistry (JET_A, SAF, or a future custom blend). Multiple engines
may share one FuelTank instance — see rotor_system.jl's shared
fuel_tank_id mechanism — in which case calls from either engine deplete
the same underlying mass_kg.
"""
mutable struct FuelTank
    properties ::FuelProperties
    capacity_kg ::Float64
    mass_kg     ::Float64
end

"""
    FuelTank(capacity_L, initial_L; properties=JET_A)

Construct a tank from litres. Mission-variable — pass whatever
`fuel_required_L()` (mission_planner.jl) computes for the planned route,
not a fixed constant.
"""
function FuelTank(capacity_L::Real, initial_L::Real; properties::FuelProperties = JET_A)
    cap_kg  = capacity_L * properties.density_kg_per_L
    init_kg = initial_L  * properties.density_kg_per_L
    FuelTank(properties, cap_kg, init_kg)
end

"""
    fuel_mass_for_energy(tank, energy_J) → kg

Mass of fuel (at this tank's chemistry) needed to release `energy_J`
joules of fuel energy. Does not mutate the tank or know about thermal
efficiency — that conversion (shaft energy → fuel energy demand) is the
caller's (powerplant.jl's) job.
"""
fuel_mass_for_energy(tank::FuelTank, energy_J::Real) =
    energy_J / tank.properties.energy_density_J_per_kg

"""
    fuel_energy_for_mass(tank, mass_kg) → J

Inverse of fuel_mass_for_energy. Useful for endurance estimates.
"""
fuel_energy_for_mass(tank::FuelTank, mass_kg::Real) =
    mass_kg * tank.properties.energy_density_J_per_kg

"""
    refuel!(tank::FuelTank, mass_kg) → Float64 (actual kg added, clamped to capacity)

Add fuel to the tank, clamped so mass_kg never exceeds capacity_kg.
For ground refuel between missions, not for in-flight use. Named
refuel! rather than fill! to avoid any ambiguity with Base.fill!
(array filling) — a plain, unrelated fill! would otherwise collide
with that existing Base function for every downstream caller.
"""
function refuel!(tank::FuelTank, mass_kg::Real)
    added = min(mass_kg, tank.capacity_kg - tank.mass_kg)
    tank.mass_kg += added
    added
end

"""
    fuel_fraction(tank) → [0,1]

Current fuel mass as a fraction of tank capacity. For gauge display.
"""
fuel_fraction(tank::FuelTank) = tank.capacity_kg > 0.0 ? tank.mass_kg / tank.capacity_kg : 0.0

"""
    fuel_capacity_L(tank) → L

Tank capacity in litres, back-converted from capacity_kg using this
tank's fuel density. For gauge display / mission planning.
"""
fuel_capacity_L(tank::FuelTank) = tank.capacity_kg / tank.properties.density_kg_per_L

"""
    fuel_mass_L(tank) → L

Current fuel volume in litres.
"""
fuel_mass_L(tank::FuelTank) = tank.mass_kg / tank.properties.density_kg_per_L

end # module Fuel
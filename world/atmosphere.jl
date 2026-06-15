# atmosphere.jl:   Troposphere-only ISA model (0-11000 m MSL)
# AUTHOR:          DANIEL DESAI
# UPDATED:         2026-05-10
# VERSION:         0.1.0
#
#
# Airport elevation: ATM.airport_alt_m sets the MSL baseline.
# ODE state 2 (alt) is always AGL (above ground level).
# The rho() / temp_c() / mach() shorthands convert AGL→MSL internally,
# so no call site in fly.jl or the subsystems needs to change.
#
# Wind: log-law shear profile referenced to ground level.
# Dryden turbulence: ODE states appended by fly.jl (see DRYDEN section).


using Random

# ── Troposphere Constants ──────────────────────────────────────────────
const ATM_T0    = 288.15    # ISA sea-level temperature (K)
const ATM_P0    = 101325.0  # ISA sea-level pressure (Pa)
const ATM_RHO0  = 1.225     # ISA sea-level density (kg/m³)
const ATM_L     = 0.0065    # Temperature lapse rate (K/m)
const ATM_G     = 9.80665   # Gravity (m/s²)
const ATM_R     = 287.058   # Dry-air gas constant (J/kg·K)
const ATM_Rv    = 461.5     # Water-vapour gas constant (J/kg·K)
const ATM_GAMMA = 1.4       # Ratio of specific heats
const ATM_Z0    = 0.03      # Surface roughness, open terrain/airport (m)

# ── Structs ────────────────────────────────────────────────────────────
Base.@kwdef mutable struct WindModel
    u                    :: Float64 = 0.0    # East component at reference_alt_m (m/s)
    v                    :: Float64 = 0.0    # North component at reference_alt_m (m/s)
    w                    :: Float64 = 0.0    # Vertical/updraft component (m/s)
    reference_alt_m      :: Float64 = 500.0  # AGL height where u,v are defined (m)
    turbulence_intensity :: Float64 = 0.0    # Dryden σ (m/s); 0.0 = disabled
    gust_amplitude       :: Float64 = 0.0    # Discrete gust peak (m/s) — future
end

Base.@kwdef mutable struct AtmosphereModel
    wind             :: WindModel = WindModel()
    ambient_temp_c   :: Float64   = 15.0     # Ground-level temperature (°C); ISA = 15
    ambient_pressure :: Float64   = ATM_P0   # Ground-level pressure (Pa)
    humidity_pct     :: Float64   = 50.0     # Relative humidity (0–100 %)
    airport_alt_m    :: Float64   = 0.0      # Airport MSL elevation (m)    [FEAT-1]
                                              # e.g. KSFO=4, KDEN=1656, LEMD=610, LSZH=432
    airport_icao     :: String    = "XXXX"   # ICAO code shown in banner & cockpit header

    # ── Dryden RNG (FEAT-3) ───────────────────────────────────────────
    # Seeded at startup by init_dryden_rng!(ATM, seed). Re-sampled every
    # DRYDEN_DT_S seconds by a PeriodicCallback in fly.jl (DRYDEN label).
    dryden_rng       :: MersenneTwister = MersenneTwister(0)
    dryden_noise     :: Vector{Float64} = zeros(3)   # [wu, wv, ww] held-noise sample
end

# Global instance — mutate before calling main():
#   ATM.airport_icao     = "KDEN"
#   ATM.airport_alt_m    = 1656.0   # Denver (5 431 ft)
#   ATM.ambient_temp_c   = 32.0     # hot day
#   ATM.wind.u           = 8.0      # 8 m/s headwind
#   ATM.wind.turbulence_intensity = 1.5   # light turbulence (m/s RMS)
const ATM = AtmosphereModel()

# ── AGL ↔ MSL Helpers ─────────────────────────────────────────────────
"""
    agl_to_msl(alt_agl, atm) → m MSL
Converts ODE alt (AGL) to MSL for atmosphere functions.
"""
agl_to_msl(alt_agl::Real, atm::AtmosphereModel=ATM) =
    alt_agl + atm.airport_alt_m

"""
    msl_to_agl(alt_msl, atm) → m AGL
Used when setting initial conditions: u0[2] = msl_to_agl(field_elevation_msl).
"""
msl_to_agl(alt_msl::Real, atm::AtmosphereModel=ATM) =
    alt_msl - atm.airport_alt_m

# ── ISA Troposphere Physics ────────────────────────────────────────────
# All functions below accept MSL altitude.
# Valid throughout the troposphere (0–11 000 m MSL).

"""
    atm_temperature(alt_msl, atm) → K
Temperature with ambient offset applied at ground level.
"""
function atm_temperature(alt_msl::Real, atm::AtmosphereModel=ATM)
    T_surface = atm.ambient_temp_c + 273.15              # K at airport elevation
    alt_above = max(alt_msl - atm.airport_alt_m, 0.0)   # height above airport (m)
    return T_surface - ATM_L * alt_above
end

"""
    atm_vapour_pressure(atm) → Pa
Partial pressure of water vapour at ground level (Buck equation × RH).
"""
function atm_vapour_pressure(atm::AtmosphereModel=ATM)
    T_c = atm.ambient_temp_c
    e_s = 611.21 * exp((18.678 - T_c / 234.5) * (T_c / (257.14 + T_c)))
    return (atm.humidity_pct / 100.0) * e_s
end

"""
    atm_pressure(alt_msl, atm) → Pa
Barometric pressure via hypsometric formula. Troposphere only.
"""
function atm_pressure(alt_msl::Real, atm::AtmosphereModel=ATM)
    # Derive station pressure at airport elevation from QNH via ISA hypsometric.
    # ambient_pressure is QNH (Pa, MSL-equivalent); airport_alt_m lifts it to field.
    p_station = atm.ambient_pressure *
                (1.0 - ATM_L * atm.airport_alt_m / ATM_T0)^(ATM_G / (ATM_L * ATM_R))
    # Lapse from airport elevation upward using actual surface temperature.
    T_surface = atm.ambient_temp_c + 273.15
    alt_above = max(alt_msl - atm.airport_alt_m, 0.0)
    return p_station * (1.0 - ATM_L * alt_above / T_surface)^(ATM_G / (ATM_L * ATM_R))
end

"""
    atm_density(alt_msl, atm) → kg/m³
Density with density-altitude correction. Virtual temperature Tv accounts
for humidity reducing air density.
"""
function atm_density(alt_msl::Real, atm::AtmosphereModel=ATM)
    alt = max(alt_msl, atm.airport_alt_m)
    T   = atm_temperature(alt, atm)
    P   = atm_pressure(alt, atm)
    # Humidity decay with altitude (simple exponential scale height ~2 km)
    e   = atm_vapour_pressure(atm) * exp(-alt / 2000.0)
    Tv  = T / max(1.0 - 0.378 * (e / max(P, 1.0)), 0.01)
    return P / (ATM_R * Tv)
end

"""
    atm_speed_of_sound(alt_msl, atm) → m/s
"""
atm_speed_of_sound(alt_msl::Real, atm::AtmosphereModel=ATM) =
    sqrt(ATM_GAMMA * ATM_R * atm_temperature(alt_msl, atm))

# ── Wind: Log-Law Boundary Layer ──────────────────────────────────────
"""
    atm_wind(atm, alt_agl) → (u, v, w) m/s
Log-law wind profile. alt_agl is height above the local runway/ground.
u and v reach full values at wind.reference_alt_m AGL.
"""
function atm_wind(atm::AtmosphereModel, alt_agl::Real)
    agl    = max(alt_agl, zero(alt_agl))
    z0     = ATM_Z0
    z_ref  = max(atm.wind.reference_alt_m, z0 + 0.1)
    ln_ref = log(z_ref / z0)
    shear  = ifelse(agl > z0,
                 log(max(agl, z0 * 1.001) / z0) / ln_ref,
                 zero(agl))
    return (atm.wind.u * shear,
            atm.wind.v * shear,
            atm.wind.w)
end

"""
    atm_airspeed(vx, atm, alt_agl) → m/s
Forward airspeed corrected for the headwind component.
"""
function atm_airspeed(vx::Real, atm::AtmosphereModel, alt_agl::Real)
    wu, _, _ = atm_wind(atm, alt_agl)
    return vx - wu
end

# ── Dryden Turbulence (MIL-HDBK-1797B §3.7, low-altitude) ────────────
#
# Three turbulence velocity components are appended to the ODE state
# vector by fly.jl (states N+1, N+2, N+3 where N = 18 + terrain_agl).
# They are first-order Ornstein-Uhlenbeck processes driven by white noise:
#
#   dturb_i/dt = -(V/L_i)·turb_i + σ·sqrt(2V/L_i)·w_i(t)
#
# where w_i(t) is unit-variance white noise held for DRYDEN_DT_S seconds.
# Scale lengths L_i(alt) follow MIL-HDBK-1797B Table B-I (low altitude).
# σ is set via ATM.wind.turbulence_intensity (0 = disabled).


const DRYDEN_DT_S    = 0.05    # noise hold period (s) — 20 Hz resample rate
const ATM_TURB_H_MIN = 5.0     # AGL below which turbulence is suppressed (m)
const ATM_TURB_H_MAX = 305.0   # scale-length upper bound — 1 000 ft (m)

"""
    init_dryden_rng!(atm, seed)
Seed the Dryden RNG and draw the first noise sample.
Call from fly.jl after ATM fields are populated from test_card.json.
"""
function init_dryden_rng!(atm::AtmosphereModel, seed::Integer)
    atm.dryden_rng   = MersenneTwister(Int(seed))
    atm.dryden_noise .= randn(atm.dryden_rng, 3)
    nothing
end

"""
    atm_turbulence_scales(alt_agl) → (L_u, L_v, L_w) m
MIL-HDBK-1797B Table B-I low-altitude scale lengths.
  L_w = h  (vertical, rises linearly with height)
  L_u = L_v = h / (0.177 + 0.000823·h)^1.2   (longitudinal/lateral)
"""
function atm_turbulence_scales(alt_agl::Real)
    h   = clamp(Float64(alt_agl), ATM_TURB_H_MIN, ATM_TURB_H_MAX)
    L_w = h
    L_u = h / (0.177 + 0.000823 * h)^1.2
    L_v = L_u
    return (L_u, L_v, L_w)
end

"""
    atm_turbulence_deriv(turb_u, turb_v, turb_w, V, alt_agl, atm, noise)
        → (dturb_u, dturb_v, dturb_w)

First-order Dryden shaping filter ODE RHS (MIL-HDBK-1797B eq. B-1).
Called from build_ode() in fly.jl with the held noise vector.

Arguments:
  turb_u/v/w  ODE states for the three gust components (m/s)
  V           airspeed magnitude (m/s); clamp prevents /0 at hover
  alt_agl     height above ground (m) — used for scale length lookup
  atm         AtmosphereModel — reads .wind.turbulence_intensity
  noise       pre-sampled unit-variance noise [wu, wv, ww]
              (from ATM.dryden_noise, resampled by PeriodicCallback)

Returns zero derivatives when turbulence_intensity == 0 or V < 1 m/s
so the states are frozen rather than driven, avoiding /0 singularities.
"""
function atm_turbulence_deriv(turb_u::Real, turb_v::Real, turb_w::Real,
                               V::Real, alt_agl::Real,
                               atm::AtmosphereModel=ATM,
                               noise::AbstractVector=atm.dryden_noise)
    σ = atm.wind.turbulence_intensity
    if σ <= 0.0 || alt_agl < ATM_TURB_H_MIN || V < 1.0
        return (zero(turb_u), zero(turb_v), zero(turb_w))
    end
    L_u, L_v, L_w = atm_turbulence_scales(alt_agl)
    Vc = max(Float64(V), 1.0)
    β_u = Vc / L_u;  β_v = Vc / L_v;  β_w = Vc / L_w
    # First-order Ornstein-Uhlenbeck: dX = -β·X·dt + σ·√(2β)·dW
    dturb_u = -β_u * turb_u + σ * sqrt(2.0 * β_u) * noise[1]
    dturb_v = -β_v * turb_v + σ * sqrt(2.0 * β_v) * noise[2]
    dturb_w = -β_w * turb_w + σ * sqrt(2.0 * β_w) * noise[3]
    return (dturb_u, dturb_v, dturb_w)
end

# ── AGL-aware Shorthands ──────────────────────────────────────────────
# Call sites in fly.jl and subsystems pass AGL altitude (ODE state 2).
# These wrappers apply the airport offset transparently.   [FEAT-2]
rho(alt_agl)     = atm_density(agl_to_msl(alt_agl), ATM)
temp_c(alt_agl)  = atm_temperature(agl_to_msl(alt_agl), ATM) - 273.15
mach(v, alt_agl) = v / atm_speed_of_sound(agl_to_msl(alt_agl), ATM)
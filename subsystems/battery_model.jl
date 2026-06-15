# battery_model.jl:   Battery and electrical system model
# AUTHOR:              DANIEL DESAI
# UPDATED:             2026-05-10
# VERSION:             0.1.0
#
# Equivalent-circuit model:
#   - Peukert-corrected current integration for SoC
#   - OCV-IR voltage model
#   - Steady-state I²R thermal model with exponential lag
#
# Pack spec: 250 kWh / 800V nominal
# 

Base.@kwdef struct BatteryParams
    capacity_kwh         :: Float64 = 160.0   # Usable pack capacity (kWh)
    v_nominal            :: Float64 = 800.0   # Nominal voltage (V)
    r_internal_ohm       :: Float64 = 0.03    # Pack internal resistance (Ω)
    soc_init             :: Float64 = 1.0     # Initial state of charge (0–1)
    soc_min              :: Float64 = 0.05    # Minimum safe SoC (reserve)
    k_peukert            :: Float64 = 1.05    # Peukert exponent (1.0 = ideal)
    eta_discharge        :: Float64 = 0.97    # Round-trip discharge efficiency
    r_thermal_degc_per_w :: Float64 = 0.02    # Thermal resistance (°C/W)
    t_ambient_c          :: Float64 = 25.0    # Ambient temperature (°C)
    tau_thermal_s        :: Float64 = 30.0    # Thermal time constant (s)
    i_max_a              :: Float64 = 400.0   # Max continuous current (A)
end

const BP = BatteryParams()

# Derived
capacity_ah(bp::BatteryParams) = (bp.capacity_kwh * 1000.0) / bp.v_nominal
const CAP_AH = capacity_ah(BP)

# ── Electrical Model ───────────────────────────────────────────────────

"""
    battery_current(power_kw, bp) → A
Draw current from power demand. Clamped to i_max.
"""
function battery_current(power_kw, bp::BatteryParams=BP)
    return min(power_kw * 1000.0 / bp.v_nominal, bp.i_max_a)
end

"""
    terminal_voltage(soc, current_a, bp) → V
OCV-IR model: V = V_nom·SoC - I·R_int
"""
function terminal_voltage(soc, current_a, bp::BatteryParams=BP)
    return bp.v_nominal * soc - current_a * bp.r_internal_ohm
end

"""
    soc_derivative(soc, power_kw, bp) → dSoC/dt (1/s)
Peukert-corrected capacity integration for discharge.
Regen (power_kw < 0) is treated as linear charge at 70% efficiency
(matching rotor_system.jl `bem_thrust_aggregate` regen path) —
Peukert distortion does not apply to charging current.
Clipped at soc_min (discharge) and 1.0 (charge).
"""
function soc_derivative(soc, power_kw, bp::BatteryParams=BP)
    cap  = capacity_ah(bp)
    if power_kw < 0.0
        # Regen: negative power_kw → charging current, linear, 70% efficiency
        i_regen = abs(power_kw) * 1000.0 / bp.v_nominal * 0.70
        dsoc    = (i_regen / cap) / 3600.0
    else
        # Discharge: Peukert-corrected — i is guaranteed non-negative here
        i     = battery_current(power_kw, bp)
        i_eff = i^bp.k_peukert / (cap^(bp.k_peukert - 1.0) + 1e-6)
        dsoc  = -(i_eff / cap) / 3600.0
    end
    # Clamp at limits
    dsoc = ifelse(soc <= bp.soc_min && dsoc < 0.0, 0.0,
           ifelse(soc >= 1.0        && dsoc > 0.0, 0.0, dsoc))
    return dsoc
end

# ── Thermal Model ──────────────────────────────────────────────────────

"""
    steady_state_temp(power_kw, bp) → °C
Steady-state pack temperature from I²R dissipation.
"""
function steady_state_temp(power_kw::Float64, bp::BatteryParams=BP)
    i = battery_current(power_kw, bp)
    return bp.t_ambient_c + (i^2 * bp.r_internal_ohm) * bp.r_thermal_degc_per_w
end

"""
    temp_derivative(temp_c, power_kw, bp) → d(T)/dt (°C/s)
First-order thermal lag toward steady-state temperature.
Use as an ODE state for live temperature tracking.
"""
function temp_derivative(temp_c::Float64, power_kw::Float64, bp::BatteryParams=BP)
    t_ss = steady_state_temp(power_kw, bp)
    return (t_ss - temp_c) / bp.tau_thermal_s
end

"""
    smooth_temp_series(ts, power_kw_series, bp) → Vector{°C}
Post-processing: applies exponential smoothing to a power time series.
Used in fly.jl postprocess() to avoid the t=0 initialisation spike.
"""
function smooth_temp_series(ts::Vector{Float64}, power_kw_series::Vector{Float64},
                             bp::BatteryParams=BP)
    temps    = similar(power_kw_series)
    temps[1] = bp.t_ambient_c
    for i in 2:length(ts)
        dt       = ts[i] - ts[i-1]
        α        = dt / (bp.tau_thermal_s + dt)
        t_ss     = steady_state_temp(power_kw_series[i], bp)
        temps[i] = temps[i-1] + α * (t_ss - temps[i-1])
    end
    return temps
end
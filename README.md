# WINDBLADE

Advanced Air Mobility Model-Based Prototyping

## Architecture

WINDBLADE is a multi-language eVTOL simulation environment. Each language allocation owns a distinct concern:

| Layer | Language | Role |
|---|---|---|
| Launcher / GUI | Python | Browser-based mission planning UI, test runner |
| Physics / ODE | Julia | Subsystem models, state integrator, glass cockpit UI|
| Autopilot | C++ | Flight controller compiled to `autopilot.so` |
| HOTAS input | C | Manual flight controls reader compiled to `controls/hotas` |

The Julia ODE solver calls into `autopilot.so` via `@ccall` at each save step. The Python launcher (`windblade.py`) orchestrates builds, planning, and the Julia subprocess. Manual flight reads the HOTAS device via a separate C process piped into the Julia ODE loop.

## Repository Layout

```
windblade/
├── windblade.py                        # Entry point — launcher and browser GUI
├── test_flight.py                      # Automated flight test runner
├── fly.jl                              # ODE integrator, state vector, controls dispatch
├── glass_cockpit.jl                    # GLMakie HMI — MIL-STD-3009 NVG palette
├── Makefile                            # Builds autopilot.so and hotas binary
│
├── controls/
│   ├── autopilot.cpp                   # Flight controller (compiled to autopilot.so)
│   └── hotas.c                         # Thrustmaster T.Flight HOTAS One reader
│
├── planning/
│   ├── mission_planner.jl              # Phase scheduler and timing constants
│   ├── navigation.jl                   # Waypoint guidance and nav map state
│   └── test_card.json                  # Mission parameters
│
├── subsystems/
│   ├── airframe.jl                     # Aerodynamics and body forces
│   ├── actuators.jl                    # Actuator model and PID tuning constants
│   ├── battery_model.jl                # Equivalent-circuit battery and SoC model
│   ├── landing_gear.jl                 # Three-point strut contact model
│   └── propulsion/
│       ├── blades.jl                   # Blade element momentum (BEM) implementation
│       ├── powerplant.jl               # Motor/engine backends (electric, turboshaft)
│       ├── fuel.jl                     # Fuel chemistry and tank capacity
│       ├── rotor_system.jl             # Powerplant top-level model
│       ├── rotor_mixer.jl              # Wrench-to-RPM control allocator
│       └── rotor_config.csv            # Per-rotor geometry and power parameters
│
└── world/
    ├── atmosphere.jl                   # Troposphere ISA model
    └── terrain.jl                      # Piecewise-linear ground-track elevation
```

## State Vector
 
The ODE integrates 22 states:
 
| # | Symbol | Description | Units |
|---|---|---|---|
| 1 | `vx` | Forward speed | m/s |
| 2 | `alt` | Altitude AGL | m |
| 3 | `tilt` | Rotor tilt angle (0=hover, π/2=cruise) | rad |
| 4 | `dtilt` | Tilt rate | rad/s |
| 5 | `pitch` | Pitch angle (+nose up) | rad |
| 6 | `dpitch` | Pitch rate | rad/s |
| 7 | `roll` | Roll angle | rad |
| 8 | `droll` | Roll rate | rad/s |
| 9 | `yaw` | Yaw angle | rad |
| 10 | `dyaw` | Yaw rate | rad/s |
| 11 | `thrust_lag` | Actual rotor thrust — first-order spool lag | N |
| 12 | `soc` | Battery state of charge | 0–1 |
| 13 | `τ` | Mission clock (negative = pre-hover) | s |
| 14 | `x` | Ground-track position, forward | m |
| 15 | `y` | Ground-track position, rightward | m |
| 16 | `ωx` | Body roll rate | rad/s |
| 17 | `ωy` | Body pitch rate | rad/s |
| 18 | `ωz` | Body yaw rate | rad/s |
| 19 | `terrain_agl` | Terrain elevation at departure datum | m |
| 20 | `turb_u` | Dryden longitudinal gust velocity | m/s |
| 21 | `turb_v` | Dryden lateral gust velocity | m/s |
| 22 | `turb_w` | Dryden vertical gust velocity | m/s |
 
States 19–22 added with Dryden turbulence (MIL-HDBK-1797B). The C++ autopilot interface receives states 1–18 + terrain AGL only — do not reorder.
 
Attitude rates (states 6, 8, 10) are propagated using the full Euler angle kinematic transform from body-frame rates (`ωx, ωy, ωz` — states 16–18) as:
 
```
φ̇ = ωx + sin(φ)tan(θ)ωy + cos(φ)tan(θ)ωz
θ̇ = cos(φ)ωy − sin(φ)ωz
ψ̇ = sin(φ)/cos(θ)·ωy + cos(φ)/cos(θ)·ωz
```
 
(φ=roll, θ=pitch, ψ=yaw)


## Dependencies

**Julia (1.9+)**
- DifferentialEquations.jl
- GLMakie.jl
- ForwardDiff.jl

**Python (3.10+)**
- numpy
- matplotlib

**C++ build**
- g++ with C++17

**HOTAS (optional)**
- Linux only — reads `/dev/input/js0` via joydev kernel interface
- Tested with Thrustmaster T.Flight HOTAS One

## Build

```bash
# Build autopilot shared library and HOTAS binary
make

# Build targets individually
make autopilot
make hotas
```

## Usage

Launcher (browser GUI):

```bash
python3 windblade.py
```

Automated test runner:

```bash
# Full run — autopilot, glass cockpit
julia --threads auto fly.jl

# No GUI — CSV output only
julia --threads auto fly.jl --no-gui

# Manual HOTAS input
julia --threads auto fly.jl --manual

# Pass weather and cruise parameters directly
python3 test_flight.py --auto \
    --dep-metar "KAXX 151155Z 00000KT 10SM CLR M01/M10 A3018 RMK AO2 T10141096" \
    --arr-metar "KSAF 151153Z 24005KT 10SM CLR 13/M09 A3005 RMK AO2 T01281094" \
    --speed-kmh 300 --alt-ft 11500 --hover-m 30

# Skip recompile
python3 test_flight.py --no-build

# Custom output directory
python3 test_flight.py --out results/r1
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | All checks passed |
| 1 | One or more test criticalities failed |
| 2 | Build failed |
| 3 | Sim failed / no CSV produced |
| 4 | Flight planning failed |
| 10 | Config invalid |

## Rotor Configuration

Rotor geometry is defined in `subsystems/propulsion/rotor_config.csv`. Default configuration is an all-electric rotor fleet, however mixed fleets are supported as in the below example with two turbine-electric rotors (R1/R2) and four electric rotors (R3–R6):

**Turbine-electric (R1/R2):**

| Parameter | Value |
|---|---|
| Radius | 1.82 m |
| Blades | 6 |
| Chord | 0.183 m |
| Twist root/tip | 16° / 6° |
| Pitch offset | 4.4° |
| Max power | 500 kW per rotor |
| Hover RPM | 960 |
| Powerplant | `turbine_electric` |

**Electric, stock (R3–R6):**

| Parameter | Value |
|---|---|
| Radius | 1.45 m |
| Blades | 5 |
| Chord | 0.175 m |
| Twist root/tip | 16° / 6° |
| Pitch offset | 4.4° |
| Max power | 280 kW per rotor |
| Hover RPM | 960 |
| Powerplant | `electric` |

Edit `rotor_config.csv` and reload the Rotor Config tab in the GUI to apply changes without restart. A `turbine_electric` row requires fuel tank (`FuelTank`) and fuel chemistry (`FuelProperties`) parameters set in `fuel.jl`. `powerplant.jl` implements a `TurboshaftEngine` model (Gagg–Ferrar altitude lapse, derived SFC from thermal efficiency). Both files are loaded unconditionally by `rotor_system.jl` regardless of fleet composition.

Edit `lift_only_rotors lift_shutoff_tilt cruise_only_rotors` and `cruise_activation_tilt` in `rotor_mixer.jl` to configure aircraft with lift-only and/or pusher rotors.

## Author

DANIEL DESAI — v0.1.2

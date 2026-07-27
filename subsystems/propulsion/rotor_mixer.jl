# rotor_mixer.jl:     Wrench-to-RPM control allocator
# AUTHOR:             DANIEL DESAI
# UPDATED:            2026-07-26
# VERSION:            0.2.0
#
# Post-solve mapping: takes the aggregate wrench demand from the flight
# controller (thrust magnitude + three moment demands) and distributes it
# across the 6 RotorFleet units by solving the geometry matrix.
#
#
# ── Conventions from rotor_system.jl ──────────────────────────────────
#   Thrust :  T = kT · ρ · ω² · R⁴
#   Torque :  Q = kQ · ρ · n² · D⁵   →   Q/T = (kQ/kT)·D = (kQ/kT)·2R  [m]
#   Power  :  P_ind  = k_induced · T · vi / eta_rotor
#             P_prof = c_profile · ρ · A · (ω·R)³
#   spin_dir: +1 = CCW from above, -1 = CW from above
#
# ── Default arm geometry (_default_unit in rotor_system.jl) ───────────
#   id  arm_x_m  arm_y_m  spin  role
#    1   +3.0    +4.5     +1    fwd-L  CCW
#    2   +3.0    -4.5     -1    fwd-R  CW
#    3    0.0    +5.5     -1    mid-L  CW   ← spin_dir=-1 (CW)
#    4    0.0    -5.5     +1    mid-R  CCW
#    5   -3.0    +4.5     +1    aft-L  CCW
#    6   -3.0    -4.5     -1    aft-R  CW
#
# ── Body-frame coordinate convention ──────────────────────────────────
#   x : forward (nose)    Mx (roll)  : right-wing-down positive
#   y : right             My (pitch) : nose-up positive
#   z : up                Mz (yaw)   : nose-right positive
#
# ══ PER-ROTOR MODE (v0.2.0) ═══════════════════════════════════════════
# Each rotor carries two independent booleans, sourced from the `lift` and
# `thrust` columns of rotor_config.csv (via rotor_system.jl → RotorUnit):
#
#   lift   thrust   mode            effective tilt τᵢ   behaviour
#   ────   ──────   ─────────────   ─────────────────   ───────────────────────
#   true   true     TILTROTOR       τᵢ = tilt_rad       v0.1.5 behaviour: the
#                                                       nacelle sweeps through
#                                                       transition
#   true   false    LIFT-ONLY       τᵢ = 0  (fixed up)  never tilts; keeps
#                                                       running in horizontal
#                                                       flight to supply the
#                                                       lift and control the
#                                                       wings don't
#   false  true     THRUST-ONLY     τᵢ = π/2 (fixed)    BETA-style pusher;
#                                                       idles below
#                                                       cruise_activation_tilt,
#                                                       powers up through
#                                                       transition
#   false  false    OUT OF SERVICE  —                   ω = 0, P = 0, zeroed
#                                                       column in B. Mass and
#                                                       inertia are untouched
#                                                       (airframe.jl), so a dead
#                                                       rotor still carries its
#                                                       weight and moment
#
# Mixed fleets are the point: e.g. R1/R2 tilt, R3/R4 lift-only, R5/R6 pusher.
# "Transition" is unchanged — the autopilot, or the pilot on the same button,
# sweeps the same `tilt_rad` schedule. Rotors that do not physically tilt
# ignore it except as a phase cue, which is what `cruise_activation_tilt` reads.
#
# CAUTION: the allocator honours whatever mix the CSV specifies, but it cannot
# invent authority the arms do not provide. Check that the lift-producing set
# still straddles the CG in x and y, or the solve will trade total lift away to
# hold pitch. With the stock arm geometry, "front tilt / mid lift / rear
# thrust" leaves R1/R2 as the only lift forward of the CG in hover and the
# allocator has to drive them to zero — see Test 8 for a topology that trims.
#
# ── B-matrix rows (5×6 as of v0.2.0) ──────────────────────────────────
#   Rotor i produces thrust Tᵢ along its own axis aᵢ = (sin τᵢ, 0, cos τᵢ)
#   and a reaction torque −spin_dirᵢ · τ_armᵢ · Tᵢ about that same axis.
#   Rotors are assumed to lie in the CG x-y plane (RotorUnit has no arm_z),
#   so an x-force produces yaw but no pitch.
#
#   [1] Fx : sin τᵢ                                  forward force
#   [2] Fz : cos τᵢ                                  lift   (the old row 1)
#   [3] Mx : arm_y·cos τᵢ − spin·τ_arm·sin τᵢ        thrust roll + reaction roll
#   [4] My : arm_x·cos τᵢ                            thrust pitch
#   [5] Mz : −arm_y·sin τᵢ − spin·τ_arm·cos τᵢ       x-force yaw + reaction yaw
#
#   The two reaction terms are one physical torque projected onto the rotated
#   spin axis: at τ=0 it is pure yaw (identical to v0.1.5), at τ=π/2 it is pure
#   roll — exactly the torque a pusher prop dumps into the airframe for the
#   wings to trim out. It is zero in hover, so hover is bit-identical to v0.1.5.
#
# ── Why the thrust row split in two ───────────────────────────────────
#   v0.1.5 solved for a single scalar T_total because every rotor shared one
#   tilt angle, so thrust direction was a fleet-level property. With mixed modes
#   each column points somewhere different and the demand must be resolved into
#   body axes before the solve. The controller interface is unchanged: it still
#   hands over a thrust *magnitude* plus the tilt, and allocate_wrench splits it
#       Fx_demand = T_total · sin(tilt_rad)
#       Fz_demand = T_total · cos(tilt_rad)
#   For an all-tiltrotor fleet this is algebraically identical to the old
#   single-row problem — the sin²+cos² residuals recombine exactly — so the
#   stock six-tiltrotor config is unchanged.
#
# ── Allocator formulation ─────────────────────────────────────────────
#   Solves:  T = arg min_{T ≥ 0} ‖BT - w‖_{Λp⁻²}
#   where:
#     w  = [Fx, Fz, M_roll, M_pitch, M_yaw]ᵀ   wrench demand vector
#     B  ∈ ℝ⁵ˣ⁶                                control effectiveness matrix
#     Λp = diag(p₁,…,p₆)                       per-rotor power-weight matrix
#     ‖v‖_{Λp⁻²} = √(vᵀ Λp⁻² v)              Λp⁻²-weighted norm
#
#   Solved via Lawson-Hanson NNLS after substitution T_sub = Λp⁻¹T:
#     min ‖(BΛp)T_sub - w‖₂  s.t.  T_sub ≥ 0,  then  T = Λp·T_sub
#
# ── Files to change in the project ────────────────────────────────────
#   rotor_mixer.jl   ← this file
#   rotor_system.jl  ← RotorUnit needs `lift::Bool` / `thrust::Bool` parsed
#                      from rotor_config.csv (both default true). If those
#                      fields are absent this file falls back to all-true, so
#                      it stays loadable against the current rotor_system.jl.
#   fly.jl           ← see "fly.jl integration" at the bottom of this file
# =====================================================================

using LinearAlgebra: pinv, Diagonal, I
using StaticArrays: SVector, SMatrix, MVector, MMatrix, @MVector

const _TILT_VERT = 0.0                # lift-only rotors: axis straight up
const _TILT_HORZ = Float64(π / 2)     # thrust-only rotors: axis straight fwd

# ── AllocatorParams ────────────────────────────────────────────────────
"""
    AllocatorParams

Tuning constants for the control allocator.
"""
Base.@kwdef struct AllocatorParams

    # Attitude error → moment demand scaling (N·m / rad).
    # Sized so a 5° (≈0.087 rad) error produces ~300–400 N·m, which is
    # enough to correct a 2177 kg / Ixx=3500 kg·m² vehicle in ~2 s.
    roll_moment_scale  :: Float64 = 3500.0   # ≈ Ixx · wn²  (wn=1 rad/s)
    pitch_moment_scale :: Float64 = 4200.0   # ≈ Iyy estimated
    yaw_moment_scale   :: Float64 = 1500.0   # ≈ Izz estimated

    # Row weights in the B matrix (scales rows before NNLS solve).
    # Reduce a weight to deprioritise that channel when rotors saturate.
    #
    # NOTE (carried over from v0.1.5 — flagged, deliberately NOT changed):
    # these scale the rows of B but not the demand vector w, so the solve is
    # really BT ≈ W⁻¹w. w_yaw = 0.5 therefore *doubles* the effective yaw
    # target rather than halving that channel's priority. Textbook weighted LS
    # would scale both sides. The moment_scale gains are tuned around the
    # current behaviour, so it is left alone; if w is ever scaled to match,
    # halve yaw_moment_scale in the same commit.
    w_fx     :: Float64 = 1.0   # forward-force row (new in v0.2.0)
    w_thrust :: Float64 = 1.0   # lift row
    w_roll   :: Float64 = 1.0
    w_pitch  :: Float64 = 1.0
    w_yaw    :: Float64 = 0.5   # limited authority on fixed-pitch rotors

    # ω limits (rad/s): R≈1 m, ω_nom≈110 rad/s, tip ≈110 m/s.
    omega_min :: Float64 =  20.0   # flight-idle floor
    omega_max :: Float64 = 180.0   # structural / acoustic limit

    # ── Tilted-axis coupling (regression control) ─────────────────────
    # Enables the two force/torque terms that only exist once a thrust axis is
    # off vertical: the reaction torque's roll component (−spin·τ_arm·sin τ)
    # and the yaw from an x-force on a y arm (−arm_y·sin τ). v0.1.5 had
    # neither, modelling a fully tilted rotor as having no yaw authority at
    # all — which is wrong, differential thrust on forward-pointing rotors
    # yaws the aircraft exactly like a twin.
    #
    # Set false to reproduce v0.1.5 bit-for-bit on an all-tiltrotor fleet, for
    # A/B revalidation of a tuned mission. Measured effect with the stock fleet
    # and realistic in-loop demands: 0 in hover, up to ~12% of peak ω through
    # transition, concentrated in the yaw channel.
    #
    # MUST stay true for any fleet with a fixed pusher — an off-centreline
    # pusher's yaw and torque reaction live entirely in these terms.
    tilt_axis_coupling :: Bool = true

    # ── Mode overrides (testing / what-if only) ───────────────────────
    # Production source of truth is the `lift` / `thrust` column pair in
    # rotor_config.csv, carried onto RotorUnit by rotor_system.jl. Leave both
    # `nothing` and the fleet decides. Set an NTuple{6,Bool} to force a
    # configuration without touching the CSV — the self-test does exactly this
    # so it runs against an unmodified rotor_system.jl.
    lift_flags   :: Union{Nothing,NTuple{6,Bool}} = nothing
    thrust_flags :: Union{Nothing,NTuple{6,Bool}} = nothing

    # Lift-only rotors (lift=true, thrust=false)
    # These never tilt. They keep running in horizontal flight, supplying
    # whatever lift and moment the wings do not — so by default they are NEVER
    # shut off (Inf) and simply fall to `omega_min` when the allocator has
    # nothing for them to do. Set a finite angle to recover the v0.1.5
    # behaviour where lift rotors declutch and windmill (regenerating) past it.
    lift_shutoff_tilt :: Float64 = Inf

    # Thrust-only rotors (lift=false, thrust=true)
    # Pusher props. Held at `omega_min` with no commanded thrust until tilt
    # reaches `cruise_activation_tilt`, then they carry the Fx demand. Overlap
    # with any lift shutoff is intentional — both sets briefly live during
    # transition ensures no thrust deficit at handoff.
    cruise_activation_tilt :: Float64 = deg2rad(45.0)

    # ── Per-rotor power-preference weights (Λp diagonal) ──────────────
    # Diagonal entries of Λp = diag(p₁,…,p₆), the column-weighting matrix
    # used in the NNLS allocator: min ‖Λp⁻¹ · T‖₂  s.t.  BT = w, T ≥ 0.
    #
    # A larger weight tells the allocator to "prefer" that rotor — it will
    # be assigned proportionally more thrust before saturating neighbours.
    # Set weights proportional to each rotor's rated shaft power so that,
    # e.g., a turbine-electric "Super rotor" (746 kW) gets 2.66× the weight
    # of a stock electric rotor (280 kW).
    #
    # Only the *ratios* matter — the allocator normalises internally.
    # Equal weights reproduce the standard uniform NNLS behaviour.
    power_weights :: NTuple{6,Float64} = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # Tikhonov regularisation for the NNLS solve (ε in ‖BΛpT_sub - w‖² + ε²‖T_sub‖²).
    # Pulls the solution away from degenerate corners where the active-set solver
    # flips rotors in/out between ODE steps, causing power spikes.
    # Larger ε → smoother allocation, slightly less optimal wrench tracking.
    # Smaller ε → more optimal but risks oscillation.
    # Tune: start at 0.01 and increase if spikes persist; decrease if wrench
    # error becomes visible. At ε=0.0 the pure unregularised NNLS is recovered.
    eps_reg :: Float64 = 0.01
    #
    # ── Autorotation (only reachable with a finite lift_shutoff_tilt) ──
    # A windmilling rotor in edgewise flow reaches an equilibrium speed where
    # aerodynamic driving torque (from the oncoming flow) balances profile drag.
    # Rather than the inflow-ratio formulation (which requires knowing blade
    # twist and lift-curve slope), we use a simpler momentum-based model:
    #
    #   ω_auto = autorotate_k · √(vx / R)
    #
    # This captures the key physics: autorotation RPM scales with √(airspeed)
    # because driving torque ∝ vx² and drag torque ∝ ω², so equilibrium gives
    # ω ∝ √vx. autorotate_k is a proportionality constant tuned so that at
    # nominal cruise speed the rotor spins at a sensible fraction of ω_nom
    # (typically 20–40%).
    #
    # Calibration: at vx=80 m/s, R=1 m, ω_nom=110 rad/s, targeting ω≈30 rad/s:
    #   k = 30 / √(80/1) ≈ 3.35
    #
    # Recovered electrical power uses the actuator-disk (wind-turbine) model:
    #   P_regen = -autorotate_eta · ½ · ρ · A · vx³ · Cp
    # where Cp is the rotor power coefficient (Betz limit = 0.593).
    # This correctly captures energy extraction from the freestream, which
    # dominates over shaft friction at cruise speeds.
    # At vx=80 m/s, R=1 m, ρ=1.225, Cp=0.05, η=0.70:
    #   P ≈ 0.70 × 0.5 × 1.225 × π × 80³ × 0.05 ≈ 8.7 kW per rotor
    #
    # NOTE: power extraction implies braking force F = P/vx. At Cp=0.35
    # (Betz-optimal) this is ~1700 N per rotor — unacceptable in cruise.
    # Cp=0.05 represents a drag-minimised, lightly-loaded autorotating rotor
    # (blades at low pitch) where regen is incidental rather than optimised.
    # The resulting braking force (~55 N per rotor) is negligible relative to
    # other drag sources. If regen drag should be captured explicitly, add it
    # as a force term in fly.jl and set autorotate_eta = 0.0 here.
    autorotate_k   :: Float64 = 3.35   # rad/s per √(m/s / m), see calibration above
    autorotate_eta :: Float64 = 0.70   # generator efficiency for regen (0–1)
    autorotate_Cp  :: Float64 = 0.05   # rotor power coefficient — drag-minimised autorotation
end

const ALLOC = AllocatorParams(power_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
# power_weights are the Λp diagonal entries, set to rated shaft power (kW) per rotor class:
#   R1/R2: TurboshaftEngine P_sl_W = 746 kW (1000 hp design point, powerplant.jl default)
#   R3–R6: ElectricMotor P_max_W  = 280 kW (rotor_system.jl default)
# Only ratios matter (allocator normalises internally); using rated kW directly
# is self-documenting and survives rotor config changes without manual rescaling.
#
# Rotor modes now come from rotor_config.csv. The stock CSV is six rows of
# lift=TRUE/thrust=TRUE → six tiltrotors → the validated 2026-06-29 all-6-rotor
# allocation, unchanged.

# ── Rotor mode resolution ─────────────────────────────────────────────
# Fall back to `true` when RotorUnit has no such field, so this file still
# loads against a pre-v0.2.0 rotor_system.jl. `hasproperty` with a literal
# Symbol on a concrete type constant-folds — no per-call cost in the ODE.
@inline _lift_of(u)   = hasproperty(u, :lift)   ? Bool(u.lift)   : true
@inline _thrust_of(u) = hasproperty(u, :thrust) ? Bool(u.thrust) : true

"""
    rotor_flags(fleet, ap) → (lift::NTuple{6,Bool}, thrust::NTuple{6,Bool})

Per-rotor mode flags, taken from the fleet (i.e. from rotor_config.csv) unless
`ap.lift_flags` / `ap.thrust_flags` override them.
"""
@inline function rotor_flags(fleet::RotorFleet, ap::AllocatorParams=ALLOC)
    lf = ap.lift_flags   === nothing ? ntuple(i -> _lift_of(fleet.units[i]),   6) : ap.lift_flags
    tf = ap.thrust_flags === nothing ? ntuple(i -> _thrust_of(fleet.units[i]), 6) : ap.thrust_flags
    return lf, tf
end

"""
    rotor_tilt(lift, thrust, tilt_rad) → Real

Effective tilt of one rotor's thrust axis. Tiltrotors follow the nacelle
command; lift-only rotors are fixed vertical; thrust-only rotors fixed
horizontal.
"""
@inline function rotor_tilt(lift::Bool, thrust::Bool, tilt_rad::Real)
    lift && thrust && return tilt_rad
    lift           && return _TILT_VERT
    thrust         && return _TILT_HORZ
    return _TILT_VERT          # out of service — the column is zeroed anyway
end

"""
    rotor_in_service(lift, thrust) → Bool

`false` only for lift=false, thrust=false: the rotor is dead weight. It still
contributes mass and inertia (airframe.jl) but no force, torque or power.
"""
@inline rotor_in_service(lift::Bool, thrust::Bool) = lift || thrust

"""
    rotor_active(lift, thrust, tilt_rad, ap) → Bool

`true` when the allocator may command thrust from this rotor.
  * tiltrotor      — always
  * lift-only      — until `ap.lift_shutoff_tilt` (Inf by default → always)
  * thrust-only    — from `ap.cruise_activation_tilt` upward
  * out of service — never
"""
@inline function rotor_active(lift::Bool, thrust::Bool,
                              tilt_rad::Float64, ap::AllocatorParams=ALLOC) :: Bool
    rotor_in_service(lift, thrust) || return false
    lift && thrust && return true
    lift           && return tilt_rad <  ap.lift_shutoff_tilt
    return                   tilt_rad ≥ ap.cruise_activation_tilt
end

"""
    rotor_active(i, fleet, tilt_rad, ap) → Bool

Index form. NOTE the signature change: v0.1.5 was `rotor_active(i, tilt_rad, ap)`.
The fleet is now required because modes live on the rotors, not in the params.
"""
@inline function rotor_active(i::Int, fleet::RotorFleet,
                              tilt_rad::Float64, ap::AllocatorParams=ALLOC) :: Bool
    lf, tf = rotor_flags(fleet, ap)
    rotor_active(lf[i], tf[i], tilt_rad, ap)
end

"""
    autorotating(lift, thrust, tilt_rad, ap) → Bool

A lift-only rotor past a finite `lift_shutoff_tilt`: declutched and windmilling.
Never true under the default `lift_shutoff_tilt = Inf`.
"""
@inline autorotating(lift::Bool, thrust::Bool, tilt_rad::Float64,
                     ap::AllocatorParams=ALLOC) =
    lift && !thrust && tilt_rad ≥ ap.lift_shutoff_tilt

# ── Autorotation helpers (opt-in, see lift_shutoff_tilt) ──────────────
"""
    autorotate_omega(u, vx, ap) → Float64

Windmilling RPM for a declutched lift rotor at forward speed `vx` (m/s).
  ω = autorotate_k · √(vx / R)
Clamped to [omega_min, omega_max].
"""
@inline function autorotate_omega(u, vx::Float64, ap::AllocatorParams=ALLOC) :: Float64
    ω = ap.autorotate_k * sqrt(max(vx, 0.0) / (u.radius_m + 1e-6))
    clamp(ω, ap.omega_min, ap.omega_max)
end

"""
    autorotate_kw(u, vx, ρ, ap) → Float64

Regenerative power (kW) for a windmilling lift rotor, actuator-disk
(wind-turbine) model. Negative = returned to the battery bus.
  P_regen = -autorotate_eta · ½ · ρ · A · vx³ · Cp
"""
@inline function autorotate_kw(u, vx::Float64, ρ::Float64,
                                ap::AllocatorParams=ALLOC) :: Float64
    A   = π * u.radius_m^2
    P   = ap.autorotate_eta * 0.5 * ρ * A * vx^3 * ap.autorotate_Cp
    return -P / 1000.0   # negative kW — energy returned to bus
end

"""
    idle_kw(u, ω, ρ) → Float64

Profile power of a rotor turning at the idle floor with no commanded thrust.
Small, but four mid rotors idling across a cruise leg is real bus load, so it
is not reported as zero.
"""
@inline idle_kw(u, ω::Float64, ρ::Float64) =
    u.c_profile * ρ * (π * u.radius_m^2) * (ω * u.radius_m)^3 / 1000.0

# ── Geometry matrix ────────────────────────────────────────────────────
"""
    build_B(fleet, tilt_rad, ap) → SMatrix{5,6,Float64}

Control effectiveness matrix. Column i is rotor i's contribution to the wrench
vector w = [Fx, Fz, Mx, My, Mz]ᵀ per unit thrust, using that rotor's own axis
angle τᵢ (see the per-rotor mode table in the header).
"""
function build_B(fleet::RotorFleet,
                 tilt_rad::Float64,
                 ap::AllocatorParams=ALLOC) :: SMatrix{5,6,Float64}

    lf, tf = rotor_flags(fleet, ap)
    B = MMatrix{5,6,Float64}(undef)

    for i in 1:6
        u  = fleet.units[i]
        rx = u.arm_x_m
        ry = u.arm_y_m
        sd = Float64(u.spin_dir)   # +1=CCW, -1=CW

        # Yaw torque arm: Q/T = (kQ/kT)·D where D = 2·R
        # Follows from the propeller coefficient convention used in blades.jl:
        #   T = kT·ρ·n²·D⁴  and  Q = kQ·ρ·n²·D⁵  (n in rev/s, D = 2R)
        # The n²·D factors cancel exactly, leaving a pure length in metres.
        torque_arm = (u.kQ / u.kT) * 2.0 * u.radius_m   # [m], ≈ 0.382 m default

        # Zero the column when the rotor is out of service, or gated off for
        # this phase (pusher below activation tilt, lift rotor past shutoff).
        on = rotor_active(lf[i], tf[i], tilt_rad, ap) ? 1.0 : 0.0
        τ  = rotor_tilt(lf[i], tf[i], tilt_rad)
        st = sin(τ) * on
        ct = cos(τ) * on
        cp = ap.tilt_axis_coupling ? st : 0.0   # off-vertical coupling terms

        B[1, i] = ap.w_fx     *  st                                # forward force
        B[2, i] = ap.w_thrust *  ct                                # lift
        B[3, i] = ap.w_roll   * ( ry * ct - sd * torque_arm * cp)  # roll  + reaction
        B[4, i] = ap.w_pitch  * ( rx * ct)                         # pitch
        B[5, i] = ap.w_yaw    * (-ry * cp - sd * torque_arm * ct)  # yaw   + reaction
    end

    return SMatrix{5,6,Float64}(B)
end

"""
    moment_authority(fleet, tilt_rad, ap) → Float64

Fraction of hover moment authority the fleet still holds at this tilt: the best
`cos τᵢ` over the rotors that are actually running.

For an all-tiltrotor fleet this is exactly `cos(tilt_rad)` — the v0.1.5 blend,
which exists because at 90° the rotors point forward and a full moment demand
makes the solve rank-deficient. A fleet with lift-only rotors keeps them
vertical, so this stays at 1.0 and those rotors retain full roll/pitch/yaw
authority in horizontal flight, which is the entire reason to run them there.

`moment_authority(nothing, …)` reproduces the v0.1.5 cos(tilt) blend.

Cheap enough to call every `build_ode` evaluation (fly.jl's integrator uses
`Rodas5P(autodiff=AutoFiniteDiff())`, which perturbs and re-evaluates with
plain Float64 — it never constructs Dual numbers, so there is no AD-safety
concern here to begin with).
"""
@inline moment_authority(::Nothing, tilt_rad::Real, ap::AllocatorParams=ALLOC) = cos(tilt_rad)

@inline function moment_authority(fleet::RotorFleet, tilt_rad::Real,
                                  ap::AllocatorParams=ALLOC)
    lf, tf = rotor_flags(fleet, ap)
    m = 0.0
    for i in 1:6
        rotor_active(lf[i], tf[i], Float64(tilt_rad), ap) || continue
        m = max(m, cos(rotor_tilt(lf[i], tf[i], tilt_rad)))
    end
    return m
end

"""
    fleet_thrust_fraction(fleet, tilt_rad, ap) → (fx_frac, fz_frac)

Cheap per-step stand-in for the allocator's thrust direction, for use inside
`build_ode` where a full NNLS solve is not affordable — `allocate_wrench_vx`
is only evaluated at the saving callback's ~10 Hz, while `build_ode` runs on
every integrator step (and every finite-difference perturbation of one).
fly.jl does not carry per-rotor RPM as ODE state — only an aggregate
`thrust_lag` scalar — so there is no per-rotor thrust to sum here even if the
NNLS solve were cheap enough; this approximates what that solve would give.

Approximates each active rotor's share of a single aggregate `thrust_cmd` by
its power_weight — the split `allocate_wrench` converges to under zero
differential moment demand — then blends sin/cos τᵢ by that share:

    fx_frac = Σ pᵢ·sin τᵢ / Σ pᵢ      (active i only)
    fz_frac = Σ pᵢ·cos τᵢ / Σ pᵢ

Gating matches `rotor_active` exactly: an idle pusher below
`cruise_activation_tilt` contributes nothing, so forward thrust only appears
once the pusher has actually spooled up, not the instant `tilt_rad` starts
sweeping. A dead rotor (lift=false, thrust=false) is excluded entirely.

For an all-tiltrotor fleet every τᵢ = tilt_rad regardless of power_weight, so
this reduces algebraically to EXACTLY `(sin(tilt_rad), cos(tilt_rad))` — the
v0.1.5 scalar model, bit-identical for the stock fleet. Verified in
`alloc_selftest` (Test 11).

Returns `(0.0, 0.0)` if every rotor is inactive — should not occur for a
valid config, but a config error should not also divide by zero.
"""
@inline function fleet_thrust_fraction(fleet::RotorFleet, tilt_rad::Float64,
                                       ap::AllocatorParams=ALLOC)
    lf, tf = rotor_flags(fleet, ap)
    pw = ap.power_weights
    fx = 0.0; fz = 0.0; psum = 0.0
    for i in 1:6
        rotor_active(lf[i], tf[i], tilt_rad, ap) || continue
        τ = rotor_tilt(lf[i], tf[i], tilt_rad)
        fx += pw[i] * sin(τ)
        fz += pw[i] * cos(τ)
        psum += pw[i]
    end
    psum < 1e-9 && return (0.0, 0.0)
    return (fx / psum, fz / psum)
end

# ── Control allocator ──────────────────────────────────────────────────
"""
    allocate_wrench(T_total, M_roll, M_pitch, M_yaw,
                    fleet, tilt_rad, rho_rel, ap)
        → (rpms::NTuple{6,Float64}, kws::NTuple{6,Float64})

Maps the wrench demand to per-rotor ω (rad/s) and electrical power (kW).
Units match `rotor_rpm_each` and `rotor_kw_each` in rotor_system.jl.

Signature is unchanged from v0.1.5. `T_total` is still the thrust *magnitude*
demanded along the nacelle axis; it is resolved into body-axis Fx/Fz here using
`tilt_rad` before the solve, so rotors with a fixed axis get the right share.
Solves

    T = arg min_{T ≥ 0} ‖BT - w‖_{Λp⁻²},   w = [Fx, Fz, Mx, My, Mz]ᵀ

via Lawson-Hanson NNLS after substitution T_sub = Λp⁻¹T, with Tikhonov
regularisation (ε = ap.eps_reg) to suppress inter-step corner-flipping.

NOTE: kW computed here uses the hover induced-velocity estimate (vx=0).
Use `allocate_wrench_vx` in the saving callback where vx is available.
"""
function allocate_wrench(T_total::Float64,
                          M_roll::Float64, M_pitch::Float64, M_yaw::Float64,
                          fleet::RotorFleet,
                          tilt_rad::Float64,
                          rho_rel::Float64,
                          ap::AllocatorParams=ALLOC) :: Tuple{NTuple{6,Float64}, NTuple{6,Float64}}
    ρ = max(rho_rel * 1.225, 0.01)
    lf, tf = rotor_flags(fleet, ap)

    # ── 1. Solve: min ‖Λp⁻¹T‖₂  s.t.  BT = w, T ≥ 0 ─────────────────
    #    Substituting T_sub = Λp⁻¹T:
    #      min ‖T_sub‖₂  s.t.  (B·Λp)·T_sub = w, T_sub ≥ 0
    #    then recover  T = Λp · T_sub
    #
    #    Tikhonov regularisation augments the system to suppress corner-flipping:
    #      [BΛp    ] T_sub ≈ [w]
    #      [λ·I₆   ]         [0]
    #    This adds a ‖T_sub‖² penalty pulling the solution toward the interior.
    B   = build_B(fleet, tilt_rad, ap)
    Λp  = Diagonal(SVector{6,Float64}(ap.power_weights))
    BΛp = B * Λp                                        # 5×6, weighted effectiveness matrix

    # Resolve the commanded thrust magnitude into body axes. For an
    # all-tiltrotor fleet (every τᵢ = tilt_rad) the sin/cos residuals recombine
    # exactly into the v0.1.5 single-thrust-row problem.
    Fx_d = T_total * sin(tilt_rad)
    Fz_d = T_total * cos(tilt_rad)
    w    = SVector{5,Float64}(Fx_d, Fz_d, M_roll, M_pitch, M_yaw)

    ε       = ap.eps_reg
    A_reg   = SMatrix{11,6,Float64}(vcat(Matrix(BΛp), ε .* Matrix(I, 6, 6)))
    b_reg   = SVector{11,Float64}(vcat(Vector(w), zeros(6)))

    T_sub = nnls(A_reg, b_reg)                          # solution in substituted space (T̃ = Λp⁻¹T)
    T_vec = Λp * T_sub                                  # recover physical thrust: T = Λp · T_sub

    # ── 2. Thrust → ω ─────────────────────────────────────────────────
    rpms = ntuple(6) do i
        u = fleet.units[i]
        rotor_in_service(lf[i], tf[i])           || return 0.0            # dead rotor
        rotor_active(lf[i], tf[i], tilt_rad, ap) || return ap.omega_min   # idling
        ω = sqrt(max(T_vec[i], 0.0) / (u.kT * ρ * u.radius_m^4 + 1e-9))
        clamp(ω, ap.omega_min, ap.omega_max)
    end

    # ── 3. Power (hover-mode induced velocity, vx=0) ───────────────────
    kws = ntuple(6) do i
        u = fleet.units[i]
        rotor_in_service(lf[i], tf[i]) || return 0.0
        rotor_active(lf[i], tf[i], tilt_rad, ap) ||
            return idle_kw(u, ap.omega_min, ρ)          # spinning, unloaded
        ω_i    = rpms[i]
        A_i    = π * u.radius_m^2
        T_i    = u.kT * ρ * ω_i^2 * u.radius_m^4
        vi_h   = sqrt(T_i / (2.0 * ρ * A_i + 1e-6))
        P_ind  = u.k_induced * T_i * vi_h / u.eta_rotor
        P_prof = u.c_profile * ρ * A_i * (ω_i * u.radius_m)^3
        (P_ind + P_prof) / 1000.0
    end

    return rpms, kws
end

# ── Lawson-Hanson NNLS ─────────────────────────────────────────────────────
# Solves min ‖Ax - b‖₂  s.t.  x ≥ 0  for small (m×n) systems.
# Ref: Lawson & Hanson, "Solving Least Squares Problems", 1974, Algorithm NNL.
# Accepts any m×6 system — called with 5×6 (bare) or 11×6 (Tikhonov augmented).
function nnls(A::SMatrix{m,6,Float64}, b::SVector{m,Float64}) :: SVector{6,Float64} where {m}
    n = 6
    x = @MVector zeros(Float64, 6)       # current solution (all passive)
    P = @MVector zeros(Bool, 6)          # true = active (free) set

    w = A' * (b - A * SVector(x))        # gradient of residual: A'(b - Ax)

    iter = 0
    while true
        # Find largest positive gradient component outside active set
        t = 0;  best = 0.0
        for i in 1:n
            if !P[i] && w[i] > best;  best = w[i];  t = i;  end
        end
        (t == 0 || best ≤ 1e-10) && break    # KKT satisfied

        P[t] = true                           # move t into active set

        # Inner loop: enforce x ≥ 0 on the active set
        while true
            iter += 1;  iter > 60 && break   # safety valve

            # Solve unconstrained LS on active columns
            P_idx = SVector{6,Bool}(P)
            A_P   = A[:, P_idx]               # m × |P| submatrix
            s_P   = pinv(A_P) * b

            # If all active components positive, accept and break inner loop
            all(s_P .≥ 0.0) && (for i in 1:n; x[i] = P[i] ? s_P[count(P[1:i])] : 0.0; end; break)

            # Find step length α to keep x ≥ 0, deactivate hitting components
            α = Inf
            for i in 1:n
                P[i] || continue
                si = s_P[count(P[1:i])]
                si < 0.0 && (α = min(α, x[i] / (x[i] - si)))
            end

            # Interpolate and deactivate any zero-crossers
            for i in 1:n
                if P[i]
                    si = s_P[count(P[1:i])]
                    x[i] = x[i] + α * (si - x[i])
                    x[i] < 1e-10 && (x[i] = 0.0; P[i] = false)
                end
            end
        end

        w = A' * (b - A * SVector(x))        # recompute gradient
    end

    return SVector{6,Float64}(x)
end

# ── vx-aware variant (use this in the saving callback) ────────────────
"""
    allocate_wrench_vx(T_total, M_roll, M_pitch, M_yaw,
                       fleet, tilt_rad, vx, alt, ap)
        → (rpms::NTuple{6,Float64}, kws::NTuple{6,Float64})

Identical to `allocate_wrench` but takes physical `vx` (m/s) and `alt` (m AGL)
so the edgewise induced-velocity correction in kW matches `rotor_kw_each`
exactly. Always prefer this variant in the callback.
"""
function allocate_wrench_vx(T_total::Float64,
                              M_roll::Float64, M_pitch::Float64, M_yaw::Float64,
                              fleet::RotorFleet,
                              tilt_rad::Float64,
                              vx::Float64, alt::Float64,
                              ap::AllocatorParams=ALLOC) :: Tuple{NTuple{6,Float64}, NTuple{6,Float64}}

    ρ      = rho(alt)          # from atmosphere.jl — same as rotor_kw_each
    rho_r  = ρ / 1.225
    v      = max(vx, 0.0)
    lf, tf = rotor_flags(fleet, ap)

    rpms, _ = allocate_wrench(T_total, M_roll, M_pitch, M_yaw,
                               fleet, tilt_rad, rho_r, ap)

    # Override non-producing rotors with the right RPM:
    #   out of service               → 0 (undriven, contributes nothing)
    #   lift-only past shutoff       → windmilling at a vx-dependent ω
    #   thrust-only below activation → idle floor, not yet spooled
    rpms = ntuple(6) do i
        rotor_in_service(lf[i], tf[i]) || return 0.0
        autorotating(lf[i], tf[i], tilt_rad, ap) &&
            return autorotate_omega(fleet.units[i], v, ap)
        rotor_active(lf[i], tf[i], tilt_rad, ap) || return ap.omega_min
        return rpms[i]
    end

    kws = ntuple(6) do i
        u = fleet.units[i]
        rotor_in_service(lf[i], tf[i]) || return 0.0
        autorotating(lf[i], tf[i], tilt_rad, ap) &&
            return autorotate_kw(u, v, ρ, ap)            # negative kW (regen)
        rotor_active(lf[i], tf[i], tilt_rad, ap) ||
            return idle_kw(u, ap.omega_min, ρ)
        ω_i = rpms[i]
        A_i = π * u.radius_m^2
        T_i = u.kT * ρ * ω_i^2 * u.radius_m^4

        vi_h = sqrt(T_i / (2.0 * ρ * A_i + 1e-6))
        vi   = vi_h^2 / sqrt(vi_h^2 + (v / 2.0)^2 + 1e-6)   # edgewise correction

        P_ind  = u.k_induced * T_i * vi / u.eta_rotor
        P_prof = u.c_profile * ρ * A_i * (ω_i * u.radius_m)^3
        (P_ind + P_prof) / 1000.0
    end

    return rpms, kws
end

# ── Realised forces (fly.jl force summation) ──────────────────────────
"""
    fleet_forces(rpms, fleet, tilt_rad, ρ, ap) → (Fx, Fz, Mx, My, Mz)

Body-frame totals actually produced by the fleet at the given per-rotor ω.
Mirrors `build_B` exactly, so the ODE sees the same geometry the allocator
solved against.

This replaces the `T_total·sin(tilt)` / `T_total·cos(tilt)` resolution
somewhere that DOES carry actual per-rotor RPM as state — currently that's
only the saving callback / postprocess path (`rpms` from
`allocate_wrench_vx`), since `build_ode` itself only tracks an aggregate
`thrust_lag` scalar and uses `fleet_thrust_fraction` instead (see below).
Generic in element type, so this is safe to call with whatever numeric type
`rpms` happens to hold, Dual or not — it just never receives one in the
current architecture.
"""
function fleet_forces(rpms, fleet::RotorFleet, tilt_rad, ρ,
                      ap::AllocatorParams=ALLOC)
    lf, tf = rotor_flags(fleet, ap)
    z  = zero(eltype(rpms)) * zero(typeof(ρ))
    Fx = Fz = Mx = My = Mz = z

    for i in 1:6
        u = fleet.units[i]
        rotor_in_service(lf[i], tf[i]) || continue

        τ      = rotor_tilt(lf[i], tf[i], tilt_rad)
        st, ct = sin(τ), cos(τ)
        cp     = ap.tilt_axis_coupling ? st : zero(st)
        sd     = Float64(u.spin_dir)
        ta     = (u.kQ / u.kT) * 2.0 * u.radius_m
        T      = u.kT * ρ * rpms[i]^2 * u.radius_m^4

        Fx += T *  st
        Fz += T *  ct
        Mx += T * ( u.arm_y_m * ct - sd * ta * cp)
        My += T * ( u.arm_x_m * ct)
        Mz += T * (-u.arm_y_m * cp - sd * ta * ct)
    end
    return (Fx, Fz, Mx, My, Mz)
end

# ── Wrench demand builder ──────────────────────────────────────────────
"""
    build_wrench(u, thrust_cmd, pitch_cmd, roll_cmd [, yaw_rate_cmd, tilt_rad, ap, fleet])
        → NTuple{4,Float64}  (T, Mx, My, Mz)

Reconstructs the wrench demand from controller outputs and ODE state.
State indices follow NOTES.md:
  u[5]=pitch  u[7]=roll  u[10]=dyaw (ωz in 6-DOF build)

`tilt_rad` — current nacelle tilt (rad).

`fleet` — pass the RotorFleet so moment demands scale by the authority the
fleet actually retains at this tilt (`moment_authority`). Omit it and the
v0.1.5 `cos(tilt_rad)` blend applies, which zeroes moment demand in cruise:
correct for an all-tiltrotor fleet, WRONG for any fleet with lift-only rotors,
since those keep full authority and are meant to be trimming the aircraft in
horizontal flight. Pass the fleet.

Called from the saving callback (~10 Hz) to reconstruct per-rotor RPM/kW for
CSV export and the cockpit — pure Float64 there, no AD constraint.

`build_ode` does NOT call this function. It only tracks an aggregate
`thrust_lag` scalar (no per-rotor RPM as ODE state), and computes its own
moment terms inline using `moment_authority` directly in place of a bare
`cos(tilt)` — see the "fly.jl integration" section at the bottom of this
file for exactly what changed there and why. (fly.jl's integrator is
`Rodas5P(autodiff=AutoFiniteDiff())`, which perturbs and re-evaluates with
plain Float64 rather than propagating Dual numbers, so there was never an
AD-safety constraint on `build_ode` internals to begin with — the original
note above about "Dual numbers from AutoFiniteDiff" conflated the two; worth
flagging in case a future switch to `AutoForwardDiff()` is actually planned,
since several functions in this file — `build_B`, `allocate_wrench`,
`rotor_active`, `autorotating` — are Float64-typed and would need loosening
to `Real` first.)
"""
function build_wrench(u::AbstractVector,
                      thrust_cmd::Float64,
                      pitch_cmd::Float64,
                      roll_cmd::Float64,
                      yaw_rate_cmd::Float64 = 0.0,
                      tilt_rad::Float64     = 0.0,
                      ap::AllocatorParams   = ALLOC,
                      fleet                 = nothing) :: NTuple{4,Float64}

    # Blends moment authority away as the rotors that generate it tilt over.
    # At tilt=90° an all-tiltrotor fleet has none — passing full moment demands
    # produces a rank-deficient problem and oscillating rotor saturation.
    # A fleet with untilting lift rotors holds ct = 1.
    ct = moment_authority(fleet, tilt_rad, ap)

    M_pitch = ap.pitch_moment_scale * (pitch_cmd    - Float64(u[5]))  * ct
    M_roll  = ap.roll_moment_scale  * (roll_cmd     - Float64(u[7]))  * ct
    M_yaw   = ap.yaw_moment_scale   * (yaw_rate_cmd - Float64(u[10])) * ct

    return (thrust_cmd, M_roll, M_pitch, M_yaw)
end

# ── Self-test ──────────────────────────────────────────────────────────
"""
    alloc_selftest()

Eleven sanity checks. Run from the REPL after loading rotor_system.jl:

    julia> include("subsystems/rotor_system.jl")
    julia> include("rotor_mixer.jl")
    julia> alloc_selftest()

Tests 8–10 drive the mode logic through `lift_flags` / `thrust_flags`, so they
run against an unmodified rotor_system.jl.
"""
function alloc_selftest()
    println("\n=== Control Allocator Self-Test ===\n")

    T_hov   = RP.hover_thrust_N   # 26 055 N
    rho_sl  = 1.0                 # sea-level ρ_rel
    rho_luk = 0.737               # Lukla 9334 ft: ρ≈0.902 kg/m³ → ρ_rel≈0.737

    pass = Bool[]

    # ── Test 1: symmetric hover — equal RPM ───────────────────────────
    rpms1, kws1 = allocate_wrench(T_hov, 0.0, 0.0, 0.0, FLEET, 0.0, rho_sl)
    rng1  = maximum(rpms1) - minimum(rpms1)
    ok1   = rng1 < 1.0
    T_chk = sum(FLEET.units[i].kT * 1.225 * rpms1[i]^2 * FLEET.units[i].radius_m^4
                for i in 1:6)
    push!(pass, ok1)
    println("Test 1 — Symmetric hover, no moments (expect uniform RPM):")
    println("  ω (rad/s) : ", round.(rpms1, digits=1))
    println("  kW        : ", round.(kws1,  digits=1))
    println("  RPM spread: $(round(rng1, digits=3)) rad/s  — $(ok1 ? "✓ PASS" : "✗ FAIL")")
    println("  Thrust check: $(round(T_chk,digits=0)) N  (target $(round(T_hov,digits=0)) N)")
    println()

    # ── Test 2: roll demand — left side spins faster ───────────────────
    # +Mx = right-wing-down → left rotors (y>0: R1,R3,R5) spin up
    rpms2, _ = allocate_wrench(T_hov, 500.0, 0.0, 0.0, FLEET, 0.0, rho_sl)
    left  = (rpms2[1] + rpms2[3] + rpms2[5]) / 3.0   # R1,R3,R5  (arm_y > 0)
    right = (rpms2[2] + rpms2[4] + rpms2[6]) / 3.0   # R2,R4,R6  (arm_y < 0)
    ok2   = left > right
    push!(pass, ok2)
    println("Test 2 — Roll +500 N·m (right-wing-down), hover:")
    println("  ω (rad/s) : ", round.(rpms2, digits=1))
    println("  Left mean (R1,R3,R5): $(round(left, digits=1))  Right mean: $(round(right, digits=1))")
    println("  Left > Right: $(ok2 ? "✓ PASS" : "✗ FAIL")  (expect true — left side lifts right wing)")
    println()

    # ── Test 3: pitch demand — forward rotors spin faster ───────────────
    # +My = nose-up → forward rotors (arm_x > 0: R1,R2) spin up.
    rpms3, _ = allocate_wrench(T_hov, 0.0, 500.0, 0.0, FLEET, 0.0, rho_sl)
    fwd  = (rpms3[1] + rpms3[2]) / 2.0   # R1,R2  arm_x=+3.0
    aft  = (rpms3[5] + rpms3[6]) / 2.0   # R5,R6  arm_x=−3.0
    ok3  = fwd > aft
    push!(pass, ok3)
    println("Test 3 — Pitch +500 N·m (nose-up), hover:")
    println("  ω (rad/s) : ", round.(rpms3, digits=1))
    println("  Fwd mean (R1,R2): $(round(fwd, digits=1))  Aft mean (R5,R6): $(round(aft, digits=1))")
    println("  Fwd > Aft: $(ok3 ? "✓ PASS" : "✗ FAIL")  (expect true — fwd rotors lift nose)")
    println()

    # ── Test 4: yaw demand — CW rotors spin faster ─────────────────────
    # +Mz = nose-right. CW rotors (spin_dir=−1): R2,R3,R6.
    rpms4, _ = allocate_wrench(T_hov, 0.0, 0.0, 300.0, FLEET, 0.0, rho_sl)
    cw_mean  = (rpms4[2] + rpms4[3] + rpms4[6]) / 3.0   # spin_dir=−1
    ccw_mean = (rpms4[1] + rpms4[4] + rpms4[5]) / 3.0   # spin_dir=+1
    ok4 = cw_mean > ccw_mean
    push!(pass, ok4)
    println("Test 4 — Yaw +300 N·m (nose-right), hover:")
    println("  ω (rad/s) : ", round.(rpms4, digits=1))
    println("  CW  mean (R2,R3,R6): $(round(cw_mean,  digits=1))")
    println("  CCW mean (R1,R4,R5): $(round(ccw_mean, digits=1))")
    println("  CW > CCW: $(ok4 ? "✓ PASS" : "✗ FAIL")  (expect true — CW rotors produce +Mz)")
    println()

    # ── Test 5: density altitude — Lukla RPM scaling ───────────────────
    rpms5, _ = allocate_wrench(T_hov, 0.0, 0.0, 0.0, FLEET, 0.0, rho_luk)
    expected = 1.0 / sqrt(rho_luk)   # ≈ 1.164
    actual   = rpms5[1] / rpms1[1]
    ok5 = abs(actual - expected) < 0.01
    push!(pass, ok5)
    println("Test 5 — Symmetric hover at Lukla (ρ_rel=$(rho_luk)):")
    println("  ω (rad/s) : ", round.(rpms5, digits=1))
    println("  RPM ratio Lukla/SL : $(round(actual,   digits=4))")
    println("  Expected (1/√ρ_rel): $(round(expected, digits=4))")
    println("  Match (±0.01): $(ok5 ? "✓ PASS" : "✗ FAIL")")
    println()

    # ── Test 6: stock all-tiltrotor fleet unchanged at cruise ──────────
    # The Fx/Fz split must recombine into the old single-thrust-row problem:
    # Σ T ≈ demand and all six rotors still loaded.
    tilt_cruise = deg2rad(85.0)
    T_dem6      = T_hov * 0.3
    rpms6, kws6 = allocate_wrench_vx(T_dem6, 0.0, 0.0, 0.0,
                      FLEET, tilt_cruise, 80.0, 300.0)
    ρ6   = rho(300.0)
    T_s6 = sum(FLEET.units[i].kT * ρ6 * rpms6[i]^2 * FLEET.units[i].radius_m^4 for i in 1:6)
    all_live6 = all(rpms6[i] > ALLOC.omega_min for i in 1:6)
    trim6     = abs(T_s6 - T_dem6) / T_dem6 < 0.05
    ok6 = all_live6 && trim6
    push!(pass, ok6)
    println("Test 6 — Stock all-tiltrotor fleet at cruise (tilt=85°, vx=80 m/s):")
    println("  ω (rad/s)  : ", round.(rpms6, digits=1))
    println("  kW         : ", round.(kws6,  digits=1))
    println("  Σ thrust   : $(round(T_s6, digits=0)) N  (demand $(round(T_dem6, digits=0)) N)")
    println("  All six loaded: $(all_live6 ? "✓" : "✗")   Σ T within 5%: $(trim6 ? "✓" : "✗")")
    println("  $(ok6 ? "✓ PASS" : "✗ FAIL")")
    println()

    # ── Test 7: power-weighted allocation prefers high-power rotors ───────
    ap_biased = AllocatorParams(power_weights = (4.0, 4.0, 1.0, 1.0, 1.0, 1.0))
    rpms7, _ = allocate_wrench(T_hov, 0.0, 0.0, 0.0, FLEET, 0.0, rho_sl, ap_biased)
    T7 = ntuple(i -> FLEET.units[i].kT * 1.225 * rpms7[i]^2 * FLEET.units[i].radius_m^4, 6)
    hi_mean = (T7[1] + T7[2]) / 2.0
    lo_mean = (T7[3] + T7[4] + T7[5] + T7[6]) / 4.0
    ok7 = hi_mean > lo_mean * 1.5   # expect ~4× loading ratio
    push!(pass, ok7)
    println("Test 7 — Power-weighted hover (Λp: R1/R2 = 4×, R3–R6 = 1×):")
    println("  ω (rad/s)          : ", round.(rpms7, digits=1))
    println("  Thrust R1/R2 mean  : $(round(hi_mean, digits=0)) N")
    println("  Thrust R3–R6 mean  : $(round(lo_mean, digits=0)) N")
    println("  Hi/Lo ratio        : $(round(hi_mean / max(lo_mean,1e-6), digits=2))×  (expect ≈ 4×)")
    println("  $(ok7 ? "✓ PASS" : "✗ FAIL")")
    println()

    # ── Test 8: BETA topology — fixed lift + fixed pusher, no tilting ──
    # R1/R2 (fwd) and R5/R6 (aft) lift-only; R3/R4 (mid) thrust-only pushers.
    # Nothing tilts. "Transition" is purely the pushers spooling up while the
    # wings take over lift and the lift rotors unload.
    #
    # Arm geometry matters when you pick a mixed fleet. This one trims: the
    # four lift rotors straddle the CG in x (±3.0) and y (±4.5), and the two
    # pushers sit at ∓5.5 in y so their thrust yaw and torque reaction cancel
    # pairwise. Note that "front tilt / mid lift / rear thrust" does NOT trim
    # in hover with the stock arms — R1/R2 at arm_x=+3.0 would be the only
    # lift forward of the CG and the allocator has to zero them to hold pitch.
    tilt_cr = deg2rad(85.0)
    ap_beta = AllocatorParams(
        lift_flags    = (true,  true,  false, false, true,  true ),
        thrust_flags  = (false, false, true,  true,  false, false),
        power_weights = (746.0, 746.0, 280.0, 280.0, 280.0, 280.0),
    )
    # Hover: pushers idle at omega_min, the four lift rotors carry the aircraft
    rpms8h, kws8h = allocate_wrench(T_hov, 0.0, 0.0, 0.0, FLEET, 0.0, rho_sl, ap_beta)
    push_idle = all(rpms8h[i] ≈ ap_beta.omega_min for i in (3, 4))
    lift_up   = all(rpms8h[i] >  ap_beta.omega_min for i in (1, 2, 5, 6))
    # Cruise: pushers spooled and carrying the Fx demand
    rpms8c, kws8c = allocate_wrench_vx(T_hov * 0.3, 0.0, 0.0, 0.0,
                        FLEET, tilt_cr, 80.0, 300.0, ap_beta)
    push_spun = all(rpms8c[i] > ap_beta.omega_min for i in (3, 4))
    ok8 = push_idle && lift_up && push_spun
    push!(pass, ok8)
    println("Test 8 — BETA topology (R1/R2 + R5/R6 lift-only, R3/R4 pushers, no tilt):")
    println("  Hover  ω : ", round.(rpms8h, digits=1), "  (R3/R4 expect $(ap_beta.omega_min))")
    println("  Hover  kW: ", round.(kws8h,  digits=1))
    println("  Cruise ω : ", round.(rpms8c, digits=1), "  (R3/R4 expect > $(ap_beta.omega_min))")
    println("  Cruise kW: ", round.(kws8c,  digits=1))
    println("  Pushers idle in hover  : $(push_idle ? "✓" : "✗")")
    println("  Lift rotors carrying   : $(lift_up   ? "✓" : "✗")")
    println("  Pushers spooled cruise : $(push_spun ? "✓" : "✗")")
    println("  $(ok8 ? "✓ PASS" : "✗ FAIL")")
    println()

    # ── Test 9: lift-only rotors keep control authority in cruise ──────
    # The requirement: they keep running in horizontal flight to supply the
    # lift and control the wings don't. moment_authority must hold at 1.0, and
    # a roll demand at "tilt"=85° must still be answered differentially by the
    # untilted lift rotors — where an all-tiltrotor fleet would have gone deaf.
    ma_beta  = moment_authority(FLEET, tilt_cr, ap_beta)
    ma_stock = moment_authority(FLEET, tilt_cr, ALLOC)
    rpms9, _ = allocate_wrench_vx(T_hov * 0.3, 800.0, 0.0, 0.0,
                        FLEET, tilt_cr, 80.0, 300.0, ap_beta)
    roll_split = (rpms9[1] + rpms9[5]) - (rpms9[2] + rpms9[6])   # +y arms vs −y
    ok9 = ma_beta > 0.99 && ma_stock < 0.10 && roll_split > 1.0
    push!(pass, ok9)
    println("Test 9 — Lift-only authority in horizontal flight (tilt=85°):")
    println("  moment_authority, BETA fleet : $(round(ma_beta,  digits=3))  (expect ≈ 1.0)")
    println("  moment_authority, all-tilt   : $(round(ma_stock, digits=3))  (expect ≈ cos85° = 0.087)")
    println("  ω under +800 N·m roll : ", round.(rpms9, digits=1))
    println("  (R1+R5) − (R2+R6) : $(round(roll_split, digits=1)) rad/s  (expect > 0)")
    println("  $(ok9 ? "✓ PASS" : "✗ FAIL")")
    println()

    # ── Test 10: out-of-service rotor ─────────────────────────────────
    # R6 dead: draws nothing, turns nothing, but the fleet keeps flying on the
    # other five. Σ thrust is reported, not asserted — losing a corner rotor
    # breaks the hover symmetry, so the allocator trades total lift against
    # holding roll/pitch, and how much it can recover is a property of the
    # airframe rather than of this allocator.
    ap_oos = AllocatorParams(
        lift_flags   = (true, true, true, true, true, false),
        thrust_flags = (true, true, true, true, true, false),
    )
    rpms10, kws10 = allocate_wrench(T_hov, 0.0, 0.0, 0.0, FLEET, 0.0, rho_sl, ap_oos)
    T_10 = sum(FLEET.units[i].kT * 1.225 * rpms10[i]^2 * FLEET.units[i].radius_m^4 for i in 1:6)
    dead_still = rpms10[6] == 0.0 && kws10[6] == 0.0
    some_up    = count(rpms10[i] > ALLOC.omega_min for i in 1:5) ≥ 3
    ok10 = dead_still && some_up
    push!(pass, ok10)
    println("Test 10 — R6 out of service (lift=FALSE, thrust=FALSE):")
    println("  ω (rad/s) : ", round.(rpms10, digits=1))
    println("  kW        : ", round.(kws10,  digits=1))
    println("  R6 stopped, 0 kW  : $(dead_still ? "✓" : "✗")")
    println("  Fleet still loaded: $(some_up   ? "✓" : "✗")")
    println("  Σ thrust  : $(round(T_10, digits=0)) N of $(round(T_hov, digits=0)) N demanded")
    println("              ($(round(100*T_10/T_hov, digits=1))% — the shortfall is the")
    println("               engine-out lift margin, report it, do not tune it away)")
    println("  $(ok10 ? "✓ PASS" : "✗ FAIL")")
    println()

    # ── Test 11: fleet_thrust_fraction — the build_ode-facing helper ───
    # (a) stock all-tiltrotor fleet: must reduce EXACTLY to sin/cos(tilt)
    #     across a tilt sweep, since build_ode substitutes this for the old
    #     scalar formula and hover/cruise must not move.
    # (b) BETA topology below cruise_activation_tilt: pusher idle, so
    #     fx_frac ≈ 0 even though tilt_rad itself is already sweeping.
    # (c) BETA topology above: fx_frac matches the pusher's power-weight share.
    stock_ok = true
    for tdeg in 0:15:90
        t = deg2rad(Float64(tdeg))
        fx, fz = fleet_thrust_fraction(FLEET, t)
        stock_ok &= abs(fx - sin(t)) < 1e-9 && abs(fz - cos(t)) < 1e-9
    end
    fx_below, _ = fleet_thrust_fraction(FLEET, deg2rad(30.0), ap_beta)   # < 45°
    fx_above, _ = fleet_thrust_fraction(FLEET, deg2rad(60.0), ap_beta)   # > 45°
    beta_ok = abs(fx_below) < 1e-9 && fx_above > 0.5
    ok11 = stock_ok && beta_ok
    push!(pass, ok11)
    println("Test 11 — fleet_thrust_fraction (build_ode-facing helper):")
    println("  Stock fleet matches sin/cos(tilt) over 0–90°: $(stock_ok ? "✓" : "✗")")
    println("  BETA pusher idle below activation (tilt=30°) : fx_frac=$(round(fx_below,digits=3))  (expect 0.0)")
    println("  BETA pusher active above activation (tilt=60°): fx_frac=$(round(fx_above,digits=3))  (expect > 0.5)")
    println("  $(ok11 ? "✓ PASS" : "✗ FAIL")")
    println()

    println("=== $(count(pass))/$(length(pass)) tests passed ===\n")
end

# ══ fly.jl integration (applied 2026-07-26) ═══════════════════════════
#
# build_ode does NOT track per-rotor RPM — only an aggregate `thrust_lag`
# scalar (state 11) — and its integrator is
# `Rodas5P(autodiff=AutoFiniteDiff())`, which perturbs and re-evaluates with
# plain Float64 rather than propagating Dual numbers. Both of those turned
# out to matter: they rule out calling the NNLS-based `allocate_wrench` (or
# `fleet_forces` on a per-rotor RPM state that does not exist) every
# integrator step, but they also mean there was never a real AD-safety
# constraint on `build_ode` internals — the "ForwardDiff-safe" language
# in earlier drafts of this file conflated the two libraries. Corrected in
# this file's docstrings; flagging here in case a switch to
# `AutoForwardDiff()` is genuinely planned, since `build_B`, `allocate_wrench`,
# `rotor_active`, `autorotating` are Float64-typed and would need loosening to
# `Real` first.
#
# What was actually wired into fly.jl, in place of the "wire in fleet_forces"
# plan sketched in an earlier draft of this file:
#
# 1. Moments — `ct = cos(tilt); ct_eff = MANUAL ? 1.0 : ct` became
#      ct_eff = MANUAL ? 1.0 : moment_authority(FLEET, tilt, ALLOC)
#    Manual (HOTAS) flight keeps full authority regardless of tilt, as before
#    — untouched. For AUTO, this is the fix that actually matters: without it
#    a fleet with lift-only rotors goes deaf in roll/pitch/yaw through cruise,
#    even though those rotors are still upright and able to answer. For an
#    all-tiltrotor fleet this is bit-identical to the old `cos(tilt)` (proved
#    by Test 9's `ma_stock` check against cos85°).
#
# 2. Forces — every `thrust_act * sin(tilt)` / `thrust_act * cos(tilt)` that
#    represents ROTOR thrust direction (not wing aero, not ground effect,
#    which are untouched) was replaced with `thrust_act * fx_frac` /
#    `thrust_act * fz_frac`, computed once per build_ode call as
#      fx_frac, fz_frac = fleet_thrust_fraction(FLEET, tilt, ALLOC)
#    Bit-identical to v0.1.5 for the stock fleet (fleet_thrust_fraction
#    reduces exactly to sin/cos(tilt) — see its docstring). For a mixed
#    fleet, forward thrust now only appears once a pusher has actually
#    reached `cruise_activation_tilt`, and vertical thrust is not falsely
#    attributed to a pusher's share.
#
#    ONE formula was deliberately left untouched: the transition-phase Fz
#      thrust_act + wing_lift(...) - WEIGHT_N
#    has no cos(tilt)/fz_frac factor at all — a pre-existing, documented
#    choice to keep the physics consistent with what the autopilot was tuned
#    assuming (see the comment at that line). Introducing fz_frac there would
#    change the validated KAXX→KSAF transition profile, which is out of scope
#    for this pass. If lift rotors ever get their own AP-visible spool logic
#    distinct from thrust_lag, revisit this line specifically.
#
# 3. Saving callback (~10 Hz, real per-rotor RPM) — the build_wrench call
#    gained the fleet argument:
#      T_w, Mx_w, My_w, Mz_w = build_wrench(u, thrust_cmd, pitch_cmd, roll_cmd,
#                                           0.0, tilt_f, ALLOC, FLEET)
#    same reasoning as (1), for the telemetry/battery path.
#
# 4. fleet_forces (per-rotor RPM → Fx/Fz/Mx/My/Mz) is NOT currently called
#    anywhere in fly.jl — there is no per-rotor RPM ODE state for it to
#    consume. Kept as a utility for the saving callback (e.g. a telemetry
#    cross-check: does the fleet's realised wrench at the allocator's output
#    RPM match what was demanded?) or for a future architecture that does
#    carry per-rotor state. Generic in element type regardless.
#
# 5. Mass properties — unchanged. An out-of-service rotor still carries its
#    weight and inertia; that comes from airframe.jl, which knows nothing
#    about the lift/thrust flags and should keep it that way.
#
# 6. Symbols changed in v0.2.0 — grep before you build:
#      REMOVED  lift_only_rotors, cruise_only_rotors  (AllocatorParams fields)
#      REMOVED  lift_rotor_active(i, tilt, ap), cruise_rotor_active(i, tilt, ap)
#      CHANGED  rotor_active(i, tilt, ap) → rotor_active(i, fleet, tilt, ap)
#      CHANGED  build_B returns SMatrix{5,6} (was 4×6)
#      CHANGED  RotorUnit gains `lift::Bool`, `thrust::Bool` fields (both
#               default true) — see rotor_system.jl
#      NEW      rotor_flags, rotor_tilt, rotor_in_service, autorotating,
#               moment_authority, fleet_thrust_fraction, idle_kw, fleet_forces
#      NEW      AllocatorParams fields: lift_flags, thrust_flags, w_fx,
#               tilt_axis_coupling
#      DEFAULT  lift_shutoff_tilt 70° → Inf (lift rotors now keep running)
#
#    Revalidation: fly the KAXX→KSAF leg once with
#      AllocatorParams(tilt_axis_coupling = false)
#    to reproduce v0.1.5 exactly (both the moment terms AND the Fx/Fz terms —
#    fleet_thrust_fraction and moment_authority are both provably
#    tilt-axis-coupling-independent no-ops for the stock fleet regardless of
#    this switch, so the switch only affects the mixed-mode terms in build_B),
#    then again with the default to see what the tilted-axis yaw coupling
#    costs. Hover is unchanged either way.
#
# 7. Not modelled: freewheeling drag of a dead rotor, any yaw asymmetry from a
#    failed lift rotor beyond what the allocator trims out, and — because
#    `fleet_thrust_fraction` is a power-weighted analytic blend rather than a
#    real per-step solve — any transient where the ODE's Fx/Fz assumption
#    (proportional-to-power split, zero differential moment demand) and the
#    saving callback's actual NNLS-solved RPM diverge under a hard turn or a
#    saturated rotor. The two paths agree at equilibrium; they are not
#    guaranteed to agree instant-by-instant. Worth an eye during the BETA
#    topology's first validation flight.
# =====================================================================

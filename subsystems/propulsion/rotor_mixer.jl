# rotor_mixer.jl:     Wrench-to-RPM control allocator
# AUTHOR:             DANIEL DESAI
# UPDATED:            2026-06-29
# VERSION:            0.1.2
#
# Post-solve mapping: takes the aggregate wrench demand from the flight
# controller (total thrust + three moment demands) and distributes it
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
# ── B-matrix rows ─────────────────────────────────────────────────────
#   [0] thrust  : B[1,i] = 1          (all rotors contribute)
#   [1] roll Mx : B[2,i] = arm_y · ct  (+y arm, +thrust → right-wing-down)
#   [2] pitch My: B[3,i] = -arm_x · ct (+x arm, +thrust → nose-down → negate)
#   [3] yaw Mz  : B[4,i] = -spin_dir · torque_arm · ct
#                           CCW (spin=+1) spinning faster → CCW reaction on frame
#                           = nose-LEFT = -Mz → contribution is -spin_dir · τ_arm
#
# ── cos(tilt) blend ───────────────────────────────────────────────────
#   At hover  (tilt=0):   full roll/pitch/yaw authority from RPM differential.
#   At cruise (tilt=π/2): rotors point forward, their thrust adds in x not z,
#                         so body-frame moment contributions → 0. Aerodynamics
#                         takes over. cos(tilt) captures this continuously.
#
# ── Files to change in the project ────────────────────────────────────
#   control_allocator.jl   ← this file (new, place alongside fly.jl)
#   fly.jl                 ← see "fly.jl integration" section at bottom
#   rotor_system.jl        ← NO changes needed (all required fields exist)
# =====================================================================

using StaticArrays
using LinearAlgebra: pinv, Diagonal

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

    # Row weights in the pseudoinverse (scales rows of B before inversion).
    # Reduce a weight to deprioritise that channel when rotors saturate.
    w_thrust :: Float64 = 1.0
    w_roll   :: Float64 = 1.0
    w_pitch  :: Float64 = 1.0
    w_yaw    :: Float64 = 0.5   # limited authority on fixed-pitch rotors

    # ω limits (rad/s). S4-class: R≈1 m, ω_nom≈110 rad/s, tip ≈110 m/s.
    omega_min :: Float64 =  20.0   # flight-idle floor
    omega_max :: Float64 = 180.0   # structural / acoustic limit

    # Daisy-chain reallocation passes after clipping negative thrusts.
    realloc_iters :: Int = 2

    # Lift-only rotor shutdown
    # Indices of rotors that provide lift only and do not produce thrust in
    # cruise. In the default configuration these are R3 and R4 (mid-L /
    # mid-R, arm_x=0). All six rotors share the same nacelle tilt angle —
    # the mid rotors do not fold or reorient; they windmill (autorotate) in
    # cruise rather than tilting out of the way.
    # These rotors are held at omega_min once tilt exceeds `lift_shutoff_tilt`
    # so the pseudoinverse does not assign them spurious small thrust values.
    # Set to an empty tuple for a fully-tilting all-rotor configuration.
    lift_only_rotors   :: NTuple{2,Int}  = (3, 4)
    lift_shutoff_tilt  :: Float64        = deg2rad(70.0)   # ≈ 70° → shut down

    # ── Per-rotor power-preference weights ────────────────────────────────
    # Diagonal of the column-weighting matrix Λp used in the weighted
    # pseudoinverse: min ‖W⁻¹ · T_vec‖₂.
    #
    # A larger weight tells the allocator to "prefer" that rotor — it will
    # be assigned proportionally more thrust before saturating neighbours.
    # Set weights proportional to each rotor's rated shaft power so that,
    # e.g., a turbine-electric "Super rotor" (630 kW) gets 2.25× the weight of
    # a stock electric rotor (280 kW).
    #
    # Only the *ratios* matter — the allocator normalises internally.
    # Equal weights reproduce the original uniform pseudoinverse behaviour.
    power_weights :: NTuple{6,Float64} = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # Autorotation model for windmilling lift-only rotors in cruise.
    #
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

const ALLOC = AllocatorParams(power_weights = (280.0, 280.0, 280.0, 280.0, 280.0, 280.0))
# power_weights set to rated shaft power (kW) per rotor class:
#   R1/R2: TurboshaftEngine P_sl_W = 746 kW (1000 hp design point, powerplant.jl default)
#   R3–R6: ElectricMotor P_max_W  = 280 kW (rotor_system.jl default)
# Only ratios matter (allocator normalises internally); using rated kW directly
# is self-documenting and survives rotor config changes without manual rescaling.

# ── Lift-rotor autorotation helpers ───────────────────────────────────
"""
    lift_rotor_active(i, tilt_rad, ap) → Bool

Returns `false` when rotor `i` is a lift-only rotor and tilt has exceeded
`ap.lift_shutoff_tilt` — i.e. the rotor is windmilling, not producing thrust.
"""
@inline function lift_rotor_active(i::Int,
                                   tilt_rad::Float64,
                                   ap::AllocatorParams=ALLOC) :: Bool
    i ∈ ap.lift_only_rotors || return true
    return tilt_rad < ap.lift_shutoff_tilt
end

"""
    autorotate_omega(u, vx, ap) → Float64

Windmilling RPM for a lift-only rotor at forward speed `vx` (m/s).
  ω = autorotate_k · √(vx / R)
Clamped to [omega_min, omega_max].
"""
@inline function autorotate_omega(u, vx::Float64, ap::AllocatorParams=ALLOC) :: Float64
    ω = ap.autorotate_k * sqrt(max(vx, 0.0) / (u.radius_m + 1e-6))
    clamp(ω, ap.omega_min, ap.omega_max)
end

"""
    autorotate_kw(u, vx, ρ, ap) → Float64

Regenerative power (kW) for a windmilling lift-only rotor using the
actuator-disk (wind-turbine) model. Negative = returned to battery bus.
  P_regen = -autorotate_eta · ½ · ρ · A · vx³ · Cp
"""
@inline function autorotate_kw(u, vx::Float64, ρ::Float64,
                                ap::AllocatorParams=ALLOC) :: Float64
    A   = π * u.radius_m^2
    P   = ap.autorotate_eta * 0.5 * ρ * A * vx^3 * ap.autorotate_Cp
    return -P / 1000.0   # negative kW — energy returned to bus
end

# ── Geometry matrix ────────────────────────────────────────────────────
"""
    build_B(fleet, tilt_rad, ap) → SMatrix{4,6,Float64}

Control effectiveness matrix. Column i is rotor i's contribution to
the wrench [T_total, Mx, My, Mz] per unit thrust.
"""
function build_B(fleet::RotorFleet,
                 tilt_rad::Float64,
                 ap::AllocatorParams=ALLOC) :: SMatrix{4,6,Float64}

    ct = cos(tilt_rad)   # hover=1, cruise=0

    B = MMatrix{4,6,Float64}(undef)

    for i in 1:6
        u  = fleet.units[i]
        rx = u.arm_x_m
        ry = u.arm_y_m
        sd = Float64(u.spin_dir)   # +1=CCW, -1=CW

        # Yaw torque arm: Q/T = (kQ/kT)·D where D = 2·R
        # Follows from the propeller coefficient convention used in blades.jl:
        #   T = kT·ρ·n²·D⁴  and  Q = kQ·ρ·n²·D⁵  (n in rev/s, D = 2R)
        # The n²·D factors cancel exactly, leaving a pure length in metres.
        # The previous formula (kQ/kT)·ω_nom·R incorrectly included ω_nom,
        # inflating the yaw arm by ω_nom/2 ≈ 65× and over-weighting yaw authority.
        torque_arm = (u.kQ / u.kT) * 2.0 * u.radius_m   # [m], τᵢ ≈ 0.382 m default

        # Lift-only rotors contribute nothing to the wrench in cruise —
        # zero their column so the pseudoinverse ignores them entirely.
        active = lift_rotor_active(i, tilt_rad, ap) ? 1.0 : 0.0

        B[1, i] = ap.w_thrust * 1.0            * active
        B[2, i] = ap.w_roll   * ry * ct        * active
        B[3, i] = ap.w_pitch  * rx * ct        * active
        B[4, i] = ap.w_yaw    * (-sd) * torque_arm * ct * active
    end

    return SMatrix{4,6,Float64}(B)
end

# ── Power-weighted pseudoinverse ───────────────────────────────────────
"""
    build_Bp_weighted(B, ap) → SMatrix{6,4,Float64}

Computes the power-weighted pseudoinverse of the control effectiveness
matrix B.

The standard `pinv(B)` minimises ‖T_vec‖₂ uniformly across all rotors.
Here we minimise ‖W⁻¹ · T_vec‖₂ instead, where Λp = diag(power_weights
normalised so max = 1). This biases the solution toward rotors with
higher rated power: they receive proportionally larger thrust assignments
before the solver burdens lower-rated neighbours.

Derivation:
  Define scaled matrix  B̃ = B · Λp          (4×6)
  Solve  min ‖T̃‖₂  subject to  B · Λp · T̃ = wrench
  Recover  T_vec = W · T̃  →  T_vec = Λp · pinv(B̃) · wrench

So the weighted pseudoinverse is:
  Bp_w = Λp · pinv(B · Λp)

Called from `allocate_wrench` in place of the bare `pinv(B)`.
"""
function build_Bp_weighted(B::SMatrix{4,6,Float64},
                           ap::AllocatorParams) :: SMatrix{6,4,Float64}
    # Normalise so the largest weight = 1 (preserves total thrust scale).
    w_max  = maximum(ap.power_weights)
    w_norm = ap.power_weights ./ max(w_max, 1e-9)
    lambda_p = Diagonal(SVector{6,Float64}(w_norm))   # 6×6

    # Scaled effectiveness matrix and its pseudoinverse.
    B_scaled  = B * lambda_p                                          # 4×6
    Bp_scaled = SMatrix{6,4,Float64}(pinv(Matrix(B_scaled)))  # 6×4

    # Un-scale: multiply back by Λp to recover T_vec in original units.
    return SMatrix{6,4,Float64}(lambda_p * Bp_scaled)
end
"""
    allocate_wrench(T_total, M_roll, M_pitch, M_yaw,
                    fleet, tilt_rad, rho_rel, ap)
        → (rpms::NTuple{6,Float64}, kws::NTuple{6,Float64})

Maps a 4-DOF wrench to per-rotor ω (rad/s) and electrical power (kW).
Units match `rotor_rpm_each` and `rotor_kw_each` in rotor_system.jl.

NOTE: kW computed here uses the hover induced-velocity estimate (vx=0).
Use `allocate_wrench_vx` in the saving callback where vx is available.
"""
function allocate_wrench(T_total::Float64,
                          M_roll::Float64, M_pitch::Float64, M_yaw::Float64,
                          fleet::RotorFleet,
                          tilt_rad::Float64,
                          rho_rel::Float64,
                          ap::AllocatorParams=ALLOC) :: Tuple{NTuple{6,Float64}, NTuple{6,Float64}}

    ρ = max(rho_rel * 1.225, 0.01)   # kg/m³

    # ── 1. Power-weighted pseudoinverse solution ───────────────────────
    B  = build_B(fleet, tilt_rad, ap)
    Bp = build_Bp_weighted(B, ap)   # prefers high-power rotors via Λp·pinv(B·Λp)
    w  = SVector{4,Float64}(T_total, M_roll, M_pitch, M_yaw)
    T_vec = MVector{6,Float64}(Bp * w)   # per-rotor thrust (N), may have negatives

    # ── 2–3. Clip negatives, redistribute lost wrench ──────────────────
    for _ in 1:ap.realloc_iters
        any(<(0.0), T_vec) || break

        T_lost = MVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
        n_free = 0
        for i in 1:6
            if T_vec[i] < 0.0
                for r in 1:4; T_lost[r] += B[r, i] * T_vec[i]; end
                T_vec[i] = 0.0
            else
                n_free += 1
            end
        end
        n_free == 0 && break

        # Spread lost wrench across free rotors via pseudoinverse
        δ = Bp * SVector{4,Float64}(T_lost) / n_free
        for i in 1:6
            T_vec[i] > 0.0 && (T_vec[i] -= δ[i])
        end
    end
    for i in 1:6; T_vec[i] = max(T_vec[i], 0.0); end

    # ── 4. Thrust → ω (rotor_system.jl formula: T = kT·ρ·ω²·R⁴) ──────
    rpms = ntuple(6) do i
        u = fleet.units[i]
        # Windmilling rotors: omega_min here (no vx available in this variant).
        # Use allocate_wrench_vx in the saving callback for correct autorotation RPM.
        if !lift_rotor_active(i, tilt_rad, ap)
            return ap.omega_min
        end
        ω = sqrt(max(T_vec[i], 0.0) / (u.kT * ρ * u.radius_m^4 + 1e-9))
        clamp(ω, ap.omega_min, ap.omega_max)
    end

    # ── 5. Power (hover-mode induced velocity, vx=0) ───────────────────
    kws = ntuple(6) do i
        u   = fleet.units[i]
        # Windmilling in cruise: no vx here so report 0.0 rather than guess.
        # allocate_wrench_vx returns the correct negative regen value.
        lift_rotor_active(i, tilt_rad, ap) || return 0.0
        ω_i = rpms[i]
        A_i = π * u.radius_m^2
        T_i = u.kT * ρ * ω_i^2 * u.radius_m^4

        vi_h   = sqrt(T_i / (2.0 * ρ * A_i + 1e-6))
        P_ind  = u.k_induced * T_i * vi_h / u.eta_rotor
        P_prof = u.c_profile * ρ * A_i * (ω_i * u.radius_m)^3
        (P_ind + P_prof) / 1000.0
    end

    return rpms, kws
end

# ── vx-aware variant (use this in the saving callback) ────────────────
"""
    allocate_wrench_vx(T_total, M_roll, M_pitch, M_yaw,
                       fleet, tilt_rad, vx, alt, ap)
        → (rpms::NTuple{6,Float64}, kws::NTuple{6,Float64})

Identical to `allocate_wrench` but takes physical `vx` (m/s) and
`alt` (m AGL) so the edgewise induced-velocity correction in kW matches
`rotor_kw_each` exactly. Always prefer this variant in the callback.
"""
function allocate_wrench_vx(T_total::Float64,
                              M_roll::Float64, M_pitch::Float64, M_yaw::Float64,
                              fleet::RotorFleet,
                              tilt_rad::Float64,
                              vx::Float64, alt::Float64,
                              ap::AllocatorParams=ALLOC) :: Tuple{NTuple{6,Float64}, NTuple{6,Float64}}

    ρ     = rho(alt)          # from atmosphere.jl — same as rotor_kw_each
    rho_r = ρ / 1.225
    v     = max(vx, 0.0)

    rpms, _ = allocate_wrench(T_total, M_roll, M_pitch, M_yaw,
                               fleet, tilt_rad, rho_r, ap)

    # Override windmilling rotors with vx-dependent autorotation RPM
    rpms = ntuple(6) do i
        lift_rotor_active(i, tilt_rad, ap) ? rpms[i] :
            autorotate_omega(fleet.units[i], v, ap)
    end

    kws = ntuple(6) do i
        u   = fleet.units[i]
        # Windmilling: negative kW (regenerative)
        if !lift_rotor_active(i, tilt_rad, ap)
            return autorotate_kw(u, v, ρ, ap)
        end
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

# ── Wrench demand builder ──────────────────────────────────────────────
"""
    build_wrench(u, thrust_cmd, pitch_cmd, roll_cmd [, yaw_rate_cmd, tilt_rad], ap)
        → NTuple{4,Float64}  (T, Mx, My, Mz)

Reconstructs the 4-DOF wrench from controller outputs and ODE state.
State indices follow NOTES.md:
  u[5]=pitch  u[7]=roll  u[10]=dyaw (ωz in 6-DOF build)

`tilt_rad` — current nacelle tilt (rad). Moment demands are scaled by
cos(tilt_rad) so they blend to zero as rotors lose body-frame moment
authority in cruise. Defaults to 0 (hover, full authority).

Called from two sites:
  1. build_ode (fly.jl) — to derive M_x/My/Mz for the Euler equations.
     Uses Dual numbers from AutoFiniteDiff; must remain ForwardDiff-safe
     (no Float64 coercions on arguments derived from u).
  2. Saving callback / postprocess — pure Float64, no constraint.
"""
function build_wrench(u::AbstractVector,
                      thrust_cmd::Float64,
                      pitch_cmd::Float64,
                      roll_cmd::Float64,
                      yaw_rate_cmd::Float64 = 0.0,
                      tilt_rad::Float64     = 0.0,
                      ap::AllocatorParams   = ALLOC) :: NTuple{4,Float64}

    # cos(tilt) blends moment authority to zero as rotors tilt to cruise.
    # At tilt=90° the rotors point forward and have no body-frame moment
    # authority — passing full moment demands produces a rank-deficient
    # pseudoinverse problem that causes oscillating rotor saturation.
    ct = cos(tilt_rad)

    M_pitch = ap.pitch_moment_scale * (pitch_cmd    - Float64(u[5])) * ct
    M_roll  = ap.roll_moment_scale  * (roll_cmd     - Float64(u[7])) * ct
    M_yaw   = ap.yaw_moment_scale   * (yaw_rate_cmd - Float64(u[10])) * ct

    return (thrust_cmd, M_roll, M_pitch, M_yaw)
end

# ── Self-test ──────────────────────────────────────────────────────────
"""
    alloc_selftest()

Five sanity checks. Run from the REPL after loading rotor_system.jl:

    julia> include("subsystems/rotor_system.jl")
    julia> include("control_allocator.jl")
    julia> alloc_selftest()
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
    # Rotors thrust upward. More thrust at the nose lifts it; less at
    # the tail lets it drop. B[3,i] = +arm_x so fwd rotors (+x) get a
    # positive pitch coefficient and spin up for +My demand.
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
    # CW spinning faster → more CW reaction on frame → nose turns right (+Mz).
    # Our B-matrix row: B[4,i] = −spin_dir · torque_arm → for CW (spin=−1): +torque_arm
    # So CW rotors have positive coefficient in the Mz row → they spin faster for +Mz.
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

    # ── Test 6: lift-rotor autorotation in cruise ─────────────────────
    # At tilt > lift_shutoff_tilt, R3 and R4 must windmill at
    #   ω = autorotate_k · √(vx/R)
    # which at vx=80 m/s, R=1 m gives ≈30 rad/s — well below omega_max.
    # Their kW must be negative (regen).
    tilt_cruise = deg2rad(85.0)
    vx_cruise   = 80.0   # m/s ≈ 288 km/h
    alt_cruise  = 300.0  # m AGL
    rpms6, kws6 = allocate_wrench_vx(T_hov * 0.3, 0.0, 0.0, 0.0,
                      FLEET, tilt_cruise, vx_cruise, alt_cruise, ALLOC)
    ρ6          = rho(alt_cruise)
    lift_ids    = ALLOC.lift_only_rotors

    # Expected autorotation ω for one of the lift rotors
    ω_auto_exp  = autorotate_omega(FLEET.units[lift_ids[1]], vx_cruise, ALLOC)

    lift_rpm_ok = all(abs(rpms6[i] - ω_auto_exp) < 1.0 for i in lift_ids)
    lift_kw_neg = all(kws6[i] < 0.0                    for i in lift_ids)
    tilt_spin   = all(rpms6[i] > ALLOC.omega_min
                      for i in 1:6 if i ∉ lift_ids)
    ok6 = lift_rpm_ok && lift_kw_neg && tilt_spin
    push!(pass, ok6)
    println("Test 6 — Lift-rotor autorotation at cruise (tilt=$(round(rad2deg(tilt_cruise),digits=0))°, vx=$(vx_cruise) m/s):")
    println("  ω (rad/s)        : ", round.(rpms6, digits=1))
    println("  kW               : ", round.(kws6,  digits=2))
    println("  Lift RPM ≈ autorotate ($(round(ω_auto_exp,digits=1)) rad/s): $(lift_rpm_ok ? "✓" : "✗")")
    println("  Lift kW negative (regen): $(lift_kw_neg ? "✓" : "✗")")
    println("  Tilting rotors spinning:  $(tilt_spin  ? "✓" : "✗")")
    println("  $(ok6 ? "✓ PASS" : "✗ FAIL")")
    println()

    # ── Test 7: power-weighted allocation prefers high-power rotors ───────
    # R1 & R2 rated 4× higher than R3–R6. Under a symmetric hover demand,
    # R1/R2 should carry substantially more thrust than the stock rotors.
    # Note: ALLOC itself now uses (746,746,280,280,280,280) — ratio ≈2.66×.
    # This test uses an explicit 4× ratio to verify the mechanism cleanly.
    ap_biased = AllocatorParams(power_weights = (4.0, 4.0, 1.0, 1.0, 1.0, 1.0))
    rpms7, _ = allocate_wrench(T_hov, 0.0, 0.0, 0.0, FLEET, 0.0, rho_sl, ap_biased)
    T7 = ntuple(i -> FLEET.units[i].kT * 1.225 * rpms7[i]^2 * FLEET.units[i].radius_m^4, 6)
    hi_mean = (T7[1] + T7[2]) / 2.0
    lo_mean = (T7[3] + T7[4] + T7[5] + T7[6]) / 4.0
    ok7 = hi_mean > lo_mean * 1.5   # expect ~4× loading ratio
    push!(pass, ok7)
    println("Test 7 — Power-weighted hover (R1/R2 rated 4× R3–R6):")
    println("  ω (rad/s)          : ", round.(rpms7, digits=1))
    println("  Thrust R1/R2 mean  : $(round(hi_mean, digits=0)) N")
    println("  Thrust R3–R6 mean  : $(round(lo_mean, digits=0)) N")
    println("  Hi/Lo ratio        : $(round(hi_mean / max(lo_mean,1e-6), digits=2))×  (expect ≈ 4×)")
    println("  $(ok7 ? "✓ PASS" : "✗ FAIL")")
    println()

    println("=== $(count(pass))/$(length(pass)) tests passed ===\n")
end

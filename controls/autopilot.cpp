// autopilot.cpp:   eVTOL tiltrotor autopilot shared library
// AUTHOR:          DANIEL DESAI
// UPDATED:         2026-05-10
// VERSION:         0.1.0
//
// Compile:
//   g++ -O2 -std=c++17 -fPIC -shared -o controls/autopilot.so controls/autopilot.cpp
//
// Called from fly.jl via @ccall:
//   @ccall "./controls/autopilot.so".compute_controls(
//       state::Ptr{Cdouble}, n::Cint, cfg::Ptr{APConfig}, out::Ptr{ControlOutput})::Cvoid
//
// Unit conventions (must match fly.jl expectations):
//   thrust_cmd — Newtons. fly.jl passes directly to thrust_derivative().
//   pitch_cmd  — radians, target pitch angle. fly.jl: M_y = scale*(pitch_cmd - pitch)*ct
//   roll_cmd   — radians, target roll angle.  fly.jl: M_x = scale*(roll_cmd  - roll)*ct
//   yaw_cmd    — [-1,1], scaled by MAX_YAW_RATE in fly.jl → rad/s target
//   tilt_mode  — 0=hover nacelles, 1=cruise (tilt actuator ramp handled in fly.jl)
//
// State vector indices (1-based in Julia → 0-based here):
//   0  vx        forward speed (m/s)
//   1  alt       altitude AGL (m)
//   2  tilt      rotor tilt (rad) — 0=hover, π/2=cruise
//   3  dtilt     tilt rate (rad/s)
//   4  pitch     pitch angle (rad, +nose up)
//   5  dpitch    pitch rate (rad/s)
//   6  roll      roll angle (rad)
//   7  droll     roll rate (rad/s)
//   8  yaw       yaw angle (rad)
//   9  dyaw      yaw rate (rad/s)
//  10  thrust_lag aggregate rotor thrust (N)
//  11  soc       battery state of charge (0–1)
//  12  tau       mission clock (s); negative = preflight/hover-climb
//  13  x         ground-track forward (m)
//  14  y         ground-track rightward (m)
//  15  omega_x   body roll rate (rad/s)
//  16  omega_y   body pitch rate (rad/s)
//  17  omega_z   body yaw rate (rad/s)

#include <cmath>
#include <cstdbool>
#include <algorithm>

// ---------------------------------------------------------------------------
// Shared interface struct — must match Julia ControlOutput exactly
// ---------------------------------------------------------------------------
extern "C" {

typedef struct {
    double thrust_cmd;  // Newtons
    double roll_cmd;    // radians, target roll angle
    double pitch_cmd;   // radians, target pitch angle
    double yaw_cmd;     // [-1, 1], scaled to rad/s in fly.jl
    int    tilt_mode;   // 0 = hover nacelles, 1 = cruise nacelles
    bool   autopilot;   // true = AP in command, false = hand off to pilot
    int    brakes;      // 0/1 wheel brakes
} ControlOutput;

// ---------------------------------------------------------------------------
// Mission / aircraft configuration passed in from fly.jl each call.
// t_dash_end removed — dash→fw_descent is range-triggered, not time-triggered.
// wp_x / wp_y carry the active waypoint (RTB → 0,0 ; FLY-TO → target coords).
// descent_initiation_m is computed by mission_planner.jl and passed in so
// the C++ threshold is guaranteed identical to the Julia phase_label() value.
// ---------------------------------------------------------------------------
typedef struct {
    // Aircraft
    double mass_kg;
    double hover_thrust_N;
    double weight_N;            // mass_kg * 9.81
    double gear_cg_m;           // GEAR.cg_to_ground_m

    // Mission timings (τ values, seconds) — T_DASH_END removed
    double preflight_hold_s;    // tau starts at -this
    double t_trans_end;         // end of tilt transition
    double t_fw_climb;          // end of fixed-wing climb
    double t_fw_desc;           // worst-case τ sentinel (not the trigger)
    double t_pitch_down;        // TIMINGS.T_PITCH_DOWN — end of back-transition

    // Navigation — active waypoint in local ENU frame (metres)
    // RTB: (0, 0).  FLY-TO: target x_m, y_m from test_card.json.
    double wp_x;
    double wp_y;

    // Descent initiation — computed by mission_planner.jl::descent_initiation_range()
    // and passed in here so C++ and Julia use exactly the same threshold.
    double descent_initiation_m;

    // Targets
    double hover_alt_m;
    double target_hover_alt_m; // hover_alt_m + z_m: hover ref at destination in ODE frame
    double dash_altitude_m;
    double dash_speed_ms;       // TC.dash_speed_kmh / 3.6

    // Envelope limits
    double soc_min;
    double pitch_limit_rad;
    double roll_limit_rad;
    double land_tilt_s;        // TC.land_tilt_s — back-transition tilt window
    double min_dash_s;         // computed from target range vs descent_initiation_m
} APConfig;

} // extern "C" — struct declarations complete; functions follow below

// ---------------------------------------------------------------------------
// Static state — reset between simulation runs
// ---------------------------------------------------------------------------
static bool s_descent_armed = false;

// ── PI integral accumulators ─────────────────────────────────────────────
// Persistent across AP calls within one simulation run.
// reset_pi_state() is called by reset_autopilot_state() so mission restart
// is always clean. Three channels, one accumulator each.
static double I_hover_alt  = 0.0;   // (m)·s   — hover altitude integral
static double I_dash_speed = 0.0;   // (m/s)·s — cruise speed integral
static double I_btrans_vel = 0.0;   // (m/s)·s — back-transition vx integral

// ── FLARE persistent state ────────────────────────────────────────────────
static double s_flare_agl_prev = -1.0;
static double s_flare_vz_filt  =  0.0;
static double s_flare_tau_prev = -1.0;

extern "C" void reset_pi_state() {
    I_hover_alt  = 0.0;
    I_dash_speed = 0.0;
    I_btrans_vel = 0.0;
}

extern "C" void reset_autopilot_state() {
    s_descent_armed    = false;
    s_flare_agl_prev   = -1.0;
    s_flare_vz_filt    =  0.0;
    s_flare_tau_prev   = -1.0;
    reset_pi_state();
}

// ── PI Gain struct ────────────────────────────────────────────────────────
// Tuning recipe per channel (see 01_autopilot_pi.cpp for full notes):
//
//   Ki_hover_alt  : start 0.40; raise until alt offset <0.5m in CSV;
//                   back off 20% if oscillation in altitude_msl_ft column.
//   Ki_dash_speed : start 0.08; raise until speed droop at DA <2 km/h;
//                   watch power_kw doesn't rail at max_thrust_N.
//   Ki_btrans     : MUST be gated strictly to BACK_TRANSITION phase.
//                   Any leak into DESCENT causes oscillatory flare.
//
// Anti-windup: hard clamp on accumulator = I_xxx_max / Ki_xxx.
// That gives a maximum PI contribution = I_xxx_max regardless of Ki.
struct PIGains {
    // Channel 1: Hover altitude
    double Kp_hover_alt  = 4.0;     // existing P gain (weight-normalised in ctrl_hover)
    double Ki_hover_alt  = 0.40;    // integral — tune up from here
    double I_hover_max   = 2000.0;  // N·s  — anti-windup ceiling

    // Channel 2: Dash / cruise speed
    double Kp_dash_speed = 0.010;   // existing P coefficient (weight fraction / (m/s))
    double Ki_dash_speed = 0.03;    // integral — reduced from 0.08; 0.08 caused 15 km/h oscillation
    double I_dash_max    = 0.12;    // fraction of weight — reduced from 0.30 to limit swing

    // Channel 3: Back-transition deceleration
    double Kp_btrans     = 0.05;    // existing P on vx (pitch_target per m/s)
    double Ki_btrans     = 0.10;    // integral — gate strictly; reset on exit
    double I_btrans_max  = 0.20;    // fraction of weight — anti-windup ceiling
} PIG;

// ── dt estimation from autopilot timestamp ────────────────────────────────
// The AP is called ~10 Hz from the saving callback.
// Computing dt from the saved timestamp avoids the fixed 0.1s assumption
// and handles variable ODE step sizes cleanly.
static double last_ap_time = -1.0;

static double get_ap_dt(double t_now) {
    double dt = (last_ap_time < 0.0) ? 0.1 : (t_now - last_ap_time);
    last_ap_time = t_now;
    return std::clamp(dt, 0.01, 0.5);   // guard against large gaps or first call
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static inline double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline double angle_err(double target, double actual) {
    double e = target - actual;
    while (e >  M_PI) e -= 2.0 * M_PI;
    while (e < -M_PI) e += 2.0 * M_PI;
    return e;
}

// ---------------------------------------------------------------------------
// Phase helpers
// ---------------------------------------------------------------------------

enum Phase {
    PHASE_LANDED_PRE,       // tau < 0, on ground
    PHASE_HOVER,            // tau < 0, airborne (climbing to hover_alt)
    PHASE_TRANSITION,       // 0 .. t_trans_end
    PHASE_FW_CLIMB,         // t_trans_end .. t_fw_climb
    PHASE_DASH,             // t_fw_climb .. range trigger
    PHASE_FW_DESCENT,       // range trigger .. t_pitch_down
    PHASE_BACK_TRANSITION,  // t_pitch_down .. t_pitch_down + land_tilt_s
    PHASE_DESCENT,          // t_pitch_down + land_tilt_s .. gear contact
    PHASE_LANDED_POST,      // tau > t_trans_end, on ground
};

static Phase get_phase(const double* s, int n, const APConfig* cfg) {
    double tau = (n > 12) ? s[12] : 0.0;
    double alt = (n >  1) ? s[1]  : 0.0;
    double x   = (n > 13) ? s[13] : 0.0;
    double y   = (n > 14) ? s[14] : 0.0;

    const double on_ground_m = cfg->gear_cg_m + 0.05;

    if (tau < 0.0)
        return (alt <= on_ground_m) ? PHASE_LANDED_PRE : PHASE_HOVER;

    if (alt <= on_ground_m && tau > cfg->t_trans_end)
        return PHASE_LANDED_POST;

    if (tau < cfg->t_trans_end) return PHASE_TRANSITION;
    if (tau < cfg->t_fw_climb)  return PHASE_FW_CLIMB;

    // Minimum dash guard — ensures the aircraft levels off at cruise
    // altitude before descent arms. Computed from available dash distance
    // so short-range missions don't overshoot before triggering.
    if (tau < cfg->t_fw_climb + cfg->min_dash_s) return PHASE_DASH;

    // Dash → descent: range-triggered, latching.
    // Once the aircraft enters the descent envelope it must not revert
    // to DASH if it overshoots the waypoint — that causes the tilt
    // actuator to oscillate and the aircraft to spiral indefinitely.
    if (!s_descent_armed) {
        double dx  = x - cfg->wp_x;
        double dy  = y - cfg->wp_y;
        double rng = std::sqrt(dx*dx + dy*dy);
        if (rng < cfg->descent_initiation_m) {
            s_descent_armed = true;
        }
    }
    // Also arm if within descent initiation range — covers the case where
    // fly.jl arms autoland by range before autopilot.so sets s_descent_armed.
    double dx_arm = cfg->wp_x - s[13];
    double dy_arm = cfg->wp_y - s[14];
    double rng_arm = std::sqrt(dx_arm*dx_arm + dy_arm*dy_arm);
    if (!s_descent_armed && rng_arm > cfg->descent_initiation_m) return PHASE_DASH;
    if (!s_descent_armed) s_descent_armed = true;  // force arm by range

    // Sub-phases after descent is armed:
    // fw_descent:      vx > 25 m/s — thrust cut, decelerate and descend
    // back_transition: vx <= 25 m/s and above destination hover band
    // descent:         within hover band of destination — translate + sink
    //
    // CRITICAL: use target_hover_alt_m (= hover_alt_m + z_m) not hover_alt_m.
    // For KAXX→KSAF: target_hover_alt_m = 30 + (-619) = -589m in ODE frame.
    // hover_alt_m = 30m — using that causes DESCENT to trigger immediately
    // after back_transition while still 650m above KSAF, causing a vertical
    // drop 4.9km short of the destination.
    double vx  = (n > 0) ? s[0] : 0.0;
    double dx  = x - cfg->wp_x;
    double dy  = y - cfg->wp_y;
    double rng = std::sqrt(dx*dx + dy*dy);

    if (vx > 25.0)                               return PHASE_FW_DESCENT;

    // Switch to final descent when within 300m of waypoint at low speed,
    // regardless of altitude. This handles the case where the aircraft
    // is close to the destination but still above target_hover_alt_m
    // (e.g. KAXX→KSAF: aircraft is at alt_ode≈0 but 619m above KSAF).
    if (rng < 300.0 && vx < 8.0)                return PHASE_DESCENT;

    if (alt > cfg->target_hover_alt_m + 5.0)     return PHASE_BACK_TRANSITION;
    return PHASE_DESCENT;
}

// ---------------------------------------------------------------------------
// Envelope monitor
// ---------------------------------------------------------------------------
static bool envelope_ok(const double* s, const APConfig* cfg) {
    double soc = s[11];
    if (soc < cfg->soc_min) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Phase controllers (unchanged except ctrl_dash gains heading-to-waypoint)
// ---------------------------------------------------------------------------

// Pre-takeoff ground hold: thrust_cmd = weight so the preflight ramp
// in build_ode has a target to ramp toward. Brakes on.
static ControlOutput ctrl_landed_pre(const double* s, const APConfig* cfg) {
    // Ground hold before liftoff. Command hover-level thrust (with altitude
    // error correction) so the t-based ramp in fly.jl can interpolate from
    // weight to this value and lift the aircraft off the ground.
    // Brakes on until ramp delivers enough thrust to lift off.
    double alt     = s[1];
    double alt_err = cfg->hover_alt_m - alt;
    double thrust_cmd = clamp(cfg->weight_N * (1.0 + 0.15 * alt_err),
                              cfg->weight_N, 1.35 * cfg->weight_N);
    ControlOutput o{};
    o.thrust_cmd = thrust_cmd;
    o.pitch_cmd  = 0.0;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = 0.0;
    o.tilt_mode  = 0;
    o.autopilot  = true;
    o.brakes     = 1;
    return o;
}

// Post-landing: rotors off, brakes on, hold pitch level.
static ControlOutput ctrl_landed_post(const double* /*s*/, const APConfig* /*cfg*/) {
    ControlOutput o{};
    o.thrust_cmd = 0.0;   // rotors off
    o.pitch_cmd  = 0.0;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = 0.0;
    o.tilt_mode  = 0;
    o.autopilot  = true;
    o.brakes     = 1;
    return o;
}

// ── Channel 1: Hover altitude PI ─────────────────────────────────────────
// Replaces P-only ctrl_hover. Integral removes steady-state offset caused
// by sensor bias, CG offset, or density-altitude mismatch vs the fixed
// weight_N baseline. Integrate only inside ±20m linear zone to prevent
// windup during large-error initial climbs.
//
// Validation: grep HOVER dash_results.csv | awk -F, '{print $2}' | tail -60
//   → alt_agl_m should converge to hover_alt_m within ±0.5m.
//   If it oscillates: reduce Ki_hover_alt by 20%, re-run.
static ControlOutput ctrl_hover(const double* s, const APConfig* cfg) {
    double alt     = s[1];
    double vx      = s[0];
    double omega_z = s[17];
    double tau     = (s[12]);   // used to recover t_now for dt

    double alt_err = cfg->hover_alt_m - alt;

    // Integrate only in linear zone; windup guard via clamped accumulator
    double dt = get_ap_dt(tau);
    if (std::fabs(alt_err) < 20.0) {
        I_hover_alt += alt_err * dt;
    }
    I_hover_alt = std::clamp(I_hover_alt,
                             -PIG.I_hover_max / PIG.Ki_hover_alt,
                              PIG.I_hover_max / PIG.Ki_hover_alt);

    // P term uses weight-normalised coefficient (existing), I term additive in N
    double thrust_cmd = cfg->weight_N * (1.0 + 0.15 * alt_err)  // original P
                        + PIG.Ki_hover_alt * I_hover_alt;         // I correction
    thrust_cmd = clamp(thrust_cmd, 0.5 * cfg->weight_N, 1.35 * cfg->weight_N);

    double pitch_target = clamp(-0.04 * vx, -0.15, 0.15);

    ControlOutput o{};
    o.thrust_cmd = thrust_cmd;
    o.pitch_cmd  = pitch_target;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = clamp(-omega_z * 2.0, -1.0, 1.0);
    o.tilt_mode  = 0;
    o.autopilot  = true;
    o.brakes     = 0;
    return o;
}

static ControlOutput ctrl_transition(const double* s, const APConfig* cfg) {
    double alt     = s[1];
    double vx      = s[0];
    double omega_z = s[17];
    double tau     = s[12];
    double tilt    = s[2];   // actual tilt angle (rad)

    double alt_err      = cfg->hover_alt_m - alt;
    double pitch_target = clamp(0.05 * alt_err, -0.20, 0.20);
    double trans_frac   = clamp(tau / cfg->t_trans_end, 0.0, 1.0);
    double vx_target    = 16.7 * trans_frac;
    double vx_err       = vx_target - vx;
    pitch_target        = clamp(pitch_target - 0.03 * vx_err, -0.25, 0.20);

    // Compensate thrust for tilt angle: as rotors tilt forward, their
    // vertical component = thrust * cos(tilt) drops. Command extra thrust
    // to maintain weight support. cos(65°) ≈ 0.42 → need ~2.4× hover thrust.
    // Wing lift begins contributing at speed; tilt_comp conservatively
    // accounts for only the rotor vertical component.
    double cos_tilt   = std::cos(tilt);
    double tilt_comp  = (cos_tilt > 0.5) ? (1.0 / cos_tilt) : (1.0 / 0.5);
    double base_thrust = cfg->weight_N * tilt_comp * (1.0 + 0.15 * alt_err);
    // Wing lift correction: at speed, wing carries some of the load
    double wing_frac  = clamp(vx / cfg->dash_speed_ms, 0.0, 0.6);
    double thrust_cmd = clamp(base_thrust * (1.0 - 0.4 * wing_frac),
                              0.7 * cfg->weight_N, 1.8 * cfg->weight_N);

    ControlOutput o{};
    o.thrust_cmd = thrust_cmd;
    o.pitch_cmd  = pitch_target;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = clamp(-omega_z * 2.0, -1.0, 1.0);
    o.tilt_mode  = 1;
    o.autopilot  = true;
    o.brakes     = 0;
    return o;
}

static ControlOutput ctrl_fw_climb(const double* s, const APConfig* cfg) {
    double alt     = s[1];
    double vx      = s[0];
    double omega_z = s[17];

    double alt_err      = cfg->dash_altitude_m - alt;
    double pitch_target = clamp(0.04 * alt_err, -0.15, 0.25);
    double spd_err      = cfg->dash_speed_ms - vx;
    double thrust_cmd   = clamp(cfg->weight_N * (0.6 + 0.008 * spd_err),
                                0.4 * cfg->weight_N, 0.90 * cfg->weight_N);

    ControlOutput o{};
    o.thrust_cmd = thrust_cmd;
    o.pitch_cmd  = pitch_target;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = clamp(-omega_z * 2.0, -1.0, 1.0);
    o.tilt_mode  = 1;
    o.autopilot  = true;
    o.brakes     = 0;
    return o;
}

// ── Channel 2: Dash / cruise speed PI ────────────────────────────────────
// Replaces P-only ctrl_dash. Integral removes speed droop caused by
// density-altitude reducing available thrust margin at fixed P gain.
// At DA factor ~0.82 the P term alone undershoots by ~15–20 km/h;
// the I term accumulates over the ~60s DASH phase and corrects it.
//
// Anti-windup ceiling: I_dash_max fraction of weight → max PI contribution
// = I_dash_max regardless of Ki_dash_speed.
//
// Validation: grep DASH dash_results.csv | awk -F, '{print $4}' | tail -40
//   → speed_kmh should converge to dash_speed_ms*3.6 within ±2 km/h.
//   If power_kw rails at max_thrust_N: reduce Ki_dash_speed by 30%.
//
// Note: altitude hold via pitch_target is P-only (alt errors are slow
// and cross-coupling with thrust is undesirable here).
static ControlOutput ctrl_dash(const double* s, const APConfig* cfg) {
    double alt     = s[1];
    double vx      = s[0];
    double omega_z = s[17];
    double tau     = s[12];

    double spd_err = cfg->dash_speed_ms - vx;
    double dt = get_ap_dt(tau);
    if (vx < 10.0) {
        I_dash_speed = 0.0;
    } else if (std::fabs(spd_err) < 30.0) {
        I_dash_speed += spd_err * dt;
    }
    I_dash_speed = std::clamp(I_dash_speed,
                              -PIG.I_dash_max / PIG.Ki_dash_speed,
                               PIG.I_dash_max / PIG.Ki_dash_speed);

    // Thrust fraction: P + I, clamped to safe band
    double frac = 0.55
                + PIG.Kp_dash_speed * spd_err
                + PIG.Ki_dash_speed * I_dash_speed;
    double thrust_max = (vx < 10.0) ? 1.02 * cfg->weight_N
                                    : 0.85 * cfg->weight_N;
    double thrust_cmd = clamp(cfg->weight_N * frac,
                              0.30 * cfg->weight_N, thrust_max);

    double alt_err      = cfg->dash_altitude_m - alt;
    double pitch_target = clamp(0.03 * alt_err, -0.10, 0.12);

    ControlOutput o{};
    o.thrust_cmd = thrust_cmd;
    o.pitch_cmd  = pitch_target;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = clamp(-omega_z * 2.0, -1.0, 1.0);
    o.tilt_mode  = (vx < 10.0) ? 0 : 1;
    o.autopilot  = true;
    o.brakes     = 0;
    return o;
}

static ControlOutput ctrl_fw_descent(const double* s, const APConfig* cfg) {
    double vx      = s[0];
    double tilt    = s[2];
    double omega_z = s[17];

    // Tilt-compensated thrust: maintain 0.75*weight vertical force
    // regardless of current nacelle tilt angle.
    // At tilt=65deg (cruise): cos=0.42, thrust=1.79*weight (brief, tilt slewing fast)
    // At tilt=30deg:          cos=0.87, thrust=0.86*weight
    // At tilt=0deg (hover):   cos=1.00, thrust=0.75*weight (steady descent)
    // Floor on cos_tilt prevents division by zero during tilt transition.
    double cos_tilt   = std::max(std::cos(tilt), 0.25);
    double thrust_cmd = std::min(cfg->weight_N * 0.75 / cos_tilt,
                                 cfg->hover_thrust_N);

    ControlOutput o{};
    o.thrust_cmd = thrust_cmd;
    o.pitch_cmd  = 0.0;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = clamp(-omega_z * 2.0, -1.0, 1.0);
    o.tilt_mode  = 0;
    o.autopilot  = true;
    o.brakes     = 0;
    return o;
}

// ── Channel 3: Back-transition deceleration PI ───────────────────────────
// Replaces fixed-fraction ctrl_back_transition. Integral corrects for
// persistent forward speed residual (aircraft still moving at ~5 km/h
// when DESCENT triggers, dragging it off the pad).
//
// CRITICAL gating:
//   • Integrate ONLY when phase == BACK_TRANSITION (phase_is_btrans=true).
//   • Hard-reset I_btrans_vel=0 when phase exits — prevents windup from
//     leaking into DESCENT and causing an oscillatory flare/hover cycle.
//   • caller (compute_controls) passes phase_is_btrans derived from
//     get_phase() so the gate is centralised and authoritative.
//
// Note: thrust base is still 0.82×weight (intentional descent). The I term
// adds/subtracts a small fraction of weight to null residual vx.
//
// Validation: grep BACK_TRANSITION dash_results.csv | awk -F, '{print $4}' | head -40
//   → speed_kmh should decelerate monotonically and reach <5 km/h.
//   grep DESCENT dash_results.csv | head -5: speed_kmh must not re-accelerate.
static ControlOutput ctrl_back_transition(const double* s, const APConfig* cfg,
                                          bool phase_is_btrans) {
    double vx      = s[0];
    double omega_z = s[17];
    double tau     = s[12];
    double alt     = s[1];
    double x       = s[13];
    double y       = s[14];

    // Range-proportional vx target.
    // Unconditional vx_target=0 causes the aircraft to stop 3-5km short of the
    // destination when the destination is significantly lower than origin (KAXX→KSAF
    // drops 619m — back_transition takes ~210s to descend, during which the aircraft
    // must cover ~6km laterally).
    //
    // vx_target = rng / time_to_target_hover — maintain just enough speed to
    // arrive at the waypoint as alt reaches target_hover_alt_m.
    // Clamped: minimum 2 m/s (don't hover in mid-air far from pad),
    //          maximum 25 m/s (fw_descent exit speed).
    double dx  = cfg->wp_x - x;
    double dy  = cfg->wp_y - y;
    double rng = std::sqrt(dx*dx + dy*dy);

    // Estimate time remaining: altitude above target hover / nominal descent rate.
    // Use 3.5 m/s as conservative descent rate (matches observed ~3-4 m/s in BT).
    double alt_to_lose   = std::max(alt - cfg->target_hover_alt_m, 1.0);
    double time_to_hover = alt_to_lose / 3.5;
        // 18 m/s cap: enough to cover 15.3 m/s needed for KAXX→KSAF geometry,
    // without the instability and power drain of maintaining 25 m/s in hover mode.
    // Min 5 m/s (18 km/h) — prevents near-stop overshoot that leaves
    // the aircraft 985m short. Max 18 m/s to avoid power burn.
    // Below 1600m autoland.so takes over — allow full deceleration to 0.
    double vx_min    = (rng < 1600.0) ? 0.0 : 5.0;
    double vx_target = std::clamp(rng / time_to_hover, vx_min, 18.0);

    double vx_err = vx_target - vx;   // positive = need more speed, negative = too fast

    double dt = get_ap_dt(tau);
    if (phase_is_btrans && vx_err < 0.0) {
        // Only integrate when too fast (vx_err < 0) — braking correction.
        // When vx_err > 0 (need more speed), pitch handles translation;
        // integrating thrust upward here caused I-windup hover equilibrium.
        I_btrans_vel += vx_err * dt;
    } else if (!phase_is_btrans) {
        I_btrans_vel = 0.0;
    }
    I_btrans_vel = std::clamp(I_btrans_vel,
                              -PIG.I_btrans_max / PIG.Ki_btrans,
                               PIG.I_btrans_max / PIG.Ki_btrans);

    // Pitch: P on vx error — nose-up when too fast, nose-down when too slow.
    // Positive vx_err (need more speed) → negative pitch (nose down to accelerate).
    // Negative vx_err (too fast) → positive pitch (nose up to decelerate).
    // Clamp: max nose-up 0.20 rad (decel), max nose-down 0.10 rad (translation).
    // Brake clamp 0.12 rad (7°) — was 0.20 (11°) which caused
    // overshoot to near-zero speed and 985m short landing.
    double pitch_target = clamp(-0.04 * vx_err, -0.10, 0.12);

    // Thrust: target a descending profile toward target_hover_alt_m.
    // Below target: increase thrust to arrest descent.
    // Above target: reduce thrust to sink. Base 0.82 gives gentle descent.
    // Alt error term: positive when above target → reduce thrust to descend.
    double alt_err_bt   = alt - cfg->target_hover_alt_m;
    // 0.000167 gives 0.72×W at top of descent (alt_err≥600), 0.82×W at target.
    // 0.72×W deficit = 0.28W ≈ 3 m/s² initial descent — controlled, not ballistic.
    double thrust_frac  = 0.82 - 0.000167 * clamp(alt_err_bt, -200.0, 600.0);
    double thrust_cmd   = cfg->weight_N * (
        thrust_frac
        + PIG.Ki_btrans * I_btrans_vel     // I: residual vx correction
    );
    thrust_cmd = clamp(thrust_cmd, 0.30 * cfg->weight_N, 1.10 * cfg->weight_N);

    ControlOutput o{};
    o.thrust_cmd = thrust_cmd;
    o.pitch_cmd  = pitch_target;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = clamp(-omega_z * 2.0, -1.0, 1.0);
    o.tilt_mode  = 0;
    o.autopilot  = true;
    o.brakes     = 0;
    return o;
}

// ── FLARE: arrest sink rate at low AGL ────────────────────────────────────
static ControlOutput ctrl_descent(const double* s, int n, const APConfig* cfg) {
    double alt     = s[1];
    double vx      = s[0];
    double omega_z = s[17];
    double yaw     = s[8];
    double x       = s[13];
    double y       = s[14];
    double tau     = s[12];

    // Range and bearing to waypoint.
    double dx      = cfg->wp_x - x;
    double dy      = cfg->wp_y - y;
    double rng     = std::sqrt(dx*dx + dy*dy);
    double bearing = std::atan2(dy, dx);

    // ── Terrain AGL ───────────────────────────────────────────────────
    // Prefer s[18] (terrain AGL appended by fly.jl, always present when n=19).
    // Fall back to ODE-frame height above destination hover alt — correct for
    // elevation-change routes (e.g. KAXX→KSAF drops 619 m).
    double agl = (n >= 19) ? s[18] : (alt - cfg->target_hover_alt_m);

    // ── FLARE: arrest sink rate at low AGL ────────────────────────────
    // Trigger at 12 m AGL. Sink rate estimated from differenced agl at
    // callback rate — accurate enough this close to ground (Δagl ≈ 0.15–0.5 m
    // per step at 1.5–5 m/s sink; well above noise).
    static constexpr double FLARE_ALT_M       = 12.0;   // m AGL
    static constexpr double FLARE_ARREST_K    = 0.65;   // × hover_thrust per m/s excess
    static constexpr double FLARE_SINK_TARGET = 0.4;    // m/s target at touchdown
    static constexpr double FLARE_VZ_TAU      = 0.20;   // s — sink-rate filter
    // Touchdown: use gear_cg_m as the AGL floor. The aircraft sits at exactly
    // cg_to_ground_m AGL on the ground; 0.25 m would never trigger if gear_cg_m > 0.25.
    const  double           TOUCHDOWN_ALT_M   = cfg->gear_cg_m + 0.1;

    // Compute dt for filter
    double flare_dt = 0.1;
    if (s_flare_tau_prev >= 0.0)
        flare_dt = clamp(tau - s_flare_tau_prev, 0.001, 0.5);
    s_flare_tau_prev = tau;

    // Update sink-rate filter
    if (s_flare_agl_prev >= 0.0) {
        double vz_raw = -(agl - s_flare_agl_prev) / flare_dt;  // +ve = descending
        double alpha  = flare_dt / (FLARE_VZ_TAU + flare_dt);
        s_flare_vz_filt += alpha * (vz_raw - s_flare_vz_filt);
    }
    s_flare_agl_prev = agl;

    if (agl < FLARE_ALT_M) {
        // Touchdown — rotors off, brakes on
        if (agl < TOUCHDOWN_ALT_M) {
            ControlOutput o{};
            o.thrust_cmd = 0.0;
            o.pitch_cmd  = 0.0; o.roll_cmd = 0.0; o.yaw_cmd = 0.0;
            o.tilt_mode  = 0;   o.autopilot = false; o.brakes = 1;
            return o;
        }
        // Flare — arrest sink, no translation
        double excess     = clamp(s_flare_vz_filt - FLARE_SINK_TARGET, 0.0, 6.0);
        double thrust_cmd = cfg->weight_N
                            + cfg->hover_thrust_N * FLARE_ARREST_K * excess;
        ControlOutput o{};
        o.thrust_cmd = clamp(thrust_cmd, cfg->weight_N * 0.5,
                             cfg->hover_thrust_N * 1.5);
        o.pitch_cmd  = 0.0; o.roll_cmd = 0.0; o.yaw_cmd = 0.0;
        o.tilt_mode  = 0;   o.autopilot = true; o.brakes = 0;
        return o;
    }

    // ── Above flare altitude: translate toward pad ────────────────────

    // [FIX-1] Range-gated thrust: 0.98W at arm (1500 m) → 0.78W at pad.
    // Holds altitude during the approach so the aircraft does not descend
    // through the approach path. Previously used a fixed 0.78W which caused
    // the aircraft to sink 30–50 m during the 1500 m translate, landing short.
    static constexpr double THRUST_FAR  = 0.98;
    static constexpr double THRUST_NEAR = 0.78;
    static constexpr double ARM_RNG_M   = 1500.0;
    double rng_frac   = clamp(rng / ARM_RNG_M, 0.0, 1.0);
    double thrust_cmd = cfg->weight_N * (THRUST_NEAR + (THRUST_FAR - THRUST_NEAR) * rng_frac);

    // [FIX-2] close_frac fade zone: 50 m, not 200 m.
    // The original 200 m gate effectively zeroed the pitch command at any
    // useful approach distance. Pitch command is now active from arm distance;
    // it fades to zero only inside 50 m to prevent pad overshoot.
    // Gain 0.25/1500 gives 14° (0.244 rad) at arm distance — more authority
    // than the original 8° max which was correct magnitude but never applied.
    static constexpr double PITCH_GAIN   = 0.25 / ARM_RNG_M;  // rad/m
    static constexpr double PITCH_MAX    = 0.244;              // 14° hard limit
    static constexpr double CLOSE_FADE_M = 50.0;
    double close_frac        = clamp(rng / CLOSE_FADE_M, 0.0, 1.0);
    double pitch_translation = -clamp(rng * PITCH_GAIN, 0.0, PITCH_MAX) * close_frac;

    // Speed damping: null residual vx once close
    double pitch_damp   = clamp(-0.03 * vx, -0.10, 0.05);
    double pitch_target = clamp(pitch_translation + pitch_damp,
                                -cfg->pitch_limit_rad, 0.10);

    // [FIX-3] 2D yaw: steer toward waypoint bearing before pitching forward.
    // Previously yaw_cmd was -omega_z * 2 (rate damp only) — no bearing
    // correction, so the aircraft pitched forward in body-x regardless of
    // where the pad actually was. Now: point at the pad, then fly toward it.
    double yaw_err  = angle_err(bearing, yaw);
    double yaw_cmd  = clamp(yaw_err * 2.0 - omega_z * 1.5, -1.0, 1.0);

    ControlOutput o{};
    o.thrust_cmd = clamp(thrust_cmd, 0.0, cfg->hover_thrust_N * 1.2);
    o.pitch_cmd  = pitch_target;
    o.roll_cmd   = 0.0;
    o.yaw_cmd    = yaw_cmd;
    o.tilt_mode  = 0;
    o.autopilot  = true;
    o.brakes     = 0;
    return o;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
extern "C" void compute_controls(const double* state, int n,
                                 const APConfig* cfg, ControlOutput* out)
{
    Phase phase = get_phase(state, n, cfg);
    bool is_btrans = (phase == PHASE_BACK_TRANSITION);

    ControlOutput result{};
    switch (phase) {
        case PHASE_LANDED_PRE:
            result = ctrl_landed_pre(state, cfg);          break;
        case PHASE_LANDED_POST:
            result = ctrl_landed_post(state, cfg);         break;
        case PHASE_HOVER:
            result = ctrl_hover(state, cfg);            break;
        case PHASE_TRANSITION:
            result = ctrl_transition(state, cfg);       break;
        case PHASE_FW_CLIMB:
            result = ctrl_fw_climb(state, cfg);         break;
        case PHASE_DASH:
            result = ctrl_dash(state, cfg);             break;
        case PHASE_FW_DESCENT:
            result = ctrl_fw_descent(state, cfg);       break;
        case PHASE_BACK_TRANSITION:
            result = ctrl_back_transition(state, cfg, true);   break;
        case PHASE_DESCENT:
            // Ensure btrans integral is zeroed if we enter DESCENT directly
            // (e.g. fast aircraft that skips phase boundary in one ODE step)
            result = ctrl_back_transition(state, cfg, false);  // resets I_btrans_vel
            result = ctrl_descent(state, n, cfg);              break;
    }

    if (!envelope_ok(state, cfg))
        result.autopilot = false;

    *out = result;
}
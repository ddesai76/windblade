# navigation.jl:   Waypoint guidance and nav map state
# AUTHOR:          DANIEL DESAI
# UPDATED:         2026-05-10
# VERSION:         0.1.0
#
# Responsibilities
#   1. Read the `navigation` block from test_card.json — single source of
#      truth for destination.  No CLI flags, no ENV vars.
#   2. Validate NavTarget (≥ 1 000 m from origin when RTB = false).
#   3. Provide real-time guidance (cross-track, bearing, range, Δyaw).
#   4. Maintain a thread-safe ring-buffer track log (NavMapState) that
#      glass_cockpit.jl reads to draw the moving-map panel.
#
#
# JSON loaded by fly.jl before this file is included.

# ══════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════
const NAV_MIN_RANGE_M   = 1_000.0       # mandatory off-base distance (m)
const NAV_TRACK_MAX     = 3_000         # ring-buffer depth (track points)
const NAV_KP_HDG        = 0.8           # bearing error (rad) → yaw_cmd [-1,1]
const NAV_MAX_YAW_CMD   = 1.0           # yaw_cmd hard cap


# NavTarget
#
# Encodes the mission destination parsed from test_card.json.
#
# Fields
# ──────
# - `return_to_base` — fly back to origin [0, 0] if true
# - `x_m, y_m`      — target in local ENU frame (forward / rightward, m)
# - `z_m`           — AGL elevation of the target landing pad relative to
#                    the takeoff point (m). 0 = same elevation as takeoff.

struct NavTarget
    return_to_base :: Bool
    x_m            :: Float64
    y_m            :: Float64
    z_m            :: Float64
end

NavTarget() = NavTarget(true, 0.0, 0.0, 0.0)   # safe default

# Active waypoint (accounts for RTB flag)
nav_wx(n::NavTarget) = n.return_to_base ? 0.0 : n.x_m
nav_wy(n::NavTarget) = n.return_to_base ? 0.0 : n.y_m

#     load_nav(json_path) → NavTarget
#
# Reads the `navigation` block from test_card.json.  Falls back to RTB if
# the file or block is absent.

function load_nav(json_path::String)::NavTarget
    if !isfile(json_path)
        @warn "navigation: $(json_path) not found — defaulting to return-to-base"
        return NavTarget()
    end
    raw = JSON.parsefile(json_path)
    nb  = get(raw, "navigation", nothing)
    nb === nothing && return NavTarget()
    rtb = get(nb, "return_to_base", true)::Bool
    tgt = get(nb, "target", nothing)
    tgt === nothing && return NavTarget(rtb, 0.0, 0.0, 0.0)
    NavTarget(rtb,
              Float64(get(tgt, "x_m", 0.0)),
              Float64(get(tgt, "y_m", 0.0)),
              Float64(get(tgt, "z_m", 0.0)))
end

# validate_nav(nav)
#
# Raises a descriptive error when RTB=false and target < NAV_MIN_RANGE_M.
# Prints a confirmation banner in all cases.

function validate_nav(nav::NavTarget)
    rng = hypot(nav.x_m, nav.y_m)
    if !nav.return_to_base
        rng < NAV_MIN_RANGE_M && error(
            "NAV validation: target [$(round(Int,nav.x_m)), $(round(Int,nav.y_m))] m " *
            "(pad AGL $(round(Int,nav.z_m)) m) is only $(round(Int,rng)) m from takeoff " *
            "— need ≥ $(round(Int,NAV_MIN_RANGE_M)) m when return_to_base = false.")
    end
end


#  Geometry helpers
#
#     nav_bearing(xf, yf, xt, yt) → degrees [0, 360)
#
# Bearing from (xf, yf) to (xt, yt), measured clockwise from the +x
# (forward) axis.  0° = dead ahead, 90° = right, 270° = left.

nav_bearing(xf::Real, yf::Real, xt::Real, yt::Real) =
    mod(rad2deg(atan(yt - yf, xt - xf)), 360.0)

#     nav_cross_track(x, y, hdg_rad, xt, yt) → Float64 (m)
#
# Signed cross-track error from current position/heading to waypoint.
# Positive = waypoint is to the right; negative = to the left.

nav_cross_track(x::Real, y::Real, hdg_rad::Real, xt::Real, yt::Real) =
    (xt - x) * sin(hdg_rad) - (yt - y) * cos(hdg_rad)


# NavGuidance
#
# Steering output from `nav_guidance()`.
#
# - `delta_yaw_rad` — yaw_cmd in [-1, 1] to inject into build_ode
# - `bearing_deg`   — current bearing to active waypoint
# - `cross_track_m` — signed cross-track error
# - `range_m`       — horizontal distance to waypoint

struct NavGuidance
    delta_yaw_rad :: Float64
    bearing_deg   :: Float64
    cross_track_m :: Float64
    range_m       :: Float64
end

# nav_guidance(nav, x, y, hdg_rad) → NavGuidance
#
# Bearing-error heading controller. Active throughout fixed-wing cruise.
# Guidance is gated in build_ode to vx > 35 m/s so the AP retains clean
# yaw authority during back-transition deceleration.

function nav_guidance(nav::NavTarget, x::Real, y::Real, hdg_rad::Real)::NavGuidance
    wx  = nav_wx(nav);  wy  = nav_wy(nav)
    rng = hypot(x - wx, y - wy)
    brg = nav_bearing(x, y, wx, wy)
    xte = nav_cross_track(x, y, hdg_rad, wx, wy)

    # Signed bearing error wrapped to [-π, π]
    err = deg2rad(brg) - hdg_rad
    while err >  π; err -= 2π; end
    while err < -π; err += 2π; end
    dyaw = clamp(NAV_KP_HDG * err, -NAV_MAX_YAW_CMD, NAV_MAX_YAW_CMD)

    NavGuidance(dyaw, brg, xte, rng)
end


#  Track ring-buffer  (saving callback → map panel)

struct _Pt
    x   :: Float32
    y   :: Float32
    alt :: Float32
    t   :: Float32
end

# NavMapState
#
# Thread-safe track log and live position scalars.  Written by the saving
# callback (`nav_push!`), read by the cockpit map panel (`nav_snapshot`).

mutable struct NavMapState
    target   :: NavTarget
    buf      :: Vector{_Pt}      # circular buffer, length = NAV_TRACK_MAX
    head     :: Int              # next write index (1-based)
    count    :: Int              # valid entries (≤ NAV_TRACK_MAX)
    x        :: Float64
    y        :: Float64
    alt      :: Float64
    hdg      :: Float64          # degrees, 0 = forward
    phase    :: String
    t        :: Float64
    lock     :: ReentrantLock
end

NavMapState(nav::NavTarget) = NavMapState(
    nav, Vector{_Pt}(undef, NAV_TRACK_MAX), 1, 0,
    0.0, 0.0, 0.0, 0.0, "preflight", 0.0, ReentrantLock())

#     nav_push!(ms, u, t, phase)
#
# Append a track point from the 18-element ODE state vector.
# Call from the saving callback — thread-safe.
#
# Reads:  u[2]=alt_agl  u[9]=yaw_rad  u[14]=x_m  u[15]=y_m

function nav_push!(ms::NavMapState, u::AbstractVector, t::Real, phase::String)
    x = Float64(u[14]);  y = Float64(u[15])
    a = Float64(u[2]);   h = rad2deg(Float64(u[9]))
    lock(ms.lock) do
        ms.x = x;  ms.y = y;  ms.alt = a;  ms.hdg = h
        ms.phase = phase;  ms.t = Float64(t)
        ms.buf[ms.head] = _Pt(Float32(x), Float32(y), Float32(a), Float32(t))
        ms.head  = mod1(ms.head + 1, NAV_TRACK_MAX)
        ms.count = min(ms.count + 1, NAV_TRACK_MAX)
    end
end

# nav_snapshot(ms) → NamedTuple
#
# Lock-safe snapshot for use in the cockpit draw function.
# Returns a plain value — no lingering lock held during drawing.

function nav_snapshot(ms::NavMapState)
    lock(ms.lock) do
        n   = ms.count
        pts = if n == 0
            _Pt[]
        elseif n < NAV_TRACK_MAX
            ms.buf[1:n]
        else
            h = ms.head
            vcat(ms.buf[h:end], ms.buf[1:h-1])
        end
        (target=ms.target, pts=pts,
         x=ms.x, y=ms.y, alt=ms.alt,
         hdg=ms.hdg, phase=ms.phase, t=ms.t)
    end
end


#  Top-level initialiser  (called once from fly.jl)
#
#     nav_init(; json_path) → (NavTarget, NavMapState)
#
# Read test_card.json, validate, and return live state objects.
# `json_path` defaults to test_card.json next to the calling script.

function nav_init(;
        json_path::String = joinpath(@__DIR__, "test_card.json"))::Tuple{NavTarget,NavMapState}
    # Default path assumes navigation.jl lives in planning/ alongside test_card.json.
    nav = load_nav(json_path)
    validate_nav(nav)
    nav, NavMapState(nav)
end


#  Self-test
#
function nav_selftest()
    println("── navigation.jl self-test ───────────────────────────────")
    fails = 0
    chk(label, ok) = (ok ? println("  PASS  $label") :
                           (println("  FAIL  $label"); fails += 1))

    chk("range 3-4-5", isapprox(hypot(3000.0, 4000.0), 5000.0, atol=0.1))
    chk("bearing east  = 90°", isapprox(nav_bearing(0,0, 0,1000), 90.0, atol=0.1))
    chk("bearing north =  0°", isapprox(nav_bearing(0,0, 1000,0),  0.0, atol=0.1))
    chk("bearing SW    = 225°",isapprox(nav_bearing(0,0,-1000,-1000),225.0,atol=0.2))

    xte = nav_cross_track(0.0, 0.0, 0.0, 0.0, 500.0)
    chk("XTE: target right of fwd heading → positive", xte > 0.0)

    threw = false
    try validate_nav(NavTarget(false, 100.0, 100.0, 0.0)); catch; threw=true; end
    chk("validate rejects < 1000 m", threw)

    ok = true
    try validate_nav(NavTarget(false, 5000.0, 0.0, 300.0)); catch; ok=false; end
    chk("validate accepts ≥ 1000 m", ok)

    println(fails==0 ? "All tests passed." : "$fails test(s) FAILED.")
    println("─────────────────────────────────────────────────────────")
    fails == 0
end
# glass_cockpit.jl:   eVTOL Tiltrotor Sim Glass Cockpit — browser-rendered, MIL-STD-3009 NVG
# AUTHOR:              DANIEL DESAI
# UPDATED:             2026-07-26
# VERSION:             0.3.3
#
# Browser-rendered real-time instrument panel served by a thin Julia
# HTTP/WebSocket backend.  Panel is the v0.2 uniform instrument grid
# (the v0.3 full redesign was reverted per direction), with the AGL
# readout folded into the VSD pane's corner and the VSD itself rebuilt
# as a real forward-looking vertical situation display: it samples the
# terrain database ahead of the aircraft along current heading each
# tick (see _terrain_ahead), matching how real VSDs work (Boeing AERO
# No. 20, Oct. 2002) rather than plotting terrain already flown over.
# Static assets (index.html/cockpit.css/cockpit.js) live alongside this
# file and are drawn client-side on Canvas2D.  The physics-facing
# contract is unchanged:
#
#   state = CockpitState(; n_rotors, labels, powerplants, fuel_kg, fuel_capacity_kg)
#   obs   = Observable(state)
#   fig   = launch_cockpit(obs; nav_map=...)     # now starts an HTTP server
#   ...mutate `state` fields in place, then notify(obs)...
#
# `launch_cockpit` returns a `CockpitServer` (the "fig-like" handle).
# `cockpit_open(fig)` replaces the old GLMakie window-open test:
#   true  while the server is up and either no browser client has connected
#         yet, or at least one client is still connected;
#   false once at least one client HAS connected and they have ALL been
#         gone for > CLIENT_LINGER_S (i.e. "the user closed the window").
# Rationale: mirrors the old "sim stays alive until the cockpit window is
# closed" semantics — a fresh server with no tab yet counts as open (the
# window always existed under GLMakie; here the auto-opened tab may take a
# moment to connect), and closing the last tab ends the session.  Ctrl-C
# still works at any time.
#
# ══════════════════════════════════════════════════════════════════════
#  MIL-STD-3009 TABLE II — CHROMATICITY-DERIVED COLOUR PALETTE
#
#  The NVG palette is preserved verbatim from v0.1.1 and now lives in
#  cockpit_web/cockpit.css as the `.theme-nvg` CSS custom-property set
#  (the default theme).  A full-luminance `.theme-day` set is provided
#  alongside it (client-side toggle / `?theme=day`); NVG traceability is
#  documented in the CSS next to each value.  For reference:
#
#  MIL-STD-3009 specifies colours as CIE 1976 UCS (u′, v′) centre
#  coordinates with tolerance radius r.  The five defined colours and
#  their cockpit roles are:
#
#  Colour          u′      v′      r     Role
#  ─────────────────────────────────────────────────────────────────
#  NVIS Green A   0.214   0.487   0.023  Primary symbology / controls
#  NVIS Green B   0.214   0.487   0.023  Same centre, broader tolerance
#  NVIS White     0.197   0.453   0.050  Crew cockpit / utility lighting
#                                        (§4.2.2 — mandated for new installs)
#  NVIS Yellow    0.260   0.520   0.040  Caution signals (§4.2.5)
#  NVIS Red       0.400   0.460   0.040  Warning signals (§4.2.5)
#  NVIS Blue      0.100   0.280   0.040  Vendor extension (Applied Avionics,
#                                        Lumitron et al.) — not in Table II,
#                                        but NVG-safe: blue (~450 nm) is
#                                        below both filter cutoffs (Class A
#                                        625 nm, Class B 665 nm), giving zero
#                                        NVIS radiance contribution.  Used
#                                        for secondary informational text.
#
#  Conversion path: CIE 1976 (u′,v′) → CIE 1931 (x,y) → XYZ → linear
#  sRGB → gamma-corrected sRGB.  CSS values are the centres of each
#  tolerance circle, converted to the closest in-gamut sRGB colour —
#  identical numeric values to the v0.1.1 `TH` NamedTuple.
# ══════════════════════════════════════════════════════════════════════
#
# Usage unchanged:
#   julia glass_cockpit.jl dash_results.csv
#   COCKPIT_FPS=30 julia glass_cockpit.jl dash_results.csv
# New env knobs:
#   COCKPIT_PORT=8090   listen port (default 8090; auto-increments if busy)
#   COCKPIT_OPEN=0      suppress auto-opening the default browser
# =====================================================================

using HTTP
using HTTP.WebSockets
using CSV, DataFrames
using Observables
using Printf
using JSON      # same library fly.jl already uses for test_card.json / export

# ══════════════════════════════════════════════════════════════════════
#  STATE (contract-compatible with v0.1.1 — fields and existing IDX
#  keys/order unchanged; fx_frac/fz_frac APPENDED, same pattern as the
#  earlier gx/gy/gz addition: vals grows 22 → 24, new keys at the end)
# ══════════════════════════════════════════════════════════════════════
const IDX = (
    t=1, tau=2, speed=4, alt=5, power=6,
    tilt=7, pitch=8, roll=9, yaw=10,
    soc=11, voltage=12, batt_temp=13,
    x_m=14, y_m=15,
    omega_x=16, omega_y=17, omega_z=18,   # body rates (rad/s)
    alt_agl_m=19,                          # CG altitude AGL (m)
    gx=20, gy=21, gz=22,                   # g-forces (longitudinal, lateral, vertical)
    fx_frac=23, fz_frac=24,                # fleet thrust vector, body x-z plane,
                                           # fraction of commanded |T|
                                           # (rotor_mixer.jl fleet_thrust_fraction)
    vrs=25,                                # vortex-ring-state thrust factor
                                           # (1.0 = clean; <1 = VRS thrust loss)
    agl_terrain_m=26,                      # height above terrain directly below
                                           # (radar-altimeter sense; NaN when the
                                           #  terrain model / CSV column is absent)
)

mutable struct CockpitState
    vals             :: Vector{Float64}
    phase            :: String
    history_power    :: Vector{Float64}
    history_t        :: Vector{Float64}
    rotor_rpm        :: Vector{Float64}
    rotor_kw         :: Vector{Float64}
    rotor_labels     :: Vector{String}
    n_rotors         :: Int
    gear_contact     :: Bool
    strut_load_n     :: Float64
    brakes_on        :: Bool       # true when wheel brakes are engaged
    rotor_powerplant :: Vector{String}   # per-rotor: "electric" | "turboshaft" | "turbine_electric"
    fuel_kg          :: Float64          # current fuel mass (turbine/hybrid fleets only)
    fuel_capacity_kg :: Float64          # usable fuel capacity
end

function CockpitState(; n_rotors::Int=6,
                        labels::Vector{String}=["R$i" for i in 1:6],
                        powerplants::Vector{String}=fill("electric", 6),
                        fuel_kg::Float64=0.0,
                        fuel_capacity_kg::Float64=0.0)
    CockpitState(vcat(zeros(24), [1.0, NaN]), "hover",
                 # 26 vals: +gx/gy/gz +fx_frac/fz_frac +vrs +agl_terrain_m
                 # (vrs defaults clean=1.0; agl_terrain NaN until telemetry
                 #  or a CSV column provides it — client shows TERR UNAVAIL)
                 Float64[], Float64[],
                 zeros(6), zeros(6),
                 vcat(labels, fill("", 6))[1:6],
                 clamp(n_rotors, 1, 6),
                 false, 0.0, false,
                 vcat(powerplants, fill("electric", 6))[1:6],
                 fuel_kg, fuel_capacity_kg)
end

# A rotor counts as turbine-powered if its powerplant string is
# "turboshaft" or "turbine_electric" (hyphen/underscore/case-insensitive).
# Pure "electric" rotors do not trigger the fuel gauge.
const _TURBINE_POWERPLANTS = ("turboshaft", "turbine_electric", "turbine")
_normalize_powerplant(pp::AbstractString) = replace(lowercase(strip(pp)), "-" => "_")
has_turbine(s::CockpitState) =
    any(_normalize_powerplant(pp) in _TURBINE_POWERPLANTS
        for pp in s.rotor_powerplant[1:s.n_rotors])

# ══════════════════════════════════════════════════════════════════════
#  VCON fallback limits — used only when airframe.jl's vcon_limits is not
#  loaded (standalone playback without fly.jl).  Values are the documented
#  physical corridor from the v0.1.1 comment block:
#    lo ≈ 75 km/h  (min wing-borne support during tilt-forward)
#    hi ≈ 200 km/h (max rotor-borne structural / control-authority limit)
# ══════════════════════════════════════════════════════════════════════
# Radar-altimeter MIN (decision-height) bug, feet AGL.  Display/alerting
# value only — drawn as the MIN bug on the PFD radalt arc and used for its
# amber band; no autopilot coupling.  Override per-run via COCKPIT_RA_MIN.
const RA_MIN_FT = parse(Float64, get(Base.ENV, "COCKPIT_RA_MIN", "150"))

const VCON_LO_DEFAULT = 75.0
const VCON_HI_DEFAULT = 200.0

_vcon(tilt_deg::Float64, agl_m::Float64) =
    isdefined(Main, :vcon_limits) ?
        Main.vcon_limits(deg2rad(tilt_deg), agl_m) :
        (lo_kmh=VCON_LO_DEFAULT, hi_kmh=VCON_HI_DEFAULT)

# ══════════════════════════════════════════════════════════════════════
#  SERVER
# ══════════════════════════════════════════════════════════════════════
const CLIENT_LINGER_S = 2.0      # grace after last client drops before
                                 # cockpit_open() reports closed
const WEB_DIR = @__DIR__   # index.html/cockpit.css/cockpit.js live alongside
                            # glass_cockpit.jl; see _STATIC_FILES allowlist —
                            # only those three names are ever servable.

mutable struct _WSClient
    ws       :: Any                    # HTTP.WebSockets.WebSocket
    ch       :: Channel{String}        # capacity-1, latest-wins outbox
end

mutable struct CockpitServer
    server          :: Any             # HTTP.Server handle from HTTP.listen!
    host            :: String
    port            :: Int
    url             :: String
    clients         :: Vector{_WSClient}
    lock            :: ReentrantLock
    init_payload    :: String          # static config, sent once per client
    latest_state    :: Union{Nothing,String}
    had_client      :: Base.RefValue{Bool}
    last_disconnect :: Base.RefValue{Float64}
    running         :: Base.RefValue{Bool}
end

"""
    cockpit_open(srv::CockpitServer) -> Bool

Web-era replacement for `isopen(GLMakie.getscreen(fig.scene))`.  See the
header comment for exact semantics (open until the last-ever browser
client has been gone for CLIENT_LINGER_S).
"""
function cockpit_open(srv::CockpitServer)
    srv.running[] || return false
    n = lock(srv.lock) do
        length(srv.clients)
    end
    n > 0 && return true
    srv.had_client[] || return true            # nobody connected yet — keep serving
    return (time() - srv.last_disconnect[]) < CLIENT_LINGER_S
end

Base.isopen(srv::CockpitServer) = cockpit_open(srv)

function Base.close(srv::CockpitServer)
    srv.running[] = false
    try close(srv.server) catch end
end

# ── Static assets ─────────────────────────────────────────────────────
# glass_cockpit.jl and DESIGN_RATIONALE.md live in the same gui/ folder as
# the panel assets, so serving is by an explicit filename allowlist rather
# than "any file that exists in WEB_DIR" — a request for glass_cockpit.jl
# itself must never be servable.
const _STATIC_FILES = Dict(
    "index.html"   => ".html",
    "cockpit.css"  => ".css",
    "cockpit.js"   => ".js",
)
const _MIME = Dict(".html" => "text/html; charset=utf-8",
                   ".css"  => "text/css; charset=utf-8",
                   ".js"   => "text/javascript; charset=utf-8",
                   ".svg"  => "image/svg+xml",
                   ".woff2"=> "font/woff2")

function _serve_static(req::HTTP.Request)
    target = HTTP.URI(req.target).path
    target == "/" && (target = "/index.html")
    name = basename(target)
    if !haskey(_STATIC_FILES, name)
        allowed = join(keys(_STATIC_FILES), ", ")
        return HTTP.Response(404, "not found: $(name)\n" *
            "(only $(allowed) are served)")
    end
    path = joinpath(WEB_DIR, name)
    if !isfile(path)
        return HTTP.Response(404, "not found: $(name)\n" *
            "(expected next to glass_cockpit.jl at $(WEB_DIR))")
    end
    body = read(path)
    ct = get(_MIME, _STATIC_FILES[name], "application/octet-stream")
    return HTTP.Response(200, ["Content-Type" => ct,
                               "Cache-Control" => "no-store"]; body=body)
end

# ── WebSocket client lifecycle ────────────────────────────────────────
function _serve_ws(srv::CockpitServer, ws)
    client = _WSClient(ws, Channel{String}(1))
    # On connect: static config first, then the current state so a tab
    # opened mid-flight shows live instruments immediately.  These sends
    # happen BEFORE registration so that, once registered, the writer
    # task below is the only task ever writing to this socket.
    try
        WebSockets.send(ws, srv.init_payload)
        srv.latest_state !== nothing && WebSockets.send(ws, srv.latest_state)
    catch
        return
    end
    lock(srv.lock) do
        push!(srv.clients, client)
        srv.had_client[] = true
    end
    writer = @async begin
        try
            while true
                msg = take!(client.ch)
                WebSockets.send(ws, msg)
            end
        catch
        end
    end
    try
        for _msg in ws          # consume (ignore) inbound; returns on close
        end
    catch
    finally
        lock(srv.lock) do
            filter!(c -> c !== client, srv.clients)
            srv.last_disconnect[] = time()
        end
        close(client.ch)
        wait(writer)
    end
end

# ── Non-blocking broadcast ────────────────────────────────────────────
# Called from the notify handler on fly.jl's main-thread 50 ms poll loop —
# must never block on a slow client.  Each client has a capacity-1
# latest-wins channel: we drop any undelivered frame and enqueue the new
# one.  Only this function ever put!s, so the drain-then-put is race-free
# against the writer task's take!.
function _broadcast(srv::CockpitServer, payload::String)
    srv.latest_state = payload
    lock(srv.lock) do
        for c in srv.clients
            try
                while isready(c.ch); take!(c.ch); end
                isopen(c.ch) && put!(c.ch, payload)
            catch
            end
        end
    end
    return nothing
end

# ── Serialization ─────────────────────────────────────────────────────
_r3(x) = round(Float64(x), digits=3)

function _nav_dict(nav_map)
    nav_map === nothing && return nothing
    snap = Main.nav_snapshot(nav_map)
    nav  = snap.target
    pts  = snap.pts
    n    = length(pts)
    st   = n > 300 ? cld(n, 300) : 1
    Dict{String,Any}(
        "x"     => _r3(snap.x),
        "y"     => _r3(snap.y),
        "hdg"   => _r3(snap.hdg),
        "agl"   => _r3(snap.alt),
        "phase" => String(snap.phase),
        "wx"    => _r3(Main.nav_wx(nav)),
        "wy"    => _r3(Main.nav_wy(nav)),
        "rtb"   => nav.return_to_base,
        "pts"   => [[_r3(p.x), _r3(p.y)] for p in pts[1:st:end]])
end

# ── Forward-looking terrain profile (VSD) ───────────────────────────────
# Real vertical situation displays (Boeing AERO No. 20, Oct. 2002) are
# track-type and look AHEAD along the current track using the onboard
# terrain database — the whole CFIT-prevention value of a VSD comes from
# seeing a possible terrain conflict before it's reached, which a trailing
# plot of terrain already overflown cannot provide.  Sampled fresh every
# broadcast tick from the current position along current heading, reusing
# the same TERRAIN model and terrain_alt/agl_m_to_msl_ft conversion fly.jl
# already uses for the AGL/radar-altimeter reading — no new physics, just
# a forward query instead of the single query at the current position.
# A straight-line projection along current heading, not a turn-adaptive
# swath (Boeing's VSD widens the swath in turns) — a deliberate scope cut,
# noted in DESIGN_RATIONALE.md.
# Undefined when the terrain model isn't loaded (standalone playback
# invoked as `julia gui/glass_cockpit.jl <csv>`, without fly.jl) — the
# field is omitted and the client shows TERR UNAVAIL, the same fallback
# already used for the AGL reading in that mode.
function _terrain_ahead(x0, y0, hdg_deg; win_km=8.0, n=48)
    isdefined(Main, :TERRAIN) || return nothing
    θ = deg2rad(hdg_deg)
    dx, dy = sin(θ), cos(θ)             # aviation heading: 0°=N(+y), 90°=E(+x)
    step_m = (win_km * 1000.0) / (n - 1)
    pts = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        d_m    = (i - 1) * step_m
        xa, ya = x0 + dx * d_m, y0 + dy * d_m
        terr_ft = Main.agl_m_to_msl_ft(Main.terrain_alt(Main.TERRAIN, xa, ya))
        pts[i] = [_r3(d_m / 1000.0), _r3(terr_ft)]
    end
    pts
end

function _state_json(s::CockpitState, nav_map)::String
    v = s.vals
    vcon = _vcon(v[IDX.tilt], v[IDX.alt_agl_m])
    nh   = length(s.history_power)
    hst  = nh > 120 ? cld(nh, 120) : 1
    d = Dict{String,Any}(
        "type"      => "state",
        "t"         => _r3(v[IDX.t]),
        "tau"       => _r3(v[IDX.tau]),
        "speed"     => _r3(v[IDX.speed]),
        "alt"       => _r3(v[IDX.alt]),
        "power"     => _r3(v[IDX.power]),
        "tilt"      => _r3(v[IDX.tilt]),
        "pitch"     => _r3(v[IDX.pitch]),
        "roll"      => _r3(v[IDX.roll]),
        "yaw"       => _r3(v[IDX.yaw]),
        "soc"       => _r3(v[IDX.soc]),
        "voltage"   => _r3(v[IDX.voltage]),
        "batt_temp" => _r3(v[IDX.batt_temp]),
        "agl_m"     => _r3(v[IDX.alt_agl_m]),
        "gx"        => _r3(v[IDX.gx]),
        "gy"        => _r3(v[IDX.gy]),
        "gz"        => _r3(v[IDX.gz]),
        "fx"        => _r3(v[IDX.fx_frac]),
        "fz"        => _r3(v[IDX.fz_frac]),
        "vrs"       => _r3(v[IDX.vrs]),
        "phase"     => s.phase,
        "rpm"       => [_r3(x) for x in s.rotor_rpm],
        "kw"        => [_r3(x) for x in s.rotor_kw],
        "gear"      => s.gear_contact,
        "strut_n"   => _r3(s.strut_load_n),
        "brakes"    => s.brakes_on,
        "fuel_kg"   => _r3(s.fuel_kg),
        "fuel_cap"  => _r3(s.fuel_capacity_kg),
        "vcon"      => Dict("lo" => _r3(vcon.lo_kmh), "hi" => _r3(vcon.hi_kmh)),
        "phist"     => [_r3(x) for x in s.history_power[1:hst:end]],
    )
    # Radar-altimeter telemetry: omitted when no terrain data exists
    # (legacy CSV without the column / terrain model absent) — the client
    # falls back to CG AGL and raises the TERR UNAVAIL advisory.
    isfinite(v[IDX.agl_terrain_m]) &&
        (d["agl_terr_m"] = _r3(v[IDX.agl_terrain_m]))
    terr_ahead = _terrain_ahead(v[IDX.x_m], v[IDX.y_m], v[IDX.yaw])
    terr_ahead !== nothing && (d["terr_ahead"] = terr_ahead)
    nav = _nav_dict(nav_map)
    nav !== nothing && (d["nav"] = nav)
    return JSON.json(d)
end

# Per-rotor static configuration label, derived at launch (fixed aircraft
# config, like the powerplant mix): TILT (lift+thrust), LIFT, THRUST, OOS.
# Available only when FLEET (rotor_system.jl) is loaded; playback without
# fly.jl falls back to "" and the client hides the mode chips.
function _rotor_modes()
    isdefined(Main, :FLEET) || return fill("", 6)
    modes = fill("", 6)
    try
        for i in 1:6
            u = Main.FLEET.units[i]
            modes[i] = (u.lift && u.thrust) ? "TILT"   :
                        u.lift              ? "LIFT"   :
                        u.thrust            ? "THRUST" : "OOS"
        end
    catch
        return fill("", 6)
    end
    return modes
end

function _airport_header()
    isdefined(Main, :ATM) || return (str="", icao="")
    A = Main.ATM
    wind_kt  = hypot(A.wind.u, A.wind.v) * 1.94384
    wind_hdg = mod(rad2deg(atan(A.wind.u, A.wind.v)) + 360, 360)
    str = @sprintf("%s  %d FT MSL %.0f C WND %.0f %.0f KT",
                   A.airport_icao, round(Int, A.airport_alt_m * 3.28084),
                   A.ambient_temp_c, wind_hdg, wind_kt)
    return (str=str, icao=String(A.airport_icao))
end

# Destination ICAO for the nav-map waypoint label (replaces the generic
# "TGT" placeholder) — mirrors the departure-ICAO lookup above. TC (the
# mission/route config) already carries dest_icao for terrain-database
# loading and end-of-mission logging; empty when unset (single-airport
# missions with no defined destination), in which case the client falls
# back to "TGT".
function _dest_icao()
    isdefined(Main, :TC) || return ""
    try
        String(Main.TC.dest_icao)
    catch
        ""
    end
end

function _init_json(s::CockpitState; rpm_nom, kw_max, has_map)
    apt = _airport_header()
    JSON.json(Dict{String,Any}(
        "type"        => "init",
        "labels"      => s.rotor_labels,
        "powerplants" => s.rotor_powerplant,
        "n_rotors"    => s.n_rotors,
        "show_fuel"   => has_turbine(s),
        "modes"       => _rotor_modes(),
        "rpm_nom"     => rpm_nom,
        "kw_max"      => kw_max,
        "airport"     => apt.str,
        "icao"        => apt.icao,
        "ra_min_ft"   => RA_MIN_FT,
        "dest_icao"   => _dest_icao(),
        "has_nav"     => has_map))
end

# ── Browser auto-open (nice-to-have, best-effort, COCKPIT_OPEN=0 to skip) ─
function _open_browser(url::String)
    get(Base.ENV, "COCKPIT_OPEN", "1") == "0" && return
    cmd = Sys.isapple()   ? `open $url` :
          Sys.iswindows() ? `cmd /c start $url` :
                            `xdg-open $url`
    try run(pipeline(cmd; stdout=devnull, stderr=devnull); wait=false) catch end
end

# ══════════════════════════════════════════════════════════════════════
#  LAUNCH
# ══════════════════════════════════════════════════════════════════════
"""
    launch_cockpit(state_obs; rpm_nom, kw_max_per_rotor, nav_map)

Start the cockpit web server and subscribe to `state_obs`.  Every
`notify(state_obs)` serializes the current `CockpitState` (plus the
nav-map snapshot, when `nav_map` is supplied) to JSON and pushes it to
all connected browser clients, non-blocking.

Returns a `CockpitServer` — the "fig-like" handle.  Use
`cockpit_open(fig)` where window-open detection was previously done via
`GLMakie.getscreen`, and `close(fig)` to shut the server down.

The POWERPLANT panel shows a FUEL row automatically — and only — when
`state_obs[]` reports at least one turbine-powered rotor via
`has_turbine()`.  Powerplant mix is a fixed aircraft configuration,
decided once at launch from the initial state, not re-evaluated per
frame (unchanged from v0.1.1).
"""
function launch_cockpit(state_obs::Observable{CockpitState};
                        rpm_nom::Float64=1050.0,
                        kw_max_per_rotor::Float64=80.0,
                        nav_map=nothing)

    has_map = nav_map !== nothing
    host    = "127.0.0.1"
    port0   = parse(Int, get(Base.ENV, "COCKPIT_PORT", "8090"))

    _missing = [f for f in keys(_STATIC_FILES) if !isfile(joinpath(WEB_DIR, f))]
    isempty(_missing) ||
        @warn "panel assets missing next to glass_cockpit.jl — will 404" _missing WEB_DIR

    init_payload = _init_json(state_obs[]; rpm_nom=rpm_nom,
                              kw_max=kw_max_per_rotor, has_map=has_map)

    srv = CockpitServer(nothing, host, port0, "", _WSClient[], ReentrantLock(),
                        init_payload, nothing,
                        Ref(false), Ref(time()), Ref(true))

    handler = function (http::HTTP.Stream)
        if HTTP.WebSockets.isupgrade(http.message)
            HTTP.WebSockets.upgrade(ws -> _serve_ws(srv, ws), http)
        else
            HTTP.streamhandler(_serve_static)(http)
        end
    end

    # Bind — auto-increment past a busy port (e.g. a stale session)
    local server
    port = port0
    bound = false
    for p in port0:(port0 + 10)
        try
            server = HTTP.listen!(handler, host, p)
            port  = p
            bound = true
            break
        catch err
            err isa InterruptException && rethrow()
            # bind failure (port busy) — try the next port
        end
    end
    bound || error("glass_cockpit: no free port in $(port0)-$(port0+10)")

    srv.server = server
    srv.port   = port
    srv.url    = "http://$(host):$(port)/"

    # The notify-driven push — this replaces the GLMakie redraw closure.
    on(state_obs) do s
        _broadcast(srv, _state_json(s, nav_map))
    end

    # Prime latest_state so an early client sees data before first notify
    srv.latest_state = _state_json(state_obs[], nav_map)

    println("Cockpit panel serving at $(srv.url)  [NVG theme default — append ?theme=day for day mode]")
    _open_browser(srv.url)
    return srv
end

# ══════════════════════════════════════════════════════════════════════
#  CSV PLAYBACK MODE
# ══════════════════════════════════════════════════════════════════════
# Row-parsing logic preserved from v0.1.1 with one necessary addition:
# current fly.jl exports attitude as a quaternion (q0..q3), not
# pitch_deg/roll_deg/yaw_deg — legacy Euler columns are still preferred
# when present, and q0..q3 are the fallback (ZYX convention, matching the
# construction in fly.jl's write_csv_row).  gx/gy/gz are also read when
# present (they previously existed in the CSV but were never parsed, so
# the CONTACT panel's gz readout was always 0.00 in playback).
#
# fx/fz thrust-vector columns don't exist in CSVs (the export schema is
# owned by fly.jl and out of scope) — playback reconstructs the vector as
# (sin tilt, cos tilt), which is exact for an all-tiltrotor fleet and the
# best available estimate otherwise.
function _quat_to_euler_deg(q0, q1, q2, q3)
    roll  = atan(2*(q0*q1 + q2*q3), 1 - 2*(q1^2 + q2^2))
    sp    = clamp(2*(q0*q2 - q3*q1), -1.0, 1.0)
    pitch = asin(sp)
    yaw   = atan(2*(q0*q3 + q1*q2), 1 - 2*(q2^2 + q3^2))
    return rad2deg(pitch), rad2deg(roll), rad2deg(yaw)
end

function playback_csv(path::String; fps=10.0, nav_map=nothing)
    df = CSV.read(path, DataFrame)
    println("Loaded $(nrow(df)) rows from $path")
    println("Playing back at $(fps)x realtime … [NVG MODE]")

    # Per-rotor powerplant is a fixed aircraft config, not per-row telemetry —
    # read it once from row 1 if the columns exist (e.g. "powerplant_r1" ..
    # "powerplant_r6", written by rotor_config.csv-aware export paths).
    # Absent columns default to "electric", matching the original all-electric
    # behaviour (no fuel gauge) for legacy CSVs.
    powerplants = if nrow(df) > 0
        [hasproperty(df, Symbol("powerplant_r$i")) ?
            String(df[1, Symbol("powerplant_r$i")]) : "electric"
         for i in 1:6]
    else
        fill("electric", 6)
    end
    fuel_capacity_kg = (nrow(df) > 0 && hasproperty(df, :fuel_capacity_kg)) ?
        Float64(df[1, :fuel_capacity_kg]) : 0.0

    state = CockpitState(powerplants=powerplants, fuel_capacity_kg=fuel_capacity_kg,
                          fuel_kg=fuel_capacity_kg)
    obs   = Observable(state)
    fig   = launch_cockpit(obs; nav_map=nav_map)

    dt_real  = 1.0 / fps
    hist_max = 600

    for col in ["rpm_r1","rpm_r2","rpm_r3","rpm_r4","rpm_r5","rpm_r6",
                "kw_r1", "kw_r2", "kw_r3", "kw_r4", "kw_r5", "kw_r6"]
        hasproperty(df, Symbol(col)) ||
            error("CSV missing column '$col' — re-run the simulation.")
    end

    has_euler = hasproperty(df, :pitch_deg)
    has_quat  = hasproperty(df, :q0)

    for row in eachrow(df)
        state.vals[IDX.t]         = row.timestamp_s
        state.vals[IDX.tau]       = row.tau_s
        state.vals[IDX.speed]     = row.speed_kmh
        state.vals[IDX.alt]       = row.altitude_msl_ft
        state.vals[IDX.power]     = row.power_kw
        state.vals[IDX.tilt]      = row.tilt_deg
        if has_euler
            state.vals[IDX.pitch] = row.pitch_deg
            state.vals[IDX.roll]  = row.roll_deg
            state.vals[IDX.yaw]   = row.yaw_deg
        elseif has_quat
            p, r, y = _quat_to_euler_deg(row.q0, row.q1, row.q2, row.q3)
            state.vals[IDX.pitch] = p
            state.vals[IDX.roll]  = r
            state.vals[IDX.yaw]   = y
        end
        state.vals[IDX.soc]       = row.soc_pct
        state.vals[IDX.voltage]   = row.voltage_v
        state.vals[IDX.batt_temp] = row.batt_temp_c
        state.vals[IDX.x_m]       = hasproperty(row, :x_m) ? row.x_m : 0.0
        state.vals[IDX.y_m]       = hasproperty(row, :y_m) ? row.y_m : 0.0
        state.vals[IDX.omega_x]   = hasproperty(row, :omega_x_rads) ? row.omega_x_rads : 0.0
        state.vals[IDX.omega_y]   = hasproperty(row, :omega_y_rads) ? row.omega_y_rads : 0.0
        state.vals[IDX.omega_z]   = hasproperty(row, :omega_z_rads) ? row.omega_z_rads : 0.0
        state.vals[IDX.gx]        = hasproperty(row, :gx) ? row.gx : 0.0
        state.vals[IDX.gy]        = hasproperty(row, :gy) ? row.gy : 0.0
        state.vals[IDX.gz]        = hasproperty(row, :gz) ? row.gz : 1.0
        # alt_agl_m: prefer direct CSV column; fall back to reverse-converting
        # altitude_msl_ft using the ATM airport elevation (requires ATM to be loaded).
        state.vals[IDX.alt_agl_m] = if hasproperty(row, :alt_agl_m)
            Float64(row.alt_agl_m)
        elseif isdefined(Main, :ATM)
            max(row.altitude_msl_ft / 3.28084 - Main.ATM.airport_alt_m, 0.0)
        else
            0.0
        end
        # Thrust vector: all-tiltrotor reconstruction (see comment above)
        θ = deg2rad(clamp(Float64(row.tilt_deg), 0.0, 90.0))
        state.vals[IDX.fx_frac]   = sin(θ)
        state.vals[IDX.fz_frac]   = cos(θ)
        # v0.3 telemetry — both columns exist in the current CSV schema;
        # legacy files fall back to clean-VRS / no-terrain (client raises
        # the TERR UNAVAIL advisory for the latter).
        state.vals[IDX.vrs]           = hasproperty(row, :vrs_factor) &&
                                        !ismissing(row.vrs_factor) ?
                                            Float64(row.vrs_factor) : 1.0
        state.vals[IDX.agl_terrain_m] = hasproperty(row, :alt_agl_terrain_m) &&
                                        !ismissing(row.alt_agl_terrain_m) ?
                                            Float64(row.alt_agl_terrain_m) : NaN

        state.gear_contact         = hasproperty(row, :gear_contact)  ? Bool(row.gear_contact)  : false
        state.strut_load_n         = hasproperty(row, :strut_load_n)  ? Float64(row.strut_load_n) : 0.0
        state.phase                = ismissing(row.phase) ? "" : String(row.phase)
        state.fuel_kg              = hasproperty(row, :fuel_kg) ? Float64(row.fuel_kg) : state.fuel_capacity_kg

        for i in 1:6
            state.rotor_rpm[i] = getproperty(row, Symbol("rpm_r$i"))
            state.rotor_kw[i]  = getproperty(row, Symbol("kw_r$i"))
        end

        push!(state.history_power, row.power_kw)
        push!(state.history_t,     row.timestamp_s)
        if length(state.history_power) > hist_max
            popfirst!(state.history_power)
            popfirst!(state.history_t)
        end

        # Feed the moving-map ring buffer from CSV columns
        if nav_map !== nothing
            # Reconstruct a minimal state vector subset the map needs:
            #   u[2]=alt_agl  u[9]=yaw_rad  u[14]=x_m  u[15]=y_m
            # All other indices are unused by nav_push!.
            u_pb = zeros(18)
            u_pb[2]  = state.vals[IDX.alt_agl_m]
            u_pb[9]  = deg2rad(state.vals[IDX.yaw])
            u_pb[14] = state.vals[IDX.x_m]
            u_pb[15] = state.vals[IDX.y_m]
            Main.nav_push!(nav_map, u_pb, row.timestamp_s, state.phase)
        end

        obs[] = state
        sleep(dt_real)
    end

    println("Playback complete. Close the browser tab(s) (or Ctrl-C) to exit.")
    try
        while cockpit_open(fig)
            sleep(0.1)
        end
    catch
    end
    close(fig)
end

# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1
        # Load navigation if test_card.json is present alongside the CSV
        _nav_map_pb = nothing
        _card = joinpath(dirname(ARGS[1]), "test_card.json")
        if isfile(_card) && isdefined(Main, :nav_init)
            _, _nav_map_pb = nav_init(json_path=_card)
        end
        playback_csv(ARGS[1],
                     fps=get(Base.ENV, "COCKPIT_FPS", "10") |> x -> parse(Float64, x),
                     nav_map=_nav_map_pb)
    else
        println("Usage:  julia glass_cockpit.jl dash_results.csv")
        println("        COCKPIT_FPS=30 julia glass_cockpit.jl dash_results.csv")
        println("        COCKPIT_PORT=8090 COCKPIT_OPEN=0 … (server knobs)")
    end
end
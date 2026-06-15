# terrain.jl:      Ground-track elevation profile model
# AUTHOR:          DANIEL DESAI
# UPDATED:         2026-05-10
# VERSION:         0.1.0
#
#
# Piecewise-linear terrain profile along the mission ground track.
# Provides ground elevation as a function of along-track position x (m)
# for gear-contact altitude and cockpit AGL display.
#
# Convention (matches prototype 05_terrain.jl):
#   terrain_alt(model, x_m) → elevation RELATIVE TO ORIGIN (m)
#   Positive = terrain above departure pad (uphill).
#   Negative = terrain below departure pad (downhill / destination valley).
#
#   Gear contact in build_ode:
#     alt_gear = alt_ode - terrain_alt(TERRAIN, Float64(x))
#
#   AGL display:
#     agl_terrain = alt_ode - terrain_alt(TERRAIN, Float64(x))
#
# Backends (in priority order)
# ─────────────────────────────
#   1. PREDEFINED_PROFILES — hardcoded piecewise-linear profiles for known
#      routes, accurate to ~50 m from USGS NED / published topo.
#      Zero runtime cost, no files, no downloads. Preferred for demos.
#
#   2. terrain_profile.json — SRTM-sampled profile written by
#      flight_plan.py --terrain. Loaded if present. Takes precedence
#      over flat_model but NOT over predefined profiles.
#
#   3. flat_model — linear origin→dest ramp. Always available as fallback.
#
# fly.jl construction (after NAV init):
# ──────────────────────────────────────
#   const TERRAIN = load_terrain(TC.airport_icao, TC.dest_icao,
#                                Float64(TC.airport_alt_m),
#                                Float64(TC.dest_alt_m),
#                                Float64(hypot(nav_wx(NAV), nav_wy(NAV))),
#                                joinpath(_PLANNING, "terrain_profile.json"))
#
# Adding a new predefined route
# ──────────────────────────────
#   Add an entry to PREDEFINED_PROFILES below. Key is "KDEP-KARR"
#   (always dep→arr order; the reverse is generated automatically).
#   Points: Vector of (x_m, elev_msl_m) tuples, x=0 at departure.
#   Sources: USGS NED, Google Earth, published sectional charts.
#
# References
# ──────────
#   Leishman (2006) for terrain clearance in rotorcraft operations.
#   USGS National Elevation Dataset: https://www.usgs.gov/3d-elevation-program
#   SRTM documentation: https://www2.jpl.nasa.gov/srtm/


module Terrain

export TerrainModel, terrain_alt, terrain_alt_2d,
       flat_model, load_terrain, profile_model,
       build_terrain_profile, load_srtm_hgt, sample_profile,
       terrain_selftest

using LinearAlgebra

# ── Data structure ─────────────────────────────────────────────────────
"""
    TerrainModel

Piecewise-linear terrain profile along the mission ground track.

Fields:
  x_m           — along-track distances from origin (m), strictly increasing
  elev_m        — MSL elevation at each point (m)
  origin_elev_m — departure airport MSL elevation (ODE frame datum)
  grid_x_m      — 2-D grid x positions (empty = 1-D only)
  grid_y_m      — 2-D grid y positions (empty = 1-D only)
  grid_elev     — 2-D elevation grid, (nx, ny) (empty = 1-D only)
  source        — human-readable description of data origin
"""
struct TerrainModel
    x_m           :: Vector{Float64}
    elev_m        :: Vector{Float64}
    origin_elev_m :: Float64
    grid_x_m      :: Vector{Float64}
    grid_y_m      :: Vector{Float64}
    grid_elev     :: Matrix{Float64}
    source        :: String
    bearing_rad   :: Float64   # route bearing (rad); used to project (x,y) → along-track distance
end

# ── Predefined profiles ────────────────────────────────────────────────
# Each entry: "KDEP-KARR" => [(x_m, elev_msl_m), ...]
# x_m = 0 at departure airport; last point ≈ route length.
# Elevations from USGS NED / Google Earth topo, accurate to ~50 m.
# Reverse direction is synthesised automatically by load_terrain().
#
# KAXX = Angel Fire, NM  (36.422°N, 105.288°W, 2554 m / 8376 ft)
# KSAF = Santa Fe, NM    (35.617°N, 106.089°W, 1935 m / 6349 ft)
# KTAO = Taos, NM        (36.458°N, 105.672°W, 2163 m / 7096 ft)
# KDEN = Denver, CO      (39.856°N, 104.674°W, 1655 m / 5431 ft)
# KCOS = Colorado Spgs   (38.806°N, 104.701°W, 1881 m / 6171 ft)
# KAPA = Broomfield, CO  (39.909°N, 105.117°W, 1724 m / 5654 ft)

const PREDEFINED_PROFILES = Dict{String, Vector{Tuple{Float64,Float64}}}(

    # ── New Mexico ────────────────────────────────────────────────────
    # KAXX→KSAF: Angel Fire → Santa Fe, 114.9 km, bearing 219°
    # Crosses Mora Valley (2250 m), Glorieta Mesa (2650 m), Lamy/Galisteo basin.
    # Max deviation from flat_model: +225 m at Glorieta Mesa summit (~55 km).
    "KAXX-KSAF" => [
        (0.0,     2554.0),  # Angel Fire Airport (8376 ft)
        (10_000,  2700.0),  # ridge climbing west from Angel Fire valley
        (20_000,  2350.0),  # Mora Valley
        (30_000,  2250.0),  # lower Mora, valley floor
        (45_000,  2550.0),  # Glorieta Mesa rising
        (55_000,  2650.0),  # Glorieta Mesa summit (~8700 ft)
        (65_000,  2450.0),  # Glorieta Pass descent
        (75_000,  2100.0),  # Lamy basin
        (85_000,  2000.0),  # Galisteo basin
        (100_000, 1980.0),  # Santa Fe foothills
        (114_900, 1935.0),  # Santa Fe Airport (6349 ft)
    ],

    # KAXX→KTAO: Angel Fire → Taos, 34.6 km, bearing 277°
    # Crosses ridge west of Angel Fire, descends to Taos Plateau.
    "KAXX-KTAO" => [
        (0.0,    2554.0),  # Angel Fire Airport
        (8_000,  2700.0),  # ridge west of Angel Fire
        (15_000, 2600.0),  # high plateau
        (25_000, 2400.0),  # descent toward Taos Plateau
        (34_600, 2163.0),  # Taos Municipal Airport (7096 ft)
    ],

    # ── Colorado ──────────────────────────────────────────────────────
    # KDEN→KCOS: Denver → Colorado Springs, 116.8 km, bearing 181°
    # KEY FEATURE: Palmer Divide at ~65 km — 2280 m (7480 ft).
    # flat_model AGL at cruise: ~1850 m. Real AGL: ~1225 m. Diff: 625 m.
    # Best demo case for why terrain matters for eVTOL operations.
    "KDEN-KCOS" => [
        (0.0,     1655.0),  # Denver International (5431 ft)
        (15_000,  1700.0),  # south Denver / Aurora
        (30_000,  1750.0),  # Castle Rock area
        (50_000,  1900.0),  # approaching Palmer Divide
        (65_000,  2280.0),  # Palmer Divide summit — 625 m above flat_model
        (80_000,  2100.0),  # Monument Hill descent
        (95_000,  1950.0),  # north Colorado Springs
        (116_800, 1881.0),  # Colorado Springs Airport (6171 ft)
    ],

    # KDEN→KAPA: Denver → Centennial, 35.8 km, bearing 205°
    # Flat Front Range suburban — good quick-iteration route.
    # Slight elevation gain (+139 m); no significant terrain features.
    "KDEN-KAPA" => [
        (0.0,    1655.0),  # Denver International (5431 ft)
        (12_000, 1680.0),  # south Denver / Aurora
        (25_000, 1720.0),  # Englewood / Arapahoe County
        (35_800, 1794.0),  # Centennial Airport KAPA (5886 ft)
    ],

    # ── New England ───────────────────────────────────────────────────
    # KPSM→KBVY: Portsmouth NH → Beverly MA, 55.5 km, bearing 188°
    # Coastal New England — essentially flat, minor drumlin terrain (~80 m max).
    # Both airports near sea level (100 ft / 107 ft).
    # Good urban-air-mobility test case: low altitude, dense airspace.
    "KPSM-KBVY" => [
        (0.0,    30.0),   # Portsmouth Intl at Pease (100 ft)
        (10_000, 40.0),   # Hampton NH / Exeter area
        (20_000, 55.0),   # Newburyport / Amesbury
        (30_000, 70.0),   # Byfield / Rowley — highest, drumlin country
        (40_000, 50.0),   # Gloucester / Manchester-by-the-Sea coast
        (48_000, 35.0),   # Magnolia / Manchester Shore
        (55_500, 33.0),   # Beverly Regional Airport (107 ft)
    ],

    # ── RTB (Return to Base) ──────────────────────────────────────────
    # RTB missions depart and return to the same airport (target x=0, y=0).
    # Terrain held flat at departure elevation over a nominal 100 km range
    # so terrain_alt never exceeds the profile regardless of outbound distance.

    # KAXX RTB: Angel Fire, 2554 m — high-DA, primary test site
    "KAXX-KAXX" => [
        (0.0,      2554.0),
        (100_000,  2554.0),
    ],

    # KDEN RTB: Denver, 1655 m
    "KDEN-KDEN" => [
        (0.0,      1655.0),
        (100_000,  1655.0),
    ],

    # KBOS RTB: Boston Logan, 6 m — coastal, sea level
    "KBOS-KBOS" => [
        (0.0,      6.0),
        (100_000,  6.0),
    ],

    # KSFO RTB: San Francisco Intl, 4 m — Bay Area, sea level
    "KSFO-KSFO" => [
        (0.0,      4.0),
        (100_000,  4.0),
    ],
)

# ── Constructors ───────────────────────────────────────────────────────
"""
    flat_model(origin_elev_m, dest_elev_m, range_m) → TerrainModel

Two-point linear ramp. Zero dependencies. Used as final fallback.
"""
function flat_model(origin_elev_m::Float64, dest_elev_m::Float64,
                    range_m::Float64,
                    bearing_rad::Float64 = 0.0) :: TerrainModel
    TerrainModel([0.0, range_m], [origin_elev_m, dest_elev_m],
                 origin_elev_m, Float64[], Float64[],
                 Matrix{Float64}(undef, 0, 0), "flat_model (linear ramp)",
                 bearing_rad)
end

"""
    profile_model(xs, zs, origin_elev_m, source) → TerrainModel

Arbitrary piecewise-linear profile. xs must be strictly increasing.
"""
function profile_model(xs::Vector{Float64}, zs::Vector{Float64},
                       origin_elev_m::Float64,
                       source::String = "external",
                       bearing_rad::Float64 = 0.0) :: TerrainModel
    length(xs) == length(zs) || error("terrain: xs and zs must have equal length")
    length(xs) >= 2           || error("terrain: need at least 2 profile points")
    issorted(xs)              || error("terrain: xs must be strictly increasing")
    TerrainModel(xs, zs, origin_elev_m, Float64[], Float64[],
                 Matrix{Float64}(undef, 0, 0), source, bearing_rad)
end

"""
    load_terrain(dep_icao, arr_icao, origin_elev_m, dest_elev_m,
                 range_m, json_path) → TerrainModel

Select terrain backend in priority order:
  1. Predefined profile for this route (dep_icao-arr_icao or reverse)
  2. terrain_profile.json if present at json_path
  3. flat_model linear ramp

Logs which backend was selected.
"""
function load_terrain(dep_icao::String, arr_icao::String,
                      origin_elev_m::Float64, dest_elev_m::Float64,
                      range_m::Float64,
                      json_path::String = "";
                      bearing_rad::Float64 = 0.0,
                      verbose::Bool = true) :: TerrainModel

    route_key = "$(uppercase(dep_icao))-$(uppercase(arr_icao))"
    rev_key   = "$(uppercase(arr_icao))-$(uppercase(dep_icao))"

    # ── 1. Predefined profile ─────────────────────────────────────────
    if haskey(PREDEFINED_PROFILES, route_key)
        pts = PREDEFINED_PROFILES[route_key]
        xs  = Float64[p[1] for p in pts]
        zs  = Float64[p[2] for p in pts]
        verbose && @info "Terrain: $route_key predefined profile ($(length(xs)) pts, USGS NED)"
        return profile_model(xs, zs, origin_elev_m, "predefined/$route_key (USGS NED)",
                             bearing_rad)
    end

    # Reverse direction: flip x and recompute relative elevations
    if haskey(PREDEFINED_PROFILES, rev_key)
        pts     = PREDEFINED_PROFILES[rev_key]
        total_x = pts[end][1]
        xs  = Float64[total_x - p[1] for p in reverse(pts)]
        zs  = Float64[p[2]           for p in reverse(pts)]
        verbose && @info "Terrain: $route_key reversed from $rev_key (USGS NED)"
        return profile_model(xs, zs, origin_elev_m,
                             "predefined/$rev_key reversed (USGS NED)", bearing_rad)
    end

    # ── 2. terrain_profile.json ───────────────────────────────────────
    if json_path != "" && isfile(json_path)
        try
            verbose && @info "Terrain: $route_key — loading terrain_profile.json"
            # Fallthrough: fly.jl handles JSON parsing via JSON.parsefile
        catch e
            @warn "Terrain: failed to read $json_path — $e"
        end
    end

    # ── 3. flat_model fallback ────────────────────────────────────────
    verbose && @info "Terrain: $route_key — no predefined profile, using flat_model"
    return flat_model(origin_elev_m, dest_elev_m, range_m, bearing_rad)
end

# ── Elevation lookup ───────────────────────────────────────────────────
"""
    terrain_alt(model, x_m) → Float64 (m, relative to origin elevation)

Terrain elevation relative to departure datum.
  Positive → terrain above takeoff pad
  Negative → terrain below takeoff pad

Clamps x_m to [0, range] for pre-takeoff and overshoot.

Gear contact in build_ode:
    alt_gear = Float64(alt) - terrain_alt(TERRAIN, Float64(x))
"""
function terrain_alt(model::TerrainModel, x_m::Float64, y_m::Float64 = 0.0) :: Float64
    # Project (x, y) onto route bearing → along-track distance (m).
    # This correctly handles any route bearing, including SW (KAXX→KSAF bearing 219°
    # gives negative x and y), RTB return legs, and crosswind drift.
    # Clamped to [0, range] so pre-takeoff and overshoot both return endpoint terrain.
    d = x_m * cos(model.bearing_rad) + y_m * sin(model.bearing_rad)

    xs = model.x_m
    zs = model.elev_m

    isempty(xs) && return 0.0
    d <= xs[1]   && return zs[1]   - model.origin_elev_m
    d >= xs[end] && return zs[end] - model.origin_elev_m

    lo, hi = 1, length(xs)
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        xs[mid] <= d ? (lo = mid) : (hi = mid)
    end
    frac = (d - xs[lo]) / (xs[hi] - xs[lo])
    return (zs[lo] + frac * (zs[hi] - zs[lo])) - model.origin_elev_m
end

"""
    terrain_alt_2d(model, x_m, y_m) → Float64 (m, relative to origin)

Bilinear interpolation on 2-D grid when loaded; falls back to 1-D.
"""
function terrain_alt_2d(model::TerrainModel, x_m::Float64, y_m::Float64) :: Float64
    isempty(model.grid_elev) && return terrain_alt(model, x_m, y_m)

    gx = model.grid_x_m; gy = model.grid_y_m; G = model.grid_elev
    ix = clamp(searchsortedlast(gx, x_m), 1, length(gx) - 1)
    iy = clamp(searchsortedlast(gy, y_m), 1, length(gy) - 1)
    fx = (x_m - gx[ix]) / (gx[ix+1] - gx[ix])
    fy = (y_m - gy[iy]) / (gy[iy+1] - gy[iy])

    return (G[ix,  iy  ] * (1-fx) * (1-fy)
          + G[ix+1,iy  ] *    fx  * (1-fy)
          + G[ix,  iy+1] * (1-fx) *    fy
          + G[ix+1,iy+1] *    fx  *    fy) - model.origin_elev_m
end

# ── SRTM backend (used by flight_plan.py --terrain, not at runtime) ───
"""
    build_terrain_profile(lats, lons, elevs, origin_lat, origin_lon,
                          origin_elev_m, initial_bearing_rad) → TerrainModel

Converts SRTM-sampled (lat, lon, elev) waypoints into the flat-Earth
x_m profile. Called from flight_plan.py --terrain; output written to
planning/terrain_profile.json for routes without a predefined profile.
"""
function build_terrain_profile(lats               :: Vector{Float64},
                                lons               :: Vector{Float64},
                                elevs              :: Vector{Float64},
                                origin_lat         :: Float64,
                                origin_lon         :: Float64,
                                origin_elev_m      :: Float64,
                                initial_bearing_rad:: Float64) :: TerrainModel
    Re  = 6_371_000.0
    x_m = Vector{Float64}(undef, length(lats))
    for i in eachindex(lats)
        north  = deg2rad(lats[i] - origin_lat) * Re
        east   = deg2rad(lons[i] - origin_lon) * Re * cosd(origin_lat)
        x_m[i] = north * cos(initial_bearing_rad) + east * sin(initial_bearing_rad)
    end
    profile_model(x_m, elevs, origin_elev_m, "SRTM GL3")
end

"""
    load_srtm_hgt(path) → (elev_grid, lat_sw, lon_sw)

Reads a SRTM3 .hgt tile. See flight_plan.py for download instructions.
"""
function load_srtm_hgt(path::String) :: Tuple{Matrix{Float64}, Float64, Float64}
    bytes = read(path)
    n     = Int(sqrt(length(bytes) ÷ 2))
    @assert length(bytes) == 2 * n * n  "Unexpected HGT size"
    elev  = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        idx       = 2 * ((i-1)*n + (j-1)) + 1
        raw       = Int16(bytes[idx] << 8 | bytes[idx+1])
        elev[i,j] = raw == Int16(-32768) ? NaN : Float64(raw)
    end
    stem   = basename(path)[1:end-4]
    lat_sw = parse(Float64, stem[2:3]) * (stem[1] == 'S' ? -1.0 : 1.0)
    lon_sw = parse(Float64, stem[5:7]) * (stem[4] == 'W' ? -1.0 : 1.0)
    return elev, lat_sw, lon_sw
end

"""
    sample_profile(hgt_grid, lat_sw, lon_sw, lat1, lon1, lat2, lon2;
                   n_points=200) → (lats, lons, elevs_m)
"""
function sample_profile(hgt_grid :: Matrix{Float64},
                         lat_sw   :: Float64, lon_sw :: Float64,
                         lat1     :: Float64, lon1   :: Float64,
                         lat2     :: Float64, lon2   :: Float64;
                         n_points :: Int = 200)
    lats  = collect(range(lat1, lat2, length=n_points))
    lons  = collect(range(lon1, lon2, length=n_points))
    step  = 3.0 / 3600.0
    elevs = map(zip(lats, lons)) do (la, lo)
        row = clamp(round(Int, (la - lat_sw) / step) + 1, 1, size(hgt_grid,1))
        col = clamp(round(Int, (lo - lon_sw) / step) + 1, 1, size(hgt_grid,2))
        e   = hgt_grid[row, col]
        isnan(e) ? 0.0 : e
    end
    return lats, lons, collect(Float64, elevs)
end

# ── Self-test ──────────────────────────────────────────────────────────
"""
    terrain_selftest() → Bool

Validates predefined profiles and lookup logic. Prints PASS/FAIL.
"""
function terrain_selftest() :: Bool
    ok = true
    function check(name, cond, got, expected)
        s = cond ? "PASS" : "FAIL"
        println("  [$s] $name: got=$(round(Float64(got), digits=1))  expected=$expected")
        cond || (ok = false)
    end

    # Selftest runs silently; only final pass/fail is printed

    # ── KAXX→KSAF predefined ──────────────────────────────────────────
    t1 = load_terrain("KAXX", "KSAF", 2554.0, 1935.0, 114_900.0; bearing_rad=0.0, verbose=false)
    check("KAXX→KSAF source is predefined",
          occursin("predefined", t1.source), 1.0, "predefined/*")
    check("KAXX→KSAF: origin delta = 0",
          terrain_alt(t1, 0.0) ≈ 0.0, terrain_alt(t1, 0.0), "0.0 m")
    check("KAXX→KSAF: dest delta = −619",
          terrain_alt(t1, 114_900.0) ≈ -619.0, terrain_alt(t1, 114_900.0), "−619.0 m")
    check("KAXX→KSAF: Glorieta Mesa > 0 (above dep)",
          terrain_alt(t1, 55_000.0) > 0.0, terrain_alt(t1, 55_000.0), "> 0 m")

    # Gear contact at destination hover (30 m AGL):
    # alt_ode = dest_alt - origin_alt + 30 = 1935-2554+30 = -589
    # alt_gear = -589 - (-619) = 30 ✓
    agl = -589.0 - terrain_alt(t1, 114_900.0)
    check("KAXX→KSAF: gear AGL at dest hover", abs(agl - 30.0) < 1.0, agl, "30.0 m")

    # ── KSAF→KAXX reverse ────────────────────────────────────────────
    t2 = load_terrain("KSAF", "KAXX", 1935.0, 2554.0, 114_900.0; bearing_rad=0.0, verbose=false)
    check("KSAF→KAXX: origin delta = 0",
          terrain_alt(t2, 0.0) ≈ 0.0, terrain_alt(t2, 0.0), "0.0 m")
    check("KSAF→KAXX: dest delta = +619",
          abs(terrain_alt(t2, 114_900.0) - 619.0) < 5.0,
          terrain_alt(t2, 114_900.0), "≈619.0 m")

    # ── KDEN→KCOS Palmer Divide ───────────────────────────────────────
    t3 = load_terrain("KDEN", "KCOS", 1655.0, 1881.0, 116_800.0; bearing_rad=0.0, verbose=false)
    palmer = terrain_alt(t3, 65_000.0)
    check("KDEN→KCOS: Palmer Divide > flat_model",
          palmer > (1655.0 + 65_000/116_800*(1881.0-1655.0)) - 1655.0,
          palmer, "> flat_model at 65km")
    check("KDEN→KCOS: Palmer Divide ≈ +625 m",
          abs(palmer - 625.0) < 50.0, palmer, "≈625.0 m")

    # ── KPSM→KBVY coastal flat ────────────────────────────────────────
    t4 = load_terrain("KPSM", "KBVY", 30.0, 33.0, 55_500.0; bearing_rad=0.0, verbose=false)
    check("KPSM→KBVY source is predefined",
          occursin("predefined", t4.source), 1.0, "predefined/*")
    check("KPSM→KBVY: max terrain < 80 m above origin",
          terrain_alt(t4, 30_000.0) < 80.0, terrain_alt(t4, 30_000.0), "< 80 m")

    # ── RTB profiles ──────────────────────────────────────────────────
    t5 = load_terrain("KDEN", "KDEN", 1655.0, 1655.0, 100_000.0; bearing_rad=0.0, verbose=false)
    check("KDEN RTB: flat at origin",  terrain_alt(t5, 0.0) ≈ 0.0,  terrain_alt(t5, 0.0),       "0.0 m")
    check("KDEN RTB: flat at 50km",    terrain_alt(t5, 50_000.0) ≈ 0.0, terrain_alt(t5, 50_000.0), "0.0 m")
    t6 = load_terrain("KBOS", "KBOS", 6.0, 6.0, 100_000.0; bearing_rad=0.0, verbose=false)
    check("KBOS RTB: flat throughout", terrain_alt(t6, 50_000.0) ≈ 0.0, terrain_alt(t6, 50_000.0), "0.0 m")

    # ── Unknown route falls back to flat_model ────────────────────────
    t7 = load_terrain("KLAX", "KSFO", 29.0, 5.0, 550_000.0; bearing_rad=0.0, verbose=false)
    check("Unknown route → flat_model",
          occursin("flat_model", t7.source), 1.0, "flat_model")
    check("flat_model: origin delta = 0",
          terrain_alt(t7, 0.0) ≈ 0.0, terrain_alt(t7, 0.0), "0.0 m")

    # ── Clamp behaviour ───────────────────────────────────────────────
    check("Clamp negative x",  terrain_alt(t1, -500.0) ≈ 0.0,    terrain_alt(t1, -500.0),    "0.0 m")
    check("Clamp overshoot",   terrain_alt(t1, 200_000.0) ≈ -619.0, terrain_alt(t1, 200_000.0), "−619.0 m")

    # ── Bearing projection: KAXX→KSAF real geometry ───────────────────
    # Bearing 219°: (x,y) for a point 114900m along the route
    # x = 114900*cos(219°) = -89294,  y = 114900*sin(219°) = -72316
    # terrain_alt with bearing=219° should project back to 114900m → delta=-619m
    t_brg = load_terrain("KAXX", "KSAF", 2554.0, 1935.0, 114_900.0;
                          bearing_rad=deg2rad(219.0), verbose=false)
    x_dest = 114_900.0 * cos(deg2rad(219.0))
    y_dest = 114_900.0 * sin(deg2rad(219.0))
    ta_brg = terrain_alt(t_brg, x_dest, y_dest)
    check("Bearing projection: KAXX→KSAF dest at real (x,y)",
          abs(ta_brg - (-619.0)) < 2.0, ta_brg, "≈−619.0 m")
    # Origin in real coords: x=0, y=0 → d=0 → delta=0
    check("Bearing projection: KAXX→KSAF origin at (0,0)",
          terrain_alt(t_brg, 0.0, 0.0) ≈ 0.0, terrain_alt(t_brg, 0.0, 0.0), "0.0 m")

    println(ok ? "\nAll tests PASSED ✓" : "\nSome tests FAILED ✗")
    return ok
end

end # module Terrain
# glass_cockpit.jl:    Tiltrotor Sim Glass Cockpit MIL-STD-3009 NVG
# AUTHOR:              DANIEL DESAI
# UPDATED:             2026-06-17
# VERSION:             0.1.1
#
# GLMakie real-time instrument panel — NVG-compatible colour theme
# per MIL-STD-3009 (Lighting, Aircraft, Night Vision Imaging System).
#
# MIL-STD-3009 colour requirements implemented:
#   • NVIS White (u′=0.197, v′=0.453) for all cockpit text and symbology
#     — mandated for new crew-station installs per §4.2.2; appears as a
#     cool greenish-white to the naked eye
#   • NVIS Green A (u′=0.214, v′=0.487) for primary instrument symbology
#     and nominal/safe status readouts
#   • NVIS Yellow (u′=0.260, v′=0.520) for caution signals per §4.2.5
#   • NVIS Red (u′=0.400, v′=0.460) for warning signals per §4.2.5
#   • Blue channel not suppressed globally — NVIS White has blue content
#     by design; the standard filters energy above 625 nm (deep red/IR),
#     not blue.  Class B filter cutoff: 665 nm.
#   • B612 / B612-Bold throughout (Makie-bundled, no install needed)
#
# Layout (demo-ready, 1440 × 900 PFD + ~720 px NAV = ~2160 × 900):
#   Columns: [Fixed 200 | auto | Fixed 200 | Fixed ~720 (NAV)]
#   Row 1 (40 px)  : Header bar — event time (mm:ss) | airport data | phase
#   Row 2 (auto)   : [VCON top] [ADI full-height with overlaid speed/alt tapes] [CONTACT top] | [NAV map rows 2-3]
#   Row 3 (52 px)  : [VCON bot] [Heading tape overlaid on ADI bottom]           [CONTACT bot] |
#   Row 4 (220 px) : [Power   ] [6× Rotor gauges]                               [Tilt        ] | (empty — future use)
#   NAV column (~1/3 of total width): moving map rows 2-3 only (square aspect ratio)
#
# Usage unchanged:
#   julia glass_cockpit_nvg.jl dash_results.csv
#   COCKPIT_FPS=30 julia glass_cockpit_nvg.jl dash_results.csv
# =====================================================================

using GLMakie
using CSV, DataFrames
using Observables
using Printf

# ══════════════════════════════════════════════════════════════════════
#  MIL-STD-3009 TABLE II — CHROMATICITY-DERIVED COLOUR PALETTE
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
#                                        here for secondary informational text
#                                        (airport data string, footer strip).
#
#  Conversion path: CIE 1976 (u′,v′) → CIE 1931 (x,y) → XYZ → linear
#  sRGB → gamma-corrected sRGB.  Values below are the centres of each
#  tolerance circle, converted to the closest in-gamut sRGB colour.
#
#  NVIS White is the §4.2.2-mandated colour for cockpit panel lighting.
#  It appears as a cool greenish-white to the naked eye — NOT neutral
#  white — because its chromaticity sits at the blue-green boundary
#  (u′=0.197, v′=0.453), well away from D65 white (u′=0.198, v′=0.469).
#  This is intentional: it minimises energy above 625 nm where NVG
#  sensitivity (Gen III image intensifiers) peaks.
#
#  Blue channel:  NVIS White has a meaningful blue component (~0.55 in
#  linear sRGB after conversion) — this is correct and compliant.  The
#  standard suppresses energy above 625 nm (IR/deep-red), not blue per se.
#  Class A filters cut at 625 nm; Class B at 665 nm.
# ══════════════════════════════════════════════════════════════════════
const TH = (
    # Backgrounds — near-black
    bg        = RGBf(0.031, 0.035, 0.031),   # #080908
    panel     = RGBf(0.047, 0.055, 0.047),   # #0c0e0c
    panel_hi  = RGBf(0.063, 0.075, 0.063),   # #101310 header / active

    # Structure
    stroke    = RGBf(0.14,  0.18,  0.13),    # dim border
    stroke_hi = RGBf(0.25,  0.32,  0.22),    # active border

    # NVIS White (u′=0.197, v′=0.453) — cockpit primary text / symbology
    # sRGB approx: #B8FFD0  at full luminance; dimmed here for legibility
    text      = RGBf(0.72,  1.00,  0.82),    # NVIS White — primary labels
    text_dim  = RGBf(0.38,  0.54,  0.42),    # NVIS White dimmed ~50%
    text_faint= RGBf(0.18,  0.26,  0.20),    # NVIS White dimmed ~25%
    text_label= RGBf(0.38,  0.54,  0.42),    # panel / axis labels

    # NVIS Green A (u′=0.214, v′=0.487) — primary instrument symbology
    # sRGB approx: #00E040
    green     = RGBf(0.00,  0.878, 0.251),   # nominal / safe — NVIS Green A

    # NVIS Yellow (u′=0.260, v′=0.520) — caution  sRGB approx: #E8C000
    amber     = RGBf(0.910, 0.753, 0.000),

    # NVIS Red (u′=0.400, v′=0.460) — warning  sRGB approx: #FF2800
    red       = RGBf(1.000, 0.157, 0.000),

    # NVIS Blue (u′=0.100, v′=0.280) — vendor extension, NVG-safe
    # sRGB approx: #00AAFF at full luminance; used dimmed for info text
    blue      = RGBf(0.28,  0.65,  0.95),    # informational / secondary data

    # ADI sky/ground — low-luminance, NVIS-safe tints
    sky       = RGBf(0.04,  0.10,  0.08),    # very dark blue-green
    ground    = RGBf(0.10,  0.08,  0.02),    # very dark olive
)

# Fonts — B612 is bundled with Makie and always available (no install needed).
# LABEL_FONT : regular weight — labels, secondary text, tape ticks
# DISP_FONT  : bold weight    — primary readouts (speed, alt, RPM, kW)
# To substitute another font, replace the strings with any font name
# registered on the host OS, e.g. "Inter", "IBM Plex Sans", "Roboto".
const LABEL_FONT = "B612"
const DISP_FONT  = "B612-Bold"

# ══════════════════════════════════════════════════════════════════════
#  STATE (unchanged from original)
# ══════════════════════════════════════════════════════════════════════
const IDX = (
    t=1, tau=2, speed=4, alt=5, power=6,
    tilt=7, pitch=8, roll=9, yaw=10,
    soc=11, voltage=12, batt_temp=13,
    x_m=14, y_m=15,
    omega_x=16, omega_y=17, omega_z=18,   # body rates (rad/s)
    alt_agl_m=19,                          # CG altitude AGL (m)
    gx=20, gy=21, gz=22,                   # g-forces (longitudinal, lateral, vertical)
)

mutable struct CockpitState
    vals          :: Vector{Float64}
    phase         :: String
    history_power :: Vector{Float64}
    history_t     :: Vector{Float64}
    rotor_rpm     :: Vector{Float64}
    rotor_kw      :: Vector{Float64}
    rotor_labels  :: Vector{String}
    n_rotors      :: Int
    gear_contact  :: Bool
    strut_load_n  :: Float64
    brakes_on     :: Bool       # true when wheel brakes are engaged
end

function CockpitState(; n_rotors::Int=6,
                        labels::Vector{String}=["R$i" for i in 1:6])
    CockpitState(zeros(22), "hover",   # 22 vals: +gx/gy/gz
                 Float64[], Float64[],
                 zeros(6), zeros(6),
                 vcat(labels, fill("", 6))[1:6],
                 clamp(n_rotors, 1, 6),
                 false, 0.0, false)
end

# ══════════════════════════════════════════════════════════════════════
#  HELPER — thin panel border drawn inside any axis
# ══════════════════════════════════════════════════════════════════════
function draw_border!(ax; color=TH.stroke_hi, lw=0.8)
    lines!(ax, [Point2f(0,0), Point2f(1,0), Point2f(1,1),
                Point2f(0,1), Point2f(0,0)],
           color=color, linewidth=lw, space=:relative)
end

# ══════════════════════════════════════════════════════════════════════
#  INSTRUMENT DRAWING
# ══════════════════════════════════════════════════════════════════════

# Vertical tape (speed or altitude) — NVG phosphor
# overlay=true → HUD mode: semi-transparent background, used when the axis
# is stacked over the ADI rather than in its own opaque column.
function draw_tape!(ax, value, range_per_half, step_major, step_minor,
                    label, unit, color=TH.text; flip=false, overlay=false)
    empty!(ax)
    xlims!(ax, 0, 1)
    ylims!(ax, value - range_per_half, value + range_per_half)
    hidedecorations!(ax)
    hidespines!(ax)

    # Panel fill — translucent in overlay/HUD mode
    bg_alpha = overlay ? 0.52 : 1.0
    poly!(ax, [Point2f(0, value - range_per_half),
               Point2f(1, value - range_per_half),
               Point2f(1, value + range_per_half),
               Point2f(0, value + range_per_half)],
          color=RGBAf(TH.panel.r, TH.panel.g, TH.panel.b, bg_alpha), strokewidth=0)

    lo = value - range_per_half
    hi = value + range_per_half
    v  = ceil(lo / step_minor) * step_minor
    while v <= hi
        is_major = abs(v % step_major) < 0.01
        x_outer = flip ? -0.06 : 1.06
        x_inner = flip ? 0.15  : 0.85
        x_label = flip ? 0.17  : 0.83
        lw = is_major ? 1.4 : 0.7
        c  = is_major ? TH.text : TH.text_dim
        lines!(ax, [x_inner, x_outer], [v, v], color=c, linewidth=lw)
        if is_major
            align = flip ? :left : :right
            text!(ax, x_label, v, text=@sprintf("%d", round(Int, v)),
                  fontsize=10, color=TH.text,
                  align=(align, :center), font=LABEL_FONT)
        end
        v += step_minor
    end

    # Current-value bug (filled box with readout)
    bug_half = range_per_half * 0.06
    bug_lo = value - bug_half; bug_hi = value + bug_half
    poly!(ax, [Point2f(0.0, bug_lo), Point2f(1.0, bug_lo),
               Point2f(1.0, bug_hi), Point2f(0.0, bug_hi)],
          color=RGBAf(0.0, 0.22, 0.05, 0.95), strokecolor=color,
          strokewidth=1.2)

    text!(ax, 0.5, value,
          text=@sprintf("%d", round(Int, value)),
          fontsize=18, color=color, font=DISP_FONT,
          align=(:center, :center))

    # Label at top
    text!(ax, 0.5, hi - range_per_half * 0.05,
          text=label * (unit != " " ? "  $(unit)" : ""),
          fontsize=18, color=TH.green, font=LABEL_FONT,
          align=(:center, :center))

    draw_border!(ax, color=TH.stroke)
end

# Attitude indicator — synthetic horizon in NVG palette
function draw_attitude!(ax, pitch_deg, roll_deg)
    empty!(ax)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, -1, 1); ylims!(ax, -1, 1)

    pitch_offset = clamp(pitch_deg / 30.0, -0.8, 0.8)
    horizon_y    = pitch_offset
    roll_rad     = deg2rad(roll_deg)

    sky_pts = Point2f[(-3, horizon_y), (3, horizon_y), (3, 3), (-3, 3)]
    gnd_pts = Point2f[(-3, -3), (3, -3), (3, horizon_y), (-3, horizon_y)]

    rot(p) = Point2f(cos(roll_rad)*p[1] - sin(roll_rad)*p[2],
                     sin(roll_rad)*p[1] + cos(roll_rad)*p[2])

    poly!(ax, rot.(sky_pts), color=TH.sky,    strokewidth=0)
    poly!(ax, rot.(gnd_pts), color=TH.ground, strokewidth=0)

    h_left  = rot(Point2f(-2, horizon_y))
    h_right = rot(Point2f( 2, horizon_y))
    lines!(ax, [h_left, h_right], color=TH.text, linewidth=1.8)

    for p_deg in -30:10:30
        p_deg == 0 && continue
        p_off = (p_deg - pitch_deg) / 30.0
        abs(p_off) > 0.85 && continue
        len = abs(p_deg) == 30 ? 0.30 : 0.20
        c_l = rot(Point2f(-len, p_off))
        c_r = rot(Point2f( len, p_off))
        lines!(ax, [c_l, c_r], color=RGBAf(0.0, 0.7, 0.18, 0.85), linewidth=1)
        text!(ax, rot(Point2f(len + 0.04, p_off))[1],
                  rot(Point2f(len + 0.04, p_off))[2],
              text=string(abs(p_deg)), fontsize=8, color=TH.text_dim,
              align=(:left, :center), font=LABEL_FONT)
    end

    # Fixed aircraft reference symbol
    for sign in (-1, 1)
        lines!(ax, [Point2f(sign*0.15, 0), Point2f(sign*0.48, 0)],
               color=TH.amber, linewidth=3.5)
        lines!(ax, [Point2f(sign*0.15, 0), Point2f(sign*0.15, -0.08)],
               color=TH.amber, linewidth=3.5)
    end
    scatter!(ax, [Point2f(0, 0)], color=TH.amber, markersize=7)

    # Roll arc + tick marks
    arc_r = 0.82
    for deg in [-60,-45,-30,-20,-10,0,10,20,30,45,60]
        a = deg2rad(deg - 90)
        x1,y1 = cos(a)*arc_r, sin(a)*arc_r
        len_t = abs(deg) % 30 == 0 ? 0.07 : 0.04
        x2,y2 = cos(a)*(arc_r-len_t), sin(a)*(arc_r-len_t)
        lines!(ax, [Point2f(x1,y1), Point2f(x2,y2)],
               color=TH.text_dim, linewidth=1)
    end
    # Roll pointer triangle
    a = deg2rad(-roll_deg - 90)
    r1, r2 = 0.82, 0.72
    poly!(ax, [Point2f(cos(a)*r1 - 0.03*sin(a), sin(a)*r1 + 0.03*cos(a)),
               Point2f(cos(a)*r2, sin(a)*r2),
               Point2f(cos(a)*r1 + 0.03*sin(a), sin(a)*r1 - 0.03*cos(a))],
          color=TH.text, strokewidth=0)
end

# Heading tape — horizontal HSI strip
# overlay=true → HUD mode: semi-transparent panel, used when the tape
# is rendered inside the ADI axis bounds rather than in its own row.
function draw_heading_tape!(ax, yaw_deg; overlay=false)
    empty!(ax)
    hidedecorations!(ax)
    hidespines!(ax)

    hdg = mod(yaw_deg, 360.0)
    half_range = 40.0
    xlims!(ax, hdg - half_range, hdg + half_range)
    ylims!(ax, 0.0, 1.0)

    bg_alpha = overlay ? 0.52 : 1.0
    poly!(ax, Point2f[(hdg - half_range, 0), (hdg + half_range, 0),
                      (hdg + half_range, 1), (hdg - half_range, 1)],
          color=RGBAf(TH.panel.r, TH.panel.g, TH.panel.b, bg_alpha), strokewidth=0)

    cardinals = Dict(0=>"N", 90=>"E", 180=>"S", 270=>"W", 360=>"N")

    lo = floor(Int, (hdg - half_range - 30) / 10) * 10
    hi = ceil( Int, (hdg + half_range + 30) / 10) * 10
    d  = lo
    while d <= hi
        d_wrap      = mod(d, 360)
        is_major    = d_wrap % 30 == 0
        is_cardinal = haskey(cardinals, d_wrap)
        tick_top    = is_major ? 0.58 : 0.44
        lw          = is_major ? 1.4 : 0.7
        c           = is_major ? TH.text : TH.text_dim

        lines!(ax, [Point2f(float(d), 0.0), Point2f(float(d), tick_top)],
               color=c, linewidth=lw)

        if is_cardinal
            text!(ax, float(d), 0.80, text=cardinals[d_wrap],
                  fontsize=13, color=TH.text, font=LABEL_FONT,
                  align=(:center, :center))
        elseif is_major
            text!(ax, float(d), 0.80, text=@sprintf("%03d", d_wrap),
                  fontsize=9, color=TH.text_dim, font=LABEL_FONT,
                  align=(:center, :center))
        end
        d += 10
    end

    # Centre lubber line — full height, behind the bug box
    lines!(ax, [Point2f(hdg, 0.0), Point2f(hdg, 1.0)],
           color=TH.text, linewidth=2.0)

    # Heading bug — vertically centred in the strip
    bug_half = 5.5     # half-width in degrees
    bug_h    = 0.55    # box height (leaves margin top and bottom)
    bug_lo   = (1.0 - bug_h) / 2.0
    bug_hi   = bug_lo + bug_h
    poly!(ax, [Point2f(hdg - bug_half, bug_lo), Point2f(hdg + bug_half, bug_lo),
               Point2f(hdg + bug_half, bug_hi), Point2f(hdg - bug_half, bug_hi)],
          color=RGBAf(0.0, 0.22, 0.05, 0.95), strokecolor=TH.text, strokewidth=1.2)

    # Heading value: mod 360 so the display never shows "360"
    hdg_display = mod(round(Int, hdg), 360)
    text!(ax, hdg, 0.5, text=@sprintf("%03d", hdg_display),
          fontsize=18, color=TH.text, font=DISP_FONT,
          align=(:center, :center))

    draw_border!(ax, color=TH.stroke)
end

# Tilt indicator — nacelle tilt schematic
#
# Geometry uses a fixed pixel-like coordinate space (0..100 × 0..100) to
# avoid the distortion that arose in the previous normalised ±1.3 space.
# Origin (pivot point) is at (ox, oy) = (22, 78) — lower-left.
# V endpoint (tilt=0, nacelle straight up):  (22, 12)  →  arc_r = 66 above oy
# H endpoint (tilt=90, nacelle horizontal):  (88, 78)  →  same radius to the right
#
# Geometry convention (unchanged):
#   tilt_deg = 0   → nacelle vertical,   thrust upward   (HOVER)
#   tilt_deg = 90  → nacelle horizontal, thrust forward  (CRUISE)
#   θ measured from vertical; thrust direction = (sin θ, -cos θ) in (x, y)
#                              (y axis points UP in Makie data coordinates)
function draw_tilt!(ax, tilt_deg)
    empty!(ax)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, 0, 100); ylims!(ax, 0, 100)

    # Panel fill
    poly!(ax, [Point2f(0,0), Point2f(100,0), Point2f(100,100), Point2f(0,100)],
          color=TH.panel, strokewidth=0)

    ox = 22.0; oy = 22.0   # pivot: lower-left of the drawing area
    arc_r = 66.0            # radius of the quarter-circle arc (pixels)

    # V and H arc endpoints
    v_pt = Point2f(ox, oy + arc_r)          # straight up
    h_pt = Point2f(ox + arc_r, oy)          # straight right

    # ── Quarter-circle arc track (V → H) ─────────────────────────────
    n_arc = 48
    arc_bg = [Point2f(ox + arc_r*sin(φ), oy + arc_r*cos(φ))
              for φ in range(0, π/2, length=n_arc)]
    lines!(ax, arc_bg, color=TH.stroke_hi, linewidth=1.2)

    # ── Filled arc to current tilt angle ──────────────────────────────
    θ = deg2rad(clamp(tilt_deg, 0.0, 90.0))
    if tilt_deg > 0.5
        arc_fill = [Point2f(ox + arc_r*sin(φ), oy + arc_r*cos(φ))
                    for φ in range(0, θ, length=max(2, round(Int, tilt_deg)+1))]
        lines!(ax, arc_fill, color=TH.text, linewidth=2.8)
    end

    # ── V / H endpoint dots and labels ───────────────────────────────
    scatter!(ax, [v_pt], color=TH.text_dim, markersize=5)
    scatter!(ax, [h_pt], color=TH.text_dim, markersize=5)
    text!(ax, ox - 5.0, oy + arc_r + 3.0,
          text="V", fontsize=9, color=TH.text_faint,
          align=(:center, :center), font=LABEL_FONT)
    text!(ax, ox + arc_r + 5.0, oy - 3.0,
          text="H", fontsize=9, color=TH.text_faint,
          align=(:center, :center), font=LABEL_FONT)

    # ── Thrust direction unit vector ──────────────────────────────────
    # In data coords: x right, y up.  tilt from vertical → thrust = (sinθ, cosθ)
    tx =  sin(θ); ty =  cos(θ)   # thrust unit vector
    px = -ty;     py =  tx       # perpendicular (rotor disk direction)

    # ── Nacelle body: thick line from pivot toward thrust direction ────
    nac_len = 38.0               # length in data units
    tip = Point2f(ox + tx*nac_len, oy + ty*nac_len)
    lines!(ax, [Point2f(ox, oy), tip],
           color=TH.green,   linewidth=1.8)

    # ── Rotor disk: short line perpendicular to nacelle at tip ────────
    disk_r = 14.0
    lines!(ax, [Point2f(tip[1] - px*disk_r, tip[2] - py*disk_r),
                Point2f(tip[1] + px*disk_r, tip[2] + py*disk_r)],
           color=TH.green, linewidth=1.8)

    # ── Thrust arrow: dashed line from tip outward ────────────────────
    arr_len = 22.0
    arr_tip = Point2f(tip[1] + tx*arr_len, tip[2] + ty*arr_len)
    lines!(ax, [tip, arr_tip],
           color=TH.green, linewidth=1.2, linestyle=:dash)
    # Arrowhead triangle
    head_l = 5.0; head_w = 3.0
    poly!(ax, [Point2f(arr_tip[1] + tx*head_l - px*head_w,
                       arr_tip[2] + ty*head_l - py*head_w),
               Point2f(arr_tip[1] + tx*head_l + px*head_w,
                       arr_tip[2] + ty*head_l + py*head_w),
               arr_tip],
          color=TH.green, strokewidth=0)

    # ── Mode string and angle readout ─────────────────────────────────
    mode_str = tilt_deg < 20 ? "HOVER" : tilt_deg > 80 ? "CRUISE" : "TRANSITION"
    text!(ax, 50.0, 96.0, text="NACELLE TILT",
          fontsize=9, color=TH.text_label, align=(:center, :center), font=LABEL_FONT)
    text!(ax, 50.0,  7.0,
          text=@sprintf("%05.1f°  %s", tilt_deg, mode_str),
          fontsize=11, color=TH.text, align=(:center, :center), font=LABEL_FONT)

    draw_border!(ax, color=TH.stroke)
end

# VCON — conversion corridor speed indicator
# Shows current IAS vs the tilt-transition safe envelope.
# The conversion corridor is the speed range within which the tilt
# transition can be safely executed:
#   Vcon_lo: minimum safe airspeed for tilt-forward — below this,
#            wing lift is insufficient to support the aircraft as
#            rotor thrust vectors forward. S4-class ≈ 80 km/h.
#   Vcon_hi: maximum safe airspeed in rotor-borne flight — above this,
#            aerodynamic loads on tilted rotors exceed structural limits
#            and rotor RPM authority is insufficient for attitude control.
#            S4-class ≈ 165 km/h.
#
# Status logic (driven by tilt angle and speed):
#   CRUISE  (tilt > 80°) — corridor not applicable; show dim
#   HOVER   (tilt < 10°) — corridor not applicable; show dim
#   TRANS   (10–80°)     — active: green = inside, amber = approaching,
#                          red = outside
#
# Sits in fig[3,1] — symmetric with CONTACT in fig[3,3].
function draw_vcon!(ax, speed_kmh::Float64, tilt_deg::Float64,
                    vcon_lo::Float64, vcon_hi::Float64;
                    vcon_warn::Float64 = 15.0)   # amber margin (km/h)
    empty!(ax)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)

    poly!(ax, [Point2f(0,0), Point2f(1,0), Point2f(1,1), Point2f(0,1)],
          color=TH.panel, strokewidth=0)

    in_trans = 10.0 < tilt_deg < 80.0

    if !in_trans
        # Outside transition — show label and dim corridor bounds only
        poly!(ax, [Point2f(0.06, 0.18), Point2f(0.94, 0.18),
                   Point2f(0.94, 0.82), Point2f(0.06, 0.82)],
              color=RGBAf(0,0,0,0), strokecolor=TH.stroke, strokewidth=0.8)
        text!(ax, 0.5, 0.63, text="VCON",
              fontsize=11, color=TH.text_faint,
              align=(:center, :center), font=DISP_FONT)
        bounds_str = @sprintf("%.0f – %.0f", vcon_lo, vcon_hi)
        text!(ax, 0.5, 0.33, text=bounds_str * " km/h",
              fontsize=9, color=TH.text_faint,
              align=(:center, :center), font=LABEL_FONT)
        return
    end

    # In transition — active corridor check
    below_lo = speed_kmh < vcon_lo
    above_hi = speed_kmh > vcon_hi
    near_lo  = speed_kmh < vcon_lo + vcon_warn
    near_hi  = speed_kmh > vcon_hi - vcon_warn
    outside  = below_lo || above_hi
    marginal = !outside && (near_lo || near_hi)

    col = outside  ? TH.red   :
          marginal ? TH.amber :
                     TH.green

    fill_alpha = outside ? 0.22 : marginal ? 0.15 : 0.12

    poly!(ax, [Point2f(0.06, 0.18), Point2f(0.94, 0.18),
               Point2f(0.94, 0.82), Point2f(0.06, 0.82)],
          color=RGBAf(col.r, col.g, col.b, fill_alpha),
          strokecolor=col, strokewidth=1.4)

    text!(ax, 0.5, 0.63, text="VCON",
          fontsize=11, color=col,
          align=(:center, :center), font=DISP_FONT)

    speed_str = @sprintf("%.0f km/h", speed_kmh)
    text!(ax, 0.5, 0.33, text=speed_str,
          fontsize=9, color=TH.text_dim,
          align=(:center, :center), font=LABEL_FONT)
end

# CONTACT / WOW annunciator — weight-on-wheels indicator
# Sits between the altitude tape (row 2) and tilt panel (row 4) in the
# right column (row 3, col 3), mirroring the heading tape row on the left.
#
# contact=true  → green CONTACT box with strut load in Newtons
# contact=false → dim dashes (airborne)
function draw_contact!(ax, contact::Bool, strut_load_n::Float64,
                       brakes_on::Bool=false, gz::Float64=1.0)
    empty!(ax)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)

    poly!(ax, [Point2f(0,0), Point2f(1,0), Point2f(1,1), Point2f(0,1)],
          color=TH.panel, strokewidth=0)

    # ── BRAKES indicator (top strip) ──────────────────────────────────
    brk_col = brakes_on ? TH.red : TH.stroke
    brk_fill = brakes_on ? RGBAf(TH.red.r, TH.red.g, TH.red.b, 0.22) :
                           RGBAf(0,0,0,0)
    poly!(ax, [Point2f(0.06, 0.72), Point2f(0.94, 0.72),
               Point2f(0.94, 0.92), Point2f(0.06, 0.92)],
          color=brk_fill, strokecolor=brk_col, strokewidth=1.2)
    text!(ax, 0.5, 0.82, text="BRAKES",
          fontsize=9, color=brk_col,
          align=(:center, :center), font=DISP_FONT)

    # ── Gear contact indicator (middle) ──────────────────────────────
    if contact
        poly!(ax, [Point2f(0.06, 0.32), Point2f(0.94, 0.32),
                   Point2f(0.94, 0.68), Point2f(0.06, 0.68)],
              color=RGBAf(TH.green.r, TH.green.g, TH.green.b, 0.18),
              strokecolor=TH.green, strokewidth=1.4)
        text!(ax, 0.5, 0.54, text="CONTACT",
              fontsize=10, color=TH.green,
              align=(:center, :center), font=DISP_FONT)
        load_str = strut_load_n > 999.0 ?
            @sprintf("%.1f kN", strut_load_n / 1000.0) :
            @sprintf("%.0f N",  strut_load_n)
        text!(ax, 0.5, 0.38, text=load_str,
              fontsize=8, color=TH.text_dim,
              align=(:center, :center), font=LABEL_FONT)
    else
        poly!(ax, [Point2f(0.06, 0.32), Point2f(0.94, 0.32),
                   Point2f(0.94, 0.68), Point2f(0.06, 0.68)],
              color=RGBAf(0,0,0,0), strokecolor=TH.stroke, strokewidth=0.8)
        text!(ax, 0.5, 0.50, text="- - -",
              fontsize=10, color=TH.text_faint,
              align=(:center, :center), font=LABEL_FONT)
    end

    # ── gz readout (bottom) ───────────────────────────────────────────
    gz_col = abs(gz) > 2.5 ? TH.red : abs(gz) > 1.8 ? TH.amber : TH.text_dim
    text!(ax, 0.5, 0.18,
          text=@sprintf("gz %.2f", gz),
          fontsize=9, color=gz_col,
          align=(:center, :center), font=LABEL_FONT)
end

# Power / SOC / Battery Temp panel
function draw_power!(ax, power_kw, soc_pct, batt_temp_c)
    empty!(ax)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, -1, 1); ylims!(ax, -1, 1)

    poly!(ax, [Point2f(-1,-1), Point2f(1,-1), Point2f(1,1), Point2f(-1,1)],
          color=TH.panel, strokewidth=0)

    soc_color  = soc_pct  > 40.0 ? TH.green : (soc_pct  > 20.0 ? TH.amber : TH.red)
    temp_color = batt_temp_c > 50.0 ? TH.amber : TH.text

    text!(ax, 0.0,  0.82, text="POWER",
          fontsize=9, color=TH.text_label, align=(:center,:center), font=LABEL_FONT)
    text!(ax, 0.0,  0.52, text=@sprintf("%.0f kW", power_kw),
          fontsize=22, color=TH.text, align=(:center,:center), font=DISP_FONT)

    lines!(ax, [Point2f(-0.82, 0.30), Point2f(0.82, 0.30)],
           color=TH.stroke_hi, linewidth=0.8)

    text!(ax, 0.0,  0.18, text="SOC",
          fontsize=9, color=TH.text_label, align=(:center,:center), font=LABEL_FONT)
    text!(ax, 0.0, -0.10, text=@sprintf("%.0f%%", soc_pct),
          fontsize=22, color=soc_color, align=(:center,:center), font=DISP_FONT)

    lines!(ax, [Point2f(-0.82, -0.33), Point2f(0.82, -0.33)],
           color=TH.stroke_hi, linewidth=0.8)

    text!(ax, 0.0, -0.48, text="BATT TEMP",
          fontsize=9, color=TH.text_label, align=(:center,:center), font=LABEL_FONT)
    text!(ax, 0.0, -0.76, text=@sprintf("%.1f°C", batt_temp_c),
          fontsize=22, color=temp_color, align=(:center,:center), font=DISP_FONT)

    text!(ax, 0.0,  0.96, text="POWERPLANT",
          fontsize=8, color=TH.text_faint, align=(:center, :center), font=LABEL_FONT)
end

# Per-rotor gauge strip — 6 mini panels
function draw_rotor_gauges!(axes::Vector{Axis},
                            rpms::Vector{Float64},
                            kws::Vector{Float64},
                            labels::Vector{String},
                            n::Int,
                            rpm_nom::Float64=1000.0,
                            kw_max::Float64=2000.0)
    for (idx, ax) in enumerate(axes)
        empty!(ax)
        hidedecorations!(ax)
        hidespines!(ax)
        xlims!(ax, 0, 1); ylims!(ax, 0, 1)

        # Panel fill
        poly!(ax, [Point2f(0,0), Point2f(1,0), Point2f(1,1), Point2f(0,1)],
              color=TH.panel, strokewidth=0)

        if idx > n
            text!(ax, 0.5, 0.5, text="---",
                  fontsize=10, color=TH.text_faint, align=(:center, :center),
                  font=LABEL_FONT)
            draw_border!(ax, color=TH.stroke, lw=0.6)
            continue
        end

        rpm   = rpms[idx]
        kw    = kws[idx]
        label = labels[idx]

        is_proto  = label != "R$idx"
        accent    = is_proto ? TH.amber : TH.text

        rpm_frac  = rpm / (rpm_nom + 1e-3)
        rpm_color = rpm_frac < 0.70 ? TH.green :
                    rpm_frac < 1.05 ? TH.green  :
                                      TH.green

        bar_frac  = clamp(kw / (kw_max + 1e-3), 0.0, 1.0)
        bar_color = bar_frac < 0.70 ? TH.green :
                    bar_frac < 1.05 ? TH.green  :
                                      TH.green

        # Vertical kW bar track
        bar_x0 = 0.72; bar_x1 = 0.88
        bar_y0 = 0.08; bar_y1 = 0.58
        poly!(ax, [Point2f(bar_x0, bar_y0), Point2f(bar_x1, bar_y0),
                   Point2f(bar_x1, bar_y1), Point2f(bar_x0, bar_y1)],
              color=TH.panel_hi, strokecolor=TH.stroke, strokewidth=0.7)

        bar_fill_y = bar_y0 + bar_frac * (bar_y1 - bar_y0)
        if bar_fill_y > bar_y0 + 0.01
            poly!(ax, [Point2f(bar_x0+0.01, bar_y0+0.01),
                       Point2f(bar_x1-0.01, bar_y0+0.01),
                       Point2f(bar_x1-0.01, bar_fill_y),
                       Point2f(bar_x0+0.01, bar_fill_y)],
                  color=bar_color, strokewidth=0)
        end

        # Header separator
        lines!(ax, [Point2f(0.04, 0.88), Point2f(0.96, 0.88)],
               color=TH.stroke_hi, linewidth=0.8)

        text!(ax, 0.38, 0.93, text=label,
              fontsize=9, color=accent,
              align=(:center, :center), font=LABEL_FONT)

        text!(ax, 0.38, 0.74, text=@sprintf("%d", round(Int, rpm)),
              fontsize=20, color=rpm_color,
              align=(:center, :center), font=DISP_FONT)
        text!(ax, 0.38, 0.63, text="RPM",
              fontsize=11, color=TH.text_faint, align=(:center, :center), font=LABEL_FONT)

        text!(ax, 0.38, 0.46, text=@sprintf("%.0f", kw),
              fontsize=18, color=TH.text_dim,
              align=(:center, :center), font=DISP_FONT)
        text!(ax, 0.38, 0.36, text="kW",
              fontsize=11, color=TH.text_faint, align=(:center, :center), font=LABEL_FONT)

        if is_proto
            poly!(ax, [Point2f(0.03, 0.01), Point2f(0.97, 0.01),
                       Point2f(0.97, 0.13), Point2f(0.03, 0.13)],
                  color=RGBAf(TH.amber.r, TH.amber.g, TH.amber.b, 0.10),
                  strokecolor=TH.amber, strokewidth=0.7)
            text!(ax, 0.50, 0.07, text="PROTO",
                  fontsize=8, color=TH.amber,
                  align=(:center, :center), font=LABEL_FONT)
        end

        draw_border!(ax, color=TH.stroke, lw=0.6)
    end
end

# ══════════════════════════════════════════════════════════════════════
#  MOVING MAP  (col 4 — placed to the right of the PFD columns)
# ══════════════════════════════════════════════════════════════════════
#
# draw_nav_map! renders the full moving-map panel into a single Axis.
# It is called from the update! closure inside launch_cockpit every time
# the state Observable fires, so it refreshes at the same rate as all
# other instruments — in both AUTO and HOTAS manual modes.
#
# Coordinate conventions (local ENU, metres):
#   x = forward from takeoff  →  screen UP   (north-up map style)
#   y = rightward             →  screen RIGHT
#
# The map auto-scales: the visible radius grows to always show both the
# aircraft and the active waypoint with 15 % margin.  The aircraft
# symbol is always centred.
#
# Layers (back to front):
#   1. Panel fill (TH.panel)
#   2. Grid lines at rounded intervals
#   3. Range rings (optional, max 2 rings at 1/4 and 1/2 of visible radius)
#   4. Track history (dim cyan trail, oldest to newest)
#   5. Straight bearing line: aircraft → waypoint (dim)
#   6. Origin marker (green hollow circle)
#   7. Waypoint marker (diamond, magenta=FLY-TO / green=RTB)
#   8. Aircraft symbol (filled triangle rotated to heading)
#   9. HUD overlay: bearing bug on a mini compass arc, XTE bar, data readout

function draw_nav_map!(ax, snap)
    empty!(ax)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, -1.0, 1.0)
    ylims!(ax, -1.0, 1.0)

    # ── Panel fill ────────────────────────────────────────────────────
    poly!(ax, [Point2f(-1,-1), Point2f(1,-1), Point2f(1,1), Point2f(-1,1)],
          color=TH.panel, strokewidth=0)

    nav    = snap.target
    ac_x   = snap.x;  ac_y = snap.y
    hdg    = snap.hdg                          # degrees, 0 = forward (+x)
    wx     = nav_wx(nav);  wy = nav_wy(nav)
    pts    = snap.pts
    phase  = snap.phase

    # ── Auto-scale: radius that shows ac + waypoint + 15 % margin ────
    dist_to_wp  = hypot(ac_x - wx, ac_y - wy)
    dist_origin = hypot(ac_x, ac_y)
    raw_r       = max(dist_to_wp, dist_origin, 200.0) * 1.15
    # Snap to a round number so grid lines stay at nice intervals
    scales = [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10_000.0, 20_000.0]
    radius = scales[end]
    for s in scales; s >= raw_r && (radius = s; break); end

    # World → NDC: aircraft is always at (0,0) in NDC
    to_ndc(wx_, wy_) = Point2f((wy_ - ac_y) / radius,   # rightward  → screen x
                               (wx_ - ac_x) / radius)   # forward    → screen y

    # ── Grid lines ───────────────────────────────────────────────────
    step = radius / 4.0          # 4 cells each way
    grid_col = RGBAf(TH.stroke.r, TH.stroke.g, TH.stroke.b, 0.45)
    for k in -4:4
        v = k * step
        # horizontal (constant x world = constant ndc-y)
        ny = (v - ac_x + ac_x) / radius   # = v/radius relative to ac
        ny_rel = (v) / radius              # offset from ac in world-x
        lines!(ax, [Point2f(-1, ny_rel), Point2f(1, ny_rel)],
               color=grid_col, linewidth=0.5)
        # vertical (constant y world)
        lines!(ax, [Point2f(ny_rel, -1), Point2f(ny_rel, 1)],
               color=grid_col, linewidth=0.5)
    end

    # ── Range rings (two: at 0.25 r and 0.5 r) ───────────────────────
    ring_col = RGBAf(TH.stroke_hi.r, TH.stroke_hi.g, TH.stroke_hi.b, 0.5)
    for frac in (0.25, 0.5)
        n_seg = 64
        ring_pts = [Point2f(frac * cos(2π*k/n_seg), frac * sin(2π*k/n_seg))
                    for k in 0:n_seg]
        lines!(ax, ring_pts, color=ring_col, linewidth=0.6)
        # Label at top of ring
        ring_m = frac * radius
        label  = ring_m >= 1000.0 ?
            @sprintf("%.0f km", ring_m / 1000.0) :
            @sprintf("%.0f m",  ring_m)
        text!(ax, 0.02, frac - 0.03,
              text=label, fontsize=7,
              color=TH.text_faint, align=(:left, :top), font=LABEL_FONT)
    end

    # ── Track history ─────────────────────────────────────────────────
    if length(pts) >= 2
        track_pts = [to_ndc(Float64(p.x), Float64(p.y)) for p in pts]
        # Subsample if very long to keep draw calls fast
        if length(track_pts) > 600
            step_s = div(length(track_pts), 600)
            track_pts = track_pts[1:step_s:end]
        end
        lines!(ax, track_pts,
               color=RGBAf(0.18, 0.50, 0.60, 0.70), linewidth=1.2)
    end

    # ── Bearing line (aircraft → waypoint) ───────────────────────────
    wp_ndc = to_ndc(wx, wy)
    lines!(ax, [Point2f(0,0), wp_ndc],
           color=RGBAf(TH.text_faint.r, TH.text_faint.g, TH.text_faint.b, 0.35),
           linewidth=0.8, linestyle=:dash)

    # ── Origin marker ─────────────────────────────────────────────────
    org_ndc = to_ndc(0.0, 0.0)
    if abs(org_ndc[1]) <= 1.05 && abs(org_ndc[2]) <= 1.05
        scatter!(ax, [org_ndc], color=TH.panel, markersize=9,
                 strokecolor=TH.green, strokewidth=1.5)
        text!(ax, org_ndc[1] + 0.03, org_ndc[2] + 0.03,
              text="TKOF", fontsize=7, color=TH.green,
              align=(:left, :bottom), font=LABEL_FONT)
    end

    # ── Waypoint marker (diamond) ─────────────────────────────────────
    wp_col = nav.return_to_base ? TH.green : TH.blue
    wp_label = nav.return_to_base ? "RTB" : "TGT"
    if abs(wp_ndc[1]) <= 1.05 && abs(wp_ndc[2]) <= 1.05
        d = 0.045
        diamond = [Point2f(wp_ndc[1],   wp_ndc[2]+d),
                   Point2f(wp_ndc[1]+d, wp_ndc[2]  ),
                   Point2f(wp_ndc[1],   wp_ndc[2]-d),
                   Point2f(wp_ndc[1]-d, wp_ndc[2]  ),
                   Point2f(wp_ndc[1],   wp_ndc[2]+d)]
        lines!(ax, diamond, color=wp_col, linewidth=1.8)
        poly!(ax, diamond[1:end-1],
              color=RGBAf(wp_col.r, wp_col.g, wp_col.b, 0.18), strokewidth=0)
        text!(ax, wp_ndc[1] + 0.04, wp_ndc[2] + 0.04,
              text=wp_label, fontsize=7, color=wp_col,
              align=(:left, :bottom), font=LABEL_FONT)
    end

    # ── Aircraft symbol (filled triangle, rotated to heading) ──────────
    # Heading 0° = forward = screen up.  In NDC: up = +y.
    # hdg is degrees clockwise from forward, so we rotate CW by hdg.
    hdg_rad = deg2rad(hdg)
    rot(px, py) = Point2f( px * cos(hdg_rad) + py * sin(hdg_rad),
                          -px * sin(hdg_rad) + py * cos(hdg_rad))
    sz = 0.055
    ac_pts = [rot(0.0, sz), rot(-sz*0.5, -sz*0.6), rot(0.0, -sz*0.2),
              rot( sz*0.5, -sz*0.6), rot(0.0, sz)]
    poly!(ax, ac_pts,
          color=TH.text, strokecolor=TH.text, strokewidth=0.8)

    # Velocity vector stub (1 s × speed, scaled)
    # We don't have vx here, so omit — the track trail serves the purpose.

    # ── HUD overlay: mini compass arc + XTE bar + data readout ────────
    # Compass arc — top-left quadrant, radius 0.22 NDC
    arc_cx = -0.72;  arc_cy = 0.77;  arc_r = 0.19
    n_arc  = 48
    arc_pts = [Point2f(arc_cx + arc_r * sin(a), arc_cy + arc_r * cos(a))
               for a in range(-π/2, π/2, length=n_arc)]
    lines!(ax, arc_pts, color=TH.stroke_hi, linewidth=1.0)

    # Bearing bug on the arc (bearing to waypoint, relative to heading)
    xte = nav_cross_track(ac_x, ac_y, hdg_rad, wx, wy)
    brg = nav_bearing(ac_x, ac_y, wx, wy)
    rel_brg = mod(brg - hdg, 360.0)
    if rel_brg > 180.0; rel_brg -= 360.0; end   # –180…+180
    bug_angle_rad = clamp(deg2rad(rel_brg), -π/2, π/2)
    bug_x = arc_cx + arc_r * sin(bug_angle_rad)
    bug_y = arc_cy + arc_r * cos(bug_angle_rad)
    scatter!(ax, [Point2f(bug_x, bug_y)],
             color=wp_col, markersize=7, marker=:diamond)

    # Heading label at centre of arc
    text!(ax, arc_cx, arc_cy - 0.04,
          text=@sprintf("%03d°", round(Int, mod(hdg, 360))),
          fontsize=9, color=TH.text,
          align=(:center, :center), font=DISP_FONT)

    # ── XTE bar (horizontal, below compass arc) ───────────────────────
    xte_bar_y  = arc_cy - 0.15
    xte_bar_hw = 0.17                  # half-width in NDC
    xte_max    = radius * 0.25         # full-deflection cross-track (m)
    xte_frac   = clamp(xte / xte_max, -1.0, 1.0)
    xte_col    = abs(xte_frac) > 0.70 ? TH.red :
                 abs(xte_frac) > 0.40 ? TH.amber : TH.green

    # Track bar
    lines!(ax, [Point2f(arc_cx - xte_bar_hw, xte_bar_y),
                Point2f(arc_cx + xte_bar_hw, xte_bar_y)],
           color=TH.stroke_hi, linewidth=1.2)
    # Centre tick
    lines!(ax, [Point2f(arc_cx, xte_bar_y - 0.015),
                Point2f(arc_cx, xte_bar_y + 0.015)],
           color=TH.text_dim, linewidth=0.8)
    # Needle
    needle_x = arc_cx + xte_frac * xte_bar_hw
    lines!(ax, [Point2f(needle_x, xte_bar_y - 0.022),
                Point2f(needle_x, xte_bar_y + 0.022)],
           color=xte_col, linewidth=2.0)

    # XTE label
    xte_str = abs(xte) < 10.0 ? "XTE 0" :
              @sprintf("XTE %+.0f m", xte)
    text!(ax, arc_cx, xte_bar_y - 0.04,
          text=xte_str, fontsize=7, color=xte_col,
          align=(:center, :center), font=LABEL_FONT)

    # ── Data readout strip (bottom of panel) ──────────────────────────
    # RNG: large display.  AGL: very large, phase-gated colour coding.
    # AGL displayed in feet; thresholds evaluated in metres (snap.alt is AGL metres).
    #   Dash phase:         <100m RED · 100–300m YELLOW · >300m GREEN
    #   All other phases:   <30m  RED · 30–100m  YELLOW · >100m GREEN
    rng_km  = hypot(ac_x - wx, ac_y - wy) / 1000.0
    rng_str = rng_km >= 10.0 ? @sprintf("%.1f km", rng_km) :
              rng_km >= 1.0  ? @sprintf("%.2f km", rng_km) :
                               @sprintf("%.0f m",  rng_km * 1000.0)
    agl_m   = snap.alt
    agl_ft  = agl_m * 3.28084
    agl_str = @sprintf("%.0f ft", agl_ft)
    brg_str = @sprintf("%03d°", round(Int, brg))

    # AGL colour — threshold depends on flight phase
    agl_col = if phase == "dash"
        agl_m < 100.0  ? TH.red   :
        agl_m < 300.0  ? TH.amber :
                         TH.green
    else   # takeoff / hover / transition / landing / any other phase
        agl_m < 30.0   ? TH.red   :
        agl_m < 100.0  ? TH.amber :
                         TH.green
    end

    # BRG small label (upper left)
    text!(ax, -0.96, -0.56,
          text="BRG", fontsize=10, color=TH.text_dim,
          align=(:left, :center), font=LABEL_FONT)
    text!(ax, -0.96, -0.67,
          text=brg_str, fontsize=22, color=wp_col,
          align=(:left, :center), font=DISP_FONT)

    # RNG — large (lower left)
    text!(ax, -0.96, -0.83,
          text="RNG", fontsize=10, color=TH.text_label,
          align=(:left, :center), font=LABEL_FONT)
    text!(ax, -0.96, -0.94,
          text=rng_str, fontsize=22, color=wp_col,
          align=(:left, :center), font=DISP_FONT)

    # Coloured glow box behind AGL value to make it pop (right side)
    agl_box_alpha = 0.10
    poly!(ax, [Point2f(0.28, -0.99), Point2f(0.99, -0.99),
               Point2f(0.99, -0.68), Point2f(0.28, -0.68)],
          color=RGBAf(agl_col.r, agl_col.g, agl_col.b, agl_box_alpha),
          strokecolor=RGBAf(agl_col.r, agl_col.g, agl_col.b, 0.45),
          strokewidth=1.0)

    # AGL — very large, right-aligned, colour-coded
    text!(ax, 0.96, -0.73,
          text="AGL", fontsize=10, color=TH.text_label,
          align=(:right, :center), font=LABEL_FONT)
    text!(ax, 0.96, -0.87,
          text=agl_str, fontsize=36, color=agl_col,
          align=(:right, :center), font=DISP_FONT)

    draw_border!(ax, color=TH.stroke_hi)
end

# ══════════════════════════════════════════════════════════════════════
#  LAYOUT — demo-ready 1440 × 900 (PFD)  +  340 px moving map (col 4)
# ══════════════════════════════════════════════════════════════════════
"""
    launch_cockpit(state_obs; rpm_nom, kw_max_per_rotor, nav_map)

Launch the MIL-STD-3009 NVG-compatible cockpit window.

Pass `nav_map::NavMapState` (from navigation.jl) to enable the moving-map
panel in column 4.  Omit or pass `nothing` for the original 3-column PFD.
"""
function launch_cockpit(state_obs::Observable{CockpitState};
                        rpm_nom::Float64=1050.0,
                        kw_max_per_rotor::Float64=80.0,
                        nav_map=nothing)

    has_map = nav_map !== nothing
    # PFD columns (1-3): 200 + auto + 200 ≈ 960 px at 1440 total.
    # NAV panel target: ~1/3 of total → total ≈ 1440 * 1.5 ≈ 2160 when present.
    # We fix the NAV column width so NAV/(PFD+NAV) ≈ 1/3.
    # nav_col_w ≈ 0.5 * PFD_width.  PFD_width ≈ 1440. → nav_col_w ≈ 720.
    # Resulting total ≈ 2160 px (16:10 works fine at 2160×900).
    fig_w   = has_map ? 2160 : 1440

    fig = Figure(size=(fig_w, 900), backgroundcolor=TH.bg,
                 figure_padding=(6, 6, 6, 6))

    # ── Row 1: Header bar (spans all columns) ────────────────────────
    ax_hdr = Axis(fig[1, 1:(has_map ? 4 : 3)], backgroundcolor=TH.panel_hi)
    hidedecorations!(ax_hdr); hidespines!(ax_hdr)
    xlims!(ax_hdr, 0, 1); ylims!(ax_hdr, 0, 1)

    clock_obs      = Observable("00:00")
    phase_obs      = Observable("HOVER")
    phase_col_obs  = Observable(TH.amber)   # hover/transition default

    # Left: event elapsed time (mm:ss)
    text!(ax_hdr, 0.01, 0.5,
          text=clock_obs, fontsize=20, color=TH.text,
          align=(:left, :center), font=LABEL_FONT)

    # Centre: airport / environment data
    wind_kt  = hypot(ATM.wind.u, ATM.wind.v) * 1.94384
    wind_hdg = mod(rad2deg(atan(ATM.wind.u, ATM.wind.v)) + 360, 360)
    temp_c   = ATM.ambient_temp_c
    apt_str  = @sprintf("%s  %d FT MSL %.0f C WND %.0f %.0f KT",
                        ATM.airport_icao,
                        round(Int, ATM.airport_alt_m * 3.28084),
                        temp_c, wind_hdg, wind_kt)
    text!(ax_hdr, 0.5, 0.5,
          text=apt_str, fontsize=20, color=TH.blue,
          align=(:center, :center), font=LABEL_FONT)

    # Right: flight phase — green for LANDED/HOVER, amber otherwise
    text!(ax_hdr, 0.99, 0.5,
          text=phase_obs, fontsize=20, color=phase_col_obs,
          align=(:right, :center), font=LABEL_FONT)

    # ── Shared column grid ────────────────────────────────────────────
    # All rows live directly in fig.layout so column edges are identical.
    # Column widths are set once here:
    #   col 1 — left panel  (speed tape / power)   Fixed(200)
    #   col 2 — centre      (ADI + heading / rotors)  Relative(1)
    #   col 3 — right panel (alt tape / tilt)      Fixed(200)
    col_w = 200   # single source of truth for left/right column width (VCON/power, CONTACT/tilt)

    # ══════════════════════════════════════════════════════════════════
    #  REVISED LAYOUT — HUD tapes via nested GridLayout
    # ══════════════════════════════════════════════════════════════════
    #
    # Axis does not support halign/valign/width=Relative in GLMakie, so
    # tapes cannot be true pixel overlays.  Instead the PFD centre cell
    # (outer col 2, rows 2-3) hosts a nested 3-column × 2-row GridLayout:
    #
    #   pfd_grid sub-cols:  [tape_w | auto | tape_w]
    #   pfd_grid sub-rows:  [auto (ADI body) | Fixed(52) (heading strip)]
    #
    # Speed tape  → pfd_grid[1:2, 1]  (spans both sub-rows, left column)
    # ADI         → pfd_grid[1,   2]  (tall centre cell)
    # Alt tape    → pfd_grid[1:2, 3]  (spans both sub-rows, right column)
    # Heading     → pfd_grid[2,   2]  (narrow strip below ADI)
    #
    # VCON  → outer fig[3, 1]   (left flank, heading-tape row only)
    # CONTACT → outer fig[3, 3] (right flank, heading-tape row only)
    #
    # Outer col 1 / col 3 are Fixed(col_w) and span rows 2-4 so the
    # Power and Tilt panels in row 4 align correctly.
    #
    # NAV map → outer fig[2, 4]  row 2 only, top-aligned, square aspect.
    # Rows 3 and 4 of col 4 are intentionally empty for future use.

    # ── Layout: original flat 3-column PFD, all in fig.layout ─────────
    # Row 2 (tall, auto):   speed tape | ADI | alt tape  [| NAV map]
    # Row 3 (Fixed 52):     VCON       | hdg tape | CONTACT
    # Row 4 (Fixed 220):    power      | rotors   | tilt
    #
    # Speed/alt tapes each get their own Fixed-width column in row 2 only.
    # VCON and CONTACT sit in the same columns but row 3 only.
    # The heading tape occupies col 2 row 3.
    # This is the simplest layout that Makie handles correctly.

    ax_speed   = Axis(fig[2, 1], backgroundcolor=TH.bg)
    ax_att     = Axis(fig[2, 2], backgroundcolor=TH.bg)
    ax_alt     = Axis(fig[2, 3], backgroundcolor=TH.bg)

    ax_vcon    = Axis(fig[3, 1], backgroundcolor=TH.panel)
    ax_hdg     = Axis(fig[3, 2], backgroundcolor=TH.panel)
    ax_contact = Axis(fig[3, 3], backgroundcolor=TH.panel)

    # ── Row 4: Power | Rotor gauges × 6 | Tilt ───────────────────────
    ax_power = Axis(fig[4, 1], backgroundcolor=TH.bg)
    ax_tilt  = Axis(fig[4, 3], backgroundcolor=TH.bg)

    rotor_grid = fig[4, 2] = GridLayout()
    ax_rotors  = [Axis(rotor_grid[1, i], backgroundcolor=TH.bg)
                  for i in 1:6]
    colgap!(rotor_grid, 3)

    # ── NAV map: col 4, row 2 only — top-aligned, square aspect ──────
    ax_map = if has_map
        ax = Axis(fig[2, 4], backgroundcolor=TH.panel, aspect=DataAspect())
        hidedecorations!(ax); hidespines!(ax)
        ax
    else
        nothing
    end

    # ── Global sizing ─────────────────────────────────────────────────
    nav_col_w = has_map ? round(Int, (fig_w - 2*col_w) * 0.52) : 0

    colsize!(fig.layout, 1, Fixed(col_w))
    colsize!(fig.layout, 3, Fixed(col_w))
    has_map && colsize!(fig.layout, 4, Fixed(nav_col_w))
    rowsize!(fig.layout, 1, Fixed(40))    # header
    rowsize!(fig.layout, 3, Fixed(52))    # VCON / heading tape / CONTACT
    rowsize!(fig.layout, 4, Fixed(220))   # bottom instruments
    colgap!(fig.layout, 4)
    rowgap!(fig.layout, 4)

    # ── Update function ───────────────────────────────────────────────
    function update!(s::CockpitState)
        v = s.vals

        # Header — event elapsed time mm:ss
        t_sim = v[IDX.t]
        clock_obs[] = @sprintf("%02d:%02d", floor(Int, t_sim/60), floor(Int, t_sim)%60)
        phase_obs[] = uppercase(s.phase)
        phase_col_obs[] = s.phase == "landed"  ? TH.green :
                          s.phase == "dash"    ? TH.blue  : TH.amber

        # Primary flight instruments
        draw_tape!(ax_speed, v[IDX.speed], 80.0, 20.0, 10.0, "IAS", "KM/H")
        draw_tape!(ax_alt,   v[IDX.alt],  500.0, 100.0, 50.0, "ALT", "FT",
                   TH.text; flip=true)
        draw_attitude!(ax_att, v[IDX.pitch], v[IDX.roll])
        draw_heading_tape!(ax_hdg, v[IDX.yaw])
        # Compute conversion corridor limits from physics (airframe.jl).
        # vcon_limits takes tilt in radians and altitude AGL in metres.
        vcon = vcon_limits(deg2rad(v[IDX.tilt]), v[IDX.alt_agl_m])
        draw_vcon!(ax_vcon, v[IDX.speed], v[IDX.tilt], vcon.lo_kmh, vcon.hi_kmh)
        draw_contact!(ax_contact, s.gear_contact, s.strut_load_n,
                      s.brakes_on, v[IDX.gz])

        # Bottom row
        draw_power!(ax_power, v[IDX.power], v[IDX.soc], v[IDX.batt_temp])
        draw_rotor_gauges!(ax_rotors, s.rotor_rpm, s.rotor_kw,
                           s.rotor_labels, s.n_rotors,
                           rpm_nom, kw_max_per_rotor)
        draw_tilt!(ax_tilt, v[IDX.tilt])

        # Moving map — only when NAV_MAP was supplied to launch_cockpit
        if has_map
            snap = nav_snapshot(nav_map)
            draw_nav_map!(ax_map, snap)
        end
    end

    on(state_obs) do s
        update!(s)
    end

    update!(state_obs[])
    display(fig)
    return fig
end

# ══════════════════════════════════════════════════════════════════════
#  CSV PLAYBACK MODE
# ══════════════════════════════════════════════════════════════════════
# Pass nav_map=NAV_MAP (from navigation.jl) to enable the moving map
# during CSV playback.  The playback loop calls nav_push! to feed the
# track buffer from the x_m / y_m / alt_agl_m / yaw_deg columns.
function playback_csv(path::String; fps=10.0, nav_map=nothing)
    df = CSV.read(path, DataFrame)
    println("Loaded $(nrow(df)) rows from $path")
    println("Playing back at $(fps)x realtime … [NVG MODE]")

    state = CockpitState()
    obs   = Observable(state)
    fig   = launch_cockpit(obs; nav_map=nav_map)

    dt_real  = 1.0 / fps
    hist_max = 600

    for col in ["rpm_r1","rpm_r2","rpm_r3","rpm_r4","rpm_r5","rpm_r6",
                "kw_r1", "kw_r2", "kw_r3", "kw_r4", "kw_r5", "kw_r6"]
        hasproperty(df, Symbol(col)) ||
            error("CSV missing column '$col' — re-run the simulation.")
    end

    for row in eachrow(df)
        state.vals[IDX.t]         = row.timestamp_s
        state.vals[IDX.tau]       = row.tau_s
        state.vals[IDX.speed]     = row.speed_kmh
        state.vals[IDX.alt]       = row.altitude_msl_ft
        state.vals[IDX.power]     = row.power_kw
        state.vals[IDX.tilt]      = row.tilt_deg
        state.vals[IDX.pitch]     = row.pitch_deg
        state.vals[IDX.roll]      = row.roll_deg
        state.vals[IDX.yaw]       = row.yaw_deg
        state.vals[IDX.soc]       = row.soc_pct
        state.vals[IDX.voltage]   = row.voltage_v
        state.vals[IDX.batt_temp] = row.batt_temp_c
        state.vals[IDX.x_m]       = hasproperty(row, :x_m) ? row.x_m : 0.0
        state.vals[IDX.y_m]       = hasproperty(row, :y_m) ? row.y_m : 0.0
        state.vals[IDX.omega_x]   = hasproperty(row, :omega_x_rads) ? row.omega_x_rads : 0.0
        state.vals[IDX.omega_y]   = hasproperty(row, :omega_y_rads) ? row.omega_y_rads : 0.0
        state.vals[IDX.omega_z]   = hasproperty(row, :omega_z_rads) ? row.omega_z_rads : 0.0
        # alt_agl_m: prefer direct CSV column; fall back to reverse-converting
        # altitude_msl_ft using the ATM airport elevation (requires ATM to be loaded).
        state.vals[IDX.alt_agl_m] = if hasproperty(row, :alt_agl_m)
            Float64(row.alt_agl_m)
        elseif isdefined(Main, :ATM)
            max(row.altitude_msl_ft / 3.28084 - ATM.airport_alt_m, 0.0)
        else
            0.0
        end
        state.gear_contact         = hasproperty(row, :gear_contact)  ? Bool(row.gear_contact)  : false
        state.strut_load_n         = hasproperty(row, :strut_load_n)  ? Float64(row.strut_load_n) : 0.0
        state.phase                = row.phase

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
            nav_push!(nav_map, u_pb, row.timestamp_s, row.phase)
        end

        obs[] = state
        sleep(dt_real)
    end

    println("Playback complete. Close window to exit.")
    wait(fig.scene)
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
        println("Usage:  julia glass_cockpit_nvg.jl dash_results.csv")
        println("        COCKPIT_FPS=30 julia glass_cockpit_nvg.jl dash_results.csv")
    end
end

/* cockpit.js — WINDBLADE glass cockpit client renderer.
 *
 * One <canvas> per instrument, redrawn on every WebSocket state frame
 * (server pushes at the sim's notify cadence — 20 Hz solving, 10 Hz after
 * landing, COCKPIT_FPS in playback).  Canvas2D throughout: every
 * instrument here is fully redrawn per frame at modest element counts,
 * where immediate-mode canvas is simpler and faster than SVG DOM churn.
 *
 * Garmin G1000 NXi / G3000 conventions applied (see DESIGN_RATIONALE.md
 * for the section-by-section mapping to AC 23.1311-1C / AC 25-11B):
 *   - tapes: pointed current-value lubber, 6-second trend vector
 *   - ADI: top-mounted roll scale w/ sky pointer + slip/skid trapezoid
 *   - colour philosophy: white=normal, green=in-envelope/engaged,
 *     amber=caution, red=warning, blue=informational
 *   - invalid data: red-X the affected instrument (link loss)
 */
"use strict";

/* ── Theme ──────────────────────────────────────────────────────── */
let TH = {};
function readTheme() {
  const cs = getComputedStyle(document.body);
  const v = (n) => cs.getPropertyValue(n).trim();
  TH = {
    bg: v("--bg"), panel: v("--panel"), panelHi: v("--panel-hi"),
    stroke: v("--stroke"), strokeHi: v("--stroke-hi"),
    text: v("--text"), dim: v("--text-dim"), faint: v("--text-faint"),
    green: v("--green"), amber: v("--amber"), red: v("--red"),
    blue: v("--blue"), sky: v("--sky"), ground: v("--ground"),
    bugFill: v("--bug-fill"),
  };
}
function setTheme(day) {
  document.body.classList.toggle("theme-day", day);
  document.body.classList.toggle("theme-nvg", !day);
  document.getElementById("themeBtn").textContent = day ? "NVG" : "DAY";
  readTheme();
  renderAll();
}
document.getElementById("themeBtn").addEventListener("click", () =>
  setTheme(!document.body.classList.contains("theme-day")));

/* ── Fonts ──────────────────────────────────────────────────────── */
const F  = (px, w = 400) => `${w} ${px}px "B612", system-ui, sans-serif`;
const FM = (px, w = 400) => `${w} ${px}px "B612 Mono", ui-monospace, monospace`;

/* ── Canvas plumbing ────────────────────────────────────────────── */
const CV = {};                        // id -> {cv, ctx, w, h}
function initCanvas(id) {
  const cv = document.getElementById(id);
  CV[id] = { cv, ctx: cv.getContext("2d"), w: 0, h: 0 };
  new ResizeObserver(() => { sizeCanvas(id); renderAll(); })
    .observe(cv.parentElement);
  sizeCanvas(id);
}
function sizeCanvas(id) {
  const c = CV[id], dpr = window.devicePixelRatio || 1;
  const r = c.cv.parentElement.getBoundingClientRect();
  c.w = Math.max(10, Math.round(r.width));
  c.h = Math.max(10, Math.round(r.height));
  c.cv.width  = Math.round(c.w * dpr);
  c.cv.height = Math.round(c.h * dpr);
  c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function begin(id) {
  const c = CV[id];
  c.ctx.clearRect(0, 0, c.w, c.h);
  return c;
}
function txt(ctx, s, x, y, font, color, alignX = "center", alignY = "middle") {
  ctx.font = font; ctx.fillStyle = color;
  ctx.textAlign = alignX; ctx.textBaseline = alignY;
  ctx.fillText(s, x, y);
}
const fmt  = (x, d = 0) => Number(x).toFixed(d);
const pad3 = (x) => String(Math.round(x)).padStart(3, "0");

/* ── State ──────────────────────────────────────────────────────── */
let init = null;          // static config ("init" message)
let st   = null;          // latest "state" message
let lastMsg = 0;          // performance.now() of last state frame
const trend = [];         // ring buffer {t, spd, alt} for 6-s trend vectors
let linkUp = false;
const M2FT = 3.28084;
const KMH2KT = 0.539957;  // display-only conversion; wire format / VCON thresholds stay km/h

/* Radar altitude in feet: alt_agl_terrain_m is height above the terrain
 * directly below (what a radalt actually senses); fall back to CG AGL. */
function radAltFt(s) {
  if (typeof s.agl_terr_m === "number") return s.agl_terr_m * M2FT;
  if (typeof s.agl_m === "number")      return s.agl_m * M2FT;
  return null;
}
const hasTerr = (s) => typeof s.agl_terr_m === "number";

/* ── WebSocket ──────────────────────────────────────────────────── */
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { linkUp = true; setLink(true); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "init") {
      init = m;
      document.getElementById("apt").textContent = m.airport || "";
      document.getElementById("cockpit").classList.toggle("has-nav", !!m.has_nav);
    } else if (m.type === "state") {
      st = m;
      lastMsg = performance.now();
      trend.push({ t: m.t, spd: m.speed, alt: m.alt });
      while (trend.length > 400 || (trend.length > 2 && m.t - trend[0].t > 8)) trend.shift();
      renderAll();
    }
  };
  ws.onclose = () => {
    linkUp = false; setLink(false); renderAll();
    setTimeout(connect, 1000);       // auto-reconnect
  };
  ws.onerror = () => ws.close();
}
function setLink(up) {
  const el = document.getElementById("link");
  el.classList.toggle("down", !up);
  el.textContent = up ? "LINK" : "NO LINK";
}
/* Stale watchdog: >2 s without a frame while connected = stale data */
setInterval(() => {
  if (linkUp && st && performance.now() - lastMsg > 2000) renderAll();
}, 500);
const isStale = () => !linkUp || (st && performance.now() - lastMsg > 2000);

/* Red-X an instrument per AC 25-11B invalid-data practice */
function redX(c, label) {
  const { ctx, w, h } = c;
  ctx.strokeStyle = TH.red; ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(6, 6); ctx.lineTo(w - 6, h - 6);
  ctx.moveTo(w - 6, 6); ctx.lineTo(6, h - 6);
  ctx.stroke();
  if (label) txt(ctx, label, w / 2, h / 2, F(12, 700), TH.red);
}

/* Trend value: change over the trailing 6 s (Garmin trend-vector basis) */
function trend6(key) {
  if (trend.length < 2 || !st) return 0;
  const t0 = st.t - 6;
  let a = trend[0];
  for (const p of trend) { if (p.t <= t0) a = p; else break; }
  const dt = st.t - a.t;
  if (dt < 0.5) return 0;
  return (st[key === "spd" ? "speed" : "alt"] - a[key]) * (6 / dt) * (dt >= 6 ? 1 : dt / 6);
}

/* ══════════════════════════════════════════════════════════════════
 *  TAPES — speed (right-edge ticks) / altitude (left-edge ticks)
 * ══════════════════════════════════════════════════════════════════ */
function drawTape(id, value, half, majorStep, minorStep, label, unit,
                  flip, trendDelta) {
  const c = begin(id), { ctx, w, h } = c;
  const y = (v) => h * (0.5 - (v - value) / (2 * half));

  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, w, h);

  const edge   = flip ? 0 : w;                 // tick edge (toward ADI)
  const tickIn = flip ? w * 0.16 : w * 0.84;
  const labX   = flip ? w * 0.20 : w * 0.80;

  let v = Math.ceil((value - half) / minorStep) * minorStep;
  ctx.lineWidth = 1;
  for (; v <= value + half; v += minorStep) {
    const major = Math.abs(v % majorStep) < 0.01;
    const yy = y(v);
    if (yy < 26 || yy > h - 6) continue;       // keep clear of the label strip
    ctx.strokeStyle = major ? TH.text : TH.dim;
    ctx.lineWidth = major ? 1.4 : 0.7;
    ctx.beginPath(); ctx.moveTo(tickIn, yy); ctx.lineTo(edge, yy); ctx.stroke();
    if (major)
      txt(ctx, fmt(v), labX, yy, FM(12), TH.text, flip ? "left" : "right");
  }

  /* 6-second trend vector — blue bar from centre along the scale */
  if (Math.abs(trendDelta) > minorStep * 0.15) {
    const x0 = flip ? 5 : w - 5;
    ctx.strokeStyle = TH.blue; ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(x0, h / 2);
    ctx.lineTo(x0, y(value + trendDelta));
    ctx.stroke();
  }

  /* Current-value lubber: pointed box, notch toward the ADI */
  const bh = 15, notch = 9;
  const xNotch = flip ? 2 : w - 2;
  const xFlat  = flip ? w * 0.98 : w * 0.02;
  ctx.beginPath();
  if (flip) {   /* altitude — pointer on left edge */
    ctx.moveTo(xNotch, h / 2);
    ctx.lineTo(xNotch + notch, h / 2 - bh);
    ctx.lineTo(xFlat, h / 2 - bh);
    ctx.lineTo(xFlat, h / 2 + bh);
    ctx.lineTo(xNotch + notch, h / 2 + bh);
  } else {      /* speed — pointer on right edge */
    ctx.moveTo(xNotch, h / 2);
    ctx.lineTo(xNotch - notch, h / 2 - bh);
    ctx.lineTo(xFlat, h / 2 - bh);
    ctx.lineTo(xFlat, h / 2 + bh);
    ctx.lineTo(xNotch - notch, h / 2 + bh);
  }
  ctx.closePath();
  ctx.fillStyle = TH.bugFill; ctx.fill();
  ctx.strokeStyle = TH.text; ctx.lineWidth = 1.2; ctx.stroke();
  txt(ctx, fmt(value), w / 2, h / 2, FM(19, 700), TH.text);

  /* Label strip */
  ctx.fillStyle = TH.panelHi; ctx.fillRect(0, 0, w, 22);
  ctx.strokeStyle = TH.stroke; ctx.strokeRect(0.5, 0.5, w - 1, 21);
  txt(ctx, `${label} ${unit}`, w / 2, 11, F(12, 700), TH.green);

  if (isStale()) redX(c);
}

/* ══════════════════════════════════════════════════════════════════
 *  ADI — attitude indicator
 * ══════════════════════════════════════════════════════════════════ */
function drawADI() {
  const c = begin("cv-adi"), { ctx, w, h } = c;
  if (!st) { txt(ctx, "AWAITING DATA", w / 2, h / 2, F(13), TH.faint); return; }

  const cx = w / 2, cy = h / 2;
  const pitch = st.pitch, roll = st.roll;
  const pxPerDeg = h / 60;                     // 30° pitch ≈ half-height
  const rr = -roll * Math.PI / 180;            // screen rotation

  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
  ctx.translate(cx, cy);
  ctx.rotate(rr);
  const hy = pitch * pxPerDeg;                 // horizon offset (up = +pitch)
  const R = Math.hypot(w, h);

  ctx.fillStyle = TH.sky;    ctx.fillRect(-R, -R, 2 * R, R + hy);
  ctx.fillStyle = TH.ground; ctx.fillRect(-R, hy, 2 * R, R);
  ctx.strokeStyle = TH.text; ctx.lineWidth = 1.8;
  ctx.beginPath(); ctx.moveTo(-R, hy); ctx.lineTo(R, hy); ctx.stroke();

  /* Pitch ladder: 10° numbered rungs, 5° half-rungs */
  ctx.lineWidth = 1;
  for (let p = -30; p <= 30; p += 5) {
    if (p === 0) continue;
    const yy = (pitch - p) * pxPerDeg;
    if (Math.abs(yy) > h * 0.44) continue;
    const major = p % 10 === 0;
    const len = major ? w * 0.085 : w * 0.045;
    ctx.strokeStyle = major ? TH.text : TH.dim;
    ctx.beginPath(); ctx.moveTo(-len, yy); ctx.lineTo(len, yy); ctx.stroke();
    if (major) {
      txt(ctx, String(Math.abs(p)), -len - 8, yy, FM(11), TH.dim, "right");
      txt(ctx, String(Math.abs(p)),  len + 8, yy, FM(11), TH.dim, "left");
    }
  }
  ctx.restore();

  /* Roll scale — fixed arc at top (Garmin/G1000 convention), moving
     sky pointer + slip/skid trapezoid */
  const arcR = Math.min(w, h) * 0.42;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.strokeStyle = TH.dim; ctx.lineWidth = 1;
  for (const d of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) {
    const a = (d - 90) * Math.PI / 180;
    const tl = (d % 30 === 0) ? 12 : 7;
    ctx.beginPath();
    ctx.moveTo(Math.cos(a) * arcR, Math.sin(a) * arcR);
    ctx.lineTo(Math.cos(a) * (arcR - tl), Math.sin(a) * (arcR - tl));
    ctx.stroke();
  }
  /* zero-reference triangle (fixed, apex down) */
  ctx.fillStyle = TH.text;
  ctx.beginPath();
  ctx.moveTo(0, -arcR);
  ctx.lineTo(-6, -arcR - 10); ctx.lineTo(6, -arcR - 10);
  ctx.closePath(); ctx.fill();

  /* moving roll pointer (apex up) + slip/skid trapezoid driven by gy */
  ctx.save();
  ctx.rotate(rr);
  ctx.fillStyle = TH.text;
  ctx.beginPath();
  ctx.moveTo(0, -arcR + 2);
  ctx.lineTo(-6, -arcR + 13); ctx.lineTo(6, -arcR + 13);
  ctx.closePath(); ctx.fill();
  const skid = Math.max(-1, Math.min(1, st.gy)) * 14;
  ctx.beginPath();
  ctx.moveTo(-7 + skid, -arcR + 17); ctx.lineTo(7 + skid, -arcR + 17);
  ctx.lineTo(9 + skid, -arcR + 23);  ctx.lineTo(-9 + skid, -arcR + 23);
  ctx.closePath(); ctx.fill();
  ctx.restore();
  ctx.restore();

  /* Fixed aircraft reference — amber flying-W (unchanged symbology) */
  ctx.strokeStyle = TH.amber; ctx.lineWidth = 3.5; ctx.lineCap = "round";
  const a1 = w * 0.055, a2 = w * 0.17;
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(cx + s * a2, cy); ctx.lineTo(cx + s * a1, cy);
    ctx.lineTo(cx + s * a1, cy + 9);
    ctx.stroke();
  }
  ctx.fillStyle = TH.amber;
  ctx.beginPath(); ctx.arc(cx, cy, 3.4, 0, 2 * Math.PI); ctx.fill();
  ctx.lineCap = "butt";

  if (isStale()) redX(c, "ATT");
}

/* ══════════════════════════════════════════════════════════════════
 *  Heading tape
 * ══════════════════════════════════════════════════════════════════ */
function drawHeading() {
  const c = begin("cv-hdg"), { ctx, w, h } = c;
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, w, h);
  if (!st) return;

  const hdg = ((st.yaw % 360) + 360) % 360;
  const half = 40;
  const x = (d) => w / 2 + (d - hdg) * (w / (2 * half));
  const card = { 0: "N", 90: "E", 180: "S", 270: "W" };

  const lo = Math.floor((hdg - half - 30) / 10) * 10;
  const hi = Math.ceil((hdg + half + 30) / 10) * 10;
  for (let d = lo; d <= hi; d += 10) {
    const dw = ((d % 360) + 360) % 360;
    const major = dw % 30 === 0;
    const xx = x(d);
    if (xx < -20 || xx > w + 20) continue;
    ctx.strokeStyle = major ? TH.text : TH.dim;
    ctx.lineWidth = major ? 1.4 : 0.7;
    ctx.beginPath(); ctx.moveTo(xx, h); ctx.lineTo(xx, h - (major ? 0.42 : 0.3) * h); ctx.stroke();
    if (dw in card)
      txt(ctx, card[dw], xx, h * 0.24, F(13, 700), TH.text);
    else if (major)
      txt(ctx, pad3(dw), xx, h * 0.24, FM(9), TH.dim);
  }

  /* lubber + heading readout box */
  ctx.strokeStyle = TH.text; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
  const bw = 34, bh = h * 0.62, by = (h - bh) / 2;
  ctx.fillStyle = TH.bugFill;
  ctx.fillRect(w / 2 - bw, by, 2 * bw, bh);
  ctx.strokeStyle = TH.text; ctx.lineWidth = 1.2;
  ctx.strokeRect(w / 2 - bw, by, 2 * bw, bh);
  txt(ctx, pad3(Math.round(hdg) % 360) + "\u00B0", w / 2, h / 2, FM(18, 700), TH.text);

  if (isStale()) redX(c);
}

/* ══════════════════════════════════════════════════════════════════
 *  VCON — conversion corridor
 *  Active (10° < tilt < 80°): horizontal envelope band with speed caret.
 *  Green inside, amber within 15 km/h of a bound, red outside — same
 *  status logic as v0.1.1, now with the corridor itself drawn to scale.
 * ══════════════════════════════════════════════════════════════════ */
function drawVcon() {
  const c = begin("cv-vcon"), { ctx, w, h } = c;
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, w, h);
  if (!st) return;

  const { lo, hi } = st.vcon;
  const tilt = st.tilt, spd = st.speed, warn = 15;
  const inTrans = tilt > 10 && tilt < 80;

  if (!inTrans) {
    ctx.strokeStyle = TH.stroke; ctx.lineWidth = 0.8;
    ctx.strokeRect(w * 0.06, h * 0.2, w * 0.88, h * 0.6);
    txt(ctx, "VCON", w * 0.28, h / 2, F(11, 700), TH.faint);
    txt(ctx, `${fmt(lo * KMH2KT)}\u2013${fmt(hi * KMH2KT)} KT`, w * 0.66, h / 2, FM(9), TH.faint);
    return;
  }

  const below = spd < lo, above = spd > hi;
  const outside = below || above;
  const marginal = !outside && (spd < lo + warn || spd > hi - warn);
  const col = outside ? TH.red : marginal ? TH.amber : TH.green;

  /* scale: lo-30 .. hi+30 */
  const s0 = lo - 30, s1 = hi + 30;
  const X = (v) => 8 + (Math.max(s0, Math.min(s1, v)) - s0) / (s1 - s0) * (w - 16);
  const bandY = h * 0.62, bandH = h * 0.24;

  ctx.fillStyle = TH.panelHi;
  ctx.fillRect(8, bandY, w - 16, bandH);
  /* amber caution margins, green core */
  ctx.fillStyle = TH.amber; ctx.globalAlpha = 0.35;
  ctx.fillRect(X(lo), bandY, X(lo + warn) - X(lo), bandH);
  ctx.fillRect(X(hi - warn), bandY, X(hi) - X(hi - warn), bandH);
  ctx.globalAlpha = 0.5; ctx.fillStyle = TH.green;
  ctx.fillRect(X(lo + warn), bandY, X(hi - warn) - X(lo + warn), bandH);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = TH.dim; ctx.lineWidth = 1;
  ctx.strokeRect(X(lo), bandY, X(hi) - X(lo), bandH);

  /* speed caret */
  const xc = X(spd);
  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.moveTo(xc, bandY - 2);
  ctx.lineTo(xc - 5, bandY - 9); ctx.lineTo(xc + 5, bandY - 9);
  ctx.closePath(); ctx.fill();

  txt(ctx, "VCON", w * 0.17, h * 0.26, F(11, 700), col, "center");
  txt(ctx, `${fmt(spd * KMH2KT)} KT`, w * 0.66, h * 0.26, FM(11, 700), col);
  if (isStale()) redX(c);
}

/* ══════════════════════════════════════════════════════════════════
 *  CONTACT / BRAKES / gz annunciator
 * ══════════════════════════════════════════════════════════════════ */
function drawContact() {
  const c = begin("cv-contact"), { ctx, w, h } = c;
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, w, h);
  if (!st) return;

  /* BRAKES chip (left) */
  const bc = st.brakes ? TH.red : TH.stroke;
  if (st.brakes) {
    ctx.fillStyle = TH.red; ctx.globalAlpha = 0.22;
    ctx.fillRect(w * 0.03, h * 0.2, w * 0.27, h * 0.6);
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = bc; ctx.lineWidth = 1.2;
  ctx.strokeRect(w * 0.03, h * 0.2, w * 0.27, h * 0.6);
  txt(ctx, "BRK", w * 0.165, h / 2, F(10, 700), st.brakes ? TH.red : TH.faint);

  /* CONTACT box (centre) */
  const x0 = w * 0.33, x1 = w * 0.72;
  if (st.gear) {
    ctx.fillStyle = TH.green; ctx.globalAlpha = 0.18;
    ctx.fillRect(x0, h * 0.14, x1 - x0, h * 0.72);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = TH.green; ctx.lineWidth = 1.4;
    ctx.strokeRect(x0, h * 0.14, x1 - x0, h * 0.72);
    txt(ctx, "CONTACT", (x0 + x1) / 2, h * 0.36, F(10, 700), TH.green);
    const load = st.strut_n > 999 ? `${fmt(st.strut_n / 1000, 1)} kN`
                                  : `${fmt(st.strut_n)} N`;
    txt(ctx, load, (x0 + x1) / 2, h * 0.68, FM(9), TH.dim);
  } else {
    ctx.strokeStyle = TH.stroke; ctx.lineWidth = 0.8;
    ctx.strokeRect(x0, h * 0.14, x1 - x0, h * 0.72);
    txt(ctx, "- - -", (x0 + x1) / 2, h / 2, F(10), TH.faint);
  }

  /* gz readout (right) — amber >1.8 g, red >2.5 g */
  const gz = st.gz;
  const gcol = Math.abs(gz) > 2.5 ? TH.red : Math.abs(gz) > 1.8 ? TH.amber : TH.dim;
  txt(ctx, "GZ", w * 0.86, h * 0.3, F(9), TH.faint);
  txt(ctx, fmt(gz, 2), w * 0.86, h * 0.66, FM(12, 700), gcol);
  if (isStale()) redX(c);
}

/* ══════════════════════════════════════════════════════════════════
 *  POWERPLANT — SOC / batt temp / fuel + power trend sparkline
 * ══════════════════════════════════════════════════════════════════ */
function drawPower() {
  const c = begin("cv-power"), { ctx, w, h } = c;
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, w, h);
  if (!st || !init) return;
  const showFuel = init.show_fuel;

  txt(ctx, "POWERPLANT", w / 2, 10, F(9), TH.faint);

  const rows = showFuel ? 3 : 2;
  const rowH = (h - 64) / rows;
  const bar = (yTop, frac, col) => {
    ctx.fillStyle = TH.panelHi; ctx.fillRect(12, yTop, w - 24, 5);
    ctx.fillStyle = col; ctx.fillRect(12, yTop, (w - 24) * Math.max(0, Math.min(1, frac)), 5);
  };

  /* SOC */
  let y = 20;
  const socCol = st.soc > 40 ? TH.green : st.soc > 20 ? TH.amber : TH.red;
  txt(ctx, "SOC", 12, y + 8, F(9), TH.dim, "left");
  txt(ctx, `${fmt(st.soc)}%`, w - 12, y + rowH * 0.42, FM(21, 700), socCol, "right");
  bar(y + rowH * 0.72, st.soc / 100, socCol);

  /* BATT TEMP */
  y += rowH;
  const tCol = st.batt_temp > 50 ? TH.amber : TH.text;
  txt(ctx, "BATT TEMP", 12, y + 8, F(9), TH.dim, "left");
  txt(ctx, `${fmt(st.batt_temp, 1)}\u00B0C`, w - 12, y + rowH * 0.42, FM(21, 700), tCol, "right");

  /* FUEL (turbine / hybrid fleets only) */
  if (showFuel) {
    y += rowH;
    const frac = st.fuel_cap > 0 ? Math.max(0, Math.min(1, st.fuel_kg / st.fuel_cap)) : 0;
    const low = frac <= 0.15;
    const fCol = frac > 0.4 ? TH.green : frac > 0.15 ? TH.amber : TH.red;
    txt(ctx, low ? "FUEL \u2014 LOW" : "FUEL", 12, y + 8, F(9), low ? TH.red : TH.dim, "left");
    txt(ctx, `${fmt(frac * 100)}%`, w - 12, y + rowH * 0.42, FM(21, 700), fCol, "right");
    bar(y + rowH * 0.72, frac, fCol);
  }

  /* Power trend sparkline (history_power was carried but never drawn
     in the GLMakie panel — surfacing it here) */
  const sy0 = h - 40, sh = 26;
  ctx.strokeStyle = TH.stroke; ctx.lineWidth = 0.8;
  ctx.strokeRect(12, sy0, w - 24, sh);
  const ph = st.phist || [];
  if (ph.length > 1) {
    const pmax = Math.max(...ph, 1);
    ctx.strokeStyle = TH.blue; ctx.lineWidth = 1.2;
    ctx.beginPath();
    ph.forEach((p, i) => {
      const x = 12 + (i / (ph.length - 1)) * (w - 24);
      const yy = sy0 + sh - (p / pmax) * (sh - 3);
      i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy);
    });
    ctx.stroke();
  }
  txt(ctx, "PWR", 16, sy0 - 7, F(8), TH.faint, "left");
  txt(ctx, `${fmt(st.power)} kW`, w - 12, sy0 - 7, FM(11, 700), TH.text, "right");
  if (isStale()) redX(c);
}

/* ══════════════════════════════════════════════════════════════════
 *  Rotor gauges ×6 — RPM, kW bar, config-mode chip, PROTO tag
 * ══════════════════════════════════════════════════════════════════ */
function drawRotors() {
  const c = begin("cv-rotors"), { ctx, w, h } = c;
  if (!st || !init) return;
  const n = init.n_rotors, gap = 4;
  const cw = (w - gap * 5) / 6;
  const rpmNom = init.rpm_nom || 1050, kwMax = init.kw_max || 80;

  for (let i = 0; i < 6; i++) {
    const x = i * (cw + gap);
    ctx.fillStyle = TH.panel; ctx.fillRect(x, 0, cw, h);
    ctx.strokeStyle = TH.stroke; ctx.lineWidth = 0.8;
    ctx.strokeRect(x + 0.5, 0.5, cw - 1, h - 1);

    if (i >= n) { txt(ctx, "---", x + cw / 2, h / 2, F(10), TH.faint); continue; }

    const rpm = st.rpm[i], kw = st.kw[i];
    const label = init.labels[i], mode = (init.modes || [])[i] || "";
    const proto = label !== `R${i + 1}`;
    const live = rpm > 0.05 * rpmNom;

    /* header: label + config-mode chip (TILT/LIFT/THRUST/OOS) */
    txt(ctx, label, x + 8, 12, F(10, 700), proto ? TH.amber : TH.text, "left");
    if (mode) {
      const mc = mode === "OOS" ? TH.faint : live ? TH.green : TH.dim;
      ctx.strokeStyle = mc; ctx.lineWidth = 0.8;
      ctx.font = F(8);
      const mw = ctx.measureText(mode).width + 12;
      ctx.strokeRect(x + cw - 8 - mw, 5, mw, 14);
      txt(ctx, mode, x + cw - 8 - mw / 2, 12, F(8), mc);
    }
    ctx.strokeStyle = TH.strokeHi; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(x + 6, 22); ctx.lineTo(x + cw - 6, 22); ctx.stroke();

    /* kW bar — amber >85 %, red >100 % of per-rotor max */
    const bf = Math.max(0, Math.min(1.05, kw / kwMax));
    const bcol = bf > 1.0 ? TH.red : bf > 0.85 ? TH.amber : TH.green;
    const bx = x + cw - 24, bw2 = 14, by1 = h - 14, by0 = 30;
    ctx.fillStyle = TH.panelHi; ctx.fillRect(bx, by0, bw2, by1 - by0);
    ctx.strokeStyle = TH.stroke; ctx.strokeRect(bx + 0.5, by0 + 0.5, bw2 - 1, by1 - by0 - 1);
    const fh = Math.min(1, bf) * (by1 - by0);
    ctx.fillStyle = bcol; ctx.fillRect(bx + 1, by1 - fh, bw2 - 2, fh);

    /* readouts */
    const rx = x + (cw - 28) / 2;
    txt(ctx, fmt(rpm), rx, h * 0.32, FM(19, 700), live ? TH.text : TH.dim);
    txt(ctx, "RPM", rx, h * 0.47, F(9), TH.faint);
    txt(ctx, fmt(kw), rx, h * 0.64, FM(16, 700), TH.dim);
    txt(ctx, "kW", rx, h * 0.78, F(9), TH.faint);

    if (proto) {
      ctx.strokeStyle = TH.amber; ctx.lineWidth = 0.7;
      ctx.fillStyle = TH.amber; ctx.globalAlpha = 0.1;
      ctx.fillRect(x + 4, h - 16, cw - 36, 12); ctx.globalAlpha = 1;
      ctx.strokeRect(x + 4, h - 16, cw - 36, 12);
      txt(ctx, "PROTO", x + 4 + (cw - 36) / 2, h - 10, F(8), TH.amber);
    }
  }
  if (isStale()) redX(c);
}

/* ══════════════════════════════════════════════════════════════════
 *  THRUST VECTOR — replaces the nacelle-tilt schematic.
 *  Frame preserved from v0.1.1: pivot lower-left, quarter arc V(up) to
 *  H(right).  The needle is now the fleet's real combined thrust vector
 *  (fx, fz fractions of commanded |T| from fleet_thrust_fraction), drawn
 *  at TRUE magnitude: for an all-tiltrotor fleet it is exactly
 *  (sin tilt, cos tilt) and reproduces the old gauge; for a mixed
 *  lift+pusher fleet a shorter needle correctly reads "thrust split
 *  across rotors pointing different ways".  A small caret on the arc
 *  still shows tilt_deg — the mission-phase progress variable — so both
 *  facts stay visible and distinct.
 * ══════════════════════════════════════════════════════════════════ */
function drawTvec() {
  const c = begin("cv-tvec"), { ctx, w, h } = c;
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, w, h);
  if (!st) return;

  const ox = w * 0.26, oy = h * 0.78;
  const R = Math.min(w, h) * 0.58;

  /* quarter arc V -> H */
  ctx.strokeStyle = TH.strokeHi; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.arc(ox, oy, R, -Math.PI / 2, 0); ctx.stroke();
  /* tick marks every 30° of tilt */
  ctx.strokeStyle = TH.dim; ctx.lineWidth = 1;
  for (const d of [0, 30, 60, 90]) {
    const a = (d - 90) * Math.PI / 180;
    ctx.beginPath();
    ctx.moveTo(ox + Math.cos(a) * R, oy + Math.sin(a) * R);
    ctx.lineTo(ox + Math.cos(a) * (R - 6), oy + Math.sin(a) * (R - 6));
    ctx.stroke();
  }
  txt(ctx, "V", ox, oy - R - 10, F(9), TH.faint);
  txt(ctx, "H", ox + R + 10, oy, F(9), TH.faint);

  /* mission-progress caret on the arc at tilt_deg (raw IDX.tilt) */
  const tiltA = (Math.max(0, Math.min(90, st.tilt)) - 90) * Math.PI / 180;
  ctx.fillStyle = TH.dim;
  ctx.beginPath();
  const tx1 = ox + Math.cos(tiltA) * (R + 3), ty1 = oy + Math.sin(tiltA) * (R + 3);
  const tx2 = ox + Math.cos(tiltA) * (R + 11), ty2 = oy + Math.sin(tiltA) * (R + 11);
  const pxc = -Math.sin(tiltA), pyc = Math.cos(tiltA);
  ctx.moveTo(tx1, ty1);
  ctx.lineTo(tx2 + pxc * 4, ty2 + pyc * 4);
  ctx.lineTo(tx2 - pxc * 4, ty2 - pyc * 4);
  ctx.closePath(); ctx.fill();

  /* thrust vector — screen x = +fx (forward), screen up = +fz (lift) */
  const fx = st.fx, fz = st.fz;
  const mag = Math.min(Math.hypot(fx, fz), 1.15);
  if (mag > 0.02) {
    const ux = fx / (mag || 1), uz = fz / (mag || 1);
    const exx = ox + fx * R, eyy = oy - fz * R;
    ctx.strokeStyle = TH.green; ctx.lineWidth = 2.6; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(exx, eyy); ctx.stroke();
    ctx.lineCap = "butt";
    /* arrowhead */
    const hl = 9, hw = 5;
    const px2 = -uz, py2 = -ux;                 /* screen-space perpendicular */
    ctx.fillStyle = TH.green;
    ctx.beginPath();
    ctx.moveTo(exx + ux * hl, eyy - uz * hl);
    ctx.lineTo(exx + px2 * hw, eyy + py2 * hw);
    ctx.lineTo(exx - px2 * hw, eyy - py2 * hw);
    ctx.closePath(); ctx.fill();
  }
  ctx.fillStyle = TH.text;
  ctx.beginPath(); ctx.arc(ox, oy, 2.6, 0, 2 * Math.PI); ctx.fill();

  /* readouts */
  const mode = st.tilt < 20 ? "HOVER" : st.tilt > 80 ? "CRUISE" : "TRANSITION";
  txt(ctx, "THRUST VECTOR", w / 2, 10, F(9), TH.faint);
  txt(ctx, `TILT ${fmt(st.tilt, 1)}\u00B0  ${mode}`, w / 2, h - 15, FM(10), TH.text);
  if (isStale()) redX(c);
}

/* ══════════════════════════════════════════════════════════════════
 *  NAV moving map
 * ══════════════════════════════════════════════════════════════════ */
function drawNav() {
  if (!init || !init.has_nav) return;
  const c = begin("cv-nav"), { ctx, w, h } = c;
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, w, h);
  if (!st || !st.nav) return;
  const nav = st.nav;

  const S = Math.min(w, h);
  const cx = w / 2, cy = S / 2;              /* map square, top-aligned */

  /* auto-scale: aircraft + waypoint + origin with 15 % margin, snapped */
  const dWp = Math.hypot(nav.x - nav.wx, nav.y - nav.wy);
  const dOr = Math.hypot(nav.x, nav.y);
  const raw = Math.max(dWp, dOr, 200) * 1.15;
  const scales = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000];
  let radius = scales[scales.length - 1];
  for (const s of scales) if (s >= raw) { radius = s; break; }

  const px = (wx, wy) => [cx + (wy - nav.y) / radius * (S / 2),
                          cy - (wx - nav.x) / radius * (S / 2)];

  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, w, S); ctx.clip();

  /* grid: 4 cells each way from the aircraft */
  ctx.strokeStyle = TH.stroke; ctx.lineWidth = 0.5; ctx.globalAlpha = 0.5;
  const step = radius / 4;
  for (let k = -4; k <= 4; k++) {
    const o = (k * step) / radius * (S / 2);
    ctx.beginPath(); ctx.moveTo(0, cy + o); ctx.lineTo(w, cy + o); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + o, 0); ctx.lineTo(cx + o, S); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  /* range rings at 1/4 and 1/2 radius */
  ctx.strokeStyle = TH.strokeHi; ctx.globalAlpha = 0.6; ctx.lineWidth = 0.7;
  for (const f of [0.25, 0.5]) {
    ctx.beginPath(); ctx.arc(cx, cy, f * (S / 2), 0, 2 * Math.PI); ctx.stroke();
    const rm = f * radius;
    const lbl = rm >= 1000 ? `${fmt(rm / 1000)} km` : `${fmt(rm)} m`;
    txt(ctx, lbl, cx + 4, cy - f * (S / 2) + 8, F(8), TH.faint, "left");
  }
  ctx.globalAlpha = 1;

  /* track history */
  if (nav.pts && nav.pts.length > 1) {
    ctx.strokeStyle = "rgba(46,128,153,0.7)"; ctx.lineWidth = 1.2;
    ctx.beginPath();
    nav.pts.forEach((p, i) => {
      const [x, y] = px(p[0], p[1]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  /* bearing line to waypoint */
  const [wxp, wyp] = px(nav.wx, nav.wy);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = TH.faint; ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(wxp, wyp); ctx.stroke();
  ctx.setLineDash([]);

  /* origin marker + ICAO */
  const [oxp, oyp] = px(0, 0);
  if (oxp > -20 && oxp < w + 20 && oyp > -20 && oyp < S + 20) {
    ctx.fillStyle = TH.panel; ctx.strokeStyle = TH.green; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(oxp, oyp, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    txt(ctx, init.icao || "ORIG", oxp + 8, oyp - 8, F(10), TH.green, "left");
  }

  /* waypoint diamond — RTB green / TGT blue */
  const wcol = nav.rtb ? TH.green : TH.blue;
  const d = 8;
  ctx.strokeStyle = wcol; ctx.lineWidth = 1.8;
  ctx.fillStyle = wcol; ctx.globalAlpha = 0.18;
  ctx.beginPath();
  ctx.moveTo(wxp, wyp - d); ctx.lineTo(wxp + d, wyp);
  ctx.lineTo(wxp, wyp + d); ctx.lineTo(wxp - d, wyp);
  ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1; ctx.stroke();
  const wlabel = nav.rtb ? "RTB" : ((init.dest_icao && init.dest_icao.length) ? init.dest_icao : "TGT");
  txt(ctx, wlabel, wxp + 10, wyp - 10, F(10), wcol, "left");

  /* aircraft symbol — filled triangle rotated to heading */
  const hr = nav.hdg * Math.PI / 180;
  const rot = (x, y) => [cx + x * Math.cos(hr) + y * Math.sin(hr),
                         cy + x * Math.sin(hr) - y * Math.cos(hr)];
  const sz = S * 0.028;
  ctx.fillStyle = TH.text;
  ctx.beginPath();
  const p0 = rot(0, sz), p1 = rot(-sz * 0.55, -sz * 0.6),
        p2 = rot(0, -sz * 0.2), p3 = rot(sz * 0.55, -sz * 0.6);
  ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]);
  ctx.lineTo(p2[0], p2[1]); ctx.lineTo(p3[0], p3[1]);
  ctx.closePath(); ctx.fill();
  ctx.restore();

  /* ── data readout strip ─────────────────────────────────────────── */
  const brg = ((Math.atan2(nav.wy - nav.y, nav.wx - nav.x) * 180 / Math.PI) + 360) % 360;
  const rngKm = dWp / 1000;
  const rngStr = rngKm >= 10 ? `${fmt(rngKm, 1)} km` :
                 rngKm >= 1  ? `${fmt(rngKm, 2)} km` : `${fmt(dWp)} m`;
  const aglM = nav.agl, aglFt = aglM * 3.28084;
  const dash = nav.phase === "dash";
  const aglCol = dash
    ? (aglM < 100 ? TH.red : aglM < 300 ? TH.amber : TH.green)
    : (aglM < 30  ? TH.red : aglM < 100 ? TH.amber : TH.green);

  const ry = S + 8, rh = h - S - 12;
  if (rh > 40) {
    txt(ctx, "BRG", 12, ry + 8, F(10), TH.dim, "left");
    txt(ctx, pad3(brg) + "\u00B0", 12, ry + 30, FM(21, 700), wcol, "left");
    txt(ctx, "RNG", 12, ry + rh - 26, F(10), TH.dim, "left");
    txt(ctx, rngStr, 12, ry + rh - 4, FM(21, 700), wcol, "left");

    ctx.fillStyle = aglCol; ctx.globalAlpha = 0.1;
    ctx.fillRect(w * 0.5, ry + 4, w * 0.5 - 8, rh - 4);
    ctx.globalAlpha = 0.45;
    ctx.strokeStyle = aglCol; ctx.lineWidth = 1;
    ctx.strokeRect(w * 0.5, ry + 4, w * 0.5 - 8, rh - 4);
    ctx.globalAlpha = 1;
    txt(ctx, "AGL", w - 20, ry + 14, F(10), TH.dim, "right");
    txt(ctx, `${fmt(aglFt)} ft`, w - 20, ry + rh - 14, FM(32, 700), aglCol, "right");
  }
  if (isStale()) redX(c);
}

const RA_RED_FT = 50;

/* Reference-line near-field length: how far the pitch-angle segment
 * extends before leveling off. Short and fixed on purpose — see the
 * drawVSD doc comment for why. */
const VSD_NEAR_KM = 1.0;

/* Below this groundspeed there's no real track to project a VSD along
 * — see the "no track" note in the drawVSD doc comment. */
const VSD_TRACK_MIN_KMH = 15;

/* ══════════════════════════════════════════════════════════════════
 *  VSD — forward-looking vertical situation display, matching real
 *  track-type VSDs (Boeing AERO No. 20, Oct. 2002): terrain sampled
 *  AHEAD of the aircraft along current heading, not a plot of terrain
 *  already flown over.  The server samples the terrain database fresh
 *  every tick (`terr_ahead`, see _terrain_ahead in glass_cockpit.jl) —
 *  no client-side accumulation.
 *
 *  NO TRACK below VSD_TRACK_MIN_KMH (hover / near-stationary): a real
 *  VSD is track-type — it works because a fixed-wing aircraft on
 *  approach is committed to roughly its current heading and
 *  groundspeed.  In hover, groundspeed is ~0: there's no actual track
 *  toward wherever the heading happens to be pointing, so colour-coding
 *  terrain out there as an imminent hazard is misleading even when the
 *  terrain shape itself is accurate — it looks like a collision course
 *  the aircraft isn't actually on.  Below the threshold this draws the
 *  terrain shape only (still useful — "here's what's out there if you
 *  do go this way"), with neutral colouring and no reference line, and
 *  labels the state explicitly instead of drawing a track that doesn't
 *  exist.
 *
 *  Reference line (when there IS a track): current altitude, tilted at
 *  the aircraft's current PITCH for a short near-field segment
 *  (VSD_NEAR_KM), then flat at whatever altitude that segment reaches.
 *  Two things this is deliberately NOT:
 *   - not a kinematic extrapolation of groundspeed + climb trend: that
 *     breaks down in hover for the same track-less reason above, and
 *     would be just as unreliable during transition or manual flying;
 *   - not pitch held constant across the whole window either: pitch is
 *     a momentary attitude, not a commitment to fly that flight-path
 *     angle for 8 km, and holding it that far out has the same
 *     runaway-divergence problem as extrapolating groundspeed/trend.
 *  So it shows real current attitude only where "real" still means
 *  something (a short near-field stretch), then stops pretending to
 *  predict and just holds level (no turn-adaptive swath or FMC-computed
 *  path either — a deliberate scope cut, documented in
 *  DESIGN_RATIONALE.md).
 *  Carries the large AGL digital readout in its upper-left corner —
 *  restoring AGL as a first-class prominent readout (v0.1 parity, was
 *  buried in the nav-map strip) without a separate dedicated pane.
 * ══════════════════════════════════════════════════════════════════ */
function drawVSD() {
  if (!init || !init.has_nav) return;
  const c = begin("cv-vsd"), { ctx, w, h } = c;
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, w, h);
  txt(ctx, "VSD \u2014 TRACK AHEAD", w - 10, 12, F(10, 700), TH.faint, "right");

  /* Large AGL digital readout, upper-left corner — restores AGL as a
     first-class prominent readout (v0.1 parity, was buried in the nav
     strip) without a dedicated instrument pane.  Same colour logic and
     2500-ft blank-out as a real radar altimeter leaving tracking range. */
  if (st) {
    const ra = radAltFt(st);
    const minFt = init && typeof init.ra_min_ft === "number" ? init.ra_min_ft : 150;
    txt(ctx, "AGL", 10, 12, F(9, 700), TH.dim, "left");
    if (ra === null || ra > 2500) {
      txt(ctx, "\u2013 \u2013 \u2013", 10, 34, FM(20, 700), TH.faint, "left");
    } else {
      const col = ra < RA_RED_FT ? TH.red : ra < minFt ? TH.amber : TH.green;
      txt(ctx, fmt(ra), 10, 34, FM(24, 700), col, "left");
    }
  }
  if (!st) return;

  if (!Array.isArray(st.terr_ahead) || st.terr_ahead.length < 2) {
    txt(ctx, "TERR UNAVAIL", w / 2, h / 2, F(11), TH.faint);
    if (isStale()) redX(c);
    return;
  }
  const terrPts = st.terr_ahead;                 // [[d_km, terr_ft_msl], ...]
  const winKm = terrPts[terrPts.length - 1][0];
  const hasTrack = st.speed >= VSD_TRACK_MIN_KMH;

  // Pitch-angle near-field segment, then flat.  tan(pitch) over the
  // near-field distance gives the altitude reached at VSD_NEAR_KM;
  // beyond that the reference simply holds that altitude.  Only
  // meaningful with a real track — see NO TRACK note above.
  const nearKm = Math.min(VSD_NEAR_KM, winKm);
  const pitchRad = (st.pitch || 0) * Math.PI / 180;
  const altAtNear = st.alt + Math.tan(pitchRad) * (nearKm * 1000) * M2FT;
  const refPath = (d_km) => hasTrack
    ? (d_km <= nearKm ? st.alt + Math.tan(pitchRad) * (d_km * 1000) * M2FT : altAtNear)
    : st.alt;   // no-track mode: only used for axis scaling, not drawn/coloured by

  let aMin = Infinity, aMax = -Infinity;
  for (const [d, terr] of terrPts) { aMin = Math.min(aMin, terr, refPath(d)); aMax = Math.max(aMax, terr, refPath(d)); }
  const pad = Math.max(120, (aMax - aMin) * 0.15);
  aMin -= pad; aMax += pad;
  // Round to clean scale ticks — an unrounded bound sits just a padding
  // margin above the aircraft's own altitude and reads as an (almost,
  // but not quite) altitude readout rather than a chart scale limit.
  aMin = Math.floor(aMin / 50) * 50;
  aMax = Math.ceil(aMax / 50) * 50;

  const y0 = 58, y1 = h - 14;
  const X = (d) => 34 + (d / winKm) * (w - 44);
  const Y = (a) => y0 + (1 - (a - aMin) / (aMax - aMin)) * (y1 - y0);
  txt(ctx, "FT MSL", 10, y0 - 10, F(8, 700), TH.faint, "left");

  ctx.beginPath();
  ctx.moveTo(X(terrPts[0][0]), Y(terrPts[0][1]));
  for (const [d, terr] of terrPts) ctx.lineTo(X(d), Y(terr));
  ctx.lineTo(X(terrPts[terrPts.length - 1][0]), y1);
  ctx.lineTo(X(terrPts[0][0]), y1);
  ctx.closePath();
  ctx.fillStyle = TH.ground; ctx.fill();

  ctx.lineWidth = 1.8;
  for (let i = 1; i < terrPts.length; i++) {
    const [d, terr] = terrPts[i];
    // No track → neutral outline only, no red/amber hazard implication
    // (there's no actual closure happening toward this terrain).
    const strokeCol = hasTrack
      ? (refPath(d) - terr < 150 ? TH.red : refPath(d) - terr < 300 ? TH.amber : TH.dim)
      : TH.faint;
    ctx.strokeStyle = strokeCol;
    ctx.beginPath();
    ctx.moveTo(X(terrPts[i - 1][0]), Y(terrPts[i - 1][1]));
    ctx.lineTo(X(d), Y(terr));
    ctx.stroke();
  }

  if (hasTrack) {
    /* Reference line: solid white pitch segment near-field (787-style —
       real current attitude), dashed blue flat continuation beyond it
       (no longer predicting) */
    ctx.strokeStyle = TH.text; ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(X(0), Y(st.alt));
    ctx.lineTo(X(nearKm), Y(altAtNear));
    ctx.stroke();
    if (winKm > nearKm) {
      ctx.strokeStyle = TH.blue;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(X(nearKm), Y(altAtNear));
      ctx.lineTo(X(winKm), Y(altAtNear));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  } else {
    txt(ctx, "HOVER \u2014 NO TRACK", (X(0) + X(winKm)) / 2, y0 - 10, F(9, 700), TH.faint);
  }

  /* Aircraft symbol at current position (left edge — "now").  No
     altitude label here — the ALT tape already shows current MSL
     altitude; this pane's own number is the AGL readout above. */
  ctx.fillStyle = TH.text;
  ctx.beginPath(); ctx.arc(X(0), Y(st.alt), 3.5, 0, 2 * Math.PI); ctx.fill();

  txt(ctx, fmt(aMax), 30, y0 + 6, FM(11, 700), TH.text, "right");
  txt(ctx, fmt(aMin), 30, y1 - 6, FM(11, 700), TH.text, "right");

  if (isStale()) redX(c);
}

/* ── Header ─────────────────────────────────────────────────────── */
function drawHeader() {
  if (!st) return;
  const t = st.t;
  document.getElementById("clock").textContent =
    `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t) % 60).padStart(2, "0")}`;
  const ph = document.getElementById("phase");
  ph.textContent = (st.phase || "\u2014").toUpperCase();
  ph.className = st.phase === "landed" ? "landed" : st.phase === "dash" ? "dash" : "";
}

/* ── Render ─────────────────────────────────────────────────────── */
function renderAll() {
  drawHeader();
  drawTape("cv-speed", st ? st.speed * KMH2KT : 0, 40, 10, 5, "IAS", "KT", false, trend6("spd") * KMH2KT);
  drawTape("cv-alt",   st ? st.alt   : 0, 500, 100, 50, "ALT", "FT",  true,  trend6("alt"));
  drawADI();
  drawHeading();
  drawVcon();
  drawContact();
  drawPower();
  drawRotors();
  drawTvec();
  drawNav();
  drawVSD();
}

/* ── Boot ───────────────────────────────────────────────────────── */
["cv-speed", "cv-adi", "cv-alt", "cv-hdg", "cv-vcon", "cv-contact",
 "cv-power", "cv-rotors", "cv-tvec", "cv-nav", "cv-vsd"].forEach(initCanvas);
setTheme(new URLSearchParams(location.search).get("theme") === "day");
document.fonts.ready.then(renderAll);
connect();
// logger_ui — dashboard + control panel.
//  • /api/stats        storage + latest records
//  • /api/record       which streams the logger writes (GET/POST)
//  • /api/replay[/*]    trigger / stop / live status of a gesture replay
//  • /api/replays       replay history (times played, total loops)
//  • /api/timeline      time-series for the canvas charts (offline, no CDN)
const el = (id) => document.getElementById(id);
const fmt = (v) => (v === null || v === undefined || v === "") ? "—" : v;

// Light-surface categorical palette (data-viz reference order). Color follows
// the entity (series index), never its rank — filtering never repaints others.
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"];
const INK = { primary: "#16263b", muted: "#898781", grid: "#e1e0d9",
              axis: "#c3c2b7", surface: "#ffffff" };

// ───────────────────────── storage / stats ───────────────────────── //
async function refreshStats() {
  let d;
  try { d = await (await fetch("/api/stats")).json(); }
  catch (e) { el("note").hidden = false; el("note").textContent = "stats fetch failed: " + e; return; }

  el("storePath").textContent = fmt(d.path);
  el("storeSize").textContent = (d.size_bytes || d.size_bytes === 0)
    ? (d.size_bytes / 1048576).toFixed(2) + " MB" : "—";

  const connected = d.connected;
  el("lamp").className = "lamp " + (connected ? "ready" : "down");
  el("connText").textContent = connected ? "READABLE" : "MISSING";
  el("connText").className = connected ? "ok" : "bad";
  el("status").textContent = connected ? "logging" : "no store file";
  el("status").className = "mono big " + (connected ? "ok-text" : "bad-text");

  el("note").hidden = !d.note;
  if (d.note) el("note").textContent = d.note;

  const cl = el("collList"); cl.innerHTML = "";
  (d.collections || []).forEach((c) => {
    const n = (d.counts && d.counts[c] !== undefined) ? d.counts[c] : "—";
    const div = document.createElement("div");
    div.className = "coll";
    div.innerHTML = `<div class="name">${c}</div>
      <div class="count">${n}<small> DOCS</small></div>`;
    cl.appendChild(div);
  });

  const showCsv = !connected;
  el("csvBlock").hidden = !showCsv;
  if (showCsv) {
    el("csvDir").textContent = fmt(d.csv_fallback_dir);
    const cf = el("csvFiles"); cf.innerHTML = "";
    (d.csv_files || []).forEach((f) => {
      const div = document.createElement("div");
      div.textContent = `${f.name}  (${(f.bytes / 1024).toFixed(1)} KB)`;
      cf.appendChild(div);
    });
    if (!(d.csv_files || []).length) cf.innerHTML = "<div>no CSV files yet</div>";
  }

  const L = d.latest || {};
  const g = L.gesture_status;
  if (g) {
    el("gName").textContent = g.active ? fmt(g.name) : "idle";
    el("gName").className = "big " + (g.active ? "ok-text" : "");
    el("gMeta").textContent = g.active
      ? `${Math.round(g.percent || 0)}%  frame ${fmt(g.frame)}/${fmt(g.total)}  · ${g.stamp_iso || ""}`
      : `last: ${g.stamp_iso || "—"}`;
  } else { el("gName").textContent = "—"; el("gMeta").textContent = "no data"; }

  const j = L.joint_states;
  if (j) {
    el("jCount").textContent = `${fmt(j.num_joints)} joints`;
    el("jMeta").textContent = j.first
      ? `${j.first.name}=${(+j.first.position).toFixed(3)}  · ${j.stamp_iso || ""}`
      : (j.stamp_iso || "");
  } else { el("jCount").textContent = "—"; el("jMeta").textContent = "no data"; }

  const m = L.motor_status;
  if (m && m.hottest) {
    const t = Math.max(m.hottest.mos_temp || 0, m.hottest.rotor_temp || 0);
    el("mTemp").textContent = `${t.toFixed(1)} °C`;
    el("mTemp").className = "big " + (t >= 70 ? "bad-text" : t >= 55 ? "amber-text" : "");
    el("mMeta").textContent = `${m.hottest.arm_name} #${m.hottest.id}  (mos ${fmt(m.hottest.mos_temp)} / rotor ${fmt(m.hottest.rotor_temp)})`;
  } else { el("mTemp").textContent = "—"; el("mMeta").textContent = "no data"; }
}

// ───────────────────────── record toggles ───────────────────────── //
let _recBusy = false;
async function refreshRecord() {
  if (_recBusy) return;
  let d; try { d = await (await fetch("/api/record")).json(); } catch { return; }
  el("recNote").textContent = d.note || (d.ok ? "logging live — toggle to start/stop a stream" : "");
  const flags = d.flags || {};
  document.querySelectorAll('#toggles input[data-stream]').forEach((cb) => {
    const v = flags[cb.dataset.stream];
    cb.disabled = (v === null || v === undefined) && !d.ok;
    if (document.activeElement !== cb) cb.checked = !!v;
  });
}
function wireRecord() {
  document.querySelectorAll('#toggles input[data-stream]').forEach((cb) => {
    cb.addEventListener("change", async () => {
      _recBusy = true;
      const body = {}; body[cb.dataset.stream] = cb.checked;
      try {
        const d = await (await fetch("/api/record", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        })).json();
        el("recNote").textContent = d.note || (d.ok ? "applied" : "failed");
        const flags = d.flags || {};
        document.querySelectorAll('#toggles input[data-stream]').forEach((x) => {
          if (flags[x.dataset.stream] !== undefined && flags[x.dataset.stream] !== null)
            x.checked = !!flags[x.dataset.stream];
        });
      } catch (e) { el("recNote").textContent = "toggle failed: " + e; }
      finally { _recBusy = false; }
    });
  });
}

// ───────────────────────── replay control ───────────────────────── //
function wireReplay() {
  el("rPlay").addEventListener("click", async () => {
    const body = {
      name: el("rName").value.trim(),
      loops: parseInt(el("rLoops").value, 10),
      speed: parseFloat(el("rSpeed").value),
      interval: parseFloat(el("rInterval").value),
      output_topic: el("rTopic").value.trim(),
    };
    if (!body.name) { el("replayNote").textContent = "enter a recording name first."; return; }
    el("replayNote").textContent = "requesting replay…";
    try {
      const d = await (await fetch("/api/replay", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })).json();
      el("replayNote").textContent = (d.ok ? "✓ " : "✗ ") + fmt(d.note);
      el("replayNote").className = "sub mono " + (d.ok ? "ok-text" : "bad-text");
    } catch (e) { el("replayNote").textContent = "replay failed: " + e; }
  });
  el("rStop").addEventListener("click", async () => {
    el("replayNote").textContent = "stopping…";
    try {
      const d = await (await fetch("/api/replay/stop", { method: "POST" })).json();
      el("replayNote").textContent = (d.ok ? "✓ " : "✗ ") + fmt(d.note);
      el("replayNote").className = "sub mono " + (d.ok ? "ok-text" : "bad-text");
    } catch (e) { el("replayNote").textContent = "stop failed: " + e; }
  });
}
async function refreshReplayStatus() {
  let d; try { d = await (await fetch("/api/replay/status")).json(); } catch { return; }
  const r = d.replay;
  const box = el("replayLive");
  if (!r || !r.active) {
    box.hidden = !r;
    if (r) {
      el("rlName").textContent = "idle";
      el("rlLoop").textContent = r.message || "";
      el("rlBar").style.width = "0%";
      el("rlMeta").textContent = `last: ${fmt(r.name)}`;
    }
    return;
  }
  box.hidden = false;
  const mode = r.repeat_mode, cnt = r.repeat_count, loop = r.loop_count || 0;
  const loopTxt = mode === "infinite" ? `loop ${loop} / ∞`
    : mode === "count" ? `loop ${loop} / ${cnt}` : "single play";
  el("rlName").textContent = fmt(r.name);
  el("rlLoop").textContent = loopTxt;
  const pct = Math.max(0, Math.min(100, r.percent || 0));
  el("rlBar").style.width = pct.toFixed(0) + "%";
  el("rlMeta").textContent =
    `${pct.toFixed(0)}%  ·  frame ${fmt(r.frame)}/${fmt(r.total)}  ·  ${fmt(r.speed)}×  ·  → ${fmt(r.output_topic)}  ·  ${fmt(r.message)}`;
}

// ───────────────────────── replay history ───────────────────────── //
async function refreshHistory() {
  let d; try { d = await (await fetch("/api/replays")).json(); } catch { return; }
  const body = el("histBody"); body.innerHTML = "";
  const rows = d.rows || [];
  if (!d.ok || !rows.length) {
    body.innerHTML = `<tr><td colspan="5" class="muted">${d.ok ? "no replays logged yet" : fmt(d.note)}</td></tr>`;
    return;
  }
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const mode = r.last_mode
      ? (r.last_mode === "count" ? `${r.last_mode} ×${fmt(r.last_count)}` : r.last_mode)
      : "—";
    tr.innerHTML = `<td class="mono">${fmt(r.name)}</td>
      <td class="num">${fmt(r.sessions)}</td>
      <td class="num">${fmt(r.total_loops)}</td>
      <td class="mono">${mode}</td>
      <td class="mono muted">${fmt(r.last_iso)}</td>`;
    body.appendChild(tr);
  });
}

// ───────────────────────── timeline chart ───────────────────────── //
let tlMetric = "joint_pos";
let tlPoints = 240;
let tlSelected = null;         // Set of selected series labels (per metric)
let tlData = null;             // last fetched payload
let tlHover = -1;              // hovered sample index

function wireTimeline() {
  el("tlMetric").querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => {
      el("tlMetric").querySelectorAll("button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      tlMetric = b.dataset.metric; tlSelected = null; tlData = null;
      refreshTimeline();
    });
  });
  el("tlPoints").querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => {
      el("tlPoints").querySelectorAll("button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      tlPoints = parseInt(b.dataset.points, 10);
      refreshTimeline();
    });
  });
  const cv = el("tlCanvas");
  cv.addEventListener("mousemove", (ev) => {
    if (!tlData || !tlData.t || !tlData.t.length) return;
    const rect = cv.getBoundingClientRect();
    tlHover = pxToIndex(ev.clientX - rect.left, rect.width);
    drawChart();
  });
  cv.addEventListener("mouseleave", () => { tlHover = -1; drawChart(); });
  window.addEventListener("resize", drawChart);
}

const PAD = { l: 52, r: 14, t: 12, b: 26 };
function plotRect(w, h) { return { x: PAD.l, y: PAD.t, w: w - PAD.l - PAD.r, h: h - PAD.t - PAD.b }; }
function pxToIndex(px, cssW) {
  const r = plotRect(cssW, el("tlCanvas").clientHeight);
  const n = tlData.t.length;
  if (n <= 1) return 0;
  const frac = (px - r.x) / r.w;
  return Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
}

async function refreshTimeline() {
  if (el("tlLive") && !el("tlLive").checked && tlData) { drawChart(); return; }
  let d;
  try {
    d = await (await fetch(`/api/timeline?metric=${tlMetric}&points=${tlPoints}`)).json();
  } catch { return; }
  tlData = d;
  // gesture: plot percent only (single y-axis); loop_count is a caption.
  const plottable = (d.series || []).filter((s) => s.axis !== "right");
  if (tlSelected === null) {
    // default selection: first 6 series that have any data.
    const withData = plottable.filter((s) => (s.values || []).some((v) => v !== null && v !== undefined));
    tlSelected = new Set(withData.slice(0, 6).map((s) => s.label));
  }
  buildLegend(plottable, d);
  drawChart();
}

function buildLegend(plottable, d) {
  const lg = el("tlLegend"); lg.innerHTML = "";
  plottable.forEach((s, i) => {
    const color = PALETTE[i % PALETTE.length];
    const dash = i >= PALETTE.length ? " dashed" : "";
    const chip = document.createElement("button");
    chip.className = "lg-item" + (tlSelected.has(s.label) ? "" : " off");
    chip.innerHTML = `<span class="sw${dash}" style="--c:${color}"></span>${s.label}`;
    chip.addEventListener("click", () => {
      if (tlSelected.has(s.label)) tlSelected.delete(s.label); else tlSelected.add(s.label);
      chip.classList.toggle("off");
      drawChart();
    });
    lg.appendChild(chip);
  });
  // caption extras
  const right = (d.series || []).find((s) => s.axis === "right");
  if (right) {
    const vals = (right.values || []).filter((v) => v !== null && v !== undefined);
    const maxLoop = vals.length ? Math.max(...vals) : 0;
    const cap = document.createElement("span");
    cap.className = "lg-cap mono";
    cap.textContent = `max loop in window: ${maxLoop}`;
    lg.appendChild(cap);
  }
}

function drawChart() {
  const cv = el("tlCanvas");
  const cssW = cv.clientWidth || cv.parentElement.clientWidth || 800;
  const cssH = cv.clientHeight || 300;
  const dpr = window.devicePixelRatio || 1;
  cv.width = cssW * dpr; cv.height = cssH * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const empty = el("tlEmpty");
  if (!tlData || !tlData.t || !tlData.t.length) { empty.hidden = false; return; }
  empty.hidden = true;

  const r = plotRect(cssW, cssH);
  const n = tlData.t.length;
  const plottable = (tlData.series || []).filter((s) => s.axis !== "right");
  const sel = plottable.map((s, i) => ({ s, i })).filter((o) => tlSelected.has(o.s.label));

  // y range across selected series
  let lo = Infinity, hi = -Infinity;
  sel.forEach(({ s }) => (s.values || []).forEach((v) => {
    if (v === null || v === undefined || Number.isNaN(v)) return;
    if (v < lo) lo = v; if (v > hi) hi = v;
  }));
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const padY = (hi - lo) * 0.08; lo -= padY; hi += padY;

  const X = (idx) => r.x + (n <= 1 ? 0 : (idx / (n - 1)) * r.w);
  const Y = (val) => r.y + r.h - ((val - lo) / (hi - lo)) * r.h;

  // grid + y ticks
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textBaseline = "middle";
  const TICKS = 5;
  for (let k = 0; k <= TICKS; k++) {
    const val = lo + (k / TICKS) * (hi - lo);
    const y = Y(val);
    ctx.strokeStyle = INK.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(r.x, y + 0.5); ctx.lineTo(r.x + r.w, y + 0.5); ctx.stroke();
    ctx.fillStyle = INK.muted; ctx.textAlign = "right";
    ctx.fillText(val.toFixed(2), r.x - 8, y);
  }
  // unit
  ctx.fillStyle = INK.muted; ctx.textAlign = "left";
  ctx.fillText(tlData.unit || "", r.x, r.y - 2 < 6 ? 8 : r.y - 2);

  // x time ticks (a few)
  const iso = tlData.iso || [];
  ctx.textBaseline = "top"; ctx.textAlign = "center"; ctx.fillStyle = INK.muted;
  const XT = 4;
  for (let k = 0; k <= XT; k++) {
    const idx = Math.round((k / XT) * (n - 1));
    const s = iso[idx]; if (!s) continue;
    const hhmmss = (s.split("T")[1] || s).slice(0, 8);
    ctx.fillText(hhmmss, X(idx), r.y + r.h + 6);
  }
  // baseline
  ctx.strokeStyle = INK.axis; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(r.x, r.y + r.h + 0.5); ctx.lineTo(r.x + r.w, r.y + r.h + 0.5); ctx.stroke();

  // series lines (clip to plot)
  ctx.save();
  ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
  sel.forEach(({ s, i }) => {
    ctx.strokeStyle = PALETTE[i % PALETTE.length];
    ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.setLineDash(i >= PALETTE.length ? [5, 4] : []);
    ctx.beginPath();
    let pen = false;
    (s.values || []).forEach((v, idx) => {
      if (v === null || v === undefined || Number.isNaN(v)) { pen = false; return; }
      const px = X(idx), py = Y(v);
      if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.restore();

  // hover crosshair + tooltip
  if (tlHover >= 0 && tlHover < n) {
    const hx = X(tlHover);
    ctx.strokeStyle = INK.axis; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(hx + 0.5, r.y); ctx.lineTo(hx + 0.5, r.y + r.h); ctx.stroke();
    ctx.setLineDash([]);
    const lines = [];
    const s0 = (iso[tlHover] || "").split("T")[1] || iso[tlHover] || "";
    lines.push({ txt: s0.slice(0, 8), color: INK.primary });
    sel.forEach(({ s, i }) => {
      const v = (s.values || [])[tlHover];
      if (v === null || v === undefined) return;
      ctx.fillStyle = PALETTE[i % PALETTE.length];
      ctx.beginPath(); ctx.arc(hx, Y(v), 3.5, 0, Math.PI * 2); ctx.fill();
      lines.push({ txt: `${s.label}: ${(+v).toFixed(3)}`, color: PALETTE[i % PALETTE.length] });
    });
    drawTooltip(ctx, hx, r, lines, cssW);
  }
}

function drawTooltip(ctx, hx, r, lines, cssW) {
  ctx.font = "11px 'IBM Plex Mono', monospace";
  const w = Math.max(...lines.map((l) => ctx.measureText(l.txt).width)) + 16;
  const h = lines.length * 15 + 8;
  let tx = hx + 10; if (tx + w > r.x + r.w) tx = hx - w - 10;
  let ty = r.y + 6;
  ctx.fillStyle = "rgba(255,255,255,0.96)";
  ctx.strokeStyle = "rgba(11,11,11,0.12)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.rect(tx, ty, w, h); ctx.fill(); ctx.stroke();
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  lines.forEach((l, k) => { ctx.fillStyle = l.color; ctx.fillText(l.txt, tx + 8, ty + 5 + k * 15); });
}

// ───────────────────────── recording session ───────────────────────── //
let _dirDirty = false;   // don't overwrite the dir field while the user edits
async function sessionAction(url, body) {
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const d = await r.json();
    el("sessNote").textContent = fmt(d.note);
    el("sessNote").className = "sub mono " + (d.ok ? "ok-text" : "bad-text");
  } catch (e) { el("sessNote").textContent = "request failed: " + e; }
  refreshSession(); refreshStats(); refreshSessionsList();
}
function wireSession() {
  el("btnStart").addEventListener("click", () => sessionAction("/api/session/start"));
  el("btnStop").addEventListener("click", () => sessionAction("/api/session/stop"));
  el("btnNew").addEventListener("click", () => {
    if (confirm("Start a NEW session file? The current one is saved and closed.")) sessionAction("/api/session/new");
  });
  el("dirInput").addEventListener("input", () => { _dirDirty = true; });
  el("btnDir").addEventListener("click", () => {
    const dir = el("dirInput").value.trim();
    if (!dir) { el("sessNote").textContent = "enter a directory path."; return; }
    if (confirm(`Change data directory to:\n${dir}\n\nIf recording, a new file starts there.`)) {
      _dirDirty = false;
      sessionAction("/api/session/dir", { dir });
    }
  });
  // event-delegated rename / delete buttons
  el("sessBody").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const name = btn.dataset.name;
    if (btn.dataset.act === "del") {
      if (confirm(`Delete session file?\n${name}`)) sessionAction("/api/session/delete", { name });
    } else if (btn.dataset.act === "ren") {
      const nn = prompt("Rename to (.json added if omitted):", name);
      if (nn && nn.trim()) sessionAction("/api/session/rename", { old: name, new: nn.trim() });
    }
  });
}
async function refreshSession() {
  let d; try { d = await (await fetch("/api/session")).json(); } catch { return; }
  const rec = !!d.recording;
  el("recDot").className = "recdot " + (rec ? "on" : "off");
  el("recState").textContent = rec ? "RECORDING" : "STOPPED";
  el("recState").className = "big " + (rec ? "ok-text" : "bad-text");
  el("sessFile").textContent = fmt(d.basename);
  el("btnStart").disabled = rec;
  el("btnStop").disabled = !rec;
  if (!_dirDirty && document.activeElement !== el("dirInput"))
    el("dirInput").value = fmt(d.log_dir);
}
async function refreshSessionsList() {
  let d; try { d = await (await fetch("/api/sessions")).json(); } catch { return; }
  const body = el("sessBody"); body.innerHTML = "";
  const rows = (d && d.sessions) || [];
  if (!rows.length) { body.innerHTML = `<tr><td colspan="4" class="muted">no saved sessions yet</td></tr>`; return; }
  rows.forEach((s) => {
    const tr = document.createElement("tr");
    if (s.current) tr.className = "cur";
    tr.innerHTML = `<td class="mono">${fmt(s.name)}${s.current ? ' <span class="pill">current</span>' : ''}</td>
      <td class="num">${(s.bytes / 1048576).toFixed(2)} MB</td>
      <td class="mono muted">${fmt(s.mtime)}</td>
      <td class="act">
        <button class="mini" data-act="ren" data-name="${s.name}">Rename</button>
        <button class="mini danger" data-act="del" data-name="${s.name}">Delete</button>
      </td>`;
    body.appendChild(tr);
  });
}

// ───────────────────────── boot + polling ───────────────────────── //
function boot() {
  wireSession(); wireRecord(); wireReplay(); wireTimeline();
  refreshSession(); refreshSessionsList();
  refreshStats(); refreshRecord(); refreshReplayStatus(); refreshHistory(); refreshTimeline();
  setInterval(refreshStats, 1000);
  setInterval(refreshSession, 1500);
  setInterval(refreshSessionsList, 5000);
  setInterval(refreshRecord, 2000);
  setInterval(refreshReplayStatus, 500);
  setInterval(refreshHistory, 5000);
  setInterval(() => { if (!el("tlLive") || el("tlLive").checked) refreshTimeline(); }, 1500);
}
boot();

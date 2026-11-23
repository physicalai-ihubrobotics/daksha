// Camera Setup UI — scan RealSense hardware, preview feeds live, drag each
// feed into a role slot, save the serial->role map to cameras.yaml.

let DEVICES = [];                 // [{serial,name,usb,status,detail}]
let ASSIGN = {};                  // role_key -> serial|null
let SAVED = {};                   // role_key -> serial|null in cameras.yaml
let LAST_SCAN = {};               // last /api/scan reply

function takeScan(data) {
  LAST_SCAN = data;
  SAVED = data.saved || {};
}

const el = (id) => document.getElementById(id);
const streamURL = (serial) => `/stream/${serial}?t=${Date.now()}`;

// MJPEG streams never end, and a browser keeps at most 6 connections open per
// server, so every stream past that makes scan/assign/save wait in a queue.
// Only live cameras stream, once each: tiles hold the streams, and role slots
// copy frames from their tile (mirrorSlots) instead of opening their own.
function setStream(img, serial) {
  const want = serial || '';
  if (img.dataset.stream === want) return;
  img.dataset.stream = want;
  img.hidden = !serial;
  if (serial) img.src = streamURL(serial);
  else img.removeAttribute('src');   // closes the connection
}

function toast(msg, kind) {
  const t = el('toast');
  t.textContent = msg;
  t.className = 'toast' + (kind ? ' ' + kind : '');
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, 3200);
}

// --- render detected cameras ------------------------------------------- //
function roleOfSerial(serial) {
  for (const k in ASSIGN) if (ASSIGN[k] === serial) return k;
  return null;
}
function roleLabel(key) {
  const r = ROLES.find((x) => x.key === key);
  return r ? r.label : key;
}

function buildDeviceCard(d) {
  const card = document.createElement('div');
  card.className = 'dev';
  card.draggable = true;
  card.dataset.serial = d.serial;
  card.innerHTML = `
    <div class="feed">
      <span class="pill"></span>
      <span class="usb"></span>
      <span class="feed-note" style="position:absolute; inset:0; display:flex; align-items:center;
        padding:0 12px; color:#d9b38c; font-size:13px"></span>
      <img alt="feed" hidden>
    </div>
    <div class="meta">
      <div class="name"></div>
      <div class="serial"></div>
      <div class="assignedtag" hidden></div>
    </div>`;
  card.addEventListener('dragstart', (e) => {
    card.classList.add('dragging');
    e.dataTransfer.setData('text/plain', d.serial);
    e.dataTransfer.effectAllowed = 'move';
  });
  card.addEventListener('dragend', () => card.classList.remove('dragging'));
  return card;
}

// Incremental update: reuse existing tiles (so their MJPEG <img> keeps
// streaming) and only add/remove/restyle as the device set changes.
function renderDevices() {
  const grid = el('devices');
  el('devCount').textContent = DEVICES.length;
  el('devEmpty').hidden = DEVICES.length > 0;

  const seen = new Set();
  for (const d of DEVICES) {
    seen.add(d.serial);
    let card = grid.querySelector(`.dev[data-serial="${d.serial}"]`);
    if (!card) { card = buildDeviceCard(d); grid.appendChild(card); }

    const assignedRole = roleOfSerial(d.serial);
    card.classList.toggle('assigned', !!assignedRole);
    const st = d.status || 'connecting';
    const pill = card.querySelector('.pill');
    pill.className = 'pill ' + st;
    pill.textContent = st === 'live' ? 'live' : st;
    setStream(card.querySelector('.feed img'), st === 'live' ? d.serial : null);
    card.querySelector('.feed-note').textContent =
      st === 'live' ? '' : (st === 'busy' ? 'device busy' : st);
    card.querySelector('.feed-note').title = d.detail || '';
    card.querySelector('.usb').textContent = 'USB ' + (d.usb || '?');
    card.querySelector('.name').textContent = d.name || 'RealSense';
    card.querySelector('.serial').textContent = d.serial;
    const tag = card.querySelector('.assignedtag');
    tag.hidden = !assignedRole;
    if (assignedRole) tag.innerHTML = '&#10003; ' + roleLabel(assignedRole);
  }
  grid.querySelectorAll('.dev').forEach((c) => {
    if (!seen.has(c.dataset.serial)) c.remove();
  });
}

// --- render role slots ------------------------------------------------- //
function renderSlots() {
  for (const r of ROLES) {
    const slot = document.getElementById('slot-' + r.key);
    const drop = slot.querySelector('.slot-drop');
    const empty = slot.querySelector('.slot-empty');
    const img = slot.querySelector('.slot-feed');
    const serialEl = slot.querySelector('.slot-serial');
    const serial = ASSIGN[r.key];

    if (serial) {
      drop.classList.add('filled');
      empty.hidden = true;
      img.hidden = false;
      if (img.dataset.serial !== serial) {   // mirrorSlots fills in the frames
        img.removeAttribute('src');
        img.dataset.serial = serial;
      }
      const dev = DEVICES.find((d) => d.serial === serial);
      serialEl.textContent = serial + (dev ? '' : ' (offline)');
    } else {
      drop.classList.remove('filled');
      empty.hidden = false;
      img.hidden = true;
      img.removeAttribute('src');
      img.dataset.serial = '';
      serialEl.textContent = '';
    }
    // The slots show the unsaved draft; say when cameras.yaml still differs.
    const saved = SAVED[r.key] || null;
    let pending = slot.querySelector('.slot-pending');
    if (!pending) {
      pending = document.createElement('div');
      pending.className = 'slot-pending';
      pending.style.cssText = 'font-size:11px; color:#b45309; margin-top:2px';
      serialEl.after(pending);
    }
    pending.hidden = !('saved' in LAST_SCAN) || saved === (serial || null);
    pending.textContent = `cameras.yaml has ${saved || 'none'}: press Save to write this change`;
  }
}

// Copies the current frame of each assigned camera's tile into its slot
// thumbnail, so a slot never opens a second stream for the same camera.
const mirrorCanvas = document.createElement('canvas');
function mirrorSlots() {
  for (const r of ROLES) {
    const img = document.querySelector(`#slot-${r.key} .slot-feed`);
    const serial = ASSIGN[r.key];
    const tile = serial && document.querySelector(`.dev[data-serial="${serial}"] .feed img`);
    if (!img || !tile || !tile.dataset.stream || !tile.naturalWidth) {
      if (img && img.getAttribute('src')) img.removeAttribute('src');
      continue;
    }
    mirrorCanvas.width = 168;
    mirrorCanvas.height = Math.round(168 * tile.naturalHeight / tile.naturalWidth);
    try {
      mirrorCanvas.getContext('2d').drawImage(tile, 0, 0, mirrorCanvas.width, mirrorCanvas.height);
      img.src = mirrorCanvas.toDataURL('image/jpeg', 0.7);
    } catch (e) { /* frame not ready yet */ }
  }
}

// --- drag & drop on slots ---------------------------------------------- //
// The whole role row accepts a drop, not just its thumbnail box.
function wireSlots() {
  document.querySelectorAll('.slot').forEach((slot) => {
    const drop = slot.querySelector('.slot-drop');
    slot.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      drop.classList.add('over');
    });
    slot.addEventListener('dragleave', (e) => {
      if (!slot.contains(e.relatedTarget)) drop.classList.remove('over');
    });
    slot.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('over');
      const serial = e.dataTransfer.getData('text/plain');
      if (serial) assign(slot.dataset.role, serial);
    });
  });
}

// --- server calls ------------------------------------------------------ //
async function assign(role, serial) {
  try {
    const res = await fetch('/api/assign', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role, serial }),
    });
    const data = await res.json();
    if (!data.ok) return toast(data.error || 'assign failed', 'err');
    ASSIGN = data.assignments;
    renderSlots();
    renderDevices();
  } catch (e) { toast('assign failed: ' + e, 'err'); }
}

function clearSlot(role) { assign(role, ''); }

// retryBusy (the Rescan button) asks the server to try busy cameras again,
// e.g. after the camera bringup was stopped.
async function scan(retryBusy) {
  try {
    const res = await fetch(retryBusy ? '/api/scan?rescan=1' : '/api/scan');
    const data = await res.json();
    takeScan(data);
    DEVICES = data.devices || [];
    ASSIGN = data.assignments || {};
    const note = el('note');
    if (data.note) { note.textContent = data.note; note.hidden = false; }
    else note.hidden = true;
    const anyLive = DEVICES.some((d) => d.status === 'live');
    el('lamp').className = 'lamp' + (anyLive ? ' ready' : (DEVICES.length ? ' busy' : ''));
    renderDevices();
    renderSlots();
  } catch (e) { toast('scan failed: ' + e, 'err'); }
}

async function rescan() {
  const b = el('btnRescan');
  b.disabled = true; const t = b.textContent; b.textContent = 'Scanning…';
  await scan(true);
  b.disabled = false; b.textContent = t;
  toast('Hardware rescanned', 'ok');
}

async function save() {
  const b = el('btnSave');
  b.disabled = true; const t = b.textContent; b.textContent = 'Saving…';
  try {
    const res = await fetch('/api/save', { method: 'POST' });
    const data = await res.json();
    if (!data.ok) toast(data.error || 'save failed', 'err');
    else {
      const n = (data.written || []).length;
      toast(`Saved ${(data.changed || []).join(', ')} to ${n} file${n === 1 ? '' : 's'} (src + install)`, 'ok');
      await scan();   // refresh what cameras.yaml now holds
    }
  } catch (e) { toast('save failed: ' + e, 'err'); }
  finally { b.disabled = false; b.textContent = t; }
}

// --- boot -------------------------------------------------------------- //
wireSlots();
scan();
setInterval(mirrorSlots, 500);
// Light background refresh so live/busy status pills stay current without
// reloading the already-running MJPEG <img> streams.
setInterval(async () => {
  try {
    const res = await fetch('/api/scan');
    const data = await res.json();
    takeScan(data);
    DEVICES = data.devices || [];
    ASSIGN = data.assignments || ASSIGN;
    renderDevices();
    renderSlots();
  } catch (e) { /* keep last good render */ }
}, 5000);

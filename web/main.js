// Glue: input -> Rust engine (wasm) -> scene buffers -> WebGL2 raytracing shader.
import { VERT, TRACE, SHOW } from './shader.js';

const canvas = document.getElementById('c');
const hud = document.getElementById('hud');
const msgEl = document.getElementById('msg');
const crosshair = document.getElementById('crosshair');
const startEl = document.getElementById('start');
const errEl = document.getElementById('err');

// Phones and tablets get touch controls (no mouse capture); ?touch=1 / ?touch=0 forces either.
const params = new URLSearchParams(location.search);
const touchParam = params.get('touch');
const touchMode = touchParam !== null ? touchParam === '1'
  : navigator.userAgentData?.mobile
    || /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)
    || (matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches);
if (touchMode) document.body.classList.add('touch');

// ---- errors: say what went wrong on the page instead of leaving a black canvas
let started = false; // the frame loop is running
let dead = false;    // a fatal error was shown
function showError(text) {
  errEl.textContent = text;
  errEl.style.display = 'block';
  startEl.style.display = 'none';
}
function hideError() {
  errEl.style.display = 'none';
  updateStart();
}
function fatal(text) {
  dead = true;
  showError(text);
  throw new Error(text);
}
// anything thrown before the frame loop runs stops the page: show it (fatal() already did)
addEventListener('error', (e) => { if (!started && !dead) showError(`rooms could not start: ${e.message || e.error}`); });
addEventListener('unhandledrejection', (e) => {
  if (!started && !dead) showError(`rooms could not start: ${e.reason?.message || e.reason}`);
});

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => { /* still works, just not installable */ });

// ---- engine
let X;
try {
  const res = await fetch('rooms.wasm');
  // an error page (HTML) would only give a confusing compile error below
  if (!res.ok) throw new Error(`the server answered ${res.status} ${res.statusText}`.trim());
  const wasm = await WebAssembly.instantiate(await res.arrayBuffer(), {});
  X = wasm.instance.exports;
  X.init();
} catch (err) {
  fatal(`could not load the engine (rooms.wasm): ${err.message || err}\nreload the page to try again`);
}
globalThis.__X = X; // debugging hook (console / tests)

// ---- WebGL2
const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: false, powerPreference: 'high-performance' });
if (!gl) fatal('WebGL2 is not available in this browser.\nTurn on hardware acceleration in the browser settings (chrome://gpu shows its state).');
globalThis.__gl = gl;

function program(vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader error: ' + gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link error: ' + gl.getProgramInfoLog(p));
  const loc = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, '');
    loc[name] = gl.getUniformLocation(p, name);
  }
  return { p, loc };
}

function floatTex(w, h, internal, format) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, gl.FLOAT, null);
  return t;
}

// Every GL object lives here; setupGL() builds them all again after a lost context comes back.
let traceP, showP, occTex, occVersion = -1;
// scene data texture (R32F, 1024 wide; grows as needed)
const DATA_W = 1024;
let dataTex = null, dataRows = 0, dataBuf = new Float32Array(0);
// accumulation ping-pong targets
let acc = [];
let sinceSetup = 0; // frames drawn since setupGL (the first one may include driver warm-up)
function setupGL() {
  if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('this GPU/browser cannot render to float textures (EXT_color_buffer_float)');
  traceP = program(VERT, TRACE);
  showP = program(VERT, SHOW);
  gl.bindVertexArray(gl.createVertexArray());
  occTex = gl.createTexture();
  occVersion = -1;
  dataTex = null;
  dataRows = 0;
  acc = [];
  sinceSetup = 0;
}
try { setupGL(); } catch (err) { fatal(err.message); }

function uploadData(ptr, len) {
  const rows = Math.max(1, Math.ceil(len / DATA_W));
  if (rows > dataRows) {
    if (dataTex) gl.deleteTexture(dataTex);
    dataRows = rows;
    dataTex = floatTex(DATA_W, rows, gl.R32F, gl.RED);
    dataBuf = new Float32Array(DATA_W * rows);
  }
  dataBuf.set(new Float32Array(X.memory.buffer, ptr, len));
  gl.bindTexture(gl.TEXTURE_2D, dataTex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, DATA_W, rows, gl.RED, gl.FLOAT, dataBuf);
}

// occupancy: 3D R8 texture of the grid cells (1 = free); re-uploaded when the engine says it changed
function uploadOcc(version) {
  if (version === occVersion) return;
  occVersion = version;
  const nx = X.occ_dim(0), ny = X.occ_dim(1), nz = X.occ_dim(2);
  const src = new Uint8Array(X.memory.buffer, X.occ_ptr(), nx * ny * nz);
  const bytes = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) bytes[i] = src[i] * 255;
  gl.bindTexture(gl.TEXTURE_3D, occTex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, bytes);
}

function makeTargets(w, h) {
  for (const a of acc) { gl.deleteTexture(a.tex); gl.deleteFramebuffer(a.fb); }
  acc = [0, 1].map(() => {
    const tex = floatTex(w, h, gl.RGBA32F, gl.RGBA);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fb };
  });
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// ---- resolution: render height steps; width follows the window's shape
// Start low and let auto-res climb: a slow GPU at 720p can take seconds for one raytrace draw,
// and a draw over ~2 s trips the driver watchdog, which kills the WebGL context.
const RES = [180, 270, 360, 540, 720, 900, 1080];
const START_RES = 2; // 360p
const AUTO_MAX = 4;  // auto-res climbs to 720p at most
const resParam = params.get('res');
let resIdx = Number(resParam ?? START_RES);
if (!(resIdx >= 0 && resIdx < RES.length)) resIdx = START_RES;
let autoRes = resParam === null; // adapt to frame rate; [ or ] sets it by hand, both together go back to auto
let autoMax = AUTO_MAX;          // lowered for good when a resolution proved too slow
// The first frames after a change are timed synchronously, so a too-slow resolution is dropped at once.
const PROBE_FRAMES = 3, SLOW_FRAME_MS = 100;
let probe = 0;
const probePixel = new Uint8Array(4);
function resize() {
  const h = RES[resIdx];
  canvas.height = h;
  canvas.width = Math.max(1, Math.round(h * innerWidth / innerHeight));
  canvas.style.imageRendering = h < 540 ? 'pixelated' : 'auto';
  if (!gl.isContextLost()) makeTargets(canvas.width, canvas.height);
  probe = PROBE_FRAMES;
  updateResButton();
}
addEventListener('resize', resize);

let msgTimer;
function msg(text, ms = 2200) {
  if (ui) return; // the save / load panel is using the message area
  msgEl.textContent = text;
  msgEl.style.opacity = 1;
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => (msgEl.style.opacity = 0), ms);
}

// ---- lost WebGL context (GPU watchdog, driver reset, too many contexts): say so and rebuild
let ctxLost = false, lostTimer;
canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault(); // without this the browser never restores the context
  ctxLost = true;
  endAction();
  endGesture();
  // the usual cause is a frame that took too long: come back lower, and don't climb back there
  if (autoRes && resIdx > 0) {
    autoMax = Math.min(autoMax, resIdx - 1);
    resIdx = Math.max(0, resIdx - 2);
  }
  showError('the graphics card stopped responding (WebGL context lost)\ntrying to recover…');
  clearTimeout(lostTimer);
  lostTimer = setTimeout(() => {
    if (ctxLost) showError('the graphics card did not come back (WebGL context lost)\nreload the page to try again');
  }, 6000);
});
canvas.addEventListener('webglcontextrestored', () => {
  try {
    setupGL();
    resize();
  } catch (err) {
    showError(`could not restore the graphics: ${err.message}\nreload the page to try again`);
    return;
  }
  ctxLost = false;
  clearTimeout(lostTimer);
  hideError();
  msg(`graphics restored at ${RES[resIdx]}p`);
});

// run something right after the next frame is drawn (the WebGL canvas is only readable then)
const afterFrame = [];

// ---- rooms in browser storage
const STORE = 'rooms.saves.v1';
const CURRENT = 'rooms.current.v1'; // the room you are in, autosaved so a page reload brings it back
let currentName = null; // name of the room last saved or loaded
function readSaves() {
  try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; }
}
function writeSaves(saves) {
  try { localStorage.setItem(STORE, JSON.stringify(saves)); return true; } catch { msg('could not save: browser storage is full'); return false; }
}
function roomText() {
  const p = X.save();
  return new TextDecoder().decode(new Uint8Array(X.memory.buffer, p, X.save_len()));
}
function loadText(text) {
  const bytes = new TextEncoder().encode(text);
  const p = X.alloc(bytes.length);
  new Uint8Array(X.memory.buffer, p, bytes.length).set(bytes);
  return X.load(bytes.length) === 1;
}
function autosave() {
  try { localStorage.setItem(CURRENT, JSON.stringify({ text: roomText(), name: currentName })); } catch { /* storage full: skip */ }
}
setInterval(autosave, 2000);
addEventListener('pagehide', autosave);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') autosave(); });
try {
  const cur = JSON.parse(localStorage.getItem(CURRENT));
  if (cur && loadText(cur.text)) { currentName = cur.name ?? null; X.clear_history(); }
} catch { /* nothing saved yet */ }

function stamp() {
  const d = new Date(), p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function newRoom() {
  X.reset();
  currentName = null;
  autosave();
  msg('new room');
}
function screenshot() {
  afterFrame.push((c) => c.toBlob((b) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `rooms-${stamp()}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    msg('screenshot saved');
  }, 'image/png'));
}
function defaultName() { return currentName || `room ${Object.keys(readSaves()).length + 1}`; }
// these return the message to show
function saveAs(name) {
  const saves = readSaves();
  saves[name] = { text: roomText(), at: Date.now() };
  if (!writeSaves(saves)) return 'could not save: browser storage is full';
  currentName = name;
  autosave();
  return `saved "${name}"`;
}
function loadSave(name) {
  const r = readSaves()[name];
  if (!(r && loadText(r.text))) return `could not load "${name}"`;
  currentName = name;
  autosave();
  return `loaded "${name}"`;
}
function deleteSave(name) {
  const saves = readSaves();
  delete saves[name];
  writeSaves(saves);
  if (currentName === name) currentName = null;
}
function savedNames() {
  const saves = readSaves();
  return Object.keys(saves).sort((a, b) => saves[b].at - saves[a].at);
}

// ---- save / load panel, shown in the message area at the bottom
// ui = { kind: 'save', text } | { kind: 'load', names, i, del }   (null when closed)
let ui = null;
function showPanel() {
  clearTimeout(msgTimer);
  msgEl.style.opacity = 1;
  if (ui.kind === 'save') {
    msgEl.textContent = `save as: ${ui.text}▏\nenter save · esc cancel`;
    return;
  }
  const items = [...ui.names, FILE_ROW];
  const from = Math.max(0, Math.min(ui.i - 3, items.length - 7));
  const rows = items.slice(from, from + 7).map((n, j) => (from + j === ui.i ? `> ${n} <` : n));
  const onFile = ui.i === ui.names.length;
  const hint = ui.del ? `delete "${ui.names[ui.i]}"? press delete again`
    : onFile ? '↑↓ choose · enter pick a .room file · esc close'
    : '↑↓ choose · enter load · e export · del delete · esc close';
  msgEl.textContent = rows.join('\n') + '\n' + hint;
}
function closePanel(text) {
  ui = null;
  if (text) msg(text);
  else msgEl.style.opacity = 0;
}
function openSave() {
  keys = 0;
  ui = { kind: 'save', text: defaultName() };
  showPanel();
}
const FILE_ROW = 'load .room file…';
const filePicker = Object.assign(document.createElement('input'), { type: 'file', accept: '.room,text/plain', id: 'roomfile', hidden: true });
document.body.append(filePicker);
filePicker.addEventListener('change', async () => {
  const f = filePicker.files[0];
  filePicker.value = '';
  if (!f) return;
  const text = await f.text();
  const name = f.name.replace(/\.room$/i, '') || 'imported room';
  if (!loadText(text)) { msg(`"${f.name}" is not a rooms file`); return; }
  const saves = readSaves();
  saves[name] = { text, at: Date.now() };
  writeSaves(saves);
  currentName = name;
  autosave();
  msg(`loaded "${name}" from file`);
});
function exportRoom(name) {
  const r = readSaves()[name];
  if (!r) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([r.text], { type: 'text/plain' }));
  a.download = `${name}.room`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function openLoad() {
  const names = savedNames();
  keys = 0;
  ui = { kind: 'load', names, i: Math.max(0, names.indexOf(currentName)), del: false };
  showPanel();
}
function panelKey(e) {
  if (e.key === 'Escape') { closePanel(); return; }
  if (ui.kind === 'save') {
    if (e.key === 'Enter') {
      const name = ui.text.trim();
      if (!name) return;
      closePanel(saveAs(name));
      return;
    } else if (e.key === 'Backspace') ui.text = ui.text.slice(0, -1);
    else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && ui.text.length < 40) ui.text += e.key;
  } else {
    const n = ui.names.length + 1; // + the "load .room file" row
    const onFile = ui.i === ui.names.length;
    if (e.key === 'l' || e.key === 'L') { closePanel(); return; }
    if ((e.key === 'e' || e.key === 'E') && !onFile) { exportRoom(ui.names[ui.i]); closePanel(`exported "${ui.names[ui.i]}.room"`); return; }
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { ui.i = (ui.i - 1 + n) % n; ui.del = false; }
    else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { ui.i = (ui.i + 1) % n; ui.del = false; }
    else if (e.key === 'Enter' && onFile) { closePanel(); filePicker.click(); return; }
    else if (e.key === 'Enter') { closePanel(loadSave(ui.names[ui.i])); return; }
    else if (e.key === 'Delete' && !onFile) {
      if (!ui.del) ui.del = true;
      else {
        deleteSave(ui.names[ui.i]);
        ui.names.splice(ui.i, 1);
        ui.i = Math.min(ui.i, ui.names.length);
        ui.del = false;
      }
    }
  }
  showPanel();
}

// ---- keyboard
const KEYS = { KeyW: 1, ArrowUp: 1, KeyS: 2, ArrowDown: 2, KeyA: 4, ArrowLeft: 4, KeyD: 8, ArrowRight: 8 };
let keys = 0;
let showHelp = false;
let preview = false;
const brackets = new Set();
function setPreview(on) { if (on !== preview) { preview = on; X.set_preview(on ? 1 : 0); } }

function undo() { if (!X.undo()) msg('nothing to undo'); }
function redo() { if (!X.redo()) msg('nothing to redo'); }
function undoRedo(e) {
  if (e.code === 'KeyY' || e.shiftKey) redo();
  else undo();
}
function setRes(i) {
  autoRes = false;
  resIdx = i;
  resize();
}

addEventListener('keydown', (e) => {
  if (ui) { e.preventDefault(); panelKey(e); return; }
  if (e.ctrlKey || e.metaKey) {
    if (e.code === 'KeyZ' || e.code === 'KeyY') { e.preventDefault(); undoRedo(e); }
    return; // every other ctrl combination is left to the browser
  }
  if (!e.repeat) {
    if (e.code === 'KeyK') { openSave(); return; }
    if (e.code === 'KeyL') { openLoad(); return; }
    if (e.code === 'KeyN') newRoom();
    if (e.code === 'KeyP') screenshot();
  }
  if (KEYS[e.code]) { keys |= KEYS[e.code]; e.preventDefault(); }
  if (e.key === 'Alt') e.preventDefault(); // keep Firefox/Windows from focusing the menu bar
  if (e.key === 'Shift' && locked) setPreview(true);
  if (e.code === 'BracketLeft' || e.code === 'BracketRight') {
    brackets.add(e.code);
    if (brackets.size === 2) {
      if (!autoRes) { autoRes = true; updateResButton(); msg('automatic resolution'); }
    } else if (!e.repeat) {
      if (e.code === 'BracketLeft' && resIdx > 0) setRes(resIdx - 1);
      else if (e.code === 'BracketRight' && resIdx < RES.length - 1) setRes(resIdx + 1);
      else autoRes = false;
    }
  }
  if (e.code === 'KeyH') showHelp = !showHelp;
});
addEventListener('keyup', (e) => {
  if (KEYS[e.code]) keys &= ~KEYS[e.code];
  if (e.key === 'Alt') e.preventDefault();
  if (e.key === 'Shift') setPreview(false);
  brackets.delete(e.code);
});
addEventListener('blur', () => { keys = 0; brackets.clear(); setPreview(false); });

// ---- mouse: pointer lock (click to capture, Esc releases); actions hit the crosshair
const C = [0.5, 0.5];
const SENS = 0.0025;
let locked = false;
let mode = null; // 'repeat' | 'tiles' | 'shift' (click or drag undecided) | 'erase' | 'win'
let shiftMoved = 0;
let pushDir = 1, pushStart = 0, repeatTimeout, repeatInterval;

function updateStart() {
  startEl.textContent = 'click to capture';
  startEl.style.display = touchMode || locked || errEl.style.display === 'block' ? 'none' : 'block';
}

// Holding a click repeats it, with steps growing logarithmically with hold time:
// 25 cm at first, ~1.5 m per repeat after 1 s, ~3 m after 10 s (about 9 repeats per second).
function pushSteps() {
  const held = Math.max(0, (performance.now() - pushStart - 380) / 1000);
  return Math.max(1, Math.round(1 + 4 * Math.log(1 + held / 0.5)));
}
function report(r) {
  if (r === 2) msg('you are in the way');
  else if (r === 3) msg("can't go further that way");
  else if (r === 4) msg('ceilings stay at least 2 m up');
}
// the region pressed on is locked (grip) until mouseup, so it moves as one piece
function doPush() {
  const steps = mode === 'repeat' ? pushSteps() : 1;
  let r = X.grip_step(pushDir * steps);
  if (r !== 1 && steps > 1) r = X.grip_step(pushDir); // too far for a big step: try a small one
  report(r);
}
// push (dir 1) or pull (-1) the surface under the crosshair, repeating while held; inside group_begin
function startPush(dir) {
  if (!X.grip_begin(...C)) return false;
  pushDir = dir;
  mode = 'repeat';
  pushStart = performance.now();
  doPush();
  repeatTimeout = setTimeout(() => { if (mode === 'repeat') repeatInterval = setInterval(doPush, 110); }, 380);
  return true;
}
function endAction() {
  if (mode === 'win') X.win_end();
  X.grip_end();
  X.tile_end();
  X.group_end();
  clearTimeout(repeatTimeout);
  clearInterval(repeatInterval);
  mode = null;
}
// window results shared by alt+click and the touch long press; true when a window is held
function winBegin(sx, sy) {
  const r = X.win_begin(sx, sy);
  if (r === 3) msg('no windows in the first 2 m above the floor');
  else if (r === 4) msg('enough windows');
  else if (r === 5) msg('too many windows in this room');
  else if (r === 0) msg('windows go on walls or the ceiling');
  else if (r === 6) msg('no room for a 1 × 1 m window here');
  return r === 1 || r === 2;
}
document.addEventListener('pointerlockchange', () => {
  locked = document.pointerLockElement === canvas;
  crosshair.style.display = locked ? 'block' : 'none';
  updateStart();
  X.set_hover(locked ? 0.5 : -1, locked ? 0.5 : -1);
  if (!locked) { endAction(); keys = 0; setPreview(false); if (ui) closePanel(); } // Esc while captured
});
document.addEventListener('pointerlockerror', () => msg('the browser refused to capture the mouse; click again'));

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('mousedown', (e) => {
  if (touchMode) return; // touch controls below
  e.preventDefault();
  if (!locked) { canvas.requestPointerLock(); return; }
  X.group_begin(); // everything until mouseup is one undo step
  if (e.altKey) {
    if (e.button === 2) { if (X.win_delete(...C)) msg('window removed'); return; }
    if (winBegin(...C)) mode = 'win';
    return;
  }
  if (e.shiftKey && e.button === 0) { mode = 'shift'; shiftMoved = 0; return; } // click: lines, drag: erase
  const t = X.tile_begin(...C); // on the floor: toggle a mirror tile, dragging paints the same onto more tiles
  if (t) { if (t === 3) msg('too many mirror tiles'); mode = 'tiles'; return; }
  startPush(e.button === 2 ? -1 : 1);
});
document.addEventListener('mousemove', (e) => {
  if (!locked) return;
  const dx = e.movementX, dy = e.movementY;
  if (mode === 'shift') {
    shiftMoved += Math.abs(dx) + Math.abs(dy);
    if (shiftMoved > 6) { mode = 'erase'; X.erase(...C); }
  }
  if (mode === 'tiles') {
    // sweep in small steps so fast moves don't skip tiles
    const n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / 3));
    for (let i = 0; i < n; i++) {
      X.look((dx / n) * SENS, (dy / n) * SENS);
      X.tile_drag(...C);
    }
    return;
  }
  if (mode === 'erase') {
    // sweep in small steps so fast moves don't skip line segments
    const n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / 3));
    for (let i = 0; i < n; i++) {
      X.look((dx / n) * SENS, (dy / n) * SENS);
      X.erase(...C);
    }
  } else {
    X.look(dx * SENS, dy * SENS);
    if (mode === 'win') X.win_drag(...C);
  }
});
document.addEventListener('mouseup', () => {
  if (touchMode) return;
  if (mode === 'shift') X.add_cross(...C);
  endAction();
});

// ---- touch: two sticks (walk, look), push / pull buttons, and gestures on the room itself.
// On the room: tap a wall = preview lines there, tap the preview again = cut them; drag = erase
// lines; long press = window, keep holding and drag to size it; long press on a window removes it;
// on the floor: mirror tile, drag paints.
let look = [0, 0];       // look stick deflection
let lookRate = [0, 0];   // ... eased toward it, so finger jitter and touch sampling don't jerk the view
let gesture = null;      // { id, x0, y0, x, y, t0, kind: 'pending' | 'tiles' | 'erase' | 'win' | 'none' }
let previewAt = null;    // [x, y] in css pixels of the previewed lines (touch)
const TAP_MOVE = 12, NEAR = 40, LONG_PRESS = 450;
const LOOK_YAW = 2.6, LOOK_PITCH = 1.7; // rad/s at full deflection

function toScreen(x, y) { return [x / innerWidth, y / innerHeight]; }
function clearPreview() {
  if (!touchMode) return;
  setPreview(false);
  previewAt = null;
  X.set_hover(0.5, 0.5); // back to the crosshair, the push / pull target
}
function showPreviewAt(x, y) {
  previewAt = [x, y];
  X.set_hover(...toScreen(x, y));
  setPreview(true);
}
// call fn on screen points from (x0, y0) to (x1, y1) every few pixels, so fast drags skip nothing
function sweep(x0, y0, x1, y1, fn) {
  const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 3));
  for (let i = 1; i <= n; i++) fn(...toScreen(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n));
}
// Long press: checked from the frame loop, which runs after the input events queued behind a slow
// frame (a timer can fire before such a pointerup and turn a quick tap into a long press).
function checkLongPress(now) {
  const g = gesture;
  if (g?.kind !== 'pending' || now - g.t0 < LONG_PRESS) return;
  clearPreview();
  if (X.win_delete(...toScreen(g.x0, g.y0))) { // on a window: remove it
    g.kind = 'none';
    navigator.vibrate?.(15);
    msg('window removed');
    return;
  }
  g.kind = winBegin(...toScreen(g.x0, g.y0)) ? 'win' : 'none';
  if (g.kind === 'win') { navigator.vibrate?.(15); X.win_drag(...toScreen(g.x, g.y)); }
}
function endGesture() {
  if (!gesture) return;
  if (gesture.kind === 'win') X.win_end();
  X.tile_end();
  X.group_end();
  gesture = null;
}
// Browsers only allow fullscreen and orientation lock from a user gesture: try on every touch
// until it holds (an installed app is already fullscreen and landscape from the manifest).
function landscape() {
  const lock = () => screen.orientation?.lock?.('landscape').catch(() => {});
  if (document.fullscreenElement || matchMedia('(display-mode: fullscreen), (display-mode: standalone)').matches) { lock(); return; }
  document.documentElement.requestFullscreen?.({ navigationUI: 'hide' }).then(lock, () => {});
}

function setupTouch() {
  document.addEventListener('pointerup', landscape, true);
  document.addEventListener('contextmenu', (e) => e.preventDefault());
  crosshair.style.display = 'block';
  X.set_hover(0.5, 0.5);

  canvas.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    if (menuEl.classList.contains('open')) { closeMenu(); return; } // that tap only closes the menu
    if (gesture || mode || ctxLost) return; // one action at a time
    canvas.setPointerCapture(e.pointerId);
    const x = e.clientX, y = e.clientY;
    const confirming = previewAt && Math.hypot(x - previewAt[0], y - previewAt[1]) < NEAR;
    if (!confirming) clearPreview();
    X.group_begin(); // everything until the finger lifts is one undo step
    gesture = { id: e.pointerId, x0: x, y0: y, x, y, t0: performance.now(), kind: 'pending' }; // from now: the press may have waited behind a slow frame
    const t = X.tile_begin(...toScreen(x, y));
    if (t) {
      clearPreview();
      if (t === 3) msg('too many mirror tiles');
      gesture.kind = 'tiles';
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    const g = gesture;
    if (!g || e.pointerId !== g.id) return;
    const px = g.x, py = g.y, x = e.clientX, y = e.clientY;
    g.x = x;
    g.y = y;
    if (g.kind === 'pending' && Math.hypot(x - g.x0, y - g.y0) > TAP_MOVE) {
      clearPreview();
      g.kind = 'erase';
      sweep(g.x0, g.y0, x, y, X.erase);
    } else if (g.kind === 'erase') sweep(px, py, x, y, X.erase);
    else if (g.kind === 'tiles') sweep(px, py, x, y, X.tile_drag);
    else if (g.kind === 'win') X.win_drag(...toScreen(x, y));
  });
  const up = (e) => {
    const g = gesture;
    if (!g || e.pointerId !== g.id) return;
    if (g.kind === 'pending' && e.type === 'pointerup') {
      if (previewAt && Math.hypot(g.x - previewAt[0], g.y - previewAt[1]) < NEAR) {
        X.add_cross(...toScreen(...previewAt));
        clearPreview();
      } else showPreviewAt(g.x, g.y);
    }
    endGesture();
  };
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);

  // sticks: x right, y down, each -1..1 with a small dead zone
  function stick(el, onMove) {
    const knob = el.firstElementChild;
    let id = null;
    const move = (e) => {
      const r = el.getBoundingClientRect(), R = r.width / 2;
      let dx = (e.clientX - r.left - R) / R, dy = (e.clientY - r.top - R) / R;
      const len = Math.hypot(dx, dy);
      if (len > 1) { dx /= len; dy /= len; }
      knob.style.transform = `translate(${dx * R * 0.55}px, ${dy * R * 0.55}px)`;
      const m = Math.min(len, 1); // (dx, dy) is now this long
      const k = m < 0.12 ? 0 : (m - 0.12) / 0.88 / m;
      onMove(dx * k, dy * k);
    };
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      if (id !== null) return;
      id = e.pointerId;
      el.setPointerCapture(id);
      closeMenu();
      clearPreview();
      move(e);
    });
    el.addEventListener('pointermove', (e) => { if (e.pointerId === id) move(e); });
    const release = (e) => {
      if (e.pointerId !== id) return;
      id = null;
      knob.style.transform = '';
      onMove(0, 0);
    };
    el.addEventListener('pointerup', release);
    el.addEventListener('pointercancel', release);
  }
  stick(document.getElementById('walk'), (x, y) => X.set_stick(x, -y));
  stick(document.getElementById('look'), (x, y) => { look = [x, y]; });

  // push / pull: like holding the mouse button on the surface under the crosshair
  function hold(el, dir) {
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      closeMenu();
      clearPreview();
      if (gesture || mode || ctxLost) return;
      X.group_begin();
      if (startPush(dir)) el.classList.add('on');
      else { X.group_end(); msg('nothing to push here'); }
    });
    const release = () => {
      if (el.classList.contains('on')) endAction();
      el.classList.remove('on');
    };
    el.addEventListener('pointerup', release);
    el.addEventListener('pointercancel', release);
  }
  hold(document.getElementById('push'), 1);
  hold(document.getElementById('pull'), -1);

  // buttons act on tap
  const tap = (id, fn) => document.getElementById(id).addEventListener('click', (e) => {
    e.preventDefault();
    clearPreview();
    if (gesture || mode) return;
    fn();
  });
  tap('undo', () => { closeMenu(); undo(); });
  tap('redo', () => { closeMenu(); redo(); });
  tap('menubtn', openMenu);
  tap('m-help', () => { closeMenu(); openHelp(); });
  document.getElementById('help').addEventListener('click', (e) => { e.preventDefault(); closeHelp(); });
  tap('m-save', () => {
    closeMenu();
    const name = prompt('save as', defaultName())?.trim();
    if (name) msg(saveAs(name));
  });
  tap('m-load', () => { closeMenu(); openList(); });
  tap('m-new', () => { closeMenu(); newRoom(); });
  tap('m-shot', () => { closeMenu(); screenshot(); });
  // resolution: auto, then each fixed step, then auto again
  tap('m-res', () => {
    if (autoRes) setRes(0);
    else if (resIdx < RES.length - 1) setRes(resIdx + 1);
    else { autoRes = true; updateResButton(); }
  });

  // the first time, explain the gestures
  try {
    if (!localStorage.getItem(HELP_SEEN)) { openHelp(); localStorage.setItem(HELP_SEEN, '1'); }
  } catch { openHelp(); }
}
const HELP_SEEN = 'rooms.touchhelp.v1';
function openHelp() { document.getElementById('help').classList.add('open'); }
function closeHelp() { document.getElementById('help').classList.remove('open'); }

// ---- touch menu (save, load, new, screenshot, resolution) and the saved rooms list
const menuEl = document.getElementById('menu');
const listEl = document.getElementById('list');
// the menu takes the menu button's place (tap the room or pick an item to close it)
function openMenu() { menuEl.classList.add('open'); document.getElementById('menubtn').style.display = 'none'; }
function closeMenu() { menuEl.classList.remove('open'); document.getElementById('menubtn').style.display = ''; }
function updateResButton() {
  const b = document.getElementById('m-res');
  if (b) b.textContent = autoRes ? `res: auto (${RES[resIdx]}p)` : `res: ${RES[resIdx]}p`;
}
function openList() {
  listEl.replaceChildren();
  const button = (text, cls, fn) => {
    const b = Object.assign(document.createElement('button'), { textContent: text, className: cls });
    b.addEventListener('click', (e) => { e.preventDefault(); fn(); });
    return b;
  };
  const close = () => { listEl.classList.remove('open'); };
  const title = Object.assign(document.createElement('div'), { className: 'title', textContent: 'load a room' });
  listEl.append(title);
  for (const name of savedNames()) {
    const row = Object.assign(document.createElement('div'), { className: 'row' });
    row.append(
      button(name + (name === currentName ? ' ·' : ''), 'name', () => { close(); msg(loadSave(name)); }),
      button('export', 'small', () => { exportRoom(name); msg(`exported "${name}.room"`); }),
      button('delete', 'small', () => { if (confirm(`delete "${name}"?`)) { deleteSave(name); row.remove(); } }),
    );
    listEl.append(row);
  }
  const foot = Object.assign(document.createElement('div'), { className: 'row' });
  foot.append(button(FILE_ROW, 'name', () => { close(); filePicker.click(); }), button('close', 'small', close));
  listEl.append(foot);
  listEl.classList.add('open');
}

// ---- frame loop
const HELP = [
  ['Esc', 'release mouse'],
  ['mouse', 'look around'],
  ['wasd/arrows', 'walk'],
  ['click', 'push'],
  ['right click', 'pull'],
  ['click floor', 'mirror tile, drag to paint more'],
  ['shift+click', 'vertical and horizontal line'],
  ['shift+drag', 'erase lines'],
  ['alt+click', 'window, hold and aim to resize (whole metres)'],
  ['k', 'save'],
  ['l', 'load'],
  ['n', 'new room'],
  ['p', 'screenshot'],
  ['ctrl+z', 'undo'],
  ['ctrl+y', 'redo'],
  ['[ ]', 'resolution, both together: auto'],
].map(([k, d]) => k.padEnd(14) + d).join('\n');
let last = performance.now(), fpsT = last, frames = 0, fpsShown = 0, slow = 0, fast = 0, frameNo = 0;
let peakFps = 0; // best recent frame rate: about the display's refresh rate

const U_LEN = 192 + 16 * 32;
let uView = new Float32Array(0);
function render() {
  const w = canvas.width, h = canvas.height;
  const p = X.frame(w, h);
  if (uView.buffer !== X.memory.buffer || uView.byteOffset !== p) uView = new Float32Array(X.memory.buffer, p, U_LEN); // only when wasm memory grows
  const u = uView;
  uploadData(X.data_ptr(), X.data_len());
  uploadOcc(u[31]);
  const src = acc[frameNo & 1], dst = acc[(frameNo + 1) & 1];
  frameNo++;

  const L = traceP.loc;
  gl.useProgram(traceP.p);
  gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
  gl.viewport(0, 0, w, h);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dataTex); gl.uniform1i(L.uData, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, src.tex); gl.uniform1i(L.uPrev, 1);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_3D, occTex); gl.uniform1i(L.uOcc, 2);
  gl.uniform2f(L.uRes, w, h);
  gl.uniform3f(L.uPos, u[0], u[1], u[2]);
  gl.uniform3f(L.uF, u[3], u[4], u[5]);
  gl.uniform3f(L.uR, u[6], u[7], u[8]);
  gl.uniform3f(L.uU, u[9], u[10], u[11]);
  gl.uniform1f(L.uAspect, u[12]);
  gl.uniform1f(L.uTan, u[13]);
  gl.uniform1f(L.uAlpha, u[14]);
  gl.uniform1f(L.uFrame, u[15]);
  gl.uniform1i(L.uNWin, u[16]);
  gl.uniform1i(L.uNVpl, u[17]);
  gl.uniform1i(L.uVplOff, u[18]);
  gl.uniform3i(L.uN, u[20], u[21], u[22]);
  gl.uniform3i(L.uCutOff, u[23], u[24], u[25]);
  gl.uniform3i(L.uCamCell, u[26], u[27], u[28]);
  gl.uniform1i(L.uNPlanes, u[29]);
  gl.uniform1i(L.uPlaneOff, u[30]);
  gl.uniform4f(L.uHover, u[32], u[33], u[34], u[35]);
  gl.uniform1i(L.uHoverOff, u[36]);
  gl.uniform1i(L.uHoverN, u[37]);
  gl.uniform3f(L.uTileHover, u[39], u[46], u[47]);
  gl.uniform1i(L.uMirOff, u[176]);
  gl.uniform1i(L.uMirN, u[177]);
  gl.uniform3f(L.uSunTo, u[181], u[182], u[183]);
  gl.uniform2f(L.uSunSeen, u[184], u[185]);
  gl.uniform4f(L.uPreview, u[40], u[41], u[42], u[43]);
  gl.uniform2f(L.uPreview2, u[44], u[45]);
  gl.uniform4fv(L.uWin, u.subarray(192, 192 + 16 * 32));
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  gl.useProgram(showP.p);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, w, h);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dst.tex);
  gl.uniform1i(showP.loc.uAcc, 0);
  gl.uniform2f(showP.loc.uRes, w, h);
  gl.uniform1f(showP.loc.uFrame, u[15]);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  for (const f of afterFrame.splice(0)) f(canvas);
}

// Time a frame to completion (readPixels waits for the GPU); drop the resolution at once if it is slow.
function probeFrame(t0) {
  probe--;
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, probePixel);
  const ms = performance.now() - t0;
  if (sinceSetup <= 1 || ms <= SLOW_FRAME_MS || !autoRes || resIdx === 0) return;
  // cost follows the pixel count: aim for ~40 ms
  const want = RES[resIdx] * Math.sqrt(40 / ms);
  let i = resIdx - 1;
  while (i > 0 && RES[i] > want) i--;
  autoMax = Math.min(autoMax, resIdx - 1);
  resIdx = i;
  resize();
  slow = fast = 0;
  console.info(`rooms: a frame took ${Math.round(ms)} ms, rendering at ${RES[i]}p`);
}

let crashed = false;
function loop(now) {
  requestAnimationFrame(loop); // first, so an error below can't stop the loop
  const dt = Math.max(0, (now - last) / 1000);
  last = now;
  if (ctxLost) return;
  try {
    if (look[0] || look[1] || lookRate[0] || lookRate[1]) {
      const c = (a) => Math.sign(a) * Math.abs(a) ** 1.6; // gentle near the centre
      const k = 1 - Math.exp(-Math.min(dt, 0.1) / 0.05);  // ~50 ms easing
      for (const i of [0, 1]) {
        lookRate[i] += (c(look[i]) - lookRate[i]) * k;
        if (Math.abs(lookRate[i]) < 1e-4 && !look[i]) lookRate[i] = 0;
      }
      X.look(lookRate[0] * LOOK_YAW * Math.min(dt, 0.1), lookRate[1] * LOOK_PITCH * Math.min(dt, 0.1));
    }
    if (gesture) checkLongPress(performance.now());
    X.update(dt, keys);
    const t0 = performance.now();
    render();
    sinceSetup++;
    if (probe > 0) probeFrame(t0);
  } catch (err) {
    if (!crashed) { crashed = true; console.error(err); msg('engine error, see the browser console (reload to recover)'); }
    return;
  }
  frames++;
  if (now - fpsT > 1000) {
    const fps = frames * 1000 / (now - fpsT);
    fpsShown = Math.round(fps);
    if (autoRes) {
      // step down after 2 slow seconds; step back up (to 720p at most) after 3 fast ones.
      // Touch screens aim for the display's full rate: 35-50 fps on a 60 Hz phone shows every
      // other frame twice and judders, though the counter looks fine; a level that proved too
      // slow there is not climbed back to.
      peakFps = Math.max(peakFps * 0.995, fps);
      const slowFps = touchMode ? 0.85 * Math.min(peakFps, 125) : 27, fastFps = touchMode ? 0.95 * Math.min(peakFps, 125) : 55;
      slow = fps < slowFps ? slow + 1 : 0;
      fast = fps > fastFps ? fast + 1 : 0;
      if (slow >= 2 && resIdx > 0) {
        if (touchMode) autoMax = Math.min(autoMax, resIdx - 1);
        resIdx--; resize(); slow = 0;
      } else if (fast >= 3 && resIdx < autoMax) { resIdx++; resize(); fast = 0; }
    }
    frames = 0;
    fpsT = now;
  }
  hud.textContent = `rooms · ${fpsShown}fps@${canvas.width}x${canvas.height}` + (touchMode ? '' : ` · 'H' help` + (showHelp ? '\n\n' + HELP : ''));
}

resize();
if (touchMode) setupTouch();
updateStart();
started = true;
requestAnimationFrame(loop);

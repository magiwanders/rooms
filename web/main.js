// Glue: input -> Rust engine (wasm) -> scene buffers -> WebGL2 raytracing shader.
import { VERT, TRACE, SHOW } from './shader.js';

const canvas = document.getElementById('c');
const hud = document.getElementById('hud');
const msgEl = document.getElementById('msg');
const crosshair = document.getElementById('crosshair');
const startEl = document.getElementById('start');

function fatal(text) {
  startEl.textContent = text;
  startEl.style.display = 'block';
  throw new Error(text);
}

// ---- engine
const wasm = await WebAssembly.instantiate(await (await fetch('rooms.wasm')).arrayBuffer(), {});
const X = wasm.instance.exports;
X.init();
globalThis.__X = X; // debugging hook (console / tests)

// ---- WebGL2
const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: false, powerPreference: 'high-performance' });
if (!gl) fatal('WebGL2 is not available in this browser');
if (!gl.getExtension('EXT_color_buffer_float')) fatal('this GPU/browser cannot render to float textures (EXT_color_buffer_float)');

function program(vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) fatal('shader error: ' + gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) fatal('link error: ' + gl.getProgramInfoLog(p));
  const loc = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, '');
    loc[name] = gl.getUniformLocation(p, name);
  }
  return { p, loc };
}
const traceP = program(VERT, TRACE);
const showP = program(VERT, SHOW);
gl.bindVertexArray(gl.createVertexArray());

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

// scene data texture (R32F, 1024 wide; grows as needed)
const DATA_W = 1024;
let dataTex = null, dataRows = 0, dataBuf = new Float32Array(0);
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
let occTex = gl.createTexture(), occVersion = -1;
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

// accumulation ping-pong targets
let acc = [];
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
const RES = [180, 270, 360, 540, 720, 900, 1080];
const DEFAULT_RES = 4; // 720p
const params = new URLSearchParams(location.search);
const resParam = params.get('res');
let resIdx = Number(resParam ?? DEFAULT_RES);
if (!(resIdx >= 0 && resIdx < RES.length)) resIdx = DEFAULT_RES;
let autoRes = resParam === null; // adapt to frame rate; [ or ] sets it by hand, both together go back to auto
function resize() {
  const h = RES[resIdx];
  canvas.height = h;
  canvas.width = Math.max(1, Math.round(h * innerWidth / innerHeight));
  canvas.style.imageRendering = h < 540 ? 'pixelated' : 'auto';
  makeTargets(canvas.width, canvas.height);
}
addEventListener('resize', resize);
resize();

let msgTimer;
function msg(text) {
  if (ui) return; // the save / load panel is using the message area
  msgEl.textContent = text;
  msgEl.style.opacity = 1;
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => (msgEl.style.opacity = 0), 2200);
}

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
  const from = Math.max(0, Math.min(ui.i - 3, ui.names.length - 7));
  const rows = ui.names.slice(from, from + 7).map((n, j) => (from + j === ui.i ? `> ${n} <` : n));
  const hint = ui.del ? `delete "${ui.names[ui.i]}"? press delete again` : '↑↓ choose · enter load · del delete · esc close';
  msgEl.textContent = rows.join('\n') + '\n' + hint;
}
function closePanel(text) {
  ui = null;
  if (text) msg(text);
  else msgEl.style.opacity = 0;
}
function openSave() {
  keys = 0;
  ui = { kind: 'save', text: currentName || `room ${Object.keys(readSaves()).length + 1}` };
  showPanel();
}
function openLoad() {
  const saves = readSaves();
  const names = Object.keys(saves).sort((a, b) => saves[b].at - saves[a].at);
  if (!names.length) { msg('no saved rooms yet'); return; }
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
      const saves = readSaves();
      saves[name] = { text: roomText(), at: Date.now() };
      if (writeSaves(saves)) { currentName = name; autosave(); closePanel(`saved "${name}"`); }
      else closePanel('could not save: browser storage is full');
      return;
    } else if (e.key === 'Backspace') ui.text = ui.text.slice(0, -1);
    else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && ui.text.length < 40) ui.text += e.key;
  } else {
    const n = ui.names.length;
    if (e.key === 'l' || e.key === 'L') { closePanel(); return; }
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { ui.i = (ui.i - 1 + n) % n; ui.del = false; }
    else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { ui.i = (ui.i + 1) % n; ui.del = false; }
    else if (e.key === 'Enter') {
      const name = ui.names[ui.i], r = readSaves()[name];
      if (r && loadText(r.text)) { currentName = name; autosave(); closePanel(`loaded "${name}"`); }
      else closePanel(`could not load "${name}"`);
      return;
    } else if (e.key === 'Delete') {
      if (!ui.del) ui.del = true;
      else {
        const name = ui.names[ui.i], saves = readSaves();
        delete saves[name];
        writeSaves(saves);
        if (currentName === name) currentName = null;
        ui.names.splice(ui.i, 1);
        if (!ui.names.length) { closePanel(`deleted "${name}"`); return; }
        ui.i = Math.min(ui.i, ui.names.length - 1);
        ui.del = false;
      }
    }
  }
  showPanel();
}

// ---- keyboard
const KEYS = { KeyW: 1, ArrowUp: 1, KeyS: 2, ArrowDown: 2, KeyA: 4, ArrowLeft: 4, KeyD: 8, ArrowRight: 8 };
const MIST = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]; // density (1/m)
let mistIdx = 5;
let keys = 0;
let showHelp = false;
let preview = false;
const brackets = new Set();
function setPreview(on) { if (on !== preview) { preview = on; X.set_preview(on ? 1 : 0); } }

function undoRedo(e) {
  const redo = e.code === 'KeyY' || e.shiftKey;
  if (!(redo ? X.redo() : X.undo())) msg(redo ? 'nothing to redo' : 'nothing to undo');
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
      if (!autoRes) { autoRes = true; msg('automatic resolution'); }
    } else if (!e.repeat) {
      autoRes = false;
      if (e.code === 'BracketLeft' && resIdx > 0) { resIdx--; resize(); }
      if (e.code === 'BracketRight' && resIdx < RES.length - 1) { resIdx++; resize(); }
    }
  }
  if (e.code === 'KeyH') showHelp = !showHelp;
  const more = e.code === 'Equal' || e.code === 'NumpadAdd', less = e.code === 'Minus' || e.code === 'NumpadSubtract';
  if ((more && mistIdx < MIST.length - 1) || (less && mistIdx > 0)) {
    mistIdx += more ? 1 : -1;
    X.invalidate();
    msg(mistIdx === 0 ? 'no mist' : `mist ${mistIdx}/${MIST.length - 1}`);
  }
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
let mode = null; // 'repeat' | 'shift' (click or drag undecided) | 'erase' | 'win'
let shiftMoved = 0;
let pushDir = 1, pushStart = 0, repeatTimeout, repeatInterval;

// Holding a click repeats it, with steps growing logarithmically with hold time:
// 25 cm at first, ~1.5 m per repeat after 1 s, ~3 m after 10 s (about 9 repeats per second).
function pushSteps() {
  const held = Math.max(0, (performance.now() - pushStart - 380) / 1000);
  return Math.max(1, Math.round(1 + 4 * Math.log(1 + held / 0.5)));
}
function doPush() {
  const steps = mode === 'repeat' ? pushSteps() : 1;
  let r = X.push(...C, pushDir * steps);
  if (r !== 1 && steps > 1) r = X.push(...C, pushDir); // too far for a big step: try a small one
  if (r === 2) msg('you are in the way');
  else if (r === 3) msg("can't go further that way");
}
function endAction() {
  if (mode === 'win') X.win_end();
  X.group_end();
  clearTimeout(repeatTimeout);
  clearInterval(repeatInterval);
  mode = null;
}
document.addEventListener('pointerlockchange', () => {
  locked = document.pointerLockElement === canvas;
  crosshair.style.display = locked ? 'block' : 'none';
  startEl.style.display = locked ? 'none' : 'block';
  X.set_hover(locked ? 0.5 : -1, locked ? 0.5 : -1);
  if (!locked) { endAction(); keys = 0; setPreview(false); if (ui) closePanel(); } // Esc while captured
});
document.addEventListener('pointerlockerror', () => msg('the browser refused to capture the mouse; click again'));

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('mousedown', (e) => {
  e.preventDefault();
  if (!locked) { canvas.requestPointerLock(); return; }
  X.group_begin(); // everything until mouseup is one undo step
  if (e.altKey) {
    if (e.button === 2) { if (X.win_delete(...C)) msg('window removed'); return; }
    const r = X.win_begin(...C);
    if (r === 3) msg('no windows in the first 2 m above the floor');
    else if (r === 4) msg('enough windows');
    else if (r === 0) msg('windows go on walls or the ceiling');
    else mode = 'win';
    return;
  }
  if (e.shiftKey && e.button === 0) { mode = 'shift'; shiftMoved = 0; return; } // click: lines, drag: erase
  pushDir = e.button === 2 ? -1 : 1;
  mode = 'repeat';
  pushStart = performance.now();
  doPush();
  repeatTimeout = setTimeout(() => { if (mode === 'repeat') repeatInterval = setInterval(doPush, 110); }, 380);
});
document.addEventListener('mousemove', (e) => {
  if (!locked) return;
  const dx = e.movementX, dy = e.movementY;
  if (mode === 'shift') {
    shiftMoved += Math.abs(dx) + Math.abs(dy);
    if (shiftMoved > 6) { mode = 'erase'; X.erase(...C); }
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
  if (mode === 'shift') X.add_cross(...C);
  endAction();
});

// ---- frame loop
const HELP = [
  ['Esc', 'release mouse'],
  ['mouse', 'look around'],
  ['wasd/arrows', 'walk'],
  ['click', 'push'],
  ['right click', 'pull'],
  ['shift+click', 'vertical and horizontal line'],
  ['shift+drag', 'erase lines'],
  ['alt+click', 'window, hold and aim to resize'],
  ['- +', 'mist'],
  ['k', 'save'],
  ['l', 'load'],
  ['n', 'new room'],
  ['p', 'screenshot'],
  ['ctrl+z', 'undo'],
  ['ctrl+y', 'redo'],
  ['[ ]', 'resolution, both together: auto'],
].map(([k, d]) => k.padEnd(14) + d).join('\n');
let last = performance.now(), fpsT = last, frames = 0, fpsShown = 0, slow = 0, fast = 0, frameNo = 0;

function render() {
  const w = canvas.width, h = canvas.height;
  const u = new Float32Array(X.memory.buffer, X.frame(w, h), 176).slice();
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
  gl.uniform1f(L.uArea, u[19]);
  gl.uniform1f(L.uSigma, MIST[mistIdx]);
  gl.uniform3i(L.uN, u[20], u[21], u[22]);
  gl.uniform3i(L.uCutOff, u[23], u[24], u[25]);
  gl.uniform3i(L.uCamCell, u[26], u[27], u[28]);
  gl.uniform1i(L.uNPlanes, u[29]);
  gl.uniform1i(L.uPlaneOff, u[30]);
  gl.uniform4f(L.uHover, u[32], u[33], u[34], u[35]);
  gl.uniform1i(L.uHoverOff, u[36]);
  gl.uniform1i(L.uHoverN, u[37]);
  gl.uniform4f(L.uPreview, u[40], u[41], u[42], u[43]);
  gl.uniform2f(L.uPreview2, u[44], u[45]);
  gl.uniform4fv(L.uWin, u.subarray(48, 176));
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

let crashed = false;
function loop(now) {
  requestAnimationFrame(loop); // first, so an error below can't stop the loop
  const dt = Math.max(0, (now - last) / 1000);
  last = now;
  try {
    X.update(dt, keys);
    render();
  } catch (err) {
    if (!crashed) { crashed = true; console.error(err); msg('engine error, see the browser console (reload to recover)'); }
    return;
  }
  frames++;
  if (now - fpsT > 1000) {
    const fps = frames * 1000 / (now - fpsT);
    fpsShown = Math.round(fps);
    if (autoRes) {
      // step down after 2 slow seconds; step back up (to 720p at most) after 3 fast ones
      slow = fps < 27 ? slow + 1 : 0;
      fast = fps > 55 ? fast + 1 : 0;
      if (slow >= 2 && resIdx > 0) { resIdx--; resize(); slow = 0; }
      else if (fast >= 3 && resIdx < DEFAULT_RES) { resIdx++; resize(); fast = 0; }
    }
    frames = 0;
    fpsT = now;
  }
  hud.textContent = `rooms · ${fpsShown}fps@${canvas.width}x${canvas.height} · 'H' help` + (showHelp ? '\n\n' + HELP : '');
}
requestAnimationFrame(loop);

// GPU side of the renderer: the same raytracer the engine uses for picking
// (a walk through the rectilinear 3D grid of free/solid cells), run per pixel.
// The scene comes from Rust each frame: a uniform block, a float data texture
// (cut positions, per-plane line grids, hover rectangles, VPLs) and a 3D
// occupancy texture. Output accumulates across frames while nothing changes.

export const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const TRACE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;

uniform sampler2D uData;   // R32F, 1024 wide
uniform sampler3D uOcc;    // R8, 1 = free
uniform sampler2D uPrev;   // previous accumulation
uniform vec2 uRes;
uniform vec3 uPos, uF, uR, uU;
uniform float uAspect, uTan, uAlpha, uFrame, uArea;
uniform int uNWin, uNVpl, uVplOff;
uniform ivec3 uN, uCutOff, uCamCell; // cut counts, their offsets in uData, camera cell
uniform int uNPlanes, uPlaneOff;     // plane table: (axis, sign, pos, nu, nv, off, -, -)
uniform vec4 uHover;                 // on, axis, sign, pos
uniform int uHoverOff, uHoverN;      // rects (u0, u1, v0, v1) in uData
uniform vec4 uPreview;               // on, axis, sign, pos
uniform vec2 uPreview2;              // u, v
uniform vec4 uWin[32];               // per window: (axis, sign, pos, area) (ua, va, -, -) (u0, u1, v0, v1) (sun direction, -)
out vec4 outColor;

const float PI = 3.14159265;
const float INF = 1e30;
// keep SUN / SKY in sync with src/lib.rs
const float SUN = 4.0, SKY = 8.0, SKY_VIEW = 6.0;
uniform float uSigma;          // mist density (1/m), set from the page
#define SIGMA uSigma
const float SUN_SPREAD = 0.06; // apparent sun size: soft edges on sunlit patches
const float SHAFT = 45.0;      // sun strength for the light shafts in the air (stronger than on surfaces)
// the shafts scatter a little even with no mist (dust in the air), more with mist
#define SHAFT_SIGMA (0.012 + 3.0 * uSigma)
const float AMB_FOG = 0.05;    // haze glow per m^2 of window
const float FOG_BASE = 0.035;  // faint glow of the mist itself: far space fades to haze
const vec3 SUNCOL = vec3(1.0, 0.88, 0.70);
const vec3 SKYCOL = vec3(0.75, 0.86, 1.0);
const int VPL_PER_PIXEL = 3;
const int FOG_STEPS = 3;

uint rs;
float rnd() {
  rs = rs * 747796405u + 2891336453u;
  uint w = ((rs >> ((rs >> 28u) + 4u)) ^ rs) * 277803737u;
  w = (w >> 22u) ^ w;
  return float(w) * (1.0 / 4294967296.0);
}

float D(int i) { return texelFetch(uData, ivec2(i & 1023, i >> 10), 0).r; }
float cutv(int a, int i) { return D(uCutOff[a] + i); }
bool isFree(ivec3 c) { return texelFetch(uOcc, c, 0).r > 0.5; }
vec3 unitv(int a, float s) { vec3 v = vec3(0.0); v[a] = s; return v; }
ivec2 uvAxes(int a) { return a == 0 ? ivec2(2, 1) : (a == 1 ? ivec2(0, 2) : ivec2(0, 1)); }

// cell index of x along axis a, or -1 outside the grid
int findCell(int a, float x) {
  int lo = 0, hi = uN[a] - 1;
  if (x < cutv(a, 0) || x >= cutv(a, hi)) return -1;
  while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (cutv(a, mid) <= x) lo = mid; else hi = mid; }
  return lo;
}
ivec3 cellAt(vec3 p) { return ivec3(findCell(0, p.x), findCell(1, p.y), findCell(2, p.z)); }

struct Hit { float t; int axis; int k; ivec3 cell; };  // k: cut index of the face plane; cell: free cell in front

// first face along o + t d, starting in free cell c
bool trace(vec3 o, vec3 d, ivec3 c, float tmax, out Hit h) {
  h = Hit(INF, 0, 0, c);
  if (c.x < 0 || c.y < 0 || c.z < 0) return false;
  vec3 tn = vec3(INF);
  for (int a = 0; a < 3; a++) {
    if (d[a] > 1e-9) tn[a] = (cutv(a, c[a] + 1) - o[a]) / d[a];
    else if (d[a] < -1e-9) tn[a] = (cutv(a, c[a]) - o[a]) / d[a];
  }
  for (int it = 0; it < 400; it++) {
    int m = tn.x < tn.y ? (tn.x < tn.z ? 0 : 2) : (tn.y < tn.z ? 1 : 2);
    float t = tn[m];
    if (t >= tmax) return false;
    bool fwd = d[m] > 0.0;
    int k = fwd ? c[m] + 1 : c[m];
    ivec3 nc = c;
    nc[m] += fwd ? 1 : -1;
    if (nc[m] < 0 || nc[m] >= uN[m] - 1 || !isFree(nc)) { h = Hit(t, m, k, c); return true; }
    c = nc;
    tn[m] = fwd ? (cutv(m, c[m] + 1) - o[m]) / d[m] : (cutv(m, c[m]) - o[m]) / d[m];
  }
  return false;
}

float hitSign(Hit h, vec3 d) { return d[h.axis] > 0.0 ? -1.0 : 1.0; }
float hitPos(Hit h) { return cutv(h.axis, h.k); }

int windowAt(Hit h, float s, vec3 p) {
  float pos = hitPos(h);
  for (int w = 0; w < uNWin; w++) {
    vec4 a = uWin[4 * w];
    if (int(a.x) != h.axis || a.y != s || abs(a.z - pos) > 1e-3) continue;
    vec4 b = uWin[4 * w + 1], r = uWin[4 * w + 2];
    float u = p[int(b.x)], v = p[int(b.y)];
    if (u > r.x && u < r.y && v > r.z && v < r.w) return w;
  }
  return -1;
}

vec3 winPoint(int w, float fu, float fv) {
  vec4 a = uWin[4 * w], b = uWin[4 * w + 1], r = uWin[4 * w + 2];
  vec3 p = vec3(0.0);
  p[int(a.x)] = a.z;
  p[int(b.x)] = mix(r.x, r.y, fu);
  p[int(b.y)] = mix(r.z, r.w, fv);
  return p;
}

// light arriving at p (normal n, free cell c) straight from the windows
vec3 direct(vec3 p, vec3 n, ivec3 c) {
  vec3 e = vec3(0.0);
  for (int w = 0; w < uNWin; w++) {
    vec4 a = uWin[4 * w], b = uWin[4 * w + 1], r = uWin[4 * w + 2];
    vec3 ld = normalize(-uWin[4 * w + 3].xyz + SUN_SPREAD * (vec3(rnd(), rnd(), rnd()) - 0.5));  // toward the sun
    int ax = int(a.x), ua = int(b.x), va = int(b.y);
    float ws = a.y, wpos = a.z, area = a.w;
    // sun beam: does the line back toward the sun leave through this window?
    float cb = dot(n, ld);
    if (cb > 0.0 && ld[ax] * ws < -1e-4) {
      float t = (wpos - p[ax]) / ld[ax];
      vec3 q = p + ld * t;
      if (t > 0.0 && q[ua] > r.x && q[ua] < r.y && q[va] > r.z && q[va] < r.w) {
        Hit h;
        if (trace(p, ld, c, t + 0.02, h) && windowAt(h, hitSign(h, ld), p + ld * h.t) == w) e += SUNCOL * (SUN * cb);
      }
    }
    // sky through the window (area light, one sample)
    vec3 l = winPoint(w, rnd(), rnd()) - p;
    float r2 = max(dot(l, l), area * 0.5), dist = sqrt(r2);
    l /= dist;
    float cp = dot(n, l), cw = -ws * l[ax];
    if (cp > 0.0 && cw > 0.0) {
      Hit h;
      if (!trace(p, l, c, dist - 0.01, h)) e += SKYCOL * (SKY * area * cp * cw / r2);
    }
  }
  return e;
}

// one bounce: a few random VPLs (spots where window light lands)
vec3 indirect(vec3 p, vec3 n, ivec3 c) {
  if (uNVpl == 0) return vec3(0.0);
  vec3 e = vec3(0.0);
  for (int k = 0; k < VPL_PER_PIXEL; k++) {
    int o = uVplOff + 9 * min(int(rnd() * float(uNVpl)), uNVpl - 1);
    vec3 lp = vec3(D(o), D(o + 1), D(o + 2));
    vec3 ln = vec3(D(o + 3), D(o + 4), D(o + 5));
    vec3 dv = lp - p;
    float d2 = dot(dv, dv), dist = sqrt(d2);
    vec3 dir = dv / dist;
    float cp = dot(n, dir), cl = -dot(ln, dir);
    if (cp <= 0.0 || cl <= 0.0) continue;
    Hit h;
    if (!trace(p, dir, c, dist - 0.01, h)) e += vec3(D(o + 6), D(o + 7), D(o + 8)) * (cp * cl / (PI * max(d2, 0.25)));
  }
  return e * (float(uNVpl) / float(VPL_PER_PIXEL));
}

int countLE(int off, int n, float x) { int i = 0; while (i < n && D(off + i) <= x) i++; return i; }

// is (u, v) on a drawn line of the plane (axis, s, pos)?
bool onLine(int axis, float s, float pos, float u, float v, float width) {
  for (int i = 0; i < uNPlanes; i++) {
    int t = uPlaneOff + 8 * i;
    if (int(D(t)) != axis || D(t + 1) != s || abs(D(t + 2) - pos) > 1e-3) continue;
    int nu = int(D(t + 3)), nv = int(D(t + 4)), off = int(D(t + 5));
    int usO = off, vsO = off + nu, usegO = off + nu + nv, vsegO = usegO + nu * (nv + 1);
    for (int j = 0; j < nu; j++)
      if (abs(u - D(usO + j)) < width && D(usegO + j * (nv + 1) + countLE(vsO, nv, v)) > 0.5) return true;
    for (int j = 0; j < nv; j++)
      if (abs(v - D(vsO + j)) < width && D(vsegO + j * (nu + 1) + countLE(usO, nu, u)) > 0.5) return true;
    return false;
  }
  return false;
}

vec3 albedo(Hit h, float s, vec3 p) {
  float pos = hitPos(h);
  ivec2 uv = uvAxes(h.axis);
  float u = p[uv.x], v = p[uv.y];
  vec3 a;
  if (h.axis == 1 && s > 0.0 && pos < 1e-3) {
    // the ground
    a = vec3(0.40, 0.38, 0.35);
    vec2 f = fract(p.xz / 0.6);
    float w = 0.01 + 0.002 * h.t;
    if (f.x < w || f.x > 1.0 - w || f.y < w || f.y > 1.0 - w) a *= 0.8;
    return a;
  }
  a = (h.axis == 1 && s < 0.0) ? vec3(0.80, 0.79, 0.76) : vec3(0.78, 0.74, 0.64);
  if (onLine(h.axis, s, pos, u, v, 0.004 + 0.003 * h.t)) a *= 0.3;
  if (uHover.x > 0.5 && int(uHover.y) == h.axis && uHover.z == s && abs(uHover.w - pos) < 1e-3) {
    for (int i = 0; i < uHoverN; i++) {
      int o = uHoverOff + 4 * i;
      if (u >= D(o) && u <= D(o + 1) && v >= D(o + 2) && v <= D(o + 3)) {
        a = a * vec3(0.93, 1.0, 1.12) + vec3(0.02, 0.03, 0.05);
        break;
      }
    }
  }
  return a;
}

void clip(inout float ta, inout float tb, float a, float b, float lo, float hi) {
  if (abs(b) < 1e-9) { if (a < lo || a > hi) tb = ta - 1.0; return; }
  float t1 = (lo - a) / b, t2 = (hi - a) / b;
  ta = max(ta, min(t1, t2));
  tb = min(tb, max(t1, t2));
}

// mist: base haze, sun shafts (marched only inside each beam's light prism), sky glow
vec3 fog(vec3 o, vec3 d, float thit) {
  vec3 l = vec3(0.8, 0.82, 0.85) * ((FOG_BASE + AMB_FOG * uArea) * (1.0 - exp(-SIGMA * thit)));
  for (int w = 0; w < uNWin; w++) {
    vec4 a = uWin[4 * w], b = uWin[4 * w + 1], r = uWin[4 * w + 2];
    vec3 beam = uWin[4 * w + 3].xyz, ld = -beam;
    int ax = int(a.x), ua = int(b.x), va = int(b.y);
    float ws = a.y, wpos = a.z, area = a.w;
    // s(t): distance along ld from the ray point to the window plane
    float s0 = (wpos - o[ax]) / ld[ax], s1 = -d[ax] / ld[ax];
    float ta = 0.0, tb = ld[ax] * ws < -1e-4 ? thit : -1.0; // no shaft if the window faces away from the sun
    clip(ta, tb, s0, s1, 0.0, INF);
    clip(ta, tb, o[ua] + ld[ua] * s0, d[ua] + ld[ua] * s1, r.x, r.y);
    clip(ta, tb, o[va] + ld[va] * s0, d[va] + ld[va] * s1, r.z, r.w);
    if (tb > ta) {
      float g = 0.55, cs = dot(beam, -d);
      float ph = (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * cs, 1.5));
      float dt = (tb - ta) / float(FOG_STEPS), j = rnd();
      for (int k = 0; k < FOG_STEPS; k++) {
        float t = ta + (float(k) + j) * dt;
        vec3 x = o + d * t;
        Hit h;
        if (trace(x, ld, cellAt(x), s0 + s1 * t + 0.02, h) && windowAt(h, hitSign(h, ld), x + ld * h.t) == w)
          l += SUNCOL * (SHAFT * ph * SHAFT_SIGMA * dt * exp(-SIGMA * t));
      }
    }
    float tl = min(thit, 25.0), t = rnd() * tl;
    vec3 x = o + d * t;
    vec3 lv = winPoint(w, rnd(), rnd()) - x;
    float r2 = max(dot(lv, lv), area), dist = sqrt(r2);
    vec3 ldir = lv / dist;
    float cw = -ws * ldir[ax];
    Hit h;
    if (cw > 0.0 && !trace(x, ldir, cellAt(x), dist - 0.01, h))
      l += SKYCOL * (SKY * area * cw / r2 / (4.0 * PI) * SIGMA * tl * exp(-SIGMA * t));
  }
  return l;
}

vec3 radiance(vec3 o, vec3 d) {
  Hit h;
  if (!trace(o, d, uCamCell, INF, h)) return vec3(0.0);
  vec3 p = o + d * h.t;
  float s = hitSign(h, d);
  vec3 L;
  if (windowAt(h, s, p) >= 0) {
    L = SKYCOL * SKY_VIEW;
  } else {
    vec3 n = unitv(h.axis, s);
    vec3 pp = p + n * 1e-3;
    L = albedo(h, s, p) * (direct(pp, n, h.cell) + indirect(pp, n, h.cell)) / PI;
    if (uPreview.x > 0.5 && int(uPreview.y) == h.axis && uPreview.z == s && abs(uPreview.w - hitPos(h)) < 1e-3) {
      ivec2 uv = uvAxes(h.axis);
      float w = 0.003 + 0.002 * h.t;
      if (abs(p[uv.x] - uPreview2.x) < w || abs(p[uv.y] - uPreview2.y) < w) L = L * 0.4 + vec3(0.30, 0.24, 0.14);
    }
  }
  return L * exp(-SIGMA * h.t) + fog(o, d, h.t);
}

void main() {
  ivec2 pix = ivec2(gl_FragCoord.xy);
  rs = uint(pix.x) * 1973u + uint(pix.y) * 9277u + uint(uFrame) * 26699u + 1u;
  rnd(); rnd();
  vec2 sc = (vec2(pix) + vec2(rnd(), rnd())) / uRes;
  vec3 d = normalize(uF + uR * ((2.0 * sc.x - 1.0) * uTan) + uU * ((2.0 * sc.y - 1.0) * uTan * uAspect));
  vec3 L = min(radiance(uPos, d), vec3(40.0)); // clamp rare fireflies
  vec3 prev = texelFetch(uPrev, pix, 0).rgb;
  outColor = vec4(mix(prev, L, uAlpha), 1.0);
}`;

export const SHOW = `#version 300 es
precision highp float;
uniform sampler2D uAcc;
uniform vec2 uRes;
uniform float uFrame;
out vec4 outColor;
const float EXPOSURE = 2.2;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + uFrame * 0.618) * 43758.5453); }
void main() {
  vec3 c = texelFetch(uAcc, ivec2(gl_FragCoord.xy), 0).rgb;
  vec2 q = gl_FragCoord.xy / uRes * 2.0 - 1.0;
  float vig = 1.0 - 0.22 * dot(q, q);
  vec3 m = 1.0 - exp(-c * EXPOSURE * vig);
  m = pow(m, vec3(1.0 / 2.2)) + (hash(gl_FragCoord.xy) - 0.5) / 255.0;
  outColor = vec4(m, 1.0);
}`;

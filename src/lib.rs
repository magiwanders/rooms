//! rooms: a liminal room-shaping toy. This crate is the engine: geometry,
//! editing, picking, collisions, undo and light placement. The per-pixel
//! raytracer runs on the GPU (web/shader.js); every frame Rust packs the scene
//! into a uniform block, a float data texture and a 3D occupancy texture.
//!
//! Geometry model
//! - Space is a rectilinear 3D grid: sorted cut positions per axis, and one
//!   free/solid flag per cell. Outside the grid is solid. The ground is y = 0.
//! - A face is where a free cell meets a solid cell (or the grid edge). Faces
//!   lie on planes identified by (axis, normal sign, position).
//! - Lines belong to a plane: each plane has its own 2D line grid (lines run
//!   across the whole plane, split into segments at intersections; segments
//!   can be removed one by one). Drawing a line also adds a grid cut there.
//! - A region is a connected patch of face cells on one plane, bounded by the
//!   face's edges and by line segments. Pushing a region carves the cells
//!   behind it one STEP deep; pulling fills the cells in front of it. Its
//!   lines and windows move with it.

#![allow(static_mut_refs)]

use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------- tunables
const STEP: f32 = 0.25; // push/pull amount per click (m)
const MAXD: f32 = 3000.0; // max distance of any plane from the origin (m)
const EYE: f32 = 1.65; // eye height for a 1.75 m tall player
const BODY_R: f32 = 0.22;
const SPEED: f32 = 3.2; // top walking speed (m/s)
const ACCEL: f32 = 9.0; // speeding up (m/s^2), plus ACCEL_K per m/s of target speed
const ACCEL_K: f32 = 1.5;
const DECEL: f32 = 7.0; // slowing down when keys are released (m/s^2), plus DECEL_K per m/s of speed
const DECEL_K: f32 = 2.5;
/// Holding a move key keeps raising the top speed, logarithmically:
/// SPEED * (1 + WALK_GROWTH * ln(1 + held / WALK_TAU)). ~17 m/s after 1 s, ~28 after 3 s, ~42 after 10 s.
const WALK_GROWTH: f32 = 4.0;
const WALK_TAU: f32 = 0.5;
const SUBSTEP: f32 = 0.1; // max movement per collision check (m), so fast walking can't tunnel
const HFOV_TAN: f32 = 0.767; // tan(75deg / 2)
const MIN_WIN_Y: f32 = 2.0; // no windows in the first 2 m
const MAX_WINS: usize = 32; // per room (the shader samples a few per pixel, so this stays cheap)
/// Outside light reaching a window: OUTSIDE_MIN for one facing away from the sun, up to 1
/// for one facing it with nothing outside blocking the sun.
const OUTSIDE_MIN: f32 = 0.3;
const MAX_CUTS: usize = 160; // per axis
const MAX_UNDO: usize = 300;
const MAX_HOVER_RECTS: usize = 256;
const EPS: f32 = 1e-4;
// light: keep SUN / SKY in sync with web/shader.js
const SUN: f32 = 4.0;
const SKY: f32 = 8.0;
const BOUNCE: f32 = 2.2; // boost on the one-bounce VPL light (stands in for more bounces)
const VPL_SUN: usize = 16; // virtual point lights per window from the sun beam
const VPL_SKY: usize = 8; // ... and from sky light
const SUNCOL: V = v(1.0, 0.88, 0.70);
/// Direction sunlight travels (normalized in sun_dir()): one sun for every window.
/// Windows that don't face it get sky light but no beam.
const SUN_DIR: V = v(0.2, -0.85, 0.75);
fn sun_dir() -> V {
    SUN_DIR.norm()
}
const SKYCOL: V = v(0.75, 0.86, 1.0);
const INF: f32 = 1e30;
/// Uniform block layout (floats), mirrored in web/main.js.
const U_LEN: usize = 192 + 16 * MAX_WINS; // windows from 192 on, 16 floats each
/// Floor tiles are TILE x TILE metres (the grout lines in the shader match). Any tile
/// can be toggled into a mirror; the mirror mask is sent as a box of at most
/// MIRROR_SPAN x MIRROR_SPAN tiles.
const TILE: f32 = 0.6;
const MIRROR_SPAN: i32 = 512;
const MIRROR_R: f32 = 0.85; // how much light a mirror tile reflects
/// A window's largest side is WIN_SIZE_PER_M times the height of its centre (1 m at 2 m):
/// growing with distance from the eye, so the largest window looks about the same size
/// on screen at any height (its area grows with the square of the height).
const WIN_SIZE_PER_M: f32 = 0.5;
/// Highest a room can reach: the largest ceiling window there throws a beam about as wide
/// as the largest mirror area the renderer shows (MIRROR_SPAN tiles), so it can still be
/// reflected in full when it reaches the floor. (512 * 0.6 m / 0.5 = 614.4 m)
const MAX_HEIGHT: f32 = MIRROR_SPAN as f32 * TILE / WIN_SIZE_PER_M;
/// Largest half-width / half-height of a window centred at height y.
fn win_half_max(y: f32) -> f32 {
    0.5 * WIN_SIZE_PER_M * y.max(0.0)
}

// ---------------------------------------------------------------- vectors
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct V {
    x: f32,
    y: f32,
    z: f32,
}
const fn v(x: f32, y: f32, z: f32) -> V {
    V { x, y, z }
}
impl V {
    #[inline]
    fn get(self, a: usize) -> f32 {
        match a {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
    #[inline]
    fn set(&mut self, a: usize, val: f32) {
        match a {
            0 => self.x = val,
            1 => self.y = val,
            _ => self.z = val,
        }
    }
    #[inline]
    fn dot(self, o: V) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline]
    fn cross(self, o: V) -> V {
        v(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    #[inline]
    fn norm(self) -> V {
        self * (1.0 / self.dot(self).sqrt())
    }
    #[inline]
    fn mulv(self, o: V) -> V {
        v(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}
impl Add for V {
    type Output = V;
    fn add(self, o: V) -> V {
        v(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for V {
    type Output = V;
    fn sub(self, o: V) -> V {
        v(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for V {
    type Output = V;
    fn mul(self, s: f32) -> V {
        v(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for V {
    type Output = V;
    fn neg(self) -> V {
        v(-self.x, -self.y, -self.z)
    }
}
fn unit(a: usize, s: f32) -> V {
    let mut r = V::default();
    r.set(a, s);
    r
}

/// In-plane axes (u, v) for a plane whose normal is along `axis`.
/// Walls: u is horizontal, v is height. Floors/ceilings: u = x, v = z.
fn uva(axis: usize) -> (usize, usize) {
    match axis {
        0 => (2, 1),
        1 => (0, 2),
        _ => (0, 1),
    }
}

// ---------------------------------------------------------------- rng
fn rnd(s: &mut u32) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    (*s >> 8) as f32 * (1.0 / 16_777_216.0)
}
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}
fn cos_sample(n: V, r: &mut u32) -> V {
    let a = if n.x.abs() > 0.5 { v(0.0, 1.0, 0.0) } else { v(1.0, 0.0, 0.0) };
    let t = a.cross(n).norm();
    let b = n.cross(t);
    let phi = 2.0 * PI * rnd(r);
    let r2 = rnd(r);
    let sr = r2.sqrt();
    t * (phi.cos() * sr) + b * (phi.sin() * sr) + n * (1.0 - r2).sqrt()
}

/// Number of cuts <= x, i.e. the interval index of x.
fn find(cuts: &[f32], x: f32) -> usize {
    let mut i = 0;
    while i < cuts.len() && cuts[i] <= x {
        i += 1;
    }
    i
}

/// A point inside interval r of `cuts` (intervals extend to infinity at both ends).
fn rep(cuts: &[f32], r: usize) -> f32 {
    match (r == 0, r == cuts.len()) {
        (true, true) => 0.0,
        (true, false) => cuts[0] - 1.0,
        (false, true) => cuts[r - 1] + 1.0,
        _ => 0.5 * (cuts[r - 1] + cuts[r]),
    }
}

// ---------------------------------------------------------------- grid
#[derive(Clone)]
struct Grid {
    c: [Vec<f32>; 3], // cut positions per axis, sorted, at least 2 each
    occ: Vec<u8>,     // 1 = free; index i + nx * (j + ny * k)
}

impl Grid {
    fn new() -> Self {
        Grid { c: [vec![-1.5, 1.5], vec![0.0, 3.0], vec![-1.5, 1.5]], occ: vec![1] }
    }
    fn n(&self, a: usize) -> usize {
        self.c[a].len() - 1
    }
    fn idx(&self, i: [usize; 3]) -> usize {
        i[0] + self.n(0) * (i[1] + self.n(1) * i[2])
    }
    fn free(&self, i: [usize; 3]) -> bool {
        self.occ[self.idx(i)] == 1
    }
    fn cell_of(&self, a: usize, x: f32) -> Option<usize> {
        let c = &self.c[a];
        if !(x >= c[0] && x < c[c.len() - 1]) {
            return None; // outside, or NaN
        }
        Some(c.partition_point(|&q| q <= x) - 1)
    }
    fn cell_at(&self, p: V) -> Option<[usize; 3]> {
        Some([self.cell_of(0, p.x)?, self.cell_of(1, p.y)?, self.cell_of(2, p.z)?])
    }
    fn free_at(&self, p: V) -> bool {
        self.cell_at(p).is_some_and(|i| self.free(i))
    }
    fn find_cut(&self, a: usize, x: f32) -> Option<usize> {
        self.c[a].iter().position(|&q| (q - x).abs() < EPS)
    }

    /// Rebuild occupancy with axis `a` resized to `n_new` layers; `map` gives the
    /// old layer for each new one (None = solid).
    fn remap(&mut self, a: usize, n_new: usize, map: impl Fn(usize) -> Option<usize>) {
        let mut dims = [self.n(0), self.n(1), self.n(2)];
        dims[a] = n_new;
        let mut occ = vec![0u8; dims[0] * dims[1] * dims[2]];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let mut src = [i, j, k];
                    if let Some(l) = map(src[a]) {
                        src[a] = l;
                        occ[i + dims[0] * (j + dims[1] * k)] = self.occ[self.idx(src)];
                    }
                }
            }
        }
        self.occ = occ;
    }

    /// Add a cut at x on axis a (no-op if one is there). Returns its index.
    fn insert_cut(&mut self, a: usize, x: f32) -> Option<usize> {
        if let Some(k) = self.find_cut(a, x) {
            return Some(k);
        }
        if self.c[a].len() >= MAX_CUTS {
            return None;
        }
        let n = self.n(a);
        let len = self.c[a].len();
        let k = self.c[a].partition_point(|&q| q < x);
        if k == 0 {
            self.remap(a, n + 1, |l| if l == 0 { None } else { Some(l - 1) });
        } else if k == len {
            self.remap(a, n + 1, |l| if l == n { None } else { Some(l) });
        } else {
            self.remap(a, n + 1, |l| Some(if l < k { l } else { l - 1 }));
        }
        self.c[a].insert(k, x);
        Some(k)
    }

    fn layer(&self, a: usize, l: usize) -> Vec<u8> {
        let n = [self.n(0), self.n(1), self.n(2)];
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let mut out = Vec::with_capacity(n[b] * n[c]);
        for y in 0..n[c] {
            for x in 0..n[b] {
                let mut i = [0; 3];
                i[a] = l;
                i[b] = x;
                i[c] = y;
                out.push(self.occ[self.idx(i)]);
            }
        }
        out
    }

    /// Remove cut k on axis a if nothing distinguishes the two sides.
    fn try_remove_cut(&mut self, a: usize, k: usize) -> bool {
        let n = self.n(a);
        if n <= 1 {
            return false;
        }
        let removable = if k == 0 {
            self.layer(a, 0).iter().all(|&o| o == 0)
        } else if k == n {
            self.layer(a, n - 1).iter().all(|&o| o == 0)
        } else {
            self.layer(a, k - 1) == self.layer(a, k)
        };
        if !removable {
            return false;
        }
        if k == 0 {
            self.remap(a, n - 1, |l| Some(l + 1));
        } else {
            self.remap(a, n - 1, |l| Some(if l < k { l } else { l + 1 }));
        }
        self.c[a].remove(k);
        true
    }

    /// First face hit along o + t d: (t, axis, plane cut index, free cell in front).
    fn trace(&self, o: V, d: V, tmax: f32) -> Option<(f32, usize, usize, [usize; 3])> {
        let mut c = self.cell_at(o)?;
        if !self.free(c) {
            return Some((0.0, 0, c[0], c));
        }
        let mut tn = [INF; 3];
        for a in 0..3 {
            let da = d.get(a);
            if da > 1e-9 {
                tn[a] = (self.c[a][c[a] + 1] - o.get(a)) / da;
            } else if da < -1e-9 {
                tn[a] = (self.c[a][c[a]] - o.get(a)) / da;
            }
        }
        loop {
            let m = if tn[0] < tn[1] { if tn[0] < tn[2] { 0 } else { 2 } } else if tn[1] < tn[2] { 1 } else { 2 };
            let t = tn[m];
            if t >= tmax {
                return None;
            }
            let fwd = d.get(m) > 0.0;
            let k = if fwd { c[m] + 1 } else { c[m] };
            let out = if fwd { c[m] + 1 >= self.n(m) } else { c[m] == 0 };
            if out {
                return Some((t, m, k, c));
            }
            let mut nc = c;
            if fwd { nc[m] += 1 } else { nc[m] -= 1 }
            if !self.free(nc) {
                return Some((t, m, k, c));
            }
            c = nc;
            tn[m] = if fwd { (self.c[m][c[m] + 1] - o.get(m)) / d.get(m) } else { (self.c[m][c[m]] - o.get(m)) / d.get(m) };
        }
    }

    /// Sunlight from outside: does the ray from `o` (just outside a window) along `d`
    /// leave the grid without passing through any free cell, i.e. any other part of
    /// the room? A wing of the room standing in the way shades the window.
    fn sun_clear(&self, o: V, d: V) -> bool {
        // enter the grid's bounding box if starting outside it
        let (mut t0, mut t1) = (0.0f32, INF);
        for a in 0..3 {
            let (lo, hi) = (self.c[a][0], self.c[a][self.c[a].len() - 1]);
            let da = d.get(a);
            if da.abs() < 1e-9 {
                if o.get(a) < lo || o.get(a) > hi {
                    return true;
                }
            } else {
                let (mut ta, mut tb) = ((lo - o.get(a)) / da, (hi - o.get(a)) / da);
                if ta > tb {
                    std::mem::swap(&mut ta, &mut tb);
                }
                t0 = t0.max(ta);
                t1 = t1.min(tb);
            }
        }
        if t1 <= t0 {
            return true;
        }
        let o = o + d * (t0 + 1e-4);
        let Some(mut c) = self.cell_at(o) else { return true };
        let mut tn = [INF; 3];
        for a in 0..3 {
            let da = d.get(a);
            if da > 1e-9 {
                tn[a] = (self.c[a][c[a] + 1] - o.get(a)) / da;
            } else if da < -1e-9 {
                tn[a] = (self.c[a][c[a]] - o.get(a)) / da;
            }
        }
        loop {
            if self.free(c) {
                return false;
            }
            let m = if tn[0] < tn[1] { if tn[0] < tn[2] { 0 } else { 2 } } else if tn[1] < tn[2] { 1 } else { 2 };
            let fwd = d.get(m) > 0.0;
            if (fwd && c[m] + 1 >= self.n(m)) || (!fwd && c[m] == 0) {
                return true;
            }
            if fwd { c[m] += 1 } else { c[m] -= 1 }
            tn[m] = if fwd { (self.c[m][c[m] + 1] - o.get(m)) / d.get(m) } else { (self.c[m][c[m]] - o.get(m)) / d.get(m) };
        }
    }

    /// Is (iu, iv) a face cell of the plane at cut k on `axis` with normal sign s?
    fn is_face(&self, axis: usize, s: f32, k: usize, iu: usize, iv: usize) -> bool {
        let (ua, va) = uva(axis);
        let n = self.n(axis);
        let (free_l, solid_l) = if s > 0.0 { (k as isize, k as isize - 1) } else { (k as isize - 1, k as isize) };
        if free_l < 0 || free_l >= n as isize {
            return false;
        }
        let mut i = [0; 3];
        i[ua] = iu;
        i[va] = iv;
        i[axis] = free_l as usize;
        if !self.free(i) {
            return false;
        }
        if solid_l < 0 || solid_l >= n as isize {
            return true;
        }
        i[axis] = solid_l as usize;
        !self.free(i)
    }
}

// ---------------------------------------------------------------- lines
/// The lines drawn on one plane: a 2D grid of infinite lines, split into
/// segments at intersections.
#[derive(Clone, Default)]
struct Lines {
    us: Vec<f32>,         // u lines (constant u), sorted
    vs: Vec<f32>,         // v lines
    useg: Vec<Vec<bool>>, // per u line: segment present, per v interval
    vseg: Vec<Vec<bool>>, // per v line: segment present, per u interval
}

impl Lines {
    fn cut_u(&self, x: f32) -> Option<usize> {
        self.us.iter().position(|&u| (u - x).abs() < EPS)
    }
    fn cut_v(&self, x: f32) -> Option<usize> {
        self.vs.iter().position(|&w| (w - x).abs() < EPS)
    }
    fn transpose(&mut self) {
        std::mem::swap(&mut self.us, &mut self.vs);
        std::mem::swap(&mut self.useg, &mut self.vseg);
    }
    /// Add a u line at x; `present` = all its segments drawn. Returns its index.
    fn insert_u(&mut self, x: f32, present: bool) -> usize {
        if let Some(i) = self.cut_u(x) {
            if present {
                self.useg[i].iter_mut().for_each(|b| *b = true);
            }
            return i;
        }
        let idx = self.us.partition_point(|&u| u < x);
        self.us.insert(idx, x);
        self.useg.insert(idx, vec![present; self.vs.len() + 1]);
        for f in &mut self.vseg {
            let b = f[idx];
            f.insert(idx, b);
        }
        idx
    }
    fn insert_v(&mut self, x: f32, present: bool) -> usize {
        self.transpose();
        let i = self.insert_u(x, present);
        self.transpose();
        i
    }
    /// Is there a drawn u-line segment at u = x covering height w?
    fn blocks_u(&self, x: f32, w: f32) -> bool {
        self.cut_u(x).is_some_and(|i| self.useg[i][find(&self.vs, w)])
    }
    fn blocks_v(&self, x: f32, u: f32) -> bool {
        self.cut_v(x).is_some_and(|j| self.vseg[j][find(&self.us, u)])
    }
    fn erase(&mut self, u: f32, w: f32, r: f32) -> bool {
        let (iu, iv) = (find(&self.us, u), find(&self.vs, w));
        let mut changed = false;
        for i in 0..self.us.len() {
            if (u - self.us[i]).abs() < r && self.useg[i][iv] {
                self.useg[i][iv] = false;
                changed = true;
            }
        }
        for j in 0..self.vs.len() {
            if (w - self.vs[j]).abs() < r && self.vseg[j][iu] {
                self.vseg[j][iu] = false;
                changed = true;
            }
        }
        if changed {
            self.cleanup();
        }
        changed
    }
    fn cleanup_u(&mut self) -> bool {
        for i in 0..self.us.len() {
            if self.useg[i].iter().all(|&b| !b) && self.vseg.iter().all(|f| f[i] == f[i + 1]) {
                self.us.remove(i);
                self.useg.remove(i);
                for f in &mut self.vseg {
                    f.remove(i + 1);
                }
                return true;
            }
        }
        false
    }
    /// Drop lines with no drawn segment that don't split anything.
    fn cleanup(&mut self) {
        loop {
            let a = self.cleanup_u();
            self.transpose();
            let b = self.cleanup_u();
            self.transpose();
            if !a && !b {
                break;
            }
        }
    }
    /// Union with another plane's lines.
    fn merge_from(&mut self, o: &Lines) {
        for &x in &o.us {
            self.insert_u(x, false);
        }
        for &x in &o.vs {
            self.insert_v(x, false);
        }
        for (i, &x) in o.us.iter().enumerate() {
            let j = self.cut_u(x).unwrap();
            for r in 0..=self.vs.len() {
                if o.useg[i][find(&o.vs, rep(&self.vs, r))] {
                    self.useg[j][r] = true;
                }
            }
        }
        for (i, &x) in o.vs.iter().enumerate() {
            let j = self.cut_v(x).unwrap();
            for r in 0..=self.us.len() {
                if o.vseg[i][find(&o.us, rep(&self.us, r))] {
                    self.vseg[j][r] = true;
                }
            }
        }
    }
    fn is_empty(&self) -> bool {
        self.us.is_empty() && self.vs.is_empty()
    }
}

/// Plane key: (normal axis, normal sign, position in mm).
type Key = (u8, i8, i64);
fn key(axis: usize, s: f32, pos: f32) -> Key {
    (axis as u8, if s > 0.0 { 1 } else { -1 }, (pos * 1000.0).round() as i64)
}

// ---------------------------------------------------------------- windows
#[derive(Clone)]
struct Win {
    axis: usize,
    s: f32,   // normal sign (points into the room)
    pos: f32, // plane position
    uc: f32,
    vc: f32,
    hu: f32,
    hv: f32,
}

impl Win {
    fn beam(&self) -> V {
        sun_dir()
    }
    fn area(&self) -> f32 {
        4.0 * self.hu * self.hv
    }
    /// Height of the window's centre above the ground.
    fn height(&self) -> f32 {
        if self.axis == 1 { self.pos } else { self.vc }
    }
    fn point(&self, fu: f32, fv: f32) -> V {
        let (ua, va) = uva(self.axis);
        let mut p = V::default();
        p.set(self.axis, self.pos);
        p.set(ua, self.uc + (2.0 * fu - 1.0) * self.hu);
        p.set(va, self.vc + (2.0 * fv - 1.0) * self.hv);
        p
    }
}

/// A region locked for a held click or an E-drag: it keeps moving as one piece even if
/// it lines up with a neighbouring face on the way (which would otherwise merge them).
#[derive(Clone)]
struct Grip {
    axis: usize,
    s: f32,
    pos: f32,
    rects: Vec<[f32; 4]>,
}

/// Everything an edit can change (for undo/redo).
#[derive(Clone)]
struct Snap {
    grid: Grid,
    lines: HashMap<Key, Lines>,
    wins: Vec<Win>,
    mirrors: HashSet<(i32, i32)>,
}

/// A virtual point light: a spot where window light lands, re-emitting it.
struct Vpl {
    p: V,
    n: V,
    power: V,
}

struct Hit {
    t: f32,
    axis: usize,
    k: usize,         // cut index of the face's plane
    cell: [usize; 3], // free cell in front of the face
    s: f32,           // normal sign
    pos: f32,         // plane position
    p: V,
    n: V,
}

impl Hit {
    fn key(&self) -> Key {
        key(self.axis, self.s, self.pos)
    }
    fn uv(&self) -> (f32, f32) {
        let (ua, va) = uva(self.axis);
        (self.p.get(ua), self.p.get(va))
    }
    fn is_ground(&self) -> bool {
        self.axis == 1 && self.s > 0.0 && self.pos.abs() < EPS
    }
}

// ---------------------------------------------------------------- state
struct State {
    grid: Grid,
    lines: HashMap<Key, Lines>,
    wins: Vec<Win>,
    mirrors: HashSet<(i32, i32)>, // floor tiles that are mirrors
    tile_last: Option<(i32, i32)>, // tile under the crosshair last during a click-drag
    tile_paint: bool,              // what the first click made its tile (mirror or not): a drag paints that
    tile_hover: Option<(i32, i32)>,
    pos: V,
    vel: (f32, f32), // horizontal velocity (x, z)
    walk_t: f32,     // how long a move key has been held (s)
    yaw: f32,
    pitch: f32,
    w: usize,
    h: usize,
    frame: u32,
    nacc: u32,
    dirty: bool,
    occ_version: u32,
    hover: (f32, f32),
    hover_desc: Vec<f32>, // [axis, sign, pos, rects (u0, u1, v0, v1)...]
    preview: bool,
    preview_at: [f32; 6],
    drag_win: Option<usize>,
    grip: Option<Grip>,  // the region held by a click or an E-drag
    extrude_hint: bool,  // E held: highlight the region under the crosshair differently
    ubuf: Vec<f32>,
    data: Vec<f32>,
    undo: Vec<Snap>,
    redo: Vec<Snap>,
    group_open: bool,  // a mouse press is in progress: its edits form one undo step
    group_saved: bool, // ... and that step's snapshot is already stored
    io: Vec<u8>,       // save/load text exchanged with the page
}

static mut STATE: Option<State> = None;
fn st() -> &'static mut State {
    unsafe { STATE.as_mut().expect("init() not called") }
}

impl State {
    fn new() -> Self {
        State {
            grid: Grid::new(),
            lines: HashMap::new(),
            mirrors: HashSet::new(),
            tile_last: None,
            tile_paint: true,
            tile_hover: None,
            wins: vec![Win { axis: 2, s: 1.0, pos: -1.5, uc: 0.25, vc: 2.62, hu: 0.32, hv: 0.18 }],
            pos: v(-0.55, EYE, 1.05),
            vel: (0.0, 0.0),
            walk_t: 0.0,
            yaw: 0.42,
            pitch: 0.12,
            w: 0,
            h: 0,
            frame: 0,
            nacc: 0,
            dirty: true,
            occ_version: 1,
            hover: (-1.0, -1.0),
            hover_desc: vec![],
            preview: false,
            preview_at: [0.0; 6],
            drag_win: None,
            grip: None,
            extrude_hint: false,
            ubuf: vec![0.0; U_LEN],
            data: vec![],
            undo: vec![],
            redo: vec![],
            group_open: false,
            group_saved: false,
            io: vec![],
        }
    }

    // ---- save / load (plain text: whitespace-separated numbers)
    fn serialize(&self) -> String {
        use std::fmt::Write;
        let mut o = String::from("rooms 1\n");
        let _ = writeln!(o, "{} {} {} {}", self.pos.x, self.pos.z, self.yaw, self.pitch);
        for a in 0..3 {
            let _ = write!(o, "{}", self.grid.c[a].len());
            for x in &self.grid.c[a] {
                let _ = write!(o, " {x}");
            }
            o.push('\n');
        }
        // occupancy, run-length encoded
        let occ = &self.grid.occ;
        let mut runs = vec![];
        let mut i = 0;
        while i < occ.len() {
            let mut j = i;
            while j < occ.len() && occ[j] == occ[i] {
                j += 1;
            }
            runs.push((occ[i], j - i));
            i = j;
        }
        let _ = write!(o, "{}", runs.len());
        for (val, n) in runs {
            let _ = write!(o, " {val} {n}");
        }
        o.push('\n');
        let _ = writeln!(o, "{}", self.lines.len());
        let mut planes: Vec<_> = self.lines.iter().collect();
        planes.sort_by_key(|(k, _)| **k);
        for (k, l) in planes {
            let _ = write!(o, "{} {} {} {}", k.0, k.1, k.2, l.us.len());
            for x in &l.us {
                let _ = write!(o, " {x}");
            }
            let _ = write!(o, " {}", l.vs.len());
            for x in &l.vs {
                let _ = write!(o, " {x}");
            }
            for seg in l.useg.iter().chain(l.vseg.iter()) {
                for &b in seg {
                    let _ = write!(o, " {}", b as u8);
                }
            }
            o.push('\n');
        }
        let _ = writeln!(o, "{}", self.wins.len());
        for w in &self.wins {
            let _ = writeln!(o, "{} {} {} {} {} {} {}", w.axis, w.s, w.pos, w.uc, w.vc, w.hu, w.hv);
        }
        // mirror tiles (optional section, absent in older saves)
        let mut tiles: Vec<_> = self.mirrors.iter().collect();
        tiles.sort();
        let _ = write!(o, "{}", tiles.len());
        for (i, j) in tiles {
            let _ = write!(o, " {i} {j}");
        }
        o.push('\n');
        o
    }

    /// Parse a saved room; on any error nothing changes.
    fn deserialize(&mut self, text: &str) -> Option<()> {
        let mut t = text.split_whitespace();
        if t.next()? != "rooms" || t.next()? != "1" {
            return None;
        }
        let mut num = || -> Option<f32> { t.next()?.parse::<f32>().ok().filter(|x| x.is_finite()) };
        let int = |x: f32| -> Option<usize> { (x >= 0.0 && x.fract() == 0.0 && x < 1e8).then_some(x as usize) };
        let (px, pz, yaw, pitch) = (num()?, num()?, num()?, num()?);
        let mut c: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        for ca in &mut c {
            let n = int(num()?)?;
            if !(2..=MAX_CUTS).contains(&n) {
                return None;
            }
            for _ in 0..n {
                ca.push(num()?);
            }
            if ca.windows(2).any(|w| w[0] >= w[1]) {
                return None;
            }
        }
        let cells = (c[0].len() - 1) * (c[1].len() - 1) * (c[2].len() - 1);
        let mut occ = Vec::with_capacity(cells);
        for _ in 0..int(num()?)? {
            let val = int(num()?)?;
            let n = int(num()?)?;
            if val > 1 || occ.len() + n > cells {
                return None;
            }
            occ.extend(std::iter::repeat(val as u8).take(n));
        }
        if occ.len() != cells {
            return None;
        }
        let mut lines = HashMap::new();
        for _ in 0..int(num()?)? {
            let axis = int(num()?)?;
            let sign = num()?;
            let pos = num()?;
            if axis > 2 || sign.abs() != 1.0 {
                return None;
            }
            let mut l = Lines::default();
            for _ in 0..int(num()?)? {
                l.us.push(num()?);
            }
            for _ in 0..int(num()?)? {
                l.vs.push(num()?);
            }
            for _ in 0..l.us.len() {
                l.useg.push((0..=l.vs.len()).map(|_| num().map(|b| b != 0.0)).collect::<Option<_>>()?);
            }
            for _ in 0..l.vs.len() {
                l.vseg.push((0..=l.us.len()).map(|_| num().map(|b| b != 0.0)).collect::<Option<_>>()?);
            }
            lines.insert((axis as u8, sign as i8, pos as i64), l);
        }
        let mut wins = vec![];
        for _ in 0..int(num()?)? {
            let axis = int(num()?)?;
            let w = Win { axis, s: num()?, pos: num()?, uc: num()?, vc: num()?, hu: num()?, hv: num()? };
            if axis > 2 || w.s.abs() != 1.0 || wins.len() >= MAX_WINS {
                return None;
            }
            wins.push(w);
        }
        let mut mirrors = HashSet::new();
        if let Some(n) = num() {
            for _ in 0..int(n)? {
                let (i, j) = (num()?, num()?);
                if i.fract() != 0.0 || j.fract() != 0.0 {
                    return None;
                }
                mirrors.insert((i as i32, j as i32));
            }
        }
        self.mirrors = mirrors;
        self.grid = Grid { c, occ };
        self.lines = lines;
        self.wins = wins;
        self.pos = v(px, EYE, pz);
        self.yaw = yaw;
        self.pitch = pitch.clamp(-1.45, 1.45);
        self.vel = (0.0, 0.0);
        self.walk_t = 0.0;
        self.drag_win = None;
        self.changed();
        Some(())
    }

    // ---- undo
    fn snap(&self) -> Snap {
        Snap { grid: self.grid.clone(), lines: self.lines.clone(), wins: self.wins.clone(), mirrors: self.mirrors.clone() }
    }
    fn changed(&mut self) {
        self.dirty = true;
        self.occ_version = self.occ_version.wrapping_add(1);
    }
    /// Record `before` as an undo step for an edit that just succeeded.
    fn commit(&mut self, before: Snap) {
        self.changed();
        if self.group_open && self.group_saved {
            return;
        }
        self.group_saved = self.group_open;
        self.undo.push(before);
        if self.undo.len() > MAX_UNDO {
            self.undo.remove(0);
        }
        self.redo.clear();
    }
    /// Put back a snapshot taken at the start of an edit that didn't happen.
    fn restore_quiet(&mut self, s: Snap) {
        self.grid = s.grid;
        self.lines = s.lines;
        self.wins = s.wins;
        self.mirrors = s.mirrors;
    }
    fn restore(&mut self, s: Snap) {
        self.grid = s.grid;
        self.lines = s.lines;
        self.wins = s.wins;
        self.mirrors = s.mirrors;
        self.drag_win = None;
        self.changed();
    }

    // ---- mirror tiles
    fn tile_of(p: V) -> (i32, i32) {
        ((p.x / TILE).floor() as i32, (p.z / TILE).floor() as i32)
    }
    fn is_mirror(&self, p: V) -> bool {
        self.mirrors.contains(&State::tile_of(p))
    }
    /// The ground tile under the screen point, if the crosshair is on the ground.
    fn ground_tile(&self, sx: f32, sy: f32) -> Option<(i32, i32)> {
        self.pick(sx, sy).filter(|h| h.is_ground()).map(|h| State::tile_of(h.p))
    }
    /// Make a tile a mirror (on) or plain floor; false if it would make the mirror area
    /// too large to send. No-op (true) if it already is.
    fn set_tile(&mut self, t: (i32, i32), on: bool) -> bool {
        if self.mirrors.contains(&t) == on {
            return true;
        }
        if on && !self.mirrors.is_empty() {
            let (mut i0, mut i1, mut j0, mut j1) = (t.0, t.0, t.1, t.1);
            for &(i, j) in &self.mirrors {
                i0 = i0.min(i);
                i1 = i1.max(i);
                j0 = j0.min(j);
                j1 = j1.max(j);
            }
            if i1 - i0 >= MIRROR_SPAN || j1 - j0 >= MIRROR_SPAN {
                return false;
            }
        }
        let before = self.snap();
        if on {
            self.mirrors.insert(t);
        } else {
            self.mirrors.remove(&t);
        }
        self.commit(before);
        true
    }

    // ---- camera
    fn basis(&self) -> (V, V, V) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let f = v(sy * cp, sp, -cy * cp);
        let r = v(cy, 0.0, sy);
        (f, r, r.cross(f))
    }
    fn aspect(&self) -> f32 {
        if self.w > 0 { self.h as f32 / self.w as f32 } else { 9.0 / 16.0 }
    }
    fn ray(&self, sx: f32, sy: f32) -> V {
        let (f, r, u) = self.basis();
        let x = (2.0 * sx - 1.0) * HFOV_TAN;
        let y = (1.0 - 2.0 * sy) * HFOV_TAN * self.aspect();
        (f + r * x + u * y).norm()
    }

    // ---- queries
    fn trace(&self, o: V, d: V, tmax: f32) -> Option<Hit> {
        let (t, axis, k, cell) = self.grid.trace(o, d, tmax)?;
        let s = if d.get(axis) > 0.0 { -1.0 } else { 1.0 };
        Some(Hit { t, axis, k, cell, s, pos: self.grid.c[axis][k], p: o + d * t, n: unit(axis, s) })
    }
    fn pick(&self, sx: f32, sy: f32) -> Option<Hit> {
        self.trace(self.pos, self.ray(sx, sy), INF)
    }
    fn window_at(&self, h: &Hit) -> Option<usize> {
        let (u, w) = h.uv();
        self.wins.iter().position(|x| {
            x.axis == h.axis && x.s == h.s && (x.pos - h.pos).abs() < EPS && (u - x.uc).abs() < x.hu && (w - x.vc).abs() < x.hv
        })
    }
    fn body_ok(&self, x: f32, z: f32) -> bool {
        for &y in &[0.15f32, 0.9, 1.72] {
            if !self.grid.free_at(v(x, y, z)) {
                return false;
            }
            for k in 0..8 {
                let a = k as f32 * PI / 4.0;
                if !self.grid.free_at(v(x + BODY_R * a.cos(), y, z + BODY_R * a.sin())) {
                    return false;
                }
            }
        }
        true
    }

    /// Face cells of the region containing the hit face cell.
    fn region(&self, h: &Hit) -> Vec<(usize, usize)> {
        let (ua, va) = uva(h.axis);
        let g = &self.grid;
        let empty = Lines::default();
        let lines = self.lines.get(&h.key()).unwrap_or(&empty);
        let (nu, nv) = (g.n(ua), g.n(va));
        let start = (h.cell[ua], h.cell[va]);
        let mut seen = HashSet::new();
        let mut stack = vec![start];
        seen.insert(start);
        while let Some((iu, iv)) = stack.pop() {
            let vm = 0.5 * (g.c[va][iv] + g.c[va][iv + 1]);
            let um = 0.5 * (g.c[ua][iu] + g.c[ua][iu + 1]);
            let mut next = Vec::with_capacity(4);
            if iu + 1 < nu && !lines.blocks_u(g.c[ua][iu + 1], vm) {
                next.push((iu + 1, iv));
            }
            if iu > 0 && !lines.blocks_u(g.c[ua][iu], vm) {
                next.push((iu - 1, iv));
            }
            if iv + 1 < nv && !lines.blocks_v(g.c[va][iv + 1], um) {
                next.push((iu, iv + 1));
            }
            if iv > 0 && !lines.blocks_v(g.c[va][iv], um) {
                next.push((iu, iv - 1));
            }
            for c in next {
                if !seen.contains(&c) && g.is_face(h.axis, h.s, h.k, c.0, c.1) {
                    seen.insert(c);
                    stack.push(c);
                }
            }
        }
        seen.into_iter().collect()
    }

    /// Move a region one STEP along its normal: dir +1 away from the viewer
    /// (carve), -1 toward (fill). 0 = not pushable, 1 = ok, 2 = player in the
    /// way, 3 = limit (floor level, window height, size).
    fn push_region(&mut self, h: &Hit, dir: f32) -> i32 {
        if h.is_ground() {
            return 0;
        }
        let cells = self.region(h);
        self.push_cells(h.axis, h.s, h.pos, cells, dir).0
    }

    /// Move the given face cells of plane (m, s, p) by `dir` steps (see push_region).
    /// Returns the result code and the new plane position.
    fn push_cells(&mut self, m: usize, s: f32, p: f32, cells: Vec<(usize, usize)>, dir: f32) -> (i32, f32) {
        if m == 1 && s > 0.0 && p.abs() < EPS {
            return (0, p);
        }
        let (ua, va) = uva(m);
        let mut q = p - s * STEP * dir;
        if m == 1 && q < 0.0 && p > EPS {
            q = 0.0; // a big step down stops at ground level
        }
        if (m == 1 && (q < -EPS || q > MAX_HEIGHT + EPS)) || q.abs() > MAXD {
            return (3, p);
        }
        if self.grid.insert_cut(m, q).is_none() {
            return (3, p);
        }
        let kp = self.grid.find_cut(m, p).unwrap();
        let kq = self.grid.find_cut(m, q).unwrap();
        let (lo, hi) = (kp.min(kq), kp.max(kq));
        let val = if dir > 0.0 { 1 } else { 0 };
        for &(iu, iv) in &cells {
            for l in lo..hi {
                let mut i = [0; 3];
                i[m] = l;
                i[ua] = iu;
                i[va] = iv;
                let ix = self.grid.idx(i);
                self.grid.occ[ix] = val;
            }
        }
        // lines and windows travel with the region
        if let Some(src) = self.lines.get(&key(m, s, p)).cloned() {
            self.lines.entry(key(m, s, q)).or_default().merge_from(&src);
        }
        let set: HashSet<(usize, usize)> = cells.into_iter().collect();
        for w in &mut self.wins {
            if w.axis == m && w.s == s && (w.pos - p).abs() < EPS {
                let iu = self.grid.cell_of(ua, w.uc);
                let iv = self.grid.cell_of(va, w.vc);
                if let (Some(iu), Some(iv)) = (iu, iv) {
                    if set.contains(&(iu, iv)) {
                        w.pos = q;
                        if m == 1 && q < MIN_WIN_Y {
                            return (3, p);
                        }
                    }
                }
            }
        }
        if !self.body_ok(self.pos.x, self.pos.z) {
            return (2, p);
        }
        (1, q)
    }

    /// The cells of a region as world-space rectangles (one per run of cells in a row),
    /// so they survive grid cuts being added or removed.
    fn cells_to_rects(&self, axis: usize, mut cells: Vec<(usize, usize)>) -> Vec<[f32; 4]> {
        let (ua, va) = uva(axis);
        let g = &self.grid;
        cells.sort_by_key(|&(iu, iv)| (iv, iu));
        let mut out = vec![];
        let mut i = 0;
        while i < cells.len() {
            let (u0, iv) = cells[i];
            let mut u1 = u0;
            while i + 1 < cells.len() && cells[i + 1] == (u1 + 1, iv) {
                i += 1;
                u1 += 1;
            }
            out.push([g.c[ua][u0], g.c[ua][u1 + 1], g.c[va][iv], g.c[va][iv + 1]]);
            i += 1;
        }
        out
    }

    /// Face cells of plane (axis, s, cut k) whose centres lie in the rectangles.
    fn rects_to_cells(&self, axis: usize, s: f32, k: usize, rects: &[[f32; 4]]) -> Vec<(usize, usize)> {
        let (ua, va) = uva(axis);
        let g = &self.grid;
        let mut out = vec![];
        for iv in 0..g.n(va) {
            let vm = 0.5 * (g.c[va][iv] + g.c[va][iv + 1]);
            for iu in 0..g.n(ua) {
                let um = 0.5 * (g.c[ua][iu] + g.c[ua][iu + 1]);
                if rects.iter().any(|r| um > r[0] && um < r[1] && vm > r[2] && vm < r[3]) && g.is_face(axis, s, k, iu, iv) {
                    out.push((iu, iv));
                }
            }
        }
        out
    }

    /// Keep only line segments that divide visible face area, i.e. with face
    /// cells of their plane on both sides. Segments on hidden planes or along a
    /// face's edge are dropped, so nothing lingers behind walls.
    fn prune_lines(&mut self) {
        let g = &self.grid;
        for (k, l) in self.lines.iter_mut() {
            let (axis, s, pos) = (k.0 as usize, k.1 as f32, k.2 as f32 / 1000.0);
            let (ua, va) = uva(axis);
            let Some(kp) = g.c[axis].iter().position(|&q| (q - pos).abs() < 6e-4) else {
                *l = Lines::default();
                continue;
            };
            let face = |iu: usize, iv: usize| g.is_face(axis, s, kp, iu, iv);
            for i in 0..l.us.len() {
                let ku = g.find_cut(ua, l.us[i]).filter(|&ku| ku >= 1 && ku < g.n(ua));
                for r in 0..=l.vs.len() {
                    if l.useg[i][r] {
                        let keep = ku.is_some_and(|ku| {
                            (0..g.n(va)).any(|iv| {
                                find(&l.vs, 0.5 * (g.c[va][iv] + g.c[va][iv + 1])) == r && face(ku - 1, iv) && face(ku, iv)
                            })
                        });
                        l.useg[i][r] = keep;
                    }
                }
            }
            for j in 0..l.vs.len() {
                let kv = g.find_cut(va, l.vs[j]).filter(|&kv| kv >= 1 && kv < g.n(va));
                for r in 0..=l.us.len() {
                    if l.vseg[j][r] {
                        let keep = kv.is_some_and(|kv| {
                            (0..g.n(ua)).any(|iu| {
                                find(&l.us, 0.5 * (g.c[ua][iu] + g.c[ua][iu + 1])) == r && face(iu, kv - 1) && face(iu, kv)
                            })
                        });
                        l.vseg[j][r] = keep;
                    }
                }
            }
            l.cleanup();
        }
        self.lines.retain(|_, l| !l.is_empty());
    }

    /// Drop windows whose wall is gone (a push moved the face away from under them),
    /// so they don't linger invisibly, blocking their surface and counting toward the limit.
    fn prune_windows(&mut self) {
        // a window moved lower (e.g. its ceiling pulled down) shrinks to the size allowed there
        for w in &mut self.wins {
            let cap = win_half_max(w.height()).max(0.06);
            w.hu = w.hu.min(cap);
            w.hv = w.hv.min(cap);
        }
        let g = &self.grid;
        self.wins.retain(|w| {
            let (ua, va) = uva(w.axis);
            match (g.find_cut(w.axis, w.pos), g.cell_of(ua, w.uc), g.cell_of(va, w.vc)) {
                (Some(k), Some(iu), Some(iv)) => g.is_face(w.axis, w.s, k, iu, iv),
                _ => false,
            }
        });
    }

    /// How much light reaches a window from outside: more the more it faces the sun,
    /// less where another part of the room blocks the sun from it (OUTSIDE_MIN..1).
    fn outside_light(&self, w: &Win) -> f32 {
        let to_sun = -sun_dir();
        let facing = unit(w.axis, -w.s).dot(to_sun).max(0.0);
        if facing <= 0.0 {
            return OUTSIDE_MIN;
        }
        let out = unit(w.axis, -w.s) * 2e-3;
        let clear = (0..16)
            .filter(|k| self.grid.sun_clear(w.point((k % 4) as f32 / 4.0 + 0.125, (k / 4) as f32 / 4.0 + 0.125) + out, to_sun))
            .count() as f32
            / 16.0;
        OUTSIDE_MIN + (1.0 - OUTSIDE_MIN) * facing * clear
    }

    /// Drop grid cuts and lines that no longer matter.
    fn cleanup(&mut self) {
        self.prune_windows();
        self.prune_lines();
        // cuts that carry a line must stay
        let mut keep: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        for (k, l) in &self.lines {
            let (ua, va) = uva(k.0 as usize);
            keep[ua].extend_from_slice(&l.us);
            keep[va].extend_from_slice(&l.vs);
        }
        for a in 0..3 {
            let mut k = 0;
            while k < self.grid.c[a].len() {
                let x = self.grid.c[a][k];
                if !keep[a].iter().any(|&q| (q - x).abs() < EPS) && self.grid.try_remove_cut(a, k) {
                    continue;
                }
                k += 1;
            }
        }
    }

    // ---- lighting
    fn vpls(&self) -> Vec<Vpl> {
        let mut out = Vec::new();
        let mut r = hash(self.nacc.wrapping_mul(7919) ^ 0xA511_E9B3) | 1;
        let albedo = |h: &Hit| {
            if h.is_ground() {
                v(0.40, 0.38, 0.35)
            } else if h.axis == 1 && h.s < 0.0 {
                v(0.80, 0.79, 0.76)
            } else {
                v(0.78, 0.74, 0.64)
            }
        };
        for w in &self.wins {
            let nin = unit(w.axis, w.s);
            let beam = w.beam();
            let cw = beam.dot(nin);
            let side = (VPL_SUN as f32).sqrt() as usize;
            for k in 0..if cw > 0.0 { VPL_SUN } else { 0 } {
                let fu = ((k % side) as f32 + rnd(&mut r)) / side as f32;
                let fv = ((k / side) as f32 + rnd(&mut r)) / side as f32;
                if !self.grid.sun_clear(w.point(fu, fv) - nin * 2e-3, -beam) {
                    continue; // another part of the room shades this bit of the window
                }
                if let Some(mut h) = self.trace(w.point(fu, fv) + nin * 2e-3, beam, INF) {
                    let mut pw = SUN * w.area() * cw / VPL_SUN as f32 * BOUNCE;
                    if h.is_ground() && self.is_mirror(h.p) {
                        // the beam bounces off a mirror tile and lands somewhere else
                        let up = v(beam.x, -beam.y, beam.z);
                        match self.trace(h.p + v(0.0, 2e-3, 0.0), up, INF) {
                            Some(h2) => {
                                h = h2;
                                pw *= MIRROR_R;
                            }
                            None => continue,
                        }
                    }
                    out.push(Vpl { p: h.p + h.n * 2e-3, n: h.n, power: SUNCOL.mulv(albedo(&h)) * pw });
                }
            }
            for _ in 0..VPL_SKY {
                let q = w.point(rnd(&mut r), rnd(&mut r)) + nin * 2e-3;
                if let Some(h) = self.trace(q, cos_sample(nin, &mut r), INF) {
                    if self.window_at(&h).is_none() {
                        let pw = SKY * self.outside_light(w) * w.area() * PI / VPL_SKY as f32 * BOUNCE;
                        out.push(Vpl { p: h.p + h.n * 2e-3, n: h.n, power: SKYCOL.mulv(albedo(&h)) * pw });
                    }
                }
            }
        }
        out
    }

    /// What the crosshair is on, as the region's plane + rectangles.
    fn make_hover(&self) -> Vec<f32> {
        if self.hover.0 < 0.0 {
            return vec![];
        }
        let Some(h) = self.pick(self.hover.0, self.hover.1) else { return vec![] };
        if h.is_ground() {
            return vec![];
        }
        let (ua, va) = uva(h.axis);
        let g = &self.grid;
        let mut cells = self.region(&h);
        cells.sort_by_key(|&(iu, iv)| (iv, iu));
        let mut out = vec![h.axis as f32, h.s, h.pos];
        let mut i = 0;
        while i < cells.len() && (out.len() - 3) / 4 < MAX_HOVER_RECTS {
            let (u0, iv) = cells[i];
            let mut u1 = u0;
            while i + 1 < cells.len() && cells[i + 1] == (u1 + 1, iv) {
                i += 1;
                u1 += 1;
            }
            out.extend_from_slice(&[g.c[ua][u0], g.c[ua][u1 + 1], g.c[va][iv], g.c[va][iv + 1]]);
            i += 1;
        }
        out
    }

    fn make_preview(&self) -> [f32; 6] {
        if !self.preview {
            return [0.0; 6];
        }
        match self.pick(0.5, 0.5) {
            Some(h) if !h.is_ground() => {
                let (u, w) = h.uv();
                [1.0, h.axis as f32, h.s, h.pos, u, w]
            }
            _ => [0.0; 6],
        }
    }

    /// Per-frame bookkeeping + pack the scene for the shader.
    fn frame(&mut self, w: usize, h: usize) {
        if w != self.w || h != self.h {
            self.w = w;
            self.h = h;
            self.dirty = true;
        }
        let hd = self.make_hover();
        if hd != self.hover_desc {
            self.hover_desc = hd;
            self.dirty = true;
        }
        let th = if self.hover.0 >= 0.0 { self.ground_tile(self.hover.0, self.hover.1) } else { None };
        if th != self.tile_hover {
            self.tile_hover = th;
            self.dirty = true;
        }
        let pd = self.make_preview();
        if pd != self.preview_at {
            self.preview_at = pd;
            self.dirty = true;
        }
        if self.dirty {
            self.nacc = 0;
            self.dirty = false;
        }
        let vpl = self.vpls();
        let alpha = 1.0 / (self.nacc as f32 + 1.0);
        self.nacc = (self.nacc + 1).min(2000);
        self.frame = self.frame.wrapping_add(1);

        // data texture: cuts x/y/z, plane table + line grids, hover rects, VPLs
        let mut d = std::mem::take(&mut self.data);
        d.clear();
        let mut u = [0.0f32; U_LEN];
        for a in 0..3 {
            u[20 + a] = self.grid.c[a].len() as f32;
            u[23 + a] = d.len() as f32;
            d.extend_from_slice(&self.grid.c[a]);
        }
        let planes: Vec<(&Key, &Lines)> = self.lines.iter().collect();
        let table = d.len();
        d.resize(table + planes.len() * 8, 0.0);
        for (pi, (k, l)) in planes.iter().enumerate() {
            let off = d.len();
            d.extend_from_slice(&l.us);
            d.extend_from_slice(&l.vs);
            for seg in &l.useg {
                d.extend(seg.iter().map(|&b| b as u8 as f32));
            }
            for seg in &l.vseg {
                d.extend(seg.iter().map(|&b| b as u8 as f32));
            }
            let t = table + pi * 8;
            d[t..t + 6].copy_from_slice(&[
                k.0 as f32,
                k.1 as f32,
                k.2 as f32 / 1000.0,
                l.us.len() as f32,
                l.vs.len() as f32,
                off as f32,
            ]);
        }
        u[29] = planes.len() as f32;
        u[30] = table as f32;
        let hover_off = d.len();
        if self.hover_desc.len() >= 3 {
            u[32] = 1.0;
            u[33..36].copy_from_slice(&self.hover_desc[0..3]);
            d.extend_from_slice(&self.hover_desc[3..]);
            u[37] = ((self.hover_desc.len() - 3) / 4) as f32;
        }
        u[36] = hover_off as f32;
        if !self.mirrors.is_empty() {
            let i0 = self.mirrors.iter().map(|t| t.0).min().unwrap();
            let i1 = self.mirrors.iter().map(|t| t.0).max().unwrap();
            let j0 = self.mirrors.iter().map(|t| t.1).min().unwrap();
            let j1 = self.mirrors.iter().map(|t| t.1).max().unwrap();
            let (w, h) = ((i1 - i0 + 1) as usize, (j1 - j0 + 1) as usize);
            let off = d.len();
            d.resize(off + w * h, 0.0);
            for &(i, j) in &self.mirrors {
                d[off + (i - i0) as usize + (j - j0) as usize * w] = 1.0;
            }
            u[176..181].copy_from_slice(&[off as f32, i0 as f32, j0 as f32, w as f32, h as f32]);
        }
        if let Some((i, j)) = self.tile_hover {
            u[39] = 1.0;
            u[46] = i as f32;
            u[47] = j as f32;
        }
        let vpl_off = d.len();
        for l in &vpl {
            d.extend_from_slice(&[l.p.x, l.p.y, l.p.z, l.n.x, l.n.y, l.n.z, l.power.x, l.power.y, l.power.z]);
        }
        self.data = d;

        let (f, r, up) = self.basis();
        u[0..12].copy_from_slice(&[self.pos.x, self.pos.y, self.pos.z, f.x, f.y, f.z, r.x, r.y, r.z, up.x, up.y, up.z]);
        u[12] = self.aspect();
        u[13] = HFOV_TAN;
        u[14] = alpha;
        u[15] = (self.frame % (1 << 20)) as f32;
        u[16] = self.wins.len() as f32;
        u[17] = vpl.len() as f32;
        u[18] = vpl_off as f32;
        u[19] = self.wins.iter().map(|w| w.area()).sum();
        let cam = self.grid.cell_at(self.pos);
        for a in 0..3 {
            u[26 + a] = cam.map_or(-1.0, |c| c[a] as f32);
        }
        u[31] = (self.occ_version % (1 << 20)) as f32;
        u[40..46].copy_from_slice(&self.preview_at);
        u[38] = self.extrude_hint as u8 as f32;
        for (i, w) in self.wins.iter().enumerate() {
            let (ua, va) = uva(w.axis);
            let b = w.beam();
            let o = 192 + i * 16;
            u[o..o + 16].copy_from_slice(&[
                w.axis as f32,
                w.s,
                w.pos,
                w.area(),
                ua as f32,
                va as f32,
                self.outside_light(w),
                0.0,
                w.uc - w.hu,
                w.uc + w.hu,
                w.vc - w.hv,
                w.vc + w.hv,
                b.x,
                b.y,
                b.z,
                0.0,
            ]);
        }
        self.ubuf.copy_from_slice(&u);
    }
}

// ---------------------------------------------------------------- exports
#[no_mangle]
pub extern "C" fn init() {
    unsafe { STATE = Some(State::new()) };
}

/// Advance bookkeeping for a w x h frame and pack the scene. Returns the
/// uniform block (U_LEN floats); see data_* and occ_* for the textures.
#[no_mangle]
pub extern "C" fn frame(w: u32, h: u32) -> *const f32 {
    let s = st();
    s.frame(w.max(1) as usize, h.max(1) as usize);
    s.ubuf.as_ptr()
}

#[no_mangle]
pub extern "C" fn data_ptr() -> *const f32 {
    st().data.as_ptr()
}

#[no_mangle]
pub extern "C" fn data_len() -> u32 {
    st().data.len() as u32
}

/// Occupancy bytes (1 = free), x fastest; dims are the cut counts minus one.
#[no_mangle]
pub extern "C" fn occ_ptr() -> *const u8 {
    st().grid.occ.as_ptr()
}

#[no_mangle]
pub extern "C" fn occ_dim(a: u32) -> u32 {
    st().grid.n(a.min(2) as usize) as u32
}

/// keys bitmask: 1 fwd, 2 back, 4 left, 8 right
#[no_mangle]
pub extern "C" fn update(dt: f32, keys: u32) {
    let s = st();
    let dt = if dt.is_finite() { dt.clamp(0.0, 0.1) } else { 0.0 };
    let (fx, fz) = (s.yaw.sin(), -s.yaw.cos());
    let (rx, rz) = (s.yaw.cos(), s.yaw.sin());
    let (mut mx, mut mz) = (0.0f32, 0.0f32);
    if keys & 1 != 0 {
        mx += fx;
        mz += fz;
    }
    if keys & 2 != 0 {
        mx -= fx;
        mz -= fz;
    }
    if keys & 4 != 0 {
        mx -= rx;
        mz -= rz;
    }
    if keys & 8 != 0 {
        mx += rx;
        mz += rz;
    }
    // inertia: steer the velocity toward the wanted one at a limited rate;
    // the longer a key is held, the higher the top speed
    let len = (mx * mx + mz * mz).sqrt();
    s.walk_t = if len > 1e-6 { s.walk_t + dt } else { 0.0 };
    let top = SPEED * (1.0 + WALK_GROWTH * (1.0 + s.walk_t / WALK_TAU).ln());
    let (tx, tz) = if len > 1e-6 { (mx / len * top, mz / len * top) } else { (0.0, 0.0) };
    let (dx, dz) = (tx - s.vel.0, tz - s.vel.1);
    let dist = (dx * dx + dz * dz).sqrt();
    let speed = (s.vel.0 * s.vel.0 + s.vel.1 * s.vel.1).sqrt();
    let rate = if len > 1e-6 { ACCEL + ACCEL_K * top } else { DECEL + DECEL_K * speed } * dt;
    s.vel = if dist <= rate || dist < 1e-9 { (tx, tz) } else { (s.vel.0 + dx / dist * rate, s.vel.1 + dz / dist * rate) };
    if s.vel.0.abs() < 1e-4 && s.vel.1.abs() < 1e-4 {
        s.vel = (0.0, 0.0);
        return;
    }
    // move in small sub-steps so high speeds can't pass through thin walls
    let travel = (s.vel.0 * s.vel.0 + s.vel.1 * s.vel.1).sqrt() * dt;
    let n = (travel / SUBSTEP).ceil().clamp(1.0, 200.0) as usize;
    let h = dt / n as f32;
    for _ in 0..n {
        // if an undo left the player inside a wall, let them walk out
        let stuck = !s.body_ok(s.pos.x, s.pos.z);
        let nx = s.pos.x + s.vel.0 * h;
        if stuck || s.body_ok(nx, s.pos.z) {
            s.pos.x = nx;
        } else {
            s.vel.0 = 0.0; // bumped into a wall: stop along that axis, keep sliding along it
        }
        let nz = s.pos.z + s.vel.1 * h;
        if stuck || s.body_ok(s.pos.x, nz) {
            s.pos.z = nz;
        } else {
            s.vel.1 = 0.0;
        }
    }
    s.dirty = true;
}

#[no_mangle]
pub extern "C" fn look(dyaw: f32, dpitch: f32) {
    let s = st();
    s.yaw += dyaw;
    s.pitch = (s.pitch - dpitch).clamp(-1.45, 1.45);
    s.dirty = true;
}

#[no_mangle]
pub extern "C" fn set_hover(sx: f32, sy: f32) {
    st().hover = (sx, sy);
}

/// Ctrl held: preview the vertical + horizontal line under the crosshair.
#[no_mangle]
pub extern "C" fn set_preview(on: i32) {
    st().preview = on != 0;
}

/// dir > 0 push away, < 0 pull closer, by |dir| steps of 25 cm (held clicks grow it).
/// 0 = nothing pushable, 1 = ok, 2 = blocked by the player, 3 = at a limit.
#[no_mangle]
pub extern "C" fn push(sx: f32, sy: f32, dir: f32) -> i32 {
    let s = st();
    let Some(h) = s.pick(sx, sy) else { return 0 };
    let before = s.snap();
    let r = s.push_region(&h, dir);
    if r == 1 {
        s.cleanup();
        s.commit(before);
    } else {
        s.restore_quiet(before);
    }
    r
}

/// Click on the ground: toggle the floor tile under the crosshair into / out of a
/// mirror. 1 = toggled, 0 = not on the ground, 3 = mirror area too large.
#[no_mangle]
pub extern "C" fn tile_begin(sx: f32, sy: f32) -> i32 {
    let s = st();
    s.tile_last = None;
    let Some(t) = s.ground_tile(sx, sy) else { return 0 };
    s.tile_last = Some(t);
    s.tile_paint = !s.mirrors.contains(&t);
    let paint = s.tile_paint;
    if s.set_tile(t, paint) { 1 } else { 3 }
}

/// Dragging after tile_begin paints: every tile the crosshair passes becomes what the
/// first click made its tile (passing a tile again changes nothing). 1 if one changed.
#[no_mangle]
pub extern "C" fn tile_drag(sx: f32, sy: f32) -> i32 {
    let s = st();
    let t = s.ground_tile(sx, sy);
    if t == s.tile_last {
        return 0;
    }
    s.tile_last = t;
    let Some(t) = t else { return 0 };
    let changed = s.mirrors.contains(&t) != s.tile_paint;
    let paint = s.tile_paint;
    (s.set_tile(t, paint) && changed) as i32
}

#[no_mangle]
pub extern "C" fn tile_end() {
    st().tile_last = None;
}

/// Press on a surface: lock its region for grip_step (held clicks, E-drag).
/// Returns 1 if there is something to move there, 0 otherwise (e.g. the ground).
#[no_mangle]
pub extern "C" fn grip_begin(sx: f32, sy: f32) -> i32 {
    let s = st();
    s.grip = None;
    let Some(h) = s.pick(sx, sy) else { return 0 };
    if h.is_ground() {
        return 0;
    }
    let cells = s.region(&h);
    let rects = s.cells_to_rects(h.axis, cells);
    s.grip = Some(Grip { axis: h.axis, s: h.s, pos: h.pos, rects });
    1
}

/// Move the gripped region by `dir` steps of 25 cm (+ away, - toward).
/// Same codes as push(); 0 if nothing is gripped or the region no longer exists.
#[no_mangle]
pub extern "C" fn grip_step(dir: f32) -> i32 {
    let s = st();
    let Some(g) = s.grip.clone() else { return 0 };
    let before = s.snap();
    // the region's edges may have been cleaned out of the grid (e.g. when it was flush
    // with its neighbours): put them back so its cells can be found again
    let (ua, va) = uva(g.axis);
    for r in &g.rects {
        for (a, x) in [(ua, r[0]), (ua, r[1]), (va, r[2]), (va, r[3])] {
            if s.grid.insert_cut(a, x).is_none() {
                s.restore_quiet(before);
                return 3;
            }
        }
    }
    let Some(k) = s.grid.find_cut(g.axis, g.pos) else { return 0 };
    let cells = s.rects_to_cells(g.axis, g.s, k, &g.rects);
    if cells.is_empty() {
        s.restore_quiet(before);
        return 0;
    }
    let (r, q) = s.push_cells(g.axis, g.s, g.pos, cells, dir);
    if r == 1 {
        s.cleanup();
        s.commit(before);
        if let Some(gr) = s.grip.as_mut() {
            gr.pos = q;
        }
    } else {
        s.restore_quiet(before);
    }
    r
}

#[no_mangle]
pub extern "C" fn grip_end() {
    st().grip = None;
}

/// Distance to the surface under the screen point (m), or -1.
#[no_mangle]
pub extern "C" fn pick_dist(sx: f32, sy: f32) -> f32 {
    st().pick(sx, sy).map_or(-1.0, |h| h.t)
}

/// E held: highlight the region under the crosshair as extrudable.
#[no_mangle]
pub extern "C" fn set_extrude_hint(on: i32) {
    let s = st();
    if s.extrude_hint != (on != 0) {
        s.extrude_hint = on != 0;
        s.dirty = true;
    }
}

/// Ctrl+click: vertical = 1 adds a vertical line (screen-vertical on floors/ceilings).
#[no_mangle]
pub extern "C" fn add_line(sx: f32, sy: f32, vertical: i32) -> i32 {
    let s = st();
    let Some(h) = s.pick(sx, sy) else { return 0 };
    if h.is_ground() {
        return 0;
    }
    let mut cut_u = vertical != 0;
    if h.axis == 1 {
        // on a ceiling, a "vertical" line runs along the view direction
        let (f, _, _) = s.basis();
        let along_x = f.x.abs() > f.z.abs();
        cut_u = if vertical != 0 { !along_x } else { along_x };
    }
    let (ua, va) = uva(h.axis);
    let (u, w) = h.uv();
    let (axis, at) = if cut_u { (ua, u) } else { (va, w) };
    let before = s.snap();
    let Some(k) = s.grid.insert_cut(axis, at) else { return 0 };
    let x = s.grid.c[axis][k];
    let l = s.lines.entry(h.key()).or_default();
    let had = if cut_u { l.blocks_u(x, w) } else { l.blocks_v(x, u) };
    if had {
        s.grid = before.grid;
        s.lines = before.lines;
        return 0;
    }
    if cut_u {
        l.insert_u(x, true);
    } else {
        l.insert_v(x, true);
    }
    s.cleanup();
    s.commit(before);
    1
}

/// Shift+click: a vertical and a horizontal line through the crosshair (one undo step).
#[no_mangle]
pub extern "C" fn add_cross(sx: f32, sy: f32) -> i32 {
    let s = st();
    let Some(h) = s.pick(sx, sy) else { return 0 };
    if h.is_ground() {
        return 0;
    }
    let (ua, va) = uva(h.axis);
    let (u, w) = h.uv();
    let before = s.snap();
    let (Some(ku), Some(kv)) = (s.grid.insert_cut(ua, u), s.grid.insert_cut(va, w)) else {
        s.restore_quiet(before);
        return 0;
    };
    let (xu, xv) = (s.grid.c[ua][ku], s.grid.c[va][kv]);
    let l = s.lines.entry(h.key()).or_default();
    let had = l.blocks_u(xu, w) && l.blocks_v(xv, u);
    l.insert_u(xu, true);
    l.insert_v(xv, true);
    if had {
        s.restore_quiet(before);
        return 0;
    }
    s.cleanup();
    s.commit(before);
    1
}

/// Shift+sweep: remove line segments under the crosshair.
#[no_mangle]
pub extern "C" fn erase(sx: f32, sy: f32) -> i32 {
    let s = st();
    let Some(h) = s.pick(sx, sy) else { return 0 };
    let (u, w) = h.uv();
    let before = s.snap();
    let Some(l) = s.lines.get_mut(&h.key()) else { return 0 };
    if !l.erase(u, w, 0.05 + 0.012 * h.t) {
        return 0;
    }
    s.cleanup();
    s.commit(before);
    1
}

/// Alt+press: start a window (or grab an existing one to resize).
/// 0 = not on a wall/ceiling, 1 = created, 2 = grabbed existing, 3 = too low,
/// 4 = enough windows (this surface already has one), 5 = the room has MAX_WINS.
#[no_mangle]
pub extern "C" fn win_begin(sx: f32, sy: f32) -> i32 {
    let s = st();
    let Some(h) = s.pick(sx, sy) else { return 0 };
    if let Some(i) = s.window_at(&h) {
        s.drag_win = Some(i);
        return 2;
    }
    if h.axis == 1 && h.s > 0.0 {
        return 0; // floors and sills face up: no windows there
    }
    if s.wins.len() >= MAX_WINS {
        return 5;
    }
    // one window per surface (the region under the crosshair)
    let (ua, va) = uva(h.axis);
    let region: HashSet<(usize, usize)> = s.region(&h).into_iter().collect();
    let taken = s.wins.iter().any(|x| {
        x.axis == h.axis
            && x.s == h.s
            && (x.pos - h.pos).abs() < EPS
            && matches!((s.grid.cell_of(ua, x.uc), s.grid.cell_of(va, x.vc)), (Some(iu), Some(iv)) if region.contains(&(iu, iv)))
    });
    if taken {
        return 4;
    }
    let (u, w) = h.uv();
    let low = if h.axis == 1 { h.pos < MIN_WIN_Y } else { w < MIN_WIN_Y + 0.06 };
    if low {
        return 3;
    }
    let before = s.snap();
    s.wins.push(Win { axis: h.axis, s: h.s, pos: h.pos, uc: u, vc: w, hu: 0.06, hv: 0.06 });
    s.drag_win = Some(s.wins.len() - 1);
    s.commit(before);
    1
}

/// Alt+aim: resize the grabbed window around its centre.
#[no_mangle]
pub extern "C" fn win_drag(sx: f32, sy: f32) {
    let s = st();
    let Some(i) = s.drag_win else { return };
    let d = s.ray(sx, sy);
    let w = &s.wins[i];
    let da = d.get(w.axis);
    if da.abs() < 1e-6 {
        return;
    }
    let t = (w.pos - s.pos.get(w.axis)) / da;
    if t <= 0.0 {
        return;
    }
    let p = s.pos + d * t;
    let (ua, va) = uva(w.axis);
    let cap = win_half_max(w.height()).max(0.06);
    let hu = (p.get(ua) - w.uc).abs().clamp(0.06, cap);
    let mut hv = (p.get(va) - w.vc).abs().clamp(0.06, cap);
    if w.axis != 1 {
        hv = hv.min(w.vc - MIN_WIN_Y).max(0.02);
    }
    if (w.hu, w.hv) == (hu, hv) {
        return;
    }
    let before = s.snap();
    let w = &mut s.wins[i];
    w.hu = hu;
    w.hv = hv;
    s.commit(before);
}

#[no_mangle]
pub extern "C" fn win_end() {
    st().drag_win = None;
}

/// Alt+right click on a window removes it.
#[no_mangle]
pub extern "C" fn win_delete(sx: f32, sy: f32) -> i32 {
    let s = st();
    let Some(h) = s.pick(sx, sy) else { return 0 };
    if let Some(i) = s.window_at(&h) {
        let before = s.snap();
        s.wins.remove(i);
        s.commit(before);
        return 1;
    }
    0
}

/// Mouse press started / ended: edits in between form one undo step.
#[no_mangle]
pub extern "C" fn group_begin() {
    let s = st();
    s.group_open = true;
    s.group_saved = false;
}

#[no_mangle]
pub extern "C" fn group_end() {
    let s = st();
    s.group_open = false;
    s.group_saved = false;
}

/// Ctrl+Z. Returns 1 if something was undone.
#[no_mangle]
pub extern "C" fn undo() -> i32 {
    let s = st();
    let Some(prev) = s.undo.pop() else { return 0 };
    let cur = s.snap();
    s.redo.push(cur);
    s.restore(prev);
    1
}

/// Ctrl+Shift+Z / Ctrl+Y. Returns 1 if something was redone.
#[no_mangle]
pub extern "C" fn redo() -> i32 {
    let s = st();
    let Some(next) = s.redo.pop() else { return 0 };
    let cur = s.snap();
    s.undo.push(cur);
    s.restore(next);
    1
}

/// Something outside the engine changed the image (e.g. mist density): restart accumulation.
#[no_mangle]
pub extern "C" fn invalidate() {
    st().dirty = true;
}

/// Serialize the room; returns a pointer to UTF-8 text of save_len() bytes.
#[no_mangle]
pub extern "C" fn save() -> *const u8 {
    let s = st();
    s.io = s.serialize().into_bytes();
    s.io.as_ptr()
}

#[no_mangle]
pub extern "C" fn save_len() -> u32 {
    st().io.len() as u32
}

/// Buffer for the page to write a saved room into before load(len).
#[no_mangle]
pub extern "C" fn alloc(len: u32) -> *mut u8 {
    let s = st();
    s.io = vec![0; len as usize];
    s.io.as_mut_ptr()
}

/// Load the room text written into the alloc() buffer. 1 = ok (undoable), 0 = invalid.
#[no_mangle]
pub extern "C" fn load(len: u32) -> i32 {
    let s = st();
    let text = String::from_utf8_lossy(&s.io[..(len as usize).min(s.io.len())]).into_owned();
    let before = s.snap();
    if s.deserialize(&text).is_none() {
        return 0;
    }
    s.commit(before);
    1
}

/// Forget undo/redo (after restoring the autosaved room at startup).
#[no_mangle]
pub extern "C" fn clear_history() {
    let s = st();
    s.undo.clear();
    s.redo.clear();
}

/// N: back to the starting room (undoable).
#[no_mangle]
pub extern "C" fn reset() {
    let s = st();
    let before = s.snap();
    let fresh = State::new();
    s.grid = fresh.grid;
    s.lines = fresh.lines;
    s.wins = fresh.wins;
    s.mirrors = fresh.mirrors;
    s.pos = fresh.pos;
    s.yaw = fresh.yaw;
    s.pitch = fresh.pitch;
    s.vel = (0.0, 0.0);
    s.walk_t = 0.0;
    s.drag_win = None;
    s.commit(before);
}

/// Grid extent for tests/console: the first (end 0) or last (end 1) cut on axis a.
#[no_mangle]
pub extern "C" fn extent(a: u32, end: u32) -> f32 {
    let c = &st().grid.c[a.min(2) as usize];
    if end == 0 { c[0] } else { c[c.len() - 1] }
}

/// Camera state for tests: 0 x, 1 z, 2 yaw, 3 pitch.
#[no_mangle]
pub extern "C" fn cam(i: u32) -> f32 {
    let s = st();
    [s.pos.x, s.pos.z, s.yaw, s.pitch][i.min(3) as usize]
}

/// Place the camera (debug/testing): eye at (x, EYE, z).
#[no_mangle]
pub extern "C" fn set_cam(x: f32, z: f32, yaw: f32, pitch: f32) {
    let s = st();
    s.pos = v(x, EYE, z);
    s.vel = (0.0, 0.0);
    s.walk_t = 0.0;
    s.yaw = yaw;
    s.pitch = pitch;
    s.dirty = true;
}

// ---------------------------------------------------------------- tests
#[cfg(test)]
mod tests {
    use super::*;

    /// Fresh state, standing at (x, z).
    fn setup(x: f32, z: f32) {
        init();
        let s = st();
        s.w = 16;
        s.h = 9;
        s.pos = v(x, EYE, z);
    }
    /// Turn to look straight at `target` (the crosshair is then on it).
    fn aim(target: V) {
        let s = st();
        let d = (target - s.pos).norm();
        s.pitch = d.y.asin();
        s.yaw = d.x.atan2(-d.z);
    }
    fn hit_at(target: V) -> Hit {
        aim(target);
        st().pick(0.5, 0.5).unwrap()
    }
    fn free(p: V) -> bool {
        st().grid.free_at(p)
    }

    /// From the room centre: two vertical lines and one horizontal line on the
    /// east wall, then push the lower middle square back 2 m into a passage.
    fn passage() {
        setup(0.0, 0.0);
        aim(v(1.5, EYE, -0.4));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        aim(v(1.5, EYE, 0.4));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        aim(v(1.5, 2.0, 0.0));
        assert_eq!(add_line(0.5, 0.5, 0), 1);
        for _ in 0..8 {
            aim(v(3.5, 1.0, 0.0));
            assert_eq!(push(0.5, 0.5, 1.0), 1);
        }
        assert!(free(v(3.4, 1.0, 0.0)), "passage is 2 m deep");
        assert!(!free(v(3.4, 1.0, 0.6)), "but only between the lines");
        assert!(!free(v(3.4, 2.2, 0.0)), "and below the horizontal line");
    }

    #[test]
    fn passage_side_wall_pushes_along_its_own_normal() {
        passage();
        let h = hit_at(v(2.5, 1.0, 0.4));
        assert_eq!((h.axis, h.s), (2, -1.0), "the passage's +z side wall faces -z");
        aim(v(2.5, 1.0, 0.4));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        assert!(free(v(2.5, 1.0, 0.55)), "passage got wider");
        assert!(!free(v(2.5, 2.5, 0.0)), "wall above the passage untouched");
        assert!(free(v(3.4, 1.0, 0.0)), "back of the passage untouched");
        assert!(!free(v(3.6, 1.0, 0.0)));
    }

    #[test]
    fn passage_ceiling_and_back_wall_push() {
        passage();
        aim(v(2.5, 2.0, 0.0));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        assert!(free(v(2.5, 2.1, 0.0)), "passage ceiling raised");
        aim(v(3.5, 1.0, 0.0));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        assert!(free(v(3.6, 1.0, 0.0)), "back wall pushed further");
    }

    #[test]
    fn lines_on_a_passage_wall_split_it() {
        passage();
        // vertical line halfway along the passage's side wall, then push only the far half
        aim(v(2.5, 1.0, 0.4));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        aim(v(3.2, 1.0, 0.4));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        assert!(free(v(3.2, 1.0, 0.55)), "far half pushed");
        assert!(!free(v(1.8, 1.0, 0.55)), "near half not");
    }

    #[test]
    fn region_merges_after_erasing_a_segment() {
        setup(0.0, 0.0);
        // two vertical lines: three regions on the east wall
        aim(v(1.5, 1.6, 0.0));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        aim(v(1.5, 1.6, 0.6));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        // erase the one at z = 0: the left and middle regions join
        aim(v(1.5, 1.6, 0.0));
        assert_eq!(erase(0.5, 0.5), 1);
        aim(v(1.5, 1.6, -0.5));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        assert!(free(v(1.6, 1.6, -0.5)) && free(v(1.6, 1.6, 0.3)), "joined region moved");
        assert!(!free(v(1.6, 1.6, 1.0)), "region past the remaining line did not");
    }

    #[test]
    fn edge_lines_disappear() {
        setup(0.0, 0.0);
        aim(v(1.5, 1.6, 0.0));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        aim(v(1.5, 1.6, 0.5));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        // the line now runs along the recess edge: it divides nothing, so it is gone
        assert!(st().lines.is_empty());
        assert_eq!(st().grid.c[2].len(), 3, "the grid keeps the z cut only because geometry needs it");
        assert!(st().grid.c[2][1].abs() < 1e-3);
    }

    #[test]
    fn ground_cannot_be_pushed_and_windows_stay_high() {
        setup(0.0, 0.0);
        aim(v(0.5, 0.0, 0.5));
        assert_eq!(push(0.5, 0.5, 1.0), 0);
        aim(v(1.5, 1.0, 0.0));
        assert_eq!(win_begin(0.5, 0.5), 3, "too low");
        aim(v(1.5, 2.5, 0.0));
        assert_eq!(win_begin(0.5, 0.5), 1);
        win_end();
        // pull the ceiling down to 2 m, then the player is in the way
        for _ in 0..4 {
            aim(v(0.5, st().grid.c[1].last().copied().unwrap(), 0.5));
            assert_eq!(push(0.5, 0.5, -1.0), 1);
        }
        assert!(free(v(0.0, 1.95, 0.0)) && !free(v(0.0, 2.05, 0.0)));
        aim(v(0.5, 2.0, 0.5));
        assert_eq!(push(0.5, 0.5, -1.0), 1, "1.75 m still clears a 1.75 m player");
        aim(v(0.5, 1.75, 0.5));
        assert_eq!(push(0.5, 0.5, -1.0), 2, "the player is in the way");
    }

    #[test]
    fn undo_redo_and_grouping() {
        setup(0.0, 0.0);
        let g0 = st().grid.c[0].clone();
        aim(v(1.5, 1.6, 0.0));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        aim(v(1.75, 1.6, 0.0));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        group_begin();
        for _ in 0..3 {
            aim(v(*st().grid.c[0].last().unwrap(), 1.6, 0.0));
            assert_eq!(push(0.5, 0.5, 1.0), 1);
        }
        group_end();
        assert_eq!(*st().grid.c[0].last().unwrap(), 1.5 + 5.0 * STEP);
        assert_eq!(undo(), 1);
        assert_eq!(*st().grid.c[0].last().unwrap(), 1.5 + 2.0 * STEP);
        assert_eq!(undo(), 1);
        assert_eq!(undo(), 1);
        assert_eq!(st().grid.c[0], g0);
        assert_eq!(undo(), 0);
        assert_eq!(redo(), 1);
        assert_eq!(redo(), 1);
        assert_eq!(redo(), 1);
        assert_eq!(*st().grid.c[0].last().unwrap(), 1.5 + 5.0 * STEP);
        assert_eq!(redo(), 0);
        assert_eq!(undo(), 1);
        aim(v(1.5 + 2.0 * STEP, 1.6, 0.3));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        assert_eq!(redo(), 0, "a new edit drops the redo history");
    }

    #[test]
    fn one_sun_for_all_windows() {
        setup(0.0, 0.0);
        // a second window on the opposite (south) wall faces away from the sun
        aim(v(0.0, 2.5, 1.5));
        assert_eq!(win_begin(0.5, 0.5), 1);
        win_end();
        let s = st();
        assert_eq!(s.wins[0].beam(), s.wins[1].beam());
        let lit_from = |w: &Win| unit(w.axis, w.s).dot(w.beam()) > 0.0;
        assert!(lit_from(&s.wins[0]), "north window lets the sun in");
        assert!(!lit_from(&s.wins[1]), "south window gets sky light only");
        // so every sun VPL comes from the north window: 16 sun + 8 sky, plus up to 8 sky for the south one
        assert!(s.vpls().len() <= VPL_SUN + 2 * VPL_SKY);
    }

    /// Passage, then erase the wall's lines and push the wall: no old divisions come back.
    #[test]
    fn erased_lines_do_not_come_back_from_behind() {
        passage();
        // erase every remaining line segment on the original wall plane
        let k = key(0, -1.0, 1.5);
        st().lines.remove(&k);
        st().cleanup();
        // nothing may linger on the planes the passage was pushed through
        for (key, _) in &st().lines {
            assert!(!(key.0 == 0 && key.2 > 1500 && key.2 < 3500), "lines left on hidden plane {key:?}");
        }
        // push the whole wall (it is one region now) a few steps
        for _ in 0..3 {
            aim(v(*st().grid.c[0].last().unwrap() - 0.01, 2.6, -1.0));
            let h = st().pick(0.5, 0.5).unwrap();
            aim(h.p);
            assert_eq!(push(0.5, 0.5, 1.0), 1);
        }
        let h = hit_at(v(2.25, 2.6, -1.0));
        assert_eq!((h.axis, h.pos), (0, 2.25));
        assert!(st().lines.get(&h.key()).is_none(), "the pushed wall has no lines");
        // ... and the whole wall moved, including where the old square was drawn
        assert!(free(v(2.2, 2.6, 0.0)) && free(v(2.2, 2.6, 1.0)));
    }

    #[test]
    fn walking_has_inertia() {
        setup(0.0, 1.0);
        st().yaw = 0.0; // facing -z
        update(0.05, 1);
        let v1 = -st().vel.1;
        assert!(v1 > 0.0 && v1 < SPEED, "speeds up gradually");
        for _ in 0..7 {
            update(0.05, 1);
        }
        let v8 = -st().vel.1;
        assert!(v8 > SPEED, "walking speed within half a second: {v8}");
        let z = st().pos.z;
        update(0.05, 0);
        assert!(st().pos.z < z && -st().vel.1 < v8, "keeps gliding, slower, after release");
        for _ in 0..40 {
            update(0.05, 0);
        }
        assert_eq!(st().vel, (0.0, 0.0), "comes to rest");
        // walking into a wall stops at the wall
        for _ in 0..100 {
            update(0.05, 1);
        }
        assert!(st().pos.z > -1.5 + BODY_R - 0.01 && st().vel.1 == 0.0);
    }

    #[test]
    fn bad_frame_times_are_harmless() {
        setup(0.0, 0.0);
        for dt in [-0.016, 0.0, f32::NAN, f32::INFINITY, 5.0] {
            update(dt, 0);
            update(dt, 1);
        }
        assert!(st().pos.x.is_finite() && st().pos.z.is_finite());
        assert!(!st().grid.free_at(v(f32::NAN, 1.0, 0.0)));
    }

    #[test]
    fn holding_a_key_keeps_speeding_up() {
        setup(0.0, 0.0);
        // a long corridor ahead: push the north wall 400 m away in big steps
        for _ in 0..16 {
            aim(v(0.0, 1.6, *st().grid.c[2].first().unwrap()));
            assert_eq!(push(0.5, 0.5, 100.0), 1);
        }
        assert!(st().grid.c[2][0] <= -400.0);
        st().yaw = 0.0;
        let mut speeds = vec![];
        for second in 0..12 {
            for _ in 0..20 {
                update(0.05, 1);
            }
            if [0, 3, 11].contains(&second) {
                speeds.push(-st().vel.1);
            }
        }
        assert!(speeds[0] > 2.0 * SPEED && speeds[1] > speeds[0] && speeds[2] > speeds[1], "{speeds:?}");
        assert!(speeds[2] > 20.0, "fast after a while: {speeds:?}");
        // releasing brakes quickly even from high speed
        for _ in 0..40 {
            update(0.05, 0);
        }
        assert_eq!(st().vel, (0.0, 0.0));
        assert!(st().grid.free_at(st().pos), "never left the free space");
    }

    #[test]
    fn fast_walking_does_not_tunnel_through_thin_walls() {
        setup(0.0, 0.0);
        st().yaw = 0.0;
        st().walk_t = 60.0; // already at full tilt
        st().vel = (0.0, -30.0);
        for _ in 0..20 {
            update(0.1, 1);
        }
        assert!(st().pos.z > -1.5, "stopped at the north wall");
    }

    #[test]
    fn big_steps_down_stop_at_ground_level() {
        setup(0.0, 0.0);
        // make a raised sill: pull a lower strip of the east wall in, then push its top down hard
        aim(v(1.5, 0.5, 0.0));
        assert_eq!(add_line(0.5, 0.5, 0), 1);
        aim(v(1.5, 0.25, 0.0));
        assert_eq!(push(0.5, 0.5, -1.0), 1); // a 25 cm deep, ~0.5 m high ledge
        aim(v(1.4, *st().grid.c[1].get(1).unwrap(), 0.0));
        let r = push(0.5, 0.5, 20.0);
        assert_eq!(r, 1);
        assert!(st().grid.free_at(v(1.4, 0.1, 0.0)), "ledge carved down to the ground, not below");
    }

    #[test]
    fn save_load_round_trip() {
        passage();
        aim(v(1.5, 2.5, -1.0));
        assert_eq!(win_begin(0.5, 0.5), 1);
        win_end();
        aim(v(-1.5, 1.6, 0.3));
        assert_eq!(add_cross(0.5, 0.5), 1);
        let text = st().serialize();
        let (grid, wins, nlines) = (st().grid.clone(), st().wins.len(), st().lines.len());
        reset();
        assert_ne!(st().grid.occ, grid.occ);
        let ptr = alloc(text.len() as u32);
        unsafe { std::ptr::copy_nonoverlapping(text.as_ptr(), ptr, text.len()) };
        assert_eq!(load(text.len() as u32), 1);
        assert_eq!(st().grid.c, grid.c);
        assert_eq!(st().grid.occ, grid.occ);
        assert_eq!((st().wins.len(), st().lines.len()), (wins, nlines));
        assert_eq!(st().serialize(), text, "identical after a round trip");
        // garbage is rejected and changes nothing
        let bad = b"rooms 1 0 0 0 0 2 0 1 2 0 1 2 0 1 5 1 1";
        let ptr = alloc(bad.len() as u32);
        unsafe { std::ptr::copy_nonoverlapping(bad.as_ptr(), ptr, bad.len()) };
        assert_eq!(load(bad.len() as u32), 0);
        assert_eq!(st().grid.occ, grid.occ);
        // and loading is undoable
        assert_eq!(undo(), 1);
    }

    #[test]
    fn shift_click_cross_splits_into_four() {
        setup(0.0, 0.0);
        aim(v(1.5, 1.6, 0.0));
        assert_eq!(add_cross(0.5, 0.5), 1);
        aim(v(1.5, 0.8, 0.7));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        assert!(free(v(1.6, 0.8, 0.7)));
        assert!(!free(v(1.6, 2.4, 0.7)) && !free(v(1.6, 0.8, -0.7)) && !free(v(1.6, 2.4, -0.7)));
        assert_eq!(undo(), 1);
        assert_eq!(undo(), 1, "cross was one step");
        assert!(st().lines.is_empty());
    }

    #[test]
    fn one_window_per_surface() {
        setup(0.0, 0.0);
        aim(v(1.5, 2.5, -0.8));
        assert_eq!(win_begin(0.5, 0.5), 1);
        win_end();
        aim(v(1.5, 2.5, 0.8));
        assert_eq!(win_begin(0.5, 0.5), 4, "the east wall already has a window");
        // split the wall with a line: the other half is its own surface
        aim(v(1.5, 1.6, 0.0));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        aim(v(1.5, 2.5, 0.8));
        assert_eq!(win_begin(0.5, 0.5), 1);
        win_end();
        // grabbing an existing window to resize still works
        aim(v(1.5, 2.5, 0.8));
        assert_eq!(win_begin(0.5, 0.5), 2);
        win_end();
    }

    #[test]
    fn gripped_region_keeps_moving_past_a_flush_position() {
        setup(0.0, 0.0);
        // a recess: line at z = 0, push the +z half back 2 steps
        aim(v(1.5, 1.6, 0.0));
        assert_eq!(add_line(0.5, 0.5, 1), 1);
        aim(v(1.5, 1.6, 0.6));
        assert_eq!(push(0.5, 0.5, 2.0), 1);
        // grip it and pull it 4 steps: through the flush position and out into the room
        aim(v(2.0, 1.6, 0.6));
        assert_eq!(grip_begin(0.5, 0.5), 1);
        for i in 0..4 {
            let r = grip_step(-1.0);
            assert_eq!(r, 1, "step {i}, grip {:?}", st().grip.as_ref().map(|g| (g.pos, g.rects.clone())));
        }
        grip_end();
        assert!(!free(v(1.4, 1.6, 0.6)), "the gripped half now sticks out 50 cm into the room");
        assert!(free(v(1.4, 1.6, -0.6)), "the other half of the wall did not move");
    }

    #[test]
    fn a_wing_of_the_room_shades_the_window() {
        setup(0.0, 0.0);
        let sun_vpls = |s: &State| {
            let w = &s.wins[0];
            let nin = unit(w.axis, w.s);
            (0..16).filter(|k| s.grid.sun_clear(w.point((*k % 4) as f32 / 4.0 + 0.125, (*k / 4) as f32 / 4.0 + 0.125) - nin * 2e-3, -w.beam())).count()
        };
        assert_eq!(sun_vpls(st()), 16, "nothing outside: full sun");
        // the sun comes from above and from -z (behind the north window): build a tall wing
        // out there by pushing the west wall's upper part far out and then north
        // (simplest: carve free cells directly in the grid where the sun ray passes)
        let g = &mut st().grid;
        for (a, x) in [(0, -4.0), (0, 4.0), (1, 6.0), (1, 12.0), (2, -6.0), (2, -3.0)] {
            g.insert_cut(a, x).unwrap();
        }
        let (i0, i1) = (g.find_cut(0, -4.0).unwrap(), g.find_cut(0, 4.0).unwrap());
        let (j0, j1) = (g.find_cut(1, 6.0).unwrap(), g.find_cut(1, 12.0).unwrap());
        let (k0, k1) = (g.find_cut(2, -6.0).unwrap(), g.find_cut(2, -3.0).unwrap());
        for i in i0..i1 {
            for j in j0..j1 {
                for k in k0..k1 {
                    let ix = g.idx([i, j, k]);
                    g.occ[ix] = 1;
                }
            }
        }
        assert_eq!(sun_vpls(st()), 0, "the wing above and behind the window blocks the sun");
    }

    #[test]
    fn floor_tiles_toggle_into_mirrors() {
        setup(0.3, 0.3);
        aim(v(0.9, 0.0, 0.3)); // tile (1, 0)
        group_begin();
        assert_eq!(tile_begin(0.5, 0.5), 1);
        assert!(st().mirrors.contains(&(1, 0)));
        // the drag paints mirrors: onto the next tile, and back onto the first changes nothing
        aim(v(0.3, 0.0, 0.3)); // tile (0, 0)
        assert_eq!(tile_drag(0.5, 0.5), 1);
        aim(v(0.31, 0.0, 0.32)); // same tile: nothing
        assert_eq!(tile_drag(0.5, 0.5), 0);
        aim(v(0.9, 0.0, 0.3)); // back onto (1, 0): already a mirror, stays one
        assert_eq!(tile_drag(0.5, 0.5), 0);
        tile_end();
        group_end();
        assert!(st().mirrors.contains(&(0, 0)) && st().mirrors.contains(&(1, 0)));
        // starting on a mirror erases instead: the drag clears tiles it passes
        aim(v(0.3, 0.0, 0.3));
        group_begin();
        assert_eq!(tile_begin(0.5, 0.5), 1);
        aim(v(0.9, 0.0, 0.3));
        assert_eq!(tile_drag(0.5, 0.5), 1);
        aim(v(1.5 - 0.01, 0.0, 0.3)); // tile (2, 0): plain floor, stays plain
        assert_eq!(tile_drag(0.5, 0.5), 0);
        tile_end();
        group_end();
        assert!(st().mirrors.is_empty());
        assert_eq!(undo(), 1);
        assert!(st().mirrors.contains(&(0, 0)) && st().mirrors.contains(&(1, 0)));
        // walls are not tiles
        aim(v(1.5, 1.6, 0.0));
        assert_eq!(tile_begin(0.5, 0.5), 0);
        // saved with the room, and the whole sweep is one undo step
        let text = st().serialize();
        assert!(text.trim_end().ends_with("2 0 0 1 0"));
        assert_eq!(undo(), 1);
        assert!(st().mirrors.is_empty());
        // older saves without the mirror section still load
        let old = text.trim_end().rsplit_once('\n').unwrap().0.to_string();
        let ptr = alloc(old.len() as u32);
        unsafe { std::ptr::copy_nonoverlapping(old.as_ptr(), ptr, old.len()) };
        assert_eq!(load(old.len() as u32), 1);
    }

    #[test]
    fn sun_bounces_off_a_mirror_tile() {
        setup(-1.2, 1.2);
        let count_floor = |s: &State| s.vpls().iter().take(VPL_SUN).filter(|l| l.n.y > 0.5).count();
        let before = count_floor(st());
        assert!(before > 8, "the sun patch starts on the floor");
        // mirror every tile in the patch area
        for i in -3..3 {
            for j in -3..3 {
                st().mirrors.insert((i, j));
            }
        }
        assert_eq!(count_floor(st()), 0, "the beam now bounces up off the mirrors");
    }

    #[test]
    fn windows_whose_wall_moves_away_are_removed() {
        setup(0.0, 0.0);
        // a window on the east wall, then split the wall right through the window's
        // left edge and push the half without the window's centre: the window stays
        aim(v(1.5, 2.5, 0.0));
        assert_eq!(win_begin(0.5, 0.5), 1);
        win_end();
        assert_eq!(st().wins.len(), 2);
        // push the whole east wall (one region): the window travels with it
        aim(v(1.5, 1.0, -1.0));
        assert_eq!(push(0.5, 0.5, 1.0), 1);
        assert_eq!(st().wins.len(), 2);
        assert_eq!(st().wins[1].pos, 1.75);
        // a window left on a vanished face is removed
        st().wins.push(Win { axis: 0, s: -1.0, pos: 1.5, uc: 0.5, vc: 2.5, hu: 0.1, hv: 0.1 });
        st().cleanup();
        assert_eq!(st().wins.len(), 2, "orphan window dropped");
        // and its surface takes a window again: many surfaces, one window each, beyond the old limit of 8
        let mut placed = 0;
        for k in 0..12 {
            let z = -1.4 + 0.25 * k as f32;
            aim(v(1.75, 1.6, z));
            if add_line(0.5, 0.5, 1) == 1 {
                placed += 1;
            }
        }
        assert!(placed >= 10);
        let mut wins = 0;
        for k in 0..12 {
            let z = -1.4 + 0.25 * k as f32 + 0.12;
            aim(v(1.75, 2.6, z));
            if win_begin(0.5, 0.5) == 1 {
                wins += 1;
            }
            win_end();
        }
        assert!(st().wins.len() > 8, "{} windows", st().wins.len());
        let _ = wins;
    }

    #[test]
    fn windows_facing_away_from_the_sun_get_less_light() {
        setup(0.0, 0.0);
        aim(v(0.0, 2.5, 1.5)); // south wall: faces away from the sun
        assert_eq!(win_begin(0.5, 0.5), 1);
        win_end();
        let s = st();
        let (north, south) = (s.outside_light(&s.wins[0]), s.outside_light(&s.wins[1]));
        assert!(north > 0.7 && (south - OUTSIDE_MIN).abs() < 1e-6, "north {north}, south {south}");
    }

    #[test]
    fn window_size_limit_grows_with_height() {
        setup(0.0, 0.0);
        // on a wall at 2.5 m: at most 1.25 m wide
        aim(v(1.5, 2.5, 0.0));
        assert_eq!(win_begin(0.5, 0.5), 1);
        aim(v(1.5, 2.5, 1.45)); // drag far to the side
        win_drag(0.5, 0.5);
        win_end();
        let w = st().wins.last().unwrap().clone();
        assert!((w.hu - 0.625).abs() < 1e-4, "half width {}", w.hu);
        // raise the ceiling to 20 m: a ceiling window there may be 10 m wide
        aim(v(-0.5, 3.0, -0.5));
        assert_eq!(push(0.5, 0.5, 68.0), 1);
        assert_eq!(*st().grid.c[1].last().unwrap(), 20.0);
        aim(v(-0.5, 20.0, -0.5));
        assert_eq!(win_begin(0.5, 0.5), 1);
        aim(v(-1.45, 20.0, -1.45));
        win_drag(0.5, 0.5);
        win_end();
        let w = st().wins.last().unwrap().clone();
        assert!(w.hu > 0.9 && w.hu <= 5.0, "half width {}", w.hu);
        // pulling that ceiling down to 4 m shrinks its window to fit (at most 2 m)
        aim(v(-0.5, 20.0, -0.5));
        assert_eq!(push(0.5, 0.5, -64.0), 1);
        let w = st().wins.last().unwrap().clone();
        assert!(w.hu <= 1.0 + 1e-4 && w.hv <= 1.0 + 1e-4);
    }

    #[test]
    fn rooms_stop_at_the_height_limit() {
        setup(0.0, 0.0);
        assert!((MAX_HEIGHT - 614.4).abs() < 1e-3);
        aim(v(0.5, 3.0, 0.5));
        assert_eq!(push(0.5, 0.5, 2440.0), 1); // up to 613 m
        aim(v(0.5, *st().grid.c[1].last().unwrap(), 0.5));
        assert_eq!(push(0.5, 0.5, 8.0), 3, "can't go past {MAX_HEIGHT} m (615 m)");
    }

    #[test]
    fn grid_stays_small() {
        setup(0.0, 0.0);
        for _ in 0..4 {
            aim(v(1.5, 1.6, 0.0));
            assert_eq!(push(0.5, 0.5, 1.0), 1);
            aim(v(1.75, 1.6, 0.0));
            assert_eq!(push(0.5, 0.5, -1.0), 1);
        }
        assert_eq!(st().grid.c[0], vec![-1.5, 1.5], "push + pull leave no stray cuts");
    }
}

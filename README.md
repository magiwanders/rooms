# rooms

**Reshape a liminal room from the inside.**

Play it at **https://rooms.magiwanders.com** (a desktop browser with WebGL2, a mouse or touchpad and a keyboard).

![A passage pushed into the wall, lit by the high window](docs/screenshot.png)

You start in a small room, 3 × 3 × 3 m, as a 1.75 m tall person. A single small window sits high on one wall and is the only light. There are only two rules: no window may be within the first two metres above the floor, and there is no door to the outside. Everything else is up to you: push walls and ceilings away, draw lines on them to split them into regions, push those regions back into passages, alcoves and halls, and open more windows high up. Mist and shafts of light show how big the space has become.

## Controls

Press **H** in the game for this list.

| Input | Action |
|---|---|
| Click / Esc | capture / release the mouse; the mouse turns the view, actions hit the crosshair |
| WASD / arrows | walk, with inertia; holding keeps raising the speed (logarithmically) |
| Click / right click | push / pull the surface region under the crosshair; holding repeats with growing steps |
| Shift (held) | preview the vertical and horizontal line under the crosshair |
| Shift + click / Shift + drag | add both lines / erase the line segments swept by the crosshair |
| Alt + click | window (hold and aim to resize); Alt + right click removes one; one window per surface |
| - / + | mist, 10 mild steps (light shafts stay faintly visible with no mist) |
| K / L / N | save (type a name) / list of saved rooms (arrows, Enter) / new room |
| P | screenshot (PNG download) |
| Ctrl + Z / Ctrl + Y | undo / redo |
| [ / ] | lower / raise resolution; both together: automatic again |

Rooms are kept in the browser's local storage. The room you are in is saved every two seconds, so a page reload brings it back.

## How it works

- **Engine (Rust, WebAssembly, no crates).** Space is a 3D grid of free and solid cells with uneven spacing. Wherever free meets solid there is a face, so every surface you create (the sides and ceiling of a passage, the back of an alcove) behaves like any other wall. Lines belong to the plane a face lies on and split into segments at intersections; a region is a patch of a face bounded by its edges and its lines. Pushing a region carves the cells behind it, pulling fills the cells in front, and its lines and windows move with it. Lines that no longer divide a visible face are removed, so nothing lingers behind walls. Undo keeps one snapshot per mouse press.
- **Renderer (WebGL2 fragment shader).** The same grid walk the engine uses for picking, run per pixel on the GPU. Light only comes through the windows: one sun for all windows, sky light from each window, one bounce of light through virtual point lights placed where window light lands, and single-scattering mist with exact light shafts. Frames accumulate while nothing changes, so a still view clears up and a moving one is grainy. The resolution adapts to the frame rate (720p at most unless you raise it).
- **No game framework and no JavaScript libraries.** `src/lib.rs` is the engine, `web/shader.js` the renderer, `web/main.js` the input, storage and WebGL glue.

## Run it locally

You need Rust with the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`) and Python 3.

```
./build.sh     # builds web/rooms.wasm
./serve.sh     # serves web/ on http://127.0.0.1:8642 (no caching)
cargo test     # geometry, editing, undo and save/load tests
```

In the browser console, `__X` is the engine (for example `__X.set_cam(x, z, yaw, pitch)`).

## Deployment

https://rooms.magiwanders.com is served by Caddy as static files from `/var/www/rooms` on the author's home server. `./deploy.sh` builds the WebAssembly and copies `web/` there; there is no backend.

## License

[CC BY-SA 4.0](LICENSE).

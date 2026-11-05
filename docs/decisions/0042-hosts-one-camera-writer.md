# 0042: Hosts: one map with one camera writer, a thin native API, and the web on one thread

Decision 0040's hosts part. The host and map camera handoff was written
four times: `orb-api`'s `Map` (which C, Kotlin and Swift reach),
`orb-demo`'s `State`, `orb-web`'s `UiRole` with `web/src/map.ts`, and
`orb-bench`. The copies had drifted into bugs:

- native `Idle` fired on every frame of an animation (`set_camera` re-armed
  it);
- the web's camera stood on a coarse copy of the terrain, refreshed at most
  every 30 frames while moving;
- the first frame of each style load and each DEM arrival drew with the
  previous camera;
- gesture times were the call's arrival, not the event's, and Android
  dropped `MotionEvent` times and historical samples;
- the native demo dropped the style's roll, Apple's view didn't recover
  from device loss, and worker mode had neither recovery nor Compat.

## Target

**One host-facing `Map`, and one writer for its camera.** `orb_runtime::Map`
owns the `Camera` and the `Interaction`. Each frame:

1. applies arrived data, DEM and surface changes included;
2. adopts the style's camera if nothing has moved it;
3. advances gestures and animations at the frame's timestamp;
4. draws.

`Idle` means no work and the camera at rest, and fires once. Hosts no longer
push the camera, adopt the style's camera or copy the surface and terrain.

**`orb-api` is thin native glue** over that `Map`: `input(Input,
event_time_ms)`, the camera moves, queries whose results arrive as events,
images, `render(&mut Target, frame_time_ms)` (which rebuilds its GPU state
after device loss itself), `needs_frame` and `poll_event` (warnings are
events too). `orb-capi` mirrors it, and C, Kotlin and Swift are generated
from it (decision 0018). Hosts translate input with its timestamps, pace
frames and own surfaces.

**The web draws on the page.** Render-worker mode is deleted (below). The
page runs the map (camera, gestures, synchronous hit tests, drawing); jobs,
fetch, glyphs and placement run in workers, which is the threading model
of every host.

## Render-worker mode is deleted

Decision 0021 measured it a frame slower than main-thread mode at p50, and
1–1.5 frames at p95, and made main mode the default. It was the main reason
the camera had a second copy: the page's ui role kept the camera, a picking
index the render worker published to it (`HitPublication`/`HitIndex`), and a
coarse elevation pyramid, which the approximate `unproject` and the 30-frame
publication cadence came from. Deleted with it: the `thread` option, the
render worker, the WebGPU-in-worker probe, picking publication, frame
acknowledgements' camera, and the page's approximate terrain (decision
0019's pyramid stays for the placement worker). DOM annotations now project
with the map's camera on its terrain, in the frame the canvas draws.

Measured after the deletion (`pnpm -C web measure --runs=2`, headless
Chrome 156, Apple M6, the two runs' worse median):

| | pan p50 / p95 | pinch p50 / p95 | rotate p50 / p95 | skew p95 (pan / pinch / rotate) | main busy: load / 3 gestures |
|---|---|---|---|---|---|
| Orbis | 38.8 / 54.1 ms | 37.5 / 39.5 | 37.6 / 54.8 | 1.6 / 8.8 / 2.7 ms | 79–108 / 157 ms |
| GL JS 6.16.0 | 51.7 / 87.2 | 56.5 / 460.7 | 52.9 / 74.2 | 1.6 / 8.8 / 3.4 | 152–189 / 280 |

Main mode before the deletion measured pan 38.8 / 55.6, pinch
38.8 / 372.6 and rotate 39.1 / 48.8 ms; the pinch tail is gone. Idle
draws nothing (no animation frames and no map frames in the quiet
windows).

## The map owns its camera

`orb_runtime::Map` holds the `orb_camera::Camera` store and the
`orb_interact::Interaction` recognizers. `Map::render` takes the frame's
timestamp and, in order, applies the messages that arrived (a style, DEM
tiles, a scene document's surface), brings the store onto that surface and
terrain and the frame's viewport, advances gestures and animations to the
timestamp, and draws with the camera that leaves. A style installed while
nothing has moved the camera is adopted at install, before the frame that
shows it. Input carries its event's timestamp (`Map::input(input, time)`),
on the same clock as frame times: the display link's on each platform
(`requestAnimationFrame` and `event.timeStamp`; Choreographer and
`MotionEvent` uptime; `CADisplayLink` and `NSEvent`/`UITouch` uptime).
Animations the app starts (`ease_to`, `fly_to`) begin at the next frame's
time, so a call between frames, or after a long idle, starts from its first
frame.

`Idle` fires once per rest: a frame is idle when its data work is done and
no animation or glide needs another frame, and the event waits until no
gesture holds the camera either. A camera that moved since the last frame
makes the next rest a new idle. `orb-api`'s test `idle_comes_once_after_an_ease`
renders a 400 ms ease on a 60 Hz clock: 25 frames, one `Idle`, on the
frame it arrives on (before, 37 of 38 frames emitted it).

One event stream: `MapEvent` carries the camera's moves
(`Camera(CameraEvent)`), clicks and double clicks, and warnings, which were
a second channel (`take_warnings`). Deleted: `Map::camera_mut`,
`set_camera`, `constrain_camera` and `StyleLoad { camera }`; hosts that
need an exact camera (fixtures, benches) use `jump_to_state`, which keeps
to the constraints and the terrain as every move does.

On the web the page's camera copy (`UiRole`) merges into the wasm
`OrbisMap`: one object per map takes input with its `timeStamp`, draws
each animation frame at `requestAnimationFrame`'s timestamp, and returns
typed `FrameStats` instead of a positional array. Camera targets and
options cross as the package's objects, read by name. Edits and image
calls are synchronous (they throw on a bad call); queries, snapshots and
regions stay asynchronous. Each map has its own wake route, and `remove()`
can be called twice and resolves running animations false. A job worker
whose request throws answers it with the error, so the map's request
completes.

`orb-api`'s `Map::render_until_idle` is the one way tools and tests draw
until the map is idle: a map waiting only on jobs needs no frame until
their wake, so loops that stopped when `needs_frame` turned false stopped
before the first tile.


## One clock: the frame's

Frames drive everything a frame shows. Fades (labels, raster tiles),
paint transitions, the pattern crossfade and the shaders' time run on the
frame timestamp the host passes, counted from the map's first frame; a
paint transition, like an animation, starts with the frame after its edit.
The wall clock is left for CPU budgets (tile intake, placement) and network
retry backoff. `orb-api`'s still output times its frames from the host's
last frame plus the wall time since, so a map keeps one clock. A test that
draws on a synthetic 60 Hz clock now sees the same frames under any load: a
400 ms ease ends on its 25th frame.

The page builds an object per event for the package (camera progress
events come every frame); their keys and names are interned, so the page
decodes each string once rather than per event (14 ms of `TextDecoder`
over a three-gesture profile before).

## The native demo is a host like the others

`orb-demo` draws through `orb-api` as C, Kotlin and Swift hosts do: a
`Runtime`, a `Map` and a window surface `Target`; winit input goes to
`Map::input` with its time; the flight loop is `fly_to` legs. Its own
device, frame loop and camera handoff are gone (about 1,500 lines). Still
pictures, its old `--snapshot` and `--a11y`, are `orbis render` (`--a11y`
writes the semantic snapshot); its tests, which drive the runtime directly,
moved to `orb-runtime/tests` (`frames`, `edits`).

## The fixtures draw through `orb-api`

`cargo xtask rt` makes one `orb-api` `Runtime` per suite (Modern, or Compat
for `gl` and `llvmpipe`) and a `Map` per test into a texture `Target`, set
up from GL JS's map options, which `MapOptions` now carries
(`fade_duration`, `cross_source_collisions`, `local_ideograph_font_family`,
`antialias`, `terrain_skirts`). The mirror's `local://` assets resolve
through `RuntimeOptions::url_transform`, the first of the host's fetch
hooks. Queries go through `Map::query_rendered` and its event. The native
render suite is unchanged at 1191 passes, 104 failures and 50 skips, test
for test (62.9 s → 58.0 s); the query suite at 71 of 83.

## The benches draw through `orb-api` and the package

`cargo xtask bench` makes an `orb-api` `Runtime` and `Map` and draws into a
texture `Target`; `bench.html` drives the `orbis` package, its frames
experiment included (`harness.wasm` keeps only the fixture driver). GPU
timing moved from the bench's `Gpu` wrapper into `orb-hal-wgpu`
(`WgpuGpu::set_timing`, `take_timings`): every submission's CPU time and,
with timestamp queries, every pass's GPU time. Hosts reach it as
`Runtime::set_gpu_timing` and `gpu_timings` natively and `setGpuTiming` and
`gpuTimings` in the package, beside `triggerRepaint` and a `pixelRatio`
option. The HAL lost its query sets, `PassTimestamps` and
`Pass::ResolveQuerySet`, which only the bench used. A take resolves its
queries once the GPU has finished: Apple GPUs write a render pass's end
sample after its fragment work, and a resolve right behind it read the
previous frame's. The cull counters (chunks kept, vertex inputs) left the
frames report: they were read out of the renderer's argument buffer, which
no host API reaches. Frame bench through `orb-api`, M6 Metal, medians of
2 × 50 frames (GPU ms; before → after): z12 0.450 → 0.611, z15
0.733 → 0.730, z15 pitch 60 0.834 → 0.840, z17 0.638 → 0.759, z16 pitch 60
with buildings 1.560 → 1.537.

## The Chrome fixtures draw through the package

`cargo xtask rt --platform chrome` opens `web/demo/fixture.html`, which draws
each fixture through the `orbis` package: GL JS's map options at creation,
operations as package calls, frames until loaded or idle, and the new
`readPixels` (the next frame as the canvas shows it). `harness.wasm`, its
`FixtureRenderer`, the `harness` feature and `orb_text::matrix` (decision
0009's script matrix digest) are deleted; the identity check of every
native pass against Chrome's picture covers what the digest did. Chrome
passes 1208 against 1206 before (88 fail, 49 skip), every one of the 1206
native passes identical; the query suite passes 71 of 83 in both. The run
takes 118 s, from 71 s: frames go through the package's animation frames.

## `render` recovers from device loss

A frame that finds the GPU lost (a hung compile, a lost EGL context, a
destroyed device) asks for another, and that frame makes the new device
(plan §8.8): `orb-api` releases a window target's surface on the old device
first, since a window takes no second EGL surface, then texture targets are
made again and window targets fail with `TargetLost`, which hosts already
answer by making another surface. `Error::DeviceLost`, the public
`Runtime::recover` and the C API's `recover` (ABI version 8) are gone, as are
the device-loss branches of the Android and Apple views and `orb-demo`. On
the Android emulator (Compat) a forced loss shows a complete frame again
1.9 s later; natively a texture target redraws in 10 ms.

Texture targets keep two frames in flight (a frame waits for the one two
before it, `WgpuGpu::end_frame`), as a display's swapchain does, and the
bench no longer waits for the GPU itself; its work-done proxy for devices
without timestamp queries goes with `Runtime::finish`, since every native
backend has them. Its earlier wait, the whole GPU after every second
frame, idled it between frames. Same-session medians of five runs each
(GPU ms, z12 and z17, M6 under load): the old bench 0.982 and 0.994; the
bench through `orb-api` with that wait 0.978 and 1.157; two in flight on
today's `main` 0.981 and 1.004 (again under heavier load: the old bench
1.423 and 1.769, the new 1.335 and 1.878). The z12 and z17 rise reported above was
the wait, not the renderer. The full bench at load average 23: z12 0.476,
z15 1.087, z15 pitch 60 1.527, z17 1.271, z16 pitch 60 with buildings
4.194. The cull counters stay out: the cull shader writes them on the GPU,
and a frame's stats would need a readback of them every frame.

## After the rebuild

`pnpm -C web measure --runs=3`, headless Chromium on the M6, 800×600 at
1×, medians of three runs, under load from other sessions (load average
about 22):

| | pan p50 / p95 ms | pinch p50 / p95 ms | rotate p50 / p95 ms | main busy, load / gestures ms |
|---|---|---|---|---|
| Orbis | 37.8 / 40.2 | 36.3 / 52.6 | 36.5 / 39.1 | 88.6 / 187.1 |
| GL JS | 54.4 / 87.4 | 50.4 / 61.2 | 53.6 / 387.1 | 157.7 / 262.9 |

Input to photon is 15–17 ms (about a display frame) sooner than GL JS's at
the median, and the main thread is busy 56% as long during load and 71% as
long during the gestures. Neither map draws a frame while idle, before or
after the gestures.

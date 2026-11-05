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

# 0021: Interaction, the web package, and the default thread mode

M3 needs gestures and camera animations on every host, a typed web package
over both thread modes, and plan §7.2's measurements to fix which mode is
the default.

## Interaction and the camera

`orb-camera` holds the single-writer camera store (`Camera`): constraints
(zoom, pitch, `maxBounds` and the world's edge), `jumpTo`, `easeTo` and
`flyTo` after GL JS (its default easing, van Wijk and Nuij's curve with
ρ = 1.42), glides for inertia and smooth wheel zoom whose properties ease on
independent schedules, `cameraForBounds` with padding, and GL JS-style
`move…`/`zoom…`/`rotate…`/`pitch…` start, progress and end events reported
once a frame. Fits look straight down as GL JS does, then refine numerically
for a pitched camera so the whole box stays inside the padding.

`orb-interact` is the gesture layer, host-independent: pointer, touch,
wheel and key input with timestamps in, moves on the store out. It ports
GL JS's handlers' behaviour and constants (click tolerance 3 px, double
taps within 500 ms and 30 px, pinch arming at 0.1 levels, rotation at 25 px
of arc, the wheel/trackpad heuristics and rates, the 160 ms inertia buffer
and 60 ms velocity window, keyboard steps). Feel profiles change rates and
inertia: `DESKTOP` is GL JS's; `IOS` glides pans like `UIScrollView`
(0.998 per ms decay); `ANDROID`'s decay (0.9935 per ms) is matched to
`OverScroller`'s spline at 2,000 px/s on a 2.75× display [inferred]. User
input interrupts animations (`AnimationEnd { interrupted: true }`).
Not ported: box zoom and cooperative gestures.

Hosts own the store and push its state to the map when it moves; until the
app or the user moves it, the map takes the style's camera and hosts adopt
it from `MapEvent::StyleLoad`. The native demo, the web ui role and part B's
`orb-api` all use the same pair.

## The web package

`web/src` is the `orbis` package: `createMap(options)` → a map with
`camera`, `edit`, typed events (`on`/`off`/`once`, `events(type)` async
iterators), synchronous `project`, `unproject` and `hitTest`, async
`queryRenderedFeatures`/`querySourceFeatures`, and DOM `annotations`. The
ui role (`UiRole` in `orb-web`) is the camera authority on the page in both
modes, per plan §7.1. In main-thread mode the map draws in the same
animation frame; in worker mode the page sends the camera once a frame and
the render worker draws it as it arrives, skipping its own animation
frame's draw when a camera drew already (waiting for the worker's frame
cost a frame). Frame acknowledgements carry the frame's camera, which
places DOM annotations, and the map's events; GL JS-style `idle` waits for
a frame that shows the camera at rest. Picking snapshots carry their camera
and publish once it rests three frames (or the map is idle); points picked
while it moves map through the ground to the snapshot.

`core.wasm` grows 627 → 663 KiB Brotli over M3 so far (the ui role,
gestures, animations and queries; the build at the start of M3 against
this one, which also carries other M3 work merged meanwhile).

## Measurements (plan §7.2)

`pnpm -C web measure` drives `web/demo/measure.html`: the Tokyo extract at
z15 with a blue circle drawn by the map and a cyan DOM annotation at the
same point. DevTools dispatches one-second touch gestures (pan 300 px,
pinch 1.5×, a 90° twist) while it captures composited frames; the page logs
where the circle should be after each input, from its camera, so a frame's
circle gives the input it shows. Latency is the frame's swap time minus
that input's timestamp on the browser's clock (input-to-composited-frame;
display scan-out is not included); skew is the annotation's offset from the
circle as input time; main busy time is DevTools' `TaskDuration`.

Headless Chromium 156, Apple M6, 60 Hz, 800×600, medians of three runs:

| | pan p50 / p95 | pinch p50 / p95 | rotate p50 / p95 | skew p95 (pan / pinch / rotate) | main busy: load / 3 gestures |
|---|---|---|---|---|---|
| Orbis, worker | 54.2 / 69.2 ms | 53.4 / 57.3 | 53.3 / 57.1 | 1.6 / 9.2 / 3.3 ms | 21 / 81 ms |
| Orbis, main | 37.1 / 40.0 | 36.9 / 41.4 | 85.6 / 391 | 1.6 / 8.8 / 2.8 | 276 / 433 |
| GL JS 6.16.0 | 53.2 / 71.6 | 48.9 / 59.4 | 53.0 / 369 | 1.5 / 9.2 / 2.4 | 139 / 287 |

An empty page receiving the same input costs 0 / 25 ms. An earlier run
gave the worker pan 55.1 / 57.7 ms against main 37.6 / 40.1. GL JS's and
main-thread Orbis's rotate tails are frames shown while the main thread is
busy at the twist's start.

Chrome 156 on a moto g 5G (2024, Adreno 619, 90 Hz), main busy only, with
the injected input's own cost (101–110 ms per gesture on an empty page)
subtracted: Orbis worker 150–210 ms load and 380 ms over three gestures;
GL JS 760–840 and 1,470; Orbis main 1,300–1,690 and 1,610–1,650. Frame capture on
the phone dropped most frames (often one or two per gesture), so its
latency and skew are unmeasured.

Pacing: a 2 s `easeTo` draws 122 frames (p50 16.7 ms apart, p95 18.2 in
worker mode; 16.6 / 17.2 on the main thread) and stops within one frame of
arrival. Idle means zero frames: no main-thread animation frames and no map
frames in 3 s after every gesture, in both modes and in GL JS; natively
(`orb-demo`'s `idle_maps_ask_for_no_frames`), no wake-ups and no camera or
gesture frames in 2 s.

## The default

§7.2's rule keeps the worker default if its p95 latency penalty is at most
one frame and its main busy time at most 25% of GL JS's. Measured: the
worker's penalty is one frame at p50 (pan +17.1 ms, pinch +16.5 at 60 Hz),
but its pan p95 penalty is 1.1–1.75 frames across runs (pinch 0.96); its
main busy time is 19% of GL JS's on the Mac and 25% on the phone. The busy
criterion holds only just, and the pan tail fails it, so by the rule **`thread: "main"`
is the default**. `"worker"` and `"auto"` (the worker where OffscreenCanvas
and WebGPU in workers work) remain, with the same API. Main-thread Orbis
keeps the main thread busier than GL JS (1.7× on the Mac, 1.4× on the
phone), which the worker mode exists to avoid; the worker mode's skew
stays under a frame (DOM annotations on frame acknowledgements).

## Measured again at the end of M3

The numbers above predate decision 0023 (picking on demand, coalesced
worker camera draws, budgeted tile intake, job workers in main-thread
mode), decision 0024 (picking without the camera: nothing published at a
gesture's end) and a leaner worker mode: the render worker spawns the job
and placement workers (the page made all seven, about 1 ms each), frame
acknowledgements are one transferred array sent only when the page needs
them, and annotations write the DOM only when they move. Medians of five
runs, the same page and gestures, Apple M6, Chrome 156 at 60 Hz:

| | pan p50 / p95 | pinch p50 / p95 | rotate p50 / p95 | skew p95 (pan / pinch / rotate) | main busy: load / 3 gestures |
|---|---|---|---|---|---|
| headless, worker | 55.9 / 57.8 ms | 55.1 / 56.3 | 54.7 / 57.7 | 1.6 / 8.8 / 3.9 ms | 7.2 / 70.4 ms |
| headless, main | 38.5 / 41.1 | 38.0 / 40.5 | 37.5 / 39.3 | 1.6 / 8.5 / 3.0 | 91.1 / 125.7 |
| headless, GL JS 6.16.0 | 54.4 / 334.1 | 52.7 / 62.3 | 54.2 / 405.9 | 1.6 / 9.3 / 2.3 | 162.5 / 210.0 |
| window, worker | 56.4 / 66.9 | 56.5 / 405.9 | 55.2 / 57.3 | 1.6 / 8.8 / 3.1 | 8.2 / 83.6 |
| window, main | 40.0 / 42.2 | 39.3 / 42.0 | 38.6 / 40.8 | 1.5 / 8.5 / 2.6 | 104.4 / 162.4 |
| window, GL JS 6.16.0 | 53.2 / 73.8 | 50.9 / 62.0 | 55.2 / 421.2 | 1.6 / 8.5 / 2.3 | 157.0 / 281.4 |

"Window" is `pnpm -C web measure --headed`: Playwright's Chromium in a
window on the display. Busy times without frame capture (`--no-capture`,
five runs, headless) agree: worker 13.2 / 70–78 ms, GL JS 143–156 /
270–281, an empty page receiving the same input −3 / 27–28.

The worker's p95 penalty over main-thread Orbis is one frame or a little
more: headless 1.00 (pan), 0.95 (pinch) and 1.10 frames (rotate); in a
window 1.48 (pan) and 0.99 (rotate), with one 406 ms pinch tail. Its p50
penalty is one frame throughout (16–17 ms). Its main busy time, the empty
page's subtracted, is 15% of GL JS's over load and gestures headless and
16% in a window; during load alone it is 4.4–5.2% (7–8 ms against
157–163), within plan §18 row 6's 10%. The rule's busy criterion holds;
its latency criterion still fails at the tail (pan in a window, rotate
headless), so **`thread: "main"` stays the default**. Main-thread mode's
busy time during load is 56–66% of GL JS's (§18 row 6 asks ≤40%).

What remains on the page in worker mode during gestures is per-frame work
that each frame needs: the animation frame that advances the camera and
posts it, the acknowledgement that places DOM annotations (a message task
and its deserialization), and the style, layout and commit an annotation's
move costs. JS and wasm on the page are about 15 ms of the 70; the rest is
the browser's (#65).

Firefox 157 on macOS runs the map but takes 15 s to idle (#60); its
latency is unmeasured (no driver for a stock Firefox here). Safari: the
owner enabled remote automation, but `safaridriver` still fails with
"session not created: The session timed out while connecting to a Safari
instance … Finding or launching a compatible local Safari and waiting for
its RWIApplication to appear and become ready" (#37). The moto g's Chrome
stopped bringing WebGPU maps to idle during this pass (a build from
before M3's changes as well; GL JS's WebGL map loads), so the phone's
numbers above are decision 0021's (#37).

Open: Chrome on Windows, Safari on macOS and iOS, Firefox latency and
Android latency (a capture that keeps up: a high-speed camera or
`screenrecord` with visible touches); the worker's latency tail.

## Since

Decision 0042 deleted the render-worker mode and its ui role: the camera,
gestures and drawing share one `Map` on the page, and hosts no longer push
the camera to the map or adopt the style's.

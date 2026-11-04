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

Open: Chrome on Windows, Safari on macOS and iOS, and Firefox are
unmeasured (no automation here: Playwright's Firefox has no WebGPU, its
WebKit isn't Safari, and `safaridriver` needs remote automation enabled);
Android latency needs a capture that keeps up (a high-speed camera or
`screenrecord` with visible touches). M3's exit criterion of worker-mode
main busy at most 10% of GL JS's is not met (19–25%).

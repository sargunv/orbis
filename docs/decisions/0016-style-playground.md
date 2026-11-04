# 0016: Style playground and published picking

The M1 demo needs live JSON edits and hover feedback, with the same edits
visible in a native viewer. It uses the existing `Map::edit`, GL JS command
transactions and `edit.js`, including `setStyle`'s diff. It adds no second
style mutation implementation.

## Picking publication

Tile jobs attach selected feature geometry and promoted ids to their mesh
results. The packed worker result transports this metadata with its geometry;
property-row jobs can refresh picking dimensions without triangulating again.
Resident meshes own the metadata, so eviction and stale job rejection follow
the existing mesh lifecycle. Image and symbol nodes contribute no geometry.

The render publication projects the resident geometry through DataToWorld and
the current camera into CSS pixels. Projected screen coordinates use
32-bit floats, matching GPU precision and halving coordinate transfer bytes;
world projection remains in double precision. The index stores bounding boxes, polygon
rings and holes, line segments, circles, and extrusion roofs/walls. It observes
layer visibility and ordering, translated geometry, and world wrapping. Its
cache key covers camera, viewport, style revision and mesh/row publication.
Color-only feature-state writes retain the index. Camera and style mutations
do not change queries until a frame publishes them.

`Map::hit_test` and `OrbisMap.hitTest(x, y)` return the topmost geometric hit:
layer, source, optional source layer, optional promoted id, and generation.
The render worker transfers changed picking snapshots with its frame ack;
the page's `PublishedHitIndex` uses the same Rust picking implementation.
`orbis.hitTest(x, y)` is therefore synchronous in either thread mode. Retained
snapshots advance their generation with each frame acknowledgement. No tile
decode, GPU readback or worker round trip is needed for a pointer query.
Symbol placement can append its separately published index at `query.rs`'s
publication seam. Full properties and `queryRendered` remain later work;
the 81 query fixtures were not run as hitTest does not implement that API.
Exact line joins/caps/offsets, pitched circle dimensions, state/composite
size expressions and intersecting extrusion depth are tracked in
[#8](https://github.com/sargunv/orbis/issues/8).

## Playground and native link

`playground.html` is the fourth Vite entry. Its textarea applies whole-style
diffs after a 200 ms pause. Failed JSON or style validation leaves the map
and accepted document intact. Hover clears the previous feature's `hover`
key and sets the new one in one transaction. The bundled self-contained
GeoJSON style includes blocks, a route and stations with stable feature ids.
The page reports the generation and rendered frames until it becomes visible.

The development server optionally writes accepted style and hover snapshots
to `ORB_PLAYGROUND_STYLE`. This endpoint requires a same-origin PUT and has a
4 MiB request limit. It exists only in Vite's development server. The native
`--watch-style` host polls the file every 16 ms, then applies transactions on
its event loop. Full snapshots avoid missed clears or edits if file changes
coalesce. The file also accepts a bare style object or a GL JS transaction
array. `--snapshot` with `--watch-style` applies the current file once before
writing the image, allowing headless validation of the same bridge.

The frame count starts when the host accepts a transaction. Editor debounce,
file polling and scheduling between browser and native processes are outside
that count; this is not a cross-process latency claim. Pages neither calls
the endpoint nor depends on a native process. Launch commands are in
`AGENTS.md`.

## Measurements and verification

On Apple M6, native offscreen and headless Chromium with WebGPU:

| Path | Paint | Layout | Constant → data binding | Feature state |
| --- | --- | --- | --- | --- |
| Playground, Chrome main | 1 frame | 1 frame | 1 frame | 1 frame |
| Playground, Chrome worker | 1 frame | 1 frame | 1 frame | 1 frame |
| Tokyo, native thread pool | 1 frame | 2 frames | 2 frames, 0 relayouts | 1 frame |
| Dev file bridge, native snapshot | 1 frame | — | — | 1 frame |

The small playground's cached tile jobs complete before its next scheduled
frame. The Tokyo sequence uses the existing edit tests from decision 0015;
job latency can change the frame count without adding a relayout. In both
Chrome thread modes the playground's paint edit uploads 0 meshes/0 rows,
the binding flip uploads 0 meshes/2 rows, and layout uploads 2 meshes/0 rows.

Native's 400-write benchmark measured 0.0018 ms per edit and 0.0135 ms for
edit plus the additional frame work. Chrome's 500-write run measured
0.0022 ms per edit and 0.0058 ms including additional frame work, under the
0.05 ms target. The Chrome difference-of-totals measurement is noisy, as
decision 0015 explains. Combined-build reruns measured edit calls of
0.0005 ms native and 0.0026 ms Chrome; their frame differences were
−0.0014 ms and −0.0044 ms respectively, below the noise floor rather than
negative work. The earlier positive measurements bound the observed cost. Full-style file snapshot application costs about
0.12–0.16 ms, separate from the feature-state-only write measurement.

`pnpm -C web playground` checks both thread modes, synchronous topmost picking,
hover pixel changes and clearing, invalid JSON recovery, and edit generations.
`playground:native` verifies the dev server's emitted style/state transactions
through native offscreen rendering. A windowed smoke run reported an occluded
surface in this environment; window presentation was not verified.
Native tests also verify published hit stability before rendering, layer
visibility, camera changes, snapshot transport round trips and malformed
snapshot rejection. `cargo xtask check` covers both targets and crate boundaries.

Before concurrent symbol work, `cargo xtask web` reported 1448 KiB raw and
470 KiB Brotli for the playground's `core.wasm`: +11 KiB Brotli over the
459 KiB completed style-edit baseline in decision 0015. Symbol typing moved
the combined build to 479 KiB; the text-stage integration moved it to 552 KiB,
above the 550 KiB ceiling.

Frame reports are read-only, avoiding unused JS setters. Binaryen optimizes
to convergence and strips tool metadata. The tile and hit headers share one
string decoder, rather than a generic copy for each cursor. These measured
reductions brought the text-stage combined build to 549 KiB Brotli without
changing the ceiling.

With the glyph atlas, raster jobs and final picking geometry,
`cargo xtask web` reports 1718 KiB raw and 550 KiB Brotli (integer KiB).
The size remains above the 550 KiB ceiling. The owner's size policy now
reports warnings instead of blocking builds or publication (decision 0003).
The initial 441 KiB estimate predates both the finished edit API and text-stage
work; the isolated playground increment was 11 KiB over the 459 KiB edit baseline.
Size cuts are recorded in decision 0003.
`harness.wasm` (bench and fixture driver) is outside the map payload budget.

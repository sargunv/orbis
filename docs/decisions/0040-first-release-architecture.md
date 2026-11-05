# 0040: The first-release architecture

The first release (plan §20) asks for a codebase that is simpler, net negative
in lines, correct for every camera and fast. A whole-system review covered
four areas: the frame and GPU core, the data pipeline, hosts and API, and an
audit for shortcuts and bad behaviour (#79). This record is the target it
settled on. Each part's own record holds the details as it lands.

## What the review found

- **The core.** `Map` owns the `Renderer` and runs the whole frame:
  - `map.rs` is 4.6k lines, and `orb-runtime` is 21k against the plan's 6k;
  - the runtime makes about 50 calls into renderer internals;
  - every tile row is rewritten every frame;
  - zooming rebuilds every cull candidate;
  - terrain drapes and label placement update only once the camera rests;
  - about 110 MB of GPU memory holds a 2 MB scene.
- **The data pipeline.** v8 is the model in practice:
  - the style exists in four or five forms;
  - each layer type's properties are listed about seven times;
  - paint is evaluated in five places, one of them a GPU interpreter that
    only runs for paint mixing feature state with feature data;
  - there are six hand-written wire codecs;
  - label state lives per tile mesh instead of per label, which causes #55.
- **Hosts.** The camera handoff between host and map is written four times,
  and the copies have drifted into bugs: idle fires on every animation frame,
  terrain lags under the camera, and gesture timestamps are wrong. Render
  worker mode is the main reason the handoff exists, and it measures a frame
  slower than main-thread mode.

## Target

**Objects (plan §5.4, now enforced).**
- **`Map` is CPU only.** It holds the scene, camera and interaction, tile
  states, jobs, feature state, labels, CPU elevation, queries and events. It
  owns the camera: it applies arrived data, adopts the style's camera, advances
  gestures at the frame timestamp, plans, then draws. `Idle` fires once.
- **`plan_frame(&Map, &View)`** is a pure function of the camera and the
  resident set.
- **`Renderer`** is the narrow GPU owner of all residency. `orb-api` keeps it
  beside the target, and dropping it frees everything.

**Every frame is right for its camera.**
- Tile rows are camera-independent: absolute origins as hi/lo f32, with the
  camera in the frame block.
- Terrain drapes are keyed by content, at a fixed zoom per terrain tile.
  Viewport-anchored hillshade is shaded in the terrain pass.
- Label state is keyed by cross-tile identity. Placement runs whenever its
  inputs change, including the set of drawn buckets and the ground.
- The pattern and dash crossfade depends on zoom only.

**One native scene model.**
- `orb-v8` is a front end onto `orb-style::Scene`, whose layers are
  table-driven by `PropertyId`.
- Edits are typed transactions.
- Paint has one path: a constant, curve or column per node, with curves
  evaluated on the GPU at the frame zoom.
- The GPU expression interpreter is deleted. Paint that reads both feature
  state and feature data is patched on the CPU in the same frame as the state
  write, as GL JS (`updatePaintArrays`) and MapLibre Native do.

**One wire codec and one job host.**
- Worker messages use postcard; mesh streams stay raw and transferable.
- Job code stays in `orb-runtime` but apart from the renderer: its modules
  (the worker, tile, symbol and glyph jobs, features, GeoJSON, the job host)
  name no renderer, scene or GPU type, which `cargo xtask check`'s boundary
  rules enforce. Moving it to a crate of its own was dropped: job workers
  load the same `core.wasm` as the page, so it saves no size; it adds lines
  rather than removing them; and no existing crate may hold it under plan
  §5.3 (`orb-layout` is job-side and may not read the style, and
  `orb-jobs` would bring the style and text crates into the GPU HAL's
  build).
- Native `PoolJobs` and the browser's job workers share one job host.

**Hosts.**
- Render-worker mode is deleted (D11, D12, §7.1–7.2, decision 0021). Main
  mode is a frame faster, and its main-thread busy time is 30–37% of
  GL JS's. The web has two roles: the page and the workers.
- `orb-api` is a thin `Map` and `Runtime`: input carries event timestamps, and
  `render` recovers from device loss itself. C, Kotlin and Swift are generated
  from it.
- Hosts translate input, pace frames and own surfaces, nothing more.
- Batteries (annotations, the location layer) are a module on `Map`, with no
  `orb-batteries` crate.
- Snapshots and `Tile` rendering are `Map` API, on decision 0039's modes.
- The HTTP cache and offline regions sit on a `Storage` host service.

**Tooling.** One way to do each thing:
- `orb-demo`, the fixtures and the bench go through `orb-api`;
- `harness.wasm`, the native playground bridge (decision 0016), the macOS
  Swift demo app, the redirect and probe pages, and the settled bench
  experiments are deleted;
- the full binding scenario runs in Rust; other languages run smoke tests.

## How it lands

Three chains rebuild toward this in parallel:
- **core:** the Map/Renderer split, rows, cull, offscreen groups, reflected
  layouts and memory;
- **data:** the scene model, the paint path, the codec and the tile jobs;
- **hosts:** camera ownership, a thin API, the web, the demo and the tooling.

Feature lanes build into the seams above. Each step is validated against the
fixture runs and benches it touches. The estimate is about −18k lines before
features add some back, against 113k at 0eeb3de (#78).

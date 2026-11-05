# 0038: Pipelines compile off the drawing thread, only what the nodes use, under a watchdog

M4 measured cold pipeline creation on the moto g (Adreno 619) at 17.0 s on
Vulkan and 17.0 s on GLES, all of it on the thread that draws. On Android
that is the UI thread, so Compat's first launch raised a "not responding"
dialog (#49). A driver that hung in its compiler wasn't caught by the crash
sentinel (#51). Plan §8.4 and §8.8 already asked for lazy, asynchronous
pipelines in first-visible order and a watchdog; this records how they were
built and what they measured.

## The HAL: create returns at once, readiness is polled

`Gpu::create_render_pipeline` and `create_compute_pipeline` queue a
compile and return the id. Commands may name a pipeline only once
`pipeline_ready` says it compiled. `poll_pipelines` takes finished compiles
and counts pipelines made (it only grows) and pending.
`prioritize_pipeline` moves a queued compile to the front, where the
implementation keeps a queue.

- **wgpu, natively:** worker threads (half the cores, at most four) take
  jobs from a queue and make the pipelines with wgpu, whose device is
  shared between threads. A job owns everything it needs; shader modules
  are parsed by naga on the first worker that needs them. Validation
  errors and panics come back as failures. `orb-hal-wgpu` and `orb-hal-gl`
  may own threads (plan §5.3); nothing below `orb-runtime` does.
- **wgpu, in the browser:** wgpu has no async creation (gfx-rs/wgpu#3794).
  The HAL builds the same request as a WebGPU dictionary and calls
  `createRenderPipelineAsync` / `createComputePipelineAsync`, which compile
  away from the GPU process's command stream. Once the promise resolves,
  wgpu makes its own pipeline from the same descriptor; Dawn keys pipelines
  (and shader modules and layouts) by content, so that returns the compiled
  one. Upstream wgpu has a PR for this (gfx-rs/wgpu#10438); once it lands,
  the dictionaries go.
- **GL, natively (GLES through EGL):** a compile thread links programs on a
  context that shares the executor's objects (surfaceless, or a 1 × 1
  pbuffer), reads binaries back for the program cache, and `glFinish`es
  before handing the program over. The executor then sets the program's
  uniform bindings on its own context. `KHR_parallel_shader_compile` isn't
  enough here: Android's drivers and the emulator's ANGLE lack it.
- **GL, in the browser (WebGL2):** with `KHR_parallel_shader_compile`, the
  executor compiles and links without asking for status and polls
  `COMPLETION_STATUS_KHR`. Without it, one program links per frame: a frame
  waits for one compile, not all of them.

## The renderer: what the nodes use, in draw order, drawn when ready

The renderer no longer makes every family's pipelines with itself. The cull
pipeline is asked for first; then, when nodes change, each node's pipeline
in draw order: its paint class (table or bytecode) for fill, fill outline,
line and circle; the symbol pipeline; raster, hillshade or color relief;
heatmap and extrusion pairs; the drape atlas's variants for draped nodes.
The DEM prepare pipelines come with the first DEM page; terrain, sky and
scattering with their first use. While pipelines compile, each frame moves
those of nodes with meshes in the scene to the front, the bottom layers
first (they cover the most of the view).

A node whose pipeline hasn't compiled records nothing and is counted as
waiting; there is no placeholder and no fade. When pipelines arrive, the
bundles are recorded again. Passes that need pipelines (terrain, the DEM
prepare, extrusion and heatmap passes, the sky) skip the frame and keep
their work: DEM jobs stay queued, and terrain starts a new drape
generation, because the map placed the drape only for the slots it asked
to redraw that frame. A frame that left anything out reports `waited`, so
the map is neither loaded nor idle; snapshots, fixtures and tests see only
complete frames. `FrameStats` and the frame reports of the C ABI and the
web package count pipelines pending and layers waiting; the debug overlay
and the Android reference app show them.

A map waiting only for compiles needs no frames. Natively the compile
threads (wgpu's and GL's) wake every map of the runtime when a pipeline
arrives, so hosts that draw on demand stop drawing until then. On the moto
g, frames drawn during a cold load fell from 4,363 to 76–87, and the
symbol compile from 37.3 s to 7.0–7.4 s. The browser hosts still draw while
compiles are pending (#70).

## Fewer and smaller pipelines

Liberty on the Tokyo extract now makes 5 pipelines instead of 16 (cull,
fill, fill outline, line, symbol); the Alps terrain style makes 16.

- **Only what the nodes use** (above): circle, heatmap, raster, hillshade,
  color relief, the DEM prepare and extrusion are no longer made for styles
  without them.
- **One symbol pipeline instead of three.** The symbol shader's `PASS`
  (halo and fill together, halo only, fill only) was an `override`, so
  each value was a pipeline (a Compat program). It only chose between two
  branches of the fragment shader, so it is now a uniform in a group of its
  own, one bind group per value.
- **One call site per heavy function in the symbol shader.** Adreno's
  compilers inline every call (decision 0017 found seven inlined copies
  crashed the Vulkan driver). `line_glyph` called `place_glyph`, a loop over
  the line with projections inside, five times; it now calls it once from a
  loop over the placements it needs. The next vertex in an offset line was
  projected twice with the same arguments; it is projected once.

The symbol pipeline on the Adreno 619 (Vulkan, compiled on the UI thread,
cold), as the shader changed: 28.5 s at this change's base, 8.4 s with one
`place_glyph`, 6.7 s with the shared projection. It still dominates (#69).

## The watchdog

Each compile thread notes what it compiles and since when. A compile
running past `orb_hal::device::COMPILE_WATCHDOG` (60 s, about 8× the
slowest compile measured on the Adreno 619 after this change) marks the
driver hung: the device is lost, and `orb-api` leaves the crash sentinel
armed. `Runtime::recover` then chooses a device as at the next start, where
the sentinel demotes it to Compat; a loss for any other reason clears the
marker first. The sentinel is now cleared once a frame has no pipelines
pending, rather than at the first frame, since that frame comes before the
compiles end. A test injects a compile that never returns
(`ORB_FAULT_INJECTION=hang`): frames go on (44 in 0.3 s), the watchdog
fires, and both the recovered device and the next runtime are demoted. A
hung thread is never joined. Compat is the last tier, so its watchdog only
reports.

## Measured

Tokyo extract, Liberty, from launch (or navigation) to the first frame that
draws a layer, to the last pipeline, and to idle.

| Where | Cold: first draw, pipelines, idle | Warm: first draw, idle |
|---|---|---|
| moto g, Modern (Vulkan) | 1.02–1.32 s, 8.18–8.36 s, 8.18–8.36 s | 0.30–0.54 s, 1.08–1.41 s |
| moto g, Compat (GLES) | 0.81–0.87 s, 13.27–13.40 s, 13.27–13.40 s | 0.24–0.27 s, 1.25–1.29 s |
| emulator (API 36), Compat | 0.73–1.08 s, 1.85–2.42 s, 1.85–2.68 s | 1.00–1.08 s, 2.12–3.56 s |
| emulator (API 36), Modern | 10.6 s, 10.6 s, 20.8 s | 1.53 s, 2.64 s |
| Chrome (headless, Apple M6), WebGPU | 124–157 ms, —, 0.86–1.01 s | 121–155 ms, 0.87–1.07 s |
| Chrome (headless, Apple M6), WebGL2 | 123–357 ms, —, 1.47–1.52 s | 138–156 ms, 0.91–0.94 s |
| macOS (Apple M6, Metal) | 14 ms, 101 ms, 679 ms | 15 ms, 670–685 ms |

M4's baseline on the moto g: 17.03 s of Vulkan compiles before the first
frame, and Compat's first launch at 18.4 s to idle with the UI thread
blocked for 17.0 s. Now the UI thread never waits for a compile: the
first frame draws layers within 1.3 s while the rest compile, and the
UI thread's 99th-percentile frame time (`dumpsys gfxinfo`) is 150 ms,
cold or warm. The emulator's Modern path is the exception: its Vulkan
transport runs every call in order on the host, so the first frame waits
behind the compiles, and it sometimes stops presenting during load, at
`main` too (#76); Compat there draws at once. Chrome's frames kept their
16.7 ms cadence while pipelines compiled, in both thread modes; on macOS
the longest frame while compiling took 5.9 ms. `pnpm -C web pipelines`
(#26) times the GPU queue's wait during a cold load in headless Chrome:
126 ms in total, 30 ms at most for one submission, against 225 ms in total
before. Each of the 5 pipelines is compiled with `createRenderPipelineAsync`
first and made by wgpu after, and the page spends 0.6 ms in pipeline calls,
so Dawn hands wgpu the compiled pipelines. Metal's shader cache is
shared by every process on the machine, so macOS cold means the first run
after a shader change.

Every fixture keeps its status from `main` natively on Metal (1161 pass,
113 fail), in Chrome on WebGPU (1161, 113) and on WebGL2 (1126, 67).

## The pipeline-load gate (plan §18 row 10)

Set from these numbers, on an Adreno 619 class phone, Tokyo extract:

- No frame waits for a compile, natively or in the browser.
- First frame drawing layers: ≤1.5 s cold, ≤0.6 s warm, Modern and Compat.
- Every pipeline compiled: ≤10 s cold; to idle: ≤1.5 s warm.

Modern meets all three. Compat meets the first two; its last program links
at 13.3 s cold, over the gate, because of the symbol shader (#69). Chrome
on Android, Chrome on Windows (D3D12) and Safari on iOS are still unmeasured
(#26).

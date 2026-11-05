# Cross-platform render measurements

Run `cargo xtask rt` from the workspace (native by default), or select paths
and platforms with `cargo xtask rt --filter line-width --platform all`.
Platforms are `native`, `chrome`, `lavapipe`, and `all`. Chrome also runs native
for a fresh identity baseline. Multiple platforms print one combined table.
The binary runs in release, sequentially,
with one GPU device/queue and shared fetch/job pools. Each test gets a fresh
`Map::new(services, None)` and GPU resource owner so that camera state and scene
allocations do not accumulate between tests. Each map also gets its own
`PoolJobs` style registry because style ids are map-local; the underlying
fetch and executor pools remain shared. There are no renderer changes.

The suite reads the GL JS fixture mirror in `fixtures/gl-js/`, which is fetched
rather than committed ([`fixtures/README.md`](../../fixtures/README.md));
`cargo xtask rt` fetches it first when it is missing or differs from the pin.
At the pinned revision the harness lives at
[`test/integration/render/render.test.ts`](https://github.com/maplibre/maplibre-gl-js/blob/2b68611a4e2d4f7f076f6c6ebf758e2d53f91471/test/integration/render/render.test.ts)
(the old `run_render_tests.ts` path is absent). Its defaults are 512 × 512 logical
pixels, pixelRatio 1, allowed 0.00025, threshold 0.1285, timeout 60000 ms.
Physical dimensions are floored after multiplying by pixelRatio; the camera
uses the logical viewport. Readback is unpremultiplied before comparison.
As upstream does, multiple `expected*.png` files use the smallest difference.

The comparator is a Rust port of **pixelmatch 8.0.0**, the version selected by
that revision's package-lock: OKLab HyAB with toe-corrected lightness, upstream
sRGB and cube-root lookup tables, default checkerboard alpha compositing,
`includeAA: false`, Rec.601 intensity-slope AA exclusion, and the default
red/yellow/grayscale diff output. The harness passes only `threshold` (default
0.1285); all those other options are pixelmatch defaults. Seeded RGBA images
check both mismatch counts and full diff-image hashes against upstream JS.
The JSON report identifies the comparator and effective defaults explicitly.

Parser and runtime plan diagnostics determine scope; original messages are
retained alongside stable, deduplicated skip reasons. GeoJSON property-string
conversion warnings are informational; source filters are skips.
`operations` apply after the map first goes idle, as GL JS's harness applies
them after `load`: `wait` and `idle` render until idle, `setZoom`,
`setCenter`, `setBearing` and `setPitch` move the camera, and every other
call goes through `Map::edit` as a one-call transaction (the same
`orb-style` command parser the web package's `edit` uses), natively and in
Chrome alike. Parser warnings that edits raise (an added symbol layer) skip as
the initial style's do. Operations this runner cannot apply (`addImage`,
waits for events or milliseconds) and metadata options it cannot honour are
skipped, including report cropping, non-default camera limits, debug views and
unknown options. Timeout metadata and custom thresholds are supported. Other
silent renderer omissions remain failures for diagnosis. Load errors and timeouts
are failures, not unsupported-feature skips. `local://` and GL JS's test-server
origin `http://localhost:2900/` are mapped to the assets directory by a Fetch
wrapper, including URLs requested from TileJSON, without rewriting styles.

`docs/scratch/rt/summary.json` contains selected platform reports and the
Chrome/native identity report. Each platform also writes `<platform>/summary.json`
with effective metadata, status, diagnostics, skip reasons, differing pixel
fraction, frames, tiles drawn, per-test time, aggregates and suite wall time.
Suite time includes GPU/browser initialization and image/report preparation;
Cargo/wasm builds are excluded. `container_seconds` additionally includes
Docker image build, startup, Cargo build and execution. All rendered tests save
`<platform>/images/<test>/actual.png`; failures also get
`<platform>/failures/<test>/{actual,diff}.png`. Each selected platform's directory
is replaced on its next run. Skip counts count a test once per reason and can
sum to more than the skip total. Pass rates use pass / (pass + fail). Fixture
failures exit zero; malformed inputs, missing expected images, initialization,
transport and report I/O errors exit nonzero. `cargo xtask check` uses neither
Chromium nor Docker.

## Chrome and identity

`cargo xtask rt --platform chrome` builds `core.wasm` first. Install tools with
`mise install`, `pnpm -C web install` and `pnpm -C web exec playwright install chromium`.
The native Rust runner owns discovery, metadata, scope, alpha conversion,
expected-image selection and pixelmatch. It starts one Node/Playwright process
(`web/scripts/render-fixtures.mjs`) with a line-oriented JSON protocol. Node
serves the demo build and the fixture assets over loopback (their JSON's
`local://` URLs pointing at it) and opens `web/demo/fixture.html` in headless
Chromium with animation frames unpaced. The page draws each fixture through the
`orbis` package as a host would: a map with GL JS's options, the operations as
package calls, frames until loaded or idle as natively, and `readPixels` for
straight RGBA (alpha unassociated by truncation, as GL JS's harness and the
native runner do). Videos stay on their first frame. A failed render reloads
the page before the next request.

Identity considers every native pass, even if the Chrome expected-image test
failed. It compares the two actual images using that fixture's unchanged
`threshold` and `allowed`, and reports `match`, `different`, or `unavailable`.
Every mismatch gets `identity/<test>/{native,chrome,diff}.png`; unavailable images
are reported explicitly. Chrome's test status and expected comparison remain
separate from identity. Main-thread mode is sufficient for this platform check.
Worker coverage would additionally exercise OffscreenCanvas, packed tile-job
messages, worker wakeups, and the JS-to-GPU upload path; it is not exercised by
this runner.

## lavapipe

`cargo xtask rt --platform lavapipe` builds the checked-in
[`lavapipe/Dockerfile`](lavapipe/Dockerfile) and runs the same native binary under
Linux Vulkan. The image pins Rust to the workspace toolchain, uses Debian slim
with only compiler/runtime and Mesa Vulkan packages, and restricts the Vulkan
loader to Mesa's lavapipe ICD. It needs no display or mounted GPU. The checkout
is mounted read-only except for the lavapipe report directory. Named Docker
volumes `orbis-rt-target`, `orbis-rt-registry` and `orbis-rt-git` cache Linux builds
and downloads without mixing native host artifacts. Repeated runs need only
incremental Cargo compilation. Docker must be running; on this Mac the measured
container uses Colima's Linux aarch64 VM.

## Sprite/pattern extension

On the initial 278-fixture mirror, Metal expanded from 99/101 passing (177
skips) to 131/133 (145 skips); Chrome also passed 131/133, with all 131 native
passes matching Chrome. All 32 newly enabled cases passed.

With the expanded 422-fixture mirror, expression engine, raster,
scene-eviction and circle/heatmap work, both platforms have the following
results. The before column is the baseline without sprites and patterns. Goldens,
comparison thresholds and allowed differences are unchanged.

| Prefix | Before pass / fail / skip | Metal after | Chrome after |
| --- | ---: | ---: | ---: |
| background- | 7 / 0 / 11 | 17 / 0 / 1 | 17 / 0 / 1 |
| circle- | 58 / 0 / 5 | 58 / 0 / 5 | 58 / 0 / 5 |
| color-relief | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 |
| fill- | 28 / 1 / 37 | 43 / 1 / 22 | 43 / 1 / 22 |
| heatmap- | 17 / 0 / 0 | 17 / 0 / 0 | 17 / 0 / 0 |
| hillshade- | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| line- | 66 / 1 / 75 | 75 / 1 / 66 | 75 / 1 / 66 |
| raster- | 34 / 2 / 1 | 34 / 2 / 1 | 34 / 2 / 1 |
| fill-extrusion- | 0 / 0 / 52 | 0 / 0 / 52 | 0 / 0 / 52 |
| Total | 237 / 4 / 181 | 271 / 4 / 147 | 271 / 4 / 147 |

All 34 newly enabled cases pass. The original 99 passes remain passing.
Chrome/native identity is **271 match, 0 differ, 0 unavailable**. Suite wall
times were **5.260 s native** and **6.474 s Chrome** on the final run.
The four existing failures remain: `fill-outline-color/multiply`,
`line-miter-limit/default`, `raster-anisotropic-filtering/pitch-45`, and
`raster-npot-mipmap/pitch-60`. Lavapipe was not rerun for this extension.

Additional browser checks render `line-pattern/pitch` and
`fill-pattern/case-data-expression` through main-thread and worker modes;
both pairs have **zero differing pixels**. The pitched line worker case uses
five tile jobs, 0.4 KiB of packed headers in wasm, one cull dispatch, bundle
replay and zero paint writes after idle.

Against a build without sprites and patterns, with the same toolchain and Brotli compressor,
`core.wasm` grows from **1,224,682 to 1,271,395 bytes raw** and from
**398,484 to 414,340 bytes Brotli**: **+15,856 bytes (15.48 KiB)** compressed.
This includes low-zoom and deterministic-publication fixes made alongside
it. The final artifact is **404.63 KiB**, leaving **46,460 bytes
(45.37 KiB)** under the 450 KiB gate. `cargo xtask web` and `cargo xtask check`
pass, including all nine demo camera, eviction and snapshot regressions.

## M0 platform measurement

At M0, macOS aarch64 host, Chromium 156.0.8078.4 WebGPU,
Linux aarch64 lavapipe in Colima (4 vCPU, 8 GiB), Mesa 25.0.7 / LLVM 19.
All runs use the same 278 mirrored fixtures, with 101 in scope and 177 skips.
No fixture, scope, comparison, tolerance or renderer changes were made.

| Platform | Prefix | Pass | Fail | Skip | Pass rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Metal | background- | 5 | 0 | 13 | 100.00% |
| Metal | fill- | 28 | 1 | 89 | 96.55% |
| Metal | line- | 66 | 1 | 75 | 98.51% |
| Metal | Total | 99 | 2 | 177 | 98.02% |
| Chrome WebGPU | background- | 5 | 0 | 13 | 100.00% |
| Chrome WebGPU | fill- | 28 | 1 | 89 | 96.55% |
| Chrome WebGPU | line- | 66 | 1 | 75 | 98.51% |
| Chrome WebGPU | Total | 99 | 2 | 177 | 98.02% |
| lavapipe Vulkan | background- | 5 | 0 | 13 | 100.00% |
| lavapipe Vulkan | fill- | 29 | 0 | 89 | 100.00% |
| lavapipe Vulkan | line- | 66 | 1 | 75 | 98.51% |
| lavapipe Vulkan | Total | 100 | 1 | 177 | 99.01% |

Chrome/native identity: **99 match, 0 differ, 0 unavailable**, covering every
Metal native pass. Each comparison counts zero differing pixels under the
fixture's pixelmatch threshold; this is not a claim of byte-for-byte equality.
The identity baseline is the host native backend, not lavapipe.

Failures left open for renderer work:

- `fill-outline-color/multiply`: 2 / 4,096 pixels (0.048828%) on Metal and
  Chrome, above allowed 0.025%. Lavapipe has zero differing pixels and passes.
  This is an observed backend edge/AA difference; its cause is not established.
- `line-miter-limit/default`: 98 / 65,536 pixels (0.149536%) on all three
  platforms, above allowed 0.025%.

Warm-cache suite wall times: Metal **0.836 s**, Chrome **2.469 s**, lavapipe
**4.766 s**. Docker build/start/Cargo/run totals **4.998 s** with Cargo completing
in 0.05 s; the initial cold Linux release compilation took 31.79 s. The Docker
image occupies 359,191,712 bytes locally (toolchain plus Mesa, no browser).
`core.wasm` measures 1,006 KiB raw / 332 KiB Brotli including the new entry point.
`cargo xtask check` passes without launching Docker or Chromium. An additional
render-only check uses a 31 × 17 logical viewport at pixel ratio 1.5: its
46 × 25 output has 4,600 correct premultiplied RGBA bytes, verifying fractional
physical-size rounding, padded readback rows and alpha independently of fixtures.

## GeoJSON / pixelmatch 8 measurement

macOS aarch64, native release runner, 278 mirrored fixtures. Before uses
pixelmatch 6 YIQ; after uses pixelmatch 8 OKLab HyAB and GeoJSON tiling.
Pass rate is pass / (pass + fail). The first measurement is without the
fill-antialiasing, translate and browser-worker work:

| Prefix | Before pass | Before fail | Before skip | Before rate | After pass | After fail | After skip | After rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| background- | 5 | 0 | 13 | 100.00% | 5 | 0 | 13 | 100.00% |
| fill- | 1 | 0 | 117 | 100.00% | 17 | 8 | 93 | 68.00% |
| line- | 29 | 1 | 112 | 96.67% | 52 | 11 | 79 | 82.54% |
| Total | 35 | 1 | 242 | 97.22% | 74 | 19 | 185 | 79.57% |

Measured runner wall time (excluding Cargo): 0.346 s before, 0.707 s after.
In-scope fixtures increase from 36 to 93 (+57); passes increase by 39. The
212 `source:geojson` skip reasons disappear; 155 of those tests still require
other unsupported features. All 19 failures are pixel mismatches, with no
load failures or timeouts. Upstream pixelmatch 8 JS independently reproduces
the Rust mismatch counts on all 19 actual/expected image pairs.

Fill-antialiasing, translate and browser-worker work landed alongside it. To
separate that from this implementation, the suite was measured again with that
work alone and with this change:

| Prefix | Before pass | Fail | Skip | Rate | Final after pass | Fail | Skip | Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| background- | 5 | 0 | 13 | 100.00% | 5 | 0 | 13 | 100.00% |
| fill- | 1 | 0 | 117 | 100.00% | 28 | 1 | 89 | 96.55% |
| line- | 33 | 1 | 108 | 97.06% | 56 | 11 | 75 | 83.58% |
| Total | 39 | 1 | 238 | 97.50% | 89 | 12 | 177 | 88.12% |

The final measured wall time is 0.836 s, versus 0.401 s without it.
In-scope tests increase from 40 to 101 (+61), and passes increase by 50.
There are 12 pixel failures and no source load errors or timeouts. Upstream
pixelmatch 8 JS also reproduces every final failing fixture's mismatch count.

Additional validation: `cargo xtask check` passes natively and for wasm32.
`cargo xtask web` produces 903 KiB raw / 285 KiB Brotli wasm. Headless Chromium
WebGPU checks cover inline data, URL data with nested properties, and missing
URL documents in main and worker modes (six runs). Main/worker canvas pixels
are identical in every pair; successful inline/URL images are identical too.
Inline data makes zero source fetches. The URL source makes one fetch in main
mode and two total in worker mode (one per job worker receiving its four tiles).
Property-conversion warnings cross the packed-result path, and missing source
documents report errors and settle idle in both modes.

The original 19 failing fixtures are described below as 15 observed patterns,
grouped by likely cause. The final column counts failures with the work that landed alongside it:
seven outline/AA failures now pass through those independent changes;
`fill-outline-color/multiply` retains only a 2-pixel discrepancy. Per-feature
line layout is still evaluated without a feature in runtime code. The other
geometry/AA diagnoses are hypotheses from actual/expected/diff images.
No renderer fixes were made in this implementation. Each failure appears once.

| Likely cause | Pattern | Original tests | Still failing | Fixture names |
| --- | --- | ---: | ---: | --- |
| Fill outlines absent from node paint | Constant and zoom-function outline colors missing | 2 | 0 | `fill-outline-color/literal`, `fill-outline-color/function` |
| Fill outlines absent from node paint | Outline opacity and multiplied alpha missing | 2 | 1 | `fill-outline-color/opacity`, `fill-outline-color/multiply` |
| Fill outlines absent from node paint | Per-feature outline colors missing | 1 | 0 | `fill-outline-color/property-function` |
| Fill outlines absent from node paint | Composite outline colors missing | 1 | 0 | `fill-outline-color/zoom-and-property-function` |
| Fill outlines / edge coverage | Implicit fill-colored outline missing | 1 | 0 | `fill-outline-color/fill` |
| Fill outlines / edge coverage | Transparent overlapping polygon edges differ | 1 | 0 | `fill-layer-opacity/default` |
| Line layout evaluated without a feature | Per-feature end caps use fallback | 1 | 1 | `line-cap/property-function` |
| Line layout evaluated without a feature | Per-feature joins use fallback | 1 | 1 | `line-join/property-function` |
| Line layout evaluated without a feature | Per-feature miter/round limits use fallback | 2 | 2 | `line-miter-limit/property-function`, `line-round-limit/property-function` |
| Acute-join geometry / thresholds | Miter-limit corner geometry differs | 2 | 2 | `line-miter-limit/default`, `line-offset/sharp-corners-miter-limit` |
| Offset extrusion / edge coverage | Small negative-offset vector-tile discrepancy | 1 | 1 | `line-offset/literal-negative` (already failed before; 28 pixels after) |
| Offset extrusion | Closed-loop corner/closure mismatch | 1 | 1 | `line-offset/loop` |
| Offset extrusion | Offset lines expose pre-offset reference geometry | 1 | 1 | `line-offset/matches-offset-geometry` |
| Offset extrusion | Sharp bevel corners differ | 1 | 1 | `line-offset/sharp-corners-bevel` |
| Offset extrusion | Sharp round corners differ | 1 | 1 | `line-offset/sharp-corners-round` |

## Raster and DEM measurement

macOS aarch64 (Metal) and Chromium WebGPU through the same
runner. Before, every `raster-`, `hillshade-` and `color-relief` fixture was
skipped as unsupported; the other families are unchanged.

| Prefix | Before pass / fail / skip | Metal after | Chrome after | After rate |
| --- | ---: | ---: | ---: | ---: |
| background- | 7 / 0 / 11 | 7 / 0 / 11 | 7 / 0 / 11 | 100.00% |
| color-relief | 0 / 0 / 7 | 7 / 0 / 0 | 7 / 0 / 0 | 100.00% |
| fill- | 28 / 1 / 37 | 28 / 1 / 37 | 28 / 1 / 37 | 96.55% |
| hillshade- | 0 / 0 / 20 | 20 / 0 / 0 | 20 / 0 / 0 | 100.00% |
| line- | 66 / 1 / 75 | 66 / 1 / 75 | 66 / 1 / 75 | 98.51% |
| raster- | 0 / 0 / 37 | 34 / 2 / 1 | 34 / 2 / 1 | 94.44% |
| Total | 101 / 2 / 319 | 162 / 4 / 256 | 162 / 4 / 256 | 97.59% |

Chrome/native identity: **162 match, 0 differ**. The two raster failures,
`raster-npot-mipmap/pitch-60` (55.4%) and `raster-anisotropic-filtering/pitch-45`
(0.35%), need mipmaps (#3); `raster-masking/overlapping-zoom` is skipped for
its `operations` metadata. Warm suite wall times: Metal 2.6 s, Chrome 4.7 s
(1.2 s before, rendering no image fixtures). The Chrome runner uses
the main-thread decode path; worker mode, checked separately with a raster +
hillshade + color-relief style, gives identical pixels with every texel copied
from transferred `ImageBitmap`s (decision 0007).

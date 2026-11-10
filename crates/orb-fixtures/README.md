# orb-fixtures

Orbis against the reference engines' test suites: MapLibre GL JS's render and
query fixtures, and MapLibre Native's render, query and expression suites, so it
is graded against both engines, and like for like with mln-rs, which runs
Native's. Use it through `cargo xtask rt`:

```sh
cargo xtask rt                                   # GL JS render suite, native
cargo xtask rt --filter line-width               # tests whose id contains a substring
cargo xtask rt --platform native|chrome|lavapipe|all
cargo xtask rt --suite query [--platform native|chrome|all]
cargo xtask rt --suite native-render [--platform native|chrome|lavapipe|all]
cargo xtask rt --suite native-query [--platform native|chrome|all]
cargo xtask rt --suite native-expression
```

`cargo xtask rt` fetches the mirror a suite reads (`fixtures/gl-js/` or
`fixtures/native/`) first when it is missing or differs from its pin
([`fixtures/README.md`](../../fixtures/README.md)); `native-render` fetches GL
JS's too, to compare shared tests with its images. The runner works in release,
one test after another, on one GPU device and queue with shared fetch and job
pools. Each test gets a fresh map and GPU resource owner, so no camera state or
allocations carry between tests.

## GL JS's render suite

The harness follows GL JS's
[`test/integration/render/render.test.ts`](https://github.com/maplibre/maplibre-gl-js/blob/2b68611a4e2d4f7f076f6c6ebf758e2d53f91471/test/integration/render/render.test.ts)
at the pinned commit. Its defaults are 512 × 512 logical pixels, pixelRatio 1,
allowed 0.00025, threshold 0.1285 and a 60 000 ms timeout. Physical sizes are
floored after multiplying by pixelRatio; the camera uses the logical viewport.
Readback is unpremultiplied before comparison, truncating and wrapping modulo
256 as GL JS's harness does in its `Uint8Array`: the globe's atmosphere adds
light over transparent pixels, which then hold more colour than alpha, and the
expected images keep those wrapped bytes. When a test has several
`expected*.png` files, the smallest difference counts, as upstream.

The comparator (`src/pixelmatch.rs`) is a Rust port of **pixelmatch 8.0.0**, the
version that GL JS's package-lock selects: OKLab HyAB with toe-corrected
lightness, upstream's sRGB and cube-root lookup tables, checkerboard alpha
compositing, `includeAA: false`, Rec.601 intensity-slope anti-aliasing detection
and the red/yellow/grayscale diff output. The harness passes only `threshold`;
everything else is a pixelmatch default. Its tests check mismatch counts and
diff-image hashes against upstream JS on seeded images. Its licence is
[`LICENSE-pixelmatch`](LICENSE-pixelmatch).

**Scope.** Parser and runtime diagnostics decide what is skipped; the report
keeps the original messages next to stable, deduplicated skip reasons. GeoJSON
property-conversion warnings are informational, and a warning that a GeoJSON
source's data is missing or invalid (so the source draws nothing) is one GL JS
raises too, so it neither skips nor fails a test. Operations this runner cannot
apply (waits for named events, `addImage` without an asset path) and metadata it
cannot honour (report cropping, non-default camera limits, debug views, unknown
options) skip the test. Timeouts and custom thresholds are honoured. A lint
(`src/dem.rs`) decodes the DEM tiles near each test's camera with the encoding
its style declares; a tile outside Earth's elevations (−12 000 to 9 000 m) skips
the test as `fixture:implausible-dem`, its diagnostics naming the tile and what
the other encodings read. Seven tests decode GL JS's terrarium-encoded Grand
Canyon tiles as `mapbox`, mountains 800 km high in GL JS's expected images too.
Load errors and timeouts are failures, not skips; so is anything the renderer
silently leaves out.

**Operations** apply once the map has settled (idle, or loaded in a test with
fades), as GL JS applies them after `load`: `["wait"]` and `idle` settle again;
`["wait", ms]` renders until a frame lands exactly `ms` after the one before it;
`setZoom`, `setCenter`, `setBearing` and `setPitch` move the camera; every other
call goes through `Map::edit` as a one-call transaction, the same command parser
the web package's `edit` uses. Warnings an edit raises skip the test as the
initial style's would. GL JS's harness draws once when a wait's timer fires and
reads that frame, so a fading or transitioning test's closing frames keep the
last frame's time: a transition is read where its waits leave it however long
frames take. The Chrome runner waits on the wall clock, as GL JS's harness does,
so a slow frame there can still move a transition along.

**Assets.** A fetch wrapper maps `local://` and GL JS's test-server origin
`http://localhost:2900/` to the mirror's `assets/`, including URLs that TileJSON
returns, without rewriting styles.

## GL JS's query suite

`--suite query` reads `fixtures/gl-js/query`. Each style settles as a render
test does, then `queryRenderedFeatures` runs at the test's point or box, and the
features are compared with `expected.json` the way GL JS's query harness
compares them: without `layer`, coordinates within 1e-6 degrees. A test passes
when the same features come back; whether they come in GL JS's order is reported
separately, and so are failures that differ only in geometry. It runs natively
and in Chrome.

Those geometry-only failures are overzoomed polygons. GL JS slices an overzoomed
tile into its children and clips each polygon to the child's buffer; then it
re-encodes the slice, and `vt-pbf` truncates each coordinate delta, so every
vertex after a clip point moves by up to one unit. Orbis's slices keep a polygon
its data tile holds whole unclipped and answer with it whole, so these answers
can't match GL JS's to 1e-6 degrees. The same shift decides
`line-width-features-in/tilt-inside`: GL JS's copy of the border Burlington
shares with Ocean moves one unit (half a pixel) west, off the query box, so only
Ocean answers.

## Native's suites

The `native-*` suites read `fixtures/native/` and follow Native's render-test
runner (`render-test/` at the pinned commit) where its rules change outcomes.
Test ids are Native's (`render-tests/…`); its manifest (`manifest.json`,
upstream's `macos-webgpu-wgpu.json`: Native's WebGPU backend on wgpu on macOS,
the configuration closest to Orbis's) gives the platform expectations and ignore
lists.

- **Metadata** (`src/native.rs`): 512 × 512, pixelRatio 1, `allowed` 0.00015;
  `mapMode: "tile"` draws one 512-pixel tile per tile the style's camera covers
  at its zoom (`Map::render_tile`) and lays them side by side. Keys Native's
  parser doesn't read (`threshold`, `fadeDuration`, `timeout`) are ignored as it
  ignores them. `debug`, `collisionDebug` and `showOverdrawInspector` turn their
  overlays on whenever the key is there, so those render tests skip; query
  answers don't depend on them. An ignore-list entry whose reason starts with
  `skip` skips as `native:skip`; the other ignored tests run, and the reports
  count them apart (Native fails them itself).
- **Rendering**: the map has no fades and no local ideograph font, as the
  runner's frontend. Operations apply to the loaded style before the first
  frame; `wait` and the feature-state operations draw a still first
  (`Map::snapshot`, Native's static map mode), and the picture is a still.
  `addImage` and `updateImage` load `mapbox://render-tests/<path>` from the
  cache and replace an image of that name. `source-replace` reruns a passing
  test with each pair of sources swapped. Operations Native's runner rejects
  skip as `native-runner:<name>`; ones Orbis can't apply (level-of-detail
  tuning) as `operation:<name>`.
- **Requests** come from the mirror's `assets/` only, as the runner reads its
  cache only: nothing goes to the network.
- **Comparison** (`src/pixelmatch_cpp.rs`): the pixelmatch Native links,
  pixelmatch-cpp at `61f433cb` (ISC;
  [`LICENSE-pixelmatch-cpp`](LICENSE-pixelmatch-cpp)): YIQ distance, alpha
  blended with white, the 2009 anti-aliasing detector, on premultiplied pixels
  (expected PNGs premultiplied as Native's `decodeImage` does) at threshold
  0.1285 whatever the test says. The expected images are those of the last of
  the test directory and the manifest's expectation directories that has any; a
  test passes when one of them is within `allowed`. Warnings and load errors
  stay in the report without failing a test: Native's runner judges the picture
  alone.
- **Second opinion**: a test GL JS's mirror shares is compared with GL JS's
  expected images too, by the same metric. Each test's `gl_js` entry says
  whether the two engines' images agree, whether the styles are the same, and
  whether Orbis's image matches GL JS's; the summary counts, among the tests
  whose references disagree, which engine Orbis matches.
- **Query suite**: each test renders as a render test does, draws the settled
  map as one interactive frame (queries read the last one), then queries its
  point or box. The answer must equal `expected.json` in order, numbers to six
  significant figures (Native's `stripPrecision` and `deepEqual`); failures that
  differ only in order are reported.
- **Expression suite**: each `test.json` through orb-expr's conformance runner
  (`orb_expr::conformance`, the `conformance` feature) with Native's
  expectations: evaluation errors are the bare message, and formatted sections
  predate `verticalAlign`. Native's runner also checks each expression's
  serialization and a round trip through it; Orbis doesn't serialize
  expressions, so those checks are left out, as they are for the style-spec
  suite in `cargo test -p orb-expr`.

## Platforms

- **native**: the host's wgpu backend (Metal, Vulkan or D3D12).
- **chrome**: builds `core.wasm`, then starts one Node and Playwright process
  (`web/scripts/render-fixtures.mjs`) that speaks line-oriented JSON. Node
  serves the demo build and the mirror's assets over loopback and opens
  `web/demo/fixture.html` in headless Chromium with animation frames unpaced.
  The page draws each fixture through the `orbis` package as a host would and
  returns the canvas's premultiplied RGBA from
  `readPixels({ premultiplied: true })`, which Rust unpremultiplies as it does
  the native readback. Videos stay on their first frame. A failed render reloads
  the page. Discovery, metadata, scope, expected-image selection and pixelmatch
  stay in Rust. Install the tools with `mise install`, `pnpm -C web install` and
  `pnpm -C web exec playwright install chromium`.
- **lavapipe** (render suites): builds
  [`lavapipe/Dockerfile`](lavapipe/Dockerfile) and runs the native runner under
  Linux on Mesa's software Vulkan, with no display or GPU. The image pins the
  workspace's Rust toolchain on Debian slim and limits the Vulkan loader to
  lavapipe's ICD. The checkout is mounted read-only except for the report
  directory; the named volumes `orbis-rt-target`, `orbis-rt-registry` and
  `orbis-rt-git` cache Linux builds and downloads. Docker must be running.
- **all**: every platform above.

Chrome always runs native too, for an **identity** check: for every native pass,
the native and Chrome images are compared with the suite's comparison and that
fixture's `threshold` and `allowed`, and reported as `match`, `different` or
`unavailable`. Each mismatch writes `identity/<test>/{native,chrome,diff}.png`.
Identity is separate from Chrome's own pass or fail.

For Native's suites the page applies operations as Native's runner does and
returns `snapshot()`'s pixels, which it gets straight and orb-fixtures
premultiplies again; tile mode skips there (`host:tile-mode`).

## Reports

Results go to `docs/scratch/rt/` (gitignored). `summary.json` combines the
selected platforms and the identity report; each platform also writes
`<platform>/summary.json` with each test's effective metadata, status,
diagnostics, skip reasons, differing-pixel fraction, frames, tiles drawn and
time, plus aggregates and the suite's wall time. Every rendered test saves
`<platform>/images/<test>/actual.png`; failures also save
`<platform>/failures/<test>/{actual,diff}.png`. The query suite writes under
`query/<platform>/`, Native's suites under `native-render/`,
`native-query/<platform>/` and `native-expression/`; their summaries add
`native_ignored` (how the tests Native ignores fared) and, for `native-render`,
`gl_js` (the second opinion). A platform's directory is replaced on its next
run.

Pass rates are pass / (pass + fail). A test with several skip reasons counts
once per reason, so skip-reason counts can sum to more than the skip total.
Fixture failures exit zero; malformed inputs, missing expected images and
initialization, transport or report I/O errors exit nonzero. `cargo xtask check`
runs neither Chromium nor Docker.

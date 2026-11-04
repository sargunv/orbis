# 0012: Immutable web demo assets

GitHub Pages caches every response for 600 seconds, including HTML, and
does not let this project change the response headers. Fixed JS, worker
and wasm URLs let a fresh page run old code or pair new glue with old wasm.

Use Vite 8 (rolldown) to build the three HTML entry points from `web/demo/`
into `web/dist/`, with `/orbis/` as the deployment base. All external runtime
assets get content-hashed names. The wasm-bindgen glue and shared modules
are bundled into hashed JS; module workers use Vite's `new Worker(new
URL(..., import.meta.url))` handling. Wasm-bindgen's default wasm URL and
the page's explicit `compileStreaming` URL both pass through Vite's asset
handling. The page still compiles core once and posts the module to workers;
render/job workers still communicate through transferred MessagePorts.

Hash the PMTiles archives too. A validated header establishes format and
offset bounds, but does not establish that a cached, valid archive contains
the data intended by a new style. A small build plugin emits each archive
through Vite, rewrites the style's `pmtiles://` URL to that hashed asset,
and emits the rewritten JSON as another hashed asset. Its root-relative
URL works from both the HTML and workers in `assets/`. Changing archive
content therefore changes the archive URL, style hash and importing JS.
The committed styles remain usable by native hosts. Development serves
the original styles and archives through Vite; production uses the hashes.

Publishing overlays `dist/` onto `gh-pages` and retains old assets. Cached
HTML and already open tabs can request dependencies after a new publish;
deleting the previous build would turn a cache mismatch into a 404. Existing
fixed-name assets are frozen during the migration for old HTML. Retention
is deliberate and currently has no automatic pruning: archive/wasm changes
grow the published branch, while unchanged content reuses the same names.

The snapshot and bench scripts build and serve `dist/` with Range support.
The fixture driver serves the same generated bindings directly because it
uses exports the demo bundle removes. No production isolation headers or
shared memory are needed. The pre-existing optional precise bench mode
continues to use local isolation for timer precision only.

Verification: all four main/worker × MVT/MLT snapshots have zero differing
pixels against the pre-Vite build. Chrome fixtures retain 352 passes, six
existing pixel failures and 525 skips; all 352 native passes match Chrome.
The build regression test changes an archive byte and checks that archive,
style and importing JS hashes all change.

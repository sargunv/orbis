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

Publishing deploys `dist/` as the whole site from a GitHub Actions workflow
on every push to `main` (`.github/workflows/pages.yml`, running `cargo xtask
pages`); previous builds' assets are not retained. An earlier version
overlaid each build onto a `gh-pages` branch and kept every old asset, so
cached HTML and open tabs could still fetch theirs; that branch is gone.
Retention is dropped because hashing already bounds what can go missing. A
deploy removes only assets whose content changed. Startup assets (JS, wasm,
workers) are fetched with their HTML and expire with it, so cached HTML is
paired with its own cached code. What an open tab fetches later (archive
ranges, the lazy shaper and word data) changes
rarely and keeps its URL when unchanged. When one of those did change, the
open tab's next fetch of it 404s and a reload recovers. That is acceptable
for a demo, and avoids the alternatives: carrying old assets in an Actions
cache makes each deploy depend on cache state, and the branch grew without
bound.

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

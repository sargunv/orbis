# Fixtures

Read-only mirrors of the reference engines' test suites and of the C API Orbis
implements, fetched, not committed:

```sh
cargo xtask fixtures            # all
cargo xtask fixtures gl-js      # or one
cargo xtask fixtures native
cargo xtask fixtures mln-ffi
```

- `fixtures/gl-js/`: the
  [MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js) render and query
  tests and the assets their styles reference.
- `fixtures/native/`: the
  [MapLibre Native](https://github.com/maplibre/maplibre-native) render, query
  and expression tests (`metrics/integration/`), the platform expectations and
  ignore lists its macOS WebGPU manifest names, and the resources of its
  render-test cache as files.
- `fixtures/mln-ffi/`: the C API's headers and C ABI suite from
  [mln-rs](https://github.com/maplibre/mln-rs), which owns them, with Unity and
  the Vulkan headers the suite builds with.

Each mirror is built from the tarball of a pinned commit and checked against a
pinned content hash: SHA-256 over every file's path, length and bytes, in path
order, `SOURCE.md` left out. The task checks that every tarball entry sits under
that commit's root directory, builds the mirror in `fixtures/<name>.partial/`,
and moves it into place only if the hash matches. When a mirror already matches,
the task only re-hashes it (well under a second), so `cargo xtask rt` runs it
first. Each fetch writes `fixtures/<name>/SOURCE.md`, which lists what the
mirror contains. Don't edit a mirror by hand; the next fetch replaces one that
differs from its pin. To move a pin, change its `COMMIT`, fetch, review the
mirror left in `fixtures/<name>.partial/`, and set its `MIRROR_SHA256` to the
hash the fetch reports.

## GL JS

Pinned in `xtask/src/fixtures.rs` (`COMMIT`, currently `2b68611a`). The task
downloads about 160 MiB and writes about 131 MiB: every render and query test
(1736 and 137), and the assets their styles reference through `local://` URLs,
including the `@fontsource` files GL JS's test server serves from npm, checked
against the `integrity` hash in GL JS's `package-lock.json`.

## MapLibre Native

Pinned in `xtask/src/native_fixtures.rs` (`COMMIT`, currently `8e144d0e`,
2026-11-02 master: the commit mln-rs grades against). The task downloads about
190 MiB and writes about 78 MiB: every test of the three suites, the manifest
`metrics/macos-webgpu-wgpu.json` (as `manifest.json`) with the expectation
directories and ignore lists it names, and `assets/`. Native's runner reads
every request from `metrics/cache-style.db` (offline, cache only), not from the
files beside the tests, so the task reads that SQLite database and writes each
resource and tile at a path its URL gives: `http://<path>`, the runner's
spelling of `local://`, at `assets/<path>`; `https://` and `mapbox://` URLs
under `assets/https/` and `assets/mapbox/`.

## The C API

Pinned in `xtask/src/mln_ffi.rs` (`COMMIT`, currently `e0c2a22b`, where mln-rs
took maplibre-native-ffi `eae96a42`'s API). The task downloads mln-rs's, Unity's
and the Vulkan headers' tarballs and writes about 4 MiB: `crates/mln_c/include`
as `include/`, and the suite and its GPU fixtures from
`tools/c-abi-test/testdata`. Moving the pin moves the API: run
`cargo xtask mln-ffi declarations` after it to regenerate `orb-ffi`'s
declarations and stubs.

## What reads them

Only `cargo xtask rt` (`orb-fixtures`), `web/scripts/render-fixtures.mjs`, which
it drives, and `cargo xtask mln-ffi`. `cargo test --workspace` runs offline from
a fresh clone, on small inputs committed next to each crate (see
[THIRD_PARTY.md](../THIRD_PARTY.md)).

## Licences

The mirrors are not part of this repository's distribution. Their files keep
their upstream terms: GL JS's tests are BSD-3-Clause, and Native's and the C
API's BSD-2-Clause; some asset families (OpenStreetMap, JAXA, GEBCO and
Mapterhorn tiles, Mapbox tiles, styles and imagery, the Arial Unicode MS glyphs
and some sprites) carry their own or unclear terms upstream.

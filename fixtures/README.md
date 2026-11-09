# Fixtures

Two read-only mirrors of the reference engines' test suites, fetched, not
committed:

```sh
cargo xtask fixtures            # both
cargo xtask fixtures gl-js      # or one
cargo xtask fixtures native
```

- `fixtures/gl-js/`: the [MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js)
  render and query tests and the assets their styles reference.
- `fixtures/native/`: the [MapLibre Native](https://github.com/maplibre/maplibre-native)
  render, query and expression tests (`metrics/integration/`), the platform
  expectations and ignore lists its macOS WebGPU manifest names, and the
  resources of its render-test cache as files.

Each mirror is built from the tarball of a pinned commit and checked
against a pinned content hash: SHA-256 over every file's path, length and
bytes, in path order, `SOURCE.md` left out. The task checks that every
tarball entry sits under that commit's root directory, builds the mirror in
`fixtures/<name>.partial/`, and moves it into place only if the hash
matches. When a mirror already matches, the task only re-hashes it (well
under a second), so `cargo xtask rt` runs it first. Each fetch writes
`fixtures/<name>/SOURCE.md`, which lists what the mirror contains. Don't
edit a mirror by hand; the next fetch replaces one that differs from its
pin. To move a pin, change its `COMMIT`, fetch, review the mirror left in
`fixtures/<name>.partial/`, and set its `MIRROR_SHA256` to the hash the
fetch reports.

## GL JS

Pinned in `xtask/src/fixtures.rs` (`COMMIT`, currently `2b68611a`). The
task downloads about 160 MiB and writes about 95 MiB: the render and query
test directories whose names start with the prefixes listed there, and the
assets their styles reference through `local://` URLs, including the
`@fontsource` files GL JS's test server serves from npm, checked against the
`integrity` hash in GL JS's `package-lock.json`.

## MapLibre Native

Pinned in `xtask/src/native_fixtures.rs` (`COMMIT`, currently `8e144d0e`,
2026-11-02 master: the commit mln-rs grades against). The task downloads
about 190 MiB and writes about 78 MiB: every test of the three suites, the
manifest `metrics/macos-webgpu-wgpu.json` (as `manifest.json`) with the
expectation directories and ignore lists it names, and `assets/`. Native's
runner reads every request from `metrics/cache-style.db` (offline, cache
only), not from the files beside the tests, so the task reads that SQLite
database and writes each resource and tile at a path its URL gives:
`http://<path>`, the runner's spelling of `local://`, at `assets/<path>`;
`https://` and `mapbox://` URLs under `assets/https/` and `assets/mapbox/`.

## What reads them

Only `cargo xtask rt` (`orb-fixtures`) and `web/scripts/render-fixtures.mjs`,
which it drives. `cargo test --workspace` runs offline from a fresh clone, on
small inputs committed next to each crate (see [THIRD_PARTY.md](../THIRD_PARTY.md)).

## Licences

The mirrors are not part of this repository's distribution. Their files keep
their upstream terms: GL JS's tests are BSD-3-Clause and Native's
BSD-2-Clause; some asset families (OpenStreetMap, JAXA, GEBCO and Mapterhorn
tiles, Mapbox tiles, styles and imagery, the Arial Unicode MS glyphs and some
sprites) carry their own or unclear terms upstream.

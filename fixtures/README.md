# Fixtures

`fixtures/gl-js/` holds a read-only mirror of the
[MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js) render and query
tests and the assets their styles reference. It is fetched, not committed:

```sh
cargo xtask fixtures
```

The task downloads the tarball of the commit pinned in
`xtask/src/fixtures.rs` (`COMMIT`, currently `2b68611a`), about 160 MiB, and
writes the mirror, about 80 MiB, without unpacking the rest. It checks that
every tarball entry sits under that commit's root directory and that the
`@fontsource` package it adds matches the `integrity` hash in GL JS's
`package-lock.json`. Then it hashes
the result (`MIRROR_SHA256`: SHA-256 over every file's path, length and
bytes, in path order) and moves it into place only if the hash matches the
pin. When `fixtures/gl-js/` already matches, the task only re-hashes it
(well under a second), so `cargo xtask rt` runs it first.

`fixtures/gl-js/SOURCE.md`, written by each fetch, lists what the mirror
contains: the test prefixes, counts per prefix and any `local://` references
that name nothing upstream. Don't edit the mirror by hand; the next fetch
replaces a mirror that differs from the pin. To move the pin, change `COMMIT`,
run `cargo xtask fixtures`, review the mirror it leaves in
`fixtures/gl-js.partial/`, and set `MIRROR_SHA256` to the hash it reports.

## What reads it

Only `cargo xtask rt` (`orb-fixtures`: the render suite natively, in Chrome
and on lavapipe; the query suite natively and in Chrome) and `web/scripts/render-fixtures.mjs`, which it drives.
`cargo test --workspace` runs offline from a fresh clone, on small inputs
committed next to each crate (see [THIRD_PARTY.md](../THIRD_PARTY.md)).

## Licences

The mirror is not part of this repository's distribution. Its files keep
their upstream terms: GL JS's tests are BSD-3-Clause; some asset families
(OpenStreetMap, JAXA, GEBCO and Mapterhorn tiles, Mapbox tiles and imagery,
the Arial Unicode MS glyphs and some sprites) carry their own or unclear terms
upstream.

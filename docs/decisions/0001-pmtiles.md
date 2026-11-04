# 0001: PMTiles v3 and a reproducible MVT/MLT demo

Use a small in-house PMTiles reader in `orb-source`, driven by host range reads
through `orb-runtime::TileLoader`. Archive parsing is pure: `Reader::lookup`
returns a directory range, a tile range/codec/type, or a missing tile;
`Reader::insert` accepts the resulting header/directory bytes. It understands
v3 Hilbert tile ids, run lengths, four directory levels, gzip and uncompressed
MVT/MLT tiles. It explicitly rejects unsupported versions, tile types and codecs.

This refines the `pmtiles-rs` choice in plan §10.1 without changing the crate
graph or service boundaries. Owning these few parsing operations gives us
explicit size/offset validation and no archive-library async runtime, HTTP
client or platform dependencies. `flate2` with its default pure Rust
`miniz_oxide` backend is already in our dependency graph. It supports wasm32,
validates gzip framing/checksums, and is bounded during inflation. Avoid a
second codec implementation or JavaScript-only decompression path.

`Fetch::get_range` is a required host service. HTTP ranges must return 206 and
a matching Content-Range; a 200 response is rejected before its body is read.
Bodies stream within the requested length, which must be 1..=64 MiB. Native
file ranges use seek/read. Inflation is also capped at 64 MiB. Root directories
fit in the first 16 KiB; decoded directory entries are capped at 8 MiB. A reader
keeps its root and at most 32 leaf directories within an 8 MiB leaf cache. The
runtime keeps at most eight archive readers, and each browser job worker owns
its own loader and caches. Tile payloads are not retained in that cache.

`TileRequest::location` distinguishes ordinary tile URLs from PMTiles archive
URLs; its existing tile coordinate identifies the tile inside an archive.
PoolJobs and browser WorkerJobs both use TileLoader, with the archive kind
carried explicitly over the worker protocol. Header zooms, bounds and type
resolve sources with TileJSON precedence. Source metadata and tile generations
are separate so a renderer rebuild cannot discard pending source resolution.

Commit two small central Tokyo archives (5,660,823 bytes combined) rather than
making tests depend on a moving planet archive. `cargo xtask extract` selects
11 tiles using 18 range reads from the pinned OpenFreeMap planet, downloads
2,941,370 bytes, and creates both archives. Attribution, licences and exact
coordinates live beside the archives in web/demo/data. The default demo uses
reduced OpenFreeMap Liberty; native `--format mvt|mlt` and web `?format=mvt|mlt`
select the same data. Pages deployment includes these public data assets.

For exact MVT/MLT columns, the high-level mlt-core encoder is insufficient:
it narrows positive signed integer columns, e.g. aerodrome_label.ele, to
unsigned, and stringifies mixed scalar columns. Keep the runtime decoder's
wire types intact. The extraction tool uses mlt-core's typed staging API
(exact `=0.22.1`, tool-only `__private` feature), preserving types, geometry,
order and F64 precision and refusing mixed scalar columns. This confines an
upstream API stability constraint to reproducible tooling. A future mlt-core
upgrade must re-run the full extract parity check before changing that pin.

All 11 full TileFrames are equal. Native MVT/MLT snapshots and all four web
format/thread combinations have zero differing pixels at the default view,
including web versus native with the line renderer.
On this Apple Metal host, the release all-tile decode pass measured 55.889 ms
MVT / 56.282 ms MLT; these one-pass timings exclude gzip and are descriptive,
not a speed gate. core.wasm measured 971 KiB raw / 320 KiB brotli with the line and
GeoJSON work. The first native MVT snapshot took
363 ms and the immediately following MLT snapshot took 62 ms, including
loading/render/readback; pipeline warming makes these unsuitable as a format
speed comparison. Each final frame had 75 draws, one dispatch and 252 candidates.
Browser worker mode uploaded 1,348.5 KiB directly from JS buffers and copied
8.4 KiB of job headers into the render worker's wasm memory; the packed
payload stayed in JS.

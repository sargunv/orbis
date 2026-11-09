# Central Tokyo PMTiles demo

The two committed archives contain the same 21 OpenMapTiles tiles, every tile
at zooms 0–14 that touches bounds `[139.725, 35.677, 139.755, 35.697]`, so the
area draws from the whole world down to street level. The initial
view is `139.74,35.685,z15`, by the Imperial Palace in central Tokyo.
Together the archives are 9,139,857 bytes. No network connection is needed
for the default native demo after checkout, or for web tile data, glyphs and
sprites once served.

- MVT: **4,695,260 bytes**, original uncompressed tile payloads, repacked with gzip.
- MLT: **4,444,597 bytes**, encoded from those MVT payloads with `mlt-core 0.22.1`.
- Source: [OpenFreeMap planet build `20261101_080013_pt`](https://btrfs.openfreemap.com/areas/planet/20261101_080013_pt/tiles.pmtiles).
  The planet is 87,352,499,305 bytes; extraction downloaded 4,890,744 bytes in
  31 validated HTTP range requests. The upstream archive is periodically retired;
  the committed extracts keep the demo and tests independent of its availability.
- Data: **© OpenStreetMap contributors**, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/),
  [attribution and licence](https://www.openstreetmap.org/copyright).
  Tiles supplied by [OpenFreeMap](https://openfreemap.org/), using
  [OpenMapTiles](https://openmaptiles.org/). These extracts are distributed under
  ODbL 1.0; the engine's code licence does not apply to the map data.
- Style: OpenFreeMap Liberty, fetched from
  <https://tiles.openfreemap.org/styles/liberty>, reduced to
  background/fill/line layers without patterns, dashes or gradients, with
  source/camera replaced; `cargo xtask demo-data` then adds Liberty's
  symbol layers, in Liberty's order. It inherits OSM Liberty's BSD code and
  CC BY design licences; see [STYLE-LICENSE.md](STYLE-LICENSE.md).
  Upstream style SHA-256: `408245464e76791801a64393c403433d09e8110eb9516040f8dc2b62584eee2a`.
- `alps.json` and `alps-flight.json`: the terrain entry (`?style=alps&flight`
  on the web, `--flight` natively), written for this
  repository. It draws OpenFreeMap's planet tiles (data as above) over
  [Mapterhorn](https://mapterhorn.com/)'s terrain tiles, fetched at run time
  and not committed: Terrarium-encoded WebP from open elevation data,
  around the Alps mainly Copernicus GLO-30 (© DLR e.V. 2010–2014 and
  © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by
  the European Union and ESA), swisstopo's swissALTI3D (open government
  data) and Austria's BEV ALS-DGM (CC BY 4.0); every source and licence is
  in [Mapterhorn's attribution list](https://mapterhorn.com/attribution).
  The flight's legs are `flyTo` options looped until the user takes the
  camera.
- `osm-carto.json`: OpenStreetMap's standard raster tiles from
  `tile.openstreetmap.org`, under its [tile usage policy](https://operations.osmfoundation.org/policies/tiles/):
  attributed, fetched only as the map shows them, with the browser's
  referrer. Written for this repository.
- `playground.json`: a made-up Tokyo district drawn with hover state, the
  style editor's end-to-end check (`pnpm -C web e2e --only=editor`) and
  `orb-runtime`'s edit tests draw; written for this repository. No picker
  entry shows it.
- Labels: `sprites/` is OpenFreeMap's Liberty sprite (`ofm_f384/ofm`, 1× and
  2×; icons as in [STYLE-LICENSE.md](STYLE-LICENSE.md)). `glyphs/` holds
  OpenFreeMap's Noto Sans Regular, Bold and Italic glyph PBFs, cut to the
  glyphs the extract's names and refs use in the layers each font labels
  (complex scripts and ideographs draw from local fonts, as GL JS's default
  `localIdeographFontFamily` has them, and are left out): about 1.5 MB. Noto Sans is © Google, under the
  [SIL Open Font License 1.1](https://openfontlicense.org/).

Reproduce from the repository root:

```sh
cargo xtask demo-data   # HTTP ranges → both archives and manifest; Liberty's labels, sprite and glyph subsets
cargo test -p xtask     # the MLT archive decodes to the same columns as the MVT one
cargo run --release -p orb-cli -- render web/demo/data/tokyo-mvt.json -o /tmp/tokyo-mvt.png --at 139.74,35.685,15
cargo run --release -p orb-cli -- render web/demo/data/tokyo-mlt.json -o /tmp/tokyo-mlt.png --at 139.74,35.685,15
cargo xtask web
pnpm -C web snapshot /tmp/tokyo-web-mvt.png --format=mvt
pnpm -C web snapshot /tmp/tokyo-web-mlt.png --format=mlt \
  --compare=/tmp/tokyo-web-mvt-map.png
```

The native demo defaults to this MVT extract; `--format mlt` selects MLT.
An explicit style URL still loads that style. The styles' tile, glyph and
sprite URLs are relative to the style file. The web demo accepts
`?format=mvt|mlt` alongside an optional `style` URL.
Run `pnpm -C web dev` after `cargo xtask web` for local work at `/orbis/`.
`pnpm -C web build` writes the production site to `web/dist/`; its styles
reference content-hashed archives, so new and cached styles cannot select
different generations of the same archive URL. Serve that directory at
`/orbis/` with HTTP Range support (and expose Content-Range for cross-origin
archives). The snapshot script builds and serves that production layout. The script
writes both a screenshot with the HUD and a `-map.png` without overlays;
`--compare` compares decoded map pixels exactly and fails on any difference.

`cargo test -p xtask` decodes every manifest tile from both archives and
checks that their layers are equal.

The MLT tool preserves feature order, geometry, numeric signedness and float
precision. The high-level encoder narrows positive signed values to unsigned;
we use its typed staging API, currently exposed by its `__private` feature,
**only in xtask**, with an exact version pin. No private encoder code is linked
into core.wasm. Mixed scalar columns are rejected because MLT v1 cannot retain
them; they are never silently stringified. The real extract has none.

## Other coordinate reference systems (`crs/`)

`arctic.json` is a v8 style whose root `projection` names a coordinate
reference system and whose sources name their `tileMatrixSet`, written
for this repository; the web demo opens it as `?style=arctic`, the native
demo as `file://$PWD/web/demo/data/crs/arctic.json`. It draws on an
EPSG:3413 (NSIDC polar stereographic) surface:
[NASA EOSDIS GIBS](https://earthdata.nasa.gov/gibs) Blue Marble shaded relief,
MEaSUREs Greenland ice velocity, coastlines and reference labels in GIBS's
EPSG:3413 tile matrix sets (declared inline: 512-pixel tiles from
±4,194,304 m), with a graticule from inline GeoJSON tiled on the surface,
and OpenFreeMap's planet tiles' boundaries and labels, reprojected from
Web Mercator (the `GpuPerVertex` plan).

`rd-new.json`, written for this repository and not in the web demo's
list, draws on the Dutch grid, Amersfoort / RD New (EPSG:28992, on the
Bessel datum): [PDOK](https://www.pdok.nl/)'s grey BRT Achtergrondkaart
(© Kadaster, CC BY 4.0) in its own tile matrix set, under OpenFreeMap's
roads and building outlines reprojected from Web Mercator, which land on
the Kadaster's buildings to within a metre or so where the datum shift is
right (and 100 m off where it isn't). Open it natively or with
`cargo run --release -p orb-cli -- render web/demo/data/crs/rd-new.json -o rd.png --at 5.3872,52.1552,16`.

`gibs-geographic.json`, likewise, draws GIBS Blue Marble shaded relief on
a geographic (EPSG:4326) plane from its 500 m tile matrix set exactly as
GIBS's capabilities list it: 288° tiles at level 0, whose matrices (2×1,
3×2, 5×3, then 10×5 on) hold only the tiles that reach the world. The
level 0–2 tiles reach past the antimeridian and the South Pole into
black, which stays off the map. Open it with
`cargo run --release -p orb-cli -- render web/demo/data/crs/gibs-geographic.json -o gibs.png --at 0,0,0 --size 900x500`.

They fetch their tiles at run time; nothing here is extracted or committed.
GIBS imagery is NASA's and free of use restrictions; GIBS asks to be credited.
OpenFreeMap's tiles and glyphs are as above (© OpenMapTiles, © OpenStreetMap
contributors, ODbL).

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
- `alps.json` and `alps-flight.json`: the terrain demo (`?style=alps&flight`
  on the web, `--flight` natively; decision 0019), written for this
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
- `alps-sunlit.json`: a scene document over `alps.json` that lights it by a
  morning sun over Zermatt and an ambient sky (decision 0026), written for
  this repository; `?style=alps-sunlit&flight`.
- `manhattan.json`: a scene document over OpenFreeMap's Liberty (data as
  above): Midtown lit by an October afternoon sun that casts shadows
  (decision 0046), labels hidden behind buildings, written for this
  repository; `?style=manhattan`.
- `tokyo-landmarks.json` and `models/`: a scene document over OpenFreeMap's
  Liberty (data as above): Shiba Park lit by a November afternoon sun, with
  three glTF landmarks (a lattice tower, a torii and a pagoda) that the
  demo's entry places at GeoJSON points in a `model` layer (decision 0047);
  `?style=landmarks`. The models are built from boxes, frusta and prisms by
  `cargo xtask demo-data models`; both are written for this repository.
- `scripts.json`: the Cairo, Delhi and Bangkok entries' style
  (`?style=cairo|delhi|bangkok`, or `scripts`), written for this repository. It draws OpenFreeMap's
  planet tiles with labels in Arabic, Devanagari, Thai and Latin from
  `font-faces` files: Noto Sans subsets from `@fontsource` 5.3.0 on
  jsDelivr (© Google, [SIL Open Font License 1.1](https://openfontlicense.org/)),
  fetched at run time and not committed.
- `osm-carto.json`: OpenStreetMap's standard raster tiles from
  `tile.openstreetmap.org`, under its [tile usage policy](https://operations.osmfoundation.org/policies/tiles/):
  attributed, fetched only as the map shows them, with the browser's
  referrer. Written for this repository.
- `playground.json`: the style editor's sample, a made-up Tokyo district
  drawn with hover state (`?style=playground`); written for this repository.
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

`cargo xtask check` includes full TileFrame equality for every manifest tile
and a native offscreen pixel comparison at the default view. Timings from
`tile-parity` include decode into TileFrame, excluding range reads and gzip.

The MLT tool preserves feature order, geometry, numeric signedness and float
precision. The high-level encoder narrows positive signed values to unsigned;
we use its typed staging API, currently exposed by its `__private` feature,
**only in xtask**, with an exact version pin. No private encoder code is linked
into core.wasm. Mixed scalar columns are rejected because MLT v1 cannot retain
them; they are never silently stringified. The real extract has none.

## Other projections (`crs/`)

Scene documents (plan §11.4) with their v8 styles inline, written for this
repository; the web demo opens them as `?style=arctic`, `?style=nz`,
`?style=equal-earth`, `?style=blue-marble` and `?style=greenland`, the
native demo as
`file://$PWD/web/demo/data/crs/arctic.json`.

- `arctic.json` draws on an EPSG:3413 (NSIDC polar stereographic) surface:
  [NASA EOSDIS GIBS](https://earthdata.nasa.gov/gibs) Blue Marble shaded relief,
  MEaSUREs Greenland ice velocity, coastlines and reference labels in GIBS's
  EPSG:3413 tile matrix sets (declared inline: 512-pixel tiles from
  ±4,194,304 m), with a graticule from inline GeoJSON tiled on the surface,
  and OpenFreeMap's planet tiles' boundaries and labels, reprojected from
  Web Mercator (the `GpuPerVertex` plan).
- `nz.json` draws OpenFreeMap's planet tiles on an NZTM2000 (EPSG:2193)
  surface, reprojected from Web Mercator.
- `blue-marble.json` draws a v8 globe: GIBS Blue Marble shaded relief from
  its EPSG:4326 tiles (declared inline from level 3, where its matrices
  become a quadtree of 36° tiles), to the poles, with OpenFreeMap's planet
  tiles' boundaries and labels.
- `equal-earth.json` draws OpenFreeMap's Liberty style (by URL) on an Equal
  Earth (EPSG:8857) world map, reprojected from Web Mercator.
- `greenland.json` keeps Web Mercator: OpenFreeMap's planet tiles under the
  same ice velocity raster, warped from EPSG:3413 (the `RasterWarp` plan).

They fetch their tiles at run time; nothing here is extracted or committed.
GIBS imagery is NASA's and free of use restrictions; GIBS asks to be credited.
OpenFreeMap's tiles and glyphs are as above (© OpenMapTiles, © OpenStreetMap
contributors, ODbL).

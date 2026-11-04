# GL JS golden vectors

Expected results for `orb-symbol`, computed by GL JS's own functions at the
pinned commit (`xtask/src/fixtures.rs`, `2b68611a`). License: GL JS is
BSD-3-Clause (`crates/orb-text/tests/data/LICENSE-gl-js.txt`).

- `leaves.json`: the line-by-line leaves. `clipLine`, `checkMaxAngle`,
  `getAnchors`, `getCenterAnchor`, `mergeLines`,
  `findPoleOfInaccessibility`, `shapeIcon`, `fitIconToText`,
  `applyTextFit`, `getIconQuads`, `CollisionFeature` and
  `evaluateVariableOffset`, on seeded pseudo-random inputs plus the inputs
  of GL JS's own unit tests.
- `layout-<name>.json`: whole layouts. Each case in `layout-cases.json` names
  a tile under `tiles/`, a source layer and a symbol layer's `layout`. The
  generator builds a `SymbolStyleLayer` and `SymbolBucket`, populates it from
  the tile, resolves glyphs from the PBF ranges under `glyphs/`, and runs
  `performSymbolLayout`. The file records the evaluated inputs per feature
  (text sections, icon, sort key, every data-driven layout value at the
  zooms layout reads), the glyphs and atlas rectangles, and every output
  table: symbol instances, collision boxes, placed symbols, layout vertices,
  glyph offsets, line vertices, anchor offsets and sort-key ranges.

## Inputs

- `tiles/14-14552-6450.mvt` (20,324 bytes): tile 14/14552/6450 of
  `web/demo/data/tokyo-mvt.pmtiles` (central Tokyo, OpenMapTiles schema),
  uncompressed and cut to five layers with only the fields `class`, `rank`,
  `name`, `name_en`, `name:ru`, `ref` and `ref_length`: `place` points inside
  the tile of class `city` or `quarter`, or `neighbourhood` with "2-chome" in
  `name_en` (22); `poi` points inside the tile with `name:ru` or of class
  `bank` or `museum` (77); `transportation_name` of class `primary`,
  `secondary`, `trunk` or `motorway` (49); `transportation` of class
  `primary`, `trunk` or `motorway` (111); and `landuse` of class
  `university` or `neighbourhood` (72). Data © OpenStreetMap contributors,
  ODbL 1.0 (`web/demo/data/README.md`).
- `glyphs/Noto Sans Regular/`: OpenFreeMap's Noto Sans Regular glyph PBFs
  (`https://tiles.openfreemap.org/fonts/Noto Sans Regular/<range>.pbf`),
  each cut to the 213 code points the cases request (143 KB in all). Noto
  Sans is under the SIL Open Font License 1.1
  (`crates/orb-text/tests/data/fonts/OFL.txt`). Generating with the full
  ranges gives the same files.

The cases' layouts are the same as GL JS's own fixtures use, mapped onto the
OpenMapTiles layers: place labels on `place`, road labels on
`transportation_name`, road icons and repeated text (`mergeLines`) on
`transportation`, polygon and outline labels on `landuse`, and POI labels
on `poi`. `rank` varies sizes, sort keys and icon sizes, folded with `%` to
small values; the Cyrillic cases read `name:ru`; the vertical cases draw
real CJK glyphs.

`src/tests/golden.rs` and `src/tests/layouts.rs` feed the same inputs to
`orb-symbol` and compare every field: doubles bit for bit, and quantized
vertex values exactly. Vertex texture coordinates are compared relative to
each glyph's or image's atlas rectangle, since `orb-symbol` never sees an
atlas. Symbol keys are compared as equivalence classes, because GL JS keeps
16 bits of a text hash and Orbis keeps a 64-bit `(source-layer, id, text)`
hash.

Numbers are written so they parse back to the same double: shortest
round-trip decimals, and `"-0"`, `"NaN"`, `"Infinity"` as strings.

## Regenerating

You need Node (the `mise` pin) and a GL JS checkout at the pinned commit
with its dependencies and generated code:

```sh
gljs=$(mktemp -d)
git -C "$gljs" init -q
git -C "$gljs" fetch -q --depth 1 https://github.com/maplibre/maplibre-gl-js \
  2b68611a4e2d4f7f076f6c6ebf758e2d53f91471
git -C "$gljs" checkout -q FETCH_HEAD
(cd "$gljs" && npm ci --ignore-scripts && npm run codegen)
crates/orb-symbol/tests/data/generate.sh "$gljs"
```

`generate.sh` copies `generate.ts` into the checkout, bundles it with GL
JS's own rolldown, and writes the JSON files here. Running it twice gives
identical files. To add a whole-layout case, add it to `layout-cases.json`,
regenerate, and list it in `src/tests/layouts.rs`.

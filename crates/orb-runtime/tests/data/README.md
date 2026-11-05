# Runtime test data

`src/testing.rs` serves this directory for `local://` URLs.

- `tiles/14-14552-6450.mvt`: the `place` layer of tile 14/14552/6450 from
  `web/demo/data/tokyo-mvt.pmtiles`, central Tokyo, uncompressed, keeping the
  52 features whose `class` is `city` or `quarter` and whose `name_en` is
  ASCII, with only those two fields (1,944 bytes). Labels in `name_en` then
  need only glyph range 0–255. Data © OpenStreetMap contributors, ODbL 1.0,
  from OpenFreeMap's OpenMapTiles build (`web/demo/data/README.md`).
- `glyphs/Noto Sans Regular/0-255.pbf`: unchanged from
  `https://tiles.openfreemap.org/fonts/Noto Sans Regular/0-255.pbf`. Noto Sans
  is © The Noto Project Authors, under the SIL Open Font License 1.1
  (`crates/orb-text/tests/data/fonts/OFL.txt`).
- `extrusion.json`: a style of three GeoJSON blocks around the origin with
  `fill-extrusion` heights, for the frame tests' camera-only uploads and
  scene lights (`tests/frames.rs`).

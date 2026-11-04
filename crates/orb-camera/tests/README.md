# Cover reference vectors

`data/covers.json` contains 66 **tile sets returned by GL JS itself**, at the
commit pinned in `xtask/src/fixtures.rs`. `generate-covers.mjs` constructs the
upstream `MercatorTransform` and calls upstream `coveringTiles` in Node; it
contains no copy of their camera, culling or LOD implementation. SHA-256 checks
reject changed reference implementations before generation. Neither Node nor
upstream sources are needed to run the Rust tests.

The cases cover zooms 2.2, 10.3, 14.48 and 18 at pitches 0°, 45°, 60°, 60.1°,
80° and 85°, north-up and rotated 30°. Additional cases cover 256/1024 px
source tiles, source min/max zoom, a wider FOV, antimeridian crossings, repeated
worlds, 1200 m elevated content, the close-to-ground camera at zoom 24, GL JS's
default maxPitch=60 clamp and our maxPitch=85 interaction clamp. Cameras in the
JSON are the resolved upstream cameras (after its viewport constraints).
The Rust comparison filters source minzoom just as `TilePyramid` does, and
compares sorted `(wrap, zoom, column, row)` sets, not traversal order.

Regenerate from the pinned tarball (from the repository root, Node 24):

```sh
TASK_GL=$(mktemp -d)
curl -Lf https://codeload.github.com/maplibre/maplibre-gl-js/tar.gz/2b68611a4e2d4f7f076f6c6ebf758e2d53f91471 | tar -xz --strip-components=1 -C "$TASK_GL"
# Keep upstream source untouched; only replace the package manifest to avoid
# installing its browser test/build tooling for this CPU-only reference run.
mv "$TASK_GL/package.json" "$TASK_GL/package.upstream.json"
node -e 'require("fs").writeFileSync(process.argv[1], JSON.stringify({private:true,type:"module"}))' "$TASK_GL/package.json"
npm install --prefix "$TASK_GL" --ignore-scripts --save-exact \
  tsx@4.20.6 gl-matrix@3.4.4 @mapbox/point-geometry@1.1.0 \
  @maplibre/maplibre-gl-style-spec@26.5.1 @mapbox/vector-tile@3.0.0 \
  @mapbox/tiny-sdf@2.0.7 @mapbox/unitbezier@1.0.0 \
  @unicode/unicode-17.0.0@2.1.0 regenerate@1.5.0 bidi-js@1.0.3 \
  earcut@3.0.2 kdbush@4.0.2 murmurhash-js@1.0.0 \
  potpack@2.1.0 quickselect@3.0.0 tinyqueue@3.0.0
(cd "$TASK_GL" && node build/generate-struct-arrays.ts && node build/generate-unicode-data.ts && node build/generate-style-code.ts)
"$TASK_GL/node_modules/.bin/tsx" crates/orb-camera/tests/generate-covers.mjs "$TASK_GL"
cargo test -p orb-camera
```

The generated types above are needed because upstream's transform imports its
symbol projection helpers; those helpers are never used by the cover generator.
Upstream source is BSD-3-Clause. Only the generator and numeric outputs are
committed here.

`cargo test -p orb-camera --test covers manhattan_cover_counts -- --nocapture`
prints per-zoom ideal tile counts for the live Liberty reproduction. The
maxzoom=24 counts represent the runtime's overzoomed vector ideals; maxzoom=14
also shows the canonical source-limited cover. Loaded fallback/ancestor tiles
and background placements in the demo's HUD are separate counters.

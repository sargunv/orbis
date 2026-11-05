# Decisions

One record per significant decision, `NNNN-<slug>.md`: the question, the
options, the evidence and the outcome. Only decisions that touch the plan's
architecture or that the owner should know about; routine design choices live
in code and commit messages.

- [0001](0001-pmtiles.md): PMTiles v3 and a reproducible MVT/MLT demo
- [0002](0002-vertex-pulling.md): Vertex pulling as the Modern default
- [0003](0003-wgpu-in-the-browser.md): wgpu's WebGPU backend in the browser
- [0004](0004-webgpu-in-workers.md): WebGPU in workers
- [0005](0005-expression-locale-and-conformance.md): Expression locale service and conformance
- [0006](0006-tile-publication.md): Tile fallback publication and request lifetime
- [0007](0007-image-tiles.md): Image tiles: decode, upload and DEM preparation
- [0008](0008-image-atlas.md): Shared image pages and deferred image references
- [0009](0009-text-pipeline.md): The text pipeline
- [0010](0010-line-textures.md): Line distances, dash SDFs and clipped gradient ramps
- [0011](0011-fill-extrusion.md): Lit v8 extrusions and nearest-surface layer opacity
- [0012](0012-web-demo-assets.md): Immutable web demo assets
- [0013](0013-segmented-lines.md): Segment-instanced line extrusion
- [0014](0014-pitched-cover-and-sky.md): Pitched cover and flat-surface sky
- [0015](0015-style-edits.md): Style edits without reloads, and feature state
- [0016](0016-style-playground.md): Style playground and published picking
- [0017](0017-gpu-bytecode.md): GPU expression bytecode and the cost model
- [0018](0018-bindings.md): Bindings: a Diplomat front end and our own Swift and Kotlin/JNI emitters
- [0019](0019-terrain.md): Terrain: draped slots, DEM references and a camera on the ground
- [0020](0020-any-crs.md): Any CRS: planes, tile matrix sets and the planner
- [0021](0021-interaction-and-thread-mode.md): Interaction, the web package, and the default thread mode
- [0022](0022-style-zoom-per-view.md): Style zoom per view (updates D15)
- [0023](0023-frame-time-regressions.md): Picking on demand, budgeted tile intake, and jobs off the page
- [0024](0024-picking-without-the-camera.md): Picking without the camera
- [0025](0025-atmosphere.md): The atmosphere: one sky and fog model for every surface
- [0026](0026-lighting.md): Lighting: one model, v8's light lowered onto it
- [0027](0027-dense-tiles-in-a-full-scene.md): Dense tiles in a full scene
- [0028](0028-compat.md): The Compat tier: Modern's commands beneath the HAL, a whole compat.wasm, and the device policy
- [0029](0029-device-access.md): Android device access through devices-cli
- [0030](0030-globe.md): The globe: v8 `projection` on a sphere, through patches in space
- [0031](0031-paint-transitions.md): Paint transitions over by-name paint in the Style IR

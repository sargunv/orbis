# 0011: Lit v8 extrusions and nearest-surface layer opacity

The behavioral reference is MapLibre GL JS
`2b68611a4e2d4f7f076f6c6ebf758e2d53f91471`, including
`fill_extrusion_bucket.ts`, the solid/pattern extrusion shaders and
`draw_fill_extrusion.ts`. Native and browser use the same layout, scene and
WGSL paths. All requested paint properties are implemented: color, opacity,
height, base, translate and anchor, vertical-gradient and image patterns.

## Geometry and paint

Runtime first normalizes source geometry onto the pinned 8192-unit tile grid,
so integer snapping and distance resets match across source extents.
Each vertex is four u32 words (16 bytes): packed i16 tile position, feature
slot, packed signed normal x/y with bottom/top in x's low bit, and packed
normal z / u16 edge distance. Roofs reuse the clipped fill/earcut path, with
winding reversed to account for tile Y pointing south. Walls clip original
ring segments independently to the tile square, retaining the original
normal and pattern distance phase. Clipping therefore creates no synthetic
walls. Real edges on 0 or extent remain walls; the pin suppresses synthetic
source-buffer edges outside that square. Extrusion fragment clipping includes
both rectangle bounds because a real vertical wall can lie exactly on a shared
edge. Depth, rather than repeated blending, resolves those surfaces.

The paint row has nine words: color, height and base each at two layout zooms,
then three deferred image ids. Source/composite expressions share the existing
property-table curves. Heights and bases clamp independently to zero. The
normal and top bit supply the reference's vertical gradient. Named patterns
share decision 0008's atlas, namespace and deferred row population; wall phase
uses cumulative edge distance and elevation, roofs use tile coordinates, and
both use the existing two-image zoom crossfade. Atlas reads stay in the
fragment stage so both vertex pulling and the indexed benchmark pipelines fit
core WebGPU's eight vertex-storage-buffer limit.

GL JS's v8 vertical metre is based on the IUGG mean Earth radius, 6371008.8 m.
Projected XY and the camera's ground resolution use the WGS84 semi-major axis.
The extrusion scale includes their ratio (about 1.001119), without changing
ground projection or camera resolution. Omitting that distinction moved roof
edges and failed seven otherwise correct fixtures, including a pattern case.
The scale is explicitly named `extrusion_scale` in the frame block.

CPU source coverage and GPU chunk culling test elevated volumes. Known
constant/camera paint and observed per-feature maxima bound each source's
height. This retains roofs whose ground footprints are below the viewport at
high pitch. Unknown data-dependent heights outside loaded tiles still need a
conservative source envelope. Camera far distance follows the pin's finite
horizon; fullscreen solid backgrounds stop where that frustum meets ground.

## Depth, opacity and lighting

Modern uses reversed-Z: the extrusion vertex shader maps camera depth to
`w - z`; depth clears to zero, compares GreaterEqual and writes. Back faces
are culled with counterclockwise outward winding. Other families retain their
2D passes without depth. A standard-Z experiment did not improve the remaining
fixture comparisons, so there is no exception to plan §8.3's convention.

Each extrusion node draws nearest surfaces into an RGBA8 layer target, with
one shared Depth32Float target across extrusion nodes. The main bundle
composites each layer in its actual style position, applying layer opacity
once. This avoids repeated alpha accumulation from overlapping roof/wall
surfaces. Zero-opacity layers have no cull candidates and cannot occlude.
The pin instead uses a depth-only pass followed by a depth/color/stencil pass
for translucent or patterned extrusions. Our offscreen path has a known tie
semantics difference: equal-depth surfaces within one layer are last-wins;
the pin's stencil ownership is first-wins. This is tracked in #5.

The default is 4× MSAA. The multisampled color attachment is transient and
discarded after resolve; a single extrusion layer's depth is transient and
discarded too. Multiple extrusion layers retain shared depth between passes.
The host may explicitly disable 3D antialiasing, which lazily selects a
single-sample pipeline and rebuilds dependent targets/bundles once. The
fixture harness does so to match GL JS's default antialias-off map option;
production and demo rendering remain 4×.

The typed style light evaluates anchor, spherical position, color and
intensity. The shader rotates viewport lights using frame bearing and ports
the reference's luminance-dependent directional lighting, ambient color
addition and vertical gradient. Pattern lighting uses the pin's fixed
luminance and normal divisor. Premultiplied paint RGB is retained even though
solid extrusion output alpha is one, matching `no-alpha-no-multiply`.
No camera-only light or paint uploads are needed.

The HAL adds only WebGPU-shaped attachment store/discard operations,
transient-attachment usage, and optional front/back face culling. Existing
2D callers explicitly store and keep culling disabled. No per-tile stencil,
new GPU features, threads, or worker/GPU dependencies below Runtime are added.

## Validation and measurements

`cargo xtask check` passes: native and wasm clippy, workspace tests, shader
validation and the 24-crate boundary checks. The demo regression loads an
actual translucent extrusion fixture and moves bearing/pitch: one dispatch,
bundle replay, zero paint writes, placement writes only. Switching 1×/4×
recreates targets and bundles once, then replays without paint writes.

Each cell below is **pass / fail / skip**. Native and Chrome have identical
counts in every row. Work landing alongside it
added patterns, line textures, MVT v1 decoding and symbol fixture coverage.
All original 237 passing cases and all 313 current-main passing cases remain
passing; extrusions add 39. All 352 final passing images match across platforms.
Comparison thresholds and expected images are unchanged.

| Family | Initial native / Chrome | Current main | Final native / Chrome |
| --- | --- | --- | --- |
| background | 7 / 0 / 11 | 17 / 0 / 1 | 17 / 0 / 1 |
| circle | 58 / 0 / 5 | 58 / 0 / 5 | 58 / 0 / 5 |
| color-relief | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 |
| fill | 28 / 1 / 37 | 43 / 1 / 22 | 43 / 1 / 22 |
| fill-extrusion | 0 / 0 / 52 | 0 / 0 / 52 | 39 / 2 / 11 |
| heatmap | 17 / 0 / 0 | 17 / 0 / 0 | 17 / 0 / 0 |
| hillshade | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| line | 66 / 1 / 75 | 113 / 1 / 28 | 113 / 1 / 28 |
| raster | 34 / 2 / 1 | 34 / 2 / 1 | 34 / 2 / 1 |
| sprites | — | 4 / 0 / 6 | 4 / 0 / 6 |
| icon | — | 0 / 0 / 136 | 0 / 0 / 136 |
| symbol | — | 0 / 0 / 37 | 0 / 0 / 37 |
| text | — | 0 / 0 / 278 | 0 / 0 / 278 |
| **Total** | **237 / 4 / 181** | **313 / 4 / 566** | **352 / 6 / 525** |

Extrusion's remaining failures are `height/steep-pitch-near-horizon`
(7441/262144 pixels, 2.8385%) and `vertical-gradient/east-wall`
(three silhouette pixels out of 4096, 0.07324%). Seven skips need nonzero
rounded-corner-distance, which explicitly warns; the other four need global
state and/or fixture operations. These and the depth-tie/source-envelope
gaps are tracked in [#5](https://github.com/sargunv/orbis/issues/5).

The reduced Tokyo extract does **not** retain Liberty's `building-3d` layer.
For the extrusion measurement only, the original Liberty layer is restored
with its original data-driven height/base, color and 0.8 opacity, using the
same committed Tokyo archive. The production reduced style is unchanged.
At `139.74,35.685,16,30,60`, 1024×768, Apple M6 / Metal, default 4× MSAA,
three runs of 50 warmup frames and 300 samples give these medians of run
medians in milliseconds:

| View | CPU render | GPU total | Cull | Extrusion pass | Main pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reduced extract | 0.082374 | 0.120959 | 0.005833 | 0 | 0.115084 |
| Restored building-3d | 0.094333 | 0.411833 | 0.005959 | 0.193875 | 0.211750 |

The restored view has one dispatch, 78 draws (including extrusion and
composite), 619 candidates, 19 placement writes, zero paint writes and bundle
replay after warmup. Scene counts: 395 meshes, 556106 vertex words, 244638
index words, 2088 property words. The ignored `orb-bench`
`tokyo_extrusion_frame_cost` test reproduces the measurement; `ORB_BENCH_STYLE`
selects a style file. These are whole-view costs, including extra source
coverage and the fullscreen composite, not a synthetic extrusion-only draw.
Headless Chrome's default 4× production path also renders this restored style
in main and worker modes with zero differing pixels, one dispatch, 78 draws,
19 tiles, bundle replay and zero camera-only paint writes.

The final `cargo xtask rt --platform chrome` run takes 11.990 s for native
and 10.856 s for Chrome (whole-suite times, not frame benchmarks).

Size results use `cargo xtask web` and the size gate's exact Rust Brotli
compressor (quality 11/window 22). A separate clean build without extrusions
isolates the extrusion delta from atlas/decoder work landing alongside it:

| Measure | Without extrusions | With extrusions | Delta |
| --- | ---: | ---: | ---: |
| Raw core.wasm | 1,288,106 B | 1,307,636 B | +19,530 B |
| Brotli core.wasm | 420,050 B | 426,110 B | +6,060 B (5.92 KiB) |

Final size is **416.12 KiB Brotli**, leaving **34,690 B (33.88 KiB)** under
450 KiB. Compared with the original recorded 398,484 B (~389 KiB) baseline,
the total increase is 27,626 B (26.98 KiB), including all intervening main work.


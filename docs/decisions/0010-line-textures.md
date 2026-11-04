# 0010: Line distances, dash SDFs and clipped gradient ramps

The behavioral reference is MapLibre GL JS
`2b68611a4e2d4f7f076f6c6ebf758e2d53f91471`: `line_bucket.ts`,
`line_atlas.ts`, `draw_line.ts`, `color_ramp.ts`, and the line SDF, gradient,
and combined gradient/SDF shaders. These paths run from the same crates on
native and web.

A line vertex is five u32 words (20 bytes), up from 16 bytes:

| Word | Contents |
| --- | --- |
| 0 | Signed i16 tile x/y |
| 1 | Signed i8 extrusion x/y and bounded offset normal x/y, scale 63 |
| 2 | Round/up flags in bits 0/1; half-distance in bits 2–15 |
| 3 | Feature slot, indexing the paint row |
| 4 | f32 progress within the original tiled line, before layout clipping |

Ordinary packed distances are in normalized 8192 tile units, quantized by
2. Sharp-corner subdivisions update distance, and long lines duplicate a
vertex pair at distance zero after 16384 units, as the pin does. With
GeoJSON `lineMetrics`, the packed field instead holds
`(clip_start + (clip_end - clip_start) * local_progress) * 32767 / 2`,
truncated to 14 bits, without resets. The f32 progress preserves precision
across extremely short clip ranges. Layout clipping retains the distance
before the clip; chunks retain the same vertex indices and strip continuity.
The feature slot replaces GL JS's separate split-index attribute, saving a
second extra word. Layout and shader comments document the format together.

Dashes occupy a separate **256 × 512 R8 texture (128 KiB)** in `orb-scene`,
shared by a Runtime's maps on the same device. The existing image table
provides the stable row indirection, and the existing sampler filters the
SDF. Using scalar R8 rather than an RGBA image page saves three quarters of
the texel storage and preserves the reference's 256-column SDF resolution.
Regular dashes use one row, round dashes fifteen. The port includes odd
arrays, zero-length segments and gaps, round dots, periodic horizontal
sampling, floor width, integer-zoom scaling, SDF gamma and the 300 ms zoom
crossfade. The atlas reuses rows when a style namespace retires; exhaustion
reports a warning. No atlas bindings change when a key is added or removed.

Jobs return typed image keys: sprite name, dash array/cap, or ramp pixels
and sampling mode. The packed worker protocol carries the same keys; the
scene resolves them and patches property rows even when the stream remains
host-owned. Data-driven dash arrays evaluate at layout zoom minus one,
layout zoom, and layout zoom plus one. As in the pin, constant dash arrays
use the cap's constant fallback for their SDF; data-driven dashes evaluate
caps per feature. Patterns take precedence over dash and gradient paint.

Gradient ramps share decision 0008's RGBA image pages and table. A job
evaluates the layer's expression over each GeoJSON-vt clip range. The pin
does not evaluate gradient expressions against feature properties: its
per-line rows differ because their clip ranges differ. Continuous ramps
have 256 samples. Step ramps use nearest sampling and a power-of-two
resolution from the reference's line-length/overzoom heuristic, bounded at
16384 samples. To fit the existing pages, long ramps fold into 256-wide
rectangles; explicit linear-address sampling preserves filtering across
row boundaries. This storage layout and fixed resolution ceiling differ
from GL JS's standalone texture and device-dependent texture-size ceiling.
All enabled gradient fixtures, including cross-continental narrow steps,
transparency, tile boundaries and combined dashes, pass unchanged comparisons.

Resident tiles lease their generated ramps. Eviction, failed tile upload,
style replacement and map destruction release those leases; the final
release returns the table row and exact padded rectangle for reuse. Named
sprites remain style-scoped. Unchanged camera coverage and paint cause no
texture allocation, table upload or command recording. The GPU regression
covers shared leases, rectangle reuse with another image still live,
namespace retirement and repeated SDF-row reuse.

The remaining newly enabled failure is `line-dasharray/diagonal-dash`: its
MVT v1 tiles are rejected by the existing v2-only decoder. MVT v1 support
was added afterwards. The renderer's fixture
comparison thresholds and mirrored expected images are unchanged.


## Validation and measurements

`cargo xtask check` passes, including native/wasm clippy, shader validation, worker round-trips, GPU atlas
lifetime tests and all nine demo tests. The point-family camera test now also
covers dashes and combined gradient/dash paint. The bounded-scene test uses
1.75× the final view's vertex words: 2× no longer forced eviction with the
20-byte format. It still compares exact bounded/unbounded pixels.

The original 422-test corpus gives the following results. Each cell is
**pass / fail / skip**; native and Chrome have identical counts in every row.
All 271 original passes remain passing; all 308 passing images match across
platforms. Comparisons and golden images are unchanged.

| Family | Native before | Native after | Chrome before | Chrome after |
| --- | --- | --- | --- | --- |
| `background-` | 17 / 0 / 1 | 17 / 0 / 1 | 17 / 0 / 1 | 17 / 0 / 1 |
| `circle-` | 58 / 0 / 5 | 58 / 0 / 5 | 58 / 0 / 5 | 58 / 0 / 5 |
| `color-relief` | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 |
| `fill-` | 43 / 1 / 22 | 43 / 1 / 22 | 43 / 1 / 22 | 43 / 1 / 22 |
| `fill-extrusion-` | 0 / 0 / 52 | 0 / 0 / 52 | 0 / 0 / 52 | 0 / 0 / 52 |
| `heatmap-` | 17 / 0 / 0 | 17 / 0 / 0 | 17 / 0 / 0 | 17 / 0 / 0 |
| `hillshade-` | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| `line-` | 75 / 1 / 66 | 112 / 2 / 28 | 75 / 1 / 66 | 112 / 2 / 28 |
| `raster-` | 34 / 2 / 1 | 34 / 2 / 1 | 34 / 2 / 1 | 34 / 2 / 1 |
| **Total** | **271 / 4 / 147** | **308 / 5 / 109** | **271 / 4 / 147** | **308 / 5 / 109** |

The focused line subfamilies, on both native and Chrome:

| Subfamily | Before | After |
| --- | --- | --- |
| `line-dasharray/` | 1 / 0 / 20 | 19 / 1 / 1 |
| `line-gradient/` | 0 / 0 / 9 | 9 / 0 / 0 |
| `line-gradient-dasharray/` | 0 / 0 / 7 | 7 / 0 / 0 |
| `line-cap/line-dasharray/` | 0 / 0 / 1 | 1 / 0 / 0 |
| `line-pattern/with-dasharray` | 0 / 0 / 1 | 1 / 0 / 0 |

The remaining skipped dash fixture requires harness `setZoom` operations.
Crossfade behavior is implemented using the existing zoom history, and the
existing zoom-history unit test still passes. The four pre-existing failures
are fill outline multiply, default line miter limit, and the two raster
mipmap cases. The fifth is the MVT v1 decoder issue described above.

The mirror expanded to **883 tests** with symbols/text/icons:
**312 pass / 5 fail / 566 skip** on both platforms, **312/312** passing images
matching. Four additional sprite fixtures already work through the existing
pattern path; the line implementation contributes the same 37 added passes.
Final suite times: **5.470 s native**, **6.605 s Chrome**. The original baseline
suite took 10.274 s native and 6.651 s Chrome; these timings include GPU/shader
warmup and are not a performance benchmark.

`core.wasm` before and after this work, using the size gate's
Brotli compressor (quality 11, window 22):

| Measure | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Raw wasm | 1,271,395 B | 1,288,217 B | +16,822 B |
| Brotli wasm | 414,340 B | 420,908 B | +6,568 B (6.41 KiB) |

The result is **411.04 KiB Brotli**, **39,892 B (38.96 KiB)** below the requested
450 KiB target. No lavapipe rerun or Pages publication for this extension.

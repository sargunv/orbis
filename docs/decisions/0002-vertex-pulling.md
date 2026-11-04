# 0002: Vertex pulling as the Modern default

Status: decided on an Apple M6 (native Metal and Chrome 156) and an Adreno
619 (Chrome 156 on Android): meshes (fill, fill outline, line and
fill-extrusion nodes) draw compacted indices (C), kept by the cull
dispatch; instanced quads (circles, heatmaps, icons, glyphs) and image
tiles keep vertex pulling. Mali and PowerVR are unmeasured
([#16](https://github.com/sargunv/orbis/issues/16)).

## Question

Plan §8.3 draws every Modern render node with one `drawIndirect` and vertex
pulling: 768 vertices per visible chunk, the vertex shader fetching its index
from storage and emitting degenerate vertices past the chunk's last index. That
gives up fixed-function index fetch and the post-transform vertex cache, and
pads chunks. M0 exit criterion 8 asks for the first GPU timings of that choice
against GL JS's way: one indexed draw per (layer, tile), culled on the CPU.

## Options

- **A. Vertex pulling** (the M0 default): one cull/compact dispatch, then a
  replayed bundle of one `drawIndirect` per render node.
- **B. Per-tile indexed:** CPU frustum test per (node, mesh, tile placement),
  then one `drawIndexed` per survivor over the scene's index arena, recorded
  every frame as GL JS does. The vertex shader reads its vertex by
  `vertex_index`.
- **C. Re-compacted indices:** the visible chunks' indices in a per-node
  index buffer, rewritten only when the visible set changes, and one
  indexed draw per node from a replayed bundle. The first round culled
  chunks on the CPU and compacted with a separate pass; the final form is
  GPU-driven (below).

## Rule, fixed before measuring

Vertex pulling stays the Modern default if, on the measured GPU, for every
view:

1. A's median GPU time per frame (cull and main passes) is at most 1.2× B's,
   or at most 0.25 ms above it; and
2. A's median CPU time per frame is no worse than B's.

If A loses rule 1 in any view, vertex pulling is not dropped outright: the
family that loses moves to C or to M+ multi-draw through the per-family cost
model of plan §8.3, and that is decided with phone numbers in hand. If C beats
A by more than 1.2× in every view, C becomes a candidate default for fills.

One desktop-class GPU decides nothing for phones: Adreno and Mali timings come
at the phone step, and the rule is applied again there.

## C, GPU-driven

Every candidate of an indexed node, a chunk in one tile placement, has a
fixed slot in one compacted index buffer, laid out in draw order when the
candidates are rebuilt (tile arrival, departure or a placement change).
Indices are `run << run_shift | vertex`, a run being one mesh in one
placement; the node's slice of the visible list holds its run table
(a chunk of the mesh for its bases, and the placement), so the vertex
shader finds its vertex without an instance index (core WebGPU's indirect
draws need `first_instance = 0`). The one cull dispatch, a workgroup per
node and per batch of 64 indexed candidates, tests each candidate as
vertex pulling's cull does and rewrites a slot only when its chunk enters
the view (the chunk's tagged indices) or leaves it (hidden indices, which
the vertex shader discards). It writes `drawIndexedIndirect` arguments up
to the last visible slot, over arguments cleared on the GPU first. The
bundle holds one indexed indirect draw per node and is recorded again only
when commands change, as before; camera-only frames keep the M0
invariants (one dispatch, bundle replay, CPU writes only the frame block
and placement rows).

Two findings on the way:

- **The first round's ~32 ms compaction was first use, not copying.** The
  CPU-culled C created its 64 MiB compacted buffer on the first C frame of
  each view, and the browser zero-fills a buffer on first use. Forcing a
  rebuild on later switches measured the copy itself on the Adreno 619:
  1.57–1.70 ms at z12 (884k indices), 0.20 ms at z15, 0.26–0.33 ms pitched,
  0.07 ms at z17.
- **An indexed indirect draw with no indices but one instance loses the
  device on the Adreno 619** (Chrome 156, Vulkan): `VK_ERROR_DEVICE_LOST`
  as soon as a node had candidates but none in view. Such a node now keeps
  all-zero arguments, as vertex pulling's do. Found by bisecting on the
  phone: not the index values, not the cull pass, not rewrites or rebuilds
  alone.

## Method

`cargo xtask bench` (`crates/orb-bench`), and the same loop in a browser
(`web/demo/bench.html`, `pnpm -C web bench`; `--android` drives an
adb-connected phone's Chrome over DevTools). One map visits five views of
the committed Tokyo extract with Liberty's labels on: z12, z15, z15 pitched
60°, z17, and z16 pitched 60° with Liberty's `building-3d` layer added.
Each view renders 3 rounds × 100 measured frames after 50 warm-up per
submission, rounds rotating which submission goes first, 2048×1536 offscreen,
two frames in flight; then the camera pans around a 128 px circle in 120
frames (6.7 px per frame), once to load the path and 3 rounds per submission
measured, with the cull pass's counters (`cull.wgsl`: chunks kept, indices
written, vertex inputs) copied out after every frame. GPU times are
timestamp queries around every pass; Chrome quantizes them (65.5 µs), so
browser cells are means and native cells medians.

The submission decides only how the mesh families draw: per-tile (B)
draws fills and lines per tile and everything else as A does, recording its
main pass every frame, so all three draw the same labels and every frame
matched A's pixels in every view and on every device. B never drew
extrusions, so the 3D-buildings view compares A and C on that family.

## Evidence

ms per frame; GPU is the whole frame (cull, extrusion and main passes), CPU
is `Map::render`. Apple M6 native: medians; Chrome: means.

| view | submission | M6 native GPU | M6 native CPU | M6 Chrome GPU | M6 Chrome CPU | Adreno 619 GPU | Adreno 619 CPU |
|---|---|---|---|---|---|---|---|
| z12 | A pulling | 0.562 | 0.093 | 0.531 | 0.052 | 37.265 | 4.156 |
| z12 | B per-tile | 0.419 | 0.106 | 0.388 | 0.078 | 24.639 | 5.265 |
| z12 | C compacted | 0.426 | 0.094 | 0.394 | 0.049 | 25.235 | 4.274 |
| z15 | A pulling | 0.377 | 0.081 | 0.336 | 0.052 | 23.276 | 2.588 |
| z15 | B per-tile | 0.338 | 0.103 | 0.300 | 0.077 | 21.091 | 4.980 |
| z15 | C compacted | 0.333 | 0.079 | 0.293 | 0.035 | 20.979 | 3.108 |
| z15 pitch 60 | A pulling | 0.452 | 0.132 | 0.408 | 0.100 | 26.698 | 5.717 |
| z15 pitch 60 | B per-tile | 0.389 | 0.164 | 0.344 | 0.133 | 23.880 | 7.416 |
| z15 pitch 60 | C compacted | 0.386 | 0.133 | 0.343 | 0.079 | 23.242 | 4.877 |
| z17 | A pulling | 0.295 | 0.095 | 0.242 | 0.068 | 15.506 | 3.352 |
| z17 | B per-tile | 0.260 | 0.107 | 0.224 | 0.093 | 15.407 | 4.343 |
| z17 | C compacted | 0.261 | 0.095 | 0.224 | 0.059 | 14.959 | 3.474 |
| z16 pitch 60, 3D buildings | A pulling | 1.333 | 0.152 | 0.791 | 0.095 | 37.809 | 6.095 |
| z16 pitch 60, 3D buildings | B per-tile | 1.299 | 0.176 | 0.756 | 0.149 | 36.701 | 7.962 |
| z16 pitch 60, 3D buildings | C compacted | 1.285 | 0.152 | 0.737 | 0.100 | 34.219 | 6.028 |

The extrusion pass alone (3D-buildings view; B draws extrusions as A):
A 0.563, C 0.540 (M6 native); A 0.308, C 0.284 (M6 Chrome); A 13.613,
C 11.892 (Adreno 619): C is 1.04×, 1.08× and 1.14× faster.

**Rule 1** (A ≤ 1.2× B or ≤ 0.25 ms above): met on the M6 natively
(1.03–1.34×, at most +0.144 ms) and in Chrome (1.05–1.37×, at most
+0.142 ms); **not met on the Adreno 619 at z12** (1.51×, +12.6 ms); met there
elsewhere by the ratio (1.01–1.12×, +0.1 to +2.8 ms).

**Rule 2** (A's CPU ≤ B's), now that labels draw: met everywhere. With
125–328 draws a frame, B's CPU frame is 0.01–0.05 ms over A's on the M6 and
1.0–2.4 ms over it on the Adreno 619.

**C against both:** C's GPU frame is 0.93–1.02× B's everywhere: within
+0.007 ms on the M6, and on the Adreno 619 under it except at z12
(+0.6 ms), where C's cull pass takes 0.64 ms against A's 0.31 (the batched
grid, its atomics and the arguments clear). Against A it is 1.04–1.35×
faster on the M6 and 1.04–1.48× on the Adreno 619 (−0.55 to −12.0 ms). Its
CPU frame is A's within 0.002 ms on the M6 natively, the lowest of the
three in Chrome on the M6 but for the 3D-buildings view (+0.005 ms on A),
and −0.84 to +0.52 ms from A's on the Adreno 619; under B's everywhere.

**C under the moving camera** (360 frames per view):

| view | frames rewriting | indices per such frame, mean (max) | cull pass rewriting, mean or median (max): M6 native / M6 Chrome / Adreno 619 | cull pass otherwise: M6 native / M6 Chrome / Adreno 619 |
|---|---|---|---|---|
| z12 | 324 | 22,420 (927,339) | 0.034 (0.217) / 0.029 (0.131) / 1.301 (8.454) | 0.013 / 0.035 / 1.209 |
| z15 | 264 | 4,157 (173,858) | 0.006 (0.053) / 0.009 (0.066) / 0.359 (1.704) | 0.006 / 0.010 / 0.292 |
| z15 pitch 60 | 270 | 29,258 (173,966) | 0.007 (0.078) / 0.025 (0.131) / 0.539 (1.901) | 0.007 / 0.021 / 0.423 |
| z17 | 129 | 982 (11,511) | 0.005 (0.007) / 0.004 (0.066) / 0.198 (0.262) | 0.005 / 0.005 / 0.189 |
| z16 pitch 60, 3D buildings | 306 | 50,063 (303,136) | 0.011 (0.097) / 0.028 (0.131) / 1.030 (3.539) | 0.010 / 0.013 / 0.792 |

At 6.7 px per frame, chunks enter or leave the view in 36–90% of frames, and
those frames cost the cull pass little more than the others: on the Adreno
1.30 against 1.21 ms at z12. The maxima are frames where the tile cover
changed and every slot was rewritten (927k indices at z12): 8.5 ms of cull
pass on the Adreno 619, 0.2 ms on the M6, on the frames tiles arrive or
leave, against vertex pulling's 12 ms more on every z12 frame. The first
round's CPU-culled C took 1.6 ms on the phone for every visible-set change.

## Outcome

By the rule, the mesh families leave vertex pulling: A loses rule 1 on the
Adreno 619. WebGPU core has no multi-draw, so they move to C, which matches
B's GPU time within 0.6 ms everywhere and keeps A's GPU-driven CPU frame:

- **Fill, fill outline and line:** compacted indices, the one path; their
  vertex-pulled entry points and pipelines are deleted.
- **Fill-extrusion:** compacted indices too. B never covered it, so the
  rule's comparison does not apply; C is faster than A in its pass on every
  measured GPU (1.04–1.14×), and it keeps one path for every mesh family.
- **Circle, heatmap, symbol icons and text:** instanced quads without index
  data, so vertex pulling (six vertices per instance) remains their path;
  not a choice C applies to.
- **Raster, hillshade, color-relief:** image tiles stay pulled, unmeasured:
  a tile is one chunk, 768 vertex invocations beside a tile's worth of
  fragments.
- **B (per-tile) is deleted**, with its shaders, its run table and the
  `Submission` switch: it is no default's alternative on the Modern tier,
  and Compat will draw per tile in its own way when it exists.

§18 row 4's GPU target, at most 1.1× per-tile indexed draws, is met by the
chosen path on the measured GPUs: C is at most 1.02× B on the M6 (native
and Chrome) and on the Adreno 619. Its invocation target holds by
construction rather than count: C's vertex inputs are 0.96–1.00× B's
indices, through fixed-function index fetch and the post-transform cache,
with no padding. Mali and PowerVR remain unmeasured
([#16](https://github.com/sargunv/orbis/issues/16)). The bench now measures
the default path only (`cargo xtask bench`, `pnpm -C web bench
[--android]`), still and moving; the three-way comparison, to re-run on
those GPUs, is the commit before this outcome
(`git log --grep "bench under a moving camera"`).

Open, not blocking: C's cull pass costs the Adreno 619 about twice vertex
pulling's (0.64 against 0.31 ms at z12, 1.2 ms while moving), and a tile
change rewrites every slot (up to 8.5 ms there). Preserving unchanged
candidates' slots across rebuilds would cut the latter
([#22](https://github.com/sargunv/orbis/issues/22)).

## First round

Labels removed, the views loaded by a fresh map each, and C culled on the
CPU. Kept for the record; the outcome above supersedes its outcome.

### Method


`cargo xtask bench` (`crates/orb-bench`) loads a style through
`orb-runtime`'s `Map` for each view until the map is idle, then renders
camera-only frames offscreen into an `Rgba8Unorm` target. The evidence below
loaded OpenFreeMap Liberty at Seattle (`cargo xtask bench --
https://tiles.openfreemap.org/styles/liberty --at -122.3321,47.6062`); the
default is now the committed Tokyo extract at (139.74, 35.685), which the
browser bench uses too. For each view it runs 3 rounds; each round renders 50
warm-up and 100 measured frames with each submission, starting with the next
submission each round, so 300 measured frames per cell. Two submissions are
in flight at a time, as with a swapchain.

- **GPU time:** timestamp queries at the beginning and end of every pass,
  summed per frame (A: cull + main; B, C: main). The bench wraps the HAL
  `Gpu` and adds the timestamps to the passes the renderer submits; the
  renderer does not know it is timed.
- **CPU render:** wall time of `Map::render`, all of the frame's CPU work:
  frame block, tile placement, culling where it runs on the CPU, building
  commands, encoding and submitting (less the bench's own timestamp
  bookkeeping). **CPU encode+submit** is the part inside `Gpu::submit`:
  HAL → wgpu encoding and `queue.submit`.
- **Vertex inputs:** A: 768 × visible chunks (chunks counted by C's CPU
  cull, a port of `cull.wgsl`'s test); B and C: indices drawn. Vertex shader
  invocations are lower for B and C by the post-transform reuse.
- **Pixels ≠ A:** every submission's final frame is compared with A's.

**In the browser**, `web/demo/bench.html` runs the same loop: `orb-bench` is a
library that both the native binary and `orb-web`'s `harness` feature call
(`harness.wasm`, built by `cargo xtask web` next to `core.wasm`, which it does
not grow). The page loads the Tokyo extract on its own thread, prints the
adapter and user agent, applies the rule, and copies the report as Markdown.
`pnpm -C web bench` runs it in headless Chromium.

- **Timestamps** where the browser offers `timestamp-query`. Chrome quantizes
  them (to 65.5 µs on a Mac, unless WebGPU's unsafe APIs are allowed) and
  `performance.now()` (100 µs, 5 µs cross-origin isolated). The bench
  measures both clocks' resolution and, where either is coarser than 1 µs,
  shows and judges by means over every measured frame instead of medians:
  quantization does not bias a mean. In headless Chrome on the M6, quantized
  means agree with unquantized medians within 0.003 ms in every cell, and CPU
  means at 100 µs with those at 5 µs within 0.006 ms.
- **Without timestamps** (Safari may lack them) the GPU column is a labelled
  proxy, not GPU time: wall time from the start of each frame until the GPU
  reports its work done (`onSubmittedWorkDone`), one frame in flight. It
  adds CPU work and queue latency to every submission, so it dilutes the
  rule's ratios; its differences track the GPU's (headless Chrome: +0.083,
  +0.020, +0.047, +0.026 ms against +0.077, +0.029, +0.061, +0.011 ms).

All three draw the same scene, buffers and fragment shaders; B and C use
indexed variants of the vertex entry points (`indexed.wgsl`). Liberty draws
92 render nodes today; symbol and pattern layers are not drawn yet.

### Evidence


Apple M6, Metal (wgpu 30), 2048×1536 px (1024×768 logical at 2×). Medians
over 300 frames; p90 in parentheses.

| view | submission | GPU | main pass | CPU render | CPU encode+submit | draws | dispatches | vertex inputs |
|---|---|---|---|---|---|---|---|---|
| z12 | A pulling | 0.251 (0.252) | 0.246 | 0.056 | 0.035 | 92 | 1 | 494,592 |
| z12 | B per-tile | 0.199 (0.199) | 0.199 | 0.056 | 0.033 | 197 | 0 | 643,900 |
| z12 | C compacted | 0.180 (0.180) | 0.180 | 0.055 | 0.020 | 60 | 0 | 413,502 |
| z15 | A pulling | 0.234 (0.235) | 0.229 | 0.048 | 0.034 | 92 | 1 | 284,928 |
| z15 | B per-tile | 0.193 (0.194) | 0.193 | 0.045 | 0.028 | 126 | 0 | 358,151 |
| z15 | C compacted | 0.189 (0.190) | 0.189 | 0.043 | 0.019 | 58 | 0 | 227,543 |
| z15 pitch 60 | A pulling | 0.365 (0.368) | 0.359 | 0.052 | 0.034 | 92 | 1 | 686,592 |
| z15 pitch 60 | B per-tile | 0.265 (0.268) | 0.265 | 0.074 | 0.044 | 303 | 0 | 627,835 |
| z15 pitch 60 | C compacted | 0.256 (0.259) | 0.256 | 0.066 | 0.023 | 68 | 0 | 528,319 |
| z17 | A pulling | 0.116 (0.117) | 0.111 | 0.055 | 0.043 | 92 | 1 | 155,136 |
| z17 | B per-tile | 0.099 (0.099) | 0.099 | 0.040 | 0.023 | 78 | 0 | 184,107 |
| z17 | C compacted | 0.094 (0.095) | 0.094 | 0.034 | 0.017 | 41 | 0 | 121,335 |

All times in ms. Every submission's frame matched A's pixel for pixel in
every view. The views draw 7, 4, 8 and 2 tiles.

- **GPU:** A takes 1.26×, 1.21×, 1.38× and 1.17× B's time (+0.052, +0.041,
  +0.100, +0.017 ms), and 1.23–1.43× C's. The cull dispatch is 0.005 ms;
  the rest is the main pass. At 1024×768 (1×), where fragments weigh less,
  A/B is 1.43×, 1.33×, 1.47× and 1.39× (+0.049, +0.032, +0.073,
  +0.018 ms).
- **Padding:** A runs 20–30% more vertex inputs than C draws indices (plan
  §8.3 inferred up to ~15%), and without post-transform reuse.
- **CPU:** every submission's frame is under 0.08 ms. B's grows with its draw
  count (303 draws pitched: +0.022 ms over A); A's is flat with 92 nodes, and
  most of A's encode time is wgpu replaying the bundle's 368 commands, which
  wgpu-native re-encodes at execution [inferred]. Without symbols, B issues
  far fewer draws than GL JS's ~450 at z15 (#8474), so the CPU walk vertex
  pulling removes barely exists in this scene yet.
- **C's rebuild:** the compaction pass takes 0.017–0.079 ms of GPU time on
  the frame the visible set changes. Its CPU cost on such frames, and how
  often a moving camera changes the chunk-level visible set, are not
  measured.

**Tokyo extract, Apple M6**, native Metal against Chrome 156
WebGPU (headless, quantized: means). 2048×1536 px; 300 measured frames per
cell; ms. The extract draws 75 render nodes and 2, 2, 4 and 1 tiles.

| view | submission | GPU native | GPU Chrome | CPU render native | CPU render Chrome |
|---|---|---|---|---|---|
| z12 | A pulling | 0.363 | 0.342 | 0.041 | 0.012 |
| z12 | B per-tile | 0.282 | 0.265 | 0.034 | 0.019 |
| z12 | C compacted | 0.276 | 0.261 | 0.038 | 0.060 |
| z15 | A pulling | 0.162 | 0.153 | 0.048 | 0.011 |
| z15 | B per-tile | 0.132 | 0.124 | 0.041 | 0.030 |
| z15 | C compacted | 0.133 | 0.124 | 0.038 | 0.039 |
| z15 pitch 60 | A pulling | 0.244 | 0.233 | 0.055 | 0.014 |
| z15 pitch 60 | B per-tile | 0.183 | 0.173 | 0.052 | 0.044 |
| z15 pitch 60 | C compacted | 0.180 | 0.171 | 0.045 | 0.066 |
| z17 | A pulling | 0.078 | 0.075 | 0.064 | 0.016 |
| z17 | B per-tile | 0.066 | 0.064 | 0.032 | 0.021 |
| z17 | C compacted | 0.066 | 0.064 | 0.030 | 0.026 |

- Chrome's GPU times are 3–7% under native's, with the same ratios: A/B
  1.17–1.35×, all within the 0.25 ms clause. Every frame matches A's pixels
  in both.
- In Chrome, A's CPU frame is the cheapest of the three in every view
  (−0.005 to −0.030 ms against B): replaying a bundle is one call into the
  browser, while B's draws and C's CPU cull run as wasm. Natively A is
  0.003–0.033 ms behind B, as at Seattle, so rule 2 holds in Chrome and not
  natively.
- The browser bench found that `compact.wgsl` did not compile under Tint
  (`a << b | c` needs parentheses; naga accepts it), so C never ran in a
  browser before; fixed with it.

#### Adreno 619 (moto g 5G 2024, Chrome 156, Android 15)

The browser bench, run on the phone over adb and Chrome DevTools:
Tokyo extract, 2048×1536 offscreen, 300 measured frames per cell, GPU
timestamps quantized to 65.5 µs, so means. Times in ms.

| view | A pulling GPU | B per-tile GPU | C compacted GPU | A/B | A−B | A CPU | B CPU | C CPU |
|---|---|---|---|---|---|---|---|---|
| z12 | 14.918 | 11.523 | 11.277 | 1.29× | +3.395 | 0.458 | 0.540 | 0.680 |
| z15 | 6.474 | 5.395 | 5.294 | 1.20× | +1.079 | 0.478 | 0.662 | 0.479 |
| z15 pitch 60 | 9.885 | 7.953 | 7.641 | 1.24× | +1.932 | 0.545 | 1.085 | 0.899 |
| z17 | 2.637 | 2.208 | 2.198 | 1.19× | +0.429 | 0.427 | 0.503 | 0.521 |

- Pixels matched A's in every cell.
- Rule 1 fails in z12, z15 and z15 pitch 60; rule 2 holds everywhere (A's
  CPU is the lowest, by 0.08–0.54 ms).
- C's per-frame GPU time is B's or lower, but its compaction pass took
  31.8–32.7 ms on the frame the visible set changed: three to fifteen
  frames of budget, every time the camera moves enough to change the
  visible chunks. That's far above the 0.017–0.079 ms it takes on the M6,
  so it's an inefficiency in `compact.wgsl` on this GPU, not a property of
  C.
- A's vertex inputs are close to B's (371k vs 394k at z12), yet A is
  slower: the cost is the storage-buffer index and vertex fetch without
  fixed-function index fetch and post-transform reuse, more than padding.

### Outcome


Against the rule on the Apple M6:

1. **GPU: met**, through the absolute clause only. A is more than 1.2× B in
   three of four views, but never more than 0.10 ms above it.
2. **CPU: not met** at z15 (+0.003 ms) and z17 (+0.015 ms); tied at z12 and
   better pitched (−0.022 ms).
3. **C beats A by more than 1.2× in every view** (1.23–1.43×), so C is a
   candidate default, for fills and, here, lines too.

So the rule does not confirm vertex pulling, and by its last clause one
desktop GPU does not overturn it either. For now:

- Vertex pulling stays the Modern default. The default path changed only
  by sharing its per-family vertex functions with the indexed entry points,
  and the index arena gaining `INDEX` usage. `Submission::PerTileIndexed` and `Submission::CompactedIndices`
  stay in `orb-render` as measured alternatives, and lead into Compat's
  per-tile draws.
- The CPU margins are microseconds in a scene without symbols, so rule 2 is
  re-checked when symbol layers draw (M1+).
- Next evidence: Adreno and Mali timings at the phone step (the same
  bench, on Android once it runs there), C's rebuild cost under a moving
  camera, and a per-family split of the main pass. Those decide between
  pulling, C, and M+ multi-draw per family (plan §8.3's cost model).

On the Adreno 619, A loses rule 1 by 1.20–1.29× (0.4–3.4 ms) in three of
four views, so by the rule the losing families move off vertex pulling.
WebGPU core has no multi-draw, so the candidate is C, which matches B per
frame. That makes the decision:

- **Fills and lines move to compacted indices (C)** once its compaction pass
  costs well under a frame on the Adreno 619 under a moving camera, and C is
  kept GPU-driven: culling and compaction on the GPU, indexed indirect draws,
  bundle replay. The default switches in the change that fixes compaction,
  with the phone re-measured.
- Until then vertex pulling stays the default, since C's 32 ms rebuild is
  worse than A's 0.4–3.4 ms.
- No Mali device is available; Mali numbers come when one is.

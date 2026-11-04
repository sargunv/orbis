# 0013: Segment-instanced line extrusion

Status: measured and shelved. CPU extrusion remains the default for solid,
dashed, gradient and patterned lines. The rule was fixed before implementation
or measurement; the experimental code is not part of the tree.

## Question

Can points stored once, with segment/join/cap instances extruded in the vertex
shader, replace GL JS's CPU extrusion without losing its rendering semantics?
The reference is the pinned GL JS port and the unchanged render comparisons
described in decisions 0002 and 0010. The CPU path remains the default while
this experiment runs. Both paths use the same paint and fragment shading.

## Rule, fixed before measuring

Apply these gates independently to solid, dashed, gradient (including combined
gradient/dash), and patterned lines. A variant becomes the default only if:

1. **Correctness:** on native and Chrome it passes at least every fixture that
   the CPU path passes, with unchanged goldens, scope and comparison thresholds.
   Equal aggregate counts with different failing cases do not pass this gate.
   The entire default render suite must retain all its existing passes.
2. **Memory:** actual occupied line geometry bytes (points, instances, indices
   and chunk records, excluding shared paint/atlases and unused arena capacity)
   are at most 75% of CPU extrusion's total across the Tokyo extract's four
   benchmark views and live planet views. Report totals and bytes per tile;
   no individual view may grow. Report upload bytes separately if different.
3. **GPU:** with identical submission/culling, viewport, style and camera,
   frame GPU time is at most 1.10× CPU extrusion's or at most 0.10 ms above it,
   in every measured native/Chrome view. Use decision 0002's rotating rounds,
   warm-up and timestamps; use means for quantized browser clocks. Report
   line-only measurements as well as complete-map frame timings so fills do
   not conceal a line regression.
4. **CPU:** steady camera-only CPU frame time is at most 1.10× CPU extrusion's
   or at most 0.02 ms above it in every view. Tile layout CPU time must not
   grow by more than 10%; width-only paint changes must not rebuild geometry.
5. **Portability/maintenance:** the shared native/wasm implementation passes
   `cargo xtask check`; record `cargo xtask web`'s core.wasm raw and Brotli
   delta. No hidden CPU-extruded fallback counts as a segmented pass.

If a variant passes correctness/memory and is within 20% of a timing boundary,
an Adreno 619 browser run decides before switching that variant.
Desktop timing alone cannot establish a phone win. If correctness or memory
fails decisively, shelve without requiring the phone. Keep experimental code
only if the measured benefit justifies a supported opt-in path; otherwise
delete it and retain the reproducible method and data here. Record unsupported
join/cap cases and the reason rather than weakening comparisons.

## Interpretation of the rule

`AGENTS.md` treats GL JS as “a reference, not a pixel target”. That is
consistent with using the existing fixture tolerances,
not requiring byte-identical pixels. Sparse raw pixel differences in benchmark
images are not a rejection criterion. The final prototype meets the fixture
pass-set gate anyway; this decision does not hinge on the last few pixels.
The fixed performance rule, rather than a stricter visual interpretation,
rejects it. No goldens, scope or thresholds were changed.

## Implementation and method

The prototype's experimental setting is style metadata `orbis:segmented-lines: true`, carried
in the same style JSON to tile workers. The default remains CPU extrusion.
The fixture driver enables it with `ORB_LINE_GEOMETRY=segmented`; benchmark
settings `lines=true` and `onlylines=true` compare both paths.

- Each clipped point is a 20-byte record: packed position, feature slot,
  distance, progress, and join/endpoint metadata. A closed line has a closing
  record with its own distance; sharp-corner helper points are stored once.
- A segment quad or join/cap piece is a 16-byte record of four point/recipe
  references, plus a 4-byte instance index. Six vertex invocations synthesize
  its triangles; a fan wedge uses a degenerate second triangle. The CPU chooses
  topology and join class, but stores no extrusion or offset vectors.
- WGSL reconstructs segment normals, miters, bevels, flipbevels, fakeround
  interpolation, round/square caps and bounded offset normals, including the
  existing signed-byte quantization. Paint, projection, translation, dash SDF,
  gradient ramp and pattern sampling use the CPU path's shared vertex shading
  and fragment shader. There is no hidden CPU-extruded fallback.
- The existing GPU chunk cull and indirect bundle draw are retained, with
  256 instances per chunk and the same line bounds/margins/clip semantics.
  Both sides use `Submission::VertexPulling`, the current default; this is
  independent of decision 0002's alternatives.
- Tokyo uses the committed MVT extract and reduced Liberty style at
  (139.74, 35.685). The live planet uses OpenFreeMap Liberty at Seattle
  (−122.3321, 47.6062). Views are z12, z15, z15/pitch60 and z17, at
  2048×1536 physical pixels (1024×768 logical, ratio 2).
- Apple M6, native Metal/wgpu 30 and headless Chrome 156 WebGPU. Three rounds
  alternate which geometry runs first, with 50 warm-up and 100 measured frames
  per path per round: 300 measured frames per cell. The HAL wrapper and GPU
  timestamps are decision 0002's. Native uses medians (1 ns timestamp clock);
  Chrome uses means (65.5 µs GPU and 100 µs CPU clock quantization).
- Both complete maps and line-only styles are timed. Variant runs apply
  `[2,2]` dashes, a linear red→blue progress gradient, or the pinned Emerald
  `generic_icon` pattern to the same Tokyo polylines and original paint.
  These isolate the fragment variants without changing geometry selection.
- Browser timings used an unbundled benchmark page and driver with the
  prototype's wasm, removed with the experiment along with its variant styles
  and sprites.

The full per-view GPU/CPU timings, p90s, pixel counts, occupied geometry bytes,
tile counts, source hashes and fixture breakdown are retained in
[data/0013-segmented-lines.json](data/0013-segmented-lines.json).

## Correctness

These counts use the fixture scope matched at measurement.
Native and Chrome have the same counts in each cell. Variant classification
uses actual paint: pattern first, gradient (including combined dash) next,
dash next, otherwise solid. Skips retain their original scope.

| Variant | CPU pass / in scope | Segmented pass / in scope | Skipped |
| --- | ---: | ---: | ---: |
| Solid | 67 / 68 | 67 / 68 | 23 |
| Dashed | 21 / 21 | 21 / 21 | 2 |
| Gradient, including dashed gradient | 16 / 16 | 16 / 16 | 1 |
| Pattern | 9 / 9 | 9 / 9 | 2 |
| **All lines** | **113 / 114** | **113 / 114** | **28** |

Every CPU pass remains passing. The sole shared failure is
`line-miter-limit/default`; no join/cap subfamily loses a CPU pass. All 120
passing images in the `line-` substring run (which also selects eight passing
fill cases) match native/Chrome under the existing comparison.

An initial implementation without sharp-corner subdivisions passed 104/114.
Adding them recovered seven cases. Preserving the preceding helper point when
rounding the next subdivision recovered the last two gradient cases (4 pixels
in `line-gradient/gradient-step`, 12 in
`line-gradient-dasharray/gradient-shared-source`). This was a small geometric
context correction, not a tolerance change or extra fragment special case.

This is fixture parity, not proof for every polyline: in particular, the
prototype resets long-line distance for the next point without GL JS's
additional duplicate pair at the reset. A production version would need reset
instances for that case. It does not justify more work once the independent
GPU gate fails decisively.

## Memory

Occupied line geometry includes point/instance vertex words, index words and
32-byte GPU chunk records. Paint, atlases and unused arena capacity are excluded.
The same geometry bytes are uploaded. Existing arena reservations are fixed,
so these numbers describe used capacity and upload traffic, not a reduction in
the renderer's total allocated device memory. Totals include resident line
meshes; the per-tile figure divides by distinct currently placed line tiles,
so it is an occupancy average rather than the size of a particular tile key.
Native and Chrome agree exactly. All four paint variants have identical
geometry sizes.

| Data / view | Placed line tiles | CPU bytes | Segmented bytes | CPU / segmented bytes per placed tile |
| --- | ---: | ---: | ---: | ---: |

| Tokyo z12 | 2 | 11,206,920 | 5,405,196 | 5,603,460.0 / 2,702,598.0 |
| Tokyo z15 | 6 | 1,126,272 | 582,808 | 187,712.0 / 97,134.7 |
| Tokyo z15 pitch 60 | 9 | 1,764,472 | 901,092 | 196,052.4 / 100,121.3 |
| Tokyo z17 | 9 | 111,968 | 55,864 | 12,440.9 / 6,207.1 |
| Planet Seattle z12 | 9 | 12,335,984 | 6,087,056 | 1,370,664.9 / 676,339.6 |
| Planet Seattle z15 | 6 | 2,025,168 | 1,061,360 | 337,528.0 / 176,893.3 |
| Planet Seattle z15 pitch 60 | 25 | 6,582,376 | 3,467,660 | 263,295.0 / 138,706.4 |
| Planet Seattle z17 | 6 | 189,992 | 99,936 | 31,665.3 / 16,656.0 |

Across the four views, Tokyo falls from 14,209,632 to 6,944,960 bytes
(**51.1% less**); the planet falls from 21,133,520 to 10,716,012 bytes
(**49.3% less**). Each view shrinks. The 75% gate passes comfortably, although
the result is about 2× smaller rather than the plan's hoped-for several-fold.

## Frame time

All times are ms, written **CPU extrusion / segmented**. “CPU frame” is the
host's complete `Map::render` wall time less timing bookkeeping; the detailed
report also retains encode/submit time. These are paired measurements on one
machine, not cross-platform CPU speed claims.

| Data / view | Native GPU | Chrome GPU | Native CPU frame | Chrome CPU frame |
| --- | ---: | ---: | ---: | ---: |
| Tokyo z12 | 0.507 / 0.957 | 0.483 / 0.861 | 0.225 / 0.256 | 0.021 / 0.025 |
| Tokyo z15 | 0.256 / 0.333 | 0.235 / 0.305 | 0.170 / 0.171 | 0.032 / 0.028 |
| Tokyo z15 pitch 60 | 0.313 / 0.421 | 0.291 / 0.382 | 0.204 / 0.209 | 0.036 / 0.033 |
| Tokyo z17 | 0.136 / 0.155 | 0.138 / 0.143 | 0.210 / 0.220 | 0.031 / 0.031 |
| Planet Seattle z12 | 0.490 / 0.902 | 0.490 / 0.861 | 0.406 / 0.401 | 0.041 / 0.038 |
| Planet Seattle z15 | 1.027 / 1.324 | 0.647 / 0.922 | 0.320 / 0.329 | 0.030 / 0.028 |
| Planet Seattle z15 pitch 60 | 2.055 / 3.000 | 1.279 / 2.117 | 0.580 / 0.588 | 0.054 / 0.056 |
| Planet Seattle z17 | 0.796 / 0.874 | 0.434 / 0.511 | 0.271 / 0.271 | 0.036 / 0.036 |

The Tokyo **line-only** z12 runs make the per-variant result explicit. Full
four-view line-only tables are in the data file.

| Variant | Native GPU | Chrome GPU | Native CPU frame | Chrome CPU frame |
| --- | ---: | ---: | ---: | ---: |
| Solid | 0.452 / 0.896 | 0.428 / 0.800 | 0.186 / 0.225 | 0.020 / 0.018 |
| Dashed | 0.541 / 0.992 | 0.513 / 0.889 | 0.234 / 0.249 | 0.023 / 0.020 |
| Gradient | 0.471 / 0.918 | 0.447 / 0.814 | 0.200 / 0.225 | 0.023 / 0.019 |
| Pattern | 0.552 / 0.994 | 0.514 / 0.883 | 0.234 / 0.258 | 0.019 / 0.025 |

Every variant fails the GPU gate at z12: segmented takes **1.80–1.98×**
native GPU time and **1.72–1.87×** Chrome GPU time, adding **0.367–0.451 ms**.
Both the 1.10× and +0.10 ms alternatives fail. Complete-map Tokyo and planet
views confirm that this is material outside the isolated line test too.
The cost is in the main pass: reconstructing normals and joins for six
invocations per piece costs more than reading the pre-extruded records.

CPU frame time brings no consistent compensating benefit; it is close in
Chrome, while native Tokyo z12 grows 0.225→0.256 ms (13.8%, +0.031 ms),
failing both CPU alternatives. CPU times vary with host load, so the decisive
result is the repeatable GPU gap.

The pure tile build after MVT decoding (all selected Tokyo layers and paint,
no I/O/GPU) uses three alternating rounds, 10 warm-up and 100 measured layouts
per path per round:

| Tile | CPU median | Segmented median | Ratio | CPU / segmented p90 |
| --- | ---: | ---: | ---: | ---: |
| 12/3637/1612 | 10.700875 | 11.946333 | 1.116× | 12.407083 / 14.456709 |
| 14/14551/6451 | 1.578458 | 1.716750 | 1.088× | 1.672125 / 1.838791 |

The z12 tile also misses the 10% layout gate. Both paths already apply width
in shared vertex paint code; changing width does not require geometry rebuild
on the CPU path either. GPU extrusion adds no new width-edit capability here.

## Outcome and size

**Shelve all four variants and remove the experimental code and settings.**
Correctness and occupied-memory gates pass, but every variant loses the GPU
gate decisively; native CPU frame/layout measurements do not rescue it. The
Adreno 619 step is unnecessary for this result: the desktop GPU gap is far
outside the close-result band. No phone or Mali timing is claimed.

Against the same base, the prototype grew `core.wasm` from
**1,308,032 raw / 426,225 Brotli bytes** to
**1,349,349 raw / 440,984 Brotli bytes**: **+41,317 raw / +14,759 Brotli**
bytes. Exact Brotli byte counts here use Node's Brotli encoder at quality 11,
window 22 on `cargo xtask web` output; its Rust budget compressor can differ
slightly. Keeping an unsupported opt-in renderer would retain that payload and
its extra topology/shader/worker/cull surface for a path that loses the timing
rule. The tree retains only this record and data. This result rejects this
particular design on the measured hardware, not all possible procedural line
renderers.

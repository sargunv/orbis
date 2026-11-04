# 0017: GPU expression bytecode and the cost model

Plan §8.4's GPU interpreter and §11.3's `GpuBytecode` binding plan: which
paint expressions the GPU evaluates, how they reach it, how exact they are,
what they cost, and the threshold the cost model starts with.

## The compiler (`orb-expr::gpu`)

`Program::compile` lowers an expression's SSA IR to bytecode, or returns a
`Fallback` saying why it stays on the CPU: `Unsupported` (operator, operand
or result type), `Limit`, `InexactBranch`, `StateBranch`. Limits: 32
instructions, 16 labels, branches or arguments per instruction, 16 live
registers (linear-scan allocation over the SSA order). Literals are a
constant pool, not instructions, and `let`/`var` are aliases, so a typical
`match` on `class` is two instructions.

The subset: `get`, `has`, `id`, `geometry-type`, `feature-state` (literal
keys), `zoom`, comparisons, `!`, `all`, `any`, `case`, `match`, `coalesce`,
`step`, `interpolate` (rgb; linear, exponential, cubic-bezier), `in` with a
literal array, the arithmetic and maths operators, type assertions,
`to-number` and `to-color`, `typeof` compared with a type name (legacy
functions' defaults), and `error`. Results are numbers, colours or booleans.
Coercions are pushed down to pack time, through selections, so the GPU never
parses a string: `["get", "color"]` as a colour packs the parsed colour, and
`["coalesce", ["feature-state", "color"], "black"]` packs state as a colour
with nulls kept apart for `coalesce`.

A render node's programs share one `Layout`: its input columns, its raw
feature-state values and one string dictionary (strings are ids; one outside
the dictionary matches no label). `Layout::encode` emits the words the
interpreter runs: a header with the property's default, instructions,
operand tables, constants, and an input table resolved to the node's
columns and state block.

## Precision (decided with the owner)

The CPU evaluator stays f64, as GL JS's numbers and Native's `double`. Every
branch the GPU takes must equal the VM's; outputs after the branch may be
f32. The owner's rule allowed "a simpler principled scheme" for float
inputs than checking f32-exactness at pack time. Orbis uses one: **branch
operands carry the exact f64.** A number is its f64 bits beside its f32, and
comparisons, `match`, `step`, `in` and interpolation stops compare the bits
through a total order (−0 = +0, NaN unordered). Inputs, literals and the
camera zoom (the frame block carries its f64 bits) are exact, so no literal
or input ever falls back for precision. What remains of the rule:

- **Arithmetic feeding a branch falls back at compile time**
  (`InexactBranch`): `["<", ["+", ["get", "n"], 1], 2]`. Interpolation
  inputs may be arithmetic: the output is continuous in them.
- **Numeric feature state feeding a branch stays on the CPU**
  (`StateBranch`), as decided, so classification depends only on the
  expression. Boolean state branches are interpreted. With exact f64 state
  values in the table, lifting this is a one-line change if the owner wants
  it.
- Unknown strings share an id, so two inputs never compare as strings, and
  strings never order (no collation on the GPU): `Unsupported`.

What it affects: of the 425 pinned expression fixtures that compile, none
falls back for precision; among the render fixtures' data-driven fill, line
and circle paint, one value falls back with `InexactBranch` and none with
`StateBranch`.

A colour is two unorm16 pairs of its straight components; the shader
premultiplies. A colour whose premultiplied and straight components disagree
beyond rounding (`to-color` of an array with alpha 0) draws transparent where
the VM keeps its premultiplied RGB.

## The interpreter (`expr.wgsl`)

A register machine over `vec4<u32>` values (tag, payload), with error values
for VM errors: every operator scans its operands in the VM's order, so an
error in an untaken branch is never selected, as the VM evaluates lazily.
The property's default replaces an error, null, NaN or a value of the wrong
type, as GL JS's `fit` does. `orb-shaders` composes it only into
bytecode-class shaders (fill, fill outline, line, circle and their indexed
variants); the table class gets stand-ins, and the renderer creates the
bytecode pipelines for the first node that needs them. The HAL has no async
pipeline creation, so that first frame compiles them.

Each shader evaluates all its interpreted slots in one loop over the mask
(`gpu_values`), one call site with a data-dependent trip count. With one call
per slot (seven in the line and circle shaders) Chrome's GPU process
crashed in the Adreno 619's Vulkan driver (shader compiler EV031.35.01.12)
when the pipelines were created, while the same interpreter compiled as a
compute shader. Drivers that inline compile the interpreter once per
pipeline this way.

## The equivalence suite

`orb-expr`'s `conformance_gpu` runs the pinned fixtures through
`expr.wgsl` in a compute dispatch and compares with the VM: tags, booleans,
null and errors exactly, numbers within 1e-4 relative, colour channels
within 1e-4. It adds eight precision probes: values f32 would misbranch on
(0.1 against 0.10000000000000002, 2²⁴ + 1, 2³² + 1, a step at 0.1, a zoom of
10.499999999 against a stop at 10.5, `in`), boolean state, and state colours.

| | Apple M6 (Metal) | Chrome 156 (WebGPU, Metal) |
|---|---|---|
| fixtures compiling on the CPU | 425 of 583 | 425 of 583 |
| GPU-evaluable | 136 (+8 probes) | 136 (+8 probes) |
| agree with the VM | 144/144 fixtures, 447/447 inputs | 144/144, 447/447 |
| fall back | 289, all `Unsupported` | same |

The 289: 146 have value, null, string, array, object, formatted, image, padding or
projection results; the rest use operators outside the subset (`within`,
`distance`, collators, string comparisons of two inputs, `index-of`,
`length`, `at`, lab/hcl interpolation, `rgba` with computed channels,
unfolded `global-state`, heatmap density, elevation, `to-boolean`).

## Binding (`orb-style::CostModel`)

`Binding::Bytecode` joins `Constant`, `Camera`, `Source`, `Composite` and
`State`. `CostModel::binding` is the one place that decides it, for the
diff's change classes and the renderer's plan alike: a data-driven fill,
fill outline, line or circle paint value whose program compiles is
interpreted when it has at most `max_ops` instructions, else it is property
rows. A value that reads feature state and feature data has no row binding,
so it is interpreted whatever its length (#6, for the GPU-evaluable
subset); otherwise it is still drawn without state. The map's model
(`Map::set_cost_model`; `CostModel::ROWS`, `ALL`, `DEFAULT`) travels with
the style to job workers. The default is `ROWS` (below).

Interpreted values evaluate at the camera's exact zoom, as camera curves do;
property rows keep GL JS's composite approximation, values at the tile's
zoom and the next.

**Columns.** Tile jobs pack a node's input columns after each feature's
property row: two words per value (an f64, or a NaN-boxed tag and payload),
three for a colour. A mesh's row stride is its own (chunk word 7, indexed
runs), so rows and stride agree through every edit. Jobs also report each
numeric column's range; the culling margin of an interpreted width is an
interval bound of its expression over those ranges at the camera zoom.

**Edits.** Programs live in one property-arena table per style revision.

| Edit | Cost |
|---|---|
| program changes, same columns (other colours, widths, operators) | program table write: 1 frame, no jobs |
| a flip to bytecode or a change of the columns read | column job (rows only) |
| bytecode → constant or camera | uniform; rows keep their columns unread |
| mixed state and data, `setFeatureState` | one state row, 1 frame |

Measured on the Tokyo extract (`crates/orb-demo/src/edits.rs`): recolouring
an interpreted `match` takes 1 frame and no jobs; the flip from a constant
5 column jobs and no relayout; back to a constant 1 frame and no jobs.
Feature-state rows gain the raw state values programs read, after the
state-lookup values.

## Render fixtures

With symbols drawn: native 850 / 168 / 126, Chrome 833 /
185 / 126, 833 of 850 passes identical, with the same failing set before
and after, with the default model and natively with `CostModel::ALL` (every
compilable value interpreted). Before the symbol pass landed, with every
compilable value interpreted: 521 / 9 / 614 on both, 521 of 521 identical,
also unchanged. 66 fixture styles have interpreted paint under `ALL` (72
nodes, 77 slots); none mixes state and data, so under the default no
fixture interprets.

## Cost and the threshold

**Method.** `cargo xtask bench --experiment paint` (and
`bench.html?experiment=paint` in a browser, the same loop): decision 0002's
Tokyo extract (its style before Liberty's labels were added to it),
2048×1536, two views, two maps per view: `CostModel::ROWS`
and `CostModel::ALL`. The reduced Liberty style has no data-driven paint, so
after measuring it as it is, every fill and line colour is replaced, as an
edit of both maps: a `match` on `class` with eight labels (two
instructions), then programs of 5, 9, 17 and 31 instructions on the feature
id (`to-number`, multiply-add steps, `interpolate`). Rounds alternate rows
and bytecode; 3 × 100 frames after 50 warm-up each. Interpreting every fill
and line colour is a stress: real styles interpret some properties of some
layers.

**Rule, fixed before the phone was measured:** the threshold is the largest
swept length for which interpreting every fill and line colour costs at
most 1.25× the rows frame's GPU time, or at most 0.25 ms more, in both
views, on every measured GPU (decision 0002's form).

GPU time per frame in ms (M6: medians, native Metal; Adreno 619: means of
timestamps quantized by Chrome, moto g 5G 2024, Chrome 156, over adb):

| view | paint | M6 rows | M6 bytecode | M6 ratio, Δ | Adreno rows | Adreno bytecode | Adreno ratio, Δ |
|---|---|---|---|---|---|---|---|
| z15 | style | 0.257 | 0.257 | 1.00×, +0.000 | 12.511 | 12.504 | 1.00×, −0.007 |
| z15 | `match`, 2 ops | 0.257 | 0.280 | 1.09×, +0.023 | 12.833 | 33.917 | 2.64×, +21.08 |
| z15 | 5 ops | 0.257 | 0.302 | 1.17×, +0.045 | 12.800 | 55.307 | 4.32×, +42.51 |
| z15 | 9 ops | 0.257 | 0.348 | 1.35×, +0.090 | 12.771 | 80.625 | 6.31×, +67.85 |
| z15 | 17 ops | 0.257 | 0.440 | 1.71×, +0.182 | 12.833 | 124.388 | 9.69×, +111.6 |
| z15 | 31 ops | 0.257 | 0.601 | 2.34×, +0.344 | 12.786 | 196.657 | 15.38×, +183.9 |
| z15 pitch 60 | style | 0.313 | 0.313 | 1.00×, +0.000 | 15.766 | 15.601 | 0.99×, −0.164 |
| z15 pitch 60 | `match`, 2 ops | 0.313 | 0.338 | 1.08×, +0.024 | 16.225 | 52.286 | 3.22×, +36.06 |
| z15 pitch 60 | 5 ops | 0.314 | 0.366 | 1.17×, +0.053 | 16.259 | 89.738 | 5.52×, +73.48 |
| z15 pitch 60 | 9 ops | 0.313 | 0.439 | 1.40×, +0.126 | 16.230 | 133.043 | 8.20×, +116.8 |
| z15 pitch 60 | 17 ops | 0.313 | 0.580 | 1.85×, +0.266 | 16.221 | 205.127 | 12.65×, +188.9 |
| z15 pitch 60 | 31 ops | 0.313 | 0.827 | 2.64×, +0.514 | 17.041 | 322.897 | 18.95×, +305.9 |

CPU render time does not change (M6 0.055–0.077 ms; Adreno 1.6–3.4 ms, the
phone's noise). The `match` frames equal the rows frames pixel for pixel;
the arithmetic programs differ by at most 1–3/255 in a channel (rows
quantize colours to 8 bits). On the M6 interpretation costs about 0.01 ms
per instruction per frame with every fill and line interpreted, and the
rule passes through 9 instructions. On the Adreno 619 it costs about
5.5 ms per instruction at z15, and even the `match` doubles the frame: the rule
passes at no length.

**The threshold is 0** (`CostModel::DEFAULT = ROWS`): data-driven paint is
property rows wherever rows can hold it, and only values that read feature
state and feature data are interpreted. `CostModel::ALL` opts in, and with
it a program edit is a buffer write.

The register file's form decides much of the cost, differently per GPU.
Four registers as named fields selected by `switch` (instead of the
dynamically indexed 16-register array) made the Adreno's `match` 1.98×
(+12.5 ms) instead of 2.64× and every length about 25–40% cheaper, but made
the M6's `match` 16× (+3.9 ms): Metal kept the array in registers and spilled
the fields. The array stays; neither form meets the rule on the Adreno. The
next candidates are the 50-odd storage-buffer loads a `match` makes per
vertex (programs in a uniform buffer, compact label tables) and a
per-device threshold (the M6 would take 9); the owner decides the latter
([#15](https://github.com/sargunv/orbis/issues/15)).

On the phone the equivalence suite also passes (Chrome 156, the
interpreter as a compute shader).

**Mali and PowerVR:** no device exists to measure; recorded as open in the
plan (M1 exit criterion) and [#16](https://github.com/sargunv/orbis/issues/16).

## Size

`core.wasm` at the same base: 562 → 578 KiB Brotli (+16 KiB), 1770 →
1839 KiB raw (+69 KiB): about 20 KiB raw (4 KiB compressed) of interpreter
WGSL, the rest the compiler, layouts, packing and bounds (`orb_expr` grows
from 6.5% to 7.3% of the pre-`wasm-opt` code). `core.wasm` was already over
the 550 KiB ceiling at that base. Stripping comments from composed
WGSL would cut every shader, not only this one.

## Open

- The interpreter's cost on Adreno 6xx ([#15](https://github.com/sargunv/orbis/issues/15)), and on Mali-G52/G57 and PowerVR ([#16](https://github.com/sargunv/orbis/issues/16)).
- Extrusion, heatmap and symbol paint stay rows.
- `StateBranch` (numeric state in branches), lab/hcl interpolation, `rgba`
  with computed channels and string operators fall back.
- Async pipeline creation: the HAL creates bytecode pipelines synchronously
  on first use.
- While a column job is in flight after a flip, a node's new programs read
  rows packed for the old layout for a frame or two, as table flips read
  stale rows today.

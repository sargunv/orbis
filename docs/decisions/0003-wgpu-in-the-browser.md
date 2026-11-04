# 0003: wgpu's WebGPU backend in the browser

Status: decided at M0: keep wgpu on the web; the web-sys
contingency is not needed.

## Question

Plan §8.1 draws browser WebGPU through wgpu's WebGPU backend, with a thin
web-sys WebGPU HAL as the contingency "if wgpu's browser overhead breaks the M0
size gate". M0 exit criterion 6 asks for `core.wasm`'s size gated with a twiggy
report, wgpu's browser overhead recorded, and the contingency decided.

## Rule, fixed before measuring

Keep wgpu if `core.wasm` meets the size target (450 KB brotli, plan §14.3) with
room for M2's text and symbols (+60–80 KB), and wgpu's own code is not among
the parts that would have to shrink to stay there. Otherwise write the web-sys
HAL.

## Method

`cargo xtask web` builds `core.wasm` (opt-level z, LTO, one codegen unit,
SIMD128, `wasm-opt -Oz`), reports raw and brotli size, warns above the 450 KiB
target and above the 550 KiB ceiling (originally it failed at 450, then at
550), and attributes the code of the build before wasm-opt (which still has function
names) to crates with `twiggy top`. Shares are approximate: generic code
monomorphized from `core`/`alloc` is attributed to them or to "(generic)", not
to the crate that instantiated it.

## Evidence

At M0, the main-thread and worker roles in one module, with
style parsing, expressions, MVT, MLT, GeoJSON, PMTiles, layout, the scene and
the renderer:

- `core.wasm`: 989 KiB raw, **326 KiB brotli**; `core.js` glue 84 KiB raw,
  15.6 KiB gzip.
- Code by crate before wasm-opt (1,442 KiB): (generic) 28.4%, `core` 11.8%,
  `fastpfor` 10.4%, `alloc` 9.0%, `orb-expr` 7.8%, `mlt-core` 6.5%, `js-sys`
  3.5%, **`wgpu` 3.5%**, `orb-v8` 2.9%, `orb-tile` 2.2%, `orb-runtime` 2.2%,
  `hashbrown` 1.8%; `wgpu-types` 0.7%.
- wgpu's WebGPU backend is a binding layer: about 4% of the code, plus most of
  the JS glue. MLT decoding (`mlt-core` and `fastpfor`) is 52 KiB brotli (#1).

## Outcome

Keep wgpu. `core.wasm` is 124 KiB under the target with M2 still to come, and
wgpu is a small share of it; a web-sys HAL would save at most a few percent
while adding a second WebGPU implementation to maintain. The levers if size
gets tight are elsewhere: MLT's unrolled FastPFor kernel (#1), formatting and
panic machinery in `core`/`alloc` (plan §14.3 rules out `format!` in bundled
paths), and `orb-expr`. The M0 hypothesis of ≤300 KB is exceeded by 26 KB;
the build warns above the 450 KB target and the 550 KB ceiling but never
fails, so size is watched without blocking work or complicating code.

## Size since M0

`cargo xtask web`'s numbers (brotli quality 11, window 22). With the text
stack linked (`orb-text`, `orb-symbol`, decision 0009), `core.wasm` was
1,695 KiB raw, 546 KiB brotli. Cuts taken because they also simplify or
clean up the code, each against the one before:

| Cut | Raw | Brotli |
| --- | ---: | ---: |
| `orb-expr`'s evaluator takes `&dyn Feature`, one copy instead of one per feature type | −34 KiB | −2.1 KiB |
| The render fixture driver moves to `harness.wasm` with the bench | −20 KiB | −7.9 KiB |
| PMTiles gzip through `miniz_oxide` with a bytewise CRC-32, without `flate2` and `crc32fast` (whose 16 KiB slice-by-16 table is most of this) | −29 KiB | −19.2 KiB |

The gzip cut costs native decode time: the bytewise CRC-32 takes 13.7 ms
over the Tokyo MVT extract's 7 MiB of tiles where `crc32fast`'s hardware
path took 0.25 ms (gunzip 20.2 → 32.4 ms in all, on an M-series Mac), about
0.2 ms on a large tile in a job worker.

With symbol placement also linked in the meantime, `core.wasm` is now
1,636 KiB raw, **522 KiB brotli**: 72 KiB over the target, 28 KiB under the
ceiling. `harness.wasm` is 544 KiB brotli. Of the brotli bytes, about 100 KiB
are data (WGSL, Unicode and CRC tables, strings) and the rest code.

Measured and not taken, against the 522 KiB build:

- **MLT decoding** (`mlt-core`, `fastpfor`, `fsst-rs`, `geo`) is 236 KiB raw,
  50.5 KiB brotli. Two ways to shrink it:
  - A looped bit-unpacking kernel in `fastpfor` instead of its unrolled one
    (`decode_page_scalar`, 150 KB before wasm-opt) saves 118 KiB raw,
    15.6 KiB brotli. It belongs upstream (`fastpfor` on wasm has no SIMD
    kernel, so it uses the scalar one); a local fork is not worth carrying.
    Its effect on decode speed is not measured.
  - A lazily loaded `mlt.wasm`, fetched by job workers when a source is MLT
    (plan §14.3's lazy-module pattern), would take the whole 50.5 KiB out of
    `core.wasm`, less a few KiB to receive decoded tiles. A standalone build
    of `orb-tile`'s MLT decoder is 298 KiB raw, 76 KiB brotli, plus 1.6 KiB
    of JS glue: MVT pages would load about 47 KiB less, MLT pages about
    27 KiB more and one more request before their first tile. It needs a
    second wasm module, a wire format for `TileFrame` and a loading path in
    the job workers, so it is left for the owner to weigh on its merits.
- **WGSL comments and indentation** are 18 KiB of the 66 KiB of shader source;
  stripping them at build time saves about 4.7 KiB brotli but needs a build
  script and shifts the line numbers in shader errors.
- **Unicode range tables:** `unicode-bidi`'s `(char, char, BidiClass)` table
  is 18 KiB raw at 12 bytes a row; `orb-text`'s generated `(u32, u32)` tables
  are about 28 KiB raw. Repacking either saves a few KiB brotli; the first is
  upstream's, the second churns text code for little.
- **Formatting machinery:** `Debug`/`Display` impls are about 36 KiB raw and
  float formatting 14 KiB, but most are reached through dependencies (`mlt-core`'s
  error type alone is 10 KiB) or needed by diagnostics and `to-string`; taking
  `format!` out of first-party error paths would not unlink them.
- **Duplicated generic code** (118 sort instantiations, 51 KiB raw; drop and
  clone glue) compresses well: removing the evaluator's second copy saved
  34 KiB raw but only 2.1 KiB brotli.
- **Toolchain:** `opt-level = "s"` is 42 KiB brotli larger than `"z"`;
  `wasm-opt --converge` (already on) saves about 2 KiB.

## Plan to meet 550 KiB by M5

M2's exit accepts an initial payload over the 450 KiB target with a recorded
plan to meet the 550 KiB ceiling by M5. At M2's end,
`core.wasm` is 2,003 KiB raw, **627 KiB brotli**, 77 KiB over the ceiling.
Since the 522 KiB build above: drawing labels (glyph atlas, placement, the
symbol pass) +40, M2's fixture work, world copies and the placement
worker's codec +14, GPU bytecode paint (decision 0017) +17, the reader for
the lazily loaded Southeast Asian word data +11, local glyphs (the canvas
rasterizer and its plumbing) +9, `font-faces` (calling `shaper.wasm`) +7,
the semantic snapshot +4, and other M2 follow-ups +3. Code before wasm-opt
(2,486 KiB) by crate: (generic) 19.1%, `core` 15.8%, `alloc` 11.4%,
`orb-runtime` 7.0%, `orb-expr` 6.8%, `hashbrown` 6.1%, `fastpfor` 6.0%,
`orb-v8` 4.2%, `mlt-core` 3.8%, `orb-symbol` 2.4%.

What M3–M5 add to the initial payload is mostly host code (interaction and
camera animation, queries, the web package's API, device-loss handling); the
heavy M3–M5 pieces are already lazy by design (`compat.wasm` in M4, plan
§14.3). Allow 15–25 KiB for it.

The plan, in order of size and certainty; each step is built when its
milestone needs it, not as size-only machinery now:

1. **Lazy `mlt.wasm`** (measured: −50.5 KiB from `core.wasm`, less a few KiB
   of glue to receive decoded tiles). Job workers load it with the first MLT
   source, as the lazy shaper loads with the first `font-faces` style. MVT
   pages load about 47 KiB less; MLT pages one more request and about 27 KiB
   more in all. It needs a `TileFrame` wire format and a module-loading path
   in the job workers, the same path `shaper.wasm` now uses (step 4), by
   M4. **627 → ~579.**
2. **The looped `fastpfor` kernel, upstream** (measured: −15.6 KiB). With
   MLT lazy it shrinks `mlt.wasm` rather than `core.wasm`; propose it
   upstream with a decode-speed measurement. No local fork.
3. **Placement in the placement worker's own module.** In worker mode the
   render worker never runs `orb-symbol`'s placer, yet `core.wasm` carries it
   for `thread: "main"`. A `placement.wasm` that both modes load (the worker,
   or the main thread when placing inline) takes placement out of the initial
   payload. Estimated from twiggy's share of `orb-symbol`'s placement code:
   −6 to −10 KiB. By M5. **~579 → ~571.**
4. **Lazy text modules stay lazy, and take their readers with them.**
   HarfRust and skrifa (`shaper.wasm`, 405 KiB measured) and the Southeast
   Asian word-break data (569 KiB) load on first use and never enter the
   initial payload. The word data's reader (ICU4X's postcard
   deserialization, +11 KiB) moves into a `seasia.wasm` with the data, as the
   shaper's code lives in its module: a word-break call from a job worker
   goes through it. By M5. **~571 → ~560.** The text core (UCD tables, bidi,
   Arabic shaping, PBF, EDT; 45 KiB) and the local-glyph rasterizer stay
   initial: every label needs the first, and the default
   `localIdeographFontFamily` draws CJK with the second.
5. **Small cuts if the allowance runs out:** stripping WGSL comments at build
   time (measured −4.7 KiB), repacking `orb-text`'s UCD ranges (a few KiB),
   and taking first-party `format!` out of hot error paths.

With steps 1, 3 and 4, `core.wasm` lands near 560 KiB before M3–M5's host
code (15–25 KiB), so about 575–585: 25–35 KiB over the ceiling. Step 5
recovers about 8. The rest needs one more split, measured before M5:
compiling paint bytecode in the job workers (which already hold the style)
rather than the render worker, or a module for the job side's decoding and
layout that the render worker never instantiates. M2 ends with this gap
known: meeting 550 by M5 is a plan with a 20–30 KiB step still to be
chosen, not yet a measured certainty. The 450 KiB target is not reachable
without splitting the renderer itself; the ceiling is the M5 commitment,
the target stays a direction (plan §14.3).

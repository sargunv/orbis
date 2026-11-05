# 0009: The text pipeline

Status: decided at M2 phase 1. `orb-text` implements the job-side
stages below. Phase 2 added symbol layout (`orb-symbol`, [Symbol
layout](#symbol-layout)), then the [runtime text stage](#runtime-text-stage),
[placement](#placement) and [drawing labels](#drawing-labels).

## Question

Plan §12 fixes the shape of the pipeline (itemize → fonts → shape → wrap →
rasterize/pack) and §5.3 fixes where code may live. Four things it leaves
open are decided here. First, where each stage runs and what data crosses
between stages. Second, how glyphs are identified. Third, how rasterization
and atlas packing are deduplicated when web job workers share nothing.
Fourth, what fixture parity with the pinned GL JS (`2b68611a`) requires.

## What the pinned GL JS does

The pin is newer than the GL JS the plan surveyed. Three changes matter
for identity:

- **Grapheme clusters are the unit of layout.** `TaggedString` splits text
  with `Intl.Segmenter` and then rejoins clusters at virama/ZWJ and before
  spacing marks (`canCombineGraphemes`). Line breaking, trimming and bidi
  reordering all count in clusters.
- **Bidi is built in.** `bidi-js` supplies embedding levels; GL JS applies
  rules L1 to L3 itself, reordering whole clusters. Arabic presentation-form
  shaping is built in too (`arabic_shaping.ts`, generated from UCD 17.0.0).
  The RTL plugin survives only as a deprecated override.
- **Only single code points come from glyph PBFs.** `GlyphManager` draws a
  glyph locally with TinySDF on a browser canvas when any of these holds: the
  style has no `glyphs` URL; the key is a multi-code-point cluster; the code
  point is in a complex-shaping block (Indic, Thai, Khmer, …); it is a local
  ideograph and `localIdeographFontFamily` is set; or its PBF range failed.
  A `font-faces` file that covers the code point wins over all of these. Each
  tile also asks for every code point of a cluster, so that a cluster no
  provider can draw falls back to one glyph per code point.

GL JS rasterizes on the main thread and packs a fresh glyph atlas per tile
(`GlyphAtlas` per `WorkerTile`). Orbis does neither.

## Decision

### Stages, crates and lanes

| Stage | Crate | Lane | In → out |
|---|---|---|---|
| Text plan | `orb-runtime` (style) | [S], once per style | v8 layout → glyphs URL template, font stacks, writing modes, `localIdeographFontFamily` policy |
| Text prep | `orb-text` | [J], per symbol layer per tile | `Label` (sections) → transformed `Label` + `GlyphRequests` |
| Glyph resolve | `orb-runtime` job host + `orb-text` | [J] | `GlyphRequests` → `GlyphMap` (metrics + `GlyphKey` per request); suspends on missing ranges |
| Shape + wrap | `orb-text` | [J] | `Label` + `GlyphMap` + `ShapeOptions` → `Shaping` (positioned glyphs, box) |
| Symbol layout | `orb-symbol` | [J] | evaluated `SymbolFeature`s + `GlyphMap` + `TileImages` → `SymbolBucket`: instances, placed symbols, quads, collision boxes, lines, keys |
| Package | `orb-runtime` | [J] → [S], [J] → [P] | symbol streams + tile glyph-key list → scene; collision SoA → placement |
| Glyph raster | `orb-text` (SDF) + job host | [J], one deduplicated class | raster request per provider unit → SDF bitmaps + metrics |
| Atlas | `orb-scene` | [S] | glyph keys → stable table rows; bitmaps → R8 array-texture pages |
| Placement | `orb-symbol` | [P] | collision SoA + camera snapshot → state-word diffs |
| Draw | `orb-render` | GPU | quads + glyph table + state words → SDF pass |

`orb-text` is a job-side core crate: no I/O, no threads, no GPU, no scene
dependencies, and `#![forbid(unsafe_code)]`. Fetching goes through the job
host's `Fetch`, the same path GeoJSON source documents take today (§7.3).

### Data between stages

- **`Label`**: sections of text or inline image, each with an optional font
  stack, scale and vertical alignment. The runtime builds it from
  `orb-expr`'s `Formatted`, so `orb-text` does not depend on `orb-expr`.
- **`GlyphRequests`**: the deduplicated `(font stack, variant, cluster)` keys
  a tile needs, collected exactly as GL JS's
  `calculateGlyphDependencies` does: clusters, their code points, and
  vertical-punctuation substitutes when vertical forms may be used.
- **`GlyphMap`**: each request resolved to `None` (unavailable) or a `Glyph`
  `{key: GlyphKey, metrics, bitmap: [w, h], double_resolution}`. Metrics are
  in 24 px-em units, as in GL JS (`width, height, left, top, advance`).
  They are `f64`, so inline-image sizes divided by a pixel ratio round
  exactly as GL JS's doubles do. The bitmap size and resolution are what
  `orb-symbol`'s quads need from the atlas, and the atlas position is not.
- **`Shaping`**: positioned lines of positioned glyphs. Each glyph has a
  request index or an inline image index, plus `x, y, vertical, scale,
  section`, and metrics copied in. It also carries the text box and the
  writing mode. Positions are `f64`, the same doubles GL JS computes;
  `orb-symbol` quantizes when it writes vertex streams.
- **Tile result**: the tile's `GlyphMap` keys in `GlyphId` order, and
  quads that refer to them by that `GlyphId` (unresolved entries are never
  referenced); images likewise by their `TileImages` index. The scene maps
  each index to an atlas row on upload, as decision 0008 does for images.
  Quads never contain atlas coordinates. Their texel rectangles are
  relative to the glyph's or image's rectangle with one texel of padding on
  every side, so glyph rows carry that padding, as GL JS's glyph atlas
  does.

### Glyph identity

Plan §12's two key kinds become three:

```rust
enum GlyphKey {
    /// A glyph from a v8 `glyphs` PBF range, as served: one code point.
    Pbf { stack: FontStack, code_point: u32 },
    /// A glyph of a font file (style `font-faces`, host system fonts).
    Font { font: FontId, glyph: u32, variant: Variant },
    /// A cluster the browser draws with a font the page cannot read.
    Cluster { stack: FontStack, variant: Variant, text: Box<str> },
}
```

- `Pbf` is new. A PBF glyph is identified by its stack and code point, and
  its SDF is fixed by the server.
- `Font` is the plan's `FontGlyph`. `FontId` is a hash of the font file's
  bytes plus its face index, so two workers loading the same file agree.
- `Cluster` carries the cluster text instead of the plan's hash. Identity
  then cannot collide, and the scene interns it to a `u32` row anyway.

`Variant` is `Default | Vertical`. GL JS asks for the vertical variant only
when the stack has `font-faces`; Orbis does the same for parity.

### Deduplication and ownership

- **The scene owns the glyph atlas.** One per `Renderer`, as in decision
  0008: a stable glyph table (`page, rect, scale, sdf_kind`) and R8
  array-texture pages filled by a shelf packer. This dedups packing by
  construction: one owner, no cross-worker races, and no per-tile atlas.
  A key seen for the first time gets a reserved transparent row
  immediately. Its bitmap fills that row later, with no geometry rebuild.
- **The scene requests rasterization.** When an uploaded tile names keys
  with no bitmap, the scene issues one raster request per provider unit:
  a PBF range `(stack, range)`, a font `(FontId, glyphs…)`, or a batch of
  clusters. It sends the request to job worker `hash(unit) mod N` (§7.3
  affinity). Because only the scene issues requests, each glyph is
  rasterized once per runtime, however many tiles and workers use it.
- **Job workers learn metrics without the atlas.** A glyph's metrics are a
  pure function of its provider data, so each worker derives them itself:
  - PBF: the job host fetches the range, and `orb-text` parses it into a
    metrics table and drops the bitmaps, unless this worker owns the range's
    raster unit. Fetches are deduplicated per worker, as GeoJSON documents
    are. Natively, `PoolJobs` shares one catalog across its threads.
  - Font bytes: skrifa reads advances and bounds; HarfRust supplies shaped
    advances.
  - Web clusters: the worker's own `OffscreenCanvas` `measureText`. These
    are the same values TinySDF derives its metrics from, so they match the
    owner's rasterization.

  Metrics are replicated (about 2 KiB per PBF range); bitmaps are not. A tile
  whose ranges are still loading reports them as dependencies, and the job
  host resumes it when they arrive, as GL JS's worker awaits `getGlyphs`.
- **Failure resolves; it never stalls.** A range that fails falls back to
  local drawing, as in GL JS, and the runtime reports a warning.

### Font chain

`orb-text` ports GL JS's provider choice exactly. A pure function picks the
provider for each request:

1. `font-faces` file covering the code point, if the style declares any.
2. Local: no `glyphs` URL, a multi-code-point cluster, a complex-shaping
   block, or a local ideograph with `localIdeographFontFamily` set.
3. PBF range `code_point / 256` of the `glyphs` template. GL JS
   substitutes `{fontstack}` (stack names joined by `,`) and `{range}`
   (`start-end`); Orbis does the same.
4. Local again, if the range fails.

"Local" means the host's `LocalGlyphs` service (`orb-host`): it measures
a text in a CSS font and returns its coverage, as TinySDF reads a canvas.
Natively, `SystemFonts` (`orb-host-native`) finds the face with fontdb,
falling back per code point to the platform's script fonts as a browser
does, and draws it with HarfRust, skrifa and zeno. On the web,
`CanvasGlyphs` (`orb-host-web`) draws on an `OffscreenCanvas` in the job
and render workers. The font is GL JS's: the stack's names plus
`sans-serif`, weight and style sniffed from the names
(`LocalFont::for_text`), or `localIdeographFontFamily` for an ideograph in
the default stack. That option defaults to `sans-serif`, as in GL JS. This refines plan §12 item 2,
whose order (style fonts → `text-font` → system → PBF) put PBF last. GL JS
consults PBF before system fonts for single code points, and fixture parity
depends on that order.

### Shaping

- **Advance shaping** (implemented). A line-by-line port of `shapeText`,
  `TaggedString`, `determineLineBreaks`, `verticalizePunctuation`, vertical
  orientation runs, bidi L1–L3 and Arabic presentation forms. Clusters a
  provider cannot draw as one glyph fall back to one glyph per code point.
- **HarfRust** (built for font files). A `font-faces` file draws each
  grapheme cluster as one atlas glyph, a `Font` key: HarfRust shapes the
  cluster with its default features and guessed script and direction,
  skrifa outlines it unhinted, zeno fills its coverage at 48 px, and
  TinySDF's EDT makes the SDF, as GL JS's `font-faces` path draws one
  cluster on a canvas. Native hosts use system fonts the same way for
  local glyphs. Whole-run shaping (kerning, cursive attachment across
  clusters) would change fixture output and is a later opt-in. Line
  breaking stays in clusters, so `determineLineBreaks` does not change.
- **`font-faces`** (built). `orb-v8` parses the declarations: a URL, or
  URLs with CSS `unicode-range`s; entries without a URL or a valid range
  are dropped with GL JS's warnings. The job side asks for a declared file
  before the glyphs URL: each name of the stack in turn, each of its files
  covering the cluster's first code point, skipping files that failed. A
  tile suspends until its files arrive, as for PBF ranges. The host's
  `FontFiles` service opens them (TrueType, OpenType, WOFF and WOFF2 via
  `wuff`) and measures and covers clusters. Natively it is `orb-text`
  linked in (`orb_host::FontFileFaces`). On the web it is `shaper.wasm`
  (`orb-shaper`), which a job worker imports the first time a file opens,
  so maps without `font-faces` never download it. A raster job names the
  file, and its owner worker opens the file if a tile hasn't. The
  `'vert'` alternates GL JS draws for vertical text with `font-faces` are
  not drawn: vertical variants stay unavailable.

### Segmentation

- Graphemes: ICU4X `GraphemeClusterSegmenter`, with compiled data, plus GL
  JS's `canCombineGraphemes` correction.
- Bidi: `unicode-bidi` (Unicode 16) for levels and paragraphs. GL JS's L1–L3
  logic is ported on top, with `bidi-js` behaviour as the oracle.
- Character properties: generated tables from UCD 17.0.0, the same version
  GL JS uses (`cargo xtask unicode` writes `orb-text/src/unicode/tables.rs`).
  They reproduce GL JS's derived sets: RTL and cursive scripts,
  complex-shaping blocks, ideographic breaking, local ideographs, vertical
  orientation, wordless scripts, joining types, presentation forms and
  mirroring. They also cover the JS regex classes it uses (`\s`, `\p{P}`,
  `\p{S}`, `\p{Nd}`, `\p{Lu}`, `\p{Mc}`, `\p{Cf}`, `\p{sc=Arab}`).
  To validate them, GL JS's own `build/generate-unicode-data.ts` was run on
  `@unicode/unicode-17.0.0`, and every predicate was compared over all
  1,112,064 scalar values. They match exactly, as do the joining types,
  presentation forms and lam-alef ligatures.
- One GL JS quirk is reproduced on purpose. The npm package lists
  `Unknown` (unassigned, private use, noncharacters) as a script containing
  right-to-left defaults, so GL JS counts every such code point as RTL.
  Inline images are private-use characters, so any label with an image
  takes GL JS's bidi path. Orbis does the same, because fixture output
  depends on it.
- Word breaks: GL JS consults the word segmenter only between two characters
  of a script written without spaces (Line_Break=SA, Balinese, Javanese,
  Tibetan). `orb-text` uses ICU4X's dictionary segmenter for these, loading
  only the Southeast Asian dictionaries and never the 5 MB CJ dictionary.
  Without that data no break is offered inside such runs, which matches GL
  JS without `Intl.Segmenter`. The runtime lays out again once the data
  arrives. Natively the data is compiled in (`orb-text`'s
  `sea-dictionaries`, a native-target dependency of `orb-runtime`). On the
  web, `cargo xtask web` writes the same dictionaries as postcard into
  `seasia.bin` (`sea_dictionary_bytes`, a small container ICU4X reads
  through a `BufferProvider`); a tile whose labels need word breaks reports
  its URL with its glyph ranges (`Built::Fetch`), the job worker fetches it
  once and shares it across styles, and the tile is laid out again
  (`WordBreaker::from_bytes`). The loaded data breaks words exactly as the
  compiled data does (tested on Thai, Lao, Khmer and Burmese). ICU4X's
  dictionary is not the browser's ICU4C engine: for `ราชอาณาจักรไทย` it also
  breaks after `ราช`. GL JS's test only asks for the break after
  `ราชอาณาจักร`, which both find.

### Identical across platforms

With the same inputs (font bytes, PBF ranges, SE-Asian data), these produce
identical results natively and in wasm: itemization, bidi, segmentation,
shaping, line breaking, glyph keys, metrics, quads, collision features and
placement. Layout uses only IEEE `f64` arithmetic, with JS `Math.max`/`min`
NaN semantics reproduced, and no platform maths. PBF SDF bitmaps are copied
byte for byte. SDFs that Orbis computes from outlines with the shared EDT are
identical too. Only browser-canvas clusters differ, because the browser draws
their alpha; those fixtures carry `expected-web-*` tags (plan §12).

### Lazy on the web

| Piece | Where | Measured (brotli) | Loaded when |
|---|---|---|---|
| UCD tables, graphemes, bidi, shaping, PBF parse, EDT | `core.wasm` | +46,447 B (+45 KiB) | always |
| SE-Asian dictionaries | `seasia.bin`: ICU4X's four dictionaries as postcard (built) | 569 KiB (1,011 KiB raw); +11 KiB in `core.wasm` to read it | first label with two neighbouring Line_Break=SA characters |
| or SE-Asian LSTM models | ICU4X postcard blob | +299,742 B (293 KiB) | same |
| HarfRust + skrifa + WOFF/WOFF2 | `shaper.wasm` (`orb-shaper`, built) | 405 KiB (1,284 KiB raw); +7 KiB in `core.wasm` to call it | first `font-faces` file |
| Cluster rasterization | `CanvasGlyphs` in `core.wasm` (`OffscreenCanvas`) | +9 KiB with the local-glyph plumbing | always |

The SE-Asian "module" is a data blob, not code; its data is nearly all of
the cost. The dictionary code stays in `core.wasm`. `shaper.wasm` came out
at 2.7–4× the estimate: twiggy finds HarfRust's shaping tables and
skrifa's hinting, autohinting, CFF and COLR paths linked even though
glyphs are drawn unhinted, and the WOFF2 (brotli) and WOFF (zlib)
decoders. It loads only for styles that declare files, so it is reported,
not trimmed yet.

Method: `cargo xtask web` (opt-level z, LTO, `wasm-opt -Oz`),
with brotli at quality 11 and window 22, the gate's settings. Nothing calls
`orb-text` yet, so a temporary exported function in `orb-web` ran its whole
path (requests, provider choice, PBF parse, shaping in both writing modes,
TinySDF), and was then removed. The baseline is 1,271,395 B raw and
413,518 B br. With the text core it is 1,410,024 B raw (+138,629) and
459,965 B br (+46,447), which the gate reports as 449 KiB, just under the
450 KiB target. Twiggy, before wasm-opt, attributes the code to `orb_text`
(45.7 KB), `unicode_bidi` (22.7 KB) and ICU4X (9.8 KB); the rest is tables
in `.rodata`. Two more builds measured the SE-Asian data compiled in:
dictionaries 2,484,234 B raw and 1,064,904 B br; LSTM 1,767,187 B raw and
759,707 B br.

The plan estimated SE-Asian data at +150–400 KB; the dictionaries are 591
KiB. Compressed tries don't compress well. ICU4X's LSTM models are half the
size, but further from the browser's ICU4C dictionary engine, which is what
GL JS's word breaks come from. Choosing between them is the owner's call.
Whichever is chosen loads on demand, so the initial payload doesn't change.

Levers for the 45 KiB, if the initial payload needs them: pack the UCD
ranges as `u32` starts with `u16` lengths (about 28 KB raw now), and move
bidi plus Arabic shaping into a lazy module. That lazy split only pays off
if labels with inline images stop needing bidi, which GL JS parity
currently requires.

### Fixture parity

- **PBF glyphs.** The fixture mirror carries each test's
  `local://glyphs/{fontstack}/{range}.pbf`. Orbis copies their SDFs
  unchanged, so atlas contents match GL JS's byte for byte; layout parity
  comes from the shaping port.
- **Shaping vectors.** GL JS's `test/integration/symbol-shaping` expectations
  (10 JSON files) and their `fontstack-glyphs.json` are vendored into
  `orb-text/tests/data`. They are compared field by field, natively and in
  Chrome.
- **Local glyphs.** `text-local-glyphs/{cjk,missing,no-glyphs}` and
  `text-local-ideographs/cjk-symbols` set `localIdeographFontFamily`. Those,
  plus every fixture whose text has clusters or complex-script letters,
  draw with the CI machine's system fonts through TinySDF. `orb-text` ports
  TinySDF's EDT exactly (`sdf::tiny_sdf`), so given the same alpha coverage
  the SDF matches. The coverage comes from the platform's fonts, so these
  fixtures compare against GL JS's `expected-mac.png` on macOS and differ
  from it by antialiasing only (0.03–0.4% of pixels), natively and in
  Chrome; they are counted as failures and left so.

### Symbol layout

`orb-symbol` (phase 2) ports `performSymbolLayout`, `addFeature`,
`addSymbol` and the bucket's feature preparation (the point-outside-tile
skip, `mergeLines` for `line` placement, the stable `symbol-sort-key`
sort). The leaves are ported line by line: `quads.ts`, `get_anchors.ts`,
`check_max_angle.ts`, `clip_line.ts`, `merge_lines.ts`,
`collision_feature.ts`, polylabel with its `tinyqueue` order,
`variable_text_anchor.ts`, and `shapeIcon`, `fitIconToText` and
`applyTextFit`. It is a job-side core crate over `orb-text` and `orb-tile`.

- **In.** `layout(layer, features, style, tile, context)`. The runtime
  evaluates everything an expression decides. Per layer, `LayerLayout`
  holds the non-data-driven layout values and each size property's
  `SizeKind`: constant, source, camera, or composite with its covering
  stops. Per feature, `SymbolFeature` holds the `Label` (transformed), the
  icon name, the sort key, the stable id (the promoted id, absent for
  clusters), and `FeatureLayout`. That has every data-driven value, with
  `text-size` and `icon-size` sampled at the tile zoom, zoom + 1, zoom 18
  and the composite stops, as GL JS samples them. `add_glyph_requests`
  gives the runtime the tile's requests, including GL JS's
  `needsVerticalForms`. `TileImages` numbers the tile's images.
- **Out.** A `SymbolBucket` of flat `Pod` tables, each one `Vec` of fixed
  rows, so the web copies each once into a transferable buffer (§10.4):

  | Table | Row | Read by |
  |---|---|---|
  | `instances: SymbolInstance` | 96 B: key, anchor, feature, text box scale, circle diameter, height offset, placed symbol per justification / vertical / icon, 4 collision boxes, anchor-offset range, flags | placement |
  | `text`/`icon` `.placed: PlacedSymbol` | 44 B: anchor, segment, writing mode, quad range, line range, packed size, line offset, instance, associated icon | placement, GPU |
  | `.quads: Quad` | 48 B: 4 corners × 32, texel rect, stretch pixel offsets, min font scale, glyph id or image index, placed symbol, section, SDF and image flags | GPU (one instance each) |
  | `.glyph_offsets: f32` | per quad, along-line offset | placement (first and last glyph), GPU compute |
  | `lines: [i16; 2]` | line vertices, one copy per clipped line | placement, GPU compute |
  | `anchor_offsets: AnchorOffset` | 12 B: variable anchor, offset | placement |
  | `sort_key_ranges: SortKeyRange` | 16 B | placement, draw order |

  Quantization is GL JS's: corners `round(v × 32)` as `i16`, boxes and
  anchors truncated to `i16`, texels to `u16`, sizes `× 128` rounded and
  capped. Vertex streams, index buffers, opacity arrays and collision debug
  buffers are not produced; the GPU pass draws instanced quads under state
  words.
- **Keys.** `SymbolInstance::key` is a 64-bit FNV-1a hash of `(source
  layer, stable id, text)`, with numbers that are equal in JavaScript
  hashed equally. GL JS hashes the text with murmur3, mixes in the promoted
  id, and keeps 16 bits; cross-tile matching also compares anchors, so the
  wider key only removes false candidates.
- **Deviations from GL JS**, all deliberate:
  - Line vertices are shared by the anchors of one line. GL JS copies the
    line per anchor, with a `tileUnitDistanceFromAnchor` it never reads.
  - Placed symbols carry one packed size, the vertex form (rounded,
    capped). GL JS's placement reads a truncated copy of the unrounded
    value. The difference is below 1/128 px of font size.
  - GL JS turns a vertical placed text index of 0 into −1
    (`placedTextSymbolIndices.vertical || -1`); `orb-symbol` keeps the index.
  - Indices are `u32`, so there is no 65,535-glyph cap or warning. NaN sort
    keys sort as 0, where JavaScript's comparator gives no order.
  - Globe line subdivision (`subdivideVertexLine` with a granularity) is
    left to M7. Mercator's granularity is 0, which copies the line.
  - `orb-tile` drops zero-area rings and repeated closing vertices that GL
    JS keeps. Only line labels along such degenerate rings could differ.
- **Identity.** Trigonometry goes through `libm`, so natively and in wasm
  the same code computes every `sin`, `cos` and `atan2`. Against V8's
  `Math`, the golden vectors agree bit for bit.
- **Goldens.** `orb-symbol/tests/data/generate.ts` runs GL JS's own
  functions in Node at the pin and writes the vectors (2.2 MB). It covers
  1,095 cases of 12 leaf functions, and 21 whole-layout cases.
  The whole-layout cases run `performSymbolLayout` on the mirrored tiles
  `14-4357-6245` and `14-9579-5520` with the mirrored PBF glyphs. They
  cover point, line, line-center and polygon placement, overscaling,
  repeated text, variable anchors and anchor offsets, `icon-text-fit` with
  nine-patch stretching and proportional text fit, vertical point and line
  labels, inline images, sort keys, and source, camera and composite sizes.
  Every instance, placed symbol, quantized vertex, collision box, glyph
  offset, line vertex and anchor offset matches GL JS exactly: 45 tests,
  natively and in headless Chrome.
- **Web size.** Measured like phase 1, with a temporary export in `orb-web`.
  The text path alone (requests, provider, PBF parse, shaping
  in both modes) adds 148,555 B raw and 47,024 B br to the 1,288,106 B raw
  and 420,050 B br baseline. Adding `orb_symbol::layout` adds another
  46,683 B raw and 16,527 B br (16.1 KiB). Before wasm-opt, twiggy
  attributes 44.4 KB to `orb_symbol` and 28.9 KB to `libm`.

### Runtime text stage

`orb-runtime` (step 2) builds `orb-symbol`'s inputs in the tile job, and
`orb-v8` types every symbol layout and paint property of the pinned spec.

- **Nodes.** A symbol layer is two render nodes: its icons (layer key × 2)
  and then its text (× 2 + 1), the order GL JS's `drawSymbols` draws them
  in. Both nodes share one bucket per tile. Each node carries per-feature
  property rows over the shared row machinery: opacity, colour, halo colour,
  halo width and halo blur, 10 words a row. A data-driven paint edit is
  then a column job, as for the other families. The `InstanceRows` table maps
  each symbol instance to its feature's row.
- **Evaluation.** Before any node is laid out, the job evaluates each symbol
  layer's selected features as `SymbolBucket.populate` and
  `performSymbolLayout` do. That covers `text-field` (plain strings resolve
  `{tokens}`), the label with `text-transform` and Arabic shaping,
  `icon-image`, the sort key when `sortFeaturesByKey` holds, the stable
  promoted id, and every data-driven layout value at the tile zoom. Sizes
  are sampled at the tile zoom, zoom + 1, zoom 18 and the covering stops.
  The layer-wide layout follows `SymbolStyleLayer.recalculate`. `SizeKind`
  comes from the property's kind and zoom curve (`getCoveringZoomStops`).
  Against GL JS, the runtime's inputs are exact on all 22 whole-layout
  golden cases: 776 features, every value bit for bit.
- **Glyphs.** Each worker keeps a catalog of PBF metrics by range URL. Its
  threads share it natively, and so do a style's revisions. A tile whose
  labels need ranges the catalog lacks returns `Built::Glyphs(urls)`. The
  job host (`PoolJobs`, or the web `JobWorker`) fetches each URL once per
  style, adds it, and builds again from the cached tile. A failed range
  leaves its glyphs undrawn and warns. Glyphs that need a local font
  (clusters, complex-shaping scripts, local ideographs, styles without
  `glyphs`, failed ranges) are measured by the job worker's `LocalGlyphs`
  and cached per font and text; the raster owner draws their bitmaps on
  its own `LocalGlyphs`, as it decodes PBF bitmaps. A host without local
  fonts leaves them undrawn and warns once.
- **Sprites.** The map sends the sprites' metadata (`SpriteIcons`, with the
  device pixel ratio) to the jobs once every sprite has resolved. Sources
  with symbol layers request no tiles before then, as GL JS's worker awaits
  `getImages`.
- **Package.** A layer's bucket travels as 11 word streams: instances; text
  placed symbols, quads and glyph offsets; the same three for icons; lines;
  anchor offsets; sort-key ranges; and instance rows. With them go the
  bucket's scalars, the glyph keys by tile-local id, and the image names.
  Across web workers they ride in the packed tile after the meshes.
- **Deviations.**
  - Overscaling is 1. Orbis slices an overzoomed tile to its own extent, so
    every tile lays out at its own scale. GL JS lays out the whole ancestor
    at its overscaled size, so labels near a child tile's edges can differ.
    A slice keeps lines a tile past its square, so line labels anchor from
    the line's start when that lies within a tile.
  - No Southeast Asian word breaks: no segmentation data is loaded, which
    matches GL JS without `Intl.Segmenter`.

### Placement

`orb-symbol::placement` (step 4) ports GL JS's `placement.ts`,
`collision_index.ts`, `grid_index.ts`, `cross_tile_symbol_index.ts` and
the parts of `projection.ts` that collision circles need. `Placer` keeps
the ingested buckets (`PlacementTables`: instances, placed symbols, line
vertices, text glyph offsets, anchor offsets, sort-key ranges; never
quads), the cross-tile index and the last commit. `place(&Frame)` runs one
placement against an immutable camera snapshot. Each drawn tile carries
its tile-to-clip matrix and its text size at the camera's zoom
(`ZoomSize`), which the scene side evaluates, so placement needs no
expressions. It returns the buckets whose state changed.

- **Order.** As in GL JS: layers from the top of the style down, each
  layer's tiles finest zoom first, then by wrap, column and row. Sort-key
  layers place by key across tiles; `viewport-y` layers by rotated anchor.
  Overlap modes, `*-ignore-placement`, `*-optional`, text and icon
  translation, variable anchors with the previous anchor tried first,
  vertical writing modes, collision groups per source without
  `crossSourceCollisions`, and fades with zoom adjustment all follow the
  pinned GL JS.
- **State.** Three words per placed symbol. The first holds the commit's
  opacity (7 bits, as GL JS packs it), whether the symbol fades in, and
  flags: hidden (another orientation, justification or anchor draws),
  shifted, a quarter turn, and the text slot an icon follows. The other
  two hold a variable anchor's shift in ems of the label's size, which the
  GPU scales by the render size each frame, as GL JS's
  `updateVariableAnchors` does on the CPU. `Committed::fade_change` is GL
  JS's `symbolFadeChange`.
- **Deviations.**
  - The cross-tile index scans a key's symbols instead of building a
    KDBush beyond 128. The scan finds the same match (the first unused one
    within tolerance), so only speed differs.
  - Collision debug output and `queryRenderedSymbols` are not ported yet.
  - No terrain: elevation is the symbol's height offset.

### Drawing labels

Steps 3–5 put labels on screen through one path, natively and on the web.

- **Glyph atlas.** The renderer owns a `GlyphAtlas`: R8 SDF bitmaps in
  growable 512² array pages, shelf packed with one texel of padding and
  one more transparent texel around that, which edge filtering reaches (it
  would otherwise read whichever glyph was packed next, which depends on
  arrival order). Atlas positions still depend on that order, so labels of
  the same view can differ by one level per channel between histories. Its
  row table follows the scene's property arena, 65,536 rows by default.
  The map leases rows by `GlyphKey` per resident bucket. A new PBF glyph
  reserves a transparent row and joins its range's raster job:
  `Jobs::raster`, one per range URL with the code points wanted, run on
  `PoolJobs`' executor or on the web job worker `hash(url) mod N`. Bitmaps
  land in their rows later; nothing that names the row changes.
- **Upload.** Each symbol stream (text, icons) of a tile becomes one
  symbol mesh. Its header and tables (quads, placed symbols, glyph offsets,
  lines, glyph rows by id, packed image rectangles by index, instance row
  slots, the draw order, for icons the size of the text each one is fitted
  to, and for text the colours `format` sections override `text-color`
  with, per flagged quad) sit in the vertex arena. Paint rows, the
  feature-state remap and three state words per placed symbol sit in the
  property arena. Chunks are runs of up to 256 quads; with
  `symbol-sort-key`, each holds one key, and the scene orders a node's
  candidates by key across tiles, as GL JS orders its sort-key segments.
- **World copies.** Each drawn world copy of a tile places as a tile of
  its own, as GL JS's wrapped tiles do: the placer keeps cross-tile ids
  and states per bucket and wrap, re-keyed on wrap jumps. A symbol mesh
  keeps a block of state words per copy, a power of two of them, grown
  when the copies drawn span more wraps; a placement draws the block of
  its wrap (in its tile row) modulo the block count.
- **Lane.** `PlacementLane` hosts `orb_symbol::Placer`: a thread natively
  (`ThreadPlacement`); on the web, a placement worker of its own beside the
  job workers (`WorkerPlacement` in the render worker posts to orb-web's
  `PlacementWorker`, tables, frames and commits packed by
  `pack_tables`/`pack_frame`/`pack_committed`), or the caller's thread with
  `thread: "main"` (`InlinePlacement`). The web always gives placement its
  own worker; plan §7.1's fallback to a job worker on machines with fewer
  than four cores is not built. Each resident bucket's tables are ingested
  once; on the web the render worker reads them from the tile's buffer and
  posts a packed copy to the placement worker. The map rebuilds the placement
  frame every frame: each symbol layer's drawn buckets, with their
  tile-to-clip matrices and text sizes at the camera's zoom. It asks for a
  placement when the frame differs from the last one placed, waiting for
  fades as GL JS's `stillRecent` does. Commits write their changed state
  words. The frame block carries `symbol_fade_change`. A map is idle only
  once labels have placed for the current frame, rasters are in, and
  fades are done. Fixtures run with `fadeDuration` 0, as GL JS's harness
  does.
- **GPU pass.** `symbol.wgsl` ports GL JS's `symbol_sdf`, `symbol_icon` and
  `symbol_text_and_icon` into one shader. The sizes are constant, source,
  camera (capped at the tile's layout size) or composite (between the
  tile's covering zooms). It handles pitch and rotation alignment, map
  rotation in the shader, `u_is_offset`, and halo and fill in one pass. A
  second halo pass runs, as in GL JS, when glyphs may overlap (negative
  letter spacing or a halo wider than 1). Icons in text sample the image
  pages. State words hide symbols and fade them by `symbol_fade_change`;
  a vertex below 0.1 opacity is dropped, as GL JS drops it.
- **Images.** Each sprite image has two rows: a pattern row with wrapped
  padding and an icon row with transparent padding, since icon quads
  cover their padding texel as GL JS's do; icon rows keep one more
  transparent texel around that, which edge filtering reaches. Icons
  sample nearest unless
  GL JS's `iconsNeedLinear` holds (scaled or SDF icons, zoom-dependent
  inline images, pitched views).
- **Draw order.** Where GL JS sorts by viewport y (`symbol-z-order`
  `viewport-y`, or `auto` with overlap allowed and no sort key), placement
  also returns each mesh's quad order and the map writes it to the mesh's
  order table, which the shader reads its quad through. Such a mesh's
  chunks are culled as one, so no chunk breaks the order.
- **Cost.** On the Tokyo demo with Liberty's labels (Apple M6, 2048×1536,
  `cargo xtask bench` medians of submission A against the same views
  without symbol layers), the symbol pass adds 0.012 ms of GPU time at z12,
  0.080 at z15, 0.093 at z15 pitched 60° and 0.107 at z17, and 50 draws;
  CPU render time rises by 0.01–0.02 ms. The bench compares submissions
  without labels, since the indexed alternatives draw none.
- **Labels bench** (`--experiment labels`, Tokyo, 2048×1536). The camera
  turns 0.25° a frame, so every frame asks for a placement.
  `FrameReport::placement_us` times the render thread's share: building the
  frame, comparing it with the last one, and applying commits. The
  placement itself runs on the lane, and glyph rasterization is the jobs'.
  Median / p90 / max per frame, in ms:

  | device | z15 | z15 pitch 60 |
  |---|---|---|
  | Apple M6, native | 0.005 / 0.006 / 0.014 | 0.006 / 0.006 / 0.011 |
  | Apple M6, Chrome (5 µs clock) | 0.010 / 0.015 / 0.060 | 0.010 / 0.015 / 0.055 |
  | Adreno 619, Chrome 156 (moto g 5G 2024) | 0.495 / 0.845 / 1.095 | 0.470 / 0.775 / 1.450 |

  The 0.2 ms budget holds on the M6 and not on the phone (#21). Turning
  writes 3–10 KB a frame (placements and changed state words) and no
  paint; line labels cost the CPU nothing per frame, since the vertex
  shader walks them. Hidden symbols cost 0.04–0.06 ms of GPU time on the
  M6 and 0.8–2.2 ms on the Adreno with nearly every symbol hidden. Showing
  every symbol costs 4–21 ms on the Adreno. Fragment work therefore
  follows the visible symbols; vertex work follows the quads of the
  visible chunks (#9 has the table).
- **Deviations**, all deliberate:
  - Line labels walk their line in the vertex shader, not in a compute
    pass. Each vertex places its glyph from the anchor, with GL JS's
    first-and-last-glyph tests for flipping and for vertical alternatives.
    That costs a few segment projections per vertex, against an
    allocation of glyph transforms, an indirect dispatch list and an extra
    pipeline. It also works unchanged on Compat.
  - Hidden symbols drop their vertices; chunks are culled, quads are not
    compacted by state word. Compaction would add a compute pass per
    commit, and `compact.wgsl` takes about 32 ms on the Adreno 619 when the
    visible set changes (decision 0002). Until that pass is fixed it
    would cost more than the 0.8–2.2 ms it saves (#9).
  - The vertical line label decides "steep" from its own first and last
    glyphs, where GL JS uses its horizontal sibling's.
  - A line label's walk in the vertex shader ignores `text-translate`
    (#12). Its `text-offset` joins segments with GL JS's miter
    (`findOffsetIntersectionPoint`), as the collision walk does.
  - With `symbol-sort-key`, halo and fill draw per quad in one pass:
    GL JS draws a segment's halos, then its fills, so within one key a
    label's halo can cover a neighbouring glyph's fill in Orbis.
  - A NaN `text-size` draws at a fallback size where GL JS draws
    nothing; it no longer keeps placement from settling (#12).
- **Fixtures.** Of the 451 mirrored `symbol-`, `text-` and `icon-` tests,
  408 pass, 15 fail and 28 are skipped (90.5%), with native and Chrome
  pixel-identical (all 1,010 passing fixtures match). The failures are
  listed with their causes in #12. Four are the local-glyph fixtures,
  which differ from GL JS's macOS images by antialiasing, as does
  `text-field/formatted-grapheme-sections` (`font-faces`, 0.46% of
  pixels: zeno's coverage against the browser's rasterizer); the two
  variable-anchor `pitched` fixtures fail intermittently under load
  (#18). GeoJSON keeps nested array and object properties as values.

## Next phases

Each step is built on the interfaces above and lands with its own tests.

1. **Symbol layout**: done, see [Symbol layout](#symbol-layout).
2. **Runtime text stage**: done, see [Runtime text stage](#runtime-text-stage).
3. **Glyph atlas**: done, see [Drawing labels](#drawing-labels).
4. **Placement lane**: done, see [Placement](#placement) and [Drawing
   labels](#drawing-labels).
5. **GPU symbol path**: done, see [Drawing labels](#drawing-labels).
6. **Fixtures**: done for the mirrored tests, see [Drawing
   labels](#drawing-labels). Local glyphs are drawn, and
   `font-faces` files with HarfRust per cluster, in `shaper.wasm` on the
   web.

## Alternatives rejected

- **A glyph atlas per tile** (GL JS): it re-uploads every glyph per tile and
  gives each tile its own texture binding, which conflicts with plan §8.2.
- **Atlas coordinates in quads**: layout would wait on packing, and a
  repack would mean rebuilding geometry (the same argument as decision 0008).
- **Job workers request bitmaps themselves**: N workers would rasterize the
  same glyph. The scene is the one place that knows what is resident.
- **Broadcasting metrics from the raster owner**: it saves at most N−1
  fetches of a ~20–80 KB range (mostly answered from the HTTP cache), but
  adds a message round trip before any tile can shape.
- **`icu_properties` for character classes**: GL JS's sets are derived (block
  unions with exceptions). Generated range tables reproduce them exactly
  and are smaller than the tries the same answers would need.

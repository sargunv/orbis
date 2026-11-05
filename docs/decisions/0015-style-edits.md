# 0015: Style edits without reloads, and feature state

M1 exit criterion 3: one frame per paint edit, no relayout on a binding flip,
feature-state write ≤0.05 ms. Before this, every style change was a full
`set_style_json` reload. This records how edits reach the renderer and the
jobs, where decoded tiles live for relayouts, and how feature and global
state are stored and drawn, with the deviations from plan §10.2 and §11.3.

## The document and its patches (`orb-style`)

`StyleDoc` keeps a v8 style as JSON (root members, sources and layers apart)
and as `orb-v8`'s typed IR, the published snapshot. Layers and sources carry
stable keys: a layer keeps its key through edits and moves, and a layer whose
type, source or source layer changes is a new layer, as GL JS's diff removes
and re-adds it; a source whose definition changes is a new source. A
`Transaction` is a list of edits (paint, layout, filter, zoom range, add,
remove or move a layer, add or remove a source, global state, feature state,
`setStyle`), parsed from GL JS map calls by `Transaction::from_commands`, the
format of the render tests' `operations` and of the web package's `edit`.
`StyleDoc::apply` applies all of them or, on the first error, none; it
re-parses only the layers it touched (`orb_v8::parse_layer`), diffs their JSON
against the old (span-insensitive), and returns a `Patch` of changes, each
with its S6 `Class`: `Block` (uniforms), `State` (state table), `Column`
(property rows), `Relayout` (one layer), `Nodes` (layers added, removed,
moved), `Reload` (a source), `Restyle` (sprite or light changes). Paint
changes carry their old and new `Binding` (`Constant`, `Camera`, `Source`,
`Composite`, `State`). `setStyle` is the same diff against a whole new
document, keeping the keys of unchanged sources and same-identity layers.

**Global state resolves at parse (plan §11.2's S2).** `orb-expr` folds
`["global-state", k]` to the style's value when parsing with a state, so a
value that reads only global state is a constant, and one that also reads
feature data is a source value with the literal in it. A fold that fails on
a global-state value (an unset colour) stays unfolded and falls back to the
default where it's evaluated, as in GL JS. `setGlobalStateProperty`
re-parses the layers whose JSON reads the key and diffs them like any edit:
a paint uniform (block write), a column job for data-driven paint, a
relayout for filters and layout. There is no GPU global-state table: the
values are uniforms or rows already, and jobs fold the same values when they
re-parse the update.

## Jobs, revisions and cached tiles (`orb-runtime`)

Each map style version is a revision. `Jobs::set_style` takes a
`StyleUpdate`: the key order of layers and sources, the JSON of layers and
sources new in the revision, and the global state. In-process jobs share the
map's parsed snapshot; browser job workers apply the update to their previous
revision (`TileWorker::update`). Workers keep each tile's decoded `TileFrame`
per source (an LRU of 256), in the job worker that owns `hash(source key,
tile)` (plan §10.4), with the GeoJSON indexes, across revisions. A
`TileRequest` names the nodes to lay out (`nodes`, by key) and whether only
their property rows (`rows_only`); a cached tile builds without fetch or
decode, an evicted one is fetched again. Results carry the revision they were
laid out with and the node keys they cover.

The map keeps per node key a layout and a rows epoch, the revisions since
which earlier results are stale. A patch applies as:

- **Block**: node uniforms on the next frame.
- **Column**: rows epoch bumped; a rows-only job per resident tile with a mesh
  of the node; `Scene::set_mesh_props` replaces the rows (reallocating when a
  flip to data-driven needs rows the mesh never had) and patches their images
  and state remap. Geometry stays.
- **Relayout**: both epochs bumped; a job for the layer's nodes per resident
  tile; their meshes are replaced in place.
- **Nodes**: new node keys are stale everywhere and laid out from the cached
  tiles. On the next frame `Renderer::remap_nodes` renumbers the nodes,
  keeping every retained node's meshes, uniforms, image pages and ramps; the
  removed nodes' meshes leave first. New nodes of image sources get quads for
  the resident image tiles.
- **Reload**: a removed source's tiles leave the scene next frame; a new one
  starts resolving. Other sources keep their tiles.
- **Restyle**: `install` of the edited document.

A load that arrives laid out before an edit relays out its stale nodes from
the cache. `Map::edit` returns the generation; `FrameReport::generation` is
the newest whose jobs have all landed (plan §7.1), and `idle` waits for them.

## Feature state (plan §10.2)

Per source and source layer, feature id → state index, keyed by GL JS's
`String(id)`: the tile's id column, a vector source's `promoteId` property
(one key or per layer), or GeoJSON's `generateId`/`promoteId`. Every feature
of a resident mesh whose node reads state is interned when the mesh uploads
and counted by the meshes that refer to it; an index with neither references
nor state is freed. A mesh's slot → index remap lives in the property arena,
its base in chunk word 6 (indexed submissions carry it in the run record), so
a state write never touches a mesh.

The state table is a range of the property arena with a row layout compiled
per style revision: per node that reads state, a block of one flag word (non
zero once the feature has state) and eight words per state-lookup property,
its values at the table's integer zooms. The shaders (fill, fill
outline, line, circle, fill-extrusion) read a feature's block through the
remap when the node's `StateLookup` mask has the property's slot, else the
node constant, the value with no state. **A write is one row**: the map
merges the state, and the next frame evaluates the node's state-lookup
properties for that feature and writes its row (4 + 8 bytes per property).

**Deviation from plan §10.2.** The plan's row holds raw state (≤32 booleans,
≤4 f32, string enums) for a GPU evaluator to read. Orbis has no GPU expression
backend yet, and the fixtures' state values are colours
(`["coalesce", ["feature-state", "color"], "black"]`), so the row holds the
evaluated values instead: exact for any state type, at the cost of
evaluating on write and a row size of the state-lookup properties rather
than of the keys. Zoom-dependent values are evaluated at eight integer
zooms, from five below the camera's floor to two above, so drape slots of
other zooms read their own (#85), and rewritten (rows with state only) when
that floor changes; fallback tiles of other zooms use the same rows.

**Binding classes.** `StateLookup` is a value that reads `feature-state` but
no feature properties, id or geometry (`Deps::PROPERTIES` in `orb-expr`).
Values that read both are interpreted on the GPU when their program
compiles, reading raw state values the row holds after the lookups
([decision 0017](0017-gpu-bytecode.md)); the rest draw without state, with
a warning: plan §10.2's CPU re-evaluation of the affected rows in the owner
job worker is left open in an issue
([#6](https://github.com/sargunv/orbis/issues/6)). Heatmap radius
and weight don't read the table.

## The web package

`OrbisMap.edit(commands)` applies a JSON array of map calls as one
transaction and returns its generation; frame acks carry `generation`.
`web/demo/edit.js` builds `edit(tx => …)`, `setFeatureState`,
`removeFeatureState` and `setGlobalState` on it, applied on the main thread
or posted to the render worker (`{type: "edit", id, commands}` →
`{type: "edited", id, generation | error}`). `pnpm -C web edits` drives
every class in both modes.

## Measurements

Apple M-series laptop, macOS, release builds; Chrome headless with WebGPU.

**Feature-state write** (scene thread: the edit call plus the next frame's
time over a frame without a write, a 1600-point GeoJSON grid over the
66-layer Tokyo style):

| Host | Edit call | Write total | Target |
| --- | --- | --- | --- |
| native (`cargo test --release -p orb-runtime --test edits feature_state_write_cost -- --ignored`), medians of 400, four runs | 0.0005–0.0019 ms | 0.0027–0.0097 ms | ≤0.05 ms |
| Chrome main thread (`pnpm -C web edits`), means of 500, three runs | 0.0020–0.0026 ms | 0.003–0.004 ms; the frame part is within the ±0.01 ms run-to-run noise of a 0.03–0.05 ms frame | ≤0.05 ms |

For comparison a uniform paint edit (`water` `fill-color`) costs 0.12–0.21 ms
in the edit call (re-parsing one layer and publishing a snapshot) and its
frame rewrites every node uniform (+0.1 ms).

**Frames and jobs per class** (`crates/orb-demo/src/edits.rs`, Tokyo z15,
1024×768; frames are until the frame shows the edit in full, so they include
waiting for jobs on the native pool; every case's pixels equal a fresh load
of the edited style, zero differing pixels):

| Edit | Class | Frames | Loads | Relayouts | Column jobs | Meshes | Rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paint colour, zoom curve, visibility, background | Block | 1 | 0 | 0 | 0 | 0 | 0 |
| `line-color` constant → `match` on `class` | Column | 3 | 0 | **0** | 5 | 0 | 5 |
| `line-color` back to a constant | Block | 1 | 0 | 0 | 0 | 0 | 0 |
| `setFilter` on `road_minor` | Relayout | 2 | 0 | 6 | 0 | 6 | 0 |
| add a layer, move one, remove one | Nodes | 1 | 0 | 6 (new layer only) | 0 | 6 | 0 |
| add a GeoJSON source and layer | Reload | — | 6 (that source) | 0 | 0 | ≤6 | 0 |
| `setFeatureState` | State | 1 | 0 | 0 | 0 | 0 | 1 table row, 0 paint writes |

## Fixtures

The mirror adds GL JS's `runtime-styling`, `feature-state`,
`remove-feature-state`, `global-state`, `filter`, `visibility` and
`zoom-history` directories at the same pin (1118 render tests, from 883;
1144 with the `sky` directory mirrored alongside).
`orb-fixtures` applies `operations` through `Map::edit` after the first idle,
natively and in Chrome; GL JS's test-server origin maps to the assets like
`local://`; a root `transition` of zero duration is honoured.

| | pass / fail / skip |
| --- | --- |
| Tests with `operations` (250 of the 1144) | **126 / 1 / 123**, native and Chrome alike |
| The 883 mirrored before, before this work (decision 0011) | 352 / 6 / 525 |
| The same 883, after | 368 / 7 / 508 |
| All 1144, after | **519 / 7 / 618** native; 519 / 7 / 618 Chrome |
| Chrome/native identity | 519 of 519 passes match |

The 883 gain 16 passes from global state and operations they already used
(`fill-pattern/global-state`, `line-pattern/global-state`, extrusion global
state); the added failure is the extrusion pattern case below.

Per new directory (native = Chrome): runtime-styling 94/0/83, visibility
25/0/6, feature-state 8/0/1, global-state 6/0/0, filter 4/0/3,
remove-feature-state 3/0/0, zoom-history 2/0/0; `sky` 9/0/17. The one failing operations
test, `fill-extrusion-pattern/global-state` (5.60%), fails identically when
its edited style loads fresh: an extrusion pattern case newly in scope, not
the edit path. Skips of the operations tests are mostly symbol layers, glyphs, `addImage`
and waits for events or milliseconds.

## Size

`core.wasm` grows from 416 to 459 KiB Brotli (+43 KiB; 1277 → 1414 KiB raw),
over the 450 KiB warning and under the 550 KiB ceiling, with the cuts taken since in
decision 0003. About 25 KiB is the
runtime's revisions, frame cache, node remapping and state store, the rest
`OrbisMap.edit` making `orb-style`'s transaction parser and diff reachable
(`orb_style` alone is 1.3% of the raw module, `hashbrown` grows 15 KiB raw).

## Open

- Paint reading both feature state and feature data: CPU re-evaluation in
  the owner job worker ([#6](https://github.com/sargunv/orbis/issues/6)).
- Transitions with a duration (plan's `orb-style` row): edits apply at once
  ([#7](https://github.com/sargunv/orbis/issues/7)).
- `fill-extrusion-pattern/global-state`, newly in scope
  ([#5](https://github.com/sargunv/orbis/issues/5)).
- `setStyle` with a URL, `addImage`/`removeImage`, sprite and light edits
  (full reload today), inline sources in `addLayer`.
- A uniform edit clones the snapshot's layers; `Arc` layers would make it
  O(edited).

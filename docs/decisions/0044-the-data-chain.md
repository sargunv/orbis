# 0044: The data chain: one paint path, one scene model, one codec

Decision 0040's data part, as it lands. Each section is added by the step that
builds it.

## No GPU interpreter; mixed paint through the state table

The GPU expression interpreter (decision 0017) is deleted: `orb-expr::gpu`,
`expr.wgsl`, the bytecode pipeline class, `CostModel`, `Binding::Bytecode`,
the program table and the frame's f64 zoom bits. At the measured default
(`CostModel::ROWS`) it ran only for paint that reads both feature state and
feature data, on the fill, line and circle families, and it had a history of
Adreno compiler crashes.

That paint now takes the path state lookups already took, which GL JS 5.24
(`SourceCache.prepare` → `bucket.update` → `updatePaintArrays`) and MapLibre
Native follow on their render threads:

- The job evaluates it per feature with no state, into the property rows.
- For nodes with such paint, each mesh also carries, per feature slot, an
  `orb_expr::FeatureData`: the feature's id, geometry type and the
  properties those expressions read (all of them when one reads a computed
  key or `properties`).
- Interning a mesh's features stores that data on the feature's
  state-table entry. When the feature has state, its row holds those paint
  slots evaluated on the CPU with its state and data, at the integer camera
  zoom and the next. The shader already prefers a feature's state block
  over its row, so a state write shows in the next frame with no job.
- It works for every family that reads state: fill, fill outline,
  extrusion, line, circle and both symbol nodes. The warning "drawn without
  state" is gone.

`orb-demo`'s `paint_reads_feature_state_and_data_together` checks a
`case(feature-state, match(get), match(get))` circle colour: the frame right
after a hundred state writes is idle, writes one state row each and runs no
job, and its pixels equal the hovered branch drawn alone.

Removed: about 3k lines, and one pipeline class. A culling margin for such
paint now uses the stateless rows' maximum, as state lookups use the
stateless constant.

## Table-driven layer properties

`orb-v8`'s layers were one hand-typed struct per layer type, with the
property list repeated in the parser, the spec lists, the by-name paint
table (decision 0031), the bytecode list, and the renderer's source, filter,
source-layer and pattern matches. Now one table (`orb-v8/src/spec.rs`, the
`properties!` macro) lists every layer property once: its layer type,
section, value type, default, range and flags (data-driven, per-light list,
colour ramp). A `PropertyId` indexes it, and a typed `Key<T>` names each
entry (`FILL_COLOR: Key<Color>`).

- A `Layer` is `{ ty, source, source_layer, filter, minzoom, maxzoom,
  visibility, props }`: one untyped `Slot` (a constant `Value`, an
  expression, or a per-light list) per property of its type, in table
  order. `layer.get(KEY)` reads one as its type, with the table's default.
  `layer.list` and `layer.ramp` read the per-light and ramp properties.
- Parsing, unknown-key validation, `property_spec` and the diff read the
  table. A layer type Orbis doesn't model can't occur: every v8 type is in
  it.
- Transitions hold the prior `Slot` and ease `Value`s (numbers, colours,
  pairs) for properties the table marks as transitioning. They skip a
  replaced style's layer of another type under the same id.
- The fixed light, sky and terrain fields keep an owned typed
  `Property<T>` over the same `Slot`.

A constant that doesn't fit its type, NaN included, now falls back to the
default, as an evaluated value does: `text-size/nan` passes. Reviewed with
two Sol rounds.

## One native document: `orb_style::Scene`

The style lived in `orb-v8` as the typed IR, with `orb-style`'s `StyleDoc`
on top of it, and the scene document's projection, tile matrix sets, lights
and atmosphere held as fields on `Map` and merged into what each frame drew.
Now:

- `orb-style` is the native document: `Scene` (the IR, renamed, with the
  scene's `projection` and `tile_matrix_sets`), its table-driven layers,
  sources, lights, atmosphere, terrain, sprites and fonts. It depends on no
  front end.
- `orb-v8` is the front end. It parses v8 and scene documents into a
  `Scene`, and keeps the JSON documents of record: `StyleDoc`, its
  transactions and its diff moved here from `orb-style`.
- A scene document decorates its style: `StyleDoc` keeps what it sets
  (projection, low-zoom surface, tile matrix sets, lights, atmosphere) and
  applies it to every snapshot, on parse and after each transaction. The
  style's globe stays under the scene's, as before.
- `Map` keeps no scene fields: its `Surface` is derived from the scene at
  install, and frames read the scene's lights and atmosphere. A scene's
  surface now arrives with its style, at `StyleLoad`, rather than when the
  scene document alone has loaded. `set_projection` edits the document.

Render fixtures are unchanged (1196 / 99 / 50 natively against the same
base), and the alps terrain demo's snapshot matches a base run byte for
byte (label fades vary between runs on both).

## Edits: changes by `PropertyId`, typed scene ops

- The edit vocabulary is native: `Change`, `Class`, `Binding`, `Patch`,
  the layer and source keys and `FeatureStateOp` live in `orb-style`. A
  paint change names its `PropertyId` and takes its bindings from the old
  and new layers' parsed slots, not from re-parsing their JSON against the
  spec. A layout change names its `PropertyId` too, and properties the
  table doesn't have no longer make changes.
- The scene document's setters are typed ops: `Op::SetPlane` draws the
  scene on another plane, through `Map::edit` like any other edit.
- GL JS edit calls (`setPaintProperty`, `setFilter`, …) still carry
  JSON, and `StyleDoc` still diffs JSON to find what changed: global state
  folds into expressions at parse (decision 0015), so a global-state write
  re-parses the layers that read it, and their JSON is what it re-parses.
  Values are typed once parsed; the document of record stays JSON.

`pnpm -C web e2e --only=edits` and the `orb-demo` edit tests keep their
classes and job counts; the runtime-styling fixtures pass, 177 of 177.

## Transitions as overrides; the paint program to come

A frame that runs a paint transition cloned the whole style, inline GeoJSON
included, to overlay the eased values. The scene's layers and sources are
now shared (`Arc`), so the overlay copies pointers and only the layers it
eases. Transitions belong to their style: a new style (`set_style_json`,
a style URL, a scene document) ends them (#79), and an edit's restyle keeps
them.

The rest of the paint path waits on the renderer's node layouts, which the
core chain reflects from WGSL (`orb_shaders::layout`, its B5). The
interface this chain proposes for it:

- `Plan` (render nodes and their bindings) moves to `orb-style`, with
  the families it names; `orb-shaders`' reflected layouts are the small
  crate both sides depend on, holding the row and node-uniform structs.
- Each node's paint property lowers to one program: a constant, a zoom
  curve (stops, interpolation and outputs; GL JS's `interpolate` and `step`
  over `zoom`), or a column of the node's typed rows. Camera expressions a
  curve can't hold stay CPU-evaluated per frame, as GL JS evaluates them.
- Node uniforms hold the programs, written on a revision or a transition
  step; shaders evaluate curves at the frame's zoom. One packer writes
  every family's uniforms from the reflected layout.
- A transition step writes the eased constant into the node's program, so
  the overlay above goes too.

## Sparse columns and a byte budget for frames

A property column keeps a presence bitmap and only its present values,
indexed by the bitmap's rank (a count per 64 features), so `Column::get`
stays O(1) and absent cells take a bit. The layer builder collects
`(feature, value)` pairs rather than one slot per feature for every key.
Workers keep decoded frames in one LRU per worker, keyed by source, the
revision its data began at and tile, within 64 MiB by `TileFrame::bytes`,
in place of 256 frames per source.

Frame bytes against the encoded tiles' bytes (`orbis render`, Tokyo at z12
and z14): the Tokyo extract 5.59× → 3.31× (MVT) and 6.73× → 3.99× (MLT),
OpenFreeMap Liberty 5.44× → 3.30×, VersaTiles 3.34× → 2.57×. Decoding the
same tiles: 98 → 51 ms, 146 → 55 ms, 191 → 119 ms and 89 → 46 ms. Tile
builds are unchanged within their noise.

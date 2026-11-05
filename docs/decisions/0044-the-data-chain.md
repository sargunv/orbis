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

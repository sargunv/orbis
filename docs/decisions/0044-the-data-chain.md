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

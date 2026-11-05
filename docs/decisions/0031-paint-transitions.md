# 0031: Paint transitions over by-name paint in the Style IR

v8 eases a paint change over the style's root `transition` or a layer's
`<property>-transition` (default 300 ms, no delay). Edits through the API
should animate the same way. This records how transitions sit on the
typed IR, and the by-name access they need, which is the native model's
first shape for addressing a paint property.

## By-name paint access

**Superseded by [decision 0044](0044-the-data-chain.md):** properties are
addressed by `PropertyId` in one table, and transitions ease untyped
values. The by-name table below is gone.

The Style IR keeps paint typed per layer kind (`Property<f32>`,
`Property<Color>`, …). `LayerKind::paint(name)` and `paint_mut(name)`
(`orb-v8/src/paint.rs`) address a property by its spec name and return it
as one of four shapes: number, colour, optional colour or pair
(`PaintRef`, `PaintMut`, and the owned `PaintValue`). One table maps names
to fields. Nothing is copied into a second, untyped representation: a
by-name read or write reaches the typed field the renderer already uses.

The table covers the properties that interpolate. Discrete values
(patterns, enums, `*-translate-anchor`) change at once, so they aren't in
it. Later by-name needs, such as the scene document's per-layer
overrides, extend this table, not a parallel map.

## Transitions

- **Parsing**: `Style.transition` is the root `transition` over the
  default. `Layer.transitions` holds each `*-transition` as an override
  (duration and delay, each optional) of the root.
- **Start**: an edit's `Change::Paint` (from `setPaintProperty`, a style
  diff, the editor or the scene document) starts a transition from the
  property's value before the edit. A zero duration and delay changes at
  once. A transition already running for that property restarts from its
  current eased value, as GL JS's chain of priors does.
- **Each frame**: while transitions run, the runtime clones the plan's
  style and overwrites each transitioning property with a constant: the
  prior and new values at the camera's zoom, eased by GL JS's
  `easeCubicInOut`. Paint writes from that clone, then drops it. Numbers
  and pairs interpolate linearly and colours premultiplied, as in GL JS.
  The map isn't idle while a transition runs.
- **Per-feature values**: properties whose prior or new value depends on
  feature data or feature state change at once. GL JS doesn't animate
  feature-state changes either: `setFeatureState` takes effect on the next
  frame, so this is the same behaviour.

Data-driven transitions (GL JS interpolates `possiblyEvaluated` values
per feature) are left out until a style needs them. They would need both
values in the per-feature paint buffers.

## Cost

A running transition deep-clones the style and rewrites all paint once
per frame. It writes no tiles or bundles. The clone of the Tokyo demo
style (90 layers) takes 240 µs, measured in release on an M-series Mac
over 100 clones. If that shows up in frame times, share layers behind
`Arc` and clone only the transitioning ones. When no transition runs,
nothing changes.

## Fixtures

`background-color/transition` passes native and in Chrome. All 162
fixtures that declare a transition pass natively, including
runtime-styling at 163 of 163. Fixture runners read a frame
mid-transition, because GL JS harness time moves only with its waits.

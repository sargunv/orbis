# 0033: Layer blends: one composite, backdrop reads by splitting the main pass

The style spec added `fill-layer-blend` and `line-layer-blend`
(maplibre-style-spec#1798, values from #1950, issue #1661). That was after
our GL JS pin, so the mirror has no fixtures for them. They take seven
modes: `normal`, `multiply`, `screen`, `overlay`, `color`, `luminosity`
and `plus-lighter`. The property is a constant, not transitionable. Each
mode is W3C Compositing 1's. GL JS hasn't implemented them yet
(maplibre-gl-js#48).

## One composite for opacity and blend

Layer opacity already drew a layer into a texture of its own and
composited it once (`layer_composite.rs`, formerly `layer_opacity.rs`).
Blending uses that same composite. A group is a run of one layer's nodes
below opacity 1 or in a mode other than normal. The renderer's API is
`set_layer_composite(node, layer, opacity, Blend)`, and it doesn't care
about the node's family. The IR holds `layer_blend` beside
`layer_opacity` on fill and line paint, and the runtime lowers both.

- **Normal** composites inside the main pass with fixed-function
  premultiplied blending, as layer opacity did.
- **Every other mode** needs the map beneath it, and the target, a
  surface texture, can't be sampled. At each such group the main pass
  splits: what came before draws into a canvas, one of two textures used
  in turn. The next pass opens with a full-screen draw. It reads the
  previous canvas and the group's texture and writes the composited
  result with blending off: `co = cs(1 − ab) + cb(1 − as) + as·ab·B(Cb, Cs)`
  for the separable and non-separable modes, and the clamped sum for
  `plus-lighter`. Then the main bundle's next segment draws on top. The
  last pass draws into the target, so a map without such groups keeps
  its one pass and allocates no canvas.

One shader computes every mode exactly, in the same space as the rest of
the map's blending. Fixed-function blending could do `screen` and
`plus-lighter` exactly, and `multiply` only over an opaque backdrop, but
not `overlay`, `color` or `luminosity`. One path that reads the backdrop
is simpler than two.

## Cost

Each blending group adds a full-screen draw and a pass break. While any
group blends, the map holds two canvases the size of the target: 15 MB
at 1200×800 CSS px and pixel ratio 2, in RGBA8. Normal blending and plain
layer opacity cost what they did.

## Beyond fill and line

The renderer composites a node of any family, so the scene document's
`blend-mode` can lower onto the same call for any layer, raster and
hillshade included (plan §11.4, the composition module with z-groups).
It isn't parsed yet. Nothing needs it before that module, and it should
come with z-groups, which would composite a group of layers through the
same path.

On terrain, a draped layer that composites once gets a drape stack of its
own (decision 0019). The stack's terrain layer, drawn into its texture as
every stack's is, becomes the group's texture, composited at the layer's
opacity and in its blend in place of the stack's plain composite.
`fill-layer-opacity/terrain` and `line-layer-opacity/terrain` now pass.
The cost is a drape slot per terrain tile and a terrain pass for each such
layer.

## Validation

- **Spec test**: `layer_blends_parse` ports the style spec's
  `layer-blend` validation test. Every mode parses on both properties,
  and `erase` (removed from the spec) is an error.
- **Render test**: `layer_blends_follow_w3c_compositing` renders pairs of
  blending layers over an opaque base. That exercises both canvases and
  two splits. Each pixel matches the W3C formulas computed on the CPU
  within 2/255, and pixels outside the layers are untouched.
- **Fixtures**: the layer-opacity fixtures are unchanged: 30 pass, with
  the globe and terrain cases failing as before, native, Chrome and
  lavapipe.
- **Demo**: "Liberty at dusk" applies the blends to live Liberty through
  GL JS calls. It checked out on WebGPU and WebGL2.

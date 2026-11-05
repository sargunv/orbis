# 0047: Model layers: instanced draws, one 3D target per layer, the map's material

The scene document's `model` layer (plan §9.4, §11.4; issue #88) draws glTF
models at point features. This records how it plugs into the narrow
renderer (decision 0043) and the shape its materials take.

## Decision

- **A node per model.** A model layer is one render node per model its
  `model-id` can pick: the one a constant names, else each of the
  document's models. Each holds the layer's points that pick it, as a
  circle node holds its points (`TileBuild`, no second upload path). A
  layer has up to 64 node keys (`style::node_key`), so it draws at most 63
  models, with a warning past that.
- **Exact instances.** A model node draws its model's indices once per
  visible point: a new draw kind, `NodeDraw::Instances`. The cull packs
  the points before each visible chunk into the chunk's visible entry
  (placements fit 16 bits) and fills the list's tail with a sentinel; the
  vertex shader finds its point by binary search (`pull_instance`). Point
  families' 256 padded instances per chunk would run a heavy model's
  vertices 255 times for a lone landmark. A node draws twice, its solid
  indices (opaque and double-sided, contiguous) and its blended ones,
  from 16 draw-argument words per node (was 8): the cull's storage
  bindings are all in use, so both draws' arguments live in the args
  buffer.
- **One 3D target per layer.** A layer's model nodes draw into one colour
  target over the depth 3D layers share, and composite once at
  `model-opacity`, as an extrusion layer does. A target per node cost a
  full-screen multisampled pass and composite each, and blended a
  layer's overlapping models at its opacity twice.
- **The map's material.** The fragment fills a `Material` (`pbr.wgsl`:
  albedo, alpha, metallic, roughness, occlusion, normal, emitted light and
  a self-lit share) and lights it with `pbr_light` under the style's
  `Lights` (decision 0026). It is what a material buffer would hold per
  pixel for one lighting pass over the whole map (#91): any layer's
  material maps onto it. glTF's unlit and `model-emissive-strength` are
  both the self-lit share.
- **Loading.** `Jobs::model` fetches the document, its buffers and its
  images, decodes the images in the core (`orb_host::decode_image`) and
  packs the model (`orb-3d`): on the native pool, and on the web in a job
  worker, whose reply carries the packed model as one transferred buffer.
  The map loads a style's models once per renderer, uploads each by its
  index in the document (`Renderer::set_model`), and isn't idle until
  they've arrived. Compat draws no models and warns.

## Measured

Apple M6, 2048×1536, `orb-bench` over Shiba Park with Liberty's lit 3D
buildings and the demo's three landmarks (1,740 vertices), against the
same scene without them: GPU median +0.18 to +0.67 ms per frame (the
layer's 3D pass and composite), CPU render within noise. The three models
hold 66 KiB of GPU memory. A target per node had cost +0.77 to +2.1 ms.

## Left

Shadows cast by and onto models and labels hidden behind them (R2),
feature state in model paint, `world-3d` placement, and models on the
globe's sphere beyond its local frame.

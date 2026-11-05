# 0048: The lit map: a scene's lights shade every layer's fragments

#91 asks for the whole map to be one lit surface: every layer a material,
one lighting model over all of it. Decision 0026 lit only 3D content
(extrusions, terrain, models) with a scene's lights; flat layers stayed
their v8 colours. Decision 0043 left room for a material buffer.

## Decision

- **A scene's lights light every layer.** As in Mapbox GL JS v3's lights, a
  scene document's `lights` shade the whole map; a layer's
  `emissive-strength` is how much of it shows unlit. A v8 `light` alone
  still lights only extrusions, so v8 styles look as they did.
- **Materials are scene-document properties, by layer.** The scene's
  `layers` member takes `roughness` and `metalness` (0 to 1) beside
  `emissive-strength`; a layer's v8 colours are its albedo. The default is
  matte paint: roughness 1, metalness 0.
- **Forward, in the main pass.** A lit layer's fragment fills the
  `Material` of its colour and its node's material and shades it with
  `pbr_light` under the frame's lights (`map_out_lit.wgsl`), where the
  unlit map draws it. Families' fragments return
  `map_out(colour, node, position)`; a shader's lit composition
  (`Shader::lit`) shades, its classic one passes the colour, so the unlit
  path is unchanged. Node materials are a storage array in the lit
  pipelines' own group 2. Map layers face up and share a pixel's point,
  so blending lit colours is lighting their blended albedo for diffuse
  light; specular differs slightly.
- **Shadows on the ground stay one pass.** The ground pass (decision 0046)
  takes the directional light's share of level ground's lighting away
  where shadows fall, which for matte lit ground is exactly its shadowed
  shading; layers don't each sample the cascades.
- 3D layers, heatmaps and layers composited once arrive shaded and
  composite as they do unlit; draped layers on terrain draw into the drape
  unlit, which the terrain's own lighting shades. Compat draws unlit and
  warns.

## The material buffer, tried first

A deferred version drew the layers into a material buffer (the albedo,
then roughness, metalness and the self-lit share) in a pass of its own,
and one full-screen draw in the main pass shaded it. It cost 0.8 to
2.5 ms more than the unlit map on the bench below: a second full pass with
two attachments written and read back, while the main pass's own cost
(attachments, labels) stayed. A map has a few layers per pixel and one
cheap lighting model, which forward shading suits; it also gave up blend
modes and terrain, which forward keeps.

## Measured

Apple M6, 2048×1536, `orb-bench` over central Tokyo with Liberty's 3D
buildings, the scene's lights without shadows against the same scene
unlit: GPU median 1.15 → 1.41 ms at z12, 1.36 → 1.57 ms at z15, 3.21 →
3.43 ms pitched 60°, 0.95 → 1.09 ms at z17, 3.08 → 3.23 ms with Liberty's
buildings pitched. Shading per fragment rather than per pixel costs 0.14
to 0.26 ms there; a mid-range phone at 1080p should see a few tenths of a
millisecond. Sampling the shadow cascades in every layer's fragments cost
1.2 ms more at z15 than the ground pass.

## Next

3D layers and models in the same lighting model, procedural detail (water,
windows at night).

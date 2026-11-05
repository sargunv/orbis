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
  `emissive-strength`, `windows` for extrusions and `waves` for fills and
  lines; a layer's v8 colours are its albedo. The default is
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
- **Extrusions and models share the model.** An extrusion's fragment
  hands `surface_out` its paint colour as albedo, its normal, its shadow
  and its occlusion (walls near the ground get less ambient light, as
  GL JS's vertical gradient darkens them: 0.7 at the ground to 1 at 40 m);
  the lit composition shades it with its node's material, the classic one
  returns GL JS's lit colour as before. Models already fill `Material`
  from glTF. A v8 `light` alone keeps GL JS's extrusion response.
- **Lit windows are a material property.** An extrusion's `windows` (0 to
  1) is the share of its windows lit: a procedural grid on its walls
  (3 m bays, 3.5 m floors from the second, a hash per window and
  building) emits a warm light, its edges a pixel wide so far walls blend
  to the share rather than alias. It depends on the wall alone, so every
  frame shows the same windows lit.
- **Water's waves are a material property too.** A fill's or line's
  `waves` is its water's RMS slope: six sines whose wave vectors are
  whole cycles per ground cell (64 quanta, about 128 m on Earth whatever
  the surface's unit), so they tile the world exactly at any distance
  from the origin and stay put under every camera
  (`ground_point`). They run at deep water's speed with the frame's time
  alone. A wave shorter than a few pixels fades out and its slope widens
  the glint (GGX's α² takes twice the variance), so far water glitters
  rather than aliases. A map showing waves draws frames continuously:
  the frame report's `animated` keeps `needs_frame` set while a layer
  with waves has a mesh in a tile placed that frame, so water out of
  view, or a style without waves, idles; `idle` still means loaded, so
  snapshots settle.
- **Every material mirrors the sky.** `pbr_light` adds the ambient light
  mirrored by Schlick's Fresnel with roughness, for dielectrics as for
  metals: smooth water takes the sky's colour toward grazing angles,
  matte paint about 1% of the ambient.
- 3D layers, heatmaps and layers composited once arrive shaded and
  composite as they do unlit; draped layers on terrain draw into the drape
  unlit, which the terrain's own lighting shades. Compat draws unlit and
  warns. A translucent 3D layer keeps its nearest faces and lays them over
  the map at its opacity (decision 0043), lit or not: the map beneath,
  ground shadows included, shows through, never its own far walls.
- **A scene paints the style it extends.** A style layer's entry in the
  scene's `layers` takes a `paint` object, set over the layer's own paint
  when the scene extends its style, so a scene can make a style it doesn't
  own fit its lights: Lit Tokyo makes Liberty's buildings (0.8 opaque,
  85% light) solid and a warmer mid-tone, whose shaded faces then read
  clearly darker than sunlit ones.

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

Extrusions lit by their material measured, on a loaded machine whose
medians swung threefold between runs, at 1.31 ms (z12) and 1.60 ms (z15)
in its quiet runs, as above; the pitched 3D-buildings view wants a quiet
machine.

## Next

The day–night globe.

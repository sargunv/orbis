# 0048: The lit map: a scene's lights shade every layer through one material buffer

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
- **One material buffer, one lighting pass.** The map's layers draw, in
  style order and blended as their colours are, into a material buffer of
  two attachments: the albedo, and roughness, metalness and the self-lit
  share, premultiplied by coverage (`map_out_material.wgsl`). One
  full-screen draw in the main pass (`map_light.wgsl`) divides the
  coverage out and shades it with `pbr_light` under the frame's lights and
  the directional light's shadows on the ground, over the sky. Labels draw
  after it, unlit. The families' fragments return `map_out(colour, node)`;
  a shader's lit composition (`Shader::lit`) defines it as material, its
  classic one as colour, so the unlit path is unchanged.
- **Per-node materials** are a storage array in a group of the lit
  pipelines' own (group 2), so the scene group and Compat are untouched.
- **First cut.** 3D layers, heatmaps and layers composited once arrive
  already shaded and composite in as self-lit; blend modes composite
  normally. The map draws unlit with terrain, and on Compat, which warns.

## Measured

Apple M6, 2048×1536, `orb-bench` over central Tokyo with Liberty's 3D
buildings, the scene's lights without shadows against the same scene
unlit: GPU median 1.24 → 2.02 ms at z12, 1.70 → 3.05 ms at z15, 3.15 →
5.69 ms at z15 pitched 60°, 1.02 → 1.91 ms at z17. The lit surface pass
takes 0.8 to 2.2 ms of that: the layers drawn into two attachments in a
pass of their own, where the unlit map drew them in the main pass. A
mid-range phone at 1080p should expect several milliseconds more per
frame. Dropping the normal attachment, which carried nothing yet, saved
little: the cost is the extra pass and the composites moved into it, not
bandwidth alone.

## Next

3D layers, terrain and models writing their materials and normals into the
buffer (one lighting model for everything, shadows on walls from the
pass), procedural detail (water, windows at night), and the cost of the
separate pass.

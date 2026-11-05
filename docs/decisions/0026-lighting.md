# 0026: Lighting: one model, v8's light lowered onto it

Plan §9.4 and §11.4 put lights in the scene document (`lights`,
`*-emissive-strength`, then shadows and glTF's materials in M8). v8 has one
flat `light` for extrusions. This records the model they share. Extending
it later must not mean a second, parallel path.

## The model

The Style IR's `Lights` (`orb-v8`) holds:
- an ambient light and a directional light, each a colour and a level;
- the directional light's direction: a position (radius, azimuth, polar)
  or the sun at a UTC time over the camera's target;
- its anchor (map or viewport);
- whether the lights shade the terrain;
- each layer's emissive strength.

The frame block carries the evaluated lights (`FrameLights`). One WGSL
module (`lighting.wgsl`) shades with them; extrusions and terrain call it.

The response is GL JS's flat extrusion shading, which v8 defines. A face
away from the light gets `a` of the ambient colour. A face toward it gets
`max(a + d − luminance, 1)` of the directional colour: lighter surfaces
gain less, so lit faces don't wash out. Faces in between are mixed by n·l,
with GL JS's wall gradient, 0.03 lift and floor for tinted lights. Emissive
strength mixes back to the surface's own colour. Directions use GL JS's
axes (`sphericalToCartesian` and its inward wall normals), and terrain
normals follow them.

## Lowering

- **v8 `light`** (the style compiler): GL JS shades a face
  `mix(1 − i, max(1 − lum + i, 1), n·l)` of the light's colour. That is the
  response with ambient level `1 − i` and directional level `2i`, both
  of the light's colour, at its position and anchor. Levels are a linear
  map of the declared intensity (`Level`), so zoom-dependent intensities
  lower as they are. v8's light shades extrusions only, as in GL JS. The
  `light` fixtures (15) and extrusion fixtures are unchanged, native and
  Chrome.
- **The scene document's `lights`** replace the style's. Each is at most
  one `ambient` and one `directional` (`direction: [azimuth, polar]`) or
  `sun` (`time`, RFC 3339 UTC), with `color` and `intensity` as declared.
  They also shade the terrain surface by its DEM normals, and so what is
  draped on it. `layers.<id>.emissive-strength` applies to extrusions. The
  shape follows the style spec's lights proposal (style-spec#1797).

```json
{"style": "alps.json",
 "lights": [
   {"type": "ambient", "properties": {"color": "#dfe9ff", "intensity": 0.45}},
   {"type": "sun", "properties": {"time": "2024-07-15T06:30:00Z", "intensity": 0.75}}],
 "layers": {"buildings": {"emissive-strength": 0.1}}}
```

The sun's position uses the Astronomical Almanac's low-precision formulas
(about 0.01° from 1950 to 2050). It fades out over its last two degrees.

## Where it grows

The levels and colours stay the model's terms:
- shadows (M+) attenuate the directional term;
- glTF's PBR materials read the same lights with their own response;
- emissive strength reaches other families when they light.

Flat 2D layers are not lit; on terrain they take the terrain's light
through the drape. Values in the scene document are constants for now.

## Not done

- Emissive strength for draped layers (they take the terrain's light
  together), and lights on flat 2D layers.
- More than one ambient or directional light (the parser warns).

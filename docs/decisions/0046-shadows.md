# 0046: Shadows from the directional light

Plan §8.5's `Shadow` node and decision 0036's reading of style-spec#1797:
real shadows from the lights model (decision 0026), cast by extrusions and
glTF models onto extrusions, models and the ground.

## Style

The scene document's `directional` and `sun` lights take `cast-shadows`
(bool, default false) and `shadow-intensity` (0 to 1, default 1), Mapbox GL
JS's names. They lower into `DirectionalLight::shadow` and the frame's
`FrameShadow`. v8's own `light` casts none, so no v8 fixture changes.

```json
{"type": "sun", "properties": {"time": "2024-10-20T20:30:00Z", "intensity": 0.6, "cast-shadows": true}}
```

A light lower than 2° casts nothing; the sun fades out there anyway.

## Shadow maps

Two cascades, a 2048² `Depth32Float` array, 32 MiB while shadows are cast,
freed when they aren't. Each frame fits them on the CPU (`orb-render`'s
`shadow` module) from the frame alone:

- Cascade 0 covers view depth up to 1.5 × the target's, cascade 1 up to 4 ×.
  Each also holds the last tenth of the one before, where receivers blend
  into it.
- Each slice is cut to the heights shadows fall between, which the runtime
  gives: the terrain's range around the target (sea level without terrain)
  up to 600 m of buildings above it. That keeps sky out of the fit.
- A bounding sphere of what is left sets the cascade's size. It depends on
  zoom and pitch only, never bearing, so turning doesn't change texel size.
- Seen along the light, the sphere's centre snaps to whole texels of
  absolute world space, so edges don't crawl as the camera pans.
- Toward the light the orthographic view reaches past the sphere by the
  height range over the light's elevation, so tall casters outside it
  still cast into it.

Nothing depends on earlier frames or on whether the camera moves.

The casters are the extrusion nodes' meshes and the model nodes' instances
with the cull's indirect arguments, drawn depth-only from the light
(`extrusion_caster.wgsl`, `model_caster.wgsl`; no face culling: walls alone
leave a box's middle lit from a high sun), one bundle per cascade, recorded
with the main bundle and replayed. Models cast with their solid and masked
materials (masked texels below the cutoff cast nothing); blended ones cast
nothing. A model caster shares `model.wgsl`'s placement with the lit draw,
and binds the cascade in group 3, after the model's own group. The
casters' scene group binds a 1×1 stand-in where the map goes, as a pass
can't sample what it draws into.

The cull keeps an extrusion chunk or model instance whose shadow falls in the view even when
the chunk itself is outside: the frame carries the shadow's reach per unit
of height (`shadow_reach`), and the cull also tests the chunk's corners
moved by it. Camera-only frames keep one dispatch and replay the bundles.

## Receivers

The scene group binds the shadow uniform, the map and a comparison sampler
(bindings 16–18). `shadow_receive.wgsl` reads them; Compat composes
`shadow_none.wgsl` instead.

- **Extrusions:** the vertex shader shades each vertex lit and with none of
  the directional light. The fragment mixes the two by `shadow_at`, a 3×3
  comparison filter after a normal and light offset of about a texel. Lit,
  the colour is exactly what it was, so extrusion fixtures are unchanged.
- **Models:** `shadow_at` at the geometric normal is `PbrLights.shadow`, which
  scales `pbr_light`'s directional term: the lighting path the whole map's
  materials take (#91), not a separate one.
- **The ground:** one full-screen draw in the main pass at the first 3D
  layer's place, blended multiplicatively (`BlendState::Multiply`). It
  rebuilds each pixel's ground point from the terrain's depth, or the view
  ray on the ground plane, and multiplies by `1 − share · (1 − lit)`. Here
  `share` is the directional light's part of level ground's lighting in
  decision 0026's response.

## Where not

- **On a globe:** none, while the view shows any globe; the map warns once
  as they turn off.
- **Compat:** none, with a warning (plan §8.7).
- **On other surfaces:** shadows need nothing beyond core WebGPU (a depth
  texture array and a comparison sampler), so they run on Modern, not only
  M+ as the plan first declared.

## Measured

Apple M6, native, 2048×1536 px, Liberty over Midtown (z14.52, pitch 61,
bearing −18.1), `--timing work-done` (wall time to the GPU's work done):

| | proxy median | p90 | CPU render | GPU memory | visible chunks |
|---|---|---|---|---|---|
| no shadows | 3.39 ms | 3.48 ms | 0.34 ms | 119.5 MiB | 4,149 |
| shadows | 6.40 ms | 6.91 ms | 0.51 ms | 151.5 MiB | 4,583 |

Most of the 3 ms is the extrusions drawn twice more, once per cascade, with
the camera's culled chunks.

## Not done

- Casters in tiles the view's cover leaves out cast nothing: the cull widens
  only within resident tiles (#92).
- Per-cascade culling or a lighter caster vertex path, which would cut the
  casters' cost (#93).

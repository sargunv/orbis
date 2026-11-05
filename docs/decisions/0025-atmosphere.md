# 0025: The atmosphere: one sky and fog model for every surface

Plan §8.6 asks for one `Atmosphere` render-graph node, "one sky/fog model on
every surface", where GL JS has a sky shader and a separate fog inside its
Mercator terrain shader (#4337). This records how Orbis draws it and where
the globe joins it.

## What it draws

`orb_render::Atmosphere` holds one uniform from the style's `sky`
(`AtmospherePaint`: `sky-color`, `horizon-color`, `fog-color`,
`sky-horizon-blend`, `horizon-fog-blend`, `fog-ground-blend`) and from the
frame's view (`AtmosphereView`). Two shaders read it through the WGSL
`atmosphere` module:

- the sky, drawn before the map layers: GL JS's gradient from the horizon
  colour to the sky colour over `sky-horizon-blend` of half the viewport,
  above the horizon;
- terrain fog: GL JS's `terrain.fragment.glsl`, a fog depth from the fog
  range along the view, blended in linear space from the fog colour toward
  the horizon colour past `horizon-fog-blend`, from `fog-ground-blend` on,
  faded in from 60° to 70° of pitch.

As in GL JS, fog applies to the terrain surface (and so to what is draped on
it), not to extrusions, symbols or a flat map. Without a `sky` there is no
sky and no fog, and the host's backing colour shows (decision 0014).

## The surface's horizon

The surface tells the atmosphere where its horizon is. A flat surface gives
a line: a point on it and its normal toward the sky, in logical pixels. This
is GL JS's `u_horizon`/`u_horizon_normal`, so camera roll (#46) only has to
pass a turned line.

The globe joins here rather than adding a second sky. GL JS 6 draws its
globe atmosphere in the same sky shader: with `u_sky_blend` (the
projection's globe blend) it measures the sky from the globe's limb and
fades it out where the view ray misses a shell 100 km above the sphere,
less `atmosphere-blend`. When R1's globe surface lands, `AtmosphereView`
gains its sphere in view space (centre, radius, blend). `AtmospherePaint`
gains `atmosphere-blend`, which v8 already parses. The sky shader takes
GL JS's terms, and terrain fog stays off on the globe as in GL JS. Until
then the three `sky/atmosphere-blend` fixtures and `terrain/fog-sky-blend-globe`
skip on `property:projection`.

## Interplay found on the way

A viewport-anchored hillshade (v8's default anchor) turns its light with
the bearing. Draped on terrain, its slots kept the light they were drawn
with. The bearing now joins the drape's look whenever such a hillshade is
draped, so its slots redraw once the camera stops turning, as zoom changes
do (decision 0019).

## Fixtures

Native, after the change (identical to before it):

| directory | pass | fail | skip |
| --- | --- | --- | --- |
| `sky` (+ `terrain/fog*`) | 12 | 9 | 10 |
| `terrain` | 9 | 19 | 15 |
| `light` | 15 | 0 | 0 |

The nine sky failures are `fog-color`, `fog-ground-blend` and
`horizon-fog-blend` (three variants each), each 0.8–1.3% of pixels off. In
`fog-color/default`'s diff the differing pixels lie on the far ridge line
against the sky, where the far terrain's silhouette differs (#41), and sky
and fog colours match elsewhere. The skips are roll (#46) and the globe
(above).

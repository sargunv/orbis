# 0035: Stars and a backdrop in the atmosphere, and the scene document's `atmosphere`

The globe drew on whatever the host put behind the canvas. There was no
space, no stars and no colour behind the sky. style-spec#1889 proposes
`sky.star-opacity` and `sky.backdrop-color`. The proposal's points:
- stars as a fixed decorative pattern that turns with the globe;
- stars that appear as the sky fades on the way to space;
- a night style can show them near the ground with a transparent sky.

This records Orbis's model for them. It also records the scene document's
`atmosphere`, which plan §11.4 lists and which carries them.

## The model

The Style IR's one `Atmosphere` (decision 0025) gains two terms:
- `stars`, a brightness from 0 (none) to 1;
- `backdrop`, the colour behind everything.

The renderer composites, back to front: the backdrop, then the stars over
it, then the sky, then the map. The backdrop shows wherever neither the
map nor the sky covers it. Transparent keeps the host's backing (decision
0014). Stars show where the view ray leaves the surface: past a globe's
limb, or above a flat surface's horizon. The sky covers them where it is
opaque. On the globe the sky fades to space with altitude (decision 0025),
so stars appear by themselves on the way out. A transparent `sky-color`
shows them near the ground.

The scene document's `atmosphere` takes v8 `sky`'s properties by the same
names, plus `star-intensity` and `backdrop-color`. It replaces the style's
lowered `sky`, as `lights` replaces `light` (decision 0026). It goes
through the same parser and into the same IR. v8's `sky` keeps the new
terms at their defaults and warns on the new names. If the spec adopts
#1889, `star-opacity` and `backdrop-color` lower onto the same two terms
in `sky.rs`.

```json
{"style": "https://tiles.openfreemap.org/styles/liberty",
 "atmosphere": {"star-intensity": ["interpolate", ["linear"], ["zoom"], 3, 1, 7, 0],
                "backdrop-color": "#02040b"}}
```

## The stars

The starfield is procedural, made from first principles for a map that
zooms, tilts and turns:

- **Fixed to the sky.** The pattern lives on the celestial sphere: the
  view ray in Earth-fixed (ECEF) axes, turned about the pole. It turns
  with the globe and the camera, and panning across a flat map moves it
  as a traveller's sky moves. A flat surface gets its ray from its
  inverse view-projection, from east, north and up at the target.
- **Turned by the real time, when there is one.** With a scene `sun` at a
  time (decision 0026), the turn is Greenwich sidereal time. Stars and
  the sun's light then agree as the time changes. Without a time the turn
  is 0. The pattern isn't a catalogue, so the time sets their motion, not
  real constellations.
- **Stable angular density.** Cells on an equal-angle cube cover nearly
  equal solid angles. There are two layers, 64 and 20 cells per face
  edge. A hashed share of cells hold a star, kept off the cell's edges,
  giving about 9,300 stars over the sky: the naked eye's count. A
  60°-wide view shows the same few hundred at any zoom or canvas size.
- **Sharp at any resolution.** Each star is a point spread about one
  device pixel wide, measured from the angle between neighbouring
  pixels' rays. Stars stay dots and never smear or alias as the field of
  view or pixel ratio changes. Brightness follows a power law, with
  many faint stars and few bright ones, tinted blue-white to orange.
- **Brightness.** Stars add light over the backdrop and clamp at the
  target's 1.0. Targets are 8-bit today. If an HDR target comes, the
  same intensity can exceed 1 with no change to the model.

Rejected:
- **A star catalogue or a sky image.** These are heavy, and only correct
  with the body's orientation and time. A catalogue could later replace
  the procedural field behind the same `stars` term.
- **A backdrop gradient.** Nothing needs one yet.
- **A glow around the globe.** The scattering pass already draws the
  atmosphere's limb.
- **Fading the stars near the ground on their own.** The sky already does
  it, and a separate rule would rule out night skies, as #1889 argues.

## Cost

The sky pass runs whenever an atmosphere exists. It now computes the stars
only where they can show and `stars > 0`. Per pixel that is two hashed
cells, each a few transcendental functions and an `exp`. That is well
under the scattering pass's 5 × 3 sample loop. The pass's GPU time wasn't
measured: the bench doesn't drive a scene's globe yet.

## Validation

- **Render test**: `stars_and_backdrop_show_behind_a_clear_sky`. Stars
  light more than 20 pixels above a flat view's horizon and none below
  it. An opaque backdrop replaces the host's backing where nothing draws.
  The existing sky gradient test is unchanged.
- **Parse test**: `scenes_add_stars_and_a_backdrop_which_v8_sky_lacks`.
  The sidereal angle is checked against J2000.0's 280.46°.
- **Fixtures**: unchanged, since v8 styles keep both terms off.
- **Demo**: the Globe entry is now a scene document over live Liberty
  with a sky, an atmosphere blend that eases out from zoom 2 to 8, stars
  that fade by zoom 7 and a deep-space backdrop. Checked in Chrome on
  WebGPU (with scattering) and WebGL2 (without, as Compat declares),
  and natively on a pitched flat night sky.

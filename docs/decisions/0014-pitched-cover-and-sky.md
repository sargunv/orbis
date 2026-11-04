# 0014: Pitched cover and flat-surface sky

Status: implemented. It fixed pitched covers that dropped detail too close to
the camera and drew nothing above the horizon.

The live Liberty style over Manhattan lost detail too quickly when pitched
from 60° to 80°, and transparent pixels above the horizon appeared black.
The behavioural reference is MapLibre GL JS commit
`2b68611a4e2d4f7f076f6c6ebf758e2d53f91471`.

Retain the plan §9.3 design: the source TMS supplies tile spans, the camera
supplies world units per pixel, and one generic depth-first traversal uses
`CoverDecision`. Select detail from the nearest horizontal tile distance,
vertical eye distance, eye-to-target distance, FOV and pitch. No tile index or
coordinate is converted to a Mercator-specific representation. The one
projection-specific conversion, v8 height metres to projected world units,
now belongs to `Projection::style_height_scale`; this replaces the runtime's
previous inline WGS84/mean-sphere conversion.

The reference keeps nominal detail through `min(60°, 78.5° - fov/2)`. Above
that, its default LOD curve uses 9.314 maximum zoom levels on screen and a
3.0 maximum/minimum tile count ratio. The prior approximation misplaced a
parenthesis in the exponent: `2 * (ratio - 1)` had become `2 * ratio - 1`.
That dropped detail too aggressively. The integral uses the same ten midpoint
samples as the reference, and desired tile zoom is floored, bounded below by
zero and above by the source/TMS limit. Source minimum zoom is still applied
by `TilePyramid`, without replacing below-minimum ideals with finer ones.

The finite flat-surface horizon is a view policy: 89.25° maximum sample angle,
0.85 apparent-horizon factor, 1.01 far-plane margin and at most three repeated
worlds on each side. The far plane also covers a plane at least 100 style
metres below the eye when the eye is close to the ground. Missing that last
rule clipped two rows from the upstream sky fixtures. Frustum rejection now
checks the tile box's axes as well as frustum planes; plane tests alone let
far-away boxes survive at corners. Near the horizon the culling volume grows
toward the larger of 500 style metres and the known maximum content height,
as upstream does, so elevated roofs can retain their tiles.

These are flat-surface distance/pitch operations, independent of coordinate
names. Future surface bounding volumes and horizon queries remain the place
to supply curved-surface geometry in M6–M8; this does not add a second cover
or change the plan's architecture, tiers or crate graph.

The pinned GL JS defaults `maxPitch` to 60°. Its API accepts explicit limits
up to 180°, labels pitches above 60° experimental, and uses the 89.25° finite
horizon angle to protect its matrix calculations near horizontal. Orbis keeps
its existing 85° interaction cap. Golden covers include the GL JS default
clamp and the 85° cap, and camera tests exercise finite frames, unprojection
and bounded covers through 85°, including at zoom 24.

For a style with `sky`, draw the reference's premultiplied sky/horizon gradient
before map layers. Its horizon colour weight is `(1 - distance/blend)^2`,
where blend is `sky-horizon-blend * viewport_height/2`. The apparent sky
horizon and the finite-frustum ground edge are distinct camera values.
The sky uses the existing frame uniform's padding, and style uniforms change
only when evaluated sky properties change. Camera-only frames still replay
map bundles and write no paints.

Absent `sky` means **transparent**, not the spec's blue sky default; `sky: {}`
uses `#88C6FC`, white horizon, and blend 0.8. GL JS clears transparent and draws
background layers on the ground. Its fog is a terrain pass and its atmosphere
is a globe pass, so neither belongs in the current flat-surface renderer.
The WebGPU canvas now uses premultiplied compositing, exposing the page's
background instead of treating transparent sky as opaque black. The native
demo and its snapshots choose a white backing with `Map::set_clear_color`;
the engine's default and fixture output remain transparent. Explicit dark
style colours and transparent engine pixels retain their intended meaning.

The cover generator and reproduction instructions live in
`crates/orb-camera/tests/README.md`. Its 66 tile-set goldens come from executing
the unmodified pinned GL JS transform and covering function in Node. The
fixture mirror now includes upstream's 26 sky cases; nine flat-sky cases are
supported (including global state), while terrain, globe, roll and unsupported
operation cases remain explicit skips. Root sky edits and changes to global
state read by sky follow the style engine's root restyle policy; layer-only
edits retain the existing sky. A wait-only fixture operation is inert because the runner already
settles the map. Existing pitched extrusion fixtures stay in the suite.

Measured Manhattan ideals, viewport 1024×768, target -74.0113/40.7075,
zoom 14.48, bearing 30°, vector cover maximum 24:

| Pitch | Before, zoom:count | After, zoom:count |
| --- | --- | --- |
| 60° | 14:15 | 14:15 |
| 80° | 1:1, 5:1, 9:1, 11:4, 12:1, 14:2, 15:1, 16:6 | 9:1, 11:2, 12:5, 13:3, 14:2, 15:1, 16:8, 17:5 |

The remaining loss of buildings in the distance follows the reference:
Liberty's building layer starts at tile zoom 14. Increasing far-tile detail
beyond the golden sets would change that policy rather than fix a camera bug.

Verification with the style engine work:

| Suite | Pass | Fail | Skip |
| --- | ---: | ---: | ---: |
| Native baseline (883 mirrored cases) | 352 | 6 | 525 |
| Native after (909 mirrored cases) | 374 | 6 | 529 |
| Chrome after | 374 | 6 | 529 |

No baseline pass regressed. Chrome/native identity matched all 374 passing
images. The steep-pitch extrusion fixture improved from 7,441 differing
pixels (2.8385%) to 3,582 (1.3664%), still outside upstream's tolerance. The
six pre-existing failures remain; 13 older global-state cases also became
runnable through the concurrent style engine change. The four camera-only
frame tests pass, including a style with sky through pitches 60°, 80° and 85°.

The live 60° and 80° screenshots were inspected. Sky is transparent over the
page's backing colour at 80°, and the 60° view is visually unchanged. HUD
placement counts (including background and resident parts) are 35 → 32 at
60° and 39 → 43 at 80°; these are not the distinct ideal tile counts above.
The snapshot command captures before/after images, with and without the HUD,
and reaches idle, then exits nonzero on the 26 existing unsupported
symbol/glyph style warnings. No GPU or network errors occurred in the live
captures.

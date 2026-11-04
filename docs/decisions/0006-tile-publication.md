# 0006: Tile fallback publication and request lifetime

Status: implemented. It fixed maps flickering between zoom levels while
zooming, and tiles that stayed unloaded until the map moved.

The demo could replace fine fallback coverage with a late coarse ancestor,
lose coverage across a zoom-out of more than one level, and forget loading
entries without cancelling their host requests. Returning to a forgotten tile
started another job; an earlier failed result could mark the new entry empty
before its successful result arrived. The successful result was then rejected,
leaving a stationary view empty. Hosts also stopped scheduling after a frame
that reported loading work, depending entirely on subsequent wake callbacks.

The deterministic source replay records cover, tile, wrap and regions. With a
fixed z2 cover, two ready z3 tiles disappeared when a z0 result arrived. Moving
from z3 to z1 produced no draws. The request replay pans away and back before
completion and delivers older failures before the latest success; it exposes
the rejected successful result. The JavaScript animation-clock test exposes a
loading frame with no next frame scheduled.

Use the GL JS-like child-before-parent fallback policy: ready ideal tiles win;
otherwise keep loaded descendants and fill only uncovered regions with the
nearest ready ancestor. Search only branches containing cached loaded tiles,
including empty tiles, so work is bounded by cached tiles times pyramid depth.
Fallback regions do not overlap; more than four clip rectangles require
additional placement rows, preserving the existing GPU row format. Expanding
those regions to whole quadrants would overlap finer tiles and double-blend
translucent geometry. Zoom hysteresis is unnecessary for these reproductions.

Keep in-flight entries until completion or a whole-pyramid drain. Cache
eviction cannot cancel host jobs, so it must not forget their ownership. They
do not consume the ready-cache budget; completed unused entries are evictable.
Map style/tile generations still reject results from a replaced scene. This
retains metadata for outstanding jobs, which already consume host resources;
future cancellable jobs can release both together.

Native and both browser modes continue scheduling while the map is non-idle.
Every arriving result still wakes the host, including results arriving after
idle. The offscreen frame test delays jobs, releases them in reverse order,
checks one wake per result and verifies publication without camera input.

Follow-up, for retained child tiles that blocked their ready parent after
zooming out: the Tokyo z15 → z12 reproduction differed
from a fresh z12 load by 132,470 of 786,432 pixels. The publication policy was
working: the actual ideal cover differed. A centred zoom unnecessarily
round-tripped the known camera target through the projection inverse, changing
latitude from 35.685 to 35.684999999999995. At the exact refinement threshold,
the fresh view selected two z11 ideals; the zoomed view selected one z11 and
four z12 ideals. Ready ideals already replaced descendants and made them
evictable; cached resident mesh counts did not identify what was drawing.
Centred zooms now retain the exact target and update only the resolution.

The pixel comparison also exposed shared clip edges: inclusive maximum edges
allowed adjacent regions to blend the same translucent fragment twice, making
pixels depend on tile arrival order. Clip rectangles are now minimum-inclusive
and maximum-exclusive, implementing the non-overlapping partition above. A
pixel-centred shared-edge render test guards this rule. The Tokyo zoom-out,
zoom-in and pan tests compare settled frames against freshly loaded views;
the source test verifies complete ideal publication and descendant eviction.

Publication order must also reach the scene: the runtime forwards the
pyramid's ancestor-before-descendant tile order when the drawn tile list
changes, and the scene orders each node's meshes accordingly. Arrival and
arena allocation order no longer decide blending at a seam. Camera-only
frames with the same tiles do not reorder meshes or rebuild candidates.

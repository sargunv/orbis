# 0027: Dense tiles in a full scene

On the live Liberty map, pitched over lower Manhattan at z14.95, one
tile-shaped block lost its 3D buildings while its roads and labels drew,
"usually repairing itself on zoom in and out, then happening again at the
exact same spot". Browsing that area in headless Chrome for a few rounds
reproduced it: z14 tiles 4821–4828 / 6155–6158 "didn't fit in scene memory
with the tiles in view", were marked missing, and their z13 parent (which
has no building layer) stood in until the entry left the cache.

## Causes

- **One contiguous block per layer.** OpenFreeMap merges a dense z14 tile's
  buildings into a few multipolygons; tile 14/4822/6157's building layer is
  16 features, one of 36,347 points. Its extrusion mesh needs 0.5–0.9 M
  contiguous vertex words. With the arena 70–80% full of the tiles in view,
  its largest free block was 145–510 k words though 2–5 M were free.
- **Giving up too early.** Right after a zoom, the tiles still loading keep
  their loaded descendants as stand-ins, which hold the room the arrivals
  need. An arrival that didn't fit evicted its own stand-ins, then was
  marked missing for good; the hole outlived the transient pressure.

## What changed

- A node's selection lays out in meshes of at most 2,048 input points
  (`tile::MESH_POINTS`, about 40 k words for extrusions). A larger feature
  splits by its parts into pieces that keep its properties and id; picking
  still names the tile's feature. Only the geometry decides the split, so
  property-row jobs (paint and feature-state edits) split alike and pair
  with the tile's meshes in order.
- A tile that doesn't fit while other tiles' jobs are in flight waits,
  writing nothing, and tries again on the next frames, within the upload
  budget, as those arrivals free their stand-ins. The map isn't idle while
  one waits. Only with nothing else in flight does it evict its own
  stand-ins and, if it still doesn't fit, give up.
- Giving up is a map `error` event as well as a warning ("doesn't fit in
  scene memory with the tiles in view; a coarser tile stands in"), as is a
  layer an edit can't fit and the first frame the cull drops chunks for
  want of candidate or compacted-index room.

## Measurements

Headless Chrome 156, Apple M6, main thread, 1280×800 at devicePixelRatio 2:
six rounds of easing between z12.5 and z15.5 around 40.717, −73.997, then
back to z14.95 at pitch 60. Before: 4–9 tiles dropped per run, the same z14
tiles each time. After: none in 12 rounds over two runs, every tile drawn
with its buildings. Uploads stay about 210 MiB a round.

## Open

The scene's cache is bounded by tile count (128), not memory, and the
arenas are fixed (16 M vertex words): a pitched dense view still keeps them
70–80% full. Plan §8's budget manager, sizing caches by memory, is the
lasting answer; until then a view whose own tiles exceed the scene draws
coarser tiles and says so.

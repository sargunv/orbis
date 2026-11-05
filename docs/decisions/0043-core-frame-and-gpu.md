# 0043: The core chain: camera-independent GPU state and a narrow renderer

Decision 0040 set the first-release target. This record holds the core
chain's part of it, the frame and the GPU scene, and what each step changed.

## Decision

- **A camera-only frame writes the frame block, then one cull dispatch and
  the recorded bundle draw it.** Nothing else in the scene depends on the
  camera:
  - **Tile rows hold the tile's place in the world.** The origin is a whole
    number of quanta (an i32) and an f32 remainder. The quantum is a power
    of two sized to the surface, with room for 64 world spans. The frame
    block holds the camera alike. Shaders subtract the whole quanta as
    integers, then the remainders.
    - An f32 hi/lo pair was tried first. On Metal, which compiles with fast
      math, the compiler reassociated the subtraction and the low parts
      vanished: at z22 a 26 cm camera move drew nothing. Integer whole
      quanta can't be reordered away.
    - Globe anchors split the same way, in ECEF metres.
    - The camera's world units per pixel moved to the frame: a row keeps
      only a factor of it, 1 on the map and a drape slot's scale in a drape.
    - The pattern origin still depends on the integer zoom, so rows change
      when the camera crosses one.
  - **Changed rows only.** The scene compares each placement's row with the
    one it holds, and writes the changed rows in one write in `prepare`.
  - **Cull parameters live in the node rows.** Margin, elevation and
    visibility are patched in place. The candidate lists are laid out again
    only when tiles or placements come or go, or a node changes how it
    draws. Hidden nodes keep their candidates; the cull and the heatmap's
    colour pass skip them by a flag.
    - The margin is still evaluated at the frame zoom on the CPU. With
      paint lowered to per-node curves (the data chain), it becomes the
      curve's maximum or a GPU evaluation.
- **Every frame is right for its camera, tested.**
  `crates/orb-api/tests/path.rs` moves a map to a camera frame by frame,
  lets it settle, and compares it with a fresh map at rest at that camera,
  to one step in a channel. It also checks that camera-only frames write no
  rows, and that a z22 corner draws where the CPU projects it.
  - It found the pyramid drawing a missing tile's nearest *ready* ancestor,
    so a coarse tile cached during a gesture stood in where a fresh map
    loads the nearest ancestor in the tileset. A missing tile's stand-in is
    now that ancestor, whatever else is resident.

## Measured

Apple M6, 2048×1536, `cargo xtask bench --experiment gestures`, render CPU:

| path | p50 | p95 | candidate rebuilds | rows written per frame |
|---|---|---|---|---|
| zoom z12 → z16 → z12 | 1.36 → 1.15 ms | 3.71 → 2.37 ms | 360 → 35 of 360 | 9.9 → 0.5 |
| rotate and tilt at z14 | 0.97 → 0.84 ms | 1.61 → 1.23 ms | 54 → 54 | 20.5 → 1.7 |

The rows still written are tiles arriving or leaving, and the background's
cover. Native fixtures are unchanged: 1179 pass, 116 fail, 50 skip.

## Still to land

The Map/Renderer split (`plan_frame`, a narrow `Renderer` that owns all
residency), reflected layouts, one offscreen group, one slot allocator,
lazy pools and atlases, one warp per tile, one native thread pool, and
pipeline failures as map errors. Each step updates this record.

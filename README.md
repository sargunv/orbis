# Orbis

**Experimental. Don't use Orbis for real maps: use
[MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js) or
[MapLibre Native](https://github.com/maplibre/maplibre-native).**

Orbis is a ground-up Rust renderer for the
[MapLibre style spec](https://maplibre.org/maplibre-style-spec/) v8. One core
renders on wgpu: in the browser through wasm and WebGPU, and natively through
Vulkan, Metal and D3D12. AI agents write and steer it; a person sets them going
and reviews the result.

**Scope.** The v8 spec, including terrain, the globe, sky and the light on
extrusions, plus what the spec could reasonably grow into, such as other
projections, coordinate reference systems and tile matrix sets.

**Status.** Experimental and unpublished; every API can change. Mostly works
well, though 3d terrain is rough and there's no WebGL compatibility for older
hosts.

The live demo is at <https://code.sargunv.dev/orbis/>.

## Design principles

- **A frame is a function of style, data and camera.** It doesn't depend on the
  camera's path, the order tiles arrived in, or anything else in its history, so
  it's deterministic and testable. Exceptions are explicit and opt-in.
- **The spec's restrictions are portability notes, not limits.** Many of the
  spec's "this property can't vary by feature" or "this can't animate" rules
  come from how the existing engines were built. Orbis accepts whatever its own
  design can honour, and tells you when a style won't work in the other engines.
  New syntax stays opt-in.
- **Both MapLibre engines are references, not targets.** GL JS and Native are
  the guide to how styles should look. Where they differ, Orbis follows
  whichever handles the case well; where both share a known limitation, it aims
  for what the style author most plausibly meant.
- **Modern GPUs first, compatibility added downward.** GL JS and Native grew up
  on WebGL 1 and OpenGL ES 2 and still carry those constraints. Orbis is
  designed for today's GPU APIs and reaches older hardware through compatibility
  layers, not by designing around the oldest one.

## How it differs

MapLibre GL JS and MapLibre Native, and
[mln-rs](https://github.com/louwers/mln-rs), a file-by-file Rust port of Native,
share one design: worker threads turn each tile into buffers, and every frame
the CPU walks the visible tiles of each layer and issues their draw calls. Orbis
carries out the principles above with a different design:

- **The GPU drives each frame.** All tiles' geometry lives in a few large
  buffers, and the draw commands are recorded once and replayed. Each frame, a
  compute shader keeps the visible pieces and writes the draw counts, so the CPU
  never loops over tiles and layers. That matters most in the browser, where
  every draw call is expensive.
- **Style values are read, not baked.** GL JS and Native bake each data-driven
  property into its own vertex attribute and shader variant, which is why the
  spec keeps many properties constant. Orbis gives each feature a row of values
  that shaders read through one generic path, so it can let more properties vary
  by feature, zoom or camera, and animate them.
- **Label placement never holds up a frame.** Deciding which labels fit is the
  costliest work a camera move triggers. GL JS does it on the page's main thread
  and Native on its render thread, both re-placing only every few hundred
  milliseconds to protect the frame rate. Orbis places labels on a thread of
  their own and keeps drawing while it runs.
- **Projections are first class.** Geometry is computed relative to the camera
  in double precision, so the globe, other projections and coordinate systems
  aren't special cases of Web Mercator.

## Build and run

[mise](https://mise.jdx.dev) provides the pinned tools (the Rust toolchain comes
from `rust-toolchain.toml`):

```sh
mise install

mise run demo:desktop                  # the map app in a native window
mise run demo:web                      # the web demo at http://127.0.0.1:5173/orbis/
cargo xtask check                      # formatting, clippy (native and wasm32), tests, boundary rules
```

[AGENTS.md](AGENTS.md) lists the other commands.

## Licence

Orbis is under the [BSD-3-Clause licence](LICENSE).

The repo contains OpenStreetMap-derived demo data under ODbL; see
[THIRD_PARTY.md](THIRD_PARTY.md).

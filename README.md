# Orbis

**Experimental. Don't use Orbis for real maps: use
[MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js) or
[MapLibre Native](https://github.com/maplibre/maplibre-native).**

Orbis is a ground-up Rust renderer for the
[MapLibre style spec](https://maplibre.org/maplibre-style-spec/) v8. One core
renders on wgpu: in the browser through wasm and WebGPU, and natively through
Vulkan, Metal and D3D12. AI agents write and steer it; a person sets them
going and reviews the result.

**Scope.** The v8 spec, including terrain, the globe, sky and the light on
extrusions, plus what the spec could reasonably grow into, such as other
projections, coordinate reference systems and tile matrix sets.

**Status.** Experimental and unpublished; every API can change. Mostly works
well, though 3d terrain is rough and there's no WebGL compatibility for older
hosts.

The live demo is at <https://code.sargunv.dev/orbis/>.

## How it differs

MapLibre GL JS and MapLibre Native, and
[mln-rs](https://github.com/louwers/mln-rs), a file-by-file Rust port of
Native, share one design: the CPU prepares each tile and issues each tile's
draw calls every frame. Orbis was designed from scratch around a few
different ideas:

- **The GPU drives each frame.** Tiles go into large shared buffers once.
  Each frame, a compute pass picks what's visible and the GPU draws it from
  pre-recorded command lists, so moving the camera costs the CPU very little.
- **Three phases.** Work happens once per tile (shaping text, building
  geometry, on worker threads), once per camera for labels (placement, on its
  own worker), or every frame in shaders. Each style property lives in the
  phase that can honour it, which is why Orbis can animate and data-drive
  more properties than the spec asks for.
- **The same picture however you got there.** A map you've panned and zoomed
  draws exactly like a fresh map at that camera. Nothing on screen depends
  on the path the camera took.
- **Plain workers on the web.** No shared memory, so no cross-origin
  isolation headers, and the map runs in a worker to keep the page
  responsive.
- **Projections are first class.** Geometry is computed relative to the
  camera in double precision, so the globe, other projections and coordinate
  systems aren't special cases of Web Mercator.
- **Both engines are the reference.** Orbis runs GL JS's and Native's test
  suites, and where they disagree it aims for whichever looks better.

## Build and run

[mise](https://mise.jdx.dev) provides the pinned tools (the Rust toolchain
comes from `rust-toolchain.toml`):

```sh
mise install
pnpm -C web install

cargo run --release -p orb-demo        # native window over the bundled Tokyo extract
cargo xtask web && pnpm -C web dev     # web demo at http://127.0.0.1:5173/orbis/
cargo xtask check                      # fmt, clippy (native and wasm32), tests, boundary rules
```

[AGENTS.md](AGENTS.md) lists the other commands.

## Licence

Orbis is under the [BSD-3-Clause licence](LICENSE).

The repo contains OpenStreetMap-derived demo data under
ODbL; see [THIRD_PARTY.md](THIRD_PARTY.md).

# Orbis

**Experimental. Don't use Orbis for real maps: use
[MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js) or
[MapLibre Native](https://github.com/maplibre/maplibre-native).**

Orbis is a ground-up Rust renderer for the
[MapLibre style spec](https://maplibre.org/maplibre-style-spec/) v8. One core
renders on wgpu: in the browser through wasm and WebGPU, and natively through
Vulkan, Metal and D3D12. AI agents write and steer it; a person sets them
going and reviews the result.

MapLibre is weighing a Rust renderer made by porting MapLibre Native's C++
core file by file ([mln-rs](https://github.com/louwers/mln-rs)). Orbis takes
the other road, a design of its own for the same spec, and sets out to beat
that port where users feel it: a smaller download, faster loading, faster
tile processing and faster rendering, in 2D and in 3D.

**Scope.** The v8 spec, including terrain, the globe, sky and the light on
extrusions, plus what the spec could reasonably grow into: other
projections, coordinate reference systems and tile matrix sets. A style's
root `projection` can name a CRS such as `{"type": "EPSG:3413"}`, and a
source's `tileMatrixSet` can name or define its tile grid. Nothing beyond
that: no style dialect of its own and no lighting beyond v8's `light`.

**Status.** Experimental and unpublished; every API can change. The C,
Kotlin and Swift bindings will be replaced by
[maplibre-native-ffi](https://github.com/maplibre/maplibre-native-ffi)'s C
API. `cargo xtask rt` measures it against GL JS's render fixtures,
`cargo xtask web` reports its download size and `cargo xtask bench` its
frame times. The live demo is at
<https://code.sargunv.dev/orbis/>.

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

Orbis is under the [BSD-2-Clause licence](LICENSE), as MapLibre Native is.
It ports code from MapLibre GL JS and other projects, tests against GL JS's
fixtures (fetched by `cargo xtask fixtures`, not committed), and ships
OpenStreetMap-derived demo data under ODbL; see
[THIRD_PARTY.md](THIRD_PARTY.md).

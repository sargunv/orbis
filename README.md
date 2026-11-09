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

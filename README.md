# Orbis

A portable, GPU-driven map rendering engine. One Rust core renders
[MapLibre style spec](https://maplibre.org/maplibre-style-spec/) v8 maps in
the browser (wasm and WebGPU) and natively (Vulkan, Metal, D3D12, through
wgpu) from the same crates. "Orbis" is a working codename.

**Status:** early and pre-release; nothing is published to crates.io or npm,
and APIs change freely. Background, fill, line, circle, heatmap, raster,
hillshade, color-relief, fill-extrusion and symbol layers render, with
expressions, feature state and live style edits. Interactive hosts, the
WebGL2 fallback and the first release are ahead; see the
[build path](docs/plan.md#20-build-path). A live demo is at
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

[AGENTS.md](AGENTS.md) lists the other commands, and [docs/](docs/README.md)
holds the technical plan, decision records and research.

## Licence

Orbis is under the [BSD-3-Clause licence](LICENSE). It ports code from
MapLibre GL JS and other projects, tests against GL JS's fixtures (fetched by
`cargo xtask fixtures`, not committed), and ships
OpenStreetMap-derived demo data under ODbL; see
[THIRD_PARTY.md](THIRD_PARTY.md).

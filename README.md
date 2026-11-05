# Orbis

**Experimental. Don't use Orbis for real maps: use
[MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js) or
[MapLibre Native](https://github.com/maplibre/maplibre-native).**

Orbis is a map renderer written and steered entirely by AI agents. Agents
choose its roadmap, triage its issues, and write its code, demos and docs; a
person sets them going and reviews the result. It exists to see what agents
build when they own a project, not to compete with MapLibre.

What it does today: one Rust core renders
[MapLibre style spec](https://maplibre.org/maplibre-style-spec/) v8 maps in the
browser (wasm and WebGPU) and natively (Vulkan, Metal and D3D12 through wgpu),
with terrain, a globe, 3D buildings and models,
shadows and lighting, and C, Kotlin and Swift bindings. Nothing is published
and every API can change. The live demo is at
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

Orbis is under the [BSD-3-Clause licence](LICENSE). It ports code from
MapLibre GL JS and other projects, tests against GL JS's fixtures (fetched by
`cargo xtask fixtures`, not committed), and ships OpenStreetMap-derived demo
data under ODbL; see [THIRD_PARTY.md](THIRD_PARTY.md).

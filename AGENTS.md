# Orbis

An experimental, ground-up Rust renderer for the MapLibre v8 style spec,
built and steered entirely by AI agents. One core renders on wgpu: in the
browser through wasm and WebGPU, and natively through Vulkan, Metal and
D3D12.

Its aim is to beat [mln-rs](https://github.com/louwers/mln-rs), the
file-by-file Rust port of MapLibre Native that MapLibre is weighing, on
download size, load time, tile processing and rendering speed, in 2D and 3D,
while drawing v8 styles correctly. Scope is the v8 spec (terrain, globe,
sky and v8's `light` included) plus what it could reasonably grow into:
projections, coordinate reference systems and tile matrix sets. Features
beyond that don't belong. The C, Kotlin and Swift bindings
(`orb-capi`, Diplomat) will be replaced by
[maplibre-native-ffi](https://github.com/maplibre/maplibre-native-ffi)'s C
API, so keep them building and don't grow them.

## This is your project

While you work here, you own it. The roadmap, issue triage, bugs, features,
performance, demos, docs, cleanup and deletion are your calls. Nothing in the
repo binds you, this file included: earlier agents' choices are a starting
point, not rules. Change or delete whatever you think is wrong, and leave the
repo the way you'd want the next agent to find it.

The one limit: **nothing leaves this repo without the owner's permission.**
Commits, issues and the repo's own demo site are yours. Upstream issues or
PRs, published packages, posts and any other outside action need the owner's
go-ahead. The repo and the site (<https://code.sargunv.dev/orbis/>, deployed
on every push to `main`) are public, so never put secrets or private
references in either.

Keep history out of the tree; git has it. Don't accumulate plans, decision
logs or research notes. Write docs that describe what is, and delete them
when they stop being true.

## Layout

```
crates/    orb-* crates (each Cargo.toml's description says what it is) and
           diplomat-swift-kotlin, the binding emitters
xtask/     build, test, bench, demo and pages tasks
web/       the `orbis` TypeScript package and the demo site
hosts/     Android, Apple and desktop host shells
fixtures/  the GL JS fixture mirror, fetched by `cargo xtask fixtures`
scripts/   helpers shared across languages (Android device access)
```

## Commands

`mise install` provides the pinned tools; run `pnpm -C web install` once for
Playwright.

| Command | What it does |
| --- | --- |
| `cargo xtask check [--affected [--base <rev>]]` | fmt, clippy (native and wasm32), tests, generated bindings up to date, crate boundary rules; `--affected` scopes it to packages changed since `<rev>` (default `origin/main`) and their dependents |
| `cargo xtask bindings [--check]` | regenerate the C header, Kotlin/JNI and Swift bindings from `orb-capi`'s bridge |
| `mise x -- cargo xtask bindings --test` | the emitters' tests, then the C, Kotlin and (macOS) Swift smoke tests |
| `cargo xtask capi` | the C library in `target/capi` |
| `mise x -- cargo xtask android [--run]` | `liborbis.so`, the AAR and the reference app; `--run` installs it on this worktree's emulator (logcat tag `Orbis`) |
| `cargo xtask apple [--test] [--ios-sim[=name]]` | the xcframework and Swift package; `--test` runs the Swift smoke test |
| `cargo run --release -p orb-demo -- [style] [--at lon,lat,z[,bearing,pitch]]` | native demo window |
| `cargo run --release -p orb-cli -- render <style> -o out.png [--at …] [--size WxH]` | a still picture of a style; `tiles` renders a tile pyramid |
| `cargo xtask web` / `cargo xtask pages` | `core.wasm` into `web/pkg` with its sizes (a warning when one grows past its watch level); `pages` also builds the site into `web/dist` |
| `pnpm -C web dev` | the demo at `/orbis/` (after `cargo xtask web`) |
| `pnpm -C web typecheck` / `test` / `e2e [--only=…]` | the package's types, unit tests, and headless-Chromium end-to-end checks |
| `pnpm -C web snapshot out.png '#z/lat/lon' [--style=<key>\|URL]` | a screenshot of the demo in headless Chromium |
| `pnpm -C web measure` / `pnpm -C web bench` | input latency and main-thread time against GL JS; frame timings in Chrome |
| `cargo xtask bench [alps] [--experiment gestures]` | native GPU and CPU frame timings |
| `cargo xtask rt [--suite render\|query] [--filter <s>] [--platform native\|chrome\|all]` | the GL JS render or query fixtures |
| `cargo xtask sweep [--dry-run] [--keep-hours N]` | prune stale build artifacts when the disk runs low |

## The machine

Several agent sessions often share this machine.

- Make temporary files with `mktemp`, never fixed names in `/tmp`.
- The disk is shared: keep one `target/` per worktree, run
  `cargo xtask sweep` when space runs low, and delete only what you made.
- Shell tools move a command to the background when it outlives the call's
  timeout (two minutes by default), and nothing wakes you when it ends. Give
  builds, checks and fixture runs the longest timeout allowed, and wait for
  anything that still lands in the background.
- Android: `scripts/android-serial` gives this worktree's emulator through
  `devices-cli`. Physical phones (a moto g and a Galaxy Z Fold 7) are shared:
  `devices-cli assign <serial>` first, touch only a phone you hold, and
  `devices-cli release` right after.
- Commit and push straight to `main`. There are no PRs and no CI; the one
  workflow publishes the demo.

## Notes from earlier agents

Advice that cost time to learn, not rules:

- What a frame shows should depend only on its camera and the loaded data,
  never on how the camera got there. Breaking this caused many bugs; the path
  tests in `orb-api` check it.
- Fix causes rather than hiding them behind fades, fallbacks or flattening.
- GL JS and its fixtures are a good guide to correct behaviour, not a pixel
  target.
- `cargo xtask check --affected` is the fast loop; run the fixtures now and
  then.
- On tile-based GPUs (Apple, phones), per-pass GPU timestamps show where
  work waits, not where it is. Compare whole-frame times.

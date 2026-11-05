# Orbis

A portable, GPU-driven map rendering engine: one Rust core that renders v8
MapLibre styles in the browser (wasm + WebGPU) and natively (Vulkan, Metal,
D3D12), with a reduced WebGL2/GLES fallback. Orbis is a working codename.

`docs/plan.md` is the technical plan: decisions D1–D24, architecture, and build
path M0–M12. Work is tracked in GitHub issues and milestones on
`sargunv/orbis`.

## Project map

The layout follows `docs/plan.md` §5.2:

```
crates/         orb-* crates, one per row of plan §5.2, and diplomat-swift-kotlin
                (the project-agnostic binding emitters, decision 0018)
xtask/          build, test, demo and pages tasks
fixtures/       GL JS fixture mirror, fetched by `cargo xtask fixtures`, not committed
scripts/        helpers the commands share across languages (Android device access)
web/            TypeScript host package and the demo site
hosts/          Android, Apple and desktop host shells
docs/           plan, research, decisions
  scratch/      gitignored scratch artifacts
```

## Commands

`mise install` provides the pinned tools (wasm-bindgen, wasm-opt, twiggy,
node, pnpm, a JDK, Gradle, kotlinc, CMake, Go and devices-cli); run
`pnpm -C web install` once for Playwright.

| Command | What it does |
| --- | --- |
| `cargo xtask check [--affected [--base <rev>]]` | fmt, clippy (native and wasm32), tests, generated bindings up to date, boundary rules; offline, no fixture mirror needed; `--affected` scopes it to the packages changed since `<rev>` (default `origin/main`) and their dependents |
| `cargo xtask bindings [--check]` | regenerate the C header (`crates/orb-capi/include`), Kotlin/JNI and Swift bindings from `orb-capi`'s bridge; run after changing it; `--check` fails when they're stale |
| `mise x -- cargo xtask bindings --test` | the emitters' Swift and Kotlin compile-and-run tests, then the C (pkg-config and CMake), Kotlin (JVM) and, on macOS, Swift smoke tests |
| `cargo xtask capi` | the C library in `target/capi`: static lib, headers, pkg-config and CMake files |
| `mise x -- cargo xtask android [--run]` | `liborbis.so` (arm64-v8a, NDK), the AAR and the Android reference app; `--run` installs it on this worktree's emulator and starts it (logcat tag `Orbis`) |
| `cargo xtask apple [--test] [--ios-sim[=name]]` | `hosts/apple/OrbisC.xcframework` (macOS, iOS, iOS simulator) and the Swift package; `--test` runs the Swift smoke test; `--ios-sim` builds the iOS app and runs it on a simulator (whose Metal lacks indirect draws: it reports that, M4's Compat tier is its path) |
| `cargo run --release -p orb-demo -- [style] [--at lon,lat,z[,bearing,pitch]] [--cache <dir>]` | native demo window through `orb-api`; `--cache` keeps state there (the HTTP cache, pipelines) |
| `… -p orb-demo -- file://$PWD/web/demo/data/alps.json --flight web/demo/data/alps-flight.json` | the terrain flight over the Alps (web: `?style=alps&flight`) |
| `cargo run --release -p orb-cli -- render <style> -o out.png [--at lon,lat,z[,bearing,pitch]] [--size WxH] [--a11y out.json]` | `orbis`: a still picture of a style or scene document (Still mode); `--a11y` also writes its semantic snapshot |
| `… -p orb-cli -- tiles <style> -o dir --tms <id\|tms.json> --zoom z[-z] [--bbox w,s,e,n]` | a tile pyramid on the scene's plane (Tile mode), `dir/z/col/row.png` |
| `cargo xtask web` | build `core.wasm` (size report, warns above plan §14.3) and `compat.wasm` into `web/pkg` |
| `cargo xtask pages` | `cargo xtask web`, then the Vite build into `web/dist`: the site `.github/workflows/pages.yml` deploys on every push to `main` |
| `pnpm -C web dev` | Vite dev server at `/orbis/` (run `cargo xtask web` first); the style editor at `/orbis/?style=playground&edit` |
| `pnpm -C web typecheck` / `pnpm -C web test` | type-check the `orbis` package (`web/src`); its unit tests and the build test |
| `pnpm -C web e2e [--only=scenario,edits,editor,offline] [--configs=webgpu,webgl2]` | headless Chromium: the binding scenario through the `orbis` package (`web/demo/scenario.html`) on each renderer, every edit class through the map app and the cost of a feature-state write, and the style editor's controls, JSON, hover and frames per edit; `offline` (opt-in, live network) downloads the live map's view as an offline region and reopens it with the network cut (`cargo test -p orb-api --test regions -- --ignored` natively) |
| `pnpm -C web snapshot out.png '#z/lat/lon' [--style=<key>\|URL] [--size=WxH] [--reopen[=offline]] [--still]` | the map app on the Tokyo extract (the site defaults to the live map) in headless Chromium, screenshot; `--reopen` times a reload from the browser's cache, `=offline` with the network cut; `--still` also writes the map's own snapshot (`<out>-still.png`) |
| `pnpm -C web measure [--android [--physical] \| --headed] [--runs=3] [--pipelines]` | plan §7.2's measurements in Chrome: input-to-photon latency (pan, pinch, rotate), DOM overlay skew, main-thread busy time and idle frames, for Orbis and the pinned GL JS (`web/demo/measure.html`); `--headed` in a window on the display; `--pipelines` instead times pipeline creation (#26) cold and warm: shaders and pipelines made, first draw, last pipeline, idle and the GPU queue's wait |
| `node web/scripts/open.mjs <app> measure.html pipelines` | a measurement page in a browser no driver controls (Firefox, Safari), its result posted back |
| `cargo xtask fixtures [expressions]` | fetch and verify the pinned GL JS fixture mirror into `fixtures/gl-js` (gitignored; no work when it matches; `fixtures/README.md`); `expressions` bundles the style-spec expression fixtures into `orb-expr` |
| `cargo xtask rt [--suite render\|query] [--filter <substring>] [--platform native\|chrome\|lavapipe\|all]` | run the GL JS render fixtures, or with `--suite query` the query fixtures (native and Chrome); fetches the mirror first; report in `docs/scratch/rt` |
| `cargo xtask bench` | GPU/CPU still-frame timings of five views; `--experiment gestures` for rotate, tilt and zoom frame times with tiles arriving (decision 0023); `alps --experiment gestures` for the terrain demo's rotate, pan and flight |
| `pnpm -C web bench [--precise] [--android [--physical]] [--experiment=gestures [--workers=N]]` | the same bench in headless Chromium, or in this worktree's Android emulator's Chrome (`web/demo/bench.html`; keep its screen on; `--physical` for an assigned phone, below); `--experiment=gestures` drives the `orbis` package (`--style=alps` for the terrain paths) |
| `cargo xtask demo-data` / `cargo xtask unicode` | regenerate the committed Tokyo extracts and their labelled styles; orb-text's character tables from the UCD |

## How we work

Progress over excessive rigor, but never at the cost of dirty shortcuts.

- **No hacks or tech debt for short-term progress.** Fix things in a principled
  way, or leave them open in an issue. No speculative abstraction, and no
  backward-compatibility machinery while there are no users.
- **Clean up as you go.** Delete dead code, stale docs and stale config in the
  same change that makes them stale.
- **Leave it smaller.** The first release (plan §20) wants the codebase
  simpler and net negative in lines. Prefer deleting, merging duplicate paths
  and collapsing indirection over adding code. When you touch a module,
  leave it simpler than you found it.
- **Move forward before polishing every detail.** Bugs will turn up; fix them
  forward.
- **Keep module interfaces neat and narrow**, so that a later bug squash can go
  module by module.
- **Keep the loop fast.** Validate what you touched and what it plausibly
  affects downstream, not everything every time: fmt, clippy (native and
  wasm32) and tests for the crates you changed and their dependents
  (`cargo xtask check --affected`), plus the specific check your change needs
  (a web build and a browser screenshot for web UI; a device run only for
  device-specific code such as GPU paths, hosts and drivers). Then rebase and
  push. Don't re-run checks after a clean rebase or a rejected push; rebase
  and push again. Re-check only if you resolved a conflict in code. If `main`
  breaks, fix it forward. The full `cargo xtask check` and the heavier checks
  (fixture runs, lavapipe, size budgets, other platforms) run occasionally and
  at milestone ends.
  Run long commands (checks, benches, fixture runs) in the foreground and
  wait for them: an agent that ends its turn while a background command runs
  is not woken when it finishes, and stalls. Shell tools move a command to the
  background when it outlives the call's timeout (two minutes by default), so
  give long commands the longest timeout the tool allows. If one still ends up
  in the background, wait for it before ending your turn (the host's
  background-wait tool, or poll its output file until it exits).
- **Commit and push straight to `main`.** No PRs, no CI; the one workflow
  only publishes the demo.
- **GL JS is a reference, not a pixel target.** What matters is that the map
  behaves well and styles render recognizably and correctly. The pinned GL JS
  and its fixtures are the best guide to that, but pixel-perfect parity isn't
  a goal. Don't complicate code to chase the last few pixels or reproduce
  GL JS internals; a fixture that differs slightly for a principled reason
  is fine once it's understood and noted.
- **Every frame is right for its camera.** A moving camera is the normal
  state of a map. What a frame shows must depend on the camera of that frame,
  not on whether the camera is moving or how it got there: no "update once
  the camera stops", no caches keyed on the camera that go stale mid-gesture,
  no gesture-only modes. If something looks wrong while moving, find the
  still frame that looks wrong and fix that.
- **Share the disk.** Many agents build on one machine. Don't make extra
  checkouts or second `target/` directories to build old commits; compare
  against a baseline by checking the commit out in your own worktree (or
  ask for the numbers), and delete scratch builds you made as soon as
  you're done. If the disk fills, stop and report it; delete only files you
  created.
- **Measure.** Report sizes, timings and counters as numbers.

### Devices

Functional Android checks run on this worktree's own emulator.
`scripts/android-serial` prints the adb serial every script uses:
`ANDROID_SERIAL` when you set one, otherwise `devices-cli get android --api
$ORB_ANDROID_EMULATOR_API`, which boots an emulator for this worktree on demand
and stops it when the worktree goes away. `devices-cli ls` shows what runs,
`devices-cli release` stops this worktree's devices, and `devices-cli gc`
collects stale ones. `cargo xtask android --run` installs and starts the
reference app there; the emulator's software Vulkan draws only the background
and crashes Chrome's WebGPU GPU process (#54), so it checks the build, the
install and the load path, not rendering fidelity or performance, and
`bench`/`measure --android` on the emulator fail the same way.

Physical phones are for performance numbers from a real slow phone, which is
rarely needed since the implementation is universal: a moto g (Adreno 619) and a
Galaxy Z Fold 7 (Adreno 830) are connected. Assigning is how they serialize
across agents: `devices-cli assign <serial>`, run the measurement with
`--physical` (which fails rather than fall back to the emulator), then
`devices-cli release` right after. Never hard-code a serial, change device
settings (brightness, stay-awake) or force-stop apps you didn't start.

## Orchestration

Opus is the project manager and orchestrator, and does most of the work
directly. After M2, work runs in two parallel lanes (plan §20), rendering and
platform, with one or two sub-lanes each at a time.

- **Delegate** through T3 orchestration: implementation work, grind work
  included, to Opus 5.5, and easy or mechanical work to DeepSeek Flash on pi
  (`opencode-go/deepseek-v4.1-flash`, not OpenCode). Avoid big parallel
  fan-outs of Opus.
- **Sol** (`gpt-6.1-sol` on Codex) reviews and hunts bugs for a second
  opinion; it doesn't write code. Routine review loops are not worth the
  tokens. Use a Sol loop (the `sol-loop` skill) for critical pieces, such as
  `unsafe`, concurrency, GPU resource lifetimes, or keystone interfaces.

## Architecture rules

These come from the plan and change only with the owner's sign-off, as do the
tiers, the crate graph and D1–D24. Details the work refines or proves wrong
(numbers, sub-steps, specifics) are updated in `docs/plan.md` in the same
commit.

- Web and native are built from the same crates at every milestone (D3).
- Below `orb-runtime`: no threads, no I/O, no GPU handles. Use `unsafe` only
  where plan §5.3 allows it. No public item names a CRS.

## Issues and decisions

Use issues and milestones as they're useful: for follow-ups, bugs found but not
yet fixed, and decisions only the owner can make. Commits that finish an issue
end with `Closes #n`. Record significant decisions, the ones the owner should
know about, in `docs/decisions/NNNN-<slug>.md` and mention them in the
milestone summary. Several agents add records at once: after your last rebase,
before pushing, check `origin/main` and renumber yours (file, title, README
and every reference) if the number has been taken.

Milestone descriptions carry exit criteria. Treat them as goals and report
against them when a milestone wraps up. Every push to `main` publishes the
demo to Pages (https://code.sargunv.dev/orbis/) through the Pages workflow. The
site is public; never publish anything sensitive to it.

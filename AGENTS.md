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
web/            TypeScript host package and the demo site
hosts/          Android, Apple and desktop host shells
docs/           plan, research, decisions
  scratch/      gitignored scratch artifacts
```

## Commands

`mise install` provides the pinned tools (wasm-bindgen, wasm-opt, twiggy,
node, pnpm, a JDK, Gradle, kotlinc, CMake); run `pnpm -C web install` once for
Playwright.

| Command | What it does |
| --- | --- |
| `cargo xtask check` | fmt, clippy (native and wasm32), tests, generated bindings up to date, boundary rules; offline, no fixture mirror needed |
| `cargo xtask bindings` | regenerate the C header (`crates/orb-capi/include`), Kotlin/JNI and Swift bindings from `orb-capi`'s bridge; run after changing it |
| `mise x -- cargo xtask bindings test` | the emitters' Swift and Kotlin compile-and-run tests |
| `mise x -- cargo xtask android [--run[=serial]]` | `liborbis.so` (arm64-v8a, NDK), the AAR and the Android reference app; `--run` installs and starts it (logcat tag `Orbis`) |
| `cargo xtask apple [--test] [--ios-sim[=name]]` | `hosts/apple/OrbisC.xcframework` (macOS, iOS, iOS simulator) and the Swift package; `--test` runs the Swift binding scenario; `--ios-sim` builds the iOS app and runs it on a simulator (whose Metal lacks indirect draws: it reports that, M4's Compat tier is its path) |
| `hosts/apple/.build/release/OrbisDemo [--snapshot out.png] [tokyo \| style]` | macOS reference app (after `cargo xtask apple`): live Liberty as on the web, `tokyo` for the extract; `--snapshot` renders into a texture target, prints timings and pipeline counts |
| `mise x -- cargo xtask capi --test` | the C library in `target/capi` (static lib, headers, pkg-config, CMake), the C smoke test through both, and the Kotlin binding scenario on the JVM |
| `cargo run --release -p orb-demo -- [style] [--at lon,lat,z[,bearing,pitch]]` | native demo window |
| `… -p orb-demo -- --snapshot out.png` | render offscreen until idle, write a PNG |
| `… -p orb-demo -- file://$PWD/web/demo/data/alps.json --flight web/demo/data/alps-flight.json` | the terrain flight over the Alps (web: `?style=alps&flight`) |
| `cargo xtask web` | build `core.wasm` (size report, warns above plan §14.3) and `harness.wasm` (bench and fixture entry points) into `web/pkg` |
| `pnpm -C web snapshot out.png '#z/lat/lon' [--thread=main\|worker] [--style=<key>\|URL] [--size=WxH]` | the map app on the Tokyo extract (the site defaults to the live map) in headless Chromium, screenshot; `--page=playground.html\|cities.html` follows an old page's redirect |
| `ORB_PLAYGROUND_STYLE=<mktemp path> pnpm -C web dev` | the map app with its style editor at `/orbis/?style=playground&edit`; writes accepted styles and hover state to that file (dev only) |
| `cargo run --release -p orb-demo -- file://$PWD/web/demo/data/playground.json --watch-style <same path>` | native viewer beside the style editor; follows that file as diffed transactions; logs generation, frames and edit ms |
| `pnpm -C web playground` | headless style editor checks (controls, JSON, hover, frames per edit) in main and worker modes |
| `ORB_PLAYGROUND_STYLE=<mktemp path> pnpm -C web playground:native` | dev bridge smoke test (build `orb-demo --release` first) |
| `pnpm -C web edits` | every edit class through `globalThis.orbis.edit` in both thread modes, and the cost of a feature-state write |
| `cargo xtask fixtures` | fetch and verify the pinned GL JS fixture mirror into `fixtures/gl-js` (gitignored; no work when it matches; `fixtures/README.md`) |
| `cargo xtask rt [--suite render\|query] [--filter <substring>] [--platform native\|chrome\|lavapipe\|all]` | run the GL JS render fixtures, or with `--suite query` the query fixtures (native and Chrome); fetches the mirror first; report in `docs/scratch/rt` |
| `cargo xtask pages` | `cargo xtask web`, then the Vite build into `web/dist`: the site `.github/workflows/pages.yml` deploys on every push to `main` |
| `pnpm -C web dev` | Vite dev server at `/orbis/` (run `cargo xtask web` first) |
| `pnpm -C web typecheck` / `pnpm -C web test` | type-check the `orbis` package (`web/src`); its unit tests and the build test |
| `cargo xtask bench` | GPU/CPU frame timings, still and with the camera moving (decision 0002); `--experiment paint` for decision 0017; `--experiment gestures` for rotate, tilt and zoom frame times with tiles arriving (decision 0023) |
| `pnpm -C web bench [--precise] [--android[=serial]]` | the same bench in headless Chromium, or in an adb-connected Android device's Chrome (`web/demo/bench.html`; keep its screen on); `--experiment=gestures` drives the `orbis` package in both thread modes |
| `pnpm -C web measure [--android[=serial]] [--runs=3]` | plan §7.2's measurements in Chrome: input-to-photon latency (pan, pinch, rotate), DOM overlay skew, main-thread busy time and idle frames, for Orbis worker and main-thread mode and the pinned GL JS (`web/demo/measure.html`) |

## How we work

Progress over excessive rigor, but never at the cost of dirty shortcuts.

- **No hacks or tech debt for short-term progress.** Fix things in a principled
  way, or leave them open in an issue. No speculative abstraction, and no
  backward-compatibility machinery while there are no users.
- **Clean up as you go.** Delete dead code, stale docs and stale config in the
  same change that makes them stale.
- **Move forward before polishing every detail.** Bugs will turn up; fix them
  forward.
- **Keep module interfaces neat and narrow**, so that a later bug squash can go
  module by module.
- **Keep the loop fast.** Validate what you touched and what it plausibly
  affects downstream, not everything every time: fmt, clippy (native and
  wasm32) and tests for the crates you changed and their dependents
  (`cargo xtask check --affected`; until it lands, `cargo clippy -p` and
  `cargo test -p` for those crates), plus the specific check your change needs
  (a web build and a browser screenshot for web UI; a device run only for
  device-specific code such as GPU paths, hosts and drivers). Then rebase and
  push. Don't re-run checks after a clean rebase or a rejected push; rebase
  and push again. Re-check only if you resolved a conflict in code. If `main`
  breaks, fix it forward. The full `cargo xtask check` and the heavier checks
  (fixture runs, lavapipe, size budgets, other platforms) run occasionally and
  at milestone ends.
- **Commit and push straight to `main`.** No PRs, no CI; the one workflow
  only publishes the demo.
- **GL JS is a reference, not a pixel target.** What matters is that the map
  behaves well and styles render recognizably and correctly. The pinned GL JS
  and its fixtures are the best guide to that, but pixel-perfect parity isn't
  a goal. Don't complicate code to chase the last few pixels or reproduce
  GL JS internals; a fixture that differs slightly for a principled reason
  is fine once it's understood and noted.
- **Measure.** Report sizes, timings and counters as numbers.

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

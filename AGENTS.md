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
crates/         orb-* crates, one per row of plan §5.2
xtask/          build, test, demo and pages tasks
fixtures/       GL JS fixture mirror, fetched by `cargo xtask fixtures`, not committed
web/            TypeScript host package and the demo site
hosts/          Android, Apple and desktop host shells
docs/           plan, research, decisions
  scratch/      gitignored scratch artifacts
```

## Commands

`mise install` provides the pinned tools (wasm-bindgen, wasm-opt, twiggy,
node, pnpm); run `pnpm -C web install` once for Playwright.

| Command | What it does |
| --- | --- |
| `cargo xtask check` | fmt, clippy (native and wasm32), tests, boundary rules; offline, no fixture mirror needed |
| `cargo run --release -p orb-demo -- [style] [--at lon,lat,z[,bearing,pitch]]` | native demo window |
| `… -p orb-demo -- --snapshot out.png` | render offscreen until idle, write a PNG |
| `cargo xtask web` | build `core.wasm` (size report, warns above plan §14.3) and `harness.wasm` (bench and fixture entry points) into `web/demo/pkg` |
| `pnpm -C web snapshot out.png '#z/lat/lon' [--thread=main\|worker]` | web demo in headless Chromium, screenshot |
| `ORB_PLAYGROUND_STYLE=<mktemp path> pnpm -C web dev` | playground at `/orbis/playground.html`; write accepted style and hover snapshots to that file (dev only) |
| `cargo run --release -p orb-demo -- file://$PWD/web/demo/data/playground.json --watch-style <same path>` | native viewer beside the playground; follows that file as diffed transactions; logs generation, frames and edit ms |
| `pnpm -C web playground` | headless playground checks in main and worker modes |
| `ORB_PLAYGROUND_STYLE=<mktemp path> pnpm -C web playground:native` | dev bridge smoke test (build `orb-demo --release` first) |
| `pnpm -C web snapshot out.png --page=playground.html` | headless playground screenshot |
| `cargo xtask fixtures` | fetch and verify the pinned GL JS fixture mirror into `fixtures/gl-js` (gitignored; no work when it matches; `fixtures/README.md`) |
| `cargo xtask rt [--filter <substring>] [--platform native\|chrome\|lavapipe\|all]` | run the GL JS render fixtures (fetches the mirror first); report in `docs/scratch/rt` |
| `cargo xtask pages` | build with Vite into `web/dist` and publish to the public Pages site; waits for GitHub's deployment |
| `pnpm -C web dev` | Vite dev server at `/orbis/` (run `cargo xtask web` first) |
| `cargo xtask bench` | GPU/CPU frame timings, still and with the camera moving (decision 0002); `--experiment paint` for decision 0017 |
| `pnpm -C web bench [--precise] [--android[=serial]]` | the same bench in headless Chromium, or in an adb-connected Android device's Chrome (`web/demo/bench.html`; keep its screen on) |

## How we work

Progress over excessive rigor, but never at the cost of dirty shortcuts.

- **No hacks or tech debt for short-term progress.** Fix things in a principled
  way, or leave them open in an issue. No speculative abstraction, and no
  backward-compatibility machinery before the first release.
- **Clean up as you go.** Delete dead code, stale docs and stale config in the
  same change that makes them stale.
- **Move forward before polishing every detail.** Bugs will turn up; fix them
  forward.
- **Keep module interfaces neat and narrow**, so that a later bug squash can go
  module by module.
- **Keep the loop fast.** Before pushing, run `cargo xtask check` (fmt,
  clippy natively and for wasm32, tests, and the plan §5.3 boundary rules).
  Run the heavier checks (fixture runs, lavapipe, size budgets, other
  platforms) occasionally and at milestone ends, not on every push.
- **Commit and push straight to `main`.** No PRs, no CI.
- **GL JS is a reference, not a pixel target.** What matters is that the map
  behaves well and styles render recognizably and correctly. The pinned GL JS
  and its fixtures are the best guide to that, but pixel-perfect parity isn't
  a goal. Don't complicate code to chase the last few pixels or reproduce
  GL JS internals; a fixture that differs slightly for a principled reason
  is fine once it's understood and noted.
- **Measure.** Report sizes, timings and counters as numbers.

## Orchestration

Opus is the project manager and orchestrator, and does most of the work
directly.

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
milestone summary.

Milestone descriptions carry exit criteria. Treat them as goals and report
against them when a milestone wraps up. Publish the demo to Pages
(https://code.sargunv.dev/orbis/, `gh-pages` branch) when there's something
significant to see. The Pages site is public; never publish anything sensitive
to it.

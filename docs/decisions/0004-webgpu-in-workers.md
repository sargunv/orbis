# 0004: WebGPU in workers

Status: recorded at M0; worker mode stays the web default where
the probe passes.

## Question

Plan §7.2 renders the web scene in a dedicated worker on an `OffscreenCanvas`
by default, with `thread: "main"` as the fallback. M0 exit criterion 7 asks
which browsers actually offer WebGPU in workers. (Whether worker mode stays the
default is decided at M3 by latency measurements; this record only covers
availability.)

## Rule

Worker mode is the default in a browser when a dedicated worker gets a WebGPU
adapter and device and the page can `transferControlToOffscreen`; otherwise the
page falls back to main-thread mode. `web/demo/main.js` probes this at start-up.

## Evidence

`https://code.sargunv.dev/orbis/probe.html` (`web/demo/probe.html`), on macOS:

| Browser | Page: adapter, device | Worker: adapter, device | OffscreenCanvas transfer | Storage buffers per stage | Notes |
|---|---|---|---|---|---|
| Chrome 156 (headless, Playwright) | yes, yes | yes, yes | yes | 8+ | SwiftShader without flags; Metal with `--enable-unsafe-webgpu` |
| Safari 26.6.2 | yes, yes | yes, yes | yes | 44 | vendor/architecture `apple`; `timestamp-query`, `shader-f16` |
| Firefox 159 | yes, yes | yes, yes | yes | 9 | vendor/architecture hidden; `timestamp-query`, `shader-f16` |

Playwright's patched Firefox 157 exposes no `navigator.gpu` at all, so it can't
stand in for release Firefox in automation.

## Outcome

All three engines offer WebGPU in dedicated workers on macOS, so worker mode is
the default everywhere it was probed. Two constraints carry forward:

- Firefox's 9 storage buffers per stage sits at WebGPU's default of 8 that the
  plan assumes (§4.2). The draw pipelines bind 7 in the vertex stage; new
  bindings need a budget, not an assumption of more.
- Firefox hides adapter vendor and architecture, so the quirks table (§8.8)
  can't key off them there.

Phones (Chrome Android, Safari iOS 26) are probed at the phone step.

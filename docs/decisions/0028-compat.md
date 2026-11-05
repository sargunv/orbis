# 0028: The Compat tier: Modern's commands beneath the HAL, a whole compat.wasm, and the device policy

M4 builds the Compat tier (plan D8, §8.7), device-loss recovery and the
device policy (§8.8). This records how it was built, where the work refined
the plan, and what it measured.

## One command stream, emulated beneath the HAL

Compat draws Modern's render nodes, bundles and shaders. What WebGL2 and
GLES 3.0 lack is emulated inside `orb-hal-gl`, under the WebGPU-shaped
`Gpu` trait:

- **Storage buffers are data textures.** Every storage buffer a pipeline
  binds is an R32UI texture 4096 words wide. Draw shaders read the scene's
  tables through a `fetch` module: `fetch.wgsl` indexes storage arrays on
  Modern, `fetch_compat.wgsl` reassembles rows with `texelFetch` on Compat.
  The plan had arenas bound as vertex and index buffers; pulling vertices
  from data textures keeps one shader per family and needed no attribute
  layouts.
- **Indirect draws read CPU copies.** `orb-render`'s Compat cull is a port
  of `cull.wgsl` over CPU copies `orb-scene` keeps of its tile, chunk,
  candidate, node and index tables (on Compat only). It writes what the
  dispatch writes: visible lists, compacted index slots and draw arguments.
  `orb-hal-gl` turns `DrawIndirect` into direct draws from its copy of the
  arguments.
- **Bundles are recorded command lists**; a GL state cache after GL JS's
  skips redundant state.
- **DEM tiles are prepared by drawing** into each layer
  (`create_layer_view`, new in the HAL) rather than by compute.
- **Line labels need no CPU pass.** Modern already places glyphs along
  lines in the symbol vertex shader, and that shader runs on Compat too, so
  §8.6's "CPU line labels" never had to be built.

WGSL becomes GLSL ES 3.00 offline. With `orb-shaders`' `glsl` feature, a
build script translates each Compat composition with naga, one program per
shader and paint class (and, until decision 0038, symbol `PASS` value). A naga IR pass gives a texture
sampled with two samplers one alias per extra sampler, since GLSL ES has
only combined samplers. ES 3.00 lacks `bitCount`, `findLSB` and `ldexp`, so
the shaders call them through `builtins.wgsl`, which Compat replaces with
polyfills. GL fixes a texture's target when it is made, so `TextureDesc`
carries the sampled view dimension, as WebGPU compatibility mode's
`textureBindingViewDimension` does. WebGL rejects element indices from
2^31, so Compat's compacted indices fit in 31 bits.

Modern is unchanged by all this. Its fixtures on Metal keep their statuses,
apart from #18's known flake. The rows, sizes and counters are the same.
The `fetch` indirection costs Modern nothing measurable: 16 Vulkan
pipelines on the Adreno 619 take 17.03 s cold against 16.88 s at the base,
and about 1.3 s warm on both. On the web, `core.wasm` grew by 3 KiB Brotli
(724 against 721) for the two-tier canvas and its recovery entry points.

## compat.wasm is a whole renderer

Rust's wasm32 target has no dynamic linking, so a lazy side module can't
share the core's state. `compat.wasm` is therefore `orb-web` built with
`compat` instead of `modern`: the renderer role on WebGL2. Only the map
loads it, where WebGPU gives no adapter or a WebGPU device is lost for
good; job and placement workers keep `core.wasm`. It weighs 841 KiB
Brotli against `core.wasm`'s 817 KiB at the same commit, where §14.3
assumed "+60–90 KB".
Pages with WebGPU never fetch it.

## Device policy

`orb-hal::device` holds data and pure logic; hosts supply the adapter
facts and the storage:

- **Allowlist.** Vulkan is Modern on Android 12+ for Qualcomm, ARM and
  Intel GPUs, and for Imagination GPUs from Android 16. Samsung Xclipse is
  pending. Everything else draws on GLES.
- **Quirks table.** It holds Qualcomm's broken indirect validation
  (skipped), and #23's two findings: the Adreno 830's cull miscompile,
  fixed by a shader rewrite and kept out by a test, and Samsung Internet's
  missing `TRANSIENT_ATTACHMENT`, dropped when the browser lacks it.
- **Crash sentinel.** Before a Modern device makes its first pipelines,
  `orb-api` writes a marker naming the device and driver to the host's
  state directory, and clears it once a frame has drawn. If the marker is
  found for the same device at the next start, that device starts on
  Compat; a driver update tries Modern again. A test injects the crash it
  exists for: a child process aborts while making its first pipelines
  (`ORB_FAULT_INJECTION=pipelines`), and the next runtime demotes to
  Compat and draws the same pixels. Pipelines are not yet made on a
  watchdog thread (issue).

## Device loss

`Map::reset_gpu` drops every GPU resource and installs the style again
under the same style id. The jobs therefore lay tiles out from their
cached frames, refetching only what they lack. `orb-api`'s
`Runtime::recover` makes a new device by the same policy, restarting the
pipeline or program cache; maps and texture targets rebuild on their next
frame. On the web, the package first tries the same tier on the same
canvas. Failing that, it switches to WebGL2 in a fresh canvas with a new
`compat.wasm` instance and its own workers. The map emits a capability
event with the tier (C ABI event kind 11).

Loss to first and complete frame, Tokyo extract:

| Where | Path | First frame | Complete |
|---|---|---:|---:|
| headless Chrome, Apple M6 | WebGPU → WebGPU | 18 ms | 698 ms |
| headless Chrome, Apple M6 | WebGPU → WebGL2 (lazy compat.wasm) | 126–159 ms | 865–883 ms |
| headless Firefox 157, macOS | WebGL2 → WebGL2 | 0.8–1.7 s | 1.6–2.6 s |
| moto g (Adreno 619), GLES, during a pan | new device / complete | 166–564 ms | 1.45–1.67 s |
| `orb-api` test, Apple M6 (debug) | Modern / Compat (ANGLE) | | 29–60 / 358–557 ms |

The plan's targets are ≤1 s on a mid-range phone and ≤2 s for a WebGPU →
WebGL2 switch. Desktop meets both; the phone's GLES recovery takes
1.45–1.67 s to a complete frame, over the 1 s target.

## Native Compat

Compat natively is GLES 3.0 through EGL, with libEGL loaded at run time:
Mesa (llvmpipe in Docker), Android's driver, and ANGLE on macOS for
development. A desktop GL 3.3 core profile isn't built: Mesa and NVIDIA
expose GLES through EGL wherever they expose GL. Compat draws into
offscreen textures and Android windows; desktop windows on Compat are
open. Program binaries are cached the way Vulkan pipelines are. On the
Adreno 619, 15 programs compile in 17.0 s cold (the first launch blocks
the UI thread) and load from the cache in about 99 ms warm; load to idle
goes from 18.4 s cold to 1.35–1.47 s warm.

## Fixtures

The Compat subset is the mirror minus what hits a declared gap. Those
fixtures skip with the gap's warning as their reason (`compat:terrain`,
`compat:heatmap-float-targets`).

| Platform | Pass | Fail | Skip | In scope |
|---|---:|---:|---:|---:|
| llvmpipe (Mesa GLES 3, Docker) | 1018 | 31 | 147 | 97.04% |
| Chrome WebGL2 (ANGLE on Metal) | 1022 | 27 | 147 | 97.43% |
| ANGLE on Metal, native EGL | 995 | 92 | 109 | 91.54% before DEM prepare and gap warnings |

`liborbis.so` (arm64) is 11.74 MB with both backends, against 10.86 MB
Modern-only at the base. The plan's target is ≤8 MB.

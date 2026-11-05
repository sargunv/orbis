# Orbis: technical plan for a portable, GPU-driven map engine

*A technical plan: what to build, how it fits together, and in what order.*

**Codename and framing.** "Orbis" is a neutral codename; identifiers use placeholders `orb-` (crates), `orb_` (C ABI), `orbis` (web package, unpublished). Orbis is a new, third map engine alongside MapLibre Native and GL JS. It replaces neither, reads the same inputs (style spec v8, MVT, MLT, PMTiles, glyph and sprite protocols), learns from both, and owes neither API compatibility.

**Tags.** **[verified]**: seen in code, git, GitHub, a measured artifact or a primary page; **[verified-2nd]**: a secondary web source; **[reviewer-reported]**: a red-team claim consistent with sources, not reproduced; **[inferred]**: a judgement, estimate or target. Every size, time and throughput target is [inferred] unless marked.

---

## 0. Summary

### 0.1 Thesis

Some things are structurally out of reach for both existing engines:
- **One engine on native and web** without Emscripten, COOP/COEP or JSPI. Native-on-wasm is 2.21 MB br and needs all three [verified].
- **A GPU-driven frame.** GL JS issues 449.6 state-changing draws per frame for Liberty at Tokyo z15 (#8474); Native's drawables rewrite measured 0.86× legacy CPU throughput (#2787); 996 renderables were drawn for 4 visible labels (#3223) [verified].
- **A geodetic world.** Both engines wrap Mercator tile space onto a unit sphere [verified]; ECEF/ellipsoid scenes and any OGC TileMatrixSet need a different world model.
- **Guest embedding** on a host's GPU device and depth buffer, and an **accessibility tree** built from placement.

Three ideas:
1. **One small Rust core implements every semantic concern once.** Sans-thread, never blocking, compiled natively and to plain `wasm32-unknown-unknown`. Hosts are thin; bindings are generated from one annotated API.
2. **The GPU owns per-frame work.** Geometry, properties, feature state and symbol state live in persistent GPU buffers; compute culls and compacts; draws are indirect, from compacted indices for meshes and with vertex pulling for instanced quads; a camera-only frame writes a few small blocks and replays an unchanged command stream. The design targets WebGPU, Vulkan, Metal and D3D12; older APIs get a separate reduced tier that never constrains the main path.
3. **The world is a surface, not a square.** Each source declares an OGC TileMatrixSet; the world is a plane, sphere or WGS84 ellipsoid; a per-source planner maps data onto it. Web Mercator is one registry entry with an affine fast path.

### 0.2 What it is and is not

| It is | It is not |
|---|---|
| An embeddable interactive engine that **runs from one core in browsers (wasm + WebGPU) and on desktop, iOS and Android** | a GL JS or Native replacement, fork or API clone |
| A consumer of **unmodified v8 styles**, GL JS semantics by default, within a declared subset per release | a source of new keys in v8 JSON; new capabilities go in a separate **scene document** |
| Designed from day one for any CRS, globe/ellipsoid, 3D and guest embedding, **shipped as later milestones** | a general 3D engine, GIS toolkit, router or editor |
| A low-level engine with host helpers and generated bindings | first-party declarative UI SDKs or controls (wrappers such as maplibre-compose own those) |
| GPU-driven on modern APIs, with a reduced Compat tier elsewhere | shaped by WebGL2/GLES 3.0 limits |

Also out: a JS VM in styles (Tangram-ES embedded Duktape and JavaScriptCore) [verified]; SharedArrayBuffer, JSPI or Memory64 on the default web path; SQLite offline databases; Mapbox-only spec features; WebGL1 and GLES 2.

### 0.3 Build order

The plan is a buildout; it tracks no release. M0–M2 build the 2D core: v8 layers background, fill, line, circle, symbol, raster, heatmap, hillshade, color-relief, fill-extrusion; all 88 expression operators; feature and global state; MVT, MLT, GeoJSON, raster, raster-dem, PMTiles, MBTiles; the full text stack and placement; in browsers (WebGPU) and natively. Then two lanes run in parallel (§20). The **platform lane** puts an interactive map on every target from one core (M3–M5): gestures, camera, queries, events; a TS package, Rust crate, C ABI (0.x), generated Kotlin and Swift bindings, Android and Apple host helpers; browsers on WebGL2 via Compat; Windows, macOS, Linux, Android, iOS; then hardening, batteries and the offline cache. The **rendering lane** widens what renders (M6–M8): any CRS and servers; globe, ellipsoid, terrain and atmosphere; 3D content and lighting. Guest embedding, accessibility host mappings, data layers and stable contracts follow (M9–M12).

---

## 1. Decisions at a glance

Themes: **ARCH** architecture, **PORT** portability, **PERF** performance, **FLEX** future scenes, **DEVEX** maintainability, **SPEC** style and data contracts.

| # | Decision | Key evidence | Theme |
|---|---|---|---|
| D1 | A new, third engine; contracts shared, code shared only by borrowing liberally licensed pieces | 6.8% lineage left between the engines [verified] | ARCH |
| D2 | Rust (stable) core, WGSL shaders, TypeScript only in the web host; no Emscripten | Native-on-wasm: 2.21 MB br, COOP/COEP, JSPI [verified] | PORT |
| D3 | **Portable from M0**: every milestone renders and is tested in a browser and natively from the same crates | the FFI's late web port needed pthreads, a 512 MB heap and JSPI [verified] | PORT |
| D4 | Sans-thread core: scene state machine + pure jobs behind an `Executor` (Owned, HostDriven, Web, Inline); sync snapshot reads; async commands with generations | ffi#760's model in production; 14–16 hops per uncached tile on Native Android [verified] | ARCH |
| D5 | **Modern tier is the design centre**: WebGPU core; Vulkan 1.1+, Metal, D3D12; storage buffers in every stage, compute, indirect draws | WebGPU default in Chrome, Safari 26, Firefox Windows/macOS (gpuweb wiki) [verified] | PERF, PORT |
| D6 | **GPU-driven submission for every 2D family**: ≤256-triangle chunks, one cull/compact dispatch, one indirect draw per (node, arena page) — indexed from compacted indices for meshes, vertex-pulled for instanced quads (decision 0002) — commands replayed as a bundle | GM2.8 cut uploads ~430 → ~8 yet saved 4.7% of `Painter.render` [verified] | PERF |
| D7 | **M+ fast paths** (multi-draw-indirect, binding arrays, subgroups, `shader-f16`) detected at runtime; never required | MDI not in WebGPU core; `subgroups` in Chrome 134 [verified-2nd] | PERF |
| D8 | **Compat tier**: one `glow` executor (WebGL2, GLES 3.0, GL 3.3) on the same data; CPU cull, direct draws, data textures; declared reduced features; lazy on the web | no WebGPU on iOS < 26 or Firefox Linux/Android; Chrome trusts Android Vulkan only on 12+; wgpu's GL path +1.7 MB raw [verified] | PORT |
| D9 | **Unsupported**: WebGL1, GLES 2, no wasm SIMD, Android < API 24, iOS < 16; WebGPU compat mode is not a tier | compat mode has 0 vertex-stage storage buffers by default [verified] | PORT |
| D10 | WGSL only; a build-time composer picks storage-pull or attribute/texture fetch; naga emits GLSL ES 3.00 for Compat; `override` specialization per binding-mask class | 18,639 lines of shader text in 4 dialects; 26–28 ms sync builds per GL JS load (#8338) [verified] | DEVEX |
| D11 | Web: **render worker + ui role on main** by default; `thread: "main"` behind the same API; confirmed by measurement in M3 | GL JS main busy 2,089 ms vs worker 609 ms (#8426) [verified]; +1 frame latency in workers [reviewer-reported] | PERF, PORT |
| D12 | One `core.wasm` in three roles; share-nothing workers; SIMD128; ≤450 KB br target, 550 ceiling, watched from M0 | GL JS 247 KB br; maplibre-rs WebGPU 401 KB br without symbols [verified] | PORT |
| D13 | Columnar `TileFrame` on `mlt-core`; columns upload as GPU tables; GeoArrow zero-copy natively, copy-once on the web | both engines re-materialize MLT per feature [verified] | PERF, SPEC |
| D14 | Unmodified v8 → typed IR, frozen spec and fixture snapshot per release, GL JS defaults, opt-in deviations; new capabilities in a **scene document** | 35.2M style-spec downloads/month; 32 expression divergences [verified] | SPEC |
| D15 | `Surface` = Plane / Sphere / Ellipsoid; OGC 2D-TMS per source; DataToWorld planner; ground-resolution camera; **style `zoom` is a per-view style input**: GL JS's Web Mercator convention on Web Mercator and the v8 globe, the plane's own scale on other planes (decision 0022) | #168 is GL JS's most-reacted issue; planar CRS needs "no new GLSL" (#8723) [verified] | FLEX |
| D16 | Font-first text in jobs: ICU4X everywhere, bidi, HarfRust by glyph ID, SDF/MSDF, one atlas | RTL reversed on Native (#4513); TinySDF 27.6% of GL JS main-thread load (#8426) [verified] | FLEX |
| D17 | Deterministic CPU collision on a placement lane; GPU compaction, fades, anchors, line-label placement | GPU collision breaks 448/414 symbol fixtures [verified] | PERF |
| D18 | One annotated Rust API (a Diplomat bridge) generates the C ABI and the Kotlin/JNI and Swift bindings; Orbis owns those two emitters (decision 0018) | Diplomat's Kotlin is JNA at 369 ns/call vs JNI 4.1 ns, no Swift backend [verified, decision 0018] | PORT, DEVEX |
| D19 | Guest embedding over raw Vulkan/Metal/D3D12 handles; Orbis records its own command buffers | wgpu 30.0.1 cannot wrap foreign command buffers [verified] | FLEX |
| D20 | One extension model, per-runtime registries, built-ins use it, WGSL plugin shaders | maplibre-plugins writes every plugin twice [verified] | FLEX |
| D21 | Read-only fixture mirror with own tags; real-world style suite as gate; Modern runs all, Compat its subset; web/native identity checks | 1,186 of 1,250 shared render tests identical in meaning [verified] | DEVEX |
| D22 | One Cargo build + `xtask`; no submodules; CI budgets for size, dependencies, LOC, `unsafe` | Native: 9+ build systems, CI p50 105–115 min [verified] | DEVEX |
| D23 | Accessibility snapshot from M2; platform mappings at M10 | no accessibility tree in either engine [verified] | FLEX |
| D24 | Offline = PMTiles extracts + HTTP cache | Native's per-hit fsync: 1,011 of 1,327 ms cold open (#4748) [verified] | DEVEX |

---

## 2. Design principles

1. **One implementation per concern, one owning crate**; CI keeps a concern → crate registry.
2. **The core never blocks, spawns threads or touches the OS** below `orb-runtime`; kernels build for `wasm32-unknown-unknown` and `wasm32-wasip2` in CI.
3. **Every milestone runs on the web and natively**, from the same crates, under CI.
4. **The GPU does per-frame work; the CPU does per-change work.** "Objects touched on [S] per frame" and "commands recorded per camera-only frame" are CI counters from the first renderer commit.
5. **Design for the modern API; degrade explicitly.** Compat implements a declared subset from Modern's data and may not add requirements upstream. Gaps are warnings, never blank screens.
6. **Plain data crosses every boundary**; closures never do (#8533) [verified].
7. **No coordinate system in public types or ABIs**; a lint bans "mercator" in public items (Mercator leaks into stable ABIs today: `mln_projected_meters_*`, plugin `tile_matrix`) [verified].
8. **Host services are traits with conformance suites**, never link-time file selection (a Qt bidi stub reached an iOS build, ffi#788) [verified].
9. **No silent fallbacks**: structured warnings tied to `capabilities.json`; `strict` makes them errors (Native renders globe styles flat silently, `parser.cpp:100-112`) [verified].
10. **Correctness by construction**: SoA, arenas, generational indices, fuzzing; no runtime canaries (#4749: canaries cost 37% of app CPU) [verified].
11. **Spec-heavy subsystems before GPU breadth**: style, expressions, text and placement land before 3D and embedding modes.

---

## 3. Scope

| M0–M5: the 2D map on every target | Rendering lane (M6–M8), M9–M12 or later |
|---|---|
| Web (WebGPU; WebGL2 Compat), Windows, macOS, Linux, Android, iOS | OpenHarmony, tvOS, visionOS-specific hosts |
| background, fill (patterns), line (dash, gradient, pattern), circle, symbol, raster, heatmap, hillshade, color-relief, fill-extrusion (v8 flat shading) | non-Mercator `projection`, `terrain`, `sky` (M6–M7), lighting beyond v8 (M8) |
| 88 operators; feature and global state; `promoteId`/`generateId` | scene-document modules: CRS, lighting, 3D (M6–M8), a11y hints (M10) |
| MVT, MLT, GeoJSON (clustering, `updateData`), raster, raster-dem, PMTiles, MBTiles (native) | GeoArrow data layers (M11), COG, GeoParquet |
| full text stack and placement | curve text, variable-font axes |
| gestures, camera animations, queries, events, idle, view annotations | accessibility host mappings (M10; the snapshot exists from M2) |
| TS package, Rust crate, C ABI 0.x, Kotlin and Swift bindings, host helpers; surface and owned-texture targets | SharedDevice, HostDriven integrations, multi-view, host-texture import (M9); stable ABIs, plugin ABI (M12) |

After M12: point clouds, splats, ProjIR, expression-function and wasm plugins, a SharedArrayBuffer build, XR and game-engine packages. Never: first-party declarative UI SDKs, SQLite offline, routing/geocoding/editing, a JS VM.

---

## 4. Platform baselines and GPU tiers

### 4.1 Facts that drive the decisions

| Capability | State (source) | Consequence |
|---|---|---|
| WebGPU, Chromium | Mac/Windows/ChromeOS since 113; Android 12+ (ARM/Qualcomm/Intel) since 121; Imagination on Android 16+ since 139; Xclipse "TBD"; Linux Intel Gen12+ 144, NVIDIA/Wayland 147, others flagged; Windows ARM64 flagged (gpuweb wiki) [verified] | Modern on most desktops and current Android browsers |
| WebGPU, Safari / Firefox | Safari 26 on by default on macOS, iOS/iPadOS, visionOS; Firefox Windows 141, macOS 145 (26+)/147, Linux Nightly, Android flagged (same) [verified] | Compat on Firefox Linux/Android and iOS < 26 |
| WebGPU compatibility mode | Chrome 146; `maxStorageBuffersInVertexStage` defaults to 0, core 8 (WebGPU spec) [verified] | not a tier (D9) |
| WebViews | WKWebView (iOS 26) appears to follow the system default [verified-2nd]; Android WebView core WebGPU unconfirmed | runtime probe; Compat covers the rest |
| Optional WebGPU features | `subgroups` (Chrome 134), `shader-f16`, per-adapter texture compression; multi-draw-indirect experimental, not core; no binding arrays in core [verified-2nd] | M+ fast paths only |
| Browser shader compilation | WGSL translated at pipeline creation (Tint, WebKit, naga) then FXC/DXC or driver; tens to hundreds of ms per pipeline on Windows D3D12 [reviewer-reported] | pipeline count and first-frame order are design constraints |
| WebGL2 | universal; GL JS WebGL2-only since v6 [verified] | Compat floor on the web |
| Wasm | SIMD128, bulk memory, EH everywhere (Safari exnref 18.4); memory never shrinks; threads need SAB → COOP/COEP (Safari lacks `credentialless`, WebKit 230550) and nightly `build-std`; no JSPI (WebKit 283054) or Memory64 in Safari; wasm needs CSP `'wasm-unsafe-eval'` [verified] | SIMD128 required; share-nothing workers; wasm32; async-native core |
| OffscreenCanvas + WebGL2 in workers | Safari 17.2+ (compose#1169: 17.0.1 failed, 17.2 worked) [verified] | Compat worker rendering from 17.2 |
| `Intl.Segmenter` | grapheme/word/sentence only, no UAX #14 lines; per-browser dictionary data (ECMA-402) [verified-2nd] | ICU4X segmentation in wasm |
| Android Vulkan | ~87% of devices had Vulkan ≥1.1, 8.85% none; Android 16 launch chipsets must support Vulkan 1.4 features (developer.android.com; Android Authority) [verified-2nd] | the Android Modern API, on an allowlist |
| Android drivers | Native defaults Android to Vulkan; Exynos CPU encode 9–14× (#4184); MediaTek compiler crashes (#4464); Adreno `DEVICE_LOST` (#4442) [verified] | allowlist + quirks + crash sentinel |
| Android ANGLE | Android 15+ ships ANGLE as an optional GLES layer; AOSP has "ANGLE as system driver" machinery [verified-2nd] | GLES there runs on the same Vulkan driver: escapes compiler bugs, not crashes |
| Native desktop / Apple | Metal on every supported Apple device; D3D12/Vulkan on Windows 10+; Mesa Vulkan on Linux, lavapipe in CI [verified for wgpu] | Modern |
| wgpu | 29.0.0, 30.0.0, 30.0.1; Bevy 0.19/0.20 on wgpu 30; Slint ships `unstable-wgpu-29` and `-30`; `as_hal_mut` exposes only wgpu's own encoder [verified] | integrations independent of wgpu majors |
| Rust ecosystem | HarfRust 0.14.0 (rustybuzz archived), ICU4X 2.3.1, skrifa, `mlt-core` (MIT OR Apache-2.0), pmtiles-rs 0.24.1 [verified] | every subsystem covered |

**Reading** [inferred]: every target platform has a modern API on most current devices. The remainder (iOS < 26 browsers, Firefox Linux/Android, Android < 12 or flagged drivers, WebViews without WebGPU) is shrinking and identifiable. Shaping every algorithm for it would cost the main path storage buffers in the vertex stage, compute-fed indirect draws and a single submission model; serving it with a reduced tier costs one GL executor, CPU cull kernels and a texture-fetch shader variant.

### 4.2 Tiers and floors

| Tier | APIs | Requires | Provides |
|---|---|---|---|
| **M (Modern)** | browser WebGPU core; Vulkan 1.1+ (Android: allowlisted 12+ devices); Metal (iOS 16+, macOS 13+); D3D12 (Windows 10+) | storage buffers in vertex/fragment/compute (WebGPU core default 8 per stage); compute; `drawIndirect`; 128 MiB storage bindings and 256 MiB buffers (WebGPU core defaults [inferred from spec; re-check]); 256-layer texture arrays | everything in the build path; all §18 budgets |
| **M+** (optional) | per adapter, web and native | any of `multi_draw_indirect(_count)`, partially bound binding arrays, `subgroups`, `shader-f16`, `indirect-first-instance`, timestamp queries, compressed formats, native multiview | fewer padded invocations, cheaper compaction, larger atlases; shadows/OIT (declared M+-only) |
| **C (Compat)** | WebGL2; GLES 3.0+ below the allowlist or after a crash-sentinel trip; GL 3.3 (Linux without Vulkan, toolkit GL contexts) | WebGL2-class GL; `EXT_color_buffer_float` for heatmap | the declared subset of §8.7, no performance budgets |
| Unsupported | WebGL1, GLES 2; Safari < 16.4 (no SIMD); Android < API 24; iOS < 16 | — | a structured "unsupported device" error |

**Floors** [inferred]: Android API 24 (matches maplibre-compose's minSdk 24 [verified]), arm64 + x86_64, texture targets API 26 (`AHardwareBuffer`); iOS 16; macOS 13; Windows 10 21H2 x64/arm64; Linux glibc 2.31/musl. Browsers for Modern: Chrome/Edge 113+ desktop, 121+ Android; Safari 26; Firefox 141 (Windows)/147 (macOS). Compat: Chrome/Edge 110+, Firefox 128 ESR+, Safari 16.4+ (worker rendering 17.2+).

**Selection order:** Web WebGPU → WebGL2. Apple Metal. Android Vulkan (allowlisted) → GLES 3.0. Windows D3D12 → Vulkan → GL. Linux Vulkan → GL. Every choice and fallback is reported in `map.diagnostics` and to host telemetry hooks.

**CI tiers.** Tier 1 (every PR on software rasterizers, nightly on GPUs): Linux lavapipe (Modern) and llvmpipe (Compat), macOS Metal, Windows WARP, headless Chrome WebGPU/SwiftShader and WebGL2; Android and iOS builds plus device-farm smoke from M3. Tier 2 (nightly): arm64 Linux/Windows, Android emulator, an API 24–26 GLES device, Firefox and Safari, desktop GL. Tier 3 (weekly): `wasm32-wasip2` kernels, tvOS, FreeBSD, OHOS.

---

## 5. Architecture

### 5.1 Layering

```
        ┌──────────────────────── hosts (thin; low-level) ────────────────────────────────┐
        │ web TS host │ Android helpers │ Apple helpers │ desktop │ engines/toolkits │ servers │
        └──────┬──────────────┬───────────────────────────────────┬─────────────────────┘
               │ wasm-bindgen │ C ABI (generated)                  │ Rust API
        ┌──────┴──────────────┴───────────────────────────────────┴─────────────────────┐
        │ orb-api (Rust) → orb-capi (Diplomat bridge) → C, Kotlin/JNI, Swift; orb-web    │
        ├────────────────────────────────────────────────────────────────────────────────┤
        │ orb-runtime: Runtime / Map / Renderer / Target, commands, snapshots, events    │
        ├──────────────┬──────────────┬─────────────────┬────────────────────────────────┤
        │ scene side   │ job side     │ placement lane  │ host services (traits)         │
        │ style, scene │ decode,      │ cross-tile idx, │ Fetch, Storage, ImageDecode,   │
        │ doc, camera, │ layout, text,│ collision,      │ Fonts, Locale, Clock, Log, A11y│
        │ planner,graph│ symbols, 3D  │ fades           │                                │
        ├──────────────┴──────────────┴─────────────────┴────────────────────────────────┤
        │ orb-render: GPU-driven Modern path │ CPU-fed Compat path (declared subset)     │
        ├────────────────────────────────────────────────────────────────────────────────┤
        │ orb-hal (WebGPU-shaped) ─ orb-hal-wgpu (WebGPU, Vk, Mtl, D3D12) ─ orb-hal-gl    │
        └────────────────────────────────────────────────────────────────────────────────┘
```

Core owns style, expressions, sources, decode, layout, text, placement, camera, projections, gestures, the renderer, queries, state, memory policy, events, the accessibility snapshot and engine-level batteries. Hosts own views and surfaces, raw input capture, platform accessibility trees and service overrides. Gestures are in core because at least 7 implementations exist today with different feel (flutter#907), and Compose lost a 300 ms tap delay only once it owned gestures (compose#829) [verified]; hosts that own input drive the camera directly.

### 5.2 Crates

LOC are CI-enforced ceilings [inferred]; "From" is the milestone that introduces the crate; code reused or ported from other projects is listed in §19.

| Crate | Responsibility | LOC | From |
|---|---|---|---|
| `orb-schema` | style schema (JSON Schema + expression signatures) from a pinned `v8.json` and the scene-document schema | 2k + gen | M0 |
| `orb-v8` | v8 parse/validate with JSON-path diagnostics, legacy migration, lowering, deviation flags; scene-document merge | 6k | M0 |
| `orb-style` | Style IR, persistent `StyleDoc` with dirty bits, transactions, diff → patch, transitions, composition | 8k | M1 |
| `orb-expr` | SSA expression IR; VM; SIMD column evaluator; GPU bytecode; binding-mask classifier; cost model | 9k | M0 |
| `orb-geo` | f64 geodesy, ECEF/ENU, WGS84, RTC and Jacobian helpers | 3k | M0 |
| `orb-crs` | `Surface`, projection families + WGSL, descriptors, EPSG table, OGC TMS registry, DataToWorld planner, subdivision, tile cover | 9k | M0 |
| `orb-camera` | single-writer camera store, resolution truth, animations against `Surface`, multi-view, terrain constraints | 4k | M0 |
| `orb-interact` | recognizers, inertia, feel profiles, keyboard | 3k | M3 |
| `orb-source` | `TileKey {tms, z, col, row}`, tile state machine, priorities, caches keyed by data TMS, 3D traversal hooks | 5k | M0 |
| `orb-tile` | `TileFrame`; MVT, MLT, GeoJSON (geojson-vt 6 port), GeoArrow, overzoom, DEM, KTX2 | 7k | M0 |
| `orb-layout` | layout families, earcut, clip-at-layout, **chunking**, property tables, hit index, subdivision | 9k | M0 |
| `orb-text` | fonts, itemize/bidi/ICU4X segmentation, HarfRust `Shaper`, line breaking, glyph service, atlas | 10k | M2 |
| `orb-symbol` | SoA symbol layout, stable keys, collision, placement scopes, fades, sort-key permutations | 9k | M2 |
| `orb-3d` | glTF model layer, georeferenced placement, 3D Tiles 1.1 | 10k | M8 |
| `orb-scene` | GPU scene tables (§8.2), arenas, staging, upload scheduler, memory budget manager | 7k | M0 |
| `orb-render` | render-graph compiler; Modern GPU-driven path; Compat path; terrain, lighting, atmosphere; built-in layers as plugins | 15k + 4k | M0, M4 |
| `orb-hal` | WebGPU-shaped trait, caps (M/M+/C), allowlist, quirks, crash sentinel, web upload, external resources | 2.5k | M0 |
| `orb-hal-wgpu` | browser WebGPU, Vulkan, Metal, D3D12; raw-handle import | 3.5k | M0 |
| `orb-hal-gl` | `glow` executor for WebGL2, GLES 3.0, GL 3.3; GL state cache | 5k | M4 |
| `orb-shaders` | WGSL modules and composer, projection modules, interpreter, `override` tables, naga build step, reflection | 8k WGSL + 1.5k | M0 |
| `orb-jobs` | `Executor`, priorities, cancellation, web slicing, native pool, worker adaptor, inline | 3k | M0 |
| `orb-runtime` | Runtime/Map/Renderer/Target, commands, generations, snapshots, events, idle | 6k | M0 |
| `orb-a11y` | semantic snapshot; AccessKit adapter | 2k | M2, M10 |
| `orb-host` (+`-native`, `-web`) | service traits + conformance suites; ureq/image/fontdb natively; fetch/Cache API/`createImageBitmap` on the web | 7k | M0 |
| `orb-plugin` | per-runtime registries; native plugin ABI | 3k | M1, M12 |
| `orb-batteries` | location layer, annotation source, PMTiles region downloader, snapshotter | 3k | M5 |
| `orb-api`, `orb-capi`, `orb-web` | safe Rust embedding API (runtime, map, targets, frames, events); the annotated Diplomat bridge and its C ABI; wasm-bindgen surface, roles, payload table | 6k + gen | M0, M3 |
| `diplomat-swift-kotlin` | project-agnostic Swift and Kotlin/JNI emitters over Diplomat's HIR (decision 0018) | 4k | M3 |
| tools | `orb-fixtures` (`cargo xtask rt`), real-world suite, bench, `xtask`, `orb-viewer`, LSP | 8k | M0 |

Total ceiling ≈ 155k lines of Rust + ~8k WGSL at M12 [inferred]; for scale, Native's core is 166,355 lines plus 259,225 of platform code and GL JS is 70,918 [verified].

### 5.3 Boundary rules (CI-enforced)

- Below `orb-runtime`: no threads, no I/O, no GPU handles. Only `orb-hal-*` touches GPU APIs, only `orb-host-*` the OS or browser, only `orb-capi`/`orb-web` export symbols.
- Job crates are share-nothing and never depend on scene crates.
- No public item names a CRS.
- **Compat may not constrain Modern**: Compat consumes `orb-scene` data and layout output; nothing in the Modern path depends on Compat code or features; a layout change needed only by Compat is rejected.
- `unsafe` only in `orb-hal-*`, `orb-capi`, `orb-web`, `orb-jobs` and audited SIMD kernels; elsewhere `#![forbid(unsafe_code)]`.
- `cargo deny` licence allowlist (MIT, Apache-2.0, BSD, Zlib, ISC, Unicode-3.0 for ICU4X and UCD data, file-level MPL-2.0), crate-graph check, `cargo-semver-checks` on `orb-api`, public-item lint, per-module size reports.

### 5.4 Object model

Four nouns, refined from ffi#706 [verified]:
- **Runtime:** executor, host services, registries, shared caches (tiles by data TMS, glyph pages, pipelines); serves many maps.
- **Map:** style and scene document, camera, sources, feature/global state, queries, accessibility snapshot; **no GPU state**. Queries run on its published snapshot; Native puts them on `Renderer`, forcing the FFI into warm-up renders [verified].
- **Renderer:** GPU tables, arenas, atlases, pipelines, current placement. It **outlives targets** (backgrounding, surface swaps; ffi#706, compose#1377) and rebuilds from cached `TileFrame`s after device loss.
- **Target:** surface, owned texture, imported host texture with optional host depth (M9), or multi-view (M9): `CAMetalLayer`, `ANativeWindow`, HWND, Wayland, `<canvas>`/OffscreenCanvas, `MTLTexture`, `VkImage`, `ID3D12Resource`, GL texture, `GPUTexture`, `wgpu::Texture`.

### 5.5 Hooks built at M0 for later milestones

| Later | Hook at M0 [inferred cost: small] |
|---|---|
| Any CRS (M6) | `TileKey` carries a TMS; caches keyed by data TMS; `Surface::Plane{crs}`; the planner exists with `Affine`; tile rows map data → camera-relative world, never clip; "mercator" lint |
| Globe/ellipsoid (M7) | camera truth `{target: Geodetic, resolution_m_per_px, …}`; 128-byte non-affine row reserved; projection is a WGSL module behind `project_tile()`; f64 CPU, camera-relative f32 GPU |
| 3D (M8) | depth-capable passes; per-view frame block; registered layout families |
| Guest embedding (M9) | Renderer never owns presentation; HAL external-resource trait stubbed; `HostDriven` executor used in tests; no process-global state |
| Accessibility (M10) | placement emits the semantic snapshot from M2 |
| Data layers (M11) | source-agnostic chunk records and cull compute; columnar property tables |
| Plugins (M12) | built-in families register through the internal registry the plugin ABI will expose |

---

## 6. Data flows

Lanes: **[H]** host UI or browser main (with the ui role), **[S]** scene, **[J]** jobs, **[P]** placement, **[GPU]**.

### 6.1 Uncached vector tile to pixels (web, Modern, worker placement)

Lanes: MAIN [H + ui], RENDER-W [S], JOB-W k [J], PLACE-W [P] (dedicated when `hardwareConcurrency ≥ 4`), GPU.

1. **MAIN:** rAF → `orb-interact` → `orb-camera` (ui role, camera authority) → `CameraState v42` (~200 B) → RENDER-W.
2. **RENDER-W:** cover over the source TMS from cached volumes → tile K is P0 → job to JOB-W `hash(source, K) mod N`, token `(K, gen)`, 2–3-job pull lookahead.
3. **JOB-W k:** `fetch` or PMTiles range read → `mlt-core` decodes MLT into `TileFrame` columns (MVT transcoded in one pass); every stage is a resumable ≤4 ms slice.
4. **JOB-W k:** per layout family: filter as a column scan → paint into property tables → streams clipped at the tile edge (synthetic-edge flags) → **≤256-triangle chunks** with bounds and 24–32 B records.
5. **JOB-W k:** itemize (ICU4X, bidi) → shape (HarfRust or advance-only) from metrics the worker derives itself (decision 0009) → glyph keys in the tile result; the scene asks each raster unit's owner worker for SDF/MSDF bitmaps of keys it hasn't got.
6. **JOB-W k → RENDER-W:** streams, chunk records and tables **copied once** out of job wasm memory into transferable `ArrayBuffer`s; `TileFrame` stays in the job worker for relayout and queries.
7. **JOB-W k → PLACE-W:** collision SoA + stable keys.
8. **RENDER-W:** budgeted upload (≤1.5 ms at 60 Hz, P0 first) via `GPUQueue.writeBuffer` straight from the `ArrayBuffer` → arena ranges → tile row → chunk ids appended to each render node's **candidate list** → buffer released.
9. **PLACE-W:** cross-tile matching → placement on the latest camera → symbol-state diff (u32 per symbol) + sort-key permutations → RENDER-W.
10. **RENDER-W:** frame block + changed rows; diff applied (≤0.2 ms) → **one cull/compact dispatch** → replay the pass bundle (≈ one `drawIndirect` per node) → present.
11. **GPU:** Cull/Compact → Prepare → Layer2D → Symbols → ViewAnnotations.
12. **RENDER-W → MAIN:** frame ack `{generation, camera, viewport}` with picking's changes (tiles, drawn list, labels, layer table; decision 0024); accessibility delta.

Main-path messages: 4 (+2 per glyph batch), against 14–16 hops per uncached tile on Native Android [verified baseline]. Worker frames usually present one rAF later than main-thread rendering [reviewer-reported; measured in M3].

### 6.2 Steady camera-only frame (Modern, native Owned)

| Lane | ms [inferred] | Step |
|---|---|---|
| [H] | 0.00–0.05 | input → camera intent |
| [S] | 0.00–0.10 | integrate camera; cover check against cached volumes |
| [S] | 0.10–0.15 | frame block + changed rows (16–60 × 64 B ≈ 1–4 KB) |
| [S] | 0.15–0.25 | 1 dispatch + bundle replay (~60–90 `drawIndirect`) + submit |
| [GPU] | ~0.1 | cull/compact ~1–2k candidates (~50 KB of records) |
| [GPU] | ~0.4–5 | Layer2D in style order → symbols (single-pass halo) |
| [P] | parallel | placement every 100–300 ms or on settle, 1–10 ms |

Native on a Pixel 7 Pro with Americana: 3.39 ms async CPU encode; 9.8 ms CPU + 7.2 ms GPU serialized, with a per-drawable tweaker walk [verified]. The bundle is re-recorded only when nodes, pipelines or bind groups change; tile arrival changes GPU data, not commands.

### 6.3 Projection switch, Mercator → globe → Equal Earth (M7)

1. **[H]** `tx.setProjection({type: "equal-earth", lowZoom: "vertical-perspective", transition: [3, 5]})`.
2. **[S]** planner: WebMercatorQuad sources → `GpuPerVertex` (promoted for the blend); `WorldCRS84Quad` sources → `Geodetic`; graph re-planned; projection modules bound.
3. **[J]** re-subdivision only, cached per `(tile, family, curvature class)`; no network, no re-decode.
4. **[S]** affected rows switch 64 B affine → 128 B non-affine (RTC centre, Jacobian, Hessian); camera kept by target and ground resolution.
5. **[GPU]** per-vertex `mix(project_a, project_b, t)`; first correct frame ≤2 frames. When `t` settles on an affine member, plans and rows demote.

GL JS reloads every tile on a projection switch (`style.ts:1806-1811`); Native has no projections [verified].

### 6.4 Game engine / XR stereo frame (M9)

1. **Game thread:** predicted head pose → `orb_map_set_views(map, views[2], geo_frame)`; `orb_runtime_poll(rt, 2000 µs)` → union-frustum cover, jobs to the engine's `JobSink`, results applied within budget.
2. **Placement lane:** `world-3d` scope; screen collision as the union across eyes; anchors occluded by a CPU DEM ray walk.
3. **Render thread:** `orb_renderer_render(r, &frame)`: Orbis records **its own** command buffers on the shared device (imported from raw handles into its own wgpu version), culls per view, renders into host colour and against host depth in the host's convention, and submits before the host's consuming pass or signals a semaphore/fence.
4. **GPU:** engine geometry and map layers occlude each other.

---

## 7. Execution model

### 7.1 Roles and modes

| Role | Owned (apps, servers) | HostDriven (engines, XR; M9) | Web (default) | Inline |
|---|---|---|---|---|
| [H] | UI thread; reads snapshots | game thread | main: TS host, DOM, input, **ui role** of `core.wasm` (camera authority, gestures, sync project/unproject, hit index, a11y mirror) | caller |
| [S] | one core thread | `runtime.poll(budget_us)` | **render worker** with the OffscreenCanvas, or main with `thread: "main"` | caller |
| Target | = [S] or surface owner | engine render thread | = [S] | caller |
| [J] | pool `min(cores − 1, 6)` | host `JobSink` or core pool | N workers `clamp(hardwareConcurrency − 2, 1, 4)`, ≤4 ms slices | inline |
| [P] | pinned lane | pinned job | dedicated worker if ≥4 cores, else a short-slice job worker | inline |

`thread: "main"` is automatic when OffscreenCanvas transfer or WebGPU in workers is missing, and used by hosts that supply their own `GPUDevice` or WebGL2 context (Compose web runs GL JS in Skia's context today) [verified need].

**Invariants** (ffi#760's model, "easiest of the four migrations" for a production Go tile renderer) [verified]: commands from any thread return completions with a **generation**; snapshot reads are synchronous and never touch mutable state, and can be fenced on a generation; no synchronous GPU waits (readback, `mapAsync`, adapter/device requests and pipeline creation are async, so JSPI is never needed); one versioned update per command group; no host pumping (ffi#352).

### 7.2 Where the web scene renders

**Default: render worker + ui role on main**, because GL JS's main thread is busy 2,089 ms during load against 609 ms in its worker, TinySDF alone 27.6% (#8426), and app frameworks compete for main [verified]. Nothing in Orbis's API forces the scene onto main.

Costs and answers:
- **Gesture latency** (+1 frame [reviewer-reported]): the ui role owns the camera, so DOM consumers see it immediately; only the canvas lags.
- **Overlay skew**: a transferred OffscreenCanvas cannot commit atomically with the DOM. Default overlays are **engine-drawn view annotations** (composited in-frame, collided with labels, depth-tested); DOM overlays are opt-in on frame acks, documented ±1 frame.
- **Terrain (M7)**: the ui role receives a compact elevation pyramid (min/max + 32×32 u16 coarse DEM per covered tile, ~2 KB each); sync `unproject` under terrain returns `{lngLat, approximate: true}`, `unprojectExact` is async.

**M3 rule:** measure input-to-photon latency (pan, pinch, rotate), overlay skew and main busy time on Chrome (Windows, Android), Safari (macOS, iOS) and Firefox (Windows). The worker default stands if its p95 penalty is ≤1 frame and main busy time is ≤25% of GL JS's; otherwise `thread: "main"` becomes the default. The API is identical either way. [measured at M3, decision 0021] In headless Chrome 156 on an Apple M6 the worker's penalty is one frame at p50 (pan +17.1 ms, pinch +16.5 at 60 Hz) but 1.1–1.75 frames at the pan p95; its main busy time is 19% of GL JS 6.16.0's there and 25% on a moto g 5G (2024); DOM annotations on frame acknowledgements skew by under a frame (p95 1.6–9.2 ms). By the rule, **`thread: "main"` is the default**; `"worker"` and `"auto"` stay. Windows, Safari, iOS and Firefox, and Android latency, are unmeasured. [built at M3, decision 0023] Both modes send tile jobs and labels to the job and placement workers; only where the map draws differs (main mode had run jobs on the page, 30–190 ms tasks per live tile).

**One binary, three roles:** `compileStreaming` overlapped with the style fetch; the compiled `Module` is posted to workers; main instantiates the ui role at no extra download. `wasm-split` only if M2 shows instantiation or memory cost on low-end phones.

### 7.3 Jobs

- **DAG per tile:** `Fetch → Decode → LayoutFamily[i] → TextPrep → SymbolLayout → Package → {Upload, PlacementIngest, QueryIndex}`; glyph raster, glTF decode and ingest are separate deduplicated classes.
- **Priorities:** P0 visible at ideal resolution; P1 fallback parents/children; P2 predicted along an active flyTo/easeTo (gl-js #116); P3 prefetch. Native round-robins tags with no priority; GL JS is FIFO [verified].
- **Tile publication:** retain loaded descendants before using an ancestor for missing regions; ready ideal tiles replace fallbacks. Centred zooms preserve the exact camera target so projection roundoff cannot change the ideal cover; clip rectangles exclude their maximum edges so adjacent regions cannot double-blend. Descendant search follows cached branches at any depth, including empty coverage, and clipping partitions remain exact (at most four rectangles per placement row, with extra rows when needed). In-flight entries survive cache eviction until their host jobs complete, preventing duplicate attempts from completing the wrong entry. Native and both web hosts schedule frames while the map is non-idle; every result also wakes the host. See [decision 0006](decisions/0006-tile-publication.md).
- **Cancellation:** `(tile key, generation)` at stage boundaries and every ~1k features natively; on the web (no SAB) at slice boundaries, plus dropping queued work through the lookahead.
- **Affinity:** `hash(source, tile) mod N` keeps `TileFrame`s for relayout and queries; GeoJSON/data-layer columns pinned per source, sharded when large; glyph raster units (a PBF range, a font, a cluster batch) at `hash(unit) mod N`, requested once by the scene, which owns the atlas (decision 0009). GeoJSON indexes currently live in each `TileWorker`, once per (style, source), with source-level URL fetch deduplication in `PoolJobs` and each browser job worker; the map owns tile selection, not the index. Browser job hosts use `TileWorker::has_data` to avoid fetching inline or already indexed source data, and `data_url` to deduplicate whole-source fetches. Source diagnostics cross workers in the packed result header. Each worker keeps its tiles' decoded `TileFrame`s per source (an LRU of 256) across style revisions, so relayout and column jobs for edits neither fetch nor decode ([decision 0015](decisions/0015-style-edits.md)).
- **Worker count:** GL JS cut to one worker (#2354) [verified], likely because of unbudgeted result intake on main [inferred]; Orbis budgets every intake; M3 re-runs the A/B.

### 7.4 Frame budgets on [S] (60 Hz; scaled by refresh)

| Stage | Budget | Overflow |
|---|---|---|
| Camera, rows, frame block | 0.1 ms affine; 0.2 ms globe (~300 non-affine rows) | — |
| Cover check | 0.1 ms | re-cover over 2 frames |
| Apply job results, candidate lists | 0.3 ms | defer |
| Uploads | 1.5 ms (0.8 at 120 Hz) | P0 first; parents keep drawing |
| Placement diff | 0.2 ms | next frame |
| Pipeline polling | 0.1 ms | layer waits and fades in; never a wrong variant |
| Compat CPU cull | 0.3 ms | tile-level culling |
| Offscreen re-renders (drapes, DEM) | 2 tiles/frame | nearest first (generalizes GL JS #8368) |
| Encode/replay + submit | Modern ≤0.1 ms; Compat remainder | — |

These answer Native's p1 encode at 15–27 ms and GL JS's 102 ms worst frame [verified].

[built at M3] Arrived tiles apply within the 1.5 ms upload budget, at least
one a frame, the rest in order on the next frames (edits' results apply
whole). A tile makes room for all its meshes before writing any: the scene
checks that its arena ranges and chunk slots fit, evicting unused tiles
until they do, where a tile that ran out part way used to write itself again
for every tile it evicted (about 250 rewrites in a 6 s rotation at Seattle,
40–100 MiB in one frame).

### 7.5 Invalidation, idle, pacing

The scene store is persistent and versioned; dirty sets derive from version deltas, including whether placement must rerun (generalizing GL JS #8208). `idle` = no P0/P1 jobs, no transitions, placement settled, every resource settled **or failed** (the GL JS #8716 class) [verified]. Idle means zero frames; timers run only with symbol layers (ffi#735); one 30 fps icon must not cause 88 renders/s (#8208) [verified]. Pacing: Choreographer + `Surface.setFrameRate` on Android, never `Thread.sleep` (#4617); `CADisplayLink` ranges on Apple; rAF only while dirty on the web; host-driven in engines. Opt-in thermal adaptation: 0.75× render scale and 30 Hz under `thermalState`/`getThermalHeadroom` pressure.

---

## 8. GPU

### 8.1 HAL and implementations

`orb-hal` is a small WebGPU-shaped trait (~2.5k lines): buffers, textures, samplers, bind groups, render and compute pipelines (lazy, async, with `override` constants), passes, bundles, indirect draws, timestamps, async readback, multi-view targets. Capabilities are detected at runtime into `Caps {tier: M | C, fast_paths, limits}`; no compile-time backends. Two extensions: a **web upload** path (`upload_from_js`: `GPUQueue.writeBuffer` on the underlying `GPUBuffer` via web-sys interop with wgpu, or `bufferSubData` on Compat, so payloads never pass through wasm memory; proven in M0) and **external resources** (M9: import device, queue, colour and depth from raw Vulkan/Metal/D3D12 handles; semaphore/fence handoff).

| Implementation | Covers | Notes |
|---|---|---|
| `orb-hal-wgpu` | browser WebGPU; Vulkan, Metal, D3D12 | backends chosen at runtime, feature-gated per platform (iOS Metal only; Android Vulkan); on the web only wgpu's WebGPU backend, no naga, no GL. maplibre-rs's WebGPU-only build is 401 KB br with wgpu inside [verified] |
| `orb-hal-gl` | WebGL2, GLES 3.0+, GL 3.3 | one hand-written `glow` executor over offline GLSL ES 3.00, GL state cache after GL JS (BSD-3); Compat only; lazy on the web |

Contingencies: a thin web-sys WebGPU HAL if wgpu's browser overhead breaks the M0 size gate; a thin Metal backend if wgpu's iOS size or performance misses M3. Rejected: wgpu's GL backend (+1.7 MB raw / +610 KB gz in the browser [verified]; its least mature backend [reviewer-reported]); TS command-stream executors; Dawn ("tried Dawn, gave 5 MB extra size", #317).

### 8.2 The GPU scene (Modern)

Everything a frame needs lives in persistent buffers; a camera-only frame changes only the frame block, changed tile rows and diffs.

| Table | Contents | Written |
|---|---|---|
| Frame block (uniform) | per view: camera-relative view-projection, camera ECEF as hi/lo f32, time, `fade_t`, viewport | every frame |
| Tile table | **affine row, 88 B**: 2×3 data → camera-relative-world affine, world units per pixel, elevation ref (DEM tile and range), packed clip rect, fade, flags, extent/scale, and the tile's warp patches (M6 planar `GpuPerVertex`: biquadratic patches in the property arena, written once per tile; decision 0020). **Non-affine row, 128 B (M7)**: `C − camera`, Jacobian 3×2, Hessian terms, clip rect, fade, flags, projection params. A flat view writes 1–4 KB, a pitched globe 19–38 KB | changed rows on camera change |
| Chunk table | per chunk (≤256 triangles, ≤256 instances or one line run): `{arena_page, first_index, index_count, tile_slot, bbox_tile[4], feature_slot_base, flags, lod}`, 24–32 B | tile arrival/eviction |
| Node candidate lists | per render node: chunk ids of resident tiles | tile arrival; style edits |
| Visible lists + indirect args | per node, GPU-written | every frame, by compute |
| Arenas | per (source, family, stream kind): vertex and u32 index pages of 16–64 MB, segregated-fit suballocator | upload |
| Property tables | per (tile, layer) columns by `feature_slot`; f16 with `shader-f16` | upload; paint edits |
| Feature/global state | 24 B rows by `state_index`; per-tile slot → state remap | state writes |
| Symbol tables | instances, glyph quads, one u32 state word per symbol `{opacity, target, anchor, orientation, flags}`, glyph-rect table | upload; placement diffs |
| Atlases | texture arrays: glyph pages (r8 SDF, rgba8 MSDF/colour), icons/patterns, raster tiles, DEM | upload |

**"Bindless enough" on core WebGPU:** shaders index `(layer, rect)` through tables, so one bind group per pass serves every tile of a family and layout never waits for atlas allocation. Atlases are shared per Runtime, keeping Native's #3198 win (7–8 MB → 1.88 MB on an iPhone 16 Pro Max) [verified]. **M+:** partially bound binding arrays (Vulkan, Metal, D3D12 via wgpu) for mixed compressed raster formats and more pages than one array holds.

The initial icon/pattern atlas has eight 1024² RGBA8 pages (32 MiB), a shelf
packer, one-texel padding, and 8192 stable 32-byte image rows. Jobs return names
and property-word offsets; scene upload resolves ids even before pixels arrive.
Style namespaces retire independently and recycle rows and empty pages. Host
services fetch sprite JSON/PNG (including namespaced sheets and `@2x`) and
decode straight RGBA8; scene upload premultiplies. See
[decision 0008](decisions/0008-image-atlas.md).

### 8.3 GPU-driven submission

Per Modern frame:
1. **Cull + compact, one dispatch.** One thread per candidate `(node, chunk)`: frustum, horizon and clip-rect test of `bbox_tile` through the tile row, fade, LOD; survivors append to the node's visible list (workgroup prefix sum; subgroup ballot where available) and write its indirect args. Symbol nodes compact by state word, so hidden symbols cost nothing downstream (#4628 halved symbol time just by skipping hidden drawables) [verified].
2. **Draw** one indirect draw per (node, arena page). Meshes (fill, fill outline, line, fill-extrusion) draw **compacted indices** with `drawIndexedIndirect`: each candidate has a fixed slot in one index buffer, which the cull dispatch rewrites only when its chunk enters or leaves the view (the chunk's indices tagged with their run, a mesh in one placement, or hidden indices), so the post-transform cache applies, and a camera-only frame that changes no chunk's visibility writes no indices ([decision 0002](decisions/0002-vertex-pulling.md)). Instanced quads (points, circles, heatmaps, icons, glyphs) and image tiles use **vertex pulling**: `drawIndirect` with `instanceCount` = visible chunks (`vertexCount = 768`, degenerate past a chunk's last index) or their quads (`vertexCount = 6`); the vertex shader maps instance → chunk → vertex. No multi-draw or `indirect-first-instance` needed (core WebGPU requires `firstInstance = 0` in indirect draws without that feature [inferred from spec; re-check]).
3. **Replay.** The pass's ~60–90 commands are identical across camera-only frames, so they are recorded once as a render bundle; tile arrival changes buffers, not commands.

**Paths.** Modern: 1 dispatch + 1 bundle per frame and ~60–90 GPU draws for Liberty at Tokyo z15, against GL JS's 449.6 state-changing draws (#8474) [verified]. M+: compute writes per-chunk `DrawIndexedIndirect` args for one `multi_draw_indexed_indirect_count` per (node, page), with no padding and fixed-function index fetch. Compat: CPU SIMD cull over the same chunk records (~0.1 ms per 2k chunks [inferred], ≤0.3 ms budget), then `multiDrawElementsWEBGL` (~60–90 draws) or per-(node, tile) `drawElements` (≈ tiles × layers, uniform changes only); arenas bind as vertex/index buffers, tables are read with `texelFetch`, hidden symbols early-out in the vertex shader (that cut Native's SDF pass from 60% to 13% of frame, #3576) [verified].

**Costs, measured from M0** on Adreno, Mali, PowerVR and Apple GPUs: padded invocations (20–30% for Liberty in Seattle [measured, Apple M6]), and lost post-transform reuse (fill vertices are reused ~1.5–2× [inferred], so vertex work can roughly double). With Liberty's labels over the Tokyo extract at 2048×1536, vertex pulling's frame took 1.03–1.37× the GPU time of per-tile indexed draws on an Apple M6 (at most +0.14 ms) and up to 1.51× on an Adreno 619 (+12.6 ms at z12), so by decision 0002's rule the mesh families moved to compacted indices: 1.04–1.48× faster than pulling, at most 1.02× per-tile draws, with vertex pulling's flat GPU-driven CPU frame ([decision 0002](decisions/0002-vertex-pulling.md)) [measured]. Under a panning camera, chunks enter or leave the view in 36–90% of frames at little extra cull cost; a tile change rewrites every slot (up to 8.5 ms of cull pass on the Adreno at z12). Mali and PowerVR are unmeasured (#16); M+ multi-draw remains an option for them. M1 measured segment-instanced line extrusion in the vertex shader: it matches CPU extrusion's 113/114 line fixtures on native and Chrome and roughly halves occupied geometry bytes, but its line-only z12 GPU time is 1.80–1.98× CPU extrusion natively and 1.72–1.87× in Chrome on Apple M6. The experiment is shelved and its code removed; CPU extrusion remains the default for all four paint variants ([decision 0013](decisions/0013-segmented-lines.md)). Width edits already avoid relayout on the CPU path too.

The M0 line port follows the fixture mirror's GL JS commit
`2b68611a4e2d4f7f076f6c6ebf758e2d53f91471`: caps, joins and both limits
evaluate per feature at tile zoom. Each vertex uses 20 bytes: packed i16
position, four signed bytes for extrusion and offset normals (scale 63), flags
and quantized line distance, feature slot, and f32 local line progress. Offset normals share a bounded join miter and ease toward
segment normals near hairpins. Solid lines extrude in tile units using
`world_per_px / tile.scale`, matching upstream's continuous `transform.zoom`
ratio; upstream's integer `tileZoom` ratio applies to pattern/dash atlas scaling.

Patterns use that integer-zoom scale, floor line width, tile-relative split
pixel offsets, and the pinned version's 300 ms two-image zoom crossfade. Fill,
fill-outline and line nodes sample the shared image array; every background
layer uses persistent fill quads in stack order, with no clear-colour shortcut.
Backgrounds draw on every covered tile, as GL JS's do, so nothing is drawn
beyond the world's edge at the poles; patterned backgrounds use tiled
sampling.
The tile cover follows GL JS's constant zoom through the FOV-dependent pitch
threshold and its default variable zoom above it. Vector overzoom slices the
ancestor before layout, preserving the reference's child-relative line phase.

Dashes use a Runtime-shared 256 × 512 R8 SDF atlas (128 KiB), with typed
job keys resolved through the shared image table. Constant and data-driven
dash arrays use the pinned floor-width scaling and zoom crossfade. GeoJSON
`lineMetrics` provides clip ranges for gradient ramps, evaluated once per
clipped line in jobs. RGBA ramps share the image pages; step ramps use nearest
sampling and 256–16384 samples, folded into 256-wide rectangles. Resident tile
leases return ramp rows and rectangles on eviction. Full-precision local
progress supplements the reference's packed scaled distance; camera-only
frames retain persistent geometry, textures and bundles. See
[decision 0010](decisions/0010-line-textures.md).

**Batching** across tiles in style order makes commands scale with layers (iOS drops everything above 482 symbol layers today, #4410); `symbol-sort-key` becomes instance order via permuted ranges from placement (gl-js #2478) [verified]. **No per-tile stencil**, because `GPURenderBundleEncoder` cannot set stencil references, viewport, scissor or blend constants [reviewer-reported]: fills and lines are clipped at layout with synthetic-edge flags (outlines, AA, caps and joins skip them); points draw only from their owner tile; loading fallback parents use ≤4 clip rects per placement row and a fragment test, splitting larger partitions across disjoint rows. Today one stencil quad is drawn per tile per layer group, and hidden layers drew 19–40 extra (#8369) [verified]. **Depth** only for real 3D (terrain, extrusions, models, globe occlusion), reversed-Z on Modern, the host's convention when embedded; this avoids GL JS's `glDepthRange` slicing and globe `gl_Position.z` hacks [verified]. **AA:** analytic coverage for lines, fills and SDF text; 4× MSAA with memoryless attachments for 3D on TBDR GPUs.

The circle and heatmap point path uses an 8-byte record (packed i16 anchor,
feature slot) and one u32 index per point. Chunks hold at most 256 points;
the cull dispatch emits six vertices and `256 × visibleChunks` instances,
with padding rejected before paint evaluation. Source buffer points outside
`[0, extent)` are omitted. Placement rectangles test the unshifted anchor,
using half-open bounds, and culling widens the owned anchors before testing
the view, so footprints cross tile and fallback boundaries without duplication.
Circle sort-key runs merge across tiles in the candidate list on scene changes.
The seven circle paint columns use the same two-zoom composite rows as lines;
seven 32-byte curves bring its uniform to 304 bytes (512-byte node slots).

Flat v8 extrusions now use four-word roof/wall vertices, per-feature color,
height/base and deferred pattern ids. Original wall segments clip independently,
so tile clipping introduces no synthetic walls. CPU cover and the single GPU
cull test elevated volumes; camera-only frames still replay bundles with only
frame/placement writes. The v8 vertical scale retains GL JS's mean Earth radius
without changing ground coordinates. See [decision 0011](decisions/0011-fill-extrusion.md).

### 8.4 Shaders

- **WGSL only**, as modules: `project` (per family), `fetch` (storage pull | attribute + `texelFetch`), `prop`, `expr` (GPU interpreter), `sdf`, `pack`, `lighting`, `atmosphere`, `cull`. A build-time **composer** (naga_oil-style imports and flags; WESL evaluated in M0) [inferred] lets layer shaders be written once against `fetch`; Compat gets the attribute/texture implementation without constraining Modern.
- **Build:** naga validates every composed module, emits GLSL ES 3.00 for Compat, and reflection emits `#[repr(C)]` Rust structs with offset checks, TS layouts and a plugin manifest. Today each UBO layout is written at least five times (49 C++ structs; `GlobalPaintParamsUBO` in 8 WGSL headers) [verified].
- **Specialization:** each data-driven property's binding mode is an `override` constant; pipelines are keyed by (layer type, **binding-mask class** ∈ {constant/curve, table, bytecode, mixed}, projection class, feature flags, pass); translators drop dead branches. **Only the bytecode class compiles the GPU interpreter** (≤32 ops, ≤16 stops), after Native's `GPUExpression` (#2202) [verified]; a new expression in an existing class is a buffer write. A typical basemap needs 25–40 pipelines [inferred], CI-counted; [measured at M2] most pipelines are created with the renderer, so Liberty (Tokyo extract and live planet, flat and pitched with extrusions) creates 16–17 and Americana (356 layers, about 400 render nodes, hillshade and extrusions) 17, counted by `WgpuGpu::pipelines_created` in `orb-demo --snapshot`; circle alone has up to 2^7 variants in Native [verified]. M1 measures interpreter register pressure and occupancy on Mali-G52/G57, PowerVR and Adreno 6xx to set the cost-model threshold. [measured at M1, decision 0017] The interpreter (`expr.wgsl`) is composed only into the bytecode class of the fill, fill outline, line and circle shaders (pipelines created for the first node that needs them); per node a uniform mask selects interpreted slots and a property-arena table holds the programs, so a program edit that keeps its input columns is a buffer write. One call site per pipeline: seven inlined copies crashed the Adreno 619's Vulkan driver. Cost with every fill and line colour interpreted: Apple M6 +0.023 ms for a `match` on `class` and about 0.01 ms per instruction; Adreno 619 2.6× the rows frame for that `match` and about 5.5 ms per instruction. The threshold is therefore 0 (`CostModel::ROWS`): rows wherever rows can hold a value, and the interpreter for values that read feature state and feature data. Mali and PowerVR are unmeasured: no device ([#16](https://github.com/sargunv/orbis/issues/16)); cutting the Adreno cost is [#15](https://github.com/sargunv/orbis/issues/15).
- **Runtime translation:** natively wgpu runs naga, then drivers compile (passthrough SPIR-V/metallib once stable); browsers translate at pipeline creation, slowly on Windows D3D12 [reviewer-reported]. **First frame:** the cull pipeline first, then layers in order of first-visible coverage; a layer without its pipeline is not drawn and fades in later; warm starts use browser blob caches and native pipeline caches. The load gate is set from M3 measurements.
- **Projection modules** expose `project_tile`, `project_tile_3d`, `local_scale`, `extrusion_normal`, `horizon_clip`; layer shaders never branch on projection (24 of GL JS's 71 GLSL files contain `#ifdef GLOBE`) [verified].
- **Seeds:** Native's WGSL (4,567 lines, BSD-2); GL JS globe math incl. the Weierstrass form that fixed Mali precision (#7916); single-pass halo (#7436).

### 8.5 Render graph

Compiled per (style IR, projection plan, scene features, tier); empty passes culled, transient targets pooled.

| Node | Modern | Compat |
|---|---|---|
| `Upload` (budgeted, P0 first) | queue writes / staging ring | `bufferSubData` into idle ranges + data-texture mirrors |
| `Cull` (chunks, symbol compaction, line-label placement, fades) | compute | CPU SIMD; VS early-out; CPU line labels |
| `Prepare` (DEM borders backfilled from neighbours and hillshade slopes into texture arrays, on tile arrival; heatmap density) | compute for DEM; render-to-texture for heatmap | render-to-texture |
| `Terrain`/`Drape` (M7; fingerprint cache, GL JS #7549: hit rate 0.2% → 94.6% [verified]) | yes | no |
| `Layer2D` (per style layer and pass, batched across tiles) | indirect | direct |
| `Group` (blend modes, z-groups, layer opacity; gl-js #48, #2108) | yes | yes |
| `Opaque3D`/`Transparent3D` (instanced extrusions; glTF, 3D Tiles at M8) | yes | extrusions only |
| `Symbols` (single-pass halo; depth-tested with `symbol-occlusion`) | yes | screen-space only |
| `ViewAnnotations` (host images at anchors, collided with labels) | yes | yes |
| `Atmosphere` (M7; one sky/fog model on every surface; today fog exists only in GL JS's Mercator terrain shader, #4337 [verified]) | yes | no |
| `Shadow` (M8) | M+ | no |
| `Overlay`/`Custom` (debug, focus ring, plugin passes) | yes | yes |

Extrusion nodes resolve their nearest lit surfaces with reversed-Z depth to
per-layer RGBA8 targets, then composite once at layer opacity in style order.
They share depth across 3D layers; 2D passes have no depth. Production uses 4×
MSAA with transient/discarded multisample color (and depth for a single layer).
Hosts may disable 3D antialiasing; fixtures use one sample to match GL JS's
default map option. Known coplanar ownership and fixture gaps are recorded in
[decision 0011](decisions/0011-fill-extrusion.md) and #5.

Heatmap density is regenerated after culling, before `Layer2D`, into one
persistent `rgba16float` target per heatmap node, at one quarter of the
physical viewport dimensions on each axis (minimum 1 texel). Gaussian
kernels use additive ONE/ONE blending and GL JS's effective-zero cutoff
`1 / 255 / 16`. A 256-texel RGBA8 ramp evaluates `heatmap-color` over
`heatmap-density`; a fullscreen composite occupies that layer's position in
the main bundle. Both passes replay bundles on camera-only frames; only a
viewport resize replaces density targets and composite bindings. The HAL
requires renderable, filterable, blendable `rgba16float` (guaranteed by core
WebGPU; Compat requires `EXT_color_buffer_float`); no `float32-blendable`
feature is needed. An unavailable capability skips heatmaps with a diagnostic.

Circle/heatmap validation (macOS native and Chrome 156): circle
58 pass / 0 fail / 5 skip (four global-style-state fixtures; one tile-boundary
debug fixture), heatmap 17 / 0 / 0. Both families previously skipped entirely
(63 and 17). All 162 previous passes remain; the full mirror is now
237 pass / 4 fail / 181 skip on each platform, with all 237 native passes
pixel-identical in Chrome. `core.wasm` is 389.077 KiB Brotli against the
starting checkout's 379.992 KiB (+9.085 KiB, including the intervening
scene-capacity fix), below the 450 KiB gate. The original camera-only test
and a circle/heatmap camera-only test pass: one cull dispatch, no bundle
recording or paint writes, placement writes equal tiles drawn.

### 8.6 GPU work for symbols

| Work | Today [verified] | Modern | Compat |
|---|---|---|---|
| Hidden symbols | drawn (#3223) | compacted in `Cull` | VS early-out |
| Line-label glyph placement | CPU every frame + buffer re-upload, both engines | compute: one workgroup per placed label walks its path and writes glyph transforms | CPU SIMD, visible labels, ≤0.3 ms |
| Anchors, fades | CPU dynamic-buffer rewrites | state word; VS computes the current value from `fade_t` | same, from a texture |
| Halo + fill | Native two passes; GL JS one (−30% GPU, #7436) | one pass | one pass |
| Collision | on the critical thread | **CPU, placement lane, deterministic** | same |

Modern-only research, adopted only if measured better: GPU-projected collision boxes read back asynchronously for pitched/globe placement; Hi-Z occlusion of `world-3d` anchors (M8); approximate GPU placement for data layers (M11, opt-in, never basemap labels).

### 8.7 The Compat tier

**Shares** everything above `orb-render`: style, expressions, `TileFrame`, layout, chunk records, text, placement, camera, planner, queries. Arenas bind as vertex/index buffers; tile rows, properties, feature and symbol state are mirrored into RGBA32F/RGBA32UI textures; layer shaders are the same WGSL composed with the attribute/texture `fetch`.

**Gives up** (declared in `capabilities.json`; each a warning, never a blank screen): GPU culling, compaction and indirect draws (CPU cull, direct draws, no Modern performance budgets); GPU line-label placement (CPU, budgeted); terrain, atmosphere and lighting beyond v8 flat extrusion shading; ellipsoid/ECEF scenes, glTF, 3D Tiles, shadows, OIT, `world-3d` labels; data layers above ~250k features; SharedDevice and multi-view. It **keeps** v8 globe (sphere) and planar any-CRS, because `Affine`, `CpuResample`, `RasterWarp` and `GpuPerVertex` are vertex-shader work. Heatmap needs `EXT_color_buffer_float`.

**Rules:** LOC ceilings (~4k in `orb-render`, 5k in `orb-hal-gl`); a declared fixture subset by directory; new Modern features need no Compat version unless they are in the v8 2D subset; Compat never changes Modern data layouts.

**Why keep it:** without it, iOS < 26 browsers, Firefox on Linux and Android, Android devices outside the allowlist and WebViews without WebGPU would get no map (§4.1). Its cost is mostly testing, which the shared fixtures and runners absorb.

### 8.8 Fallback, device loss, quirks

1. **Android allowlist:** Vulkan is Modern only on Android 12+ with known-good GPUs and drivers, seeded from Chrome's WebGPU Android rollout (Qualcomm, ARM, Intel on 12+; Imagination on 16+; Xclipse pending) and Native's incidents; it grows only from farm and field telemetry.
2. **Quirks table** (ANGLE `FeaturesGL` style), seeded from Goldfish, Adreno `DEVICE_LOST` (#4442), MediaTek and Huawei compiler crashes (#4464, #4215), Exynos encode (#4184), PowerVR banding (#4506), Mali precision (#7916), SwiftShader UBO order (FFI patch 0011), CGL `bufferSubData` stalls (#8468), ANGLE-Metal `flat` varyings (#8364) [verified]. First entry (M3): native devices on Qualcomm Vulkan skip wgpu's indirect-call validation, after which the driver draws nothing from `drawIndexedIndirect` (Adreno 619; gfx-rs/wgpu#8801 on 650/740). [measured] Two more from the field ([#23](https://github.com/sargunv/orbis/issues/23)), worked around for every device: an Adreno 830 (Snapdragon 8 Elite, Chrome 156, Vulkan) miscompiles the cull shader when a function that indexes function-scope arrays in loops is inlined twice, so the dispatch writes no draws or loses the device; the cull shader keeps no function-scope arrays (an `orb-shaders` test checks it; a Dawn report with a standalone repro is still to file). Browsers without `GPUTextureUsage.TRANSIENT_ATTACHMENT` (Samsung Internet 30, Chromium 143) reject textures that use it, so `orb-hal-wgpu` drops that hint there.
3. **Crash sentinel:** a marker written before first pipeline creation and cleared after; if found at next launch, the device drops to Compat. #4464 and #4215 crash inside driver compilers, so offline compilation cannot avoid them [verified]. Pipelines are created on a watchdog thread.
4. **Device loss:** streams are freed after upload; the Renderer rebuilds by re-running layout from cached `TileFrame`s (or refetching from the HTTP cache), visible tiles first. On the web it may switch WebGPU → WebGL2 (Figma's design) [verified precedent] and emits a capability-change event. Targets [inferred]: ≤1 s to a correct visible frame on a mid-range phone; ≤2 s for a WebGPU → WebGL2 switch including the lazy Compat module.

### 8.9 Memory, wgpu policy, observability

- **Memory:** staging via `queue.write_buffer` or async-mapped rings; per-frame uniform rings (replacing Metal's per-update `MTLBuffer`, `mtl/buffer_resource.cpp:98-128`) [verified]; KTX2/Basis rasters transcoded in jobs to ASTC/ETC2/BC (DEM stays R16/RG8). A budget manager covers every arena, table, atlas, target pool, `TileFrame` cache and column set with cross-cache LRU (mobile 192 MB GPU / 128 MB CPU, desktop 512 MB GPU [inferred]), bounding the #4684 (hillshade +6 MB/s down to 5 fps) and #8052 leak classes [verified]. Since wasm memory never shrinks, web workers have high-water budgets (mobile 64 MB per job worker, 96 MB render worker; 2× desktop) and are recycled when over; large ingests use disposable workers.
- **wgpu:** each major tracked within ~8 weeks on a farm-gated branch (majors every 3–4 months [verified]); engines integrate over raw handles; `wgpu::Device` sharing for N and N−1 through `orb-wgpu-N` adapters (Slint's pattern); wgpu types never cross the C ABI (maplibre-rs's churn, #311, came from a forked Bevy render graph [verified]).
- **Observability:** CPU timers and GPU timestamps; the §16 counters plus per-worker wasm memory and pipelines created cold/warm; a stats API and HUD; debug labels for RenderDoc, Xcode, PIX and Chrome captures.

---

## 9. Scene and CRS model

### 9.1 One scene, many appearances

A scene is a **Surface**, optional **Terrain**, an ordered layer list, **lights**, an **atmosphere**, and a camera or external views.

| Mode | Surface | Milestone |
|---|---|---|
| Flat 2D | `Plane{crs}`: Web Mercator, any projected CRS, `simple` for games/pixel CRSs (#1228, #3890) | M0 (Mercator), M6 |
| 2.5D | pitched plane; extrusions, terrain, models with real depth | M1, M7, M8 |
| Globe | `Sphere` (v8 `globe`) or `Ellipsoid(WGS84)` (scene document); horizon culling; atmosphere | M7 |
| Local 3D | `Ellipsoid` + ENU `GeoFrame`; glTF and 3D Tiles in metres; engine-local origins | M8, M9 |
| Adaptive | blend of two projections over a zoom range, with plan promotion | M7 |

Aviation, drones (#3161), glTF georeferencing, 3D Tiles and XR assume WGS84; Native's globe proposal left the ellipsoid out (#4525) [verified]. `Surface` operations (`intersect_ray`, `tangent_frame`, `geodesic_interpolate`, `horizon`) are written once, so the ellipsoid is cheap; v8 `globe` compiles to `Sphere` to match GL JS goldens.

A `Projection` is plain data plus f64 `forward`/`inverse`, `affinity(tms)` (affine or a curvature class), `world_bounds` (wrap periods, singularities), `tile_volume`, `jacobian_hessian` for the precision series, and a WGSL module implementing `project_tile()`.

**Spaces:** data (tile-local integers in the source TMS, or local metres for georeferenced 3D), world (ECEF for sphere/ellipsoid, projected metres for planes; f64 on CPU, camera-relative f32 on GPU), screen (per view). Tile rows map data → camera-relative world; the frame block maps world → clip. This is harp.gl's `TilingScheme × Projection` split [verified].

### 9.2 TileMatrixSets and the planner (M6–M7)

- `"tileMatrixSet": "<OGC id>" | {inline 2D-TMS}` per source in the scene document; default `WebMercatorQuad`. The parser honours the registry traps [verified]: `WorldCRS84Quad` is 2×1 at z0; `EuropeanETRS89_LAEAQuad` lists northing first; `GNOSISGlobalGrid` coalesces polar rows; non-standard `cornerOfOrigin`. Quad and variable-width TMSs cover LINZ, PDOK, NASA GIBS (EPSG:3413/3031) and UTM. URL tokens: `{z}{x}{y}`, `{tileMatrix}{tileRow}{tileCol}`, `{bbox}` in source units (as #8723 adds). Caches are keyed by data TMS. Martin serves named TMS grids (martin#3280) [verified].

| Plan | When | Cost | Precedent |
|---|---|---|---|
| `Affine` (M0) | data CRS = view plane CRS | one affine row per tile | GL JS #8287/#8723, "no new GLSL" [verified] |
| `CpuResample` (M6) | no GPU kernel, exactness, or server output in a foreign CRS | reprojection once per tile in jobs | Mapbox flat projections [verified] |
| `RasterWarp` (M6) | raster/DEM in a foreign CRS | error-bounded mesh per tile, sampled on the GPU | OpenLayers `Triangulation.js` [verified] |
| `GpuPerVertex` (M6 planar, M7 surfaces) | non-affine with a GPU kernel | curvature-bounded subdivision in jobs + WGSL | GL JS globe |
| `Geodetic` (M7) | geodetic TMSs on sphere/ellipsoid | lon/lat → ECEF per vertex; **real poles** | — |

The planner picks the cheapest plan within the projection's declared pixel error and reports it in diagnostics. Subdivision is cached per `(tile, family, curvature class)`; Mercator sources on a sphere get explicit caps from `world_bounds().singularities` instead of ±32767 sentinel vertices [verified hack]. GPU subdivision in compute is a later option.

[implemented at M6, decision 0020] Tile matrix sets are OGC quadtrees (a first matrix of any size and zoom; non-quadtree sets are #29); a scene document v0 sets the surface's projection and each source's set. `Affine`, `RasterWarp` (a grid refined to a quarter pixel at 512 px per tile, tile edges at fixed steps so neighbours share vertices) and planar `GpuPerVertex` are built; `CpuResample` is not yet, since `GpuPerVertex` covers every planar pair and no case needs exact output in a foreign CRS. Planar `GpuPerVertex` evaluates 1–256 biquadratic patches per tile through f64 positions at their 3×3 nodes (the precision series in Lagrange form, so neighbouring patches and tiles meet exactly), with geometry cut along a grid in jobs; tiles lay out at their style zoom. Measured: the 39 descriptors agree with PROJ 9.8.1 within 0.1 mm forward and a nanodegree inverse; Mercator fixtures are unchanged.

### 9.3 Projections, precision, zoom

- **Descriptors** round-trip (`{"type":"equal-earth"}`, `{"type":"EPSG:2193"}`, LCC with parallels). Built-in families (f64 Rust + WGSL): web-mercator, equirectangular, equal-earth, natural-earth, TM/UTM, LCC, Albers, LAEA, polar stereographic/UPS, orthographic/vertical-perspective, identity; they cover #168, #5764, #1228, #3890, #8330, style-spec#1623 and #1865 [verified demand]. An EPSG table from PROJ data maps codes to families; unknown codes are errors, never a silent Mercator fallback; every family passes a CPU/GPU agreement test.
- **Adaptive** (style-spec#1866 shape `{type, lowZoom, transition: [z0, z1]}`): v8 `"globe"` is web-mercator with a vertical-perspective low zoom over [11, 12] on a `Sphere`, following GL JS (the docs say 10 → 12; maplibre-rs RFC #339) [verified drift]. Blends across incompatible surfaces are errors (today any array projection silently becomes a globe) [verified].
- **Precision:** affine rows store the f64 camera-relative affine as f32 (deck.gl's approach, exact at any zoom); non-affine rows carry f64-derived `C − camera`, `J`, `H`, and the shader evaluates `(C − camera) + J·d + ½dᵀHd`. [implemented at M6 for planes, decision 0020] The series is held as biquadratic patches through exact node positions rather than as derivatives at a centre: Taylor series of adjacent tiles disagreed on their shared edge and cracked. GL JS hands the globe to Mercator at z11–12 because f32 on a unit sphere gives "one value per 2.5 m" [verified]; Orbis keeps that handover for v8 globes until z18–z22 agreement passes on the farm.
- **Camera and zoom:** `CameraState {target: Geodetic, resolution_m_per_px, bearing, pitch, roll, fov, altitude?, padding}` as single-writer snapshots, pitch/roll unclamped (Native Android hard-codes three 60° clamps, #4530) [verified]. **Style `zoom` is a style-evaluation input derived per view, not a camera quantity** (decision 0022, which updates D15). On Web Mercator views and the v8 globe it is GL JS's convention (ground resolution at the target against the parallel; the globe keeps `1/cos(lat)` scaling, `globe_utils.ts:84`), so v8 stops, `minzoom`/`maxzoom` and tile selection match GL JS there. On other flat planes it is the zoom at which Web Mercator's ground resolution at the plane's reference latitude (its latitude of true scale, else its natural origin) equals the view's: one number for the whole view, constant as it pans, so zoom stops mean roughly the map scale they mean in Web Mercator near the plane's area of use. Camera zoom limits bound the same zoom; tiles lay out at the view's zoom where they span 512 px. Scene-document layers may use `["resolution"]`. Tile LOD per source comes from ground resolution and its TMS; cover is one DFS over cached volumes (GL JS globe cover 50.7 Hz vs Mercator 157 Hz, #8287) [verified]. The flat-surface implementation matches pinned GL JS tile sets using tile distance, pitch and FOV (66 generated cover goldens); frustum-box axis checks and elevated-content bounds keep high-pitch cover finite through the demo's 85° cap. Flat sky uses the style's gradient, or transparent host compositing when absent (decision 0014); fog/atmosphere remain with terrain/globe. External views (M9) use a union-frustum cover. `within`/`distance` stay planar by default (geodesic by flag), with one Earth radius per purpose (the engines disagree today) [verified].

### 9.4 Terrain, lighting, 3D content

- **Terrain (M7):** CDLOD quadtree over the surface TMS with skirts; min/max elevation pyramid for cover and camera constraints (also on the web ui role); a real depth texture (GL JS packs depth into RGBA at CSS resolution) [verified]; layers `drape` (cached RTT) or `project` (per-vertex elevation). **The renderer never writes the camera** (GL JS's `Map._render` does, causing jumps, #8549); elevation queries use a CPU DEM ray walk, never `readPixels` (108–117 ms stalls, #8387) [verified]. [implemented at M7 under Mercator, decision 0019] Terrain tiles are a quadtree of 512-pixel tiles with skirts, one grid vertex per DEM pixel; draped layers draw into 1024² slots of one mipmapped atlas through drape placements (ordinary tile placements targeting a slot), redrawn only when their content changes or, for zoom-only changes, once the camera stops; each drape stack's terrain composites in style order over the 3D depth extrusions share. Map placements name their DEM tile, so circles, heatmaps and extrusions stand on the ground. `CameraState::elevation` holds the target's height: at rest it stands on the terrain, keeping its elevation while the DEM under it loads; gestures end with the eye in place.
- **Flat v8 extrusion lighting (available):** style light anchor, spherical position, color and intensity; reference luminance/ambient terms and optional vertical wall gradient. Viewport anchoring rotates from frame bearing, so camera motion needs no paint uploads. All extrusion paint properties, including shared-atlas patterns, are rendered; 39 of 41 enabled pinned fixtures pass on native and Chrome, with 11 explicit skips. See [decision 0011](decisions/0011-fill-extrusion.md).
- **Lighting (M8):** ambient + directional + sun from time and location; PBR metallic-roughness for glTF; lit extrusions and terrain; `*-emissive-strength`; cascaded shadows on M+.
- **3D content (M8):** glTF `model` layer (instanced from points or one georeferenced placement; meshopt, quantization, KTX2; Draco native-only); `clip` volumes; 3D Tiles 1.1 (implicit tiling, `EXT_mesh_features`/`EXT_structural_metadata`, screen-space-error traversal in `orb-source`; vector content ingested into `TileFrame`). Placement scopes `viewport | tile | region | world-3d`; `world-3d` labels are ECEF billboards collided in screen space and occluded by depth.

---

## 10. Data formats

### 10.1 `TileFrame`

The single in-memory vector representation, after MLT's FeatureTable [verified]: per layer an id column, geometry types, `i16x2 | i32x2` vertices with part/ring offsets and optional triangles, typed property columns (`Bool`, `I64`, `F64`, dictionary `Str`, `Json`) with present bitmaps, and a string dictionary; one arena per tile.

| Input | Path |
|---|---|
| MLT | `mlt-core` streams into columns (columnar API, mlt-tile-spec #1901); no per-feature objects; avoids Native's MULTIPOINT → LineString bug (`vector_mlt_tile_data.cpp:~59`) [verified] |
| MVT | single-pass protobuf → columns |
| GeoJSON | Rust port of `@maplibre/geojson-vt` 6.3.0 → `TileFrame` directly (no GL JS `fromVectorTileJs` PBF re-encode); base tiling implemented, `updateData` and clustering remain planned [verified] |
| Overzoom | implemented: pure TileFrame child slicing with buffered clipping, retained ids/properties, and rounding on the normalized 8192 grid (GL JS writes a new MVT for queries) |
| PMTiles / MBTiles | PMTiles v3 reader in `orb-source` (pure header/directory/Hilbert logic, bounded directory cache); runtime satisfies host range reads. Both hosts require matching 206/Content-Range before reading, with a 64 MiB compressed/decompressed cap (a 200 answer to a Range request pulled an 83 GB archive into RAM, #4374) [verified]; gzip via `flate2`/`miniz_oxide`, MVT and MLT tile types; MBTiles via native storage |
| Rasters / DEM | host decode in jobs (`createImageBitmap` in job workers, the bitmap transferred and copied with `copyExternalImageToTexture`; `image` natively) → per-source texture-array pages, one tile per layer, drawn as one quad per tile and node; KTX2 transcode later (decision 0007) |
| Fonts, icons | Sprites implemented through host Fetch + ImageDecode, URL or namespaced array, retina selection, SDF, pixel ratio, stretch/content/text-fit metadata retained in shared image pages. M2: TTF/OTF/WOFF2 and unicode-range subsets; glyph PBF as a provider; SVG icons via usvg (discussion #175) |
| GeoArrow (M11), glTF / 3D Tiles (M8) | §10.3, §9.4 |

GeoJSON JSON parsing/conversion uses `orb_expr::json` in `orb-v8`, which passes typed features to `orb-tile` (the existing `orb-expr` → `orb-tile` dependency stays acyclic). GL JS source defaults are extent 8192, maxzoom 18, buffer 128 pixels (2048 tile units), tolerance 0.375 pixels (6 tile units), indexMaxZoom 5 and indexMaxPoints 100000. TileFrame rings have implicit closure; GeoJSON object/array property values become JSON strings with a warning, and null values are absent.

Raw decode is roughly a wash (MLT median-tile decode with gzip is 113–117% of MVT; earcut ~2% of GL JS worker time, #8425) [verified]; the wins are no per-feature allocation, filters and paint as column scans (a `["==", ["get","class"], "primary"]` filter becomes a dictionary-code compare), columns uploaded directly as GPU tables, and selective decode for queries. Pre-tessellation is opt-in per source (+22–36% size, +27–145% decode) [verified]. Render streams use tile-local integer positions, 32-bit indices (no 65k segmenting), `feature_slot` ranges and a per-tile hit index (8–16 B per feature); they are freed after upload.

### 10.2 Feature and global state

Per source, id → `state_index` (promoteId values, interned strings, the MLT id column for `generateId`); a global table with a per-style compiled row layout (≤32 booleans, ≤4 f32, string enums; typically 24 B); a per-tile slot → state remap only for sources whose style reads `feature-state`; arbitrary JSON state falls back to CPU re-evaluation of the affected ranges. A write is 24 bytes (target ≤0.05 ms) against 11.9 ms in GL JS after #7550 [verified]. Feature state also drives icon opacity, size scale and in-atlas `icon-image` swaps (gl-js #5542) [verified demand]; `promoteId`/`generateId` work everywhere (native#4737 open) [verified].

[measured at M1, decision 0015] Implemented per source and source layer, keyed by GL JS's `String(id)` (tile ids, vector `promoteId`, GeoJSON `generateId`/`promoteId`); every feature of a resident state-reading mesh is interned and counted by its meshes, and each mesh's slot → index remap lives in the property arena. The row holds the **evaluated** state-lookup values rather than raw state (there is no GPU evaluator yet, and state values are often colours): per node that reads state, a flag word and two words per property, at the integer camera zoom and the next. A write is one row: 0.003–0.010 ms natively; in Chrome the edit call is 0.002–0.003 ms and the frame's share within ±0.01 ms noise. Paint reading both state and feature data is interpreted on the GPU when its program compiles, reading raw state values the row adds after the lookups ([decision 0017](decisions/0017-gpu-bytecode.md)); otherwise it draws without state ([#6](https://github.com/sargunv/orbis/issues/6)). Global state has no table: it resolves at S2 (§11.2).

### 10.3 Data layers (M11)

A `columnar` source family: Arrow/GeoArrow batches (zero-copy natively with a release callback; on the web transferred to a disposable ingest worker and copied once per shard), packed geometry aligned to MLT streams (ffi#584); jobs build a Hilbert sort, a packed R-tree over chunk bounds, ≤256-element chunks, LOD bands and property columns; the GPU culls them with the tile cull compute and draws instances. Data-layer symbols join basemap collision; feature state, queries and the accessibility snapshot are shared. Modern-only: GPU aggregation (heatmap density, hexbin, grid) and opt-in `symbol-placement-mode: "gpu-approximate"`. Targets [inferred, low confidence]: 10M points at 60 fps on an M-series laptop, 2M on a mid phone; ~250k on Compat.

### 10.4 Data across web boundaries

| Data | Lives in | Copies |
|---|---|---|
| `TileFrame` columns | owner job worker | 0 after decode |
| Streams, chunk records, property tables | job wasm → **one copy** into a transferable `ArrayBuffer` → render worker's payload table → GPU; then freed | 1 |
| GeoArrow batches | disposable ingest worker, one copy per shard | 1 |
| Collision SoA + keys | placement worker | 1 |
| Metadata, arena ranges, candidate lists, hit index | render worker wasm | 1 |
| Camera, hit index, a11y snapshot, elevation pyramid (M7) | ui role on main | 1 per update |

A SharedArrayBuffer build (opt-in, after M12) would remove the job → render copy and allow in-job cancellation; no target depends on it.

---

## 11. Style handling

### 11.1 v8 as unmodified input, plus a separate scene document

`orb-v8` compiles v8 losslessly into a typed Style IR, the engine contract. Orbis never adds keys to v8 JSON; anything v8 cannot express lives in a separately versioned **scene document** that references an unmodified v8 style, so a v8 document means the same thing in every engine (rejected options in §22). v8 cannot be the executed model: it bakes in Mercator (`scheme`, `{bbox-epsg-3857}`), lacks TMS, surface, composition, lighting, 3D and a11y, and carries legacy coercions the engines disagree on.

**Frozen semantics.** Each release pins a spec version, a fixture snapshot (GL JS render and query tests, Native's unique tests, the spec's expression fixtures) and a GL JS version. Precedence: fixtures → spec text → GL JS behaviour, all at the pinned versions (the spec's TS engine is the reference) [verified]. Defaults match GL JS; deviations are opt-in scene-document flags (`geodesicMeasures`, `exactCompositeInterpolation`, `symbolSortKey: "index"`). Unsupported keys warn via per-tier `capabilities.json` and error under `strict` (data-driven `icon-rotation-alignment` validates in GL JS but renders as a constant) [verified]. A generated support matrix replaces hand-edited `sdk-support` cells that are stale in both directions [verified].

### 11.2 IR and pipeline

IR (internal, versioned, CBOR-cacheable, buildable from Rust without JSON): **SourceDecl** (kind, TMS, provider, attribution); **LayerProgram** (layout kernel, filter IR, paint binding plan and class, pipeline spec, pass, query hook, placement scope, a11y role); **SceneDecl** (surface, projection + blend, terrain, lights, atmosphere, constraints, flags); **Composition** (imports, slots, namespaced state; style-spec#591 open) [verified].

Pipeline: S0 merge (v8 + scene document; the scene document decorates, never edits v8 semantics) → S1 parse/validate (JSON-path diagnostics, also as standalone wasm/C) → S2 resolve (imports, state defaults, sprites, fonts, TMS and projection descriptors) → S3 lower (legacy functions/filters → expressions in one place) → S4 plan (render graph, pipeline classes, tier-specific nodes) → S5 prepare (async pipelines in first-visible order) → S6 diff (paint → block/table write; filter/layout → relayout of one family; source → reload). Expressions compile once and ship to workers as bytecode; GL JS parses each at validation, on main and in every worker, and Native clears every source on a new style and has no validator [verified]. `map.edit(tx => …)` publishes one snapshot, giving wrapper reconcilers a diff target (removing the flutter#477 class) [verified].

[implemented at M1, decision 0015] `orb-style`'s `StyleDoc` keeps the document as JSON and typed IR with stable layer and source keys; a transaction (also parsed from GL JS map calls) re-parses only the layers it touches and diffs them into changes classed `Block`, `State`, `Column`, `Relayout`, `Nodes`, `Reload` or `Restyle`. Global state resolves at S2: `orb-expr` folds `global-state` reads at parse, so a global-state write is an edit of the layers that read it. Jobs receive each revision's new layers and sources; results carry their revision and per-node epochs decide staleness; node renumbering keeps every other node's meshes.

### 11.3 Expressions

The 89 public v8 operators (style-spec 26.5.1) with `ParsingContext` semantics, ported from the TS reference into an SSA DAG with a dependency bitset (`zoom, pitch, feature, feature_state, global_state, elevation, line_progress, heatmap_density, resolution, image`). Backends: a register VM with no per-feature allocation; a SIMD column evaluator (dictionary inputs once per entry); the GPU bytecode interpreter in bytecode-class pipelines. Paint binding plans: `Constant`, `CameraCurve`, `TableLookup`, `TableCurve`, `GpuBytecode`, `StateLookup`, `Cpu`. A constant ↔ data-driven flip is a block write plus at most a column job and an async pipeline, never a relayout (GL JS reloads the source and Native relayouts today, `style.ts:1414-1425`) [verified]. [measured at M1] On the Tokyo extract a flip is 5 column jobs and 5 row uploads, zero relayouts and zero mesh uploads; `StateLookup` is a value that reads `feature-state` but no feature data ([decision 0015](decisions/0015-style-edits.md)). [implemented at M1, decision 0017] `GpuBytecode`: `orb-expr::gpu` lowers the GPU-evaluable subset (≤32 instructions, ≤16 stops, 16 registers); `orb-style::CostModel` is the one binding decision, shared by the diff and the renderer. Input columns follow each feature's property row (per-mesh row stride); branch operands compare as exact f64 bits, so every branch equals the VM's and only arithmetic or numeric feature state feeding a branch falls back. Interpreted values evaluate at the camera's exact zoom. The pinned fixtures through the interpreter: 136 GPU-evaluable, all agreeing with the VM natively, in Chrome and on an Adreno 619. Every fixture runs on every backend that can evaluate it. Locale operators use the host `Locale` service with one conformance suite (`number-format` has 5 backends today and the FFI's browser build falls back to `std::to_string`, ffi#797) [verified]. No JS VM; expression-function plugins later through a reserved extern-op slot.

M1 CPU implementation: all 583 pinned style-spec 26.5.1 fixtures pass natively and as wasm in headless Chrome. `orb-expr::Locale` provides the host boundary for comparison, number formatting, locale resolution and case mapping, with a deterministic built-in implementation; `parse_with_locale` and `EvaluationContext::locale` use the same service for folding and runtime evaluation. Inputs include feature/global state, tile-local geometry, line progress, density, elevation, accumulated values, images and script support. Literal feature-property keys remain column-bound. The fixture bundle and native/wasm test share one source and comparison implementation; the GPU interpreter runs the GPU-evaluable fixtures against the VM natively and in Chrome ([decision 0017](decisions/0017-gpu-bytecode.md)); the column evaluator remains later M1 work. See [decision 0005](decisions/0005-expression-locale-and-conformance.md) for locale coverage, harness choice and size measurements.

### 11.4 The scene document

```json
{
  "$schema": "https://…/orbis-scene/0.1.json",
  "style": "https://tiles.example/liberty/style.json",
  "sources": { "linz": { "tileMatrixSet": "NZTM2000Quad" } },
  "projection": { "type": "EPSG:2193" },
  "surface": "plane",
  "layers+": [ { "id": "fleet", "type": "circle", "source": "fleet-arrow", "after": "roads" } ],
  "a11y": { "layers": { "poi": { "role": "landmark", "importance": ["get", "rank"] } } }
}
```

It references or embeds an unchanged v8 style; it may decorate sources (TMS) and layers (a11y), add layers and sources using its modules, and set scene properties; it may not change v8 meanings. Every module has an API equivalent (`tx.setSurface(...)`).

| Module | Content | Milestone |
|---|---|---|
| CRS/TMS | `tileMatrixSet`, descriptor `projection` (#1866 shape), `surface`; demand #168, discussion #163, GL JS #8723 [verified] | M6–M7 |
| Composition, groups | `imports`, `slot`s, namespaced state (style-spec#591); `blend-mode`, z-groups (gl-js #48) | M5 |
| Lighting, atmosphere | `lights`, `atmosphere` (unifies `sky`/`fog`), `*-emissive-strength` (style-spec#1797) | M7–M8 |
| 3D content | `model`, `clip`, `3d-tiles` | M8 |
| Data | `columnar` source | M11 |
| Accessibility | layer `a11y` (`role`, `label`, `importance`) | M10 |
| Symbols, text | `placement-scope`, `symbol-occlusion`, `symbol-placement-mode`; variable fonts; SVG icons (discussion #175) | M8+ |

CRS/TMS and composition are the candidates for spec RFCs with fixtures. An offline `orb-v8 import --dialect mapbox-v8` tool converts same-key, different-schema features (`projection`, `sky`) into a v8 style plus a scene document.

---

## 12. Text and symbols

**One pipeline in jobs.** Native shapes `font-faces` in an LTR-only HarfBuzz island on the render thread (#4513) with a `char16_t` GlyphID; GL JS rasterizes clusters on main with TinySDF [verified]. Orbis keys glyphs by `Pbf(font_stack, code_point)` for glyph PBF ranges, `Font(font_id, glyph_index: u32, variant)` for font files, or `Cluster(font_stack, variant, text)` for web system fonts the page cannot read ([decision 0009](decisions/0009-text-pipeline.md)).
1. **Itemize:** script runs (UAX #24); bidi (UAX #9, `unicode-bidi`, GL JS's `bidi-js` as oracle); grapheme/word/line segmentation from **ICU4X everywhere, including wasm**, with dictionary/LSTM breaks for Thai, Lao, Khmer, Burmese; GL JS's `canCombineGraphemes` kept.
2. **Fonts:** a chain that always resolves, in the pinned GL JS's order: style `font-faces` by unicode-range → local fonts for clusters, complex-shaping scripts and (with `localIdeographFontFamily`) ideographs → the `text-font` glyph PBF range → local fonts again if the range fails. It reports fallbacks (Native stalls tiles forever when a range fails, #4443) [verified].
3. **Shape:** HarfRust with direction and BCP-47 language wherever font bytes exist (always natively); advance-only for PBF stacks.
4. **Wrap** with the shared Knuth-Plass-lite `determineLineBreaks`, then reorder visually.
5. **Rasterize** in a deduplicated glyph service (SDF; MSDF for icons, large and pitched/3D text; colour for COLRv1/emoji) and **pack** into the shared array-texture atlas.

**Web size** [measured at M2 phase 1, decision 0009]: the text core (UCD tables, ICU4X graphemes, `unicode-bidi`, shaping, PBF, SDF) adds 46,447 B br (+45 KiB) to `core.wasm`; SE-Asian data lazy: ICU4X's four dictionaries +604,939 B br (591 KiB), or its LSTM models +299,742 B br (293 KiB); HarfRust lazy (+100–150, from harfbuzzjs at 147 KB br [verified]) when a run has font bytes; PBF-only styles with system fonts use browser cluster rasterization on a worker OffscreenCanvas. **Identity:** with the same font bytes, segmentation, shaping, breaks, layout and placement are identical everywhere; browser-rasterized cluster pixels carry `expected-web-*` tags. **Fixed by construction** [verified]: RTL reversal and Kurdish forms (#4513, gl-js #6232), astral text, the Qt bidi stub (ffi#788), TinySDF on main (a worker prototype cut the worst frame 102 → 44 ms, #8426), never-resolving glyph ranges; tables generated from UCD (GL JS uses 17.0.0, Native hand-codes 12.0.0).

**Symbol layout** is ported once with **GL JS as behavioural reference** (ahead on 15 of 25 ledger items: overlap modes, KDBush cross-tile index #7797, linear `mergeLines` #8436, single-pass halo #7436, globe/terrain collision), plus Native's shared atlas and `TilePlacement` [verified]; high-lineage leaves (`quads`, `check_max_angle`, `intersection_tests`) are ported line by line with golden vectors; SoA without canaries; stable keys `(source-layer, id | promoteId, text)`, where the id is the promoted one, as in GL JS, because raw tile ids need not be stable across zooms. [measured at M2 phase 2, decision 0009] `orb-symbol` matches GL JS's `performSymbolLayout` exactly on 21 whole-layout golden cases and 1,095 leaf cases, natively and in Chrome, with 64-bit keys. It writes flat `Pod` tables per lane and adds 16,527 B br to `core.wasm`; the placement leaves of `intersection_tests` come with the placement lane. The runtime text stage evaluates each symbol layer's features in the tile job, exactly as GL JS's `populate` does on all 22 golden cases. Tiles suspend on missing glyph PBF ranges, and each worker fetches a range once; sprite metadata reaches the jobs before tiles with symbol layers are requested. Placement ports GL JS's collision, cross-tile index and fades into `orb-symbol` and runs on the placement lane (a thread natively, a placement worker of its own on the web). Each drawn world copy places apart, with state words per copy. The symbol pass draws glyph and icon quads with state words and walks line labels in the vertex shader rather than a compute pass (decision 0009). Placement, the upload path and the shader add 40 KiB br to `core.wasm` (522 → 562 KiB); fixture fixes, world copies and the placement worker's codec add 14 more (576 KiB). Local glyphs (clusters, complex scripts, ideographs) draw through a host `LocalGlyphs` service: system fonts with HarfRust natively, an `OffscreenCanvas` on the web (+9 KiB). `font-faces` files draw from the same bytes everywhere through a host `FontFiles` service: `orb-text` natively, `shaper.wasm` on the web (405 KiB br, loaded with the first declared file; `core.wasm` +7 KiB). On the Cairo, Delhi and Bangkok style (`?style=cairo|delhi|bangkok`) web and native frames differ in 10–28 pixels, each by 1/255. Of the 451 mirrored symbol, text and icon fixtures, 408 pass and 14 fail (90.5%), natively and in Chrome alike; the symbol pass costs 0.01–0.11 ms of GPU time on the Tokyo demo with Liberty's labels. Render-thread placement work while the camera turns (`FrameReport::placement_us`) is 0.005 ms median natively and 0.010 ms in Chrome on the Apple M6, and 0.5 ms on the Adreno 619 (moto g, Chrome 156; over budget, #21). A Pixel 7-class run is open. Hidden symbols drop per vertex rather than being compacted: 0.04 ms of GPU time on the M6 and 0.8–2.2 ms on the Adreno with nearly all hidden (#9).

**Placement** runs on a dedicated lane holding the cross-tile index (KDBush with a promoteId fast path; 10k symbols 466 → 1.5 ms, #8470) [verified] and the collision grid, against immutable camera snapshots, time-sliced so newer snapshots pre-empt. Output: double-buffered state-word diffs and sort-key permutations; fades hide 1–2 frames; cadence follows angular velocity. Scopes `viewport`, `tile` (M6), `region`, `world-3d` (M8); `placement: "sync"` for still output. Each commit emits the accessibility snapshot. Later: Slug-style curve text for large, 3D and XR text (GL JS discussion #7282).

---

## 13. Interaction, queries, accessibility

**Interaction (M3).** `orb-interact` turns pointer, touch, wheel and keyboard into camera intents with inertia and **feel profiles** (iOS-like, Android-like, desktop, custom). [built at M3, decision 0021] It drives `orb-camera`'s single-writer store, which holds constraints, `jumpTo`/`easeTo`/`flyTo`, glides and `cameraForBounds` and reports GL JS-style camera events; GL JS's handlers are the reference for each gesture's rates and thresholds. Bounds constraints hold the rotated viewport's extent, where GL JS ignores bearing, so a turned map never shows past them (pitch is ignored, as in GL JS). Independent property animations (removing FFI patch 0015, #4637); closed-form pitch-aware fitting (ffi#764) and horizon-aware visible regions (ffi#622) [verified]; globe-aware panning (versors near poles, GL JS #8154) and terrain constraints from M7. Input-owning hosts drive the camera directly.

**Queries.** `hitTest(point)` is synchronous on the published hit index and symbol index. `project`/`unproject` are synchronous and exact without terrain; with terrain, synchronous approximate on the web ui role plus an async exact variant. `queryRendered` is async in the tile's owner worker (features) and on the placement lane (symbols), with full properties from `TileFrame` (no whole-tile re-decode on main and no PBF re-encode, both done by GL JS today) [verified]. `querySource` is a column scan with the compiled filter. Queries go through the inverse DataToWorld plan and carry a snapshot generation. 3D picking (M8): CPU ray tests on terrain; an async id-buffer pass for models.

The M1 playground publishes non-symbol fill, line, circle and extrusion hit
geometry with resident tile results, including promoted ids. `Map::hit_test`
and `OrbisMap.hitTest(x, y)` synchronously read the last rendered frame in
CSS pixels; camera and style mutations leave it intact until the next
render. [built at M3, decision 0024] Picking does not depend on the camera.
Tile jobs pack each node's features in tile-local data units with their
bounds (`MeshHits`); a frame records the tiles it drew and their clip
regions in world space; placement commits record the labels they show as
world anchors with pixel boxes (a line label as its anchor's circle); and a
layer table holds the paint that sets pixel tolerances. A query unprojects
its point or box with its camera, culls by world bounds and projects only
the candidates for the exact test. The render worker publishes these parts
as they change (`Map::publish_hits`: tiles as they load, change or go, the
drawn list, labels with commits, the layer table with the style), never for
a camera move, and the page's ui role answers `hitTest` with the camera it
has now: 0.2–0.3 ms per hit test on live Liberty, and 0.1–0.2 ms of
publication at a gesture's end where a screen-space snapshot cost 52 ms.
`Map::query_rendered` (the package's `queryRenderedFeatures`, a point or a
box) takes the index's features, topmost first, and asks each tile's owner
job worker for them: geometry and every property from its cached
`TileFrame` (or the tile fetched again), as GeoJSON with GL JS's `source`,
`sourceLayer`, `layer` and `state`. `Map::query_source` filters a source's
drawn tiles in their workers. Of GL JS's 83 mirrored query fixtures, 58
pass natively and in Chrome alike (`cargo xtask rt --suite query`); the
rest are paint semantics (#8): line gap width and offset, feature-state
dimensions, circle pitch scaling, extrusion box sorting, viewport
translate anchors.

The demo site is one map app (`web/demo/index.html`), built on the
`orbis` package: a picker over the live basemaps, grouped by provider, and
Orbis's own showcases (terrain, other projections, complex scripts, the
offline Tokyo extract, whose camera is constrained to the extract); an
attribution control built from the active style's sources; feature popups
and hover and selection highlights through feature state; a debug overlay
(`?debug`) with the frame counters, the thread mode and the build. Its
style editor (`?edit`, which `playground.html` now redirects to) lists the
style's layers by source layer, with visibility toggles and paint controls
generated from the v8 spec's property types; each change is one
`map.edit` transaction, and the editor reports how many frames it took to
show (one, for paint). Its JSON tab replaces the whole style; errors keep
the previous map. In development only, setting `ORB_PLAYGROUND_STYLE`
enables a same-origin Vite endpoint that writes the accepted style and
hover state to the chosen file; `orb-demo --watch-style` polls it and
applies diffed transactions on its event loop. The native log and browser
status report generation and rendered frames after accepting an edit;
file polling is outside that frame count. Pages has no file endpoint or
native dependency.
See [decision 0016](decisions/0016-style-playground.md) for measurements and
the initial picking scope.

**Accessibility.** Neither engine has an accessibility tree; Native's introspection (#4460) is the only seed [verified]. From M2, each placement commit produces a **semantic snapshot**: placed labels, POIs and features with an `a11y.role`, each with text, role, importance (sort key), bounds, bearing/distance; a reading order; a region summary. [built at M2] `orb-a11y` holds the snapshot and `Map::semantic_snapshot` builds it from the last commit's shown state words: per label or icon on screen, its text as the data has it (before `text-transform` and shaping; the icon's name for an icon alone), a role (`road` along lines, `poi` with an icon, `label` otherwise; style-assigned `a11y.role`s are M10's), its layer and layer rank, its sort key, its collision box in viewport pixels, and its great-circle bearing and distance from the view's center. The reading order is upper layers first, then lower sort keys, then top to bottom and left to right; the summary is the center, zoom, counts per role and the first five texts. Tiles carry each instance's text to the render side (two tables). Hosts get it as JSON: `orbis.semanticSnapshot()` on the web demo (main-thread and worker modes alike) and `orb-demo --snapshot … --a11y out.json`. `core.wasm` +4 KiB. **M10** maps it through AccessKit (UIA, NSAccessibility, AT-SPI), Android `AccessibilityNodeProvider` and iOS `UIAccessibilityElement` helpers, and an off-screen ARIA list with roving focus and a live region; adds keyboard feature navigation with an engine-drawn focus ring; maps reduced motion (cross-fade, no inertia), text scaling (a `text-size` multiplier with re-placement), contrast and colour-vision palettes (layer-block remaps, not style reloads); adds contrast lints and expected-tree fixtures.

---

## 14. Embedding and APIs

### 14.1 C ABI and bindings

`orb-capi` is a `#[diplomat::bridge]` module over `orb-api` (decision 0018). Its items carry contracts as Diplomat backend attributes under the reserved name `contract`, which code generators skip and Orbis's emitters read: an execution category per method (`command | query | snapshot | event_batch | render_driver | lifecycle | immediate`, the categories of the FFI codegen branch's `MLN_BINDING` contracts [verified]) and a thread rule per type (`owner`: one thread, the one that created it; `any`). Ownership is Diplomat's: `Box` owned, references borrowed, lifetimes tying a borrow to its owner.

**C conventions** are Diplomat's: opaque pointers released with `*_destroy`; results as `{ union { T ok; E err; }; bool is_ok; }` with an `OrbError` status enum and a per-thread last message; UTF-8 string views in and `DiplomatWrite` buffers out; plain by-value structs. From maplibre-native-ffi (BSD-2) [verified] Orbis keeps the execution model of ffi#760 (the core owns execution, commands return generations, queries live on `Map`), event batches drained by polling, host-supplied graphics, and the mutation currency: style JSON and transactions through `OrbMap_edit` rather than generated per-property setters (Android 9,328 + JNI 4,381 lines in Native) [verified]. Generational handles and `size`-prefixed structs are not used; extensible structs are an M12 question. 0.x until M12.

```c
OrbRuntime_create_result rt = OrbRuntime_create();
OrbMap* map = OrbMap_create(rt.ok);
OrbMap_load_style_url(map, (DiplomatStringView){ url, strlen(url) });
OrbTarget_create_metal_layer_result t = OrbTarget_create_metal_layer(rt.ok, (uintptr_t)layer, w, h, scale);
OrbFrame f = OrbMap_render(map, t.ok);   // per display-link tick while OrbMap_needs_frame(map)
```

**Bindings:** Rust (`orb-api`, M0); TypeScript over `orb-web` (M0 low-level, M3 package); C (Diplomat's header, static library, pkg-config and CMake config, M3); Kotlin (classes over a generated JNI C shim; Android API 24+, also the desktop JVM, M3); Swift (classes over the C header through a module map, M3). The Kotlin and Swift emitters are `diplomat-swift-kotlin`: Rust, over `diplomat_core`'s HIR, no Orbis names, configured by the caller, with golden and compile-and-run tests, so they can be published or upstreamed. Diplomat's Kotlin backend (JNA: 369 ns per call vs 4.1 ns through JNI, `java.lang.ref.Cleaner` needs API 33) and its missing Swift backend are why (decision 0018). Diplomat's C++, Dart and .NET backends remain available. UniFFI is rejected (RustBuffer serialization on hot paths, weak C/Go/Zig/.NET). Every binding runs one scenario suite.

### 14.2 Graphics integration

- **Apps (M3):** surface targets with platform pacing.
- **Toolkits:** owned-texture targets from M3; imported toolkit textures (`AHardwareBuffer`, `IOSurface`/`MTLTexture`, D3D12 shared handles, `GPUTexture`) at M9; toolkit GL contexts through Compat. No process-global side effects: no exported C++ symbols and HarfRust instead of C HarfBuzz, so the symbol clash that broke Compose iOS text (#3777) cannot happen [verified incident]. On the web, host-device mode takes a toolkit's `GPUDevice` or WebGL2 context (the supported version of Compose monkey-patching GL JS) [verified].
- **Game engines (M9):** raw Vulkan/Metal/D3D12 device and queue handles are imported into **Orbis's own** pinned wgpu, so the host's wgpu version does not matter; per frame the host supplies colour target(s), depth and its convention; **Orbis records and submits its own command buffers** (wgpu cannot wrap a foreign one; `as_hal_mut` forbids mixing [verified]) and synchronizes by same-queue order or semaphore/fence; `GeoFrame` anchors engine origins to ECEF/ENU.

  | Host | Device access | Target exchange | Sync [inferred; verified per engine in M9] |
  |---|---|---|---|
  | Bevy | `as_hal` on `RenderDevice`/`RenderQueue`, or shared `wgpu::Device` for N/N−1 | Bevy-created texture | same queue, render-graph node |
  | Slint | `unstable-wgpu-N` | texture import | same queue |
  | Qt Quick 6 (RHI) | `QSGRendererInterface` handles | native texture wrapping | `beforeRendering`; same queue |
  | Godot 4 | `RenderingDevice.get_driver_resource()` | `texture_create_from_extension()` | same queue; no general external semaphore |
  | Unity | `IUnityGraphicsVulkanV2` / `Metal` / `D3D12v7` | `Texture2D.CreateExternalTexture` | plugin render events; D3D12 frame fence |
  | Unreal | `IVulkanDynamicRHI` / `ID3D12DynamicRHI` | RDG external textures | `EnqueueLambda`; fences partly exposed |

- **XR prerequisites (M9; packages later):** multi-view targets, union-frustum cover with per-view cull, native multiview where available (two passes on WebGPU), `world-3d` labels, host-depth occlusion, runtime-driven timing; WebXR via `XRWebGLBinding` first, `XRGPUBinding` (Editor's Draft) by detection.
- **Servers (M6):** Owned/Inline; headless Vulkan, Metal, D3D12, lavapipe; `Still`/`Tile` with tile-scope placement; deterministic, any-TMS output; a `wasm32-wasip2` kernel component (validator, tiler, MLT transcode).

### 14.3 The web package

```ts
import { createMap } from "orbis";               // placeholder, unpublished
const map = await createMap({ container, style: "https://…/style.json",   // or a scene document
                              renderer: "auto",  // "webgpu" | "webgl2"
                              thread: "auto" }); // "worker" (default) | "main"
map.capabilities;                                // tier, fast paths, declared gaps (+ "capabilities" event)
map.camera.current; map.project(lngLat); map.unproject(px);   // sync, ui role
await map.camera.flyTo({ center, zoom });        // resolves on arrival or interruption
await map.edit(tx => tx.setPaint("roads", "line-color", ["get", "color"]));  // resolves at generation g
const hit = map.hitTest([x, y]);                 // sync
const full = await map.queryRendered([x, y]);    // async, owner worker
map.annotations.add({ image, lngLat });          // engine-drawn view annotation (default overlay)
map.overlays.add(element, lngLat);               // DOM overlay on frame acks (±1 frame in worker mode)
for await (const e of map.events("idle")) { … }
```

[built at M3, decision 0021] `web/src` is the `orbis` package as above, with `map.camera` as a control (`current`, `jumpTo`, `easeTo`, `flyTo`, `fitBounds`, `cameraForBounds`, constraints), `queryRenderedFeatures`/`querySourceFeatures`, and DOM `annotations` (engine-drawn annotations are later work); the demo and playground use it.

- **Wasm features:** required SIMD128, bulk memory, sign-ext, non-trapping float-to-int, multi-value, reference types; not used: threads/SAB, exception handling (`panic=abort`), JSPI, Memory64, relaxed SIMD (would need a second build), GC. WebGPU in dedicated workers is in the spec [inferred]; probed at runtime from M0.
- **Modules:** TS host and `core.wasm` (ui, render and jobs roles; Modern HAL) initial; lazy `compat.wasm` (glow executor, Compat nodes, CPU cull, GLSL text), `shaper.wasm` (HarfRust), `seasia.wasm` (ICU4X SE-Asian data), `world3d.wasm` (M7–M8).
- **Payload** (brotli, TS host + `core.wasm`) [inferred]: TS host 30–50; style + expressions + scene document 70–100; decode + layout incl. `mlt-core` 50–80; symbols + placement + bidi + glyphs 60–80; ICU4X rules 30–60; scene tables + graph + WGSL + `orb-hal-wgpu` 80–120; glue 20–30 → **target ≤450 KB, ceiling 550 KB**. Lazy: Compat +60–90, HarfRust +100–150, SE-Asian 591 measured (dictionaries; 293 with LSTM models, decision 0009), 3D +40–80. Watched, not gated: `cargo xtask web` reports and warns above the target and the ceiling, never fails (decision 0003); size cuts are taken when they are simple or improve the code anyway. M0 slice measured (hypothesis ≤300 KB); with the text stack and symbol placement, 522 KB after the cuts in decision 0003, MLT decoding 50 of it (decision 0003). References [verified]: GL JS 6.11 247; maplibre-rs WebGPU 401 (no symbols/expressions); maplibre-rs WebGL (wgpu GL + naga) 846; Native-on-wasm 2,211. Rules: `opt-level="z"`, LTO, `wasm-opt -Oz`, `panic=abort` without strings, no `format!`/`serde_json` in bundled paths, no ICU4C/sqlite/FreeType/codecs, no runtime shader translator, `twiggy` per PR.
- **Demo build and publication:** Vite 8 (rolldown) bundles the four HTML entries from `web/demo/` to `web/dist/`. JS (including wasm-bindgen glue and shared modules), module workers, wasm, extract styles and PMTiles archives have content-hashed filenames. A build plugin rewrites the styles to hashed archive URLs under `/orbis/`. `cargo xtask web` retains the wasm size report in `web/pkg/`; `cargo xtask pages` adds the Vite build, and a GitHub Actions workflow runs it and deploys `dist/` as the whole site on every push to `main`; previous builds' assets are not retained (decision [0012](decisions/0012-web-demo-assets.md)). The snapshot and bench harnesses build and serve `dist/` with HTTP Range support; the fixture driver continues to serve the generated bindings directly so its full test surface is available.
- **Startup and memory:** `compileStreaming` overlapped with the style fetch; one compiled `Module` posted to workers; cull and first-visible pipelines first; load gate set in M3 from the hypothesis ≤1.3× GL JS 6.x on a mid-range Android Chrome (GL JS: 355 ms on headless SwiftShader, #8208) [verified baseline]. Memories start ≤32 MB and grow (no fixed 512 MB heap as in the FFI's Emscripten build) [verified].
- **Also:** `'wasm-unsafe-eval'` documented; portable custom layers (`render(pass, frame)` with device, encoder, local-frame matrices, projection WGSL, depth; `{gl, …}` on Compat), the bar of WebGLOverlayView and Mapbox v3.32 `CustomLayerRenderParameters` [verified]; no controls kit; ESM with self-contained workers (GL JS #8060, #8225, #8672) [verified], a single-file CDN build, a bundler matrix in CI.

### 14.4 Native hosts and batteries

No first-party declarative UI SDK; host helpers for wrappers (maplibre-compose, swiftui-dsl, Flutter, Qt): **Android (Kotlin, M3)** `SurfaceView`/`TextureView` targets, `AHardwareBuffer` textures (API 26+), a `MotionEvent` forwarder, Choreographer pacing, lifecycle (Renderer survives surface loss), accessibility helper (M10); **Apple (Swift, M3)** `CAMetalLayer` and `MTLTexture` targets, gesture forwarder, `CADisplayLink`, accessibility helpers (M10); **desktop** winit host (also the demo) and AccessKit (M10). Sizes [inferred]: Android arm64 `.so` ≤8 MB (2D, Vulkan + GLES), ≤12 MB full; iOS ≤7 MB; vs Native's 33.2 MB multiBackend AAR, 15.9 MB Vulkan `.so`, and wgpu-native's 5.86 MB with all backends [verified]. **Batteries** (M5, engine-level plugins implemented once; reimplemented 4–6 times across SDKs today) [verified]: location-indicator layer, managed annotation source, PMTiles region downloader, snapshotter. Offline is PMTiles extracts + an HTTP cache with batched metadata writes.

---

## 15. Extensibility

Registries are per runtime and removable (Native's `LayerManager::get()` is a singleton, GL JS's protocol registry page-global, and Native's plugin ABI says registration "cannot be undone") [verified]. Axes: **protocol** (`fetch(url | range)`, run in the job that needs bytes, no worker → main → worker bounce as in GL JS `ajax.ts:262-286`); **source provider** (`TileFrame`/raster/DEM per `TileKey` in a declared TMS; contours without maplibre-contour's MVT encode-decode hop); **decoder**; **layer type** (property descriptors, a layout kernel emitting chunks into host arenas, WGSL against `fetch`, optional Compat support declared, optional frame hook and query predicate), so plugin layers ride the GPU-driven path; **projection** (descriptor or kernel + WGSL); **render-graph node**; **host services**; expression functions later.

Built-ins are plugins from M1 and optional families are Cargo features and lazy web modules (Native's AAR grew 8.4 → 17.5 MiB without a modular build, #3866) [verified]. The **public plugin ABI (M12)** keeps `plugin_api.h` v1 discipline (`struct_size`, host ABI range, status enums, borrowed-in/copied-out, no STL/vtables/exceptions) and adds what downstream patched in (textures, render targets, sourceless frame-driven layers, rotation, frame queries, large uniform blocks, attributes by name; maplibre-plugins: 6 patches, ~3.3k lines) [verified]. Plugins import `orb::project_tile`, never a Mercator matrix. Later: one plugin source compiled to native and wasm (particles today is 8,857 TS + 4,932 Zig + a 2,183-line JS paint engine) [verified], and sandboxed wasm data plugins. Escape hatch: a `Custom` node and raw backend handles natively.

---

## 16. Testing and conformance

- **`orb-fixtures`**: a read-only mirror of GL JS render and query tests, Native's unique tests and the spec's 583 expression fixtures, pinned per release. Orbis keeps its own tags (`expected-orb.png` → `expected-orb-compat.png` → `expected-web-*`, perceptual tolerances) and ignore lists, each linked to an issue; maplibre-rs #331 vendored the GL JS tests the same way [verified]. Seeds: GL JS's 1,718 render tests and Native's 87 unique; 1,186 of 1,250 shared are semantically identical; query tests 136/126 [verified]. A declared operation vocabulary replaces "call any `map[name](...)`" and Native's ~41 hand-parsed ops [verified].
- **Real-world style suite** (gates M5's Mercator styles and M6's non-Mercator ones; whichever lands first builds it): pinned OpenFreeMap Liberty, Bright, Positron; Protomaps light and dark; Americana; OSM Bright; VersaTiles Colorful; from M6 NZTM2000Quad, EPSG:3413 and UTM-grid styles; local PMTiles extracts; 30 cameras each. **Yardstick:** Orbis's perceptual difference from pinned GL JS ≤ Native's difference from the same GL JS on ≥90% of snapshots (M5), ≥95% (M12). Non-Mercator styles use reviewed goldens and GL JS #8723 output once merged.
- **Orbis suites:** CRS (Mercator vs GL JS on a 4,000-camera sample, the #8723 method; CPU/GPU agreement z0–z22; TMS traps; poles; transitions); 3D (Khronos glTF samples, lighting, occlusion); text (Arabic, Kurdish, Hebrew, Devanagari, Khmer, Burmese, Thai, vertical CJK, astral, emoji); **GPU-path equivalence** (cull/compact compute vs a CPU reference; GPU interpreter vs VM); accessibility trees; host-service, plugin and binding conformance.
- **Runners:** one Rust runner, every PR, on Modern (lavapipe, Metal, WARP, Chrome WebGPU/SwiftShader) and Compat (llvmpipe, Chrome WebGL2); real GPUs nightly. **Web/native identity:** the same fixture through `core.wasm` in Chrome and natively must agree within tolerance. Golden vectors for layout, chunking and placement; golden draw streams and read-back indirect args; property tests; fuzzing of every input format, styles, scene documents and expressions (Native's #4440/#4403 crash classes) [verified]; Miri/loom on scheduler and atlases; sanitizers on the C shims.
- **Deterministic perf CI from M0:** frozen scenarios over local extracts (Americana, Liberty, a #4749-like nautical chart, a CJK/Arabic/Devanagari city; later polar, globe + terrain, ellipsoid + glTF, 5M points); PR gates on counters (commands per camera-only frame, draws, dispatches, bundle re-records, pipelines per style, bytes uploaded, objects touched, visible chunks, glyphs, instruction counts, sizes), never name-filtered (Native's PR regex `^[^A][^P][^I].*` skips every render benchmark) [verified]. A nightly device matrix (Pixel 7a, Tab A7 Lite class, an API 24–26 GLES device, Exynos S24, MediaTek, budget Mali and PowerVR, iPhone, M-series Mac, Windows NVIDIA, three browsers) records p50/p99 frame time, tile-arrival p99, GPU time, per-worker memory and a power proxy. GL JS and Native packages run read-only in the same harness.

---

## 17. Devex and build

- **Cargo + `xtask`:** `cargo test` (unit, kernel, binding, headless conformance); `cargo xtask web` (`core.wasm`, lazy modules, TS host via Vite/rolldown); `cargo xtask android|apple` (AAR, XCFramework; Gradle and SwiftPM only package); `cargo xtask demo [--web]` (the same demo natively and in a browser); `cargo xtask rt --filter text-arabic` (a corpus slice on both). A static library + header + pkg-config/CMake config serves CMake, Bazel, BUCK2 and Meson. No submodules (Native: 38 submodules, ~800 MB vendor, 9+ build systems), no Emscripten (the FFI's emsdk is "several gigabytes"), stable Rust pinned with `rust-toolchain.toml` + mise [verified].
- **CI:** the PR lane under 20 minutes (Native's p50 is 105–115 min; its Windows CI is off, #3670) [verified]; packaging on path changes; device farm nightly.
- **Tools:** `orb-viewer` frame debugger (command stream, render graph, per-node timings, tile/chunk/visible-list inspector via indirect-arg readback, placement overlay, plan per source, pipeline classes, tile timeline) natively, as a browser devtools panel and as `capture()`; compiler-style style diagnostics in the validator, runtime and an LSP (`orb-v8` in wasm); stats API and HUD; hot reload of styles, scene documents and WGSL; generated API docs and support matrix; `AGENTS.md` (41% of Native's merged PRs in six months disclosed AI assistance) [verified].
- **Budgets** [inferred]: crate LOC ceilings (§5.2); ≤60 native / ≤25 web direct dependencies with a licence allowlist; clean build ≤4 min, kernel `cargo check` ≤10 s; MSRV stable − 4; `cargo-semver-checks` and C ABI diffs; per-module size reports; `unsafe` confined and documented; exactly two render configurations (M+ fast paths must be result-identical to Modern); wgpu within ~8 weeks of each major. One version train (semver for crate, npm package and bindings; calendar epochs for the ABI).

---

## 18. Performance budgets

**Invariants:** camera-only frames touch O(changed tiles) on [S], write the frame block and changed rows, dispatch once and replay an unchanged command stream (Modern); per-stage budgets defer rather than stall; nothing blocks [S] or [H] (no readback, sync compile, cross-thread `get()`, or fsync per cache hit); idle = zero frames; every stage, job, cache, worker and payload has a budget.

"Today" is [verified] (`gap-perf-evidence` and cited issues); targets [inferred], Modern unless noted. Confidence: **H** mechanism removes the work; **M** GPU/driver dependent; **L** unmeasured. "Gate": first enforcing milestone.

| # | Bottleneck | Today | Target | Conf. | Gate |
|---|---|---|---|---|---|
| 1 | Steady CPU frame, Pixel 7 Pro class, Americana | Native 3.39 ms async encode (was ~2.2 ms) | ≤0.5 ms on [S]; hypothesis ≤0.25 ms | M | M2 |
| 2 | Serialized frame, same device | 9.8 ms CPU + 7.2 ms GPU = 56 fps | ≤2 ms CPU + ≤5.5 ms GPU | M | M5 |
| 3 | Draws, Liberty Tokyo z15 | GL JS 449.6 state-changing; `Painter.render` 1.66–1.75 ms | ~60–90 indirect draws and commands per pass (Modern) | H count, M time | M0 / M2 |
| 4 | Vertex-pulling overhead | meshes draw compacted indices: ≤1.02× per-tile indexed draws on Apple M6 and Adreno 619 (decision 0002); Mali/PowerVR unmeasured (#16) | fills: ≤2.3× unique-vertex invocations; GPU time ≤1.1× per-tile indexed draws on Mali/PowerVR | L | M1 |
| 5 | Tile-arrival spikes | Native p1 encode 15–27 ms; GL JS worst 102 ms; S24 worst 50–56 ms | p99 ≤1.5× median; worst ≤33 ms on S24 class | M | M5 |
| 6 | Web main thread during load | GL JS 2,089 ms; TinySDF 27.6% | worker mode ≤10% of GL JS; main mode ≤40% | M | M3 |
| 7 | Placement on the critical thread | Native unsliced, guards 37% CPU; GL JS 2 ms/frame | ≤0.2 ms per frame to apply | H | M2 |
| 8 | Hidden symbols drawn | 996 for 4 visible | ∝ visible | H | M2 |
| 9 | Line-label reprojection | CPU every frame, both engines | 0 CPU Modern; ≤0.3 ms Compat | M | M2 |
| 10 | Shader compiles at load | GL JS 26–28 ms sync; MediaTek crash | no blocking natively; web first-full-frame gate from M3 data | L | M3 |
| 11 | Feature-state write | 11.9 ms (GL JS after #7550) | ≤0.05 ms | H | M1 |
| 12 | Constant ↔ data-driven flip / paint edit | source reparse or relayout; ~40 frames via FFI (ffi#736) | no relayout; 1 frame | H | M1 |
| 13 | Layer-count scaling | iOS drops >482 symbol layers (#4410) | commands ∝ layers | H | M2 |
| 14 | Idle repaint | 88 renders/s for one 30 fps icon (#8208) | renders = changes | H | M3 |
| 15 | Cold open with cache | Native 1,327 ms, 1,011 ms fsync (#4748) | ≤450 ms | H | M5 |
| 16 | Web payload | GL JS 247 KB br; Native-on-wasm 2,211 KB br | ≤450 KB br target, 550 ceiling | L | M5 |
| 17 | Android binary | Native Vulkan `.so` 15.9 MB; multiBackend AAR 33 MB | ≤8 MB 2D, ≤12 MB full | L | M4 |
| 18 | Device-loss recovery | unmeasured in both engines | ≤1 s native mid phone; ≤2 s WebGPU → WebGL2 | M | M4 |
| 19 | Pipelines per basemap | runtime permutations; circle up to 2^7 | ≤40, CI-counted | M | M2 |
| 20 | Projection switch | full reload (GL JS); unsupported (Native) | ≤2 frames, no refetch | H | M7 |
| 21 | Globe tile cover | 50.7 Hz vs Mercator 157 Hz | ≤1.5× Mercator | M | M7 |
| 22 | Extrusion memory at z17 | 0.9–1.6 GB (#4107) | ≤300 MB | M | M8 |
| 23 | Large client data | no native API; GL JS slowest in a GeoJSON study | 10M points 60 fps laptop, 2M mid phone; ~250k Compat | L | M11 |

Not fixed: GPU fill rate on very low-end devices (Tab A7 Lite runs Americana at 18–23 fps) beyond render scale and overdraw control; driver bugs beyond the allowlist, quirks table and farm.

---

## 19. Relationship to existing projects (technical)

| Project | What Orbis reuses or learns |
|---|---|
| maplibre-style-spec | v8 at a pinned `v8.json`; the 583 expression fixtures; the TS engine as oracle; CRS/TMS and composition as candidate RFCs with fixtures |
| MLT | `mlt-core` (MIT OR Apache-2.0) as a dependency via its columnar API (#1901); fixes upstream, not a fifth decoder |
| GL JS (BSD-3) | fixtures mirrored read-only; globe math (#7916), subdivision, `canCombineGraphemes`, symbol leaves and placement semantics, WebGL state cache, RTT fingerprints (#7549), CSC camera (#8694), #8723's planar-CRS approach |
| MapLibre Native (BSD-2) | WGSL seeds (4,567 lines), driver workarounds as quirks data, `plugin_api.h` discipline, `DynamicTextureAtlas` design, `TilePlacement`, unique render tests |
| maplibre-native-ffi (BSD-2) | the ffi#760 execution model; execution categories; event batches; JSON as mutation currency (decision 0018 replaced its handle and status conventions and its emitters with Diplomat's) |
| maplibre-rs | prior art: a Rust/wgpu renderer with a 401 KB br WebGPU-only wasm build, a GL JS fixture harness (#331), a globe RFC (#339); its forked Bevy render graph and wgpu churn (#311) inform §8.9 |
| Martin | serves MLT, fonts, sprites and named TMS grids (martin#3280); a natural source of WOFF2 font subsets and KTX2 rasters |
| deck.gl, OpenLayers, Cesium, harp.gl, Open Mobile Maps | common-space precision; raster triangulation; RTC frames and 3D Tiles; `TilingScheme × Projection`; OMM 4.0 (KMP, wasm, CRS in styles) as the closest prior art |

Bugs found through mirrored fixtures or ported algorithms are reported to the owning engine with the fixture attached.

---

## 20. Build path

Each milestone is a vertical slice rendering on the web and natively from the same crates, with CI for both. No dates, no costs: a milestone is done when its exit criteria pass. Fixture percentages refer to the frozen snapshot pinned at the milestone's start, restricted to the directories it declares. The path is a buildout and tracks no release.

After M2 the path splits into two lanes that run in parallel, each with one or two sub-lanes at a time:
- **Rendering lane** (style spec → rendering breadth): M6 any CRS and servers → M7 globe, ellipsoid, terrain, atmosphere → M8 3D content and lighting; M11 data layers.
- **Platform lane** (bindings → platforms): M3 interactive hosts → M4 Compat and device robustness → M5 hardening and batteries; M9 guest embedding, M10 accessibility host mappings.

Rendering-lane milestones do not wait for M3–M5. Where an exit criterion needs the other lane's work (Compat coverage in M6–M8 and M11 needs M4; any-CRS gestures and queries in M6, and M11's phone demo, need M3), whichever of the two milestones finishes second carries it.

> **As tracked:** the GitHub milestones M0–M8 carry working versions of these exit criteria, used as goals reported at each milestone's end rather than hard gates (emulators, simulators and one phone instead of a device farm; no Windows in M0–M8). The GitHub milestones follow the same two lanes. The architecture is unchanged.

| # | Milestone | Lane | Proves | Depends on |
|---|---|---|---|---|
| **M0** | First light | core | sans-thread core; wasm + WebGPU and native wgpu from one crate graph; GPU-driven submission; web upload path; CI on both | — |
| M1 | Style engine and 2D layers | core | expression/style compiler, binding classes, feature state, every non-symbol 2D layer | M0 |
| M2 | Text and symbols | core | text stack, placement lane, GPU symbol path, identical layout on web and native | M1 |
| M3 | Interactive hosts everywhere | platform | gestures, camera, queries; web package; desktop, Android, iOS hosts; C ABI and bindings | M2 |
| M4 | Compat and device robustness | platform | WebGL2/GLES path that leaves Modern untouched; allowlist, quirks, sentinel, device loss | M3 |
| M5 | Hardening and batteries | platform | real-world styles, budgets and size ceilings on the device matrix; batteries, offline cache, composition | M4 |
| M6 | Any CRS and servers | rendering | TMS per source, planner, planar families, Still/Tile | M2 |
| M7 | Globe, ellipsoid, terrain | rendering | surfaces, non-affine rows, precision, terrain incl. the web ui role, atmosphere | M6 |
| M8 | 3D content and lighting | rendering | glTF, lit extrusions, `world-3d` labels, 3D Tiles 1.1, shadows (M+) | M7 |
| M9 | Guest embedding | platform | SharedDevice, host depth, HostDriven, multi-view, toolkit texture import | M4 (full value after M8) |
| M10 | Accessibility host mappings | platform | AccessKit, Android, iOS, ARIA from one snapshot | M3 |
| M11 | Data layers | rendering | GeoArrow → GPU chunks sharing placement with the basemap | M2 |
| M12 | Stable contracts | both | plugin ABI, C ABI 1.0, scene document 1.0 | M5, M8–M11 |

M9–M11 need only the listed milestone for their core and can be reordered within and across lanes; M12 joins the lanes.

**M0 First light** (the smallest slice that proves the architecture).
- *Builds:* workspace, `xtask` and CI from the first commit (Linux lavapipe, macOS Metal, Windows WARP, headless Chrome WebGPU/SwiftShader; size, counter and boundary gates; kernels built for `wasm32-unknown-unknown` and `wasm32-wasip2`). `orb-runtime`, `orb-jobs` (native pool; sliced web workers; inline; a `HostDriven` test driver), `orb-host` (ureq; `fetch`; PMTiles range reads). `orb-tile` (MVT, MLT via `mlt-core`), `orb-source`, `orb-layout` (earcut, clip-at-layout, GL JS line extrusion, chunking). `orb-schema`/`orb-v8`/`orb-expr` subset for background, fill and line (constants, zoom `interpolate`/`step`, `get`, `match`, `case`, comparison filters; VM and column evaluator; constant, curve and table classes). `orb-geo`, `orb-crs` (WebMercatorQuad, `Affine`, cover), `orb-camera`. `orb-hal` + `orb-hal-wgpu` with the web upload path; `orb-shaders` (composer; background, fill, line, cull); `orb-scene` (arenas, affine rows, chunk table, candidate lists); `orb-render` (Upload, Cull, Layer2D; vertex-pulled `drawIndirect`; bundle replay). `orb-web` (one `core.wasm`, three roles, both thread modes). Fixture mirror, headless runner for native and Chrome, winit demo, browser demo, one perf scenario.
- *Demo:* OpenFreeMap Liberty (or Protomaps light) reduced to background, fill and line, over a local PMTiles extract with an MVT/MLT switch; `cargo xtask demo` in a native window on macOS, Windows, Linux; `cargo xtask demo --web` in Chrome and Safari 26 with a worker/main toggle; mouse pan and wheel zoom; a HUD with draws, dispatches, bytes uploaded and objects touched.
- *Exit:* (1) ≥80% of mirrored `background-*`, `fill-*` (no patterns) and `line-*` (no patterns, dashes, gradients) fixtures using only M0 features pass on lavapipe, Metal, WARP and Chrome WebGPU; (2) every passing fixture matches between `core.wasm` in Chrome and native within tolerance; (3) MVT and MLT of the same data give identical columns and pixels; (4) camera-only frames show one dispatch, one bundle replay, zero recorded commands, objects touched ≤ changed tiles, GPU draws = Σ(nodes × pages); (5) zero payload bytes pass through render-worker wasm memory; (6) `core.wasm` size gated with a `twiggy` report, wgpu's browser overhead recorded and the web-sys contingency decided; (7) WebGPU-in-worker availability recorded for Chrome, Safari 26, Firefox; (8) first vertex-pulling vs per-tile indexed GPU timings on an Adreno, a Mali and an Apple GPU.
- *Render-platform evidence:* `cargo xtask rt --platform all` runs the same Rust scope/metadata/pixelmatch on Metal, headless Chrome WebGPU, and Docker lavapipe. Of 101 in-scope mirrored fixtures (177 skipped), Metal and Chrome pass 99 (98.02%); lavapipe passes 100 (99.01%). Every one of the 99 host-native passes matches Chrome with zero pixelmatch differences. `fill-outline-color/multiply` fails by 2 pixels on Metal/Chrome and passes on lavapipe; `line-miter-limit/default` fails by 98 pixels everywhere. These renderer differences remain open. Main-thread browser fixtures exercise async offscreen texture readback; worker transport/upload coverage remains separate. See [runner instructions and measurements](../crates/orb-fixtures/README.md).
- *PMTiles/demo evidence:* committed central Tokyo OpenFreeMap extract, 11 tiles at z10–14; 2,815,031 bytes MVT and 2,845,792 bytes MLT. `cargo xtask extract` reproduces it via bounded range reads; `cargo xtask tile-parity` checks identical full TileFrames for every tile. Native `--format` and web `?format=` switch archives. The native GPU test and web snapshot `--compare` require zero MVT/MLT pixel differences. Both browser thread modes passed; 75 draws, one dispatch, 252 candidates at the default z15 view. See [decision 0001](decisions/0001-pmtiles.md) and [extract provenance](../web/demo/data/README.md). Other M0 exit criteria remain independent.

**M1 Style engine and 2D layers.**
- *Builds:* full `orb-expr`, `orb-style` and `orb-v8` (validation, migration, `capabilities.json`); GeoJSON (geojson-vt 6 port), raster, raster-dem; the remaining non-symbol layers and patterns; sprites; feature and global state; built-ins through the internal registry; `Prepare` compute; the segmented-line experiment (measured and shelved, decision 0013); LSP v0.
- *Demo:* a style playground: browser and native viewer side by side; JSON edits update both within one frame; hover highlights via feature state.
- *Exit:* 583/583 expression fixtures through every applicable backend on native and in Chrome; ≥90% of mirrored fixtures in the M1 layer directories on Modern, web and native; one frame per paint edit, no relayout on a binding flip, feature-state write ≤0.05 ms; interpreter occupancy measured on Mali-G52/G57, PowerVR, Adreno 6xx and the cost-model threshold set; §18 row 4 met on fills or the per-family alternative chosen; segmented lines adopted at ≥ the CPU path's pass rate, or shelved with data.
- *Vertex pulling per family (exit criterion 5):* the per-family alternative is chosen ([decision 0002](decisions/0002-vertex-pulling.md)). Vertex pulling lost the rule on the Adreno 619 (1.51× per-tile draws at z12, +12.6 ms), so fill, fill outline, line and fill-extrusion nodes draw compacted indices kept by the cull dispatch: on the Apple M6 (native, Chrome) and the Adreno 619 their frame is at most 1.02× per-tile draws and 1.04–1.48× faster than pulling, the CPU frame is pulling's, and camera-only frames keep one dispatch and bundle replay. Point families and image tiles keep vertex pulling; per-tile draws and the submission switch are deleted. Render fixtures are unchanged natively and in Chrome (1010 pass, 26 fail, 108 skipped). Mali and PowerVR are unmeasured (#16).
- *Raster/DEM evidence:* `raster`, `raster-dem` (TileJSON, templates, `pmtiles://` images, `tileSize`, `scheme`, `bounds`, zooms, the three DEM encodings) and the `raster`, `hillshade` (all five methods, up to four lights) and `color-relief` layers render natively and in Chrome: raster 34/36, hillshade 20/20, color-relief 7/7, every pass identical between Chrome and Metal; the two raster failures need mipmaps. Image pixels reach the GPU from transferred `ImageBitmap`s in worker mode (decision 0007).

Sprites/patterns evidence: the initial fixture mirror expands
from 99/101 to 131/133 passing on native and Chrome. After the expanded mirror,
expression engine, raster and circle/heatmap rebase, both pass 271/275 in scope
(147 skipped), up from the recorded 237/241 baseline; all 34 newly enabled cases pass.
Background 17/17, fill 43/44, line 75/76; the original 99 passes remain passing.
All 271 native passes match Chrome, with unchanged fixture comparisons. Main
and worker modes render the pitched line and data-driven fill patterns with
zero differing pixels. Shared image tables, job references and lifecycle are
recorded in [decision 0008](decisions/0008-image-atlas.md). No lavapipe rerun for
this extension.

Dashed/gradient line evidence: the original 422-test mirror
moves from 271/275 to 308/313 in scope on native and Chrome, with every
original pass retained and 308/308 passing images matching. Line fixtures
move from 75/76 to 112/114; all 9 gradient and 7 combined gradient/dash cases
pass. The added failure used MVT v1 tiles, since supported. After the symbol/text/icon
mirror rebase, the 883-test suite gives 312 pass, 5 fail, 566 skip on each
platform, with 312 matching passes. `core.wasm` grows by 6,568 B Brotli to
420,908 B (411.04 KiB), leaving 39,892 B below 450 KiB. `cargo xtask check`,
including the expanded camera-only and atlas lifetime tests, passes. Details
and per-family tables: [decision 0010](decisions/0010-line-textures.md).

Style edits evidence, exit criterion 3: `Map::edit` applies a
transaction without reloads. Uniform paint and visibility edits take one frame
and no jobs; a constant ↔ data-driven flip re-evaluates rows from cached
tiles (5 column jobs, 0 relayouts); a filter relays out one layer (6 jobs, 0
loads); layer add/move/remove lays out only the new layer, and retained
nodes (extrusions included) keep their meshes; a new source loads only its
tiles; every case's pixels equal a fresh load of the edited style. A
feature-state write is one table row: 0.003–0.010 ms native, about 0.004 ms
in Chrome (target ≤0.05 ms). The web package's `edit`, `setFeatureState` and
`setGlobalState` work in both thread modes. The mirror adds GL JS's
runtime-styling, feature/global-state, filter, visibility and zoom-history
directories (1144 render tests with `sky`); of the 250 with `operations`, 127
are in scope and 126 pass natively and in Chrome. All mirrored tests: 519
pass, 7 fail, 618 skip on both, all 519 passes identical across platforms;
the 883 mirrored before go from 352 / 6 / 525 to 368 / 7 / 508. `core.wasm` grows 416 → 459 KiB Brotli. See
[decision 0015](decisions/0015-style-edits.md).

GPU bytecode evidence, exit criteria 1 and 4: the interpreter evaluates the
136 GPU-evaluable pinned expression fixtures (and eight precision probes)
exactly as the VM decides their branches: 447/447 inputs agree on Metal, in
Chrome 156 and on an Adreno 619; 289 fall back, none for precision. Render
fixtures are unchanged with every compilable value interpreted and at the
default, native and Chrome alike. The cost-model threshold is set from an
Apple M6 and an Adreno 619 at 0; Mali-G52/G57 and PowerVR are open (no
device, [#16](https://github.com/sargunv/orbis/issues/16)). Paint reading
feature state and feature data draws its state when it compiles (#6).
`core.wasm` grows 562 → 578 KiB Brotli. See
[decision 0017](decisions/0017-gpu-bytecode.md).

**M2 Text and symbols.**
- *Builds:* `orb-text`; `orb-symbol`; the GPU symbol path (compaction, line-label compute, state words); the web placement worker; lazy shaper and SE-Asian modules; the semantic snapshot.
- *Demo:* Liberty and Americana with labels on web and native; Cairo, Delhi and Bangkok side by side with identical label layout.
- *Exit:* ≥90% of mirrored symbol/text/icon fixtures (448 GL JS paths) on web and native; RTL/complex-script matrix green with byte-identical layout vectors between web and native; placement apply ≤0.2 ms, hidden-symbol work ∝ visible, zero CPU for line labels; §18 row 1 on a Pixel 7-class device; initial payload within 450 KB br or a recorded plan to meet 550 by M5; ≤40 pipelines per real-world basemap.

**M3 Interactive hosts everywhere.**
- *Builds:* `orb-interact`; camera animations and fitting; queries; events and idle; view annotations; the web package; Android and Apple helpers with generated Kotlin and Swift bindings; C ABI 0.x; owned-texture targets; pacing; the Diplomat evaluation.
- *Demo:* one reference app (same style, gestures, tap-to-query, hover, annotations) on Chrome, Edge, Safari 26 and Firefox (Windows, macOS), native macOS/Windows/Linux, Android (allowlisted Vulkan) and iOS; the mobile apps written against the generated bindings.
- *Exit:* Tier 1 CI green including Android and iOS builds and farm smoke runs; §7.2 latency, skew and main-busy measurements recorded and the default fixed by its rule; cold and warm pipeline creation measured on Chrome/Windows D3D12, Safari iOS, Chrome Android and the load gate set; worker-count A/B recorded; binding scenario suite green for Rust, C, Kotlin, Swift, TS; idle = zero frames everywhere; main busy ≤10% of GL JS in worker mode; mobile binary sizes measured.

*Native hosts and bindings evidence:* `orb-api` packages a native host's frame (runtime, map with camera store, gestures, events and wake, surface or texture targets) and `orb-capi` exports it as a Diplomat bridge: 59 C functions, a static library with pkg-config and CMake files, and Kotlin/JNI and Swift bindings generated by `diplomat-swift-kotlin` (decision 0018). A trivial call costs 7.4 ns through the Swift and 6.1 ns through the Kotlin. The binding scenario (style, pixels, hit test, query, edit, click, idle, projection, errors) passes through Rust, C (pkg-config and CMake), Kotlin on the JVM and Swift; TS is open (#25). The Android host (SurfaceView, Vulkan, Choreographer) runs the reference app on the moto g (Adreno 619): live Liberty, gestures, a tap popup from `queryRenderedFeatures`, the Tokyo Station pin, zero frames while idle. The macOS app (CAMetalLayer, display link) renders through the same Swift bindings; the iOS app builds, but the iOS simulator's Metal lacks indirect execution, so devices there report that and iOS needs a device (#27). Two Android fixes: Qualcomm's Vulkan driver drew nothing from `drawIndexedIndirect` after wgpu's indirect-call validation pass (gfx-rs/wgpu#8801), which native Qualcomm devices now skip (§8.8); and fontdb finds no Android fonts, so `/system/fonts` is loaded for local CJK glyphs. Pipelines, Tokyo extract: Apple M-series (Metal) 16 in 1,581 ms on the first run after a build, 20–39 ms after (driver cache); Adreno 619 (Vulkan) 16 in 5,473 ms cold, 110–119 ms warm through the native pipeline cache hosts now keep, load to idle 6,969 → 1,257–1,668 ms. Chrome Android, Chrome/Windows D3D12 and Safari iOS are open (#26). Sizes (`dist` profile: LTO, panic=abort, stripped): Android arm64 `liborbis.so` 10.67 MB (§14.4 target ≤8 MB) and AAR 5.37 MB; `liborbis.a` 21.3–21.4 MB per Apple slice before linking; iOS arm64 app executable 9.95 MB stripped (target ≤7 MB); macOS demo 12.0 MB. Size is watched, not gated.

**M4 Compat and device robustness.**
- *Builds:* `orb-hal-gl`; Compat nodes and `fetch`; data-texture mirrors; CPU cull and line labels; lazy `compat.wasm`; allowlist, quirks table, crash sentinel; device-loss recovery incl. WebGPU → WebGL2; capability events.
- *Demo:* the reference app on Firefox Linux, Safari on iOS 18 and an Android 10 GLES device; a forced device loss mid-pan recovers.
- *Exit:* ≥90% of the declared Compat fixture subset on llvmpipe and Chrome WebGL2, with every §8.7 gap warning under test; Modern counters, sizes and fixture results unchanged by the merge; device-loss targets met and sentinel demotion tested by fault injection; Android `.so` ≤8 MB (2D, both backends); Compat LOC ceilings held.

**M5 Hardening and batteries.**
- *Builds:* hardening; batteries; offline cache; generated support matrix; docs; composition and groups.
- *Demo:* the reference app on every target with the real-world suite's public styles.
- *Exit:* real-world suite ≥90% within the yardstick on Modern (web and native) and on Compat for its supported layers; §18 rows 1–3, 5–16 and 19 on the device matrix; initial payload ≤550 KB br; fuzzers green 30 days; no open crash or blank-screen bugs on Tier 1; docs and the generated support matrix on Pages.

**M6 Any CRS and servers.**
- *Builds:* TMS registry; scene document v0; `CpuResample`, `RasterWarp`, planar `GpuPerVertex`; planar families and EPSG table; mixed TMSs; any-CRS gestures and queries; `Still`/`Tile`; the CLI; the `wasm32-wasip2` kernel component.
- *Demo:* interactive NZTM2000Quad and EPSG:3413 maps on web and native; a Mercator basemap over a warped EPSG:3413 raster; a UTM tile pyramid from the CLI.
- *Exit:* CRS suite green (4,000-camera Mercator pin, per-family agreement, TMS traps, polar data); non-Mercator real-world styles pass reviewed goldens on Modern and Compat; tile-edge label agreement and byte-identical repeat output in `Tile` mode; Mercator fixture and real-world results unchanged by the merge.

**M7 Globe, ellipsoid, terrain.**
- *Builds:* surfaces and `Geodetic`; non-affine rows; adaptive projections and v8 `globe`; horizon culling; globe gestures; terrain; the web elevation pyramid; `Atmosphere`; v8 `terrain`/`sky`/fog; KTX2; lazy `world3d.wasm`.
- *Demo:* a v8 globe zoom from space to street; an ellipsoid terrain flight; Mercator → globe → Equal Earth; `WorldCRS84Quad` data to the pole.
- *Exit:* ≥90% of mirrored globe/terrain/sky fixtures on Modern (globe also on Compat); projection switch ≤2 frames with no refetch; globe cover ≤1.5× Mercator; z18–z22 agreement on the farm before globe-at-every-zoom is enabled; no terrain camera jumps; web sync `unproject` within its documented error.

**M8 3D content and lighting.**
- *Builds:* lighting block; lit extrusions; `orb-3d` glTF layer; `world-3d` scope and depth-tested symbols; MSAA; `clip`; Hi-Z occlusion; id-buffer picking; shadows on M+; 3D Tiles 1.1 last.
- *Demo:* a lit city with glTF landmarks and occluded labels; a photogrammetry 3D Tiles set with MapLibre labels and draped roads.
- *Exit:* Khronos glTF samples; lighting and occlusion fixtures; z17 extrusion memory ≤300 MB; a 3D Tiles conformance subset; Compat warns for every 3D gap.

**M9 Guest embedding.**
- *Builds:* HAL external resources; `HostDriven` with `JobSink`; host-depth targets; multi-view with per-view cull; `GeoFrame`; `orb-wgpu-N` adapters; toolkit texture import; GL-context targets on Compat; Bevy and Slint integrations; a Qt Quick sample over the C ABI.
- *Demo:* a Bevy scene on Vulkan and Metal with mutual occlusion between map and engine objects; a Slint app; a stereo render from simulated head poses.
- *Exit:* SharedDevice import on Vulkan, Metal and D3D12 against hosts on two wgpu majors; depth interop with and without reversed-Z; same-queue and semaphore sync tests; `HostDriven` polls within budget; §14.2's table verified for Bevy, Slint, Qt.

**M10 Accessibility host mappings.**
- *Builds:* §13's mappings, keyboard navigation and focus ring, system-setting mappings, contrast lints, scene-document `a11y` hints.
- *Demo:* VoiceOver, TalkBack and NVDA navigating POIs and roads in the reference app.
- *Exit:* expected-tree fixtures; automated tree checks (AccessKit harness, UiAutomator, XCUITest, axe-core in Playwright); reduced-motion and text-scaling fixtures.

**M11 Data layers.**
- *Builds:* `columnar` sources; GeoArrow ingest; LOD; cull reuse; shared placement; GPU aggregation and approximate placement (Modern); the Compat cap.
- *Demo:* a live 5M-point fleet layer with selection in a laptop browser and natively; 2M on a mid phone.
- *Exit:* §18 row 23; ingest copies match §10.4; data labels collide with basemap labels in fixtures; the Compat cap enforced and reported.

**M12 Stable contracts.** Public plugin ABI 1.0 with conformance; C ABI 1.0; scene document 1.0; bindings stabilized. *Exit:* real-world suite ≥95% within the yardstick on Modern; two independent plugins (a layer type and a source provider) pass conformance; semver and ABI diffs clean against the 1.0 baseline.

**After M12:** wasm plugins, ProjIR, expression-function plugins, XR packages, Godot/Unity/Unreal packages, 3D Tiles 2.0, point clouds and splats, curve text, GPU subdivision, a SAB build, COG/GeoParquet providers, OpenHarmony.

---

## 21. Technical risks and mitigations

| Risk | Sev. | Mitigation |
|---|---|---|
| Vertex pulling slow on Mali/PowerVR (padding, lost reuse) | High | measured from M0; meshes moved to compacted indices on Apple and Adreno data (decision 0002); Mali/PowerVR re-measure with `pnpm -C web bench --android` (#16); instanced quads for points and glyphs; tunable chunk size |
| wgpu's browser overhead breaks the payload budget | High | M0 measurement; web-sys HAL contingency; per-module gates; lazy Compat, shaper, 3D |
| Worker rendering adds visible gesture latency | High | ui role owns the camera; view annotations; `thread: "main"` with the same API; M3 rule |
| Web pipeline creation time | High | ≤40 pipelines; cull and first-visible first; fade-in; blob caches; M3 load gate |
| WebGPU field gaps (Xclipse, Firefox Linux/Android, WebViews, workers) | High | runtime probes; Compat; mid-session switch; capability events |
| Android driver quality (#4184, #4464, #4442) | High | allowlist; quirks; sentinel; Compat on GLES; nightly farm |
| v8 long tail (symbols, text metrics) | High | GL JS behaviour with golden vectors; frozen snapshots; real-world gate |
| Compat creep constraining Modern | Medium | §5.3 rule; LOC ceilings; declared subset; M4 "Modern unchanged" exit |
| ANGLE-as-system-driver erodes GLES fallback | Medium | allowlist is the primary Android defence; Compat justified mainly by the web |
| wgpu churn; runtime naga cost; binary size | Medium | ~8-week tracking; raw-handle integrations; N/N−1 adapters; passthrough shaders; feature gating; Metal contingency |
| GPU-written state is hard to debug | Medium | CPU reference cull/compact as oracle; indirect-arg readback in tests; frame debugger |
| Interpreter occupancy on low-end GPUs | Medium | bytecode class only; M1 measurement; cost model to CPU columns |
| wasm memory never shrinks | Medium | per-worker budgets; disposable ingest; worker recycling |
| ICU4X data size | Medium | M2 measurement; SE-Asian lazy; documented fallback to browser segmentation without the identity claim for complex-script wrapping |
| HarfRust maturity (0.14) | Medium | `Shaper` trait; symbol-hidden HarfBuzz C behind a feature natively |
| Engine sync gaps (Godot, Unreal) | Medium | same-queue ordering as common denominator; per-engine verification first |
| Scope creep into a 3D engine | Medium | non-goals; M8's declared list; splats and point clouds after M12 |
| Licensing mistakes when porting | Low | `THIRD_PARTY.md`; port checklist; BSD/MIT/Apache sources; no Mapbox post-fork code |

---

## 22. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| **Baseline algorithms shaped by WebGPU compat / GLES 3.0 / WebGL2 limits** | no vertex-stage storage, compute only as an accelerator, per-tile direct draws as the reference algorithm: the main path pays for a shrinking minority (§4.1), which a separate reduced tier now serves |
| **Web last or conditional** | the web is the strictest environment; deferring it lets blocking waits, thread assumptions and large dependencies into the core |
| **Headless any-CRS renderer as the first target** | interactive portability shapes the core from M0; any-CRS reuses M0 hooks at M6 |
| **No fallback at all** | iOS < 26 browsers, Firefox Linux/Android, non-allowlisted Android devices and WebViews without WebGPU would get nothing; Compat is one executor plus CPU kernels |
| **WebGPU compatibility mode as its own tier** | a third render configuration for devices that also expose WebGL2 |
| **Per-tile bundles as primary submission; requiring MDI, bindless or mesh shaders** | ≈ tiles × layers draws; not in core WebGPU or portable via wgpu (M+ only) |
| **GPU collision by default** | non-deterministic labels; breaks 448/414 symbol fixtures and tile-edge agreement |
| **Runtime WGSL per expression; an interpreter in every shader** | a translator in the bundle and a pipeline per expression; worst-case registers everywhere |
| **wgpu's GL backend; a web-sys HAL by default; TS command-stream executors** | +1.7 MB raw in the browser; a second Modern HAL (kept as contingency); existed only for raw-GL custom layers |
| **SAB threads, JSPI or Memory64 by default; main-thread scene as the fixed choice** | not embeddable on arbitrary pages and missing in Safari; GL JS's main-thread load (#8426) |
| **Recording into a host's command buffer; one pinned wgpu major; CPU copies for device loss** | wgpu cannot wrap foreign command buffers; wgpu, Bevy and Slint move every 3–4 months; doubles mobile memory |
| **Keys in v8; a "ground" zoom default; a new language, v9-first or v8-exact internal model; `Intl.Segmenter` for line breaks** | §11.1; breaks every author's model of zoom; no line granularity |
| Fork or wasm-compile Native; fork GL JS; Bevy, three.js or Cesium as a base; C++ + Dawn; Zig, TS-everywhere or KMP core | thread-affine core, 4 shader dialects, Emscripten build (2.21 MB br, COOP/COEP, JSPI, 512 MB heap); no native story; forked-graph churn (#311), weak 2D on three.js, Cesium's own styling; "5 MB extra size" (#317); missing GPU/text crates, threads or shared memory (ffi#37) [verified] |
| vello-style paths; depth as layer order; a JS VM; sphere-only globe; Mercator-only grid; SQLite offline; UniFFI; Diplomat's JNA Kotlin; per-property codegen; a fifth MLT decoder; cesium-native-only 3D Tiles | unproven at map scale; blocks globe clipping and depth sharing; the Tangram-ES lesson; wrong for 3D Tiles and XR; no poles or national grids; PMTiles is simpler; hot-path serialization; 90–170× JNI call cost and API 33 (decision 0018); JSON setters suffice; `mlt-core` exists; no wasm |

---

## 23. Open questions

1. Where the project lives (MapLibre org, maplibre-rs, or standalone).
2. wgpu's browser overhead in a size-optimized build, and whether wgpu exposes the `GPUBuffer` the web upload path needs (M0).
3. WebGPU in dedicated workers on Safari 26 and Firefox; core WebGPU in Android WebView and WKWebView embeddings (M0, M3).
4. Compacted indices and vertex pulling on Mali and PowerVR (#16); Adreno 619 and Apple M6 measured (decision 0002). M+ multi-draw unmeasured.
5. WGSL composition: naga_oil-style vs WESL (M0).
6. `subgroups` and `shader-f16` on Safari 26 and Firefox; `WorkerGlobalScope.fonts` coverage (decides cluster-raster fallbacks to main).
7. wgpu passthrough shaders stabilizing (removes runtime naga natively).
8. ~~Diplomat vs the shared emitters (M3).~~ Diplomat's front end with Orbis-owned Swift and Kotlin/JNI emitters (decision 0018).
9. ICU4X segmentation data size in wasm (M2).
10. How fast the Android allowlist can grow, and whether Compat on Android stays worth its test cost as ANGLE-on-Vulkan spreads.
11. Whether async GPU-projected collision boxes beat CPU projection for pitched/globe placement without visible latency.
12. Martin as a source of WOFF2 font subsets (real shaping without client font configuration) and KTX2 rasters.

---

## 24. DIAGRAM DATA

### DD1. Architecture (static)

**Nodes** (group: ids): Inputs: `in_v8` v8 style, `in_scene` scene document, `in_builder` IR builder, `in_data` tiles/rasters/GeoArrow/glTF/3D Tiles. Hosts: `h_web` (package + ui role), `h_android`, `h_apple`, `h_desktop` (winit), `h_engine` (M9), `h_server` (M6). External UI: `w_wrappers` (maplibre-compose, swiftui-dsl, Flutter, Qt). Bindings: `b_supported` (Rust, C, Kotlin, Swift, TS), `b_experimental` (C++, Dart, .NET through Diplomat's own backends). API: `api`, `capi` (Diplomat bridge), `web` (roles, payload table), `emitters` (`diplomat-swift-kotlin`), external `e_diplomat`. Runtime: `rt`, `jobs`, `a11y`, `host`, `plugin`, `batteries`. Style: `schema`, `v8c`, `style`, `expr`. World: `geo`, `crs` (Surface, TMS, planner), `cam`, `inter`. Jobs (share-nothing): `src`, `tile`, `layout`, `text`, `sym`, `m3d`. GPU: `scene` (tables, arenas), `render_m` (Modern: cull/compact, indirect, bundles), `render_c` (Compat: CPU cull, direct draws, data textures), `shaders`, `hal` (caps M/M+/C), `hal_wgpu` (WebGPU, Vulkan, Metal, D3D12), `hal_gl` (WebGL2, GLES 3.0, GL 3.3). External: `c_spec`, `c_mlt`, `c_pm`, `c_glyph`, `c_fixtures`, `e_gljs`, `e_native`, `e_ffi`, `e_diplomat`.

**Edges** (from → to "label"; ⇢ dashed):
- in_v8, in_scene → v8c "compile / merge"; in_builder → style; in_data → src "TileKey (TMS)"; schema → v8c; v8c → style "IR"; style → expr "compile once"
- api → capi, web; capi → emitters → b_supported "generated"; capi ⇢ b_experimental "Diplomat backends"; e_diplomat ⇢ capi, emitters "macro, HIR, C backend"
- h_web → web; h_android, h_apple, b_supported → capi; h_desktop, h_engine, h_server → api; w_wrappers ⇢ h_android, h_apple, h_web "wrapper-owned adapter"; h_engine → hal_wgpu "raw handles, own command buffers, semaphore"
- capi, web → rt "commands → generations"; rt → jobs, style, cam, src, scene, a11y
- inter → cam → crs → geo; crs → src "cover"; crs → layout "subdivision"; crs → scene "row params"; crs → shaders "projection WGSL"
- src → jobs → tile → layout → text → sym "per-tile DAG (sliced on web)"; layout, m3d → scene "streams, chunks, tables (one copy on web)"; sym → scene "state diffs"; sym → a11y "snapshot"
- scene → render_m "GPU tables"; scene → render_c "data-texture mirrors"; render_m → hal "dispatch + indirect + bundles"; render_c → hal "direct draws"; shaders → hal_wgpu "WGSL"; shaders → hal_gl "GLSL ES 3.00"; hal → hal_wgpu, hal_gl
- plugin → src, layout, scene, crs; batteries → plugin; host → host-native, host-web; a11y → h_web, h_android, h_apple, h_desktop "platform trees (M10)"
- c_spec → v8c, schema; c_mlt, c_pm → tile; c_glyph → text; c_fixtures ⇢ rt "read-only mirror"; e_gljs ⇢ sym, crs, render_c "ported (BSD-3)"; e_native ⇢ shaders, hal "WGSL, quirks (BSD-2)"; e_ffi ⇢ capi "execution model"

**Styling:** scene-side blue, job-side green, GPU orange (Modern solid; Compat hatched inside a dashed box "Compat: may not constrain Modern"), API grey, wrappers outside Orbis, externals in their own band. Boundaries: language/ABI at `capi`/`web`, threads between scene and jobs, GPU at `hal`.

### DD2. Thread lanes

Lanes and modes: the §7.1 roles table (Owned, HostDriven, Web; Inline omitted). Channels: [H]→[S] commands with generations (web: ~200 B camera per rAF); [S]→[H] snapshots, frame acks, a11y deltas, elevation pyramid (M7); [S]↔[J] owned inputs/results (one copy + transfer on the web); [S]↔[P] camera in, state diffs out; [S]→GPU rows, one dispatch, bundle replay (Compat: CPU cull → direct draws); engine ↔ Orbis queue order or semaphore.

### DD3. Flow A: uncached tile → pixels (web, Modern; animate)

Lanes MAIN [H+ui] · RENDER-W [S] · JOB-W k [J] · PLACE-W [P] · GPU; steps §6.1 1–12. Annotations: 1 "~200 B camera"; 3 "≤4 ms slices"; 4 "≤256-tri chunks, 24–32 B records"; 6 "one copy, then transfer"; 8 "writeBuffer from ArrayBuffer; candidate lists"; 10 "1 dispatch + bundle replay; ≤0.2 ms diff"; 12 "annotations in-frame; DOM ±1 frame". Side bars: "4 messages (+2/glyph batch) vs 14–16 hops on Native Android"; "+1 frame vs main mode; M3 decides".

### DD4. Flow B: camera-only frame (flame chart)

Bars from §6.2: [H] 0–0.05 input; [S] 0–0.10 camera + cover; [S] 0.10–0.15 rows; [S] 0.15–0.25 dispatch + bundle + submit (hatched "measured M0–M2"); [GPU] 0.25–0.35 cull/compact; [GPU] 0.35–5.0 layers → symbols; [P] parallel placement band. Comparison track "Native, Pixel 7 Pro, Americana, serialized": `createRenderTree` 0–2.4 → bucket update/reprojection 2.4–3.4 → upload 3.4–4.0 → `renderLayer.update` 4.0–5.2 → tweakers ×~400 5.2–6.7 → layer-group upload 6.7–7.0 → encode 7.0–9.8 → GPU wait 9.8–17.0 (split inferred from verified totals). Track "Orbis Compat [inferred]": rows 0–0.15 → CPU cull 0.15–0.45 → direct draws 0.45–1.5 → GPU.

### DD5. Flow C: engine/XR stereo frame (M9)

Lanes: game thread · render thread · Orbis command buffers on the shared device · jobs · placement · host GPU queue; steps §6.4 1–4. Callout: "never records into the host's command buffer". Side table: §14.2 engine table.

### DD6. Build path (no dates)

Nodes M0–M12 plus "After M12", in a core band (M0–M2) that splits into two parallel swimlanes: rendering (M6, M7, M8, M11) and platform (M3, M4, M5, M9, M10); M12 joins them. Edges: M0 → M1 → M2; M2 → M3 → M4 → M5; M2 → M6 → M7 → M8; M2 → M11; M4 → M9; M3 → M10; M3 ⇢ M6, M11 "gestures, hosts"; M4 ⇢ M6, M7, M8, M11 "Compat criteria"; M8 ⇢ M9 "ECEF content"; M5, M8, M9, M10, M11 → M12 → After. Label each node with its demo and key gate (M0 "web/native identity, 1 dispatch per frame"; M3 "thread default decided"; M4 "Modern unchanged"; M5 "suite ≥90%, ≤550 KB br"; M6 "CRS suite green"). Overlay "platforms green": web WebGPU + desktop from M0; Android Vulkan + iOS from M3; WebGL2 + Android GLES from M4; servers from M6; engines from M9.

### DD7. Charts

- **Web payload, KB br (bar):** GL JS 247 · maplibre-rs WebGPU 401 · Orbis M0 slice (measured; ≤300 hypothesis) · Orbis target 450 / ceiling 550 · maplibre-rs WebGL 846 · Native-on-wasm 2,211; lazy: Compat +60–90, HarfRust +100–150, SE-Asian 591 (measured), 3D +40–80.
- **Draws per frame, Liberty Tokyo z15 (bar):** GL JS 449.6 · Orbis Modern ~60–90 (1 dispatch, 1 bundle) · Compat with multi-draw ~60–90 · Compat without ~300–450.
- **Tier coverage (heatmap; M / C / unsupported):** Chrome desktop M; Chrome Linux (Intel Gen12+, NVIDIA/Wayland) M, other Linux C; Chrome Android 12+ M, older C; Safari 26 M; Safari 16.4–18 C; Firefox Windows/macOS M, Linux/Android C; Android native allowlisted Vulkan M, other API 24+ C, < 24 unsupported; iOS 16+, macOS 13+, Windows 10+ M; Linux Vulkan M, GL-only C. Sources: gpuweb wiki [verified]; Android Vulkan ≥1.1 ~87% [verified-2nd].

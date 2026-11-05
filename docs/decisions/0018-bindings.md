# 0018: Bindings: a Diplomat front end and our own Swift and Kotlin/JNI emitters

M3 builds the C ABI and generated Kotlin and Swift bindings. Plan D18 said
the C ABI follows maplibre-native-ffi's conventions and the bindings come
from the FFI's emitters, given a Rust-annotation front end. Plan §14.1 left
Diplomat to be evaluated at M3. This records that evaluation, the
alternative that was measured against it, and the owner's decision, which
changes D18.

## What was measured

**Diplomat 0.17** (the `#[diplomat::bridge]` macro, `diplomat_core`'s HIR and
`diplomat-tool`'s backends). The prototype exposed a slice of the `Map` API:
a runtime with an offscreen device, map creation, a URL or JSON style, an
edit transaction that returns its generation, the camera as a getter and a
setter, pan, rendering into an owned texture with pixel read-back, and an
event drained by polling. That is 19 functions, 4 structs, 2 enums and 3
opaque types.

Generated code for that slice:

| Backend | Files | Lines | Bytes |
|---|---:|---:|---:|
| C | 17 | 489 | 8,966 |
| C++ | 17 | 1,658 | 53,133 |
| Kotlin (JNA) | 9 | 1,997 | 56,377 |
| JS/TS (wasm) | 21 | 2,102 | 63,851 |
| .NET | 30 | 1,891 | 53,541 |
| Swift | none: no backend | | |

The C header is direct: `OrbMap* OrbMap_create(const OrbRuntime*)`, results
as `{ union { T ok; E err; }; bool is_ok; }`, strings as
`DiplomatStringView`, output strings through `DiplomatWrite`. The
`diplomat-runtime` glue adds 9 exported functions (allocation, write
buffers); the release dylib of the prototype, the whole engine with Metal,
is 8,075,776 bytes.

Call overhead on an Apple M-series Mac with JVM 25, one run each after a
warm-up. A trivial getter
(`generation`, `u64`) and a 40-byte struct return (`camera`):

| Caller | `generation` | `camera` |
|---|---:|---:|
| C, direct | 0.77 ns | 5.65 ns |
| Swift, direct over the C header | 1.11 ns | 4.53 ns |
| Kotlin through Diplomat's JNA binding | 368.8 ns | 1,333.9 ns |
| Kotlin through a JNI shim | 4.1 ns | 7.7 ns |
| Kotlin through FFM (`java.lang.foreign`) | 4.6 ns | |

Other findings:

- **No Swift backend.** Swift can import the C header through a module map,
  but that is the C API, not a binding.
- **Kotlin is JVM-only JNA.** That is 90–170× a JNI call. The generated code
  frees objects through `java.lang.ref.Cleaner`, which needs Android API 33;
  Orbis's floor is API 24 (D9). JNA adds 2.0 MB of jar, or a 522 KB AAR with
  a 177 KB arm64 `libjnidispatch.so`.
- **The JS backend** targets the C ABI on wasm (spec ABI, or nightly Rust).
  The web package stays on wasm-bindgen (`orb-web`).
- **Annotations.** There are no execution or threading categories. Mutable
  access to an opaque needs `#[diplomat::opaque_mut]`. Kotlin and JS reject
  `&mut [u8]` out-parameters, and owned slices can't be returned, so pixels
  come back as a slice borrowed from the texture object.
- **C naming.** Pure C uses Rust type names verbatim. `abi_rename` prefixes
  symbols only, so types carry the prefix themselves (`OrbMap`).
- **Friction.** `cargo install diplomat-tool` takes 55 s. Generation takes
  under 0.1 s. In 0.17 a lowering error panics while printing its report
  ("Could not read source file src") rather than naming the problem.
  Bisection found the causes above.

**The FFI's emitters** (maplibre-native-ffi `tools/bindgen`, public, BSD-2).
They are 32,467 lines of Python over libclang. They render the FFI's 22
headers into 408 files, 121,745 lines in 9 languages, in 2.6 s (warm `uv`).
Its Kotlin binding is 34,585 generated and 12,542 handwritten lines (KMP:
JVM FFM, an Android JNI shim, K/N cinterop, wasm); Swift is 15,399 and 4,381.
The emitters name the FFI's protocol types (`mln_status`, `mln_diagnostic`,
completions, buffer views, delivery queues; about 430 references in about 50
files) and write into that repository's layout. Reusing them would mean
generalizing them upstream or vendoring a copy, and adding Python and
libclang to Orbis's toolchain.

## Decision

The owner chose **Diplomat's front end with Orbis-owned emitters**:

- **`orb-capi`** is a `#[diplomat::bridge]` module over `orb-api`, the safe
  Rust embedding API. It is the annotated API. Its C header comes from
  Diplomat's C backend, run by `cargo xtask capi`.
- **`diplomat-swift-kotlin`** is a new crate with Swift and Kotlin/JNI
  emitters over `diplomat_core`'s HIR. It is written in Rust, has no Orbis
  names in it, and takes its configuration (library name, package, module)
  from the caller. It has its own golden and compile-and-run tests, so it
  could be published or offered to Diplomat later. Kotlin calls through a
  generated JNI C shim, which runs on Android API 24+ with no JNA and no
  `Cleaner`. Swift calls the C header directly through a module map.
- **Contracts.** The execution categories and threading go on items as
  Diplomat backend attributes under a reserved name, `contract`, which no
  code generator satisfies. Diplomat's lowering skips them, and the emitters
  read them from the AST: `#[diplomat::attr(contract, command)]` on a method,
  and `#[diplomat::attr(contract, thread = "owner")]` on a type. Ownership is
  Diplomat's own model: `Box` is owned, references are borrowed, and
  lifetimes tie a borrow to its owner.

What this changes from D18 and plan §14.1:

- **C conventions are Diplomat's.** Handles are opaque pointers with
  `*_destroy`, not generational handles. Errors are result structs carrying
  an `OrbError` status enum, with the last message kept per thread. Structs
  are plain by-value structs, not `size`-prefixed. The C ABI is 0.x and may
  break until M12, and extensible structs are an M12 question.
- **No shared emitter set with the FFI.** Orbis owns two emitters; the
  FFI's stay where they are.
- **Only C, Kotlin (Android/JVM through JNI) and Swift are supported.** The
  plan's experimental languages (Dart, .NET, Go, Python, Zig) are dropped
  from M3. Diplomat's own C++, Dart and .NET backends exist if they're wanted
  later.

## Measured on the built version

`diplomat-swift-kotlin` is 2,379 lines of Rust, plus a 224-line fixture
bridge, golden files and Swift and Kotlin run tests. For `orb-capi`'s bridge
(59 C functions, destructors included, on 5 opaque types, 7 structs and 9
enums), `cargo xtask bindings` writes:

| Output | Files | Lines | Bytes |
|---|---:|---:|---:|
| C headers (Diplomat) and `orbis.h` | 44 | 1,232 | 21,757 |
| Kotlin | 23 | 1,221 | 33,701 |
| JNI shim (C) | 1 | 556 | 24,651 |
| Swift | 23 | 1,073 | 31,265 |

A trivial getter (`i64`) through the generated bindings, with each
binding's owner-thread check, Apple M-series:

| Caller | ns per call |
|---|---:|
| Swift class over the C header | 7.38 |
| Kotlin over the JNI shim, JVM 25 | 6.12 |
| Kotlin through Diplomat's JNA (prototype above) | 368.8 |

The C library (`cargo xtask capi`) builds a static library, headers, a
pkg-config file and a CMake package. The Swift and Kotlin run tests exercise
every supported shape, callbacks from another thread included.

The binding scenario (style, pixels, hit test, rendered-features query, edit
generation, click, idle, projection and errors) runs through Rust
(`crates/orb-api/tests`). C (`crates/orb-capi/tests/smoke.c`, through
pkg-config and CMake), Kotlin on the JVM over the JNI shim
(`hosts/android/orbis/src/jvmSmoke`) and Swift (`hosts/apple/Tests`) run
smoke tests, all from `cargo xtask bindings --test`. The
Android and Apple hosts are written against the generated bindings and
call nothing else of Orbis. The one exception is the JNI glue that turns a
`Surface` into an `ANativeWindow`, which only JNI can do.

Known gaps in the emitters: no `Option` of values, no borrowed opaque
returns, no callbacks with arguments, and no `Cleaner`-style release for
owner-thread objects, which must be closed on their thread.
`diplomat-tool` 0.17 passes the entry's directory as the source file when
reporting lowering errors, so its reports panic; the emitters lower with
the file and report errors as text.

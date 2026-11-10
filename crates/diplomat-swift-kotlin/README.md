# diplomat-swift-kotlin

Swift and Kotlin/JNI bindings for
[Diplomat](https://github.com/rust-diplomat/diplomat) bridges. Diplomat's macro
exports a `#[diplomat::bridge]` module as C functions and its C backend writes
their header. This crate reads the same bridge through `diplomat_core`'s HIR and
writes:

- **Swift**: classes and values over the C header, imported as a Clang module.
  Objects free themselves on `deinit`, fallible calls `throw`.
- **Kotlin**: classes over a generated JNI shim in C. No JNA, reflection or
  `java.lang.ref.Cleaner`, so it runs on Android API 21+ and any JVM. Objects
  are `AutoCloseable`.

The crate names no particular library: `KotlinConfig` and `SwiftConfig` carry
the package, library and module names. The crate docs (`src/lib.rs`) list the
supported shapes, and the contracts: thread rules, execution categories and
error messages, written as `#[diplomat::attr(contract, …)]` attributes, which
Diplomat's code generators skip.

```rust
let files = diplomat_swift_kotlin::kotlin(Path::new("src/lib.rs"), &KotlinConfig {
    package: "com.example".into(),
    library: "example".into(),        // System.loadLibrary
    native_object: "ExampleNative".into(),
    c_headers: vec!["example.h".into()],
})?;
```

Tests: `fixture/` is a bridge with every supported shape. `tests/golden.rs`
compares its bindings with `tests/golden` (`UPDATE_GOLDEN=1` rewrites them).
`tests/run.rs` builds the fixture, compiles Swift and Kotlin programs against it
and runs them (ignored by default; they need `swiftc`, or a JDK and `kotlinc`).

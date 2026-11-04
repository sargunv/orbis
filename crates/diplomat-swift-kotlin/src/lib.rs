//! Swift and Kotlin/JNI bindings for [Diplomat](https://github.com/rust-diplomat/diplomat)
//! bridges.
//!
//! Diplomat's macro turns a `#[diplomat::bridge]` module into `extern "C"`
//! functions, and its C backend writes their header. This crate reads the
//! same bridge through `diplomat_core`'s HIR and writes:
//!
//! - **Swift**: one file per type over the C header, imported as a Clang
//!   module. Opaque types are classes that free themselves on `deinit`
//!   (owner-thread ones trap if released on another thread); structs and
//!   enums are values; fallible calls `throw`.
//! - **Kotlin**: one file per type, calling a generated JNI shim in C that
//!   calls the C functions. No JNA, no reflection, no
//!   `java.lang.ref.Cleaner`: it runs on Android API 21+ and any JVM.
//!   Opaque types are `AutoCloseable`; calls on objects any thread may use
//!   hold the object's monitor, as `close` does, so none is freed under a
//!   call. Threads the library calls back on are attached to the JVM for
//!   the call and detached after.
//!
//! **Contracts.** Bridges may say how each item is used, with Diplomat
//! backend attributes under the name `contract`, which no code generator
//! satisfies, so Diplomat's lowering skips them:
//!
//! - on a type, `#[diplomat::attr(contract, thread = "owner")]`: only the
//!   thread that made an object may use it (checked on every call), or
//!   `thread = "any"` (the default for structs and enums; opaque types
//!   default to `owner`);
//! - on a method, an execution category: `command`, `query`, `snapshot`,
//!   `event_batch`, `render_driver`, `lifecycle` or `immediate`, shown in
//!   the generated docs;
//! - on a static method that writes text, `error_message`: it writes the
//!   calling thread's last error, and the bindings put it in the errors
//!   they throw.
//!
//! **Supported bridges.** Opaque types (owned `Box` returns, nullable
//! `Option<Box<T>>` returns, `&T` and `&mut T` parameters), structs of
//! scalars, enums and structs, enums, scalars, UTF-8 strings in, text out
//! through `DiplomatWrite`, borrowed scalar slices in and out (copied),
//! `Result<T, E>` with enum or unit errors, and `impl Fn() + 'static`
//! callbacks the callee may keep and call from any thread. Anything else
//! is reported as an error rather than generated.
//!
//! Nothing here names a particular library: [`KotlinConfig`] and
//! [`SwiftConfig`] carry the package, library and module names.

#![forbid(unsafe_code)]

mod kotlin;
mod model;
mod names;
mod swift;

use std::collections::BTreeMap;
use std::path::Path;

/// Where the Kotlin bindings live and what they load.
#[derive(Clone, Debug)]
pub struct KotlinConfig {
    /// The Kotlin package, such as `com.example.lib`.
    pub package: String,
    /// The native library `System.loadLibrary` loads: the JNI shim linked
    /// with the Rust library.
    pub library: String,
    /// The internal object holding the `external` functions.
    pub native_object: String,
    /// The C headers the JNI shim includes, as `#include` names.
    pub c_headers: Vec<String>,
}

/// How the Swift bindings import the C header.
#[derive(Clone, Debug)]
pub struct SwiftConfig {
    /// The Clang module of Diplomat's C headers.
    pub c_module: String,
}

/// Generated files by path relative to the output directory.
pub type Files = BTreeMap<String, String>;

/// Kotlin sources under `kotlin/<package path>/`, and the JNI shim at
/// `jni/<library>_jni.c`.
pub fn kotlin(entry: &Path, config: &KotlinConfig) -> Result<Files, Vec<String>> {
    let api = model::lower(entry, "kotlin")?;
    kotlin::emit(&api, config)
}

/// Swift sources, one per type.
pub fn swift(entry: &Path, config: &SwiftConfig) -> Result<Files, Vec<String>> {
    let api = model::lower(entry, "swift")?;
    swift::emit(&api, config)
}

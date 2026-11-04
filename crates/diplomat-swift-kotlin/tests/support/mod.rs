//! The fixture bridge and its configuration, shared by the tests.

use std::path::{Path, PathBuf};

use diplomat_swift_kotlin::{Files, KotlinConfig, SwiftConfig};

pub(crate) fn entry() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixture/src/lib.rs")
}

pub(crate) fn kotlin_config() -> KotlinConfig {
    KotlinConfig {
        package: "dev.fixture".into(),
        library: "dsk_fixture_jni".into(),
        native_object: "FixtureNative".into(),
        c_headers: vec!["fixture.h".into()],
    }
}

pub(crate) fn swift_config() -> SwiftConfig {
    SwiftConfig {
        c_module: "FixtureC".into(),
    }
}

/// Everything generated from the fixture, by path under `kotlin/`,
/// `jni/` and `swift/`.
pub(crate) fn generate() -> Files {
    let mut files = diplomat_swift_kotlin::kotlin(&entry(), &kotlin_config()).unwrap();
    for (path, text) in diplomat_swift_kotlin::swift(&entry(), &swift_config()).unwrap() {
        files.insert(format!("swift/{path}"), text);
    }
    files
}

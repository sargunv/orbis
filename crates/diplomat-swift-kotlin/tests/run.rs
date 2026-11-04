//! Builds the fixture bridge, generates its bindings, compiles a program
//! in each language against them and runs it. Ignored by default: they
//! need `swiftc`, or a JDK and `kotlinc` (`KOTLINC`, `JAVA_HOME`);
//! `cargo xtask bindings test` runs them.

mod support;

use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command) -> String {
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("{command:?}: {e}"));
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{command:?} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

/// A fresh directory with the fixture's C headers, a Clang module map and
/// the fixture's static library.
fn prepare(name: &str) -> (PathBuf, PathBuf) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("../../target/dsk-run").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // A target directory of its own, so this doesn't wait on the one
    // `cargo test` holds.
    let target = root.join("../../target/dsk-run/target");
    run(Command::new(env!("CARGO"))
        .args(["build", "-q", "-p", "dsk-fixture"])
        .env("CARGO_TARGET_DIR", &target));
    let c = dir.join("c");
    diplomat_tool::r#gen(
        &support::entry(),
        "c",
        &c,
        &Default::default(),
        Default::default(),
        true,
    )
    .unwrap();
    let mut headers: Vec<String> = std::fs::read_dir(&c)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".h") && !n.ends_with(".d.h"))
        .collect();
    headers.sort();
    let umbrella: String = headers
        .iter()
        .map(|h| format!("#include \"{h}\"\n"))
        .collect();
    std::fs::write(c.join("fixture.h"), umbrella).unwrap();
    std::fs::write(
        c.join("module.modulemap"),
        "module FixtureC {\n    header \"fixture.h\"\n    export *\n}\n",
    )
    .unwrap();
    let files = support::generate();
    for (path, text) in &files {
        let path = dir.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    (dir, target.join("debug/libdsk_fixture.a"))
}

#[test]
#[ignore = "needs swiftc"]
fn swift_bindings_run() {
    let (dir, lib) = prepare("swift");
    let swift = dir.join("swift");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/run/main.swift"),
        swift.join("main.swift"),
    )
    .unwrap();
    let sources: Vec<PathBuf> = std::fs::read_dir(&swift)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    let exe = dir.join("fixture-swift");
    run(Command::new("swiftc")
        .args(["-O", "-module-name", "FixtureSwift", "-I"])
        .arg(dir.join("c"))
        .args(&sources)
        .arg(&lib)
        .arg("-o")
        .arg(&exe));
    let out = run(&mut Command::new(&exe));
    print!("{out}");
    assert!(out.contains("swift ok"));
}

#[test]
#[ignore = "needs a JDK and kotlinc"]
fn kotlin_bindings_run() {
    let (dir, lib) = prepare("kotlin");
    let java_home = std::env::var_os("JAVA_HOME")
        .map(PathBuf::from)
        .expect("JAVA_HOME");
    let platform = if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    };
    let shim = dir.join(if cfg!(target_os = "macos") {
        "libdsk_fixture_jni.dylib"
    } else {
        "libdsk_fixture_jni.so"
    });
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
    run(Command::new(cc)
        .args(["-shared", "-fPIC", "-O2", "-Wall", "-Werror", "-I"])
        .arg(java_home.join("include"))
        .arg("-I")
        .arg(java_home.join("include").join(platform))
        .arg("-I")
        .arg(dir.join("c"))
        .arg(dir.join("jni/dsk_fixture_jni_jni.c"))
        .arg(&lib)
        .arg("-o")
        .arg(&shim));
    let mut sources = Vec::new();
    collect_kt(&dir.join("kotlin"), &mut sources);
    sources.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/run/Main.kt"));
    let jar = dir.join("fixture.jar");
    let kotlinc = std::env::var("KOTLINC").unwrap_or_else(|_| "kotlinc".into());
    run(Command::new(kotlinc)
        .args(["-nowarn", "-include-runtime", "-d"])
        .arg(&jar)
        .args(&sources));
    let out = run(Command::new(java_home.join("bin/java"))
        .arg(format!("-Djava.library.path={}", dir.display()))
        .arg("-cp")
        .arg(&jar)
        .arg("MainKt"));
    print!("{out}");
    assert!(out.contains("kotlin ok"));
}

fn collect_kt(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_kt(&path, out);
        } else if path.extension().is_some_and(|e| e == "kt") {
            out.push(path);
        }
    }
}

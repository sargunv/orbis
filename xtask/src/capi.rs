//! `cargo xtask capi [--test]`: the C library for this host (plan §17), in
//! `target/capi`:
//!
//! - `lib/liborbis.a`, built with the `dist` profile;
//! - `include/orbis/`, Diplomat's headers and the `orbis.h` umbrella;
//! - `lib/pkgconfig/orbis.pc` and `lib/cmake/orbis/orbisConfig.cmake`
//!   (`orbis::orbis`), with the system libraries rustc reports.
//!
//! `--test` compiles `crates/orb-capi/tests/smoke.c` once through
//! pkg-config and once through CMake, and runs both; then runs the Kotlin
//! binding scenario on the host's JVM, over the generated JNI shim linked
//! with the same library (a JDK, `JAVA_HOME` and `kotlinc`, from mise).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::bindings::{C_DIR, root};

const PREFIX: &str = "target/capi";

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let test = match args {
        [] => false,
        [flag] if flag == "--test" => true,
        _ => return Err("capi: usage: cargo xtask capi [--test]".into()),
    };
    let prefix = build()?;
    if test {
        smoke(&prefix)?;
    }
    Ok(())
}

fn build() -> Result<PathBuf, String> {
    let root = root();
    eprintln!("$ cargo rustc -p orb-capi --profile dist --lib --crate-type staticlib");
    let output = Command::new(env!("CARGO"))
        .current_dir(&root)
        .args([
            "rustc",
            "-p",
            "orb-capi",
            "--profile",
            "dist",
            "--lib",
            "--crate-type",
            "staticlib",
            "--",
            "--print",
            "native-static-libs",
        ])
        .output()
        .map_err(|e| format!("cargo: {e}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!("building liborbis.a failed:\n{stderr}"));
    }
    // rustc prints the libraries once per build that links; an up-to-date
    // build doesn't, so the last list is kept beside the library.
    let prefix = root.join(PREFIX);
    let libs_file = prefix.join("native-static-libs");
    let native = match stderr
        .lines()
        .find_map(|l| l.split_once("native-static-libs: ").map(|(_, libs)| libs))
    {
        Some(libs) => link_items(libs).join(" "),
        None => std::fs::read_to_string(&libs_file).map_err(|_| {
            "rustc didn't report native-static-libs; `cargo clean -p orb-capi --profile dist` and retry"
                .to_owned()
        })?,
    };
    let lib = prefix.join("lib");
    let include = prefix.join("include/orbis");
    let _ = std::fs::remove_dir_all(&include);
    for dir in [
        &lib,
        &include,
        &lib.join("pkgconfig"),
        &lib.join("cmake/orbis"),
    ] {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&libs_file, &native).map_err(|e| e.to_string())?;
    let built = root.join("target/dist/liborbis.a");
    std::fs::copy(&built, lib.join("liborbis.a"))
        .map_err(|e| format!("{}: {e}", built.display()))?;
    for header in std::fs::read_dir(root.join(C_DIR)).map_err(|e| e.to_string())? {
        let header = header.map_err(|e| e.to_string())?;
        std::fs::copy(header.path(), include.join(header.file_name()))
            .map_err(|e| e.to_string())?;
    }
    std::fs::write(
        lib.join("pkgconfig/orbis.pc"),
        format!(
            "prefix=${{pcfiledir}}/../..\nincludedir=${{prefix}}/include\nlibdir=${{prefix}}/lib\n\nName: orbis\nDescription: The Orbis map engine's C ABI (0.x)\nVersion: {}\nCflags: -I${{includedir}}/orbis\nLibs: -L${{libdir}} -lorbis {native}\n",
            env!("CARGO_PKG_VERSION")
        ),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        lib.join("cmake/orbis/orbisConfig.cmake"),
        format!(
            "# The Orbis C ABI (0.x) as the imported target orbis::orbis.\nget_filename_component(_orbis_prefix \"${{CMAKE_CURRENT_LIST_DIR}}/../../..\" ABSOLUTE)\nif(NOT TARGET orbis::orbis)\n  add_library(orbis::orbis STATIC IMPORTED)\n  set_target_properties(orbis::orbis PROPERTIES\n    IMPORTED_LOCATION \"${{_orbis_prefix}}/lib/liborbis.a\"\n    INTERFACE_INCLUDE_DIRECTORIES \"${{_orbis_prefix}}/include/orbis\"\n    INTERFACE_LINK_LIBRARIES \"{}\")\nendif()\nunset(_orbis_prefix)\n",
            cmake_libs(&native)
        ),
    )
    .map_err(|e| e.to_string())?;
    let size = std::fs::metadata(lib.join("liborbis.a"))
        .map_err(|e| e.to_string())?
        .len();
    eprintln!(
        "capi: {} ({:.1} MB liborbis.a); links {native}",
        prefix.display(),
        size as f64 / 1e6
    );
    Ok(prefix)
}

/// rustc's link flags without repeats; `-framework X` is one item.
fn link_items(native: &str) -> Vec<String> {
    let mut items: Vec<String> = Vec::new();
    let mut words = native.split_whitespace();
    while let Some(word) = words.next() {
        let item = match word {
            "-framework" => match words.next() {
                Some(name) => format!("-framework {name}"),
                None => continue,
            },
            _ => word.to_owned(),
        };
        if !items.contains(&item) {
            items.push(item);
        }
    }
    items
}

/// Link flags as a CMake list.
fn cmake_libs(native: &str) -> String {
    link_items(native).join(";")
}

fn smoke(prefix: &Path) -> Result<(), String> {
    let root = root();
    let source = root.join("crates/orb-capi/tests/smoke.c");
    let out = prefix.join("smoke");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;

    // pkg-config.
    let flags = Command::new("pkg-config")
        .env("PKG_CONFIG_PATH", prefix.join("lib/pkgconfig"))
        .args(["--cflags", "--libs", "orbis"])
        .output()
        .map_err(|e| format!("pkg-config: {e}"))?;
    if !flags.status.success() {
        return Err(String::from_utf8_lossy(&flags.stderr).into_owned());
    }
    let flags = String::from_utf8_lossy(&flags.stdout).into_owned();
    let exe = out.join("smoke-pkg-config");
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
    check(
        Command::new(&cc)
            .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"])
            .arg(&source)
            .args(flags.split_whitespace())
            .arg("-o")
            .arg(&exe),
    )?;
    check(&mut Command::new(&exe))?;

    // CMake.
    let project = out.join("cmake");
    std::fs::create_dir_all(&project).map_err(|e| e.to_string())?;
    std::fs::write(
        project.join("CMakeLists.txt"),
        format!(
            "cmake_minimum_required(VERSION 3.20)\nproject(orbis_smoke C)\nset(CMAKE_C_STANDARD 11)\nfind_package(orbis REQUIRED CONFIG)\nadd_executable(smoke \"{}\")\ntarget_link_libraries(smoke PRIVATE orbis::orbis)\n",
            source.display()
        ),
    )
    .map_err(|e| e.to_string())?;
    let build = project.join("build");
    check(
        Command::new("cmake")
            .arg("-S")
            .arg(&project)
            .arg("-B")
            .arg(&build)
            .arg(format!("-DCMAKE_PREFIX_PATH={}", prefix.display()))
            .arg("-DCMAKE_BUILD_TYPE=Release"),
    )?;
    check(Command::new("cmake").arg("--build").arg(&build))?;
    check(&mut Command::new(build.join("smoke")))?;
    eprintln!("capi: the smoke test passed through pkg-config and CMake");
    kotlin_scenario(prefix, &flags)
}

/// The Kotlin scenario on the JVM: the JNI shim and liborbis.a as a
/// shared library, the generated Kotlin and the scenario compiled with
/// kotlinc.
fn kotlin_scenario(prefix: &Path, link: &str) -> Result<(), String> {
    let root = root();
    let out = prefix.join("smoke/kotlin");
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let java_home = PathBuf::from(
        std::env::var_os("JAVA_HOME")
            .ok_or("capi: the Kotlin scenario needs JAVA_HOME (mise x --)")?,
    );
    let platform = if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    };
    let library = out.join(if cfg!(target_os = "macos") {
        "liborbis.dylib"
    } else {
        "liborbis.so"
    });
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
    check(
        Command::new(&cc)
            .args(["-shared", "-fPIC", "-O2", "-Wall", "-Werror", "-I"])
            .arg(java_home.join("include"))
            .arg("-I")
            .arg(java_home.join("include").join(platform))
            .arg(root.join("hosts/android/orbis/src/generated/jni/orbis_jni.c"))
            .args(link.split_whitespace())
            .arg("-o")
            .arg(&library),
    )?;
    let mut sources = Vec::new();
    kotlin_files(
        &root.join("hosts/android/orbis/src/generated/kotlin"),
        &mut sources,
    )?;
    sources.push(root.join("hosts/android/orbis/src/jvmScenario/kotlin/Scenario.kt"));
    let jar = out.join("scenario.jar");
    check(
        Command::new("kotlinc")
            .args(["-nowarn", "-include-runtime", "-d"])
            .arg(&jar)
            .args(&sources),
    )?;
    check(
        Command::new(java_home.join("bin/java"))
            .arg("--enable-native-access=ALL-UNNAMED")
            .arg(format!("-Djava.library.path={}", out.display()))
            .arg("-cp")
            .arg(&jar)
            .arg("ScenarioKt"),
    )?;
    eprintln!("capi: the Kotlin scenario passed on the JVM");
    Ok(())
}

fn kotlin_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            kotlin_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "kt") {
            out.push(path);
        }
    }
    Ok(())
}

fn check(command: &mut Command) -> Result<(), String> {
    eprintln!("$ {command:?}");
    let status = command.status().map_err(|e| format!("{command:?}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{command:?} failed"))
    }
}

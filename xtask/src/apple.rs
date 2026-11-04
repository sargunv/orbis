//! `cargo xtask apple [--test]`: `liborbis.a` for macOS, iOS and the iOS
//! simulator (arm64, `dist` profile) as `hosts/apple/OrbisC.xcframework`
//! with the C headers and a Clang module map, then the Swift package and
//! its macOS reference app with SwiftPM (plan §17: SwiftPM only packages).
//! `--test` also runs the package's Swift binding scenario.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::bindings::{C_DIR, root};

/// Rust target, SwiftPM's name for the platform, deployment target
/// variable and version.
const SLICES: &[(&str, &str, &str, &str)] = &[
    (
        "aarch64-apple-darwin",
        "macOS",
        "MACOSX_DEPLOYMENT_TARGET",
        "14.0",
    ),
    (
        "aarch64-apple-ios",
        "iOS",
        "IPHONEOS_DEPLOYMENT_TARGET",
        "16.0",
    ),
    (
        "aarch64-apple-ios-sim",
        "iOS Simulator",
        "IPHONEOS_DEPLOYMENT_TARGET",
        "16.0",
    ),
];

const PACKAGE: &str = "hosts/apple";

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let test = match args {
        [] => false,
        [flag] if flag == "--test" => true,
        _ => return Err("apple: usage: cargo xtask apple [--test]".into()),
    };
    let root = root();
    let headers = headers(&root)?;
    let mut libraries = Vec::new();
    for (target, platform, variable, version) in SLICES {
        let lib = static_lib(&root, target, variable, version)?;
        let size = std::fs::metadata(&lib).map_err(|e| e.to_string())?.len();
        eprintln!(
            "apple: {platform} ({target}) liborbis.a {:.1} MB",
            size as f64 / 1e6
        );
        libraries.push(lib);
    }
    let framework = root.join(PACKAGE).join("OrbisC.xcframework");
    let _ = std::fs::remove_dir_all(&framework);
    let mut command = Command::new("xcodebuild");
    command.arg("-create-xcframework");
    for lib in &libraries {
        command
            .arg("-library")
            .arg(lib)
            .arg("-headers")
            .arg(&headers);
    }
    command.arg("-output").arg(&framework);
    check(&mut command)?;

    let package = root.join(PACKAGE);
    check(
        Command::new("swift")
            .args([
                "build",
                "-c",
                "release",
                "--product",
                "OrbisDemo",
                "--package-path",
            ])
            .arg(&package),
    )?;
    let demo = package.join(".build/release/OrbisDemo");
    let size = std::fs::metadata(&demo).map_err(|e| e.to_string())?.len();
    eprintln!(
        "apple: {} ({:.1} MB, macOS arm64)",
        demo.display(),
        size as f64 / 1e6
    );
    if test {
        check(
            Command::new("swift")
                .arg("test")
                .arg("--package-path")
                .arg(&package),
        )?;
    }
    Ok(())
}

/// The C headers and a Clang module map for them.
fn headers(root: &Path) -> Result<PathBuf, String> {
    let dir = root.join("target/apple/headers");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    for header in std::fs::read_dir(root.join(C_DIR)).map_err(|e| e.to_string())? {
        let header = header.map_err(|e| e.to_string())?;
        std::fs::copy(header.path(), dir.join(header.file_name())).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        dir.join("module.modulemap"),
        "module OrbisC {\n    header \"orbis.h\"\n    export *\n}\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(dir)
}

fn static_lib(root: &Path, target: &str, variable: &str, version: &str) -> Result<PathBuf, String> {
    check(
        Command::new(env!("CARGO"))
            .current_dir(root)
            .env(variable, version)
            .args([
                "rustc",
                "-q",
                "-p",
                "orb-capi",
                "--profile",
                "dist",
                "--lib",
                "--crate-type",
                "staticlib",
                "--target",
                target,
            ]),
    )?;
    Ok(root.join("target").join(target).join("dist/liborbis.a"))
}

pub(crate) fn check(command: &mut Command) -> Result<(), String> {
    eprintln!("$ {command:?}");
    let status = command.status().map_err(|e| format!("{command:?}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{command:?} failed"))
    }
}

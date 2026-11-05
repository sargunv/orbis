//! `cargo xtask apple [--test]`: `liborbis.a` for macOS, iOS and the iOS
//! simulator (arm64, `dist` profile) as `hosts/apple/OrbisC.xcframework`
//! with the C headers and a Clang module map, then the Swift package with
//! SwiftPM. `--test` also runs the
//! package's Swift smoke test.
//! `--ios-sim[=name]` builds the iOS reference app for the simulator,
//! wraps it in an app bundle, and installs and launches it on a booted
//! simulator of that name (default "iPhone 17"), booting it if needed.

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
    let mut test = false;
    let mut simulator = None;
    for arg in args {
        match arg.as_str() {
            "--test" => test = true,
            "--ios-sim" => simulator = Some("iPhone 17".to_owned()),
            a if a.starts_with("--ios-sim=") => {
                simulator = Some(a["--ios-sim=".len()..].to_owned())
            }
            _ => return Err("apple: usage: cargo xtask apple [--test] [--ios-sim[=name]]".into()),
        }
    }
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
                "--target",
                "Orbis",
                "--package-path",
            ])
            .arg(&package),
    )?;
    if test {
        check(
            Command::new("swift")
                .arg("test")
                .arg("--package-path")
                .arg(&package),
        )?;
    }
    if let Some(name) = simulator {
        ios_simulator(&root, &package, &name)?;
    }
    Ok(())
}

/// Builds the iOS reference app for the simulator and runs it there.
fn ios_simulator(root: &Path, package: &Path, name: &str) -> Result<(), String> {
    let derived = root.join("target/apple/derived");
    check(
        Command::new("xcodebuild")
            .current_dir(package)
            .args([
                "-quiet",
                "-scheme",
                "OrbisDemoIOS",
                "-destination",
                "generic/platform=iOS Simulator",
                "-configuration",
                "Release",
                "-derivedDataPath",
            ])
            .arg(&derived)
            // The simulator slice is arm64 only.
            .args(["ARCHS=arm64", "build"]),
    )?;
    let binary = derived.join("Build/Products/Release-iphonesimulator/OrbisDemoIOS");
    let app = root.join("target/apple/OrbisDemo.app");
    let _ = std::fs::remove_dir_all(&app);
    std::fs::create_dir_all(&app).map_err(|e| e.to_string())?;
    std::fs::copy(&binary, app.join("OrbisDemoIOS"))
        .map_err(|e| format!("{}: {e}", binary.display()))?;
    std::fs::write(app.join("Info.plist"), INFO_PLIST).map_err(|e| e.to_string())?;
    let size = std::fs::metadata(&binary).map_err(|e| e.to_string())?.len();
    eprintln!(
        "apple: iOS simulator app binary {:.1} MB",
        size as f64 / 1e6
    );
    // Booting a booted device fails; that's fine.
    let _ = Command::new("xcrun")
        .args(["simctl", "boot", name])
        .status();
    check(
        Command::new("xcrun")
            .args(["simctl", "install", name])
            .arg(&app),
    )?;
    check(Command::new("xcrun").args(["simctl", "launch", name, "dev.orbis.demo"]))
}

const INFO_PLIST: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>OrbisDemoIOS</string>
    <key>CFBundleIdentifier</key><string>dev.orbis.demo</string>
    <key>CFBundleName</key><string>Orbis</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.0.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSRequiresIPhoneOS</key><true/>
    <key>MinimumOSVersion</key><string>16.0</string>
    <key>UIDeviceFamily</key><array><integer>1</integer><integer>2</integer></array>
    <key>UILaunchScreen</key><dict/>
</dict>
</plist>
"#;

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

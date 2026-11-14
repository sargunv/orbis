//! `cargo xtask android [--run [--leakcheck]]`: the Android app.
//! `orb-android` for arm64-v8a (`dist` profile, API 24) as
//! `liborbis.so`, linked by the NDK's clang into
//! `hosts/android/app/src/main/jniLibs`, then the APK with Gradle, which
//! only packages: the library in a `NativeActivity` and the web demo's data
//! as assets. `--run` installs the app on the Android device
//! `scripts/android-serial` names (this worktree's emulator from
//! devices-cli, or `ANDROID_SERIAL`) and starts it. `--run --leakcheck`
//! starts it to make, draw and drop 20 maps, and fails unless the device's
//! live GPU resources stay flat.
//!
//! Needs `ANDROID_HOME` with an NDK (`mise x --` provides it here).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::root;

const TARGET: &str = "aarch64-linux-android";
const ABI: &str = "arm64-v8a";
const API: u32 = 24;
const PROJECT: &str = "hosts/android";

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let (install_app, leakcheck) = match args {
        [] => (false, false),
        [run] if run == "--run" => (true, false),
        [run, leak] if run == "--run" && leak == "--leakcheck" => (true, true),
        _ => return Err("android: usage: cargo xtask android [--run [--leakcheck]]".into()),
    };
    let root = root();
    let bin = ndk_bin()?;
    let clang = bin.join(format!("{TARGET}{API}-clang"));

    // liborbis.so, with the NDK's clang for C dependencies and the link.
    let target_env = TARGET.replace('-', "_");
    crate::run(
        Command::new(env!("CARGO"))
            .current_dir(&root)
            .env(format!("CC_{target_env}"), &clang)
            .env(format!("AR_{target_env}"), bin.join("llvm-ar"))
            .env(
                format!("CARGO_TARGET_{}_LINKER", target_env.to_uppercase()),
                &clang,
            )
            .args([
                "rustc",
                "-p",
                "orb-android",
                "--profile",
                "dist",
                "--lib",
                "--crate-type",
                "cdylib",
                "--target",
                TARGET,
                "--",
                // Android 15 devices may use 16 KB pages.
                "-C",
                "link-arg=-Wl,-z,max-page-size=16384",
            ]),
    )?;
    let project = root.join(PROJECT);
    let jni_libs = project.join("app/src/main/jniLibs").join(ABI);
    std::fs::create_dir_all(&jni_libs).map_err(|e| e.to_string())?;
    let so = jni_libs.join("liborbis.so");
    let built = root
        .join("target")
        .join(TARGET)
        .join("dist/liborb_android.so");
    std::fs::copy(&built, &so).map_err(|e| format!("{}: {e}", built.display()))?;
    let so_size = std::fs::metadata(&so).map_err(|e| e.to_string())?.len();
    eprintln!(
        "android: {ABI} liborbis.so {:.2} MB ({so_size} bytes)",
        so_size as f64 / 1e6
    );

    let verifier = format!(
        "-PrustlsPlatformVerifier={}",
        locked_version(&root, "rustls-platform-verifier-android")?
    );
    crate::run(Command::new("gradle").current_dir(&project).args([
        "--quiet",
        &verifier,
        ":app:assembleRelease",
    ]))?;
    let apk = project.join("app/build/outputs/apk/release/app-release.apk");
    let size = std::fs::metadata(&apk)
        .map_err(|e| format!("{}: {e}", apk.display()))?
        .len();
    eprintln!(
        "android: {} {:.2} MB ({size} bytes)",
        apk.display(),
        size as f64 / 1e6
    );
    if install_app {
        install(&root, &apk, leakcheck)?;
    }
    Ok(())
}

/// The version of `package` that `Cargo.lock` pins: the Kotlin half of
/// rustls-platform-verifier must be its Rust half's.
fn locked_version(root: &Path, package: &str) -> Result<String, String> {
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).map_err(|e| e.to_string())?;
    let name = format!("name = \"{package}\"");
    let mut lines = lock.lines().skip_while(|l| *l != name).skip(1);
    lines
        .next()
        .and_then(|l| l.strip_prefix("version = \""))
        .and_then(|v| v.strip_suffix('"'))
        .map(str::to_string)
        .ok_or_else(|| format!("android: Cargo.lock pins no {package}"))
}

/// The 32-bit Android ABIs (armeabi-v7a, x86) as Rust targets, each with
/// its NDK clang's prefix.
const THIRTY_TWO_BIT: [(&str, &str); 2] = [
    ("armv7-linux-androideabi", "armv7a-linux-androideabi"),
    ("i686-linux-android", "i686-linux-android"),
];

/// Clippy over orb-ffi for the 32-bit Android ABIs, which hosts such as
/// maplibre-compose ship: there `usize` is 32 bits and `u64` aligns to 4 on
/// both, so a dependency's layout assertion or a cast that holds on 64-bit
/// fails here first. Its C dependencies (ring) build with the NDK's clang.
/// Skipped, with a note, for a target that isn't installed or without an
/// NDK.
pub(crate) fn clippy_32bit() -> Result<(), String> {
    let bin = match ndk_bin() {
        Ok(bin) => bin,
        Err(e) => {
            eprintln!("check: {e}, so no 32-bit Android clippy");
            return Ok(());
        }
    };
    for (target, clang) in THIRTY_TWO_BIT {
        if !crate::target_installed(target) {
            eprintln!(
                "check: no {target} target (`rustup target add {target}`), so no clippy for it"
            );
            continue;
        }
        let target_env = target.replace('-', "_");
        crate::run(
            Command::new(env!("CARGO"))
                .env(
                    format!("CC_{target_env}"),
                    bin.join(format!("{clang}{API}-clang")),
                )
                .env(format!("AR_{target_env}"), bin.join("llvm-ar"))
                .args([
                    "clippy",
                    "--quiet",
                    "--target",
                    target,
                    "-p",
                    "orb-ffi",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                    // Clippy flags `thread_local!`s that are already
                    // `const` on Android targets, whose thread locals are
                    // emulated.
                    "-A",
                    "clippy::missing_const_for_thread_local",
                ]),
        )?;
    }
    Ok(())
}

/// The newest NDK's LLVM tools, from `ANDROID_HOME`'s SDK.
fn ndk_bin() -> Result<PathBuf, String> {
    let sdk = PathBuf::from(
        std::env::var_os("ANDROID_HOME")
            .or_else(|| std::env::var_os("ANDROID_SDK_ROOT"))
            .ok_or("android: set ANDROID_HOME (`mise x --` provides it)")?,
    );
    let mut versions: Vec<PathBuf> = std::fs::read_dir(sdk.join("ndk"))
        .map_err(|e| format!("{}: no NDK ({e})", sdk.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    versions.sort();
    let ndk = versions.pop().ok_or("android: no NDK installed")?;
    let prebuilt = ndk.join("toolchains/llvm/prebuilt");
    let host = std::fs::read_dir(&prebuilt)
        .map_err(|e| format!("{}: {e}", prebuilt.display()))?
        .filter_map(|e| e.ok())
        .next()
        .ok_or("android: no NDK host toolchain")?;
    Ok(host.path().join("bin"))
}

/// Installs the app on the device `scripts/android-serial` names and
/// starts it: the worktree's emulator from devices-cli, or `ANDROID_SERIAL`.
fn install(root: &Path, apk: &Path, leakcheck: bool) -> Result<(), String> {
    let serial = android_serial(root)?;
    eprintln!("android: installing on {serial}");
    let adb = |args: &[&str]| {
        let mut command = Command::new("adb");
        command.arg("-s").arg(&serial).args(args);
        command
    };
    crate::run(adb(&["install", "-r"]).arg(apk))?;
    let start = [
        "shell",
        "am",
        "start",
        "-W",
        "-n",
        "dev.orbis.demo/.MainActivity",
    ];
    if !leakcheck {
        return crate::run(&mut adb(&start));
    }
    crate::run(&mut adb(&["shell", "am", "force-stop", "dev.orbis.demo"]))?;
    crate::run(&mut adb(&["logcat", "-c"]))?;
    let extras = ["--ez", "leakcheck", "true"];
    crate::run(adb(&start).args(extras))?;
    // Cold pipelines on the emulator take minutes before the first map draws.
    for _ in 0..600 {
        let log = adb(&["logcat", "-d", "-s", "Orbis"])
            .output()
            .map_err(|e| format!("adb logcat: {e}"))?;
        let log = String::from_utf8_lossy(&log.stdout);
        if let Some(line) = log.lines().find(|l| l.contains("leakcheck: live")) {
            eprintln!("android: {line}");
            let counts: Vec<&str> = line
                .rsplit('[')
                .next()
                .unwrap_or("")
                .trim_end_matches(']')
                .split(", ")
                .collect();
            return match counts.windows(2).all(|w| w[0] == w[1]) {
                true => Ok(()),
                false => Err("android: live GPU resources grew as maps were dropped".into()),
            };
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    Err("android: no leakcheck line in ten minutes".into())
}

/// The adb serial `scripts/android-serial` prints for a functional check: this
/// worktree's emulator, booted on demand.
fn android_serial(root: &Path) -> Result<String, String> {
    let output = Command::new(root.join("scripts/android-serial"))
        .output()
        .map_err(|e| format!("scripts/android-serial: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "scripts/android-serial failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

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
/// The web demo's data, packaged as the app's assets.
const DATA: &str = "web/demo/data";
/// The asset listing the data's files for the app to unpack (NDK asset
/// directories don't list subdirectories): `orb-android`'s `src/android.rs`
/// reads it.
const INDEX: &str = "orbis-files.txt";

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let (install_app, leakcheck) = match args {
        [] => (false, false),
        [run] if run == "--run" => (true, false),
        [run, leak] if run == "--run" && leak == "--leakcheck" => (true, true),
        _ => return Err("android: usage: cargo xtask android [--run [--leakcheck]]".into()),
    };
    let root = root();
    let sdk = PathBuf::from(
        std::env::var_os("ANDROID_HOME")
            .or_else(|| std::env::var_os("ANDROID_SDK_ROOT"))
            .ok_or("android: set ANDROID_HOME (`mise x -- cargo xtask android`)")?,
    );
    let bin = ndk_bin(&sdk)?;
    let clang = bin.join(format!("{TARGET}{API}-clang"));

    // liborbis.so, with the NDK's clang for C dependencies and the link.
    let target_env = TARGET.replace('-', "_");
    check(
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

    write_index(
        &root.join(DATA),
        &project.join("app/build/generated/assets"),
    )?;
    check(
        Command::new("gradle")
            .current_dir(&project)
            .args(["--quiet", ":app:assembleRelease"]),
    )?;
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

/// Writes the asset index into `out`: a line naming this build, so the app
/// unpacks each build's data once, then every file under `data`.
fn write_index(data: &Path, out: &Path) -> Result<(), String> {
    fn walk(dir: &Path, prefix: &str, files: &mut Vec<String>) -> Result<(), String> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(Result::ok)
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
            if entry.path().is_dir() {
                walk(&entry.path(), &format!("{name}/"), files)?;
            } else {
                files.push(name);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(data, "", &mut files)?;
    let build = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let index = format!("{build}\n{}\n", files.join("\n"));
    std::fs::write(out.join(INDEX), index).map_err(|e| e.to_string())
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

/// The newest NDK's LLVM tools.
fn ndk_bin(sdk: &Path) -> Result<PathBuf, String> {
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
    check(adb(&["install", "-r"]).arg(apk))?;
    let start = [
        "shell",
        "am",
        "start",
        "-W",
        "-n",
        "dev.orbis.demo/.MainActivity",
    ];
    if !leakcheck {
        return check(&mut adb(&start));
    }
    check(&mut adb(&["shell", "am", "force-stop", "dev.orbis.demo"]))?;
    check(&mut adb(&["logcat", "-c"]))?;
    let extras = ["--ez", "leakcheck", "true"];
    check(adb(&start).args(extras))?;
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

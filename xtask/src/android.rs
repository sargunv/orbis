//! `cargo xtask android [--run]`: `liborbis.so` for arm64-v8a (`dist` profile,
//! API 24, the D9 floor): `liborbis.a` linked with the generated JNI shim and
//! the ANativeWindow glue by the NDK's clang into
//! `hosts/android/orbis/src/main/jniLibs`, then the AAR and the reference app
//! with Gradle (plan §17: Gradle only packages). `--run` installs the app on
//! the Android device `scripts/android-serial` names (this worktree's emulator
//! from devices-cli, or `ANDROID_SERIAL`) and starts it.
//!
//! Needs `ANDROID_HOME` with an NDK (`mise x --` provides it here).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::apple::check;
use crate::bindings::{C_DIR, root};

const TARGET: &str = "aarch64-linux-android";
const ABI: &str = "arm64-v8a";
const API: u32 = 24;
const PROJECT: &str = "hosts/android";

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let install_app = match args {
        [] => false,
        [flag] if flag == "--run" => true,
        _ => return Err("android: usage: cargo xtask android [--run]".into()),
    };
    let root = root();
    let sdk = PathBuf::from(
        std::env::var_os("ANDROID_HOME")
            .or_else(|| std::env::var_os("ANDROID_SDK_ROOT"))
            .ok_or("android: set ANDROID_HOME (`mise x -- cargo xtask android`)")?,
    );
    let bin = ndk_bin(&sdk)?;
    let clang = bin.join(format!("{TARGET}{API}-clang"));

    // liborbis.a, with the NDK's clang for C dependencies.
    let target_env = TARGET.replace('-', "_");
    let output = Command::new(env!("CARGO"))
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
            "orb-capi",
            "--profile",
            "dist",
            "--lib",
            "--crate-type",
            "staticlib",
            "--target",
            TARGET,
            "--",
            "--print",
            "native-static-libs",
        ])
        .output()
        .map_err(|e| format!("cargo: {e}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!(
            "building liborbis.a for {TARGET} failed:\n{stderr}"
        ));
    }
    let libs_file = root.join("target").join(TARGET).join("native-static-libs");
    let native = match stderr
        .lines()
        .find_map(|l| l.split_once("native-static-libs: ").map(|(_, libs)| libs))
    {
        Some(libs) => libs.trim().to_owned(),
        None => std::fs::read_to_string(&libs_file).map_err(|e| e.to_string())?,
    };
    std::fs::write(&libs_file, &native).map_err(|e| e.to_string())?;
    let archive = root.join("target").join(TARGET).join("dist/liborbis.a");

    // liborbis.so: the JNI shim and glue, with the archive's objects they use.
    let project = root.join(PROJECT);
    let jni_libs = project.join("orbis/src/main/jniLibs").join(ABI);
    std::fs::create_dir_all(&jni_libs).map_err(|e| e.to_string())?;
    let so = jni_libs.join("liborbis.so");
    check(
        Command::new(&clang)
            .args(["-shared", "-fPIC", "-O2", "-Wall", "-Werror", "-I"])
            .arg(root.join(C_DIR))
            .arg(project.join("orbis/src/generated/jni/orbis_jni.c"))
            .arg(project.join("orbis/src/main/cpp/orbis_android.c"))
            .arg(&archive)
            .args(native.split_whitespace())
            .args([
                "-landroid",
                "-Wl,--gc-sections",
                "-Wl,--as-needed",
                // Android 15 devices may use 16 KB pages.
                "-Wl,-z,max-page-size=16384",
                "-s",
                "-o",
            ])
            .arg(&so),
    )?;
    let so_size = std::fs::metadata(&so).map_err(|e| e.to_string())?.len();
    eprintln!(
        "android: {ABI} liborbis.so {:.2} MB ({so_size} bytes)",
        so_size as f64 / 1e6
    );

    check(Command::new("gradle").current_dir(&project).args([
        "--quiet",
        ":orbis:assembleRelease",
        ":app:assembleRelease",
    ]))?;
    let aar = project.join("orbis/build/outputs/aar/orbis-release.aar");
    let apk = project.join("app/build/outputs/apk/release/app-release.apk");
    for file in [&aar, &apk] {
        let size = std::fs::metadata(file)
            .map_err(|e| format!("{}: {e}", file.display()))?
            .len();
        eprintln!(
            "android: {} {:.2} MB ({size} bytes)",
            file.display(),
            size as f64 / 1e6
        );
    }
    if install_app {
        install(&root, &apk)?;
    }
    Ok(())
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

/// Installs the reference app on the device `scripts/android-serial` names and
/// starts it: the worktree's emulator from devices-cli, or `ANDROID_SERIAL`.
fn install(root: &Path, apk: &Path) -> Result<(), String> {
    let serial = android_serial(root)?;
    eprintln!("android: installing on {serial}");
    let adb = |args: &[&str]| {
        let mut command = Command::new("adb");
        command.arg("-s").arg(&serial).args(args);
        command
    };
    check(adb(&["install", "-r"]).arg(apk))?;
    check(&mut adb(&[
        "shell",
        "am",
        "start",
        "-W",
        "-n",
        "dev.orbis.demo/.MainActivity",
    ]))
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

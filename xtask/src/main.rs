//! Workspace tasks. `cargo xtask help` lists them.

#![forbid(unsafe_code)]

mod affected;
mod android;
mod boundaries;
mod compare;
mod dxc;
mod expression_fixtures;
mod fixtures;
mod h2h;
mod labels;
mod mln_ffi;
mod native_fixtures;
mod pmtiles;
mod split;
mod support;
mod sweep;
mod unicode;
mod web;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const HELP: &str = "\
cargo xtask <task>

  check [--affected [--base <rev>]]  formatting (dprint), clippy (native,
               wasm32, and orb-hal-wgpu for Windows when its Rust target is
               installed), tests and boundary rules; --affected scopes them
               to the packages changed since <rev>'s merge base with HEAD
               (default origin/main), the working tree and untracked files,
               and every package depending on those
  clippy [--fix]  clippy over the workspace, native and wasm32 (WebGL 2 too), denying
               warnings; --fix applies its suggestions
  android [--run [--leakcheck]]  the Android app: orb-android as
               liborbis.so for arm64-v8a (NDK, API 24) in a NativeActivity,
               packaged with Gradle; --run installs it on this worktree's
               Android emulator and starts it; --leakcheck checks that maps
               dropped there leave no GPU resources behind
  web [--names] [--panics] [--gl]
               build core.wasm, split along web/deferred.txt into what a
               map's first frames run and core-deferred.wasm, and
               core-gl.wasm (WebGL 2) into web/pkg, and report sizes;
               --names also writes core.wasm whole, with its function
               names, to target/web/core-names.wasm (for `pnpm -C web coverage`);
               --panics builds both to say on the console why they panic
               (bigger: for debugging, never to ship); --gl builds
               core-gl.wasm alone
  pages        build the Pages site (web, then Vite) into web/dist; the
               Pages workflow runs it and deploys on every push to main
  rt [--suite <suite>] [--filter <substring>] [--platform native|chrome|chrome-webgl2|lavapipe|all|native-cpu|native-cpu-tex|native-webgl2|native-gl]
               the fixtures: GL JS's render (default) and query suites, or
               MapLibre Native's native-render, native-query and
               native-expression suites; lavapipe runs the render suites
  support      docs/style-support.md, the style support matrix, from the
               spec's v8.json, orb-style's property table and the latest
               fixture results in docs/scratch/rt; run it after the fixtures
  bench [args]  native benchmarks (release): --experiment frames (GPU/CPU
               frame timings; --nudge for whole frames read back,
               --antialias off for 3D layers without MSAA), gestures,
               load (style load to first complete still) or tiles (the
               extract's tiles through decode, layout and pack); --views
               <file> takes mln-rs's views.txt cameras; web/demo/bench.html
               runs the frames and gestures benches in a browser;
               --compare <rev> [--runs N] builds <rev> in target/compare
               and runs both builds N times (default 3),
               interleaved, reporting each number as <rev> → this tree
  h2h <stills|tiles|frames|size|all|web|mirror> [options]  Orbis against mln-rs
               (pinned, built into the shared h2h cache) over a local
               caching mirror of OpenFreeMap: per-process stills, tile
               blocks, nudged frames, binary sizes; `h2h help` lists options
  dxc [--arch arm64|x64]  DXC's dxcompiler.dll and dxil.dll (pinned
               release) into target/debug and target/release, where D3D12
               builds look for their shader compiler first
  sweep [--dry-run] [--target-dir <path>] [--keep-hours <n>]  prune stale build
               artifacts: keep every build compiled in the last <n> hours
               (default 6) and the 2 newest builds of each crate
  mln-ffi [--filter <substring>] [--verbose]  build orb-ffi's
               maplibre-native-c, compile maplibre-native-ffi's pinned C ABI
               suite against it and score it case by case (results in
               target/mln-ffi/results.json); `mln-ffi declarations [--check]`
               regenerates orb-ffi's declarations and stubs from the pin
  fixtures [gl-js|native|mln-ffi|expressions]  fetch and verify the pinned
               GL JS, MapLibre Native and maplibre-native-ffi mirrors
               (fixtures/gl-js, fixtures/native and fixtures/mln-ffi, not
               committed), or one of them; with
               `expressions`, bundle the pinned style-spec expression
               fixtures into orb-expr
  demo-data    reproduce the committed Tokyo MVT/MLT PMTiles extracts, then
               add Liberty's labels, sprite and glyph subsets to their styles
  unicode      regenerate orb-text's character tables from the UCD
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("check") => check(&args[1..]),
        Some("clippy") => clippy(&args[1..]),
        Some("android") => android::run(&args[1..]),
        Some("web") => web::build(web::Options {
            names: args.iter().any(|a| a == "--names"),
            panics: args.iter().any(|a| a == "--panics"),
            gl: args.iter().any(|a| a == "--gl"),
        }),
        Some("pages") => web::pages(),
        Some("rt") => rt(&args[1..]),
        Some("bench") if args.iter().any(|a| a == "--compare") => compare::bench(&args[1..]),
        Some("bench") => {
            let mut command = vec!["run", "--release", "-p", "orb-bench", "--"];
            command.extend(args.iter().skip(1).map(String::as_str));
            cargo(&command)
        }
        Some("h2h") => h2h::run(&args[1..]),
        Some("sweep") => sweep::run(&args[1..]),
        Some("dxc") => dxc::run(&args[1..]),
        Some("support") => support::run(),
        Some("mln-ffi") => mln_ffi::run(&args[1..]),
        Some("fixtures") if args.get(1).map(String::as_str) == Some("expressions") => {
            expression_fixtures::bundle()
        }
        Some("fixtures") => match args.get(1).map(String::as_str) {
            Some("gl-js") => fixtures::ensure(),
            Some("native") => native_fixtures::ensure(),
            Some("mln-ffi") => mln_ffi::ensure(),
            None => fixtures::ensure()
                .and_then(|()| native_fixtures::ensure())
                .and_then(|()| mln_ffi::ensure()),
            Some(other) => Err(format!("fixtures: unknown mirror `{other}`")),
        },
        Some("demo-data") => pmtiles::extract().and_then(|()| labels::generate()),
        Some("unicode") => unicode::generate(),
        Some("help") | None => {
            print!("{HELP}");
            Ok(())
        }
        Some(other) => Err(format!("unknown task `{other}`\n\n{HELP}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("xtask: {message}");
            ExitCode::FAILURE
        }
    }
}

fn rt(args: &[String]) -> Result<(), String> {
    let suite = args
        .windows(2)
        .find(|pair| pair[0] == "--suite")
        .map_or("render", |pair| pair[1].as_str());
    if suite.starts_with("native-") {
        native_fixtures::ensure()?;
    } else {
        fixtures::ensure()?;
    }
    // Native's render suite also compares with GL JS's images where the
    // suites share a test.
    if suite == "native-render" {
        fixtures::ensure()?;
    }
    if args.windows(2).any(|pair| {
        pair[0] == "--platform" && ["chrome", "chrome-webgl2", "all"].contains(&pair[1].as_str())
    }) {
        web::build(Default::default())?;
    }
    let mut command = vec!["run", "--release", "-p", "orb-fixtures"];
    // wgpu's GLES back end (through ANGLE on macOS) for `native-gl`.
    if args
        .windows(2)
        .any(|pair| pair[0] == "--platform" && pair[1] == "native-gl")
    {
        command.extend(["--features", "gl"]);
    }
    command.push("--");
    command.extend(args.iter().map(String::as_str));
    cargo(&command)
}

/// Clippy over core-gl.wasm's build: orb-web on WebGL 2, with `extra`
/// arguments (`--fix`'s).
fn gl_clippy(extra: &[&str]) -> Result<(), String> {
    let mut gl = vec![
        "clippy",
        "--quiet",
        "-p",
        "orb-web",
        "--target",
        "wasm32-unknown-unknown",
        "--no-default-features",
        "--features",
        "gl",
    ];
    gl.extend(extra);
    gl.extend(["--", "-D", "warnings"]);
    cargo(&gl)
}

/// The fast pre-push check from `AGENTS.md`.
fn check(args: &[String]) -> Result<(), String> {
    let mut affected = false;
    let mut base = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--affected" => affected = true,
            "--base" => {
                base = Some(
                    args.next()
                        .ok_or("check: --base takes a revision")?
                        .to_string(),
                )
            }
            other => return Err(format!("check: unknown argument `{other}`")),
        }
    }
    if !affected {
        if base.is_some() {
            return Err("check: `--base` needs `--affected`".to_string());
        }
        return full_check();
    }
    affected_check(base.as_deref().unwrap_or("origin/main"))
}

fn full_check() -> Result<(), String> {
    format_check()?;
    clippy(&[])?;
    windows_clippy()?;
    cargo(&["test", "--workspace", "--quiet"])?;
    cpu_driver_tests(true, true)?;
    mln_ffi::check_declarations()?;
    boundaries::check()
}

/// Every file's formatting, through dprint (rustfmt for Rust, oxc for the
/// web, ktfmt and ruff), as `dprint.jsonc` configures it.
fn format_check() -> Result<(), String> {
    run(Command::new("dprint").arg("check"))
        .map_err(|e| format!("{e}; `dprint fmt` fixes formatting"))
}

/// `clippy [--fix]`: clippy over the workspace natively and, without the
/// native-only crates, for wasm32 (and orb-web's WebGL 2 build), denying
/// warnings; `--fix` applies its
/// suggestions instead. hk's clippy step runs this.
fn clippy(args: &[String]) -> Result<(), String> {
    let fix = match args {
        [] => false,
        [flag] if flag == "--fix" => true,
        _ => return Err("clippy: the only option is --fix".to_string()),
    };
    let fix_args: &[&str] = if fix {
        &["--fix", "--allow-dirty", "--allow-staged"]
    } else {
        &[]
    };
    let mut native = vec!["clippy", "--workspace", "--all-targets", "--quiet"];
    native.extend(fix_args);
    native.extend(["--", "-D", "warnings"]);
    cargo(&native)?;
    let mut wasm = vec![
        "clippy",
        "--workspace",
        "--quiet",
        "--target",
        "wasm32-unknown-unknown",
    ];
    for name in boundaries::NATIVE_ONLY {
        wasm.extend(["--exclude", name]);
    }
    wasm.extend(fix_args);
    wasm.extend(["--", "-D", "warnings"]);
    cargo(&wasm)?;
    gl_clippy(fix_args)
}

/// `check --affected`: the same steps, for the packages that changed since
/// `base`'s merge base with `HEAD` and the packages depending on them.
fn affected_check(base: &str) -> Result<(), String> {
    let selection = affected::select(base)?;
    format_check()?;
    let packages = selection.packages();
    if packages.is_empty() {
        eprintln!("affected: no workspace package changed, so fmt and boundaries only");
        return boundaries::check();
    }
    eprintln!("{}", selection.report());

    let mut clippy = vec!["clippy", "--all-targets", "--quiet"];
    let mut test = vec!["test", "--quiet"];
    for name in &packages {
        clippy.extend(["-p", name.as_str()]);
        test.extend(["-p", name.as_str()]);
    }
    clippy.extend(["--", "-D", "warnings"]);
    cargo(&clippy)?;

    let wasm_packages: Vec<&String> = packages
        .iter()
        .copied()
        .filter(|name| !boundaries::NATIVE_ONLY.contains(&name.as_str()))
        .collect();
    if wasm_packages.is_empty() {
        eprintln!("affected: no selected package builds for wasm32");
    } else {
        let mut wasm = vec!["clippy", "--quiet", "--target", "wasm32-unknown-unknown"];
        for name in &wasm_packages {
            wasm.extend(["-p", name.as_str()]);
        }
        wasm.extend(["--", "-D", "warnings"]);
        cargo(&wasm)?;
        if selection.contains("orb-web") {
            gl_clippy(&[])?;
        }
    }

    if selection.contains("orb-hal-wgpu") {
        windows_clippy()?;
    }
    cargo(&test)?;
    cpu_driver_tests(
        selection.contains("orb-render"),
        selection.contains("orb-api"),
    )?;
    if selection.contains("orb-ffi") {
        mln_ffi::check_declarations()?;
    }
    boundaries::check()
}

/// The renderer's tests and `orb-api`'s path tests again on the CPU frame
/// driver with its tables in data textures, WebGL 2's path (#140), run
/// natively (`ORB_DRIVER=cpu-tex`). `orb-runtime`'s `drivers` test holds
/// the CPU driver to the GPU's word for word in every run.
fn cpu_driver_tests(render: bool, path: bool) -> Result<(), String> {
    // One run each: `--test` would narrow orb-render's too.
    let runs: [(bool, &[&str]); 2] = [
        (render, &["test", "--quiet", "-p", "orb-render"]),
        (
            path,
            &["test", "--quiet", "-p", "orb-api", "--test", "path"],
        ),
    ];
    for (_, args) in runs.into_iter().filter(|(run, _)| *run) {
        run(Command::new(env!("CARGO"))
            .args(args)
            .env("ORB_DRIVER", "cpu-tex"))?;
    }
    Ok(())
}

/// Clippy over orb-hal-wgpu for Windows, where its D3D12 code builds: it
/// needs no C toolchain, unlike the crates above it (ring). Skipped, with a
/// note, without the Rust target.
fn windows_clippy() -> Result<(), String> {
    const TARGET: &str = "x86_64-pc-windows-msvc";
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .is_ok_and(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .any(|l| l == TARGET)
        });
    if !installed {
        eprintln!("check: no {TARGET} target (`rustup target add {TARGET}`), so no Windows clippy");
        return Ok(());
    }
    cargo(&[
        "clippy",
        "--quiet",
        "--target",
        TARGET,
        "-p",
        "orb-hal-wgpu",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ])
}

/// A binary's file name on this platform (`orbis.exe` on Windows).
pub(crate) fn exe(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

/// The repository's root.
pub(crate) fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the workspace root")
        .to_path_buf()
}

pub(crate) fn cargo(args: &[&str]) -> Result<(), String> {
    run(Command::new(env!("CARGO")).args(args))
}

/// Runs `command`, echoed first as a shell would show it; an error if it
/// can't start or fails.
pub(crate) fn run(command: &mut Command) -> Result<(), String> {
    let shown = shown(command);
    eprintln!("$ {shown}");
    let status = command.status().map_err(|e| {
        format!("failed to run `{shown}` ({e}); `mise install` provides the pinned tools")
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{shown}` failed"))
    }
}

/// `command` as a shell would show it: its environment, then its
/// program's name and its arguments.
fn shown(command: &Command) -> String {
    let env = command.get_envs().filter_map(|(key, value)| {
        Some(format!(
            "{}={}",
            key.to_string_lossy(),
            value?.to_string_lossy()
        ))
    });
    let program = Path::new(command.get_program());
    let program = program.file_name().unwrap_or(program.as_os_str());
    let args = command.get_args().map(|a| a.to_string_lossy().into_owned());
    env.chain([program.to_string_lossy().into_owned()])
        .chain(args)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_shows_its_environment_program_name_and_arguments() {
        let mut command = Command::new("/usr/local/bin/cargo");
        command
            .args(["build", "-p", "orb-web"])
            .env("RUSTC_BOOTSTRAP", "1");
        assert_eq!(shown(&command), "RUSTC_BOOTSTRAP=1 cargo build -p orb-web");
    }
}

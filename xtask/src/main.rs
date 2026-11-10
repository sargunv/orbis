//! Workspace tasks. `cargo xtask help` lists them.

#![forbid(unsafe_code)]

mod affected;
mod android;
mod boundaries;
mod compare;
mod expression_fixtures;
mod fixtures;
mod h2h;
mod labels;
mod native_fixtures;
mod pmtiles;
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
  clippy [--fix]  clippy over the workspace, native and wasm32, denying
               warnings; --fix applies its suggestions
  android [--run [--leakcheck]]  the Android app: orb-android as
               liborbis.so for arm64-v8a (NDK, API 24) in a NativeActivity,
               packaged with Gradle; --run installs it on this worktree's
               Android emulator and starts it; --leakcheck checks that maps
               dropped there leave no GPU resources behind
  web [--names]  build core.wasm into web/pkg, report sizes; --names also
               writes it with its function names to target/web/core-names.wasm
               (for `pnpm -C web coverage`)
  pages        build the Pages site (web, then Vite) into web/dist; the
               Pages workflow runs it and deploys on every push to main
  rt [--suite <suite>] [--filter <substring>] [--platform native|chrome|lavapipe|all]
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
  sweep [--dry-run] [--target-dir <path>] [--keep-hours <n>]  prune stale build
               artifacts: keep every build compiled in the last <n> hours
               (default 6) and the 2 newest builds of each crate
  fixtures [gl-js|native|expressions]  fetch and verify the pinned GL JS
               and MapLibre Native fixture mirrors (fixtures/gl-js and
               fixtures/native, not committed), or one of them; with
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
        Some("web") => web::build(args.iter().any(|a| a == "--names")),
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
        Some("support") => support::run(),
        Some("fixtures") if args.get(1).map(String::as_str) == Some("expressions") => {
            expression_fixtures::bundle()
        }
        Some("fixtures") => match args.get(1).map(String::as_str) {
            Some("gl-js") => fixtures::ensure(),
            Some("native") => native_fixtures::ensure(),
            None => fixtures::ensure().and_then(|()| native_fixtures::ensure()),
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
    if args
        .windows(2)
        .any(|pair| pair[0] == "--platform" && ["chrome", "all"].contains(&pair[1].as_str()))
    {
        web::build(false)?;
    }
    let mut command = vec!["run", "--release", "-p", "orb-fixtures", "--"];
    command.extend(args.iter().map(String::as_str));
    cargo(&command)
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
    boundaries::check()
}

/// Every file's formatting, through dprint (rustfmt for Rust, oxc for the
/// web, ktfmt, clang-format and ruff), as `dprint.jsonc` configures it.
fn format_check() -> Result<(), String> {
    eprintln!("$ dprint check");
    let status = Command::new("dprint")
        .arg("check")
        .status()
        .map_err(|e| format!("failed to run dprint ({e}); `mise install` provides it"))?;
    if status.success() {
        Ok(())
    } else {
        Err("formatting differs; `dprint fmt` fixes it".to_string())
    }
}

/// `clippy [--fix]`: clippy over the workspace natively and, without the
/// native-only crates, for wasm32, denying warnings; `--fix` applies its
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
    cargo(&wasm)
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
    }

    if selection.contains("orb-hal-wgpu") {
        windows_clippy()?;
    }
    cargo(&test)?;
    boundaries::check()
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
    eprintln!("$ cargo {}", args.join(" "));
    let status = Command::new(env!("CARGO"))
        .args(args)
        .status()
        .map_err(|e| format!("failed to run cargo: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`cargo {}` failed", args.join(" ")))
    }
}

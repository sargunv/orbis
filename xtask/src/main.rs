//! Workspace tasks. `cargo xtask help` lists them.

#![forbid(unsafe_code)]

mod affected;
mod android;
mod apple;
mod bindings;
mod boundaries;
mod capi;
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

use std::process::{Command, ExitCode};

const HELP: &str = "\
cargo xtask <task>

  check [--affected [--base <rev>]]  fmt, clippy (native and wasm32), tests,
               generated bindings and boundary rules; --affected scopes them
               to the packages changed since <rev>'s merge base with HEAD
               (default origin/main), the working tree and untracked files,
               and every package depending on those
  bindings [--check | --test]  the C header, Kotlin and Swift bindings from
               orb-capi; --check fails when they're stale;
               --test runs the emitters' tests and the C, Kotlin and Swift
               smoke tests
  capi         the C library for this host: static library, headers,
               pkg-config and CMake files in target/capi
  android [--run]  liborbis.so for arm64-v8a (NDK, API 24), the AAR and
               the reference app with Gradle; --run installs it on this
               worktree's Android emulator and starts it
  apple [--test] [--ios-sim[=name]]  hosts/apple/OrbisC.xcframework (macOS,
               iOS, iOS simulator) and the Swift package; --test runs the
               Swift smoke test; --ios-sim runs the iOS app on a simulator
  web          build core.wasm into web/pkg, report sizes
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
               frame timings; --nudge for whole frames read back), gestures,
               load (style load to first complete still) or tiles (the
               extract's tiles through decode, layout and pack); --views
               <file> takes mln-rs's views.txt cameras; web/demo/bench.html
               runs the frames and gestures benches in a browser;
               --compare <rev> [--runs N] builds <rev> in a temporary
               worktree and runs both builds N times (default 3),
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
        Some("bindings") => bindings::run(&args[1..]),
        Some("capi") => capi::build().map(|_| ()),
        Some("android") => android::run(&args[1..]),
        Some("apple") => apple::run(&args[1..]),
        Some("web") => web::build(),
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
        web::build()?;
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
    cargo(&["fmt", "--all", "--check"])?;
    cargo(&[
        "clippy",
        "--workspace",
        "--all-targets",
        "--quiet",
        "--",
        "-D",
        "warnings",
    ])?;
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
    wasm.extend(["--", "-D", "warnings"]);
    cargo(&wasm)?;
    cargo(&["test", "--workspace", "--quiet"])?;
    bindings::run(&["--check".into()])?;
    boundaries::check()
}

/// `check --affected`: the same steps, for the packages that changed since
/// `base`'s merge base with `HEAD` and the packages depending on them.
fn affected_check(base: &str) -> Result<(), String> {
    let selection = affected::select(base)?;
    cargo(&["fmt", "--all", "--check"])?;
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

    cargo(&test)?;
    // The bindings are generated from orb-capi's bridge through
    // diplomat-swift-kotlin.
    if ["orb-capi", "diplomat-swift-kotlin"]
        .iter()
        .any(|name| selection.contains(name))
    {
        bindings::run(&["--check".into()])?;
    }
    boundaries::check()
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

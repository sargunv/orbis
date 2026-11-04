//! Workspace tasks. `cargo xtask help` lists them.

#![forbid(unsafe_code)]

mod android;
mod apple;
mod bindings;
mod boundaries;
mod capi;
mod expression_fixtures;
mod fixtures;
mod labels;
mod pmtiles;
mod unicode;
mod web;

use std::process::{Command, ExitCode};

const HELP: &str = "\
cargo xtask <task>

  check        fmt, clippy (native and wasm32), tests and boundary rules
  boundaries   plan §5.3 boundary rules only
  bindings [--check | test]  the C header, Kotlin and Swift bindings from orb-capi
               (decision 0018); --check fails when they're stale; test runs the
               emitters' Swift and Kotlin compile-and-run tests
  android [--run[=serial]]  liborbis.so for arm64-v8a (NDK, API 24), the AAR and
               the reference app with Gradle; --run installs and starts it
  apple [--test]  liborbis.a for macOS, iOS and the iOS simulator as
               hosts/apple/OrbisC.xcframework, then the Swift package and the
               macOS reference app; --test runs the Swift binding scenario
  capi [--test]  the C library for this host: static library, headers,
               pkg-config and CMake files in target/capi; --test builds and runs
               the C smoke test against it, through pkg-config and CMake
  fixtures     fetch and verify the pinned MapLibre GL JS fixture mirror
               (fixtures/gl-js, not committed); no work when it matches
  fixtures expressions  bundle the pinned style-spec expression fixtures
  unicode      regenerate orb-text's character tables from the UCD
  web          build core.wasm and harness.wasm into web/pkg, report sizes
  extract      reproduce the committed Tokyo MVT/MLT PMTiles archives
  tile-parity  compare every extract tile and measure decode times
  demo-labels  add Liberty's labels, its sprite and glyph subsets to the Tokyo demo
  pages        build the Pages site (web, then Vite) into web/dist; the
               Pages workflow runs it and deploys on every push to main
  rt [--suite render|query] [--filter <substring>] [--platform native|chrome|lavapipe|all]  GL JS fixtures
  bench [args]  time frames, still and moving, on the default GPU (release);
               web/demo/bench.html runs the same bench in a browser
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("check") => check(),
        Some("boundaries") => boundaries::check(),
        Some("bindings") => bindings::run(&args[1..]),
        Some("capi") => capi::run(&args[1..]),
        Some("apple") => apple::run(&args[1..]),
        Some("android") => android::run(&args[1..]),
        Some("fixtures") if args.get(1).map(String::as_str) == Some("expressions") => {
            expression_fixtures::bundle()
        }
        Some("fixtures") => fixtures::ensure(),
        Some("extract") => pmtiles::extract(),
        Some("transcode") => pmtiles::transcode(),
        Some("tile-parity") => pmtiles::verify(),
        Some("unicode") => unicode::generate(),
        Some("demo-labels") => labels::generate(),
        Some("web") => web::build(),
        Some("pages") => web::pages(),
        Some("rt") => rt(&args[1..]),
        Some("bench") => {
            let mut command = vec!["run", "--release", "-p", "orb-bench", "--"];
            command.extend(args.iter().skip(1).map(String::as_str));
            cargo(&command)
        }
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
    fixtures::ensure()?;
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
fn check() -> Result<(), String> {
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
        "--features",
        "orb-web/harness",
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

//! `cargo xtask web`: builds `core.wasm` and its JS bindings into
//! `web/demo/pkg/`, reports its size by crate (twiggy) and warns when it
//! grows past plan §14.3's numbers; then `harness.wasm`, the same module with
//! the bench page's and the fixture driver's entry points.
//!
//! `cargo xtask pages`: the above, then the Vite build of the site
//! `.github/workflows/pages.yml` deploys, into `web/dist/`.

use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::cargo;

const OUT_DIR: &str = "web/demo/pkg";

/// Plan §14.3's initial-payload target. Size is watched, not a gate: above
/// it, and above the ceiling, the build only warns.
const BROTLI_TARGET: usize = 450 * 1024;
/// Plan §14.3's ceiling for the first release.
const BROTLI_CEILING: usize = 550 * 1024;

/// Crates whose code the size report attributes separately.
const CRATES: &[&str] = &[
    "wgpu_types",
    "wgpu_core",
    "wgpu",
    "naga",
    "web_sys",
    "js_sys",
    "wasm_bindgen_futures",
    "wasm_bindgen",
    "mlt_core",
    "fastpfor",
    "hashbrown",
    "earcut",
    "miniz_oxide",
    "dlmalloc",
    "alloc",
    "core",
    "std",
];

pub(crate) fn build() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let out = root.join(OUT_DIR);
    let built = root.join("target/wasm32-unknown-unknown/wasm-release/orb_web.wasm");

    let core = module(&root, "orb-web", "core", &[])?;
    report_crates(&built);
    let js = std::fs::read(out.join("core.js")).map_err(|e| format!("core.js: {e}"))?;
    eprintln!(
        "core.wasm: {} KiB raw, {} KiB brotli; core.js {} KiB raw ({OUT_DIR})",
        core.raw / 1024,
        core.brotli / 1024,
        js.len() / 1024,
    );
    for (limit, what) in [(BROTLI_CEILING, "ceiling"), (BROTLI_TARGET, "target")] {
        if core.brotli > limit {
            eprintln!(
                "warning: core.wasm is {} KiB brotli, over the {} KiB {what} (plan §14.3)",
                core.brotli / 1024,
                limit / 1024
            );
            break;
        }
    }
    // The bench page's and the fixture driver's module: core plus their
    // entry points, outside the budget because no map page loads it.
    word_data(&out)?;
    let harness = module(&root, "orb-web", "harness", &["--features", "harness"])?;
    eprintln!(
        "harness.wasm: {} KiB raw, {} KiB brotli",
        harness.raw / 1024,
        harness.brotli / 1024
    );
    // `font-faces` drawing, loaded the first time a style declares a file.
    let shaper = module(&root, "orb-shaper", "shaper", &[])?;
    eprintln!(
        "shaper.wasm: {} KiB raw, {} KiB brotli (lazy)",
        shaper.raw / 1024,
        shaper.brotli / 1024
    );
    Ok(())
}

pub(crate) fn pages() -> Result<(), String> {
    build()?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    run(Command::new("pnpm")
        .current_dir(root.join("web"))
        .arg("build"))
}

struct Sizes {
    raw: usize,
    brotli: usize,
}

/// Builds `package` with `features`, binds it as `<name>.js` and
/// `<name>_bg.wasm` in [`OUT_DIR`] and shrinks it.
fn module(root: &Path, package: &str, name: &str, features: &[&str]) -> Result<Sizes, String> {
    let built = root.join(format!(
        "target/wasm32-unknown-unknown/wasm-release/{}.wasm",
        package.replace('-', "_")
    ));
    let mut args = vec![
        "build",
        "--quiet",
        "-p",
        package,
        "--target",
        "wasm32-unknown-unknown",
        "--profile",
        "wasm-release",
    ];
    args.extend(features);
    cargo(&args)?;
    let out = root.join(OUT_DIR);
    run(Command::new("wasm-bindgen")
        .arg(&built)
        .args(["--target", "web", "--out-name", name, "--out-dir"])
        .arg(&out))?;
    let wasm = out.join(format!("{name}_bg.wasm"));
    run(Command::new("wasm-opt")
        .args([
            "-Oz",
            "--converge",
            "--strip-producers",
            "--strip-target-features",
            "--enable-simd",
            "--enable-bulk-memory",
            "--enable-sign-ext",
        ])
        .args([
            "--enable-nontrapping-float-to-int",
            "--enable-multivalue",
            "--enable-reference-types",
        ])
        .arg(&wasm)
        .arg("-o")
        .arg(&wasm))?;
    let bytes = std::fs::read(&wasm).map_err(|e| format!("{}: {e}", wasm.display()))?;
    sizes(&bytes)
}

fn sizes(bytes: &[u8]) -> Result<Sizes, String> {
    let mut brotli = Vec::new();
    {
        let mut writer = brotli::CompressorWriter::new(&mut brotli, 4096, 11, 22);
        writer.write_all(bytes).map_err(|e| e.to_string())?;
    }
    Ok(Sizes {
        raw: bytes.len(),
        brotli: brotli.len(),
    })
}

/// Writes the Southeast Asian word data the job workers fetch when labels
/// first need it (decision 0009).
fn word_data(out: &Path) -> Result<(), String> {
    let bytes = orb_text::sea_dictionary_bytes();
    let path = out.join("seasia.bin");
    std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let s = sizes(&bytes)?;
    eprintln!(
        "seasia.bin: {} KiB raw, {} KiB brotli (lazy)",
        s.raw / 1024,
        s.brotli / 1024
    );
    Ok(())
}

/// Prints the code size per crate of the build before wasm-opt, which
/// still has function names, from `twiggy top`. Shares are what matter:
/// wasm-opt shrinks everything.
fn report_crates(wasm: &Path) {
    let output = Command::new("twiggy")
        .args(["top", "-n", "1000000", "--format", "csv"])
        .arg(wasm)
        .output();
    let Some(output) = output.ok().filter(|o| o.status.success()) else {
        eprintln!("size report skipped: twiggy not found (`mise install` provides it)");
        return;
    };
    let csv = String::from_utf8_lossy(&output.stdout);
    let mut sizes: std::collections::BTreeMap<String, u64> = Default::default();
    let mut code = 0u64;
    for line in csv.lines().skip(1) {
        // Name,ShallowSize,ShallowSizePercent,RetainedSize,RetainedSizePercent;
        // names may contain commas.
        let fields: Vec<&str> = line.rsplitn(5, ',').collect();
        let [_, _, _, size, name] = fields[..] else {
            continue;
        };
        let Ok(size) = size.parse::<u64>() else {
            continue;
        };
        if name.contains("custom section")
            || name.contains("\"function names\"")
            || name.contains("data segment")
        {
            continue;
        }
        code += size;
        let krate = name
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .find(|word| word.starts_with("orb_") || CRATES.contains(word))
            .unwrap_or("(generic)");
        *sizes.entry(krate.to_string()).or_default() += size;
    }
    let mut sizes: Vec<_> = sizes.into_iter().collect();
    sizes.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
    eprintln!("code by crate before wasm-opt ({} KiB):", code / 1024);
    for (krate, size) in sizes.iter().take(15) {
        eprintln!("  {:>6.1}%  {krate}", *size as f64 * 100.0 / code as f64);
    }
}

fn run(command: &mut Command) -> Result<(), String> {
    let name = command.get_program().to_string_lossy().into_owned();
    let status = command
        .status()
        .map_err(|e| format!("failed to run {name} ({e}); `mise install` provides it"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{name} failed"))
    }
}

//! `cargo xtask web`: builds `core.wasm` and its JS bindings into
//! `web/pkg/` and reports its sizes, raw, brotli and by crate (twiggy).
//! Size is watched, not a gate: it warns when a module grows past its
//! watch level, and never fails. `--names` also writes core.wasm with its
//! function names, for `pnpm -C web coverage`.
//!
//! `cargo xtask pages`: the above, then the Vite build of the site
//! `.github/workflows/pages.yml` deploys, into `web/dist/`.

use std::io::Write;
use std::path::Path;
use std::process::Command;

const OUT_DIR: &str = "web/pkg";
/// Where `cargo xtask web --names` writes `core.wasm` with its function names.
const NAMED_CORE: &str = "target/web/core-names.wasm";

/// The brotli sizes, in KiB, past which `core.wasm` and `shaper.wasm` warn:
/// a few percent above what each weighs now, so crossing one means
/// something grew. Find what in the per-crate report; if the growth earns
/// its bytes, raise the level to the new size plus the same margin, and
/// when a change shrinks a module, lower it.
const CORE_WATCH_KIB: usize = 830;
const SHAPER_WATCH_KIB: usize = 380;

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

/// Builds both modules; `names` also writes `core.wasm` with its function
/// names to [`NAMED_CORE`], for `pnpm -C web coverage`.
pub(crate) fn build(names: bool) -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let out = root.join(OUT_DIR);
    let built = root.join("target/wasm32-unknown-unknown/wasm-release/orb_web.wasm");

    let named = root.join(NAMED_CORE);
    // A copy from an earlier build no longer matches web/pkg's: if its
    // imports still fit core.js, coverage would measure the old code.
    if !names {
        let _ = std::fs::remove_file(&named);
    }
    let named = names.then_some(named);
    let core = module(&root, "orb-web", "core", named.as_deref())?;
    report_crates(&built);
    let js = std::fs::read(out.join("core.js")).map_err(|e| format!("core.js: {e}"))?;
    eprintln!(
        "core.wasm: {} KiB raw, {} KiB brotli; core.js {} KiB raw ({OUT_DIR})",
        core.raw / 1024,
        core.brotli / 1024,
        js.len() / 1024,
    );
    watch("core.wasm", core.brotli, CORE_WATCH_KIB);
    // `font-faces` drawing, loaded the first time a style declares a file.
    let shaper = module(&root, "orb-shaper", "shaper", None)?;
    eprintln!(
        "shaper.wasm: {} KiB raw, {} KiB brotli (lazy)",
        shaper.raw / 1024,
        shaper.brotli / 1024
    );
    watch("shaper.wasm", shaper.brotli, SHAPER_WATCH_KIB);
    Ok(())
}

/// Warns when `name`'s brotli size passes its watch level.
fn watch(name: &str, brotli: usize, level_kib: usize) {
    if brotli > level_kib * 1024 {
        eprintln!(
            "warning: {name} grew to {} KiB brotli, past its {level_kib} KiB watch level (xtask/src/web.rs)",
            brotli / 1024
        );
    }
}

pub(crate) fn pages() -> Result<(), String> {
    build(false)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    run(Command::new("pnpm")
        .current_dir(root.join("web"))
        .arg("build"))
}

struct Sizes {
    raw: usize,
    brotli: usize,
}

/// Builds `package`, binds it as `<name>.js` and `<name>_bg.wasm` in
/// [`OUT_DIR`] and shrinks it; and shrinks it into `names` too, keeping
/// its function names.
fn module(root: &Path, package: &str, name: &str, names: Option<&Path>) -> Result<Sizes, String> {
    let built = root.join(format!(
        "target/wasm32-unknown-unknown/wasm-release/{}.wasm",
        package.replace('-', "_")
    ));
    // A panic traps at once, with no message. On wasm32-unknown-unknown
    // std's stderr is a sink: the message was formatted only to be dropped,
    // and that formatting, with the `Debug` impls and source locations it
    // reaches, was 5% of core.wasm. It takes std rebuilt with the module
    // and a flag stable Rust keeps unstable, so this build alone sets
    // RUSTC_BOOTSTRAP.
    let args = [
        "build",
        "--quiet",
        "-p",
        package,
        "--target",
        "wasm32-unknown-unknown",
        "--profile",
        "wasm-release",
        "-Zbuild-std=std,panic_abort",
        "--config",
        r#"target.wasm32-unknown-unknown.rustflags = ["-Zunstable-options", "-Cpanic=immediate-abort"]"#,
    ];
    eprintln!("$ RUSTC_BOOTSTRAP=1 cargo {}", args.join(" "));
    run(Command::new(env!("CARGO"))
        .env("RUSTC_BOOTSTRAP", "1")
        .args(args))?;
    let out = root.join(OUT_DIR);
    run(Command::new("wasm-bindgen")
        .arg(&built)
        .args(["--target", "web", "--out-name", name, "--out-dir"])
        .arg(&out))?;
    let wasm = out.join(format!("{name}_bg.wasm"));
    if let Some(names) = names {
        let dir = names.parent().expect("a file in a directory");
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        run(wasm_opt(&wasm, names).arg("-g"))?;
        eprintln!("{name}.wasm with its names: {}", names.display());
    }
    run(&mut wasm_opt(&wasm, &wasm))?;
    let bytes = std::fs::read(&wasm).map_err(|e| format!("{}: {e}", wasm.display()))?;
    sizes(&bytes)
}

/// Shrinks `input` into `output`.
fn wasm_opt(input: &Path, output: &Path) -> Command {
    let mut command = Command::new("wasm-opt");
    command
        .args([
            "-Oz",
            "--converge",
            // The module makes its memory, zeroed, and Rust lays nothing
            // out below 1 MiB: true of both modules, and 2 KiB brotli.
            "--zero-filled-memory",
            "--low-memory-unused",
            "--strip-producers",
            "--strip-target-features",
            // By default binaryen inlines every function that has one
            // caller, at any size, which welds whole features into a few
            // giant functions. V8 compiles a function whole the first time
            // it runs, so a map's first frame would wait on code it never
            // runs. Kept apart, functions compile only when they run, and
            // the module compresses better.
            "--one-caller-inline-max-function-size=20",
            "--enable-simd",
            "--enable-bulk-memory",
            "--enable-sign-ext",
        ])
        .args([
            "--enable-nontrapping-float-to-int",
            "--enable-multivalue",
            "--enable-reference-types",
        ])
        .arg(input)
        .arg("-o")
        .arg(output);
    command
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

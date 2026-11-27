//! `cargo xtask web`: builds `core.wasm` and its JS bindings into
//! `web/pkg/`, split by feature (split.rs) into `core.wasm`, what plain
//! vector maps run, and a module per feature (`core-<feature>.wasm`),
//! which a map loads once its style declares the feature; reports their
//! sizes, raw, brotli and by crate (twiggy); then `core-gl.wasm`, the same
//! crates on WebGL 2 for browsers without a WebGPU adapter, split the same
//! way, and `shaper.wasm`. It fails on an entry of the split's lists that
//! names no function of either build, and writes the bindings' methods
//! the API's module holds to `api-calls.json`, for the web package's gate.
//! Size is watched, not a gate: it warns when a module grows past its
//! watch level, and never fails. `--names` also writes core.wasm whole,
//! with its function names, for `pnpm -C web coverage`.
//!
//! `cargo xtask pages`: the above, then the Vite build of the site
//! `.github/workflows/pages.yml` deploys, into `web/dist/`.

use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::{data, glue, run, split};

const OUT_DIR: &str = "web/pkg";
/// The file in [`OUT_DIR`] that names the bindings' methods the API's
/// module runs ([`write_api_calls`]).
const API_CALLS: &str = "api-calls.json";
/// Where `cargo xtask web --names` writes `core.wasm` whole, with its
/// function names.
const NAMED_CORE: &str = "target/web/core-names.wasm";

/// The brotli sizes, in KiB, past which `core.wasm`, `core-gl.wasm` and
/// `shaper.wasm` warn: a few percent above what each weighs now, so
/// crossing one means something grew. Find what in the per-crate report;
/// if the growth earns its bytes, raise the level to the new size plus the
/// same margin, and when a change shrinks a module, lower it.
const CORE_WATCH_KIB: usize = 730;
const CORE_GL_WATCH_KIB: usize = 1020;
const SHAPER_WATCH_KIB: usize = 320;

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

/// How `cargo xtask web` builds.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Options {
    /// `--names`: also core.wasm whole with its function names, to
    /// [`NAMED_CORE`], for `pnpm -C web coverage`.
    pub(crate) names: bool,
    /// `--panics`: panics say why on the console.
    pub(crate) panics: bool,
    /// `--gl`: core-gl.wasm alone, for work on WebGL 2. The split's lists
    /// go unchecked: an entry may name core.wasm's code alone.
    pub(crate) gl: bool,
}

pub(crate) fn build(options: Options) -> Result<(), String> {
    let Options { names, panics, gl } = options;
    let root = crate::root();
    let out = root.join(OUT_DIR);
    write_api_calls(&out)?;
    if gl {
        return build_gl(&root, panics).map(drop);
    }

    // A copy from an earlier build no longer matches web/pkg's: if its
    // imports still fit core.js, coverage would measure the old code.
    if !names {
        let _ = std::fs::remove_file(root.join(NAMED_CORE));
    }
    let core = module(&root, "orb-web", "core", &[], panics, Some(names))?;
    report_crates(&core.built);
    let js = std::fs::read(out.join("core.js")).map_err(|e| format!("core.js: {e}"))?;
    eprintln!(
        "core.wasm: {} KiB raw, {} KiB brotli ({} bytes); core.js {} KiB raw ({OUT_DIR})",
        core.raw / 1024,
        core.brotli / 1024,
        core.brotli,
        js.len() / 1024,
    );
    report_modules("core", &core.modules);
    watch("core.wasm", core.brotli, CORE_WATCH_KIB);
    let gl = build_gl(&root, panics)?;
    // An entry of the split's lists that names no function of either
    // build does nothing, or no longer does what it says.
    if let (Some(mut found), Some(gl)) = (core.found, gl) {
        found.extend(gl);
        let unmatched = found.unmatched();
        if !unmatched.is_empty() {
            return Err(format!(
                "{} entries of xtask/src/split.rs's lists name no function of core.wasm or core-gl.wasm: delete them, or name what they meant\n  {}",
                unmatched.len(),
                unmatched.join("\n  "),
            ));
        }
    }
    // `font-faces` drawing, loaded the first time a style declares a file.
    let shaper = module(&root, "orb-shaper", "shaper", &[], false, None)?;
    eprintln!(
        "shaper.wasm: {} KiB raw, {} KiB brotli (lazy)",
        shaper.raw / 1024,
        shaper.brotli / 1024
    );
    watch("shaper.wasm", shaper.brotli, SHAPER_WATCH_KIB);
    Ok(())
}

/// Writes [`API_CALLS`]: the bindings' methods whose code the API's
/// module holds ([`split::api_calls`]), which web/src/engine.ts makes wait
/// for the module.
fn write_api_calls(out: &Path) -> Result<(), String> {
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let path = out.join(API_CALLS);
    let json = serde_json::to_string(&split::api_calls()).map_err(|e| e.to_string())?;
    std::fs::write(&path, json + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// The same map on WebGL 2, fetched only where WebGPU has no adapter;
/// returns the entries of the split's lists that named its functions.
fn build_gl(root: &Path, panics: bool) -> Result<Option<split::Named>, String> {
    let features = ["--no-default-features", "--features", "gl"];
    let gl = module(root, "orb-web", "core-gl", &features, panics, Some(false))?;
    report_crates(&gl.built);
    eprintln!(
        "core-gl.wasm: {} KiB raw, {} KiB brotli ({} bytes; WebGL 2 only)",
        gl.raw / 1024,
        gl.brotli / 1024,
        gl.brotli,
    );
    report_modules("core-gl", &gl.modules);
    watch("core-gl.wasm", gl.brotli, CORE_GL_WATCH_KIB);
    Ok(gl.found)
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

/// Prints each feature module's size, and what it holds.
fn report_modules(build: &str, modules: &[Feature]) {
    let total: usize = modules.iter().map(|m| m.brotli).sum();
    eprintln!(
        "{build}'s feature modules ({} KiB brotli in all):",
        total / 1024
    );
    for m in modules {
        eprintln!(
            "  {build}-{}.wasm: {} KiB raw, {} KiB brotli, {} functions ({} KiB of code, {} bytes of core.wasm's data)",
            m.name,
            m.raw / 1024,
            m.brotli / 1024,
            m.functions,
            m.code / 1024,
            m.data,
        );
    }
}

pub(crate) fn pages() -> Result<(), String> {
    build(Options::default())?;
    run(Command::new("pnpm")
        .current_dir(crate::root().join("web"))
        .arg("build"))
}

struct Sizes {
    raw: usize,
    brotli: usize,
    /// The module cargo built, before wasm-opt, with its function names.
    built: std::path::PathBuf,
    /// A split module's feature modules.
    modules: Vec<Feature>,
    /// A split module's entries of the split's lists that named its
    /// functions ([`split::Named`]).
    found: Option<split::Named>,
}

/// A feature module of a split build.
struct Feature {
    name: &'static str,
    raw: usize,
    brotli: usize,
    functions: usize,
    /// Its functions' code bytes in the whole module.
    code: usize,
    /// The bytes of core.wasm's read-only data it carries.
    data: usize,
}

/// Builds `package` with `features` (cargo's feature arguments), binds it
/// as `<name>.js` and `<name>_bg.wasm` in [`OUT_DIR`] and shrinks it.
/// `split` splits it by feature into `<name>-<feature>.wasm` beside it
/// ([`split::MODULES`]); `Some(true)` also writes it whole with its
/// function names to [`NAMED_CORE`].
fn module(
    root: &Path,
    package: &str,
    name: &str,
    features: &[&str],
    panics: bool,
    split: Option<bool>,
) -> Result<Sizes, String> {
    // A panic traps at once, with no message. On wasm32-unknown-unknown
    // std's stderr is a sink: the message was formatted only to be dropped,
    // and that formatting, with the `Debug` impls and source locations it
    // reaches, was 5% of core.wasm. It takes std rebuilt with the module
    // and a flag stable Rust keeps unstable, so this build alone sets
    // RUSTC_BOOTSTRAP. Nothing reads a panic's location either: without
    // its file, line and column the modules carry no source paths of the
    // machine that built them. std is built for size too: its sorts are
    // a merge sort and a heapsort rather than driftsort and ipnsort,
    // which every element type instantiated anew (19 KiB of core.wasm),
    // and its number formatting is the compact one.
    let mut args = vec![
        "build",
        "--quiet",
        "-p",
        package,
        "--target",
        "wasm32-unknown-unknown",
        "--profile",
        "wasm-release",
        "-Zbuild-std=std,panic_abort",
        "-Zbuild-std-features=optimize_for_size",
    ];
    // `--panics`: messages, formatted and logged by orb-web's hook.
    if panics {
        args.extend(["--features", "panic-messages"]);
    } else {
        // A derived `Debug` writes only its type's or variant's name
        // (10 KiB): what a message must show has a `Display`, as a tile's
        // `z/col/row` does. LLVM folds no functions with one body into
        // one: the split takes each function by its name (split.rs). The
        // linker's relocations stay in the module, for the split to move
        // the data only a module's code reads (data.rs).
        args.extend([
            "--config",
            r#"target.wasm32-unknown-unknown.rustflags = ["-Zunstable-options", "-Cpanic=immediate-abort", "-Zlocation-detail=none", "-Zfmt-debug=shallow", "-Zmerge-functions=disabled", "-Clink-arg=--emit-relocs"]"#,
        ]);
    }
    args.extend(features);
    run(Command::new(env!("CARGO"))
        .env("RUSTC_BOOTSTRAP", "1")
        .args(&args))?;
    // Builds with other features share the path: each is read before the
    // next.
    let built = root.join(format!(
        "target/wasm32-unknown-unknown/wasm-release/{}.wasm",
        package.replace('-', "_")
    ));
    let out = root.join(OUT_DIR);
    run(Command::new("wasm-bindgen")
        .arg(&built)
        .args(["--target", "web", "--out-name", name, "--out-dir"])
        .arg(&out))?;
    expose_instantiation(&out, name)?;
    glue::shorten_imports(&out, name)?;
    glue::shorten_exports(&out, name)?;
    let wasm = out.join(format!("{name}_bg.wasm"));
    data::strip_relocations(&wasm)?;
    let read = |path: &Path| std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()));
    // A module built to say why it panics stays whole, its feature
    // modules empty, so its stacks name every function.
    let unsplit = panics && split.is_some();
    let split = split.filter(|_| !panics);
    if unsplit {
        for module in split::MODULES {
            let path = out.join(format!("{name}-{}.wasm", module.name));
            std::fs::write(&path, b"\0asm\x01\0\0\0")
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    let Some(names) = split else {
        // A module built to say why it panics names its functions in the
        // stacks it logs too (`-g`): it's never shipped.
        let mut opt = wasm_opt(&wasm, &wasm, &[]);
        if panics {
            opt.arg("-g");
        }
        run(&mut opt)?;
        let bytes = read(&wasm)?;
        return Ok(Sizes {
            raw: bytes.len(),
            brotli: brotli(&bytes)?,
            built,
            modules: Vec::new(),
            found: None,
        });
    };
    let named = root.join(NAMED_CORE);
    let dir = named.parent().expect("a file in a directory");
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    // binaryen folds functions with one body into one of their names,
    // which the split would then take for all of them: a closure of a
    // module's whose body is a closure of core.wasm's too would leave
    // core.wasm calling into the module. Whole, nothing is folded; the
    // split folds within each module (split.rs).
    // Nor does it inline a function the split's paths name into code
    // outside its module: so what reads a module's data is what read it
    // before (data.rs).
    let bytes = read(&wasm)?;
    // What the split's lists name, before wasm-opt inlines some of it and
    // after.
    let mut found = split::Named::default();
    let seeds = data::mark(
        &wasm,
        &split::not_to_inline(&bytes, &data::names(&bytes)?, &mut found)?,
    )?;
    let whole_opt = |output: &Path| {
        let mut opt = wasm_opt(
            &wasm,
            output,
            &[&format!("--no-inline={}*", data::NO_INLINE)],
        );
        opt.arg("--skip-pass=duplicate-function-elimination");
        opt
    };
    if names {
        run(whole_opt(&named).arg("-g"))?;
        eprintln!("{name}.wasm with its names: {}", named.display());
    }
    // Shrunk with its names, binaryen optimizes a little less: the split
    // takes them from the symbol map instead.
    let whole = root.join(format!("target/web/{name}-whole.wasm"));
    let symbols = whole.with_extension("symbols");
    run(whole_opt(&whole).arg(format!("--symbolmap={}", symbols.display())))?;
    let (split, paths) = split::split(
        &whole,
        &symbols,
        &wasm,
        &format!("{name}-"),
        FEATURES,
        (&built, &seeds),
        &mut found,
    )?;
    let bytes = read(&wasm)?;
    let mut modules = Vec::new();
    for (&(name, functions, code, data), path) in split.modules.iter().zip(&paths) {
        let module = read(path)?;
        modules.push(Feature {
            name,
            raw: module.len(),
            brotli: brotli(&module)?,
            functions,
            code,
            data,
        });
    }
    Ok(Sizes {
        raw: bytes.len(),
        brotli: brotli(&bytes)?,
        built,
        modules,
        found: Some(found),
    })
}

/// Exports the two steps of wasm-bindgen's `initSync` from `<name>.js`, so
/// web/src/builds.ts can give the instance imports of its own: a split
/// module's placeholders.
fn expose_instantiation(out: &Path, name: &str) -> Result<(), String> {
    let patch = |file: &str, from: &str, to: &str| {
        let path = out.join(file);
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if !text.contains(from) {
            return Err(format!(
                "{}: no `{from}`: wasm-bindgen's output changed, so update expose_instantiation",
                path.display()
            ));
        }
        std::fs::write(&path, text.replacen(from, to, 1))
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    patch(
        &format!("{name}.js"),
        "export { initSync, __wbg_init as default };",
        "export { initSync, __wbg_init as default, __wbg_get_imports, __wbg_finalize_init };",
    )?;
    patch(
        &format!("{name}.d.ts"),
        "export type SyncInitInput",
        "/** The imports an instance takes (exported by `cargo xtask web`). */\n\
         export function __wbg_get_imports(): WebAssembly.Imports;\n\
         /** Binds these bindings to `instance` of `module` and starts it. */\n\
         export function __wbg_finalize_init(instance: WebAssembly.Instance, module: WebAssembly.Module): InitOutput;\n\n\
         export type SyncInitInput",
    )
}

/// The wasm features the modules are built with.
const FEATURES: &[&str] = &[
    "--enable-simd",
    "--enable-bulk-memory",
    "--enable-sign-ext",
    "--enable-nontrapping-float-to-int",
    "--enable-multivalue",
    "--enable-reference-types",
];

/// Shrinks `input` into `output`, after the passes `first`.
fn wasm_opt(input: &Path, output: &Path, first: &[&str]) -> Command {
    let mut command = Command::new("wasm-opt");
    command
        .args(first)
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
        ])
        .args(FEATURES)
        .arg(input)
        .arg("-o")
        .arg(output);
    command
}

fn brotli(bytes: &[u8]) -> Result<usize, String> {
    let mut brotli = Vec::new();
    {
        let mut writer = brotli::CompressorWriter::new(&mut brotli, 4096, 11, 22);
        writer.write_all(bytes).map_err(|e| e.to_string())?;
    }
    Ok(brotli.len())
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

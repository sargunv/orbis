//! maplibre-native-ffi's C API on Orbis: the pinned mln-ffi mirror, the
//! declarations `orb-ffi` takes from it, and its C ABI suite scored against
//! Orbis's `maplibre-native-c`.
//!
//! The mirror in `fixtures/mln-ffi/` is not committed. It holds the pinned
//! commit's headers, its ABI suite and the GPU fixtures that suite drives
//! (`tests/native`, `tests/graphics`), the Rust -sys crate's declarations
//! (upstream's bindgen output from those headers), Unity, which the suite
//! is written in, and the Vulkan headers `tests/graphics` includes.
//!
//! mln-ffi builds its suite only inside its own CMake project, against the
//! library it builds there (maplibre/maplibre-native-ffi#1780), so this
//! module compiles it the way `cmake/mln_ffi_tests.cmake` does: the same
//! sources, file tags, registry and flags, linked against the shared
//! library from `cargo build -p orb-ffi`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::fixtures::{SOURCE_MD, entry_path, get, hex, install, open, unwrap_root};

/// The upstream repository and the commit Orbis implements.
const REPO: &str = "https://github.com/maplibre/maplibre-native-ffi";
const COMMIT: &str = "9cbe0b9c6388aabf3841cbd92dfafddf27ca76ee";

/// The mirror's content hash at [`COMMIT`], with [`UNITY`] and
/// [`VULKAN_HEADERS`].
const MIRROR_SHA256: &str = "c4bce2809b69a8c604c8e59be5cae7bc7d43bd37f0a7c1242d3d6db356200acc";

/// Unity, at the release and hash `cmake/mln_ffi_tests.cmake` pins.
const UNITY: (&str, &str) = (
    "https://github.com/ThrowTheSwitch/Unity/archive/refs/tags/v2.6.1.tar.gz",
    "b41a66d45a6b99758fb3202ace6178177014d52fc524bf1f72687d93e9867292",
);

/// The Vulkan headers `tests/graphics/graphics.c` includes on every
/// platform (mln-ffi takes them from MapLibre Native's vendor tree).
const VULKAN_HEADERS: &str =
    "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.309.tar.gz";

/// What the mirror keeps of the mln-ffi tarball, by path prefix.
const KEPT: [&str; 6] = [
    "include/",
    "tests/native/abi/",
    "tests/native/support/",
    "tests/native/plugin/",
    "tests/native/fixtures/",
    "tests/graphics/",
];
/// The -sys crate's declarations, mirrored as `sys/generated.rs`.
const SYS: &str = "bindings/rust/crates/maplibre-native-ffi-sys/src/generated.rs";

/// The ABI suite's domains in the order its registry runs them
/// (`MLN_NATIVE_ABI_DOMAINS`): plugin last, as a registration lasts for
/// the rest of the process.
const DOMAINS: [&str; 13] = [
    "base",
    "completion",
    "runtime",
    "resources",
    "map",
    "projection",
    "style",
    "render",
    "backend",
    "delivery",
    "binding",
    "platform",
    "plugin",
];

/// File-name tags that build a file only on matching targets
/// (`MLN_NATIVE_FILE_TAGS`).
const FILE_TAGS: [&str; 15] = [
    "metal",
    "opengl",
    "vulkan",
    "webgpu",
    "egl",
    "wgl",
    "webgl",
    "emscripten",
    "android",
    "ohos",
    "apple",
    "wayland",
    "linux",
    "windows",
    "icu",
];

/// The support sources every suite links (`mln_native_configure_suite`).
const SUPPORT: [&str; 12] = [
    "harness.c",
    "watchdog.cpp",
    "wait.cpp",
    "env.c",
    "render.c",
    "frames.c",
    "camera.c",
    "map.c",
    "style.c",
    "resources.c",
    "render_graphics.c",
    "http_server.c",
];

/// How long one run of the suite binary may take before it's stopped.
const RUN_TIMEOUT_SECS: u32 = 600;

/// GPU processes at once: each runtime the suite makes opens a device.
const PARALLEL_RUNS: usize = 2;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the workspace root")
        .to_path_buf()
}

fn mirror_dir() -> PathBuf {
    root().join("fixtures/mln-ffi")
}

/// Fetches the mirror into `fixtures/mln-ffi/` unless the one there matches
/// the pin.
pub(crate) fn ensure() -> Result<(), String> {
    install(
        "mln-ffi",
        COMMIT,
        MIRROR_SHA256,
        "xtask/src/mln_ffi.rs",
        mirror,
    )
}

/// Every regular file of a GitHub tarball, by its path below the root.
fn files(archive: &[u8], root: Option<&str>) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut files = BTreeMap::new();
    let mut entries = open(archive);
    for entry in entries.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry_path(&entry)?;
        let rel = match root {
            Some(root) => unwrap_root(&path, root)?,
            None => path.split_once('/').map(|(_, rest)| rest),
        };
        let Some(rel) = rel else { continue };
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|e| format!("{path}: {e}"))?;
        files.insert(rel.to_string(), bytes);
    }
    Ok(files)
}

fn write(out: &Path, rel: &str, bytes: &[u8]) -> Result<(), String> {
    let path = out.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Downloads the pinned tarballs and writes the mirror into `out`.
fn mirror(out: &Path) -> Result<(), String> {
    eprintln!("fetching {REPO} at {COMMIT}");
    let archive = get(&format!(
        "https://codeload.github.com/maplibre/maplibre-native-ffi/tar.gz/{COMMIT}"
    ))?;
    let mut kept = 0;
    for (rel, bytes) in files(&archive, Some(&format!("maplibre-native-ffi-{COMMIT}")))? {
        if KEPT.iter().any(|prefix| rel.starts_with(prefix)) || rel == "LICENSE" {
            write(out, &rel, &bytes)?;
            kept += 1;
        } else if rel == SYS {
            write(out, "sys/generated.rs", &bytes)?;
            kept += 1;
        }
    }
    if !out.join("sys/generated.rs").exists() || !out.join("include/maplibre_native_c.h").exists() {
        return Err(format!(
            "the tarball at {COMMIT} lacks {SYS} or the headers"
        ));
    }

    eprintln!("fetching {}", UNITY.0);
    let unity = get(UNITY.0)?;
    let hash = hex(&Sha256::digest(&unity));
    if hash != UNITY.1 {
        return Err(format!("Unity's tarball hashes to {hash}, not {}", UNITY.1));
    }
    for (rel, bytes) in files(&unity, None)? {
        if rel.starts_with("src/unity") || rel == "LICENSE.txt" {
            write(out, &format!("unity/{rel}"), &bytes)?;
        }
    }

    eprintln!("fetching {VULKAN_HEADERS}");
    for (rel, bytes) in files(&get(VULKAN_HEADERS)?, None)? {
        let header = rel.starts_with("include/vulkan/") || rel.starts_with("include/vk_video/");
        if (header && rel.ends_with(".h")) || rel == "LICENSE.md" {
            write(out, &format!("vulkan/{rel}"), &bytes)?;
        }
    }

    fs::write(
        out.join(SOURCE_MD),
        format!(
            "# maplibre-native-ffi's C API and ABI suite\n\n\
             Mirrored by `cargo xtask fixtures mln-ffi` from {REPO} at {COMMIT}: \
             {kept} files of `include/`, `tests/native/` and `tests/graphics/`, and the \
             Rust -sys crate's declarations as `sys/generated.rs`. `unity/` is {}, and \
             `vulkan/` the headers of {VULKAN_HEADERS}.\n",
            UNITY.0
        ),
    )
    .map_err(|e| e.to_string())?;
    eprintln!("mirrored {kept} files");
    Ok(())
}

// MARK: - Declarations

fn crate_src() -> PathBuf {
    root().join("crates/orb-ffi/src")
}

/// `cargo xtask mln-ffi declarations [--check]`: copies the mirror's
/// declarations into `orb-ffi` and generates its stubs, or checks that
/// both are up to date.
pub(crate) fn declarations(check: bool) -> Result<(), String> {
    ensure()?;
    let generated = fs::read_to_string(mirror_dir().join("sys/generated.rs"))
        .map_err(|e| format!("the mirror's sys/generated.rs: {e}"))?;
    let stubs = stubs(&generated)?;
    let outputs = [
        (crate_src().join("sys/generated.rs"), generated),
        (crate_src().join("stubs.rs"), stubs),
    ];
    let mut stale = Vec::new();
    for (path, text) in &outputs {
        if fs::read_to_string(path).ok().as_deref() != Some(text.as_str()) {
            if check {
                stale.push(
                    path.strip_prefix(root())
                        .unwrap_or(path)
                        .display()
                        .to_string(),
                );
            } else {
                fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
                eprintln!("wrote {}", path.display());
            }
        }
    }
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} out of date with the mln-ffi pin; `cargo xtask mln-ffi declarations` \
             regenerates them",
            stale.join(" and ")
        ))
    }
}

/// `cargo xtask check`'s step: the declarations match the pin, when the
/// mirror is there (fetching it needs the network).
pub(crate) fn check_declarations() -> Result<(), String> {
    if !mirror_dir().exists() {
        eprintln!(
            "check: no fixtures/mln-ffi (`cargo xtask fixtures mln-ffi`), so orb-ffi's \
             declarations aren't checked against the pin"
        );
        return Ok(());
    }
    declarations(true)
}

/// One function of the -sys crate's extern block.
struct Function {
    name: String,
    /// Its declaration without `pub fn ` and the `;`.
    signature: String,
    returns: Option<String>,
    diagnostic: bool,
}

/// The functions `generated.rs` declares, in its order.
fn functions(generated: &str) -> Result<Vec<Function>, String> {
    let start = generated
        .find("unsafe extern \"C\" {\n")
        .ok_or("generated.rs has no extern block")?;
    let block = &generated[start..];
    let block = &block[block.find('\n').unwrap_or(0) + 1..];
    let end = block
        .find("\n}\n")
        .ok_or("generated.rs's extern block doesn't end")?;
    let mut functions = Vec::new();
    for item in block[..end].split(';') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let signature = item
            .strip_prefix("pub fn ")
            .ok_or_else(|| format!("unexpected item in the extern block: {item}"))?
            .to_string();
        let name =
            signature[..signature.find('(').ok_or("a function without parameters")?].to_string();
        // The return type follows the parameters' closing parenthesis.
        let mut depth = 0;
        let mut close = None;
        for (i, c) in signature.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.ok_or_else(|| format!("{name}'s parameters don't close"))?;
        let returns = signature[close + 1..]
            .trim()
            .strip_prefix("->")
            .map(|r| r.trim().to_string());
        let diagnostic = signature[..close].contains("out_diagnostic: *mut mln_diagnostic");
        functions.push(Function {
            name,
            signature,
            returns,
            diagnostic,
        });
    }
    Ok(functions)
}

/// The module path of every function `orb-ffi` exports by hand.
fn implemented() -> Result<BTreeMap<String, String>, String> {
    let mut found = BTreeMap::new();
    let src = crate_src();
    let mut stack = vec![src.clone()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n != "sys") {
                    stack.push(path);
                }
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") || path.ends_with("stubs.rs") {
                continue;
            }
            let rel = path.strip_prefix(&src).expect("under src");
            let mut module: Vec<String> = rel
                .with_extension("")
                .iter()
                .map(|c| c.to_string_lossy().into_owned())
                .collect();
            if module.last().is_some_and(|m| m == "mod" || m == "lib") {
                module.pop();
            }
            let module = module.join("::");
            let text = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            // An export is an `extern "C" fn`, or a `command!` that
            // expands to one.
            let exports = text
                .match_indices("extern \"C\" fn mln_")
                .map(|(i, _)| i + "extern \"C\" fn ".len())
                .chain(text.match_indices("command!(").map(|(i, _)| {
                    let start = i + "command!(".len();
                    start + text[start..].len() - text[start..].trim_start().len()
                }))
                .filter(|&i| text[i..].starts_with("mln_"));
            for i in exports {
                let rest = &text[i..];
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if let Some(other) = found.insert(name.clone(), module.clone()) {
                    return Err(format!("{name} is exported by both {other} and {module}"));
                }
            }
        }
    }
    Ok(found)
}

/// `stubs.rs`: every function `orb-ffi` doesn't export by hand, as a stub
/// that returns `MLN_STATUS_UNSUPPORTED`, and a check that the ones it
/// does export have the headers' signatures.
fn stubs(generated: &str) -> Result<String, String> {
    let functions = functions(generated)?;
    let done = implemented()?;
    let declared: BTreeSet<&str> = functions.iter().map(|f| f.name.as_str()).collect();
    for name in done.keys() {
        if !declared.contains(name.as_str()) {
            return Err(format!(
                "orb-ffi exports {name}, which the headers don't declare"
            ));
        }
    }
    let mut out = String::from(
        "// @generated by `cargo xtask mln-ffi declarations` from the pinned\n\
         // headers' declarations (sys/generated.rs); do not edit.\n\
         //\n\
         // The C API functions orb-ffi doesn't implement. Each keeps its C\n\
         // signature, so callers of the whole API link, and returns\n\
         // MLN_STATUS_UNSUPPORTED with a diagnostic (or does nothing). At the end, a\n\
         // check that the functions it does implement have the headers' signatures.\n\
         \n\
         #![allow(unused_variables, clippy::missing_safety_doc)]\n\
         \n\
         use crate::sys::*;\n\n",
    );
    let stubbed: Vec<&Function> = functions
        .iter()
        .filter(|f| !done.contains_key(&f.name))
        .collect();
    for f in &stubbed {
        let body = if f.diagnostic {
            format!(
                "unsafe {{ crate::base::unsupported(out_diagnostic, \"{}\") }}",
                f.name
            )
        } else {
            match f.returns.as_deref() {
                None => String::new(),
                Some("mln_status") => "MLN_STATUS_UNSUPPORTED".into(),
                Some(r) if r.starts_with("*mut ") => "std::ptr::null_mut()".into(),
                Some(r) if r.starts_with("*const ") => "std::ptr::null()".into(),
                Some(_) => "Default::default()".into(),
            }
        };
        out.push_str(&format!(
            "/// `{}`: not implemented.\n#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn {} {{\n{body}\n}}\n\n",
            f.name, f.signature
        ));
    }
    out.push_str(
        "// A function item coerces to a function pointer type it shares with another\n\
         // only if their signatures match, so this fails to compile where a\n\
         // hand-written export differs from the header's declaration.\n\
         const _: () = {\n",
    );
    for f in &functions {
        if let Some(module) = done.get(&f.name) {
            out.push_str(&format!(
                "let _ = [crate::sys::{0}, crate::{module}::{0}];\n",
                f.name
            ));
        }
    }
    out.push_str("};\n");
    rustfmt(&out)
}

fn rustfmt(text: &str) -> Result<String, String> {
    use std::io::Write;
    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("rustfmt: {e}"))?;
    child
        .stdin
        .take()
        .expect("piped")
        .write_all(text.as_bytes())
        .map_err(|e| e.to_string())?;
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("rustfmt failed on the generated stubs".into());
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

// MARK: - The ABI suite

/// The file tags this platform builds, as mln-ffi's preset for it does:
/// Metal on Apple platforms, Vulkan elsewhere.
fn active_tags() -> Vec<&'static str> {
    if cfg!(target_vendor = "apple") {
        vec!["metal", "apple"]
    } else if cfg!(target_os = "linux") {
        vec!["vulkan", "wayland", "linux", "icu"]
    } else if cfg!(windows) {
        vec!["vulkan", "windows", "icu"]
    } else {
        vec!["vulkan"]
    }
}

fn selected(rel: &str, active: &[&str]) -> bool {
    let stem = Path::new(rel)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    match stem.rsplit_once('_') {
        Some((_, tag)) if FILE_TAGS.contains(&tag) => active.contains(&tag),
        _ => true,
    }
}

/// The ABI suite's files for this platform, relative to `tests/native/abi`,
/// in registry order, with the group symbol each defines.
fn abi_files(mirror: &Path) -> Result<Vec<(String, String)>, String> {
    let active = active_tags();
    let mut out = Vec::new();
    for domain in DOMAINS {
        let dir = mirror.join("tests/native/abi").join(domain);
        let mut names: Vec<String> = fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.ends_with(".c"))
            .collect();
        names.sort();
        for name in names {
            let rel = format!("{domain}/{name}");
            if !selected(&rel, &active) {
                continue;
            }
            let stem = name.trim_end_matches(".c");
            let symbol: String = format!("mln_test_group_{domain}_{stem}")
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect();
            out.push((rel, symbol));
        }
    }
    Ok(out)
}

fn work_dir() -> PathBuf {
    root().join("target/mln-ffi")
}

/// A C or C++ compiler command. On macOS it takes the SDK `rustc` links
/// with (`xcrun --sdk macosx`), not the one `cc` defaults to, which can be
/// a Command Line Tools SDK newer than the linker reads.
fn compiler(program: &str) -> Command {
    let mut command = Command::new(program);
    if cfg!(target_os = "macos") && std::env::var_os("SDKROOT").is_none() {
        static SDK: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let sdk = SDK.get_or_init(|| {
            let output = Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        });
        if let Some(sdk) = sdk {
            command.env("SDKROOT", sdk);
        }
    }
    command
}

fn run_tool(command: &mut Command) -> Result<(), String> {
    let output = command.output().map_err(|e| format!("{command:?}: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{command:?} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Builds `maplibre-native-c` and installs it as mln-ffi's install
/// prefix lays it out: `lib/`, `include/`.
fn install_library(mirror: &Path) -> Result<PathBuf, String> {
    crate::cargo(&["build", "--release", "-p", "orb-ffi"])?;
    let prefix = work_dir().join("install");
    let lib = prefix.join("lib");
    fs::create_dir_all(&lib).map_err(|e| e.to_string())?;
    let (built, installed) = if cfg!(target_vendor = "apple") {
        ("libmaplibre_native_c.dylib", "libmaplibre-native-c.dylib")
    } else if cfg!(windows) {
        ("maplibre_native_c.dll", "maplibre-native-c.dll")
    } else {
        ("libmaplibre_native_c.so", "libmaplibre-native-c.so")
    };
    let from = root().join("target/release").join(built);
    fs::copy(&from, lib.join(installed)).map_err(|e| format!("{}: {e}", from.display()))?;
    let include = prefix.join("include");
    if include.exists() {
        fs::remove_dir_all(&include).map_err(|e| e.to_string())?;
    }
    copy_tree(&mirror.join("include"), &include)?;
    Ok(prefix)
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    fs::create_dir_all(to).map_err(|e| format!("{}: {e}", to.display()))?;
    for entry in fs::read_dir(from).map_err(|e| format!("{}: {e}", from.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), &target).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Compiles the ABI suite against the install at `prefix` and links it:
/// the suite's executable.
fn build_suite(mirror: &Path, prefix: &Path) -> Result<PathBuf, String> {
    let build = work_dir().join("suite");
    if build.exists() {
        fs::remove_dir_all(&build).map_err(|e| e.to_string())?;
    }
    let objects_dir = build.join("obj");
    fs::create_dir_all(&objects_dir).map_err(|e| e.to_string())?;
    let native = mirror.join("tests/native");
    let support = native.join("support");

    let files = abi_files(mirror)?;
    let registry: String = files
        .iter()
        .map(|(rel, symbol)| {
            format!("MLN_TEST_REGISTRY_ENTRY({symbol}, \"tests/native/abi/{rel}\")\n")
        })
        .collect();
    fs::write(build.join("mln_native_test_registry.inc"), registry).map_err(|e| e.to_string())?;

    let backend = if cfg!(target_vendor = "apple") {
        "MLN_FFI_TEST_BACKEND_METAL=1"
    } else {
        "MLN_FFI_TEST_BACKEND_VULKAN=1"
    };
    let common: Vec<String> = vec![
        "-O1".into(),
        "-g".into(),
        "-DUNITY_INCLUDE_CONFIG_H".into(),
        format!("-D{backend}"),
        format!("-I{}", native.display()),
        format!("-I{}", support.display()),
        format!("-I{}", build.display()),
        format!("-I{}", mirror.join("unity/src").display()),
        format!("-I{}", mirror.join("tests/graphics/include").display()),
        format!("-I{}", native.join("plugin").display()),
        format!("-isystem{}", prefix.join("include").display()),
        format!("-isystem{}", mirror.join("vulkan/include").display()),
        "-Wno-everything".into(),
    ];

    // (source, extra flags) for every translation unit.
    let mut units: Vec<(PathBuf, Vec<String>)> = Vec::new();
    for (rel, symbol) in &files {
        units.push((
            native.join("abi").join(rel),
            vec![format!("-DMLN_TEST_GROUP_NAME={symbol}")],
        ));
    }
    for file in SUPPORT {
        units.push((support.join(file), Vec::new()));
    }
    if cfg!(target_os = "linux") {
        units.push((support.join("wayland_compositor.c"), Vec::new()));
    }
    units.push((
        mirror.join("tests/graphics/graphics.c"),
        vec![
            "-DMLN_TEST_GRAPHICS_EXPORTS".into(),
            "-fvisibility=hidden".into(),
        ],
    ));
    units.push((mirror.join("unity/src/unity.c"), Vec::new()));

    let objects = Mutex::new(Vec::new());
    let errors = Mutex::new(Vec::new());
    let queue = Mutex::new(units.into_iter().enumerate().collect::<Vec<_>>());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                while let Some((i, (source, extra))) = queue.lock().expect("queue").pop() {
                    let cpp = source.extension().is_some_and(|e| e == "cpp");
                    let object = objects_dir.join(format!(
                        "{i}-{}.o",
                        source.file_stem().unwrap_or_default().to_string_lossy()
                    ));
                    let mut command = compiler(if cpp { "c++" } else { "cc" });
                    command
                        .arg(if cpp { "-std=c++2b" } else { "-std=c2x" })
                        .args(&common)
                        .args(&extra)
                        .arg("-c")
                        .arg(&source)
                        .arg("-o")
                        .arg(&object);
                    match run_tool(&mut command) {
                        Ok(()) => objects.lock().expect("objects").push(object),
                        Err(e) => errors.lock().expect("errors").push(e),
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().expect("errors");
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    // The plugin the plugin group registers, a library of its own.
    let lib = prefix.join("lib");
    let plugin = lib.join(if cfg!(target_vendor = "apple") {
        "libmln-native-test-plugin.dylib"
    } else {
        "libmln-native-test-plugin.so"
    });
    run_tool(
        compiler("cc")
            .args(["-std=c2x", "-shared", "-fPIC", "-fvisibility=hidden"])
            .arg("-DMLN_NATIVE_TEST_PLUGIN_BUILDING")
            .arg(format!("-I{}", prefix.join("include").display()))
            .arg(format!("-I{}", native.join("plugin").display()))
            .arg(native.join("plugin/square_plugin.c"))
            .arg("-o")
            .arg(&plugin),
    )?;

    let exe = build.join("mln_native_abi_tests");
    let mut link = compiler("c++");
    link.args(objects.into_inner().expect("objects"))
        .arg("-o")
        .arg(&exe)
        .arg(format!("-L{}", lib.display()))
        .args(["-lmaplibre-native-c", "-lmln-native-test-plugin"])
        .arg(format!("-Wl,-rpath,{}", lib.display()))
        .arg("-lpthread");
    if cfg!(target_vendor = "apple") {
        link.args([
            "-framework",
            "Foundation",
            "-framework",
            "IOSurface",
            "-framework",
            "Metal",
            "-lobjc",
        ]);
    } else {
        link.arg("-ldl");
    }
    run_tool(&mut link)?;
    Ok(exe)
}

/// A case's outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail(String),
    Ignore(String),
    /// The process died or hung in the case.
    Crash(String),
}

impl Outcome {
    fn label(&self) -> &'static str {
        match self {
            Outcome::Pass => "pass",
            Outcome::Fail(_) => "fail",
            Outcome::Ignore(_) => "ignore",
            Outcome::Crash(_) => "crash",
        }
    }

    fn message(&self) -> &str {
        match self {
            Outcome::Pass => "",
            Outcome::Fail(m) | Outcome::Ignore(m) | Outcome::Crash(m) => m,
        }
    }
}

/// The suite's cases, `file` (`tests/native/abi/…`) to its case names in
/// run order.
fn list_cases(exe: &Path) -> Result<BTreeMap<String, Vec<String>>, String> {
    let output = Command::new(exe)
        .arg("-l")
        .output()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    let mut cases: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some((file, case)) = line.trim().rsplit_once(':')
            && file.starts_with("tests/native/")
        {
            cases
                .entry(file.to_string())
                .or_default()
                .push(case.to_string());
        }
    }
    if cases.is_empty() {
        return Err(format!(
            "the suite listed no cases:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(cases)
}

/// Parses Unity's result lines: `file:line:case:PASS`, `…:FAIL: why`,
/// `…:IGNORE`, each maybe ending in ` (n ms)`.
fn parse_results(stdout: &str, file: &str) -> BTreeMap<String, Outcome> {
    let mut out = BTreeMap::new();
    for line in stdout.lines() {
        let Some(rest) = line.strip_prefix(file).and_then(|r| r.strip_prefix(':')) else {
            continue;
        };
        let mut parts = rest.splitn(3, ':');
        let (Some(_line), Some(case), Some(result)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let result = match result.rfind(" (") {
            Some(i) if result.ends_with(" ms)") => &result[..i],
            _ => result,
        };
        let outcome = if result == "PASS" {
            Outcome::Pass
        } else if let Some(why) = result.strip_prefix("FAIL") {
            Outcome::Fail(why.trim_start_matches(':').trim().to_string())
        } else if let Some(why) = result.strip_prefix("IGNORE") {
            Outcome::Ignore(why.trim_start_matches(':').trim().to_string())
        } else {
            // A failure message that holds a colon: Unity writes
            // `case:FAIL: message` after the line number, so a `FAIL`
            // later in the line is the result.
            match result.find(":FAIL") {
                Some(i) => Outcome::Fail(result[i + 5..].trim_start_matches(':').trim().into()),
                None => continue,
            }
        };
        out.insert(case.to_string(), outcome);
    }
    out
}

/// Runs one file's cases, again past any that crash, and returns each
/// case's outcome.
fn run_file(exe: &Path, mirror: &Path, file: &str, cases: &[String]) -> BTreeMap<String, Outcome> {
    let mut outcomes = BTreeMap::new();
    let mut remaining: Vec<String> = cases.to_vec();
    let mut first = true;
    while !remaining.is_empty() {
        // The first run names the file; later ones the cases not yet run.
        let filter = if first {
            format!("/{}", file.trim_start_matches("tests/native/"))
        } else {
            remaining.join(",")
        };
        first = false;
        let output = Command::new("perl")
            .args(["-e", "alarm shift; exec @ARGV"])
            .arg(RUN_TIMEOUT_SECS.to_string())
            .arg(exe)
            .args(["-f", &filter])
            .env(
                "MLN_FFI_TEST_FIXTURE_DIR",
                mirror.join("tests/native/fixtures"),
            )
            .stdin(Stdio::null())
            .output();
        let output = match output {
            Ok(output) => output,
            Err(e) => {
                for case in remaining.drain(..) {
                    outcomes.insert(case, Outcome::Crash(format!("couldn't run: {e}")));
                }
                break;
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let results = parse_results(&stdout, file);
        for (case, outcome) in &results {
            outcomes.insert(case.clone(), outcome.clone());
        }
        remaining.retain(|c| !results.contains_key(c));
        if remaining.is_empty() {
            break;
        }
        // The first case without a result is where the run stopped.
        let crashed = remaining.remove(0);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(3).collect();
        let why = format!(
            "{} {}",
            match output.status.code() {
                Some(code) => format!("exit {code}"),
                None => "killed by a signal (or the timeout)".into(),
            },
            tail.into_iter().rev().collect::<Vec<_>>().join(" / ")
        );
        outcomes.insert(crashed, Outcome::Crash(why.trim().to_string()));
    }
    outcomes
}

/// `cargo xtask mln-ffi [--filter <s>] [--verbose]`: builds Orbis's
/// `maplibre-native-c`, compiles the pinned ABI suite against it, runs it
/// file by file and scores it case by case.
pub(crate) fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("declarations") => return declarations(args.iter().any(|a| a == "--check")),
        Some("help") => {
            eprintln!(
                "cargo xtask mln-ffi [--filter <substring>] [--verbose]\n\
                 cargo xtask mln-ffi declarations [--check]"
            );
            return Ok(());
        }
        _ => {}
    }
    let mut filter = None;
    let mut verbose = false;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--filter" => filter = Some(iter.next().ok_or("--filter takes a substring")?.clone()),
            "--verbose" => verbose = true,
            other => return Err(format!("mln-ffi: unknown argument `{other}`")),
        }
    }
    ensure()?;
    declarations(true)?;
    let mirror = mirror_dir();
    let prefix = install_library(&mirror)?;
    eprintln!("building the ABI suite against {}", prefix.display());
    let exe = build_suite(&mirror, &prefix)?;
    let cases = list_cases(&exe)?;
    let files: Vec<(&String, &Vec<String>)> = cases
        .iter()
        .filter(|(file, _)| filter.as_ref().is_none_or(|f| file.contains(f.as_str())))
        .collect();
    let total: usize = files.iter().map(|(_, c)| c.len()).sum();
    eprintln!(
        "running {total} cases in {} files, {PARALLEL_RUNS} at a time",
        files.len()
    );

    let results: Mutex<BTreeMap<String, BTreeMap<String, Outcome>>> = Mutex::default();
    let queue = Mutex::new(files.clone());
    std::thread::scope(|scope| {
        for _ in 0..PARALLEL_RUNS {
            scope.spawn(|| {
                while let Some((file, cases)) = queue.lock().expect("queue").pop() {
                    let outcomes = run_file(&exe, &mirror, file, cases);
                    let passed = outcomes.values().filter(|o| **o == Outcome::Pass).count();
                    eprintln!("{file}: {passed}/{}", cases.len());
                    results
                        .lock()
                        .expect("results")
                        .insert(file.clone(), outcomes);
                }
            });
        }
    });
    let results = results.into_inner().expect("results");
    report(&results, &cases, verbose, filter.is_none())
}

/// Prints the score by domain and file, compares it with the last full
/// run's, and saves it in `target/mln-ffi/results.json`.
fn report(
    results: &BTreeMap<String, BTreeMap<String, Outcome>>,
    cases: &BTreeMap<String, Vec<String>>,
    verbose: bool,
    full: bool,
) -> Result<(), String> {
    let mut by_domain: BTreeMap<&str, [usize; 5]> = BTreeMap::new();
    for (file, outcomes) in results {
        let domain = file
            .trim_start_matches("tests/native/abi/")
            .split('/')
            .next()
            .unwrap_or_default();
        let row = by_domain.entry(domain).or_default();
        for case in &cases[file] {
            let i = match outcomes.get(case) {
                Some(Outcome::Pass) => 0,
                Some(Outcome::Fail(_)) => 1,
                Some(Outcome::Ignore(_)) => 2,
                Some(Outcome::Crash(_)) | None => 3,
            };
            row[i] += 1;
            row[4] += 1;
        }
        if verbose {
            for case in &cases[file] {
                if let Some(o) = outcomes.get(case)
                    && *o != Outcome::Pass
                {
                    println!("  {file}:{case}: {} {}", o.label(), o.message());
                }
            }
        }
    }
    println!(
        "\n{:<12} {:>5} {:>5} {:>6} {:>5} {:>5}",
        "domain", "pass", "fail", "ignore", "crash", "all"
    );
    let mut sum = [0; 5];
    for (domain, row) in &by_domain {
        println!(
            "{domain:<12} {:>5} {:>5} {:>6} {:>5} {:>5}",
            row[0], row[1], row[2], row[3], row[4]
        );
        for (s, r) in sum.iter_mut().zip(row) {
            *s += r;
        }
    }
    println!(
        "{:<12} {:>5} {:>5} {:>6} {:>5} {:>5}",
        "total", sum[0], sum[1], sum[2], sum[3], sum[4]
    );

    let path = work_dir().join("results.json");
    let json: Value = results
        .iter()
        .map(|(file, outcomes)| {
            let cases: serde_json::Map<String, Value> = outcomes
                .iter()
                .map(|(case, o)| (case.clone(), json!([o.label(), o.message()])))
                .collect();
            (file.clone(), Value::Object(cases))
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
    if full {
        if let Some(before) = fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        {
            let passed = |v: &Value, file: &str, case: &str| v[file][case][0] == "pass";
            let mut lost = Vec::new();
            let mut gained = 0;
            for (file, outcomes) in results {
                for case in outcomes.keys() {
                    match (passed(&before, file, case), passed(&json, file, case)) {
                        (true, false) => lost.push(format!("{file}:{case}")),
                        (false, true) => gained += 1,
                        _ => {}
                    }
                }
            }
            println!(
                "since the last full run: {gained} more pass, {} no longer do",
                lost.len()
            );
            for case in lost {
                println!("  no longer passes: {case}");
            }
        }
        fs::write(
            &path,
            serde_json::to_string_pretty(&json).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

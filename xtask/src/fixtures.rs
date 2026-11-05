//! Fetches the MapLibre GL JS fixtures that Orbis tests against.
//!
//! The mirror in `fixtures/gl-js/` is not committed. `cargo xtask fixtures`
//! builds it from the pinned commit's tarball, checks it against
//! [`MIRROR_SHA256`] and moves it into place; when the mirror there already
//! matches, it does nothing.

use std::collections::BTreeSet;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use base64::Engine;
use flate2::read::GzDecoder;
use serde_json::Value;
use sha2::{Digest, Sha256, Sha512};
use tar::Archive;

/// The upstream repository and the commit the mirror is pinned to.
const REPO: &str = "https://github.com/maplibre/maplibre-gl-js";
const COMMIT: &str = "2b68611a4e2d4f7f076f6c6ebf758e2d53f91471";

/// The mirror's [`content_hash`] at [`COMMIT`]. Changing the pin changes it:
/// a fetch that doesn't match reports the hash it got, for review.
const MIRROR_SHA256: &str = "490ca5fe904f4cd58af1e9a9921806fc3be2144c2bfe38e2a1b8749f1f3feb4f";

/// The test directory prefixes the mirrored milestones need, for the render and
/// query suites alike. Matching is by prefix, so `fill-` already covers
/// `fill-extrusion-*`; `color-relief` and `sprites` have no trailing dash
/// because upstream names those directories exactly that. The style
/// engine's directories (`runtime-styling` and the feature and global state
/// ones) drive maps through `operations`.
const PREFIXES: [&str; 25] = [
    "background-",
    "circle-",
    "color-relief",
    "feature-state",
    "fill-",
    "filter",
    "global-state",
    "heatmap-",
    "hillshade-",
    "icon-",
    // `image` and `video` sources.
    "image",
    "light",
    "line-",
    // The globe and its vertical perspective, and the
    // identity and Mercator projections' tests beside them.
    "projection",
    "raster-",
    "remove-feature-state",
    "runtime-styling",
    "sprites",
    "sky",
    "symbol-",
    "terrain",
    "text-",
    "video",
    "visibility",
    "zoom-history",
];

const TESTS: &str = "test/integration/render/tests/";
const QUERY: &str = "test/integration/query/tests/";
const ASSETS: &str = "test/integration/assets/";
const GLYPHS: &str = "glyphs/";

/// The generated description inside the mirror, left out of its hash.
const SOURCE_MD: &str = "SOURCE.md";

fn fixtures_dir() -> PathBuf {
    let xtask = Path::new(env!("CARGO_MANIFEST_DIR"));
    xtask.parent().expect("the workspace root").join("fixtures")
}

/// Khronos glTF sample models (glTF-Sample-Assets at [`GLTF_COMMIT`]) for
/// `orb-3d`'s model suite, each `Models/<name>/glTF-Binary/<name>.glb` with
/// its SHA-256.
const GLTF_COMMIT: &str = "8f492390ddc7a2ae221a1369ef560589cf8c70fb";
const GLTF_SAMPLES: [(&str, &str); 12] = [
    (
        "AlphaBlendModeTest",
        "37c3577d143071b42dd46e9d942b157837eb25c6340112171d7faecaa987b14e",
    ),
    (
        "Avocado",
        "ccc9c3ce56423720b09399c2351537207cd5a65f859f9e6e2f30922762f3abd4",
    ),
    (
        "BoomBox",
        "f8b918445ebdd006768232205a62f5182d2208ca57f84c6ccc084943c0bc8f15",
    ),
    (
        "Box",
        "ed52f7192b8311d700ac0ce80644e3852cd01537e4d62241b9acba023da3d54e",
    ),
    (
        "BoxTextured",
        "b510eca2e2ef33f62f9ed57d6e7ce2d10ebb2bdebc4a8e59d347719ba81abdf4",
    ),
    (
        "DamagedHelmet",
        "a1e3b04de97b11de564ce6e53b95f02954a297f0008183ac63a4f5974f6b32d8",
    ),
    (
        "Duck",
        "65bf938f54d6073e619e76e007820bbf980cdc3dc0daec0d94830ffc4ae54ab5",
    ),
    (
        "Fox",
        "d97044e701822bac5a62696459b27d7b375aada5de8574ed4362edbba94771f7",
    ),
    (
        "MetalRoughSpheres",
        "450c05557b0823a10835e32c05caf77a7b175eba94516555c1c9bb084f563f01",
    ),
    (
        "NormalTangentMirrorTest",
        "31ce5f3c873fc55531a17ccacf001e3d1d482a508695aa826fa79cc196e0ce78",
    ),
    (
        "OrientationTest",
        "a91cf448f37de06ab61bc615e123692dd29ac185e6c69e1fdbf1cf53e41045b2",
    ),
    (
        "TextureCoordinateTest",
        "fe75a63a0423c9a682bf46c3045b6328ae9c55abefc9a50adf3ec1d6dc6ea3b9",
    ),
];

/// Fetches the glTF samples missing from `fixtures/gltf/` or differing
/// from their pins.
fn ensure_gltf() -> Result<(), String> {
    let out = fixtures_dir().join("gltf");
    fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (name, sha) in GLTF_SAMPLES {
        let path = out.join(format!("{name}.glb"));
        if fs::read(&path).is_ok_and(|bytes| hex(&Sha256::digest(&bytes)) == sha) {
            continue;
        }
        let url = format!(
            "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/{GLTF_COMMIT}/Models/{name}/glTF-Binary/{name}.glb"
        );
        let bytes = get(&url)?;
        let got = hex(&Sha256::digest(&bytes));
        if got != sha {
            return Err(format!("{name}.glb hashes to {got}, not the pinned {sha}"));
        }
        fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    eprintln!("fixtures/gltf matches glTF-Sample-Assets {GLTF_COMMIT}");
    Ok(())
}

/// Fetches the mirror into `fixtures/gl-js/` unless the one there matches the
/// pin, and the glTF samples into `fixtures/gltf/`.
pub(crate) fn ensure() -> Result<(), String> {
    ensure_gltf()?;
    let out = fixtures_dir().join("gl-js");
    if out.exists() {
        let hash = content_hash(&out)?;
        if hash == MIRROR_SHA256 {
            eprintln!("fixtures/gl-js matches {COMMIT} (sha256 {hash})");
            return Ok(());
        }
        eprintln!("fixtures/gl-js differs from the pin (sha256 {hash}); fetching it again");
    }
    let staging = fixtures_dir().join("gl-js.partial");
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    }
    mirror(&staging)?;
    let hash = content_hash(&staging)?;
    if hash != MIRROR_SHA256 {
        return Err(format!(
            "the mirror built from {COMMIT} hashes to {hash}, not the pinned {MIRROR_SHA256}; \
             it is left in {} for review. If the pin changed on purpose, set MIRROR_SHA256 \
             in xtask/src/fixtures.rs to the new hash.",
            staging.display()
        ));
    }
    if out.exists() {
        fs::remove_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    fs::rename(&staging, &out).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!("fixtures/gl-js verified (sha256 {hash})");
    Ok(())
}

/// SHA-256 over every file in the mirror but `SOURCE.md`, in path order:
/// each file's `/`-separated relative path, a NUL, its length as a
/// little-endian `u64`, then its bytes.
fn content_hash(dir: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    walk(dir, String::new(), &mut files)?;
    files.sort();
    let mut hasher = Sha256::new();
    for rel in files.iter().filter(|rel| *rel != SOURCE_MD) {
        let bytes = fs::read(dir.join(rel)).map_err(|e| format!("{rel}: {e}"))?;
        hasher.update(rel.as_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(hex(&hasher.finalize()))
}

fn walk(dir: &Path, prefix: String, files: &mut Vec<String>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|name| format!("non-UTF-8 file name {name:?}"))?;
        let rel = format!("{prefix}{name}");
        if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            walk(&entry.path(), format!("{rel}/"), files)?;
        } else {
            files.push(rel);
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn get(url: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    ureq::get(url)
        .call()
        .map_err(|e| format!("failed to fetch {url}: {e}"))?
        .body_mut()
        .with_config()
        .limit(u64::MAX)
        .reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("failed to download {url}: {e}"))?;
    Ok(bytes)
}

/// Downloads the pinned tarball and writes the mirror into `out`.
fn mirror(out: &Path) -> Result<(), String> {
    eprintln!("fetching {REPO} at {COMMIT}");
    let url = format!("https://codeload.github.com/maplibre/maplibre-gl-js/tar.gz/{COMMIT}");
    let archive = get(&url)?;

    let scanned = scan(&archive)?;
    let mut expanded = expand(&scanned.styles, &scanned.assets);

    let mut written = extract(&archive, out, &expanded.wanted)?;
    fontsource(&scanned.lock, out, &mut expanded.dangling, &mut written)?;
    if written.render_total() == 0 {
        return Err("no render tests matched the mirrored prefixes".to_string());
    }
    fs::write(
        out.join(SOURCE_MD),
        source_md(&scanned.tests, &written, &expanded.dangling),
    )
    .map_err(|e| format!("{}: {e}", out.join(SOURCE_MD).display()))?;

    eprintln!(
        "mirrored {} render tests, {} query tests, {} asset files, {:.1} MiB",
        written.render_total(),
        written.query_total(),
        written.assets,
        written.bytes as f64 / (1024.0 * 1024.0),
    );
    for reference in &expanded.dangling {
        eprintln!("dangling reference local://{reference}");
    }
    Ok(())
}

fn open(archive: &[u8]) -> Archive<GzDecoder<Cursor<&[u8]>>> {
    Archive::new(GzDecoder::new(Cursor::new(archive)))
}

fn entry_path<R: Read>(entry: &tar::Entry<R>) -> Result<String, String> {
    let path = entry.path_bytes();
    String::from_utf8(path.into_owned()).map_err(|e| format!("non-UTF-8 path in tarball: {e}"))
}

/// A tarball path without the `maplibre-gl-js-<commit>/` root GitHub wraps
/// every entry in. A different root means the tarball isn't the pinned commit.
fn unwrap_root(path: &str) -> Result<Option<&str>, String> {
    let Some((root, rest)) = path.split_once('/') else {
        return Ok(None);
    };
    if root != format!("maplibre-gl-js-{COMMIT}") {
        return Err(format!(
            "tarball entry {path} is not under the pinned commit"
        ));
    }
    Ok(Some(rest))
}

struct Scanned {
    /// The mirrored `style.json` files.
    styles: Vec<Style>,
    /// Every file under `test/integration/assets/`.
    assets: BTreeSet<String>,
    /// The repository's `package-lock.json`.
    lock: String,
    /// Test cases (one per `style.json`) per prefix.
    tests: Tests,
}

/// Test counts per prefix, for the render and query suites.
#[derive(Default)]
struct Tests {
    render: [usize; PREFIXES.len()],
    query: [usize; PREFIXES.len()],
}

/// One mirrored style: the assets it references and the glyphs it can reach.
struct Style {
    /// The `local://` URLs in the document, without the scheme.
    refs: BTreeSet<String>,
    /// Every font stack its `text-font` values can name, or `None` when one of
    /// them isn't decidable statically.
    stacks: Option<BTreeSet<String>>,
}

/// First pass: the referenced URLs, the asset tree and the per-prefix counts.
fn scan(archive: &[u8]) -> Result<Scanned, String> {
    let mut scanned = Scanned {
        styles: Vec::new(),
        assets: BTreeSet::new(),
        lock: String::new(),
        tests: Tests::default(),
    };
    let mut archive = open(archive);
    let entries = archive.entries().map_err(|e| e.to_string())?;
    for entry in entries {
        let mut entry = entry.map_err(|e| e.to_string())?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry_path(&entry)?;
        let Some(path) = unwrap_root(&path)? else {
            continue;
        };
        if let Some(rel) = path.strip_prefix(ASSETS) {
            scanned.assets.insert(rel.to_string());
            continue;
        }
        if path == "package-lock.json" {
            entry
                .read_to_string(&mut scanned.lock)
                .map_err(|e| format!("{path}: {e}"))?;
            continue;
        }
        let (rel, query) = if let Some(rel) = path.strip_prefix(TESTS) {
            (rel, false)
        } else if let Some(rel) = path.strip_prefix(QUERY) {
            (rel, true)
        } else {
            continue;
        };
        let Some(prefix) = prefix_of(rel) else {
            continue;
        };
        if !rel.ends_with("/style.json") {
            continue;
        }
        let mut style = String::new();
        entry
            .read_to_string(&mut style)
            .map_err(|e| format!("{path}: {e}"))?;
        scanned.styles.push(Style {
            refs: local_urls(&style)
                .into_iter()
                .chain(operation_images(&style))
                .collect(),
            stacks: fontstacks(&style).map_err(|e| format!("{path}: {e}"))?,
        });
        let counts = if query {
            &mut scanned.tests.query[prefix]
        } else {
            &mut scanned.tests.render[prefix]
        };
        *counts += 1;
    }
    Ok(scanned)
}

/// The index into [`PREFIXES`] of the top-level directory of a test path.
fn prefix_of(rel: &str) -> Option<usize> {
    let top = rel.split('/').next()?;
    PREFIXES.iter().position(|prefix| top.starts_with(prefix))
}

/// The `local://` paths in a style document, without the scheme.
fn local_urls(style: &str) -> Vec<String> {
    const SCHEME: &str = "local://";
    style
        .match_indices(SCHEME)
        .filter_map(|(start, _)| {
            let rest = &style[start + SCHEME.len()..];
            let end = rest.find('"')?;
            Some(rest[..end].to_string())
        })
        .collect()
}

/// The asset paths of a test's `addImage` operations, which GL JS's harness
/// loads from the assets directory.
fn operation_images(style: &str) -> Vec<String> {
    let Ok(style) = serde_json::from_str::<Value>(style) else {
        return Vec::new();
    };
    let operations = style
        .pointer("/metadata/test/operations")
        .and_then(Value::as_array);
    operations
        .into_iter()
        .flatten()
        .filter(|op| op.get(0).and_then(Value::as_str) == Some("addImage"))
        .filter_map(|op| op.get(2).and_then(Value::as_str))
        .map(|path| path.trim_start_matches("./").to_string())
        .collect()
}

/// Every font stack a style's `text-font` values name, comma-joined as in glyph
/// URLs; `None` when one of those values isn't decidable statically.
fn fontstacks(style: &str) -> Result<Option<BTreeSet<String>>, String> {
    let style: Value = serde_json::from_str(style).map_err(|e| e.to_string())?;
    let mut stacks = BTreeSet::new();
    let decided = collect_fonts(&style, &mut stacks);
    Ok(decided.then_some(stacks))
}

/// Collects the stacks of every `text-font` layer property in a document,
/// including `format` sections nested in a text field. Reports whether every
/// one of them is decidable.
fn collect_fonts(node: &Value, stacks: &mut BTreeSet<String>) -> bool {
    let mut decided = true;
    match node {
        Value::Object(map) => {
            for (key, value) in map {
                decided &= match key.as_str() {
                    "layout" | "paint" => properties(value, stacks),
                    _ => collect_fonts(value, stacks),
                };
            }
        }
        Value::Array(items) => {
            for item in items {
                decided &= collect_fonts(item, stacks);
            }
        }
        _ => {}
    }
    decided
}

/// Collects the stacks of the `text-font` properties in a layer's `layout` or
/// `paint` object. `text-font` elsewhere, as in a GeoJSON feature's own
/// properties, is data rather than a font stack.
fn properties(node: &Value, stacks: &mut BTreeSet<String>) -> bool {
    let mut decided = true;
    match node {
        Value::Object(map) => {
            for (key, value) in map {
                decided &= if key == "text-font" {
                    names(value, stacks)
                } else {
                    properties(value, stacks)
                };
            }
        }
        Value::Array(items) => {
            for item in items {
                decided &= properties(item, stacks);
            }
        }
        _ => {}
    }
    decided
}

/// Collects the stacks a `text-font` value can name, and reports whether that
/// is all of them. A string or an array of font names is a stack itself; an
/// expression is decided by the literal stacks inside it, so
/// `["step", ["zoom"], ["literal", ["A"]], 1, ["literal", ["B"]]]` names both
/// and `["get", "font"]` names none.
fn names(value: &Value, stacks: &mut BTreeSet<String>) -> bool {
    match value {
        Value::String(font) => {
            stacks.insert(font.clone());
            true
        }
        Value::Array(items) => {
            let mut found = BTreeSet::new();
            match literal_stack(items).or_else(|| literal_node(items)) {
                Some(stack) => {
                    found.insert(stack);
                }
                None => expression(items, &mut found),
            }
            let decided = !found.is_empty();
            stacks.extend(found);
            decided
        }
        _ => false,
    }
}

/// The stack an array of font names stands for, if it is one.
fn literal_stack(items: &[Value]) -> Option<String> {
    let names: Option<Vec<&str>> = items.iter().map(Value::as_str).collect();
    let names = names?;
    (!names.is_empty()).then(|| names.join(","))
}

/// The stack behind a `["literal", [...]]` node, if that is what it is.
fn literal_node(items: &[Value]) -> Option<String> {
    if items.first()?.as_str()? != "literal" {
        return None;
    }
    literal_stack(items.get(1)?.as_array()?)
}

/// Every stack a `["literal", [...]]` node nested in an expression can produce.
fn expression(items: &[Value], out: &mut BTreeSet<String>) {
    if let Some(stack) = literal_node(items) {
        out.insert(stack);
        return;
    }
    for item in items {
        if let Value::Array(inner) = item {
            expression(inner, out);
        }
    }
}

/// The mirrored assets, and the references that name nothing upstream.
struct Expanded {
    wanted: BTreeSet<String>,
    dangling: BTreeSet<String>,
}

/// Every asset file the mirrored styles reference.
///
/// `{fontstack}` stands for each font stack the referencing style can name, so
/// glyph ranges are mirrored only for fonts a test can request; for a style
/// whose stacks aren't decidable it stands for every stack under `glyphs/`, and
/// for a style that names none it stands for nothing. Any other `{...}`
/// placeholder stands for the whole directory above it, so
/// `tiles/{z}-{x}-{y}.mvt` brings in `tiles/`. A reference without a
/// placeholder is a file, or the base name of a sprite and its resolution
/// variants (`sprites/emerald` → `emerald.json`, `emerald.png`, `emerald@2x.*`).
/// A reference that matches nothing is reported rather than dropped silently.
fn expand(styles: &[Style], assets: &BTreeSet<String>) -> Expanded {
    let every = glyph_stacks(assets);
    let mut expanded = Expanded {
        wanted: BTreeSet::new(),
        dangling: BTreeSet::new(),
    };
    for style in styles {
        let stacks = match &style.stacks {
            Some(stacks) => stacks,
            None => &every,
        };
        for reference in &style.refs {
            let references: Vec<String> = if reference.contains("{fontstack}") {
                stacks
                    .iter()
                    .map(|stack| reference.replace("{fontstack}", stack))
                    .collect()
            } else {
                vec![reference.clone()]
            };
            for reference in references {
                if !select(&reference, assets, &mut expanded.wanted) {
                    expanded.dangling.insert(reference);
                }
            }
        }
    }
    expanded
}

/// Every glyph stack directory under `glyphs/`.
fn glyph_stacks(assets: &BTreeSet<String>) -> BTreeSet<String> {
    assets
        .iter()
        .filter_map(|asset| asset.strip_prefix(GLYPHS))
        .filter_map(|asset| asset.split_once('/'))
        .map(|(stack, _)| stack.to_string())
        .collect()
}

/// Adds the assets one reference stands for to `wanted`, and reports whether it
/// named anything.
fn select(reference: &str, assets: &BTreeSet<String>, wanted: &mut BTreeSet<String>) -> bool {
    let mut matches = 0;
    if reference.contains('{') {
        let literal = reference
            .split('/')
            .take_while(|segment| !segment.contains('{'))
            .collect::<Vec<_>>()
            .join("/");
        let dir = format!("{literal}/");
        for asset in assets.iter().filter(|asset| asset.starts_with(&dir)) {
            wanted.insert(asset.clone());
            matches += 1;
        }
    } else if assets.contains(reference) {
        wanted.insert(reference.to_string());
        matches += 1;
    } else {
        let (dir, base) = match reference.rsplit_once('/') {
            Some((dir, base)) => (format!("{dir}/"), base),
            None => (String::new(), reference),
        };
        for asset in assets {
            let Some(name) = asset.strip_prefix(&dir) else {
                continue;
            };
            let variant = name == base
                || name.starts_with(&format!("{base}."))
                || name.starts_with(&format!("{base}@"));
            if variant && !name.contains('/') {
                wanted.insert(asset.clone());
                matches += 1;
            }
        }
    }
    matches > 0
}

#[derive(Default)]
struct Written {
    tests: Tests,
    assets: usize,
    bytes: u64,
    /// The share of `bytes` the query tests account for.
    query_bytes: u64,
}

impl Written {
    fn render_total(&self) -> usize {
        self.tests.render.iter().sum()
    }

    fn query_total(&self) -> usize {
        self.tests.query.iter().sum()
    }
}

/// Second pass: write the mirrored tests and the assets they reference.
fn extract(archive: &[u8], out: &Path, wanted: &BTreeSet<String>) -> Result<Written, String> {
    let mut written = Written::default();
    let mut archive = open(archive);
    let entries = archive.entries().map_err(|e| e.to_string())?;
    for entry in entries {
        let mut entry = entry.map_err(|e| e.to_string())?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry_path(&entry)?;
        let Some(path) = unwrap_root(&path)? else {
            continue;
        };
        let mut query = false;
        let target = if let Some(rel) = path.strip_prefix(TESTS) {
            let Some(prefix) = prefix_of(rel) else {
                continue;
            };
            if rel.ends_with("/style.json") {
                written.tests.render[prefix] += 1;
            }
            out.join("render").join(rel)
        } else if let Some(rel) = path.strip_prefix(QUERY) {
            let Some(prefix) = prefix_of(rel) else {
                continue;
            };
            if rel.ends_with("/style.json") {
                written.tests.query[prefix] += 1;
            }
            query = true;
            out.join("query").join(rel)
        } else if let Some(rel) = path.strip_prefix(ASSETS) {
            if !wanted.contains(rel) {
                continue;
            }
            written.assets += 1;
            out.join("assets").join(rel)
        } else {
            continue;
        };
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|e| format!("{path}: {e}"))?;
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        fs::write(&target, &bytes).map_err(|e| format!("{}: {e}", target.display()))?;
        written.bytes += bytes.len() as u64;
        if query {
            written.query_bytes += bytes.len() as u64;
        }
    }
    Ok(written)
}

/// Mirrors `local://fonts/<name>/files/<file>`, which GL JS's render test
/// server serves from the `@fontsource/<name>` package its lockfile pins,
/// into `assets/fonts/`.
fn fontsource(
    lock: &str,
    out: &Path,
    dangling: &mut BTreeSet<String>,
    written: &mut Written,
) -> Result<(), String> {
    let lock: Value = serde_json::from_str(lock).map_err(|e| format!("package-lock.json: {e}"))?;
    let fonts: Vec<String> = dangling
        .iter()
        .filter(|r| r.starts_with("fonts/"))
        .cloned()
        .collect();
    for reference in fonts {
        let Some((name, file)) = reference
            .strip_prefix("fonts/")
            .and_then(|r| r.split_once("/files/"))
        else {
            continue;
        };
        let package = &lock["packages"][format!("node_modules/@fontsource/{name}")];
        let (Some(url), Some(integrity)) =
            (package["resolved"].as_str(), package["integrity"].as_str())
        else {
            continue;
        };
        let tarball = get(url)?;
        let digest = base64::engine::general_purpose::STANDARD.encode(Sha512::digest(&tarball));
        if integrity != format!("sha512-{digest}") {
            return Err(format!("{url} does not match the lockfile's {integrity}"));
        }
        let mut tarball = Archive::new(GzDecoder::new(&tarball[..]));
        let wanted = format!("package/files/{file}");
        for entry in tarball.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            if entry_path(&entry)? != wanted {
                continue;
            }
            let mut bytes = Vec::new();
            entry
                .read_to_end(&mut bytes)
                .map_err(|e| format!("{url}: {e}"))?;
            let target = out.join("assets").join(&reference);
            fs::create_dir_all(target.parent().expect("a parent"))
                .map_err(|e| format!("{}: {e}", target.display()))?;
            fs::write(&target, &bytes).map_err(|e| format!("{}: {e}", target.display()))?;
            written.assets += 1;
            written.bytes += bytes.len() as u64;
            dangling.remove(&reference);
            break;
        }
    }
    Ok(())
}

fn source_md(tests: &Tests, written: &Written, dangling: &BTreeSet<String>) -> String {
    let mut rows = String::new();
    for (i, prefix) in PREFIXES.iter().enumerate() {
        let (render, query) = (tests.render[i], tests.query[i]);
        rows.push_str(&format!("| `{prefix}*` | {render} | {query} |\n"));
    }
    let dangling = if dangling.is_empty() {
        String::new()
    } else {
        let mut list = String::new();
        for reference in dangling {
            list.push_str(&format!("- `{reference}`\n"));
        }
        format!(
            "\
## Dangling references

These `local://` URLs in mirrored styles name no file under
`test/integration/assets/` at this commit. Upstream requests them too, so they
fail to load there as well; the mirror contains nothing for them.

{list}"
        )
    };
    format!(
        "\
# MapLibre GL JS fixtures

A read-only mirror of the MapLibre GL JS render and query tests, fetched by
`cargo xtask fixtures` and not committed. Do not edit it by hand: the next
`cargo xtask fixtures` or `cargo xtask rt` replaces a mirror that differs
from the pin. To move the pin, change `COMMIT` in `xtask/src/fixtures.rs`,
fetch, and set `MIRROR_SHA256` to the hash the fetch reports.

- Repository: {REPO}
- Commit: {COMMIT}
- License: BSD-3-Clause (upstream)

## Contents

Every test directory under `test/integration/render/tests/` (mirrored into
`render/`) and `test/integration/query/tests/` (into `query/`) whose top-level
name starts with one of the prefixes in the table below, at its original nested
path (`render/<dir>/<variant>/style.json`, `expected.png`, ...); and every file
under `test/integration/assets/` that a mirrored `style.json` references through
a `local://` URL. `{{fontstack}}` stands for each `text-font` stack the
referencing style can name — the font arrays, `[\"literal\", [...]]` wrappers and
the literal stacks inside expressions — so only those glyphs are mirrored; a
style whose stacks aren't decidable mirrors every stack under `glyphs/`, and one
that names none mirrors none. Any
other `{{...}}` placeholder stands for the whole directory above it, so
`local://tiles/{{z}}-{{x}}-{{y}}.mvt` mirrors `tiles/`. `local://fonts/<name>/files/<file>`,
which GL JS's test server serves from npm, mirrors that file of the
`@fontsource/<name>` package the repository's `package-lock.json` pins into
`assets/fonts/`.

| Prefix | Render tests | Query tests |
| --- | --- | --- |
{rows}
- Render tests: {render}
- Query tests: {query} ({query_kib:.1} KiB)
- Asset files: {assets}
- Total size: {mib:.1} MiB ({bytes} bytes)

{dangling}",
        render = written.render_total(),
        query = written.query_total(),
        query_kib = written.query_bytes as f64 / 1024.0,
        assets = written.assets,
        mib = written.bytes as f64 / (1024.0 * 1024.0),
        bytes = written.bytes,
    )
}

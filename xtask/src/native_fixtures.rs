//! Fetches the MapLibre Native test suites that Orbis tests against.
//!
//! The mirror in `fixtures/native/` is not committed. `cargo xtask fixtures
//! native` builds it from the pinned commit's tarball: the integration
//! render, query and expression tests, the platform expectations and ignore
//! lists that Native's macOS WebGPU manifest names, and the resources in
//! `cache-style.db`, which Native's render-test runner reads every request
//! from (offline, cache only), as plain files.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use flate2::read::{GzDecoder, ZlibDecoder};
use rusqlite::types::ValueRef;
use serde_json::Value;

use crate::fixtures::{SOURCE_MD, entry_path, get, install, open, unwrap_root};

/// The upstream repository and the commit the mirror is pinned to: the
/// MapLibre Native commit mln-rs grades against.
const REPO: &str = "https://github.com/maplibre/maplibre-native";
const COMMIT: &str = "8e144d0eca082e8350335b2172a6dfdc939b9cef";

/// The mirror's content hash at [`COMMIT`].
const MIRROR_SHA256: &str = "1898144df1c03ebeeb37a0150f2c9cf561f4125f44c974a52f39474fd03d1e31";

/// The manifest whose expectations and ignores the mirror keeps: Native's
/// WebGPU backend on wgpu on macOS, the configuration closest to Orbis's
/// (wgpu, Metal on a Mac). Mirrored as `manifest.json`.
const MANIFEST: &str = "macos-webgpu-wgpu.json";

const METRICS: &str = "metrics/";
/// The suites under `metrics/integration/`, mirrored at the same paths.
const SUITES: [&str; 3] = ["render-tests", "query-tests", "expression-tests"];
const CACHE: &str = "cache-style.db";

/// Fetches the mirror into `fixtures/native/` unless the one there matches
/// the pin.
pub(crate) fn ensure() -> Result<(), String> {
    install(
        "native",
        COMMIT,
        MIRROR_SHA256,
        "xtask/src/native_fixtures.rs",
        mirror,
    )
}

fn root() -> String {
    format!("maplibre-native-{COMMIT}")
}

/// Downloads the pinned tarball and writes the mirror into `out`.
fn mirror(out: &Path) -> Result<(), String> {
    eprintln!("fetching {REPO} at {COMMIT}");
    let url = format!("https://codeload.github.com/maplibre/maplibre-native/tar.gz/{COMMIT}");
    let archive = get(&url)?;

    // Everything under `metrics/` the mirror might keep, by path below it.
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let root = root();
    let mut entries = open(&archive);
    for entry in entries.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry_path(&entry)?;
        let Some(rel) = unwrap_root(&path, &root)?.and_then(|p| p.strip_prefix(METRICS)) else {
            continue;
        };
        let wanted = rel == MANIFEST
            || rel == CACHE
            || rel.starts_with("expectations/")
            || rel.starts_with("ignores/")
            || SUITES
                .iter()
                .any(|suite| rel.starts_with(&format!("integration/{suite}/")));
        if !wanted {
            continue;
        }
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|e| format!("{path}: {e}"))?;
        files.insert(rel.to_string(), bytes);
    }

    // The manifest's expectation and ignore paths, relative to `metrics/`.
    let manifest = files
        .get(MANIFEST)
        .ok_or_else(|| format!("metrics/{MANIFEST} is missing"))?;
    let manifest: Value =
        serde_json::from_slice(manifest).map_err(|e| format!("metrics/{MANIFEST}: {e}"))?;
    let named = |key: &str| -> Result<Vec<String>, String> {
        manifest[key]
            .as_array()
            .ok_or_else(|| format!("metrics/{MANIFEST} has no {key}"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("metrics/{MANIFEST}: {key} holds a non-string"))
            })
            .collect()
    };
    let expectations = named("expectation_paths")?;
    let ignores = named("ignore_paths")?;
    for ignore in &ignores {
        if !files.contains_key(ignore) {
            return Err(format!(
                "metrics/{MANIFEST} names {ignore}, which is missing"
            ));
        }
    }

    let mut written = Written::default();
    for (rel, bytes) in &files {
        let keep = if let Some(suite) = SUITES
            .iter()
            .position(|suite| rel.starts_with(&format!("integration/{suite}/")))
        {
            let file = rel.rsplit('/').next().unwrap_or_default();
            if file == "style.json" || file == "test.json" {
                written.tests[suite] += 1;
            }
            true
        } else if rel == MANIFEST {
            write(out, "manifest.json", bytes, &mut written)?;
            continue;
        } else if expectations
            .iter()
            .any(|dir| rel.starts_with(&format!("{dir}/")))
        {
            if rel.ends_with(".png") || rel.ends_with(".json") {
                written.expectations += 1;
            }
            true
        } else {
            ignores.contains(rel)
        };
        if keep {
            write(out, rel, bytes, &mut written)?;
        }
    }

    let cache = files
        .get(CACHE)
        .ok_or_else(|| format!("metrics/{CACHE} is missing"))?;
    let resources = extract_cache(cache, out, &mut written)?;

    fs::write(
        out.join(SOURCE_MD),
        source_md(&written, &resources, &expectations, &ignores),
    )
    .map_err(|e| format!("{}: {e}", out.join(SOURCE_MD).display()))?;
    eprintln!(
        "mirrored {} render tests, {} query tests, {} expression tests, {} expectation files, \
         {} resources and {} tiles from {CACHE}, {:.1} MiB",
        written.tests[0],
        written.tests[1],
        written.tests[2],
        written.expectations,
        resources.resources,
        resources.tiles,
        written.bytes as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}

#[derive(Default)]
struct Written {
    /// Tests per suite, in [`SUITES`] order.
    tests: [usize; SUITES.len()],
    expectations: usize,
    bytes: u64,
}

fn write(out: &Path, rel: &str, bytes: &[u8], written: &mut Written) -> Result<(), String> {
    let target = out.join(rel);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fs::write(&target, bytes).map_err(|e| format!("{}: {e}", target.display()))?;
    written.bytes += bytes.len() as u64;
    Ok(())
}

#[derive(Default)]
struct Resources {
    resources: usize,
    tiles: usize,
    /// Files per top-level asset directory.
    dirs: BTreeMap<String, usize>,
}

/// Writes every resource and tile in Native's render-test cache under
/// `assets/`, at the path [`asset_path`] gives its URL.
fn extract_cache(db: &[u8], out: &Path, written: &mut Written) -> Result<Resources, String> {
    // SQLite opens files: stage the database inside the mirror, and remove
    // it before hashing.
    let path = out.join(CACHE);
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    fs::write(&path, db).map_err(|e| format!("{}: {e}", path.display()))?;
    let result = read_cache(&path, out, written);
    fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    result
}

fn read_cache(path: &Path, out: &Path, written: &mut Written) -> Result<Resources, String> {
    let sql = |e: rusqlite::Error| format!("{CACHE}: {e}");
    let db =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(sql)?;
    // URL → contents, so two rows naming one file must agree.
    let mut assets: BTreeMap<PathBuf, (String, Vec<u8>)> = BTreeMap::new();
    let mut resources = Resources::default();
    let mut add = |url: String, data: Vec<u8>| -> Result<(), String> {
        let rel = asset_path(&url)?;
        match assets.get(&rel) {
            Some((other, bytes)) if *bytes != data => {
                Err(format!("{url} and {other} both map to {}", rel.display()))
            }
            Some(_) => Ok(()),
            None => {
                assets.insert(rel, (url, data));
                Ok(())
            }
        }
    };

    let mut rows = db
        .prepare("SELECT url, data, compressed FROM resources ORDER BY url")
        .map_err(sql)?;
    let mut query = rows.query([]).map_err(sql)?;
    while let Some(row) = query.next().map_err(sql)? {
        let url: String = row.get(0).map_err(sql)?;
        let Some(data) = data(row.get_ref(1).map_err(sql)?, row.get(2).map_err(sql)?, &url)? else {
            continue;
        };
        add(url, data)?;
        resources.resources += 1;
    }

    let mut rows = db
        .prepare(
            "SELECT url_template, pixel_ratio, z, x, y, data, compressed FROM tiles \
             ORDER BY url_template, pixel_ratio, z, x, y",
        )
        .map_err(sql)?;
    let mut query = rows.query([]).map_err(sql)?;
    while let Some(row) = query.next().map_err(sql)? {
        let template: String = row.get(0).map_err(sql)?;
        let ratio: i64 = row.get(1).map_err(sql)?;
        let (z, x, y): (i64, i64, i64) = (
            row.get(2).map_err(sql)?,
            row.get(3).map_err(sql)?,
            row.get(4).map_err(sql)?,
        );
        let url = tile_url(&template, ratio, z, x, y)?;
        let Some(data) = data(row.get_ref(5).map_err(sql)?, row.get(6).map_err(sql)?, &url)? else {
            continue;
        };
        add(url, data)?;
        resources.tiles += 1;
    }

    for (rel, (_, bytes)) in &assets {
        let rel = Path::new("assets").join(rel);
        let top = rel
            .components()
            .nth(1)
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .unwrap_or_default();
        *resources.dirs.entry(top).or_default() += 1;
        write(out, &rel.to_string_lossy(), bytes, written)?;
    }
    Ok(resources)
}

/// A row's contents: `compressed` rows are zlib or gzip streams, as
/// Native's `util::decompress` reads them. `None` for a row without data.
fn data(value: ValueRef<'_>, compressed: bool, url: &str) -> Result<Option<Vec<u8>>, String> {
    let bytes = match value {
        ValueRef::Null => return Ok(None),
        ValueRef::Blob(b) => b,
        ValueRef::Text(t) => t,
        other => return Err(format!("{url}: data of type {:?}", other.data_type())),
    };
    if !compressed {
        return Ok(Some(bytes.to_vec()));
    }
    let mut out = Vec::new();
    let read = if bytes.starts_with(&[0x1f, 0x8b]) {
        GzDecoder::new(bytes).read_to_end(&mut out)
    } else {
        ZlibDecoder::new(bytes).read_to_end(&mut out)
    };
    read.map_err(|e| format!("{url}: {e}"))?;
    Ok(Some(out))
}

/// A tile's URL from its template, as Native's `util::tileURL` fills one in.
fn tile_url(template: &str, ratio: i64, z: i64, x: i64, y: i64) -> Result<String, String> {
    let mut url = String::new();
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        url.push_str(&rest[..start]);
        let end = rest[start..]
            .find('}')
            .ok_or_else(|| format!("unterminated token in {template}"))?;
        let token = &rest[start + 1..start + end];
        match token {
            "z" => url.push_str(&z.to_string()),
            "x" => url.push_str(&x.to_string()),
            "y" => url.push_str(&y.to_string()),
            "ratio" if ratio > 1 => url.push_str(&format!("@{ratio}x")),
            "ratio" => {}
            _ => return Err(format!("unknown token {{{token}}} in {template}")),
        }
        rest = &rest[start + end + 1..];
    }
    url.push_str(rest);
    Ok(url)
}

/// Where a cached URL's contents live under `assets/`: `http://<path>` (the
/// runner's spelling of `local://` and GL JS's test server) at `<path>`,
/// other schemes under a directory named after the scheme. Percent escapes
/// are decoded and `.` segments dropped, as a file server would.
fn asset_path(url: &str) -> Result<PathBuf, String> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| format!("{url} has no scheme"))?;
    let rest = rest.split(['?', '#']).next().unwrap_or_default();
    let mut decoded = Vec::new();
    let mut bytes = rest.bytes();
    while let Some(byte) = bytes.next() {
        decoded.push(if byte == b'%' {
            let hi = bytes.next().and_then(|c| (c as char).to_digit(16));
            let lo = bytes.next().and_then(|c| (c as char).to_digit(16));
            match (hi, lo) {
                (Some(hi), Some(lo)) => (hi * 16 + lo) as u8,
                _ => return Err(format!("{url} has a bad percent escape")),
            }
        } else {
            byte
        });
    }
    let decoded = String::from_utf8(decoded).map_err(|_| format!("{url} isn't UTF-8"))?;
    let mut path = match scheme {
        "http" => PathBuf::new(),
        "https" | "mapbox" => PathBuf::from(scheme),
        _ => return Err(format!("{url}: unknown scheme")),
    };
    for component in Path::new(&decoded).components() {
        match component {
            Component::Normal(part) => path.push(part),
            Component::CurDir => {}
            _ => return Err(format!("{url} leaves the asset tree")),
        }
    }
    Ok(path)
}

fn source_md(
    written: &Written,
    resources: &Resources,
    expectations: &[String],
    ignores: &[String],
) -> String {
    let list = |items: &[String]| {
        items
            .iter()
            .map(|i| format!("`{i}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let dirs: String = resources
        .dirs
        .iter()
        .map(|(dir, n)| format!("| `assets/{dir}/` | {n} |\n"))
        .collect();
    format!(
        "\
# MapLibre Native test suites

A read-only mirror of MapLibre Native's render, query and expression tests,
fetched by `cargo xtask fixtures native` and not committed. Do not edit it by
hand: the next `cargo xtask fixtures native` or `cargo xtask rt --suite
native-*` replaces a mirror that differs from the pin. To move the pin,
change `COMMIT` in `xtask/src/native_fixtures.rs`, fetch, and set
`MIRROR_SHA256` to the hash the fetch reports.

- Repository: {REPO}
- Commit: {COMMIT}
- License: BSD-2-Clause (upstream)

## Contents

Paths are upstream's below `metrics/`:

- `integration/render-tests/`: {render} render tests (`style.json`,
  `expected*.png`), every one.
- `integration/query-tests/`: {query} query tests (`style.json`,
  `expected.json`), every one.
- `integration/expression-tests/`: {expression} expression tests
  (`test.json`), every one.
- `manifest.json`: upstream's `metrics/{MANIFEST}`, the render-test runner's
  manifest for its WebGPU backend on wgpu on macOS. Its expectation paths
  ({expectation_paths}: {expectation_files} files) and ignore lists
  ({ignore_paths}) are mirrored too; its `base_test_path` is `integration`.
- `assets/`: every resource ({resources}) and tile ({tiles}) in upstream's
  `metrics/{CACHE}`, the offline database Native's runner reads every
  request from, decompressed. `http://<path>`, which is how the runner spells
  `local://` and `http://localhost:2900/`, is at `assets/<path>`; `https://`
  and `mapbox://` URLs are under `assets/https/` and `assets/mapbox/`. Paths
  are percent-decoded, without `.` segments. Tile rows are written at the
  URL their template gives for their `z`, `x`, `y` and pixel ratio. The
  on-disk assets beside the tests in `metrics/integration/` aren't
  mirrored: Native's runner never reads them.

| Directory | Files |
| --- | --- |
{dirs}
- Total size: {mib:.1} MiB ({bytes} bytes)
",
        render = written.tests[0],
        query = written.tests[1],
        expression = written.tests[2],
        expectation_paths = list(expectations),
        expectation_files = written.expectations,
        ignore_paths = list(ignores),
        resources = resources.resources,
        tiles = resources.tiles,
        mib = written.bytes as f64 / (1024.0 * 1024.0),
        bytes = written.bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_urls_map_to_asset_paths() {
        assert_eq!(
            asset_path("http://glyphs/Open%20Sans%2cArial/0-255.pbf").unwrap(),
            Path::new("glyphs/Open Sans,Arial/0-255.pbf")
        );
        assert_eq!(
            asset_path("mapbox://render-tests/./image/rocket.png").unwrap(),
            Path::new("mapbox/render-tests/image/rocket.png")
        );
        assert_eq!(
            asset_path("https://demotiles.maplibre.org/a/1.png").unwrap(),
            Path::new("https/demotiles.maplibre.org/a/1.png")
        );
        assert!(asset_path("http://../secret").is_err());
        assert!(asset_path("file:///etc/passwd").is_err());
        assert_eq!(
            tile_url("http://tiles/{z}-{x}-{y}{ratio}.png", 2, 1, 2, 3).unwrap(),
            "http://tiles/1-2-3@2x.png"
        );
        assert!(tile_url("http://t/{quadkey}", 1, 0, 0, 0).is_err());
    }
}

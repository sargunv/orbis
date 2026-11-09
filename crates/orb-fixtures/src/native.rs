//! MapLibre Native's render and query suites (`fixtures/native`), run as
//! Native's render-test runner (`render-test/` at the pinned commit) runs
//! them: its metadata defaults, its operations applied before the first
//! frame, still (static map mode) and tile output, every request from its
//! offline cache, the manifest's platform expectations and ignore lists, and
//! its comparison: pixelmatch-cpp on premultiplied pixels at threshold
//! 0.1285, and query answers compared in order to six significant figures.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use orb_api::{Map, MapOptions, Runtime, TileMatrixSet, TileRequest};
use serde_json::{Value, json};

use crate::fetch::Assets;
use crate::images::Image;
use crate::{
    Backend, Counts, chrome, dem, finish_skip, note_warning, operations, pixelmatch_cpp,
    record_warnings, scope, write_json,
};

/// Native's runner hard-codes this pixelmatch threshold ("Defined in GL JS").
pub(crate) const THRESHOLD: f64 = 0.1285;
/// Native's default `allowed`.
const ALLOWED: f64 = 0.00015;
/// Native's tile size: tile mode draws 512 × 512 logical pixels per tile.
const TILE: f64 = 512.0;

/// The mirror.
pub(crate) fn mirror(root: &Path) -> PathBuf {
    root.join("fixtures/native")
}

/// Native's cache, as files.
pub(crate) fn assets(root: &Path) -> PathBuf {
    mirror(root).join("assets")
}

/// The runner's manifest (`fixtures/native/manifest.json`): where platform
/// expectations are, and the ignore lists.
pub(crate) struct Manifest {
    /// The test directory comes first, then the manifest's expectation
    /// directories; the runner looks in them last to first.
    expectations: Vec<PathBuf>,
    /// Test id → reason, the first list naming a test winning, as the
    /// runner's `find_if` does.
    pub(crate) ignores: BTreeMap<String, String>,
}

impl Manifest {
    pub(crate) fn read(root: &Path) -> Result<Self, String> {
        let mirror = mirror(root);
        let path = mirror.join("manifest.json");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            format!(
                "{}: {e}; run `cargo xtask fixtures native` to fetch the Native fixture mirror",
                path.display()
            )
        })?;
        let manifest: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let strings = |key: &str| -> Vec<String> {
            manifest[key]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        };
        let mut ignores = BTreeMap::new();
        for list in strings("ignore_paths") {
            let path = mirror.join(&list);
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let list: serde_json::Map<String, Value> =
                serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            for (id, reason) in list {
                ignores
                    .entry(id)
                    .or_insert_with(|| reason.as_str().unwrap_or_default().to_string());
            }
        }
        Ok(Self {
            expectations: strings("expectation_paths")
                .into_iter()
                .map(|dir| mirror.join(dir))
                .collect(),
            ignores,
        })
    }

    /// The expected files for test `id` (in `dir`) whose names match
    /// `expected*.<extension>`: those of the last directory that has any.
    fn expected(&self, id: &str, dir: &Path, extension: &str) -> Vec<PathBuf> {
        let dirs = std::iter::once(dir.to_path_buf())
            .chain(self.expectations.iter().map(|base| base.join(id)));
        let dirs: Vec<PathBuf> = dirs.collect();
        for dir in dirs.iter().rev() {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut found: Vec<PathBuf> = entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.is_file()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("expected") && n.ends_with(extension))
                })
                .collect();
            if !found.is_empty() {
                found.sort();
                return found;
            }
        }
        Vec::new()
    }
}

/// A test's metadata as Native's `parseTestMetadata` reads it.
#[derive(Debug)]
pub(crate) struct Test {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixel_ratio: f64,
    pub(crate) allowed: f64,
    /// `mapMode: "tile"`: one 512-pixel tile per tile the camera covers.
    pub(crate) tile_mode: bool,
    pub(crate) cross_source_collisions: bool,
    pub(crate) max_pitch: f64,
    pub(crate) operations: Vec<Value>,
    /// The query box, `[x1, y1, x2, y2]`, for a query test.
    pub(crate) query: Option<[f64; 4]>,
    /// `source-replace`: runs again with each pair's sources swapped.
    pub(crate) replacements: Vec<(String, String)>,
    pub(crate) value: Value,
    pub(crate) skips: BTreeSet<String>,
}

impl Test {
    pub(crate) fn read(style: &Value) -> Self {
        let test = style
            .pointer("/metadata/test")
            .cloned()
            .unwrap_or(json!({}));
        let mut skips = BTreeSet::new();
        let mut value = json!({"width": 512, "height": 512, "pixelRatio": 1, "allowed": ALLOWED});
        if let Some(object) = test.as_object() {
            for (k, v) in object {
                value[k] = v.clone();
            }
        }
        let tile_mode = match test.get("mapMode").and_then(Value::as_str) {
            None | Some("static") => false,
            Some("tile") => true,
            // Continuous tests output no image; the suite has none.
            Some(_) => {
                skips.insert("metadata:mapMode".into());
                false
            }
        };
        let mut number = |key: &str, default: f64, min: f64, integer: bool| match test.get(key) {
            None => default,
            Some(v) => match v.as_f64() {
                Some(n) if n.is_finite() && n >= min && (!integer || n.fract() == 0.0) => n,
                _ => {
                    skips.insert(format!("metadata:{key}"));
                    default
                }
            },
        };
        // Tile mode draws whole tiles and ignores the size.
        let (width, height) = if tile_mode {
            (TILE, TILE)
        } else {
            (
                number("width", 512.0, 1.0, true),
                number("height", 512.0, 1.0, true),
            )
        };
        let pixel_ratio = number("pixelRatio", 1.0, f64::MIN_POSITIVE, false);
        let allowed = number("allowed", ALLOWED, 0.0, false);
        let max_pitch = number("maxPitch", 60.0, 0.0, false);
        // The runner turns these on when the key is present, whatever its
        // value. A query test's answer doesn't depend on them.
        for key in ["debug", "collisionDebug", "showOverdrawInspector"] {
            if test.get(key).is_some() && test.get("queryGeometry").is_none() {
                skips.insert(format!("metadata:{key}"));
            }
        }
        if test.get("axonometric") == Some(&json!(true)) {
            skips.insert("metadata:axonometric".into());
        }
        if test
            .get("skew")
            .is_some_and(|s| s.as_array().is_none_or(|s| s != &[json!(0), json!(1)]))
        {
            skips.insert("metadata:skew".into());
        }
        if test.get("queryOptions").is_some() {
            skips.insert("query:options".into());
        }
        let query = test.get("queryGeometry").map(|g| {
            let geometry = query_box(g);
            if geometry.is_none() {
                skips.insert("query:geometry".into());
            }
            geometry.unwrap_or_default()
        });
        let operations = test
            .get("operations")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        skips.extend(operations.iter().filter_map(operation_skip));
        let replacements = test
            .get("source-replace")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|r| {
                Some((
                    r.get("replace")?.as_str()?.to_string(),
                    r.get("with")?.as_str()?.to_string(),
                ))
            })
            .collect();
        for logical in [width, height] {
            if !(1.0..=8192.0).contains(&(logical * pixel_ratio).floor()) {
                skips.insert("metadata:canvas-size".into());
            }
        }
        Self {
            width: width as u32,
            height: height as u32,
            pixel_ratio,
            allowed,
            tile_mode,
            cross_source_collisions: test
                .get("crossSourceCollisions")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            max_pitch,
            operations,
            query,
            replacements,
            value,
            skips,
        }
    }

    /// The physical size of a still: the frontend's, `size × pixelRatio`
    /// truncated.
    pub(crate) fn physical_size(&self) -> (u32, u32) {
        let px = |v: u32| (f64::from(v) * self.pixel_ratio) as u32;
        (px(self.width), px(self.height))
    }
}

/// `queryGeometry`, a point or two corners, as a box.
fn query_box(g: &Value) -> Option<[f64; 4]> {
    let n = Value::as_f64;
    match g.as_array()?.as_slice() {
        [x, y] if x.is_number() => Some([n(x)?, n(y)?, n(x)?, n(y)?]),
        [a, b] => {
            let (a, b) = (a.as_array()?, b.as_array()?);
            Some([n(a.first()?)?, n(a.get(1)?)?, n(b.first()?)?, n(b.get(1)?)?])
        }
        _ => None,
    }
}

/// The operations Native's runner knows (`parseTestOperations`).
const NATIVE_OPERATIONS: &[&str] = &[
    "wait",
    "sleep",
    "addImage",
    "updateImage",
    "removeImage",
    "setStyle",
    "setCenter",
    "setZoom",
    "setBearing",
    "setPitch",
    "setRoll",
    "setCenterElevation",
    "setFilter",
    "setLayerZoomRange",
    "setLight",
    "addLayer",
    "removeLayer",
    "addSource",
    "removeSource",
    "setPaintProperty",
    "setLayoutProperty",
    "setGlobalStateProperty",
    "probeFileSize",
    "probeMemory",
    "probeMemoryStart",
    "probeMemoryEnd",
    "probeNetwork",
    "probeNetworkStart",
    "probeNetworkEnd",
    "setFeatureState",
    "getFeatureState",
    "removeFeatureState",
    "panGesture",
    "probeGFX",
    "probeGFXStart",
    "probeGFXEnd",
    "setTileLodMinRadius",
    "setTileLodScale",
    "setTileLodPitchThreshold",
    "setTileLodZoomShift",
    "setTileLodMode",
];

/// Why an operation can't run, if it can't: `native-runner:` for one
/// Native's runner rejects (it errors those tests), `operation:` for one
/// Orbis can't apply.
fn operation_skip(op: &Value) -> Option<String> {
    let name = op.get(0).and_then(Value::as_str).unwrap_or("unnamed");
    if !NATIVE_OPERATIONS.contains(&name) {
        return Some(format!("native-runner:{name}"));
    }
    let number = |i: usize| op.get(i).is_some_and(Value::is_number);
    let supported = match name {
        // Probes measure; with `ignoreProbing` on by default nothing is
        // checked.
        "wait" | "sleep" | "getFeatureState" => true,
        n if n.starts_with("probe") => true,
        "setZoom" | "setBearing" | "setPitch" | "setRoll" | "setCenterElevation" => number(1),
        "setCenter" => op
            .get(1)
            .and_then(Value::as_array)
            .is_some_and(|c| c.len() == 2),
        "addImage" | "updateImage" => op.get(2).is_some_and(Value::is_string),
        "removeImage" => op.get(1).is_some_and(Value::is_string),
        "setStyle" => op.get(1).is_some_and(|s| s.is_string() || s.is_object()),
        // Level-of-detail tuning and continuous-mode gestures have no
        // Orbis counterpart.
        n if n.starts_with("setTileLod") || n == "panGesture" => false,
        _ => orb_api::Transaction::from_commands(&format!("[{op}]")).is_ok(),
    };
    (!supported).then(|| format!("operation:{name}"))
}

/// Whether the runner renders a still before applying `op`: `wait` is one,
/// and the feature-state operations render first.
fn renders_first(op: &Value) -> bool {
    matches!(
        op.get(0).and_then(Value::as_str),
        Some("wait" | "setFeatureState" | "getFeatureState" | "removeFeatureState")
    )
}

/// Applies an operation the way Native's runner does, after any still it
/// renders first.
fn apply(map: &mut Map, op: &Value, assets: &Path) -> Result<(), String> {
    let name = op.get(0).and_then(Value::as_str).unwrap_or_default();
    match name {
        "wait" | "getFeatureState" => Ok(()),
        n if n.starts_with("probe") => Ok(()),
        // Static maps don't animate: a sleep changes nothing.
        "sleep" => Ok(()),
        "setStyle" => match &op[1] {
            Value::String(url) => {
                map.load_style_url(url);
                Ok(())
            }
            style => map
                .set_style_json(&style.to_string())
                .map_err(|e| e.to_string()),
        },
        // The runner loads images through its file source, from
        // `mapbox://render-tests/<path>`, and `Style::addImage` replaces an
        // image of the same name, its size and options too.
        "addImage" | "updateImage" => {
            let path = op[2].as_str().unwrap_or_default();
            let name = op[1].as_str().unwrap_or_default();
            if map.has_image(name) {
                map.remove_image(name).map_err(|e| e.to_string())?;
            }
            let mut op = op.clone();
            op[0] = json!("addImage");
            op[2] = json!(format!("mapbox/render-tests/{path}"));
            operations::apply(map, &op, assets).map(|_| ())
        }
        _ => operations::apply(map, op, assets).map(|_| ()),
    }
}

/// The suite's tests under `fixtures/native/integration/<suite>`, by id
/// (`render-tests/…`).
pub(crate) fn discover(
    root: &Path,
    suite: &str,
    filter: &str,
) -> Result<Vec<(String, PathBuf)>, String> {
    let base = mirror(root).join("integration");
    let dir = base.join(suite);
    if !dir.is_dir() {
        return Err(format!(
            "{} is missing; run `cargo xtask fixtures native` to fetch the Native fixture mirror",
            dir.display()
        ));
    }
    let mut paths = Vec::new();
    crate::discover(&dir, &mut paths)?;
    let mut tests: Vec<(String, PathBuf)> = paths
        .into_iter()
        .map(|path| (crate::test_id(&path, &base), path))
        .filter(|(id, _)| id.contains(filter))
        .collect();
    tests.sort();
    Ok(tests)
}

/// The skip a Native ignore list asks for (a reason starting `skip`), or
/// the reason Native ignores a test it runs.
fn ignored(manifest: &Manifest, id: &str) -> (Option<String>, Option<String>) {
    match manifest.ignores.get(id) {
        Some(reason) if reason.starts_with("skip") => {
            (Some("native:skip".into()), Some(reason.clone()))
        }
        Some(reason) => (None, Some(reason.clone())),
        None => (None, None),
    }
}

/// The map for a test, set up as the runner sets up its own.
fn map(runtime: &Runtime, test: &Test) -> Map {
    let mut map = Map::new(
        runtime,
        MapOptions {
            fade_duration: 0.0,
            cross_source_collisions: test.cross_source_collisions,
            // The runner's frontend has no local font family: every glyph
            // comes from the style's glyph ranges.
            local_ideograph_font_family: None,
            antialias: false,
            ..MapOptions::default()
        },
    );
    map.set_viewport(f64::from(test.width), f64::from(test.height));
    // Native's defaults: zoom 0 to 25.5, pitch up to 60 degrees.
    map.set_constraints(orb_api::Constraints {
        min_zoom: 0.0,
        max_zoom: 25.5,
        max_pitch: test.max_pitch,
        ..orb_api::Constraints::WORLD
    });
    map
}

/// What rendering a test gave.
pub(crate) struct Rendered {
    /// Premultiplied, as Native's frontend reads back.
    pub(crate) image: Option<Image>,
    pub(crate) failure: Option<&'static str>,
    pub(crate) map: Option<Map>,
}

/// Renders `text` natively: the style, its operations (with the stills
/// they ask for), then the still or tiles the runner compares.
fn render_native(
    runtime: &Runtime,
    test: &Test,
    text: &str,
    assets: &Path,
    reasons: &mut BTreeSet<String>,
    diagnostics: &mut Vec<String>,
) -> Result<Rendered, String> {
    let mut map = map(runtime, test);
    map.set_style_json(text).map_err(|e| e.to_string())?;
    record_warnings(&mut map, reasons, diagnostics);
    let still = |map: &mut Map, diagnostics: &mut Vec<String>| match map.snapshot(
        test.width,
        test.height,
        test.pixel_ratio,
    ) {
        Ok(image) => Some(image),
        Err(e) => {
            diagnostics.push(format!("render: {e}"));
            None
        }
    };
    for op in &test.operations {
        if renders_first(op) && still(&mut map, diagnostics).is_none() {
            return Ok(Rendered {
                image: None,
                failure: Some("render"),
                map: Some(map),
            });
        }
        record_warnings(&mut map, reasons, diagnostics);
        if !reasons.is_empty() {
            return Ok(Rendered {
                image: None,
                failure: None,
                map: Some(map),
            });
        }
        if let Err(e) = apply(&mut map, op, assets) {
            diagnostics.push(format!("operation {op}: {e}"));
        }
    }
    let image = if test.tile_mode {
        tiles(&mut map, test, diagnostics)
    } else {
        still(&mut map, diagnostics).map(|image| Image {
            width: image.width,
            height: image.height,
            pixels: image.pixels,
        })
    };
    record_warnings(&mut map, reasons, diagnostics);
    let failure = image.is_none().then_some("render");
    Ok(Rendered {
        image,
        failure,
        map: Some(map),
    })
}

/// Tile mode: the tiles at the camera's zoom that its 512-pixel view
/// covers, each drawn as a tile and laid side by side, as the runner lays
/// them out.
fn tiles(map: &mut Map, test: &Test, diagnostics: &mut Vec<String>) -> Option<Image> {
    // The style's camera, once it is in.
    if let Err(e) = map.snapshot(1, 1, 1.0) {
        diagnostics.push(format!("render: {e}"));
        return None;
    }
    let camera = map.camera();
    let z = map.zoom().max(0.0) as u8;
    let world = TILE * f64::from(1u32 << z);
    let lat = camera.target.lat.to_radians();
    let cx = (camera.target.lon + 180.0) / 360.0 * world;
    let cy = (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / std::f64::consts::PI) / 2.0 * world;
    // A view centred on a tile covers that tile alone, give or take the
    // rounding of its centre's coordinates.
    let range = |c: f64| {
        let first = ((c - TILE / 2.0) / TILE + 1e-6).floor().max(0.0) as u32;
        let last = (((c + TILE / 2.0) / TILE - 1e-6).ceil() as u32).saturating_sub(1);
        first..=last.max(first).min((1 << z) - 1)
    };
    let (xs, ys) = (range(cx), range(cy));
    let tile_px = (TILE * test.pixel_ratio) as u32;
    let columns = xs.clone().count() as u32;
    let rows = ys.clone().count() as u32;
    let mut image = Image {
        width: columns * tile_px,
        height: rows * tile_px,
        pixels: vec![0; (columns * tile_px * rows * tile_px * 4) as usize],
    };
    let tms = TileMatrixSet::by_id("WebMercatorQuad").expect("built in");
    for (row, y) in ys.enumerate() {
        for (column, x) in xs.clone().enumerate() {
            let tile = match map.render_tile(&TileRequest {
                tile_matrix_set: tms.clone(),
                z,
                col: x,
                row: y,
                size: TILE as u32,
                scale: test.pixel_ratio,
            }) {
                Ok(tile) => tile,
                Err(e) => {
                    diagnostics.push(format!("render tile {z}/{x}/{y}: {e}"));
                    return None;
                }
            };
            let stride = (image.width * 4) as usize;
            for line in 0..tile.height as usize {
                let from =
                    &tile.pixels[line * tile.width as usize * 4..][..tile.width as usize * 4];
                let at = (row * tile_px as usize + line) * stride + column * tile_px as usize * 4;
                image.pixels[at..at + from.len()].copy_from_slice(from);
            }
        }
    }
    Some(image)
}

/// The test's style with each `source-replace` pair's sources swapped, one
/// variant per pair, as the runner reruns a passing test.
fn variants(style: &Value, test: &Test) -> Vec<String> {
    test.replacements
        .iter()
        .filter_map(|(target, with)| {
            let mut style = style.clone();
            let sources = style.get_mut("sources")?.as_object_mut()?;
            let (a, b) = (sources.get(target)?.clone(), sources.get(with)?.clone());
            if !a.is_object() || !b.is_object() {
                return None;
            }
            sources.insert(target.clone(), b);
            sources.insert(with.clone(), a);
            Some(style.to_string())
        })
        .collect()
}

/// Renders one style on `backend`.
fn render(
    backend: &mut Backend,
    test: &Test,
    text: &str,
    assets: &Path,
    output: &Path,
    reasons: &mut BTreeSet<String>,
    diagnostics: &mut Vec<String>,
) -> Result<Rendered, String> {
    match backend {
        Backend::Native(runtime) => {
            render_native(runtime, test, text, assets, reasons, diagnostics)
        }
        Backend::Chrome(_) if test.tile_mode => {
            reasons.insert("host:tile-mode".into());
            Ok(Rendered {
                image: None,
                failure: None,
                map: None,
            })
        }
        Backend::Chrome(chrome) => {
            let (width, height) = test.physical_size();
            if width.max(height) > chrome.max_size {
                reasons.insert("device:canvas-size".into());
                return Ok(Rendered {
                    image: None,
                    failure: None,
                    map: None,
                });
            }
            let response = chrome.request(request(
                test,
                text,
                json!({"output": output.join("readback.rgba")}),
            ))?;
            if !response["error"].is_null() {
                diagnostics.push(response["error"].to_string());
                return Ok(Rendered {
                    image: None,
                    failure: Some("browser-render"),
                    map: None,
                });
            }
            for warning in response["warnings"].as_array().into_iter().flatten() {
                note_warning(
                    warning.as_str().unwrap_or_default().to_string(),
                    reasons,
                    diagnostics,
                );
            }
            let path = output.join("readback.rgba");
            let pixels = std::fs::read(&path).map_err(|e| e.to_string())?;
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
            let w = response["width"].as_u64().unwrap_or(0) as u32;
            let h = response["height"].as_u64().unwrap_or(0) as u32;
            if pixels.len() != w as usize * h as usize * 4 {
                return Err("Chrome returned the wrong number of RGBA bytes".into());
            }
            // The page returns straight colour; Native compares premultiplied.
            let mut image = Image {
                width: w,
                height: h,
                pixels,
            };
            image.premultiply_native();
            Ok(Rendered {
                image: Some(image),
                failure: None,
                map: None,
            })
        }
    }
}

/// The Chrome driver's request for a still (`web/demo/fixture.html`).
fn request(test: &Test, text: &str, extra: Value) -> Value {
    let mut request = json!({
        "native": true,
        "style": text,
        "width": test.width,
        "height": test.height,
        "ratio": test.pixel_ratio,
        "timeout_ms": 60000,
        "cross_source_collisions": test.cross_source_collisions,
        "max_pitch": test.max_pitch,
        "operations": test.operations,
    });
    request
        .as_object_mut()
        .expect("an object")
        .extend(extra.as_object().expect("an object").clone());
    request
}

/// The result of comparing a premultiplied image with expected images.
struct Compared {
    difference: f64,
    expected: PathBuf,
    diff: Vec<u8>,
    /// The expected image's size differs from the actual one's.
    wrong_size: bool,
}

/// Compares as `checkRenderTestResults` does: each expected image,
/// premultiplied, until one is within `allowed`; the smallest difference.
fn compare(actual: &Image, expected: &[PathBuf], allowed: f64) -> Result<Compared, String> {
    let mut best: Option<Compared> = None;
    for path in expected {
        let mut image = Image::read(path)?;
        image.premultiply_native();
        let compared = if (image.width, image.height) == (actual.width, actual.height) {
            let (count, diff) = pixelmatch_cpp::compare(
                &actual.pixels,
                &image.pixels,
                actual.width as usize,
                actual.height as usize,
                THRESHOLD,
            );
            Compared {
                difference: count as f64 / (f64::from(actual.width) * f64::from(actual.height)),
                expected: path.clone(),
                diff,
                wrong_size: false,
            }
        } else {
            Compared {
                difference: 1.0,
                expected: path.clone(),
                diff: [255, 0, 0, 255].repeat((actual.width * actual.height) as usize),
                wrong_size: true,
            }
        };
        let done = compared.difference <= allowed;
        if best
            .as_ref()
            .is_none_or(|b| compared.difference < b.difference)
        {
            best = Some(compared);
        }
        if done {
            break;
        }
    }
    best.ok_or_else(|| "no expected image".into())
}

pub(crate) fn run_render_suite(root: &Path, filter: &str, name: &str) -> Result<Value, String> {
    let manifest = Manifest::read(root)?;
    let tests = discover(root, "render-tests", filter)?;
    let output = root.join("docs/scratch/rt/native-render").join(name);
    if output.exists() {
        std::fs::remove_dir_all(&output).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(output.join("failures")).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let assets = assets(root);
    let mut backend = match name {
        "chrome" => Backend::Chrome(chrome::Chrome::new(root, &assets, Assets::Native)?),
        _ => crate::runtime(&assets, Assets::Native)?,
    };
    let gl_js = root.join("fixtures/gl-js/render");
    let mut results = Vec::new();
    let mut prefixes: BTreeMap<String, Counts> = BTreeMap::new();
    let mut skip_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut ignored_counts = Counts::default();
    let mut opinion = SecondOpinion::default();
    for (id, path) in tests {
        let result = run_render_test(
            &manifest,
            &id,
            &path,
            &output,
            &assets,
            &gl_js,
            &mut backend,
        )?;
        let status = result["status"].as_str().expect("our status");
        let rel = id.strip_prefix("render-tests/").unwrap_or(&id);
        prefixes
            .entry(crate::prefix(rel).into())
            .or_default()
            .add(status);
        if result["native_ignored"].is_string() {
            ignored_counts.add(status);
        }
        opinion.add(&result["gl_js"], status);
        for reason in result["skip_reasons"].as_array().into_iter().flatten() {
            *skip_counts
                .entry(reason.as_str().expect("our reason").into())
                .or_default() += 1;
        }
        if status == "fail" {
            eprintln!(
                "FAIL native-render {name} {id}: {} ({:.2}% pixels)",
                result["failure_reason"],
                result["difference"].as_f64().unwrap_or(0.0) * 100.0
            );
        }
        results.push(result);
    }
    let total = prefixes.values().fold(Counts::default(), |mut t, c| {
        t.pass += c.pass;
        t.fail += c.fail;
        t.skip += c.skip;
        t
    });
    let mut skip_counts: Vec<_> = skip_counts.into_iter().collect();
    skip_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let browser = match &backend {
        Backend::Chrome(c) => c.version.clone(),
        Backend::Native(_) => Value::Null,
    };
    let summary = json!({
        "suite": "native-render",
        "filter": filter,
        "platform": {"name":name,"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"browser":browser},
        "comparison": {"metric":"pixelmatch-cpp-61f433cb-yiq-premultiplied", "threshold":THRESHOLD, "default_allowed":ALLOWED},
        "total_seconds": started.elapsed().as_secs_f64(),
        "totals": total.json(),
        "native_ignored": ignored_counts.json(),
        "gl_js": opinion.json(),
        "prefixes": prefixes.iter().map(|(p,c)| (p.clone(), c.json())).collect::<BTreeMap<_,_>>(),
        "skip_reasons": skip_counts.iter().map(|(reason,count)| json!({"reason":reason,"count":count})).collect::<Vec<_>>(),
        "tests": results,
    });
    write_json(&output.join("summary.json"), &summary)?;
    Ok(summary)
}

/// One render test: skipped, or rendered (with any source-replace reruns)
/// and compared with Native's expected images, and with GL JS's where the
/// suites share it.
fn run_render_test(
    manifest: &Manifest,
    id: &str,
    path: &Path,
    output: &Path,
    assets: &Path,
    gl_js: &Path,
    backend: &mut Backend,
) -> Result<Value, String> {
    let started = Instant::now();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let style: Value = serde_json::from_str(&text).map_err(|e| format!("{id}: {e}"))?;
    let test = Test::read(&style);
    let mut result = json!({"id": id, "metadata": test.value});
    let (skip, native_ignored) = ignored(manifest, id);
    if let Some(reason) = &native_ignored {
        result["native_ignored"] = json!(reason);
    }
    let mut reasons = test.skips.clone();
    reasons.extend(skip);
    let mut diagnostics = dem::lint(&style, test.width, test.height, assets);
    if !diagnostics.is_empty() {
        reasons.insert(dem::IMPLAUSIBLE.into());
    }
    classify(&text, &mut reasons, &mut diagnostics);
    if !reasons.is_empty() {
        result["diagnostics"] = json!(diagnostics);
        finish_skip(&mut result, &reasons, started);
        return Ok(result);
    }
    let dir = path.parent().expect("style parent");
    let expected = manifest.expected(id, dir, ".png");
    if expected.is_empty() {
        return Err(format!("{id}: missing expected*.png"));
    }
    let mut failure = None;
    let mut last: Option<(Image, Compared)> = None;
    for (i, text) in std::iter::once(text.clone())
        .chain(variants(&style, &test))
        .enumerate()
    {
        let rendered = render(
            backend,
            &test,
            &text,
            assets,
            output,
            &mut reasons,
            &mut diagnostics,
        )?;
        if !reasons.is_empty() {
            result["diagnostics"] = json!(diagnostics);
            finish_skip(&mut result, &reasons, started);
            return Ok(result);
        }
        let Some(image) = rendered.image else {
            failure = rendered.failure.or(Some("render"));
            break;
        };
        let compared = compare(&image, &expected, test.allowed)?;
        if compared.wrong_size {
            failure = Some("expected-size");
        } else if compared.difference > test.allowed {
            failure = Some(if i == 0 { "pixels" } else { "source-replace" });
        }
        // Native's runner judges the picture alone: warnings and load
        // errors stay in the report without failing a test.
        last = Some((image, compared));
        if failure.is_some() {
            break;
        }
    }
    result["diagnostics"] = json!(diagnostics);
    if let Some((image, compared)) = &last {
        let straight = image.unpremultiply_native();
        let image_path = output.join("images").join(id).join("actual.png");
        std::fs::create_dir_all(image_path.parent().expect("parent")).map_err(|e| e.to_string())?;
        straight.write(&image_path)?;
        result["image"] = json!(format!("images/{id}/actual.png"));
        result["expected"] = json!(
            compared
                .expected
                .strip_prefix(mirror_root(assets))
                .unwrap_or(&compared.expected)
        );
        result["difference"] = json!(compared.difference);
        result["different_pixels"] = json!(
            (compared.difference * f64::from(image.width) * f64::from(image.height)).round() as u64
        );
        result["gl_js"] =
            second_opinion(id, &style, image, &compared.expected, test.allowed, gl_js)?;
        if failure.is_some() {
            let dir = output.join("failures").join(id);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            straight.write(&dir.join("actual.png"))?;
            Image {
                width: image.width,
                height: image.height,
                pixels: compared.diff.clone(),
            }
            .write(&dir.join("diff.png"))?;
            result["actual"] = json!(format!("failures/{id}/actual.png"));
            result["diff"] = json!(format!("failures/{id}/diff.png"));
        }
    }
    result["status"] = json!(if failure.is_some() { "fail" } else { "pass" });
    if let Some(failure) = failure {
        result["failure_reason"] = json!(failure);
    }
    result["seconds"] = json!(started.elapsed().as_secs_f64());
    Ok(result)
}

fn mirror_root(assets: &Path) -> &Path {
    assets.parent().expect("the mirror")
}

/// Skips for what the style uses that Orbis doesn't support, as the GL JS
/// suites decide them: parser diagnostics.
fn classify(text: &str, reasons: &mut BTreeSet<String>, diagnostics: &mut Vec<String>) {
    match orb_style::v8::parse(text) {
        Ok(style) => reasons.extend(
            scope::reasons(&style)
                .into_iter()
                .filter(|r| r != "property:interactive"),
        ),
        Err(errors) => {
            reasons.insert("style-parse".into());
            reasons.extend(
                errors
                    .errors
                    .iter()
                    .filter(|d| d.message.starts_with("Expression"))
                    .map(|d| scope::parse_reason(&d.message)),
            );
            reasons.extend(
                errors
                    .warnings
                    .iter()
                    .map(|w| scope::parse_reason(&w.message)),
            );
            diagnostics.extend(
                errors
                    .errors
                    .iter()
                    .chain(&errors.warnings)
                    .map(ToString::to_string),
            );
        }
    }
}

/// A shared test, judged against GL JS's expected images too: whether the
/// two engines' expectations agree (Native's metric and `allowed`), and
/// which of them Orbis's image matches. `null` for a test GL JS doesn't
/// have.
fn second_opinion(
    id: &str,
    style: &Value,
    actual: &Image,
    native_expected: &Path,
    allowed: f64,
    gl_js: &Path,
) -> Result<Value, String> {
    let rel = id.strip_prefix("render-tests/").unwrap_or(id);
    let dir = gl_js.join(rel);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(Value::Null);
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("expected") && n.ends_with(".png"))
        })
        .collect();
    if paths.is_empty() {
        return Ok(Value::Null);
    }
    paths.sort();
    let gl_style: Option<Value> = std::fs::read_to_string(dir.join("style.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    let same_style = gl_style.as_ref() == Some(style);
    let mut native = Image::read(native_expected)?;
    native.premultiply_native();
    let measure = |a: &Image, b: &Image| -> Option<f64> {
        ((a.width, a.height) == (b.width, b.height)).then(|| {
            let (count, _) = pixelmatch_cpp::compare(
                &a.pixels,
                &b.pixels,
                a.width as usize,
                a.height as usize,
                THRESHOLD,
            );
            count as f64 / (f64::from(a.width) * f64::from(a.height))
        })
    };
    // GL JS's closest expected image to Native's, and to Orbis's.
    let (mut references, mut orbis) = (None::<f64>, None::<f64>);
    for path in &paths {
        let mut image = Image::read(path)?;
        image.premultiply_native();
        if let Some(d) = measure(&image, &native) {
            references = Some(references.map_or(d, |r| r.min(d)));
        }
        if let Some(d) = measure(&image, actual) {
            orbis = Some(orbis.map_or(d, |r| r.min(d)));
        }
    }
    let (Some(references), Some(orbis)) = (references, orbis) else {
        return Ok(json!({"same_style": same_style, "comparable": false}));
    };
    Ok(json!({
        "same_style": same_style,
        "comparable": true,
        "references_difference": references,
        "references_agree": references <= allowed,
        "difference": orbis,
        "matches": orbis <= allowed,
    }))
}

/// How shared tests' expected images compare between the engines.
#[derive(Default)]
struct SecondOpinion {
    shared: usize,
    comparable: usize,
    same_style: usize,
    agree: usize,
    /// For tests whose references disagree: Orbis matches GL JS's, Native's
    /// (passes), both or neither.
    disagree: BTreeMap<&'static str, usize>,
    /// Orbis's image matches GL JS's expected image, among comparable tests.
    matches_gl_js: usize,
}

impl SecondOpinion {
    fn add(&mut self, opinion: &Value, status: &str) {
        if opinion.is_null() {
            return;
        }
        self.shared += 1;
        if opinion["same_style"] == true {
            self.same_style += 1;
        }
        if opinion["comparable"] != true {
            return;
        }
        self.comparable += 1;
        let gl_js = opinion["matches"] == true;
        if gl_js {
            self.matches_gl_js += 1;
        }
        if opinion["references_agree"] == true {
            self.agree += 1;
            return;
        }
        let key = match (status == "pass", gl_js) {
            (true, true) => "both",
            (true, false) => "native",
            (false, true) => "gl_js",
            (false, false) => "neither",
        };
        *self.disagree.entry(key).or_default() += 1;
    }

    fn json(&self) -> Value {
        json!({
            "shared": self.shared,
            "same_style": self.same_style,
            "comparable": self.comparable,
            "references_agree": self.agree,
            "references_disagree": self.disagree,
            "matches_gl_js": self.matches_gl_js,
        })
    }
}

/// Native's query suite: each test renders as a render test does, then
/// `queryRenderedFeatures` runs at its point or box; the answer must equal
/// `expected.json` in order, numbers to six significant figures, as the
/// runner's `deepEqual` compares.
pub(crate) fn run_query_suite(root: &Path, filter: &str, name: &str) -> Result<Value, String> {
    let manifest = Manifest::read(root)?;
    let tests = discover(root, "query-tests", filter)?;
    let output = root.join("docs/scratch/rt/native-query").join(name);
    if output.exists() {
        std::fs::remove_dir_all(&output).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let assets = assets(root);
    let mut backend = match name {
        "chrome" => Backend::Chrome(chrome::Chrome::new(root, &assets, Assets::Native)?),
        _ => crate::runtime(&assets, Assets::Native)?,
    };
    let mut totals = Counts::default();
    let mut ignored_counts = Counts::default();
    let mut unordered = 0;
    let mut prefixes: BTreeMap<String, Counts> = BTreeMap::new();
    let mut skip_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut results = Vec::new();
    for (id, path) in tests {
        let result = run_query_test(&manifest, &id, &path, &output, &assets, &mut backend)?;
        let status = result["status"].as_str().expect("our status");
        if result["same_features"] == true && status == "fail" {
            unordered += 1;
        }
        if result["native_ignored"].is_string() {
            ignored_counts.add(status);
        }
        for reason in result["skip_reasons"].as_array().into_iter().flatten() {
            *skip_counts
                .entry(reason.as_str().expect("our reason").into())
                .or_default() += 1;
        }
        let rel = id.strip_prefix("query-tests/").unwrap_or(&id);
        prefixes
            .entry(rel.split('/').next().unwrap_or(rel).to_string())
            .or_default()
            .add(status);
        totals.add(status);
        results.push(result);
    }
    let mut skip_counts: Vec<_> = skip_counts.into_iter().collect();
    skip_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let summary = json!({
        "suite": "native-query",
        "platform": name,
        "total_seconds": started.elapsed().as_secs_f64(),
        "totals": totals.json(),
        "native_ignored": ignored_counts.json(),
        "out_of_order": unordered,
        "prefixes": prefixes.iter().map(|(p, c)| (p.clone(), c.json())).collect::<BTreeMap<_, _>>(),
        "skip_reasons": skip_counts.iter().map(|(reason,count)| json!({"reason":reason,"count":count})).collect::<Vec<_>>(),
        "tests": results,
    });
    write_json(&output.join("summary.json"), &summary)?;
    println!("\nnative-query {name}  prefix          pass  fail  skip  in-scope pass rate");
    for (prefix, counts) in &prefixes {
        print!("{:<19} ", "");
        counts.print(prefix);
    }
    print!("{:<19} ", "");
    totals.print("TOTAL");
    println!(
        "  Native ignores {} of them: {} pass, {} fail",
        ignored_counts.pass + ignored_counts.fail + ignored_counts.skip,
        ignored_counts.pass,
        ignored_counts.fail
    );
    println!("  {unordered} failures have Native's features in another order");
    Ok(summary)
}

fn run_query_test(
    manifest: &Manifest,
    id: &str,
    path: &Path,
    output: &Path,
    assets: &Path,
    backend: &mut Backend,
) -> Result<Value, String> {
    let started = Instant::now();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let style: Value = serde_json::from_str(&text).map_err(|e| format!("{id}: {e}"))?;
    let test = Test::read(&style);
    let mut result = json!({"id": id});
    let (skip, native_ignored) = ignored(manifest, id);
    if let Some(reason) = &native_ignored {
        result["native_ignored"] = json!(reason);
    }
    let mut reasons = test.skips.clone();
    reasons.extend(skip);
    let Some(query) = test.query else {
        reasons.insert("query:geometry".into());
        finish_skip(&mut result, &reasons, started);
        return Ok(result);
    };
    let mut diagnostics = dem::lint(&style, test.width, test.height, assets);
    if !diagnostics.is_empty() {
        reasons.insert(dem::IMPLAUSIBLE.into());
    }
    classify(&text, &mut reasons, &mut diagnostics);
    if !reasons.is_empty() {
        result["diagnostics"] = json!(diagnostics);
        finish_skip(&mut result, &reasons, started);
        return Ok(result);
    }
    let actual = match backend {
        Backend::Native(runtime) => {
            let rendered = render_native(
                runtime,
                &test,
                &text,
                assets,
                &mut reasons,
                &mut diagnostics,
            )?;
            match (rendered.failure, rendered.map) {
                (None, Some(mut map)) if reasons.is_empty() => {
                    // Queries read the last interactive frame, which stills
                    // don't replace: draw the settled map as one.
                    let mut target = orb_api::Target::texture(
                        runtime,
                        test.width,
                        test.height,
                        test.pixel_ratio,
                    );
                    match map.render_until_idle(&mut target, orb_api::STILL_TIMEOUT) {
                        Ok(_) => Some(crate::query::query_native(&mut map, query)?),
                        Err(e) => {
                            diagnostics.push(format!("render: {e}"));
                            None
                        }
                    }
                }
                _ => None,
            }
        }
        Backend::Chrome(chrome) => {
            let response = chrome.request(request(&test, &text, json!({"query": query})))?;
            if response["error"].is_null() {
                let features = response["features"]
                    .as_str()
                    .ok_or("Chrome query: no features")?;
                Some(
                    serde_json::from_str(features)
                        .map_err(|e| format!("Chrome query JSON: {e}"))?,
                )
            } else {
                diagnostics.push(response["error"].to_string());
                None
            }
        }
    };
    if !reasons.is_empty() {
        result["diagnostics"] = json!(diagnostics);
        finish_skip(&mut result, &reasons, started);
        return Ok(result);
    }
    result["diagnostics"] = json!(diagnostics);
    let Some(mut actual) = actual else {
        result["status"] = json!("fail");
        result["failure_reason"] = json!("render");
        result["seconds"] = json!(started.elapsed().as_secs_f64());
        return Ok(result);
    };
    for feature in actual.as_array_mut().into_iter().flatten() {
        if let Some(f) = feature.as_object_mut() {
            f.remove("layer");
        }
    }
    let dir = path.parent().expect("style parent");
    let expected = manifest.expected(id, dir, ".json");
    if expected.is_empty() {
        return Err(format!("{id}: missing expected*.json"));
    }
    // Every expected file must match, as the runner checks them all.
    let mut equal = true;
    let mut same_features = true;
    let mut expected_count = 0;
    for path in &expected {
        let expected: Value =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{id}: {e}"))?)
                .map_err(|e| format!("{id}: {e}"))?;
        expected_count = expected.as_array().map_or(0, Vec::len);
        equal &= deep_equal(&actual, &expected);
        same_features &= unordered_equal(&actual, &expected);
    }
    result["status"] = json!(if equal { "pass" } else { "fail" });
    result["actual"] = json!(actual.as_array().map_or(0, Vec::len));
    result["expected"] = json!(expected_count);
    if !equal {
        result["failure_reason"] = json!(if same_features { "order" } else { "features" });
        result["same_features"] = json!(same_features);
        let dir = output.join("failures").join(id);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        write_json(&dir.join("actual.json"), &actual)?;
        eprintln!(
            "FAIL native-query {id}: {} features, expected {}{}",
            result["actual"],
            result["expected"],
            if same_features {
                " (another order)"
            } else {
                ""
            }
        );
    }
    result["diagnostics"] = json!(diagnostics);
    result["seconds"] = json!(started.elapsed().as_secs_f64());
    Ok(result)
}

/// Native's `stripPrecision`: six significant figures, floored twice.
pub(crate) fn strip_precision(n: f64) -> f64 {
    if n == 0.0 {
        return n;
    }
    let multiplier = 10f64.powf((6.0 - n.abs().log10().ceil()).max(0.0));
    let first = (n * multiplier).floor() / multiplier;
    (first * multiplier).floor() / multiplier
}

/// Native's `deepEqual` on JSON values: numbers equal after
/// [`strip_precision`], arrays in order, objects by key, and strings equal
/// or equal as numbers.
pub(crate) fn deep_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            strip_precision(x.as_f64().unwrap_or(f64::NAN))
                == strip_precision(y.as_f64().unwrap_or(f64::NAN))
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| deep_equal(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| deep_equal(v, w)))
        }
        (Value::String(x), Value::String(y)) => {
            x == y
                || matches!((x.parse::<f64>(), y.parse::<f64>()),
                    (Ok(x), Ok(y)) if strip_precision(x) == strip_precision(y))
        }
        _ => a == b,
    }
}

/// Whether two feature lists hold the same features in any order.
fn unordered_equal(actual: &Value, expected: &Value) -> bool {
    let (Some(a), Some(e)) = (actual.as_array(), expected.as_array()) else {
        return false;
    };
    if a.len() != e.len() {
        return false;
    }
    let mut unmatched: Vec<&Value> = e.iter().collect();
    a.iter().all(|feature| {
        unmatched
            .iter()
            .position(|e| deep_equal(feature, e))
            .map(|i| unmatched.swap_remove(i))
            .is_some()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_follows_natives_parser() {
        let t = Test::read(&json!({}));
        assert_eq!(
            (t.width, t.height, t.allowed, t.max_pitch),
            (512, 512, ALLOWED, 60.0)
        );
        let t = Test::read(&json!({"metadata":{"test":{
            "mapMode":"tile", "width": 100, "pixelRatio": 2, "debug": false,
            "fadeDuration": 100, "threshold": 0.5,
            "operations": [["wait"], ["setTileLodScale", 2], ["easeTo", {}], ["setStyle", "local://a"]]
        }}}));
        assert!(t.tile_mode);
        assert_eq!(t.physical_size(), (1024, 1024));
        let skips: Vec<&str> = t.skips.iter().map(String::as_str).collect();
        assert_eq!(
            skips,
            [
                "metadata:debug",
                "native-runner:easeTo",
                "operation:setTileLodScale"
            ]
        );
        let q = Test::read(&json!({"metadata":{"test":{"queryGeometry":[[1,2],[3,4]]}}}));
        assert_eq!(q.query, Some([1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn query_answers_compare_in_order_to_six_figures() {
        assert!(deep_equal(
            &json!([1.0000001, "2"]),
            &json!([1, "2.0000001"])
        ));
        assert!(!deep_equal(&json!([1, 2]), &json!([2, 1])));
        assert!(unordered_equal(&json!([1, 2]), &json!([2, 1])));
        assert_eq!(strip_precision(1234.5678), 1234.56);
        assert_eq!(strip_precision(0.0), 0.0);
    }
}

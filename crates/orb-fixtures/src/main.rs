//! Sequential offscreen render measurements against the read-only GL JS
//! and MapLibre Native mirrors.

#![forbid(unsafe_code)]

mod chrome;
mod dem;
mod fetch;
mod images;
mod metadata;
mod native;
mod native_expression;
mod operations;
mod pixelmatch;
mod pixelmatch_cpp;
mod platforms;
mod query;
mod scope;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use operations::Next;
use orb_api::{Event, Map, MapEvent, MapOptions, Runtime, RuntimeOptions, Target};
use serde_json::{Value, json};

use images::Image;
use metadata::Metadata;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("orb-fixtures: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut filter = String::new();
    let mut platform = "native".to_string();
    let mut suite = "render".to_string();
    const USAGE: &str = "usage: orb-fixtures [--suite render|query|native-render|native-query|native-expression] [--filter <substring>] [--platform native|chrome|lavapipe|all]";
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--suite" => suite = args.next().ok_or(USAGE)?,
            "--filter" => filter = args.next().ok_or("--filter needs a substring")?,
            "--platform" => {
                platform = args
                    .next()
                    .ok_or("--platform needs native|chrome|lavapipe|all")?
            }
            _ => return Err(format!("unknown argument {arg}; {USAGE}")),
        }
    }
    if !["native", "chrome", "lavapipe", "all"].contains(&platform.as_str()) {
        return Err(format!("unknown platform {platform}"));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|e| e.to_string())?;
    match suite.as_str() {
        "render" => platforms::run(&root, &filter, &platform, Suite::Render),
        "native-render" => platforms::run(&root, &filter, &platform, Suite::NativeRender),
        "query" | "native-query" => {
            if !["native", "chrome", "all"].contains(&platform.as_str()) {
                return Err("the query suites run natively and in Chrome".into());
            }
            for name in ["native", "chrome"] {
                if platform == name || platform == "all" {
                    if suite == "query" {
                        query::run_suite(&root, &filter, name)?;
                    } else {
                        native::run_query_suite(&root, &filter, name)?;
                    }
                }
            }
            Ok(())
        }
        // The expressions evaluate on the CPU, the same code on every
        // platform.
        "native-expression" => native_expression::run(&root, &filter),
        _ => Err(format!("unknown suite {suite}; {USAGE}")),
    }
}

/// The render suites, which run on every platform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Suite {
    /// GL JS's render tests (`fixtures/gl-js/render`).
    Render,
    /// MapLibre Native's (`fixtures/native/integration/render-tests`).
    NativeRender,
}

impl Suite {
    /// Its name for `--suite`, and its report directory under
    /// `docs/scratch/rt` (GL JS's render suite writes there directly).
    pub(crate) fn name(self) -> &'static str {
        match self {
            Suite::Render => "render",
            Suite::NativeRender => "native-render",
        }
    }

    /// Runs the suite on one platform; its summary.
    pub(crate) fn run(self, root: &Path, filter: &str, platform: &str) -> Result<Value, String> {
        match self {
            Suite::Render => run_suite(root, filter, platform),
            Suite::NativeRender => native::run_render_suite(root, filter, platform),
        }
    }
}

enum Backend {
    /// `orb-api`'s runtime (`native`, `lavapipe`).
    Native(Runtime),
    Chrome(chrome::Chrome),
}

/// The runtime every native fixture's map is made from, fetching a
/// mirror's `assets` the way its suite's test server or cache serves them.
pub(crate) fn runtime(assets: &Path, kind: fetch::Assets) -> Result<Backend, String> {
    let assets = assets.to_path_buf();
    let runtime = Runtime::new(&RuntimeOptions::default()).map_err(|e| e.to_string())?;
    runtime.set_request_transform(Some(Arc::new(move |url: &str| {
        let url = fetch::resolve(&assets, url, kind)?;
        Some(orb_api::Request {
            url,
            headers: Vec::new(),
        })
    })));
    Ok(Backend::Native(runtime))
}

/// The mirror's assets, which `addImage` operations name.
pub(crate) fn assets(root: &Path) -> PathBuf {
    root.join("fixtures/gl-js/assets")
}

fn run_suite(root: &Path, filter: &str, name: &str) -> Result<Value, String> {
    let fixtures = root.join("fixtures/gl-js/render");
    if !fixtures.is_dir() {
        return Err(format!(
            "{} is missing; run `cargo xtask fixtures` to fetch the GL JS fixture mirror",
            fixtures.display()
        ));
    }
    let output = root.join("docs/scratch/rt").join(name);
    if output.exists() {
        for entry in std::fs::read_dir(&output).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                std::fs::remove_dir_all(path)
            } else {
                std::fs::remove_file(path)
            }
            .map_err(|e| e.to_string())?;
        }
    }
    std::fs::create_dir_all(output.join("failures")).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let mut backend = match name {
        "chrome" => Backend::Chrome(chrome::Chrome::new(
            root,
            &assets(root),
            fetch::Assets::GlJs,
        )?),
        _ => runtime(&assets(root), fetch::Assets::GlJs)?,
    };
    let mut paths = Vec::new();
    discover(&fixtures, &mut paths)?;
    paths.sort();
    paths.retain(|path| test_id(path, &fixtures).contains(filter));
    let assets = assets(root);
    let mut results = Vec::new();
    let mut prefixes: BTreeMap<String, Counts> = BTreeMap::new();
    let mut skip_counts: BTreeMap<String, usize> = BTreeMap::new();
    for path in paths {
        let id = test_id(&path, &fixtures);
        let result = run_test(&path, &id, &output, &assets, &mut backend)?;
        let status = result["status"].as_str().expect("our status");
        prefixes.entry(prefix(&id).into()).or_default().add(status);
        if let Some(reasons) = result["skip_reasons"].as_array() {
            for reason in reasons {
                *skip_counts
                    .entry(reason.as_str().expect("our reason").into())
                    .or_default() += 1;
            }
        }
        if status == "fail" {
            eprintln!(
                "FAIL {name} {id}: {} ({:.2}% pixels)",
                result["failure_reason"],
                result["difference"].as_f64().unwrap_or(0.0) * 100.0
            );
        }
        results.push(result);
    }
    let total = prefixes.values().fold(Counts::default(), |mut total, c| {
        total.pass += c.pass;
        total.fail += c.fail;
        total.skip += c.skip;
        total
    });
    let mut skip_counts: Vec<_> = skip_counts.into_iter().collect();
    skip_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let browser = match &backend {
        Backend::Chrome(c) => c.version.clone(),
        _ => Value::Null,
    };
    let summary = json!({
        "filter": filter,
        "platform": {"name":name,"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"browser":browser},
        "comparison": {"metric":"pixelmatch-8.0.0-oklab-hyab", "includeAA":false, "checkerboard":true, "default_threshold":0.1285, "default_allowed":0.00025},
        "total_seconds": started.elapsed().as_secs_f64(),
        "totals": total.json(),
        "prefixes": prefixes.iter().map(|(p,c)| (p.clone(), c.json())).collect::<BTreeMap<_,_>>(),
        "skip_reasons": skip_counts.iter().map(|(reason,count)| json!({"reason":reason,"count":count})).collect::<Vec<_>>(),
        "tests": results,
    });
    write_json(&output.join("summary.json"), &summary)?;
    Ok(summary)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

fn discover(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            discover(&entry.path(), paths)?;
        } else if entry.file_name() == "style.json" {
            paths.push(entry.path());
        }
    }
    Ok(())
}

fn test_id(path: &Path, fixtures: &Path) -> String {
    path.parent()
        .expect("style parent")
        .strip_prefix(fixtures)
        .expect("fixture path")
        .to_string_lossy()
        .replace('\\', "/")
}

/// The family a fixture is of, by its top-level directory: a layer type, or
/// one of the larger groups of the rest. Longer prefixes first, so
/// `fill-extrusion-*` isn't counted as `fill-*`.
fn prefix(id: &str) -> &str {
    const FAMILIES: &[&str] = &[
        "background-",
        "circle-",
        "color-relief",
        "combinations",
        "feature-state",
        "fill-extrusion-",
        "fill-",
        "filter",
        "global-state",
        "heatmap-",
        "hillshade-",
        "icon-",
        "line-",
        "projection",
        "raster-",
        "regressions",
        "remove-feature-state",
        "runtime-styling",
        "sprites",
        "sky",
        "symbol-",
        "terrain",
        "text-",
        "visibility",
        "zoom-history",
    ];
    FAMILIES
        .iter()
        .find(|family| id.starts_with(*family))
        .copied()
        .unwrap_or("other")
}

/// A map set up as GL JS's harness sets up a test's.
pub(crate) fn map(runtime: &Runtime, metadata: &Metadata) -> Map {
    let mut map = Map::new(
        runtime,
        MapOptions {
            // GL JS's render tests run with `fadeDuration: 0` unless they
            // say.
            fade_duration: metadata.fade_duration,
            cross_source_collisions: metadata.cross_source_collisions,
            local_ideograph_font_family: metadata.local_ideographs.clone(),
            antialias: false,
            terrain_skirts: metadata.terrain_skirts,
            // GL JS's tests expect labels placed for the camera alone,
            // not as MapLibre Native's stabilisers keep them from the
            // placements while the test loads.
            stable_labels: false,
            ..MapOptions::default()
        },
    );
    // GL JS's map starts at its container's size, so the style's camera
    // keeps to the world in the test's viewport, not a default one.
    map.set_viewport(f64::from(metadata.width), f64::from(metadata.height));
    // GL JS's transform keeps to the world and the test's pitch limit as
    // the harness moves it.
    map.set_constraints(orb_api::Constraints {
        max_pitch: metadata.max_pitch,
        ..orb_api::Constraints::WORLD
    });
    map
}

fn run_test(
    path: &Path,
    id: &str,
    output: &Path,
    assets: &Path,
    backend: &mut Backend,
) -> Result<Value, String> {
    let started = Instant::now();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let metadata = Metadata::read(&value);
    let mut reasons = metadata.skips.clone();
    let mut diagnostics = dem::lint(&value, metadata.width, metadata.height, assets);
    if !diagnostics.is_empty() {
        reasons.insert(dem::IMPLAUSIBLE.into());
    }
    let mut map = match backend {
        Backend::Native(runtime) => Some(map(runtime, &metadata)),
        Backend::Chrome(_) => None,
    };
    match orb_style::v8::parse(&text) {
        Ok(style) => {
            reasons.extend(scope::reasons(&style));
            // set_style_json also exposes plan diagnostics absent from the parser.
            if let Some(map) = &mut map {
                map.set_style_json(&text)
                    .map_err(|e| format!("{id}: {e}"))?;
                record_warnings(map, &mut reasons, &mut diagnostics);
            }
        }
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
    let mut result = json!({"id":id, "metadata":metadata.value, "diagnostics":diagnostics});
    if !reasons.is_empty() {
        finish_skip(&mut result, &reasons, started);
        return Ok(result);
    }
    let (width, height) = metadata.physical_size();
    let mut failure = None;
    let rendered = match backend {
        Backend::Native(runtime) => render_native(
            runtime,
            map.as_mut().expect("a native map"),
            &metadata,
            assets,
            &mut reasons,
            &mut diagnostics,
        ),
        Backend::Chrome(chrome) if width.max(height) > chrome.max_size => {
            reasons.insert("device:canvas-size".into());
            None
        }
        Backend::Chrome(chrome) => {
            let (image, rendered) =
                chrome.render(&text, &metadata, &output.join("readback.rgba"))?;
            let Some(image) = image else {
                result["status"] = json!("fail");
                result["failure_reason"] = json!("browser-render");
                result["diagnostics"] = json!([rendered["error"]]);
                result["seconds"] = json!(started.elapsed().as_secs_f64());
                return Ok(result);
            };
            for warning in rendered["warnings"].as_array().expect("Chrome warnings") {
                let warning = warning.as_str().expect("warning string");
                note_warning(warning.to_string(), &mut reasons, &mut diagnostics);
            }
            Some((image, rendered))
        }
    };
    let Some((actual, rendered)) = rendered else {
        finish_skip(&mut result, &reasons, started);
        return Ok(result);
    };
    result["frames"] = rendered["frames"].clone();
    result["tiles_drawn"] = rendered["tiles_drawn"].clone();
    result["diagnostics"] = json!(diagnostics);
    if !reasons.is_empty() {
        finish_skip(&mut result, &reasons, started);
        return Ok(result);
    }
    if rendered["timed_out"] == true {
        failure = Some("timeout");
    }
    if !diagnostics.is_empty() {
        failure.get_or_insert("runtime-warning");
    }
    let image_path = output.join("images").join(id).join("actual.png");
    std::fs::create_dir_all(image_path.parent().expect("image parent"))
        .map_err(|e| e.to_string())?;
    actual.write(&image_path)?;
    result["image"] = json!(format!("images/{id}/actual.png"));
    let mut expected_paths = Vec::new();
    for entry in
        std::fs::read_dir(path.parent().expect("style parent")).map_err(|e| e.to_string())?
    {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("expected") && name.ends_with(".png") {
            expected_paths.push(entry.path());
        }
    }
    expected_paths.sort();
    if expected_paths.is_empty() {
        return Err(format!("{id}: missing expected*.png"));
    }
    let mut best: Option<(f64, Vec<u8>, PathBuf)> = None;
    for expected_path in &expected_paths {
        let expected = Image::read(expected_path)?;
        if (expected.width, expected.height) != (width, height) {
            continue;
        }
        let (count, diff) = pixelmatch::compare(
            &actual.pixels,
            &expected.pixels,
            width as usize,
            height as usize,
            metadata.threshold,
        );
        let difference = count as f64 / (f64::from(width) * f64::from(height));
        if best.as_ref().is_none_or(|b| difference < b.0) {
            best = Some((difference, diff, expected_path.clone()));
        }
    }
    let (difference, diff, expected) = best.unwrap_or_else(|| {
        failure = Some("expected-size");
        (
            1.0,
            [255, 0, 0, 255].repeat(width as usize * height as usize),
            expected_paths[0].clone(),
        )
    });
    result["expected"] = json!(
        expected
            .strip_prefix(path.parent().expect("style parent"))
            .expect("expected in fixture")
    );
    result["difference"] = json!(difference);
    result["different_pixels"] =
        json!((difference * f64::from(width) * f64::from(height)).round() as u64);
    if difference > metadata.allowed {
        failure.get_or_insert("pixels");
    }
    result["status"] = json!(if failure.is_some() { "fail" } else { "pass" });
    if let Some(failure) = failure {
        result["failure_reason"] = json!(failure);
        let dir = output.join("failures").join(id);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        actual.write(&dir.join("actual.png"))?;
        Image {
            width,
            height,
            pixels: diff,
        }
        .write(&dir.join("diff.png"))?;
        result["actual"] = json!(format!("failures/{id}/actual.png"));
        result["diff"] = json!(format!("failures/{id}/diff.png"));
    }
    result["seconds"] = json!(started.elapsed().as_secs_f64());
    Ok(result)
}

/// Renders the test natively and reads it back; `None` with a skip reason
/// when its canvas is larger than the device's textures.
fn render_native(
    runtime: &Runtime,
    map: &mut Map,
    metadata: &Metadata,
    assets: &Path,
    reasons: &mut BTreeSet<String>,
    diagnostics: &mut Vec<String>,
) -> Option<(Image, Value)> {
    let (width, height) = metadata.physical_size();
    if width.max(height) > runtime.stats().max_texture_size {
        reasons.insert("device:canvas-size".into());
        return None;
    }
    let mut target = Target::texture(runtime, width, height, metadata.pixel_ratio);
    let mut frames = 0;
    let mut tiles_drawn = 0;
    let mut failure = None;
    let render_started = Instant::now();
    // Tests with fades follow GL JS's harness: they start once the map has
    // loaded, fades or not, and wait in wall time; the rest settle fully.
    let fades = metadata.fade_duration > 0.0;
    let loaded = if fades { Next::Loaded } else { Next::Idle };
    let mut render = |map: &mut Map, diagnostics: &mut Vec<String>| match map
        .render(&mut target, frame_time(render_started))
    {
        Ok(report) => report,
        Err(e) => {
            diagnostics.push(format!("render: {e}"));
            None
        }
    };
    // Renders until `until`, or for a wait's wall time; `None` when the run
    // stops (a skip reason, a failed frame or the timeout).
    let mut settle = |map: &mut Map,
                      until: Next,
                      reasons: &mut BTreeSet<String>,
                      diagnostics: &mut Vec<String>,
                      frames: &mut u32| {
        let started = Instant::now();
        loop {
            let report = render(map, diagnostics)?;
            *frames += 1;
            record_warnings(map, reasons, diagnostics);
            if !reasons.is_empty() {
                return None;
            }
            let done = match until {
                Next::Continue => true,
                Next::Loaded => report.loaded,
                Next::Idle => report.idle,
                Next::Wait(ms) => started.elapsed() >= Duration::from_secs_f64(ms / 1000.0),
            };
            if done {
                return Some(report);
            }
            if render_started.elapsed() >= Duration::from_millis(metadata.timeout_ms) {
                failure = Some("timeout");
                return None;
            }
            std::thread::park_timeout(Duration::from_millis(1));
        }
    };
    let mut settled = settle(map, loaded, reasons, diagnostics, &mut frames).is_some();
    for op in &metadata.operations {
        if !settled {
            break;
        }
        match operations::apply(map, op, assets) {
            Ok(Next::Continue) => {}
            Ok(Next::Loaded) => {
                settled = settle(map, loaded, reasons, diagnostics, &mut frames).is_some();
            }
            Ok(next) => settled = settle(map, next, reasons, diagnostics, &mut frames).is_some(),
            Err(e) => diagnostics.push(format!("operation {op}: {e}")),
        }
    }
    // GL JS reads a fading test's pixels after its last operation, and a
    // transition's where its waits left it (its harness's time moves only
    // with them); the rest after one more settle.
    let last = if fades || map.transitioning() {
        Next::Continue
    } else {
        Next::Idle
    };
    if settled
        && settle(map, last, reasons, diagnostics, &mut frames).is_some()
        // Draw the state installed by the last frame's messages.
        && let Some(report) = settle(map, Next::Continue, reasons, diagnostics, &mut frames)
    {
        tiles_drawn = report.tiles_drawn;
    }
    let pixels = match target.read_pixels() {
        Ok(pixels) => pixels,
        Err(e) => {
            diagnostics.push(format!("read back: {e}"));
            vec![0; width as usize * height as usize * 4]
        }
    };
    let mut image = Image {
        width,
        height,
        pixels,
    };
    image.unpremultiply();
    let report = json!({"frames":frames,"tiles_drawn":tiles_drawn,"timed_out":failure.is_some()});
    Some((image, report))
}

/// Milliseconds since `started`: frames' time, which nothing in a test
/// animates by.
fn frame_time(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn record_warnings(map: &mut Map, reasons: &mut BTreeSet<String>, diagnostics: &mut Vec<String>) {
    while let Some(event) = map.poll_event() {
        if let Event::Map(MapEvent::Warning { message } | MapEvent::Error { message }) = event {
            note_warning(message, reasons, diagnostics);
        }
    }
}

/// A runtime warning's skip reason, if any, and the diagnostic it adds.
fn note_warning(warning: String, reasons: &mut BTreeSet<String>, diagnostics: &mut Vec<String>) {
    if let Some(reason) = scope::runtime_reason(&warning) {
        reasons.insert(reason.into());
    }
    // Edits re-parse layers: their parser warnings skip as the initial
    // style's do.
    if let Some(reason) = scope::edit_reason(&warning) {
        reasons.insert(reason);
    }
    // Parser warnings were classified before constructing the map. Keep all
    // original messages for diagnosis, including load errors (which fail),
    // but those GL JS gives too.
    if !scope::mirrors_gl_js(&warning) {
        diagnostics.push(warning);
    }
}

fn finish_skip(result: &mut Value, reasons: &BTreeSet<String>, started: Instant) {
    result["status"] = json!("skip");
    result["skip_reasons"] = json!(reasons);
    result["seconds"] = json!(started.elapsed().as_secs_f64());
}

#[derive(Default)]
struct Counts {
    pass: usize,
    fail: usize,
    skip: usize,
}

impl Counts {
    fn add(&mut self, status: &str) {
        match status {
            "pass" => self.pass += 1,
            "fail" => self.fail += 1,
            "skip" => self.skip += 1,
            _ => unreachable!(),
        }
    }
    fn rate(&self) -> Option<f64> {
        let scope = self.pass + self.fail;
        (scope != 0).then(|| self.pass as f64 / scope as f64)
    }
    fn json(&self) -> Value {
        json!({"pass":self.pass,"fail":self.fail,"skip":self.skip,"pass_rate":self.rate()})
    }
    fn print(&self, name: &str) {
        let rate = self
            .rate()
            .map_or_else(|| "n/a".into(), |r| format!("{:.2}%", r * 100.0));
        println!(
            "{name:<15} {:4}  {:4}  {:4}  {rate:>8}",
            self.pass, self.fail, self.skip
        );
    }
}

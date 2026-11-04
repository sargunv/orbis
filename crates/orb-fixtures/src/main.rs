//! Sequential offscreen render measurements against the read-only GL JS mirror.

#![forbid(unsafe_code)]

mod chrome;
mod fetch;
mod images;
mod metadata;
mod operations;
mod pixelmatch;
mod platforms;
mod query;
mod scope;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use operations::Next;
use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureUsages};
use orb_hal_wgpu::WgpuGpu;
use orb_jobs::ThreadPool;
use orb_runtime::{Map, PoolJobs, Services, TargetFormat, Viewport};
use serde_json::{Value, json};

use fetch::FixtureFetch;
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
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--suite" => suite = args.next().ok_or("--suite needs render|query")?,
            "--filter" => filter = args.next().ok_or("--filter needs a substring")?,
            "--platform" => {
                platform = args
                    .next()
                    .ok_or("--platform needs native|chrome|lavapipe|all")?
            }
            _ => {
                return Err(format!(
                    "unknown argument {arg}; usage: orb-fixtures [--suite render|query] [--filter <substring>] [--platform native|chrome|lavapipe|all]"
                ));
            }
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
        "render" => platforms::run(&root, &filter, &platform),
        "query" => {
            if !["native", "chrome", "all"].contains(&platform.as_str()) {
                return Err("the query suite runs natively and in Chrome".into());
            }
            for name in ["native", "chrome"] {
                if platform == name || platform == "all" {
                    query::run_suite(&root, &filter, name)?;
                }
            }
            Ok(())
        }
        _ => Err(format!("unknown suite {suite}")),
    }
}

enum Backend {
    Native(Box<WgpuGpu>),
    Chrome(chrome::Chrome),
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
    let mut backend = if name == "chrome" {
        Backend::Chrome(chrome::Chrome::new(root)?)
    } else {
        Backend::Native(Box::new(pollster::block_on(WgpuGpu::offscreen())?))
    };
    let mut paths = Vec::new();
    discover(&fixtures, &mut paths)?;
    paths.sort();
    paths.retain(|path| test_id(path, &fixtures).contains(filter));
    let current = std::thread::current();
    let fetch = Arc::new(FixtureFetch::new(root.join("fixtures/gl-js/assets")));
    let executor = Arc::new(ThreadPool::new("fixtures", ThreadPool::default_size()));
    let wake = Arc::new(move || current.unpark());
    let mut results = Vec::new();
    let mut prefixes: BTreeMap<String, Counts> = BTreeMap::new();
    let mut skip_counts: BTreeMap<String, usize> = BTreeMap::new();
    for path in paths {
        let id = test_id(&path, &fixtures);
        let jobs = PoolJobs::new(
            fetch.clone(),
            Arc::new(orb_host_native::NativeImageDecode),
            executor.clone(),
        );
        jobs.set_local_glyphs(orb_host_native::SystemFonts::shared());
        jobs.set_font_files(Arc::new(orb_host_native::FontFileFaces::new()));
        let services = Services {
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(orb_host_native::NativeImageDecode),
            images: Default::default(),
            fetch: fetch.clone(),
            jobs,
            wake: wake.clone(),
        };
        let result = run_test(&path, &id, &output, &mut backend, services)?;
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
    // Decision 0009's byte-identity check: one digest of text layout from
    // font bytes, natively and in wasm.
    let text_matrix = match &backend {
        Backend::Chrome(c) => c.text_matrix.clone(),
        _ => json!(orb_text::matrix::digest().to_string()),
    };
    let summary = json!({
        "filter": filter,
        "platform": {"name":name,"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"browser":browser},
        "comparison": {"metric":"pixelmatch-8.0.0-oklab-hyab", "includeAA":false, "checkerboard":true, "default_threshold":0.1285, "default_allowed":0.00025},
        "total_seconds": started.elapsed().as_secs_f64(),
        "text_matrix": text_matrix,
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

/// The layer family a fixture tests, by its top-level directory. Longer
/// prefixes first, so `fill-extrusion-*` isn't counted as `fill-*`.
fn prefix(id: &str) -> &str {
    const FAMILIES: &[&str] = &[
        "background-",
        "circle-",
        "color-relief",
        "feature-state",
        "fill-extrusion-",
        "fill-",
        "filter",
        "global-state",
        "heatmap-",
        "hillshade-",
        "icon-",
        "line-",
        "raster-",
        "remove-feature-state",
        "runtime-styling",
        "sprites",
        "sky",
        "symbol-",
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

fn run_test(
    path: &Path,
    id: &str,
    output: &Path,
    backend: &mut Backend,
    services: Services,
) -> Result<Value, String> {
    let started = Instant::now();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let metadata = Metadata::read(&value);
    let mut reasons = metadata.skips.clone();
    let mut diagnostics = Vec::new();
    let mut map = Map::new(services, None);
    map.set_antialias_3d(false);
    // GL JS's render tests run with `fadeDuration: 0` unless they say.
    map.set_fade_duration(metadata.fade_duration);
    map.set_cross_source_collisions(metadata.cross_source_collisions);
    map.set_local_ideograph_font_family(metadata.local_ideographs.as_deref());
    match orb_v8::parse(&text) {
        Ok(style) => {
            reasons.extend(scope::reasons(&style));
            // set_style_json also exposes plan diagnostics absent from the parser.
            map.set_style_json(&text)
                .map_err(|e| format!("{id}: {e}"))?;
            record_warnings(&mut map, &mut reasons, &mut diagnostics);
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
    let (mut actual, rendered) = match backend {
        Backend::Native(device) => {
            let max_size = device.caps().limits.max_texture_dimension_2d;
            if width > max_size || height > max_size {
                reasons.insert("device:canvas-size".into());
                finish_skip(&mut result, &reasons, started);
                return Ok(result);
            }
            render_native(device, &mut map, &metadata, &mut reasons, &mut diagnostics)
        }
        Backend::Chrome(chrome) => {
            if width > chrome.max_size || height > chrome.max_size {
                reasons.insert("device:canvas-size".into());
                finish_skip(&mut result, &reasons, started);
                return Ok(result);
            }
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
                if let Some(reason) = scope::runtime_reason(warning) {
                    reasons.insert(reason.into());
                }
                if let Some(reason) = scope::edit_reason(warning) {
                    reasons.insert(reason);
                }
                if !scope::mirrors_gl_js(warning) {
                    diagnostics.push(warning.to_string());
                }
            }
            (image, rendered)
        }
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
    actual.unpremultiply();
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

fn render_native(
    device: &WgpuGpu,
    map: &mut Map,
    metadata: &Metadata,
    reasons: &mut BTreeSet<String>,
    diagnostics: &mut Vec<String>,
) -> (Image, Value) {
    let (width, height) = metadata.physical_size();
    let mut gpu = WgpuGpu::new(device.device().clone(), device.queue().clone());
    let format = TargetFormat {
        color: TextureFormat::Rgba8Unorm,
        sample_count: 1,
    };
    let target = gpu.create_texture(&TextureDesc {
        label: "render fixture",
        width,
        height,
        array_layers: 1,
        format: format.color,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        sample_count: 1,
        mip_levels: 1,
    });
    let viewport = Viewport {
        width: f64::from(metadata.width),
        height: f64::from(metadata.height),
    };
    let mut frames = 0;
    let mut tiles_drawn = 0;
    let mut failure = None;
    let render_started = Instant::now();
    // Tests with fades follow GL JS's harness: they start once the map has
    // loaded, fades or not, and wait in wall time; the rest settle fully.
    let fades = metadata.fade_duration > 0.0;
    let loaded = if fades { Next::Loaded } else { Next::Idle };
    // Renders until `until`, or for a wait's wall time; false when the run
    // stops (a skip reason or the timeout).
    let mut settle = |map: &mut Map,
                      until: Next,
                      reasons: &mut BTreeSet<String>,
                      diagnostics: &mut Vec<String>,
                      frames: &mut u32| {
        let started = Instant::now();
        loop {
            // GL JS's transform keeps to the world as the harness moves it.
            map.constrain_camera(viewport);
            let report = map.render(
                &mut gpu,
                target,
                format,
                viewport,
                metadata.pixel_ratio as f32,
            );
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
        match operations::apply(map, op) {
            Ok(Next::Continue) => {}
            Ok(Next::Loaded) => {
                settled = settle(map, loaded, reasons, diagnostics, &mut frames).is_some();
            }
            Ok(next) => settled = settle(map, next, reasons, diagnostics, &mut frames).is_some(),
            Err(e) => diagnostics.push(format!("operation {op}: {e}")),
        }
    }
    // GL JS reads a fading test's pixels after its last operation; the
    // rest after one more settle.
    let last = if fades { Next::Continue } else { Next::Idle };
    if settled && settle(map, last, reasons, diagnostics, &mut frames).is_some() {
        // Draw the state installed by the last frame's messages.
        let final_report = map.render(
            &mut gpu,
            target,
            format,
            viewport,
            metadata.pixel_ratio as f32,
        );
        frames += 1;
        record_warnings(map, reasons, diagnostics);
        tiles_drawn = final_report.tiles_drawn;
    }
    let image = Image {
        width,
        height,
        pixels: pollster::block_on(gpu.read_texture(target)),
    };
    (
        image,
        json!({"frames":frames,"tiles_drawn":tiles_drawn,"timed_out":failure.is_some()}),
    )
}

fn record_warnings(map: &mut Map, reasons: &mut BTreeSet<String>, diagnostics: &mut Vec<String>) {
    for warning in map.take_warnings() {
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

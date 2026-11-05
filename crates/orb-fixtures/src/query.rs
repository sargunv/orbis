//! GL JS's query fixtures (`fixtures/gl-js/query`): each style settles as a
//! render test does, then `queryRenderedFeatures` runs at the test's point
//! or box and its features are compared with `expected.json`, as GL JS's
//! query harness compares them: without `layer`, coordinates within 1e-6
//! degrees. A test passes when the same features come back; whether they
//! come in GL JS's order is reported apart.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orb_hal_wgpu::WgpuGpu;
use orb_jobs::ThreadPool;
use orb_runtime::{Map, PoolJobs, Services};
use serde_json::{Value, json};

use crate::fetch::FixtureFetch;
use crate::metadata::Metadata;
use crate::{
    Counts, chrome, discover, finish_skip, record_warnings, render_native, scope, test_id,
    write_json,
};

/// Metadata keys only the query harness reads.
const QUERY_KEYS: &[&str] = &["queryGeometry", "queryOptions", "debug", "collisionDebug"];

/// The query box in logical pixels, `[x1, y1, x2, y2]`, from GL JS's
/// `queryGeometry`: a point or two corners.
fn geometry(test: &Value) -> Option<[f64; 4]> {
    let g = test.get("queryGeometry")?;
    let n = |v: &Value| v.as_f64();
    match g.as_array()?.as_slice() {
        [x, y] if x.is_number() => Some([n(x)?, n(y)?, n(x)?, n(y)?]),
        [a, b] => {
            let (a, b) = (a.as_array()?, b.as_array()?);
            Some([n(&a[0])?, n(&a[1])?, n(&b[0])?, n(&b[1])?])
        }
        _ => None,
    }
}

pub(crate) fn run_suite(root: &Path, filter: &str, name: &str) -> Result<Value, String> {
    let fixtures = root.join("fixtures/gl-js/query");
    if !fixtures.is_dir() {
        return Err(format!(
            "{} is missing; run `cargo xtask fixtures` to fetch the GL JS fixture mirror",
            fixtures.display()
        ));
    }
    let output = root.join("docs/scratch/rt/query").join(name);
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let mut chrome = if name == "chrome" {
        Some(chrome::Chrome::new(root, "webgpu")?)
    } else {
        None
    };
    let device = if chrome.is_none() {
        Some(pollster::block_on(WgpuGpu::offscreen())?)
    } else {
        None
    };
    let mut paths = Vec::new();
    discover(&fixtures, &mut paths)?;
    paths.sort();
    paths.retain(|path| test_id(path, &fixtures).contains(filter));
    let fetch = Arc::new(FixtureFetch::new(root.join("fixtures/gl-js/assets")));
    let executor = Arc::new(ThreadPool::new("fixtures", ThreadPool::default_size()));
    let current = std::thread::current();
    let wake = Arc::new(move || current.unpark());
    let mut totals = Counts::default();
    let mut ordered = 0;
    let mut prefixes: BTreeMap<String, Counts> = BTreeMap::new();
    let mut results = Vec::new();
    for path in paths {
        let id = test_id(&path, &fixtures);
        let test_started = Instant::now();
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let style: Value = serde_json::from_str(&text).map_err(|e| format!("{id}: {e}"))?;
        let test = style
            .pointer("/metadata/test")
            .cloned()
            .unwrap_or(json!({}));
        let mut stripped = style.clone();
        if let Some(t) = stripped
            .pointer_mut("/metadata/test")
            .and_then(Value::as_object_mut)
        {
            for key in QUERY_KEYS {
                t.remove(*key);
            }
        }
        let metadata = Metadata::read(&stripped);
        let mut reasons = metadata.skips.clone();
        let mut result = json!({"id": id});
        let Some(query) = geometry(&test) else {
            reasons.insert("query:geometry".into());
            finish_skip(&mut result, &reasons, test_started);
            prefixes.entry(prefix(&id)).or_default().add("skip");
            totals.add("skip");
            results.push(result);
            continue;
        };
        if test.get("queryOptions").is_some() {
            reasons.insert("query:options".into());
        }
        if let Ok(parsed) = orb_v8::parse(&text) {
            // The query tests' layers carry GL JS's long-gone `interactive`
            // flag, which nothing reads.
            reasons.extend(
                scope::reasons(&parsed)
                    .into_iter()
                    .filter(|r| r != "property:interactive"),
            );
        }
        if !reasons.is_empty() {
            finish_skip(&mut result, &reasons, test_started);
            prefixes.entry(prefix(&id)).or_default().add("skip");
            totals.add("skip");
            results.push(result);
            continue;
        }
        let mut diagnostics = Vec::new();
        let actual = match (&mut chrome, &device) {
            (Some(chrome), _) => chrome.query(&text, &metadata, query)?,
            (None, Some(device)) => {
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
                let mut map = Map::new(services, None);
                map.set_antialias_3d(false);
                map.set_fade_duration(metadata.fade_duration);
                map.set_cross_source_collisions(metadata.cross_source_collisions);
                map.set_style_json(&text)
                    .map_err(|e| format!("{id}: {e}"))?;
                record_warnings(&mut map, &mut reasons, &mut diagnostics);
                let mut gpu = WgpuGpu::new(device.device().clone(), device.queue().clone());
                render_native(
                    &mut gpu,
                    &mut map,
                    &metadata,
                    &mut reasons,
                    &mut diagnostics,
                );
                query_native(&map, query)?
            }
            _ => unreachable!("one backend"),
        };
        let expected: Value = serde_json::from_slice(
            &std::fs::read(path.with_file_name("expected.json"))
                .map_err(|e| format!("{id}: {e}"))?,
        )
        .map_err(|e| format!("{id}: {e}"))?;
        let mut actual = actual;
        for feature in actual.as_array_mut().into_iter().flatten() {
            if let Some(f) = feature.as_object_mut() {
                f.remove("layer");
            }
        }
        let (same, in_order) = compare(&actual, &expected);
        let status = if same { "pass" } else { "fail" };
        if same && in_order {
            ordered += 1;
        }
        result["status"] = json!(status);
        result["in_order"] = json!(in_order);
        result["actual"] = json!(actual.as_array().map_or(0, Vec::len));
        result["expected"] = json!(expected.as_array().map_or(0, Vec::len));
        result["diagnostics"] = json!(diagnostics);
        result["seconds"] = json!(test_started.elapsed().as_secs_f64());
        if !same {
            eprintln!(
                "FAIL query {name} {id}: {} features, expected {}",
                result["actual"], result["expected"]
            );
            let dir = output.join("failures").join(&id);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            write_json(&dir.join("actual.json"), &actual)?;
        }
        prefixes.entry(prefix(&id)).or_default().add(status);
        totals.add(status);
        results.push(result);
    }
    let summary = json!({
        "platform": name,
        "total_seconds": started.elapsed().as_secs_f64(),
        "totals": totals.json(),
        "in_order": ordered,
        "prefixes": prefixes.iter().map(|(p, c)| (p.clone(), c.json())).collect::<BTreeMap<_, _>>(),
        "tests": results,
    });
    write_json(&output.join("summary.json"), &summary)?;
    println!("\nquery {name}  prefix          pass  fail  skip  in-scope pass rate");
    for (prefix, counts) in &prefixes {
        print!("{:<12} ", "");
        counts.print(prefix);
    }
    print!("{:<12} ", "");
    totals.print("TOTAL");
    println!("  {ordered} of {} passes in GL JS's order", totals.pass);
    Ok(summary)
}

fn prefix(id: &str) -> String {
    id.split('/').next().unwrap_or(id).to_string()
}

/// Runs the query on a settled map and waits for its workers.
fn query_native(map: &Map, [x1, y1, x2, y2]: [f64; 4]) -> Result<Value, String> {
    let (sender, receiver) = std::sync::mpsc::channel();
    map.query_rendered(
        glam::DVec2::new(x1, y1),
        glam::DVec2::new(x2, y2),
        None,
        Box::new(move |result| {
            let _ = sender.send(result);
        }),
    );
    let json = receiver
        .recv_timeout(Duration::from_secs(30))
        .map_err(|_| "query timed out".to_string())??;
    serde_json::from_str(&json).map_err(|e| format!("query JSON: {e}"))
}

/// Whether two feature lists hold the same features, and in the same
/// order; numbers within 1e-6.
fn compare(actual: &Value, expected: &Value) -> (bool, bool) {
    let (Some(a), Some(e)) = (actual.as_array(), expected.as_array()) else {
        return (false, false);
    };
    if a.len() != e.len() {
        return (false, false);
    }
    let in_order = a.iter().zip(e).all(|(a, e)| close(a, e));
    if in_order {
        return (true, true);
    }
    let mut unmatched: Vec<&Value> = e.iter().collect();
    for feature in a {
        match unmatched.iter().position(|e| close(feature, e)) {
            Some(i) => {
                unmatched.swap_remove(i);
            }
            None => return (false, false),
        }
    }
    (true, false)
}

fn close(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (
                x.as_f64().unwrap_or(f64::NAN),
                y.as_f64().unwrap_or(f64::NAN),
            );
            (x - y).abs() <= 1e-6 * x.abs().max(1.0)
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| close(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| close(v, w)))
        }
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_compare_without_order_and_within_tolerance() {
        let a = json!([{"id": 1, "geometry": {"coordinates": [1.0000000001, 2]}}, {"id": 2}]);
        let b = json!([{"id": 2}, {"id": 1, "geometry": {"coordinates": [1, 2]}}]);
        assert_eq!(compare(&a, &b), (true, false));
        assert_eq!(compare(&b, &b), (true, true));
        assert_eq!(compare(&a, &json!([{"id": 2}])), (false, false));
        assert_eq!(
            geometry(&json!({"queryGeometry": [3, 4]})),
            Some([3.0, 4.0, 3.0, 4.0])
        );
        assert_eq!(
            geometry(&json!({"queryGeometry": [[1, 2], [5, 6]]})),
            Some([1.0, 2.0, 5.0, 6.0])
        );
    }
}

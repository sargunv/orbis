//! GL JS's query fixtures (`fixtures/gl-js/query`): each style settles as a
//! render test does, then `queryRenderedFeatures` runs at the test's point
//! or box and its features are compared with `expected.json`, as GL JS's
//! query harness compares them: without `layer`, coordinates within 1e-6
//! degrees. A test passes when the same features come back; whether they
//! come in GL JS's order is reported apart, and so are failures that
//! differ only in geometry: which tile's copy of a feature answered, or
//! how that tile clipped it.

use std::path::Path;
use std::time::{Duration, Instant};

use orb_api::{Event, Map};
use serde_json::{Value, json};

use crate::metadata::Metadata;
use crate::tally::{self, Tally};
use crate::{
    Backend, assets, discover, finish_skip, record_warnings, render_native, scope, test_id,
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
    let mut tally = Tally::start(root.join("docs/scratch/rt/query").join(name))?;
    let output = tally.output.clone();
    let mut backend = tally::backend(root, name, &assets(root), crate::fetch::Assets::GlJs)?;
    let mut paths = Vec::new();
    discover(&fixtures, &mut paths)?;
    paths.sort();
    paths.retain(|path| crate::matches(&test_id(path, &fixtures), filter));
    let assets = assets(root);
    let (mut ordered, mut shapes) = (0, 0);
    for path in paths {
        let id = test_id(&path, &fixtures);
        let result = run_test(&path, &id, &output, name, &assets, &mut backend)?;
        if result["status"] == "pass" && result["in_order"] == true {
            ordered += 1;
        }
        if result["geometry_only"] == true {
            shapes += 1;
        }
        tally.add(&prefix(&id), result);
    }
    tally.print(&format!("query {name}"));
    println!(
        "  {ordered} of {} passes in GL JS's order",
        tally.totals.pass
    );
    println!(
        "  {shapes} of {} failures differ only in geometry",
        tally.totals.fail
    );
    tally.finish(json!({
        "platform": name,
        "in_order": ordered,
        "geometry_only": shapes,
    }))
}

/// One query test: skipped, or its features compared with `expected.json`.
fn run_test(
    path: &Path,
    id: &str,
    output: &Path,
    name: &str,
    assets: &Path,
    backend: &mut Backend,
) -> Result<Value, String> {
    let started = Instant::now();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
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
    let query = geometry(&test);
    if query.is_none() {
        reasons.insert("query:geometry".into());
    }
    if test.get("queryOptions").is_some() {
        reasons.insert("query:options".into());
    }
    if let Ok(parsed) = orb_style::v8::parse(&text) {
        // The query tests' layers carry GL JS's long-gone `interactive`
        // flag, which nothing reads.
        reasons.extend(
            scope::reasons(&parsed)
                .into_iter()
                .filter(|r| r != "property:interactive"),
        );
    }
    let mut diagnostics = crate::dem::lint(&style, metadata.width, metadata.height, assets);
    if !diagnostics.is_empty() {
        reasons.insert(crate::dem::IMPLAUSIBLE.into());
        result["diagnostics"] = json!(diagnostics);
    }
    let actual = match (query.filter(|_| reasons.is_empty()), backend) {
        (None, _) => None,
        (Some(query), Backend::Chrome(chrome)) => Some(chrome.query(&text, &metadata, query)?),
        (Some(query), Backend::Native(runtime)) => {
            let mut map = crate::map(runtime, &metadata);
            map.set_style_json(&text)
                .map_err(|e| format!("{id}: {e}"))?;
            record_warnings(&mut map, &mut reasons, &mut diagnostics);
            render_native(
                runtime,
                &mut map,
                &metadata,
                assets,
                &mut reasons,
                &mut diagnostics,
            )
            .map(|_| query_native(&mut map, query))
            .transpose()?
        }
    };
    let Some(mut actual) = actual else {
        finish_skip(&mut result, &reasons, started);
        return Ok(result);
    };
    let expected: Value = serde_json::from_slice(
        &std::fs::read(path.with_file_name("expected.json")).map_err(|e| format!("{id}: {e}"))?,
    )
    .map_err(|e| format!("{id}: {e}"))?;
    for feature in actual.as_array_mut().into_iter().flatten() {
        if let Some(f) = feature.as_object_mut() {
            f.remove("layer");
        }
    }
    let (same, in_order) = compare(&actual, &expected);
    result["status"] = json!(if same { "pass" } else { "fail" });
    result["in_order"] = json!(in_order);
    result["actual"] = json!(actual.as_array().map_or(0, Vec::len));
    result["expected"] = json!(expected.as_array().map_or(0, Vec::len));
    result["diagnostics"] = json!(diagnostics);
    result["seconds"] = json!(started.elapsed().as_secs_f64());
    if !same {
        let shapes = compare(&without_geometry(&actual), &without_geometry(&expected)).0;
        result["geometry_only"] = json!(shapes);
        eprintln!(
            "FAIL query {name} {id}: {} features, expected {}{}",
            result["actual"],
            result["expected"],
            if shapes { " (geometry only)" } else { "" }
        );
        let dir = output.join("failures").join(id);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        write_json(&dir.join("actual.json"), &actual)?;
    }
    Ok(result)
}

fn prefix(id: &str) -> String {
    id.split('/').next().unwrap_or(id).to_string()
}

/// Runs the query on a settled map and waits for its workers.
pub(crate) fn query_native(map: &mut Map, [x1, y1, x2, y2]: [f64; 4]) -> Result<Value, String> {
    let id = map.query_rendered(glam::DVec2::new(x1, y1), glam::DVec2::new(x2, y2));
    let deadline = Instant::now() + Duration::from_secs(30);
    let json = loop {
        if let Some(Event::Query { result, .. }) = std::iter::from_fn(|| map.poll_event())
            .find(|e| matches!(e, Event::Query { id: i, .. } if *i == id))
        {
            break result?;
        }
        if Instant::now() >= deadline {
            return Err("query timed out".into());
        }
        std::thread::park_timeout(Duration::from_millis(1));
    };
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

/// The features without their geometry.
fn without_geometry(features: &Value) -> Value {
    let mut features = features.clone();
    for feature in features.as_array_mut().into_iter().flatten() {
        if let Some(f) = feature.as_object_mut() {
            f.remove("geometry");
        }
    }
    features
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

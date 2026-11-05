//! GL JS render.test.ts metadata defaults; unsupported options are explicit skips.

use std::collections::BTreeSet;

use serde_json::{Value, json};

#[derive(Debug)]
pub(crate) struct Metadata {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixel_ratio: f64,
    pub(crate) allowed: f64,
    pub(crate) threshold: f64,
    pub(crate) timeout_ms: u64,
    /// GL JS `fadeDuration`, milliseconds; the harness's default is 0.
    pub(crate) fade_duration: f64,
    /// GL JS `crossSourceCollisions`; on by default.
    pub(crate) cross_source_collisions: bool,
    /// GL JS `localIdeographFontFamily`; the harness passes `false` unless
    /// the test sets it.
    pub(crate) local_ideographs: Option<String>,
    /// GL JS `maxPitch`: the harness leaves it to the map's default, 60
    /// degrees, which clamps steeper style pitches.
    pub(crate) max_pitch: f64,
    pub(crate) value: Value,
    /// GL JS's `operations`, applied after the map first goes idle.
    pub(crate) operations: Vec<Value>,
    pub(crate) skips: BTreeSet<String>,
}

/// Why the runners can't apply an operation, if they can't. Camera calls,
/// `wait` and `idle` are the runners'; map calls must parse as a style
/// transaction.
fn operation_skip(op: &Value) -> Option<String> {
    let name = op.get(0).and_then(Value::as_str)?;
    let number = |i: usize| op.get(i).is_some_and(Value::is_number);
    match name {
        "wait" if op.as_array().is_some_and(|a| a.len() == 1) || number(1) => None,
        // Waiting for a named event.
        "wait" => Some("operation:wait-for".into()),
        "idle" => None,
        "setZoom" | "setBearing" | "setPitch" | "setRoll" if number(1) => None,
        "addImage" if op.get(2).is_some_and(Value::is_string) => None,
        "removeImage" if op.get(1).is_some_and(Value::is_string) => None,
        "setCenter"
            if op
                .get(1)
                .and_then(Value::as_array)
                .is_some_and(|c| c.len() == 2) =>
        {
            None
        }
        _ => orb_runtime::Transaction::from_commands(&format!("[{op}]"))
            .err()
            .map(|_| format!("operation:{name}")),
    }
}

impl Metadata {
    pub(crate) fn read(style: &Value) -> Self {
        let mut value = json!({
            "width": 512, "height": 512, "pixelRatio": 1,
            "allowed": 0.00025, "threshold": 0.1285, "timeout": 60000
        });
        let mut skips = BTreeSet::new();
        let mut operations = Vec::new();
        if let Some(test) = style.pointer("/metadata/test") {
            if let Some(test) = test.as_object() {
                for (key, v) in test {
                    value[key] = v.clone();
                    let supported = match key.as_str() {
                        "width" | "height" | "pixelRatio" | "allowed" | "threshold" | "timeout"
                        | "description" => true,
                        "operations" => match v.as_array() {
                            Some(ops) => {
                                skips.extend(ops.iter().filter_map(operation_skip));
                                operations = ops.clone();
                                true
                            }
                            None => false,
                        },
                        "maxPitch" => v.as_f64().is_some_and(|max| (0.0..=180.0).contains(&max)),
                        // Inert values of harness options can be honoured without new APIs.
                        "debug" | "showOverdrawInspector" | "showPadding" | "collisionDebug" => {
                            v == false
                        }
                        "fadeDuration" => v.as_f64().is_some_and(|ms| ms >= 0.0 && ms.is_finite()),
                        "localIdeographFontFamily" => v.is_string() || v == false || v.is_null(),
                        "crossSourceCollisions" => v.is_boolean(),
                        "continuesRepaint" => v.is_boolean(),
                        _ => false,
                    };
                    if !supported {
                        skips.insert(if key == "operations" {
                            "operations".into()
                        } else {
                            format!("metadata:{key}")
                        });
                    }
                }
            } else {
                skips.insert("metadata:test".into());
            }
        }
        let mut number = |key: &str, default: f64, min: f64, max: f64, integer: bool| match value
            [key]
            .as_f64()
        {
            Some(n) if n.is_finite() && n >= min && n <= max && (!integer || n.fract() == 0.0) => n,
            _ => {
                skips.insert(format!("metadata:{key}"));
                default
            }
        };
        let width = number("width", 512.0, 1.0, 8192.0, true) as u32;
        let height = number("height", 512.0, 1.0, 8192.0, true) as u32;
        let pixel_ratio = number("pixelRatio", 1.0, f64::MIN_POSITIVE, 8192.0, false);
        let allowed = number("allowed", 0.00025, 0.0, 1.0, false);
        let threshold = number("threshold", 0.1285, 0.0, 1.0, false);
        let timeout_ms = number("timeout", 60000.0, 1.0, u32::MAX as f64, true) as u64;
        let fade_duration = value["fadeDuration"].as_f64().unwrap_or(0.0);
        let max_pitch = value["maxPitch"].as_f64().unwrap_or(60.0);
        let cross_source_collisions = value["crossSourceCollisions"].as_bool().unwrap_or(true);
        let local_ideographs = value["localIdeographFontFamily"]
            .as_str()
            .filter(|f| !f.is_empty())
            .map(str::to_string);
        // GL JS floors its physical canvas dimensions and caps them at 8192.
        for logical in [width, height] {
            let physical = (f64::from(logical) * pixel_ratio).floor();
            if !(1.0..=8192.0).contains(&physical) {
                skips.insert("metadata:canvas-size".into());
            }
        }
        Self {
            width,
            height,
            pixel_ratio,
            allowed,
            threshold,
            timeout_ms,
            fade_duration,
            cross_source_collisions,
            local_ideographs,
            max_pitch,
            value,
            operations,
            skips,
        }
    }

    pub(crate) fn physical_size(&self) -> (u32, u32) {
        (
            (f64::from(self.width) * self.pixel_ratio).floor() as u32,
            (f64::from(self.height) * self.pixel_ratio).floor() as u32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_custom_threshold_and_physical_pixels() {
        let defaults = Metadata::read(&json!({}));
        assert_eq!(defaults.physical_size(), (512, 512));
        assert_eq!(defaults.allowed, 0.00025);
        assert_eq!(defaults.threshold, 0.1285);
        assert_eq!(defaults.max_pitch, 60.0);
        let m = Metadata::read(
            &json!({"metadata":{"test":{"width":31,"height":17,"pixelRatio":1.5,"threshold":0.01}}}),
        );
        assert_eq!(m.physical_size(), (46, 25));
        assert_eq!(m.threshold, 0.01);
        assert!(m.skips.is_empty());
        let waiting = Metadata::read(&json!({"metadata":{"test":{"operations":[["wait"]]}}}));
        assert!(waiting.skips.is_empty());
    }

    #[test]
    fn options_we_cannot_honour_and_invalid_numbers_skip() {
        let m = Metadata::read(&json!({"metadata":{"test":{
            "operations":[["wait"], ["setZoom", 3], ["setPaintProperty", "l", "fill-color", "red"],
                ["addImage", "x", 3], ["wait", "render"], ["wait", 100]],
            "reportWidth":12, "maxPitch":-1,
            "width":0, "pixelRatio":-1, "debug":true, "description":"informational"
        }}}));
        assert_eq!(m.operations.len(), 6);
        for reason in [
            "operation:addImage",
            "operation:wait-for",
            "metadata:reportWidth",
            "metadata:maxPitch",
            "metadata:width",
            "metadata:pixelRatio",
            "metadata:debug",
        ] {
            assert!(m.skips.contains(reason), "{reason}");
        }
        assert_eq!(m.skips.len(), 7);
    }
}

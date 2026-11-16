//! Render targets (the `dev.sargunv.orbis:targets` preview): the root list
//! of targets, and an image-process layer's operations.

use crate::v8::Cx;
use crate::*;
use orb_expr::json::{Json, JsonValue};

/// The root key that declares the targets.
pub(crate) const TARGETS: &str = "dev.sargunv.orbis:targets";
/// The layer key naming the target a layer draws into.
pub(crate) const DESTINATION: &str = "dev.sargunv.orbis:destination-target";
/// The layer key naming the target an image-process layer reads, or a fill
/// draws through.
pub(crate) const SOURCE: &str = "dev.sargunv.orbis:source-target";
/// The layer key holding an image-process layer's operations.
pub(crate) const OPERATIONS: &str = "dev.sargunv.orbis:operations";

/// The root `dev.sargunv.orbis:targets`: `[{"id": …}]`, ids unique.
pub(crate) fn parse_targets(cx: &mut Cx, json: &Json) -> Vec<RenderTarget> {
    let Some(items) = cx.array(json, TARGETS) else {
        return Vec::new();
    };
    let mut targets: Vec<RenderTarget> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let path = format!("{TARGETS}[{i}]");
        let Some(members) = cx.object(item, &path) else {
            continue;
        };
        for m in members {
            if m.key != "id" {
                cx.warn_unknown(&path, m);
            }
        }
        let Some(id) = item.get("id") else {
            cx.error(&path, item.span, "missing required property \"id\"");
            continue;
        };
        let Some(name) = cx.string(id, &format!("{path}.id")) else {
            continue;
        };
        if targets.iter().any(|t| t.id == name) {
            cx.error(
                format!("{path}.id"),
                id.span,
                format!("duplicate target id \"{name}\""),
            );
            continue;
        }
        targets.push(RenderTarget { id: name });
    }
    targets
}

/// An image-process layer's operations at `path`.
pub(crate) fn parse_operations(cx: &mut Cx, json: &Json, path: &str) -> Vec<Operation> {
    let Some(items) = cx.array(json, path) else {
        return Vec::new();
    };
    (items.iter().enumerate())
        .filter_map(|(i, item)| parse_operation(cx, item, &format!("{path}[{i}]")))
        .collect()
}

fn parse_operation(cx: &mut Cx, json: &Json, path: &str) -> Option<Operation> {
    let members = cx.object(json, path)?;
    let Some(ty) = json.get("type") else {
        cx.error(path, json.span, "missing required property \"type\"");
        return None;
    };
    let name = cx.string(ty, &format!("{path}.type"))?;
    let params: &[&str] = match name.as_str() {
        "select" => &["value", "tolerance"],
        "blur" | "high-pass" => &["size"],
        "noise" | "sharpen" => &["size", "amount"],
        "threshold" => &["cutoff"],
        "invert" => &[],
        "convolve" => &["kernel"],
        _ => {
            cx.error(
                format!("{path}.type"),
                ty.span,
                format!(
                    "expected one of [{}], {ty} found",
                    Operation::NAMES.join(", ")
                ),
            );
            return None;
        }
    };
    for m in members {
        if m.key != "type" && !params.contains(&m.key.as_str()) {
            cx.warn_unknown(path, m);
        }
    }
    let number = |cx: &mut Cx, key: &str, (min, max): (f64, f64), default: Option<f64>| {
        let at = format!("{path}.{key}");
        match json.get(key) {
            Some(v) => cx.number_in(v, &at, min, max).map(|n| n as f32),
            None if default.is_some() => default.map(|n| n as f32),
            None => {
                cx.error(
                    path,
                    json.span,
                    format!("missing required property \"{key}\""),
                );
                None
            }
        }
    };
    // A blur's standard deviation: a kernel of three times it a side.
    const SIZE: (f64, f64) = (0.0, 64.0);
    Some(match name.as_str() {
        "select" => {
            let Some(value) = json.get("value") else {
                cx.error(path, json.span, "missing required property \"value\"");
                return None;
            };
            let Some(color) = value.as_str().and_then(Color::parse) else {
                cx.error(
                    format!("{path}.value"),
                    value.span,
                    format!("color expected, {value} found"),
                );
                return None;
            };
            Operation::Select {
                value: color,
                tolerance: number(cx, "tolerance", (0.0, 255.0), Some(16.0))?,
            }
        }
        "blur" => Operation::Blur {
            size: number(cx, "size", SIZE, None)?,
        },
        "high-pass" => Operation::HighPass {
            size: number(cx, "size", SIZE, None)?,
        },
        "noise" => Operation::Noise {
            size: number(cx, "size", (0.0, 1024.0), None)?.max(f32::MIN_POSITIVE),
            amount: number(cx, "amount", (0.0, 1.0), Some(0.25))?,
        },
        "sharpen" => Operation::Sharpen {
            size: number(cx, "size", SIZE, None)?,
            amount: number(cx, "amount", (0.0, 16.0), Some(1.0))?,
        },
        "threshold" => Operation::Threshold {
            cutoff: number(cx, "cutoff", (0.0, 255.0), Some(128.0))?,
        },
        "invert" => Operation::Invert,
        "convolve" => {
            let Some(kernel) = json.get("kernel") else {
                cx.error(path, json.span, "missing required property \"kernel\"");
                return None;
            };
            let weights: Option<Vec<f32>> = match &kernel.value {
                JsonValue::Array(items) => {
                    items.iter().map(|w| w.as_f64().map(|w| w as f32)).collect()
                }
                _ => None,
            };
            match weights {
                Some(w) if [9, 25, 49].contains(&w.len()) => Operation::Convolve { kernel: w },
                _ => {
                    cx.error(
                        format!("{path}.kernel"),
                        kernel.span,
                        "expected an array of 9, 25 or 49 numbers",
                    );
                    return None;
                }
            }
        }
        _ => unreachable!("checked above"),
    })
}

#[cfg(test)]
mod tests {
    use crate::*;

    fn parse(
        previews: Previews,
        root: &str,
        layers: &str,
    ) -> Result<Style, crate::v8::Diagnostics> {
        crate::v8::parse_with(
            &format!(
                r#"{{"version":8,"sources":{{"v":{{"type":"vector","tiles":["x"]}}}},{root}"layers":[{layers}]}}"#
            ),
            previews,
            &[],
        )
    }

    const ON: Previews = Previews::NONE.with(Preview::Targets);
    const TARGETS: &str = r#""dev.sargunv.orbis:targets":[{"id":"paint"},{"id":"land"}],"#;

    #[test]
    fn layers_draw_into_targets_that_image_process_layers_read() {
        let style = parse(
            ON,
            TARGETS,
            r##"{"id":"water","type":"fill","source":"v","source-layer":"water",
                 "dev.sargunv.orbis:destination-target":"paint","paint":{"fill-color":"#00f"}},
               {"id":"mask","type":"dev.sargunv.orbis.image-process",
                 "dev.sargunv.orbis:source-target":"paint","dev.sargunv.orbis:destination-target":"land",
                 "dev.sargunv.orbis:operations":[
                   {"type":"select","value":"rgb(255,255,0)"},{"type":"blur","size":5},
                   {"type":"noise","size":2},{"type":"threshold","cutoff":128},{"type":"invert"},
                   {"type":"sharpen","size":1,"amount":2},{"type":"high-pass","size":3},
                   {"type":"convolve","kernel":[0,-1,0,-1,5,-1,0,-1,0]}]},
               {"id":"land","type":"fill","dev.sargunv.orbis:source-target":"land",
                 "paint":{"fill-color":"orange"}}"##,
        )
        .unwrap();
        assert!(style.warnings.is_empty(), "{:?}", style.warnings);
        let ids: Vec<&str> = style.targets.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, ["paint", "land"]);
        let [water, mask, land] = &style.layers[..] else {
            panic!("three layers");
        };
        assert_eq!(water.destination_target.as_deref(), Some("paint"));
        assert_eq!(mask.ty, LayerType::ImageProcess);
        assert_eq!(mask.source_target.as_deref(), Some("paint"));
        assert_eq!(mask.destination_target.as_deref(), Some("land"));
        assert_eq!(
            mask.operations,
            [
                Operation::Select {
                    value: Color::parse("#ff0").unwrap(),
                    tolerance: 16.0
                },
                Operation::Blur { size: 5.0 },
                Operation::Noise {
                    size: 2.0,
                    amount: 0.25
                },
                Operation::Threshold { cutoff: 128.0 },
                Operation::Invert,
                Operation::Sharpen {
                    size: 1.0,
                    amount: 2.0
                },
                Operation::HighPass { size: 3.0 },
                Operation::Convolve {
                    kernel: vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0]
                },
            ]
        );
        assert_eq!(land.source, None);
        assert_eq!(land.source_target.as_deref(), Some("land"));
    }

    #[test]
    fn the_preview_off_ignores_targets_and_refuses_image_process_layers() {
        let style = parse(
            Previews::NONE,
            TARGETS,
            r#"{"id":"water","type":"fill","source":"v","source-layer":"water",
                "dev.sargunv.orbis:destination-target":"paint"}"#,
        )
        .unwrap();
        assert!(style.targets.is_empty());
        assert_eq!(style.layers[0].destination_target, None);
        let paths: Vec<&str> = style.warnings.iter().map(|w| w.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "dev.sargunv.orbis:targets",
                "layers[0].dev.sargunv.orbis:destination-target"
            ]
        );
        let err = parse(
            Previews::NONE,
            "",
            r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"x"}"#,
        )
        .unwrap_err();
        assert!(err.errors[0].message.contains("preview"), "{err:?}");
    }

    #[test]
    fn invalid_targets_and_operations_are_errors() {
        for (root, layers) in [
            (
                r#""dev.sargunv.orbis:targets":[{"id":"a"},{"id":"a"}],"#,
                "",
            ),
            (r#""dev.sargunv.orbis:targets":[{}],"#, ""),
            ("", r#"{"id":"p","type":"dev.sargunv.orbis.image-process"}"#),
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"smudge"}]}"#,
            ),
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"blur","size":-1}]}"#,
            ),
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"convolve","kernel":[1,2]}]}"#,
            ),
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"select"}]}"#,
            ),
        ] {
            assert!(parse(ON, root, layers).is_err(), "{root} {layers}");
        }
    }

    #[test]
    fn a_fill_drawn_through_a_target_has_no_features() {
        let style = parse(
            ON,
            TARGETS,
            r#"{"id":"land","type":"fill","source":"v","source-layer":"x",
                "dev.sargunv.orbis:source-target":"land","dev.sargunv.orbis:destination-target":"paint",
                "paint":{"fill-color":["get","c"]}},
              {"id":"b","type":"fill-extrusion","source":"v","source-layer":"x",
                "dev.sargunv.orbis:destination-target":"paint"},
              {"id":"l","type":"line","source":"v","source-layer":"x",
                "dev.sargunv.orbis:source-target":"paint"}"#,
        )
        .unwrap();
        let [land, extrusion, line] = &style.layers[..] else {
            panic!("three layers");
        };
        assert_eq!(
            (land.source.as_deref(), land.source_layer.as_deref()),
            (None, None)
        );
        assert_eq!(land.destination_target, None);
        assert_eq!(land.get(FILL_COLOR).constant(), Some(Color::BLACK));
        assert_eq!(extrusion.destination_target, None);
        assert_eq!(line.source_target, None);
        let paths: Vec<&str> = style.warnings.iter().map(|w| w.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "layers[0].dev.sargunv.orbis:destination-target",
                "layers[0].paint.fill-color",
                "layers[0].source",
                "layers[0].source-layer",
                "layers[1].dev.sargunv.orbis:destination-target",
                "layers[2].dev.sargunv.orbis:source-target",
            ]
        );
    }
}

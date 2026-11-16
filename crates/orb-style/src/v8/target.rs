//! Render targets (the `dev.sargunv.orbis:targets` preview): the root list
//! of targets, and an image-process layer's operations.

use crate::v8::{Cx, layer::typed_property_at};
use crate::*;
use orb_expr::PropertyType;
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

/// A parameter's spec: a number, colour or pair, which may follow the zoom.
fn param(
    name: &'static str,
    ty: PropertyType,
    default: &'static str,
    (minimum, maximum): (f64, f64),
) -> Prop {
    Prop {
        name,
        ty,
        default: Some(default),
        interpolated: true,
        minimum: Some(minimum),
        maximum: Some(maximum),
        ..Prop::ROOT
    }
}

const NUMBER: PropertyType = PropertyType::Number;
const COLOR: PropertyType = PropertyType::Color;
const PAIR: PropertyType = PropertyType::Array(orb_expr::Type::Number, Some(2));
/// A blur's or a dilation's size: past 64 pixels it reaches no further.
const SIZE: (f64, f64) = (0.0, 64.0);

fn parse_operation(cx: &mut Cx, json: &Json, path: &str) -> Option<Operation> {
    let members = cx.object(json, path)?;
    let Some(ty) = json.get("type") else {
        cx.error(path, json.span, "missing required property \"type\"");
        return None;
    };
    let name = cx.string(ty, &format!("{path}.type"))?;
    let params: &[&str] = match name.as_str() {
        "select" => &["value", "tolerance"],
        "blur" | "high-pass" | "dilate" | "erode" => &["size"],
        "noise" | "sharpen" => &["size", "amount"],
        "threshold" => &["cutoff"],
        "invert" | "luminance" => &[],
        "convolve" => &["kernel"],
        "offset" => &["translate"],
        "combine" => &["target", "mode"],
        "opacity" | "color" => &["value"],
        "ramp" => &["stops"],
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
    let errors = cx.diagnostics.errors.len();
    // A parameter, required where `required`: a missing one is an error.
    fn read<T: PropertyValue>(
        cx: &mut Cx,
        json: &Json,
        path: &str,
        spec: &Prop,
        required: bool,
    ) -> Property<T> {
        let value = json.get(spec.name);
        if required && value.is_none() {
            let message = format!("missing required property \"{}\"", spec.name);
            cx.error(path, json.span, message);
        }
        typed_property_at(cx, value, &format!("{path}.{}", spec.name), spec)
    }
    let size = |cx: &mut Cx| read(cx, json, path, &param("size", NUMBER, "1", SIZE), true);
    let op = match name.as_str() {
        "select" => Operation::Select {
            value: read(
                cx,
                json,
                path,
                &param("value", COLOR, "\"white\"", (0.0, 0.0)),
                true,
            ),
            tolerance: read(
                cx,
                json,
                path,
                &param("tolerance", NUMBER, "16", (0.0, 255.0)),
                false,
            ),
        },
        "blur" => Operation::Blur { size: size(cx) },
        "high-pass" => Operation::HighPass { size: size(cx) },
        "dilate" => Operation::Dilate { size: size(cx) },
        "erode" => Operation::Erode { size: size(cx) },
        "noise" => Operation::Noise {
            size: read(
                cx,
                json,
                path,
                &param("size", NUMBER, "1", (0.0, 1024.0)),
                true,
            ),
            amount: read(
                cx,
                json,
                path,
                &param("amount", NUMBER, "0.25", (0.0, 1.0)),
                false,
            ),
        },
        "sharpen" => Operation::Sharpen {
            size: size(cx),
            amount: read(
                cx,
                json,
                path,
                &param("amount", NUMBER, "1", (0.0, 16.0)),
                false,
            ),
        },
        "threshold" => Operation::Threshold {
            cutoff: read(
                cx,
                json,
                path,
                &param("cutoff", NUMBER, "128", (0.0, 255.0)),
                false,
            ),
        },
        "invert" => Operation::Invert,
        "luminance" => Operation::Luminance,
        "offset" => Operation::Offset {
            translate: read(
                cx,
                json,
                path,
                &param("translate", PAIR, "[0,0]", (-512.0, 512.0)),
                true,
            ),
        },
        "opacity" => Operation::Opacity {
            value: read(
                cx,
                json,
                path,
                &param("value", NUMBER, "1", (0.0, 1.0)),
                true,
            ),
        },
        "color" => Operation::Color {
            value: read(
                cx,
                json,
                path,
                &param("value", COLOR, "\"white\"", (0.0, 0.0)),
                true,
            ),
        },
        "combine" => {
            let target = match json.get("target") {
                Some(t) => cx.string(t, &format!("{path}.target")),
                None => {
                    cx.error(path, json.span, "missing required property \"target\"");
                    None
                }
            };
            let mode = match json.get("mode") {
                None => Some(Combine::Multiply),
                Some(m) => {
                    let mode = m.as_str().and_then(Combine::from_name);
                    if mode.is_none() {
                        cx.error(
                            format!("{path}.mode"),
                            m.span,
                            format!("expected one of [{}], {m} found", Combine::NAMES.join(", ")),
                        );
                    }
                    mode
                }
            };
            Operation::Combine {
                target: target?,
                mode: mode?,
            }
        }
        "ramp" => Operation::Ramp {
            stops: ramp_stops(cx, json, path)?,
        },
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
    };
    (cx.diagnostics.errors.len() == errors).then_some(op)
}

/// A ramp's `stops`: `[[coverage, colour], …]`, coverages 0–1 in order, up
/// to eight.
fn ramp_stops(cx: &mut Cx, json: &Json, path: &str) -> Option<Vec<(f32, Color)>> {
    let at = format!("{path}.stops");
    let Some(stops) = json.get("stops") else {
        cx.error(path, json.span, "missing required property \"stops\"");
        return None;
    };
    let items = cx.array(stops, &at)?;
    let parsed: Option<Vec<(f32, Color)>> = items
        .iter()
        .map(|stop| match stop.as_array()? {
            [at, color] => Some((at.as_f64()? as f32, Color::parse(color.as_str()?)?)),
            _ => None,
        })
        .collect();
    match parsed {
        Some(stops)
            if (1..=8).contains(&stops.len())
                && stops.windows(2).all(|w| w[0].0 <= w[1].0)
                && stops.iter().all(|s| (0.0..=1.0).contains(&s.0)) =>
        {
            Some(stops)
        }
        _ => {
            cx.error(
                at,
                stops.span,
                "expected 1 to 8 [coverage, colour] stops, coverages 0 to 1 in order",
            );
            None
        }
    }
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
                   {"type":"select","value":"rgb(255,255,0)"},
                   {"type":"blur","size":["interpolate",["linear"],["zoom"],10,2,14,6]},
                   {"type":"noise","size":2},{"type":"threshold","cutoff":128},{"type":"invert"},
                   {"type":"sharpen","size":1,"amount":2},{"type":"high-pass","size":3},
                   {"type":"convolve","kernel":[0,-1,0,-1,5,-1,0,-1,0]},
                   {"type":"offset","translate":[3,-4]},{"type":"dilate","size":2},{"type":"erode","size":1},
                   {"type":"combine","target":"land","mode":"subtract"},{"type":"opacity","value":0.5},
                   {"type":"color","value":"red"},{"type":"luminance"},
                   {"type":"ramp","stops":[[0,"rgba(0,0,0,0)"],[1,"#fff"]]}]},
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
        let ops = &mask.operations;
        assert_eq!(ops.len(), 16);
        let n = |p: &Property<f32>| p.evaluate(10.0);
        let Operation::Select { value, tolerance } = &ops[0] else {
            panic!("{:?}", ops[0]);
        };
        assert_eq!(
            (value.evaluate(0.0), n(tolerance)),
            (Color::parse("#ff0").unwrap(), 16.0)
        );
        // Sizes may follow the zoom.
        let Operation::Blur { size } = &ops[1] else {
            panic!("{:?}", ops[1]);
        };
        assert_eq!((size.evaluate(10.0), size.evaluate(14.0)), (2.0, 6.0));
        assert!(
            matches!(&ops[2], Operation::Noise { size, amount } if n(size) == 2.0 && n(amount) == 0.25)
        );
        assert!(matches!(&ops[3], Operation::Threshold { cutoff } if n(cutoff) == 128.0));
        assert!(matches!(ops[4], Operation::Invert));
        assert!(
            matches!(&ops[5], Operation::Sharpen { size, amount } if n(size) == 1.0 && n(amount) == 2.0)
        );
        assert!(matches!(&ops[6], Operation::HighPass { size } if n(size) == 3.0));
        assert!(matches!(&ops[7], Operation::Convolve { kernel } if kernel.len() == 9));
        assert!(
            matches!(&ops[8], Operation::Offset { translate } if translate.evaluate(0.0) == [3.0, -4.0])
        );
        assert!(matches!(&ops[9], Operation::Dilate { size } if n(size) == 2.0));
        assert!(matches!(&ops[10], Operation::Erode { size } if n(size) == 1.0));
        assert!(
            matches!(&ops[11], Operation::Combine { target, mode: Combine::Subtract } if target == "land")
        );
        assert!(matches!(&ops[12], Operation::Opacity { value } if n(value) == 0.5));
        assert!(
            matches!(&ops[13], Operation::Color { value } if value.evaluate(0.0) == Color::parse("red").unwrap())
        );
        assert!(matches!(ops[14], Operation::Luminance));
        assert!(
            matches!(&ops[15], Operation::Ramp { stops } if stops.len() == 2 && stops[1].0 == 1.0)
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
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"combine","target":"a","mode":"xor"}]}"#,
            ),
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"ramp","stops":[[0.5,"red"],[0.2,"blue"]]}]}"#,
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

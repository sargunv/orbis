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

/// The root `dev.sargunv.orbis:targets`: `[{"id": …, "precision": …}]`,
/// ids unique.
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
            if m.key != "id" && m.key != "precision" {
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
        let precision = match item.get("precision") {
            None => Precision::Standard,
            Some(p) => match p.as_str().and_then(Precision::from_name) {
                Some(precision) => precision,
                None => {
                    cx.error(
                        format!("{path}.precision"),
                        p.span,
                        format!(
                            "expected one of [{}], {p} found",
                            Precision::NAMES.join(", ")
                        ),
                    );
                    continue;
                }
            },
        };
        targets.push(RenderTarget {
            id: name,
            precision,
        });
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

/// A parameter's spec: a number, colour or pair, which may follow the zoom
/// and the camera, as a layer-phase paint property may.
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
        extended: Inputs::CAMERA,
        preview: Some(Preview::Targets),
        ..Prop::ROOT
    }
}

const NUMBER: PropertyType = PropertyType::Number;
const COLOR: PropertyType = PropertyType::Color;
const PAIR: PropertyType = PropertyType::Array(orb_expr::Type::Number, Some(2));
/// A blur's or a dilation's size: past 64 pixels it reaches no further.
const SIZE: (f64, f64) = (0.0, 64.0);
const NOISE_ANCHOR: &str = "dev.sargunv.orbis:anchor";
const REFERENCE_ZOOM: &str = "dev.sargunv.orbis:reference-zoom";

fn noise_anchor(cx: &mut Cx, json: &Json, path: &str) -> Option<NoiseAnchor> {
    let anchor = json.get(NOISE_ANCHOR);
    let reference = json.get(REFERENCE_ZOOM);
    if let Some(anchor) = anchor
        && anchor.as_str() == Some("map")
    {
        let zoom = match reference {
            Some(zoom) => cx.number_in(zoom, &format!("{path}.{REFERENCE_ZOOM}"), 0.0, 24.0)?,
            None => 14.0,
        };
        return Some(NoiseAnchor::Map {
            reference_zoom: zoom,
        });
    }
    if let Some(reference) = reference {
        cx.error(
            format!("{path}.{REFERENCE_ZOOM}"),
            reference.span,
            format!("requires \"{NOISE_ANCHOR}\": \"map\""),
        );
    }
    match anchor {
        None => Some(NoiseAnchor::Adaptive),
        Some(a) if a.as_str() == Some("viewport") => Some(NoiseAnchor::Viewport),
        Some(a) => {
            cx.error(
                format!("{path}.{NOISE_ANCHOR}"),
                a.span,
                format!("expected one of [map, viewport], {a} found"),
            );
            None
        }
    }
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
        "blur" | "high-pass" | "dilate" | "erode" => &["size"],
        "noise" => &["size", "amount", "speed", NOISE_ANCHOR, REFERENCE_ZOOM],
        "sharpen" => &["size", "amount"],
        "threshold" => &["cutoff"],
        "invert" | "luminance" => &[],
        "convolve" => &["kernel"],
        "offset" => &["translate"],
        "combine" => &["target", "mode"],
        "opacity" | "color" => &["value"],
        "ramp" => &["stops"],
        "split" => &["red", "green", "blue", "radial"],
        "gradient" => &["kind", "angle", "center", "length", "radius"],
        "bands" => &["count", "width"],
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
            anchor: noise_anchor(cx, json, path)?,
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
            speed: read(
                cx,
                json,
                path,
                &param("speed", NUMBER, "0", (0.0, 1024.0)),
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
        "split" => {
            let mut channel = |name| {
                let spec = param(name, PAIR, "[0,0]", (-512.0, 512.0));
                read(cx, json, path, &spec, false)
            };
            let (red, green, blue) = (channel("red"), channel("green"), channel("blue"));
            let radial = match json.get("radial") {
                None => Some(false),
                Some(r) => {
                    let radial = r.as_bool();
                    if radial.is_none() {
                        cx.error(
                            format!("{path}.radial"),
                            r.span,
                            format!("expected a boolean, {r} found"),
                        );
                    }
                    radial
                }
            };
            Operation::Split {
                red,
                green,
                blue,
                radial: radial?,
            }
        }
        "gradient" => {
            let kind = match json.get("kind") {
                None => Some("linear"),
                Some(k) => {
                    let kind = k.as_str().filter(|k| ["linear", "radial"].contains(k));
                    if kind.is_none() {
                        cx.error(
                            format!("{path}.kind"),
                            k.span,
                            format!("expected one of [linear, radial], {k} found"),
                        );
                    }
                    kind
                }
            };
            let center = read(
                cx,
                json,
                path,
                &param("center", PAIR, "[0,0]", (-16384.0, 16384.0)),
                false,
            );
            // A length or radius: `None` where it is left to the view, as
            // it is where the value doesn't fit.
            let extent = |cx: &mut Cx, name| {
                json.get(name).and_then(|_| {
                    let spec = param(name, NUMBER, "1", (0.0, 16384.0));
                    Some(read(cx, json, path, &spec, false)).filter(Property::is_set)
                })
            };
            // What the other kind takes and this one doesn't.
            let ignore = |cx: &mut Cx, names: &[&str]| {
                for m in members.iter().filter(|m| names.contains(&m.key.as_str())) {
                    cx.warn_unknown(path, m);
                }
            };
            match kind? {
                "linear" => {
                    ignore(cx, &["radius"]);
                    let spec = param("angle", NUMBER, "0", (-360.0, 360.0));
                    Operation::Gradient {
                        kind: Gradient::Linear {
                            angle: read(cx, json, path, &spec, false),
                            center,
                            length: extent(cx, "length"),
                        },
                    }
                }
                _ => {
                    ignore(cx, &["angle", "length"]);
                    Operation::Gradient {
                        kind: Gradient::Radial {
                            center,
                            radius: extent(cx, "radius"),
                        },
                    }
                }
            }
        }
        "bands" => Operation::Bands {
            count: read(
                cx,
                json,
                path,
                &param("count", NUMBER, "8", (1.0, 256.0)),
                false,
            ),
            width: read(
                cx,
                json,
                path,
                &param("width", NUMBER, "0.5", (0.0, 1.0)),
                false,
            ),
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

/// A ramp's `stops`: `[[coverage, colour], …]`, up to eight, coverages
/// 0–1 and, those that are constants, in order. Either may follow the zoom
/// and the camera.
fn ramp_stops(
    cx: &mut Cx,
    json: &Json,
    path: &str,
) -> Option<Vec<(Property<f32>, Property<Color>)>> {
    let at = format!("{path}.stops");
    let Some(stops) = json.get("stops") else {
        cx.error(path, json.span, "missing required property \"stops\"");
        return None;
    };
    let items = cx.array(stops, &at)?;
    let pairs: Option<Vec<&[Json]>> = (items.iter())
        .map(|stop| stop.as_array().filter(|s| s.len() == 2))
        .collect();
    let Some(pairs) = pairs.filter(|p| (1..=8).contains(&p.len())) else {
        cx.error(at, stops.span, "expected 1 to 8 [coverage, colour] stops");
        return None;
    };
    let coverage = param("coverage", NUMBER, "0", (0.0, 1.0));
    let color = param("color", COLOR, "\"black\"", (0.0, 0.0));
    let parsed: Vec<(Property<f32>, Property<Color>)> = (pairs.iter().enumerate())
        .map(|(i, pair)| {
            let path = format!("{at}[{i}]");
            (
                typed_property_at(cx, Some(&pair[0]), &format!("{path}[0]"), &coverage),
                typed_property_at(cx, Some(&pair[1]), &format!("{path}[1]"), &color),
            )
        })
        .collect();
    // A coverage that doesn't fit draws the default, out of the order.
    let constants: Vec<Option<f32>> = (parsed.iter())
        .map(|(at, _)| (at.is_set() && at.kind() == Kind::Constant).then(|| at.evaluate(0.0)))
        .collect();
    let ordered = (constants.iter().flatten())
        .zip(constants.iter().flatten().skip(1))
        .all(|(a, b)| a <= b);
    if !ordered {
        cx.error(at, stops.span, "expected the stops' coverages in order");
        return None;
    }
    Some(parsed)
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
                   {"type":"ramp","stops":[[0,"rgba(0,0,0,0)"],[1,"#fff"]]},
                   {"type":"noise","size":8,"speed":["interpolate",["linear"],["zoom"],10,4,14,12]},
                   {"type":"split","red":[2,0],"blue":[-2,1],"radial":true},
                   {"type":"gradient","kind":"linear","angle":90},{"type":"gradient","kind":"radial"},
                   {"type":"bands","count":6,"width":0.25}]},
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
        assert_eq!(ops.len(), 21);
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
            matches!(&ops[2], Operation::Noise { size, amount, speed, anchor: NoiseAnchor::Adaptive }
                if n(size) == 2.0 && n(amount) == 0.25 && n(speed) == 0.0)
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
            matches!(&ops[15], Operation::Ramp { stops } if stops.len() == 2 && n(&stops[1].0) == 1.0)
        );
        // A noise's speed may follow the zoom.
        let Operation::Noise { speed, .. } = &ops[16] else {
            panic!("{:?}", ops[16]);
        };
        assert_eq!((speed.evaluate(10.0), speed.evaluate(14.0)), (4.0, 12.0));
        let Operation::Split {
            red,
            green,
            blue,
            radial,
        } = &ops[17]
        else {
            panic!("{:?}", ops[17]);
        };
        let pair = |p: &Property<[f32; 2]>| p.evaluate(0.0);
        assert_eq!(
            (pair(red), pair(green), pair(blue), *radial),
            ([2.0, 0.0], [0.0, 0.0], [-2.0, 1.0], true)
        );
        // A gradient's centre is the view's, its extent the view's, unless
        // given.
        assert!(matches!(&ops[18], Operation::Gradient {
            kind: Gradient::Linear { angle, center, length: None }
        } if n(angle) == 90.0 && pair(center) == [0.0, 0.0]));
        assert!(matches!(
            &ops[19],
            Operation::Gradient {
                kind: Gradient::Radial { radius: None, .. }
            }
        ));
        assert!(ops.iter().all(|op| !op.reads_camera()));
        assert!(
            matches!(&ops[20], Operation::Bands { count, width } if n(count) == 6.0 && n(width) == 0.25)
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
    fn invalid_targets_and_operations_are_errors_but_values_draw_defaults() {
        for (root, layers) in [
            (
                r#""dev.sargunv.orbis:targets":[{"id":"a"},{"id":"a"}],"#,
                "",
            ),
            (r#""dev.sargunv.orbis:targets":[{}],"#, ""),
            (
                r#""dev.sargunv.orbis:targets":[{"id":"a","precision":"double"}],"#,
                "",
            ),
            ("", r#"{"id":"p","type":"dev.sargunv.orbis.image-process"}"#),
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"smudge"}]}"#,
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
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"split","radial":"yes"}]}"#,
            ),
            (
                "",
                r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"a",
                    "dev.sargunv.orbis:operations":[{"type":"gradient","kind":"conic"}]}"#,
            ),
        ] {
            assert!(parse(ON, root, layers).is_err(), "{root} {layers}");
        }
        // A parameter's value that doesn't fit it draws the parameter's
        // default, as a property's does.
        for ops in [
            r#"{"type":"blur","size":-1}"#,
            r#"{"type":"noise","size":4,"speed":-1}"#,
            r#"{"type":"split","red":[1,2,3]}"#,
            r#"{"type":"bands","count":0}"#,
            r#"{"type":"bands","width":2}"#,
        ] {
            draws_defaults(ops);
        }
    }

    #[test]
    fn a_target_may_keep_high_precision() {
        let root = r#""dev.sargunv.orbis:targets":[{"id":"a"},{"id":"b","precision":"high"},
            {"id":"c","precision":"standard"}],"#;
        let style = parse(ON, root, "").unwrap();
        assert!(style.warnings.is_empty(), "{:?}", style.warnings);
        let precisions: Vec<_> = style.targets.iter().map(|t| t.precision).collect();
        assert_eq!(
            precisions,
            [Precision::Standard, Precision::High, Precision::Standard]
        );
    }

    #[test]
    fn a_radial_gradient_has_no_angle() {
        let style = parse(
            ON,
            TARGETS,
            r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"paint",
                "dev.sargunv.orbis:operations":[{"type":"gradient","kind":"radial","angle":45}]}"#,
        )
        .unwrap();
        let paths: Vec<&str> = style.warnings.iter().map(|w| w.path.as_str()).collect();
        assert_eq!(paths, ["layers[0].dev.sargunv.orbis:operations[0].angle"]);
        assert!(matches!(
            style.layers[0].operations[..],
            [Operation::Gradient {
                kind: Gradient::Radial { .. }
            }]
        ));
        // Nor a length; a linear one has no radius.
        let style = parse(
            ON,
            TARGETS,
            r#"{"id":"p","type":"dev.sargunv.orbis.image-process","dev.sargunv.orbis:source-target":"paint",
                "dev.sargunv.orbis:operations":[{"type":"gradient","kind":"radial","length":45},
                  {"type":"gradient","radius":45}]}"#,
        )
        .unwrap();
        let paths: Vec<&str> = style.warnings.iter().map(|w| w.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "layers[0].dev.sargunv.orbis:operations[0].length",
                "layers[0].dev.sargunv.orbis:operations[1].radius"
            ]
        );
    }

    /// One image-process layer of `ops`.
    fn operations(ops: &str) -> Result<Style, crate::v8::Diagnostics> {
        parse(
            ON,
            TARGETS,
            &format!(
                r#"{{"id":"p","type":"dev.sargunv.orbis.image-process",
                    "dev.sargunv.orbis:source-target":"paint","dev.sargunv.orbis:operations":[{ops}]}}"#
            ),
        )
    }

    /// Asserts that image-process operations `ops` parse, with a value
    /// that draws its default.
    fn draws_defaults(ops: &str) {
        let style = operations(ops).unwrap_or_else(|d| panic!("{ops}: {d}"));
        assert!(
            (style.warnings.iter()).any(|w| w.message.ends_with(crate::v8::INVALID_VALUE)),
            "{ops}: {:?}",
            style.warnings
        );
    }

    #[test]
    fn gradients_take_a_centre_and_an_extent_in_pixels() {
        let style = operations(
            r#"{"type":"gradient","center":[40,-20],"length":300,"angle":30},
               {"type":"gradient","kind":"radial","center":[-10,5],"radius":120}"#,
        )
        .unwrap();
        assert!(style.warnings.is_empty(), "{:?}", style.warnings);
        let [
            Operation::Gradient {
                kind:
                    Gradient::Linear {
                        angle,
                        center: linear,
                        length: Some(length),
                    },
            },
            Operation::Gradient {
                kind:
                    Gradient::Radial {
                        center: radial,
                        radius: Some(radius),
                    },
            },
        ] = &style.layers[0].operations[..]
        else {
            panic!("{:?}", style.layers[0].operations);
        };
        assert_eq!(
            (
                angle.evaluate(0.0),
                linear.evaluate(0.0),
                length.evaluate(0.0)
            ),
            (30.0, [40.0, -20.0], 300.0)
        );
        assert_eq!(
            (radial.evaluate(0.0), radius.evaluate(0.0)),
            ([-10.0, 5.0], 120.0)
        );
        for invalid in [
            r#"{"type":"gradient","length":-1}"#,
            r#"{"type":"gradient","kind":"radial","radius":-1}"#,
            r#"{"type":"gradient","center":[1]}"#,
        ] {
            draws_defaults(invalid);
        }
        // An extent that doesn't fit is left to the view, as an absent one.
        let style = operations(
            r#"{"type":"gradient","length":-1},{"type":"gradient","kind":"radial","radius":-1}"#,
        )
        .unwrap();
        assert!(
            matches!(
                &style.layers[0].operations[..],
                [
                    Operation::Gradient {
                        kind: Gradient::Linear { length: None, .. }
                    },
                    Operation::Gradient {
                        kind: Gradient::Radial { radius: None, .. }
                    },
                ]
            ),
            "{:?}",
            style.layers[0].operations
        );
    }

    /// Numbers, pairs and colours may follow the camera as well as the
    /// zoom, as a layer's paint may, without a warning: the operations are
    /// Orbis's own, beyond what GL JS and Native read anyway.
    #[test]
    fn operation_values_follow_the_camera() {
        let style = operations(
            r#"{"type":"gradient","angle":["*",-1,["bearing"]]},
               {"type":"offset",
                 "translate":["interpolate",["linear"],["pitch"],0,["literal",[0,0]],60,["literal",[0,-200]]]},
               {"type":"color","value":["interpolate",["linear"],["pitch"],0,"red",60,"blue"]},
               {"type":"blur","size":["interpolate",["linear"],["zoom"],10,2,14,6]}"#,
        )
        .unwrap();
        assert!(style.warnings.is_empty(), "{:?}", style.warnings);
        let ops = &style.layers[0].operations;
        let camera = |pitch, bearing| orb_expr::CameraInputs { pitch, bearing };
        let Operation::Gradient {
            kind: Gradient::Linear { angle, .. },
        } = &ops[0]
        else {
            panic!("{:?}", ops[0]);
        };
        assert_eq!(angle.evaluate_camera(0.0, camera(0.0, 90.0)), -90.0);
        let Operation::Offset { translate } = &ops[1] else {
            panic!("{:?}", ops[1]);
        };
        assert_eq!(
            translate.evaluate_camera(0.0, camera(30.0, 0.0)),
            [0.0, -100.0]
        );
        let Operation::Color { value } = &ops[2] else {
            panic!("{:?}", ops[2]);
        };
        assert_eq!(
            value.evaluate_camera(0.0, camera(60.0, 0.0)),
            Color::parse("blue").unwrap()
        );
        let reads: Vec<bool> = ops.iter().map(Operation::reads_camera).collect();
        assert_eq!(reads, [true, true, true, false]);
    }

    /// A ramp's stops are a coverage and a colour each, either a constant
    /// or following the zoom and the camera; constant coverages are in
    /// order.
    #[test]
    fn ramp_stops_follow_the_zoom_and_the_camera() {
        let style = operations(
            r##"{"type":"ramp","stops":[
                 [0,"rgba(0,0,0,0)"],
                 [["interpolate",["linear"],["zoom"],10,0.2,14,0.6],"#f00"],
                 [1,["interpolate",["linear"],["pitch"],0,"#000",60,"#fff"]]]}"##,
        )
        .unwrap();
        assert!(style.warnings.is_empty(), "{:?}", style.warnings);
        let op = &style.layers[0].operations[0];
        let Operation::Ramp { stops } = op else {
            panic!("{op:?}");
        };
        let camera = orb_expr::CameraInputs {
            pitch: 60.0,
            bearing: 0.0,
        };
        let at: Vec<f32> = stops.iter().map(|s| s.0.evaluate(12.0)).collect();
        assert_eq!(at, [0.0, 0.4, 1.0]);
        assert_eq!(
            stops[2].1.evaluate_camera(0.0, camera),
            Color::parse("#fff").unwrap()
        );
        assert!(op.reads_camera());
        for invalid in [
            r#"{"type":"ramp","stops":[[1.5,"red"]]}"#,
            r#"{"type":"ramp","stops":[[0,"nope"]]}"#,
        ] {
            draws_defaults(invalid);
        }
        for invalid in [
            r#"{"type":"ramp","stops":[]}"#,
            r#"{"type":"ramp","stops":[[0,"red","blue"]]}"#,
            r#"{"type":"ramp","stops":[[0.6,"red"],
               [["interpolate",["linear"],["zoom"],0,0.5,20,0.5],"red"],[0.4,"blue"]]}"#,
            r#"{"type":"ramp","stops":[[0,"red"],[1,"red"],[0,"red"],[1,"red"],[0,"red"],
               [1,"red"],[0,"red"],[1,"red"],[1,"red"]]}"#,
        ] {
            assert!(operations(invalid).is_err(), "{invalid}");
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

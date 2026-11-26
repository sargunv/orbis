//! Render targets (the `dev.sargunv.orbis:targets` preview): the root list
//! of targets, and an image-process layer's operations.

use crate::property::v8_enum;
use crate::v8::{Cx, layer::typed_property_at};
use crate::*;
use orb_expr::PropertyType;
use orb_expr::json::{Json, JsonValue, Member};

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
            Some(p) => match cx.enumeration(p, &format!("{path}.precision")) {
                Some(precision) => precision,
                None => continue,
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
pub(crate) fn parse_operations(cx: &mut Cx, json: &Json, path: &str) -> Operations {
    let Some(items) = cx.array(json, path) else {
        return Operations::default();
    };
    let mut reads_camera = false;
    let ops = (items.iter().enumerate())
        .filter_map(|(i, item)| {
            let (op, camera) = parse_operation(cx, item, &format!("{path}[{i}]"))?;
            reads_camera |= camera;
            Some(op)
        })
        .collect();
    Operations::new(ops, reads_camera)
}

/// An operation `type`: the keys it takes and how it is made of them.
struct Spec {
    name: &'static str,
    params: &'static [Param],
    build: fn(&mut Params) -> Option<Operation>,
}

/// A key an operation takes, and whether it must be given.
struct Param {
    kind: ParamKind,
    required: bool,
}

enum ParamKind {
    /// A number, colour or pair, which may follow the zoom and the camera,
    /// as a layer-phase paint property may: read by [`Params::value`].
    Value(Prop),
    /// A key its operation reads itself: a target, a kernel, a choice.
    Key(&'static str),
}

impl Param {
    const fn required(mut self) -> Self {
        self.required = true;
        self
    }

    const fn name(&self) -> &'static str {
        match &self.kind {
            ParamKind::Value(prop) => prop.name,
            ParamKind::Key(name) => name,
        }
    }
}

/// An optional value parameter, its default drawn where it is absent or
/// what is given doesn't fit.
const fn value(
    name: &'static str,
    ty: PropertyType,
    default: &'static str,
    range: (f64, f64),
) -> Param {
    Param {
        kind: ParamKind::Value(param(name, ty, default, range)),
        required: false,
    }
}

/// An optional key its operation reads itself.
const fn key(name: &'static str) -> Param {
    Param {
        kind: ParamKind::Key(name),
        required: false,
    }
}

/// A value's spec: a number, colour or pair, which may follow the zoom and
/// the camera, as a layer-phase paint property may.
const fn param(
    name: &'static str,
    ty: PropertyType,
    default: &'static str,
    (minimum, maximum): (f64, f64),
) -> Prop {
    let mut prop = prop(name, ty, Some(default));
    prop.interpolated = true;
    prop.minimum = Some(minimum);
    prop.maximum = Some(maximum);
    prop.phase = Some(Phase::Layer);
    prop.extended = Inputs::CAMERA;
    prop.preview = Some(Preview::Targets);
    prop
}

const NUMBER: PropertyType = PropertyType::Number;
const COLOR: PropertyType = PropertyType::Color;
const PAIR: PropertyType = PropertyType::Array(orb_expr::Type::Number, Some(2));
/// A colour's range, which colours don't read.
const ANY: (f64, f64) = (0.0, 0.0);
/// A blur's or a dilation's size: past 64 pixels it reaches no further.
const SIZE: Param = value("size", NUMBER, "1", (0.0, 64.0)).required();
const CHANNEL: (f64, f64) = (-512.0, 512.0);
const EXTENT: (f64, f64) = (0.0, 16384.0);
const NOISE_ANCHOR: &str = "dev.sargunv.orbis:anchor";
const REFERENCE_ZOOM: &str = "dev.sargunv.orbis:reference-zoom";

/// Every operation, in the order an unknown `type`'s error lists them.
static SPECS: &[Spec] = &[
    Spec {
        name: "select",
        params: &[
            value("value", COLOR, "\"white\"", ANY).required(),
            value("tolerance", NUMBER, "16", (0.0, 255.0)),
        ],
        build: |p| {
            Some(Operation::Select {
                value: p.value("value"),
                tolerance: p.value("tolerance"),
            })
        },
    },
    Spec {
        name: "blur",
        params: &[SIZE],
        build: |p| {
            Some(Operation::Blur {
                size: p.value("size"),
            })
        },
    },
    Spec {
        name: "noise",
        params: &[
            value("size", NUMBER, "1", (0.0, 1024.0)).required(),
            value("amount", NUMBER, "0.25", (0.0, 1.0)),
            value("speed", NUMBER, "0", (0.0, 1024.0)),
            key(NOISE_ANCHOR),
            key(REFERENCE_ZOOM),
        ],
        build: |p| {
            Some(Operation::Noise {
                anchor: noise_anchor(p)?,
                size: p.value("size"),
                amount: p.value("amount"),
                speed: p.value("speed"),
            })
        },
    },
    Spec {
        name: "threshold",
        params: &[value("cutoff", NUMBER, "128", (0.0, 255.0))],
        build: |p| {
            Some(Operation::Threshold {
                cutoff: p.value("cutoff"),
            })
        },
    },
    Spec {
        name: "invert",
        params: &[],
        build: |_| Some(Operation::Invert),
    },
    Spec {
        name: "sharpen",
        params: &[SIZE, value("amount", NUMBER, "1", (0.0, 16.0))],
        build: |p| {
            Some(Operation::Sharpen {
                size: p.value("size"),
                amount: p.value("amount"),
            })
        },
    },
    Spec {
        name: "high-pass",
        params: &[SIZE],
        build: |p| {
            Some(Operation::HighPass {
                size: p.value("size"),
            })
        },
    },
    Spec {
        name: "convolve",
        params: &[key("kernel").required()],
        build: |p| {
            let kernel = p.key("kernel")?;
            let weights: Option<Vec<f32>> = match &kernel.value {
                JsonValue::Array(items) => {
                    items.iter().map(|w| w.as_f64().map(|w| w as f32)).collect()
                }
                _ => None,
            };
            let weights = weights.filter(|w| [9, 25, 49].contains(&w.len()));
            if weights.is_none() {
                let path = format!("{}.kernel", p.path);
                let message = "expected an array of 9, 25 or 49 numbers";
                p.cx.error(path, kernel.span, message);
            }
            Some(Operation::Convolve { kernel: weights? })
        },
    },
    Spec {
        name: "offset",
        params: &[value("translate", PAIR, "[0,0]", CHANNEL).required()],
        build: |p| {
            Some(Operation::Offset {
                translate: p.value("translate"),
            })
        },
    },
    Spec {
        name: "dilate",
        params: &[SIZE],
        build: |p| {
            Some(Operation::Dilate {
                size: p.value("size"),
            })
        },
    },
    Spec {
        name: "erode",
        params: &[SIZE],
        build: |p| {
            Some(Operation::Erode {
                size: p.value("size"),
            })
        },
    },
    Spec {
        name: "combine",
        params: &[key("target").required(), key("mode")],
        build: |p| {
            let mode = p.choice("mode").unwrap_or(Combine::Multiply);
            let target = p.key("target")?;
            let target = p.cx.string(target, &format!("{}.target", p.path))?;
            Some(Operation::Combine { target, mode })
        },
    },
    Spec {
        name: "opacity",
        params: &[value("value", NUMBER, "1", (0.0, 1.0)).required()],
        build: |p| {
            Some(Operation::Opacity {
                value: p.value("value"),
            })
        },
    },
    Spec {
        name: "color",
        params: &[value("value", COLOR, "\"white\"", ANY).required()],
        build: |p| {
            Some(Operation::Color {
                value: p.value("value"),
            })
        },
    },
    Spec {
        name: "luminance",
        params: &[],
        build: |_| Some(Operation::Luminance),
    },
    Spec {
        name: "ramp",
        params: &[key("stops").required()],
        build: |p| {
            Some(Operation::Ramp {
                stops: ramp_stops(p)?,
            })
        },
    },
    Spec {
        name: "split",
        params: &[
            value("red", PAIR, "[0,0]", CHANNEL),
            value("green", PAIR, "[0,0]", CHANNEL),
            value("blue", PAIR, "[0,0]", CHANNEL),
            key("radial"),
        ],
        build: |p| {
            Some(Operation::Split {
                red: p.value("red"),
                green: p.value("green"),
                blue: p.value("blue"),
                radial: p.boolean("radial").unwrap_or(false),
            })
        },
    },
    Spec {
        name: "gradient",
        params: &[
            key("kind"),
            value("angle", NUMBER, "0", (-360.0, 360.0)),
            value("center", PAIR, "[0,0]", (-16384.0, 16384.0)),
            value("length", NUMBER, "1", EXTENT),
            value("radius", NUMBER, "1", EXTENT),
        ],
        build: |p| {
            let center = p.value("center");
            let kind = match p.key("kind") {
                None => GradientKind::Linear,
                Some(_) => p.choice("kind")?,
            };
            let kind = match kind {
                GradientKind::Linear => {
                    p.ignore("radius");
                    Gradient::Linear {
                        angle: p.value("angle"),
                        center,
                        length: p.extent("length"),
                    }
                }
                GradientKind::Radial => {
                    p.ignore("angle");
                    p.ignore("length");
                    Gradient::Radial {
                        center,
                        radius: p.extent("radius"),
                    }
                }
            };
            Some(Operation::Gradient { kind })
        },
    },
    Spec {
        name: "bands",
        params: &[
            value("count", NUMBER, "8", (1.0, 256.0)),
            value("width", NUMBER, "0.5", (0.0, 1.0)),
        ],
        build: |p| {
            Some(Operation::Bands {
                count: p.value("count"),
                width: p.value("width"),
            })
        },
    },
];

v8_enum!(
    /// A gradient's `kind`.
    pub(crate) GradientKind { Linear = "linear", Radial = "radial" }
);
v8_enum!(
    /// A noise's `dev.sargunv.orbis:anchor`; without one it adapts to the
    /// zoom.
    pub(crate) AnchorKind { Map = "map", Viewport = "viewport" }
);

/// An operation's keys as its [`Spec::build`] reads them.
struct Params<'a> {
    cx: &'a mut Cx,
    json: &'a Json,
    members: &'a [Member],
    path: &'a str,
    spec: &'static Spec,
    /// Whether a value read so far reads the camera beyond its zoom.
    reads_camera: bool,
    /// The parameters read or ignored so far, by their place in the spec.
    read: u64,
}

impl<'a> Params<'a> {
    /// Value parameter `name`: its default where it is absent or what is
    /// given doesn't fit.
    fn value<T: PropertyValue>(&mut self, name: &str) -> Property<T> {
        let ParamKind::Value(prop) = &self.declared(name).kind else {
            panic!("{}'s \"{name}\" is no value", self.spec.name);
        };
        let path = format!("{}.{name}", self.path);
        self.property_at(self.json.get(name), &path, prop)
    }

    /// Value parameter `name`, or `None` where it is left to the view, as
    /// it is where the value doesn't fit.
    fn extent(&mut self, name: &str) -> Option<Property<f32>> {
        self.declared(name);
        self.json.get(name)?;
        Some(self.value(name)).filter(Property::is_set)
    }

    /// A value of `prop` at `path`, noting whether it reads the camera.
    fn property_at<T: PropertyValue>(
        &mut self,
        json: Option<&Json>,
        path: &str,
        prop: &Prop,
    ) -> Property<T> {
        let value: Property<T> = typed_property_at(self.cx, json, path, prop);
        self.reads_camera |=
            (value.as_expression()).is_some_and(|e| e.reads().contains(Inputs::CAMERA));
        value
    }

    /// Key `name`, where it is given.
    fn key(&mut self, name: &str) -> Option<&'a Json> {
        self.declared(name);
        self.json.get(name)
    }

    /// Parameter `name` of the spec, noted as read.
    fn declared(&mut self, name: &str) -> &'static Param {
        let spec: &'static Spec = self.spec;
        let Some(i) = spec.params.iter().position(|p| p.name() == name) else {
            panic!("{} has no key \"{name}\"", spec.name);
        };
        self.read |= 1 << i;
        &spec.params[i]
    }

    /// Whether every value parameter has been read or ignored, so none the
    /// spec declares goes unchecked.
    fn read_all_values(&self) -> bool {
        (self.spec.params.iter().enumerate())
            .all(|(i, p)| matches!(p.kind, ParamKind::Key(_)) || self.read & 1 << i != 0)
    }

    /// Key `name`'s choice, where it is given and one of `E`'s.
    fn choice<E: Enum>(&mut self, name: &str) -> Option<E> {
        let json = self.key(name)?;
        self.cx.enumeration(json, &format!("{}.{name}", self.path))
    }

    /// Key `name`'s boolean, where it is given and one.
    fn boolean(&mut self, name: &str) -> Option<bool> {
        let json = self.key(name)?;
        let value = json.as_bool();
        if value.is_none() {
            let message = format!("expected a boolean, {json} found");
            self.cx
                .error(format!("{}.{name}", self.path), json.span, message);
        }
        value
    }

    /// Warns of key `name`, which this form of the operation doesn't take.
    fn ignore(&mut self, name: &str) {
        self.declared(name);
        if let Some(m) = self.members.iter().find(|m| m.key == name) {
            self.cx.warn_unknown(self.path, m);
        }
    }
}

/// A noise's lattice: adapting to the zoom, or anchored to the map at a
/// reference zoom or to the viewport.
fn noise_anchor(p: &mut Params) -> Option<NoiseAnchor> {
    let anchor = p.choice(NOISE_ANCHOR);
    let reference = p.key(REFERENCE_ZOOM);
    if anchor == Some(AnchorKind::Map) {
        let zoom = match reference {
            Some(zoom) => {
                let path = format!("{}.{REFERENCE_ZOOM}", p.path);
                p.cx.number_in(zoom, &path, 0.0, 24.0)?
            }
            None => 14.0,
        };
        return Some(NoiseAnchor::Map {
            reference_zoom: zoom,
        });
    }
    if let Some(reference) = reference {
        p.cx.error(
            format!("{}.{REFERENCE_ZOOM}", p.path),
            reference.span,
            format!("requires \"{NOISE_ANCHOR}\": \"map\""),
        );
    }
    Some(match anchor {
        Some(_) => NoiseAnchor::Viewport,
        None => NoiseAnchor::Adaptive,
    })
}

/// An operation and whether any of its values reads the camera beyond its
/// zoom.
fn parse_operation(cx: &mut Cx, json: &Json, path: &str) -> Option<(Operation, bool)> {
    let members = cx.object(json, path)?;
    let Some(ty) = json.get("type") else {
        cx.error(path, json.span, "missing required property \"type\"");
        return None;
    };
    let name = cx.string(ty, &format!("{path}.type"))?;
    let Some(spec) = SPECS.iter().find(|op| op.name == name) else {
        let names: Vec<&str> = SPECS.iter().map(|op| op.name).collect();
        cx.error(
            format!("{path}.type"),
            ty.span,
            format!("expected one of [{}], {ty} found", names.join(", ")),
        );
        return None;
    };
    for m in members {
        if m.key != "type" && !spec.params.iter().any(|p| p.name() == m.key) {
            cx.warn_unknown(path, m);
        }
    }
    let errors = cx.diagnostics.errors.len();
    for param in spec.params.iter().filter(|p| p.required) {
        if json.get(param.name()).is_none() {
            let message = format!("missing required property \"{}\"", param.name());
            cx.error(path, json.span, message);
        }
    }
    let mut params = Params {
        cx,
        json,
        members,
        path,
        spec,
        reads_camera: false,
        read: 0,
    };
    let op = (spec.build)(&mut params);
    debug_assert!(
        op.is_none() || params.read_all_values(),
        "{} declares a value it doesn't read",
        spec.name
    );
    let reads_camera = params.reads_camera;
    (cx.diagnostics.errors.len() == errors)
        .then_some(op)
        .flatten()
        .map(|op| (op, reads_camera))
}

/// A ramp's `stops`: `[[coverage, colour], …]`, up to eight, coverages
/// 0–1 and, those that are constants, in order. Either may follow the zoom
/// and the camera.
fn ramp_stops(p: &mut Params) -> Option<Vec<(Property<f32>, Property<Color>)>> {
    const COVERAGE: Prop = param("coverage", NUMBER, "0", (0.0, 1.0));
    const STOP_COLOR: Prop = param("color", COLOR, "\"black\"", ANY);
    let at = format!("{}.stops", p.path);
    let stops = p.key("stops")?;
    let items = p.cx.array(stops, &at)?;
    let pairs: Option<Vec<&[Json]>> = (items.iter())
        .map(|stop| stop.as_array().filter(|s| s.len() == 2))
        .collect();
    let Some(pairs) = pairs.filter(|p| (1..=8).contains(&p.len())) else {
        p.cx.error(at, stops.span, "expected 1 to 8 [coverage, colour] stops");
        return None;
    };
    let parsed: Vec<(Property<f32>, Property<Color>)> = (pairs.iter().enumerate())
        .map(|(i, pair)| {
            let path = format!("{at}[{i}]");
            (
                p.property_at(Some(&pair[0]), &format!("{path}[0]"), &COVERAGE),
                p.property_at(Some(&pair[1]), &format!("{path}[1]"), &STOP_COLOR),
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
        p.cx.error(at, stops.span, "expected the stops' coverages in order");
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
        assert_eq!(mask.destination_target.as_deref(), Some("land"));
        let Some(TargetRead::Process {
            source,
            operations: ops,
        }) = &mask.reads_target
        else {
            panic!("{:?}", mask.reads_target);
        };
        assert_eq!(source, "paint");
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
        assert!(!ops.reads_camera());
        assert!(
            matches!(&ops[20], Operation::Bands { count, width } if n(count) == 6.0 && n(width) == 0.25)
        );
        assert_eq!(land.source, None);
        assert!(matches!(&land.reads_target, Some(TargetRead::Fill(t)) if t == "land"));
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
    fn noise_anchors_to_the_zoom_the_map_or_the_viewport() {
        let style = operations(
            r#"{"type":"noise","size":4},
               {"type":"noise","size":4,"dev.sargunv.orbis:anchor":"map"},
               {"type":"noise","size":4,"dev.sargunv.orbis:anchor":"map",
                 "dev.sargunv.orbis:reference-zoom":20},
               {"type":"noise","size":4,"dev.sargunv.orbis:anchor":"viewport"}"#,
        )
        .unwrap();
        assert!(style.warnings.is_empty(), "{:?}", style.warnings);
        let anchors: Vec<NoiseAnchor> = (operations_of(&style.layers[0]).iter())
            .map(|op| match op {
                Operation::Noise { anchor, .. } => *anchor,
                op => panic!("{op:?}"),
            })
            .collect();
        assert_eq!(
            anchors,
            [
                NoiseAnchor::Adaptive,
                NoiseAnchor::Map {
                    reference_zoom: 14.0
                },
                NoiseAnchor::Map {
                    reference_zoom: 20.0
                },
                NoiseAnchor::Viewport
            ]
        );
        for invalid in [
            r#"{"type":"noise","size":4,"dev.sargunv.orbis:anchor":"globe"}"#,
            r#"{"type":"noise","size":4,"dev.sargunv.orbis:reference-zoom":12}"#,
            r#"{"type":"noise","size":4,"dev.sargunv.orbis:anchor":"viewport",
                "dev.sargunv.orbis:reference-zoom":12}"#,
            r#"{"type":"noise","size":4,"dev.sargunv.orbis:anchor":"map",
                "dev.sargunv.orbis:reference-zoom":30}"#,
        ] {
            assert!(operations(invalid).is_err(), "{invalid}");
        }
    }

    /// Each operation takes its own keys: another is a warning, a missing
    /// required one an error that drops the layer.
    #[test]
    fn operations_take_their_own_keys() {
        let style = operations(r#"{"type":"blur","size":2,"amount":1}"#).unwrap();
        let paths: Vec<&str> = style.warnings.iter().map(|w| w.path.as_str()).collect();
        assert_eq!(paths, ["layers[0].dev.sargunv.orbis:operations[0].amount"]);
        for (missing, key) in [
            (r#"{"type":"blur"}"#, "size"),
            (r#"{"type":"convolve"}"#, "kernel"),
            (r#"{"type":"combine","mode":"max"}"#, "target"),
            (r#"{"type":"ramp"}"#, "stops"),
            (r#"{"type":"offset"}"#, "translate"),
        ] {
            let err = operations(missing).unwrap_err();
            let message = format!("missing required property \"{key}\"");
            assert!(
                err.errors.iter().any(|e| e.message == message),
                "{missing}: {err}"
            );
        }
        // Every key is checked, whichever fails first.
        let err = operations(r#"{"type":"combine","mode":"bogus"}"#).unwrap_err();
        assert_eq!(err.errors.len(), 2, "{err}");
        // A kind that isn't one is its only fault: the keys aren't read as
        // another kind's.
        let err =
            operations(r#"{"type":"gradient","kind":"conic","radius":2,"length":-1}"#).unwrap_err();
        assert_eq!((err.errors.len(), err.warnings.len()), (1, 0), "{err:?}");
        let err = operations(r#"{"type":"smudge"}"#).unwrap_err();
        assert!(
            (err.errors[0].message).starts_with("expected one of [select, blur, noise,"),
            "{err}"
        );
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
            operations_of(&style.layers[0])[..],
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

    /// An image-process layer's operations.
    fn operations_of(layer: &Layer) -> &Operations {
        let Some(TargetRead::Process { operations, .. }) = &layer.reads_target else {
            panic!("{:?} processes no target", layer.id);
        };
        operations
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
        ] = &operations_of(&style.layers[0])[..]
        else {
            panic!("{:?}", operations_of(&style.layers[0]));
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
                &operations_of(&style.layers[0])[..],
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
            operations_of(&style.layers[0])
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
        let ops = operations_of(&style.layers[0]);
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
        assert!(ops.reads_camera());
        // Every value that may follow the camera is watched for it.
        for (op, reads) in [
            (r#"{"type":"gradient","angle":["*",-1,["bearing"]]}"#, true),
            (
                r#"{"type":"gradient","kind":"radial","radius":["*",2,["pitch"]]}"#,
                true,
            ),
            (
                r#"{"type":"noise","size":1,"speed":["*",2,["pitch"]]}"#,
                true,
            ),
            (r#"{"type":"split","blue":["literal",[0,0]]}"#, false),
            (
                r#"{"type":"blur","size":["interpolate",["linear"],["zoom"],10,2,14,6]}"#,
                false,
            ),
        ] {
            let style = operations(op).unwrap();
            assert_eq!(
                operations_of(&style.layers[0]).reads_camera(),
                reads,
                "{op}"
            );
        }
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
        let op = &operations_of(&style.layers[0])[0];
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
        assert!(operations_of(&style.layers[0]).reads_camera());
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
        assert!(line.reads_target.is_none());
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

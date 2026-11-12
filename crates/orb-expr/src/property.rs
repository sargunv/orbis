//! Property values: parsing against a property's spec, and classification
//! into GL JS's `PropertyValue` kinds.

use crate::eval::interpolation_factor;
use crate::ir::{Call, Deps, Interpolation, Ir, NodeId, Op};
use crate::json::Json;
use crate::legacy::lower_function;
use crate::parse::{Annotation, Parser, is_expression};
use crate::scale::CameraScale;
use crate::value::Type;
use crate::{Diagnostic, Expression};

/// A property's value type, as the v8 spec declares it.
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyType {
    Number,
    String,
    Boolean,
    Color,
    ResolvedImage,
    Formatted,
    ProjectionDefinition,
    Padding,
    NumberArray,
    ColorArray,
    VariableAnchorOffsetCollection,

    /// A string from a fixed set of values.
    Enum(&'static [&'static str]),
    /// An array with an item type and an optional fixed length.
    Array(Type, Option<u32>),
}

/// What the v8 spec says about one property that matters for parsing it.
#[derive(Clone, Debug)]
pub struct PropertySpec {
    pub ty: PropertyType,
    /// The spec default as JSON text, which legacy categorical functions
    /// fall back to.
    pub default: Option<std::borrow::Cow<'static, str>>,
    /// `zoom` is among the expression parameters.
    pub zoom: bool,
    /// The property is data-driven: it may depend on the feature.
    pub feature: bool,
    /// It may read `feature-state` (data-driven paint, as the spec's
    /// validator has it; not layout).
    pub state: bool,
    /// It may read the camera beyond its zoom (`pitch`, `bearing`), for a
    /// value the same for every feature: Orbis's own extension.
    pub camera: bool,
    /// It may read the camera and feature data in one value, as a
    /// per-feature value scaled by one of the camera
    /// ([`CameraScale`]): Orbis's own.
    pub camera_data: bool,
    /// It reads `["elevation"]` per point, which varies within a feature
    /// as its data does between them: a colour relief's ramp, an
    /// extrusion's colour.
    pub elevation: bool,
    /// The property may use `interpolate`.
    pub interpolated: bool,
    /// Strings expand `{token}`s: legacy string stops (`convertFunction`)
    /// and plain values (GL JS `resolveTokens` on non-expression values).
    pub tokens: bool,
}

impl PropertySpec {
    /// The type an expression for this property must produce.
    pub(crate) fn expected_type(&self) -> Type {
        match &self.ty {
            PropertyType::Number => Type::Number,
            PropertyType::String | PropertyType::Enum(_) => Type::String,
            PropertyType::Boolean => Type::Boolean,
            PropertyType::Color => Type::Color,
            PropertyType::ResolvedImage => Type::ResolvedImage,
            PropertyType::Formatted => Type::Formatted,
            PropertyType::ProjectionDefinition => Type::ProjectionDefinition,
            PropertyType::Padding => Type::Padding,
            PropertyType::NumberArray => Type::NumberArray,
            PropertyType::ColorArray => Type::ColorArray,
            PropertyType::VariableAnchorOffsetCollection => Type::VariableAnchorOffsetCollection,

            PropertyType::Array(item, len) => Type::array(item.clone(), *len),
        }
    }
}

/// How a property value varies (GL JS's `StylePropertyExpression` kinds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// The same everywhere.
    Constant,
    /// Varies with zoom only, through a zoom curve.
    Camera,
    /// Varies with the feature only (or the point on it, for a property
    /// that reads `["elevation"]`).
    Source,
    /// Varies with zoom and the feature (or the point), through a
    /// top-level zoom curve.
    Composite,
}

/// The top-level `step` or `interpolate` on `["zoom"]` of a camera or
/// composite value.
#[derive(Clone, Debug, PartialEq)]
pub struct ZoomCurve {
    /// The curve's input stops.
    pub stops: Vec<f64>,
    /// `None` for a `step` curve.
    pub interpolation: Option<Interpolation>,
}

/// A parsed and classified property value.
#[derive(Clone, Debug)]
pub struct PropertyExpression {
    /// For a value with a camera scale, its feature part.
    expression: Expression,
    kind: Kind,
    curve: Option<ZoomCurve>,
    scale: Option<Box<CameraScale>>,
}

const TOP_LEVEL: &str = "\"zoom\" expression may only be used as input to a top-level \"step\" or \"interpolate\" expression.";
const ONLY_ONE: &str =
    "Only one zoom-based \"step\" or \"interpolate\" subexpression may be used in an expression.";

type CurveResult = Result<Option<NodeId>, &'static str>;

/// `findZoomCurve`: the zoom curve, which may sit under `coalesce`s but no
/// other operator.
fn find_zoom_curve(ir: &Ir, id: NodeId, memo: &mut [Option<CurveResult>]) -> CurveResult {
    if let Some(result) = memo[id as usize] {
        return result;
    }
    let node = ir.node(id);
    let mut result: CurveResult = match &node.op {
        Op::Call(crate::ir::Call::Let, args) => {
            find_zoom_curve(ir, *args.last().expect("let result"), memo)
        }
        Op::Coalesce(args) => {
            let mut found = Ok(None);
            for &arg in args {
                found = find_zoom_curve(ir, arg, memo);
                if found != Ok(None) {
                    break;
                }
            }
            found
        }
        Op::Step { input, .. } | Op::Interpolate { input, .. }
            if matches!(ir.node(*input).op, Op::Zoom) =>
        {
            Ok(Some(id))
        }
        _ => Ok(None),
    };
    if result.is_ok() {
        node.op
            .for_each_child(|child| match find_zoom_curve(ir, child, memo) {
                Err(e) => result = Err(e),
                Ok(Some(curve)) => match result {
                    Ok(None) => result = Err(TOP_LEVEL),
                    Ok(Some(r)) if r == curve => {}
                    _ => result = Err(ONLY_ONE),
                },
                Ok(None) => {}
            });
    }
    memo[id as usize] = Some(result);
    result
}

/// `ir` with its camera reads resolved to `camera`'s.
pub(crate) fn fold_camera(ir: &crate::ir::Ir, camera: crate::CameraInputs) -> crate::ir::Ir {
    let mut ir = ir.clone();
    for node in &mut ir.nodes {
        let value = match node.op {
            Op::Call(Call::Pitch, _) => camera.pitch,
            Op::Call(Call::Bearing, _) => camera.bearing,
            _ => continue,
        };
        node.op = Op::Literal(crate::Value::Number(value));
    }
    ir.recount_deps();
    ir
}

impl PropertyExpression {
    /// Parses a property value: a constant, an expression or a legacy
    /// function, and classifies it (GL JS's `createPropertyExpression`, with
    /// `convertFunction` for legacy functions). Diagnostic paths are relative
    /// to the value.
    pub fn parse(json: &Json, spec: &PropertySpec) -> Result<Self, Vec<Diagnostic>> {
        Self::parse_inner(json, spec, None)
    }

    /// Parses with the style's global state resolved:
    /// `global-state` reads fold to `state`'s values, so a value that reads
    /// only global state classifies as a constant.
    pub fn parse_with_state(
        json: &Json,
        spec: &PropertySpec,
        state: &[(String, crate::Value<'static>)],
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::parse_inner(json, spec, Some(state))
    }

    fn parse_inner(
        json: &Json,
        spec: &PropertySpec,
        state: Option<&[(String, crate::Value<'static>)]>,
    ) -> Result<Self, Vec<Diagnostic>> {
        let expected = spec.expected_type();
        // String properties coerce rather than assert at the top level.
        let annotation = (spec.ty == PropertyType::String).then_some(Annotation::Coerce);
        let mut parser = Parser {
            global_state: state,
            ..Parser::default()
        };
        let root = if json.as_object().is_some() {
            let lowered = lower_function(json, spec)?;
            parser.parse(&lowered, "", Some(&expected), annotation)
        } else if is_expression(json) {
            parser.parse(json, "", Some(&expected), annotation)
        } else if let Some(text) = json.as_str().filter(|_| spec.tokens) {
            match crate::legacy::convert_token_string(json.span, text) {
                tokens if tokens.as_str().is_none() => {
                    parser.parse(&tokens, "", Some(&expected), annotation)
                }
                _ => parser.parse_constant(json, Some(&expected), annotation),
            }
        } else {
            parser.parse_constant(json, Some(&expected), annotation)
        };
        let Some(root) = root else {
            return Err(parser.errors);
        };
        let expression = Expression {
            ir: parser.ir.compact(root),
        };
        Self::classify(expression, spec).map_err(|message| {
            vec![Diagnostic {
                path: String::new(),
                span: json.span,
                message: message.into(),
            }]
        })
    }

    pub(crate) fn classify(
        expression: Expression,
        spec: &PropertySpec,
    ) -> Result<Self, &'static str> {
        let ir = &expression.ir;
        let deps = ir.node(ir.root()).deps;
        let feature_constant = !deps.contains(Deps::FEATURE);
        if !feature_constant && !spec.feature {
            return Err("data expressions not supported");
        }
        if deps.contains(Deps::FEATURE_STATE) && !spec.state {
            return Err(
                "\"feature-state\" data expressions are not supported with layout properties.",
            );
        }
        let zoom_constant = !deps.contains(Deps::ZOOM);
        if !zoom_constant && !spec.zoom {
            return Err("zoom expressions not supported");
        }
        let camera = deps.contains(Deps::CAMERA);
        if camera && !spec.camera {
            return Err("camera expressions not supported");
        }
        if camera && !feature_constant && !spec.camera_data {
            return Err("camera expressions cannot read the feature");
        }
        let elevation = spec.elevation && deps.contains(Deps::ELEVATION);
        if camera && elevation {
            return Err("camera expressions cannot read the elevation");
        }
        let mut memo = vec![None; ir.nodes.len()];
        let found = find_zoom_curve(ir, ir.root(), &mut memo)?;
        if camera && !feature_constant {
            // The zoom reads as in any value; then the feature part draws
            // per feature and the camera's scales it.
            if found.is_none() && !zoom_constant {
                return Err(TOP_LEVEL);
            }
            let (feature, scale) = crate::scale::split(ir)?;
            let mut value = Self::classify(Expression { ir: feature }, spec)?;
            value.scale = Some(Box::new(scale));
            return Ok(value);
        }
        let curve = match found {
            None if !zoom_constant => return Err(TOP_LEVEL),
            None => None,
            Some(id) => Some(match &ir.node(id).op {
                Op::Interpolate {
                    interpolation,
                    labels,
                    ..
                } => {
                    if !spec.interpolated {
                        return Err(
                            "\"interpolate\" expressions cannot be used with this property",
                        );
                    }
                    ZoomCurve {
                        stops: labels.clone(),
                        interpolation: Some(*interpolation),
                    }
                }
                Op::Step { labels, .. } => ZoomCurve {
                    stops: labels.clone(),
                    interpolation: None,
                },
                _ => unreachable!("a zoom curve is a step or interpolate"),
            }),
        };
        let kind = match (&curve, feature_constant && !elevation) {
            (None, true) if camera => Kind::Camera,
            (None, true) => Kind::Constant,
            (None, false) => Kind::Source,
            (Some(_), true) => Kind::Camera,
            (Some(_), false) => Kind::Composite,
        };
        Ok(Self {
            expression,
            kind,
            curve,
            scale: None,
        })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub(crate) fn from_parts(expression: Expression, kind: Kind, curve: Option<ZoomCurve>) -> Self {
        Self {
            expression,
            kind,
            curve,
            scale: None,
        }
    }

    /// Whether the value reads the camera beyond its zoom: `pitch` or
    /// `bearing`.
    pub fn reads_camera(&self) -> bool {
        self.scale.is_some() || self.expression.deps().contains(Deps::CAMERA)
    }

    /// The camera scale of a per-feature value that reads the camera:
    /// its rows hold [`expression`](Self::expression), the feature part.
    pub fn camera_scale(&self) -> Option<&CameraScale> {
        self.scale.as_deref()
    }

    /// The value with its camera reads resolved to `camera`'s: a camera
    /// value over its zoom curve, or a constant without one, or a
    /// per-feature value with its camera scale resolved. `None` when it
    /// reads no camera.
    pub fn at_camera(&self, camera: crate::CameraInputs) -> Option<Self> {
        if let Some(scale) = &self.scale {
            return Some(Self {
                scale: Some(Box::new(scale.at_camera(camera))),
                ..self.clone()
            });
        }
        if !self.reads_camera() {
            return None;
        }
        Some(Self {
            expression: Expression {
                ir: fold_camera(&self.expression.ir, camera),
            },
            kind: if self.curve.is_some() {
                Kind::Camera
            } else {
                Kind::Constant
            },
            curve: self.curve.clone(),
            scale: None,
        })
    }

    /// Whether the value reads `feature-state`.
    pub fn reads_feature_state(&self) -> bool {
        self.expression.deps().contains(Deps::FEATURE_STATE)
    }

    /// Whether the value reads the feature's properties, id or geometry,
    /// as opposed to only its state.
    pub fn reads_feature_properties(&self) -> bool {
        self.expression.deps().contains(Deps::PROPERTIES)
    }

    pub fn expression(&self) -> &Expression {
        &self.expression
    }

    /// The zoom curve of a camera or composite value.
    pub fn zoom_curve(&self) -> Option<&ZoomCurve> {
        self.curve.as_ref()
    }

    /// How far `zoom` is between the curve stops `lower` and `upper`, as
    /// GL JS's `interpolationFactor`; 0 for `step` curves and values without
    /// a curve.
    pub fn interpolation_factor(&self, zoom: f64, lower: f64, upper: f64) -> f64 {
        match self.curve.as_ref().and_then(|c| c.interpolation) {
            Some(interpolation) => interpolation_factor(interpolation, zoom, lower, upper),
            None => 0.0,
        }
    }
}

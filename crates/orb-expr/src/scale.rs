//! A value that reads both the camera and the feature, as one the
//! renderer draws per feature and scales per frame: `clamp(feature ×
//! factor, min, max)`, where the feature part reads no camera and the
//! factor and bounds read no feature (Orbis's own). Property rows hold
//! the feature part, laid out with the tile as any per-feature value; the
//! factor and bounds follow the camera as feature-independent values do.
//!
//! `["*", ["get", "width"], ["/", ["pitch"], 60]]` splits into
//! `["get", "width"]` and `pitch / 60`. So do
//! products and quotients of such values, a `step`, `interpolate`,
//! `match` or `case` that the camera chooses among outputs sharing one
//! feature part (constant multiples of it, or 0), or that the feature
//! chooses among outputs sharing one camera part, and `min` and `max`
//! around the whole with bounds of the camera.

use crate::ir::{Deps, Ir, NodeId, Op};
use crate::value::{Type, Value};
use crate::{CameraInputs, EvaluationContext, Expression};

/// What a per-feature value's camera reads come to.
#[derive(Clone, Debug)]
pub struct CameraScale {
    factor: Expression,
    min: Option<Expression>,
    max: Option<Expression>,
}

/// A [`CameraScale`] at one camera: a value is `clamp(feature × factor,
/// min, max)`, the bounds infinite where it has none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    pub factor: f64,
    pub min: f64,
    pub max: f64,
}

impl Scale {
    /// What leaves a value as it is.
    pub const IDENTITY: Self = Self {
        factor: 1.0,
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
    };

    /// The value of a feature whose feature part is `value`.
    pub fn apply(self, value: f64) -> f64 {
        (value * self.factor).max(self.min).min(self.max)
    }

    /// The largest magnitude a value takes where its feature part's is
    /// at most `magnitude`, as culling margins need.
    pub fn magnitude(self, magnitude: f64) -> f64 {
        let m = magnitude.abs() * self.factor.abs();
        self.apply_scaled(m).abs().max(self.apply_scaled(-m).abs())
    }

    fn apply_scaled(self, scaled: f64) -> f64 {
        scaled.max(self.min).min(self.max)
    }
}

impl CameraScale {
    /// The scale in `context` (its zoom and camera). Where the factor or
    /// a bound gives no number, the whole value fails for every feature:
    /// `default`, as any failed evaluation gives.
    pub fn at(&self, context: &EvaluationContext<'_>, default: f64) -> Scale {
        let number = |e: &Expression| match e.evaluate_with(context) {
            Ok(Value::Number(n)) if !n.is_nan() => Some(n),
            _ => None,
        };
        let bound = |e: &Option<Expression>, none: f64| e.as_ref().map_or(Some(none), number);
        match (
            number(&self.factor),
            bound(&self.min, f64::NEG_INFINITY),
            bound(&self.max, f64::INFINITY),
        ) {
            (Some(factor), Some(min), Some(max)) => Scale { factor, min, max },
            _ => Scale {
                factor: 0.0,
                min: default,
                max: default,
            },
        }
    }

    /// Its factor and bounds with their camera reads resolved to
    /// `camera`'s.
    pub(crate) fn at_camera(&self, camera: CameraInputs) -> Self {
        let fold = |e: &Expression| Expression {
            ir: crate::property::fold_camera(&e.ir, camera),
        };
        Self {
            factor: fold(&self.factor),
            min: self.min.as_ref().map(fold),
            max: self.max.as_ref().map(fold),
        }
    }
}

/// Why a value reading the camera and the feature can't be drawn.
pub(crate) const SHAPE: &str = "camera inputs draw with feature data as a value of the feature times or divided by one of the camera, chosen by either, within \"min\" and \"max\" of the camera";

/// Splits `ir`, which reads the camera and the feature, into its feature
/// part and its camera scale.
pub(crate) fn split(ir: &Ir) -> Result<(Ir, CameraScale), &'static str> {
    let mut s = Splitter { ir: ir.clone() };
    // `min` and `max` layers around the product, outermost first: whether
    // each is a `max`, and its bounds.
    let mut layers = Vec::new();
    let mut at = ir.root();
    loop {
        let node = s.ir.node(at);
        match &node.op {
            Op::Min(args) | Op::Max(args) if node.deps.contains(Deps::CAMERA) => {
                let is_max = matches!(node.op, Op::Max(_));
                let mut reading = (0..args.len()).filter(|&i| s.reads_feature(args[i]));
                let (Some(inner), None) = (reading.next(), reading.next()) else {
                    return Err(SHAPE);
                };
                let bounds: Vec<NodeId> = (args.iter().enumerate())
                    .filter(|&(i, _)| i != inner)
                    .map(|(_, &a)| a)
                    .collect();
                at = args[inner];
                layers.push((is_max, bounds));
            }
            Op::Assert(args) | Op::Coerce(args) if s.is_number_arg(args) => at = args[0],
            _ => break,
        }
    }
    let (feature, factor) = s.factor(at)?;
    let feature = feature.ok_or(SHAPE)?;
    // Inside out: `max(clamp(v, lo, hi), b)` is `clamp(v, max(lo, b),
    // max(hi, b))`, and `min(clamp(v, lo, hi), b)` is `clamp(v, lo,
    // min(hi, b))`.
    let (mut lo, mut hi): (Option<NodeId>, Option<NodeId>) = (None, None);
    for (is_max, bounds) in layers.into_iter().rev() {
        let b = match bounds[..] {
            [b] => b,
            _ if is_max => s.push(Op::Max(bounds)),
            _ => s.push(Op::Min(bounds)),
        };
        if is_max {
            lo = Some(lo.map_or(b, |lo| s.push(Op::Max(vec![lo, b]))));
            hi = hi.map(|hi| s.push(Op::Max(vec![hi, b])));
        } else {
            hi = Some(hi.map_or(b, |hi| s.push(Op::Min(vec![hi, b]))));
        }
    }
    let factor = factor.unwrap_or_else(|| s.literal(1.0));
    let expression = |root: NodeId| Expression {
        ir: s.ir.compact(root),
    };
    let scale = CameraScale {
        factor: expression(factor),
        min: lo.map(expression),
        max: hi.map(expression),
    };
    Ok((s.ir.compact(feature), scale))
}

/// A value's feature part and camera part, `None` for 1.
type Parts = (Option<NodeId>, Option<NodeId>);

struct Splitter {
    ir: Ir,
}

impl Splitter {
    fn reads_feature(&self, id: NodeId) -> bool {
        self.ir.node(id).deps.contains(Deps::FEATURE)
    }

    fn reads_camera(&self, id: NodeId) -> bool {
        self.ir.node(id).deps.contains(Deps::CAMERA)
    }

    /// One operand that is already a number: an assertion or coercion
    /// that can't fail.
    fn is_number_arg(&self, args: &[NodeId]) -> bool {
        matches!(args, [a] if self.ir.node(*a).ty == Type::Number)
    }

    fn push(&mut self, op: Op) -> NodeId {
        self.ir.push(op, Type::Number, "")
    }

    fn literal(&mut self, n: f64) -> NodeId {
        self.push(Op::Literal(Value::Number(n)))
    }

    fn product(&mut self, parts: Vec<NodeId>) -> Option<NodeId> {
        match parts[..] {
            [] => None,
            [one] => Some(one),
            _ => Some(self.push(Op::Product(parts))),
        }
    }

    fn quotient(&mut self, a: Option<NodeId>, b: Option<NodeId>) -> Option<NodeId> {
        match (a, b) {
            (a, None) => a,
            (a, Some(b)) => {
                let a = a.unwrap_or_else(|| self.literal(1.0));
                Some(self.push(Op::Div(a, b)))
            }
        }
    }

    /// `id` as its feature part times its camera part.
    fn factor(&mut self, id: NodeId) -> Result<Parts, &'static str> {
        if !self.reads_feature(id) {
            return Ok((None, Some(id)));
        }
        if !self.reads_camera(id) {
            return Ok((Some(id), None));
        }
        match self.ir.node(id).op.clone() {
            Op::Product(args) => {
                let (mut feature, mut camera) = (Vec::new(), Vec::new());
                for a in args {
                    let (f, c) = self.factor(a)?;
                    feature.extend(f);
                    camera.extend(c);
                }
                Ok((self.product(feature), self.product(camera)))
            }
            Op::Div(a, b) => {
                let ((fa, ca), (fb, cb)) = (self.factor(a)?, self.factor(b)?);
                Ok((self.quotient(fa, fb), self.quotient(ca, cb)))
            }
            Op::Neg(a) => {
                let (f, c) = self.factor(a)?;
                let minus = self.literal(-1.0);
                Ok((f, self.product(c.into_iter().chain([minus]).collect())))
            }
            Op::Assert(args) | Op::Coerce(args) if self.is_number_arg(&args) => {
                self.factor(args[0])
            }
            Op::Step {
                input,
                labels,
                outputs,
            } => self.choice(&[input], &outputs, |outputs| Op::Step {
                input,
                labels: labels.clone(),
                outputs,
            }),
            Op::Interpolate {
                color_space,
                interpolation,
                input,
                labels,
                outputs,
            } => self.choice(&[input], &outputs, |outputs| Op::Interpolate {
                color_space,
                interpolation,
                input,
                labels: labels.clone(),
                outputs,
            }),
            Op::Match {
                input,
                labels,
                mut outputs,
                otherwise,
            } => {
                outputs.push(otherwise);
                self.choice(&[input], &outputs, |mut outputs| {
                    let otherwise = outputs.pop().expect("an otherwise");
                    Op::Match {
                        input,
                        labels: labels.clone(),
                        outputs,
                        otherwise,
                    }
                })
            }
            Op::Case {
                branches,
                otherwise,
            } => {
                let tests: Vec<NodeId> = branches.iter().map(|b| b.0).collect();
                let outputs: Vec<NodeId> =
                    (branches.iter().map(|b| b.1)).chain([otherwise]).collect();
                self.choice(&tests, &outputs, |mut outputs| {
                    let otherwise = outputs.pop().expect("an otherwise");
                    Op::Case {
                        branches: tests.iter().copied().zip(outputs).collect(),
                        otherwise,
                    }
                })
            }
            _ => Err(SHAPE),
        }
    }

    /// A choice by `tests` among `outputs`: rebuilt by `choose` around new
    /// outputs. Chosen by the camera (or by neither), its outputs must
    /// share one feature part, and the choice among their camera parts is
    /// the factor; chosen by the feature, they must share one camera part,
    /// and the choice among their feature parts is the feature's.
    fn choice(
        &mut self,
        tests: &[NodeId],
        outputs: &[NodeId],
        choose: impl Fn(Vec<NodeId>) -> Op,
    ) -> Result<Parts, &'static str> {
        let by_camera = tests.iter().any(|&t| self.reads_camera(t));
        let by_feature = tests.iter().any(|&t| self.reads_feature(t));
        if by_camera && by_feature {
            return Err(SHAPE);
        }
        let parts = (outputs.iter())
            .map(|&o| self.factor(o))
            .collect::<Result<Vec<_>, _>>()?;
        if !by_feature && let Some((Some(feature), own)) = self.shared(outputs, &parts) {
            let factor = self.push(choose(own));
            return Ok((Some(feature), Some(factor)));
        }
        if !by_camera {
            let pairs: Vec<Parts> = parts.iter().map(|&(f, c)| (c, f)).collect();
            if let Some((camera, own)) = self.shared(outputs, &pairs) {
                let feature = self.push(choose(own));
                return Ok((Some(feature), camera));
            }
        }
        Err(SHAPE)
    }

    /// `outputs` as one shared part times a part of each one's own, from
    /// each output's `(shared, own)` parts: every shared part a constant
    /// multiple of one core or of its reciprocal, or a literal 0 output,
    /// which fits any. The shared part and the outputs' own parts (each
    /// times its multiple), or `None` where they share none.
    fn shared(
        &mut self,
        outputs: &[NodeId],
        parts: &[Parts],
    ) -> Option<(Option<NodeId>, Vec<NodeId>)> {
        let mut core: Option<(Option<NodeId>, bool)> = None;
        let mut own = Vec::with_capacity(outputs.len());
        for (&output, &(shared, part)) in outputs.iter().zip(parts) {
            if matches!(self.ir.node(output).op, Op::Literal(Value::Number(n)) if n == 0.0) {
                own.push(output);
                continue;
            }
            let (c, inverted, k) = self.peel(shared);
            match core {
                None => core = Some((c, inverted)),
                Some((other, i)) if i == inverted && self.same_part(other, c) => {}
                Some(_) => return None,
            }
            let k = (k != 1.0).then(|| self.literal(k));
            let part = self.product(part.into_iter().chain(k).collect());
            own.push(part.unwrap_or_else(|| self.literal(1.0)));
        }
        let shared = match core? {
            (Some(c), true) => {
                let one = self.literal(1.0);
                Some(self.push(Op::Div(one, c)))
            }
            (c, _) => c,
        };
        Some((shared, own))
    }

    /// A part as `k × core` or, inverted, `k / core`: `(core, inverted,
    /// k)`, the core `None` for 1 (a part of `None`, or a constant).
    fn peel(&self, part: Option<NodeId>) -> (Option<NodeId>, bool, f64) {
        let number = |id: NodeId| match self.ir.node(id).op {
            Op::Literal(Value::Number(n)) => Some(n),
            _ => None,
        };
        let Some(id) = part else {
            return (None, false, 1.0);
        };
        match &self.ir.node(id).op {
            Op::Literal(Value::Number(n)) => (None, false, *n),
            Op::Product(args) => {
                let k: f64 = args.iter().filter_map(|&a| number(a)).product();
                let mut rest = args.iter().filter(|&&a| number(a).is_none());
                match (rest.next(), rest.next()) {
                    (None, _) => (None, false, k),
                    (Some(&inner), None) => {
                        let (core, inverted, j) = self.peel(Some(inner));
                        (core, inverted, k * j)
                    }
                    _ => (Some(id), false, 1.0),
                }
            }
            Op::Div(a, b) => match (number(*a), number(*b)) {
                (_, Some(d)) if d != 0.0 => {
                    let (core, inverted, j) = self.peel(Some(*a));
                    (core, inverted, j / d)
                }
                (Some(n), None) => match self.peel(Some(*b)) {
                    (core @ Some(_), inverted, j) if j != 0.0 => (core, !inverted, n / j),
                    _ => (Some(id), false, 1.0),
                },
                _ => (Some(id), false, 1.0),
            },
            Op::Neg(a) => {
                let (core, inverted, j) = self.peel(Some(*a));
                (core, inverted, -j)
            }
            _ => (Some(id), false, 1.0),
        }
    }

    fn same_part(&self, a: Option<NodeId>, b: Option<NodeId>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => self.same(a, b),
            _ => false,
        }
    }

    /// Whether two nodes compute alike: the same operators over the same
    /// operands.
    fn same(&self, a: NodeId, b: NodeId) -> bool {
        if a == b {
            return true;
        }
        let (na, nb) = (self.ir.node(a), self.ir.node(b));
        let own = |op: &Op| {
            let mut op = op.clone();
            op.map_children(|_| 0);
            format!("{op:?}")
        };
        if na.ty != nb.ty || own(&na.op) != own(&nb.op) {
            return false;
        }
        let (mut ca, mut cb) = (Vec::new(), Vec::new());
        na.op.for_each_child(|c| ca.push(c));
        nb.op.for_each_child(|c| cb.push(c));
        ca.len() == cb.len() && ca.iter().zip(&cb).all(|(&x, &y)| self.same(x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::Scale;
    use crate::{EvaluationContext, Kind, PropertyExpression, PropertySpec, PropertyType, Value};

    fn width(text: &str) -> Result<PropertyExpression, String> {
        let spec = PropertySpec {
            ty: PropertyType::Number,
            default: Some("1".into()),
            zoom: true,
            feature: true,
            state: true,
            camera: true,
            camera_data: true,
            elevation: false,
            interpolated: true,
            tokens: false,
        };
        PropertyExpression::parse(&crate::json::parse(text).unwrap(), &spec)
            .map_err(|e| e[0].message.clone())
    }

    fn camera(zoom: f64, pitch: f64, bearing: f64) -> EvaluationContext<'static> {
        EvaluationContext {
            zoom,
            pitch,
            bearing,
            ..EvaluationContext::default()
        }
    }

    fn feature(properties: &str) -> crate::json::Json {
        crate::json::parse(&format!(r#"{{"properties": {properties}}}"#)).unwrap()
    }

    /// The feature part at a feature with `properties`, and the scale at
    /// `context`.
    fn split(text: &str, properties: &str, context: &EvaluationContext<'_>) -> (f64, Scale) {
        let value = width(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        let feature = feature(properties);
        let part = match value.expression().evaluate_geojson(context, &feature) {
            Ok(Value::Number(n)) => n,
            other => panic!("{text}: {other:?}"),
        };
        let scale = value.camera_scale().expect("a camera scale");
        (part, scale.at(context, 1.0))
    }

    /// Splits `text` and checks that the parts make the whole at a
    /// feature with `properties` under `context`.
    fn agrees(text: &str, properties: &str, context: &EvaluationContext<'_>) {
        let (part, scale) = split(text, properties, context);
        let number = crate::Type::Number;
        let whole = crate::Expression::parse(&crate::json::parse(text).unwrap(), Some(&number));
        let whole = whole.unwrap();
        let feature = feature(properties);
        let Ok(Value::Number(expected)) = whole.evaluate_geojson(context, &feature) else {
            panic!("{text} gives no number");
        };
        let got = scale.apply(part);
        assert!((got - expected).abs() < 1e-9, "{text}: {got} != {expected}");
    }

    #[test]
    fn a_feature_value_over_the_bearing() {
        let text = r#"["/", ["get", "w"], ["bearing"]]"#;
        let value = width(text).unwrap();
        assert_eq!(value.kind(), Kind::Source);
        assert!(value.reads_camera());
        let (part, scale) = split(text, r#"{"w": 12}"#, &camera(14.0, 0.0, 4.0));
        assert_eq!((part, scale.factor), (12.0, 0.25));
        assert_eq!(scale.apply(part), 3.0);
    }

    #[test]
    fn parts_make_the_whole() {
        let at = [camera(13.5, 50.0, 2.5), camera(3.0, 10.0, 0.25)];
        for text in [
            r#"["*", 2, ["get", "w"], ["pitch"]]"#,
            r#"["/", ["*", ["get", "w"], 3], ["*", ["pitch"], ["bearing"]]]"#,
            r#"["-", ["/", ["get", "w"], ["bearing"]]]"#,
            r#"["/", ["get", "w"], ["*", ["get", "k"], ["pitch"]]]"#,
            r#"["interpolate", ["linear"], ["pitch"], 0, 0, 60, ["*", 2, ["get", "w"]]]"#,
            r#"["interpolate", ["linear"], ["pitch"], 0, ["get", "w"], 60, ["/", ["get", "w"], 4]]"#,
            r#"["step", ["pitch"], ["get", "w"], 30, ["-", ["get", "w"]]]"#,
            r#"["case", [">", ["pitch"], 20], ["*", ["get", "w"], 2], ["get", "w"]]"#,
            r#"["match", ["get", "k"], 2, ["/", ["get", "w"], ["pitch"]], ["/", 5, ["pitch"]]]"#,
            r#"["interpolate", ["linear"], ["zoom"], 3, ["/", ["get", "w"], ["bearing"]], 15, ["/", ["*", 4, ["get", "w"]], ["bearing"]]]"#,
            r#"["interpolate", ["linear"], ["zoom"], 3, ["/", ["get", "w"], ["pitch"]], 15, ["/", ["get", "k"], ["pitch"]]]"#,
            r#"["max", 1, ["/", ["get", "w"], ["bearing"]]]"#,
            r#"["min", ["max", ["/", ["get", "w"], ["bearing"]], 1, ["/", 2, ["pitch"]]], 20]"#,
            r#"["max", ["min", ["*", ["get", "w"], ["pitch"]], 5], 8]"#,
            r#"["max", ["get", "w"], ["/", 1, ["bearing"]]]"#,
        ] {
            for properties in [r#"{"w": 12, "k": 2}"#, r#"{"w": 0.5, "k": 3}"#] {
                for context in &at {
                    agrees(text, properties, context);
                }
            }
        }
    }

    #[test]
    fn other_shapes_are_rejected() {
        for text in [
            r#"["+", ["get", "w"], ["pitch"]]"#,
            r#"["interpolate", ["linear"], ["pitch"], 0, ["get", "w"], 60, ["get", "k"]]"#,
            r#"["interpolate", ["linear"], ["get", "w"], 0, 1, 60, ["pitch"]]"#,
            r#"["max", ["get", "k"], ["*", ["get", "w"], ["pitch"]]]"#,
            r#"["^", ["get", "w"], ["pitch"]]"#,
            r#"["case", [">", ["pitch"], ["get", "w"]], 1, 2]"#,
        ] {
            assert_eq!(width(text).err().as_deref(), Some(super::SHAPE), "{text}");
        }
        // The zoom reads only in a top-level curve, as anywhere.
        let nested =
            r#"["/", ["interpolate", ["linear"], ["zoom"], 0, ["get", "w"], 10, 1], ["pitch"]]"#;
        assert!(width(nested).is_err());
    }

    #[test]
    fn a_factor_that_fails_gives_the_default() {
        let text = r#"["/", ["get", "w"], ["number", ["at", ["case", [">", ["pitch"], 5], 1, 0], ["literal", [2, "x"]]]]]"#;
        let value = width(text).unwrap();
        let scale = value.camera_scale().unwrap();
        assert_eq!(scale.at(&camera(1.0, 10.0, 1.0), 7.0).apply(123.0), 7.0);
        assert_eq!(scale.at(&camera(1.0, 0.0, 1.0), 7.0).apply(123.0), 61.5);
    }

    #[test]
    fn magnitudes_bound_both_signs() {
        let scale = Scale {
            factor: -2.0,
            min: -5.0,
            max: f64::INFINITY,
        };
        assert_eq!(scale.magnitude(3.0), 6.0);
        assert_eq!(Scale { min: -4.0, ..scale }.magnitude(3.0), 6.0);
        assert_eq!(Scale { max: 1.0, ..scale }.magnitude(3.0), 5.0);
    }
}

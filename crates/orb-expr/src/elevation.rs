//! A value over `["elevation"]` as a curve the renderer evaluates per
//! fragment: one `step` or `interpolate` reads the elevation, through a
//! chain of arithmetic with constants, and the value is known at each of
//! its stops.

use crate::ir::{Deps, Interpolation, MathFn, NodeId, Op};
use crate::property::{Kind, PropertyExpression};
use crate::{Expression, Value};

/// One step of what a curve's input does to the elevation `x`, with a
/// constant `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElevationOp {
    /// `x + c`.
    Add(f64),
    /// `x * c`.
    Mul(f64),
    /// `x - c`.
    Sub(f64),
    /// `c - x`.
    SubFrom(f64),
    /// `x / c`.
    Div(f64),
    /// `x % c`, with the sign of `x`.
    Rem(f64),
    Min(f64),
    Max(f64),
    Abs,
    Floor,
    Ceil,
    /// Halves away from zero.
    Round,
}

/// A value that reads `["elevation"]` only as the input of one curve.
#[derive(Clone, Debug)]
pub struct InputCurve {
    /// What the curve's input does to the elevation, innermost first.
    pub input: Vec<ElevationOp>,
    /// The curve's input stops; a `step`'s first is −∞.
    pub labels: Vec<f64>,
    /// `None` for a `step`.
    pub interpolation: Option<Interpolation>,
    /// The curve interpolates in RGB, so that the value between two stops
    /// is their mix.
    pub rgb: bool,
    value: PropertyExpression,
    /// The curve's node in `value`, and its input's.
    curve: NodeId,
    input_node: NodeId,
}

impl InputCurve {
    /// The value at stop `i`: the curve's output there, so that it reads
    /// the feature only if that output (or what is around the curve) does.
    pub fn stop(&self, i: usize) -> PropertyExpression {
        let mut ir = self.value.expression().ir.clone();
        let output = match &ir.node(self.curve).op {
            Op::Step { outputs, .. } | Op::Interpolate { outputs, .. } => outputs[i],
            _ => unreachable!("a curve"),
        };
        let curve = self.curve;
        for node in &mut ir.nodes {
            node.op
                .map_children(|c| if c == curve { output } else { c });
        }
        let root = match ir.root() {
            root if root == curve => output,
            root => root,
        };
        self.value.rebuilt(ir, root)
    }

    /// The value where the curve's input is `input`.
    pub fn at(&self, input: f64) -> PropertyExpression {
        let mut ir = self.value.expression().ir.clone();
        ir.nodes[self.input_node as usize].op = Op::Literal(Value::Number(input));
        let root = ir.root();
        self.value.rebuilt(ir, root)
    }
}

impl PropertyExpression {
    /// The value changed to `ir`, rooted at `root`, with its inputs counted
    /// again.
    fn rebuilt(&self, mut ir: crate::ir::Ir, root: NodeId) -> Self {
        ir.recount_deps();
        let ir = ir.compact(root);
        let feature = ir.node(ir.root()).deps.contains(Deps::FEATURE);
        let curve = self.zoom_curve().cloned();
        let kind = match (&curve, feature) {
            (None, false) => Kind::Constant,
            (None, true) => Kind::Source,
            (Some(_), false) => Kind::Camera,
            (Some(_), true) => Kind::Composite,
        };
        Self::from_parts(Expression { ir }, kind, curve)
    }

    /// The value where the daylight is `daylight` (1 the day side's).
    pub fn at_daylight(&self, daylight: f64) -> Self {
        let mut ir = self.expression().ir.clone();
        for node in &mut ir.nodes {
            if matches!(node.op, Op::Call(crate::ir::Call::Daylight, _)) {
                node.op = Op::Literal(Value::Number(daylight));
            }
        }
        let root = ir.root();
        self.rebuilt(ir, root)
    }

    /// Whether the value reads `["elevation"]`.
    pub fn reads_elevation(&self) -> bool {
        self.expression().deps().contains(Deps::ELEVATION)
    }

    /// The value as a curve over the elevation, or why it isn't one: it
    /// must read `["elevation"]` once, as the input of a `step` or
    /// `interpolate`, through `+`, `-`, `*`, `/`, `%`, `min` and `max` with
    /// constants, `abs`, `floor`, `ceil` and `round`.
    pub fn elevation_curve(&self) -> Result<InputCurve, &'static str> {
        const SHAPE: &str = "[\"elevation\"] draws per fragment once, as the input of a \"step\" or \"interpolate\", through arithmetic with constants";
        self.input_curve(Deps::ELEVATION, SHAPE)
    }

    /// Whether the value reads `["dev.sargunv.orbis:daylight"]`.
    pub fn reads_daylight(&self) -> bool {
        self.expression().deps().contains(Deps::DAYLIGHT)
    }

    /// The value as a curve over the daylight, or why it isn't one: it
    /// must read `["dev.sargunv.orbis:daylight"]` once, as the input of a
    /// `step` with one stop or a linear or exponential `interpolate` in
    /// RGB with at most two.
    pub fn daylight_curve(&self) -> Result<InputCurve, &'static str> {
        const SHAPE: &str = "[\"dev.sargunv.orbis:daylight\"] draws per label once, as the input of a \"step\" with one stop or a linear or exponential \"interpolate\" in RGB with at most two";
        let curve = self.input_curve(Deps::DAYLIGHT, SHAPE)?;
        let stops = curve.labels.len();
        let fits = match curve.interpolation {
            None => stops == 2,
            Some(Interpolation::Linear | Interpolation::Exponential(_)) => curve.rgb && stops <= 2,
            Some(Interpolation::CubicBezier(_)) => false,
        };
        if !curve.input.is_empty() || !fits {
            return Err(SHAPE);
        }
        Ok(curve)
    }

    /// The value as a curve over the per-point input `deps` names, or
    /// `shape` where it isn't one.
    fn input_curve(&self, deps: Deps, shape: &'static str) -> Result<InputCurve, &'static str> {
        let ir = &self.expression().ir;
        let reads = |id: NodeId| ir.node(id).deps.contains(deps);
        let reads_input = ir
            .nodes
            .iter()
            .filter(|n| matches!(&n.op, Op::Call(call, _) if call.input_deps() == Some(deps)));
        let mut curves = ir.nodes.iter().enumerate().filter(|(_, n)| match &n.op {
            Op::Step { input, .. } | Op::Interpolate { input, .. } => reads(*input),
            _ => false,
        });
        let (1, Some((curve, node)), None) = (reads_input.count(), curves.next(), curves.next())
        else {
            return Err(shape);
        };
        let (input, labels, interpolation, rgb) = match &node.op {
            Op::Step { input, labels, .. } => (*input, labels.clone(), None, true),
            Op::Interpolate {
                input,
                labels,
                interpolation,
                color_space,
                ..
            } => (
                *input,
                labels.clone(),
                Some(*interpolation),
                *color_space == crate::ir::ColorSpace::Rgb,
            ),
            _ => unreachable!("a curve"),
        };
        // Down the input to the one elevation read, each operator's other
        // operands constants.
        let constant = |id: NodeId| match ir.node(id).op {
            Op::Literal(Value::Number(n)) => Some(n),
            _ => None,
        };
        let mut ops = Vec::new();
        let mut at = input;
        loop {
            let op = &ir.node(at).op;
            let args = match op {
                Op::Call(call, _) if call.input_deps() == Some(deps) => break,
                Op::Sum(args) | Op::Product(args) | Op::Min(args) | Op::Max(args) => args.clone(),
                Op::Sub(a, b) | Op::Div(a, b) | Op::Rem(a, b) => vec![*a, *b],
                Op::Neg(a) | Op::Math(_, a) => vec![*a],
                // `["number", …]` and `["to-number", …]` of a number.
                Op::Assert(args) | Op::Coerce(args) if args.len() == 1 => {
                    at = args[0];
                    continue;
                }
                _ => return Err(shape),
            };
            let x = args.iter().position(|&a| reads(a)).ok_or(shape)?;
            let c: Vec<f64> = (args.iter().enumerate())
                .filter(|&(i, _)| i != x)
                .map(|(_, &a)| constant(a))
                .collect::<Option<_>>()
                .ok_or(shape)?;
            let first = c.first().copied().unwrap_or(0.0);
            ops.push(match (op, x) {
                (Op::Sum(_), _) => ElevationOp::Add(c.iter().sum()),
                (Op::Product(_), _) => ElevationOp::Mul(c.iter().product()),
                (Op::Min(_), _) => ElevationOp::Min(c.into_iter().fold(f64::INFINITY, f64::min)),
                (Op::Max(_), _) => {
                    ElevationOp::Max(c.into_iter().fold(f64::NEG_INFINITY, f64::max))
                }
                (Op::Sub(..), 0) => ElevationOp::Sub(first),
                (Op::Sub(..), _) => ElevationOp::SubFrom(first),
                (Op::Div(..), 0) => ElevationOp::Div(first),
                (Op::Rem(..), 0) => ElevationOp::Rem(first),
                (Op::Neg(_), _) => ElevationOp::Mul(-1.0),
                (Op::Math(MathFn::Abs, _), _) => ElevationOp::Abs,
                (Op::Math(MathFn::Floor, _), _) => ElevationOp::Floor,
                (Op::Math(MathFn::Ceil, _), _) => ElevationOp::Ceil,
                (Op::Math(MathFn::Round, _), _) => ElevationOp::Round,
                _ => return Err(shape),
            });
            at = args[x];
        }
        ops.reverse();
        Ok(InputCurve {
            input: ops,
            labels,
            interpolation,
            rgb,
            value: self.clone(),
            curve: curve as NodeId,
            input_node: input,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ElevationOp as E;
    use crate::{Kind, PropertyExpression, PropertySpec, PropertyType, Value};

    fn color(text: &str) -> PropertyExpression {
        let spec = PropertySpec {
            ty: PropertyType::Color,
            default: None,
            zoom: true,
            feature: true,
            state: true,
            camera: false,
            elevation: true,
            daylight: false,
            interpolated: true,
            tokens: false,
        };
        PropertyExpression::parse(&crate::json::parse(text).unwrap(), &spec).unwrap()
    }

    fn red(e: &PropertyExpression) -> f32 {
        match e.expression().evaluate(0.0).unwrap() {
            Value::Color(c) => c.premultiplied()[0],
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_value_over_the_elevation_varies_per_point() {
        // Not folded to its colour at the ground.
        let e = color(r#"["interpolate", ["linear"], ["elevation"], 0, "red", 10, "blue"]"#);
        assert_eq!(e.kind(), Kind::Source);
        assert!(e.reads_elevation());
        let e = color(
            r#"["interpolate", ["linear"], ["zoom"], 15, ["step", ["elevation"], "red", 3, "blue"], 16, "white"]"#,
        );
        assert_eq!(e.kind(), Kind::Composite);
        assert_eq!(color(r#""red""#).kind(), Kind::Constant);
    }

    #[test]
    fn a_curve_knows_its_value_at_each_stop() {
        let e = color(
            r#"["interpolate", ["exponential", 2], ["elevation"], 0, "red", 10, ["get", "c"]]"#,
        );
        let curve = e.elevation_curve().unwrap();
        assert_eq!(curve.input, []);
        assert_eq!(curve.labels, [0.0, 10.0]);
        assert_eq!(
            curve.interpolation,
            Some(crate::Interpolation::Exponential(2.0))
        );
        assert!(curve.rgb);
        let low = curve.stop(0);
        assert_eq!(low.kind(), Kind::Constant);
        assert_eq!(red(&low), 1.0);
        assert_eq!(curve.stop(1).kind(), Kind::Source);
        // Between stops, the curve itself.
        assert_eq!(curve.at(5.0).kind(), Kind::Source);
    }

    #[test]
    fn floors_read_the_elevation_through_arithmetic() {
        let e = color(r##"["step", ["%", ["elevation"], 3.5], "#d9d9d9", 3, "#a8a8a8"]"##);
        let curve = e.elevation_curve().unwrap();
        assert_eq!(curve.input, [E::Rem(3.5)]);
        assert_eq!(curve.labels, [f64::NEG_INFINITY, 3.0]);
        assert_eq!(curve.interpolation, None);
        assert_eq!(red(&curve.stop(0)), 0xd9 as f32 / 255.0);
        assert_eq!(red(&curve.stop(1)), 0xa8 as f32 / 255.0);
        assert_eq!(red(&curve.at(3.2)), 0xa8 as f32 / 255.0);
        // Innermost first, with the constant on either side.
        let e = color(
            r#"["step", ["floor", ["/", ["-", 100, ["*", ["elevation"], 2, 3]], 4]], "red", 3, "blue"]"#,
        );
        assert_eq!(
            e.elevation_curve().unwrap().input,
            [E::Mul(6.0), E::SubFrom(100.0), E::Div(4.0), E::Floor]
        );
    }

    #[test]
    fn other_reads_of_the_elevation_are_no_curve() {
        for text in [
            r#"["rgb", ["elevation"], 0, 0]"#,
            r#"["step", ["elevation"], "red", 3, ["step", ["elevation"], "blue", 6, "white"]]"#,
            r#"["case", [">", ["elevation"], 3], "red", "blue"]"#,
            r#"["step", ["*", ["elevation"], ["get", "k"]], "red", 3, "blue"]"#,
            r#"["step", ["+", ["elevation"], ["elevation"]], "red", 3, "blue"]"#,
            r#"["step", ["/", 10, ["elevation"]], "red", 3, "blue"]"#,
            r#"["step", ["sqrt", ["elevation"]], "red", 3, "blue"]"#,
        ] {
            assert!(color(text).elevation_curve().is_err(), "{text}");
        }
    }

    fn daylight(text: &str, allowed: bool) -> Result<PropertyExpression, Vec<crate::Diagnostic>> {
        let spec = PropertySpec {
            ty: PropertyType::Color,
            default: None,
            zoom: true,
            feature: true,
            state: true,
            camera: false,
            elevation: false,
            daylight: allowed,
            interpolated: true,
            tokens: false,
        };
        PropertyExpression::parse(&crate::json::parse(text).unwrap(), &spec)
    }

    /// The daylight is read per label, so a value over it varies as data
    /// does, and only where the property allows it.
    #[test]
    fn a_value_over_the_daylight_varies_per_label() {
        let text = r#"["interpolate", ["linear"], ["dev.sargunv.orbis:daylight"], 0, "white", 1, "black"]"#;
        let e = daylight(text, true).unwrap();
        assert_eq!(e.kind(), Kind::Source);
        assert!(e.reads_daylight());
        assert!(daylight(text, false).is_err());
        let curve = e.daylight_curve().unwrap();
        assert_eq!(curve.labels, [0.0, 1.0]);
        assert_eq!(curve.stop(0).kind(), Kind::Constant);
        assert_eq!(red(&curve.stop(0)), 1.0);
        // Within a zoom curve, with data at a stop.
        let e = daylight(
            r#"["interpolate", ["linear"], ["zoom"], 4, ["step", ["dev.sargunv.orbis:daylight"], "white", 0.5, ["get", "c"]], 8, "black"]"#,
            true,
        )
        .unwrap();
        assert_eq!(e.kind(), Kind::Composite);
        let curve = e.daylight_curve().unwrap();
        assert_eq!(curve.interpolation, None);
        assert_eq!(curve.stop(0).kind(), Kind::Camera);
        assert_eq!(curve.stop(1).kind(), Kind::Composite);
    }

    /// The renderer mixes two stops in RGB.
    #[test]
    fn other_reads_of_the_daylight_are_no_curve() {
        for text in [
            r#"["rgb", ["*", 255, ["dev.sargunv.orbis:daylight"]], 0, 0]"#,
            r#"["interpolate", ["linear"], ["dev.sargunv.orbis:daylight"], 0, "white", 0.5, "red", 1, "black"]"#,
            r#"["interpolate-hcl", ["linear"], ["dev.sargunv.orbis:daylight"], 0, "white", 1, "black"]"#,
            r#"["interpolate", ["linear"], ["*", 2, ["dev.sargunv.orbis:daylight"]], 0, "white", 1, "black"]"#,
            r#"["step", ["dev.sargunv.orbis:daylight"], "white", 0.3, "red", 0.6, "black"]"#,
            r#"["interpolate", ["cubic-bezier", 0, 0, 1, 1], ["dev.sargunv.orbis:daylight"], 0, "white", 1, "black"]"#,
        ] {
            let e = daylight(text, true).unwrap();
            assert!(e.daylight_curve().is_err(), "{text}");
        }
    }
}

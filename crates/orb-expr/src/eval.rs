//! A tree-walking evaluator over the IR, with GL JS's runtime semantics.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;

use crate::color::Color;
use crate::ir::{
    CmpOp, Interpolation, Ir, KeyId, LegacyFilter, MatchLabels, MathFn, Node, NodeId, Op, Subject,
};
use crate::js;
use crate::value::{Type, Value, check_subtype};

/// The feature an expression is evaluated against.
pub(crate) trait Feature<'a> {
    /// The property under a literal key of the expression, `None` if absent.
    fn get(&self, key: KeyId, name: &str) -> Option<Value<'a>>;
    /// The property under a computed key.
    fn get_by_name(&self, name: &str) -> Option<Value<'a>>;
    fn properties(&self) -> Vec<(String, Value<'static>)>;
    /// The id, `Null` if the feature has none.
    fn id(&self) -> Value<'a>;
    /// `Point`, `LineString`, `Polygon` (or a GeoJSON type name).
    fn geometry_type(&self) -> Option<&'a str>;
}

/// An evaluation failure. GL JS falls back to the property's default (or
/// `false` for a filter) when this happens.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalError {
    pub(crate) node: NodeId,
    pub(crate) kind: ErrorKind,
}

/// Errors on the common paths (a missing property under an assertion, a
/// type mismatch in a comparison) don't allocate.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ErrorKind {
    Expected {
        expected: Type,
        found: Type,
    },
    Order {
        op: CmpOp,
        lhs: &'static str,
        rhs: &'static str,
    },
    NotANumber,
    Message(String),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::Expected { expected, found } => {
                write!(
                    f,
                    "Expected value to be of type {expected}, but found {found} instead."
                )
            }
            ErrorKind::Order { op, lhs, rhs } => write!(
                f,
                "Expected arguments for \"{}\" to be (string, string) or (number, number), but found ({lhs}, {rhs}) instead.",
                op.name()
            ),
            ErrorKind::NotANumber => f.write_str("Input is not a number."),
            ErrorKind::Message(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for EvalError {}

pub(super) type Result<T> = std::result::Result<T, EvalError>;

/// One evaluation. Without a feature, properties are absent and the id and
/// geometry type `null`.
pub(crate) struct Evaluator<'c, 'a> {
    pub context: &'c crate::EvaluationContext<'a>,
    pub ir: &'a Ir,
    pub feature: Option<&'c dyn Feature<'a>>,
}

pub(super) fn error(node: NodeId, message: impl Into<String>) -> EvalError {
    EvalError {
        node,
        kind: ErrorKind::Message(message.into()),
    }
}

/// `v` is of type `ty`, as an assertion checks it.
fn is_a(v: &Value<'_>, ty: &Type) -> bool {
    match ty {
        Type::Value => true,
        Type::Array(..) => check_subtype(ty, &v.type_of()).is_ok(),
        t => t.kind() == v.kind(),
    }
}

/// `validateRGBA`.
fn validate_rgba(r: f64, g: f64, b: f64, a: Option<f64>) -> Option<String> {
    let channel = |c: f64| (0.0..=255.0).contains(&c);
    let list = |values: &[f64]| {
        values
            .iter()
            .map(|&v| js::number_to_string(v))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if !(channel(r) && channel(g) && channel(b)) {
        let values = match a {
            Some(a) => list(&[r, g, b, a]),
            None => list(&[r, g, b]),
        };
        return Some(format!(
            "Invalid rgba value [{values}]: 'r', 'g', and 'b' must be between 0 and 255."
        ));
    }
    if let Some(a) = a
        && !(0.0..=1.0).contains(&a)
    {
        return Some(format!(
            "Invalid rgba value [{}]: 'a' must be between 0 and 1.",
            list(&[r, g, b, a])
        ));
    }
    None
}

/// `findStopLessThanOrEqualTo`: the index of the last label `<= input`.
fn find_stop(labels: &[f64], input: f64, node: NodeId) -> Result<usize> {
    let last = labels.len() - 1;
    let (mut lo, mut hi) = (0isize, last as isize);
    while lo <= hi {
        let i = ((lo + hi) / 2) as usize;
        let value = labels[i];
        if value <= input {
            if i == last || input < labels[i + 1] {
                return Ok(i);
            }
            lo = i as isize + 1;
        } else if value > input {
            hi = i as isize - 1;
        } else {
            return Err(EvalError {
                node,
                kind: ErrorKind::NotANumber,
            });
        }
    }
    Ok(0)
}

/// GL JS's `interpolationFactor`: how far `input` is between the stops
/// `lower` and `upper`, shaped by the interpolation type.
pub fn interpolation_factor(
    interpolation: Interpolation,
    input: f64,
    lower: f64,
    upper: f64,
) -> f64 {
    match interpolation {
        Interpolation::Linear => exponential_factor(input, 1.0, lower, upper),
        Interpolation::Exponential(base) => exponential_factor(input, base, lower, upper),
        Interpolation::CubicBezier([x1, y1, x2, y2]) => {
            UnitBezier::new(x1, y1, x2, y2).solve(exponential_factor(input, 1.0, lower, upper))
        }
    }
}

fn exponential_factor(input: f64, base: f64, lower: f64, upper: f64) -> f64 {
    let difference = upper - lower;
    let progress = input - lower;
    if difference == 0.0 {
        0.0
    } else if base == 1.0 {
        progress / difference
    } else {
        (base.powf(progress) - 1.0) / (base.powf(difference) - 1.0)
    }
}

/// A port of `@mapbox/unitbezier`, as GL JS uses it.
struct UnitBezier {
    ax: f64,
    bx: f64,
    cx: f64,
    ay: f64,
    by: f64,
    cy: f64,
}

impl UnitBezier {
    fn new(p1x: f64, p1y: f64, p2x: f64, p2y: f64) -> Self {
        let cx = 3.0 * p1x;
        let bx = 3.0 * (p2x - p1x) - cx;
        let cy = 3.0 * p1y;
        let by = 3.0 * (p2y - p1y) - cy;
        Self {
            ax: 1.0 - cx - bx,
            bx,
            cx,
            ay: 1.0 - cy - by,
            by,
            cy,
        }
    }

    fn sample_x(&self, t: f64) -> f64 {
        ((self.ax * t + self.bx) * t + self.cx) * t
    }

    fn sample_y(&self, t: f64) -> f64 {
        ((self.ay * t + self.by) * t + self.cy) * t
    }

    fn sample_dx(&self, t: f64) -> f64 {
        (3.0 * self.ax * t + 2.0 * self.bx) * t + self.cx
    }

    fn solve_x(&self, x: f64) -> f64 {
        const EPSILON: f64 = 1e-6;
        if x < 0.0 {
            return 0.0;
        }
        if x > 1.0 {
            return 1.0;
        }
        let mut t = x;
        for _ in 0..8 {
            let x2 = self.sample_x(t) - x;
            if x2.abs() < EPSILON {
                return t;
            }
            let d2 = self.sample_dx(t);
            if d2.abs() < 1e-6 {
                break;
            }
            t -= x2 / d2;
        }
        let (mut t0, mut t1) = (0.0, 1.0);
        t = x;
        for _ in 0..20 {
            let x2 = self.sample_x(t);
            if (x2 - x).abs() < EPSILON {
                break;
            }
            if x > x2 {
                t0 = t;
            } else {
                t1 = t;
            }
            t = (t1 - t0) * 0.5 + t0;
        }
        t
    }

    fn solve(&self, x: f64) -> f64 {
        self.sample_y(self.solve_x(x))
    }
}

/// JavaScript's `typeof`, as legacy filters compare it.
fn js_typeof(v: Option<&Value<'_>>) -> &'static str {
    match v {
        None => "undefined",
        Some(Value::Bool(_)) => "boolean",
        Some(Value::Number(_)) => "number",
        Some(Value::String(_)) => "string",
        Some(_) => "object",
    }
}

/// `a op b` for an ordering `op` and two values of the same `typeof`, as
/// JavaScript compares them.
fn js_order(op: CmpOp, a: &Value<'_>, b: &Value<'_>) -> bool {
    let ordering = match (a, b) {
        (Value::Number(a), Value::Number(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(js::compare_strings(a, b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Null, Value::Null) => Some(Ordering::Equal),
        _ => None,
    };
    let Some(ordering) = ordering else {
        return false;
    };
    match op {
        CmpOp::Lt => ordering == Ordering::Less,
        CmpOp::Le => ordering != Ordering::Greater,
        CmpOp::Gt => ordering == Ordering::Greater,
        CmpOp::Ge => ordering != Ordering::Less,
        CmpOp::Eq | CmpOp::Ne => unreachable!("equality is not an ordering"),
    }
}

/// The item of a borrowed or owned array or object, keeping the borrow.
fn project<'a, T: Clone>(
    items: Cow<'a, [T]>,
    pick: impl Fn(&[T]) -> Option<&T>,
    value: impl Fn(&T) -> &Value<'static>,
) -> Option<Value<'a>> {
    match items {
        Cow::Borrowed(items) => pick(items).map(|v| value(v).reborrow()),
        Cow::Owned(items) => pick(&items).map(|v| value(v).clone()),
    }
}

impl<'a> Evaluator<'_, 'a> {
    pub(crate) fn eval_root(&self) -> Result<Value<'a>> {
        self.eval(self.ir.root())
    }

    pub(super) fn number(&self, id: NodeId) -> Result<f64> {
        match self.eval(id)? {
            Value::Number(n) => Ok(n),
            other => Err(self.mismatch(id, Type::Number, &other)),
        }
    }

    pub(super) fn boolean(&self, id: NodeId) -> Result<bool> {
        match self.eval(id)? {
            Value::Bool(b) => Ok(b),
            other => Err(self.mismatch(id, Type::Boolean, &other)),
        }
    }

    pub(super) fn string(&self, id: NodeId) -> Result<Cow<'a, str>> {
        match self.eval(id)? {
            Value::String(s) => Ok(s),
            other => Err(self.mismatch(id, Type::String, &other)),
        }
    }

    fn mismatch(&self, id: NodeId, expected: Type, found: &Value<'_>) -> EvalError {
        EvalError {
            node: id,
            kind: ErrorKind::Expected {
                expected,
                found: found.type_of(),
            },
        }
    }

    fn numbers<'s>(&'s self, ids: &'s [NodeId]) -> impl Iterator<Item = Result<f64>> + 's {
        ids.iter().map(|&id| self.number(id))
    }

    fn subject(&self, subject: Subject) -> Option<Value<'a>> {
        match subject {
            Subject::Property(key) => self
                .feature
                .as_ref()
                .and_then(|f| f.get(key, &self.ir.keys[key as usize])),
            Subject::Id => Some(self.feature.as_ref().map_or(Value::Null, |f| f.id())),
            Subject::Type => Some(
                self.feature
                    .as_ref()
                    .and_then(|f| f.geometry_type())
                    .map_or(Value::Null, |t| Value::String(Cow::Borrowed(t))),
            ),
        }
    }

    fn legacy(&self, filter: &LegacyFilter) -> bool {
        match filter {
            LegacyFilter::Compare { subject, op, value } => {
                let actual = self.subject(*subject);
                match op {
                    CmpOp::Eq => actual.is_some_and(|a| a.strict_eq(value)),
                    CmpOp::Ne => !actual.is_some_and(|a| a.strict_eq(value)),
                    _ => {
                        js_typeof(actual.as_ref()) == js_typeof(Some(value))
                            && actual.is_some_and(|a| js_order(*op, &a, value))
                    }
                }
            }
            LegacyFilter::In { subject, values } => self
                .subject(*subject)
                .is_some_and(|a| values.iter().any(|v| a.strict_eq(v))),
            LegacyFilter::Has(Subject::Id) => {
                !matches!(self.subject(Subject::Id), None | Some(Value::Null))
            }
            LegacyFilter::Has(subject) => self.subject(*subject).is_some(),
        }
    }

    pub(crate) fn eval(&self, id: NodeId) -> Result<Value<'a>> {
        // Each group of operators evaluates in its own function, which keeps
        // this recursive frame small in unoptimised builds.
        let node = self.ir.node(id);
        match &node.op {
            Op::Call(call, args) => self.eval_call(id, call, args),
            Op::Literal(v) => Ok(v.reborrow()),
            Op::Zoom => Ok(Value::Number(self.context.zoom)),
            Op::LineProgress => Ok(Value::Number(self.context.line_progress)),
            Op::Id
            | Op::GeometryType
            | Op::Properties
            | Op::Get(..)
            | Op::GetDynamic(..)
            | Op::Has(..)
            | Op::HasDynamic(..)
            | Op::GetFrom { .. }
            | Op::HasIn { .. } => self.eval_feature(&node.op),
            Op::Match { .. }
            | Op::Case { .. }
            | Op::Coalesce(..)
            | Op::Step { .. }
            | Op::Interpolate { .. } => self.eval_control(id, &node.op),
            Op::Compare { .. }
            | Op::All(..)
            | Op::Any(..)
            | Op::Not(..)
            | Op::In { .. }
            | Op::Sum(..)
            | Op::Product(..)
            | Op::Sub(..)
            | Op::Neg(..)
            | Op::Div(..)
            | Op::Rem(..)
            | Op::Pow(..)
            | Op::Min(..)
            | Op::Max(..)
            | Op::Math(..) => self.eval_logic(id, &node.op),
            Op::Coerce(..) | Op::Assert(..) | Op::Rgba { .. } | Op::ToRgba(..) => {
                self.eval_conversion(id, node)
            }
            _ => self.eval_other(id, &node.op),
        }
    }

    fn eval_feature(&self, op: &Op) -> Result<Value<'a>> {
        let feature = self.feature.as_ref();
        Ok(match op {
            Op::Id => feature.map_or(Value::Null, |f| f.id()),
            Op::GeometryType => feature
                .and_then(|f| f.geometry_type())
                .map_or(Value::Null, |t| Value::String(Cow::Borrowed(t))),
            Op::Properties => Value::Object(Cow::Owned(
                feature.map(|f| f.properties()).unwrap_or_default(),
            )),
            Op::Get(key) => feature
                .and_then(|f| f.get(*key, &self.ir.keys[*key as usize]))
                .unwrap_or(Value::Null),
            Op::GetDynamic(key) => {
                let key = self.string(*key)?;
                feature
                    .and_then(|f| f.get_by_name(&key))
                    .unwrap_or(Value::Null)
            }
            Op::Has(key) => Value::Bool(
                feature.is_some_and(|f| f.get(*key, &self.ir.keys[*key as usize]).is_some()),
            ),
            Op::HasDynamic(key) => {
                let key = self.string(*key)?;
                Value::Bool(feature.is_some_and(|f| f.get_by_name(&key).is_some()))
            }
            Op::GetFrom { key, object } => {
                let key = self.string(*key)?;
                self.member(*object, &key)?.unwrap_or(Value::Null)
            }
            Op::HasIn { key, object } => {
                let key = self.string(*key)?;
                Value::Bool(self.member(*object, &key)?.is_some())
            }
            _ => unreachable!("dispatched by eval"),
        })
    }

    fn eval_control(&self, id: NodeId, op: &Op) -> Result<Value<'a>> {
        Ok(match op {
            Op::Match {
                input,
                labels,
                outputs,
                otherwise,
            } => {
                let value = self.eval(*input)?;
                let index = match (labels, &value) {
                    (MatchLabels::Number(labels), Value::Number(n)) => labels
                        .binary_search_by(|(l, _)| l.partial_cmp(n).unwrap_or(Ordering::Less))
                        .ok()
                        .map(|i| labels[i].1),
                    (MatchLabels::String(labels), Value::String(s)) => labels
                        .binary_search_by(|(l, _)| (**l).cmp(&**s))
                        .ok()
                        .map(|i| labels[i].1),
                    _ => None,
                };
                match index {
                    Some(i) => self.eval(outputs[i as usize])?,
                    None => self.eval(*otherwise)?,
                }
            }
            Op::Case {
                branches,
                otherwise,
            } => {
                for &(test, output) in branches {
                    if self.boolean(test)? {
                        return self.eval(output);
                    }
                }
                self.eval(*otherwise)?
            }
            Op::Coalesce(args) => {
                let mut result = Value::Null;
                let mut requested_image = None;
                for (i, &arg) in args.iter().enumerate() {
                    result = self.eval(arg)?;
                    if let Value::Image {
                        name,
                        available: false,
                    } = result
                    {
                        requested_image.get_or_insert(name);
                        result = if i + 1 == args.len() {
                            Value::String(requested_image.take().expect("image name"))
                        } else {
                            Value::Null
                        };
                    }
                    if result != Value::Null {
                        break;
                    }
                }
                result
            }
            Op::Step {
                input,
                labels,
                outputs,
            } => {
                if labels.len() == 1 {
                    return self.eval(outputs[0]);
                }
                let value = self.number(*input)?;
                if value <= labels[0] {
                    return self.eval(outputs[0]);
                }
                if value >= labels[labels.len() - 1] {
                    return self.eval(outputs[labels.len() - 1]);
                }
                self.eval(outputs[find_stop(labels, value, id)?])?
            }
            Op::Interpolate {
                interpolation,
                color_space,
                input,
                labels,
                outputs,
            } => {
                if labels.len() == 1 {
                    return self.eval(outputs[0]);
                }
                let value = self.number(*input)?;
                if value <= labels[0] {
                    return self.eval(outputs[0]);
                }
                let last = labels.len() - 1;
                if value >= labels[last] {
                    return self.eval(outputs[last]);
                }
                let i = find_stop(labels, value, id)?;
                let t = interpolation_factor(*interpolation, value, labels[i], labels[i + 1]);
                let lower = self.eval(outputs[i])?;
                let upper = self.eval(outputs[i + 1])?;
                self.interpolate_typed(id, lower, upper, t, &self.ir.node(id).ty, *color_space)?
            }
            _ => unreachable!("dispatched by eval"),
        })
    }

    fn eval_logic(&self, id: NodeId, op: &Op) -> Result<Value<'a>> {
        Ok(match op {
            Op::Compare {
                op,
                lhs,
                rhs,
                untyped,
                collator,
            } => {
                let a = self.eval(*lhs)?;
                let b = self.eval(*rhs)?;
                if op.is_order() && *untyped {
                    let (lk, rk) = (a.kind(), b.kind());
                    if lk != rk || !(lk == "string" || lk == "number") {
                        return Err(EvalError {
                            node: id,
                            kind: ErrorKind::Order {
                                op: *op,
                                lhs: lk,
                                rhs: rk,
                            },
                        });
                    }
                }
                if let Some(c) = collator
                    && let (Value::String(a), Value::String(b), Value::Collator(c)) =
                        (&a, &b, self.eval(*c)?)
                {
                    let ordering = self.context.locale.compare(a, b, &c);
                    return Ok(Value::Bool(match op {
                        CmpOp::Eq => ordering == Ordering::Equal,
                        CmpOp::Ne => ordering != Ordering::Equal,
                        CmpOp::Lt => ordering == Ordering::Less,
                        CmpOp::Le => ordering != Ordering::Greater,
                        CmpOp::Gt => ordering == Ordering::Greater,
                        CmpOp::Ge => ordering != Ordering::Less,
                    }));
                }
                Value::Bool(match op {
                    CmpOp::Eq => a.strict_eq(&b),
                    CmpOp::Ne => !a.strict_eq(&b),
                    _ => js_order(*op, &a, &b),
                })
            }
            Op::All(args) => {
                for &arg in args {
                    if !self.boolean(arg)? {
                        return Ok(Value::Bool(false));
                    }
                }
                Value::Bool(true)
            }
            Op::Any(args) => {
                for &arg in args {
                    if self.boolean(arg)? {
                        return Ok(Value::Bool(true));
                    }
                }
                Value::Bool(false)
            }
            Op::Not(a) => Value::Bool(!self.boolean(*a)?),
            Op::In { needle, haystack } => self.contains(id, *needle, *haystack)?,
            Op::Sum(args) => Value::Number(
                self.numbers(args)
                    .try_fold(0.0, |acc, n| n.map(|n| acc + n))?,
            ),
            Op::Product(args) => Value::Number(
                self.numbers(args)
                    .try_fold(1.0, |acc, n| n.map(|n| acc * n))?,
            ),
            Op::Sub(a, b) => Value::Number(self.number(*a)? - self.number(*b)?),
            Op::Neg(a) => Value::Number(-self.number(*a)?),
            Op::Div(a, b) => Value::Number(self.number(*a)? / self.number(*b)?),
            Op::Rem(a, b) => Value::Number(self.number(*a)? % self.number(*b)?),
            Op::Pow(a, b) => {
                let (base, exp) = (self.number(*a)?, self.number(*b)?);
                // `Math.pow` differs from `powf` only here.
                let nan = exp.is_nan() || (base.abs() == 1.0 && exp.is_infinite());
                Value::Number(if nan { f64::NAN } else { base.powf(exp) })
            }
            Op::Min(args) => {
                Value::Number(self.numbers(args).try_fold(f64::INFINITY, |acc, n| {
                    n.map(|n| {
                        if n.is_nan() || acc.is_nan() {
                            f64::NAN
                        } else {
                            acc.min(n)
                        }
                    })
                })?)
            }
            Op::Max(args) => {
                Value::Number(self.numbers(args).try_fold(f64::NEG_INFINITY, |acc, n| {
                    n.map(|n| {
                        if n.is_nan() || acc.is_nan() {
                            f64::NAN
                        } else {
                            acc.max(n)
                        }
                    })
                })?)
            }
            Op::Math(f, a) => {
                let x = self.number(*a)?;
                Value::Number(match f {
                    MathFn::Abs => x.abs(),
                    MathFn::Floor => x.floor(),
                    MathFn::Ceil => x.ceil(),
                    // GL JS rounds halves away from zero.
                    MathFn::Round => x.round(),
                    MathFn::Sqrt => x.sqrt(),
                    MathFn::Ln => x.ln(),
                    MathFn::Log2 => x.ln() / std::f64::consts::LN_2,
                    MathFn::Log10 => x.ln() / std::f64::consts::LN_10,
                    MathFn::Sin => x.sin(),
                    MathFn::Cos => x.cos(),
                    MathFn::Tan => x.tan(),
                    MathFn::Asin => x.asin(),
                    MathFn::Acos => x.acos(),
                    MathFn::Atan => x.atan(),
                })
            }
            _ => unreachable!("dispatched by eval"),
        })
    }

    fn eval_conversion(&self, id: NodeId, node: &Node) -> Result<Value<'a>> {
        let op = &node.op;
        Ok(match op {
            Op::Coerce(args) => self.coerce(id, &node.ty, args)?,
            Op::Assert(args) => {
                let mut found = Value::Null;
                for &arg in args {
                    let v = self.eval(arg)?;
                    if is_a(&v, &node.ty) {
                        return Ok(v);
                    }
                    found = v;
                }
                return Err(self.mismatch(id, node.ty.clone(), &found));
            }
            Op::Rgba { r, g, b, a } => {
                let (r, g, b) = (self.number(*r)?, self.number(*g)?, self.number(*b)?);
                let a = a.map(|a| self.number(a)).transpose()?;
                if let Some(message) = validate_rgba(r, g, b, a) {
                    return Err(error(id, message));
                }
                Value::Color(Color::new(
                    r / 255.0,
                    g / 255.0,
                    b / 255.0,
                    a.unwrap_or(1.0),
                ))
            }
            Op::ToRgba(c) => {
                let v = self.eval(*c)?;
                let Value::Color(c) = v else {
                    return Err(self.mismatch(*c, Type::Color, &v));
                };
                let [r, g, b, a] = c.straight().map(f64::from);
                Value::Array(Cow::Owned(vec![
                    Value::Number(r * 255.0),
                    Value::Number(g * 255.0),
                    Value::Number(b * 255.0),
                    Value::Number(a),
                ]))
            }
            _ => unreachable!("dispatched by eval"),
        })
    }

    fn eval_other(&self, id: NodeId, op: &Op) -> Result<Value<'a>> {
        Ok(match op {
            Op::Concat(args) => {
                let mut out = String::new();
                for &arg in args {
                    use fmt::Write;
                    write!(out, "{}", self.eval(arg)?).expect("write to String");
                }
                Value::String(Cow::Owned(out))
            }
            Op::Upcase(s) => {
                Value::String(Cow::Owned(self.context.locale.uppercase(&self.string(*s)?)))
            }
            Op::Downcase(s) => {
                Value::String(Cow::Owned(self.context.locale.lowercase(&self.string(*s)?)))
            }
            Op::Length(v) => match self.eval(*v)? {
                Value::String(s) => Value::Number(s.chars().count() as f64),
                Value::Array(a) => Value::Number(a.len() as f64),
                other => {
                    return Err(error(
                        id,
                        format!(
                            "Expected value to be of type string or array, but found {} instead.",
                            other.type_of()
                        ),
                    ));
                }
            },
            Op::At { index, array } => {
                let i = self.number(*index)?;
                let v = self.eval(*array)?;
                let Value::Array(items) = v else {
                    return Err(self.mismatch(*array, Type::array(Type::Value, None), &v));
                };
                let len = items.len();
                if i < 0.0 {
                    return Err(error(
                        id,
                        format!(
                            "Array index out of bounds: {} < 0.",
                            js::number_to_string(i)
                        ),
                    ));
                }
                if i >= len as f64 {
                    return Err(error(
                        id,
                        format!(
                            "Array index out of bounds: {} > {}.",
                            js::number_to_string(i),
                            js::number_to_string(len as f64 - 1.0)
                        ),
                    ));
                }
                if i != i.floor() {
                    return Err(error(
                        id,
                        format!(
                            "Array index must be an integer, but found {} instead.",
                            js::number_to_string(i)
                        ),
                    ));
                }
                project(items, |items| items.get(i as usize), |v| v).expect("index in bounds")
            }
            Op::TypeOf(v) => Value::String(Cow::Owned(self.eval(*v)?.type_of().to_string())),
            Op::Legacy(filter) => Value::Bool(self.legacy(filter)),
            _ => unreachable!("dispatched by eval"),
        })
    }

    /// `object[key]`, `None` if absent.
    fn member(&self, object: NodeId, key: &str) -> Result<Option<Value<'a>>> {
        match self.eval(object)? {
            Value::Object(members) => Ok(project(
                members,
                |m| m.iter().rev().find(|(k, _)| k == key),
                |(_, v)| v,
            )),
            other => Err(self.mismatch(object, Type::Object, &other)),
        }
    }

    fn contains(&self, id: NodeId, needle: NodeId, haystack: NodeId) -> Result<Value<'a>> {
        let needle = self.eval(needle)?;
        let haystack = self.eval(haystack)?;
        if !haystack.truthy() {
            return Ok(Value::Bool(false));
        }
        if !matches!(
            needle,
            Value::Bool(_) | Value::String(_) | Value::Number(_) | Value::Null
        ) {
            return Err(error(
                id,
                format!(
                    "Expected first argument to be of type boolean, string, number or null, but found {} instead.",
                    needle.type_of()
                ),
            ));
        }
        Ok(Value::Bool(match &haystack {
            Value::String(s) => s.contains(needle.to_js_string().as_str()),
            Value::Array(items) => items.iter().any(|v| needle.strict_eq(v)),
            other => {
                return Err(error(
                    id,
                    format!(
                        "Expected second argument to be of type array or string, but found {} instead.",
                        other.type_of()
                    ),
                ));
            }
        }))
    }

    pub(super) fn interpolate<'v>(
        &self,
        id: NodeId,
        lower: Value<'v>,
        upper: Value<'v>,
        t: f64,
    ) -> Result<Value<'v>> {
        let number = |a: f64, b: f64| a + t * (b - a);
        Ok(match (&lower, &upper) {
            (Value::Number(a), Value::Number(b)) => Value::Number(number(*a, *b)),
            (Value::Color(a), Value::Color(b)) => Value::Color(Color::interpolate(*a, *b, t)),
            (Value::Array(a), Value::Array(b)) => Value::Array(Cow::Owned(
                a.iter()
                    .enumerate()
                    .map(|(i, a)| match (a, b.get(i)) {
                        (Value::Number(a), Some(Value::Number(b))) => {
                            Ok(Value::Number(number(*a, *b)))
                        }
                        (Value::Number(_), None) => Ok(Value::Number(f64::NAN)),
                        _ => Err(error(id, "Cannot interpolate non-numeric array items.")),
                    })
                    .collect::<Result<_>>()?,
            )),
            _ => {
                return Err(error(
                    id,
                    format!(
                        "Cannot interpolate {} and {}.",
                        lower.type_of(),
                        upper.type_of()
                    ),
                ));
            }
        })
    }

    fn coerce(&self, id: NodeId, ty: &Type, args: &[NodeId]) -> Result<Value<'a>> {
        match ty {
            Type::Padding
            | Type::NumberArray
            | Type::ColorArray
            | Type::VariableAnchorOffsetCollection => self.coerce_array(id, ty, args),
            Type::Formatted => Ok(Value::Formatted(
                vec![crate::FormattedSection::text(
                    self.eval(args[0])?.to_string(),
                )]
                .into(),
            )),
            Type::ProjectionDefinition => {
                let v = self.eval(args[0])?;
                let valid = matches!(v, Value::String(_))
                    || v.as_array().is_some_and(|a| {
                        a.len() == 3
                            && a[0].as_str().is_some()
                            && a[1].as_str().is_some()
                            && a[2].as_f64().is_some()
                    })
                    || matches!(&v, Value::Object(o) if ["from","to"].iter().all(|key| o.iter().rev().find(|(k,_)|k==key).is_some_and(|(_,v)|v.as_str().is_some())) && o.iter().rev().find(|(k,_)|k=="transition").is_some_and(|(_,v)|v.as_f64().is_some()));
                if valid {
                    Ok(v)
                } else {
                    Err(error(
                        id,
                        format!(
                            "Could not parse projectionDefinition from value '{}'",
                            v.to_json_string()
                        ),
                    ))
                }
            }
            Type::Boolean => Ok(Value::Bool(self.eval(args[0])?.truthy())),
            Type::String => Ok(Value::String(Cow::Owned(self.eval(args[0])?.to_string()))),
            Type::ResolvedImage => {
                let name = self.eval(args[0])?.to_string();
                Ok(if name.is_empty() {
                    Value::Null
                } else {
                    Value::Image {
                        name: Cow::Owned(name),
                        available: false,
                    }
                })
            }
            Type::Number => {
                let mut last = Value::Null;
                for &arg in args {
                    let v = self.eval(arg)?;
                    if v == Value::Null {
                        return Ok(Value::Number(0.0));
                    }
                    let n = v.to_number();
                    if !n.is_nan() {
                        return Ok(Value::Number(n));
                    }
                    last = v;
                }
                Err(error(
                    id,
                    format!("Could not convert {} to number.", last.to_json_string()),
                ))
            }
            Type::Color => {
                let mut last = Value::Null;
                let mut message = None;
                for &arg in args {
                    let v = self.eval(arg)?;
                    message = None;
                    match &v {
                        Value::Color(c) => return Ok(Value::Color(*c)),
                        Value::String(s) => {
                            if let Some(c) = Color::parse(s) {
                                return Ok(Value::Color(c));
                            }
                        }
                        Value::Array(items) => {
                            if let Some(c) = rgba_array(items, &mut message) {
                                return Ok(Value::Color(c));
                            }
                        }
                        _ => {}
                    }
                    last = v;
                }
                Err(error(
                    id,
                    message.unwrap_or_else(|| {
                        let shown = match &last {
                            Value::String(s) => s.to_string(),
                            other => other.to_json_string(),
                        };
                        format!("Could not parse color from value '{shown}'")
                    }),
                ))
            }
            other => Err(error(id, format!("Cannot coerce to {other}."))),
        }
    }
}

/// A colour from `[r, g, b]` or `[r, g, b, a]`, as `to-color` reads arrays.
pub(crate) fn rgba_array(items: &[Value<'static>], message: &mut Option<String>) -> Option<Color> {
    if items.len() < 3 || items.len() > 4 {
        *message = Some(format!(
            "Invalid rgba value {}: expected an array containing either three or four numeric values.",
            Value::Array(Cow::Borrowed(items)).to_json_string()
        ));
        return None;
    }
    let in_range =
        |v: &Value<'_>, max: f64| matches!(v, Value::Number(n) if (0.0..=max).contains(n));
    let alpha = items.get(3);
    if !items[..3].iter().all(|v| in_range(v, 255.0)) {
        let shown = if matches!(alpha, Some(Value::Number(_))) {
            &items[..4]
        } else {
            &items[..3]
        };
        *message = Some(format!(
            "Invalid rgba value [{}]: 'r', 'g', and 'b' must be between 0 and 255.",
            join_values(shown)
        ));
        return None;
    }
    if let Some(a) = alpha
        && !in_range(a, 1.0)
    {
        *message = Some(format!(
            "Invalid rgba value [{}]: 'a' must be between 0 and 1.",
            join_values(items)
        ));
        return None;
    }
    let [r, g, b] = [0, 1, 2].map(|i| items[i].as_f64().expect("checked"));
    let a = alpha.and_then(Value::as_f64).unwrap_or(1.0);
    // GL JS builds this colour as already premultiplied.
    Some(Color::from_premultiplied(
        r / 255.0,
        g / 255.0,
        b / 255.0,
        a,
    ))
}

/// `Array.prototype.join(', ')`.
fn join_values(items: &[Value<'static>]) -> String {
    items
        .iter()
        .map(|v| match v {
            Value::Null => String::new(),
            v => v.to_js_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

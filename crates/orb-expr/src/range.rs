//! The numbers an expression can give over every feature: an interval
//! for each node, from its operators alone. Inputs known without a feature
//! (zoom) evaluate exactly; feature properties are unbounded until an
//! operator bounds them (`match`, `step`, `interpolate`, `min`, …).

use crate::eval::Evaluator;
use crate::ir::{Call, Deps, Ir, MathFn, NodeId, Op};
use crate::{EvaluationContext, Expression, Type, Value};

/// A closed interval of numbers, empty when `lo > hi`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Range {
    lo: f64,
    hi: f64,
}

impl Range {
    const ALL: Self = Self {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };
    /// Gives no number: always an error, `null` or another type.
    const NONE: Self = Self {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };

    fn point(v: f64) -> Self {
        if v.is_nan() {
            Self::NONE
        } else {
            Self { lo: v, hi: v }
        }
    }

    /// From two ends that may be NaN (`inf - inf`, `0 * inf`): unbounded.
    fn new(lo: f64, hi: f64) -> Self {
        if lo.is_nan() || hi.is_nan() {
            Self::ALL
        } else {
            Self { lo, hi }
        }
    }

    fn is_none(self) -> bool {
        self.lo > self.hi
    }

    fn union(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn map_monotone(self, f: impl Fn(f64) -> f64) -> Self {
        if self.is_none() {
            return self;
        }
        Self::new(f(self.lo), f(self.hi))
    }
}

impl Expression {
    /// The largest number the expression gives at `zoom`, over every
    /// feature: `None` when its operators set no bound (a bare `get`), and
    /// negative infinity when it never gives a number.
    pub fn max_number(&self, zoom: f64) -> Option<f64> {
        let context = EvaluationContext {
            zoom,
            ..EvaluationContext::default()
        };
        let ranges = ranges(&self.ir, &context);
        let root = ranges[self.ir.root() as usize];
        (root.hi < f64::INFINITY).then_some(root.hi)
    }
}

/// Every node's range, in the arena's post order.
fn ranges(ir: &Ir, context: &EvaluationContext<'_>) -> Vec<Range> {
    let evaluator = Evaluator {
        ir,
        context,
        feature: None,
    };
    let mut out: Vec<Range> = Vec::with_capacity(ir.nodes.len());
    for (id, node) in ir.nodes.iter().enumerate() {
        let range = if node.deps.within(Deps::ZOOM) {
            match evaluator.eval(id as NodeId) {
                Ok(Value::Number(n)) => Range::point(n),
                _ => Range::NONE,
            }
        } else {
            range_of(ir, &node.op, &out)
        };
        out.push(range);
    }
    out
}

/// A feature-dependent node's range from its operands' `ranges`.
fn range_of(ir: &Ir, op: &Op, ranges: &[Range]) -> Range {
    let r = |id: &NodeId| ranges[*id as usize];
    let union = |ids: &mut dyn Iterator<Item = &NodeId>| ids.map(r).fold(Range::NONE, Range::union);
    // Arithmetic errors on an operand that gives no number.
    let all = |ids: &[NodeId]| ids.iter().map(r).all(|x| !x.is_none());
    match op {
        Op::Call(Call::Let, args) => args.last().map_or(Range::ALL, r),
        Op::Match {
            outputs, otherwise, ..
        } => union(&mut outputs.iter().chain([otherwise])),
        Op::Case {
            branches,
            otherwise,
        } => union(&mut branches.iter().map(|(_, out)| out).chain([otherwise])),
        Op::Coalesce(ids) | Op::Assert(ids) => union(&mut ids.iter()),
        // A string or other value can convert to any number.
        Op::Coerce(ids) if ids.iter().all(|&id| ir.node(id).ty == Type::Number) => {
            union(&mut ids.iter())
        }
        Op::Step {
            input,
            labels,
            outputs,
        } => {
            let input = r(input);
            if input.is_none() {
                return Range::NONE;
            }
            let above = &labels[1..];
            let first = above.partition_point(|&l| l <= input.lo);
            let last = above.partition_point(|&l| l <= input.hi);
            union(&mut outputs[first..=last].iter())
        }
        // Every interpolation stays between its two stops' outputs.
        Op::Interpolate {
            input,
            labels,
            outputs,
            ..
        } if ir.nodes[outputs[0] as usize].ty == Type::Number => {
            let input = r(input);
            if input.is_none() {
                return Range::NONE;
            }
            let end = labels.len() - 1;
            let first = labels.partition_point(|&l| l <= input.lo).saturating_sub(1);
            let last = labels.partition_point(|&l| l < input.hi).min(end);
            union(&mut outputs[first.min(last)..=last].iter())
        }
        Op::Sum(ids) if all(ids) => ids.iter().map(r).fold(Range::point(0.0), |a, b| {
            Range::new(a.lo + b.lo, a.hi + b.hi)
        }),
        Op::Product(ids) if all(ids) => ids.iter().map(r).fold(Range::point(1.0), |a, b| {
            let ends = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
            if ends.iter().any(|e| e.is_nan()) {
                return Range::ALL;
            }
            Range::new(
                ends.into_iter().fold(f64::INFINITY, f64::min),
                ends.into_iter().fold(f64::NEG_INFINITY, f64::max),
            )
        }),
        Op::Sub(a, b) if all(&[*a, *b]) => {
            let (a, b) = (r(a), r(b));
            Range::new(a.lo - b.hi, a.hi - b.lo)
        }
        Op::Neg(a) => {
            let a = r(a);
            if a.is_none() {
                a
            } else {
                Range::new(-a.hi, -a.lo)
            }
        }
        Op::Min(ids) if all(ids) => {
            let start = Range::point(f64::INFINITY);
            let min = |a: Range, b: Range| Range::new(a.lo.min(b.lo), a.hi.min(b.hi));
            ids.iter().map(r).fold(start, min)
        }
        Op::Max(ids) if all(ids) => {
            let start = Range::point(f64::NEG_INFINITY);
            let max = |a: Range, b: Range| Range::new(a.lo.max(b.lo), a.hi.max(b.hi));
            ids.iter().map(r).fold(start, max)
        }
        Op::Math(f, a) => {
            let a = r(a);
            match f {
                MathFn::Floor => a.map_monotone(f64::floor),
                MathFn::Ceil => a.map_monotone(f64::ceil),
                MathFn::Round => a.map_monotone(f64::round),
                MathFn::Sqrt => a.map_monotone(|v| v.max(0.0).sqrt()),
                MathFn::Abs if a.is_none() => a,
                MathFn::Abs if a.lo >= 0.0 => a,
                MathFn::Abs if a.hi <= 0.0 => Range::new(-a.hi, -a.lo),
                MathFn::Abs => Range::new(0.0, a.hi.max(-a.lo)),
                MathFn::Sin | MathFn::Cos if !a.is_none() => Range::new(-1.0, 1.0),
                _ => Range::ALL,
            }
        }
        _ => Range::ALL,
    }
}

#[cfg(test)]
mod tests {
    use crate::Expression;

    fn max(source: &str, zoom: f64) -> Option<f64> {
        let json = crate::json::parse(source).unwrap();
        Expression::parse(&json, Some(&crate::Type::Number))
            .unwrap()
            .max_number(zoom)
    }

    #[test]
    fn bounds_follow_the_operators() {
        assert_eq!(max("500", 0.0), Some(500.0));
        assert_eq!(max(r#"["get", "altitude"]"#, 0.0), None);
        assert_eq!(
            max(
                r#"["match", ["get", "kind"], "leo", 500000, "geo", 35786000, 0]"#,
                0.0
            ),
            Some(35_786_000.0)
        );
        assert_eq!(
            max(r#"["min", ["get", "altitude"], 600000]"#, 0.0),
            Some(600_000.0)
        );
        assert_eq!(max(r#"["max", ["get", "altitude"], 600000]"#, 0.0), None);
        assert_eq!(
            max(
                r#"["interpolate", ["linear"], ["get", "altitude"], 0, 0, 1000, 2000]"#,
                0.0
            ),
            Some(2000.0)
        );
        assert_eq!(
            max(r#"["*", 2, ["case", ["has", "a"], 10, -3]]"#, 0.0),
            Some(20.0)
        );
        assert_eq!(
            max(r#"["-", 100, ["abs", ["number", ["get", "d"], 0]]]"#, 0.0),
            Some(100.0)
        );
        assert_eq!(max(r#"["coalesce", ["get", "a"], 7]"#, 0.0), None);
        assert_eq!(max(r#"["to-number", ["get", "a"], 7]"#, 0.0), None);
    }

    #[test]
    fn zoom_selects_its_stops() {
        let curve = r#"["interpolate", ["linear"], ["zoom"],
            2, ["match", ["get", "k"], "a", 10, 1],
            4, ["match", ["get", "k"], "a", 30, 3],
            8, ["*", ["get", "h"], 2]]"#;
        assert_eq!(max(curve, 1.0), Some(10.0));
        assert_eq!(max(curve, 2.0), Some(10.0));
        assert_eq!(max(curve, 3.0), Some(30.0));
        assert_eq!(max(curve, 4.0), Some(30.0));
        assert_eq!(max(curve, 5.0), None);
        let step = r#"["step", ["zoom"], ["get", "h"], 4, 100, 8, ["get", "h"]]"#;
        assert_eq!(max(step, 3.0), None);
        assert_eq!(max(step, 4.0), Some(100.0));
        assert_eq!(max(step, 7.9), Some(100.0));
        assert_eq!(max(step, 8.0), None);
        assert_eq!(
            max(r#"["step", ["get", "h"], 1, 10, 5, 20, 2]"#, 0.0),
            Some(5.0)
        );
    }
}

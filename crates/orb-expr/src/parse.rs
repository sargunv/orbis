//! Parsing and type checking, ported from the style-spec's `ParsingContext`
//! and the `parse` methods of its expression classes. Messages match the
//! reference so the spec's fixtures compare exactly.

use std::mem;

use crate::Diagnostic;
use crate::eval::Evaluator;
use crate::ir::{Call, CmpOp, ColorSpace, Interpolation, Ir, MatchLabels, MathFn, NodeId, Op};
use crate::json::{Json, JsonValue, Span};
use crate::value::{Type, Value, check_subtype};

/// How a value-typed result is fitted to an expected type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Annotation {
    Assert,
    Coerce,
    Omit,
}

/// The public expression operators in style-spec 26.5.1.
const OPERATORS: &[&str] = &[
    "!",
    "!=",
    "%",
    "*",
    "+",
    "-",
    "/",
    "<",
    "<=",
    "==",
    ">",
    ">=",
    "^",
    "abs",
    "accumulated",
    "acos",
    "all",
    "any",
    "array",
    "asin",
    "at",
    "atan",
    "bearing",
    "boolean",
    "case",
    "ceil",
    "coalesce",
    "collator",
    "concat",
    "cos",
    "distance",
    "downcase",
    "dev.sargunv.orbis:daylight",
    "e",
    "elevation",
    "error",
    "feature-state",
    "floor",
    "format",
    "geometry-type",
    "get",
    "global-state",
    "has",
    "heatmap-density",
    "id",
    "image",
    "in",
    "index-of",
    "interpolate",
    "interpolate-hcl",
    "interpolate-lab",
    "is-supported-script",
    "join",
    "length",
    "let",
    "line-progress",
    "literal",
    "ln",
    "ln2",
    "log10",
    "log2",
    "match",
    "max",
    "min",
    "number",
    "number-format",
    "object",
    "pi",
    "pitch",
    "properties",
    "resolved-locale",
    "rgb",
    "rgba",
    "round",
    "semiliteral",
    "sin",
    "slice",
    "split",
    "sqrt",
    "step",
    "string",
    "tan",
    "to-boolean",
    "to-color",
    "to-number",
    "to-rgba",
    "to-string",
    "typeof",
    "upcase",
    "var",
    "within",
    "zoom",
];

/// Whether `json` is an expression rather than a constant: a non-empty
/// array whose head names a v8 operator (GL JS's `isExpression`).
pub(crate) fn is_expression(json: &Json) -> bool {
    json.as_array()
        .and_then(|a| a.first())
        .and_then(Json::as_str)
        .is_some_and(|op| OPERATORS.contains(&op))
}

pub(super) fn child_key(key: &str, index: usize) -> String {
    format!("{key}[{index}]")
}

enum Params {
    Fixed(Vec<Type>),
    Varargs(Type),
}

impl Params {
    fn get(&self, i: usize) -> &Type {
        match self {
            Params::Fixed(types) => &types[i],
            Params::Varargs(t) => t,
        }
    }
}

type Build = fn(&mut Parser, &[NodeId]) -> Op;

/// The fixed-signature operators (GL JS's `CompoundExpression` definitions).
fn compound(name: &str) -> Option<(Type, Vec<(Params, Build)>)> {
    use Params::{Fixed, Varargs};
    use Type::{Boolean as B, Number as N, Object as O, String as S, Value as V};
    let one = |t: Type| Fixed(vec![t]);
    let two = |a: Type, b: Type| Fixed(vec![a, b]);
    let none = || Fixed(vec![]);
    Some(match name {
        "typeof" => (S, vec![(one(V), |_, a| Op::TypeOf(a[0]))]),
        "to-rgba" => (
            Type::array(N, Some(4)),
            vec![(one(Type::Color), |_, a| Op::ToRgba(a[0]))],
        ),
        "rgb" => (
            Type::Color,
            vec![(Fixed(vec![N, N, N]), |_, a| Op::Rgba {
                r: a[0],
                g: a[1],
                b: a[2],
                a: None,
            })],
        ),
        "rgba" => (
            Type::Color,
            vec![(Fixed(vec![N, N, N, N]), |_, a| Op::Rgba {
                r: a[0],
                g: a[1],
                b: a[2],
                a: Some(a[3]),
            })],
        ),
        "has" => (
            B,
            vec![
                (one(S), |p, a| match p.literal_string(a[0]) {
                    Some(key) => Op::Has(p.ir.intern_key(&key)),
                    None => Op::HasDynamic(a[0]),
                }),
                (two(S, O), |_, a| Op::HasIn {
                    key: a[0],
                    object: a[1],
                }),
            ],
        ),
        "get" => (
            V,
            vec![
                (one(S), |p, a| match p.literal_string(a[0]) {
                    Some(key) => Op::Get(p.ir.intern_key(&key)),
                    None => Op::GetDynamic(a[0]),
                }),
                (two(S, O), |_, a| Op::GetFrom {
                    key: a[0],
                    object: a[1],
                }),
            ],
        ),
        "properties" => (O, vec![(none(), |_, _| Op::Properties)]),
        "geometry-type" => (S, vec![(none(), |_, _| Op::GeometryType)]),
        "id" => (V, vec![(none(), |_, _| Op::Id)]),
        "zoom" => (N, vec![(none(), |_, _| Op::Zoom)]),
        "line-progress" => (N, vec![(none(), |_, _| Op::LineProgress)]),
        "heatmap-density" => (N, vec![(none(), |_, _| Op::Call(Call::Heatmap, vec![]))]),
        "pitch" => (N, vec![(none(), |_, _| Op::Call(Call::Pitch, vec![]))]),
        "bearing" => (N, vec![(none(), |_, _| Op::Call(Call::Bearing, vec![]))]),
        "dev.sargunv.orbis:meters-per-pixel" => (
            N,
            vec![(none(), |_, _| Op::Call(Call::MetersPerPixel, vec![]))],
        ),
        "elevation" => (N, vec![(none(), |_, _| Op::Call(Call::Elevation, vec![]))]),
        "dev.sargunv.orbis:daylight" => {
            (N, vec![(none(), |_, _| Op::Call(Call::Daylight, vec![]))])
        }
        "accumulated" => (
            V,
            vec![(none(), |_, _| Op::Call(Call::Accumulated, vec![]))],
        ),
        "feature-state" => (
            V,
            vec![(one(S), |_, a| Op::Call(Call::FeatureState, a.to_vec()))],
        ),
        "error" => (
            Type::Error,
            vec![(one(S), |_, a| Op::Call(Call::Error, a.to_vec()))],
        ),
        "is-supported-script" => (
            B,
            vec![(one(S), |_, a| Op::Call(Call::SupportedScript, a.to_vec()))],
        ),
        "resolved-locale" => (
            S,
            vec![(one(Type::Collator), |_, a| {
                Op::Call(Call::ResolvedLocale, a.to_vec())
            })],
        ),
        "split" => (
            Type::array(S, None),
            vec![(two(S, S), |_, a| Op::Call(Call::Split, a.to_vec()))],
        ),
        "join" => (
            S,
            vec![(two(Type::array(S, None), S), |_, a| {
                Op::Call(Call::Join, a.to_vec())
            })],
        ),
        "pi" => (
            N,
            vec![(none(), |_, _| {
                Op::Literal(Value::Number(std::f64::consts::PI))
            })],
        ),
        "e" => (
            N,
            vec![(none(), |_, _| {
                Op::Literal(Value::Number(std::f64::consts::E))
            })],
        ),
        "ln2" => (
            N,
            vec![(none(), |_, _| {
                Op::Literal(Value::Number(std::f64::consts::LN_2))
            })],
        ),
        "+" => (N, vec![(Varargs(N), |_, a| Op::Sum(a.to_vec()))]),
        "*" => (N, vec![(Varargs(N), |_, a| Op::Product(a.to_vec()))]),
        "-" => (
            N,
            vec![
                (two(N, N), |_, a| Op::Sub(a[0], a[1])),
                (one(N), |_, a| Op::Neg(a[0])),
            ],
        ),
        "/" => (N, vec![(two(N, N), |_, a| Op::Div(a[0], a[1]))]),
        "%" => (N, vec![(two(N, N), |_, a| Op::Rem(a[0], a[1]))]),
        "^" => (N, vec![(two(N, N), |_, a| Op::Pow(a[0], a[1]))]),
        "ln" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Ln, a[0]))]),
        "log2" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Log2, a[0]))]),
        "log10" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Log10, a[0]))]),
        "sin" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Sin, a[0]))]),
        "cos" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Cos, a[0]))]),
        "tan" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Tan, a[0]))]),
        "asin" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Asin, a[0]))]),
        "acos" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Acos, a[0]))]),
        "atan" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Atan, a[0]))]),
        "sqrt" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Sqrt, a[0]))]),
        "abs" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Abs, a[0]))]),
        "round" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Round, a[0]))]),
        "floor" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Floor, a[0]))]),
        "ceil" => (N, vec![(one(N), |_, a| Op::Math(MathFn::Ceil, a[0]))]),
        "min" => (N, vec![(Varargs(N), |_, a| Op::Min(a.to_vec()))]),
        "max" => (N, vec![(Varargs(N), |_, a| Op::Max(a.to_vec()))]),
        "all" => (
            B,
            vec![
                (two(B, B), |_, a| Op::All(a.to_vec())),
                (Varargs(B), |_, a| Op::All(a.to_vec())),
            ],
        ),
        "any" => (
            B,
            vec![
                (two(B, B), |_, a| Op::Any(a.to_vec())),
                (Varargs(B), |_, a| Op::Any(a.to_vec())),
            ],
        ),
        "!" => (B, vec![(one(B), |_, a| Op::Not(a[0]))]),
        "upcase" => (S, vec![(one(S), |_, a| Op::Upcase(a[0]))]),
        "downcase" => (S, vec![(one(S), |_, a| Op::Downcase(a[0]))]),
        "concat" => (S, vec![(Varargs(V), |_, a| Op::Concat(a.to_vec()))]),
        _ => return None,
    })
}

fn stringify_signature(params: &Params) -> String {
    match params {
        Params::Fixed(types) => format!(
            "({})",
            types
                .iter()
                .map(Type::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Params::Varargs(t) => format!("({t}...)"),
    }
}

/// JavaScript's `String(v)` of a JSON value, for messages.
fn js_string(json: &Json) -> String {
    match &json.value {
        JsonValue::String(s) => s.clone(),
        _ => Value::from_json(json).to_js_string(),
    }
}

/// `typeof` of a JSON value.
pub(super) fn js_typeof(json: &Json) -> &'static str {
    match json.value {
        JsonValue::Bool(_) => "boolean",
        JsonValue::Number(_) => "number",
        JsonValue::String(_) => "string",
        _ => "object",
    }
}

#[derive(Debug, Default)]
pub(crate) struct Parser<'l> {
    /// When set, `global-state` reads fold to these values at parse time.
    pub global_state: Option<&'l [(String, Value<'static>)]>,
    /// The literals folded from global state.
    pub(crate) global_literals: Vec<NodeId>,
    pub(crate) ir: Ir,
    pub errors: Vec<Diagnostic>,
    pub(super) scope: Vec<(String, NodeId)>,
}

impl<'l> Parser<'l> {
    pub(crate) fn error(
        &mut self,
        path: impl Into<String>,
        span: Span,
        message: impl Into<String>,
    ) {
        self.errors.push(Diagnostic {
            path: path.into(),
            span,
            message: message.into(),
        });
    }

    /// Checks that `items` holds exactly `count` arguments after the operator.
    pub(super) fn arity(
        &mut self,
        items: &[Json],
        count: usize,
        span: Span,
        key: &str,
    ) -> Option<()> {
        let n = items.len() - 1;
        if n != count {
            let s = if count == 1 { "" } else { "s" };
            let message = format!("Expected {count} argument{s}, but found {n} instead.");
            self.error(key, span, message);
            return None;
        }
        Some(())
    }

    /// Checks that `items` holds an input and pairs after the operator
    /// (`match`, `step`, `interpolate`): at least `min` arguments, as many
    /// as `min` is even or odd.
    fn pairs_arity(&mut self, items: &[Json], min: usize, span: Span, key: &str) -> Option<()> {
        let n = items.len() - 1;
        if n < min {
            let message = format!("Expected at least {min} arguments, but found only {n}.");
            self.error(key, span, message);
            return None;
        }
        if n % 2 != min % 2 {
            let parity = if min.is_multiple_of(2) { "even" } else { "odd" };
            self.error(
                key,
                span,
                format!("Expected an {parity} number of arguments."),
            );
            return None;
        }
        Some(())
    }

    pub(super) fn ty(&self, id: NodeId) -> &Type {
        &self.ir.node(id).ty
    }

    fn literal_string(&self, id: NodeId) -> Option<String> {
        match &self.ir.node(id).op {
            Op::Literal(Value::String(s)) => Some(s.to_string()),
            _ => None,
        }
    }

    /// A literal node, inferring an empty array's item type from what the
    /// context expects.
    pub(crate) fn literal(
        &mut self,
        value: Value<'static>,
        key: &str,
        expected: Option<&Type>,
    ) -> NodeId {
        let mut ty = value.type_of();
        if let (Type::Array(_, Some(0)), Some(Type::Array(_, n))) = (&ty, expected)
            && matches!(n, None | Some(0))
        {
            ty = expected.cloned().expect("matched");
        }
        self.ir.push(Op::Literal(value), ty, key)
    }

    /// Parses `json` as a constant property value rather than an
    /// expression: a literal, fitted to the expected type like any other.
    pub(crate) fn parse_constant(
        &mut self,
        json: &Json,
        expected: Option<&Type>,
        annotation: Option<Annotation>,
    ) -> Option<NodeId> {
        let id = self.literal(Value::from_json(json), "", expected);
        self.finish(id, json.span, "", expected, annotation)
    }

    /// `ParsingContext.parse`.
    pub(crate) fn parse(
        &mut self,
        json: &Json,
        key: &str,
        expected: Option<&Type>,
        annotation: Option<Annotation>,
    ) -> Option<NodeId> {
        let parsed = match &json.value {
            JsonValue::Array(items) => {
                let Some(head) = items.first() else {
                    self.error(
                        key,
                        json.span,
                        "Expected an array with at least one element. If you wanted a literal array, use [\"literal\", []].",
                    );
                    return None;
                };
                let Some(op) = head.as_str() else {
                    self.error(
                        child_key(key, 0),
                        head.span,
                        format!(
                            "Expression name must be a string, but found {} instead. If you wanted a literal array, use [\"literal\", [...]].",
                            js_typeof(head)
                        ),
                    );
                    return None;
                };
                self.parse_op(op, items, json.span, key, expected)?
            }
            JsonValue::Object(_) => {
                self.error(
                    key,
                    json.span,
                    "Bare objects invalid. Use [\"literal\", {...}] instead.",
                );
                return None;
            }
            _ => self.literal(Value::from_json(json), key, expected),
        };
        self.finish(parsed, json.span, key, expected, annotation)
    }

    /// Fits a parsed node to the expected type, then folds it to a literal
    /// if it is constant.
    fn finish(
        &mut self,
        mut parsed: NodeId,
        span: Span,
        key: &str,
        expected: Option<&Type>,
        annotation: Option<Annotation>,
    ) -> Option<NodeId> {
        if let Some(expected) = expected {
            let actual = self.ty(parsed);
            let annotate = if matches!(
                expected,
                Type::String | Type::Number | Type::Boolean | Type::Object | Type::Array(..)
            ) && *actual == Type::Value
            {
                Some(annotation.unwrap_or(Annotation::Assert))
            } else if (matches!(expected, Type::Padding | Type::NumberArray)
                && matches!(actual, Type::Value | Type::Number | Type::Array(..)))
                || (*expected == Type::ColorArray
                    && matches!(actual, Type::Value | Type::String | Type::Array(..)))
                || (*expected == Type::VariableAnchorOffsetCollection
                    && matches!(actual, Type::Value | Type::Array(..)))
                || (*expected == Type::ProjectionDefinition
                    && matches!(actual, Type::String | Type::Array(..) | Type::Value))
                || matches!(
                    expected,
                    Type::Color | Type::ResolvedImage | Type::Formatted
                ) && matches!(actual, Type::Value | Type::String)
            {
                Some(annotation.unwrap_or(Annotation::Coerce))
            } else {
                if let Err(message) = check_subtype(expected, actual) {
                    self.error(key, span, message);
                    return None;
                }
                None
            };
            parsed = match annotate {
                Some(Annotation::Assert) => {
                    self.ir
                        .push(Op::Assert(vec![parsed]), expected.clone(), key)
                }
                Some(Annotation::Coerce) => {
                    self.ir
                        .push(Op::Coerce(vec![parsed]), expected.clone(), key)
                }
                Some(Annotation::Omit) | None => parsed,
            };
        }

        let node = self.ir.node(parsed);
        if !matches!(node.op, Op::Literal(_))
            && node.ty != Type::ResolvedImage
            && self.is_constant(parsed)
        {
            let ty = node.ty.clone();
            let evaluator = Evaluator {
                context: &crate::EvaluationContext::default(),
                ir: &self.ir,
                feature: None,
            };
            match evaluator.eval(parsed).map(Value::into_owned) {
                Ok(value) => parsed = self.ir.push(Op::Literal(value), ty, key),
                // A global-state value fails where GL JS evaluates it, and
                // falls back to the property's default there.
                Err(_) if self.reads_folded_global(parsed) => {}
                Err(e) => {
                    self.error(key, span, e.to_string());
                    return None;
                }
            }
        }
        Some(parsed)
    }

    /// Whether a node is, or has under it, a literal folded from global
    /// state.
    fn reads_folded_global(&self, id: NodeId) -> bool {
        if self.global_literals.contains(&id) {
            return true;
        }
        let mut found = false;
        self.ir
            .node(id)
            .op
            .for_each_child(|c| found |= self.reads_folded_global(c));
        found
    }

    /// `isExpressionConstant`: no inputs, and operands already literals
    /// (or, under a type annotation, constant themselves).
    fn is_constant(&self, id: NodeId) -> bool {
        let node = self.ir.node(id);
        if matches!(
            node.op,
            Op::Call(
                Call::Error | Call::Collator | Call::GlobalState(_) | Call::Geometry { .. },
                _
            )
        ) {
            return false;
        }
        if let Op::Call(Call::Var, args) = &node.op {
            return self.is_constant(args[0]);
        }
        if !node.deps.is_empty() {
            return false;
        }
        let annotation = matches!(node.op, Op::Assert(_) | Op::Coerce(_));
        let mut constant = true;
        node.op.for_each_child(|c| {
            constant &= if annotation {
                self.is_constant(c)
            } else {
                matches!(self.ir.node(c).op, Op::Literal(_))
            };
        });
        constant
    }

    fn parse_op(
        &mut self,
        op: &str,
        items: &[Json],
        span: Span,
        key: &str,
        expected: Option<&Type>,
    ) -> Option<NodeId> {
        match op {
            "literal" => {
                if items.len() != 2 {
                    self.error(
                        key,
                        span,
                        format!(
                            "'literal' expression requires exactly one argument, but found {} instead.",
                            items.len() - 1
                        ),
                    );
                    return None;
                }
                Some(self.literal(Value::from_json(&items[1]), key, expected))
            }
            "match" => self.parse_match(items, span, key, expected),
            "case" => self.parse_case(items, span, key, expected),
            "coalesce" => self.parse_coalesce(items, span, key, expected),
            "step" => self.parse_step(items, span, key, expected),
            "interpolate" | "interpolate-hcl" | "interpolate-lab" => {
                self.parse_interpolate(items, span, key, expected)
            }
            "let" | "var" | "semiliteral" | "global-state" | "index-of" | "slice" | "collator"
            | "number-format" | "format" | "image" | "within" | "distance" => {
                self.parse_extra(op, items, span, key, expected)
            }
            op if let Some(cmp) = CmpOp::from_name(op) => {
                self.parse_comparison(cmp, items, span, key)
            }
            "in" => self.parse_in(items, span, key),
            "at" => self.parse_at(items, span, key, expected),
            "length" => self.parse_length(items, span, key),
            "to-boolean" | "to-color" | "to-number" | "to-string" => {
                self.parse_coercion(op, items, span, key)
            }
            "number" | "string" | "boolean" | "object" | "array" => {
                self.parse_assertion(op, items, span, key)
            }
            _ => {
                if let Some((ty, overloads)) = compound(op) {
                    return self.parse_compound(ty, &overloads, items, span, key);
                }
                let message = format!(
                    "Unknown expression \"{op}\". If you wanted a literal array, use [\"literal\", [...]]."
                );
                self.error(child_key(key, 0), items[0].span, message);
                None
            }
        }
    }

    fn parse_compound(
        &mut self,
        ty: Type,
        overloads: &[(Params, Build)],
        items: &[Json],
        span: Span,
        key: &str,
    ) -> Option<NodeId> {
        let args = &items[1..];
        let candidates: Vec<&(Params, Build)> = overloads
            .iter()
            .filter(|(params, _)| match params {
                Params::Fixed(types) => types.len() == args.len(),
                Params::Varargs(_) => true,
            })
            .collect();

        let mut last_errors = Vec::new();
        for (params, build) in &candidates {
            // A fresh error list per signature, so a later match isn't
            // polluted by an earlier attempt.
            let saved = mem::take(&mut self.errors);
            let mut parsed = Vec::new();
            for (i, arg) in args.iter().enumerate() {
                match self.parse(arg, &child_key(key, i + 1), Some(params.get(i)), None) {
                    Some(id) => parsed.push(id),
                    None => break,
                }
            }
            if parsed.len() == args.len() {
                for (i, &arg) in parsed.iter().enumerate() {
                    if let Err(message) = check_subtype(params.get(i), self.ty(arg)) {
                        self.error(child_key(key, i + 1), args[i].span, message);
                    }
                }
            }
            let errors = mem::replace(&mut self.errors, saved);
            if parsed.len() == args.len() && errors.is_empty() {
                let op = build(self, &parsed);
                return Some(self.ir.push(op, ty, key));
            }
            last_errors = errors;
        }

        if candidates.len() == 1 {
            self.errors.extend(last_errors);
        } else {
            let shown: Vec<&Params> = if candidates.is_empty() {
                overloads.iter().map(|(p, _)| p).collect()
            } else {
                candidates.iter().map(|(p, _)| p).collect()
            };
            let signatures = shown
                .iter()
                .map(|p| stringify_signature(p))
                .collect::<Vec<_>>()
                .join(" | ");
            // Re-parse without coercions to report the actual types.
            let mut actual = Vec::new();
            for (i, arg) in args.iter().enumerate() {
                let id = self.parse(arg, &child_key(key, i + 1), None, None)?;
                actual.push(self.ty(id).to_string());
            }
            self.error(
                key,
                span,
                format!(
                    "Expected arguments of type {signatures}, but found ({}) instead.",
                    actual.join(", ")
                ),
            );
        }
        None
    }

    fn output_type(expected: Option<&Type>) -> Option<Type> {
        expected.filter(|t| **t != Type::Value).cloned()
    }

    fn parse_match(
        &mut self,
        items: &[Json],
        span: Span,
        key: &str,
        expected: Option<&Type>,
    ) -> Option<NodeId> {
        self.pairs_arity(items, 4, span, key)?;

        enum Label {
            Number(f64),
            String(String),
        }
        let mut output_type = Self::output_type(expected);
        let mut input_type: Option<Type> = None;
        let mut cases: Vec<(Label, u32)> = Vec::new();
        let mut outputs = Vec::new();
        let mut i = 2;
        while i < items.len() - 1 {
            let label_key = child_key(key, i);
            let group = &items[i];
            let labels: Vec<&Json> = match group.as_array() {
                Some(labels) => labels.iter().collect(),
                None => vec![group],
            };
            if labels.is_empty() {
                self.error(label_key, group.span, "Expected at least one branch label.");
                return None;
            }
            for label in labels {
                let (value, ty) = match &label.value {
                    JsonValue::Number(n) => {
                        if n.abs() > 9007199254740991.0 {
                            self.error(
                                label_key,
                                label.span,
                                "Branch labels must be integers no larger than 9007199254740991.",
                            );
                            return None;
                        }
                        if n.floor() != *n {
                            self.error(
                                label_key,
                                label.span,
                                "Numeric branch labels must be integer values.",
                            );
                            return None;
                        }
                        (Label::Number(*n), Type::Number)
                    }
                    JsonValue::String(s) => (Label::String(s.clone()), Type::String),
                    _ => {
                        self.error(
                            label_key,
                            label.span,
                            "Branch labels must be numbers or strings.",
                        );
                        return None;
                    }
                };
                match &input_type {
                    None => input_type = Some(ty),
                    Some(input) => {
                        if let Err(message) = check_subtype(input, &ty) {
                            self.error(label_key, label.span, message);
                            return None;
                        }
                    }
                }
                let duplicate = cases.iter().any(|(l, _)| match (l, &value) {
                    (Label::Number(a), Label::Number(b)) => a == b,
                    (Label::String(a), Label::String(b)) => a == b,
                    _ => false,
                });
                if duplicate {
                    self.error(label_key, label.span, "Branch labels must be unique.");
                    return None;
                }
                cases.push((value, outputs.len() as u32));
            }
            // GL JS parses each output at its label's index.
            let output = self.parse(&items[i + 1], &label_key, output_type.as_ref(), None)?;
            output_type.get_or_insert_with(|| self.ty(output).clone());
            outputs.push(output);
            i += 2;
        }

        let input = self.parse(&items[1], &child_key(key, 1), Some(&Type::Value), None)?;
        let last = items.len() - 1;
        let otherwise = self.parse(
            &items[last],
            &child_key(key, last),
            output_type.as_ref(),
            None,
        )?;
        let input_type = input_type.expect("at least one label");
        if *self.ty(input) != Type::Value
            && let Err(message) = check_subtype(&input_type, self.ty(input))
        {
            self.error(child_key(key, 1), items[1].span, message);
            return None;
        }

        let labels = if input_type == Type::Number {
            let mut labels: Vec<(f64, u32)> = cases
                .into_iter()
                .map(|(l, i)| match l {
                    Label::Number(n) => (n, i),
                    Label::String(_) => unreachable!("labels share one type"),
                })
                .collect();
            labels.sort_by(|a, b| a.0.total_cmp(&b.0));
            MatchLabels::Number(labels)
        } else {
            let mut labels: Vec<(Box<str>, u32)> = cases
                .into_iter()
                .map(|(l, i)| match l {
                    Label::String(s) => (s.into(), i),
                    Label::Number(_) => unreachable!("labels share one type"),
                })
                .collect();
            labels.sort_by(|a, b| a.0.cmp(&b.0));
            MatchLabels::String(labels)
        };
        let ty = output_type.expect("at least one output");
        Some(self.ir.push(
            Op::Match {
                input,
                labels,
                outputs,
                otherwise,
            },
            ty,
            key,
        ))
    }

    fn parse_case(
        &mut self,
        items: &[Json],
        span: Span,
        key: &str,
        expected: Option<&Type>,
    ) -> Option<NodeId> {
        self.pairs_arity(items, 3, span, key)?;
        let mut output_type = Self::output_type(expected);
        let mut branches = Vec::new();
        let last = items.len() - 1;
        for i in (1..last).step_by(2) {
            let test = self.parse(&items[i], &child_key(key, i), Some(&Type::Boolean), None)?;
            let result = self.parse(
                &items[i + 1],
                &child_key(key, i + 1),
                output_type.as_ref(),
                None,
            )?;
            output_type.get_or_insert_with(|| self.ty(result).clone());
            branches.push((test, result));
        }
        let otherwise = self.parse(
            &items[last],
            &child_key(key, last),
            output_type.as_ref(),
            None,
        )?;
        let ty = output_type.expect("at least one branch");
        Some(self.ir.push(
            Op::Case {
                branches,
                otherwise,
            },
            ty,
            key,
        ))
    }

    fn parse_coalesce(
        &mut self,
        items: &[Json],
        span: Span,
        key: &str,
        expected: Option<&Type>,
    ) -> Option<NodeId> {
        if items.len() < 2 {
            self.error(key, span, "Expected at least one argument.");
            return None;
        }
        let mut output_type = Self::output_type(expected);
        let mut args = Vec::new();
        for (i, arg) in items[1..].iter().enumerate() {
            // Without inferred annotations, so `null` reaches the coalesce
            // instead of failing an assertion first.
            let parsed = self.parse(
                arg,
                &child_key(key, i + 1),
                output_type.as_ref(),
                Some(Annotation::Omit),
            )?;
            output_type.get_or_insert_with(|| self.ty(parsed).clone());
            args.push(parsed);
        }
        // If an argument would have needed an annotation, annotate the
        // whole coalesce instead.
        let needs_annotation =
            expected.is_some_and(|e| args.iter().any(|&a| check_subtype(e, self.ty(a)).is_err()));
        let ty = if needs_annotation {
            Type::Value
        } else {
            output_type.expect("at least one argument")
        };
        Some(self.ir.push(Op::Coalesce(args), ty, key))
    }

    fn parse_step(
        &mut self,
        items: &[Json],
        span: Span,
        key: &str,
        expected: Option<&Type>,
    ) -> Option<NodeId> {
        self.pairs_arity(items, 4, span, key)?;
        let input = self.parse(&items[1], &child_key(key, 1), Some(&Type::Number), None)?;
        let mut output_type = Self::output_type(expected);
        let mut labels = Vec::new();
        let mut outputs = Vec::new();
        for i in (1..items.len()).step_by(2) {
            let label = if i == 1 {
                f64::NEG_INFINITY
            } else {
                match items[i].as_f64() {
                    Some(label) => label,
                    None => {
                        self.error(
                            child_key(key, i),
                            items[i].span,
                            "Input/output pairs for \"step\" expressions must be defined using literal numeric values (not computed expressions) for the input values.",
                        );
                        return None;
                    }
                }
            };
            if labels.last().is_some_and(|&last| last >= label) {
                self.error(
                    child_key(key, i),
                    items[i].span,
                    "Input/output pairs for \"step\" expressions must be arranged with input values in strictly ascending order.",
                );
                return None;
            }
            let output = self.parse(
                &items[i + 1],
                &child_key(key, i + 1),
                output_type.as_ref(),
                None,
            )?;
            output_type.get_or_insert_with(|| self.ty(output).clone());
            labels.push(label);
            outputs.push(output);
        }
        let ty = output_type.expect("at least one stop");
        Some(self.ir.push(
            Op::Step {
                input,
                labels,
                outputs,
            },
            ty,
            key,
        ))
    }

    fn parse_interpolation(&mut self, json: &Json, key: &str) -> Option<Interpolation> {
        let parts = match json.as_array() {
            Some(parts) if !parts.is_empty() => parts,
            _ => {
                self.error(
                    child_key(key, 1),
                    json.span,
                    "Expected an interpolation type expression.",
                );
                return None;
            }
        };
        match parts[0].as_str() {
            Some("linear") => Some(Interpolation::Linear),
            Some("exponential") => match parts.get(1).and_then(Json::as_f64) {
                Some(base) => Some(Interpolation::Exponential(base)),
                None => {
                    self.error(
                        child_key(&child_key(key, 1), 1),
                        parts.get(1).map_or(json.span, |p| p.span),
                        "Exponential interpolation requires a numeric base.",
                    );
                    None
                }
            },
            Some("cubic-bezier") => {
                let points: Vec<f64> = parts[1..]
                    .iter()
                    .filter_map(|p| p.as_f64().filter(|t| (0.0..=1.0).contains(t)))
                    .collect();
                if parts.len() != 5 || points.len() != 4 {
                    self.error(
                        child_key(key, 1),
                        json.span,
                        "Cubic bezier interpolation requires four numeric arguments with values between 0 and 1.",
                    );
                    return None;
                }
                Some(Interpolation::CubicBezier([
                    points[0], points[1], points[2], points[3],
                ]))
            }
            _ => {
                self.error(
                    child_key(&child_key(key, 1), 0),
                    parts[0].span,
                    format!("Unknown interpolation type {}", js_string(&parts[0])),
                );
                None
            }
        }
    }

    fn parse_interpolate(
        &mut self,
        items: &[Json],
        span: Span,
        key: &str,
        expected: Option<&Type>,
    ) -> Option<NodeId> {
        let Some(interpolation_json) = items.get(1) else {
            self.error(
                child_key(key, 1),
                span,
                "Expected an interpolation type expression.",
            );
            return None;
        };
        let interpolation = self.parse_interpolation(interpolation_json, key)?;
        self.pairs_arity(items, 4, span, key)?;
        let input = self.parse(&items[2], &child_key(key, 2), Some(&Type::Number), None)?;
        let color_space = match items[0].as_str() {
            Some("interpolate-lab") => ColorSpace::Lab,
            Some("interpolate-hcl") => ColorSpace::Hcl,
            _ => ColorSpace::Rgb,
        };
        let mut output_type =
            if color_space == ColorSpace::Rgb || expected == Some(&Type::ColorArray) {
                Self::output_type(expected)
            } else {
                Some(Type::Color)
            };
        let mut labels = Vec::new();
        let mut outputs = Vec::new();
        for i in (3..items.len()).step_by(2) {
            let Some(label) = items[i].as_f64() else {
                self.error(
                    child_key(key, i),
                    items[i].span,
                    "Input/output pairs for \"interpolate\" expressions must be defined using literal numeric values (not computed expressions) for the input values.",
                );
                return None;
            };
            if labels.last().is_some_and(|&last| last >= label) {
                self.error(
                    child_key(key, i),
                    items[i].span,
                    "Input/output pairs for \"interpolate\" expressions must be arranged with input values in strictly ascending order.",
                );
                return None;
            }
            let output = self.parse(
                &items[i + 1],
                &child_key(key, i + 1),
                output_type.as_ref(),
                None,
            )?;
            output_type.get_or_insert_with(|| self.ty(output).clone());
            labels.push(label);
            outputs.push(output);
        }
        let ty = output_type.expect("at least one stop");
        let interpolatable = match &ty {
            Type::Number
            | Type::Color
            | Type::ProjectionDefinition
            | Type::Padding
            | Type::NumberArray
            | Type::ColorArray
            | Type::VariableAnchorOffsetCollection => true,
            Type::Array(item, Some(_)) => **item == Type::Number,
            _ => false,
        };
        if !interpolatable {
            self.error(key, span, format!("Type {ty} is not interpolatable."));
            return None;
        }
        Some(self.ir.push(
            Op::Interpolate {
                color_space,
                interpolation,
                input,
                labels,
                outputs,
            },
            ty,
            key,
        ))
    }

    fn parse_comparison(
        &mut self,
        op: CmpOp,
        items: &[Json],
        span: Span,
        key: &str,
    ) -> Option<NodeId> {
        if items.len() != 3 && items.len() != 4 {
            self.error(key, span, "Expected two or three arguments.");
            return None;
        }
        let comparable = |t: &Type| match t {
            Type::String | Type::Number | Type::Value => true,
            Type::Boolean | Type::Null => !op.is_order(),
            _ => false,
        };
        let mut sides = [0; 2];
        for (side, i) in sides.iter_mut().zip([1, 2]) {
            let id = self.parse(&items[i], &child_key(key, i), Some(&Type::Value), None)?;
            if !comparable(self.ty(id)) {
                let message = format!(
                    "\"{}\" comparisons are not supported for type '{}'.",
                    op.name(),
                    self.ty(id)
                );
                self.error(child_key(key, i), items[i].span, message);
                return None;
            }
            *side = id;
        }
        let [mut lhs, mut rhs] = sides;
        let (lt, rt) = (self.ty(lhs).clone(), self.ty(rhs).clone());
        if lt.kind() != rt.kind() && lt != Type::Value && rt != Type::Value {
            self.error(
                key,
                span,
                format!("Cannot compare types '{lt}' and '{rt}'."),
            );
            return None;
        }
        if op.is_order() {
            if lt == Type::Value && rt != Type::Value {
                lhs = self.ir.push(Op::Assert(vec![lhs]), rt.clone(), key);
            } else if lt != Type::Value && rt == Type::Value {
                rhs = self.ir.push(Op::Assert(vec![rhs]), lt.clone(), key);
            }
        }
        let mut collator = None;
        if items.len() == 4 {
            let stringish = |t: &Type| matches!(t, Type::String | Type::Value);
            if !stringish(&lt) && !stringish(&rt) {
                self.error(
                    key,
                    span,
                    "Cannot use collator to compare non-string types.",
                );
                return None;
            }
            collator =
                Some(self.parse(&items[3], &child_key(key, 3), Some(&Type::Collator), None)?);
        }
        let untyped = *self.ty(lhs) == Type::Value || *self.ty(rhs) == Type::Value;
        Some(self.ir.push(
            Op::Compare {
                op,
                lhs,
                rhs,
                untyped,
                collator,
            },
            Type::Boolean,
            key,
        ))
    }

    fn parse_in(&mut self, items: &[Json], span: Span, key: &str) -> Option<NodeId> {
        self.arity(items, 2, span, key)?;
        let needle = self.parse(&items[1], &child_key(key, 1), Some(&Type::Value), None);
        let haystack = self.parse(&items[2], &child_key(key, 2), Some(&Type::Value), None);
        let (needle, haystack) = (needle?, haystack?);
        if !matches!(
            self.ty(needle),
            Type::Boolean | Type::String | Type::Number | Type::Null | Type::Value
        ) {
            let message = format!(
                "Expected first argument to be of type boolean, string, number or null, but found {} instead",
                self.ty(needle)
            );
            self.error(key, span, message);
            return None;
        }
        Some(
            self.ir
                .push(Op::In { needle, haystack }, Type::Boolean, key),
        )
    }

    fn parse_at(
        &mut self,
        items: &[Json],
        span: Span,
        key: &str,
        expected: Option<&Type>,
    ) -> Option<NodeId> {
        self.arity(items, 2, span, key)?;
        let array_type = Type::array(expected.cloned().unwrap_or(Type::Value), None);
        let index = self.parse(&items[1], &child_key(key, 1), Some(&Type::Number), None);
        let array = self.parse(&items[2], &child_key(key, 2), Some(&array_type), None);
        let (index, array) = (index?, array?);
        let Type::Array(item, _) = self.ty(array) else {
            unreachable!("checked against an array type")
        };
        let ty = (**item).clone();
        Some(self.ir.push(Op::At { index, array }, ty, key))
    }

    fn parse_length(&mut self, items: &[Json], span: Span, key: &str) -> Option<NodeId> {
        self.arity(items, 1, span, key)?;
        let input = self.parse(&items[1], &child_key(key, 1), None, None)?;
        if !matches!(self.ty(input), Type::Array(..) | Type::String | Type::Value) {
            let message = format!(
                "Expected argument of type string or array, but found {} instead.",
                self.ty(input)
            );
            self.error(key, span, message);
            return None;
        }
        Some(self.ir.push(Op::Length(input), Type::Number, key))
    }

    fn parse_coercion(
        &mut self,
        op: &str,
        items: &[Json],
        span: Span,
        key: &str,
    ) -> Option<NodeId> {
        if items.len() < 2 {
            self.error(key, span, "Expected at least one argument.");
            return None;
        }
        if matches!(op, "to-boolean" | "to-string") && items.len() != 2 {
            self.error(key, span, "Expected one argument.");
            return None;
        }
        let ty = match op {
            "to-boolean" => Type::Boolean,
            "to-color" => Type::Color,
            "to-number" => Type::Number,
            _ => Type::String,
        };
        let mut args = Vec::new();
        for (i, arg) in items.iter().enumerate().skip(1) {
            args.push(self.parse(arg, &child_key(key, i), Some(&Type::Value), None)?);
        }
        Some(self.ir.push(Op::Coerce(args), ty, key))
    }

    fn parse_assertion(
        &mut self,
        op: &str,
        items: &[Json],
        span: Span,
        key: &str,
    ) -> Option<NodeId> {
        if items.len() < 2 {
            self.error(key, span, "Expected at least one argument.");
            return None;
        }
        let mut first = 1;
        let ty = match op {
            "number" => Type::Number,
            "string" => Type::String,
            "boolean" => Type::Boolean,
            "object" => Type::Object,
            _ => {
                let item = if items.len() > 2 {
                    first += 1;
                    match items[1].as_str() {
                        Some("string") => Type::String,
                        Some("number") => Type::Number,
                        Some("boolean") => Type::Boolean,
                        _ => {
                            self.error(
                                child_key(key, 1),
                                items[1].span,
                                "The item type argument of \"array\" must be one of string, number, boolean",
                            );
                            return None;
                        }
                    }
                } else {
                    Type::Value
                };
                let mut len = None;
                if items.len() > 3 {
                    first += 1;
                    match &items[2].value {
                        JsonValue::Null => {}
                        JsonValue::Number(n)
                            if *n >= 0.0 && n.floor() == *n && *n <= f64::from(u32::MAX) =>
                        {
                            len = Some(*n as u32);
                        }
                        _ => {
                            self.error(
                                child_key(key, 2),
                                items[2].span,
                                "The length argument to \"array\" must be a positive integer literal",
                            );
                            return None;
                        }
                    }
                }
                Type::array(item, len)
            }
        };
        let mut args = Vec::new();
        for (i, arg) in items.iter().enumerate().skip(first) {
            args.push(self.parse(arg, &child_key(key, i), Some(&Type::Value), None)?);
        }
        Some(self.ir.push(Op::Assert(args), ty, key))
    }
}

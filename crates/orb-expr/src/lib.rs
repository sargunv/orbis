//! MapLibre v8 expressions (plan §11.3): a strict JSON parser, expression
//! parsing with GL JS's type checking, legacy filter and function lowering,
//! property classification, and evaluation against `orb_tile` rows.
//!
//! Semantics follow the style-spec's TypeScript reference
//! (`src/expression/`, version 26.5.1). All public v8 operators are supported;
//! the complete pinned expression suite runs natively and as wasm in Chrome.
//!
//! Two consumers shape the API:
//!
//! - The tile job selects features with [`Filter::select`] and evaluates
//!   source and composite values per feature through [`Expression::bind`],
//!   which resolves property keys to columns once per layer.
//! - The scene side evaluates constant and camera values per frame with
//!   [`Expression::evaluate`] and [`PropertyExpression::interpolation_factor`].

#![forbid(unsafe_code)]

mod color;
mod color_spaces;
mod context;
mod eval;
mod eval_extra;
mod filter;
mod geometry;
mod ir;
mod js;
pub mod json;
mod legacy;
mod locale;
mod parse;
mod parse_extra;
mod property;
mod tile;
mod value;

use std::fmt;

pub use color::Color;
pub use context::{CanonicalTile, EvaluationContext, FeatureGeometry};
pub use eval::{EvalError, interpolation_factor};
pub use filter::Filter;
pub use ir::Interpolation;
pub use locale::{BUILTIN_LOCALE, BuiltinLocale, Collator, Locale, NumberFormat};
pub use property::{Kind, PropertyExpression, PropertySpec, PropertyType, ZoomCurve};
pub use tile::{Bound, FeatureData};
pub use value::FormattedSection;
pub use value::{Type, Value};

use eval::Evaluator;
use ir::{Deps, Ir, Op};
use json::{Json, Span};
use parse::Parser;

#[cfg(test)]
mod conformance;
#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// A number as JavaScript's `String(x)` writes it, as GL JS keys feature
/// state by `String(id)`.
pub fn js_number_string(x: f64) -> String {
    js::number_to_string(x)
}

/// A parse or validation message, located by a path relative to the parsed
/// value (`[2][1]` inside an expression, `.stops[0]` inside a legacy
/// function) and by a byte span in the source text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub path: String,
    pub span: Span,
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.path, self.message)
        }
    }
}

/// A parsed, type-checked expression.
#[derive(Clone, Debug)]
pub struct Expression {
    ir: Ir,
}

impl Expression {
    /// Parses an expression that should produce `expected` (any type if
    /// `None`), fitting the result to it as GL JS does: value-typed results
    /// are asserted, strings coerced to colours.
    pub fn parse(json: &Json, expected: Option<&Type>) -> Result<Self, Vec<Diagnostic>> {
        Self::parse_with_locale(json, expected, &BUILTIN_LOCALE)
    }

    /// Parses using the host locale service, including constant folding of
    /// case mapping and number formatting. Use the same service at evaluation.
    pub fn parse_with_locale(
        json: &Json,
        expected: Option<&Type>,
        locale: &dyn Locale,
    ) -> Result<Self, Vec<Diagnostic>> {
        let mut parser = Parser::with_locale(locale);
        match parser.parse(json, "", expected, None) {
            Some(root) => Ok(Self {
                ir: parser.ir.compact(root),
            }),
            None => Err(parser.errors),
        }
    }

    /// The static type of the result.
    pub fn ty(&self) -> &Type {
        &self.ir.node(self.ir.root()).ty
    }

    pub fn is_feature_constant(&self) -> bool {
        !self.deps().contains(Deps::FEATURE)
    }

    /// Whether the expression reads the feature's geometry (`within`,
    /// `distance`), which evaluation then needs in its context.
    pub fn reads_geometry(&self) -> bool {
        self.deps().contains(Deps::GEOMETRY)
    }

    /// Whether the expression reads `image`, whose availability evaluation
    /// decides.
    pub fn reads_images(&self) -> bool {
        self.deps().contains(Deps::IMAGE)
    }

    pub(crate) fn deps(&self) -> Deps {
        self.ir.node(self.ir.root()).deps
    }

    /// The value, if the expression folded to a constant.
    pub fn constant(&self) -> Option<&Value<'static>> {
        match &self.ir.node(self.ir.root()).op {
            Op::Literal(v) => Some(v),
            _ => None,
        }
    }

    /// Evaluates at `zoom` without a feature: feature properties are
    /// absent and the id and geometry type are `null`.
    pub fn evaluate(&self, zoom: f64) -> Result<Value<'_>, EvalError> {
        self.evaluate_with(&EvaluationContext {
            zoom,
            ..EvaluationContext::default()
        })
    }

    /// Evaluates with explicit state, geometry, locale and global inputs.
    pub fn evaluate_with<'a>(
        &'a self,
        context: &EvaluationContext<'a>,
    ) -> Result<Value<'a>, EvalError> {
        Evaluator {
            ir: &self.ir,
            context,
            feature: None,
        }
        .eval_root()
    }

    /// The input stops of a top-level `interpolate`, as GL JS reads a colour
    /// ramp's `labels`.
    pub fn interpolate_labels(&self) -> Option<&[f64]> {
        match &self.ir.node(self.ir.root()).op {
            Op::Interpolate { labels, .. } => Some(labels),
            _ => None,
        }
    }

    /// Whether the root is a step ramp, requiring nearest sampling.
    pub fn is_step(&self) -> bool {
        matches!(self.ir.node(self.ir.root()).op, Op::Step { .. })
    }

    /// Binds the expression to a tile layer for evaluation per feature.
    pub fn bind<'a>(&'a self, layer: &'a orb_tile::Layer) -> Bound<'a> {
        Bound::new(&self.ir, layer)
    }

    /// The path, within the expression, of the operator that failed.
    pub fn error_path(&self, error: &EvalError) -> &str {
        &self.ir.paths[error.node as usize]
    }
}

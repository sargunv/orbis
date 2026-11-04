//! Layer filters.

use orb_tile::Layer;

use crate::legacy::{is_expression_filter, lower_filter, mixed_filter_warning};
use crate::parse::Parser;
use crate::value::{Type, Value};
use crate::{Diagnostic, Expression};

/// A compiled layer filter: an expression or a lowered legacy filter.
#[derive(Clone, Debug)]
pub struct Filter {
    expression: Expression,
}

impl Filter {
    /// Parses a filter, in expression or legacy syntax. Legacy syntax mixed
    /// into an expression filter is reported in `warnings`, as GL JS warns
    /// about it. Diagnostic paths are relative to the filter.
    pub fn parse(
        json: &crate::json::Json,
        warnings: &mut Vec<Diagnostic>,
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::parse_inner(json, warnings, None)
    }

    /// Parses with `global-state` reads folded to `state`'s values.
    pub fn parse_with_state(
        json: &crate::json::Json,
        warnings: &mut Vec<Diagnostic>,
        state: &[(String, Value<'static>)],
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::parse_inner(json, warnings, Some(state))
    }

    fn parse_inner(
        json: &crate::json::Json,
        warnings: &mut Vec<Diagnostic>,
        state: Option<&[(String, Value<'static>)]>,
    ) -> Result<Self, Vec<Diagnostic>> {
        let mut parser = Parser {
            global_state: state,
            ..Parser::default()
        };
        let root = if is_expression_filter(json) {
            warnings.extend(mixed_filter_warning(json));
            parser.parse(json, "", Some(&Type::Boolean), None)
        } else {
            lower_filter(&mut parser, json, "")
        };
        match root {
            Some(root) => Ok(Self {
                expression: Expression {
                    ir: parser.ir.compact(root),
                },
            }),
            None => Err(parser.errors),
        }
    }

    pub fn expression(&self) -> &Expression {
        &self.expression
    }

    /// The indices of the features of `layer` the filter selects at `zoom`.
    /// A feature whose evaluation fails is not selected, as in GL JS.
    pub fn select(&self, layer: &Layer, zoom: f64) -> Vec<u32> {
        if let Some(constant) = self.expression.constant() {
            return if *constant == Value::Bool(true) {
                (0..layer.len() as u32).collect()
            } else {
                Vec::new()
            };
        }
        let bound = self.expression.bind(layer);
        (0..layer.len())
            .filter(|&row| matches!(bound.evaluate(zoom, row), Ok(Value::Bool(true))))
            .map(|row| row as u32)
            .collect()
    }
}

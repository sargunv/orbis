//! The style engine's document (plan §11.2): a persistent [`StyleDoc`] that
//! holds a v8 style as JSON and as `orb-v8`'s typed IR, [`Transaction`]s
//! that edit it, and the diff of each transaction into a [`Patch`] of
//! classified [`Change`]s (S6: paint → block or table write, filter and
//! layout → relayout of one layer, source → reload).
//!
//! A transaction publishes one new snapshot ([`StyleDoc::style`]) and one
//! version. Layers and sources carry stable keys across edits, so a patch
//! names what it changes even as indices shift; a layer whose type, source
//! or source layer changes is a new layer, as GL JS's diff removes and
//! re-adds it. Feature state belongs to the map, not the document; a
//! transaction validates it and passes it through in the patch.

#![forbid(unsafe_code)]

mod cost;
mod diff;
mod doc;
mod tx;

pub use cost::CostModel;
pub use diff::{Binding, Change, Class, LayoutChange, Patch};
pub use doc::{LayerKey, SourceKey, StyleDoc};
pub use tx::{FeatureStateOp, FeatureTarget, Op, Transaction};

use orb_expr::json::{Json, JsonValue};

/// JSON equality ignoring spans: two documents mean the same value.
pub(crate) fn same(a: &Json, b: &Json) -> bool {
    match (&a.value, &b.value) {
        (JsonValue::Array(a), JsonValue::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b))
        }
        (JsonValue::Object(a), JsonValue::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| a.key == b.key && same(&a.value, &b.value))
        }
        (a, b) => a == b,
    }
}

/// [`same`] for optional members, an absent value matching only another.
pub(crate) fn same_opt(a: Option<&Json>, b: Option<&Json>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => same(a, b),
        (None, None) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests;

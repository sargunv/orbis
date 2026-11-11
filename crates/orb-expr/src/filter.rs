//! Layer filters.

use orb_tile::Layer;

use crate::legacy::{is_expression_filter, lower_filter, mixed_filter_warning};
use crate::parse::Parser;
use crate::value::{Type, Value};
use crate::{CanonicalTile, Diagnostic, Expression};

/// Property keys, sorted.
type Keys = Box<[Box<str>]>;

/// A compiled layer filter: an expression or a lowered legacy filter.
#[derive(Clone, Debug)]
pub struct Filter {
    expression: Expression,
    /// The property keys the filter reads, sorted, if they and the
    /// geometry type are all it reads of a feature (see [`FilterGroups`]).
    grouped: Option<Keys>,
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
            // Filters run per tile, which knows no camera beyond its zoom.
            Some(root) if parser.ir.node(root).deps.contains(crate::ir::Deps::CAMERA) => {
                Err(vec![Diagnostic {
                    path: String::new(),
                    span: json.span,
                    message: "camera expressions not supported".into(),
                }])
            }
            Some(root)
                if parser
                    .ir
                    .node(root)
                    .deps
                    .contains(crate::ir::Deps::DAYLIGHT) =>
            {
                Err(vec![Diagnostic {
                    path: String::new(),
                    span: json.span,
                    message: "[\"dev.sargunv.orbis:daylight\"] is not supported in filters".into(),
                }])
            }
            Some(root) => {
                let ir = parser.ir.compact(root);
                Ok(Self {
                    grouped: grouped_keys(&ir),
                    expression: Expression { ir },
                })
            }
            None => Err(parser.errors),
        }
    }

    pub fn expression(&self) -> &Expression {
        &self.expression
    }

    /// The indices of the features of `layer` the filter selects at `zoom`.
    /// A feature whose evaluation fails is not selected, as in GL JS.
    /// `tile` locates the layer's geometry for filters that read it
    /// (`within`, `distance`); without it they select nothing.
    pub fn select(&self, layer: &Layer, zoom: f64, tile: Option<CanonicalTile>) -> Vec<u32> {
        self.select_rows(layer, zoom, tile, 0..layer.len() as u32)
    }

    /// [`select`](Self::select) over `rows` of `layer` alone, in their order.
    pub fn select_rows(
        &self,
        layer: &Layer,
        zoom: f64,
        tile: Option<CanonicalTile>,
        rows: impl IntoIterator<Item = u32>,
    ) -> Vec<u32> {
        let rows = rows.into_iter().filter(|&row| (row as usize) < layer.len());
        if let Some(constant) = self.expression.constant() {
            return if *constant == Value::Bool(true) {
                rows.collect()
            } else {
                Vec::new()
            };
        }
        let bound = self.expression.bind(layer).with_tile(tile);
        rows.filter(|&row| matches!(bound.evaluate(zoom, row as usize), Ok(Value::Bool(true))))
            .collect()
    }

    /// [`select`](Self::select), evaluating a filter that reads only
    /// property keys and the geometry type once per group of `groups`
    /// rather than once per feature. `groups` holds `layer`'s groups for
    /// the filters of other style layers over it.
    pub fn select_grouped(
        &self,
        layer: &Layer,
        zoom: f64,
        tile: Option<CanonicalTile>,
        groups: &mut FilterGroups,
    ) -> Vec<u32> {
        let grouping = match &self.grouped {
            Some(keys) if self.expression.constant().is_none() => groups.of(layer, keys),
            _ => None,
        };
        let Some(grouping) = grouping else {
            return self.select(layer, zoom, tile);
        };
        let bound = self.expression.bind(layer);
        let selects: Vec<bool> = (grouping.first.iter())
            .map(|&row| matches!(bound.evaluate(zoom, row as usize), Ok(Value::Bool(true))))
            .collect();
        (0..layer.len() as u32)
            .filter(|&row| selects[grouping.of_row[row as usize] as usize])
            .collect()
    }
}

/// The property keys a filter reads, sorted, if they and the geometry type
/// are all it reads of a feature: not its id, its geometry or keys it
/// computes.
fn grouped_keys(ir: &crate::ir::Ir) -> Option<Keys> {
    use crate::ir::{Deps, LegacyFilter, Op, Subject};
    let groupable = ir.nodes.iter().all(|n| match &n.op {
        Op::Id | Op::Properties | Op::GetDynamic(_) | Op::HasDynamic(_) => false,
        Op::Legacy(
            LegacyFilter::Compare { subject, .. }
            | LegacyFilter::In { subject, .. }
            | LegacyFilter::Has(subject),
        ) => *subject != Subject::Id,
        _ => !n.deps.contains(Deps::GEOMETRY),
    });
    groupable.then(|| {
        let mut keys = ir.keys.clone();
        keys.sort_unstable();
        keys.dedup();
        keys.into_boxed_slice()
    })
}

/// A tile layer's features in groups that a filter reading only some
/// property keys and the geometry type can't tell apart: each group's
/// features have the same geometry type and the same values under those
/// keys. The filters of a tile layer's many style layers (a road network's
/// dozens) share them, so each filter evaluates once per group.
#[derive(Debug, Default)]
pub struct FilterGroups {
    /// Per set of keys: its groups, or none when nearly every feature is a
    /// group of its own.
    sets: Vec<(Keys, Option<Grouping>)>,
}

#[derive(Debug)]
struct Grouping {
    /// Each feature's group.
    of_row: Vec<u32>,
    /// Each group's first feature.
    first: Vec<u32>,
}

impl FilterGroups {
    /// `layer`'s groups under `keys`, grouped on first use.
    fn of(&mut self, layer: &Layer, keys: &[Box<str>]) -> Option<&Grouping> {
        let at = match self.sets.iter().position(|(k, _)| **k == *keys) {
            Some(at) => at,
            None => {
                self.sets.push((keys.into(), Grouping::new(layer, keys)));
                self.sets.len() - 1
            }
        };
        self.sets[at].1.as_ref()
    }
}

impl Grouping {
    /// Groups `layer`'s features by geometry type and value codes under
    /// `keys`, or none if there would be more than a quarter as many groups
    /// as features: evaluating each feature is then as cheap.
    fn new(layer: &Layer, keys: &[Box<str>]) -> Option<Self> {
        let columns: Vec<&orb_tile::Column> =
            keys.iter().filter_map(|k| layer.property(k)).collect();
        // A word of which columns are present and the geometry type, then
        // the present values' codes.
        if columns.len() > 60 {
            return None;
        }
        let limit = layer.len() / 4 + 8;
        // Fx rather than SipHash: several times faster on short keys, and
        // codes are no denial-of-service vector.
        let mut index: rustc_hash::FxHashMap<Box<[u64]>, u32> = Default::default();
        let mut key = Vec::with_capacity(columns.len() + 1);
        let mut of_row = Vec::with_capacity(layer.len());
        let mut first = Vec::new();
        for row in 0..layer.len() {
            key.clear();
            let mut head = layer.geometry_type(row) as u64;
            for (i, column) in columns.iter().enumerate() {
                if let Some(code) = column.code(row) {
                    head |= 1 << (i + 2);
                    key.push(code);
                }
            }
            key.push(head);
            let group = match index.get(key.as_slice()) {
                Some(&group) => group,
                None => {
                    if first.len() == limit {
                        return None;
                    }
                    let group = first.len() as u32;
                    index.insert(key.as_slice().into(), group);
                    first.push(row as u32);
                    group
                }
            };
            of_row.push(group);
        }
        Some(Self { of_row, first })
    }
}

#[cfg(test)]
mod tests {
    use orb_tile::{GeometryType, LayerBuilder, Value};

    use super::*;

    /// A road network: few distinct values over many features, a key on
    /// some of them, a column of mixed types and ids.
    fn roads() -> Layer {
        let mut b = LayerBuilder::new("transportation", 4096);
        let classes = ["motorway", "primary", "minor", "path"];
        let types = [GeometryType::LineString, GeometryType::Polygon];
        for i in 0..400u64 {
            b.begin_feature(Some(i), types[usize::from(i % 7 == 0)]);
            b.begin_part();
            b.begin_ring();
            b.push_vertex([0, 0]);
            b.push_vertex([1, 1]);
            b.set_property("class", Value::Str(classes[i as usize % 4].into()));
            if i % 3 == 0 {
                b.set_property("brunnel", Value::Str("bridge".into()));
            }
            b.set_property("layer", Value::I64(i as i64 % 5 - 2));
            b.set_property("ramp", Value::Bool(i % 11 == 0));
            let mixed = if i % 2 == 0 {
                Value::I64(1)
            } else {
                Value::Str("1".into())
            };
            b.set_property("mixed", mixed);
        }
        b.finish()
    }

    /// Filters select the same features in groups as one by one, those
    /// that read ids or computed keys included.
    #[test]
    fn grouped_selection_matches_each_feature() {
        let layer = roads();
        let mut groups = FilterGroups::default();
        for text in [
            r#"["==", ["get", "class"], "motorway"]"#,
            r#"["all", ["==", "$type", "LineString"], ["!=", "brunnel", "bridge"]]"#,
            r#"["match", ["get", "class"], ["primary", "minor"], ["<", ["get", "layer"], 0], false]"#,
            r#"["any", ["get", "ramp"], ["has", "brunnel"]]"#,
            r#"["==", ["get", "mixed"], 1]"#,
            r#"["<", ["id"], 30]"#,
            r#"["==", ["get", ["concat", "cl", "ass"]], "path"]"#,
            r#"[">=", ["zoom"], 12]"#,
        ] {
            let json = crate::json::parse(text).unwrap();
            let filter = Filter::parse(&json, &mut Vec::new()).unwrap();
            let one_by_one = filter.select(&layer, 10.0, None);
            assert!(!one_by_one.is_empty() || text.contains("zoom"), "{text}");
            let grouped = filter.select_grouped(&layer, 10.0, None, &mut groups);
            assert_eq!(grouped, one_by_one, "{text}");
        }
        let groups_of = |key: &str| {
            let (_, grouping) = (groups.sets.iter())
                .find(|(k, _)| **k == [Box::from(key)])
                .expect("grouped");
            grouping.as_ref().map(|g| g.first.len())
        };
        // 400 roads in 8 groups: class by geometry type.
        assert_eq!(groups_of("class"), Some(8));
        // A column of mixed types gives each feature its own group: none.
        assert_eq!(groups_of("mixed"), None);
    }
}

//! Layer filters.

use orb_tile::Layer;

use crate::ir::{Call, Deps, Ir, NodeId, Op};
use crate::legacy::{is_expression_filter, lower_filter, mixed_filter_warning};
use crate::parse::Parser;
use crate::value::{Type, Value};
use crate::{CanonicalTile, Diagnostic, Expression};

/// Property keys, sorted.
type Keys = Box<[Box<str>]>;

/// A compiled layer filter: an expression or a lowered legacy filter.
///
/// A symbol layer's filter may read the camera (`["pitch"]`,
/// `["bearing"]`) and each symbol's distance from the view's centre
/// (`["dev.sargunv.orbis:distance-from-center"]`), which a tile, laid out
/// on a job worker, doesn't know. Such a filter splits in two: the
/// conditions of its top-level `all` that read none of them select a
/// tile's features as any filter does, and the others
/// ([`CameraFilter`]) decide per symbol at each placement.
#[derive(Clone, Debug)]
pub struct Filter {
    /// What a tile selects its features by.
    expression: Expression,
    /// The property keys the filter reads, sorted, if they and the
    /// geometry type are all it reads of a feature (see [`FilterGroups`]).
    grouped: Option<Keys>,
    camera: Option<CameraFilter>,
}

/// The conditions of a symbol layer's filter that read the camera or the
/// symbol's distance from the view's centre, which placement evaluates
/// per symbol: a symbol they reject is not placed, so it neither shows nor
/// collides.
#[derive(Clone, Debug)]
pub struct CameraFilter {
    expression: Expression,
}

/// The style layer a filter selects features for, which decides whether
/// it may read the camera.
#[derive(Clone, Copy, Debug)]
pub enum FilterLayer<'a> {
    /// A symbol layer's, whose camera conditions decide per symbol at
    /// placement ([`Filter::camera`]).
    Symbol,
    /// Another layer's, which runs per tile and knows no camera beyond
    /// its zoom: its type's v8 name and its id, for messages.
    Other { ty: &'a str, id: &'a str },
}

impl CameraFilter {
    /// The conditions, as one expression.
    pub fn expression(&self) -> &Expression {
        &self.expression
    }

    /// Whether they read the feature (its properties, id or geometry
    /// type), which placement then needs captured
    /// ([`FeatureData::capture`](crate::FeatureData::capture)).
    pub fn reads_feature(&self) -> bool {
        self.expression.deps().contains(Deps::FEATURE)
    }

    /// Whether they read the camera's pitch or bearing.
    pub fn reads_camera(&self) -> bool {
        self.expression.deps().contains(Deps::CAMERA)
    }

    /// Whether they read the symbol's distance from the view's centre, so
    /// that each symbol decides apart rather than each feature.
    pub fn reads_center_distance(&self) -> bool {
        self.expression.deps().contains(Deps::CENTER_DISTANCE)
    }

    /// Whether a symbol passes: `context` holds the zoom the tile was
    /// laid out at, the camera and the symbol's distance from the view's
    /// centre; `data` its feature's, when [`reads_feature`](Self::reads_feature).
    /// A failing evaluation rejects it, as a filter's does.
    pub fn passes(
        &self,
        context: &crate::EvaluationContext<'_>,
        data: Option<&crate::FeatureData>,
    ) -> bool {
        let value = match data {
            Some(data) => self.expression.evaluate_data(context, data),
            None => self.expression.evaluate_with(context),
        };
        matches!(value, Ok(Value::Bool(true)))
    }
}

/// Whether `deps` read what only placement knows.
fn reads_camera(deps: Deps) -> bool {
    deps.contains(Deps::CAMERA) || deps.contains(Deps::CENTER_DISTANCE)
}

/// The conditions `id` holds to: the operands of nested `all`s, in order.
fn conjuncts(ir: &Ir, id: NodeId, out: &mut Vec<NodeId>) {
    match &ir.node(id).op {
        Op::All(ids) => ids.iter().for_each(|&c| conjuncts(ir, c, out)),
        _ => out.push(id),
    }
}

/// A node that holds when all of `ids` do.
fn conjunction(ir: &mut Ir, ids: &[NodeId]) -> NodeId {
    match ids {
        [] => ir.push(Op::Literal(Value::Bool(true)), Type::Boolean, ""),
        [one] => *one,
        many => ir.push(Op::All(many.to_vec()), Type::Boolean, ""),
    }
}

/// The name of the first camera operator `ir` reads.
fn camera_operator(ir: &Ir) -> &'static str {
    ir.nodes
        .iter()
        .find_map(|n| match n.op {
            Op::Call(Call::Pitch, _) => Some("pitch"),
            Op::Call(Call::Bearing, _) => Some("bearing"),
            Op::Call(Call::CenterDistance, _) => Some(crate::parse::CENTER_DISTANCE),
            _ => None,
        })
        .unwrap_or("pitch")
}

impl Filter {
    /// Parses a feature query's filter, in expression or legacy syntax.
    /// Legacy syntax mixed into an expression filter is reported in
    /// `warnings`, as GL JS warns about it. Diagnostic paths are relative
    /// to the filter. A filter that reads the camera is an error: only a
    /// symbol layer's may ([`parse_layer`](Self::parse_layer)).
    pub fn parse(
        json: &crate::json::Json,
        warnings: &mut Vec<Diagnostic>,
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::parse_inner(json, warnings, None, None)
    }

    /// Parses `layer`'s filter as [`parse`](Self::parse) does, with
    /// `global-state` reads folded to `state`'s values. A symbol layer's
    /// may read the camera and each symbol's distance from the view's
    /// centre ([`camera`](Self::camera)).
    pub fn parse_layer(
        json: &crate::json::Json,
        warnings: &mut Vec<Diagnostic>,
        layer: FilterLayer<'_>,
        state: &[(String, Value<'static>)],
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::parse_inner(json, warnings, Some(state), Some(layer))
    }

    /// `layer` is none for a query's filter.
    fn parse_inner(
        json: &crate::json::Json,
        warnings: &mut Vec<Diagnostic>,
        state: Option<&[(String, Value<'static>)]>,
        layer: Option<FilterLayer<'_>>,
    ) -> Result<Self, Vec<Diagnostic>> {
        let mut parser = Parser {
            global_state: state,
            in_filter: true,
            ..Parser::default()
        };
        let root = if is_expression_filter(json) {
            warnings.extend(mixed_filter_warning(json));
            parser.parse(json, "", Some(&Type::Boolean), None)
        } else {
            lower_filter(&mut parser, json, "")
        };
        let Some(root) = root else {
            return Err(parser.errors);
        };
        let error = |message: String| {
            Err(vec![Diagnostic {
                path: String::new(),
                span: json.span,
                message,
            }])
        };
        let (root, camera) = if !reads_camera(parser.ir.node(root).deps) {
            (root, None)
        } else {
            // Only a symbol layer's filter decides at placement; the rest
            // run per tile, which knows no camera beyond its zoom.
            let site = match layer {
                Some(FilterLayer::Symbol) => None,
                // Mapbox GL JS's wording.
                Some(FilterLayer::Other { ty, id }) => {
                    Some(format!("a filter for a {ty} layer with id: {id}"))
                }
                None => Some("a query's filter".into()),
            };
            if let Some(site) = site {
                let name = camera_operator(&parser.ir);
                return error(format!(
                    "[\"{name}\"] expression is not supported in {site}"
                ));
            }
            let mut all = Vec::new();
            conjuncts(&parser.ir, root, &mut all);
            let (camera, tile): (Vec<NodeId>, Vec<NodeId>) =
                (all.into_iter()).partition(|&c| reads_camera(parser.ir.node(c).deps));
            // Placement has the feature's data but not its geometry.
            if camera
                .iter()
                .any(|&c| parser.ir.node(c).deps.contains(Deps::GEOMETRY))
            {
                return error(
                    "a filter condition that reads the camera can't read the feature's geometry (\"within\", \"distance\"); give those conditions of their own under \"all\"".into(),
                );
            }
            let camera = conjunction(&mut parser.ir, &camera);
            let camera = CameraFilter {
                expression: Expression {
                    ir: parser.ir.compact(camera),
                },
            };
            (conjunction(&mut parser.ir, &tile), Some(camera))
        };
        let ir = parser.ir.compact(root);
        Ok(Self {
            grouped: grouped_keys(&ir),
            expression: Expression { ir },
            camera,
        })
    }

    /// The conditions placement decides per symbol, for a symbol layer's
    /// filter that reads the camera.
    pub fn camera(&self) -> Option<&CameraFilter> {
        self.camera.as_ref()
    }

    /// The filter without its [`camera`](Self::camera) conditions: what a
    /// tile selects by alone.
    #[must_use]
    pub fn without_camera(self) -> Self {
        Self {
            camera: None,
            ..self
        }
    }

    /// What a tile selects its features by: the whole filter, or for one
    /// that reads the camera, its conditions that don't.
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

    fn symbol_filter(text: &str) -> Result<Filter, Vec<Diagnostic>> {
        let json = crate::json::parse(text).unwrap();
        Filter::parse_layer(&json, &mut Vec::new(), FilterLayer::Symbol, &[])
    }

    fn at(pitch: f64, center_distance: f64) -> crate::EvaluationContext<'static> {
        crate::EvaluationContext {
            zoom: 10.0,
            pitch,
            center_distance,
            ..Default::default()
        }
    }

    /// A symbol filter's conditions that read no camera select a tile's
    /// features; the others decide per symbol at placement.
    #[test]
    fn symbol_filters_split_off_their_camera_conditions() {
        let layer = roads();
        let filter = symbol_filter(
            r#"["all", ["==", ["get", "class"], "motorway"],
                ["all", ["<", ["pitch"], 45], ["!", ["get", "ramp"]]],
                ["<", ["dev.sargunv.orbis:distance-from-center"], 1]]"#,
        )
        .unwrap();
        let tile = symbol_filter(
            r#"["all", ["==", ["get", "class"], "motorway"], ["!", ["get", "ramp"]]]"#,
        )
        .unwrap();
        assert_eq!(
            filter.select(&layer, 10.0, None),
            tile.select(&layer, 10.0, None)
        );
        assert!(tile.camera().is_none());
        let camera = filter.camera().expect("reads the camera");
        assert!(camera.reads_center_distance() && !camera.reads_feature());
        assert!(camera.passes(&at(30.0, 0.5), None));
        assert!(camera.passes(&at(30.0, -3.0), None));
        assert!(!camera.passes(&at(50.0, 0.5), None));
        assert!(!camera.passes(&at(30.0, 1.0), None));
    }

    /// A camera condition may read the feature, captured for placement,
    /// and the tile's zoom; a filter that only reads the camera selects
    /// every feature of a tile.
    #[test]
    fn camera_conditions_read_the_feature_and_zoom() {
        let layer = roads();
        let filter = symbol_filter(
            r#"["case", [">=", ["zoom"], 12], true,
                [">", ["dev.sargunv.orbis:distance-from-center"], ["get", "reach"]]]"#,
        )
        .unwrap();
        assert_eq!(filter.select(&layer, 10.0, None).len(), layer.len());
        let camera = filter.camera().unwrap();
        assert!(camera.reads_feature() && camera.reads_center_distance());
        let data = crate::FeatureData {
            id: crate::Value::Null,
            geometry_type: None,
            properties: vec![("reach".into(), crate::Value::Number(-0.5))],
        };
        assert!(camera.passes(&at(0.0, 0.0), Some(&data)));
        assert!(!camera.passes(&at(0.0, -1.0), Some(&data)));
        // Without the property the comparison fails, which rejects.
        assert!(!camera.passes(&at(0.0, 0.0), None));
        let zoomed = crate::EvaluationContext {
            zoom: 12.0,
            ..at(0.0, -1.0)
        };
        assert!(camera.passes(&zoomed, Some(&data)));
    }

    /// Only a symbol layer's filter reads the camera; the distance from the
    /// view's centre is nothing else's to read.
    #[test]
    fn camera_reads_outside_symbol_filters_are_errors() {
        let circle = FilterLayer::Other {
            ty: "circle",
            id: "c",
        };
        let message = |text: &str, layer: Option<FilterLayer<'_>>| {
            let json = crate::json::parse(text).unwrap();
            let parsed = match layer {
                Some(layer) => Filter::parse_layer(&json, &mut Vec::new(), layer, &[]),
                None => Filter::parse(&json, &mut Vec::new()),
            };
            parsed.unwrap_err().remove(0).message
        };
        let pitch = r#"["<", ["pitch"], 30]"#;
        let far = r#"["<", ["dev.sargunv.orbis:distance-from-center"], 1]"#;
        assert_eq!(
            message(pitch, None),
            "[\"pitch\"] expression is not supported in a query's filter"
        );
        assert_eq!(
            message(far, Some(circle)),
            "[\"dev.sargunv.orbis:distance-from-center\"] expression is not supported in a filter for a circle layer with id: c"
        );
        let spec = crate::PropertySpec {
            ty: crate::PropertyType::Number,
            default: None,
            inputs: crate::Inputs::ZOOM
                .union(crate::Inputs::DATA)
                .union(crate::Inputs::STATE)
                .union(crate::Inputs::CAMERA_DATA),
            interpolated: true,
            tokens: false,
        };
        let json = crate::json::parse(r#"["dev.sargunv.orbis:distance-from-center"]"#).unwrap();
        assert!(crate::PropertyExpression::parse(&json, &spec).is_err());
        // Placement captures no geometry.
        let polygon = r#"{"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}"#;
        let within = format!(r#"["any", ["within", {polygon}], ["<", ["pitch"], 30]]"#);
        let errors = symbol_filter(&within).unwrap_err();
        assert!(errors[0].message.contains("geometry"), "{errors:?}");
        let apart = format!(r#"["all", ["within", {polygon}], ["<", ["pitch"], 30]]"#);
        assert!(symbol_filter(&apart).is_ok());
        assert!(symbol_filter(r#"["dev.sargunv.orbis:distance-from-center", 1]"#).is_err());
    }
}

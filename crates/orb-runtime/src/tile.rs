//! The tile job: a decoded tile in, one mesh and property rows per render
//! node out. Pure, so it runs on any job thread or worker.

use std::collections::HashMap;

use orb_layout::{Mesh, extrusion, fill, line, point};
use orb_render::{Family, pack_color};
use orb_tile::{Layer, TileFrame, ValueRef};
use orb_v8::{Kind, LayerKind, SourceKind};

use crate::glyphs::GlyphCatalog;
use crate::style::{NodePlan, PaintProperty, StylePlan};
use crate::symbol::{SpriteIcons, SymbolBuild};

/// A laid-out tile in core memory, from [`TileWorker::build`](crate::TileWorker::build).
#[derive(Debug, Default)]
pub struct TileBuild {
    /// Source diagnostics produced by this job, including GeoJSON property
    /// conversions. Carried through packed results to the map's warnings.
    pub warnings: Vec<String>,
    /// The style revision the tile was laid out with.
    pub(crate) revision: u64,
    /// Keys of the nodes laid out; those without a mesh have no features.
    pub(crate) nodes: Vec<u32>,
    /// Only property rows: meshes have no geometry.
    pub(crate) rows_only: bool,
    /// Feature ids, as GL JS keys feature state, that meshes refer to.
    pub(crate) ids: Vec<String>,
    pub(crate) meshes: Vec<NodeMesh>,
    /// Per symbol layer laid out, by layer key: its bucket. Its icon and
    /// text nodes' meshes carry property rows only.
    pub(crate) symbols: Vec<(u32, SymbolBuild)>,
}

/// What symbol layout reads beyond the tile: the worker's glyph metrics
/// and sprite metadata.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Symbols<'a> {
    pub(crate) glyphs: &'a GlyphCatalog,
    pub(crate) icons: &'a SpriteIcons,
    pub(crate) words: WordData<'a>,
    /// The map's `localIdeographFontFamily`.
    pub(crate) ideographs: Option<&'a str>,
}

/// The Southeast Asian word data a build has, or where to fetch it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum WordData<'a> {
    Ready(&'a orb_text::WordBreaker),
    Fetch(&'a str),
    Unavailable,
}

/// Which of a source's nodes to lay out.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Select<'a> {
    /// Node keys; `None` for every node of the source.
    pub(crate) nodes: Option<&'a [u32]>,
    /// Property rows (with their images and feature ids) only.
    pub(crate) rows_only: bool,
}

/// One render node's share of a tile.
#[derive(Debug)]
pub(crate) struct NodeMesh {
    /// The node's key.
    pub(crate) node: u32,
    pub(crate) hits: Vec<crate::HitFeature>,
    /// Data units per tile side.
    pub(crate) extent: u32,
    pub(crate) mesh: Mesh,
    /// Property rows, one per selected feature.
    pub(crate) props: Vec<u32>,
    /// Words per row: the family's, then the node's input columns.
    pub(crate) row_words: u32,
    /// Per paint property, the largest number in the rows (0 for colours
    /// and properties not in the rows), for culling margins; then per input
    /// column of the node's bytecode, its largest and negated smallest
    /// number, for [`orb_expr::Expression::bound`].
    pub(crate) maxima: Vec<f32>,
    pub(crate) images: Vec<ImageRef>,
    /// For nodes that read feature state: per feature slot, its id as an
    /// index into [`TileBuild::ids`], or [`NO_ID`].
    pub(crate) state_ids: Option<Vec<u32>>,
}

/// A feature without an id: it keeps the default state.
pub(crate) const NO_ID: u32 = u32::MAX;

/// A property word and typed texture key to resolve on the scene side. Jobs
/// need no atlas allocation; generated ramps travel with the key across workers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRef {
    pub word: u32,
    pub key: orb_render::ImageKey,
}

/// The least f32 at or above `x`.
fn up(x: f64) -> f32 {
    let f = x as f32;
    if f64::from(f) < x { f.next_up() } else { f }
}

/// The feature-state key of a value: GL JS's `String(id)`.
fn id_string(value: ValueRef<'_>) -> String {
    match value {
        ValueRef::Str(s) | ValueRef::Json(s) => s.to_string(),
        ValueRef::Bool(b) => b.to_string(),
        ValueRef::I64(n) => orb_expr::js_number_string(n as f64),
        ValueRef::U64(n) => orb_expr::js_number_string(n as f64),
        ValueRef::F64(n) => orb_expr::js_number_string(n),
    }
}

/// The features a node draws from `layer`, in slot order.
fn features(plan: &StylePlan, node: &NodePlan, layer: &Layer, zoom: f64) -> Vec<u32> {
    let style_layer = &plan.style.layers[node.layer];
    let selection = match &style_layer.kind {
        LayerKind::Fill(fill) => &fill.selection,
        LayerKind::FillExtrusion(e) => &e.selection,
        LayerKind::Line(line) => &line.selection,
        LayerKind::Circle(c) => &c.selection,
        LayerKind::Heatmap(h) => &h.selection,
        LayerKind::Symbol(s) => &s.selection,
        _ => unreachable!("vector nodes have feature selections"),
    };
    let mut features: Vec<u32> = match &selection.filter {
        Some(filter) => filter.select(layer, zoom),
        None => (0..layer.len() as u32).collect(),
    };
    if let LayerKind::Circle(c) = &style_layer.kind {
        let key = c.layout.sort_key.bind(layer);
        // Stable order for equal keys, just like JS's Array.sort.
        features.sort_by(|&a, &b| {
            key.evaluate(zoom, a as usize)
                .unwrap_or(0.0)
                .total_cmp(&key.evaluate(zoom, b as usize).unwrap_or(0.0))
        });
    }
    features
}

/// A node's geometry over `features`.
fn layout(layer: &Layer, features: &[u32], kind: &LayerKind, family: Family, zoom: f64) -> Mesh {
    match kind {
        LayerKind::Fill(_) => match family {
            Family::FillOutline => fill::outline(layer, features),
            _ => fill::layout(layer, features),
        },
        LayerKind::FillExtrusion(_) => extrusion::layout_in_extent(layer, features, 8192),
        LayerKind::Circle(c) => {
            let key = c.layout.sort_key.bind(layer);
            point::layout_sorted(layer, features, |feature| {
                key.evaluate(zoom, feature as usize)
            })
        }
        LayerKind::Heatmap(_) => point::layout(layer, features),
        LayerKind::Line(l) => {
            let join = l.layout.join.bind(layer);
            let cap = l.layout.cap.bind(layer);
            let miter_limit = l.layout.miter_limit.bind(layer);
            let round_limit = l.layout.round_limit.bind(layer);
            let params = |feature: u32| line::LineLayout {
                join: match join.evaluate(zoom, feature as usize) {
                    orb_v8::LineJoin::Bevel => line::LineJoin::Bevel,
                    orb_v8::LineJoin::Round => line::LineJoin::Round,
                    orb_v8::LineJoin::Miter => line::LineJoin::Miter,
                },
                cap: match cap.evaluate(zoom, feature as usize) {
                    orb_v8::LineCap::Butt => line::LineCap::Butt,
                    orb_v8::LineCap::Round => line::LineCap::Round,
                    orb_v8::LineCap::Square => line::LineCap::Square,
                },
                miter_limit: miter_limit.evaluate(zoom, feature as usize),
                round_limit: round_limit.evaluate(zoom, feature as usize),
            };
            line::layout_in_extent(layer, features, params, 8192)
        }
        _ => unreachable!("vector nodes have feature selections"),
    }
}

/// The tile layer a node draws from, if the node lays out at zoom `z`
/// (GL JS's bucket rule: nothing outside the layer's zoom range).
fn node_layer<'f, 'p>(
    frame: &'f TileFrame,
    plan: &'p StylePlan,
    node: &NodePlan,
    zoom: f64,
) -> Option<(&'f Layer, &'p str)> {
    let style_layer = &plan.style.layers[node.layer];
    if zoom < style_layer.minzoom.floor() || zoom >= style_layer.maxzoom {
        return None;
    }
    let name = match &plan.source_def(node.source).kind {
        SourceKind::GeoJson(_) => Some(orb_tile::geojson::LAYER_NAME),
        _ => plan.source_layer(node),
    }?;
    Some((frame.layer(name)?, name))
}

/// Whether a symbol layer's bucket is needed: either of its nodes is.
fn wants_symbols(select: Select<'_>, icon_node: u32) -> bool {
    !select.rows_only
        && select
            .nodes
            .is_none_or(|n| n.contains(&icon_node) || n.contains(&(icon_node + 1)))
}

/// The column of stable ids symbol keys use: the promoted id, when the
/// source has `promoteId` (GL JS `keyWithFeatureId`).
fn stable_ids<'l>(source: &orb_v8::Source, layer: &'l Layer) -> Option<&'l orb_tile::Column> {
    match &source.kind {
        SourceKind::Vector(v) => v
            .promote_id
            .as_ref()
            .and_then(|p| p.key(layer.name()))
            .and_then(|key| layer.property(key)),
        SourceKind::GeoJson(g) if g.options.promote_id.is_some() => Some(layer.ids()),
        _ => None,
    }
}

/// Lays out the symbol layers of `source` the selection needs. Returns the
/// glyph range URLs this worker must fetch first, if any.
fn build_symbols(
    frame: &TileFrame,
    plan: &StylePlan,
    source: usize,
    zoom: f64,
    select: Select<'_>,
    symbols: Symbols<'_>,
    warnings: &mut Vec<String>,
) -> Result<Vec<(u32, SymbolBuild)>, Vec<String>> {
    let template = plan.style.glyphs.as_deref();
    let policy = orb_text::FontPolicy {
        glyphs_url: template.is_some(),
        local_ideographs: symbols.ideographs.is_some_and(|f| !f.is_empty()),
    };
    let source_def = plan.source_def(source);
    let mut prepared = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for node in &plan.nodes {
        if node.source != source
            || node.family != Family::SymbolIcon
            || !wants_symbols(select, node.key)
        {
            continue;
        }
        let LayerKind::Symbol(symbol) = &plan.style.layers[node.layer].kind else {
            unreachable!("symbol nodes are of symbol layers")
        };
        let Some((layer, _)) = node_layer(frame, plan, node, zoom) else {
            continue;
        };
        let features = features(plan, node, layer, zoom);
        if features.is_empty() {
            continue;
        }
        let ids = stable_ids(source_def, layer);
        let p = crate::symbol::prepare(symbol, layer, &features, zoom, ids);
        let faces = &plan.style.font_faces;
        for url in symbols.glyphs.missing(&p.requests, template, policy, faces) {
            if !missing.contains(&url) {
                missing.push(url);
            }
        }
        if let WordData::Fetch(url) = symbols.words
            && !missing.iter().any(|m| m == url)
            && crate::symbol::needs_word_breaks(&p)
        {
            missing.push(url.to_string());
        }
        prepared.push((node, layer, features, p));
    }
    if !missing.is_empty() {
        return Err(missing);
    }
    let mut built = Vec::with_capacity(prepared.len());
    for (node, layer, features, mut p) in prepared {
        let requests = std::mem::take(&mut p.requests);
        let resolved = symbols.glyphs.resolve(
            requests,
            template,
            policy,
            symbols.ideographs,
            &plan.style.font_faces,
        );
        if resolved.local > 0 && symbols.glyphs.warn_local() {
            warnings.push(
                "glyphs that need local fonts (clusters, complex scripts, ideographs or a missing range) are not drawn: this host has no local fonts".into(),
            );
        }
        let slots: HashMap<u32, u32> = features
            .iter()
            .enumerate()
            .map(|(slot, &f)| (f, slot as u32))
            .collect();
        let layer_key = node.key / 2;
        built.push((
            layer_key,
            crate::symbol::layout(
                p,
                layer,
                layer.name(),
                resolved.glyphs,
                symbols.icons,
                &slots,
                match symbols.words {
                    WordData::Ready(breaker) => Some(breaker),
                    _ => None,
                },
            ),
        ));
    }
    Ok(built)
}

/// Lays out the selected nodes of `source` that draw from `frame`, at tile
/// zoom `z`. Returns the glyph range URLs this worker must fetch first when
/// the tile's labels need ranges it hasn't got.
pub(crate) fn build(
    frame: &TileFrame,
    plan: &StylePlan,
    source: usize,
    z: u8,
    select: Select<'_>,
    symbols: Symbols<'_>,
) -> Result<TileBuild, Vec<String>> {
    let zoom = f64::from(z);
    let mut build = TileBuild {
        revision: plan.revision,
        rows_only: select.rows_only,
        ..TileBuild::default()
    };
    build.symbols = build_symbols(
        frame,
        plan,
        source,
        zoom,
        select,
        symbols,
        &mut build.warnings,
    )?;
    let mut ids: HashMap<String, u32> = HashMap::new();
    let source_def = plan.source_def(source);
    for node in &plan.nodes {
        if node.source != source || select.nodes.is_some_and(|n| !n.contains(&node.key)) {
            continue;
        }
        build.nodes.push(node.key);
        let style_layer = &plan.style.layers[node.layer];
        let Some((layer, name)) = node_layer(frame, plan, node, zoom) else {
            continue;
        };
        let features = features(plan, node, layer, zoom);
        if features.is_empty() {
            continue;
        }
        // The family's words, then the node's input columns.
        let family_words = plan.row_words(node)
            - node
                .gpu
                .as_ref()
                .map_or(0, |g| g.layout.column_words() as usize);
        let row_words = plan.row_words(node);
        let bytecode = plan.bytecode_mask(node);
        let mesh = match node.family {
            _ if select.rows_only => Mesh::default(),
            // Symbol geometry is the layer's bucket; its nodes carry rows.
            family @ (Family::SymbolIcon | Family::SymbolText) => {
                let bucket = build.symbols.iter().find(|(k, _)| *k == node.key / 2);
                let stream = bucket.map(|(_, b)| match family {
                    Family::SymbolIcon => &b.bucket.icon,
                    _ => &b.bucket.text,
                });
                if stream.is_none_or(|s| s.quads.is_empty()) {
                    continue;
                }
                Mesh::default()
            }
            _ => {
                let mesh = layout(layer, &features, &style_layer.kind, node.family, zoom);
                if mesh.is_empty() {
                    continue;
                }
                mesh
            }
        };

        let properties = plan.paint_properties(node);
        let mut maxima = vec![0.0f32; properties.len()];
        let pattern = plan
            .pattern(node)
            .filter(|p| matches!(p.kind(), Kind::Source | Kind::Composite));
        let line_effects = matches!(&style_layer.kind, LayerKind::Line(l) if l.paint.dasharray.as_constant() != Some(&None) || l.paint.gradient.is_some());
        let per_feature = line_effects
            || pattern.is_some()
            || bytecode != 0
            || properties.iter().any(PaintProperty::per_feature);
        let props = if per_feature {
            let mut rows = vec![0u32; features.len() * row_words];
            if let Some(gpu) = &node.gpu {
                let layout = &gpu.layout;
                layout.pack_columns(layer, &features, row_words, family_words, &mut rows);
                // Each column's range, as a maximum and a negated minimum,
                // for the culling bounds of interpreted widths.
                for [lo, hi] in layout.ranges(layer, &features) {
                    maxima.extend([up(hi), up(-lo)]);
                }
            }
            for (slot, property) in properties.iter().enumerate() {
                if !property.per_feature() || bytecode & 1 << slot != 0 {
                    continue;
                }
                let composite = property.kind() == Kind::Composite;
                let next = zoom + 1.0;
                let mut write = |i: usize, words: [u32; 2]| {
                    let at = i * row_words + slot * 2;
                    rows[at..at + 2].copy_from_slice(&words);
                };
                match property {
                    PaintProperty::Color(p) => {
                        let bound = p.bind(layer);
                        for (i, &feature) in features.iter().enumerate() {
                            let start = bound.evaluate(zoom, feature as usize).premultiplied();
                            let end = if composite {
                                bound.evaluate(next, feature as usize).premultiplied()
                            } else {
                                start
                            };
                            write(i, [pack_color(start), pack_color(end)]);
                        }
                    }
                    PaintProperty::OptionalColor(p) => {
                        let bound = p.bind(layer);
                        let color = |zoom, feature: u32| {
                            bound
                                .evaluate(zoom, feature as usize)
                                .map_or([0.0; 4], |c| c.premultiplied())
                        };
                        for (i, &feature) in features.iter().enumerate() {
                            let start = color(zoom, feature);
                            let end = if composite {
                                color(next, feature)
                            } else {
                                start
                            };
                            write(i, [pack_color(start), pack_color(end)]);
                        }
                    }
                    PaintProperty::Number(p) => {
                        let bound = p.bind(layer);
                        for (i, &feature) in features.iter().enumerate() {
                            let start = bound.evaluate(zoom, feature as usize);
                            let end = if composite {
                                bound.evaluate(next, feature as usize)
                            } else {
                                start
                            };
                            maxima[slot] = maxima[slot].max(start).max(end);
                            write(i, [start.to_bits(), end.to_bits()]);
                        }
                    }
                }
            }
            rows
        } else {
            Vec::new()
        };

        let mut props = props;
        let mut images = Vec::new();
        if let Some(pattern) = pattern {
            let bound = pattern.bind(layer);
            for (slot, &feature) in features.iter().enumerate() {
                for (j, z) in [zoom - 1.0, zoom, zoom + 1.0].into_iter().enumerate() {
                    if let Some(name) = bound.evaluate(z, feature as usize) {
                        images.push(ImageRef {
                            word: (slot * row_words
                                + if node.family == Family::Line {
                                    12
                                } else {
                                    family_words - 3
                                }
                                + j) as u32,
                            key: orb_render::ImageKey::Name(name),
                        });
                    }
                }
            }
            // Image slots are patched after upload, including host-held rows.
            props.shrink_to_fit();
        }
        if let LayerKind::Line(l) = &style_layer.kind {
            // Dash and pattern scales read the width at the tile's zoom per
            // feature (word 18); an interpreted width has no table words.
            if bytecode & 1 << 2 != 0 {
                let width = l.paint.width.bind(layer);
                for (i, &feature) in features.iter().enumerate() {
                    props[i * row_words + 18] = width.evaluate(zoom, feature as usize).to_bits();
                }
            }
            let maxzoom = match &source_def.kind {
                SourceKind::GeoJson(source) => source.options.max_zoom,
                SourceKind::Vector(source) => source.tileset.maxzoom.unwrap_or(22.0) as u8,
                _ => unreachable!("line source"),
            };
            crate::line::textures(l, layer, &features, zoom, maxzoom, &mut props, &mut images);
        }
        let state_ids = plan.reads_state(node).then(|| {
            let promote = match &source_def.kind {
                SourceKind::Vector(v) => v
                    .promote_id
                    .as_ref()
                    .and_then(|p| p.key(name))
                    .and_then(|key| layer.property(key)),
                _ => None,
            };
            features
                .iter()
                .map(|&feature| {
                    let value = match promote {
                        Some(column) => column.get(feature as usize),
                        None => layer.ids().get(feature as usize),
                    };
                    value.map_or(NO_ID, |value| {
                        let next = ids.len() as u32;
                        *ids.entry(id_string(value)).or_insert(next)
                    })
                })
                .collect()
        });
        let promote = match &source_def.kind {
            SourceKind::Vector(v) => v
                .promote_id
                .as_ref()
                .and_then(|p| p.key(name))
                .and_then(|key| layer.property(key)),
            _ => None,
        };
        let hits = if matches!(
            node.family,
            Family::Fill | Family::Line | Family::Circle | Family::FillExtrusion
        ) {
            features
                .iter()
                .filter(|&&f| match node.family {
                    Family::Fill | Family::FillExtrusion => {
                        layer.geometry_type(f as usize) == orb_tile::GeometryType::Polygon
                    }
                    Family::Line => {
                        layer.geometry_type(f as usize) != orb_tile::GeometryType::Point
                    }
                    _ => true,
                })
                .map(|&f| {
                    let i = f as usize;
                    let (kind, radius, stroke, height) = match &style_layer.kind {
                        LayerKind::Circle(c) => (
                            2,
                            c.paint.radius.bind(layer).evaluate(zoom, i),
                            c.paint.stroke_width.bind(layer).evaluate(zoom, i),
                            0.0,
                        ),
                        LayerKind::Line(l) => (
                            1,
                            l.paint.width.bind(layer).evaluate(zoom, i) / 2.0,
                            0.0,
                            0.0,
                        ),
                        LayerKind::FillExtrusion(e) => {
                            (0, 0.0, 0.0, e.paint.height.bind(layer).evaluate(zoom, i))
                        }
                        _ => (0, 0.0, 0.0, 0.0),
                    };
                    crate::HitFeature {
                        id: promote.unwrap_or(layer.ids()).get(i).map(id_string),
                        feature: f,
                        kind,
                        radius,
                        stroke,
                        height,
                        base: match &style_layer.kind {
                            LayerKind::FillExtrusion(e) => {
                                e.paint.base.bind(layer).evaluate(zoom, i)
                            }
                            _ => 0.0,
                        },
                        extent: layer.extent(),
                        parts: layer
                            .geometry(i)
                            .parts()
                            .map(|p| {
                                p.rings()
                                    .map(|r| {
                                        let mut points = r.to_vec();
                                        if kind == 2 {
                                            // Circle buckets own only vertices inside the half-open tile.
                                            points.retain(|p| {
                                                p.iter()
                                                    .all(|v| *v >= 0 && *v < layer.extent() as i32)
                                            });
                                        } else if kind == 1
                                            && layer.geometry_type(i)
                                                == orb_tile::GeometryType::Polygon
                                            && let Some(first) = points.first().copied()
                                        {
                                            points.push(first);
                                        }
                                        points
                                    })
                                    .collect()
                            })
                            .collect(),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        build.meshes.push(NodeMesh {
            hits,
            node: node.key,
            extent: if matches!(
                node.family,
                Family::Line | Family::FillExtrusion | Family::SymbolIcon | Family::SymbolText
            ) {
                8192
            } else {
                layer.extent()
            },
            mesh,
            props,
            row_words: row_words as u32,
            maxima,
            images,
            state_ids,
        });
    }
    let mut table: Vec<(String, u32)> = ids.into_iter().collect();
    table.sort_unstable_by_key(|&(_, i)| i);
    build.ids = table.into_iter().map(|(id, _)| id).collect();
    Ok(build)
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_render::LINE_ROW_WORDS;
    use orb_tile::{GeometryType, LayerBuilder, Value};

    fn no_symbols() -> Symbols<'static> {
        static GLYPHS: std::sync::LazyLock<GlyphCatalog> =
            std::sync::LazyLock::new(Default::default);
        static ICONS: std::sync::LazyLock<SpriteIcons> = std::sync::LazyLock::new(Default::default);
        Symbols {
            glyphs: &GLYPHS,
            icons: &ICONS,
            words: WordData::Unavailable,
            ideographs: None,
        }
    }

    fn plan(style: &str) -> StylePlan {
        StylePlan::of(orb_v8::parse(style).expect("valid style"))
    }

    #[test]
    fn picking_matches_bucket_geometry() {
        let plan = plan(
            r#"{"version":8,"sources":{"s":{"type":"vector","tiles":["test://{z}/{x}/{y}"]}},
          "layers":[{"id":"f","type":"fill","source":"s","source-layer":"roads"},
          {"id":"l","type":"line","source":"s","source-layer":"roads"},
          {"id":"c","type":"circle","source":"s","source-layer":"roads"}]}"#,
        );
        let mut builder = LayerBuilder::new("roads", 4096);
        for (kind, points) in [
            (GeometryType::Point, vec![[-1, 5], [5, 5]]),
            (GeometryType::LineString, vec![[0, 0], [10, 10]]),
            (GeometryType::Polygon, vec![[20, 20], [30, 20], [30, 30]]),
        ] {
            builder.begin_feature(None, kind);
            builder.begin_part();
            builder.begin_ring();
            for p in points {
                builder.push_vertex(p);
            }
        }
        let frame = TileFrame {
            layers: vec![builder.finish()],
        };
        let result = build(&frame, &plan, 0, 0, Select::default(), no_symbols()).unwrap();
        let hits = |family| {
            &result
                .meshes
                .iter()
                .find(|m| plan.nodes[plan.node(m.node).unwrap()].family == family)
                .unwrap()
                .hits
        };
        assert_eq!(hits(Family::Fill).len(), 1);
        assert_eq!(hits(Family::Line).len(), 2);
        let polygon = &hits(Family::Line)[1].parts[0][0];
        assert_eq!(polygon.first(), polygon.last());
        assert_eq!(hits(Family::Circle)[0].parts[0][0], vec![[5, 5]]);
    }

    fn frame() -> TileFrame {
        let mut b = LayerBuilder::new("roads", 4096);
        for (class, y) in [("primary", 100), ("minor", 200), ("primary", 300)] {
            b.begin_feature(None, GeometryType::LineString);
            b.begin_part();
            b.begin_ring();
            b.push_vertex([0, y]);
            b.push_vertex([4096, y]);
            b.set_property("class", Value::Str(class.into()));
        }
        TileFrame {
            layers: vec![b.finish()],
        }
    }

    #[test]
    fn patterned_jobs_return_names_without_an_atlas() {
        let plan = plan(
            r#"{"version":8,"sources":{"s":{"type":"vector","tiles":["test://{z}/{x}/{y}"]}},"layers":[{"id":"l","type":"line","source":"s","source-layer":"roads","paint":{"line-pattern":["get","class"]}}]}"#,
        );
        let tile = build(&frame(), &plan, 0, 10, Select::default(), no_symbols()).unwrap();
        let mesh = &tile.meshes[0];
        assert_eq!(mesh.images.len(), 9);
        assert_eq!(
            mesh.images[0],
            ImageRef {
                word: 12,
                key: orb_render::ImageKey::Name("primary".into())
            }
        );
        assert_eq!(
            mesh.images[3],
            ImageRef {
                word: 32,
                key: orb_render::ImageKey::Name("minor".into())
            }
        );
        assert!(mesh.images.iter().all(|i| mesh.props[i.word as usize] == 0));
    }

    const STYLE: &str = r#"{
        "version": 8,
        "sources": {"s": {"type": "vector", "tiles": ["https://example.com/{z}/{x}/{y}.pbf"]}},
        "layers": [
            {"id": "bg", "type": "background", "paint": {"background-color": "white"}},
            {"id": "primary", "type": "line", "source": "s", "source-layer": "roads",
             "filter": ["==", ["get", "class"], "primary"],
             "paint": {"line-width": ["interpolate", ["linear"], ["zoom"], 10, 2, 11, 4]}},
            {"id": "all", "type": "line", "source": "s", "source-layer": "roads",
             "paint": {"line-width": ["interpolate", ["linear"], ["zoom"],
                10, ["match", ["get", "class"], "primary", 6, 1],
                11, ["match", ["get", "class"], "primary", 8, 2]]}},
            {"id": "far", "type": "line", "source": "s", "source-layer": "roads", "minzoom": 12}
        ]
    }"#;

    #[test]
    fn filters_and_lays_out_each_node() {
        let plan = plan(STYLE);
        assert_eq!(plan.nodes[0].source, usize::MAX);
        assert_eq!(plan.nodes.len(), 4);
        let build = build(&frame(), &plan, 0, 10, Select::default(), no_symbols()).unwrap();
        let nodes: Vec<u32> = build.meshes.iter().map(|m| m.node).collect();
        // Keys of the second and third layers' nodes.
        assert_eq!(nodes, [4, 6], "the minzoom 12 layer is skipped at z10");
        // The camera-only width needs no rows.
        assert!(build.meshes[0].props.is_empty());
    }

    #[test]
    fn packs_composite_values_at_both_zooms() {
        // Rows alone: the cost model interprets nothing.
        let style = orb_v8::parse(STYLE).expect("valid style");
        let plan = StylePlan::new(
            std::sync::Arc::new(style),
            vec![1, 2, 3, 4],
            vec![1],
            0,
            orb_style::CostModel::ROWS,
        );
        let build = build(&frame(), &plan, 0, 10, Select::default(), no_symbols()).unwrap();
        let all = &build.meshes[1];
        assert_eq!(all.props.len(), 3 * LINE_ROW_WORDS);
        let width = |feature: usize| {
            let at = feature * LINE_ROW_WORDS + 2 * 2;
            (
                f32::from_bits(all.props[at]),
                f32::from_bits(all.props[at + 1]),
            )
        };
        assert_eq!(width(0), (6.0, 8.0));
        assert_eq!(width(1), (1.0, 2.0));
        assert_eq!(all.maxima[2], 8.0);
    }

    #[test]
    fn packs_input_columns_for_interpreted_paint() {
        let style = orb_v8::parse(STYLE).expect("valid style");
        let plan = StylePlan::new(
            std::sync::Arc::new(style),
            vec![1, 2, 3, 4],
            vec![1],
            0,
            orb_style::CostModel::ALL,
        );
        let node = &plan.nodes[2];
        assert_eq!(plan.bytecode_mask(node), 1 << 2, "the composite width");
        let build = build(&frame(), &plan, 0, 10, Select::default(), no_symbols()).unwrap();
        let all = &build.meshes[1];
        // One string column, `class`, after each feature's row.
        let stride = LINE_ROW_WORDS + 2;
        assert_eq!(all.props.len(), 3 * stride);
        let class = |feature: usize| all.props[feature * stride + LINE_ROW_WORDS];
        assert_eq!(class(0), class(2));
        assert_ne!(class(0), class(1));
        // No number column: its range is empty.
        assert_eq!(all.maxima.len(), 6 + 2);
    }
}

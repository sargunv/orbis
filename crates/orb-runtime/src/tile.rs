//! The tile job: a decoded tile in, one mesh and property rows per render
//! node out. Pure, so it runs on any job thread or worker.

use std::borrow::Cow;
use std::collections::HashMap;

use orb_layout::{Family, ImageKey, pack_color};
use orb_layout::{Mesh, extrusion, fill, line, point};
use orb_shaders::layout;
use orb_style::{Kind, LayerType, SourceKind};
use orb_tile::{GeometryType, Layer, LayerBuilder, TileFrame};

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
    /// Per source layer laid out, the geometry its nodes' hits point into.
    pub(crate) hits: Vec<(String, crate::query::HitGeometry)>,
    /// Per symbol layer laid out, by layer key: its bucket. Its icon and
    /// text nodes' meshes carry property rows only.
    pub(crate) symbols: Vec<(u32, SymbolBuild)>,
    /// Images the symbol layers name that the style doesn't have.
    pub(crate) missing_images: Vec<String>,
}

/// What symbol layout reads beyond the tile: the worker's glyph metrics
/// and sprite metadata.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Symbols<'a> {
    pub(crate) glyphs: &'a GlyphCatalog,
    pub(crate) icons: &'a SpriteIcons,
    /// How words break in scripts written without spaces, if they do.
    pub(crate) words: Option<&'a orb_text::WordBreaker>,
    /// The map's `localIdeographFontFamily`.
    pub(crate) ideographs: Option<&'a str>,
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
    pub(crate) hits: crate::query::MeshHits,
    /// Data units per tile side.
    pub(crate) extent: u32,
    pub(crate) mesh: Mesh,
    /// Property rows, one per selected feature.
    pub(crate) props: Vec<u32>,
    /// Words per row.
    pub(crate) row_words: u32,
    /// Per paint property, the largest number in the rows (0 for colours
    /// and properties not in the rows), for culling margins.
    pub(crate) maxima: Vec<f32>,
    pub(crate) images: Vec<ImageRef>,
    /// For nodes that read feature state: per feature slot, its id as an
    /// index into [`TileBuild::ids`], or [`NO_ID`].
    pub(crate) state_ids: Option<Vec<u32>>,
    /// For nodes with paint that reads both feature state and feature
    /// data: per feature slot, what that paint reads of the feature.
    pub(crate) data: Vec<orb_expr::FeatureData>,
}

/// A feature without an id: it keeps the default state.
pub(crate) const NO_ID: u32 = u32::MAX;

/// A property word and typed texture key to resolve on the scene side. Jobs
/// need no atlas allocation; generated ramps travel with the key across workers.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImageRef {
    pub word: u32,
    pub key: ImageKey,
}

/// The features a node draws from `layer`, in slot order.
/// Input points a node's mesh lays out at most. A dense city tile's
/// extrusions take about 20 vertex words a point: in one mesh, up to a
/// million contiguous words, which a scene holding the other tiles in view
/// may not have in one block though it has the words. Pieces of about
/// 40,000 words fit the gaps a nearly full scene leaves.
pub(crate) const MESH_POINTS: usize = 2048;

/// A node's selected features, split into meshes of at most
/// [`MESH_POINTS`] input points, in order.
#[derive(Debug)]
struct Split {
    /// When a feature alone is larger (tiles merge a block's buildings
    /// into one multipolygon), its parts go to several pieces: a layer of
    /// pieces, each with its feature's properties and id, and each piece's
    /// feature in the tile's layer. The groups index this layer then.
    pieces: Option<(Layer, Vec<u32>)>,
    groups: Vec<Vec<u32>>,
}

/// Splits `features` of `layer` into groups for [`MESH_POINTS`]. Only the
/// geometry decides it, so a job that sends only property rows splits as
/// the layout did. A single part larger than the cap stays whole, and a
/// feature larger than it stays whole unless `split_parts`.
fn split_features(layer: &Layer, features: Vec<u32>, split_parts: bool) -> Split {
    let part_points = |feature: u32| -> Vec<usize> {
        layer
            .geometry(feature as usize)
            .parts()
            .map(|part| part.rings().map(<[_]>::len).sum())
            .collect()
    };
    // (feature, parts range, points) per piece.
    let mut pieces = Vec::new();
    for &feature in &features {
        let parts = part_points(feature);
        let mut start = 0;
        let mut total = 0;
        for (i, &n) in parts.iter().enumerate() {
            if split_parts && i > start && total + n > MESH_POINTS {
                pieces.push((feature, start..i, total));
                start = i;
                total = 0;
            }
            total += n;
        }
        pieces.push((feature, start..parts.len(), total));
    }
    let mut groups = vec![Vec::new()];
    let mut total = 0;
    for (i, (_, _, n)) in pieces.iter().enumerate() {
        let group = groups.last_mut().expect("one group at least");
        if !group.is_empty() && total + n > MESH_POINTS {
            groups.push(Vec::new());
            total = 0;
        }
        groups.last_mut().expect("pushed above").push(i as u32);
        total += n;
    }
    if pieces.len() == features.len() {
        let groups = groups
            .into_iter()
            .map(|g| g.into_iter().map(|i| features[i as usize]).collect())
            .collect();
        return Split {
            pieces: None,
            groups,
        };
    }
    let properties: Vec<_> = layer.properties().collect();
    let mut builder = LayerBuilder::new(layer.name(), layer.extent());
    for (feature, parts, _) in &pieces {
        let feature = *feature as usize;
        builder.begin_feature(None, layer.geometry_type(feature));
        if let Some(id) = layer.ids().get(feature) {
            builder.set_id(id.to_owned());
        }
        for (key, column) in &properties {
            if let Some(value) = column.get(feature) {
                builder.set_property(key, value.to_owned());
            }
        }
        let geometry = layer.geometry(feature);
        for part in geometry.parts().skip(parts.start).take(parts.len()) {
            builder.begin_part();
            for ring in part.rings() {
                builder.begin_ring();
                for &p in ring {
                    builder.push_vertex(p);
                }
            }
        }
    }
    let origin = pieces.iter().map(|(feature, ..)| *feature).collect();
    Split {
        pieces: Some((builder.finish(), origin)),
        groups,
    }
}

/// Per tile layer of a frame, the feature groups its style layers' filters
/// share (`orb_expr::FilterGroups`).
#[derive(Default)]
struct Groups<'f>(Vec<(&'f Layer, orb_expr::FilterGroups)>);

impl<'f> Groups<'f> {
    fn of(&mut self, layer: &'f Layer) -> &mut orb_expr::FilterGroups {
        let at = match self.0.iter().position(|(l, _)| std::ptr::eq(*l, layer)) {
            Some(at) => at,
            None => {
                self.0.push((layer, orb_expr::FilterGroups::default()));
                self.0.len() - 1
            }
        };
        &mut self.0[at].1
    }
}

fn features(
    plan: &StylePlan,
    node: &NodePlan,
    layer: &Layer,
    (zoom, canonical): (f64, Option<orb_expr::CanonicalTile>),
    groups: &mut orb_expr::FilterGroups,
) -> Vec<u32> {
    let style_layer = &plan.style.layers[node.layer];
    let mut features: Vec<u32> = match &style_layer.filter {
        Some(filter) => filter.select_grouped(layer, zoom, canonical, groups),
        None => (0..layer.len() as u32).collect(),
    };
    // GL JS's buckets draw features by their sort key.
    let sort_key = match style_layer.ty {
        LayerType::Circle => Some(orb_style::CIRCLE_SORT_KEY),
        LayerType::Fill => Some(orb_style::FILL_SORT_KEY),
        LayerType::Line => Some(orb_style::LINE_SORT_KEY),
        _ => None,
    }
    .map(|key| style_layer.get(key));
    if let Some(key) = sort_key.filter(|k| k.constant() != Some(None)) {
        let key = key.bind(layer);
        // Stable order for equal keys, just like JS's Array.sort.
        features.sort_by(|&a, &b| {
            key.evaluate(zoom, a as usize)
                .unwrap_or(0.0)
                .total_cmp(&key.evaluate(zoom, b as usize).unwrap_or(0.0))
        });
    }
    features
}

/// What a node's geometry depends on besides its tile layer and features.
/// Two nodes with equal shapes over the same features lay out the same
/// mesh, as a road's casing and its fill do, so a tile lays it out once
/// ([`Laid`]).
#[derive(Debug, PartialEq)]
enum Shape<'f> {
    /// `bend` cuts fills for a tile drawn through a curved warp or on a
    /// globe, and extends them to the poles.
    Fill(Option<orb_layout::fill::Bend>),
    FillOutline,
    Extrusion {
        rounding: f64,
        bend: Option<orb_layout::fill::Bend>,
    },
    /// Each feature's sort key, by slot.
    Circle(Vec<Option<f32>>),
    Heatmap,
    Line {
        /// Each feature's layout properties, by slot.
        params: Vec<line::LineLayout>,
        /// The data tile's lines an overzoom slice's lay out along
        /// (`Layer::overscaled_lines`), and the bend's grid.
        overscaled: Option<(ByAddress<'f>, orb_tile::Ancestor, Option<u32>)>,
        /// Whether vertices carry their progress (`StylePlan::line_metrics`).
        metrics: bool,
    },
}

/// A layer equal only to itself.
#[derive(Debug)]
struct ByAddress<'f>(&'f Layer);

impl PartialEq for ByAddress<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl<'f> Shape<'f> {
    fn of(
        layer: &Layer,
        features: &[u32],
        style_layer: &orb_style::Layer,
        (family, overscaled, metrics): (Family, Option<(&'f Layer, orb_tile::Ancestor)>, bool),
        zoom: f64,
        bend: Option<orb_layout::fill::Bend>,
        units_per_metre: f64,
    ) -> Self {
        let l = style_layer;
        match l.ty {
            LayerType::Fill => match family {
                Family::FillOutline => Self::FillOutline,
                _ => Self::Fill(bend),
            },
            LayerType::FillExtrusion => Self::Extrusion {
                rounding: f64::from(l.get(orb_style::EXTRUSION_ROUNDED_CORNER).evaluate(zoom))
                    * units_per_metre,
                bend,
            },
            LayerType::Circle => {
                let key = l.get(orb_style::CIRCLE_SORT_KEY).bind(layer);
                Self::Circle(
                    (features.iter())
                        .map(|&f| key.evaluate(zoom, f as usize))
                        .collect(),
                )
            }
            LayerType::Heatmap => Self::Heatmap,
            LayerType::Line => {
                let join = l.get(orb_style::LINE_JOIN).bind(layer);
                let cap = l.get(orb_style::LINE_CAP).bind(layer);
                let miter_limit = l.get(orb_style::LINE_MITER_LIMIT).bind(layer);
                let round_limit = l.get(orb_style::LINE_ROUND_LIMIT).bind(layer);
                let params = |feature: usize| line::LineLayout {
                    join: match join.evaluate(zoom, feature) {
                        orb_style::LineJoin::Bevel => line::LineJoin::Bevel,
                        orb_style::LineJoin::Round => line::LineJoin::Round,
                        orb_style::LineJoin::Miter => line::LineJoin::Miter,
                    },
                    cap: match cap.evaluate(zoom, feature) {
                        orb_style::LineCap::Butt => line::LineCap::Butt,
                        orb_style::LineCap::Round => line::LineCap::Round,
                        orb_style::LineCap::Square => line::LineCap::Square,
                    },
                    miter_limit: miter_limit.evaluate(zoom, feature),
                    round_limit: round_limit.evaluate(zoom, feature),
                };
                Self::Line {
                    params: features.iter().map(|&f| params(f as usize)).collect(),
                    overscaled: overscaled
                        .map(|(data, ancestor)| (ByAddress(data), ancestor, bend.map(|b| b.cells))),
                    metrics,
                }
            }
            _ => unreachable!("vector nodes have feature selections"),
        }
    }

    /// The geometry of `features` of `layer` in this shape.
    fn layout(&self, layer: &Layer, features: &[u32]) -> Mesh {
        match self {
            Self::Fill(bend) => fill::layout_subdivided(layer, features, *bend),
            Self::FillOutline => fill::outline(layer, features),
            Self::Extrusion { rounding, bend } => {
                extrusion::layout_in_extent(layer, features, 8192, *rounding, *bend)
            }
            Self::Circle(keys) => point::layout_sorted(layer, features, |slot| keys[slot]),
            Self::Heatmap => point::layout(layer, features),
            Self::Line {
                params,
                overscaled,
                metrics,
            } => {
                let params = |slot: usize| params[slot];
                match overscaled {
                    Some((data, ancestor, cells)) => {
                        let data_features: Vec<u32> = (features.iter())
                            .map(|&f| layer.data_feature(f as usize))
                            .collect();
                        line::layout_overscaled(
                            data.0,
                            &data_features,
                            params,
                            *ancestor,
                            *cells,
                            *metrics,
                        )
                    }
                    None => line::layout_in_extent(layer, features, params, 8192, *metrics),
                }
            }
        }
    }
}

/// The meshes a tile has laid out, by tile layer, features and shape, for
/// nodes that lay out alike.
#[derive(Default)]
struct Laid<'f>(Vec<(&'f Layer, Vec<u32>, Shape<'f>, Mesh)>);

impl<'f> Laid<'f> {
    /// The mesh of `features` of `layer` in `shape`: laid out, or a copy
    /// of an equal node's.
    fn mesh(&mut self, layer: &'f Layer, features: &[u32], shape: Shape<'f>) -> Mesh {
        let earlier = self.0.iter().find(|(l, f, s, _)| {
            std::ptr::eq(*l, layer) && *s == shape && f.as_slice() == features
        });
        if let Some((_, _, _, mesh)) = earlier {
            return mesh.clone();
        }
        let mesh = shape.layout(layer, features);
        self.0.push((layer, features.to_vec(), shape, mesh.clone()));
        mesh
    }
}

/// The tile layer a node draws from, if the node lays out at `zoom` (GL
/// JS's bucket rule: nothing outside the layer's zoom range). A tile at
/// zoom `z` shows from `z` to `z + 1`; tiles of other matrix sets lay out
/// at fractional zooms and keep the layers that range overlaps.
fn node_layer<'f, 'p>(
    frame: &'f TileFrame,
    plan: &'p StylePlan,
    node: &NodePlan,
    zoom: f64,
) -> Option<(&'f Layer, &'p str)> {
    let style_layer = &plan.style.layers[node.layer];
    if zoom + 1.0 <= style_layer.minzoom.floor() || zoom >= style_layer.maxzoom {
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
fn stable_ids<'l>(source: &orb_style::Source, layer: &'l Layer) -> Option<&'l orb_tile::Column> {
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

/// The canonical tile of `canonical`'s `ancestor`, if it has one.
fn in_ancestor(
    canonical: Option<orb_expr::CanonicalTile>,
    ancestor: Option<orb_tile::Ancestor>,
) -> Option<orb_expr::CanonicalTile> {
    let up = ancestor.map_or(0, |a| a.scale.trailing_zeros());
    canonical.map(|c| orb_expr::CanonicalTile {
        z: c.z - up as u8,
        x: c.x >> up,
        y: c.y >> up,
    })
}

/// Lays out the symbol layers of `source` the selection needs. Returns the
/// glyph range URLs this worker must fetch first, if any.
#[allow(clippy::too_many_arguments)]
fn build_symbols<'f>(
    frame: &'f TileFrame,
    plan: &StylePlan,
    source: usize,
    (zoom, canonical): (f64, Option<orb_expr::CanonicalTile>),
    select: Select<'_>,
    symbols: Symbols<'_>,
    groups: &mut Groups<'f>,
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
        let symbol = &plan.style.layers[node.layer];
        let Some((layer, _)) = node_layer(frame, plan, node, zoom) else {
            continue;
        };
        // Labels lay out where GL JS lays them out (`Layer::labelled`).
        let (labelled, ancestor) = layer.labelled();
        // A layer made for this node groups its features alone.
        let mut own = orb_expr::FilterGroups::default();
        let node_groups = match &labelled {
            Cow::Borrowed(layer) => groups.of(layer),
            Cow::Owned(_) => &mut own,
        };
        let layer = &*labelled;
        let at = (zoom, in_ancestor(canonical, ancestor));
        let features = features(plan, node, layer, at, node_groups);
        if features.is_empty() {
            continue;
        }
        let ids = stable_ids(source_def, layer);
        let images = symbols.icons.names();
        let tile = in_ancestor(canonical, ancestor);
        // A slice laid out in its ancestor prepares only the labels it may
        // keep, not each of its ancestor's.
        let laid_out: Cow<'_, [u32]> = match ancestor {
            Some(_) => {
                let placement = crate::symbol::placement(symbol, zoom);
                let may = |&f: &u32| orb_symbol::may_anchor_in_tile(layer, f, placement, ancestor);
                Cow::Owned(features.iter().copied().filter(may).collect())
            }
            None => Cow::Borrowed(&features),
        };
        let p = crate::symbol::prepare(symbol, layer, &laid_out, zoom, ids, images, tile);
        let faces = &plan.style.font_faces;
        for url in symbols.glyphs.missing(&p.requests, template, policy, faces) {
            if !missing.contains(&url) {
                missing.push(url);
            }
        }
        prepared.push((node, labelled, ancestor, features, p));
    }
    if !missing.is_empty() {
        return Err(missing);
    }
    let mut built = Vec::with_capacity(prepared.len());
    for (node, layer, ancestor, features, mut p) in prepared {
        let layer = &*layer;
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
        let layer_key = crate::style::layer_key(node.key);
        built.push((
            layer_key,
            crate::symbol::layout(
                p,
                (layer, ancestor),
                resolved.glyphs,
                symbols.icons,
                &slots,
                symbols.words,
            ),
        ));
    }
    Ok(built)
}

/// Lays out the selected nodes of `source` that draw from `frame`, at the
/// tile's style `zoom`. Returns the glyph range URLs this worker must fetch
/// first when the tile's labels need ranges it hasn't got.
pub(crate) fn build(
    frame: &TileFrame,
    plan: &StylePlan,
    source: usize,
    (zoom, units_per_metre, canonical): (f64, f64, Option<orb_expr::CanonicalTile>),
    bend: Option<orb_layout::fill::Bend>,
    select: Select<'_>,
    symbols: Symbols<'_>,
) -> Result<TileBuild, Vec<String>> {
    let mut groups = Groups::default();
    let mut laid = Laid::default();
    let mut build = TileBuild {
        revision: plan.revision,
        rows_only: select.rows_only,
        ..TileBuild::default()
    };
    build.symbols = build_symbols(
        frame,
        plan,
        source,
        (zoom, canonical),
        select,
        symbols,
        &mut groups,
        &mut build.warnings,
    )?;
    for (_, symbols) in &mut build.symbols {
        for name in symbols.missing_images.drain(..) {
            if !build.missing_images.contains(&name) {
                build.missing_images.push(name);
            }
        }
    }
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
        // Symbol rows follow the features their layout selected, in the
        // ancestor's slice labels lay out from, if they do
        // (`Layer::labelled`; a slice's own layer has the same features as
        // what it lays labels out from otherwise).
        let (layer, ancestor) = match layer.ancestor_labels() {
            Some((labels, ancestor)) if node.family.is_symbol() => (labels, Some(ancestor)),
            _ => (layer, None),
        };
        // Where `within` and `distance` find the features' geometry.
        let geometry_tile = in_ancestor(canonical, ancestor);
        let selected = features(plan, node, layer, (zoom, geometry_tile), groups.of(layer));
        if selected.is_empty() {
            continue;
        }
        // An overzoom slice's lines lay out along its data tile's.
        let overscaled = (node.family == Family::Line)
            .then(|| layer.overscaled_lines())
            .flatten();
        // Large layers draw as several meshes (see `split_features`);
        // symbol nodes' rows follow their whole bucket, and overscaled
        // lines lay out whole features.
        let split = if node.family.is_symbol() {
            Split {
                pieces: None,
                groups: vec![selected],
            }
        } else {
            split_features(layer, selected, overscaled.is_none())
        };
        let tile_layer = layer;
        let tile_promote = match &source_def.kind {
            SourceKind::Vector(v) => v
                .promote_id
                .as_ref()
                .and_then(|p| p.key(name))
                .and_then(|key| layer.property(key)),
            _ => None,
        };
        let hit_layer = match build.hits.iter().position(|(n, _)| n == name) {
            Some(i) => i,
            None => {
                build
                    .hits
                    .push((name.to_string(), crate::query::HitGeometry::default()));
                build.hits.len() - 1
            }
        };
        // A feature split into pieces picks once, with all its parts.
        let mut picked = std::collections::HashSet::new();
        let (layer, origin) = match &split.pieces {
            Some((pieces, origin)) => (pieces, Some(origin.as_slice())),
            None => (layer, None),
        };
        for features in split.groups {
            let row_words = plan.row_words(node);
            let mesh = match node.family {
                _ if select.rows_only => Mesh::default(),
                // Symbol geometry is the layer's bucket; its nodes carry rows.
                family @ (Family::SymbolIcon | Family::SymbolText) => {
                    let bucket = build
                        .symbols
                        .iter()
                        .find(|(k, _)| *k == crate::style::layer_key(node.key));
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
                    let shape = Shape::of(
                        layer,
                        &features,
                        style_layer,
                        (node.family, overscaled, plan.line_metrics(node)),
                        zoom,
                        bend,
                        units_per_metre,
                    );
                    // A layer of pieces is this node's alone.
                    let mesh = match origin {
                        Some(_) => shape.layout(layer, &features),
                        None => laid.mesh(tile_layer, &features, shape),
                    };
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
            let line_effects = crate::line::has_textures(style_layer);
            // Extrusion rows carry each feature's centroid, for terrain and
            // ownership.
            let per_feature = line_effects
                || node.family == Family::FillExtrusion
                || pattern.is_some()
                || properties.iter().any(PaintProperty::per_feature);
            let props = if per_feature {
                let mut rows = vec![0u32; features.len() * row_words];
                for (slot, property) in properties.iter().enumerate() {
                    if !property.per_feature() {
                        continue;
                    }
                    let composite = property.kind() == Kind::Composite;
                    // Per feature at a zoom: its row word, and its
                    // magnitude for culling margins (none for colours).
                    let value: Box<dyn Fn(f64, usize) -> (u32, f32)> = match property {
                        PaintProperty::Color(p) => {
                            let b = p.bind(layer).with_tile(geometry_tile);
                            Box::new(move |z, f| {
                                (pack_color(b.evaluate(z, f).premultiplied()), 0.0)
                            })
                        }
                        PaintProperty::OptionalColor(p) => {
                            let b = p.bind(layer).with_tile(geometry_tile);
                            Box::new(move |z, f| {
                                let c = b.evaluate(z, f).map_or([0.0; 4], |c| c.premultiplied());
                                (pack_color(c), 0.0)
                            })
                        }
                        PaintProperty::Number(p) => {
                            let b = p.bind(layer).with_tile(geometry_tile);
                            Box::new(move |z, f| {
                                let v = b.evaluate(z, f);
                                (v.to_bits(), v)
                            })
                        }
                    };
                    for (i, &feature) in features.iter().enumerate() {
                        let f = feature as usize;
                        let (start, a) = value(zoom, f);
                        let (end, b) = if composite {
                            value(zoom + 1.0, f)
                        } else {
                            (start, a)
                        };
                        maxima[slot] = maxima[slot].max(a).max(b);
                        rows[i * row_words + slot * 2..][..2].copy_from_slice(&[start, end]);
                    }
                }
                if node.family == Family::FillExtrusion {
                    let whole = orb_layout::extrusion::whole(layer, &features, bend.is_some());
                    // Per-feature opacity and vertical gradient
                    // (`orbis:expressions`) share the flags word.
                    let opacity = style_layer.get(orb_style::EXTRUSION_OPACITY);
                    let gradient = style_layer.get(orb_style::EXTRUSION_VERTICAL_GRADIENT);
                    let per_feature = |kind| matches!(kind, Kind::Source | Kind::Composite);
                    let composite = opacity.kind() == Kind::Composite;
                    let opacity = per_feature(opacity.kind()).then(|| opacity.bind(layer));
                    let gradient = per_feature(gradient.kind()).then(|| gradient.bind(layer));
                    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
                    for (i, &feature) in features.iter().enumerate() {
                        let f = feature as usize;
                        let [x, y] = orb_layout::extrusion::centroid(layer, feature, 8192);
                        rows[i * row_words + layout::EXTRUSION_ROW_CENTROID] =
                            u32::from(x as u16) | u32::from(y as u16) << 16;
                        let mut flags = u32::from(whole[i]);
                        if let Some(gradient) = &gradient {
                            flags |= u32::from(gradient.evaluate(zoom, f)) << 1;
                        }
                        if let Some(opacity) = &opacity {
                            let at = opacity.evaluate(zoom, f);
                            let next = if composite {
                                opacity.evaluate(zoom + 1.0, f)
                            } else {
                                at
                            };
                            flags |= byte(at) << 8 | byte(next) << 16;
                        }
                        rows[i * row_words + layout::EXTRUSION_ROW_WHOLE] = flags;
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
                                    + match node.family {
                                        Family::Line => layout::LINE_ROW_PATTERN,
                                        Family::FillExtrusion => layout::EXTRUSION_ROW_PATTERN,
                                        _ => layout::FILL_ROW_PATTERN,
                                    }
                                    + j) as u32,
                                key: ImageKey::Name(name),
                            });
                        }
                    }
                }
                // Image slots are patched after upload, including host-held rows.
                props.shrink_to_fit();
            }
            if style_layer.ty == LayerType::Line {
                let l = style_layer;
                let maxzoom = match &source_def.kind {
                    SourceKind::GeoJson(source) => source.options.max_zoom,
                    SourceKind::Vector(source) => source.tileset.maxzoom.unwrap_or(22.0) as u8,
                    _ => unreachable!("line source"),
                };
                let overscaled = overscaled.map(|(data, _)| data);
                crate::line::textures(
                    l,
                    layer,
                    &features,
                    zoom,
                    maxzoom,
                    overscaled,
                    &mut props,
                    &mut images,
                );
            }
            let promote = match &source_def.kind {
                SourceKind::Vector(v) => v
                    .promote_id
                    .as_ref()
                    .and_then(|p| p.key(name))
                    .and_then(|key| layer.property(key)),
                _ => None,
            };
            let state_ids = plan.reads_state(node).then(|| {
                features
                    .iter()
                    .map(|&feature| {
                        let value = promote.unwrap_or(layer.ids()).get(feature as usize);
                        value.map_or(NO_ID, |value| {
                            let next = ids.len() as u32;
                            *ids.entry(crate::features::state_key(value)).or_insert(next)
                        })
                    })
                    .collect()
            });
            let mixed: Vec<&orb_expr::Expression> = properties
                .iter()
                .filter(|p| p.mixes_state())
                .filter_map(|p| p.expression().map(|e| e.expression()))
                .collect();
            let data = if mixed.is_empty() {
                Vec::new()
            } else {
                features
                    .iter()
                    .map(|&f| {
                        orb_expr::FeatureData::capture(layer, f as usize, mixed.iter().copied())
                    })
                    .collect()
            };
            let kind = match node.family {
                Family::Fill | Family::FillExtrusion => Some(0),
                Family::Line => Some(1),
                Family::Circle => Some(2),
                Family::SymbolIcon | Family::SymbolText => Some(crate::query::LABEL),
                _ => None,
            };
            let geometry = &mut build.hits[hit_layer].1;
            let mut hits = crate::query::MeshHits::new(hit_layer as u32);
            // Unless its producer said, a scan of the layer's vertices:
            // once per layer, not once per building.
            let clip_buffer = (node.family == Family::FillExtrusion).then(|| layer.clip_buffer());
            // What a hit test reads of each feature's paint.
            let dimensions: &[_] = match style_layer.ty {
                LayerType::Circle => &[orb_style::CIRCLE_RADIUS, orb_style::CIRCLE_STROKE_WIDTH],
                LayerType::Line => &[
                    orb_style::LINE_WIDTH,
                    orb_style::LINE_GAP_WIDTH,
                    orb_style::LINE_OFFSET,
                ],
                LayerType::FillExtrusion => {
                    &[orb_style::EXTRUSION_HEIGHT, orb_style::EXTRUSION_BASE]
                }
                _ => &[],
            };
            let dimensions: Vec<_> = (dimensions.iter())
                .map(|&key| style_layer.get(key).bind(layer))
                .collect();
            for &f in features.iter().filter(|_| kind.is_some()) {
                let i = f as usize;
                let shape = layer.geometry_type(i);
                let fits = match node.family {
                    Family::Fill | Family::FillExtrusion => shape == GeometryType::Polygon,
                    Family::Line => shape != GeometryType::Point,
                    _ => true,
                };
                // Queries read the feature from the tile's layer.
                let feature = origin.map_or(f, |origin| origin[i]);
                if !fits || (origin.is_some() && !picked.insert(feature)) {
                    continue;
                }
                let kind = kind.expect("a picked family");
                let id = tile_promote
                    .unwrap_or(tile_layer.ids())
                    .get(feature as usize);
                let at = |d: usize| dimensions[d].evaluate(zoom, i);
                let dimensions = match style_layer.ty {
                    LayerType::Circle => [at(0), at(1), 0.0, 0.0],
                    LayerType::Line => {
                        let offset = at(2);
                        [at(0) / 2.0, at(1), offset.abs(), offset]
                    }
                    // Whether it's whole, as layout keeps it, for picking
                    // to own it as drawing does.
                    LayerType::FillExtrusion => [
                        f32::from(u8::from(
                            bend.is_none() && layer.is_whole(i, clip_buffer.unwrap_or_default()),
                        )),
                        0.0,
                        at(0),
                        at(1),
                    ],
                    _ => [0.0; 4],
                };
                let parts = kind != crate::query::LABEL;
                let record = geometry.add(tile_layer, feature, id, parts);
                hits.push(geometry, record, feature, kind, dimensions);
            }
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
                data,
            });
        }
    }
    let mut table: Vec<(String, u32)> = ids.into_iter().collect();
    table.sort_unstable_by_key(|&(_, i)| i);
    build.ids = table.into_iter().map(|(id, _)| id).collect();
    Ok(build)
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_shaders::layout::LINE_ROW_WORDS;
    use orb_tile::{GeometryType, LayerBuilder, Value};

    fn no_symbols() -> Symbols<'static> {
        static GLYPHS: std::sync::LazyLock<GlyphCatalog> =
            std::sync::LazyLock::new(Default::default);
        static ICONS: std::sync::LazyLock<SpriteIcons> = std::sync::LazyLock::new(Default::default);
        Symbols {
            glyphs: &GLYPHS,
            icons: &ICONS,
            words: None,
            ideographs: None,
        }
    }

    fn plan(style: &str) -> StylePlan {
        StylePlan::of(orb_style::v8::parse(style).expect("valid style"))
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
        let result = build(
            &frame,
            &plan,
            0,
            (0.0, 1.0, None),
            None,
            Select::default(),
            no_symbols(),
        )
        .unwrap();
        let hits = |family| {
            let mesh = result
                .meshes
                .iter()
                .find(|m| plan.nodes[plan.node(m.node).unwrap()].family == family);
            hits(&result, mesh.unwrap())
        };
        assert_eq!(hits(Family::Fill).len(), 1);
        assert_eq!(hits(Family::Line).len(), 2);
        let polygon = &hits(Family::Line)[1].parts[0][0];
        assert_eq!(polygon.first(), polygon.last());
        assert_eq!(hits(Family::Circle)[0].parts[0][0], vec![[5, 5]]);
    }

    /// A dense city tile: thousands of buildings whose extrusions would need
    /// one contiguous block of hundreds of thousands of vertex words. They
    /// lay out as several meshes of at most `MESH_POINTS` input points, in
    /// feature order, and a job sending only property rows splits alike.
    #[test]
    fn dense_layers_split_into_meshes() {
        let plan = plan(
            r#"{"version":8,"sources":{"s":{"type":"vector","tiles":["test://{z}/{x}/{y}"]}},
          "layers":[{"id":"b","type":"fill-extrusion","source":"s","source-layer":"building",
            "paint":{"fill-extrusion-height":["get","h"]}}]}"#,
        );
        let mut builder = LayerBuilder::new("building", 4096);
        // 60 × 60 buildings of 8 points each: 28,800 points.
        for i in 0..3600 {
            let (x, y) = ((i % 60) * 64 + 8, (i / 60) * 64 + 8);
            builder.begin_feature(None, GeometryType::Polygon);
            builder.set_property("h", Value::F64(f64::from(10 + i % 50)));
            builder.begin_part();
            builder.begin_ring();
            for [dx, dy] in [
                [0, 0],
                [20, 0],
                [40, 0],
                [40, 20],
                [40, 40],
                [20, 40],
                [0, 40],
                [0, 20],
            ] {
                builder.push_vertex([x + dx, y + dy]);
            }
        }
        let frame = TileFrame {
            layers: vec![builder.finish()],
        };
        let full = build(
            &frame,
            &plan,
            0,
            (15.0, 1.0, None),
            None,
            Select::default(),
            no_symbols(),
        )
        .unwrap();
        let per_mesh = MESH_POINTS / 8;
        assert_eq!(
            full.meshes.len(),
            3600usize.div_ceil(per_mesh),
            "whole buildings a mesh"
        );
        let mut features = Vec::new();
        for mesh in &full.meshes {
            let hits = hits(&full, mesh);
            // Picking closes the rings.
            let points: usize = hits.iter().map(|h| h.parts[0][0].len() - 1).sum();
            assert!(points <= MESH_POINTS, "{points} points in one mesh");
            features.extend(hits.iter().map(|h| h.feature));
        }
        assert_eq!(
            features,
            (0..3600).collect::<Vec<u32>>(),
            "every feature once, in order"
        );
        let rows = build(
            &frame,
            &plan,
            0,
            (15.0, 1.0, None),
            None,
            Select {
                rows_only: true,
                ..Select::default()
            },
            no_symbols(),
        )
        .unwrap();
        let row_counts = |b: &TileBuild| -> Vec<usize> {
            b.meshes
                .iter()
                .map(|m| m.props.len() / m.row_words as usize)
                .collect()
        };
        assert_eq!(row_counts(&rows), row_counts(&full));
    }

    /// OpenFreeMap merges a dense z14 tile's buildings into a few
    /// multipolygons (lower Manhattan: one of 36,347 points). A feature
    /// over the cap splits by its parts into pieces that keep its
    /// properties and id; picking still names the tile's feature.
    #[test]
    fn merged_buildings_split_by_parts() {
        let plan = plan(
            r#"{"version":8,"sources":{"s":{"type":"vector","tiles":["test://{z}/{x}/{y}"]}},
          "layers":[{"id":"b","type":"fill-extrusion","source":"s","source-layer":"building",
            "paint":{"fill-extrusion-height":["get","h"]}}]}"#,
        );
        let mut builder = LayerBuilder::new("building", 4096);
        builder.begin_feature(Some(7), GeometryType::Polygon);
        builder.set_property("h", Value::F64(30.0));
        for i in 0..3600 {
            let (x, y) = ((i % 60) * 64 + 8, (i / 60) * 64 + 8);
            builder.begin_part();
            builder.begin_ring();
            for [dx, dy] in [[0, 0], [40, 0], [40, 40], [0, 40]] {
                builder.push_vertex([x + dx, y + dy]);
            }
        }
        builder.begin_feature(Some(8), GeometryType::Polygon);
        builder.set_property("h", Value::F64(10.0));
        builder.begin_part();
        builder.begin_ring();
        for p in [[0, 0], [8, 0], [8, 8], [0, 8]] {
            builder.push_vertex(p);
        }
        let frame = TileFrame {
            layers: vec![builder.finish()],
        };
        let result = build(
            &frame,
            &plan,
            0,
            (15.0, 1.0, None),
            None,
            Select::default(),
            no_symbols(),
        )
        .unwrap();
        // 14,400 points of the first feature in pieces of whole buildings,
        // then the second feature with the last piece. The first piece
        // picks the first feature, with all its parts, and the last the
        // second.
        let per_piece = MESH_POINTS / 4;
        let mut expected = vec![per_piece; 3600 / per_piece];
        expected.push(3600 % per_piece + 1);
        let picked: Vec<Vec<(u32, Option<String>, usize)>> = result
            .meshes
            .iter()
            .map(|m| {
                let hits = hits(&result, m).into_iter();
                hits.map(|h| (h.feature, h.id, h.parts.len())).collect()
            })
            .collect();
        assert_eq!(picked.len(), expected.len());
        assert_eq!(picked[0], [(0, Some("7".to_string()), 3600)]);
        assert!(picked[1..picked.len() - 1].iter().all(Vec::is_empty));
        assert_eq!(picked.last().unwrap(), &[(1, Some("8".to_string()), 1)]);
        // A property row a piece.
        let rows: usize = result
            .meshes
            .iter()
            .map(|m| m.props.len() / m.row_words as usize)
            .sum();
        assert_eq!(rows, expected.len() + 1);
    }

    /// A mesh's hits, pointed at its build's geometry.
    fn hits(build: &TileBuild, mesh: &NodeMesh) -> Vec<crate::query::HitFeature> {
        let geometry: Vec<_> = build
            .hits
            .iter()
            .map(|(_, g)| std::sync::Arc::new(g.clone()))
            .collect();
        let mut hits = mesh.hits.clone();
        hits.link(&geometry);
        hits.decode()
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
        let tile = build(
            &frame(),
            &plan,
            0,
            (10.0, 1.0, None),
            None,
            Select::default(),
            no_symbols(),
        )
        .unwrap();
        let mesh = &tile.meshes[0];
        assert_eq!(mesh.images.len(), 9);
        assert_eq!(
            mesh.images[0],
            ImageRef {
                word: 12,
                key: ImageKey::Name("primary".into())
            }
        );
        assert_eq!(
            mesh.images[3],
            ImageRef {
                word: 32,
                key: ImageKey::Name("minor".into())
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
        let build = build(
            &frame(),
            &plan,
            0,
            (10.0, 1.0, None),
            None,
            Select::default(),
            no_symbols(),
        )
        .unwrap();
        let nodes: Vec<u32> = build.meshes.iter().map(|m| m.node).collect();
        // Keys of the second and third layers' nodes.
        assert_eq!(
            nodes,
            [crate::style::node_key(2, 0), crate::style::node_key(3, 0)],
            "the minzoom 12 layer is skipped at z10"
        );
        // The camera-only width needs no rows.
        assert!(build.meshes[0].props.is_empty());
    }

    #[test]
    fn packs_composite_values_at_both_zooms() {
        let plan = plan(STYLE);
        let build = build(
            &frame(),
            &plan,
            0,
            (10.0, 1.0, None),
            None,
            Select::default(),
            no_symbols(),
        )
        .unwrap();
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
}

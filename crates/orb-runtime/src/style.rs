//! A v8 style compiled into what the renderer draws: a background and one
//! render node per drawable layer, in style order.
//!
//! Layers and sources carry the stable keys `orb-style` gives them, so a
//! node keeps its key across edits that move it ([`node_key`]: a layer's
//! nodes are its fill and outline, or its icons and text), and tile jobs
//! and results name nodes and sources by key.

use std::sync::Arc;

use orb_expr::Kind;
use orb_layout::Family;
use orb_shaders::layout;
use orb_style::{Color, LayerType, PropertyRef, SourceKind, Style, Value};

use crate::daylight::LabelDaylight;
use crate::elevation::ElevationColor;

/// Node keys a layer has: a fill and its outline, or a symbol layer's
/// icons and text, take one each.
const LAYER_NODES: u32 = 2;

/// The key of a layer's `n`th node.
pub(crate) fn node_key(layer_key: u32, n: u32) -> u32 {
    layer_key * LAYER_NODES + n
}

/// The key of the layer a node is of.
pub(crate) fn layer_key(node_key: u32) -> u32 {
    node_key / LAYER_NODES
}

/// A style ready to render: one revision of it. Clones are shallow, as a
/// frame's paint transitions overlay it.
#[derive(Clone, Debug)]
pub(crate) struct StylePlan {
    /// The map's style version this plan is of.
    pub(crate) revision: u64,
    pub(crate) style: Arc<Style>,
    /// Keys of the style's layers, in order.
    pub(crate) layer_keys: Vec<u32>,
    /// Keys of the style's sources, in order.
    pub(crate) source_keys: Vec<u32>,
    pub(crate) nodes: Vec<NodePlan>,
    /// The sources nodes draw from; nodes refer to them by index.
    pub(crate) sources: Vec<PlanSource>,
    /// What the compiled plan leaves out, beyond the parser's warnings.
    pub(crate) warnings: Vec<String>,
    /// The layers whose paint reads the camera beyond its zoom
    /// ([`crate::paint::at_camera`]).
    pub(crate) camera_layers: Vec<usize>,
    /// Whether any of them reads `["dev.sargunv.orbis:meters-per-pixel"]`, which a pan
    /// changes: paint that reads only the pitch and bearing isn't written
    /// again as the map pans.
    pub(crate) reads_resolution: bool,
    /// By layer, the first layer before it that lays out alike
    /// ([`orb_style::Layer::layout_group`]): the group's leader.
    pub(crate) layout_leaders: Vec<Option<usize>>,
}

/// A source some node draws from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlanSource {
    pub(crate) key: u32,
    pub(crate) id: String,
}

/// What an image source's tiles hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageKind {
    /// Colour images, drawn premultiplied.
    Raster,
    /// Elevations, encoded in RGB.
    Dem,
}

/// One style layer drawn as a render node.
#[derive(Clone, Debug)]
pub(crate) struct NodePlan {
    /// Index into the style's layers.
    pub(crate) layer: usize,
    pub(crate) family: Family,
    /// Index into [`StylePlan::sources`].
    pub(crate) source: usize,
    /// Stable across revisions.
    pub(crate) key: u32,
    /// An extrusion's colour over the elevation, drawn per fragment.
    pub(crate) elevation: Option<Arc<ElevationColor>>,
    /// A symbol node's colours over the daylight, drawn per label.
    pub(crate) daylight: Option<Arc<LabelDaylight>>,
}

/// A paint property as the renderer packs it.
#[derive(Clone, Debug)]
pub(crate) enum PaintProperty<'a> {
    Color(PropertyRef<'a, Color>),
    /// A colour that may be unset, such as `fill-outline-color` given as an
    /// expression; unset draws transparent.
    OptionalColor(PropertyRef<'a, Option<Color>>),
    Number(PropertyRef<'a, f32>),
}

impl PaintProperty<'_> {
    pub(crate) fn kind(&self) -> Kind {
        match self {
            Self::Color(p) => p.kind(),
            Self::OptionalColor(p) => p.kind(),
            Self::Number(p) => p.kind(),
        }
    }

    /// A colour property's premultiplied value at `zoom`.
    pub(crate) fn color(&self, zoom: f64) -> [f32; 4] {
        match self {
            Self::Color(p) => p.evaluate(zoom).premultiplied(),
            Self::OptionalColor(p) => p.evaluate(zoom).map_or([0.0; 4], |c| c.premultiplied()),
            Self::Number(_) => unreachable!("not a colour"),
        }
    }

    /// The property it reads.
    pub(crate) fn id(&self) -> Option<orb_style::PropertyId> {
        match self {
            Self::Color(p) => p.id(),
            Self::OptionalColor(p) => p.id(),
            Self::Number(p) => p.id(),
        }
    }

    /// Whether its row words are colours (rgba8) rather than numbers (f32).
    pub(crate) fn is_color(&self) -> bool {
        !matches!(self, Self::Number(_))
    }

    /// Whether the value comes from per-feature property rows.
    pub(crate) fn per_feature(&self) -> bool {
        matches!(self.kind(), Kind::Source | Kind::Composite) && !self.is_state_lookup()
    }

    /// Whether the value reads feature state but no feature data: it comes
    /// from the feature-state table, else from the node's constant.
    pub(crate) fn is_state_lookup(&self) -> bool {
        match self {
            Self::Color(p) => p.is_state_lookup(),
            Self::OptionalColor(p) => p.is_state_lookup(),
            Self::Number(p) => p.is_state_lookup(),
        }
    }

    /// Whether the value reads both feature state and feature data: rows
    /// hold it without state, and the state table with the feature's state
    /// and captured data.
    pub(crate) fn mixes_state(&self) -> bool {
        self.expression()
            .is_some_and(|e| e.reads_feature_state() && e.reads_feature_properties())
    }

    /// Whether features with state take the value from the state table.
    pub(crate) fn reads_state(&self) -> bool {
        self.is_state_lookup() || self.mixes_state()
    }

    pub(crate) fn expression(&self) -> Option<&orb_expr::PropertyExpression> {
        match self {
            Self::Color(p) => p.as_expression(),
            Self::OptionalColor(p) => p.as_expression(),
            Self::Number(p) => p.as_expression(),
        }
    }

    /// A state-table value packed as a row word: for a feature with
    /// `state` and captured `data` at `zoom`.
    pub(crate) fn state_word(
        &self,
        zoom: f64,
        state: &[(String, Value<'static>)],
        data: Option<&orb_expr::FeatureData>,
    ) -> u32 {
        match self {
            Self::Color(p) => {
                orb_layout::pack_color(p.evaluate_state(zoom, state, data).premultiplied())
            }
            Self::OptionalColor(p) => orb_layout::pack_color(
                p.evaluate_state(zoom, state, data)
                    .map_or([0.0; 4], |c| c.premultiplied()),
            ),
            Self::Number(p) => p.evaluate_state(zoom, state, data).to_bits(),
        }
    }

    /// The node constant of a state lookup: its value with no state.
    pub(crate) fn default_number(&self, zoom: f64) -> f32 {
        match self {
            Self::Number(p) => p.evaluate_state(zoom, &[], None),
            _ => unreachable!("not a number"),
        }
    }

    pub(crate) fn default_color(&self, zoom: f64) -> [f32; 4] {
        match self {
            Self::Color(p) => p.evaluate_state(zoom, &[], None).premultiplied(),
            Self::OptionalColor(p) => p
                .evaluate_state(zoom, &[], None)
                .map_or([0.0; 4], |c| c.premultiplied()),
            Self::Number(_) => unreachable!("not a colour"),
        }
    }
}

impl StylePlan {
    /// A plan with layer and source keys counted from one, for a style that
    /// is never edited.
    #[cfg(test)]
    pub(crate) fn of(style: Style) -> Self {
        let layers = (1..=style.layers.len() as u32).collect();
        let sources = (1..=style.sources.len() as u32).collect();
        Self::new(Arc::new(style), layers, sources, 0)
    }

    pub(crate) fn new(
        style: Arc<Style>,
        layer_keys: Vec<u32>,
        source_keys: Vec<u32>,
        revision: u64,
    ) -> Self {
        assert_eq!(layer_keys.len(), style.layers.len());
        assert_eq!(source_keys.len(), style.sources.len());
        let mut nodes = Vec::new();
        let mut sources: Vec<PlanSource> = Vec::new();
        let mut warnings = Vec::new();
        let mut warn = |index: usize, message: &str| {
            let id = &style.layers[index].id;
            warnings.push(format!("layers[{index}] ({id}): {message}"));
        };

        for (index, layer) in style.layers.iter().enumerate() {
            let key = node_key(layer_keys[index], 0);
            let family = match layer.ty {
                LayerType::Background => {
                    nodes.push(NodePlan {
                        layer: index,
                        family: Family::Fill,
                        source: usize::MAX,
                        key,
                        elevation: None,
                        daylight: None,
                    });
                    continue;
                }
                LayerType::FillExtrusion => Family::FillExtrusion,
                LayerType::Fill => Family::Fill,
                LayerType::Line => Family::Line,
                LayerType::Circle => Family::Circle,
                LayerType::Heatmap => Family::Heatmap,
                LayerType::Raster => Family::Raster,
                LayerType::Hillshade => Family::Hillshade,
                LayerType::ColorRelief => Family::ColorRelief,
                LayerType::Symbol => Family::SymbolIcon,
            };
            let source = layer.source.as_deref().expect("drawn layers have sources");
            match style.source(source).map(|s| &s.kind) {
                // A source not added yet: the layer draws once it is
                // (`orb_style` warned).
                None => continue,
                // An app's tiles hold GeoJSON for a layer naming no source
                // layer (`orb_style::TileSet::host`).
                Some(SourceKind::Vector(v)) => {
                    if layer.ty.draws_features()
                        && layer.source_layer.is_none()
                        && !v.tileset.host()
                    {
                        warn(index, "vector layers need a source-layer");
                        continue;
                    }
                }
                Some(
                    SourceKind::GeoJson(_)
                    | SourceKind::Raster(_)
                    | SourceKind::RasterDem(_)
                    | SourceKind::Image(_),
                ) => {}
            }
            if layer.ty == LayerType::Hillshade {
                let lights = [
                    orb_style::HILLSHADE_ILLUMINATION_DIRECTION.id(),
                    orb_style::HILLSHADE_ILLUMINATION_ALTITUDE.id(),
                    orb_style::HILLSHADE_SHADOW_COLOR.id(),
                    orb_style::HILLSHADE_HIGHLIGHT_COLOR.id(),
                ]
                .map(|id| layer.slot(id).items().len());
                if lights.into_iter().max().unwrap_or(1) > orb_shaders::layout::MAX_LIGHTS {
                    warn(
                        index,
                        &format!(
                            "hillshades draw at most {} light sources; the rest are ignored",
                            orb_shaders::layout::MAX_LIGHTS
                        ),
                    );
                }
            }
            let source = match sources.iter().position(|s| s.id == source) {
                Some(i) => i,
                None => {
                    let at = style.sources.iter().position(|s| s.id == source);
                    sources.push(PlanSource {
                        key: source_keys[at.expect("layers' sources exist")],
                        id: source.to_string(),
                    });
                    sources.len() - 1
                }
            };
            let elevation = match ElevationColor::of(layer) {
                Some(Ok(color)) => Some(Arc::new(color)),
                Some(Err(why)) => {
                    warn(
                        index,
                        &format!("fill-extrusion-color: {why}; it draws at the ground"),
                    );
                    None
                }
                None => None,
            };
            let mut daylight = |part| {
                let (daylight, why) = LabelDaylight::of(layer, part);
                for why in why {
                    warn(index, &why);
                }
                daylight.map(Arc::new)
            };
            nodes.push(NodePlan {
                layer: index,
                family,
                source,
                key,
                elevation,
                daylight: daylight(orb_style::ICON_PART),
            });
            // A symbol layer draws its icons, then its text (GL JS
            // `drawSymbols`).
            if layer.ty == LayerType::Symbol {
                nodes.push(NodePlan {
                    layer: index,
                    family: Family::SymbolText,
                    source,
                    key: key + 1,
                    elevation: None,
                    daylight: daylight(orb_style::TEXT_PART),
                });
            }
            // GL JS antialiases fills by drawing their outlines after them:
            // here unless no zoom, camera or feature ever wants one, the
            // outline skipping zooms and features that don't.
            if layer.ty == LayerType::Fill
                && layer.get(orb_style::FILL_ANTIALIAS).constant() != Some(false)
            {
                nodes.push(NodePlan {
                    layer: index,
                    family: Family::FillOutline,
                    source,
                    key: key + 1,
                    elevation: None,
                    daylight: None,
                });
            }
        }

        // The terrain's DEM source loads even when no layer draws it.
        if let Some(terrain) = &style.terrain
            && !sources.iter().any(|s| s.id == terrain.source)
            && let Some(at) = style.sources.iter().position(|s| s.id == terrain.source)
        {
            sources.push(PlanSource {
                key: source_keys[at],
                id: terrain.source.clone(),
            });
        }

        let camera_layers: Vec<usize> = (style.layers.iter().enumerate())
            .filter(|(_, l)| {
                l.slots()
                    .any(|(_, s)| s.as_expression().is_some_and(|e| e.reads_camera()))
            })
            .map(|(i, _)| i)
            .collect();
        let reads_resolution = camera_layers.iter().any(|&i| {
            (style.layers[i].slots()).any(|(_, s)| {
                s.as_expression()
                    .is_some_and(|e| e.reads_meters_per_pixel())
            })
        });
        let mut groups = std::collections::HashMap::new();
        let layout_leaders = (style.layers.iter().enumerate())
            .map(|(i, l)| {
                let leader = *groups.entry(l.layout_group?).or_insert(i);
                (leader != i).then_some(leader)
            })
            .collect();
        Self {
            revision,
            style,
            layer_keys,
            source_keys,
            nodes,
            sources,
            warnings,
            camera_layers,
            reads_resolution,
            layout_leaders,
        }
    }

    /// Words per feature of the node's property rows: the family's, then a
    /// circle's pitch flags or an outline's antialiasing, and the
    /// feature's translate, where they are per feature.
    pub(crate) fn row_words(&self, node: &NodePlan) -> usize {
        let family = self.family_row_words(node);
        let flags = self.circle_pitch_rows(node) || self.antialias_rows(node);
        let pitch = usize::from(flags);
        let translate = if self.translate_rows(node) {
            layout::TRANSLATE_ROW_WORDS
        } else {
            0
        };
        family + pitch + translate
    }

    /// A fill, line, extrusion, circle or symbol node's `*-translate` and
    /// its anchor.
    pub(crate) fn translate<'a>(
        &'a self,
        node: &NodePlan,
    ) -> Option<(
        PropertyRef<'a, [f32; 2]>,
        PropertyRef<'a, orb_style::TranslateAnchor>,
    )> {
        let l = &self.style.layers[node.layer];
        let (px, anchor) = translate_keys(l.ty, node.family)?;
        Some((l.get(px), l.get(anchor)))
    }

    /// Whether each feature's translate ends the node's rows, read per
    /// feature beyond the spec.
    pub(crate) fn translate_rows(&self, node: &NodePlan) -> bool {
        self.translate(node)
            .is_some_and(|(px, _)| matches!(px.kind(), Kind::Source | Kind::Composite))
    }

    /// Whether a circle node's rows carry each feature's pitch scale and
    /// alignment, read per feature beyond the spec.
    pub(crate) fn circle_pitch_rows(&self, node: &NodePlan) -> bool {
        let l = &self.style.layers[node.layer];
        node.family == Family::Circle
            && [
                orb_style::CIRCLE_PITCH_SCALE,
                orb_style::CIRCLE_PITCH_ALIGNMENT,
            ]
            .into_iter()
            .any(|key| matches!(l.get(key).kind(), Kind::Source | Kind::Composite))
    }

    /// Whether a fill outline node's rows carry each feature's
    /// `fill-antialias`, read per feature beyond the spec.
    pub(crate) fn antialias_rows(&self, node: &NodePlan) -> bool {
        let l = &self.style.layers[node.layer];
        node.family == Family::FillOutline
            && matches!(
                l.get(orb_style::FILL_ANTIALIAS).kind(),
                Kind::Source | Kind::Composite
            )
    }

    fn family_row_words(&self, node: &NodePlan) -> usize {
        match node.family {
            Family::Fill | Family::FillOutline => layout::FILL_ROW_WORDS,
            Family::Line => layout::LINE_ROW_WORDS,
            Family::FillExtrusion => {
                layout::EXTRUSION_ROW_WORDS + node.elevation.as_ref().map_or(0, |e| e.row_words())
            }
            Family::Circle => layout::CIRCLE_ROW_WORDS,
            Family::Heatmap => layout::HEATMAP_ROW_WORDS,
            Family::SymbolIcon | Family::SymbolText => {
                let daylight = node.daylight.is_some();
                layout::SYMBOL_ROW_WORDS + usize::from(daylight) * layout::SYMBOL_DAYLIGHT_ROW_WORDS
            }
            image => unreachable!("{image:?} nodes have no property rows"),
        }
    }

    /// The first of property `slot`'s two words in a row of the node: an
    /// extrusion's elevation colours after the first follow the family's
    /// words.
    pub(crate) fn row_word(&self, node: &NodePlan, slot: usize) -> usize {
        match node.family {
            Family::FillExtrusion if slot > 2 => layout::EXTRUSION_ROW_WORDS + 2 * (slot - 3),
            _ => 2 * slot,
        }
    }

    /// The index of the node with `key`.
    pub(crate) fn node(&self, key: u32) -> Option<usize> {
        self.nodes.iter().position(|n| n.key == key)
    }

    /// The index of the terrain's DEM source and its exaggeration.
    pub(crate) fn terrain(&self) -> Option<(usize, f64)> {
        let terrain = self.style.terrain.as_ref()?;
        let index = self.sources.iter().position(|s| s.id == terrain.source)?;
        Some((index, terrain.exaggeration))
    }

    /// The index of the source with `key`.
    pub(crate) fn source(&self, key: u32) -> Option<usize> {
        self.sources.iter().position(|s| s.key == key)
    }

    /// A drawn source's definition.
    pub(crate) fn source_def(&self, index: usize) -> &orb_style::Source {
        self.style
            .source(&self.sources[index].id)
            .expect("plan sources exist")
    }

    /// Whether a line node's vertices carry their progress for
    /// `line-gradient` (`orb_layout::line::METRICS_STRIDE`): those of a
    /// GeoJSON source with `lineMetrics`, as GL JS's carry `a_uv_x`.
    pub(crate) fn line_metrics(&self, node: &NodePlan) -> bool {
        node.family == Family::Line
            && matches!(&self.source_def(node.source).kind,
                SourceKind::GeoJson(g) if g.options.line_metrics)
    }

    /// Whether a node draws paint from the feature-state table, so its
    /// tiles carry feature ids.
    pub(crate) fn reads_state(&self, node: &NodePlan) -> bool {
        matches!(
            node.family,
            Family::Fill
                | Family::FillOutline
                | Family::FillExtrusion
                | Family::Line
                | Family::Circle
                | Family::SymbolIcon
                | Family::SymbolText
        ) && self
            .paint_properties(node)
            .iter()
            .any(PaintProperty::reads_state)
    }

    /// The source layer a node's features come from: `None` for GeoJSON.
    pub(crate) fn source_layer(&self, node: &NodePlan) -> Option<&str> {
        if matches!(self.source_def(node.source).kind, SourceKind::GeoJson(_)) {
            return None;
        }
        self.style.layers[node.layer].source_layer.as_deref()
    }

    /// A node's paint properties in property-row order.
    pub(crate) fn paint_properties<'a>(&'a self, node: &'a NodePlan) -> Vec<PaintProperty<'a>> {
        use PaintProperty::{Color as C, Number as N};
        use orb_style::{
            BACKGROUND_COLOR, BACKGROUND_OPACITY, CIRCLE_BLUR, CIRCLE_COLOR, CIRCLE_OPACITY,
            CIRCLE_RADIUS, CIRCLE_STROKE_COLOR, CIRCLE_STROKE_OPACITY, CIRCLE_STROKE_WIDTH,
            EXTRUSION_BASE, EXTRUSION_COLOR, EXTRUSION_HEIGHT, FILL_COLOR, FILL_OPACITY,
            FILL_OUTLINE_COLOR, HEATMAP_RADIUS, HEATMAP_WEIGHT, ICON_PART, LINE_BLUR, LINE_COLOR,
            LINE_GAP_WIDTH, LINE_OFFSET, LINE_OPACITY, LINE_WIDTH, TEXT_PART,
        };
        let l = &self.style.layers[node.layer];
        match l.ty {
            LayerType::Fill => {
                // Without an outline colour, outlines use the fill colour.
                let outline = l.get(FILL_OUTLINE_COLOR);
                let color = match (node.family, outline.constant()) {
                    (Family::Fill, _) | (_, Some(None)) => C(l.get(FILL_COLOR)),
                    _ => PaintProperty::OptionalColor(outline),
                };
                vec![color, N(l.get(FILL_OPACITY))]
            }
            // Over the elevation, a colour per stop
            // ([`crate::elevation::colour_slot`]).
            LayerType::FillExtrusion => match &node.elevation {
                Some(e) => {
                    let color = |slot| C(PropertyRef::of(slot, EXTRUSION_COLOR.id()));
                    let mut properties = vec![
                        color(&e.colors[0]),
                        N(l.get(EXTRUSION_HEIGHT)),
                        N(l.get(EXTRUSION_BASE)),
                    ];
                    properties.extend(e.colors[1..].iter().map(color));
                    properties
                }
                None => vec![
                    C(l.get(EXTRUSION_COLOR)),
                    N(l.get(EXTRUSION_HEIGHT)),
                    N(l.get(EXTRUSION_BASE)),
                ],
            },
            LayerType::Line => vec![
                C(l.get(LINE_COLOR)),
                N(l.get(LINE_OPACITY)),
                N(l.get(LINE_WIDTH)),
                N(l.get(LINE_GAP_WIDTH)),
                N(l.get(LINE_OFFSET)),
                N(l.get(LINE_BLUR)),
            ],
            LayerType::Circle => vec![
                C(l.get(CIRCLE_COLOR)),
                N(l.get(CIRCLE_RADIUS)),
                N(l.get(CIRCLE_BLUR)),
                N(l.get(CIRCLE_OPACITY)),
                C(l.get(CIRCLE_STROKE_COLOR)),
                N(l.get(CIRCLE_STROKE_WIDTH)),
                N(l.get(CIRCLE_STROKE_OPACITY)),
            ],
            LayerType::Heatmap => vec![N(l.get(HEATMAP_RADIUS)), N(l.get(HEATMAP_WEIGHT))],
            LayerType::Background => vec![C(l.get(BACKGROUND_COLOR)), N(l.get(BACKGROUND_OPACITY))],
            LayerType::Symbol => {
                let part = match node.family {
                    Family::SymbolIcon => ICON_PART,
                    _ => TEXT_PART,
                };
                // Over the daylight, the colours' day stops, then their
                // night stops ([`crate::daylight`]).
                let Some(d) = &node.daylight else {
                    return vec![
                        N(l.get(part.opacity)),
                        C(l.get(part.color)),
                        C(l.get(part.halo_color)),
                        N(l.get(part.halo_width)),
                        N(l.get(part.halo_blur)),
                    ];
                };
                vec![
                    N(l.get(part.opacity)),
                    C(d.day(l, 0, part.color)),
                    C(d.day(l, 1, part.halo_color)),
                    N(l.get(part.halo_width)),
                    N(l.get(part.halo_blur)),
                    C(d.night(l, 0, part.color)),
                    C(d.night(l, 1, part.halo_color)),
                ]
            }
            // Image families have no property rows.
            _ => Vec::new(),
        }
    }

    /// Whether a source's tiles are images, and of which kind.
    pub(crate) fn image_kind(&self, source: usize) -> Option<ImageKind> {
        match self.style.source(&self.sources[source].id)?.kind {
            SourceKind::Raster(_) | SourceKind::Image(_) => Some(ImageKind::Raster),
            SourceKind::RasterDem(_) => Some(ImageKind::Dem),
            _ => None,
        }
    }

    /// The pattern sampled by a node. An explicit fill outline colour uses
    /// the solid outline shader even when its interior is patterned.
    pub(crate) fn pattern(&self, node: &NodePlan) -> Option<PropertyRef<'_, Option<String>>> {
        let l = &self.style.layers[node.layer];
        let key = match l.ty {
            LayerType::Background => orb_style::BACKGROUND_PATTERN,
            LayerType::Fill
                if node.family != Family::FillOutline
                    || l.get(orb_style::FILL_OUTLINE_COLOR).constant() == Some(None) =>
            {
                orb_style::FILL_PATTERN
            }
            LayerType::Line => orb_style::LINE_PATTERN,
            LayerType::FillExtrusion => orb_style::EXTRUSION_PATTERN,
            _ => return None,
        };
        let p = l.get(key);
        (p.constant() != Some(None)).then_some(p)
    }
}

/// A layer type's `*-translate` and anchor properties, for those that can
/// have one per feature: fills, lines, extrusions, circles, and a symbol
/// layer's text or icon `family`.
pub(crate) fn translate_keys(
    ty: LayerType,
    family: Family,
) -> Option<(
    orb_style::Key<[f32; 2]>,
    orb_style::Key<orb_style::TranslateAnchor>,
)> {
    use orb_style::*;
    Some(match ty {
        LayerType::Fill => (FILL_TRANSLATE, FILL_TRANSLATE_ANCHOR),
        LayerType::Line => (LINE_TRANSLATE, LINE_TRANSLATE_ANCHOR),
        LayerType::FillExtrusion => (EXTRUSION_TRANSLATE, EXTRUSION_TRANSLATE_ANCHOR),
        LayerType::Circle => (CIRCLE_TRANSLATE, CIRCLE_TRANSLATE_ANCHOR),
        LayerType::Symbol if family == Family::SymbolText => {
            (TEXT_TRANSLATE, TEXT_TRANSLATE_ANCHOR)
        }
        LayerType::Symbol => (ICON_TRANSLATE, ICON_TRANSLATE_ANCHOR),
        _ => return None,
    })
}

//! A v8 style compiled into what the renderer draws: a background and one
//! render node per drawable layer, in style order.
//!
//! Layers and sources carry the stable keys `orb-style` gives them, so a
//! node keeps its key across edits that move it (`layer key × 2`, plus one
//! for a fill's outline node), and tile jobs and results name nodes and
//! sources by key.

use std::sync::Arc;

use orb_expr::gpu::{Layout, Program};
use orb_expr::{Interpolation, Kind};
use orb_render::{Curve, Family};
use orb_style::CostModel;
use orb_v8::{Color, LayerKind, Property, SourceKind, Style, Value};

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
    /// The paint the GPU interprets, if any.
    pub(crate) gpu: Option<Arc<NodeGpu>>,
}

/// A render node's bytecode paint (plan §11.3 `GpuBytecode`): per paint
/// slot its program, and the input columns and state values they share.
#[derive(Debug)]
pub(crate) struct NodeGpu {
    pub(crate) programs: Vec<Option<Program>>,
    pub(crate) layout: Layout,
}

impl NodeGpu {
    /// Bit `i`: slot `i` is interpreted.
    pub(crate) fn mask(&self) -> u32 {
        self.programs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .map(|(i, _)| 1 << i)
            .sum()
    }
}

/// A paint property as the renderer packs it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PaintProperty<'a> {
    Color(&'a Property<Color>),
    /// A colour that may be unset, such as `fill-outline-color` given as an
    /// expression; unset draws transparent.
    OptionalColor(&'a Property<Option<Color>>),
    Number(&'a Property<f32>),
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

    /// Whether the value reads both feature state and feature data, which
    /// rows evaluate without state.
    pub(crate) fn mixes_state(&self) -> bool {
        self.expression()
            .is_some_and(|e| e.reads_feature_state() && e.reads_feature_properties())
    }

    pub(crate) fn expression(&self) -> Option<&orb_expr::PropertyExpression> {
        match self {
            Self::Color(p) => p.as_expression(),
            Self::OptionalColor(p) => p.as_expression(),
            Self::Number(p) => p.as_expression(),
        }
    }

    /// The value a program falls back to: the property's default.
    pub(crate) fn default_value(&self) -> orb_expr::Value<'static> {
        match self {
            Self::Color(p) => orb_expr::Value::Color(p.default_value()),
            Self::OptionalColor(p) => {
                orb_expr::Value::Color(p.default_value().unwrap_or(orb_expr::Color::TRANSPARENT))
            }
            Self::Number(p) => orb_expr::Value::Number(f64::from(p.default_value())),
        }
    }

    /// A state lookup's value packed as a row word: for a feature with
    /// `state` at `zoom`.
    pub(crate) fn state_word(&self, zoom: f64, state: &[(String, Value<'static>)]) -> u32 {
        match self {
            Self::Color(p) => orb_render::pack_color(p.evaluate_state(zoom, state).premultiplied()),
            Self::OptionalColor(p) => orb_render::pack_color(
                p.evaluate_state(zoom, state)
                    .map_or([0.0; 4], |c| c.premultiplied()),
            ),
            Self::Number(p) => p.evaluate_state(zoom, state).to_bits(),
        }
    }

    /// The node constant of a state lookup: its value with no state.
    pub(crate) fn default_number(&self, zoom: f64) -> f32 {
        match self {
            Self::Number(p) => p.evaluate_state(zoom, &[]),
            _ => unreachable!("not a number"),
        }
    }

    pub(crate) fn default_color(&self, zoom: f64) -> [f32; 4] {
        match self {
            Self::Color(p) => p.evaluate_state(zoom, &[]).premultiplied(),
            Self::OptionalColor(p) => p
                .evaluate_state(zoom, &[])
                .map_or([0.0; 4], |c| c.premultiplied()),
            Self::Number(_) => unreachable!("not a colour"),
        }
    }

    /// How the property's row values blend with zoom.
    pub(crate) fn curve(&self) -> Curve {
        let interpolation = self
            .expression()
            .filter(|e| e.kind() == Kind::Composite)
            .and_then(|e| e.zoom_curve())
            .and_then(|c| c.interpolation);
        match interpolation {
            None => Curve::Step,
            Some(Interpolation::Linear) => Curve::Exponential(1.0),
            Some(Interpolation::Exponential(base)) => Curve::Exponential(base as f32),
            Some(Interpolation::CubicBezier(p)) => Curve::CubicBezier(p.map(|v| v as f32)),
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
        Self::new(Arc::new(style), layers, sources, 0, CostModel::DEFAULT)
    }

    pub(crate) fn new(
        style: Arc<Style>,
        layer_keys: Vec<u32>,
        source_keys: Vec<u32>,
        revision: u64,
        cost: CostModel,
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
            let (family, source) = match &layer.kind {
                LayerKind::Background(_) => {
                    nodes.push(NodePlan {
                        layer: index,
                        family: Family::Fill,
                        source: usize::MAX,
                        key: layer_keys[index] * 2,
                        gpu: None,
                    });
                    continue;
                }
                LayerKind::FillExtrusion(e) => (Family::FillExtrusion, &e.selection.source),
                LayerKind::Fill(fill) => (Family::Fill, &fill.selection.source),
                LayerKind::Line(line) => (Family::Line, &line.selection.source),
                LayerKind::Circle(c) => (Family::Circle, &c.selection.source),
                LayerKind::Heatmap(h) => (Family::Heatmap, &h.selection.source),
                LayerKind::Raster(raster) => (Family::Raster, &raster.source),
                LayerKind::Hillshade(hillshade) => (Family::Hillshade, &hillshade.source),
                LayerKind::ColorRelief(relief) => (Family::ColorRelief, &relief.source),
                LayerKind::Symbol(symbol) => (Family::SymbolIcon, &symbol.selection.source),
                LayerKind::Unsupported { .. } => continue,
            };
            let selection = match &layer.kind {
                LayerKind::Fill(fill) => Some(&fill.selection),
                LayerKind::FillExtrusion(e) => Some(&e.selection),
                LayerKind::Line(line) => Some(&line.selection),
                LayerKind::Circle(c) => Some(&c.selection),
                LayerKind::Heatmap(h) => Some(&h.selection),
                LayerKind::Symbol(s) => Some(&s.selection),
                _ => None,
            };
            match style.source(source).map(|s| &s.kind) {
                Some(SourceKind::Vector(_)) => {
                    if selection.is_some_and(|s| s.source_layer.is_none()) {
                        warn(index, "vector layers need a source-layer");
                        continue;
                    }
                }
                Some(SourceKind::GeoJson(source)) if !source.cluster => {}
                Some(SourceKind::GeoJson(_)) => {
                    warn(
                        index,
                        "GeoJSON clustering is not supported yet; the layer is skipped",
                    );
                    continue;
                }
                Some(SourceKind::Raster(_) | SourceKind::RasterDem(_)) => {}
                _ => {
                    warn(
                        index,
                        "only vector, GeoJSON, raster and raster-dem sources are drawn yet",
                    );
                    continue;
                }
            }
            if let LayerKind::Hillshade(hillshade) = &layer.kind {
                let p = &hillshade.paint;
                let lights = [
                    p.illumination_direction.len(),
                    p.illumination_altitude.len(),
                    p.shadow_color.len(),
                    p.highlight_color.len(),
                ];
                if lights.into_iter().max().unwrap_or(1) > orb_render::MAX_LIGHTS {
                    warn(
                        index,
                        &format!(
                            "hillshades draw at most {} light sources; the rest are ignored",
                            orb_render::MAX_LIGHTS
                        ),
                    );
                }
            }
            let source = match sources.iter().position(|s| s.id == *source) {
                Some(i) => i,
                None => {
                    let at = style.sources.iter().position(|s| s.id == *source);
                    sources.push(PlanSource {
                        key: source_keys[at.expect("layers' sources exist")],
                        id: source.clone(),
                    });
                    sources.len() - 1
                }
            };
            let key = layer_keys[index] * 2;
            nodes.push(NodePlan {
                layer: index,
                family,
                source,
                key,
                gpu: None,
            });
            // A symbol layer draws its icons, then its text (GL JS
            // `drawSymbols`).
            if matches!(layer.kind, LayerKind::Symbol(_)) {
                nodes.push(NodePlan {
                    layer: index,
                    family: Family::SymbolText,
                    source,
                    key: key + 1,
                    gpu: None,
                });
            }
            // GL JS antialiases fills by drawing their outlines after them.
            if let LayerKind::Fill(fill) = &layer.kind
                && fill.paint.antialias.evaluate(0.0)
            {
                nodes.push(NodePlan {
                    layer: index,
                    family: Family::FillOutline,
                    source,
                    key: key + 1,
                    gpu: None,
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

        let mut plan = Self {
            revision,
            style,
            layer_keys,
            source_keys,
            nodes,
            sources,
            warnings,
        };
        let gpu: Vec<Option<NodeGpu>> = plan.nodes.iter().map(|n| plan.gpu(n, cost)).collect();
        for (node, gpu) in plan.nodes.iter_mut().zip(gpu) {
            node.gpu = gpu.map(Arc::new);
        }
        for node in &plan.nodes {
            let mask = node.gpu.as_deref().map_or(0, NodeGpu::mask);
            if plan
                .paint_properties(node)
                .iter()
                .enumerate()
                .any(|(slot, p)| p.mixes_state() && mask & 1 << slot == 0)
            {
                let id = &plan.style.layers[node.layer].id;
                let warning = format!(
                    "layers[{}] ({id}): paint that reads both feature state and feature properties is drawn without state",
                    node.layer
                );
                if !plan.warnings.contains(&warning) {
                    plan.warnings.push(warning);
                }
            }
        }
        plan
    }

    /// The node's bytecode paint under `cost`, if any slot is interpreted.
    fn gpu(&self, node: &NodePlan, cost: CostModel) -> Option<NodeGpu> {
        let names = self.paint_names(node);
        let programs: Vec<Option<Program>> = self
            .paint_properties(node)
            .iter()
            .zip(names)
            .map(|(p, name)| cost.program(name, p.expression()?))
            .collect();
        if programs.iter().all(Option::is_none) {
            return None;
        }
        let layout = Layout::new(programs.iter().flatten());
        Some(NodeGpu { programs, layout })
    }

    /// The v8 names of a node's paint slots that may be interpreted, in
    /// [`paint_properties`](Self::paint_properties) order; empty for
    /// families without bytecode.
    pub(crate) fn paint_names(&self, node: &NodePlan) -> &'static [&'static str] {
        match (&self.style.layers[node.layer].kind, node.family) {
            (LayerKind::Fill(f), Family::FillOutline)
                if f.paint.outline_color.as_constant() != Some(&None) =>
            {
                &["fill-outline-color", "fill-opacity"]
            }
            (LayerKind::Fill(_), _) => &["fill-color", "fill-opacity"],
            (LayerKind::Line(_), _) => &[
                "line-color",
                "line-opacity",
                "line-width",
                "line-gap-width",
                "line-offset",
                "line-blur",
            ],
            (LayerKind::Circle(_), _) => &[
                "circle-color",
                "circle-radius",
                "circle-blur",
                "circle-opacity",
                "circle-stroke-color",
                "circle-stroke-width",
                "circle-stroke-opacity",
            ],
            _ => &[],
        }
    }

    /// Bit `i`: the node interprets paint slot `i` on the GPU.
    pub(crate) fn bytecode_mask(&self, node: &NodePlan) -> u32 {
        node.gpu.as_deref().map_or(0, NodeGpu::mask)
    }

    /// Words per feature of the node's property rows: the family's, then
    /// its input columns.
    pub(crate) fn row_words(&self, node: &NodePlan) -> usize {
        let family = match node.family {
            Family::Fill | Family::FillOutline => orb_render::FILL_ROW_WORDS,
            Family::Line => orb_render::LINE_ROW_WORDS,
            Family::FillExtrusion => orb_render::EXTRUSION_ROW_WORDS,
            Family::Circle => orb_render::CIRCLE_ROW_WORDS,
            Family::Heatmap => orb_render::HEATMAP_ROW_WORDS,
            Family::SymbolIcon | Family::SymbolText => orb_render::SYMBOL_ROW_WORDS,
            image => unreachable!("{image:?} nodes have no property rows"),
        };
        family
            + node
                .gpu
                .as_ref()
                .map_or(0, |g| g.layout.column_words() as usize)
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
    pub(crate) fn source_def(&self, index: usize) -> &orb_v8::Source {
        self.style
            .source(&self.sources[index].id)
            .expect("plan sources exist")
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
        ) && (self
            .paint_properties(node)
            .iter()
            .any(PaintProperty::is_state_lookup)
            || node
                .gpu
                .as_ref()
                .is_some_and(|g| g.layout.state_words() > 0))
    }

    /// The source layer a node's features come from: `None` for GeoJSON.
    pub(crate) fn source_layer(&self, node: &NodePlan) -> Option<&str> {
        if matches!(self.source_def(node.source).kind, SourceKind::GeoJson(_)) {
            return None;
        }
        match &self.style.layers[node.layer].kind {
            LayerKind::Fill(l) => l.selection.source_layer.as_deref(),
            LayerKind::FillExtrusion(l) => l.selection.source_layer.as_deref(),
            LayerKind::Line(l) => l.selection.source_layer.as_deref(),
            LayerKind::Circle(l) => l.selection.source_layer.as_deref(),
            LayerKind::Heatmap(l) => l.selection.source_layer.as_deref(),
            LayerKind::Symbol(l) => l.selection.source_layer.as_deref(),
            _ => None,
        }
    }

    /// A node's paint properties in property-row order.
    pub(crate) fn paint_properties(&self, node: &NodePlan) -> Vec<PaintProperty<'_>> {
        match &self.style.layers[node.layer].kind {
            LayerKind::Fill(fill) => {
                let p = &fill.paint;
                let color = match (node.family, p.outline_color.as_constant()) {
                    // Without an outline colour, outlines use the fill colour.
                    (Family::Fill, _) | (_, Some(None)) => PaintProperty::Color(&p.color),
                    _ => PaintProperty::OptionalColor(&p.outline_color),
                };
                vec![color, PaintProperty::Number(&p.opacity)]
            }
            LayerKind::FillExtrusion(e) => vec![
                PaintProperty::Color(&e.paint.color),
                PaintProperty::Number(&e.paint.height),
                PaintProperty::Number(&e.paint.base),
            ],
            LayerKind::Line(line) => {
                let p = &line.paint;
                vec![
                    PaintProperty::Color(&p.color),
                    PaintProperty::Number(&p.opacity),
                    PaintProperty::Number(&p.width),
                    PaintProperty::Number(&p.gap_width),
                    PaintProperty::Number(&p.offset),
                    PaintProperty::Number(&p.blur),
                ]
            }
            LayerKind::Circle(c) => {
                let p = &c.paint;
                vec![
                    PaintProperty::Color(&p.color),
                    PaintProperty::Number(&p.radius),
                    PaintProperty::Number(&p.blur),
                    PaintProperty::Number(&p.opacity),
                    PaintProperty::Color(&p.stroke_color),
                    PaintProperty::Number(&p.stroke_width),
                    PaintProperty::Number(&p.stroke_opacity),
                ]
            }
            LayerKind::Heatmap(h) => vec![
                PaintProperty::Number(&h.paint.radius),
                PaintProperty::Number(&h.paint.weight),
            ],
            LayerKind::Background(b) => vec![
                PaintProperty::Color(&b.paint.color),
                PaintProperty::Number(&b.paint.opacity),
            ],
            LayerKind::Symbol(symbol) => {
                let p = match node.family {
                    Family::SymbolIcon => &symbol.paint.icon,
                    _ => &symbol.paint.text,
                };
                vec![
                    PaintProperty::Number(&p.opacity),
                    PaintProperty::Color(&p.color),
                    PaintProperty::Color(&p.halo_color),
                    PaintProperty::Number(&p.halo_width),
                    PaintProperty::Number(&p.halo_blur),
                ]
            }
            // Image families have no property rows.
            _ => Vec::new(),
        }
    }

    /// Whether a source's tiles are images, and of which kind.
    pub(crate) fn image_kind(&self, source: usize) -> Option<ImageKind> {
        match self.style.source(&self.sources[source].id)?.kind {
            SourceKind::Raster(_) => Some(ImageKind::Raster),
            SourceKind::RasterDem(_) => Some(ImageKind::Dem),
            _ => None,
        }
    }

    /// The pattern sampled by a node. An explicit fill outline colour uses
    /// the solid outline shader even when its interior is patterned.
    pub(crate) fn pattern(&self, node: &NodePlan) -> Option<&Property<Option<String>>> {
        let p = match &self.style.layers[node.layer].kind {
            LayerKind::Background(b) => &b.paint.pattern,
            LayerKind::Fill(f)
                if node.family != Family::FillOutline
                    || f.paint.outline_color.as_constant() == Some(&None) =>
            {
                &f.paint.pattern
            }
            LayerKind::Line(l) => &l.paint.pattern,
            LayerKind::FillExtrusion(e) => &e.paint.pattern,
            _ => return None,
        };
        (p.as_constant() != Some(&None)).then_some(p)
    }
}

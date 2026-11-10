//! Node paint for a frame: each node's paint and visibility
//! at the frame's zoom and bearing, and the draped nodes' at their drape
//! zooms, written to the renderer when they change.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use orb_hal::Gpu;
use orb_render::{
    Curve, Family, FillPaint, ImageAtlas, ImageKey, LinePaint, NodePaint, PatternPaint, Renderer,
    Translate,
};
use orb_style::{
    CameraInputs, Kind, LINE_CAP, LINE_DASHARRAY, LINE_GRADIENT, LINE_PATTERN, LINE_WIDTH,
    LayerType, LineCap, TranslateAnchor, Value,
};

use crate::image;
use crate::map::LoadedStyle;
use crate::resident::SourceState;
use crate::style::{PaintProperty, StylePlan};

/// What a style's paint was last written for.
#[derive(Debug, Default)]
pub(crate) struct PaintState {
    /// The zoom and bearing node paint was written for, and whether
    /// draped nodes draped (on terrain they show at their slots' zooms,
    /// not the camera's); `None` to rewrite.
    written: Option<(f64, f64, bool)>,
    fade: Option<Crossfade>,
    /// Counts rewrites of all paint: what camera-free nodes were written
    /// in.
    epoch: u64,
    zoom_history: ZoomHistory,
    /// The draped nodes' paint at drape zooms.
    drape: DrapePaint,
    /// The camera inputs paint that reads the camera was written at.
    camera: Option<CameraInputs>,
}

impl PaintState {
    /// Every node's paint is written again with the next frame.
    pub(crate) fn invalidate(&mut self) {
        self.written = None;
    }
}

/// What `state` gives values that read the camera: its pitch, bearing
/// and ground resolution at the target.
pub(crate) fn camera_inputs(state: &orb_camera::CameraState) -> CameraInputs {
    CameraInputs {
        pitch: state.pitch,
        bearing: state.bearing,
        meters_per_pixel: state.resolution,
    }
}

/// The plan with its paint's camera reads (`["pitch"]`, `["bearing"]`,
/// `["meters-per-pixel"]`, beyond the spec) resolved to `camera`'s: `None`
/// when no paint reads the camera.
pub(crate) fn at_camera(plan: &StylePlan, camera: CameraInputs) -> Option<StylePlan> {
    if plan.camera_layers.is_empty() {
        return None;
    }
    let mut style = (*plan.style).clone();
    for &index in &plan.camera_layers {
        let layer = Arc::make_mut(&mut style.layers[index]);
        let ids: Vec<_> = layer.slots().map(|(id, _)| id).collect();
        for id in ids {
            let slot = layer.slot_mut(id);
            let Some(folded) = slot.as_expression().and_then(|e| e.at_camera(camera)) else {
                continue;
            };
            *slot = if folded.kind() == Kind::Constant {
                let value = folded.expression().evaluate(0.0).ok();
                orb_style::Slot::constant(value.map_or(Value::Null, Value::into_owned), true)
            } else {
                orb_style::Slot::expression(folded)
            };
        }
    }
    Some(StylePlan {
        style: Arc::new(style),
        ..plan.clone()
    })
}

/// Writes this frame's node paint: at `zoom` and `camera`, with
/// paint transitions eased in at `now`, and feature state as of frame time
/// `time` (the shaders'). Returns whether the pattern and dash cross-fade
/// is still fading, and images the style lacks.
pub(crate) fn frame(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    style: &mut LoadedStyle,
    transitions: &mut crate::transitions::Transitions,
    (zoom, inputs): (f64, CameraInputs),
    (now, time): (f64, f64),
    atlas: &mut ImageAtlas,
) -> (bool, Vec<String>) {
    let worker = Arc::clone(&style.worker);
    let plan = worker.plan();
    if style.states.prepare(gpu, renderer, plan, (zoom, time)) {
        style.paint.invalidate();
    }
    // Paint that reads the camera is written again as it moves.
    let bearing = inputs.bearing;
    let camera = at_camera(plan, inputs);
    if camera.is_some() && style.paint.camera != Some(inputs) {
        style.paint.camera = Some(inputs);
        style.paint.invalidate();
    }
    let plan = camera.as_ref().unwrap_or(plan);
    let crossfade = style.paint.zoom_history.update(zoom, now);
    let fading = crossfade.mix < 1.0
        && plan.nodes.iter().any(|n| {
            plan.pattern(n).is_some() || {
                let l = &plan.style.layers[n.layer];
                l.ty == LayerType::Line && l.get(orb_style::LINE_DASHARRAY).constant() != Some(None)
            }
        });
    // Paint transitions write from the style overlaid with their eased
    // values, and once more when they end.
    let overlay = transitions.overlay(plan, zoom, now);
    if overlay.is_some() || std::mem::take(&mut transitions.ran) {
        style.paint.invalidate();
    }
    let plan = overlay.as_ref().unwrap_or(plan);
    let missing = write_paint(
        gpu,
        renderer,
        (style, plan),
        (zoom, bearing),
        crossfade,
        atlas,
    );
    (fading, missing)
}

/// Writes the draped nodes' paint at each of `zooms` not written since it
/// changed, for drape layout `generation`: evaluated there,
/// with the cross-fade of a zoom at rest, and left out where their layers
/// don't draw.
pub(crate) fn drape(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (style, plan): (&mut LoadedStyle, &StylePlan),
    zooms: &std::collections::BTreeSet<u8>,
    (generation, inputs): (u64, CameraInputs),
    atlas: &mut ImageAtlas,
) {
    let bearing = inputs.bearing;
    let camera = at_camera(plan, inputs);
    let plan = camera.as_ref().unwrap_or(plan);
    let drape = &mut style.paint.drape;
    if drape.generation != Some(generation) {
        // A new atlas, or one whose slots all draw again.
        drape.generation = Some(generation);
        drape.painted.clear();
    }
    // Paint that reads the camera is draped again as it moves.
    if camera.is_some() && drape.camera != Some(inputs) {
        drape.camera = Some(inputs);
        drape.painted.clear();
    }
    if drape.bearing != bearing {
        drape.bearing = bearing;
        let turning = &drape.turning;
        drape.painted.retain(|z| !turning.contains(z));
    }
    for &zoom in zooms {
        if style.paint.drape.painted.contains(&zoom) {
            continue;
        }
        let at = PaintAt {
            zoom: f64::from(zoom),
            bearing,
            // At an integer zoom at rest: that zoom's images alone.
            crossfade: Crossfade {
                from_scale: 0.5,
                mix: 1.0,
            },
        };
        let (mut changed, mut turns) = (false, false);
        for (index, node) in plan
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.family.is_draped())
        {
            let layer = &plan.style.layers[node.layer];
            // A shaded hillshade drapes its slopes, unlit: the terrain pass
            // lights them from the frame's bearing, so its slots don't
            // turn with it.
            let shaded = renderer.shades_on_terrain(index as u32);
            let at = PaintAt {
                bearing: if shaded { 0.0 } else { at.bearing },
                ..at
            };
            let paint = (layer.is_visible_at(at.zoom))
                .then(|| node_paint(gpu, renderer, (style, plan), index as u32, at, atlas))
                .flatten();
            turns |= paint.is_some() && !shaded && turns_with_bearing(layer, at.zoom);
            changed |= renderer.set_drape_paint(gpu, index as u32, zoom, paint.as_ref());
        }
        let drape = &mut style.paint.drape;
        if changed {
            *drape.revisions.entry(zoom).or_default() += 1;
        }
        if turns {
            drape.turning.insert(zoom);
        } else {
            drape.turning.remove(&zoom);
        }
        drape.painted.insert(zoom);
    }
}

impl PaintState {
    /// What changes with the draped paint at drape zoom `zoom`: its
    /// revision, and the bearing if something drawn there turns with it.
    pub(crate) fn drape_key(&self, zoom: u8, bearing: f64) -> (u64, Option<u64>) {
        let revision = self.drape.revisions.get(&zoom).copied().unwrap_or(0);
        (
            revision,
            self.drape
                .turning
                .contains(&zoom)
                .then_some(bearing.to_bits()),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Crossfade {
    pub(crate) from_scale: f32,
    pub(crate) mix: f32,
}

/// The draped nodes' paint at the drape zooms slots are drawn at,
/// and what tells a slot's content changed with it.
#[derive(Debug, Default)]
struct DrapePaint {
    /// Zooms written since the paint last changed.
    painted: HashSet<u8>,
    /// Per zoom, a revision that changes with what is written there.
    revisions: HashMap<u8, u64>,
    /// Zooms where something drawn turns with the bearing, and the
    /// bearing they were written at.
    turning: HashSet<u8>,
    bearing: f64,
    /// The drape layout generation written for.
    generation: Option<u64>,
    /// The camera inputs paint that reads the camera was draped at.
    camera: Option<CameraInputs>,
}

#[derive(Debug, Default)]
struct ZoomHistory {
    zoom: Option<f64>,
    integer: f64,
    changed: Option<f64>,
}

impl ZoomHistory {
    fn update(&mut self, zoom: f64, now: f64) -> Crossfade {
        match self.zoom {
            None => self.integer = zoom.floor(),
            Some(old) if old.floor() < zoom.floor() => {
                self.integer = zoom.floor();
                self.changed = Some(now);
            }
            Some(old) if old.floor() > zoom.floor() => {
                self.integer = zoom.floor() + 1.0;
                self.changed = Some(now);
            }
            _ => {}
        }
        self.zoom = Some(zoom);
        let t = self
            .changed
            .map_or(1.0, |at| ((now - at) / 300.0).clamp(0.0, 1.0));
        let fraction = zoom - zoom.floor();
        if zoom > self.integer {
            Crossfade {
                from_scale: 2.0,
                mix: (fraction + (1.0 - fraction) * t) as f32,
            }
        } else {
            Crossfade {
                from_scale: 0.5,
                mix: (1.0 - (1.0 - t) * fraction) as f32,
            }
        }
    }
}

/// What a node's paint is evaluated at: a zoom, the bearing (for
/// viewport-anchored hillshade light) and the pattern cross-fade.
#[derive(Clone, Copy)]
struct PaintAt {
    zoom: f64,
    bearing: f64,
    crossfade: Crossfade,
}

/// Whether a node's paint is the same at every zoom, bearing and
/// crossfade: a vector node whose layer reads no camera (constant, per
/// feature, or per feature and zoom in rows, which don't change with the
/// camera), and draws no pattern, dash or gradient textures, which cross
/// fade with the zoom. Its visibility still follows the zoom.
fn camera_free(plan: &StylePlan, node: &crate::style::NodePlan) -> bool {
    let layer = &plan.style.layers[node.layer];
    matches!(
        node.family,
        Family::Fill
            | Family::FillOutline
            | Family::Line
            | Family::Circle
            | Family::Heatmap
            | Family::FillExtrusion
    ) && layer.slots().all(|(_, slot)| match slot.kind() {
        orb_expr::Kind::Constant | orb_expr::Kind::Source => true,
        // Rows hold it at two zooms; a state lookup's default is per zoom.
        orb_expr::Kind::Composite => slot
            .as_expression()
            .is_some_and(|e| e.reads_feature_properties()),
        orb_expr::Kind::Camera => false,
    }) && plan.pattern(node).is_none()
        && !crate::line::has_textures(layer)
}

/// Writes every node's paint and visibility for `zoom`, if it changed.
fn write_paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (style, plan): (&mut LoadedStyle, &StylePlan),
    (zoom, bearing): (f64, f64),
    crossfade: Crossfade,
    atlas: &mut ImageAtlas,
) -> Vec<String> {
    let mut missing = Vec::new();
    if style.paint.written.is_none() {
        // What draped nodes draw with at their drape zooms may have changed.
        style.paint.drape.painted.clear();
        style.paint.epoch += 1;
    }
    // Draped nodes on terrain draw at their slots' zooms, which hide them
    // where their layers don't draw ([`drape`]).
    let draping = renderer.terrain_on();
    if style.paint.written == Some((zoom, bearing, draping)) && style.paint.fade == Some(crossfade)
    {
        return missing;
    }
    // Only image nodes' paint reads the bearing (a viewport-anchored
    // hillshade's light): a turn alone repaints them.
    let turned_only = style
        .paint
        .written
        .is_some_and(|(z, _, d)| z == zoom && d == draping)
        && style.paint.fade == Some(crossfade);
    style.paint.written = Some((zoom, bearing, draping));
    style.paint.fade = Some(crossfade);
    for (index, node) in plan.nodes.iter().enumerate() {
        if turned_only && !node.family.is_image() {
            continue;
        }
        let state = style.nodes.entry(node.key).or_default();
        let index = index as u32;
        let layer = &plan.style.layers[node.layer];
        // A node its layer hides at this zoom draws nothing: its paint is
        // written when a zoom shows it.
        if node.family.is_symbol() {
            if !layer.is_visible_at(zoom) {
                renderer.set_visible(index, false);
                continue;
            }
            let lookup = style.states.lookup(index as usize);
            let paint = crate::labels::paint(plan, node, zoom, &state.maxima, lookup);
            renderer.set_visible(index, paint.is_some());
            if let Some(paint) = paint {
                renderer.set_paint(gpu, index, &NodePaint::Symbol(paint));
            }
            continue;
        }
        // A layer-wide extrusion opacity of zero hides the layer; a
        // per-feature one hides only its own buildings, in the shader.
        let visible = layer.is_visible_at(zoom)
            && (layer.ty != LayerType::FillExtrusion || {
                let opacity = layer.get(orb_style::EXTRUSION_OPACITY);
                is_per_feature(opacity.kind()) || opacity.evaluate(zoom) > 0.0
            });
        let draped = draping && node.family.is_draped();
        renderer.set_visible(
            index,
            if draped {
                layer.visibility == orb_style::Visibility::Visible
            } else {
                visible
            },
        );
        if !draped && !visible {
            continue;
        }
        // Paint that is the same for every camera was written this epoch.
        if state.painted == Some(style.paint.epoch) {
            continue;
        }
        if camera_free(plan, node) {
            state.painted = Some(style.paint.epoch);
        }
        if matches!(node.family, Family::Circle | Family::Heatmap) {
            let properties = plan.paint_properties(node).len();
            if state.maxima.len() < properties {
                state.maxima.resize(properties, 0.0);
            }
            let lookup = style.states.lookup(index as usize);
            crate::point::paint(
                gpu,
                renderer,
                plan,
                node,
                index,
                zoom,
                &state.maxima,
                lookup,
            );
            continue;
        }
        let at = PaintAt {
            zoom,
            bearing,
            crossfade,
        };
        let Some(paint) = node_paint(gpu, renderer, (style, plan), index, at, atlas) else {
            continue;
        };
        renderer.set_paint(gpu, index, &paint);
        if node.family.is_image() {
            continue;
        }
        // A pattern the same for every feature is the node's paint: the
        // style lacking it is reported here, as tiles report theirs.
        if let Some(pattern) = plan
            .pattern(node)
            .filter(|p| !matches!(p.kind(), orb_expr::Kind::Source | orb_expr::Kind::Composite))
            && visible
            && style.sprites_loading == 0
            && let Some(name) = pattern.evaluate(zoom.floor())
            && !name.is_empty()
            && !style.icons.contains(&name)
            && style.reported_missing.insert(name.clone())
        {
            missing.push(name);
        }
        // A fill or line layer below layer opacity 1, or blending other
        // than normally, composites once.
        let composite = match layer.ty {
            LayerType::Fill => Some((orb_style::FILL_LAYER_OPACITY, orb_style::FILL_LAYER_BLEND)),
            LayerType::Line => Some((orb_style::LINE_LAYER_OPACITY, orb_style::LINE_LAYER_BLEND)),
            _ => None,
        };
        if let Some((opacity, blend)) = composite {
            let (opacity, blend) = (layer.get(opacity), layer.get(blend));
            renderer.set_layer_composite(
                gpu,
                index,
                node.layer as u32,
                opacity.evaluate(zoom),
                layer_blend(blend.evaluate(zoom)),
            );
        }
    }
    missing
}

/// Whether a layer drawn at `zoom` turns with the map's bearing: a
/// viewport-anchored hillshade's light (one the terrain doesn't shade), or
/// a viewport-anchored translate.
fn turns_with_bearing(l: &orb_style::Layer, zoom: f64) -> bool {
    let translates = |px: orb_style::Key<[f32; 2]>, anchor: orb_style::Key<TranslateAnchor>| {
        let px = l.get(px);
        l.get(anchor).evaluate(zoom) == TranslateAnchor::Viewport
            && (is_per_feature(px.kind()) || px.evaluate(zoom) != [0.0; 2])
    };
    match l.ty {
        LayerType::Hillshade => {
            l.get(orb_style::HILLSHADE_ILLUMINATION_ANCHOR)
                .evaluate(zoom)
                == orb_style::IlluminationAnchor::Viewport
        }
        LayerType::Fill => translates(orb_style::FILL_TRANSLATE, orb_style::FILL_TRANSLATE_ANCHOR),
        LayerType::Line => translates(orb_style::LINE_TRANSLATE, orb_style::LINE_TRANSLATE_ANCHOR),
        _ => false,
    }
}

/// Node `index`'s paint at `at`, for a fill, line, background, extrusion
/// or image node; `None` for an image node that can't draw yet.
fn node_paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (style, plan): (&mut LoadedStyle, &StylePlan),
    index: u32,
    at: PaintAt,
    atlas: &mut ImageAtlas,
) -> Option<NodePaint> {
    let PaintAt {
        zoom,
        bearing,
        crossfade,
    } = at;
    let node = &plan.nodes[index as usize];
    let state = style.nodes.entry(node.key).or_default();
    if node.family.is_image() {
        let unpack = match &style.sources[node.source] {
            SourceState::Ready { endpoint, .. } => endpoint.dem_unpack,
            _ => None,
        };
        let context = (plan, node, index);
        return image::paint(
            gpu,
            renderer,
            context,
            zoom,
            bearing,
            unpack,
            &mut state.ramp,
        );
    }
    let properties = plan.paint_properties(node);
    if state.maxima.len() < properties.len() {
        state.maxima.resize(properties.len(), 0.0);
    }
    let lookup = style.states.lookup(index as usize);
    let maxima = &state.maxima;
    let mut table_mask = 0;
    let mut curves = [Curve::Step; 6];
    let mut numbers = [0.0f32; 6];
    let mut color = [0.0f32; 4];
    // An extrusion's further elevation colours are its curve's.
    let base_slots = match node.family {
        Family::FillExtrusion => 3,
        _ => properties.len(),
    };
    for (slot, property) in properties.iter().enumerate().take(base_slots) {
        if property.per_feature() {
            table_mask |= 1 << slot;
            curves[slot] = curve(property);
            numbers[slot] = maxima[slot];
        } else if property.is_state_lookup() {
            curves[slot] = curve(property);
            match property {
                PaintProperty::Number(_) => numbers[slot] = property.default_number(zoom),
                _ => color = property.default_color(zoom),
            }
        } else {
            match property {
                PaintProperty::Number(p) => numbers[slot] = p.evaluate(zoom),
                color_property => color = color_property.color(zoom),
            }
        }
    }
    let layer = &plan.style.layers[node.layer];
    let translate = translate(plan, node, zoom, maxima.get(properties.len()));
    let paint = match node.family {
        Family::Fill | Family::FillOutline => NodePaint::Fill(FillPaint {
            color,
            opacity: numbers[1],
            translate,
            table_mask,
            curves: [curves[0], curves[1]],
            state: lookup,
        }),
        Family::FillExtrusion => {
            let e = layer;
            // Per feature (beyond the spec), from the rows' flags: the
            // layer composites at full opacity.
            let opacity = e.get(orb_style::EXTRUSION_OPACITY);
            let gradient = e.get(orb_style::EXTRUSION_VERTICAL_GRADIENT);
            let opacity_rows = is_per_feature(opacity.kind());
            let gradient_rows = is_per_feature(gradient.kind());
            let opacity_curve = opacity_rows
                .then(|| opacity.as_expression().map(expression_curve))
                .flatten()
                .unwrap_or(Curve::Step);
            NodePaint::Extrusion(orb_render::ExtrusionPaint {
                color,
                height: numbers[1],
                base: numbers[2],
                opacity: if opacity_rows {
                    1.0
                } else {
                    opacity.evaluate(zoom)
                },
                translate,
                vertical_gradient: gradient.evaluate(zoom),
                table_mask: table_mask
                    | if opacity_rows {
                        orb_render::EXTRUSION_OPACITY_ROWS
                    } else {
                        0
                    }
                    | if gradient_rows {
                        orb_render::EXTRUSION_GRADIENT_ROWS
                    } else {
                        0
                    },
                curves: [
                    node.elevation.as_ref().map_or(curves[0], |e| e.zoom),
                    curves[1],
                    curves[2],
                    opacity_curve,
                ],
                state: lookup,
                elevation: node.elevation.as_ref().map(|e| e.paint(&properties, zoom)),
            })
        }
        Family::Line => NodePaint::Line(LinePaint {
            color,
            opacity: numbers[1],
            translate,
            width: numbers[2],
            gap_width: numbers[3],
            offset: numbers[4],
            blur: numbers[5],
            table_mask,
            curves,
            dash: None,
            gradient: false,
            metrics: plan.line_metrics(node),
            state: lookup,
        }),
        image => unreachable!("{image:?} nodes are painted above"),
    };
    let paint = line_paint(
        gpu,
        paint,
        layer,
        atlas,
        style.image_namespace.unwrap(),
        zoom,
        crossfade,
    );
    let paint = pattern_paint(
        paint,
        plan.pattern(node).as_ref(),
        style.image_namespace.unwrap(),
        atlas,
        zoom,
        crossfade,
        match layer.ty {
            LayerType::Line => layer.get(orb_style::LINE_WIDTH).evaluate(zoom.floor()),
            _ => 0.0,
        },
    );
    let paint = if layer.ty == LayerType::Background && plan.pattern(node).is_none() {
        match paint {
            NodePaint::Fill(p) => NodePaint::Background(p),
            _ => unreachable!("solid background"),
        }
    } else {
        paint
    };
    Some(paint)
}

/// The renderer's blend for a v8 layer blend.
fn layer_blend(blend: orb_style::LayerBlend) -> orb_render::Blend {
    use orb_render::Blend;
    use orb_style::LayerBlend;
    match blend {
        LayerBlend::Normal => Blend::Normal,
        LayerBlend::Multiply => Blend::Multiply,
        LayerBlend::Screen => Blend::Screen,
        LayerBlend::Overlay => Blend::Overlay,
        LayerBlend::Color => Blend::Color,
        LayerBlend::Luminosity => Blend::Luminosity,
        LayerBlend::PlusLighter => Blend::PlusLighter,
    }
}

fn pattern_paint(
    paint: NodePaint,
    property: Option<&orb_style::PropertyRef<'_, Option<String>>>,
    namespace: u64,
    atlas: &mut ImageAtlas,
    zoom: f64,
    crossfade: Crossfade,
    floor_width: f32,
) -> NodePaint {
    let Some(property) = property else {
        return paint;
    };
    let per_feature = matches!(
        property.kind(),
        orb_expr::Kind::Source | orb_expr::Kind::Composite
    );
    let image = |atlas: &mut ImageAtlas, z| {
        property
            .evaluate(z)
            .and_then(|name| atlas.resolve(namespace, &name).ok())
            .unwrap_or(0)
    };
    let mut pattern = PatternPaint {
        from: image(
            atlas,
            zoom.floor()
                + if crossfade.from_scale > 1.0 {
                    -1.0
                } else {
                    1.0
                },
        ),
        to: image(atlas, zoom.floor()),
        from_scale: crossfade.from_scale,
        to_scale: 1.0,
        mix: crossfade.mix,
        floor_width: 0.0,
    };
    // A pattern per feature reads the table's seventh column.
    let mask = if per_feature { 64 } else { 0 };
    match paint {
        NodePaint::Fill(p) => NodePaint::FillPattern(
            FillPaint {
                table_mask: p.table_mask | mask,
                ..p
            },
            pattern,
        ),
        NodePaint::Extrusion(p) => NodePaint::ExtrusionPattern(
            orb_render::ExtrusionPaint {
                table_mask: p.table_mask | mask,
                ..p
            },
            pattern,
        ),
        NodePaint::Line(p) => {
            pattern.floor_width = floor_width;
            NodePaint::LinePattern(
                LinePaint {
                    table_mask: p.table_mask | mask,
                    ..p
                },
                pattern,
            )
        }
        _ => unreachable!("base paint"),
    }
}

/// A line node's dash and gradient paint, its dashes resolved in `atlas`.
fn line_paint(
    gpu: &mut dyn Gpu,
    paint: NodePaint,
    line: &orb_style::Layer,
    atlas: &mut ImageAtlas,
    namespace: u64,
    zoom: f64,
    crossfade: Crossfade,
) -> NodePaint {
    let NodePaint::Line(mut paint) = paint else {
        return paint;
    };
    if line.ty != LayerType::Line {
        return NodePaint::Line(paint);
    }
    if line.get(LINE_PATTERN).constant() != Some(None) {
        return NodePaint::Line(paint);
    }
    paint.gradient = line.ramp(LINE_GRADIENT).is_some();
    let p = line.get(LINE_DASHARRAY);
    if p.constant() != Some(None) {
        // The pin only evaluates caps per feature for data-driven dashes.
        // Constant dashes use the cap's constant fallback for their SDF.
        let per_feature = matches!(p.kind(), Kind::Source | Kind::Composite);
        if per_feature {
            paint.table_mask |= 128;
        }
        let mut resolve = |z| {
            p.evaluate(z)
                .and_then(|array| {
                    atlas
                        .resolve_key(
                            gpu,
                            namespace,
                            &ImageKey::Dash {
                                array: array.iter().map(|v| v.to_bits()).collect(),
                                round: line.get(LINE_CAP).evaluate(zoom) == LineCap::Round,
                            },
                        )
                        .ok()
                })
                .unwrap_or(0)
        };
        paint.dash = Some(PatternPaint {
            from: if per_feature {
                0
            } else {
                resolve(
                    zoom.floor()
                        + if crossfade.from_scale > 1.0 {
                            -1.0
                        } else {
                            1.0
                        },
                )
            },
            to: if per_feature {
                0
            } else {
                resolve(zoom.floor())
            },
            from_scale: crossfade.from_scale,
            to_scale: 1.0,
            mix: crossfade.mix,
            floor_width: line.get(LINE_WIDTH).evaluate(zoom.floor()),
        });
    }
    NodePaint::Line(paint)
}

/// How a property's row values blend with zoom.
pub(crate) fn curve(property: &PaintProperty<'_>) -> Curve {
    property.expression().map_or(Curve::Step, expression_curve)
}

/// A fill, line, extrusion or circle node's translate at `zoom`. Per
/// feature (beyond the spec) it ends the rows, and `reach`, the node's
/// maximum after its slots, is the farthest any feature's goes.
pub(crate) fn translate(
    plan: &StylePlan,
    node: &crate::style::NodePlan,
    zoom: f64,
    reach: Option<&f32>,
) -> Translate {
    let Some((px, anchor)) = plan.translate(node) else {
        return Translate::default();
    };
    let viewport = anchor.evaluate(zoom) == TranslateAnchor::Viewport;
    match px.as_expression().filter(|e| is_per_feature(e.kind())) {
        Some(e) => Translate {
            px: [reach.copied().unwrap_or(0.0), 0.0],
            viewport,
            rows: Some(expression_curve(e)),
        },
        None => Translate {
            px: px.evaluate(zoom),
            viewport,
            rows: None,
        },
    }
}

/// Whether values of `kind` come from per-feature rows.
fn is_per_feature(kind: Kind) -> bool {
    matches!(kind, Kind::Source | Kind::Composite)
}

/// How a composite value's rows blend between the tile's zoom and the
/// next.
pub(crate) fn expression_curve(expression: &orb_expr::PropertyExpression) -> Curve {
    use orb_expr::Interpolation;
    let interpolation = Some(expression)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_crossfade_tracks_integer_crossings_and_settles() {
        let mut h = ZoomHistory::default();
        assert_eq!(h.update(4.5, 0.0).mix, 1.0);
        let up = h.update(5.25, 10.0);
        assert_eq!(
            up,
            Crossfade {
                from_scale: 2.0,
                mix: 0.25
            }
        );
        assert_eq!(h.update(5.25, 160.0).mix, 0.625);
        assert_eq!(h.update(5.25, 310.0).mix, 1.0);
        let down = h.update(4.75, 320.0);
        assert_eq!(
            down,
            Crossfade {
                from_scale: 0.5,
                mix: 0.25
            }
        );
        assert_eq!(h.update(4.75, 620.0).mix, 1.0);
    }
}

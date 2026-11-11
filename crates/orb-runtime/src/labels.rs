//! Labels on the scene side: resident tiles' symbol
//! meshes, the glyph rows they lease, rasterization of new glyphs, and the
//! placement lane's inputs and results.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use orb_hal::Gpu;
use orb_render::{
    GlyphAtlas, ImageAtlas, MeshId, OutOfSpace, PLACED_WORDS, Renderer, SymbolMeshData, Words,
    pack_image_rect,
};
use orb_symbol::{
    AnchorOffset, Committed, PlacedSymbol, PlacementTables, Quad, SizeKind, SortKeyRange,
    SymbolInstance,
};
use orb_text::GlyphKey;

use crate::jobs::TileResult;
use crate::placement::PlacementLane;
use crate::resident::{Resident, ResidentTile};
use crate::state::FeatureStates;
use crate::style::StylePlan;
use crate::symbol::{ResultSymbols, SymbolTable};
use crate::{GlyphBitmap, GlyphRaster, Jobs};

// Layout sizes glyph quads by the padding the atlas packs glyphs with.
const _: () = assert!(orb_symbol::GLYPH_PADDING == orb_render::GLYPH_PADDING);

/// Buckets dropped since the last frame, with the glyph rows they held.
type Retired = Arc<Mutex<Vec<(u64, Vec<u32>)>>>;

/// A symbol layer's bucket of a resident tile. Dropping it hands its glyph
/// rows and placement bucket back, however the tile leaves the scene.
#[derive(Debug)]
pub(crate) struct ResidentSymbols {
    /// The layer's key.
    pub(crate) layer: u32,
    pub(crate) bucket: u64,
    /// The largest `symbol-height-offset` its labels lift by, metres.
    pub(crate) max_height_offset: f64,
    rows: Vec<u32>,
    retired: Retired,
}

impl Drop for ResidentSymbols {
    fn drop(&mut self) {
        let rows = std::mem::take(&mut self.rows);
        if let Ok(mut retired) = self.retired.lock() {
            retired.push((self.bucket, rows));
        }
    }
}

/// A glyph as leases know it: its key and, for PBF glyphs, the style's
/// glyphs template, since two styles' stacks of one name are different
/// fonts.
type LeaseKey = (Option<String>, GlyphKey);

/// A glyph row's raster request: the glyph's index in its job, the row,
/// and the lease it was made for, since the row may be released and leased
/// again before the bitmap arrives.
pub(crate) type RasterRow = (u32, u32, u64);

/// A held bucket (see [`Labels::hold`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// Drawn as last placed until the commit of a request after this many.
    Waiting { requests: u64 },
    /// Fading out since the last commit.
    Fading,
}

/// Glyph atlas rows leased by glyph key, and the rasterization they wait
/// for.
#[derive(Debug, Default)]
struct GlyphRows {
    leases: HashMap<LeaseKey, (u32, u32)>,
    /// Per leased row, its key and lease.
    keys: HashMap<u32, (LeaseKey, u64)>,
    next_lease: u64,
    /// Per PBF range URL, the code points and rows to rasterize.
    pending: HashMap<String, Vec<(u32, u32, u64)>>,
    /// Per font stack, the local glyphs and rows to draw.
    pending_local: HashMap<String, Vec<(Box<str>, u32, u64)>>,
    /// Per declared font file, the glyphs and rows to draw.
    pending_files: HashMap<orb_text::FontId, Vec<(Box<str>, u32, u64)>>,
    /// Raster jobs out.
    in_flight: u32,
}

impl GlyphRows {
    /// A row for `key`, rasterized once (row 0 when the atlas is full).
    fn lease(&mut self, atlas: &mut GlyphAtlas, key: &GlyphKey, template: Option<&str>) -> u32 {
        let provider = matches!(key, GlyphKey::Pbf { .. })
            .then(|| template.map(str::to_string))
            .flatten();
        let lease_key = (provider, key.clone());
        if let Some((row, refs)) = self.leases.get_mut(&lease_key) {
            *refs += 1;
            return *row;
        }
        let Ok(row) = atlas.reserve() else {
            return 0;
        };
        self.next_lease += 1;
        let lease = self.next_lease;
        self.leases.insert(lease_key.clone(), (row, 1));
        self.keys.insert(row, (lease_key, lease));
        match key {
            GlyphKey::Pbf { stack, code_point } => {
                if let Some(template) = template {
                    let url = orb_text::glyphs_url(template, stack, code_point / 256);
                    self.pending
                        .entry(url)
                        .or_default()
                        .push((*code_point, row, lease));
                }
            }
            GlyphKey::Cluster { stack, text, .. } => self
                .pending_local
                .entry(stack.as_str().to_string())
                .or_default()
                .push((text.clone(), row, lease)),
            GlyphKey::Font { font, text, .. } => self
                .pending_files
                .entry(*font)
                .or_default()
                .push((text.clone(), row, lease)),
        }
        row
    }

    fn release(&mut self, gpu: &mut dyn Gpu, atlas: &mut GlyphAtlas, row: u32) {
        if row == 0 {
            return;
        }
        let (key, _) = self.keys.get(&row).expect("leased rows have keys").clone();
        let (_, refs) = self.leases.get_mut(&key).expect("leased");
        *refs -= 1;
        if *refs == 0 {
            self.leases.remove(&key);
            self.keys.remove(&row);
            atlas.release(gpu, row);
        }
    }

    /// Writes a raster job's bitmaps into the rows still waiting for them.
    fn apply(
        &mut self,
        gpu: &mut dyn Gpu,
        atlas: &mut GlyphAtlas,
        rows: &[RasterRow],
        bitmaps: &[GlyphBitmap],
    ) {
        self.in_flight -= 1;
        for b in bitmaps {
            for &(_, row, lease) in rows.iter().filter(|(i, _, _)| *i == b.index) {
                // The row may have been released, and leased again since.
                let current = self.keys.get(&row).is_some_and(|(_, l)| *l == lease);
                if current && atlas.upload(gpu, row, b.size, &b.sdf).is_err() {
                    return;
                }
            }
        }
    }
}

/// Whether two placement frames place the same, clock aside: the camera,
/// the drawn buckets, the settings and the terrain the labels stand on.
fn same_inputs(a: &orb_symbol::Frame, b: &orb_symbol::Frame) -> bool {
    a.camera == b.camera
        && a.layers == b.layers
        && a.cross_source_collisions == b.cross_source_collisions
        && a.ground == b.ground
        && a.scope == b.scope
        && a.stable_labels == b.stable_labels
}

/// Per bucket, the world copies a placement draws.
type WorldCopies = HashMap<u64, Vec<i32>>;

/// A resident bucket: the meshes its states go to, and what its text
/// size is evaluated from.
#[derive(Clone, Debug, Default)]
struct BucketMeshes {
    text: Option<MeshId>,
    icon: Option<MeshId>,
    text_size: SizeKind,
    tile_zoom: f64,
    /// The states last written, per drawn world copy, to write again when
    /// the meshes make room for more copies.
    written: HashMap<i32, orb_symbol::BucketStates>,
    /// The bucket's placement tables and, per instance, what it reads as
    /// (for [`shown`](Labels::shown)).
    tables: Arc<PlacementTables>,
    names: Vec<Box<str>>,
}

impl BucketMeshes {
    fn write(&self, gpu: &mut dyn Gpu, renderer: &mut Renderer, wrap: i32) {
        let states = &self.written[&wrap];
        for (mesh, words, order) in [
            (self.text, &states.text, &states.text_order),
            (self.icon, &states.icon, &states.icon_order),
        ] {
            let Some(mesh) = mesh else { continue };
            renderer.write_symbol_states(gpu, mesh, wrap, words);
            if !order.is_empty() {
                renderer.write_symbol_order(gpu, mesh, order);
            }
        }
    }
}

/// Where symbol buckets' tiles are: each bucket's tile units
/// ([`orb_symbol::EXTENT`]) to world, and the surface's wrap period.
pub(crate) struct BucketPlanes {
    pub(crate) to_world: HashMap<u64, glam::DAffine2>,
    /// The surface's wrap period, 0 if it doesn't wrap.
    pub(crate) period: f64,
}

/// A drawn tile's bucket of a symbol layer, for this frame's placement.
#[derive(Clone, Debug)]
pub(crate) struct SymbolTile {
    /// Tile units ([`orb_symbol::EXTENT`]) to world in the primary copy:
    /// the tile's bounds, or a warped tile's linear approximation at its
    /// centre (placement collides labels in it; the GPU draws them through
    /// the warp).
    pub(crate) to_world: glam::DAffine2,
    /// The layer's key.
    pub(crate) layer: u32,
    pub(crate) bucket: u64,
    pub(crate) tile: orb_crs::TileIndex,
    pub(crate) wrap: i32,
    /// The tile is a Web Mercator tile drawn whole (not warped), so
    /// `to_world` is exact Web Mercator: points past it go on exactly.
    pub(crate) mercator: bool,
    /// On a globe, the tile's patches on the sphere, which placement
    /// projects through.
    pub(crate) globe: Option<Arc<orb_crs::Series<glam::DVec3>>>,
}

pub(crate) fn overlap(mode: Option<orb_style::Overlap>, allow: bool) -> orb_symbol::Overlap {
    match mode {
        Some(orb_style::Overlap::Never) => orb_symbol::Overlap::Never,
        Some(orb_style::Overlap::Always) => orb_symbol::Overlap::Always,
        Some(orb_style::Overlap::Cooperative) => orb_symbol::Overlap::Cooperative,
        None if allow => orb_symbol::Overlap::Always,
        None => orb_symbol::Overlap::Never,
    }
}

/// `text-rotation-alignment` is `map`, after `auto` (GL JS `recalculate`).
pub(crate) fn text_rotates_with_map(symbol: &orb_style::Layer, zoom: f64) -> bool {
    match symbol
        .get(orb_style::TEXT_ROTATION_ALIGNMENT)
        .evaluate(zoom)
    {
        orb_style::TextRotationAlignment::Map => true,
        orb_style::TextRotationAlignment::Auto => {
            symbol.get(orb_style::SYMBOL_PLACEMENT).evaluate(zoom)
                != orb_style::SymbolPlacement::Point
        }
        _ => false,
    }
}

/// `text-pitch-alignment` is `map`, after `auto`.
pub(crate) fn text_pitches_with_map(symbol: &orb_style::Layer, zoom: f64) -> bool {
    match symbol.get(orb_style::TEXT_PITCH_ALIGNMENT).evaluate(zoom) {
        orb_style::Alignment::Map => true,
        orb_style::Alignment::Viewport => false,
        orb_style::Alignment::Auto => text_rotates_with_map(symbol, zoom),
    }
}

/// A symbol layer's placement settings at `zoom` and `camera`: values
/// that read the camera (beyond the spec) follow it at each placement.
fn layer_params(
    plan: &StylePlan,
    node: &crate::style::NodePlan,
    zoom: f64,
    camera: orb_style::CameraInputs,
) -> orb_symbol::LayerParams {
    let layer = &plan.style.layers[node.layer];
    let symbol = layer;
    let l = symbol;
    let z_order = l.get(orb_style::SYMBOL_Z_ORDER).evaluate(zoom);
    let point =
        l.get(orb_style::SYMBOL_PLACEMENT).evaluate(zoom) == orb_style::SymbolPlacement::Point;
    let writing_modes = if point {
        let mut modes = Vec::new();
        for m in l
            .get(orb_style::TEXT_WRITING_MODE)
            .evaluate(zoom)
            .unwrap_or_default()
        {
            let m = match m {
                orb_style::WritingMode::Horizontal => orb_text::WritingMode::Horizontal,
                orb_style::WritingMode::Vertical => orb_text::WritingMode::Vertical,
            };
            if !modes.contains(&m) {
                modes.push(m);
            }
        }
        if modes.is_empty() {
            modes.push(orb_text::WritingMode::Horizontal);
        }
        modes
    } else {
        Vec::new()
    };
    let translate = |p: orb_style::SymbolPart| {
        let t = l.get(p.translate).evaluate_camera(zoom, camera);
        (
            [f64::from(t[0]), f64::from(t[1])],
            l.get(p.translate_anchor).evaluate(zoom) == orb_style::TranslateAnchor::Viewport,
        )
    };
    orb_symbol::LayerParams {
        layer: crate::style::layer_key(node.key),
        follows: plan.layout_leaders[node.layer].map(|leader| plan.layer_keys[leader]),
        source: plan.sources[node.source].key,
        minzoom: layer.minzoom,
        maxzoom: if layer.maxzoom.is_finite() {
            layer.maxzoom
        } else {
            0.0
        },
        flags: orb_symbol::SymbolFlags {
            text_overlap: overlap(
                l.get(orb_style::TEXT_OVERLAP).evaluate_camera(zoom, camera),
                l.get(orb_style::TEXT_ALLOW_OVERLAP)
                    .evaluate_camera(zoom, camera),
            ),
            icon_overlap: overlap(
                l.get(orb_style::ICON_OVERLAP).evaluate_camera(zoom, camera),
                l.get(orb_style::ICON_ALLOW_OVERLAP)
                    .evaluate_camera(zoom, camera),
            ),
            text_optional: l
                .get(orb_style::TEXT_OPTIONAL)
                .evaluate_camera(zoom, camera),
            icon_optional: l
                .get(orb_style::ICON_OPTIONAL)
                .evaluate_camera(zoom, camera),
            text_ignore_placement: l
                .get(orb_style::TEXT_IGNORE_PLACEMENT)
                .evaluate_camera(zoom, camera),
            icon_ignore_placement: l
                .get(orb_style::ICON_IGNORE_PLACEMENT)
                .evaluate_camera(zoom, camera),
            avoid_edges: l
                .get(orb_style::SYMBOL_AVOID_EDGES)
                .evaluate_camera(zoom, camera),
        },
        own_flags: own_symbol_flags(symbol),
        rotate_with_map: text_rotates_with_map(symbol, zoom),
        pitch_with_map: text_pitches_with_map(symbol, zoom),
        icon_text_fit: l.get(orb_style::ICON_TEXT_FIT).evaluate(zoom)
            != orb_style::IconTextFit::None,
        z_order_viewport_y: z_order == orb_style::SymbolZOrder::ViewportY,
        draw_by_viewport_y: draws_by_viewport_y(symbol, zoom),
        sort_across_tiles: z_order != orb_style::SymbolZOrder::ViewportY
            && matches!(
                l.get(orb_style::SYMBOL_SORT_KEY).kind(),
                orb_style::Kind::Source | orb_style::Kind::Composite
            ),
        height_anchor_ground: l.get(orb_style::SYMBOL_HEIGHT_ANCHOR).evaluate(zoom)
            == orb_style::HeightAnchor::Ground,
        text_padding: l.get(orb_style::TEXT_PADDING).evaluate(zoom),
        text_translate: translate(orb_style::TEXT_PART),
        icon_translate: translate(orb_style::ICON_PART),
        writing_modes,
        variable_placement: has_variable_placement(symbol),
    }
}

/// GL JS `sortFeaturesByY`: symbols that may overlap draw ordered by
/// viewport y, unless the layer sorts by key.
/// GL JS `sortFeaturesByKey`.
fn draws_by_key(symbol: &orb_style::Layer, zoom: f64) -> bool {
    let l = symbol;
    l.get(orb_style::SYMBOL_Z_ORDER).evaluate(zoom) != orb_style::SymbolZOrder::ViewportY
        && matches!(
            l.get(orb_style::SYMBOL_SORT_KEY).kind(),
            orb_style::Kind::Source | orb_style::Kind::Composite
        )
}

fn draws_by_viewport_y(symbol: &orb_style::Layer, zoom: f64) -> bool {
    let l = symbol;
    let z_order = l.get(orb_style::SYMBOL_Z_ORDER).evaluate(zoom);
    let by_key = draws_by_key(symbol, zoom);
    let by_y = z_order == orb_style::SymbolZOrder::ViewportY
        || (z_order == orb_style::SymbolZOrder::Auto && !by_key);
    let can_overlap = overlap(
        l.get(orb_style::TEXT_OVERLAP).evaluate(zoom),
        l.get(orb_style::TEXT_ALLOW_OVERLAP).evaluate(zoom),
    ) != orb_symbol::Overlap::Never
        || overlap(
            l.get(orb_style::ICON_OVERLAP).evaluate(zoom),
            l.get(orb_style::ICON_ALLOW_OVERLAP).evaluate(zoom),
        ) != orb_symbol::Overlap::Never
        || l.get(orb_style::TEXT_IGNORE_PLACEMENT).evaluate(zoom)
        || l.get(orb_style::ICON_IGNORE_PLACEMENT).evaluate(zoom)
        // Per feature or following the camera, any symbol may.
        || symbol_flag_values(symbol)
            .any(|e| e.reads_camera() || matches!(e.kind(), orb_style::Kind::Source | orb_style::Kind::Composite));
    by_y && can_overlap
}

/// The overlap, optional, ignore-placement and avoid-edges settings the
/// layer reads per feature, beyond the spec, as
/// [`orb_symbol::LayerParams::own_flags`]:
/// each symbol collides by its own of those, and by the layer's (which
/// may follow the camera) for the rest.
pub(crate) fn own_symbol_flags(symbol: &orb_style::Layer) -> u8 {
    use orb_style::{
        ICON_ALLOW_OVERLAP, ICON_IGNORE_PLACEMENT, ICON_OPTIONAL, ICON_OVERLAP, SYMBOL_AVOID_EDGES,
        TEXT_ALLOW_OVERLAP, TEXT_IGNORE_PLACEMENT, TEXT_OPTIONAL, TEXT_OVERLAP,
    };
    use orb_symbol::SymbolFlags as F;
    let per_feature = |kind| matches!(kind, orb_style::Kind::Source | orb_style::Kind::Composite);
    let flag = |key: orb_style::Key<bool>| per_feature(symbol.get(key).kind());
    let overlap =
        |key: orb_style::Key<Option<orb_style::Overlap>>| per_feature(symbol.get(key).kind());
    [
        (
            overlap(TEXT_OVERLAP) || flag(TEXT_ALLOW_OVERLAP),
            F::TEXT_OVERLAP,
        ),
        (
            overlap(ICON_OVERLAP) || flag(ICON_ALLOW_OVERLAP),
            F::ICON_OVERLAP,
        ),
        (flag(TEXT_OPTIONAL), F::TEXT_OPTIONAL),
        (flag(ICON_OPTIONAL), F::ICON_OPTIONAL),
        (flag(TEXT_IGNORE_PLACEMENT), F::TEXT_IGNORE_PLACEMENT),
        (flag(ICON_IGNORE_PLACEMENT), F::ICON_IGNORE_PLACEMENT),
        (flag(SYMBOL_AVOID_EDGES), F::AVOID_EDGES),
    ]
    .into_iter()
    .filter(|&(own, _)| own)
    .fold(0, |mask, (_, bit)| mask | bit)
}

/// The expressions of the layer's overlap, optional and ignore-placement
/// settings.
fn symbol_flag_values(
    symbol: &orb_style::Layer,
) -> impl Iterator<Item = &orb_style::PropertyExpression> {
    let flags = [
        orb_style::TEXT_ALLOW_OVERLAP,
        orb_style::ICON_ALLOW_OVERLAP,
        orb_style::TEXT_OPTIONAL,
        orb_style::ICON_OPTIONAL,
        orb_style::TEXT_IGNORE_PLACEMENT,
        orb_style::ICON_IGNORE_PLACEMENT,
    ];
    let overlaps = [orb_style::TEXT_OVERLAP, orb_style::ICON_OVERLAP];
    (flags.into_iter())
        .filter_map(|k| symbol.get(k).as_expression())
        .chain(
            overlaps
                .into_iter()
                .filter_map(|k| symbol.get(k).as_expression()),
        )
}

/// The layer sets `text-variable-anchor` or `text-variable-anchor-offset`.
pub(crate) fn has_variable_placement(symbol: &orb_style::Layer) -> bool {
    let l = symbol;
    l.get(orb_style::TEXT_VARIABLE_ANCHOR).constant() != Some(None)
        || l.get(orb_style::TEXT_VARIABLE_ANCHOR_OFFSET).constant() != Some(None)
}

/// A bucket's text size at the camera's zoom (GL JS `evaluateSizeForZoom`).
fn zoom_size(
    property: &orb_style::PropertyRef<'_, f64>,
    kind: SizeKind,
    tile_zoom: f64,
    zoom: f64,
) -> orb_symbol::ZoomSize {
    let layout = property.evaluate(tile_zoom + 1.0);
    match kind {
        SizeKind::Constant => orb_symbol::ZoomSize {
            size: layout,
            t: 0.0,
        },
        SizeKind::Camera => orb_symbol::ZoomSize {
            size: property.evaluate(zoom).min(layout),
            t: 0.0,
        },
        SizeKind::Composite { zooms } => orb_symbol::ZoomSize {
            size: 0.0,
            t: property
                .as_expression()
                .map_or(0.0, |e| e.interpolation_factor(zoom, zooms[0], zooms[1]))
                .clamp(0.0, 1.0),
        },
        SizeKind::Source => orb_symbol::ZoomSize::default(),
    }
}

/// How far `zoom` blends a per-feature `*-translate`'s values at a tile's
/// zoom and the next, as the symbol shader's `curve_t` does.
fn translate_t(
    symbol: &orb_style::Layer,
    part: orb_style::SymbolPart,
    tile_zoom: f64,
    zoom: f64,
) -> f64 {
    let translate = symbol.get(part.translate);
    match translate.as_expression() {
        Some(e) if e.kind() == orb_style::Kind::Composite => e
            .interpolation_factor(zoom, tile_zoom, tile_zoom + 1.0)
            .clamp(0.0, 1.0),
        _ => 0.0,
    }
}

/// Labels of one map: placement lane, buckets and glyph rows.
pub(crate) struct Labels {
    pub(crate) lane: Arc<dyn PlacementLane>,
    retired: Retired,
    buckets: HashMap<u64, BucketMeshes>,
    next_bucket: u64,
    glyphs: GlyphRows,
    /// The last commit applied.
    committed: Option<Committed>,
    /// The placement out on the lane: its frame, the world copies of each
    /// bucket it draws, which its commit is applied against, and its
    /// request's number.
    in_flight: Option<(Arc<orb_symbol::Frame>, WorldCopies, u64)>,
    /// The frame the last placement was asked for, or `None` when the
    /// next frame needs placing whatever it is.
    last_frame: Option<Arc<orb_symbol::Frame>>,
    /// The frame of the last commit applied, and how many have been.
    placed: Option<Arc<orb_symbol::Frame>>,
    commits: u64,
    /// Placements asked for so far, and the number of the last one
    /// committed: a request the lane failed never commits.
    requests: u64,
    committed_request: u64,
    /// Buckets of tiles that left the draw list, still drawn with their
    /// last words until a placement asked for since commits, then fading
    /// out from there (GL JS `holdingForFade`): what replaced them shows
    /// its labels by then.
    held: HashMap<u64, Held>,
    /// The layers, buckets and tiles the last frame drew.
    drawn: Vec<(u32, u64, orb_symbol::TileId)>,
    pub(crate) options: LabelOptions,
}

/// The map's label settings, which outlive its labels (a lost device
/// makes new ones).
#[derive(Clone, Debug)]
pub(crate) struct LabelOptions {
    /// Milliseconds labels fade over (GL JS `fadeDuration`).
    pub(crate) fade_duration: f64,
    /// Whether labels of different sources collide (GL JS
    /// `crossSourceCollisions`).
    pub(crate) cross_source_collisions: bool,
    /// The map's `localIdeographFontFamily`, for local raster jobs.
    pub(crate) ideographs: Option<String>,
    /// Labels keep their place across placements, as in MapLibre Native
    /// ([`orb_symbol::Frame::stable_labels`]).
    pub(crate) stable_labels: bool,
}

impl From<&crate::MapOptions> for LabelOptions {
    fn from(options: &crate::MapOptions) -> Self {
        Self {
            fade_duration: options.fade_duration,
            cross_source_collisions: options.cross_source_collisions,
            ideographs: options.local_ideograph_font_family.clone(),
            stable_labels: options.stable_labels,
        }
    }
}

impl std::fmt::Debug for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Labels")
            .field("buckets", &self.buckets.len())
            .finish_non_exhaustive()
    }
}

/// Rows of a `Pod` table from its words, copied to the rows' alignment.
fn rows<T: bytemuck::Pod>(words: &[u32]) -> Vec<T> {
    let bytes: &[u8] = bytemuck::cast_slice(words);
    let size = std::mem::size_of::<T>();
    let count = bytes.len() / size;
    let mut out = vec![T::zeroed(); count];
    bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(&bytes[..count * size]);
    out
}

/// The layer's size at `zoom + 1`, the size collision boxes are built for
/// and camera sizes are capped at (GL JS `layoutSize`).
fn layout_size(property: &orb_style::PropertyRef<'_, f64>, tile_zoom: f64) -> f32 {
    property.evaluate(tile_zoom + 1.0) as f32
}

/// Chunks of up to `MAX_CHUNK_INSTANCES` quads, in quad order. With `symbol-sort-key` a
/// chunk holds one key, which the scene orders candidates by across tiles
/// (GL JS draws sort-key segments across tiles in key order).
fn quad_chunks(placed: &[PlacedSymbol], sort_keys: &[SortKeyRange]) -> Vec<orb_layout::Chunk> {
    // The ranges run in instance order.
    let key_of = |instance: u32| {
        let i = sort_keys.partition_point(|r| r.end <= instance);
        sort_keys
            .get(i)
            .filter(|r| r.start <= instance)
            .map(|r| r.sort_key as f32)
    };
    let quads = placed
        .iter()
        .map(|p| p.quads[0] + p.quads[1])
        .max()
        .unwrap_or(0);
    let mut owners: Vec<Option<([i16; 2], Option<f32>)>> = vec![None; quads as usize];
    for p in placed {
        let key = key_of(p.instance);
        let [start, count] = p.quads;
        for q in start..start + count {
            owners[q as usize] = Some((p.anchor, key));
        }
    }
    let mut chunks: Vec<orb_layout::Chunk> = Vec::new();
    for (q, owner) in owners.iter().enumerate() {
        let Some((anchor, key)) = *owner else {
            continue;
        };
        let q = q as u32;
        let fits = chunks.last().is_some_and(|c| {
            c.index_count < orb_layout::MAX_CHUNK_INSTANCES
                && c.sort_key == key
                && c.first_index + c.index_count == q
        });
        if !fits {
            chunks.push(orb_layout::Chunk {
                first_index: q,
                first_vertex: 0,
                index_count: 0,
                bounds: orb_layout::TileBox {
                    min: [i16::MAX; 2],
                    max: [i16::MIN; 2],
                },
                sort_key: key,
                owner: orb_layout::Owner::Every,
            });
        }
        let c = chunks.last_mut().expect("pushed above");
        c.index_count += 1;
        c.bounds.min = [0, 1].map(|i| c.bounds.min[i].min(anchor[i]));
        c.bounds.max = [0, 1].map(|i| c.bounds.max[i].max(anchor[i]));
    }
    chunks
}

impl Labels {
    /// Labels placed on `lane`, with `options`' label settings.
    pub(crate) fn new(lane: Arc<dyn PlacementLane>, options: LabelOptions) -> Self {
        Self {
            lane,
            retired: Arc::default(),
            buckets: HashMap::new(),
            next_bucket: 0,
            glyphs: GlyphRows::default(),
            committed: None,
            in_flight: None,
            last_frame: None,
            placed: None,
            commits: 0,
            requests: 0,
            committed_request: 0,
            held: HashMap::new(),
            drawn: Vec::new(),
            options,
        }
    }

    /// Uploads a tile's symbol layers into `resident`, replacing what it held
    /// of those layers, and gives their buckets to the placement lane.
    /// `reloaded`: the tile was laid out again (a data or style change), so
    /// its labels show without fading in the first time they are placed
    /// (GL JS's and MapLibre Native's `justReloaded`); a tile's first load
    /// fades them in.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        plan: &StylePlan,
        result: &TileResult,
        resident: &mut ResidentTile,
        images: &mut ImageAtlas,
        namespace: u64,
        states: &mut FeatureStates,
        tile_zoom: f64,
        reloaded: bool,
    ) -> Result<(), OutOfSpace> {
        for s in &result.symbols {
            self.upload_layer(
                gpu, renderer, plan, result, s, resident, images, namespace, states, tile_zoom,
                reloaded,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_layer(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        plan: &StylePlan,
        result: &TileResult,
        s: &ResultSymbols,
        resident: &mut ResidentTile,
        images: &mut ImageAtlas,
        namespace: u64,
        states: &mut FeatureStates,
        tile_zoom: f64,
        reloaded: bool,
    ) -> Result<(), OutOfSpace> {
        let (icon_key, text_key) = (
            crate::style::node_key(s.layer, 0),
            crate::style::node_key(s.layer, 1),
        );
        // What the tile held of the layer goes.
        resident.symbols.retain(|r| r.layer != s.layer);
        resident.meshes.retain(|m| {
            if m.node == icon_key || m.node == text_key {
                renderer.remove_mesh(resident.gpu, m.mesh);
                for &index in &m.states {
                    states.release(index);
                }
                false
            } else {
                true
            }
        });
        let Some(icon_node) = plan.node(icon_key) else {
            return Ok(());
        };
        let symbol = &plan.style.layers[plan.nodes[icon_node].layer];
        let bytes = result.host_bytes;
        let table = |t: SymbolTable| s.table(t).core_words(gpu, bytes);
        let instances: Vec<SymbolInstance> = rows(&table(SymbolTable::Instances));
        let text_placed: Vec<PlacedSymbol> = rows(&table(SymbolTable::TextPlaced));
        let icon_placed: Vec<PlacedSymbol> = rows(&table(SymbolTable::IconPlaced));
        let text_glyph_offsets: Vec<f32> = rows(&table(SymbolTable::TextGlyphOffsets));
        let lines: Vec<[i16; 2]> = rows(&table(SymbolTable::Lines));
        let anchor_offsets: Vec<AnchorOffset> = rows(&table(SymbolTable::AnchorOffsets));
        let sort_key_ranges: Vec<SortKeyRange> = rows(&table(SymbolTable::SortKeyRanges));
        if instances.is_empty() {
            return Ok(());
        }
        let name_ranges = table(SymbolTable::NameRanges);
        let name_bytes = table(SymbolTable::NameBytes);
        let name_bytes: &[u8] = bytemuck::cast_slice(&name_bytes);
        let names: Vec<Box<str>> = name_ranges
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[start, len]| {
                let (start, len) = (start as usize, len as usize);
                name_bytes
                    .get(start..start + len)
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .unwrap_or_default()
                    .into()
            })
            .collect();

        let template = plan.style.glyphs.as_deref();
        self.next_bucket += 1;
        // The leases' owner from the start: if an upload below fails, it
        // drops, and the next `retire` hands the rows back.
        let owner = ResidentSymbols {
            layer: s.layer,
            bucket: self.next_bucket,
            max_height_offset: s.meta.max_height_offset,
            rows: s
                .glyphs
                .iter()
                .map(|key| match key {
                    Some(key) => self.glyphs.lease(renderer.glyphs(), key, template),
                    None => 0,
                })
                .collect(),
            retired: Arc::clone(&self.retired),
        };
        let image_rects: Vec<u32> = s
            .images
            .iter()
            .map(|name| {
                images
                    .resolve_icon(namespace, name)
                    .ok()
                    .and_then(|id| images.rect(id))
                    .map_or(u32::MAX, |[x, y, layer]| pack_image_rect(x, y, layer))
            })
            .collect();

        let meta = &s.meta;
        let text_layout = layout_size(&symbol.get(orb_style::TEXT_SIZE), tile_zoom);
        let icon_layout = layout_size(&symbol.get(orb_style::ICON_SIZE), tile_zoom);
        let zooms = |kind: SizeKind| match kind {
            SizeKind::Composite { zooms } => zooms.map(|z| z as f32),
            _ => [0.0; 2],
        };
        let [text_lo, text_hi] = zooms(meta.text_size);
        let [icon_lo, icon_hi] = zooms(meta.icon_size);
        let f = f32::to_bits;
        // Images need filtering when drawn at another pixel ratio, rotated,
        // or as SDFs (GL JS `iconsNeedLinear`).
        let mesh_flags = u32::from(meta.icons_need_linear || meta.sdf_icons == Some(true));
        let text_params = [
            f(text_layout),
            f(text_lo),
            f(text_hi),
            f(text_layout),
            f(text_lo),
            f(text_hi),
            0,
            0,
        ];
        let icon_params = [
            f(icon_layout),
            f(icon_lo),
            f(icon_hi),
            f(text_layout),
            f(text_lo),
            f(text_hi),
            mesh_flags,
            0,
        ];
        // An icon fitted to text moves with that text: per placed icon, the
        // text's packed size.
        let mut icon_text_sizes = vec![0u32; icon_placed.len()];
        for t in &text_placed {
            if let Some(size) = icon_text_sizes.get_mut(t.associated_icon as usize) {
                *size = u32::from(t.size[0]) | u32::from(t.size[1]) << 16;
            }
        }

        let mut meshes = BucketMeshes::default();
        let streams = [
            (
                text_key,
                &text_placed,
                SymbolTable::TextQuads,
                SymbolTable::TextGlyphOffsets,
                text_params,
                &[][..],
            ),
            (
                icon_key,
                &icon_placed,
                SymbolTable::IconQuads,
                SymbolTable::IconGlyphOffsets,
                icon_params,
                &icon_text_sizes[..],
            ),
        ];
        for (key, placed, quads, offsets, params, text_sizes) in streams {
            if placed.is_empty() || plan.node(key).is_none() {
                continue;
            }
            let node = plan.node(key).expect("checked") as u32;
            let mut chunks = quad_chunks(placed, &sort_key_ranges);
            if draws_by_viewport_y(symbol, tile_zoom) {
                // Placement reorders the quads: every chunk may hold any.
                let mut all = orb_layout::TileBox {
                    min: [i16::MAX; 2],
                    max: [i16::MIN; 2],
                };
                for c in &chunks {
                    all.min = [0, 1].map(|i| all.min[i].min(c.bounds.min[i]));
                    all.max = [0, 1].map(|i| all.max[i].max(c.bounds.max[i]));
                }
                for c in &mut chunks {
                    c.bounds = all;
                }
            }
            let quads_words = s.table(quads).words(bytes);
            if quads_words.len() / (std::mem::size_of::<Quad>() as u32 / 4) == 0 {
                continue;
            }
            let rows_mesh = result.meshes.iter().find(|m| m.node == key);
            let props = rows_mesh.map_or(Words::Core(&[]), |m| m.props.words(bytes));
            let data = SymbolMeshData {
                quads: quads_words,
                placed: if key == text_key {
                    s.table(SymbolTable::TextPlaced)
                } else {
                    s.table(SymbolTable::IconPlaced)
                }
                .words(bytes),
                glyph_offsets: s.table(offsets).words(bytes),
                lines: s.table(SymbolTable::Lines).words(bytes),
                glyphs: &owner.rows,
                images: &image_rects,
                instance_rows: s.table(SymbolTable::InstanceRows).words(bytes),
                text_sizes,
                colors: if key == text_key {
                    s.table(SymbolTable::TextColors).words(bytes)
                } else {
                    Words::Core(&[])
                },
                props,
                params,
                chunks: &chunks,
            };
            debug_assert_eq!(data.placed.len(), placed.len() as u32 * PLACED_WORDS);
            // Symbol rows name no images: only their features' states.
            let mut warnings = Vec::new();
            let rows = rows_mesh.map(|m| {
                crate::resident::resolve_rows(
                    gpu,
                    plan,
                    result,
                    m,
                    images,
                    namespace,
                    states,
                    &mut warnings,
                )
            });
            debug_assert!(
                rows.as_ref().is_none_or(|r| r.ramps.is_empty()) && warnings.is_empty(),
                "symbol rows name no images"
            );
            let indices = rows.map(|r| r.states).unwrap_or_default();
            let mesh = match renderer.add_symbol_mesh(gpu, resident.gpu, node, &data, &indices) {
                Ok(mesh) => mesh,
                Err(e) => {
                    for index in indices {
                        states.release(index);
                    }
                    return Err(e);
                }
            };
            resident.meshes.push(Resident {
                node: key,
                mesh,
                hits: rows_mesh.map(|m| m.hits.clone()).unwrap_or_default(),
                layout_rev: result.revision,
                rows_rev: result.revision,
                ramps: Vec::new(),
                states: indices,
                maxima: Vec::new(),
            });
            if key == text_key {
                meshes.text = Some(mesh);
            } else {
                meshes.icon = Some(mesh);
            }
        }

        // Beyond the spec, each symbol's own translates, which end its
        // feature's rows.
        let translates = [text_key, icon_key].map(|key| {
            let Some(node) = plan.node(key).map(|i| &plan.nodes[i]) else {
                return Vec::new();
            };
            let mesh = result.meshes.iter().find(|m| m.node == key);
            let Some(mesh) = mesh.filter(|_| plan.translate_rows(node)) else {
                return Vec::new();
            };
            let props = mesh.props.core_words(gpu, bytes);
            let row_words = plan.row_words(node);
            let instance_rows = s.table(SymbolTable::InstanceRows).core_words(gpu, bytes);
            let unpack = |at: usize| {
                let word = |i| props.get(at + i).copied().unwrap_or(0);
                [0, 1].map(|i| crate::query::unpack_translate(word(i)))
            };
            (instance_rows.as_chunks::<2>().0.iter())
                .map(|&[slot, _]| {
                    unpack(
                        (slot as usize + 1) * row_words - orb_shaders::layout::TRANSLATE_ROW_WORDS,
                    )
                })
                .collect()
        });
        let bucket = owner.bucket;
        meshes.text_size = meta.text_size;
        meshes.tile_zoom = tile_zoom;
        meshes.tables = Arc::new(PlacementTables {
            instances,
            text_placed,
            icon_placed,
            text_glyph_offsets,
            lines,
            anchor_offsets,
            sort_key_ranges,
            text_size: meta.text_size,
            allow_vertical_placement: meta.allow_vertical_placement,
            reloaded,
            translates,
            ancestor_scale: meta.ancestor_scale,
        });
        meshes.names = names;
        self.lane.ingest(bucket, Arc::clone(&meshes.tables));
        self.buckets.insert(bucket, meshes);
        resident.symbols.push(owner);
        // The new bucket needs a placement.
        self.last_frame = None;
        Ok(())
    }

    /// Releases what dropped buckets held, and asks for the glyphs leased
    /// since the last frame.
    pub(crate) fn maintain(
        &mut self,
        gpu: &mut dyn Gpu,
        atlas: &mut GlyphAtlas,
        jobs: &dyn Jobs,
        raster_done: &dyn Fn(Vec<RasterRow>) -> crate::RasterDone,
        faces: &orb_style::FontFaces,
    ) {
        self.retire(gpu, atlas);
        // Each job's glyphs, deduplicated, with every row by its index.
        fn indexed<T: Ord + Clone>(rows: Vec<(T, u32, u64)>) -> (Vec<T>, Vec<RasterRow>) {
            let mut items: Vec<T> = rows.iter().map(|(item, _, _)| item.clone()).collect();
            items.sort_unstable();
            items.dedup();
            let rows = rows
                .into_iter()
                .map(|(item, row, lease)| {
                    let index = items.binary_search(&item).expect("listed above") as u32;
                    (index, row, lease)
                })
                .collect();
            (items, rows)
        }
        for (url, rows) in std::mem::take(&mut self.glyphs.pending) {
            let (code_points, rows) = indexed(rows);
            self.glyphs.in_flight += 1;
            jobs.raster(GlyphRaster::Pbf { url, code_points }, raster_done(rows));
        }
        for (stack, rows) in std::mem::take(&mut self.glyphs.pending_local) {
            let (texts, rows) = indexed(rows);
            let raster = GlyphRaster::Local {
                stack,
                texts: texts.into_iter().map(String::from).collect(),
                ideographs: self.options.ideographs.clone(),
            };
            self.glyphs.in_flight += 1;
            jobs.raster(raster, raster_done(rows));
        }
        for (font, rows) in std::mem::take(&mut self.glyphs.pending_files) {
            // A file the style no longer declares leaves its rows clear.
            let Some(url) = faces
                .iter()
                .flat_map(|(_, files)| files)
                .map(|f| f.url.as_str())
                .find(|&url| crate::glyphs::file_id(url) == font)
            else {
                continue;
            };
            let (texts, rows) = indexed(rows);
            let raster = GlyphRaster::File {
                url: url.to_string(),
                texts: texts.into_iter().map(String::from).collect(),
            };
            self.glyphs.in_flight += 1;
            jobs.raster(raster, raster_done(rows));
        }
    }

    /// Releases what dropped buckets held. Their meshes are already gone,
    /// so this runs before anything writes to a bucket's meshes.
    fn retire(&mut self, gpu: &mut dyn Gpu, atlas: &mut GlyphAtlas) {
        let retired = std::mem::take(&mut *self.retired.lock().expect("retired poisoned"));
        for (bucket, rows) in retired {
            for row in rows {
                self.glyphs.release(gpu, atlas, row);
            }
            if self.buckets.remove(&bucket).is_some() {
                self.lane.remove(bucket);
            }
            self.held.remove(&bucket);
            self.last_frame = None;
        }
    }

    /// A raster job's result arrived.
    pub(crate) fn rastered(
        &mut self,
        gpu: &mut dyn Gpu,
        atlas: &mut GlyphAtlas,
        rows: &[RasterRow],
        result: Result<Vec<GlyphBitmap>, String>,
        warnings: &mut Vec<String>,
    ) {
        match result {
            Ok(bitmaps) => self.glyphs.apply(gpu, atlas, rows, &bitmaps),
            Err(e) => {
                self.glyphs.in_flight -= 1;
                warnings.push(e);
            }
        }
    }

    /// Hands a lane that started afresh every bucket's tables again (the
    /// scene holds them), and asks for a placement.
    pub(crate) fn resend_if_restarted(&mut self) {
        if !self.lane.restarted() {
            return;
        }
        for (&bucket, meshes) in &self.buckets {
            self.lane.ingest(bucket, Arc::clone(&meshes.tables));
        }
        self.last_frame = None;
    }

    /// Whether `frame` needs placing: it differs from the last one placed
    /// (clock aside) and nothing is out on the lane. Each frame that moves
    /// the camera places, rather than every 300 ms as GL JS and MapLibre
    /// Native do: fades then smooth what a frame's collisions decide, and
    /// a label one placement drops doesn't fade out in full before the
    /// next can show it again (the gestures bench's label returns).
    pub(crate) fn wants_placement(&self, frame: &orb_symbol::Frame) -> bool {
        if self.in_flight.is_some() {
            return false;
        }
        !self
            .last_frame
            .as_ref()
            .is_some_and(|last| same_inputs(last, frame))
    }

    /// Whether the last placement asked was of `frame`'s inputs, and has
    /// committed.
    pub(crate) fn is_placed(&self, frame: &orb_symbol::Frame) -> bool {
        self.in_flight.is_none()
            && self
                .last_frame
                .as_ref()
                .is_some_and(|last| same_inputs(last, frame))
    }

    /// Records a placement asked of the lane.
    pub(crate) fn placing(&mut self, frame: &orb_symbol::Frame) {
        let mut drawn: HashMap<u64, Vec<i32>> = HashMap::new();
        for t in frame.layers.iter().flat_map(|l| &l.tiles) {
            drawn.entry(t.bucket).or_default().push(t.view.id.wrap);
        }
        let frame = Arc::new(frame.clone());
        self.requests += 1;
        self.in_flight = Some((Arc::clone(&frame), drawn, self.requests));
        self.last_frame = Some(frame);
    }

    /// The lane couldn't place the frame asked of it: the next frame asks
    /// again.
    pub(crate) fn place_failed(&mut self) {
        self.in_flight = None;
        self.last_frame = None;
    }

    /// Applies a commit: changed state words go to their meshes.
    pub(crate) fn committed(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        mut committed: Committed,
    ) {
        let Some((frame, drawn, request)) = self.in_flight.take() else {
            return;
        };
        self.placed = Some(frame);
        self.commits += 1;
        self.committed_request = request;
        // Buckets dropped since the placement was asked for have no meshes.
        self.retire(gpu, renderer.glyphs());
        let mut changed: HashMap<u64, Vec<(i32, orb_symbol::BucketStates)>> = HashMap::new();
        for (bucket, wrap, states) in std::mem::take(&mut committed.changed) {
            changed.entry(bucket).or_default().push((wrap, states));
        }
        for (bucket, states) in changed {
            let Some(meshes) = self.buckets.get_mut(&bucket) else {
                continue;
            };
            let wraps = drawn.get(&bucket).map_or(&[][..], Vec::as_slice);
            meshes.written.retain(|w, _| wraps.contains(w));
            let written: Vec<i32> = states.iter().map(|&(wrap, _)| wrap).collect();
            for (wrap, s) in states {
                meshes.written.insert(wrap, s);
            }
            // Copies whose wraps differ modulo the block count.
            let span = wraps.iter().max().zip(wraps.iter().min());
            let copies = span
                .map_or(1, |(max, min)| (max - min + 1) as u32)
                .next_power_of_two();
            let mut rewrite = false;
            for mesh in [meshes.text, meshes.icon].into_iter().flatten() {
                if renderer.symbol_copies(mesh) < copies {
                    // Out of memory: copies keep sharing their words.
                    rewrite |= renderer.set_symbol_copies(gpu, mesh, copies).is_ok();
                }
            }
            if rewrite {
                for &wrap in meshes.written.keys() {
                    meshes.write(gpu, renderer, wrap);
                }
            } else {
                for wrap in written {
                    meshes.write(gpu, renderer, wrap);
                }
            }
        }
        // A held bucket's labels a drawn tile took over hide: that tile's
        // copy carries on their fade (GL JS hides held tiles' duplicates).
        for (bucket, wrap, instances) in std::mem::take(&mut committed.handed_over) {
            let Some(meshes) = self.buckets.get_mut(&bucket) else {
                continue;
            };
            if !self.held.contains_key(&bucket) {
                continue;
            }
            let Some(states) = meshes.written.get_mut(&wrap) else {
                continue;
            };
            for &i in &instances {
                let Some(instance) = meshes.tables.instances.get(i as usize) else {
                    continue;
                };
                hide(&mut states.text, &instance.text);
                hide(&mut states.icon, &instance.icon);
            }
            meshes.write(gpu, renderer, wrap);
        }
        // Held buckets a placement since covers fade out from where they
        // are, measured from this commit as every label's fade is.
        let change = self.committed.as_ref().map_or(1.0, |c| {
            c.fade_change(committed.commit_time).clamp(0.0, 1.0) as f32
        });
        let latest = self.committed_request;
        for (&bucket, held) in &mut self.held {
            match *held {
                Held::Waiting { requests } if latest > requests => *held = Held::Fading,
                Held::Waiting { .. } => continue,
                Held::Fading => {}
            }
            let Some(meshes) = self.buckets.get_mut(&bucket) else {
                continue;
            };
            for states in meshes.written.values_mut() {
                fade_out(&mut states.text, change);
                fade_out(&mut states.icon, change);
            }
            for wrap in meshes.written.keys().copied().collect::<Vec<_>>() {
                meshes.write(gpu, renderer, wrap);
            }
        }
        self.committed = Some(committed);
    }

    /// Keeps a bucket drawn, as it was last placed, after its tile left
    /// the draw list: until a placement asked for from now commits, then
    /// while its labels fade out.
    pub(crate) fn hold(&mut self, bucket: u64) {
        self.held.entry(bucket).or_insert(Held::Waiting {
            requests: self.requests,
        });
    }

    /// The frame draws no labels, so no placement will cover the held
    /// buckets, nor fade them: they go now.
    pub(crate) fn nothing_to_place(&mut self) {
        self.held.clear();
    }

    /// Its tile draws again, or left: placement has it as any bucket.
    pub(crate) fn unhold(&mut self, bucket: u64) {
        self.held.remove(&bucket);
    }

    /// Whether a held bucket is done: faded out, or gone.
    pub(crate) fn released(&self, bucket: u64, now: f64) -> bool {
        match self.held.get(&bucket) {
            Some(Held::Waiting { .. }) => !self.buckets.contains_key(&bucket),
            Some(Held::Fading) => self.fade_change(now) >= 1.0,
            None => true,
        }
    }

    /// How far fades have run at `now`.
    pub(crate) fn fade_change(&self, now: f64) -> f32 {
        self.committed
            .as_ref()
            .map_or(1.0, |c| c.fade_change(now) as f32)
    }

    /// Records the buckets `frame` draws, for [`shown`](Self::shown).
    pub(crate) fn drawing(&mut self, frame: &orb_symbol::Frame) {
        self.drawn.clear();
        for layer in &frame.layers {
            for tile in &layer.tiles {
                self.drawn
                    .push((layer.params.layer, tile.bucket, tile.view.id));
            }
        }
    }

    /// Per text, the greatest opacity a label of it drew with in the last
    /// frame at `now`: what the symbol shader computes, for benches.
    pub(crate) fn shown(&self, now: f64) -> HashMap<Box<str>, f32> {
        use orb_symbol::state;
        let change = self.fade_change(now);
        let mut shown: HashMap<Box<str>, f32> = HashMap::new();
        let held = self.held.keys().flat_map(|&bucket| {
            let wraps = self.buckets.get(&bucket).map(|m| m.written.keys().copied());
            wraps.into_iter().flatten().map(move |wrap| (bucket, wrap))
        });
        let drawn = self
            .drawn
            .iter()
            .map(|&(_, bucket, tile)| (bucket, tile.wrap));
        for (bucket, wrap) in drawn.chain(held) {
            let Some(meshes) = self.buckets.get(&bucket) else {
                continue;
            };
            let states = meshes.written.get(&wrap);
            for (instance, name) in meshes.tables.instances.iter().zip(&meshes.names) {
                if name.is_empty() {
                    continue;
                }
                let opacity = instance
                    .text
                    .iter()
                    .filter_map(|&p| states?.text.get(p as usize * state::WORDS).copied())
                    .filter(|w| w & state::HIDDEN == 0)
                    .map(|w| {
                        let from = (w & state::OPACITY) as f32 / 127.0;
                        let sign = if w & state::PLACED != 0 { 1.0 } else { -1.0 };
                        (from + sign * change).clamp(0.0, 1.0)
                    })
                    .fold(0.0, f32::max);
                let best = shown.entry(name.clone()).or_default();
                *best = best.max(opacity);
            }
        }
        shown
    }

    /// A placement is out or glyphs are still on their way.
    pub(crate) fn settling(&self) -> bool {
        self.in_flight.is_some()
            || self.glyphs.in_flight > 0
            || !self.glyphs.pending.is_empty()
            || !self.glyphs.pending_local.is_empty()
            || !self.glyphs.pending_files.is_empty()
    }

    /// The last commit's fades are still running.
    pub(crate) fn fading(&self, now: f64) -> bool {
        let held = self.held.values().any(|h| *h == Held::Fading);
        self.committed
            .as_ref()
            .is_some_and(|c| c.fading(now) || (held && c.fade_change(now) < 1.0))
    }

    /// This frame's placement inputs: every symbol layer in style order
    /// with its drawn buckets.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn frame(
        &self,
        plan: &StylePlan,
        tiles: &[SymbolTile],
        view: &orb_camera::View,
        camera_world: glam::DVec2,
        period: f64,
        camera: &orb_camera::CameraState,
        zoom: f64,
        extrusion_scale: f64,
        now: f64,
    ) -> orb_symbol::Frame {
        let view_proj = view.view_proj();
        let viewport = view.viewport();
        let camera_inputs = crate::paint::camera_inputs(camera);
        let layers = plan
            .nodes
            .iter()
            .filter(|n| n.family == orb_render::Family::SymbolIcon)
            .map(|node| {
                let symbol = &plan.style.layers[node.layer];
                let mut drawn: Vec<&SymbolTile> = tiles
                    .iter()
                    .filter(|t| t.layer == crate::style::layer_key(node.key))
                    .collect();
                // Frames compare equal for the same tiles in any order.
                drawn.sort_by_key(|t| (t.tile, t.wrap, t.bucket));
                let tiles = drawn
                    .into_iter()
                    .filter_map(|t| {
                        let b = self.buckets.get(&t.bucket)?;
                        let a = t.to_world;
                        let origin = glam::DVec2::new(
                            a.translation.x + f64::from(t.wrap) * period,
                            a.translation.y,
                        ) - camera_world;
                        let (x, y) = (a.matrix2.x_axis, a.matrix2.y_axis);
                        let tile = glam::DMat4::from_cols(
                            glam::DVec4::new(x.x, x.y, 0.0, 0.0),
                            glam::DVec4::new(y.x, y.y, 0.0, 0.0),
                            glam::DVec4::new(0.0, 0.0, extrusion_scale, 0.0),
                            glam::DVec4::new(origin.x, origin.y, 0.0, 1.0),
                        );
                        Some(orb_symbol::PlacedTile {
                            bucket: t.bucket,
                            view: orb_symbol::TileView {
                                id: orb_symbol::TileId {
                                    z: t.tile.z,
                                    x: t.tile.col,
                                    y: t.tile.row,
                                    wrap: t.wrap,
                                },
                                matrix: (view_proj * tile).to_cols_array(),
                                globe: view.globe().zip(t.globe.clone()).map(|(globe, series)| {
                                    orb_symbol::TileGlobe {
                                        series,
                                        target: globe.target,
                                        view_proj: globe.view_proj,
                                        eye: globe.target + globe.eye,
                                        plane: globe.plane.is_some(),
                                        mercator: t.mercator.then_some(t.to_world),
                                    }
                                }),
                            },
                            text_size: zoom_size(
                                &symbol.get(orb_style::TEXT_SIZE),
                                b.text_size,
                                b.tile_zoom,
                                zoom,
                            ),
                            translate_t: [orb_style::TEXT_PART, orb_style::ICON_PART]
                                .map(|part| translate_t(symbol, part, b.tile_zoom, zoom)),
                        })
                    })
                    .collect();
                orb_symbol::LayerFrame {
                    params: layer_params(plan, node, zoom, camera_inputs),
                    tiles,
                }
            })
            .collect();
        orb_symbol::Frame {
            camera: orb_symbol::Camera {
                width: viewport.width,
                height: viewport.height,
                zoom,
                bearing: camera.bearing.to_radians(),
                pitch: camera.pitch.to_radians(),
                roll: camera.roll.to_radians(),
                camera_to_center_distance: view.camera_to_center(),
            },
            layers,
            now,
            fade_duration: self.options.fade_duration,
            cross_source_collisions: self.options.cross_source_collisions,
            lng: camera.target.lon,
            ground: None,
            scope: orb_symbol::Scope::Viewport,
            stable_labels: self.options.stable_labels,
        }
    }

    /// Placement commits applied so far.
    pub(crate) fn commits(&self) -> u64 {
        self.commits
    }

    /// Picking hits for the labels and icons the last commit shows: their
    /// anchors in world space and their collision boxes in pixels at the
    /// anchor's scale, for any camera to project. Boxes are
    /// viewport-aligned, without variable-anchor shifts; a line label is
    /// its anchor's collision circle's square.
    pub(crate) fn label_hits(
        &self,
        plan: &StylePlan,
        planes: &BucketPlanes,
    ) -> Vec<crate::query::LabelHit> {
        use orb_symbol::state;
        let Some(frame) = self.placed.as_deref() else {
            return Vec::new();
        };
        let shown = |words: &[u32], placed: u32| {
            placed != orb_symbol::NONE
                && words
                    .get(placed as usize * state::WORDS)
                    .is_some_and(|&w| w & state::HIDDEN == 0 && w & state::PLACED != 0)
        };
        let period = planes.period;
        let mut hits = Vec::new();
        for layer in &frame.layers {
            let key = crate::style::node_key(layer.params.layer, 0);
            let Some(node) = plan.node(key) else {
                continue;
            };
            let node = &plan.nodes[node];
            for tile in &layer.tiles {
                let Some(meshes) = self.buckets.get(&tile.bucket) else {
                    continue;
                };
                let id = tile.view.id;
                let Some(states) = meshes.written.get(&id.wrap) else {
                    continue;
                };
                let Some(&to_world) = planes.to_world.get(&tile.bucket) else {
                    continue;
                };
                let shift = glam::DVec2::new(f64::from(id.wrap) * period, 0.0);
                // Where each feature's first instance draws, when the
                // layer draws by viewport y; else by feature index.
                let sorted: HashMap<u32, u32> = if layer.params.draw_by_viewport_y {
                    let order =
                        orb_symbol::feature_sort_order(&meshes.tables, frame.camera.bearing);
                    let mut at = HashMap::new();
                    for (i, feature) in order.into_iter().enumerate() {
                        at.entry(feature).or_insert(i as u32);
                    }
                    at
                } else {
                    HashMap::new()
                };
                for (index, instance) in meshes.tables.instances.iter().enumerate() {
                    let text = instance.text.iter().find(|&&p| shown(&states.text, p));
                    let icon = instance.icon.iter().find(|&&p| shown(&states.icon, p));
                    if text.is_none() && icon.is_none() {
                        continue;
                    }
                    // A variable anchor's shift, in the boxes' units (GL JS
                    // `calculateVariableLayoutShift`), for the text and an
                    // icon shifted with it.
                    let anchor_shift = |words: &[u32], placed: u32| {
                        let at = placed as usize * state::WORDS;
                        (words[at] & state::SHIFTED != 0).then(|| {
                            let scale = f64::from(instance.text_box_scale);
                            [1, 2].map(|i| f64::from(f32::from_bits(words[at + i])) * scale)
                        })
                    };
                    // Each part where its translate draws it.
                    let [text_translate, icon_translate] =
                        orb_symbol::label_translates(&layer.params, tile, &meshes.tables, index);
                    let boxes: Vec<crate::query::LabelBox> = [
                        (
                            text.map(|&p| anchor_shift(&states.text, p)),
                            0,
                            text_translate,
                        ),
                        (
                            icon.map(|&p| anchor_shift(&states.icon, p)),
                            2,
                            icon_translate,
                        ),
                    ]
                    .into_iter()
                    .filter_map(|(shown, i, (translate, viewport))| {
                        let [dx, dy] = shown?.unwrap_or([0.0; 2]);
                        let b = instance.boxes[i].map(f64::from);
                        (!(b[0] == b[2] && b[1] == b[3])).then_some(crate::query::LabelBox {
                            rect: [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy],
                            translate: translate.map(|v| v as f32),
                            viewport,
                        })
                    })
                    .collect();
                    let circles = if text.is_some() {
                        orb_symbol::label_circles(frame, &layer.params, tile, &meshes.tables, index)
                    } else {
                        Vec::new()
                    };
                    if boxes.is_empty() && circles.is_empty() {
                        continue;
                    }
                    let [ax, ay] = instance.anchor.map(f64::from);
                    hits.push(crate::query::LabelHit {
                        node: key,
                        key: crate::query::FeatureKey {
                            source: plan.sources[node.source].key,
                            tile: orb_crs::TileIndex::new(id.z, id.x, id.y),
                            feature: instance.feature,
                        },
                        wrap: id.wrap,
                        rank: sorted
                            .get(&instance.feature)
                            .copied()
                            .unwrap_or(instance.feature),
                        anchor: to_world.transform_point2(glam::DVec2::new(ax, ay)) + shift,
                        boxes,
                        circles,
                    });
                }
            }
        }
        hits
    }

    /// Forgets the placement inputs, so the next frame places again.
    pub(crate) fn invalidate(&mut self) {
        self.last_frame = None;
    }
}

/// Rewrites state words to fade out from where they are `change` into
/// their fade: each shown symbol's opacity now, no longer placed.
/// Hides the placed symbols `placed` of a bucket's state words.
fn hide(words: &mut [u32], placed: &[u32]) {
    for &p in placed.iter().filter(|&&p| p != orb_symbol::NONE) {
        if let Some(w) = words.get_mut(p as usize * orb_symbol::state::WORDS) {
            *w |= orb_symbol::state::HIDDEN;
        }
    }
}

fn fade_out(words: &mut [u32], change: f32) {
    use orb_symbol::state;
    for w in words.chunks_mut(state::WORDS) {
        let flags = w[0];
        if flags & state::HIDDEN != 0 {
            continue;
        }
        let from = (flags & state::OPACITY) as f32 / 127.0;
        let sign = if flags & state::PLACED != 0 {
            1.0
        } else {
            -1.0
        };
        let now = (from + sign * change).clamp(0.0, 1.0);
        w[0] = flags & !(state::OPACITY | state::PLACED) | (now * 127.0).round() as u32;
    }
}

/// How a size property is drawn per frame at `zoom`.
fn size_paint(property: &orb_style::PropertyRef<'_, f64>, zoom: f64) -> orb_render::SymbolSize {
    let curve = || {
        let interpolation = property
            .as_expression()
            .and_then(|e| e.zoom_curve())
            .and_then(|c| c.interpolation);
        crate::paint::interpolation_curve(interpolation)
    };
    match property.kind() {
        orb_style::Kind::Constant => {
            orb_render::SymbolSize::Constant(property.evaluate(0.0) as f32)
        }
        orb_style::Kind::Source => orb_render::SymbolSize::Source,
        orb_style::Kind::Composite => orb_render::SymbolSize::Composite { curve: curve() },
        orb_style::Kind::Camera => {
            // The stops around `zoom`, which the shader blends at the
            // frame's: paint is written again when the zoom changes, so a
            // curve of any length draws exactly.
            let zooms = property
                .as_expression()
                .and_then(|e| e.zoom_curve())
                .map_or(&[][..], |c| &c.stops);
            let lower = zooms.iter().rposition(|&z| z <= zoom).unwrap_or(0);
            let upper = (lower + 1).min(zooms.len().saturating_sub(1));
            let stop = |i: usize| {
                let z = zooms.get(i).copied().unwrap_or(zoom);
                [z as f32, property.evaluate(z) as f32]
            };
            orb_render::SymbolSize::Camera {
                curve: curve(),
                stops: [stop(lower), stop(upper)],
            }
        }
    }
}

/// A symbol node's paint at `zoom` (GL JS `drawSymbols` and
/// `drawLayerSymbols`), or `None` when it draws nothing.
pub(crate) fn paint(
    plan: &StylePlan,
    node: &crate::style::NodePlan,
    zoom: f64,
    maxima: &[f32],
    state: orb_render::StateLookup,
) -> Option<orb_render::SymbolPaint> {
    let symbol = &plan.style.layers[node.layer];
    let is_text = node.family == orb_render::Family::SymbolText;
    let part = if is_text {
        orb_style::TEXT_PART
    } else {
        orb_style::ICON_PART
    };
    // GL JS skips a part whose constant opacity is 0.
    if symbol.get(part.opacity).constant() == Some(0.0) {
        return None;
    }
    let l = symbol;
    let point =
        l.get(orb_style::SYMBOL_PLACEMENT).evaluate(zoom) == orb_style::SymbolPlacement::Point;
    let (rotate, pitch, keep_upright) = if is_text {
        (
            text_rotates_with_map(symbol, zoom),
            text_pitches_with_map(symbol, zoom),
            l.get(orb_style::TEXT_KEEP_UPRIGHT).evaluate(zoom),
        )
    } else {
        let rotate = match l.get(orb_style::ICON_ROTATION_ALIGNMENT).constant() {
            Some(orb_style::Alignment::Map) => true,
            Some(orb_style::Alignment::Viewport) => false,
            _ => !point,
        };
        let pitch = match l.get(orb_style::ICON_PITCH_ALIGNMENT).evaluate(zoom) {
            orb_style::Alignment::Map => true,
            orb_style::Alignment::Viewport => false,
            orb_style::Alignment::Auto => rotate,
        };
        (
            rotate,
            pitch,
            l.get(orb_style::ICON_KEEP_UPRIGHT).evaluate(zoom),
        )
    };
    let viewport_aligned = if is_text {
        l.get(orb_style::TEXT_ROTATION_ALIGNMENT).evaluate(zoom)
            == orb_style::TextRotationAlignment::Viewport
            || (!rotate && point)
    } else {
        !rotate
    };
    let along_line = !viewport_aligned && !point;

    let mut paint = orb_render::SymbolPaint {
        color: [0.0; 4],
        halo_color: [0.0; 4],
        opacity: 0.0,
        halo_width: 0.0,
        halo_blur: 0.0,
        table_mask: 0,
        curves: [orb_render::Curve::STEP; 5],
        night_color: [0.0; 4],
        night_halo_color: [0.0; 4],
        daylight: node.daylight.as_ref().map_or([None; 2], |d| d.curves()),
        row_words: plan.row_words(node) as u32,
        // Per feature (beyond the spec), the farthest any feature's goes
        // follows the slots' maxima.
        translate: crate::paint::translate(
            plan,
            node,
            zoom,
            maxima.get(plan.paint_properties(node).len()),
        ),
        size: size_paint(
            &l.get(if is_text {
                orb_style::TEXT_SIZE
            } else {
                orb_style::ICON_SIZE
            }),
            zoom,
        ),
        text_size: size_paint(&l.get(orb_style::TEXT_SIZE), zoom),
        is_text,
        pitch_with_map: pitch,
        rotate_with_map: rotate,
        along_line,
        rotate_in_shader: rotate && !pitch && !along_line,
        is_offset: l.get(orb_style::ICON_OFFSET).is_set(),
        keep_upright,
        height_anchor_ground: l.get(orb_style::SYMBOL_HEIGHT_ANCHOR).evaluate(zoom)
            == orb_style::HeightAnchor::Ground,
        // GL JS draws halos, then fills, per sort-key segment of a tile;
        // a layer sorted by key draws them per quad instead.
        separate_halo: !draws_by_key(symbol, zoom)
            && (l
                .get(orb_style::TEXT_LETTER_SPACING)
                .constant()
                .is_some_and(|s| s * orb_text::ONE_EM < 0.0)
                || l.get(part.halo_width).constant().is_none_or(|w| w > 1.0)),
        reach_px: 0.0,
        state,
        // GL JS samples images linearly for scaled icons, zoom-dependent
        // inline images and pitched views.
        linear_icons: pitch
            || (is_text
                && matches!(
                    l.get(orb_style::TEXT_SIZE).kind(),
                    orb_style::Kind::Camera | orb_style::Kind::Composite
                ))
            || (!is_text && l.get(orb_style::ICON_SIZE).constant() != Some(1.0)),
    };
    for (slot, property) in plan.paint_properties(node).iter().enumerate() {
        // A colour's night stop shares its day stop's zoom curve, and so
        // its slot's: the one of them per feature or in the state table
        // reads it, and when both do, theirs is the colour's.
        let curve = if slot > 4 { slot - 4 } else { slot };
        if property.per_feature() {
            paint.table_mask |= 1 << slot;
            paint.curves[curve] = crate::paint::curve(property, zoom);
        }
        // A state lookup's constant is its value with no state.
        let lookup = property.is_state_lookup();
        if lookup {
            paint.curves[curve] = crate::paint::curve(property, zoom);
        }
        let number = |p: &crate::style::PaintProperty<'_>| match p {
            _ if lookup => p.default_number(zoom),
            crate::style::PaintProperty::Number(n) => n.evaluate(zoom),
            _ => 0.0,
        };
        let color = |p: &crate::style::PaintProperty<'_>| {
            if lookup {
                p.default_color(zoom)
            } else {
                p.color(zoom)
            }
        };
        match slot {
            0 => paint.opacity = number(property),
            1 => paint.color = color(property),
            2 => paint.halo_color = color(property),
            3 => {
                let max = maxima.get(3).copied().unwrap_or(0.0);
                paint.halo_width = number(property).max(property.row_maximum(zoom, max));
            }
            4 => paint.halo_blur = number(property),
            5 => paint.night_color = color(property),
            _ => paint.night_halo_color = color(property),
        }
    }
    // Culling reach: a long label, with its offsets and halo, at the
    // largest size the layer draws now.
    let size = match paint.size {
        orb_render::SymbolSize::Constant(s) => s,
        _ => {
            (if is_text {
                l.get(orb_style::TEXT_SIZE).evaluate(zoom)
            } else {
                l.get(orb_style::ICON_SIZE).evaluate(zoom)
            }) as f32
        }
    };
    paint.reach_px = size.max(16.0) * if is_text { 24.0 } else { 4.0 } + paint.halo_width;
    Some(paint)
}

#[cfg(test)]
mod tests {
    use orb_hal::TextureFormat;
    use orb_hal_wgpu::WgpuGpu;
    use orb_render::{SceneCapacity, TargetFormat};
    use orb_symbol::{BucketStates, Camera, Frame, LayerFrame, PlacedTile, TileId, TileView};

    use super::*;
    use crate::worker::TileWorker;

    /// A camera size curve of any length draws exactly: the stops around
    /// the zoom, beyond the eight the node uniform once held.
    #[test]
    fn camera_sizes_take_the_stops_around_the_zoom() {
        let stops: Vec<String> = (0..12).map(|z| format!("{z}, {}", z * 2)).collect();
        let style = orb_style::v8::parse(&format!(
            r#"{{"version": 8, "sources": {{"s": {{"type": "geojson", "data": {{"type": "FeatureCollection", "features": []}}}}}},
                "layers": [{{"id": "t", "type": "symbol", "source": "s",
                "layout": {{"text-size": ["interpolate", ["linear"], ["zoom"], {}]}}}}]}}"#,
            stops.join(", ")
        ))
        .unwrap();
        let size = style.layers[0].get(orb_style::TEXT_SIZE);
        let at = |zoom| match size_paint(&size, zoom) {
            orb_render::SymbolSize::Camera { stops, .. } => stops,
            other => panic!("{other:?}"),
        };
        assert_eq!(at(9.5), [[9.0, 18.0], [10.0, 20.0]]);
        assert_eq!(at(11.0), [[11.0, 22.0], [11.0, 22.0]]);
        assert_eq!(at(-1.0), [[0.0, 0.0], [1.0, 2.0]]);
    }

    struct Harness {
        gpu: WgpuGpu,
        renderer: Renderer,
        images: ImageAtlas,
        namespace: u64,
        states: FeatureStates,
        worker: TileWorker,
        result: TileResult,
        labels: Labels,
    }

    impl Harness {
        fn new(capacity: SceneCapacity) -> Self {
            let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
            let mut images = ImageAtlas::new(&mut gpu, 256, 1, 16);
            let namespace = images.namespace();
            let target = TargetFormat {
                color: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            };
            let mut renderer = Renderer::new(&mut gpu, target, capacity, images.bindings());
            let (worker, result) = crate::testing::symbol_tile();
            let families: Vec<_> = worker.plan().nodes.iter().map(|n| n.family).collect();
            renderer.set_nodes(&mut gpu, &families);
            Self {
                gpu,
                renderer,
                images,
                namespace,
                states: FeatureStates::default(),
                worker,
                result,
                labels: Labels::new(
                    Arc::new(crate::InlinePlacement::default()),
                    (&crate::MapOptions::default()).into(),
                ),
            }
        }

        /// An empty tile to upload buckets into.
        fn tile(&mut self) -> ResidentTile {
            let key = orb_render::TileKey {
                source: 0,
                tile: orb_crs::TileIndex::new(14, 14552, 6450),
            };
            let gpu = self.renderer.add_tile(&mut self.gpu, key, None).unwrap();
            ResidentTile::new(gpu, 14.0)
        }

        fn upload(&mut self, resident: &mut ResidentTile) -> Result<(), OutOfSpace> {
            self.labels.upload(
                &mut self.gpu,
                &mut self.renderer,
                self.worker.plan(),
                &self.result,
                resident,
                &mut self.images,
                self.namespace,
                &mut self.states,
                14.0,
                false,
            )
        }

        /// A placement of `bucket`'s tile at `wraps`, out on the lane.
        fn placing(&mut self, bucket: u64, wraps: &[i32]) {
            let frame = self.frame(bucket, wraps);
            self.labels.placing(&frame);
        }

        /// A frame placing `bucket`'s tile at `wraps`.
        fn frame(&self, bucket: u64, wraps: &[i32]) -> Frame {
            let tile = |wrap| PlacedTile {
                bucket,
                view: TileView {
                    id: TileId {
                        z: crate::testing::SYMBOL_TILE.0,
                        x: crate::testing::SYMBOL_TILE.1,
                        y: crate::testing::SYMBOL_TILE.2,
                        wrap,
                    },
                    matrix: [0.0; 16],
                    globe: None,
                },
                text_size: orb_symbol::ZoomSize::default(),
                translate_t: [0.0; 2],
            };
            let camera = orb_style::CameraInputs {
                pitch: 0.0,
                bearing: 0.0,
                meters_per_pixel: f64::NAN,
            };
            let params = layer_params(
                self.worker.plan(),
                &self.worker.plan().nodes[0],
                14.0,
                camera,
            );
            Frame {
                camera: Camera {
                    width: 512.0,
                    height: 512.0,
                    zoom: 14.0,
                    bearing: 0.0,
                    pitch: 0.0,
                    roll: 0.0,
                    camera_to_center_distance: 1.0,
                },
                layers: vec![LayerFrame {
                    params,
                    tiles: wraps.iter().map(|&w| tile(w)).collect(),
                }],
                now: 0.0,
                fade_duration: 0.0,
                cross_source_collisions: true,
                lng: 0.0,
                ground: None,
                scope: orb_symbol::Scope::Viewport,
                stable_labels: false,
            }
        }

        /// A commit showing every text symbol of `bucket` at `wraps`.
        fn commit(&self, bucket: u64, wraps: &[i32]) -> Committed {
            let placed = self.result.symbols[0].table(SymbolTable::TextPlaced);
            let crate::jobs::Stream::Core(words) = placed else {
                panic!("core streams")
            };
            let count = words.len() / PLACED_WORDS as usize;
            let states = BucketStates {
                text: [orb_symbol::state::PLACED | 127, 0, 0].repeat(count),
                ..BucketStates::default()
            };
            Committed {
                changed: wraps.iter().map(|&w| (bucket, w, states.clone())).collect(),
                fade_duration: 0.0,
                ..Committed::default()
            }
        }
    }

    /// A placement thread whose placer traps starts afresh: that placement
    /// fails, the labels hand the lane their buckets again, and the next
    /// one places them.
    #[test]
    fn a_trapped_placement_lane_places_again() {
        let mut h = Harness::new(SceneCapacity::default());
        let lane = crate::placement::ThreadPlacement::new();
        h.labels.lane = lane.clone();
        let mut resident = h.tile();
        h.upload(&mut resident).unwrap();
        let bucket = resident.symbols[0].bucket;
        let place = |h: &Harness| {
            let (send, receive) = std::sync::mpsc::channel();
            let done = Box::new(move |c| send.send(c).unwrap());
            h.labels.lane.place(h.frame(bucket, &[0]), done);
            receive.recv().unwrap()
        };
        let placed = |c: &Committed| c.changed.iter().any(|(b, _, _)| *b == bucket);
        assert!(placed(&place(&h).unwrap()));
        lane.trap_next();
        assert!(place(&h).is_err());
        h.labels.resend_if_restarted();
        assert!(placed(&place(&h).unwrap()), "the bucket, sent again");
        assert!(!h.labels.lane.restarted());
    }

    /// A relayout drops the bucket a placement was asked for; its
    /// commit must not write to the freed meshes.
    #[test]
    fn commits_skip_buckets_replaced_while_placing() {
        let mut h = Harness::new(SceneCapacity::default());
        let mut resident = h.tile();
        h.upload(&mut resident).unwrap();
        let old = resident.symbols[0].bucket;
        h.placing(old, &[0]);
        // The relayout's result lands before the commit.
        h.upload(&mut resident).unwrap();
        let commit = h.commit(old, &[0]);
        h.labels.committed(&mut h.gpu, &mut h.renderer, commit);
        assert!(!h.labels.buckets.contains_key(&old));
    }

    /// A commit applies against the frame it was asked for, even if
    /// the needs-placement marker was cleared meanwhile.
    #[test]
    fn commits_keep_world_copies_apart_after_invalidation() {
        let mut h = Harness::new(SceneCapacity::default());
        let mut resident = h.tile();
        h.upload(&mut resident).unwrap();
        let bucket = resident.symbols[0].bucket;
        h.placing(bucket, &[0, 1]);
        h.labels.invalidate();
        let commit = h.commit(bucket, &[0, 1]);
        h.labels.committed(&mut h.gpu, &mut h.renderer, commit);
        let mesh = h.labels.buckets[&bucket].text.expect("a text mesh");
        assert_eq!(h.renderer.symbol_copies(mesh), 2);
    }

    /// A bucket whose tile left the draw list keeps its labels as
    /// placed until a placement asked for since commits, then fades them
    /// out from there and is released once they have.
    #[test]
    fn held_labels_a_drawn_tile_took_over_hide_at_once() {
        use orb_symbol::state;
        let mut h = Harness::new(SceneCapacity::default());
        let mut resident = h.tile();
        h.upload(&mut resident).unwrap();
        let bucket = resident.symbols[0].bucket;
        h.placing(bucket, &[0]);
        let commit = h.commit(bucket, &[0]);
        h.labels.committed(&mut h.gpu, &mut h.renderer, commit);
        // The tile leaves for its parent, which took over its first label.
        h.labels.hold(bucket);
        h.placing(bucket, &[]);
        let covering = Committed {
            commit_time: 1000.0,
            fade_duration: 300.0,
            handed_over: vec![(bucket, 0, vec![0])],
            ..h.commit(bucket, &[])
        };
        h.labels.committed(&mut h.gpu, &mut h.renderer, covering);
        let meshes = &h.labels.buckets[&bucket];
        let words = &meshes.written[&0].text;
        let word = |p: u32| words[p as usize * state::WORDS];
        let text = |i: usize| {
            let instance = &meshes.tables.instances[i];
            instance.text.into_iter().filter(|&p| p != orb_symbol::NONE)
        };
        assert!(text(0).count() > 0);
        assert!(text(0).all(|p| word(p) & state::HIDDEN != 0));
        // The rest fade out from where they are.
        let other = (1..meshes.tables.instances.len())
            .find(|&i| text(i).count() > 0)
            .expect("another label");
        assert!(text(other).all(|p| word(p) & (state::HIDDEN | state::OPACITY) == 127));
    }

    #[test]
    fn held_buckets_wait_for_a_covering_commit_then_fade_out() {
        use orb_symbol::state;
        let mut h = Harness::new(SceneCapacity::default());
        let mut resident = h.tile();
        h.upload(&mut resident).unwrap();
        let bucket = resident.symbols[0].bucket;
        let fade = |c: Committed| Committed {
            fade_duration: 300.0,
            ..c
        };
        h.placing(bucket, &[0]);
        let commit = fade(h.commit(bucket, &[0]));
        h.labels.committed(&mut h.gpu, &mut h.renderer, commit);
        // The tile leaves while the next placement is already out: that
        // one doesn't cover its replacement.
        h.placing(bucket, &[0]);
        h.labels.hold(bucket);
        let early = Committed {
            commit_time: 1000.0,
            ..fade(h.commit(bucket, &[0]))
        };
        h.labels.committed(&mut h.gpu, &mut h.renderer, early);
        assert!(!h.labels.released(bucket, 5000.0), "still waiting");
        let shown = |h: &Harness| h.labels.buckets[&bucket].written[&0].text[0];
        assert_eq!(shown(&h) & state::PLACED, state::PLACED);
        // The next one does: the labels fade out from full.
        h.placing(bucket, &[]);
        let covering = Committed {
            changed: Vec::new(),
            commit_time: 2000.0,
            ..fade(h.commit(bucket, &[]))
        };
        h.labels.committed(&mut h.gpu, &mut h.renderer, covering);
        assert_eq!(shown(&h) & (state::PLACED | state::OPACITY), 127);
        assert!(h.labels.fading(2100.0));
        assert!(!h.labels.released(bucket, 2100.0));
        assert!(h.labels.released(bucket, 2300.0));
    }

    /// A failed upload hands its glyph leases back.
    #[test]
    fn failed_uploads_release_their_glyph_rows() {
        let mut h = Harness::new(SceneCapacity {
            vertex_words: 64,
            ..SceneCapacity::fixed()
        });
        let mut resident = h.tile();
        assert!(h.upload(&mut resident).is_err());
        h.labels.retire(&mut h.gpu, h.renderer.glyphs());
        assert!(h.labels.glyphs.leases.is_empty());
    }

    fn pbf(stack: &str, code_point: u32) -> GlyphKey {
        GlyphKey::Pbf {
            stack: orb_text::FontStack::from_joined(stack),
            code_point,
        }
    }

    fn bitmap(index: u32) -> GlyphBitmap {
        GlyphBitmap {
            index,
            size: [2, 2],
            sdf: vec![255; 4],
        }
    }

    /// A bitmap asked for one lease of a row never lands in a later
    /// lease of the same row.
    #[test]
    fn raster_replies_land_only_in_their_lease() {
        let mut h = Harness::new(SceneCapacity::default());
        let atlas = h.renderer.glyphs();
        let mut rows = GlyphRows::default();
        let template = Some("t/{fontstack}/{range}.pbf");
        let row = rows.lease(atlas, &pbf("X", 65), template);
        let first: Vec<RasterRow> = rows.pending.drain().flat_map(|(_, r)| r).collect();
        rows.release(&mut h.gpu, atlas, row);
        assert_eq!(
            rows.lease(atlas, &pbf("Y", 65), template),
            row,
            "row reused"
        );
        let second: Vec<RasterRow> = rows.pending.drain().flat_map(|(_, r)| r).collect();
        rows.in_flight = 2;
        rows.apply(&mut h.gpu, atlas, &first, &[bitmap(65)]);
        // Uploading twice into one row would panic.
        rows.apply(&mut h.gpu, atlas, &second, &[bitmap(65)]);
    }

    /// One stack name under two glyphs templates is two fonts.
    #[test]
    fn glyph_leases_are_per_glyphs_template() {
        let mut h = Harness::new(SceneCapacity::default());
        let atlas = h.renderer.glyphs();
        let mut rows = GlyphRows::default();
        let a = rows.lease(atlas, &pbf("X", 65), Some("a/{fontstack}/{range}.pbf"));
        let b = rows.lease(atlas, &pbf("X", 65), Some("b/{fontstack}/{range}.pbf"));
        assert_ne!(a, b);
        assert!(rows.pending.contains_key("b/X/0-255.pbf"));
    }
}

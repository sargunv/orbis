//! GL JS `placement.ts` and `pauseable_placement.ts`: one placement of
//! every drawn symbol layer against a camera snapshot, its commit with
//! fades, and the state each placed symbol draws with.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use orb_text::{Anchor, WritingMode};

use crate::bucket::{
    AnchorOffset, NONE, PlacedSymbol, SizeKind, SortKeyRange, SymbolInstance, SymbolWritingMode,
    instance_flags, justification,
};
use crate::geometry::{EXTENT, Point};
use crate::layout::{anchor_justification, justification_slot};
use crate::placement::collision::{
    BoxPlacement, CollisionBox, CollisionIndex, LineLabel, PlacedBox, PlacedCircles, Scratch,
    Viewport,
};
use crate::placement::cross_tile::{CrossTileIndex, Indexed};
use crate::placement::grid::{GridKey, Overlap};
use crate::placement::view::{Camera, TileView};

/// The placement lane's view of one symbol layer's tile: the tables
/// placement reads, never quads.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlacementTables {
    pub instances: Vec<SymbolInstance>,
    pub text_placed: Vec<PlacedSymbol>,
    pub icon_placed: Vec<PlacedSymbol>,
    /// Per text quad, its offset along the line.
    pub text_glyph_offsets: Vec<f32>,
    pub lines: Vec<[i16; 2]>,
    pub anchor_offsets: Vec<AnchorOffset>,
    pub sort_key_ranges: Vec<SortKeyRange>,
    pub text_size: SizeKind,
    pub allow_vertical_placement: bool,
    /// The bucket's tile was laid out again (a data or style change): its
    /// symbols show without fading in the first time it is placed (GL JS
    /// and MapLibre Native `justReloaded`). A tile's first load fades its
    /// new symbols in.
    pub reloaded: bool,
    /// Beyond the spec, each instance's own `text-translate` and
    /// `icon-translate` where the layer reads them per feature: pixels at
    /// the tile's zoom and the next, blended by
    /// [`PlacedTile::translate_t`]. Empty where the layer's apply.
    pub translates: [Vec<[[f32; 2]; 2]>; 2],
}

impl PlacementTables {
    /// Instance `i`'s own text (`part` 0) or icon (1) translate in pixels,
    /// `t` of the way to the next zoom's, if the layer reads it per
    /// feature.
    fn own_translate(&self, part: usize, i: usize, t: f64) -> Option<[f64; 2]> {
        let [a, b] = self.translates[part].get(i)?.map(|v| v.map(f64::from));
        Some([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
    }
}

/// A layer's placement settings, from its evaluated layout and paint.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LayerParams {
    /// The layer's key.
    pub layer: u32,
    /// An earlier layer that lays out the same symbols, which places them
    /// for both, as GL JS and Native place a bucket for the first of the
    /// layers it groups: this layer's symbols show where that layer's do,
    /// rather than colliding with them.
    pub follows: Option<u32>,
    /// Its source: symbols of different sources collide only with
    /// `crossSourceCollisions`.
    pub source: u32,
    /// Placed at zooms `[minzoom, maxzoom)`; 0 sets no bound.
    pub minzoom: f64,
    pub maxzoom: f64,
    /// The overlap, optional and ignore-placement settings; a symbol's
    /// own where its feature gives them ([`LayerParams::flags_of`]).
    pub flags: SymbolFlags,
    /// Which of `flags` each symbol carries its own of, read per feature
    /// beyond the spec ([`SymbolFlags::TEXT_OVERLAP`] and the rest); the
    /// others are the layer's, which may follow the camera.
    pub own_flags: u8,
    /// `text-rotation-alignment: map` (after `auto`).
    pub rotate_with_map: bool,
    /// `text-pitch-alignment: map` (after `auto`).
    pub pitch_with_map: bool,
    pub icon_text_fit: bool,
    pub z_order_viewport_y: bool,
    /// Symbols may overlap and draw ordered by viewport y (GL JS
    /// `sortFeaturesByY`): each commit sends draw orders.
    pub draw_by_viewport_y: bool,
    /// The layer sorts by `symbol-sort-key`: placement runs across tiles
    /// in key order.
    pub sort_across_tiles: bool,
    pub height_anchor_ground: bool,
    /// `text-padding`, pixels.
    pub text_padding: f64,
    /// `text-translate` and `icon-translate`, pixels, and whether each is
    /// anchored to the viewport.
    pub text_translate: ([f64; 2], bool),
    pub icon_translate: ([f64; 2], bool),
    /// `text-writing-mode` for point labels, in order.
    pub writing_modes: Vec<WritingMode>,
    /// The layer sets `text-variable-anchor` or `text-variable-anchor-offset`.
    pub variable_placement: bool,
}

/// How a symbol collides: `text-overlap`, `icon-overlap` (after the
/// `*-allow-overlap`s), and the optional and ignore-placement flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SymbolFlags {
    pub text_overlap: Overlap,
    pub icon_overlap: Overlap,
    pub text_optional: bool,
    pub icon_optional: bool,
    pub text_ignore_placement: bool,
    pub icon_ignore_placement: bool,
}

impl SymbolFlags {
    /// [`instance_flags::OWN_PLACEMENT`] and the bits after it hold them.
    const SHIFT: u32 = 9;

    /// `text-overlap` (after `text-allow-overlap`), as a bit of
    /// [`LayerParams::own_flags`]; the rest likewise.
    pub const TEXT_OVERLAP: u8 = 1;
    pub const ICON_OVERLAP: u8 = 2;
    pub const TEXT_OPTIONAL: u8 = 4;
    pub const ICON_OPTIONAL: u8 = 8;
    pub const TEXT_IGNORE_PLACEMENT: u8 = 16;
    pub const ICON_IGNORE_PLACEMENT: u8 = 32;

    /// These, but for those of `own` the instance's, where its feature
    /// gives them (beyond the spec: GL JS and Native read them per layer).
    fn of(self, own: u8, instance: &SymbolInstance) -> Self {
        if own == 0 || instance.flags & instance_flags::OWN_PLACEMENT == 0 {
            return self;
        }
        let bits = instance.flags >> Self::SHIFT;
        let mine = |flag: u8| own & flag != 0;
        let overlap = |flag: u8, b: u32, layer: Overlap| {
            if !mine(flag) {
                return layer;
            }
            match b & 3 {
                1 => Overlap::Always,
                2 => Overlap::Cooperative,
                _ => Overlap::Never,
            }
        };
        let flag = |flag: u8, bit: u32, layer: bool| {
            if mine(flag) { bits & bit != 0 } else { layer }
        };
        Self {
            text_overlap: overlap(Self::TEXT_OVERLAP, bits, self.text_overlap),
            icon_overlap: overlap(Self::ICON_OVERLAP, bits >> 2, self.icon_overlap),
            text_optional: flag(Self::TEXT_OPTIONAL, 16, self.text_optional),
            icon_optional: flag(Self::ICON_OPTIONAL, 32, self.icon_optional),
            text_ignore_placement: flag(
                Self::TEXT_IGNORE_PLACEMENT,
                64,
                self.text_ignore_placement,
            ),
            icon_ignore_placement: flag(
                Self::ICON_IGNORE_PLACEMENT,
                128,
                self.icon_ignore_placement,
            ),
        }
    }

    /// Their instance flags, which [`of`](Self::of) reads.
    pub(crate) fn instance_bits(self) -> u32 {
        let overlap = |o: Overlap| match o {
            Overlap::Never => 0,
            Overlap::Always => 1,
            Overlap::Cooperative => 2,
        };
        let bits = overlap(self.text_overlap)
            | overlap(self.icon_overlap) << 2
            | u32::from(self.text_optional) << 4
            | u32::from(self.icon_optional) << 5
            | u32::from(self.text_ignore_placement) << 6
            | u32::from(self.icon_ignore_placement) << 7;
        instance_flags::OWN_PLACEMENT | bits << Self::SHIFT
    }

    /// Whether text and icon show whatever else collides (GL JS
    /// `alwaysShowText`, `alwaysShowIcon`).
    fn always(self, has_text: bool, has_icon: bool) -> (bool, bool) {
        let text = self.text_overlap == Overlap::Always;
        let icon = self.icon_overlap == Overlap::Always;
        (
            text && (icon || !has_icon || self.icon_optional),
            icon && (text || !has_text || self.text_optional),
        )
    }
}

impl LayerParams {
    /// A symbol's overlap, optional and ignore-placement settings: its own
    /// of those the layer reads per feature, the layer's for the rest.
    pub fn flags_of(&self, instance: &SymbolInstance) -> SymbolFlags {
        self.flags.of(self.own_flags, instance)
    }
}

/// A text size evaluated at the camera's zoom for one tile (GL JS
/// `evaluateSizeForZoom`): `u_size` for constant and camera sizes, the
/// interpolation factor `u_size_t` for composite ones. Equal when the
/// bits are: frames compare to see whether their inputs changed, and a
/// NaN size (`["sqrt", -1]`) is an input that stays the same.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ZoomSize {
    pub size: f64,
    pub t: f64,
}

impl PartialEq for ZoomSize {
    fn eq(&self, other: &Self) -> bool {
        self.size.to_bits() == other.size.to_bits() && self.t.to_bits() == other.t.to_bits()
    }
}

/// One drawn tile of a layer.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlacedTile {
    pub bucket: u64,
    pub view: TileView,
    pub text_size: ZoomSize,
    /// How far between the tile's zoom and the next the camera's zoom
    /// blends per-feature text and icon translates
    /// ([`PlacementTables::translates`]).
    pub translate_t: [f64; 2],
}

/// One layer of a placement, its drawn tiles in any order.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LayerFrame {
    pub params: LayerParams,
    pub tiles: Vec<PlacedTile>,
}

/// What a placement runs against.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Frame {
    pub camera: Camera,
    /// Symbol layers in style order.
    pub layers: Vec<LayerFrame>,
    /// Milliseconds on the scene's clock.
    pub now: f64,
    /// `fadeDuration`, milliseconds.
    pub fade_duration: f64,
    pub cross_source_collisions: bool,
    /// The camera's longitude, degrees, for world copies.
    pub lng: f64,
    /// The terrain, if the map has one.
    /// Placement workers have their own elevations.
    #[serde(skip)]
    pub ground: Option<GroundRef>,
    /// Where labels collide.
    pub scope: Scope,
    /// In [`Scope::Viewport`], labels keep their place across placements
    /// as MapLibre Native keeps them: each label tries its previous
    /// variable anchor first (in tilted views, only that anchor), and in
    /// tilted views labels shown before are placed before new ones, and
    /// those before labels hidden before. What a placement shows then
    /// depends on the placements before it; without this it depends on
    /// the frame alone.
    pub stable_labels: bool,
}

/// Where a placement collides labels.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Scope {
    /// Against the view, continuing the last placement: fades, variable
    /// anchors and orientations carry over, as in GL JS.
    #[default]
    Viewport,
    /// Against the view as if nothing had been placed before, so that the
    /// same inputs always place alike: still output.
    Still,
    /// Each tile of an output grid alone, as if nothing had been placed
    /// before (tile scope): a tile places the labels anchored in it against
    /// each other in its own view, and a label that then overlaps one of a
    /// neighbouring tile's, ranked above it, is dropped. Neither step
    /// depends on which tile is being drawn, so tiles drawn apart agree on
    /// every label that crosses their shared edges, as long as their
    /// neighbourhoods hold each such label's competitors.
    Tiles(Vec<ScopeTile>),
}

/// One tile of a tile-scope placement.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScopeTile {
    /// Its column and row in the output grid: neighbours differ by one.
    pub col: i64,
    pub row: i64,
    /// A view centred on the tile whose viewport is the tile, unrotated and
    /// looking straight down, the same size for every tile.
    pub camera: Camera,
    /// The frame's layers as that view sees them.
    pub layers: Vec<LayerFrame>,
}

/// A frame's terrain. Frames with the same terrain snapshot compare
/// equal.
#[derive(Clone, Debug)]
pub struct GroundRef(pub std::sync::Arc<dyn super::Ground>);

impl PartialEq for GroundRef {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Bits of a placed symbol's state word. The two words
/// after it are the variable-anchor shift, in ems of the label's size.
pub mod state {
    /// Opacity at commit, 0 to 127.
    pub const OPACITY: u32 = 0x7f;
    /// Fading in (placed) rather than out.
    pub const PLACED: u32 = 1 << 7;
    /// Drawn by nothing: another orientation, justification or anchor is.
    pub const HIDDEN: u32 = 1 << 8;
    /// The shift words hold a variable anchor's shift.
    pub const SHIFTED: u32 = 1 << 9;
    /// The shift turns with a vertical label: an icon fitted to it.
    pub(crate) const QUARTER_TURN: u32 = 1 << 10;
    /// For an icon shifted with its text: the text's justification slot.
    pub(crate) const TEXT_SLOT_SHIFT: u32 = 11;
    /// Words per placed symbol.
    pub const WORDS: usize = 3;
}

/// The words of a fully hidden symbol.
const HIDDEN_WORDS: [u32; 3] = [state::HIDDEN, 0, 0];

/// A bucket's state words, [`state::WORDS`] per placed symbol, and, for
/// layers drawn by viewport y, the quads of each stream in draw order.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BucketStates {
    pub text: Vec<u32>,
    pub icon: Vec<u32>,
    pub text_order: Vec<u32>,
    pub icon_order: Vec<u32>,
}

/// A committed placement: the buckets whose state words changed, by world
/// copy (wrap), and how fades progress.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Committed {
    pub changed: Vec<(u64, i32, BucketStates)>,
    pub commit_time: f64,
    pub fade_duration: f64,
    /// GL JS `prevZoomAdjustment`.
    pub zoom_adjustment: f64,
    /// When a symbol was last placed or hidden differently, so a fade
    /// started (GL JS `lastPlacementChangeTime`): a commit that changes
    /// nothing keeps the earlier fades' time, as they run on.
    pub last_change_time: f64,
    /// Some symbol's opacity was committed partway: its state word holds
    /// it rounded down to 1/127, so it reaches its end only once
    /// [`fade_change`](Self::fade_change) does, which can be after the
    /// fade's last change plus its duration.
    pub partway: bool,
    /// Per world copy (bucket, wrap) this placement no longer draws, which
    /// its host may keep drawing as last placed while its labels fade out
    /// (a tile that left the draw list): the instances whose labels a
    /// drawn copy took over, matched across tiles and zooms. That copy
    /// carries the label's fade on from where it was (GL JS's held tiles'
    /// duplicates, hidden at once), so these hide now rather than fade out
    /// beside it.
    pub handed_over: Vec<(u64, i32, Vec<u32>)>,
}

impl Committed {
    /// GL JS `symbolFadeChange`: how far fades have run at `now`.
    pub fn fade_change(&self, now: f64) -> f64 {
        if self.fade_duration == 0.0 {
            1.0
        } else {
            (now - self.commit_time) / self.fade_duration + self.zoom_adjustment
        }
    }

    /// GL JS `Placement.hasTransitions`: whether fades still run at `now`,
    /// and until every opacity committed partway reaches its end: what an
    /// idle frame shows mustn't depend on when the last commit fell.
    pub fn fading(&self, now: f64) -> bool {
        now - self.last_change_time < self.fade_duration
            || (self.partway && self.fade_change(now) < 1.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OpacityState {
    opacity: f64,
    placed: bool,
}

impl OpacityState {
    fn new(prev: Option<OpacityState>, increment: f64, placed: bool, skip_fade: bool) -> Self {
        let opacity = match prev {
            Some(p) => (p.opacity + if p.placed { increment } else { -increment }).clamp(0.0, 1.0),
            None if skip_fade && placed => 1.0,
            None => 0.0,
        };
        Self { opacity, placed }
    }

    fn is_hidden(&self) -> bool {
        self.opacity == 0.0 && !self.placed
    }

    /// Still fading toward shown or hidden.
    fn partway(&self) -> bool {
        self.opacity != if self.placed { 1.0 } else { 0.0 }
    }

    /// GL JS `packOpacity`, as one word's low byte.
    fn pack(&self) -> u32 {
        if self.is_hidden() {
            return 0;
        }
        let bits = (self.opacity * 127.0).floor() as u32;
        bits | if self.placed { state::PLACED } else { 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct JointOpacity {
    text: OpacityState,
    icon: OpacityState,
}

impl JointOpacity {
    fn new(
        prev: Option<&JointOpacity>,
        increment: f64,
        text: bool,
        icon: bool,
        skip_fade: bool,
    ) -> Self {
        Self {
            text: OpacityState::new(prev.map(|p| p.text), increment, text, skip_fade),
            icon: OpacityState::new(prev.map(|p| p.icon), increment, icon, skip_fade),
        }
    }

    fn is_hidden(&self) -> bool {
        self.text.is_hidden() && self.icon.is_hidden()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct JointPlacement {
    text: bool,
    icon: bool,
    skip_fade: bool,
}

/// GL JS `VariableOffset`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct VariableOffset {
    text_offset: [f64; 2],
    width: f64,
    height: f64,
    anchor: Anchor,
    text_box_scale: f64,
}

/// What one placement decided, by cross-tile id.
#[derive(Debug, Default)]
struct Decisions {
    placements: HashMap<u32, JointPlacement>,
    opacities: HashMap<u32, JointOpacity>,
    variable_offsets: HashMap<u32, VariableOffset>,
    /// A [`SymbolWritingMode`] code.
    orientations: HashMap<u32, u8>,
    commit_time: f64,
    last_change_time: f64,
    /// Some opacity is partway (see [`Committed::partway`]).
    partway: bool,
    zoom: f64,
}

/// One world copy of a bucket, placed as a tile of its own, as GL JS
/// places each wrapped tile.
#[derive(Debug)]
struct Bucket {
    tables: Arc<PlacementTables>,
    /// The copy's identity in the cross-tile index; it survives wrap jumps.
    id: u64,
    /// Cross-tile ids, per instance.
    ids: Vec<u32>,
    /// Reloaded and not yet placed: its symbols appear without fading in
    /// (GL JS `justReloaded`).
    just_reloaded: bool,
    /// The words last committed.
    states: BucketStates,
}

/// The placement lane's state: the buckets it was given, their drawn
/// world copies, the cross-tile index and the last commit.
#[derive(Debug, Default)]
pub struct Placer {
    tables: HashMap<u64, Arc<PlacementTables>>,
    /// By bucket and wrap.
    buckets: HashMap<(u64, i32), Bucket>,
    next_copy: u64,
    /// The camera longitude of the last placement, for wrap jumps.
    lng: f64,
    index: CrossTileIndex,
    prev: Option<Decisions>,
    /// Buckets placed so far: a reloaded one's copies fade in after its
    /// first placement.
    placed: HashSet<u64>,
}

fn copy_key(t: &PlacedTile) -> (u64, i32) {
    (t.bucket, t.view.id.wrap)
}

/// Of the `dropped` copies' labels, those a copy among `kept` now has by
/// cross-tile id: per dropped copy that has any, their instances.
fn handed_over<'a>(
    dropped: Vec<(u64, i32, Vec<u32>)>,
    kept: impl Iterator<Item = &'a Bucket>,
) -> Vec<(u64, i32, Vec<u32>)> {
    if dropped.is_empty() {
        return Vec::new();
    }
    let claimed: HashSet<u32> = kept.flat_map(|b| b.ids.iter().copied()).collect();
    dropped
        .into_iter()
        .filter_map(|(bucket, wrap, ids)| {
            let instances: Vec<u32> = (ids.iter().enumerate())
                .filter(|(_, id)| claimed.contains(id))
                .map(|(i, _)| i as u32)
                .collect();
            (!instances.is_empty()).then_some((bucket, wrap, instances))
        })
        .collect()
}

/// GL JS `calculateVariableLayoutShift`.
fn layout_shift(anchor: Anchor, width: f64, height: f64, offset: [f64; 2], scale: f64) -> Point {
    let (h, v) = anchor.alignment();
    Point::new(
        -(h - 0.5) * width + offset[0] * scale,
        -(v - 0.5) * height + offset[1] * scale,
    )
}

/// GL JS `calculateVariableRenderShift` without the render size: ems.
fn render_shift(o: &VariableOffset) -> Point {
    let (h, v) = o.anchor.alignment();
    Point::new(
        -(h - 0.5) * o.width / o.text_box_scale + o.text_offset[0],
        -(v - 0.5) * o.height / o.text_box_scale + o.text_offset[1],
    )
}

/// The collision circles of a line label `frame` shows, as placement
/// places them for its camera, each `[dx, dy, radius]` in viewport pixels
/// from the label's projected anchor: GL JS's queries test a line label
/// by these. Empty for labels without collision circles.
pub fn label_circles(
    frame: &Frame,
    params: &LayerParams,
    tile: &PlacedTile,
    tables: &PlacementTables,
    index: usize,
) -> Vec<[f64; 3]> {
    let instance = &tables.instances[index];
    let center = instance.text[justification::CENTER];
    if instance.flags & instance_flags::RUNTIME_COLLISION_CIRCLES == 0 || center == NONE {
        return Vec::new();
    }
    let symbol = &tables.text_placed[center as usize];
    let view = Viewport::new(frame.camera);
    let ground = frame.ground.as_ref().map(|g| &*g.0);
    let anchor = view.anchor_px(&tile.view, instance.anchor, ground);
    let placement = BoxPlacement {
        tile: &tile.view,
        overlap: Overlap::Always,
        pixel_ratio: 512.0 / EXTENT,
        pitch_with_map: params.pitch_with_map,
        rotate_with_map: params.rotate_with_map,
        translation: translation(
            &frame.camera,
            &tile.view,
            label_translates(params, tile, tables, index)[0],
        ),
        group: None,
        height_offset: f64::from(instance.height_offset),
        height_anchor_ground: params.height_anchor_ground,
        ground,
    };
    let label = LineLabel {
        symbol,
        lines: &tables.lines,
        glyph_offsets: &tables.text_glyph_offsets,
        font_size: feature_size(tables.text_size, tile.text_size, symbol),
        circle_diameter: f64::from(instance.collision_circle_diameter),
        padding: params.text_padding,
    };
    let placed = view.place_circles(&placement, &label, &mut Scratch::default(), |_, _, _| false);
    placed
        .circles
        .into_iter()
        .map(|[x, y, r]| [x - anchor.x, y - anchor.y, r])
        .collect()
}

fn collision_box(instance: &SymbolInstance, i: usize, flag: u32) -> Option<CollisionBox> {
    (instance.flags & flag != 0).then(|| {
        let b = instance.boxes[i];
        CollisionBox {
            anchor: Point::new(f64::from(instance.anchor[0]), f64::from(instance.anchor[1])),
            x1: f64::from(b[0]),
            y1: f64::from(b[1]),
            x2: f64::from(b[2]),
            y2: f64::from(b[3]),
        }
    })
}

fn quads(placed: &[PlacedSymbol], index: u32) -> u32 {
    if index == NONE {
        0
    } else {
        placed[index as usize].quads[1]
    }
}

/// The text's and icon's `*-translate` of instance `index` of a tile's
/// tables: pixels, and whether each turns with the viewport. The layer's,
/// or beyond the spec the instance's own.
pub fn label_translates(
    params: &LayerParams,
    tile: &PlacedTile,
    tables: &PlacementTables,
    index: usize,
) -> [([f64; 2], bool); 2] {
    let layer = [params.text_translate, params.icon_translate];
    [0, 1].map(|part| {
        let (px, viewport) = layer[part];
        let own = tables.own_translate(part, index, tile.translate_t[part]);
        (own.unwrap_or(px), viewport)
    })
}

/// GL JS `translatePosition`: a pixel translation in tile units.
fn translation(camera: &Camera, tile: &TileView, translate: ([f64; 2], bool)) -> [f64; 2] {
    let (mut t, viewport) = translate;
    if t == [0.0; 2] {
        return t;
    }
    if viewport && camera.bearing != 0.0 {
        let (s, c) = (libm::sin(camera.bearing), libm::cos(camera.bearing));
        t = [t[0] * c - t[1] * s, t[0] * s + t[1] * c];
    }
    let units = tile.tile_units_per_pixel(camera);
    [t[0] * units, t[1] * units]
}

/// GL JS `evaluateSizeForFeature`.
fn feature_size(kind: SizeKind, size: ZoomSize, symbol: &PlacedSymbol) -> f64 {
    let [lower, upper] = symbol.size.map(|s| f64::from(s) / 128.0);
    match kind {
        SizeKind::Source => lower,
        SizeKind::Composite { .. } => lower + (upper - lower) * size.t,
        _ => size.size,
    }
}

impl Placer {
    /// Adds a resident bucket (or replaces one with the same id).
    pub fn ingest(&mut self, id: u64, tables: impl Into<Arc<PlacementTables>>) {
        self.tables.insert(id, tables.into());
        self.buckets.retain(|&(bucket, _), _| bucket != id);
        self.placed.remove(&id);
    }

    pub fn remove(&mut self, id: u64) {
        self.tables.remove(&id);
        self.buckets.retain(|&(bucket, _), _| bucket != id);
        self.placed.remove(&id);
    }

    /// Keeps the copies `frame` draws, re-wrapped when the camera jumped a
    /// world (GL JS `handleWrapJump`), and adds the new ones. Returns the
    /// copies it dropped, by the wrap they were drawn at, with their
    /// labels' cross-tile ids.
    fn update_copies(&mut self, frame: &Frame) -> Vec<(u64, i32, Vec<u32>)> {
        let delta = ((frame.lng - self.lng) / 360.0).round() as i32;
        self.lng = frame.lng;
        if delta != 0 {
            self.buckets = std::mem::take(&mut self.buckets)
                .into_iter()
                .map(|((bucket, wrap), b)| ((bucket, wrap + delta), b))
                .collect();
        }
        let drawn: HashSet<(u64, i32)> = frame
            .layers
            .iter()
            .flat_map(|l| &l.tiles)
            .filter(|t| self.tables.contains_key(&t.bucket))
            .map(copy_key)
            .collect();
        let mut dropped = Vec::new();
        self.buckets.retain(|&(bucket, wrap), b| {
            let keep = drawn.contains(&(bucket, wrap));
            if !keep {
                dropped.push((bucket, wrap - delta, std::mem::take(&mut b.ids)));
            }
            keep
        });
        for key in drawn {
            self.buckets.entry(key).or_insert_with(|| {
                self.next_copy += 1;
                Bucket {
                    tables: self.tables[&key.0].clone(),
                    id: self.next_copy,
                    ids: Vec::new(),
                    just_reloaded: self.tables[&key.0].reloaded && !self.placed.contains(&key.0),
                    states: BucketStates::default(),
                }
            });
        }
        dropped
    }

    /// The drawn tiles of each layer whose buckets were ingested, GL JS's
    /// order: a source's tiles by zoom, finest first.
    fn drawn(&self, layers: &[LayerFrame]) -> Vec<(LayerParams, Vec<PlacedTile>)> {
        layers
            .iter()
            .map(|l| {
                let mut tiles: Vec<PlacedTile> = l
                    .tiles
                    .iter()
                    .filter(|t| self.buckets.contains_key(&copy_key(t)))
                    .cloned()
                    .collect();
                tiles.sort_by_key(|t| {
                    let id = t.view.id;
                    (std::cmp::Reverse(id.z), id.wrap, id.x, id.y)
                });
                (l.params.clone(), tiles)
            })
            .collect()
    }

    /// Places every layer of `frame` and commits, returning the buckets
    /// whose states changed. Tiles of buckets not ingested are skipped.
    pub fn place(&mut self, frame: &Frame) -> Committed {
        if frame.scope != Scope::Viewport {
            // Still output places as if nothing had been placed before.
            self.buckets.clear();
            self.index = CrossTileIndex::default();
            self.prev = None;
        }
        let dropped = self.update_copies(frame);
        let camera = &frame.camera;
        let layers = self.drawn(&frame.layers);
        // Each layer's copies lend the index their ids.
        let mut lent = Vec::new();
        let mut keys = HashSet::new();
        for (params, tiles) in &layers {
            keys.clear();
            for t in tiles {
                // Each copy appears once per layer.
                if !keys.insert(copy_key(t)) {
                    continue;
                }
                let b = self.buckets.get_mut(&copy_key(t)).expect("drawn");
                let ids = std::mem::take(&mut b.ids);
                lent.push((copy_key(t), t.view.id, b.id, b.tables.clone(), ids));
            }
            let mut indexed: Vec<Indexed<'_>> = (lent.iter_mut())
                .map(|(_, tile, bucket, tables, ids)| Indexed {
                    tile: *tile,
                    bucket: *bucket,
                    instances: &tables.instances,
                    ids,
                })
                .collect();
            self.index.add_layer(params.layer, &mut indexed, frame.lng);
            for (key, .., ids) in lent.drain(..) {
                self.buckets.get_mut(&key).expect("lent above").ids = ids;
            }
        }
        self.index
            .retain_layers(&layers.iter().map(|(p, _)| p.layer).collect());
        let handed_over = handed_over(dropped, self.buckets.values());

        let ground = frame.ground.as_ref().map(|g| &*g.0);
        let now = match &frame.scope {
            Scope::Tiles(tiles) => self.place_tiles(frame, tiles),
            Scope::Viewport | Scope::Still => {
                let mut placer = Pass::new(camera, self.prev.as_ref(), ground);
                placer.stable = frame.stable_labels && frame.scope == Scope::Viewport;
                placer.run(&self.buckets, &layers, frame.cross_source_collisions);
                placer.now
            }
        };
        let mut now = now;
        self.follow(&layers, &mut now);
        commit(&mut now, self.prev.as_ref(), frame.now, frame.fade_duration);
        let zoom_adjustment = self
            .prev
            .as_ref()
            .map_or(0.0, |p| ((p.zoom - camera.zoom) / 1.5).max(0.0));

        let mut changed = Vec::new();
        for (params, tiles) in &layers {
            let mut seen = HashSet::new();
            for t in tiles {
                let b = self.buckets.get_mut(&copy_key(t)).expect("filtered above");
                b.just_reloaded = false;
                self.placed.insert(t.bucket);
                let states = bucket_states(&mut now, params, b, &mut seen, camera.bearing);
                if states != b.states {
                    b.states = states.clone();
                    changed.push((t.bucket, t.view.id.wrap, states));
                }
            }
        }
        let committed = Committed {
            changed,
            commit_time: frame.now,
            fade_duration: frame.fade_duration,
            zoom_adjustment,
            last_change_time: now.last_change_time,
            partway: now.partway,
            handed_over,
        };
        self.prev = Some(now);
        committed
    }

    /// Gives each symbol of a layer that follows another (see
    /// [`LayerParams::follows`]) the placement of the same symbol of that
    /// layer's bucket for the same tile.
    fn follow(&self, layers: &[(LayerParams, Vec<PlacedTile>)], now: &mut Decisions) {
        for (params, tiles) in layers {
            let Some(leader) = params.follows else {
                continue;
            };
            let Some((_, leader_tiles)) = layers.iter().find(|(p, _)| p.layer == leader) else {
                continue;
            };
            for t in tiles {
                let Some(l) = leader_tiles.iter().find(|l| l.view.id == t.view.id) else {
                    continue;
                };
                let (from, to) = (&self.buckets[&copy_key(l)], &self.buckets[&copy_key(t)]);
                if from.ids.len() != to.ids.len() {
                    continue;
                }
                for (&from, &to) in from.ids.iter().zip(&to.ids) {
                    if let Some(&p) = now.placements.get(&from) {
                        now.placements.insert(to, p);
                    }
                    if let Some(&o) = now.variable_offsets.get(&from) {
                        now.variable_offsets.insert(to, o);
                    }
                    if let Some(&o) = now.orientations.get(&from) {
                        now.orientations.insert(to, o);
                    }
                }
            }
        }
    }

    /// Tile scope (see [`Scope::Tiles`]): each tile's own placement, then
    /// the labels that overlap a higher-ranked one of another tile dropped.
    fn place_tiles(&self, frame: &Frame, tiles: &[ScopeTile]) -> Decisions {
        let ground = frame.ground.as_ref().map(|g| &*g.0);
        let mut now = Decisions {
            zoom: frame.camera.zoom,
            ..Decisions::default()
        };
        let mut records = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let layers = self.drawn(&tile.layers);
            let mut pass = Pass::new(&tile.camera, None, ground);
            pass.record = Some(Record::default());
            pass.run(&self.buckets, &layers, frame.cross_source_collisions);
            // A label two tiles both claim (an anchor on their shared edge,
            // within rounding) keeps the first tile's placement.
            let Decisions {
                placements,
                variable_offsets,
                orientations,
                ..
            } = pass.now;
            for (id, p) in placements {
                if let std::collections::hash_map::Entry::Vacant(e) = now.placements.entry(id) {
                    e.insert(p);
                    if let Some(&o) = variable_offsets.get(&id) {
                        now.variable_offsets.insert(id, o);
                    }
                    if let Some(&o) = orientations.get(&id) {
                        now.orientations.insert(id, o);
                    }
                }
            }
            records.push(pass.record.expect("set above"));
        }
        for (id, text, icon) in super::tile_scope::outranked(tiles, &records) {
            if let Some(p) = now.placements.get_mut(&id) {
                p.text = text;
                p.icon = icon;
            }
        }
        now
    }
}

/// What a tile-scope pass placed: each label's shapes as it inserted them,
/// in its view's padded pixels, and its rank.
#[derive(Debug, Default)]
pub(crate) struct Record {
    pub(crate) labels: Vec<RecordedLabel>,
    pub(crate) shapes: Vec<RecordedShape>,
    /// Labels placed so far in this pass.
    next: u64,
}

#[derive(Debug)]
pub(crate) struct RecordedLabel {
    pub(crate) id: u32,
    /// Rank among the labels of every tile, highest first: the layer
    /// (topmost first), the sort key, then (with the tile's position) the
    /// order the tile placed it in.
    pub(crate) layer: u32,
    pub(crate) sort_key: f64,
    pub(crate) order: u64,
    pub(crate) shapes: std::ops::Range<usize>,
    /// Whether text and icon show when the label is dropped: `*-overlap:
    /// always`.
    pub(crate) always_text: bool,
    pub(crate) always_icon: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ShapeGeometry {
    Box([f64; 4]),
    /// `x, y, radius`.
    Circle([f64; 3]),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RecordedShape {
    pub(crate) geometry: ShapeGeometry,
    /// As a query: its overlap mode and the collision group it tests
    /// against; as a placed shape, the key others test.
    pub(crate) overlap: Overlap,
    pub(crate) group: Option<u32>,
    pub(crate) key: GridKey,
    /// Others collide with it: not `*-ignore-placement`.
    pub(crate) blocks: bool,
}

struct Pass<'a> {
    index: CollisionIndex,
    now: Decisions,
    prev: Option<&'a Decisions>,
    camera: &'a Camera,
    ground: Option<&'a dyn super::Ground>,
    /// Tile scope: place only the labels anchored in the view, and record
    /// what they insert.
    record: Option<Record>,
    /// The layer being placed, topmost first, and its part's sort key.
    layer: u32,
    sort_key: f64,
    /// Labels keep their place ([`Frame::stable_labels`]).
    stable: bool,
}

impl<'a> Pass<'a> {
    fn new(
        camera: &'a Camera,
        prev: Option<&'a Decisions>,
        ground: Option<&'a dyn super::Ground>,
    ) -> Self {
        Self {
            index: CollisionIndex::new(*camera),
            now: Decisions {
                zoom: camera.zoom,
                ..Decisions::default()
            },
            prev,
            camera,
            ground,
            record: None,
            layer: 0,
            sort_key: 0.0,
            stable: false,
        }
    }

    /// The previous placement, when labels keep their place.
    fn stable_prev(&self) -> Option<&'a Decisions> {
        self.prev.filter(|_| self.stable)
    }

    /// The view is tilted (MapLibre Native's `isTiltedView`).
    fn tilted(&self) -> bool {
        self.camera.pitch != 0.0
    }

    /// Places `layers`, topmost first.
    fn run(
        &mut self,
        buckets: &HashMap<(u64, i32), Bucket>,
        layers: &[(LayerParams, Vec<PlacedTile>)],
        cross_source_collisions: bool,
    ) {
        for (rank, (params, tiles)) in layers.iter().rev().enumerate() {
            let zoom = self.camera.zoom;
            if params.follows.is_some()
                || (params.minzoom != 0.0 && params.minzoom > zoom)
                || (params.maxzoom != 0.0 && params.maxzoom <= zoom)
            {
                continue;
            }
            self.layer = rank as u32;
            let group = (!cross_source_collisions).then_some(params.source);
            let mut parts: Vec<(f64, &PlacedTile, std::ops::Range<usize>)> = Vec::new();
            for t in tiles {
                let b = &buckets[&copy_key(t)];
                if params.sort_across_tiles {
                    for r in &b.tables.sort_key_ranges {
                        parts.push((r.sort_key, t, r.start as usize..r.end as usize));
                    }
                } else {
                    parts.push((0.0, t, 0..b.tables.instances.len()));
                }
            }
            if params.sort_across_tiles {
                parts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            let mut seen = HashSet::new();
            for (key, t, range) in parts {
                self.sort_key = key;
                let b = &buckets[&copy_key(t)];
                self.place_part(params, group, t, b, range, &mut seen);
            }
        }
    }
    /// GL JS `markUsedOrientation` records which orientation shows; the
    /// words read it from `orientations`.
    fn mark_orientation(&mut self, id: u32, orientation: u8) {
        self.now.orientations.insert(id, orientation);
    }

    /// The order a bucket part's instances are placed in.
    fn order(&self, p: &LayerParams, b: &Bucket, range: std::ops::Range<usize>) -> Vec<usize> {
        let mut order: Vec<usize> = if p.z_order_viewport_y {
            let mut order = viewport_y_order(&b.tables, -self.camera.bearing);
            order.reverse();
            order
        } else {
            range.collect()
        };
        // MapLibre Native's `getSortedSymbols`: in a tilted view, labels
        // shown before go first, then new ones, then those hidden before.
        if let Some(prev) = self.stable_prev().filter(|_| self.tilted()) {
            order.sort_by_key(|&i| match prev.placements.get(&b.ids[i]) {
                Some(p) if p.text || p.icon => 0,
                None => 1,
                Some(_) => 2,
            });
        }
        order
    }

    /// GL JS `placeLayerBucketPart`.
    fn place_part(
        &mut self,
        p: &LayerParams,
        group: Option<u32>,
        tile: &PlacedTile,
        b: &Bucket,
        range: std::ops::Range<usize>,
        seen: &mut HashSet<u32>,
    ) {
        let t = &b.tables;
        let has_text_data = !t.text_placed.is_empty();
        let has_icon_data = !t.icon_placed.is_empty();
        let text_translation = translation(self.camera, &tile.view, p.text_translate);
        let icon_translation = translation(self.camera, &tile.view, p.icon_translate);
        let own_translates = t.translates.iter().any(|t| !t.is_empty());
        for i in self.order(p, b, range) {
            let instance = &t.instances[i];
            let id = b.ids[i];
            if seen.contains(&id) {
                continue;
            }
            // Tile scope: a tile places the labels anchored in it.
            if self.record.is_some() && !self.index.view.in_view(&tile.view, instance.anchor) {
                continue;
            }
            let c = Candidate {
                p,
                tile,
                t,
                instance,
                id,
                f: p.flags_of(instance),
                group,
                text_translation,
                icon_translation,
                ground: self.ground,
            };
            // Beyond the spec, translated each its own way.
            let c = if own_translates {
                let [text, icon] = label_translates(p, tile, t, i);
                Candidate {
                    text_translation: translation(self.camera, &tile.view, text),
                    icon_translation: translation(self.camera, &tile.view, icon),
                    ..c
                }
            } else {
                c
            };
            let text = self.place_text(&c);
            let mut place_text = text.placeable();
            // Without a text box GL JS's `offscreen` is undefined: falsy.
            let mut offscreen = text.placed.is_some_and(|b| b.offscreen);
            let circles = self.place_line_circles(&c);
            if let Some(result) = &circles {
                place_text = c.f.text_overlap == Overlap::Always
                    || (!result.circles.is_empty() && !result.collision_detected);
                offscreen &= result.offscreen;
            }
            let icon = self.place_icon(&c, &text);
            let mut place_icon = icon.is_some_and(|b| b.placeable);
            if let Some(result) = icon {
                offscreen &= result.offscreen;
            }
            (place_text, place_icon) = c.together(place_text, place_icon);

            let (always_text, always_icon) = c.f.always(has_text_data, has_icon_data);
            let text_box = text.placed.filter(|b| place_text && b.placeable);
            let icon_box = icon.filter(|b| place_icon && b.placeable);
            let circles = circles.filter(|_| place_text);
            self.insert(&c, text_box, icon_box, circles, (always_text, always_icon));
            self.now.placements.insert(
                id,
                JointPlacement {
                    text: place_text || always_text,
                    icon: place_icon || always_icon,
                    skip_fade: offscreen || b.just_reloaded,
                },
            );
            seen.insert(id);
        }
    }

    /// A label's text box, in the first writing mode (and variable anchor)
    /// that fits, or the last tried.
    fn place_text(&mut self, c: &Candidate<'_>) -> PlacedText {
        let (t, instance, id) = (c.t, c.instance, c.id);
        let mut text = PlacedText::default();
        let Some(text_box) = c.collision_box(0, instance_flags::TEXT_BOX) else {
            return text;
        };
        let vertical_text_box = c.collision_box(1, instance_flags::VERTICAL_TEXT_BOX);
        let icon_box = c.collision_box(2, instance_flags::ICON_BOX);
        let vertical_icon_box = c.collision_box(3, instance_flags::VERTICAL_ICON_BOX);
        let vertical_glyphs = quads(&t.text_placed, instance.text[justification::VERTICAL]);
        let allow_vertical = t.allow_vertical_placement && vertical_glyphs > 0;
        let offsets = &t.anchor_offsets
            [instance.anchor_offsets[0] as usize..instance.anchor_offsets[1] as usize];
        // Modes in order; the first that fits wins.
        let modes: &[WritingMode] = if allow_vertical && vertical_text_box.is_some() {
            &c.p.writing_modes
        } else {
            &[WritingMode::Horizontal]
        };
        if offsets.is_empty() {
            for &mode in modes {
                let (bx, orientation) = match mode {
                    WritingMode::Vertical => (
                        vertical_text_box.expect("checked"),
                        SymbolWritingMode::Vertical,
                    ),
                    WritingMode::Horizontal => (text_box, SymbolWritingMode::Horizontal),
                };
                let result =
                    self.index
                        .place_box(&bx, &c.base(c.f.text_overlap, c.text_translation), None);
                if result.placeable {
                    self.mark_orientation(id, orientation as u8);
                }
                text.placed = Some(result);
                text.vertical = mode == WritingMode::Vertical;
                if result.placeable {
                    break;
                }
            }
        } else {
            for &mode in modes {
                if mode == WritingMode::Vertical && text.placeable() {
                    break;
                }
                let (bx, ibox, orientation) = match mode {
                    WritingMode::Vertical => (
                        vertical_text_box.expect("checked"),
                        vertical_icon_box,
                        SymbolWritingMode::Vertical,
                    ),
                    WritingMode::Horizontal => (text_box, icon_box, SymbolWritingMode::Horizontal),
                };
                let variable_icon = if c.p.icon_text_fit && c.f.icon_overlap == Overlap::Never {
                    ibox
                } else {
                    None
                };
                // GL JS records the orientation only when the bucket may
                // place vertically.
                let orientation = if t.allow_vertical_placement {
                    orientation
                } else {
                    SymbolWritingMode::None
                };
                text.placed = self.place_variable(
                    c,
                    offsets,
                    &bx,
                    variable_icon,
                    orientation,
                    &mut text.shift,
                );
                text.vertical = mode == WritingMode::Vertical;
                if text.placeable() {
                    break;
                }
            }
            // Not placed: keep the last placement's anchor while it fades.
            if !text.placeable()
                && let Some(prev) = self.prev.and_then(|p| p.variable_offsets.get(&id))
            {
                self.now.variable_offsets.insert(id, *prev);
            }
        }
        // GL JS `updatePreviousOrientationIfNotPlaced`.
        if t.allow_vertical_placement
            && !text.placeable()
            && let Some(&o) = self.prev.and_then(|p| p.orientations.get(&id))
        {
            self.now.orientations.insert(id, o);
        }
        text
    }

    /// A line label's collision circles, if it has them.
    fn place_line_circles(&mut self, c: &Candidate<'_>) -> Option<PlacedCircles> {
        let (t, instance) = (c.t, c.instance);
        let center = instance.text[justification::CENTER];
        if instance.flags & instance_flags::RUNTIME_COLLISION_CIRCLES == 0 || center == NONE {
            return None;
        }
        let symbol = &t.text_placed[center as usize];
        let label = LineLabel {
            symbol,
            lines: &t.lines,
            glyph_offsets: &t.text_glyph_offsets,
            font_size: feature_size(t.text_size, c.tile.text_size, symbol),
            circle_diameter: f64::from(instance.collision_circle_diameter),
            padding: c.p.text_padding,
        };
        let placement = c.base(c.f.text_overlap, c.text_translation);
        Some(self.index.place_circles(&placement, &label))
    }

    /// A label's icon box, turned with its text and moved by the text's
    /// variable anchor when fitted to it.
    fn place_icon(&mut self, c: &Candidate<'_>, text: &PlacedText) -> Option<PlacedBox> {
        let icon_box = c.collision_box(2, instance_flags::ICON_BOX)?;
        let vertical_icon_box = c.collision_box(3, instance_flags::VERTICAL_ICON_BOX);
        let bx = match vertical_icon_box {
            Some(vertical) if text.vertical && text.placeable() => vertical,
            _ => icon_box,
        };
        let shift = if c.p.icon_text_fit { text.shift } else { None };
        let placement = c.base(c.f.icon_overlap, c.icon_translation);
        Some(self.index.place_box(&bx, &placement, shift))
    }

    /// Inserts what a label placed into the collision index, and in tile
    /// scope records it.
    fn insert(
        &mut self,
        c: &Candidate<'_>,
        text: Option<PlacedBox>,
        icon: Option<PlacedBox>,
        circles: Option<PlacedCircles>,
        (always_text, always_icon): (bool, bool),
    ) {
        let (f, group) = (c.f, c.group);
        let key = |overlap| GridKey {
            overlap,
            group: group.unwrap_or(0),
        };
        // Recorded in tile scope only.
        let recording = self.record.is_some();
        let mut shapes = Vec::new();
        let shape = |geometry, overlap, ignore: bool| RecordedShape {
            geometry,
            overlap,
            group,
            key: key(overlap),
            blocks: !ignore,
        };
        if let Some(b) = text {
            let (overlap, ignore) = (f.text_overlap, f.text_ignore_placement);
            self.index.insert_box(b.bounds, key(overlap), ignore);
            shapes.extend(recording.then(|| shape(ShapeGeometry::Box(b.bounds), overlap, ignore)));
        }
        if let Some(b) = icon {
            let (overlap, ignore) = (f.icon_overlap, f.icon_ignore_placement);
            self.index.insert_box(b.bounds, key(overlap), ignore);
            shapes.extend(recording.then(|| shape(ShapeGeometry::Box(b.bounds), overlap, ignore)));
        }
        if let Some(circles) = circles {
            let (overlap, ignore) = (f.text_overlap, f.text_ignore_placement);
            self.index
                .insert_circles(&circles.circles, key(overlap), ignore);
            for &circle in circles.circles.iter().filter(|_| recording) {
                shapes.push(shape(ShapeGeometry::Circle(circle), overlap, ignore));
            }
        }
        if let Some(record) = &mut self.record
            && !shapes.is_empty()
        {
            let start = record.shapes.len();
            record.shapes.extend(shapes);
            record.labels.push(RecordedLabel {
                id: c.id,
                layer: self.layer,
                sort_key: self.sort_key,
                order: record.next,
                shapes: start..record.shapes.len(),
                always_text,
                always_icon,
            });
            record.next += 1;
        }
    }

    /// GL JS `placeBoxForVariableAnchors` with `attemptAnchorPlacement`:
    /// `icon_box` is an icon fitted to the text, which must fit too.
    fn place_variable(
        &mut self,
        c: &Candidate<'_>,
        offsets: &[AnchorOffset],
        text_box: &CollisionBox,
        icon_box: Option<CollisionBox>,
        // The orientation to record, or `None`.
        orientation: SymbolWritingMode,
        shift_out: &mut Option<Point>,
    ) -> Option<PlacedBox> {
        let (instance, id, f) = (c.instance, c.id, c.f);
        let width = text_box.x2 - text_box.x1;
        let height = text_box.y2 - text_box.y1;
        let scale = f64::from(instance.text_box_scale);
        // Anchors in the style's order, so that a frame's labels depend on
        // its camera alone; or, as MapLibre Native does when labels keep
        // their place, the label's previous anchor first, and in a tilted
        // view that anchor alone.
        let reordered: Vec<AnchorOffset>;
        let mut offsets = offsets;
        if let Some(prev) = self.stable_prev().and_then(|p| p.variable_offsets.get(&id))
            && let Some(found) = offsets.iter().position(|o| o.text_anchor() == prev.anchor)
            && found != 0
        {
            let first = offsets[found];
            let rest = offsets.iter().filter(|o| o.text_anchor() != prev.anchor);
            reordered = std::iter::once(first)
                .chain(rest.filter(|_| !self.tilted()).copied())
                .collect();
            offsets = &reordered;
        }
        let passes = if f.text_overlap == Overlap::Never {
            1
        } else {
            2
        };
        let mut overlap = Overlap::Never;
        let mut placed = None;
        for _ in 0..passes {
            for o in offsets {
                let anchor = o.text_anchor();
                let text_offset = o.offset.map(f64::from);
                let shift = layout_shift(anchor, width, height, text_offset, scale);
                let text = c.base(overlap, c.text_translation);
                let result = self.index.place_box(text_box, &text, Some(shift));
                if let Some(ib) = &icon_box {
                    let icon = c.base(overlap, c.icon_translation);
                    if !self.index.place_box(ib, &icon, Some(shift)).placeable {
                        continue;
                    }
                }
                placed = Some(result);
                if result.placeable {
                    self.now.variable_offsets.insert(
                        id,
                        VariableOffset {
                            text_offset,
                            width,
                            height,
                            anchor,
                            text_box_scale: scale,
                        },
                    );
                    if orientation != SymbolWritingMode::None {
                        self.now.orientations.insert(id, orientation as u8);
                    }
                    *shift_out = Some(shift);
                    return placed;
                }
            }
            overlap = f.text_overlap;
        }
        placed
    }
}

/// One label as a pass places it: its layer, tile and bucket, and how it
/// collides.
struct Candidate<'b> {
    p: &'b LayerParams,
    tile: &'b PlacedTile,
    t: &'b PlacementTables,
    instance: &'b SymbolInstance,
    id: u32,
    f: SymbolFlags,
    /// The collision group it tests against, unless sources collide.
    group: Option<u32>,
    text_translation: [f64; 2],
    icon_translation: [f64; 2],
    ground: Option<&'b dyn super::Ground>,
}

impl<'b> Candidate<'b> {
    /// How a box or circle of the label is placed, with `overlap`.
    fn base(&self, overlap: Overlap, translation: [f64; 2]) -> BoxPlacement<'b> {
        BoxPlacement {
            tile: &self.tile.view,
            overlap,
            pixel_ratio: 512.0 / EXTENT,
            pitch_with_map: self.p.pitch_with_map,
            rotate_with_map: self.p.rotate_with_map,
            translation,
            group: self.group,
            height_offset: f64::from(self.instance.height_offset),
            height_anchor_ground: self.p.height_anchor_ground,
            ground: self.ground,
        }
    }

    /// The instance's `i`th collision box, if `flag` says it has one.
    fn collision_box(&self, i: usize, flag: u32) -> Option<CollisionBox> {
        collision_box(self.instance, i, flag)
    }

    /// Whether text and icon show, from whether each was placed: each
    /// shows only with the other unless it is optional or the other is
    /// missing.
    fn together(&self, text: bool, icon: bool) -> (bool, bool) {
        let (t, instance, f) = (self.t, self.instance, self.f);
        let vertical_glyphs = quads(&t.text_placed, instance.text[justification::VERTICAL]);
        let horizontal_glyphs = instance.text[..3]
            .iter()
            .map(|&i| quads(&t.text_placed, i))
            .max()
            .unwrap_or(0);
        let icon_without_text = f.text_optional || (horizontal_glyphs == 0 && vertical_glyphs == 0);
        let text_without_icon = f.icon_optional || quads(&t.icon_placed, instance.icon[0]) == 0;
        if !icon_without_text && !text_without_icon {
            let both = icon && text;
            (both, both)
        } else if !text_without_icon {
            (icon && text, icon)
        } else if !icon_without_text {
            (text, icon && text)
        } else {
            (text, icon)
        }
    }
}

/// Where a label's text box landed.
#[derive(Clone, Copy, Debug, Default)]
struct PlacedText {
    /// The last box tried; `None` without a text box.
    placed: Option<PlacedBox>,
    /// It was the vertical one.
    vertical: bool,
    /// The variable anchor's shift, when one placed it.
    shift: Option<Point>,
}

impl PlacedText {
    fn placeable(&self) -> bool {
        self.placed.is_some_and(|b| b.placeable)
    }
}

/// GL JS `Placement.commit`.
fn commit(now: &mut Decisions, prev: Option<&Decisions>, time: f64, fade_duration: f64) {
    now.commit_time = time;
    let increment = match prev {
        None => 1.0,
        Some(_) if fade_duration == 0.0 => 1.0,
        Some(p) => {
            let adjustment = ((p.zoom - now.zoom) / 1.5).max(0.0);
            (time - p.commit_time) / fade_duration + adjustment
        }
    };
    let empty = HashMap::new();
    let prev_opacities = prev.map_or(&empty, |p| &p.opacities);
    let mut changed = false;
    for (&id, placement) in &now.placements {
        let opacity = match prev_opacities.get(&id) {
            Some(prev) => {
                changed |= placement.text != prev.text.placed || placement.icon != prev.icon.placed;
                JointOpacity::new(Some(prev), increment, placement.text, placement.icon, false)
            }
            None => {
                // A label of a tile just laid out again, or offscreen,
                // shows at once (skip fade):
                // nothing animates, so nothing keeps frames coming.
                changed |= (placement.text || placement.icon) && !placement.skip_fade;
                JointOpacity::new(
                    None,
                    increment,
                    placement.text,
                    placement.icon,
                    placement.skip_fade,
                )
            }
        };
        now.opacities.insert(id, opacity);
    }
    for (&id, prev) in prev_opacities {
        if let std::collections::hash_map::Entry::Vacant(entry) = now.opacities.entry(id) {
            let opacity = JointOpacity::new(Some(prev), increment, false, false, false);
            if !opacity.is_hidden() {
                entry.insert(opacity);
                changed |= prev.text.placed || prev.icon.placed;
            }
        }
    }
    if let Some(prev) = prev {
        for (&id, offset) in &prev.variable_offsets {
            if !now.variable_offsets.contains_key(&id)
                && now.opacities.get(&id).is_some_and(|o| !o.is_hidden())
            {
                now.variable_offsets.insert(id, *offset);
            }
        }
        for (&id, &orientation) in &prev.orientations {
            if !now.orientations.contains_key(&id)
                && now.opacities.get(&id).is_some_and(|o| !o.is_hidden())
            {
                now.orientations.insert(id, orientation);
            }
        }
    }
    now.partway = (now.opacities.values()).any(|o| o.text.partway() || o.icon.partway());
    // Without a change nor an earlier one, nothing fades.
    now.last_change_time = if changed {
        time
    } else {
        prev.map_or(f64::NEG_INFINITY, |p| p.last_change_time)
    };
}

/// GL JS `updateBucketOpacities` with `_markDuplicates`, and the draw-time
/// choices of `updateVariableAnchors`: each placed symbol's words.
fn bucket_states(
    now: &mut Decisions,
    params: &LayerParams,
    b: &Bucket,
    seen: &mut HashSet<u32>,
    camera_bearing: f64,
) -> BucketStates {
    let t = &b.tables;
    let mut states = BucketStates {
        text: HIDDEN_WORDS.repeat(t.text_placed.len()),
        icon: HIDDEN_WORDS.repeat(t.icon_placed.len()),
        ..BucketStates::default()
    };
    let duplicate = JointOpacity::new(None, 0.0, false, false, true);
    let has = (!t.text_placed.is_empty(), !t.icon_placed.is_empty());
    let vertical = SymbolWritingMode::Vertical as u8;
    for (i, instance) in t.instances.iter().enumerate() {
        let id = b.ids[i];
        let opacity = if !seen.insert(id) {
            duplicate
        } else {
            *now.opacities.entry(id).or_insert_with(|| {
                let (text, icon) = params.flags_of(instance).always(has.0, has.1);
                JointOpacity::new(None, 0.0, text, icon, true)
            })
        };
        let orientation = now.orientations.get(&id).copied();
        let horizontal_hidden = orientation == Some(vertical);
        let vertical_hidden = orientation.is_some_and(|o| o != vertical);
        let text_hidden = opacity.text.is_hidden();
        let packed = opacity.text.pack();
        let offset = now.variable_offsets.get(&id).copied();
        // GL JS `markUsedJustification`: with several justifications, only
        // one is drawn. GL JS marks by the variable anchor, then marks
        // `left` (or vertical) again when an orientation is recorded; the
        // last mark wins.
        let used_slot = |slot: usize| -> bool {
            let auto = match (orientation, offset) {
                (Some(o), _) if o == vertical => instance.text[justification::VERTICAL],
                (Some(_), _) => instance.text[justification::LEFT],
                (None, Some(o)) => {
                    instance.text[justification_slot(anchor_justification(o.anchor))]
                }
                (None, None) => NONE,
            };
            auto == NONE || instance.text[slot] == auto
        };
        for slot in 0..4 {
            let index = instance.text[slot];
            if index == NONE {
                continue;
            }
            let is_vertical = slot == justification::VERTICAL;
            let mut word = if (is_vertical && vertical_hidden)
                || (!is_vertical && horizontal_hidden)
                || text_hidden
            {
                state::HIDDEN
            } else {
                packed
            };
            let mut shift = [0.0f32; 2];
            if !used_slot(slot) {
                word |= state::HIDDEN;
            }
            if params.variable_placement {
                // Drawn only where placement chose an anchor, and in the
                // chosen orientation (GL JS `skipOrientation`).
                let skip_orientation = t.allow_vertical_placement && orientation.is_none();
                match offset {
                    Some(o) if !skip_orientation => {
                        let s = render_shift(&o);
                        shift = [s.x as f32, s.y as f32];
                        word |= state::SHIFTED;
                    }
                    _ => word |= state::HIDDEN,
                }
            }
            let at = index as usize * state::WORDS;
            states.text[at..at + 3].copy_from_slice(&[
                word,
                shift[0].to_bits(),
                shift[1].to_bits(),
            ]);
        }

        let icon_packed = opacity.icon.pack();
        let icon_hidden = opacity.icon.is_hidden();
        let use_horizontal =
            !(params.icon_text_fit && instance.icon[1] != NONE && horizontal_hidden);
        for (k, &index) in instance.icon.iter().enumerate() {
            if index == NONE {
                continue;
            }
            let shown = if k == 0 {
                use_horizontal
            } else {
                !use_horizontal
            };
            let mut word = if shown && !icon_hidden {
                icon_packed
            } else {
                state::HIDDEN
            };
            let mut shift = [0.0f32; 2];
            if params.icon_text_fit && params.variable_placement {
                // Fitted icons move with the text drawn for them.
                let text = (0..4).find(|&slot| {
                    let ti = instance.text[slot];
                    ti != NONE
                        && t.text_placed[ti as usize].associated_icon == index
                        && states.text[ti as usize * state::WORDS] & state::HIDDEN == 0
                });
                match text {
                    Some(slot) => {
                        let ti = instance.text[slot] as usize * state::WORDS;
                        shift = [
                            f32::from_bits(states.text[ti + 1]),
                            f32::from_bits(states.text[ti + 2]),
                        ];
                        word |= state::SHIFTED | (slot as u32) << state::TEXT_SLOT_SHIFT;
                        if slot == justification::VERTICAL {
                            word |= state::QUARTER_TURN;
                        }
                    }
                    None => word |= state::HIDDEN,
                }
            }
            let at = index as usize * state::WORDS;
            states.icon[at..at + 3].copy_from_slice(&[
                word,
                shift[0].to_bits(),
                shift[1].to_bits(),
            ]);
        }
    }
    if params.draw_by_viewport_y {
        (states.text_order, states.icon_order) = draw_order(t, -camera_bearing);
    }
    states
}

/// GL JS `getSortedSymbolIndexes`: the instances by their anchors rotated
/// by `angle`, top to bottom, ties by feature index descending.
fn viewport_y_order(t: &PlacementTables, angle: f64) -> Vec<usize> {
    let (sin, cos) = (libm::sin(angle), libm::cos(angle));
    let mut order: Vec<usize> = (0..t.instances.len()).collect();
    order.sort_by_key(|&i| {
        let s = &t.instances[i];
        let y = (sin * f64::from(s.anchor[0]) + cos * f64::from(s.anchor[1])).round() as i64;
        (y, std::cmp::Reverse(s.feature))
    });
    order
}

/// GL JS's `featureSortOrder`: each instance's feature, in the order a
/// layer drawn by viewport y (`LayerParams::draw_by_viewport_y`) draws them
/// at a bearing of `bearing` radians, upper ones last. Queries answer a
/// tile's labels in it, upper ones first.
pub fn feature_sort_order(t: &PlacementTables, bearing: f64) -> Vec<u32> {
    (viewport_y_order(t, -bearing).into_iter())
        .map(|i| t.instances[i].feature)
        .collect()
}

/// GL JS `sortFeatures`: quads in their instances' [`viewport_y_order`].
fn draw_order(t: &PlacementTables, angle: f64) -> (Vec<u32>, Vec<u32>) {
    let order = viewport_y_order(t, angle);
    let (mut text, mut icon) = (Vec::new(), Vec::new());
    let push = |out: &mut Vec<u32>, placed: &[PlacedSymbol], index: u32| {
        let [start, count] = placed[index as usize].quads;
        out.extend(start..start + count);
    };
    for i in order {
        let s = &t.instances[i];
        for (slot, &index) in s.text.iter().enumerate() {
            // Justifications sharing a placed symbol draw it once.
            if index != NONE && !s.text[..slot].contains(&index) {
                push(&mut text, &t.text_placed, index);
            }
        }
        for &index in &s.icon {
            if index != NONE {
                push(&mut icon, &t.icon_placed, index);
            }
        }
    }
    (text, icon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::view::TileId;
    use bytemuck::Zeroable;

    /// A camera over tile 0/0/0 at zoom 0: 16 tile units per pixel,
    /// no perspective.
    fn frame(layers: Vec<LayerFrame>, now: f64, fade: f64) -> Frame {
        Frame {
            camera: Camera {
                width: 512.0,
                height: 512.0,
                zoom: 0.0,
                bearing: 0.0,
                pitch: 0.0,
                roll: 0.0,
                camera_to_center_distance: 1.0,
            },
            layers,
            now,
            fade_duration: fade,
            cross_source_collisions: true,
            lng: 0.0,
            ground: None,
            scope: Scope::Viewport,
            stable_labels: false,
        }
    }

    fn view(wrap: i32) -> TileView {
        let s = 1.0 / 4096.0;
        TileView {
            id: TileId {
                z: 0,
                x: 0,
                y: 0,
                wrap,
            },
            matrix: [
                s, 0.0, 0.0, 0.0, 0.0, -s, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0,
            ],
            globe: None,
        }
    }

    fn params(layer: u32) -> LayerParams {
        LayerParams {
            layer,
            follows: None,
            source: 1,
            minzoom: 0.0,
            maxzoom: 0.0,
            flags: SymbolFlags::default(),
            own_flags: 0,
            rotate_with_map: false,
            pitch_with_map: false,
            icon_text_fit: false,
            z_order_viewport_y: false,
            draw_by_viewport_y: false,
            sort_across_tiles: false,
            height_anchor_ground: true,
            text_padding: 2.0,
            text_translate: ([0.0; 2], false),
            icon_translate: ([0.0; 2], false),
            writing_modes: vec![WritingMode::Horizontal],
            variable_placement: false,
        }
    }

    /// A point label per anchor: a 64 × 16 tile-unit text box (4 × 1 px).
    fn labels(anchors: &[(u64, [i16; 2])]) -> PlacementTables {
        let mut t = PlacementTables::default();
        for (i, &(key, anchor)) in anchors.iter().enumerate() {
            t.text_placed.push(PlacedSymbol {
                anchor,
                quads: [i as u32, 1],
                ..PlacedSymbol::zeroed()
            });
            t.instances.push(SymbolInstance {
                key,
                anchor,
                text: [i as u32; 3]
                    .into_iter()
                    .chain([NONE])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                icon: [NONE; 2],
                boxes: [[-32, -8, 32, 8], [0; 4], [0; 4], [0; 4]],
                flags: instance_flags::TEXT_BOX,
                text_box_scale: 1.0,
                ..SymbolInstance::zeroed()
            });
        }
        t
    }

    fn word(states: &BucketStates, i: usize) -> u32 {
        states.text[i * state::WORDS]
    }

    fn shown(w: u32) -> bool {
        w & state::HIDDEN == 0 && w & state::PLACED != 0
    }

    fn tile(bucket: u64, wrap: i32) -> PlacedTile {
        PlacedTile {
            bucket,
            view: view(wrap),
            text_size: ZoomSize { size: 16.0, t: 0.0 },
            translate_t: [0.0; 2],
        }
    }

    /// A tile that leaves the draw list for its parent hands the parent the
    /// labels they share: the parent's copy shows on at full, and the
    /// commit names the child's copy for its host to hide rather than fade
    /// out beside it. A label only the child has fades out as usual.
    #[test]
    fn a_dropped_tile_hands_its_shared_labels_over() {
        let mut placer = Placer::default();
        // The child (1/1/1) has labels 5 and 6; the parent has 5 too, at
        // the same place (the child's origin is the parent's middle).
        placer.ingest(1, labels(&[(5, [0, 0]), (6, [200, 200])]));
        placer.ingest(2, labels(&[(5, [4096, 4096])]));
        let child = PlacedTile {
            view: TileView {
                id: TileId {
                    z: 1,
                    x: 1,
                    y: 1,
                    wrap: 0,
                },
                ..view(0)
            },
            ..tile(1, 0)
        };
        let layer = |tiles| {
            vec![LayerFrame {
                params: params(8),
                tiles,
            }]
        };
        let c = placer.place(&frame(layer(vec![child]), 0.0, 300.0));
        assert!(c.handed_over.is_empty());
        let c = placer.place(&frame(layer(vec![tile(2, 0)]), 1000.0, 300.0));
        assert_eq!(c.handed_over, vec![(1, 0, vec![0])]);
        let parent = &c.changed.iter().find(|(b, _, _)| *b == 2).unwrap().2;
        assert_eq!(
            word(parent, 0) & (state::PLACED | state::OPACITY),
            127 | state::PLACED
        );
    }

    #[test]
    fn world_copies_place_apart() {
        let mut placer = Placer::default();
        placer.ingest(1, labels(&[(1, [4096, 4096])]));
        // Wrap -3 is three worlds (six clip units) to the left: off the
        // grid, so its copy hides while wrap 0's shows. It places first.
        let mut far = tile(1, -3);
        far.view.matrix[12] -= 6.0;
        let c = placer.place(&frame(
            vec![LayerFrame {
                params: params(7),
                tiles: vec![far, tile(1, 0)],
            }],
            0.0,
            0.0,
        ));
        let by_wrap = |w: i32| &c.changed.iter().find(|(_, wrap, _)| *wrap == w).unwrap().2;
        assert!(shown(word(by_wrap(0), 0)));
        assert!(!shown(word(by_wrap(-3), 0)));
    }

    /// Beyond the spec, a label translated its own way collides where it
    /// draws: two labels at one anchor collide, until the second's own
    /// translate, blended toward the next zoom's, carries it clear.
    #[test]
    fn labels_translated_each_their_own_way_collide_where_they_draw() {
        let mut placer = Placer::default();
        let mut t = labels(&[(1, [4096, 4096]), (2, [4096, 4096])]);
        placer.ingest(1, t.clone());
        let place = |placer: &mut Placer, t: f64, now: f64| {
            let c = placer.place(&frame(
                vec![LayerFrame {
                    params: params(7),
                    tiles: vec![PlacedTile {
                        translate_t: [t, 0.0],
                        ..tile(1, 0)
                    }],
                }],
                now,
                0.0,
            ));
            let states = &c.changed[0].2;
            [0, 1].map(|i| shown(word(states, i)))
        };
        assert_eq!(place(&mut placer, 0.0, 0.0), [true, false]);
        // The second moves 0 px at the tile's zoom and 8 px down at the
        // next: its 1 px tall box clears the first's halfway there, not
        // at the tile's zoom.
        t.translates[0] = vec![[[0.0; 2]; 2], [[0.0; 2], [0.0, 8.0]]];
        placer.ingest(1, t);
        assert_eq!(place(&mut placer, 0.0, 10.0), [true, false]);
        assert_eq!(place(&mut placer, 0.5, 20.0), [true, true]);
    }

    #[test]
    fn overlapping_labels_collide_unless_allowed() {
        let mut placer = Placer::default();
        // Labels 2 and 3 overlap label 1; label 4 is clear.
        placer.ingest(
            1,
            labels(&[
                (1, [4096, 4096]),
                (2, [4100, 4100]),
                (3, [4140, 4096]),
                (4, [6000, 6000]),
            ]),
        );
        let c = placer.place(&frame(
            vec![LayerFrame {
                params: params(7),
                tiles: vec![tile(1, 0)],
            }],
            0.0,
            0.0,
        ));
        let states = &c.changed[0].2;
        let placed: Vec<bool> = (0..4).map(|i| shown(word(states, i))).collect();
        assert_eq!(placed, [true, false, false, true]);
        // Without fades, labels show in full at once.
        assert_eq!(c.fade_change(0.0), 1.0);

        let mut overlap = params(7);
        overlap.flags.text_overlap = Overlap::Always;
        let c = placer.place(&frame(
            vec![LayerFrame {
                params: overlap,
                tiles: vec![tile(1, 0)],
            }],
            10.0,
            0.0,
        ));
        let states = &c.changed[0].2;
        assert!((0..4).all(|i| shown(word(states, i))));
    }

    #[test]
    fn upper_layers_place_first_and_duplicates_draw_once() {
        let mut placer = Placer::default();
        placer.ingest(1, labels(&[(1, [4096, 4096])]));
        placer.ingest(2, labels(&[(2, [4100, 4096])]));
        // Layer 9 is above layer 8 in style order: it wins the space.
        let c = placer.place(&frame(
            vec![
                LayerFrame {
                    params: params(8),
                    tiles: vec![tile(1, 0)],
                },
                LayerFrame {
                    params: params(9),
                    tiles: vec![tile(2, 0)],
                },
            ],
            0.0,
            0.0,
        ));
        let by = |b: u64| &c.changed.iter().find(|(id, _, _)| *id == b).unwrap().2;
        assert!(!shown(word(by(1), 0)));
        assert!(shown(word(by(2), 0)));

        // The same label in a parent and a child tile is drawn once.
        let mut placer = Placer::default();
        placer.ingest(1, labels(&[(5, [4096, 4096])]));
        placer.ingest(2, labels(&[(5, [0, 0])]));
        let child = PlacedTile {
            view: TileView {
                id: TileId {
                    z: 1,
                    x: 1,
                    y: 1,
                    wrap: 0,
                },
                ..view(0)
            },
            ..tile(2, 0)
        };
        let c = placer.place(&frame(
            vec![LayerFrame {
                params: params(8),
                tiles: vec![tile(1, 0), child],
            }],
            0.0,
            0.0,
        ));
        let shown_count = c
            .changed
            .iter()
            .filter(|(_, _, s)| shown(word(s, 0)))
            .count();
        assert_eq!(shown_count, 1);
    }

    #[test]
    fn layers_that_lay_out_alike_place_together() {
        // Layer 9 lays out what layer 8 does (a second label colour, say):
        // its labels show with layer 8's rather than colliding with them,
        // and the one layer 10 above hides in layer 8 hides in layer 9.
        let mut placer = Placer::default();
        placer.ingest(1, labels(&[(1, [4096, 4096]), (2, [6000, 6000])]));
        placer.ingest(2, labels(&[(3, [4096, 4096]), (4, [6000, 6000])]));
        placer.ingest(3, labels(&[(5, [6004, 6000])]));
        let c = placer.place(&frame(
            vec![
                LayerFrame {
                    params: params(8),
                    tiles: vec![tile(1, 0)],
                },
                LayerFrame {
                    params: LayerParams {
                        follows: Some(8),
                        ..params(9)
                    },
                    tiles: vec![tile(2, 0)],
                },
                LayerFrame {
                    params: params(10),
                    tiles: vec![tile(3, 0)],
                },
            ],
            0.0,
            0.0,
        ));
        let by = |b: u64| &c.changed.iter().find(|(id, _, _)| *id == b).unwrap().2;
        assert!(shown(word(by(1), 0)) && shown(word(by(2), 0)));
        assert!(!shown(word(by(1), 1)) && !shown(word(by(2), 1)));
    }

    #[test]
    fn fades_follow_the_clock() {
        let mut placer = Placer::default();
        placer.ingest(1, labels(&[(1, [4096, 4096]), (2, [4100, 4100])]));
        let layer = |p| {
            vec![LayerFrame {
                params: p,
                tiles: vec![tile(1, 0)],
            }]
        };
        // A tile's first labels fade in from 0.
        let c = placer.place(&frame(layer(params(7)), 0.0, 300.0));
        let w = word(&c.changed[0].2, 0);
        assert_eq!((w & state::OPACITY, w & state::PLACED != 0), (0, true));
        assert_eq!(c.fade_change(150.0), 0.5);
        // Allowing overlap places label 2, which fades in from 0.
        let mut overlap = params(7);
        overlap.flags.text_overlap = Overlap::Always;
        let c = placer.place(&frame(layer(overlap.clone()), 1000.0, 300.0));
        let w = word(&c.changed[0].2, 1);
        assert_eq!((w & state::OPACITY, w & state::PLACED != 0), (0, true));
        // Half a fade later it is half in.
        let c = placer.place(&frame(layer(overlap), 1150.0, 300.0));
        assert_eq!(word(&c.changed[0].2, 1) & state::OPACITY, 63);
    }

    /// A fade committed partway is stored rounded down, so it runs on past
    /// its change's time plus the duration, until it has run a whole fade
    /// since that commit: idle frames show every label at its end, whenever
    /// the last commit fell.
    #[test]
    fn fades_committed_partway_run_to_their_end() {
        let mut placer = Placer::default();
        placer.ingest(1, labels(&[(1, [4096, 4096])]));
        let layer = || {
            vec![LayerFrame {
                params: params(7),
                tiles: vec![tile(1, 0)],
            }]
        };
        placer.place(&frame(layer(), 0.0, 300.0));
        // Almost in at the next commit: 0.997, stored as 126/127.
        let c = placer.place(&frame(layer(), 299.0, 300.0));
        assert_eq!(word(&c.changed[0].2, 0) & state::OPACITY, 126);
        assert!(c.partway);
        // The change began at 0, but the stored opacity reaches 1 only a
        // fade after the commit.
        assert!(c.fading(300.0));
        assert!(c.fading(598.0));
        assert!(!c.fading(599.0));
        // Fully in: nothing waits.
        let c = placer.place(&frame(layer(), 400.0, 300.0));
        assert!(!c.partway);
        assert!(!c.fading(400.0));
    }

    #[test]
    fn variable_anchors_fall_back_and_shift() {
        let mut t = labels(&[(1, [4096, 4096]), (2, [4096, 4112])]);
        for (i, instance) in t.instances.iter_mut().enumerate() {
            let start = t.anchor_offsets.len() as u32;
            // Label 2 would sit on label 1 centred; `top` moves it down.
            for anchor in [1, 4] {
                t.anchor_offsets.push(AnchorOffset {
                    anchor,
                    offset: [0.0; 2],
                });
            }
            instance.anchor_offsets = [start, start + 2];
            let _ = i;
        }
        let mut placer = Placer::default();
        placer.ingest(1, t);
        let mut p = params(7);
        p.variable_placement = true;
        let c = placer.place(&frame(
            vec![LayerFrame {
                params: p,
                tiles: vec![tile(1, 0)],
            }],
            0.0,
            0.0,
        ));
        let s = &c.changed[0].2;
        assert!(shown(word(s, 0)) && shown(word(s, 1)));
        assert_ne!(word(s, 1) & state::SHIFTED, 0);
        // `top` shifts down by half the box height: 8 units of scale 1.
        assert_eq!(f32::from_bits(s.text[state::WORDS + 2]), 8.0);
        assert_eq!(f32::from_bits(s.text[1]), 0.0);
    }

    #[test]
    fn line_labels_collide_as_circles() {
        // Two labels on one horizontal line, 62.5 px apart; each spans
        // glyphs from -60 to +60 layout px at 24 px.
        let mut t = PlacementTables {
            lines: vec![[0, 4096], [8192, 4096]],
            text_size: SizeKind::Constant,
            ..PlacementTables::default()
        };
        for (i, x) in [3000i16, 4000].into_iter().enumerate() {
            t.text_placed.push(PlacedSymbol {
                anchor: [x, 4096],
                segment: 0,
                quads: [i as u32 * 2, 2],
                line: [0, 2],
                ..PlacedSymbol::zeroed()
            });
            t.text_glyph_offsets.extend([-60.0, 60.0]);
            t.instances.push(SymbolInstance {
                key: i as u64,
                anchor: [x, 4096],
                text: [NONE, i as u32, NONE, NONE],
                icon: [NONE; 2],
                collision_circle_diameter: 16.0,
                flags: instance_flags::RUNTIME_COLLISION_CIRCLES,
                text_box_scale: 1.0,
                ..SymbolInstance::zeroed()
            });
        }
        let mut placer = Placer::default();
        placer.ingest(1, t.clone());
        let mut p = params(7);
        p.rotate_with_map = true;
        let size = |size| PlacedTile {
            text_size: ZoomSize { size, t: 0.0 },
            ..tile(1, 0)
        };
        let c = placer.place(&frame(
            vec![LayerFrame {
                params: p.clone(),
                tiles: vec![size(24.0)],
            }],
            0.0,
            0.0,
        ));
        let s = &c.changed[0].2;
        assert!(shown(word(s, 0)));
        assert!(
            !shown(word(s, 1)),
            "the second label's circles hit the first's"
        );
        // At a quarter of the size their circles (radius 10 px) no longer
        // reach each other.
        let mut placer = Placer::default();
        placer.ingest(1, t);
        let c = placer.place(&frame(
            vec![LayerFrame {
                params: p,
                tiles: vec![size(6.0)],
            }],
            0.0,
            0.0,
        ));
        let s = &c.changed[0].2;
        assert!(shown(word(s, 0)) && shown(word(s, 1)));
    }

    /// Tile 0/0/0 as seen from a 256-pixel view of its quadrant `(col,
    /// row)`, as tile scope places it.
    fn quadrant(col: i64, row: i64, layers: &[(LayerParams, u64)]) -> ScopeTile {
        let s = 1.0 / 2048.0;
        let (c, r) = (col as f64, row as f64);
        ScopeTile {
            col,
            row,
            camera: Camera {
                width: 256.0,
                height: 256.0,
                ..frame(Vec::new(), 0.0, 0.0).camera
            },
            layers: layers
                .iter()
                .map(|(params, bucket)| LayerFrame {
                    params: params.clone(),
                    tiles: vec![PlacedTile {
                        view: TileView {
                            matrix: [
                                s,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                -s,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                0.0,
                                -1.0 - 2.0 * c,
                                1.0 + 2.0 * r,
                                0.0,
                                1.0,
                            ],
                            ..view(0)
                        },
                        ..tile(*bucket, 0)
                    }],
                })
                .collect(),
        }
    }

    fn tile_scope(placer: &mut Placer, owners: &[(i64, i64)]) -> BucketStates {
        let layers = [(params(7), 1)];
        let mut f = frame(
            vec![LayerFrame {
                params: params(7),
                tiles: vec![tile(1, 0)],
            }],
            0.0,
            0.0,
        );
        f.scope = Scope::Tiles(
            owners
                .iter()
                .map(|&(c, r)| quadrant(c, r, &layers))
                .collect(),
        );
        placer.place(&f).changed.remove(0).2
    }

    #[test]
    fn tile_scope_agrees_across_tiles() {
        let mut placer = Placer::default();
        // Label 1 is anchored in quadrant (0, 0) and crosses into (1, 0),
        // where label 2, anchored just across the edge, overlaps it. Label 3
        // lies alone in (1, 1).
        placer.ingest(
            1,
            labels(&[(1, [4090, 2000]), (2, [4100, 2000]), (3, [6000, 6000])]),
        );
        let all = tile_scope(&mut placer, &[(0, 0), (1, 0), (0, 1), (1, 1)]);
        let placed = |states: &BucketStates| -> Vec<bool> {
            (0..3).map(|i| shown(word(states, i))).collect()
        };
        // Each quadrant places its own; the first in row-major order wins
        // across the edge.
        assert_eq!(placed(&all), [true, false, true]);
        // Drawing either side's neighbourhood decides the edge alike, and
        // again the same.
        let left = tile_scope(&mut placer, &[(0, 0), (1, 0)]);
        assert_eq!(placed(&left)[..2], [true, false]);
        // In any order.
        let right = tile_scope(&mut placer, &[(1, 1), (1, 0), (0, 0)]);
        assert_eq!(placed(&right), [true, false, true]);
        assert_eq!(
            tile_scope(&mut placer, &[(0, 0), (1, 0), (0, 1), (1, 1)]),
            all
        );
        // A label no listed tile holds isn't placed.
        assert!(!placed(&left)[2]);
    }

    #[test]
    fn still_scope_places_as_if_first() {
        let mut placer = Placer::default();
        placer.ingest(1, labels(&[(1, [4096, 4096]), (2, [4100, 4100])]));
        let layer = |overlap| {
            let mut p = params(7);
            p.flags.text_overlap = overlap;
            vec![LayerFrame {
                params: p,
                tiles: vec![tile(1, 0)],
            }]
        };
        // A fading placement, then a still one: it shows fully at once.
        placer.place(&frame(layer(Overlap::Always), 0.0, 300.0));
        let mut f = frame(layer(Overlap::Never), 10.0, 0.0);
        f.scope = Scope::Still;
        let c = placer.place(&f);
        let states = &c.changed[0].2;
        assert!(shown(word(states, 0)));
        assert_eq!(c.fade_change(10.0), 1.0);
        assert!(!shown(word(states, 1)));
    }

    #[test]
    fn reloaded_buckets_show_at_once() {
        let mut placer = Placer::default();
        let mut t = labels(&[(1, [4096, 4096])]);
        t.reloaded = true;
        placer.ingest(1, t);
        let layer = vec![LayerFrame {
            params: params(7),
            tiles: vec![tile(1, 0)],
        }];
        let c = placer.place(&frame(layer, 0.0, 300.0));
        assert_eq!(word(&c.changed[0].2, 0) & state::OPACITY, 127);
    }

    /// Where label 2 stands after two placements: centred or below its
    /// anchor (`top`), with label 1 of an upper layer pushing it below in
    /// the first, then at `block` (or gone) in the second. `None`: hidden.
    fn second_anchor(stable: bool, pitch: f64, block: Option<i16>) -> Option<f32> {
        let mut t = labels(&[(2, [4096, 4112])]);
        t.anchor_offsets = [1, 4]
            .map(|anchor| AnchorOffset {
                anchor,
                offset: [0.0; 2],
            })
            .to_vec();
        t.instances[0].anchor_offsets = [0, 2];
        let mut placer = Placer::default();
        placer.ingest(1, t);
        placer.ingest(2, labels(&[(1, [4096, 4096])]));
        if let Some(y) = block {
            placer.ingest(3, labels(&[(3, [4096, y])]));
        }
        let mut p = params(7);
        p.variable_placement = true;
        let place = |placer: &mut Placer, upper: u64, now: f64| {
            let mut f = frame(
                vec![
                    LayerFrame {
                        params: p.clone(),
                        tiles: vec![tile(1, 0)],
                    },
                    LayerFrame {
                        params: params(8),
                        tiles: vec![tile(upper, 0)],
                    },
                ],
                now,
                300.0,
            );
            f.camera.pitch = pitch;
            f.stable_labels = stable;
            placer.place(&f);
        };
        place(&mut placer, 2, 0.0);
        let states = &placer.buckets[&(1, 0)].states;
        assert_eq!(f32::from_bits(states.text[2]), 8.0, "pushed below");
        place(&mut placer, if block.is_some() { 3 } else { 0 }, 1000.0);
        let states = &placer.buckets[&(1, 0)].states;
        shown(word(states, 0)).then(|| f32::from_bits(states.text[2]))
    }

    #[test]
    fn stable_labels_try_their_previous_anchor_first() {
        // Label 1 leaves: a stable label stays below, others centre again.
        assert_eq!(second_anchor(false, 0.0, None), Some(0.0));
        assert_eq!(second_anchor(true, 0.0, None), Some(8.0));
        assert_eq!(second_anchor(true, 0.5, None), Some(8.0));
        // Something now stands below: untilted, a stable label falls back
        // to the next anchor; tilted, MapLibre Native tries the previous
        // anchor alone.
        assert_eq!(second_anchor(false, 0.5, Some(4136)), Some(0.0));
        assert_eq!(second_anchor(true, 0.0, Some(4136)), Some(0.0));
        assert_eq!(second_anchor(true, 0.5, Some(4136)), None);
    }

    #[test]
    fn stable_labels_place_shown_ones_first_in_tilted_views() {
        // Labels X and Y of one bucket overlap; label Z of an upper layer
        // overlaps X alone. With Z, Y shows; without, X wins in bucket
        // order unless, in a tilted view, Y shown before goes first.
        let shown_after = |stable: bool, pitch: f64| {
            let mut placer = Placer::default();
            placer.ingest(1, labels(&[(1, [4096, 4096]), (2, [4140, 4096])]));
            placer.ingest(2, labels(&[(3, [4050, 4096])]));
            let mut both = frame(
                vec![
                    LayerFrame {
                        params: params(7),
                        tiles: vec![tile(1, 0)],
                    },
                    LayerFrame {
                        params: params(8),
                        tiles: vec![tile(2, 0)],
                    },
                ],
                0.0,
                300.0,
            );
            both.camera.pitch = pitch;
            both.stable_labels = stable;
            placer.place(&both);
            let states = &placer.buckets[&(1, 0)].states;
            assert!(!shown(word(states, 0)) && shown(word(states, 1)));
            let mut alone = both.clone();
            alone.layers.pop();
            alone.now = 1000.0;
            placer.place(&alone);
            let states = &placer.buckets[&(1, 0)].states;
            [0, 1].map(|i| shown(word(states, i)))
        };
        assert_eq!(shown_after(false, 0.5), [true, false]);
        assert_eq!(shown_after(true, 0.0), [true, false]);
        assert_eq!(shown_after(true, 0.5), [false, true]);
    }
}

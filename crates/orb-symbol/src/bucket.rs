//! What symbol layout produces: plain tables of `Pod` rows, grouped by the
//! lane that reads them (decision 0009, plan §10.4).
//!
//! - The placement lane reads [`SymbolInstance`]s (anchors, collision boxes,
//!   circle diameters, keys), [`PlacedSymbol`]s, line vertices, glyph
//!   offsets and anchor offsets. It never sees text or textures.
//! - The GPU symbol pass reads [`PlacedSymbol`]s, [`Quad`]s, glyph offsets
//!   and line vertices: one instance per quad, one state word per placed
//!   symbol, one compute workgroup per placed line label.
//!
//! Every table is one contiguous `Vec` of fixed-size rows, so the web copies
//! each once from a job worker into a transferable buffer. Rows hold indices,
//! never pointers. Textures are tile-local glyph ids and image indices; atlas
//! coordinates never enter the data.

use bytemuck::{Pod, Zeroable};
use orb_text::Anchor;

/// An absent row index.
pub const NONE: u32 = u32::MAX;

/// GL JS `WritingMode` as a placed symbol records it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SymbolWritingMode {
    /// Icons.
    None = 0,
    /// Text that also has a vertical shaping.
    Horizontal = 1,
    /// Upright glyphs laid out along x, each quad turned a quarter turn
    /// back: point labels draw the whole symbol turned a quarter turn
    /// clockwise (GL JS's `PI / 2` dynamic angle), line labels along the
    /// line.
    Vertical = 2,
    /// Text with no vertical alternative.
    HorizontalOnly = 3,
}

impl SymbolWritingMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Horizontal,
            2 => Self::Vertical,
            3 => Self::HorizontalOnly,
            _ => Self::None,
        }
    }
}

/// Bits of [`SymbolInstance::flags`]: which collision features exist.
pub mod instance_flags {
    pub const TEXT_BOX: u32 = 1 << 0;
    pub const VERTICAL_TEXT_BOX: u32 = 1 << 1;
    pub const ICON_BOX: u32 = 1 << 2;
    pub const VERTICAL_ICON_BOX: u32 = 1 << 3;
    /// Line labels collide as circles along the line, sized by
    /// [`super::SymbolInstance::collision_circle_diameter`].
    pub const RUNTIME_COLLISION_CIRCLES: u32 = 1 << 4;
}

/// Index of a justification in [`SymbolInstance::text`].
pub mod justification {
    pub const RIGHT: usize = 0;
    pub const CENTER: usize = 1;
    pub const LEFT: usize = 2;
    pub const VERTICAL: usize = 3;
}

/// One possible placement of a feature's label and icon at one anchor (GL
/// JS `SymbolInstance`). Placement shows or hides it as a whole.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct SymbolInstance {
    /// Identity across tiles and zooms, for cross-tile matching.
    pub key: u64,
    /// Tile units, truncated as GL JS's `Int16` columns are.
    pub anchor: [i16; 2],
    /// The feature's index in the tile's source layer.
    pub feature: u32,
    /// Tile units per layout pixel of text, at the tile's zoom + 1.
    pub text_box_scale: f32,
    /// In pixels of `text-size` at zoom + 1; meaningful with
    /// [`instance_flags::RUNTIME_COLLISION_CIRCLES`].
    pub collision_circle_diameter: f32,
    /// `symbol-height-offset`, meters.
    pub height_offset: f32,
    /// Text placed symbols by [`justification`]: right, center, left and
    /// vertical. Justifications of a single-line label share one.
    pub text: [u32; 4],
    /// The icon placed symbol, and its vertical-text variant.
    pub icon: [u32; 2],
    /// Collision boxes `x1, y1, x2, y2` around the anchor in tile units:
    /// text, vertical text, icon, vertical icon. Present per
    /// [`instance_flags`].
    pub boxes: [[i16; 4]; 4],
    /// Range of this instance's rows in [`SymbolBucket::anchor_offsets`].
    pub anchor_offsets: [u32; 2],
    pub flags: u32,
}

/// One drawn variant of a label or icon: a run of quads around an anchor,
/// possibly along a line (GL JS `PlacedSymbol`).
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct PlacedSymbol {
    pub anchor: [i16; 2],
    /// The line segment the anchor is on; 0 off lines.
    pub segment: u16,
    /// A [`SymbolWritingMode`].
    pub writing_mode: u8,
    pub _pad: u8,
    /// First quad and quad count in the stream.
    pub quads: [u32; 2],
    /// First vertex and vertex count in [`SymbolBucket::lines`]; empty for
    /// symbols not along a line.
    pub line: [u32; 2],
    /// `*-size` × 128, rounded and capped as GL JS's vertex `a_size`:
    /// `[size, 0]` for source sizes, `[lower, upper]` for composite sizes.
    pub size: [u16; 2],
    /// `text-offset` or `icon-offset`, layout units.
    pub line_offset: [f32; 2],
    /// The [`SymbolInstance`] this belongs to.
    pub instance: u32,
    /// For text: the icon placed symbol that moves with it ([`NONE`] if none).
    pub associated_icon: u32,
}

/// Bits of [`Quad::flags`].
pub mod quad_flags {
    /// The texture is a signed distance field.
    pub const SDF: u16 = 1 << 0;
    /// [`super::Quad::texture`] is a tile-local image index, not a glyph id.
    pub const IMAGE: u16 = 1 << 1;
    /// Its `format` section overrides `text-color` (GL JS
    /// `FormatSectionOverride`). Layout never sets it: the runtime, which
    /// evaluates paint, does, with the colour in a table beside the quads.
    pub const COLOR: u16 = 1 << 2;
}

/// A glyph or icon quad, one GPU instance (GL JS's four `a_pos_offset`,
/// `a_data` and `a_pixeloffset` vertices, quantized identically).
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Quad {
    /// Corner offsets from the anchor, `tl, tr, bl, br`, × 32 in layout
    /// pixels at 24 px text (or the icon's size 1). A line label's glyph
    /// offset `y` is included.
    pub corners: [[i16; 2]; 4],
    /// Texels `x0, y0, x1, y1` relative to the padded glyph or image
    /// rectangle (one texel of padding around every bitmap).
    pub tex: [u16; 4],
    /// Stretchable-icon pixel offsets × 16: `tl.x, tl.y, br.x, br.y`.
    pub pixel_offsets: [i16; 4],
    /// Minimum font scale of stretched icons × 256, `x, y`.
    pub min_font_scale: [i16; 2],
    /// A tile-local glyph id ([`orb_text::GlyphId`]), or an image index
    /// with [`quad_flags::IMAGE`].
    pub texture: u32,
    /// The [`PlacedSymbol`] it belongs to.
    pub symbol: u32,
    /// The label section it draws, for `format` paint overrides.
    pub section: u16,
    pub flags: u16,
}

/// The placed symbols and quads of one kind: text or icons.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymbolStream {
    pub placed: Vec<PlacedSymbol>,
    pub quads: Vec<Quad>,
    /// Per quad: the glyph's offset along the line from the anchor, layout
    /// units (GL JS `glyphOffsetArray`); 0 for labels not along a line.
    pub glyph_offsets: Vec<f32>,
}

/// One `text-variable-anchor` candidate of an instance.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct AnchorOffset {
    /// GL JS `TextAnchorEnum`: 1 center, 2 left, 3 right, 4 top, 5 bottom,
    /// 6 top-left, 7 top-right, 8 bottom-left, 9 bottom-right.
    pub anchor: u32,
    /// Layout units.
    pub offset: [f32; 2],
}

impl AnchorOffset {
    pub fn text_anchor(&self) -> Anchor {
        match self.anchor {
            2 => Anchor::Left,
            3 => Anchor::Right,
            4 => Anchor::Top,
            5 => Anchor::Bottom,
            6 => Anchor::TopLeft,
            7 => Anchor::TopRight,
            8 => Anchor::BottomLeft,
            9 => Anchor::BottomRight,
            _ => Anchor::Center,
        }
    }
}

/// Instances sharing a `symbol-sort-key`, in key order.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct SortKeyRange {
    pub sort_key: f64,
    pub start: u32,
    pub end: u32,
}

/// A size property as the renderer evaluates it per frame (GL JS
/// `SizeData`, without the expression the scene side already has).
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SizeKind {
    /// Constant; every feature has the same size.
    #[default]
    Constant,
    /// Per feature, constant over zoom; placed symbols carry the size.
    Source,
    /// Per zoom, the same for every feature.
    Camera,
    /// Per feature and zoom: placed symbols carry the size at these two
    /// zoom stops, which cover the tile's zoom (GL JS
    /// `getCoveringZoomStops`).
    Composite { zooms: [f64; 2] },
}

/// Style-sheet problems layout ran into, for the runtime to report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Warnings {
    /// Icons of one layer mix SDF and non-SDF images.
    pub mixed_sdf_icons: bool,
    /// A `text-size` reached 255 px, the packed maximum.
    pub text_size_too_large: bool,
    /// An `icon-size` reached 255, the packed maximum.
    pub icon_size_too_large: bool,
}

/// One symbol layer's layout of one tile (GL JS `SymbolBucket`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymbolBucket {
    pub instances: Vec<SymbolInstance>,
    pub text: SymbolStream,
    pub icon: SymbolStream,
    /// Line vertices labels ride on, tile units. Instances on one line share
    /// one copy.
    pub lines: Vec<[i16; 2]>,
    pub anchor_offsets: Vec<AnchorOffset>,
    /// Empty unless the layer sorts by `symbol-sort-key`.
    pub sort_key_ranges: Vec<SortKeyRange>,
    pub text_size: SizeKind,
    pub icon_size: SizeKind,
    /// Tile units per layout pixel (GL JS `tilePixelRatio`).
    pub tile_pixel_ratio: f64,
    /// Whether the layer's icons are SDFs; `None` with no icons.
    pub sdf_icons: Option<bool>,
    /// Some label has inline images.
    pub icons_in_text: bool,
    /// Icons need linear filtering: scaled from another pixel ratio, or
    /// rotated.
    pub icons_need_linear: bool,
    /// `text-writing-mode` allows vertical point labels.
    pub allow_vertical_placement: bool,
    /// The largest `symbol-height-offset`, meters, at least 0.
    pub max_height_offset: f64,
    pub warnings: Warnings,
}

impl SymbolBucket {
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

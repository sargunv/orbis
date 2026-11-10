//! The GPU scene: persistent tables a frame reads, the arenas
//! that hold tile streams, and the bookkeeping that keeps them consistent.
//!
//! A camera-only frame writes the frame block and the tile rows; everything
//! else changes only when meshes arrive or leave. Row layouts match the
//! structs in `orb-shaders`' `types.wgsl` and `cull.wgsl`.

mod arena;
mod candidates;
mod glyph;
mod image;
mod line;
mod pages;
mod shelf;
mod symbol;
mod tables;
pub(crate) use candidates::Move;
#[cfg(cpu_driven)]
pub(crate) use candidates::{MOVED, NO_CHUNK};
pub use glyph::{GLYPH_PADDING, GLYPH_PAGE, GLYPH_ROW_WORDS, GlyphAtlas};
pub use image::{ImageAtlas, ImageBindings, ImageKey, ImageMetadata, ImageUpload};
#[cfg(cpu_driven)]
pub(crate) use tables::data_width;
pub(crate) use tables::{Backing, DataTable, Table, TableWriter, Tables};

use std::ops::Range;

use crate::driver::{self, SubmitLimits};
use arena::{Arena, Slots};
use orb_hal::{BufferDesc, BufferId, BufferUsages, Gpu, HostBytesId};
use orb_layout::Chunk;
use orb_layout::Mesh;
use orb_shaders::layout;
pub use pages::Pixels;
pub(crate) use pages::{PageTexture, TilePage};
pub use symbol::{
    PLACED_WORDS, QUAD_WORDS, SYMBOL_HEADER_WORDS, SYMBOL_STATE_WORDS, SymbolMeshData,
    pack_image_rect, symbol_header,
};

fn state_base(range: &Range<u32>) -> u32 {
    if range.is_empty() {
        NO_STATE
    } else {
        range.start
    }
}

fn write_chunk_word(gpu: &mut dyn Gpu, tables: &mut Tables, chunk: u32, word: u64, value: u32) {
    let at = u64::from(chunk) * CHUNK_WORDS as u64 + word;
    tables.write_words(gpu, Table::Chunks, at as u32, &[value]);
}

/// Per-frame values every pipeline reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameBlock {
    /// Camera-relative world → clip, column-major. In f64: reversed-Z
    /// depth is the difference of two of its rows.
    pub view_proj: [[f64; 4]; 4],
    /// Where the camera is in the world: tile rows' origins less it are
    /// camera-relative.
    pub camera: [f64; 2],
    /// How many world units the surface spans, which sets the precision
    /// positions split at (`quantum`).
    pub world_span: f64,
    /// Logical pixels.
    pub viewport: [f32; 2],
    /// World units per logical pixel at the camera target.
    pub world_per_px: f32,
    /// Physical pixels per logical pixel.
    pub pixel_ratio: f32,
    /// Style zoom of the camera.
    pub zoom: f64,
    /// Clockwise map rotation, in radians.
    pub bearing: f32,
    /// A camera-relative world offset that puts image tiles on the pixel
    /// grid, as GL JS's aligned projection matrix does.
    pub align: [f32; 2],
    /// Projected world units per v8 extrusion metre at the camera target.
    pub extrusion_scale: f32,
    /// How far symbol fades have run since the last placement commit (GL
    /// JS `symbolFadeChange`).
    pub symbol_fade_change: f32,
    /// Camera pitch, radians.
    pub pitch: f32,
    /// Camera roll about the view axis, radians.
    pub roll: f32,
    /// The camera target's clip w (orb-camera's `View::camera_to_center`).
    pub camera_to_center: f32,
    /// Milliseconds since the map began, which image fades measure from
    /// (their rows hold when each tile arrived).
    pub time: f32,
    /// The same, which feature-state fades measure from: a drape slot
    /// takes the frame's, where its `time` holds image fades at their end.
    pub state_time: f32,
    /// The terrain elevated content stands on, if any.
    pub terrain: Option<FrameTerrain>,
    /// v8's `light`.
    pub light: FrameLight,
    /// The view's globe, when it is one at its zoom.
    pub globe: Option<FrameGlobe>,
}

/// v8's `light` as shaders take it (`lighting.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameLight {
    /// Premultiplied.
    pub color: [f32; 3],
    pub intensity: f32,
    /// Its position as GL JS's `sphericalToCartesian` gives it: tile axes
    /// (y south) with the horizontal parts negated, as long as v8's radial
    /// position. Viewport anchoring is already applied.
    pub position: [f32; 3],
    /// style-spec#1797: how much extrusions' shadows darken the ground
    /// beneath them, 0 to 1, cast away from `position`.
    pub shadow_intensity: f32,
    /// maplibre/maplibre#555's ambient light on extrusions, premultiplied,
    /// and how far they take it (0 to 1) rather than GL JS's `1 −
    /// intensity` of `color`: with it, `intensity` is the sun's alone,
    /// ambient occlusion dims only the ambient light and shadows only the
    /// sun's.
    pub ambient: [f32; 3],
    pub ambient_amount: f32,
    /// Its shading of the ground (`dev.sargunv.orbis:ground-lighting`),
    /// which labels' `["dev.sargunv.orbis:daylight"]` follows.
    pub ground: Option<crate::GroundLight>,
}

impl FrameLight {
    /// The longest a shadow gets, in heights: about a sun 6° up's.
    const LONGEST_SHADOW: f32 = 10.0;

    /// Where a point a metre up falls on the ground along the light, in
    /// metres east and north, and how dark extrusions' shadows are. As the
    /// light sinks to 3° above the horizon its shadows fade out rather
    /// than streak across the map, and with it below there are none.
    pub fn shadow(&self) -> ([f32; 2], f32) {
        let [x, y, z] = self.position;
        let radius = (x * x + y * y + z * z).sqrt();
        if self.shadow_intensity <= 0.0 || z <= 0.0 {
            return ([0.0; 2], 0.0);
        }
        let (low, high) = (3f32.to_radians().sin(), 6f32.to_radians().sin());
        let fade = ((z / radius - low) / (high - low)).clamp(0.0, 1.0);
        let reach = (x.hypot(y) / z).min(Self::LONGEST_SHADOW);
        let along = [x, -y].map(|v| v / x.hypot(y).max(f32::MIN_POSITIVE) * reach);
        // With an ambient light a shadow takes away the sun's share of the
        // light on the ground (`lighting.wgsl`'s `light_ground_ambient`).
        let sun = 1.0 - self.ground_ambient_share();
        let share = 1.0 + (sun - 1.0) * self.ambient_amount;
        (along, self.shadow_intensity * fade * share)
    }

    /// The ambient share of the light on flat ground, as `lighting.wgsl`'s
    /// `light_ground_ambient` gives it before the ambient amount.
    fn ground_ambient_share(&self) -> f32 {
        let luminance = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let i = self.intensity;
        let ambient: [f32; 3] = std::array::from_fn(|k| {
            let gl = (1.0 - i) * self.color[k];
            gl + (self.ambient[k] - gl) * self.ambient_amount
        });
        let facing = self.position[2].clamp(0.0, 1.0);
        let total: [f32; 3] = std::array::from_fn(|k| ambient[k] + facing * i * self.color[k]);
        luminance(ambient) / luminance(total).max(1e-6)
    }
}

/// A low-zoom surface's terms (`project.wgsl`), as a globe's; positions on the sphere are
/// camera-relative ECEF metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameGlobe {
    /// Camera-relative globe space → clip, column-major.
    pub view_proj: [[f32; 4]; 4],
    /// The target's ECEF position: rows' anchors less it are
    /// camera-relative.
    pub target: [f64; 3],
    /// How much of a globe the view is: clip positions mix the plane's
    /// (0) and the globe's (1).
    pub amount: f32,
    /// The horizon plane as `(n, w)`: a camera-relative position `p` on
    /// the sphere faces the eye where `dot(n, p) + w >= 0`, rising to 1
    /// under the eye. Depth comes from it, as in GL JS.
    pub horizon: [f32; 4],
    /// Metres on the globe per plane world unit at the target: offsets and
    /// heights in world units scale by it.
    pub metres_per_unit: f32,
    /// The eye, camera-relative: 3D content behind the sphere hides.
    pub eye: [f32; 3],
    /// The low-zoom surface is a plane, not the sphere: positions are its
    /// world units, z up, with no horizon.
    pub plane: bool,
}

/// What elevated content reads of the terrain's DEM page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameTerrain {
    /// GL JS's unpack vector.
    pub unpack: [f32; 4],
    pub exaggeration: f32,
    /// DEM pixels per side, without borders.
    pub dem_dim: f32,
    /// The ground's elevation under the camera target, metres: the plane
    /// the ground mask lies on (`project.wgsl`'s `ground_mask_clip`).
    pub ground: f32,
    /// The lowest elevation the frame's terrain tiles can show, metres
    /// with exaggeration: on a globe, ground below sea level near the
    /// limb stays in front of the sphere that hides solid points
    /// (`project.wgsl`'s `sphere_hides`).
    pub lowest: f32,
}

const FRAME_WORDS: usize = layout::frame::WORDS;

/// Bytes of a frame block's uniform buffer.
pub(crate) const FRAME_BYTES: u64 = FRAME_WORDS as u64 * 4;

/// The power of two positions on a surface `span` world units across
/// split at: a whole number of quanta (an i32, room for 64 spans) and an
/// f32 remainder under half a quantum. Shaders subtract the camera's whole
/// quanta as integers, exactly whatever the compiler reorders, then the
/// remainders, which keeps positions to the f64 camera at zoom 22.
pub(crate) fn quantum(span: f64) -> f64 {
    2f64.powi((span * 64.0 / 2f64.powi(31)).log2().ceil() as i32)
}

/// The sphere globe positions are on is this many metres across
/// (`project.wgsl`'s `SPHERE_RADIUS`).
const GLOBE_SPAN: f64 = 2.0 * 6_378_137.0;

/// A position as whole quanta and remainders: the i32 words' bits, then
/// the f32 words' (see [`quantum`]).
fn split<const N: usize>(v: [f64; N], quantum: f64) -> [[u32; N]; 2] {
    let whole = v.map(|x| (x / quantum).round());
    let mut rest = [0u32; N];
    for i in 0..N {
        rest[i] = ((v[i] - whole[i] * quantum) as f32).to_bits();
    }
    [whole.map(|w| w as i64 as i32 as u32), rest]
}

/// Writes a frame block into a uniform buffer of [`FRAME_BYTES`]: the
/// scene's own, or another view's of the same scene (the terrain drape's).
/// `linear` when colours mix in linear light.
pub(crate) fn write_frame(gpu: &mut dyn Gpu, buffer: BufferId, frame: &FrameBlock, linear: bool) {
    gpu.write_buffer(buffer, 0, bytemuck::cast_slice(&frame_words(frame, linear)));
}

/// A frame block's words, as [`write_frame`] writes them.
pub(crate) fn frame_words(frame: &FrameBlock, linear: bool) -> Vec<u32> {
    use layout::frame as f;
    let mut b = orb_shaders::Block::new(f::WORDS);
    let view_proj = frame.view_proj.map(|column| column.map(|v| v as f32));
    b.f(f::VIEW_PROJ, view_proj.as_flattened());
    // Clip w less clip z, taken in f64: in f32 the difference cancels to a
    // few float steps, which decide coplanar faces' order by rounding.
    let depth = frame.view_proj.map(|column| (column[3] - column[2]) as f32);
    b.f(f::VIEW_DEPTH, &depth);
    b.f(f::VIEWPORT, &frame.viewport);
    b.f(f::WORLD_PER_PX, &[frame.world_per_px, frame.pixel_ratio]);
    b.f(f::ZOOM, &[frame.zoom as f32, frame.bearing]);
    b.f(f::ALIGN, &frame.align);
    b.f(f::EXTRUSION_SCALE, &[frame.extrusion_scale]);
    b.f(f::TIME, &[frame.time]);
    b.f(f::STATE_TIME, &[frame.state_time]);
    b.f(f::SYMBOL_FADE_CHANGE, &[frame.symbol_fade_change]);
    b.f(f::PITCH, &[frame.pitch]);
    b.f(f::ROLL, &[frame.roll]);
    b.f(f::CAMERA_TO_CENTER, &[frame.camera_to_center]);
    let q = quantum(frame.world_span);
    b.f(f::QUANTUM, &[q as f32]);
    b.f(f::GLOBE_QUANTUM, &[quantum(GLOBE_SPAN) as f32]);
    if let Some(t) = frame.terrain {
        b.f(f::DEM_UNPACK, &t.unpack);
        b.f(f::TERRAIN_EXAGGERATION, &[t.exaggeration]);
        b.f(f::DEM_DIM, &[t.dem_dim]);
        b.f(f::TERRAIN_GROUND, &[t.ground]);
        b.f(f::TERRAIN_LOWEST, &[t.lowest.min(0.0)]);
    }
    let [red, green, blue] = frame.light.color;
    b.f(f::LIGHT_COLOR, &[red, green, blue, frame.light.intensity]);
    b.f(f::LIGHT_POSITION, &frame.light.position);
    let [red, green, blue] = frame.light.ambient;
    b.f(
        f::LIGHT_AMBIENT,
        &[red, green, blue, frame.light.ambient_amount],
    );
    let (offset, intensity) = frame.light.shadow();
    b.f(f::SHADOW_OFFSET, &offset);
    b.f(f::SHADOW_INTENSITY, &[intensity]);
    let globe = frame.globe.map(|g| {
        b.f(f::GLOBE_VIEW_PROJ, g.view_proj.as_flattened());
        b.f(f::GLOBE_HORIZON, &g.horizon);
        b.f(f::GLOBE_TARGET, &g.target.map(|v| v as f32));
        b.f(f::GLOBE_AMOUNT, &[g.amount, g.metres_per_unit]);
        b.f(f::GLOBE_EYE, &g.eye);
        (g.target, g.plane)
    });
    let [camera_whole, camera_rest] = split(frame.camera, q);
    b.u(f::CAMERA_WHOLE, &camera_whole)
        .u(f::CAMERA_REST, &camera_rest);
    if let Some((target, plane)) = globe {
        let [whole, rest] = split(target, quantum(GLOBE_SPAN));
        b.u(f::GLOBE_TARGET_WHOLE, &whole)
            .u(f::GLOBE_TARGET_REST, &rest);
        b.u(f::GLOBE_PLANE, &[u32::from(plane)]);
    }
    if let Some(g) = frame.light.ground {
        let [x, y, z] = g.sun;
        b.f(f::GROUND_SUN, &[x, y, z, g.twilight / 2.0]);
        let [x, y, z] = g.up;
        b.f(f::GROUND_UP, &[x, y, z, g.amount]);
    }
    b.u(f::LINEAR, &[u32::from(linear)]);
    b.u(f::SURFACE_CELLS, &[surface_cells(frame.globe)]);
    b.0
}

/// A surface quad's cells per side on a globe. A single quad is a chord:
/// its middle sinks below the sphere, and seen from the eye a wide
/// kernel's peak slides toward the view's centre (a 60 px heatmap kernel
/// at zoom 1, 30° from the target, peaks 18 px short of its point). An
/// even count puts a vertex on the point itself; four cells keep the rest
/// of the footprint within a pixel or so of the sphere. GL JS tessellates
/// these as 3 × 3 uneven cells (`CircleBucket`, granularity 3), which
/// leaves that kernel's peak 4 px short.
const SURFACE_CELLS: u32 = 4;

/// The cells per side of a surface node's point quads in a view
/// (`types.wgsl`'s `point_cells`): [`SURFACE_CELLS`] on the sphere, one
/// on a plane. The frame block carries it, so the cull and the point
/// shaders read this one rule.
pub(crate) fn surface_cells(globe: Option<FrameGlobe>) -> u32 {
    if globe.is_some_and(|g| g.amount > 0.0 && !g.plane) {
        SURFACE_CELLS
    } else {
        1
    }
}

/// One placement of a tile's data: its transform into the world,
/// `origin + (x, -y) · scale` (or, with a planar warp, `origin` plus the
/// warp's quadratic patches; `scale` is then its local scale at the
/// centre), and where it may draw. On a globe, its position on the sphere
/// is `globe.origin` plus its warp's patches in space. Nothing in it
/// depends on the camera but the pattern origin's integer zoom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileRow {
    /// In world units; rows hold it as hi/lo f32, less the frame's camera.
    pub origin: [f64; 2],
    pub scale: f32,
    /// The tile's zoom level.
    pub zoom: f32,
    /// Rectangles in data units the placement draws in; unused entries are
    /// [`ClipRect::EMPTY`].
    pub clips: [ClipRect; 4],
    /// Where its content may land, in data units: a drape placement's slot,
    /// [`ClipRect::EVERYWHERE`] for a map placement. Content translated into
    /// a slot is owned where it was (`clips`) and bounded where it lands.
    pub bound: ClipRect,
    /// Global pixel origin at the integer camera zoom, modulo 2^32.
    pub pattern_origin: [u32; 2],
    pub extent: f32,
    /// The world copy it draws; symbol meshes keep state words per copy.
    pub wrap: i32,
    /// World units per logical pixel at this placement's scale, as a
    /// multiple of the frame's: pixel sizes (line widths, translations)
    /// convert with it. 1 for a map placement; a drape placement's is its
    /// slot's texels per logical pixel at the camera's zoom.
    pub px_scale: f32,
    /// The DEM tile elevated content of this placement stands on.
    pub dem: DemRef,
    /// The lowest and highest terrain elevation under the placement, in
    /// metres with exaggeration, for culling.
    pub terrain_range: [f32; 2],
    /// Which render nodes draw it.
    pub target: PlacementTarget,
    /// Its planar patches (`project.wgsl`), or [`TileWarp::NONE`].
    pub warp: TileWarp,
    /// On a globe view, where it is on the sphere.
    pub globe: Option<RowGlobe>,
    /// Which of the tiles around it at its zoom (bit `3 (dy + 1) + dx + 1`,
    /// rows down) are each placed alone over their whole square, and so
    /// draw entire the whole polygons whose centroid they hold
    /// (`orb_tile::Layer::is_whole`): this placement leaves those to
    /// them, and draws a whole polygon it doesn't own anywhere else
    /// clipped to its regions. 0 leaves nothing to a neighbour.
    pub neighbours: u16,
}

impl TileRow {
    /// Its clip as fragments test it where its content lands where it was
    /// (`types.wgsl`'s `fragment_clip`): the one rectangle it owns, within
    /// its bound, where it is affine (empty where it owns none); otherwise
    /// `None`, and fragments test its rectangles and bound.
    fn fragment_clip(&self) -> Option<ClipRect> {
        if self.warp.cells != 0 {
            return None;
        }
        let mut owned = (self.clips.iter()).filter(|c| c.min[0] < c.max[0] && c.min[1] < c.max[1]);
        let clip = match (owned.next(), owned.next()) {
            (None, _) => return Some(ClipRect::EMPTY),
            (Some(clip), None) => clip,
            (Some(_), Some(_)) => return None,
        };
        if self.bound.min == ClipRect::EVERYWHERE.min {
            return Some(*clip);
        }
        let lo = [0, 1].map(|i| clip.min[i].max(self.bound.min[i]));
        let hi = [0, 1].map(|i| clip.max[i].min(self.bound.max[i]).max(lo[i]));
        Some(ClipRect::new(lo, hi))
    }

    /// The row as `types.wgsl`'s `Tile`, its origins split at `quantum`
    /// on the plane.
    fn words(&self, quantum: f64) -> [u32; TILE_WORDS] {
        use layout::tile as t;
        let pack = |p: [i16; 2]| u32::from(p[0] as u16) | u32::from(p[1] as u16) << 16;
        let mut words = [0u32; TILE_WORDS];
        let mut put =
            |at: usize, values: &[u32]| words[at..][..values.len()].copy_from_slice(values);
        let [origin_whole, origin_rest] = split(self.origin, quantum);
        put(t::ORIGIN, &origin_whole);
        put(t::ORIGIN_REST, &origin_rest);
        put(t::SCALE, &[self.scale.to_bits(), self.zoom.to_bits()]);
        for (i, clip) in self.clips.iter().enumerate() {
            put(t::CLIPS + 2 * i, &[pack(clip.min), pack(clip.max)]);
        }
        put(t::BOUND, &[pack(self.bound.min), pack(self.bound.max)]);
        let clip = self.fragment_clip().unwrap_or(CLIP_FETCHED);
        put(t::CLIP, &[pack(clip.min), pack(clip.max)]);
        put(t::PATTERN_ORIGIN, &self.pattern_origin);
        put(t::EXTENT, &[self.extent.to_bits(), self.wrap as u32]);
        put(t::PX_SCALE, &[self.px_scale.to_bits()]);
        put(t::DEM, &self.dem.words());
        put(t::TERRAIN_RANGE, &[pack_range(self.terrain_range)]);
        put(t::WARP_BASE, &[self.warp.base, self.warp.cells]);
        if let Some(globe) = self.globe {
            let [whole, rest] = split(globe.origin, self::quantum(GLOBE_SPAN));
            put(t::GLOBE_ORIGIN, &whole);
            put(t::GLOBE_ORIGIN_REST, &rest);
            put(t::GLOBE_BASE, &[globe.warp.base, globe.warp.cells]);
            put(t::GLOBE_FLAGS, &[u32::from(!globe.on_plane)]);
        }
        put(t::NEIGHBOURS, &[u32::from(self.neighbours)]);
        words
    }
}

/// A placement on a globe: its patches in space, their anchor in ECEF
/// metres, and whether it also has a position on the
/// plane (a tile the plane can't hold, such as one at a pole, draws only
/// on the globe).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowGlobe {
    pub warp: TileWarp,
    pub origin: [f64; 3],
    pub on_plane: bool,
}

/// A tile's biquadratic patches in the property arena (`GpuPerVertex`),
/// as [`warp_words`] lays them out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileWarp {
    base: u32,
    cells: u32,
    words: u32,
}

impl Default for TileWarp {
    fn default() -> Self {
        Self::NONE
    }
}

impl TileWarp {
    /// No warp: an affine tile.
    pub const NONE: Self = Self {
        base: 0,
        cells: 0,
        words: 0,
    };

    /// Its first word in the property arena.
    pub fn base(self) -> u32 {
        self.base
    }

    /// Its patches per side, packed as [`warp_words`] has them; 0 for
    /// none.
    pub fn cells(self) -> u32 {
        self.cells
    }
}

/// Where a placement draws: on the map, or into the terrain's drape
/// textures for the render nodes of one drape stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PlacementTarget {
    #[default]
    Map,
    Drape(u8),
    /// Symbol nodes only: a tile that left the draw list keeps drawing its
    /// labels until a placement covers what replaced it (GL JS
    /// `holdingForFade`).
    Labels,
}

/// A placement's DEM: tile data unit `p` reads the DEM page's `layer` at
/// `(p / extent + offset) / 2^depth` of the DEM tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DemRef {
    pub layer: u16,
    pub depth: u8,
    pub offset: [u16; 2],
}

impl DemRef {
    /// No DEM: the ground at sea level.
    pub const NONE: Self = Self {
        layer: u16::MAX,
        depth: 0,
        offset: [0; 2],
    };

    fn words(self) -> [u32; 2] {
        [
            u32::from(self.layer) | u32::from(self.depth) << 16,
            u32::from(self.offset[0]) | u32::from(self.offset[1]) << 16,
        ]
    }
}

/// A rectangle in tile data units, minimum inclusive and maximum exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipRect {
    pub min: [i16; 2],
    pub max: [i16; 2],
}

impl ClipRect {
    pub const EMPTY: Self = Self {
        min: [i16::MAX; 2],
        max: [i16::MIN; 2],
    };

    /// Every position a data unit can name.
    pub const EVERYWHERE: Self = Self {
        min: [i16::MIN; 2],
        max: [i16::MAX; 2],
    };

    pub const fn new(min: [i16; 2], max: [i16; 2]) -> Self {
        Self { min, max }
    }
}

/// `types.wgsl`'s `CLIP_FETCHED`: a tile row's clip that fragments test
/// against its rectangles and bound.
const CLIP_FETCHED: ClipRect = ClipRect::new([i16::MIN; 2], [i16::MIN; 2]);

/// Words of a tile row (`types.wgsl`'s `Tile`, `TILE_WORDS` there).
const TILE_WORDS: usize = layout::tile::WORDS;
/// Words per biquadratic patch of a planar warp: nine nodes' positions
/// (`project.wgsl`).
const WARP_PATCH_WORDS: usize = 18;
/// Words per biquadratic patch on a globe: nine nodes' positions in space.
const GLOBE_PATCH_WORDS: usize = 27;
/// Words of a warp's window, before its patches.
const WARP_WINDOW_WORDS: usize = 4;

/// A warp as the scene takes it (`project.wgsl`): its patches per side
/// (x then y) packed, and its words: `window` (the rectangle of the tile
/// the patches span, min then max in tile fractions, y down), then the
/// patches' nodes row by row, nine positions each.
pub fn warp_words(
    cells: [u32; 2],
    window: [f32; 4],
    nodes: impl IntoIterator<Item = f32>,
) -> (u32, Vec<f32>) {
    debug_assert!(cells.iter().all(|&c| (1..=0xffff).contains(&c)));
    let words = window.into_iter().chain(nodes).collect();
    (cells[0] | cells[1] << 16, words)
}
const CHUNK_WORDS: usize = layout::chunk::WORDS;
const CANDIDATE_WORDS: usize = layout::candidate::WORDS;
const VISIBLE_WORDS: usize = layout::visible::WORDS;
const NODE_CULL_WORDS: usize = layout::node_cull::WORDS;
/// Words per node of `SceneBuffers::args`: a draw's indirect arguments
/// and the cull's counters.
pub const ARGS_WORDS: usize = 8;
/// Candidates per cull workgroup of an indexed node (`cull.wgsl`).
const INDEXED_BATCH: u32 = 64;
/// A candidate's `drawn` word before its first cull: neither drawn (1) nor
/// hidden (0), so the cull pass writes its slot (`cull.wgsl`).
const UNKNOWN: u32 = u32::MAX;

/// How many of each thing the scene can hold. Buffers are allocated once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneCapacity {
    /// Tile placements: a tile drawn in two world copies takes two.
    pub placements: u32,
    pub chunks: u32,
    /// `(chunk, placement)` pairs the cull pass considers, over all nodes.
    pub candidates: u32,
    pub nodes: u32,
    /// The vertex pool's first size; it doubles as meshes need, up to
    /// `max_vertex_words`.
    pub vertex_words: u32,
    pub index_words: u32,
    /// Slots of indexed nodes' candidates in the compacted index buffer:
    /// each candidate's chunk indices once per placement.
    pub compacted_words: u32,
    /// The property arena's first size, past the glyph table at its start.
    pub prop_words: u32,
    /// Rows of the glyph atlas's table, the property arena's first block.
    pub glyph_rows: u32,
    /// Ceilings of the growable pools, which the device's buffer limits
    /// lower further. Growth is for the tiles in view: the map evicts
    /// tiles out of view first.
    pub max_vertex_words: u32,
    pub max_index_words: u32,
    pub max_compacted_words: u32,
    pub max_prop_words: u32,
    pub max_chunks: u32,
    pub max_candidates: u32,
}

impl SceneCapacity {
    /// Pools that start at 1 MB and grow to the device's limits as the
    /// tiles in view need.
    pub fn growable() -> Self {
        let start = 1 << 18;
        Self {
            vertex_words: start,
            index_words: start,
            compacted_words: start,
            prop_words: start,
            chunks: 1 << 13,
            candidates: 1 << 14,
            max_vertex_words: u32::MAX,
            max_index_words: u32::MAX,
            max_compacted_words: u32::MAX,
            max_prop_words: u32::MAX,
            max_chunks: 1 << 20,
            max_candidates: 1 << 20,
            ..Self::fixed()
        }
    }

    /// Fixed pools at the given sizes, as tests set them: the growable
    /// pools don't grow past their first size.
    pub fn fixed() -> Self {
        Self {
            placements: 1024,
            chunks: 1 << 17,
            candidates: 1 << 18,
            // A render node per drawn layer and symbol pass: Americana
            // needs about 400.
            nodes: 1024,
            vertex_words: 16 << 20,
            index_words: 16 << 20,
            compacted_words: 16 << 20,
            prop_words: 4 << 20,
            glyph_rows: 1 << 16,
            max_vertex_words: 0,
            max_index_words: 0,
            max_compacted_words: 0,
            max_prop_words: 0,
            max_chunks: 0,
            max_candidates: 0,
        }
    }
}

/// One of the scene's pools, in bytes: in use, allocated, and the most it
/// may grow to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolMemory {
    pub used: u64,
    pub allocated: u64,
    pub ceiling: u64,
}

/// The scene's GPU memory by pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneMemory {
    pub vertices: PoolMemory,
    pub indices: PoolMemory,
    pub compacted: PoolMemory,
    pub props: PoolMemory,
    /// The fixed tables: placements, chunks, candidates, visible lists,
    /// node tables and draw arguments.
    pub tables: u64,
}

impl Default for SceneCapacity {
    /// [`growable`](Self::growable) pools.
    fn default() -> Self {
        Self::growable()
    }
}

/// The GPU memory budget on desktops, in bytes.
pub const DESKTOP_BUDGET: u64 = 512 << 20;
/// The GPU memory budget on phones, in bytes.
pub const MOBILE_BUDGET: u64 = 192 << 20;

/// The scene's buffers besides its tables ([`Tables`]): what every
/// pipeline reads of the frame, and what the driver writes for nodes to
/// draw.
///
/// The tables: tile placements; chunks; `(chunk, placement)` candidates,
/// grouped by node, with what the cull last wrote for indexed nodes'; node
/// rows; per node, the visible `(chunk, placement)` list the cull writes
/// for listed nodes, or an indexed node's runs; and the vertex, index and
/// property arenas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SceneBuffers {
    pub frame: BufferId,
    /// Per node, [`ARGS_WORDS`] words its driver writes: draw arguments
    /// (four for a draw, five for an indexed draw), then the counters of
    /// `cull.wgsl`.
    pub args: BufferId,
    /// Indexed nodes' index buffer: their candidates' slots, each the
    /// chunk's indices tagged with its run, or hidden.
    pub compacted: BufferId,
}

/// A stream of u32 words to upload: in core memory, or held by the host
/// (payloads reach the GPU without entering the render worker's wasm
/// memory).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Words<'a> {
    Core(&'a [u32]),
    Host {
        bytes: HostBytesId,
        /// Byte offset of the first word in `bytes`.
        offset: u64,
        /// Number of words.
        len: u32,
    },
}

impl Words<'_> {
    pub fn len(&self) -> u32 {
        match self {
            Self::Core(words) => words.len() as u32,
            Self::Host { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Queues the words into `buffer` from word `word_offset`: what the
    /// storage backing of [`Tables`] does.
    fn write_buffer(&self, gpu: &mut dyn Gpu, buffer: BufferId, word_offset: u32) {
        let offset = u64::from(word_offset) * 4;
        match *self {
            Self::Core(words) => gpu.write_buffer(buffer, offset, bytemuck::cast_slice(words)),
            Self::Host {
                bytes,
                offset: start,
                len,
            } => {
                if len > 0 {
                    gpu.write_buffer_from_host(
                        buffer,
                        offset,
                        bytes,
                        start..start + u64::from(len) * 4,
                    );
                }
            }
        }
    }
}

/// A mesh's streams and chunks, ready to upload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshData<'a> {
    pub vertices: Words<'a>,
    pub indices: Words<'a>,
    /// Property rows.
    pub props: Words<'a>,
    /// Words per feature of the rows. Shaders that read rows take their
    /// stride from the mesh, so rows and stride always agree.
    pub row_words: u32,
    pub chunks: &'a [Chunk],
}

/// The words and chunks of a mesh to insert, for `Scene::fits`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshSize {
    pub vertices: u32,
    pub indices: u32,
    pub props: u32,
    pub chunks: u32,
}

impl<'a> MeshData<'a> {
    /// A mesh laid out in core memory, with its property rows.
    pub fn new(mesh: &'a Mesh, props: &'a [u32], row_words: u32) -> Self {
        Self {
            vertices: Words::Core(&mesh.vertices),
            indices: Words::Core(&mesh.indices),
            props: Words::Core(props),
            row_words,
            chunks: &mesh.chunks,
        }
    }

    /// What it takes in the scene.
    pub fn size(&self) -> MeshSize {
        MeshSize {
            vertices: self.vertices.len(),
            indices: self.indices.len(),
            props: self.props.len(),
            chunks: self.chunks.len() as u32,
        }
    }
}

/// A tile's data in the scene: its meshes and placements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileId(u32);

/// A mesh resident in the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshId(u32);

/// Metres per unit of a placement's packed terrain range (`types.wgsl`'s
/// `TERRAIN_RANGE_UNIT`).
const TERRAIN_RANGE_UNIT: f32 = 4.0;

/// A terrain range as two i16s of [`TERRAIN_RANGE_UNIT`], rounded
/// outward so it still covers the range.
fn pack_range([lo, hi]: [f32; 2]) -> u32 {
    let unit = |v: f32| v.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16 as u16;
    u32::from(unit((lo / TERRAIN_RANGE_UNIT).floor()))
        | u32::from(unit((hi / TERRAIN_RANGE_UNIT).ceil())) << 16
}

/// The scene is full; evict something and retry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfSpace;

#[derive(Debug, Default)]
struct TileEntry {
    meshes: Vec<MeshId>,
    placements: Vec<Placement>,
}

/// A tile placement: its row in the placement table, and the row's values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Placement {
    pub slot: u32,
    pub row: TileRow,
}

#[derive(Debug)]
struct MeshEntry {
    tile: TileId,
    node: u32,
    chunk_slots: Vec<u32>,
    chunks: Vec<Chunk>,
    vertices: Range<u32>,
    indices: Range<u32>,
    props: Range<u32>,
    row_words: u32,
    /// Per feature slot, its feature-state index, in the property arena.
    state: Range<u32>,
    /// A symbol mesh's state words, in the property arena.
    symbol_states: Range<u32>,
    /// World copies the state words are kept for.
    symbol_copies: u32,
    /// A symbol mesh's draw-order table: its offset in the mesh's vertex
    /// words, and its quads.
    symbol_order: [u32; 2],
}

/// Chunk word that holds the first word of the mesh's feature-state
/// remap, or [`NO_STATE`].
const CHUNK_STATE_WORD: u64 = layout::chunk::STATE_BASE as u64;
const CHUNK_PROPS_WORD: u64 = layout::chunk::PROP_BASE as u64;
const CHUNK_ROW_WORDS_WORD: u64 = layout::chunk::ROW_WORDS as u64;

/// No feature-state remap: every feature has the default state.
pub(crate) const NO_STATE: u32 = u32::MAX;

#[derive(Debug)]
struct NodeEntry {
    meshes: Vec<MeshId>,
    cull: NodeCull,
}

/// How the cull pass treats a render node's chunks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeCull {
    /// Extra margin around chunk bounds, in pixels.
    pub margin_px: f32,
    /// Maximum nonnegative elevation in metres (base and roof).
    pub elevation: f32,
    pub draw: NodeDraw,
    /// Hidden nodes keep their candidates and draw nothing.
    pub visible: bool,
    /// The placements it draws: on the map, or those of its drape stack,
    /// which the cull keeps by their clip rectangles alone.
    pub target: PlacementTarget,
    /// It draws symbols: held tiles' [`PlacementTarget::Labels`]
    /// placements too.
    pub labels: bool,
    /// Its point quads lie on the surface (heatmap kernels, map-aligned
    /// circles): on a globe each draws as a grid of cells that follows the
    /// sphere (`types.wgsl`'s `SURFACE`).
    pub surface: bool,
    /// Its meshes draw past their placements' clip rectangles: a whole
    /// extrusion draws entire from the placement owning it
    /// (`extrusion.wgsl`), so the cull keeps chunks by their own bounds.
    pub unclipped: bool,
}

/// How a render node draws its visible chunks, which the cull pass
/// prepares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeDraw {
    /// Image meshes, by vertex pulling: one instance of 768 vertices per
    /// visible chunk, from the node's visible list. Their fragments clip in
    /// the image's own tile units (a warped tile's mesh is in other units),
    /// so the cull doesn't narrow their bounds to the clip rectangles.
    Images,
    /// Point families: 256 instances per visible chunk, each a quad of six
    /// vertices (or a grid of such cells for a [`NodeCull::surface`] node on
    /// a globe). Anchors are clipped, then the full footprints expanded.
    Points,
    /// Compacted indices: each candidate has a slot in the compacted index
    /// buffer, which the cull pass fills with the chunk's indices when it
    /// becomes visible and with hidden indices when it leaves the view;
    /// one `drawIndexedIndirect` covers the node's slots. Indices are
    /// `run << run_shift | vertex`, a run being a mesh in one placement.
    Indexed,
}

impl NodeCull {
    /// Whether it draws placements of `target`.
    fn draws(&self, target: PlacementTarget) -> bool {
        target == self.target || (self.labels && target == PlacementTarget::Labels)
    }

    /// The node row's draw word: how it draws, and its flags.
    fn draw_word(&self) -> u32 {
        self.draw.word()
            | if self.target == PlacementTarget::Map {
                0
            } else {
                DRAPED
            }
            | if self.visible { 0 } else { HIDDEN }
            | if self.surface { SURFACE } else { 0 }
            | if self.unclipped { UNCLIPPED } else { 0 }
    }
}

impl NodeDraw {
    fn word(self) -> u32 {
        match self {
            Self::Images => 0,
            Self::Points => 1,
            Self::Indexed => 2,
        }
    }
}

/// The cull's flag on a node's draw word: its candidates are drape
/// placements (`cull.wgsl`'s `DRAPED`).
pub(crate) const DRAPED: u32 = 1 << 8;
/// The cull's flag on a node's draw word: the node draws nothing
/// (`cull.wgsl`'s `HIDDEN`).
pub(crate) const HIDDEN: u32 = 1 << 9;
/// The cull's flag on a node's draw word: its point quads lie on the
/// surface (`cull.wgsl`'s `SURFACE`).
pub(crate) const SURFACE: u32 = 1 << 10;
/// The cull's flag on a node's draw word: its meshes draw past the clip
/// rectangles (`cull.wgsl`'s `UNCLIPPED`).
pub(crate) const UNCLIPPED: u32 = 1 << 11;

#[derive(Debug)]
pub(crate) struct Scene {
    tables: Tables,
    buffers: SceneBuffers,
    /// The driver whose frames it lays candidates out for.
    driver: driver::Kind,
    /// The frame block as the last [`prepare`](Self::prepare) wrote it.
    #[cfg(cpu_driven)]
    frame: Vec<u32>,
    capacity: SceneCapacity,
    /// Bits of a compacted index on the renderer's driver
    /// ([`SubmitLimits::index_bits`]).
    index_bits: u32,
    placement_slots: Slots,
    chunk_slots: Slots,
    vertex_arena: Arena,
    index_arena: Arena,
    prop_arena: Arena,
    /// Vertex and index ranges more than one mesh draws, by their first
    /// word: how many meshes beyond the first. The last to go frees it.
    shared: [std::collections::HashMap<u32, u32>; 2],
    /// The capacity the scene started with: the first sizes the growable
    /// pools and lists don't shrink below.
    first: SceneCapacity,
    tiles: Vec<Option<TileEntry>>,
    free_tiles: Vec<u32>,
    meshes: Vec<Option<MeshEntry>>,
    free_meshes: Vec<u32>,
    nodes: Vec<NodeEntry>,
    /// Node rows' layout (slices of the candidate lists and compacted
    /// buffer) is stale: tiles or placements came or went, or nodes changed
    /// how they draw.
    candidates_dirty: bool,
    /// The node rows as the last rebuild laid them out, with their cull
    /// parameters (margin, elevation, visibility) patched since.
    node_rows: Vec<u32>,
    /// Cull parameters changed since the node rows were last written.
    params_dirty: bool,
    /// Candidates laid out, holes aside.
    candidate_count: u32,
    candidates_dropped: u32,
    /// The ends of the node slices of the candidate lists and the
    /// compacted buffer.
    rows_end: u32,
    compacted_words: u32,
    /// Each node's slices as last laid out, so a rebuild lays out only
    /// what changed (`candidates`); empty when the next is laid out whole.
    laid: Vec<candidates::LaidNode>,
    /// Compacted words the last rebuild hid, the slots of runs that went.
    hidden: Vec<Range<u32>>,
    /// Compacted words the last rebuild moved, from slots it left to its
    /// own (`candidates`).
    moves: Vec<Move>,
    /// What moves go through, and its bytes: a buffer can't copy into
    /// itself.
    scratch: Option<(BufferId, u64)>,
    /// Each mesh's serial, by mesh slot: no two meshes take the same.
    mesh_serials: Vec<u64>,
    next_serial: u64,
    /// Rows of the cull dispatch: the most batches an indexed node has.
    cull_batches: u32,
    /// Changes whenever a pool or table moves into a larger buffer: bind
    /// groups and
    /// recorded commands over the old one must be made again.
    buffers_revision: u64,
    /// Placement rows that changed since the last `take_placements_written`.
    placements_written: u32,
    /// The tile table's rows as the GPU holds them, and rows changed since
    /// the last [`prepare`](Self::prepare), which writes them.
    tile_rows: Vec<u32>,
    rows_dirty: Vec<(u32, TileRow)>,
    /// The quantum rows' origins are split at ([`quantum`]).
    quantum: f64,
}

/// Counts for the stats HUD and memory accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneStats {
    pub meshes: u32,
    pub candidates: u32,
    /// Candidates beyond [`SceneCapacity::candidates`] or
    /// [`SceneCapacity::compacted_words`], or beyond an indexed node's runs
    /// (see [`NodeDraw::Indexed`]), not drawn.
    pub candidates_dropped: u32,
    pub vertex_words: u32,
    pub index_words: u32,
    pub compacted_words: u32,
    pub prop_words: u32,
}

impl Scene {
    /// Destroys the scene's buffers, with the scene.
    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        let b = self.buffers;
        for buffer in [b.frame, b.args, b.compacted] {
            gpu.destroy_buffer(buffer);
        }
        if let Some((scratch, _)) = self.scratch {
            gpu.destroy_buffer(scratch);
        }
        self.tables.destroy(gpu);
    }

    /// A scene for `driver`'s frames, its tables on the device's backing.
    pub(crate) fn new(gpu: &mut dyn Gpu, capacity: SceneCapacity, driver: driver::Kind) -> Self {
        // The growable pools stay within one table.
        let backing = Backing::of(gpu.caps());
        let largest = backing.largest(&gpu.caps().limits);
        let mut capacity = capacity;
        // The glyph table is the property arena's first block.
        let glyph_words = capacity.glyph_rows * GLYPH_ROW_WORDS as u32;
        capacity.prop_words += glyph_words;
        capacity.max_prop_words = capacity.max_prop_words.saturating_add(glyph_words);
        for (start, max) in [
            (&mut capacity.vertex_words, &mut capacity.max_vertex_words),
            (&mut capacity.index_words, &mut capacity.max_index_words),
            (
                &mut capacity.compacted_words,
                &mut capacity.max_compacted_words,
            ),
            (&mut capacity.prop_words, &mut capacity.max_prop_words),
        ] {
            *start = (*start).min(largest);
            *max = (*max).min(largest).max(*start);
        }
        #[cfg(cpu_driven)]
        if backing == Backing::Texture {
            let limits = &gpu.caps().limits;
            let width = tables::data_width(limits);
            fit_row_texture(&mut capacity, limits.max_texture_dimension_2d, width);
        }
        let mut buffer = |label: &str, words: u64, usage: BufferUsages| {
            gpu.create_buffer(&BufferDesc {
                label,
                size: (words * 4).max(16),
                usage,
            })
        };
        let c = capacity;
        let words = |count: u32, per: usize| u64::from(count) * per as u64;
        let buffers = SceneBuffers {
            frame: buffer(
                "frame",
                FRAME_WORDS as u64,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            args: buffer("draw args", words(c.nodes, ARGS_WORDS), driver.args_usage()),
            compacted: buffer(
                "compacted indices",
                u64::from(c.compacted_words),
                driver.compacted_usage(),
            ),
        };
        let tables = Tables::new(
            gpu,
            backing,
            [
                words(c.placements, TILE_WORDS),
                words(c.chunks, CHUNK_WORDS),
                words(c.candidates, CANDIDATE_WORDS),
                words(c.nodes, NODE_CULL_WORDS),
                words(c.candidates, VISIBLE_WORDS),
                u64::from(c.vertex_words),
                u64::from(c.index_words),
                u64::from(c.prop_words),
            ],
            driver.mirrored(),
        );
        let mut prop_arena = Arena::new(c.prop_words);
        let glyphs = prop_arena.alloc(glyph_words).expect("the glyph table fits");
        debug_assert_eq!(glyphs.start, 0);
        Self {
            tables,
            buffers,
            driver,
            #[cfg(cpu_driven)]
            frame: Vec::new(),
            capacity,
            index_bits: driver.index_bits(),
            placement_slots: Slots::new(c.placements),
            chunk_slots: Slots::new(c.chunks),
            vertex_arena: Arena::new(c.vertex_words),
            index_arena: Arena::new(c.index_words),
            prop_arena,
            shared: Default::default(),
            first: capacity,
            tiles: Vec::new(),
            free_tiles: Vec::new(),
            meshes: Vec::new(),
            free_meshes: Vec::new(),
            nodes: Vec::new(),
            candidates_dirty: true,
            node_rows: Vec::new(),
            params_dirty: false,
            candidate_count: 0,
            candidates_dropped: 0,
            rows_end: 0,
            compacted_words: 0,
            laid: Vec::new(),
            hidden: Vec::new(),
            moves: Vec::new(),
            scratch: None,
            mesh_serials: Vec::new(),
            next_serial: 0,
            cull_batches: 1,
            buffers_revision: 0,
            placements_written: 0,
            tile_rows: Vec::new(),
            rows_dirty: Vec::new(),
            quantum: 0.0,
        }
    }

    pub(crate) fn buffers(&self) -> &SceneBuffers {
        &self.buffers
    }

    pub(crate) fn tables(&self) -> &Tables {
        &self.tables
    }

    #[cfg(cpu_driven)]
    pub(crate) fn tables_mut(&mut self) -> &mut Tables {
        &mut self.tables
    }

    /// The frame block's words, as the last [`prepare`](Self::prepare)
    /// wrote them.
    #[cfg(cpu_driven)]
    pub(crate) fn frame(&self) -> &[u32] {
        &self.frame
    }

    /// The render nodes.
    #[cfg(cpu_driven)]
    pub(crate) fn node_count(&self) -> u32 {
        self.nodes.len() as u32
    }

    /// Lays the candidates out again at the next
    /// [`prepare`](Self::prepare), every slot to be written anew.
    #[cfg(cpu_driven)]
    pub(crate) fn lay_out_again(&mut self) {
        self.candidates_dirty = true;
        self.laid.clear();
    }

    pub(crate) fn capacity(&self) -> &SceneCapacity {
        &self.capacity
    }

    /// What candidates are laid out within: the pools' sizes now, and
    /// the driver's index bits.
    pub(crate) fn limits(&self) -> SubmitLimits {
        SubmitLimits::new(&self.capacity, self.index_bits)
    }

    /// The cull dispatch's grid, as of the last [`prepare`](Self::prepare):
    /// one workgroup per node, by as many rows as the indexed node with the
    /// most candidates has batches of them.
    pub(crate) fn cull_grid(&self) -> [u32; 2] {
        [self.nodes.len() as u32, self.cull_batches]
    }

    /// Orders tiles within each render node independently of upload order.
    /// Meshes of the same tile keep their relative order; tiles not listed
    /// follow the listed ones. The order persists until changed again.
    pub(crate) fn set_tile_order(&mut self, tiles: &[TileId]) {
        let order: std::collections::HashMap<_, _> = tiles
            .iter()
            .enumerate()
            .map(|(index, &tile)| (tile, index))
            .collect();
        let meshes = &self.meshes;
        for node in &mut self.nodes {
            let key = |id: &MeshId| {
                let tile = meshes[id.0 as usize].as_ref().expect("live mesh").tile;
                order.get(&tile).copied().unwrap_or(usize::MAX)
            };
            if !node.meshes.is_sorted_by_key(key) {
                node.meshes.sort_by_key(key);
                self.candidates_dirty = true;
            }
        }
    }

    /// Whether render `node` has any resident mesh.
    pub(crate) fn node_has_meshes(&self, node: u32) -> bool {
        !self.nodes[node as usize].meshes.is_empty()
    }

    pub(crate) fn stats(&self) -> SceneStats {
        SceneStats {
            meshes: (self.meshes.len() - self.free_meshes.len()) as u32,
            candidates: self.candidate_count,
            candidates_dropped: self.candidates_dropped,
            vertex_words: self.vertex_arena.used(),
            index_words: self.index_arena.used(),
            compacted_words: self.compacted_words,
            prop_words: self.prop_arena.used(),
        }
    }

    /// Makes room for `nodes` render nodes: the node cull and draw argument
    /// tables move to buffers twice the size until they fit. Their rows
    /// are written whole again ([`remap_nodes`](Self::remap_nodes) follows).
    pub(crate) fn grow_nodes(&mut self, gpu: &mut dyn Gpu, nodes: u32) {
        if nodes <= self.capacity.nodes {
            return;
        }
        let new = nodes.next_power_of_two();
        let words = |per: usize| u64::from(new) * per as u64;
        self.tables
            .resize(gpu, Table::NodeCull, words(NODE_CULL_WORDS), 0);
        let desc = BufferDesc {
            label: "draw args",
            size: (words(ARGS_WORDS) * 4).max(16),
            usage: self.driver.args_usage(),
        };
        move_buffer(gpu, &mut self.buffers.args, &desc, 0);
        self.capacity.nodes = new;
        self.candidates_dirty = true;
        self.laid.clear();
        self.buffers_revision += 1;
    }

    /// Renumbers the render nodes with their meshes: node `i` becomes
    /// `map[i]`, and the scene has `nodes.len()` nodes. Nodes mapped to
    /// `None` must hold no meshes.
    pub(crate) fn remap_nodes(&mut self, map: &[Option<u32>], nodes: &[NodeCull]) {
        assert_eq!(map.len(), self.nodes.len(), "one entry per node");
        assert!(
            nodes.len() as u32 <= self.capacity.nodes,
            "too many render nodes"
        );
        let mut entries: Vec<NodeEntry> = nodes
            .iter()
            .map(|&cull| NodeEntry {
                meshes: Vec::new(),
                cull,
            })
            .collect();
        for (old, node) in self.nodes.drain(..).enumerate() {
            match map[old] {
                Some(new) => {
                    for &id in &node.meshes {
                        self.meshes[id.0 as usize].as_mut().expect("live mesh").node = new;
                    }
                    entries[new as usize].meshes = node.meshes;
                }
                None => assert!(node.meshes.is_empty(), "removed node {old} holds meshes"),
            }
        }
        self.nodes = entries;
        self.candidates_dirty = true;
        self.laid.clear();
    }

    /// Sets how each render node is culled. Changing the number of nodes
    /// requires the scene to hold no meshes. Margins, elevations and
    /// visibility are parameters the cull reads from the node rows: they
    /// change without laying the candidates out again.
    pub(crate) fn set_nodes(&mut self, nodes: &[NodeCull]) {
        if nodes.len() != self.nodes.len() {
            assert!(
                self.nodes.iter().all(|n| n.meshes.is_empty()),
                "nodes changed while meshes are resident"
            );
            assert!(
                nodes.len() as u32 <= self.capacity.nodes,
                "too many render nodes"
            );
            self.nodes = nodes
                .iter()
                .map(|&cull| NodeEntry {
                    meshes: Vec::new(),
                    cull,
                })
                .collect();
            self.candidates_dirty = true;
            self.laid.clear();
            return;
        }
        for (i, &cull) in nodes.iter().enumerate() {
            self.set_node(i as u32, cull);
        }
    }

    /// Sets how one render node is culled; see [`set_nodes`](Self::set_nodes).
    pub(crate) fn set_node(&mut self, node: u32, cull: NodeCull) {
        let i = node as usize;
        let entry = &mut self.nodes[i];
        if entry.cull == cull {
            return;
        }
        let shape = |c: &NodeCull| (c.draw, c.target, c.labels);
        if shape(&entry.cull) != shape(&cull) {
            self.candidates_dirty = true;
        } else if let Some(row) = self
            .node_rows
            .get_mut(i * NODE_CULL_WORDS..(i + 1) * NODE_CULL_WORDS)
        {
            use layout::node_cull as n;
            row[n::MARGIN_PX] = cull.margin_px.to_bits();
            row[n::DRAW] = cull.draw_word();
            row[n::ELEVATION] = cull.elevation.to_bits();
            self.params_dirty = true;
        }
        entry.cull = cull;
    }

    /// A new tile with no meshes and no placements.
    pub(crate) fn insert_tile(&mut self) -> TileId {
        match self.free_tiles.pop() {
            Some(id) => {
                self.tiles[id as usize] = Some(TileEntry::default());
                TileId(id)
            }
            None => {
                self.tiles.push(Some(TileEntry::default()));
                TileId(self.tiles.len() as u32 - 1)
            }
        }
    }

    /// Removes a tile with its meshes and placements.
    pub(crate) fn remove_tile(&mut self, tile: TileId) {
        let entry = self.tiles[tile.0 as usize]
            .take()
            .expect("tile already removed");
        self.free_tiles.push(tile.0);
        for placement in entry.placements {
            self.placement_slots.free(placement.slot);
        }
        for mesh in entry.meshes {
            self.drop_mesh(mesh);
        }
        self.candidates_dirty = true;
    }

    /// Uploads a warp ([`warp_words`]) for tile rows to draw through:
    /// [`WARP_PATCH_WORDS`] f32 per patch on a plane, [`GLOBE_PATCH_WORDS`]
    /// on a globe.
    pub(crate) fn add_warp(
        &mut self,
        gpu: &mut dyn Gpu,
        cells: u32,
        patches: &[f32],
    ) -> Result<TileWarp, OutOfSpace> {
        let count = ((cells & 0xffff) * (cells >> 16)) as usize;
        debug_assert!(
            [WARP_PATCH_WORDS, GLOBE_PATCH_WORDS]
                .iter()
                .any(|words| patches.len() == WARP_WINDOW_WORDS + count * words)
        );
        let range = self
            .prop_arena
            .alloc(patches.len() as u32)
            .ok_or(OutOfSpace)?;
        let words: Vec<u32> = patches.iter().map(|v| v.to_bits()).collect();
        self.tables
            .write(gpu, Table::Props, range.start, Words::Core(&words));
        Ok(TileWarp {
            base: range.start,
            cells,
            words: patches.len() as u32,
        })
    }

    /// Frees a warp no row draws through any more.
    pub(crate) fn remove_warp(&mut self, warp: TileWarp) {
        if warp.cells > 0 {
            self.prop_arena.free(warp.base..warp.base + warp.words);
        }
    }

    /// Sets where a tile draws: one row per placement, none to hide it.
    /// Rows are of the tile's place in the world, not of the camera, so
    /// a frame that only moves the camera changes none; changed rows reach
    /// the GPU together in the next [`prepare`](Self::prepare).
    pub(crate) fn place_tile(&mut self, tile: TileId, rows: &[TileRow]) -> Result<(), OutOfSpace> {
        let entry = self.tiles[tile.0 as usize].as_mut().expect("unknown tile");
        let added = rows.len().saturating_sub(entry.placements.len());
        if entry.placements.len() != rows.len() {
            let needed = rows.len().saturating_sub(entry.placements.len()) as u32;
            if needed > self.placement_slots.available() {
                return Err(OutOfSpace);
            }
            while entry.placements.len() > rows.len() {
                let placement = entry.placements.pop().expect("non-empty");
                self.placement_slots.free(placement.slot);
            }
            while entry.placements.len() < rows.len() {
                entry.placements.push(Placement {
                    slot: self.placement_slots.alloc().expect("checked above"),
                    row: rows[entry.placements.len()],
                });
            }
            self.candidates_dirty = true;
        }
        let fresh = entry.placements.len() - added;
        for (i, (placement, row)) in entry.placements.iter_mut().zip(rows).enumerate() {
            if i >= fresh || placement.row != *row {
                // Which nodes draw it decides the candidates.
                self.candidates_dirty |= placement.row.target != row.target;
                placement.row = *row;
                self.rows_dirty.push((placement.slot, *row));
            }
        }
        Ok(())
    }

    /// Packs the rows that changed, or every row when the quantum did,
    /// and writes them in one write.
    fn write_rows(&mut self, gpu: &mut dyn Gpu, quantum: f64) {
        if quantum != self.quantum {
            self.quantum = quantum;
            self.rows_dirty = self
                .tiles
                .iter()
                .flatten()
                .flat_map(|t| &t.placements)
                .map(|p| (p.slot, p.row))
                .collect();
        }
        let Some(first) = self.rows_dirty.iter().map(|d| d.0).min() else {
            return;
        };
        let end = self
            .rows_dirty
            .iter()
            .map(|d| d.0)
            .max()
            .expect("not empty")
            + 1;
        let words = |slot: u32| slot as usize * TILE_WORDS;
        if self.tile_rows.len() < words(end) {
            self.tile_rows.resize(words(end), 0);
        }
        self.placements_written += self.rows_dirty.len() as u32;
        for (slot, row) in self.rows_dirty.drain(..) {
            self.tile_rows[words(slot)..words(slot + 1)].copy_from_slice(&row.words(quantum));
        }
        self.tables.write_words(
            gpu,
            Table::Tiles,
            words(first) as u32,
            &self.tile_rows[words(first)..words(end)],
        );
    }

    /// Placement rows that changed in the frames since the last call, which
    /// each frame wrote in one write.
    pub(crate) fn take_placements_written(&mut self) -> u32 {
        std::mem::take(&mut self.placements_written)
    }

    /// How many candidates render `node` had at the last layout.
    pub(crate) fn node_candidates(&self, node: u32) -> u32 {
        self.laid.get(node as usize).map_or(0, |n| n.live)
    }

    /// Compacted words the last rebuild hid: the slots of runs that went,
    /// which a driver writing compacted indices of its own hides too.
    #[cfg(cpu_driven)]
    pub(crate) fn hidden(&self) -> &[Range<u32>] {
        &self.hidden
    }

    /// Compacted words the last rebuild moved, which a driver writing
    /// compacted indices of its own moves too.
    #[cfg(cpu_driven)]
    pub(crate) fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// The buffer moves go through, at least `bytes` long: it grows to a
    /// power of two, and stays.
    pub(crate) fn scratch(&mut self, gpu: &mut dyn Gpu, bytes: u64) -> BufferId {
        match self.scratch {
            Some((buffer, size)) if size >= bytes => buffer,
            old => {
                if let Some((buffer, _)) = old {
                    gpu.destroy_buffer(buffer);
                }
                let size = bytes.next_power_of_two().max(1 << 16);
                let buffer = gpu.create_buffer(&BufferDesc {
                    label: "moved slots",
                    size,
                    // An index buffer, as the compacted indices are: WebGL
                    // copies between buffers of one kind only.
                    usage: BufferUsages::INDEX | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                });
                self.scratch = Some((buffer, size));
                buffer
            }
        }
    }

    /// Keeps only a tile's labels drawing: its map placements become
    /// [`PlacementTarget::Labels`] ones on this frame's surface (`globe`)
    /// and ground, and its drape placements go.
    pub(crate) fn hold_symbols(
        &mut self,
        tile: TileId,
        globe: Option<RowGlobe>,
        (dem, terrain_range): (DemRef, [f32; 2]),
    ) {
        let rows: Vec<TileRow> = self.tiles[tile.0 as usize]
            .as_ref()
            .expect("unknown tile")
            .placements
            .iter()
            .filter(|p| matches!(p.row.target, PlacementTarget::Map | PlacementTarget::Labels))
            .map(|p| TileRow {
                target: PlacementTarget::Labels,
                globe,
                dem,
                terrain_range,
                ..p.row
            })
            .collect();
        self.place_tile(tile, &rows)
            .expect("fewer placements always fit");
    }

    /// Uploads a mesh of `tile` for render `node`, with its property rows.
    /// Host-held words are written straight from the host; releasing them
    /// stays with the caller.
    /// Whether meshes of these sizes would all insert now, in order: the
    /// arena ranges and chunk slots [`insert_mesh`](Self::insert_mesh)
    /// would take, found without writing anything.
    pub(crate) fn fits(&self, meshes: impl IntoIterator<Item = MeshSize>) -> bool {
        let mut vertices = self.vertex_arena.clone();
        let mut indices = self.index_arena.clone();
        let mut props = self.prop_arena.clone();
        let mut chunks = 0u64;
        for mesh in meshes {
            if vertices.alloc(mesh.vertices).is_none()
                || indices.alloc(mesh.indices).is_none()
                || props.alloc(mesh.props).is_none()
            {
                return false;
            }
            chunks += u64::from(mesh.chunks);
        }
        chunks <= u64::from(self.chunk_slots.available())
    }

    pub(crate) fn insert_mesh(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: TileId,
        node: u32,
        mesh: &MeshData<'_>,
    ) -> Result<MeshId, OutOfSpace> {
        self.insert_mesh_sharing(gpu, tile, node, mesh, [None; 2])
    }

    /// [`insert_mesh`](Self::insert_mesh) drawing the vertices, the
    /// indices or both of earlier meshes (`shares`), whose words it
    /// leaves out of `mesh`.
    pub(crate) fn insert_mesh_sharing(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: TileId,
        node: u32,
        mesh: &MeshData<'_>,
        shares: [Option<MeshId>; 2],
    ) -> Result<MeshId, OutOfSpace> {
        assert!(
            (node as usize) < self.nodes.len(),
            "unknown render node {node}"
        );
        assert!(self.tiles[tile.0 as usize].is_some(), "unknown tile");
        let len = |v: usize| u32::try_from(v).map_err(|_| OutOfSpace);
        if !self.chunk_slots.reserve(len(mesh.chunks.len())?) {
            return Err(OutOfSpace);
        }
        // A shared range is the earlier mesh's; the others are allocated.
        let ranges = shares.map(|of| {
            of.map(|of| {
                let entry = self.meshes[of.0 as usize].as_ref().expect("live mesh");
                [entry.vertices.clone(), entry.indices.clone()]
            })
        });
        let vertices = match &ranges[0] {
            Some([vertices, _]) => Some(vertices.clone()),
            None => self.vertex_arena.alloc(mesh.vertices.len()),
        };
        let indices = match &ranges[1] {
            Some([_, indices]) => Some(indices.clone()),
            None => self.index_arena.alloc(mesh.indices.len()),
        };
        let props = self.prop_arena.alloc(mesh.props.len());
        let (Some(vertices), Some(indices), Some(prop_range)) =
            (vertices.clone(), indices.clone(), props.clone())
        else {
            if let (Some(range), None) = (vertices, &ranges[0]) {
                self.vertex_arena.free(range);
            }
            if let (Some(range), None) = (indices, &ranges[1]) {
                self.index_arena.free(range);
            }
            if let Some(range) = props {
                self.prop_arena.free(range);
            }
            return Err(OutOfSpace);
        };
        for (pool, (range, words, table)) in [
            (&vertices, mesh.vertices, Table::Vertices),
            (&indices, mesh.indices, Table::Indices),
        ]
        .into_iter()
        .enumerate()
        {
            match ranges[pool] {
                // Empty ranges all start at 0, where the first allocation
                // does: they hold nothing to share.
                Some(_) if range.is_empty() => {}
                Some(_) => *self.shared[pool].entry(range.start).or_default() += 1,
                None => self.tables.write(gpu, table, range.start, words),
            }
        }
        self.tables
            .write(gpu, Table::Props, prop_range.start, mesh.props);

        let mut chunk_ids = Vec::with_capacity(mesh.chunks.len());
        for chunk in mesh.chunks {
            let id = self.chunk_slots.alloc().expect("checked above");
            let pack = |p: [i16; 2]| u32::from(p[0] as u16) | u32::from(p[1] as u16) << 16;
            let row: [u32; CHUNK_WORDS] = [
                indices.start * 2 + chunk.first_index,
                chunk.index_count,
                vertices.start,
                chunk.first_vertex,
                prop_range.start,
                pack(chunk.bounds.min),
                pack(chunk.bounds.max),
                NO_STATE,
                mesh.row_words,
            ];
            self.tables
                .write_words(gpu, Table::Chunks, id * CHUNK_WORDS as u32, &row);
            chunk_ids.push(id);
        }

        let entry = MeshEntry {
            tile,
            node,
            chunk_slots: chunk_ids,
            chunks: mesh.chunks.to_vec(),
            vertices,
            indices,
            props: prop_range,
            row_words: mesh.row_words,
            state: 0..0,
            symbol_states: 0..0,
            symbol_copies: 0,
            symbol_order: [0; 2],
        };
        Ok(self.add_mesh(entry))
    }

    /// Registers an uploaded mesh with its node and tile.
    fn add_mesh(&mut self, entry: MeshEntry) -> MeshId {
        let (node, tile) = (entry.node, entry.tile);
        let id = match self.free_meshes.pop() {
            Some(id) => {
                self.meshes[id as usize] = Some(entry);
                MeshId(id)
            }
            None => {
                self.meshes.push(Some(entry));
                MeshId(self.meshes.len() as u32 - 1)
            }
        };
        self.mesh_serials.resize(self.meshes.len(), 0);
        self.mesh_serials[id.0 as usize] = self.next_serial;
        self.next_serial += 1;
        self.nodes[node as usize].meshes.push(id);
        self.tiles[tile.0 as usize]
            .as_mut()
            .expect("checked above")
            .meshes
            .push(id);
        self.candidates_dirty = true;
        id
    }

    /// Patches image references after uploading rows from either core or
    /// host memory. No vertex data or bind groups change when images arrive.
    pub(crate) fn set_mesh_image(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        word: u32,
        image: u32,
    ) {
        let entry = self.meshes[mesh.0 as usize].as_ref().expect("live mesh");
        assert!(word < entry.props.len() as u32);
        self.tables
            .write_words(gpu, Table::Props, entry.props.start + word, &[image]);
    }

    /// Replaces a mesh's property rows, keeping its geometry: a column job's
    /// result, `row_words` per feature. Image references are patched again
    /// by the caller.
    pub(crate) fn set_mesh_props(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        props: &Words<'_>,
        row_words: u32,
    ) -> Result<(), OutOfSpace> {
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        if row_words != entry.row_words {
            entry.row_words = row_words;
            for &chunk in &entry.chunk_slots {
                write_chunk_word(
                    gpu,
                    &mut self.tables,
                    chunk,
                    CHUNK_ROW_WORDS_WORD,
                    row_words,
                );
            }
        }
        if props.len() != entry.props.len() as u32 {
            let range = self.prop_arena.alloc(props.len()).ok_or(OutOfSpace)?;
            self.prop_arena
                .free(std::mem::replace(&mut entry.props, range.clone()));
            for &chunk in &entry.chunk_slots {
                write_chunk_word(gpu, &mut self.tables, chunk, CHUNK_PROPS_WORD, range.start);
            }
        }
        self.tables
            .write(gpu, Table::Props, entry.props.start, *props);
        Ok(())
    }

    /// Sets the feature-state index of each of a mesh's feature slots, or
    /// (empty) gives every feature the default state.
    pub(crate) fn set_mesh_state(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        indices: &[u32],
    ) -> Result<(), OutOfSpace> {
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        if indices.len() != entry.state.len() {
            let range = self
                .prop_arena
                .alloc(indices.len() as u32)
                .ok_or(OutOfSpace)?;
            self.prop_arena
                .free(std::mem::replace(&mut entry.state, range.clone()));
            for &chunk in &entry.chunk_slots {
                write_chunk_word(
                    gpu,
                    &mut self.tables,
                    chunk,
                    CHUNK_STATE_WORD,
                    state_base(&range),
                );
            }
        }
        if !indices.is_empty() {
            self.tables
                .write(gpu, Table::Props, entry.state.start, Words::Core(indices));
        }
        Ok(())
    }

    /// Words of the property buffer for a table outside any mesh, such as
    /// the feature-state table. The words are not cleared.
    pub(crate) fn alloc_table(&mut self, words: u32) -> Result<Range<u32>, OutOfSpace> {
        self.prop_arena.alloc(words).ok_or(OutOfSpace)
    }

    pub(crate) fn free_table(&mut self, range: Range<u32>) {
        self.prop_arena.free(range);
    }

    /// Writes words of a table from [`alloc_table`](Self::alloc_table),
    /// starting at word `at` of the property buffer.
    pub(crate) fn write_table(&mut self, gpu: &mut dyn Gpu, at: u32, words: &[u32]) {
        self.tables.write_words(gpu, Table::Props, at, words);
    }

    pub(crate) fn remove_mesh(&mut self, id: MeshId) {
        let tile = self.meshes[id.0 as usize]
            .as_ref()
            .expect("mesh already removed")
            .tile;
        self.tiles[tile.0 as usize]
            .as_mut()
            .expect("mesh outlived its tile")
            .meshes
            .retain(|&m| m != id);
        self.drop_mesh(id);
        self.candidates_dirty = true;
    }

    fn drop_mesh(&mut self, id: MeshId) {
        let entry = self.meshes[id.0 as usize]
            .take()
            .expect("mesh already removed");
        self.free_meshes.push(id.0);
        self.nodes[entry.node as usize].meshes.retain(|&m| m != id);
        for chunk in entry.chunk_slots {
            self.chunk_slots.free(chunk);
        }
        if self.release_shared(0, &entry.vertices) {
            self.vertex_arena.free(entry.vertices);
        }
        if self.release_shared(1, &entry.indices) {
            self.index_arena.free(entry.indices);
        }
        self.prop_arena.free(entry.props);
        self.prop_arena.free(entry.state);
        self.prop_arena.free(entry.symbol_states);
    }

    /// Drops a mesh's hold on a vertex (`pool` 0) or index (1) range:
    /// whether it was the last, and the range is free.
    fn release_shared(&mut self, pool: usize, range: &Range<u32>) -> bool {
        // An empty range (`0..0`: a point node's indices, a symbol's) is
        // no share of the one starting at 0.
        if range.is_empty() {
            return true;
        }
        let Some(others) = self.shared[pool].get_mut(&range.start) else {
            return true;
        };
        *others -= 1;
        if *others == 0 {
            self.shared[pool].remove(&range.start);
        }
        false
    }

    /// Writes the frame block, and the candidate lists if meshes or
    /// placements changed. Returns the number of candidates, and whether
    /// they were laid out again.
    pub(crate) fn prepare(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        linear: bool,
    ) -> (u32, bool) {
        self.write_rows(gpu, quantum(frame.world_span));
        self.moves.clear();
        let rebuilt = self.candidates_dirty;
        if self.candidates_dirty {
            self.candidates_dirty = false;
            // A list moved to a larger buffer is written whole, and the
            // cull writes every compacted slot again: they need no copy,
            // only the candidates laid out again. Room in one can show a want of it
            // in the other (candidates left out for the compacted words
            // weren't counted against the lists), so both grow until
            // everything fits or neither can.
            loop {
                let (compacted, candidates) = self.rebuild_candidates(gpu);
                let grew = (compacted > 0 && self.grow_compacted(gpu, compacted))
                    | (candidates > 0 && self.grow_candidates(gpu, candidates));
                if !grew {
                    break;
                }
            }
            if self.shrink_lists(gpu) {
                self.rebuild_candidates(gpu);
            }
        } else if std::mem::take(&mut self.params_dirty) && !self.node_rows.is_empty() {
            self.tables
                .write_words(gpu, Table::NodeCull, 0, &self.node_rows);
        }

        let words = frame_words(frame, linear);
        gpu.write_buffer(self.buffers.frame, 0, bytemuck::cast_slice(&words));
        #[cfg(cpu_driven)]
        {
            self.frame = words;
        }
        (self.candidate_count, rebuilt)
    }

    /// Lays out the candidates again: only what changed where every node's
    /// slices hold it, else everything (`candidates`). Returns the
    /// compacted index words and candidates left out for want of room.
    fn rebuild_candidates(&mut self, gpu: &mut dyn Gpu) -> (u64, u64) {
        if self.lay_out_changes(gpu) {
            return (0, 0);
        }
        self.lay_out(gpu)
    }

    /// Grows the vertex and index pools, each doubling to its ceiling (the
    /// device's buffer limits), until `meshes` would insert.
    /// Returns whether they would.
    pub(crate) fn grow_for(&mut self, gpu: &mut dyn Gpu, meshes: &[MeshSize]) -> bool {
        loop {
            if self.fits(meshes.iter().copied()) {
                return true;
            }
            let short = |arena: &Arena, words: fn(&MeshSize) -> u32| {
                let mut arena = arena.clone();
                meshes.iter().any(|m| arena.alloc(words(m)).is_none())
            };
            let vertices = short(&self.vertex_arena, |m| m.vertices);
            let indices = short(&self.index_arena, |m| m.indices);
            let props = short(&self.prop_arena, |m| m.props);
            let mut grew = false;
            if vertices {
                grew |= self.grow_pool(gpu, Pool::Vertices);
            }
            if indices {
                grew |= self.grow_pool(gpu, Pool::Indices);
            }
            if props {
                grew |= self.grow_pool(gpu, Pool::Props);
            }
            let chunks: u64 = meshes.iter().map(|m| u64::from(m.chunks)).sum();
            let available = u64::from(self.chunk_slots.available());
            if chunks > available {
                let needed = u64::from(self.capacity.chunks) + chunks - available;
                grew |= self.grow_chunks(gpu, needed);
            }
            if !grew {
                return false;
            }
        }
    }

    /// Grows what refused an upload whose sizes aren't known ahead: each
    /// pool, and the chunk table, that refused it since the last grow, to
    /// the smallest doubling that would have held what it refused, within
    /// its ceiling. Returns whether any grew; an upload retries until it
    /// fits or nothing grows.
    pub(crate) fn grow(&mut self, gpu: &mut dyn Gpu) -> bool {
        let mut grew = false;
        for pool in [Pool::Vertices, Pool::Indices, Pool::Props] {
            let (old, max) = self.pool_words(pool);
            let needed = self.arena_mut(pool).take_needed();
            let new = grown(old, u64::from(needed), max);
            if needed > old && new > old {
                self.resize_pool(gpu, pool, new);
                grew = true;
            }
        }
        let chunks = self.chunk_slots.take_needed();
        grew | (chunks > self.capacity.chunks && self.grow_chunks(gpu, u64::from(chunks)))
    }

    fn arena_mut(&mut self, pool: Pool) -> &mut Arena {
        match pool {
            Pool::Vertices => &mut self.vertex_arena,
            Pool::Indices => &mut self.index_arena,
            Pool::Props => &mut self.prop_arena,
        }
    }

    /// Forgets the pools' and chunk table's refusals: they are for the
    /// upload retrying after them, and one that made room another way
    /// asks no growth later.
    fn forget_refusals(&mut self) {
        for pool in [Pool::Vertices, Pool::Indices, Pool::Props] {
            self.arena_mut(pool).take_needed();
        }
        self.chunk_slots.take_needed();
    }

    /// Moves a pool into a buffer twice its size (or its ceiling's): the
    /// contents copy over on the GPU, at the same offsets.
    fn grow_pool(&mut self, gpu: &mut dyn Gpu, pool: Pool) -> bool {
        let (old, max) = self.pool_words(pool);
        let new = grown(old, u64::from(old) + 1, max);
        new > old && {
            self.resize_pool(gpu, pool, new);
            true
        }
    }

    /// Halves the vertex, index and property pools and the chunk table
    /// while what they hold all lies in their lowest quarter, down to their
    /// first sizes: a view that emptied gives back what it grew them to,
    /// and each keeps room to grow fourfold before it doubles again.
    /// Returns whether any shrank.
    pub(crate) fn shrink(&mut self, gpu: &mut dyn Gpu) -> bool {
        self.forget_refusals();
        let mut shrank = false;
        let first = [
            self.first.vertex_words,
            self.first.index_words,
            self.first.prop_words,
        ];
        for (pool, first) in [Pool::Vertices, Pool::Indices, Pool::Props]
            .into_iter()
            .zip(first)
        {
            let (old, _) = self.pool_words(pool);
            let held = match pool {
                Pool::Vertices => self.vertex_arena.free_from(old),
                Pool::Indices => self.index_arena.free_from(old),
                Pool::Props => self.prop_arena.free_from(old),
            };
            let new = shrunk(old, held, first);
            if new < old {
                self.resize_pool(gpu, pool, new);
                shrank = true;
            }
        }
        let old = self.capacity.chunks;
        let new = shrunk(old, self.chunk_slots.held_below(), self.first.chunks);
        if new < old {
            let words = |count: u32| u64::from(count) * CHUNK_WORDS as u64;
            let kept = words(self.chunk_slots.held_below());
            self.tables.resize(gpu, Table::Chunks, words(new), kept);
            self.chunk_slots.shrink(new);
            self.capacity.chunks = new;
            self.buffers_revision += 1;
            shrank = true;
        }
        shrank
    }

    /// A pool's size and ceiling, in words.
    fn pool_words(&self, pool: Pool) -> (u32, u32) {
        let c = &self.capacity;
        match pool {
            Pool::Vertices => (c.vertex_words, c.max_vertex_words),
            Pool::Indices => (c.index_words, c.max_index_words),
            Pool::Props => (c.prop_words, c.max_prop_words),
        }
    }

    /// Moves a pool into a buffer of `new` words, its contents below both
    /// sizes copied over on the GPU at the same offsets.
    fn resize_pool(&mut self, gpu: &mut dyn Gpu, pool: Pool, new: u32) {
        let c = &mut self.capacity;
        let (words, table, arena) = match pool {
            Pool::Vertices => (&mut c.vertex_words, Table::Vertices, &mut self.vertex_arena),
            Pool::Indices => (&mut c.index_words, Table::Indices, &mut self.index_arena),
            Pool::Props => (&mut c.prop_words, Table::Props, &mut self.prop_arena),
        };
        let old = *words;
        let keep = u64::from(old.min(new));
        self.tables.resize(gpu, table, u64::from(new), keep);
        if new > old {
            arena.grow(old, new);
        } else {
            arena.shrink(old, new);
        }
        *words = new;
        self.buffers_revision += 1;
    }

    /// Moves the chunk table into the smallest doubling of at least
    /// `needed` rows, to its ceiling, its rows copied over. Returns whether
    /// it grew.
    fn grow_chunks(&mut self, gpu: &mut dyn Gpu, needed: u64) -> bool {
        let old = self.capacity.chunks;
        let new = grown(old, needed, self.capacity.max_chunks);
        if new <= old {
            return false;
        }
        let words = |count: u32| u64::from(count) * CHUNK_WORDS as u64;
        self.tables
            .resize(gpu, Table::Chunks, words(new), words(old));
        self.chunk_slots.grow(new);
        self.capacity.chunks = new;
        self.buffers_revision += 1;
        true
    }

    /// Makes the candidate and visible lists room for at least `short`
    /// more candidates, to their ceiling. The next layout writes them
    /// whole.
    fn grow_candidates(&mut self, gpu: &mut dyn Gpu, short: u64) -> bool {
        let old = self.capacity.candidates;
        let new = grown(old, u64::from(old) + short, self.capacity.max_candidates);
        new > old && {
            self.resize_candidates(gpu, new);
            true
        }
    }

    /// Moves the candidate and visible lists into buffers of `new`
    /// candidates, nothing copied: the next layout writes them whole.
    fn resize_candidates(&mut self, gpu: &mut dyn Gpu, new: u32) {
        for (table, words) in [
            (Table::Candidates, CANDIDATE_WORDS),
            (Table::Visible, VISIBLE_WORDS),
        ] {
            self.tables
                .resize(gpu, table, u64::from(new) * words as u64, 0);
        }
        self.capacity.candidates = new;
        self.laid.clear();
        self.buffers_revision += 1;
    }

    /// Grows the compacted index buffer by at least `short` words, to its
    /// ceiling.
    fn grow_compacted(&mut self, gpu: &mut dyn Gpu, short: u64) -> bool {
        let old = self.capacity.compacted_words;
        let max = self.capacity.max_compacted_words;
        let new = grown(old, u64::from(old) + short, max);
        new > old && {
            self.resize_compacted(gpu, new);
            true
        }
    }

    /// Moves the compacted indices into a buffer of `new` words, nothing
    /// copied: the next cull writes every slot of laid-out candidates.
    /// What moves slots goes too, to be made again to the new size's
    /// moves.
    fn resize_compacted(&mut self, gpu: &mut dyn Gpu, new: u32) {
        let desc = BufferDesc {
            label: "compacted indices",
            size: u64::from(new) * 4,
            usage: self.driver.compacted_usage(),
        };
        move_buffer(gpu, &mut self.buffers.compacted, &desc, 0);
        if let Some((scratch, _)) = self.scratch.take() {
            gpu.destroy_buffer(scratch);
        }
        self.capacity.compacted_words = new;
        self.laid.clear();
        self.buffers_revision += 1;
    }

    /// Halves the candidate lists and the compacted indices while what the
    /// candidates take lies in their lowest quarter, down to their first
    /// sizes, as the pools shrink. Returns whether either shrank: the
    /// candidates are then laid out again.
    fn shrink_lists(&mut self, gpu: &mut dyn Gpu) -> bool {
        let (c, first) = (self.capacity, self.first);
        let compacted = shrunk(
            c.compacted_words,
            self.compacted_words,
            first.compacted_words,
        );
        let candidates = shrunk(c.candidates, self.rows_end, first.candidates);
        if compacted < c.compacted_words {
            self.resize_compacted(gpu, compacted);
        }
        if candidates < c.candidates {
            self.resize_candidates(gpu, candidates);
        }
        compacted < c.compacted_words || candidates < c.candidates
    }

    /// GPU bytes a mesh takes in the pools.
    pub(crate) fn mesh_bytes(&self, mesh: MeshId) -> u64 {
        self.meshes[mesh.0 as usize].as_ref().map_or(0, |m| {
            (m.vertices.len()
                + m.indices.len()
                + m.props.len()
                + m.state.len()
                + m.symbol_states.len()) as u64
                * 4
        })
    }

    /// Changes whenever a pool moved into a larger buffer.
    pub(crate) fn buffers_revision(&self) -> u64 {
        self.buffers_revision
    }

    /// The scene's GPU memory by pool.
    pub(crate) fn memory(&self) -> SceneMemory {
        let c = &self.capacity;
        let bytes = |words: u32| u64::from(words) * 4;
        let pool = |arena: &Arena, allocated: u32, ceiling: u32| PoolMemory {
            used: bytes(arena.used()),
            allocated: bytes(allocated),
            ceiling: bytes(ceiling.max(allocated)),
        };
        let props = bytes(c.prop_words);
        SceneMemory {
            vertices: pool(&self.vertex_arena, c.vertex_words, c.max_vertex_words),
            indices: pool(&self.index_arena, c.index_words, c.max_index_words),
            compacted: PoolMemory {
                used: bytes(self.compacted_words),
                allocated: bytes(c.compacted_words),
                ceiling: bytes(c.max_compacted_words.max(c.compacted_words)),
            },
            props: PoolMemory {
                used: bytes(self.prop_arena.used()),
                allocated: props,
                ceiling: bytes(c.max_prop_words.max(c.prop_words)),
            },
            tables: (u64::from(c.placements) * TILE_WORDS as u64
                + u64::from(c.chunks) * CHUNK_WORDS as u64
                + u64::from(c.candidates) * (CANDIDATE_WORDS + VISIBLE_WORDS) as u64
                + u64::from(c.nodes) * (NODE_CULL_WORDS + ARGS_WORDS) as u64)
                * 4
                + self.scratch.map_or(0, |(_, bytes)| bytes),
        }
    }
}

/// The pools that grow by copying into a larger buffer.
#[derive(Clone, Copy, Debug)]
enum Pool {
    Vertices,
    Indices,
    Props,
}

/// `old` halved while `held` lies in its lowest quarter, no lower than
/// `first`: a capacity's size once what it held went, keeping room to grow
/// fourfold before it doubles again.
fn shrunk(old: u32, held: u32, first: u32) -> u32 {
    let mut new = old;
    while new > first && held <= new / 4 {
        new = (new / 2).max(first);
    }
    new
}

/// Lowers `capacity`'s chunk and candidate ceilings, and first sizes,
/// until the row tables fit one data texture of `side` rows `width` words
/// wide, the texture backing's (`tables::RowTexture`): halving whichever
/// takes more rows. At the ceilings they take about 2,830 rows of 4096,
/// which a device of 4096 or more holds whole; a device at WebGL 2's least
/// side, 2048 rows of 2048, holds fewer chunks at once, and drops what
/// doesn't fit as at any ceiling.
#[cfg(cpu_driven)]
fn fit_row_texture(capacity: &mut SceneCapacity, side: u32, width: u32) {
    let c = capacity;
    let rows = |count: u32, per: usize| tables::rows(u64::from(count) * per as u64, width);
    let fixed = rows(c.placements, TILE_WORDS) + rows(c.nodes, NODE_CULL_WORDS);
    loop {
        let chunks = c.max_chunks.max(c.chunks);
        let candidates = c.max_candidates.max(c.candidates);
        let (chunk_rows, visible_rows) =
            (rows(chunks, CHUNK_WORDS), rows(candidates, VISIBLE_WORDS));
        if fixed + chunk_rows + visible_rows <= side || chunks.max(candidates) <= 1 {
            return;
        }
        // The one taking more rows, of those that can still halve.
        let (max, start, count) = if chunks > 1 && (chunk_rows >= visible_rows || candidates <= 1) {
            (&mut c.max_chunks, &mut c.chunks, chunks)
        } else {
            (&mut c.max_candidates, &mut c.candidates, candidates)
        };
        *max = count / 2;
        *start = (*start).min(*max);
    }
}

/// The smallest of `old`'s doublings (from 1 when it is 0) that holds
/// `need`, at most `max`: a capacity's next size.
fn grown(old: u32, need: u64, max: u32) -> u32 {
    let mut new = u64::from(old.max(1));
    while new < need {
        new *= 2;
    }
    new.min(u64::from(max)) as u32
}

/// Replaces `buffer` with a new one, its first `keep` bytes copied over on
/// the GPU at the same offsets, and destroys the old one.
fn move_buffer(gpu: &mut dyn Gpu, buffer: &mut BufferId, desc: &BufferDesc<'_>, keep: u64) {
    let moved = gpu.create_buffer(desc);
    if keep > 0 {
        gpu.submit(&[orb_hal::Pass::CopyBuffer {
            source: *buffer,
            source_offset: 0,
            destination: moved,
            destination_offset: 0,
            size: keep,
        }]);
    }
    gpu.destroy_buffer(std::mem::replace(buffer, moved));
}

#[cfg(test)]
mod tests {
    use super::{FrameLight, INDEXED_BATCH, grown};

    /// With the style's ambient light a shadow takes away the sun's share
    /// of the light on the ground, so a sun under a bright sky casts faint
    /// shadows; without it, as dark as the intensity says.
    #[test]
    fn shadows_take_away_the_suns_share() {
        let light = |ambient: f32, amount| FrameLight {
            color: [1.0; 3],
            intensity: 0.5,
            // Overhead.
            position: [0.0, 0.0, 1.0],
            shadow_intensity: 0.8,
            ambient: [ambient; 3],
            ambient_amount: amount,
            ground: None,
        };
        assert_eq!(light(0.0, 0.0).shadow().1, 0.8);
        // No sky light: the sun's whole.
        assert_eq!(light(0.0, 1.0).shadow().1, 0.8);
        // As bright a sky as the sun: half.
        assert!((light(0.5, 1.0).shadow().1 - 0.4).abs() < 1e-6);
        assert!((light(0.5, 0.5).shadow().1 - 0.6).abs() < 1e-6);
    }

    /// The cull shader's constants are the scene's: its batches, and the
    /// most indices a chunk has, which bounds its copy loop.
    #[test]
    fn the_cull_shares_its_constants() {
        let source = orb_shaders::CULL_SHADER.source();
        for (name, value) in [
            ("INDEXED_BATCH", INDEXED_BATCH),
            ("CHUNK_INDICES", orb_layout::CHUNK_INDICES),
        ] {
            assert!(
                source.contains(&format!("const {name}: u32 = {value}u;")),
                "cull.wgsl's {name} isn't {value}"
            );
        }
    }

    #[test]
    fn capacities_double_to_their_need_and_ceiling() {
        assert_eq!(grown(4, 5, 100), 8);
        assert_eq!(grown(4, 33, 100), 64);
        assert_eq!(grown(4, 330, 100), 100);
        // From nothing, without spinning.
        assert_eq!(grown(0, 3, 100), 4);
        assert_eq!(grown(0, 1, 100), 1);
        assert_eq!(
            grown(u32::MAX / 2 + 1, u64::from(u32::MAX) + 1, u32::MAX),
            u32::MAX
        );
    }

    /// The row tables fit one texture at any WebGL 2 device's side, and
    /// keep their ceilings where they already fit.
    #[cfg(cpu_driven)]
    #[test]
    fn row_tables_fit_any_texture_side() {
        use super::{
            CHUNK_WORDS, NODE_CULL_WORDS, SceneCapacity, TILE_WORDS, VISIBLE_WORDS,
            fit_row_texture, tables::rows,
        };
        let total = |c: &SceneCapacity, width: u32| {
            let rows = |count: u32, per: usize| rows(u64::from(count) * per as u64, width);
            rows(c.placements, TILE_WORDS)
                + rows(c.nodes, NODE_CULL_WORDS)
                + rows(c.max_chunks.max(c.chunks), CHUNK_WORDS)
                + rows(c.max_candidates.max(c.candidates), VISIBLE_WORDS)
        };
        for start in [SceneCapacity::growable(), SceneCapacity::fixed()] {
            for side in [4096, 8192] {
                let mut c = start;
                fit_row_texture(&mut c, side, 4096);
                assert_eq!(c, start);
            }
            let mut c = start;
            fit_row_texture(&mut c, 2048, 2048);
            assert!(total(&c, 2048) <= 2048, "{} rows", total(&c, 2048));
            assert!(c.max_chunks.max(c.chunks) >= 1 << 16);
            assert!(c.max_candidates.max(c.candidates) >= 1 << 16);
            // A side the fixed tables fill halves both down to one, and ends.
            let mut c = start;
            fit_row_texture(&mut c, 1, 2048);
            assert!(c.max_chunks.max(c.chunks) <= 1 && c.max_candidates.max(c.candidates) <= 1);
        }
    }
}

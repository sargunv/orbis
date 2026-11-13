//! A map's options and their defaults, which every front end (orb-api,
//! orb-web) starts from.

use orb_camera::{CameraState, TileLod};
use orb_style::Previews;

use crate::{Map, MissingTiles};

/// How a map starts. Each option but the camera can change later, on its
/// own setter or all at once with [`Map::set_options`].
#[derive(Clone, Debug, PartialEq)]
pub struct MapOptions {
    /// The camera; the first style's when `None`, as GL JS does.
    pub camera: Option<CameraState>,
    /// The backing colour behind transparent style pixels, premultiplied
    /// RGBA; opaque surfaces want their view's background. Transparent by
    /// default, like GL JS's canvas.
    pub clear_color: [f64; 4],
    /// How long labels fade in and out, in milliseconds (GL JS
    /// `fadeDuration`); still output doesn't fade.
    pub fade_duration: f64,
    /// Whether labels of different sources collide with each other (GL JS
    /// `crossSourceCollisions`).
    pub cross_source_collisions: bool,
    /// The CSS font family ideographs of the default font stack are drawn
    /// with instead of the style's glyphs (GL JS
    /// `localIdeographFontFamily`); `None` draws the glyphs.
    pub local_ideograph_font_family: Option<String>,
    /// Font files every font stack draws with after the style's own
    /// `font-faces` (MapLibre Native's `fallbackFontFaces`). None by
    /// default.
    pub fallback_font_faces: Vec<orb_style::FontFace>,
    /// 4× multisampling for 3D surfaces: extrusion layers, and on a
    /// style with terrain, the map's passes, which it draws in (GL JS
    /// `antialias`, which multisamples its whole canvas). Off by default,
    /// as in GL JS (MapLibre Native never multisamples): off, their edges
    /// step visibly at a pixel ratio of 1 and barely at 2, and on some
    /// tiling GPUs it costs more than the rest of the frame (under D3D12
    /// on the Adreno X1, a pitched city's frame takes 2.4× as long with
    /// it).
    pub antialias: bool,
    /// Whether terrain tiles hang skirts that hide the cracks between
    /// tiles of different detail; translucent terrain shows them (GL JS
    /// `terrainSkirtLength`).
    pub terrain_skirts: bool,
    /// The previews its styles may use: new style syntax beyond the
    /// approved v8 spec, which warn and fall back to their defaults while
    /// off. None by default.
    pub previews: Previews,
    /// Whether labels keep their place while the camera moves, as
    /// MapLibre Native keeps them: a label tries its previous variable
    /// anchor first (in tilted views, only that anchor), and in tilted
    /// views labels shown before are placed before new and hidden ones.
    /// What labels show then depends on how the camera got there; off, it
    /// depends on the camera alone. On by default; still and tile output
    /// places afresh either way.
    pub stable_labels: bool,
    /// How tiles coarsen with distance in pitched views: MapLibre Native's
    /// tile level of detail (maplibre-native-ffi's tile options), whose
    /// defaults it takes.
    pub tile_lod: TileLod,
    /// What a vector tile the tileset doesn't have (HTTP 404) draws: its
    /// nearest ancestor by default, or nothing, as GL JS and MapLibre
    /// Native draw it. Raster and raster-dem tiles draw their ancestor
    /// either way ([`MissingTiles`]).
    pub missing_tiles: MissingTiles,
}

impl Default for MapOptions {
    fn default() -> Self {
        Self {
            camera: None,
            clear_color: [0.0; 4],
            fade_duration: 300.0,
            cross_source_collisions: true,
            local_ideograph_font_family: Some("sans-serif".into()),
            fallback_font_faces: Vec::new(),
            antialias: false,
            terrain_skirts: true,
            previews: Previews::NONE,
            stable_labels: true,
            tile_lod: TileLod::default(),
            missing_tiles: MissingTiles::Parent,
        }
    }
}

impl Map {
    /// Sets every option but the camera, which only a new map takes
    /// ([`Map::new`]; move it with the camera's calls after). Options that
    /// apply to styles or tiles loaded afterwards say so on their setters.
    pub fn set_options(&mut self, options: &MapOptions) {
        self.set_clear_color(options.clear_color);
        self.set_fade_duration(options.fade_duration);
        self.set_cross_source_collisions(options.cross_source_collisions);
        self.set_local_ideograph_font_family(options.local_ideograph_font_family.as_deref());
        self.set_fallback_font_faces(options.fallback_font_faces.clone());
        self.set_antialias_3d(options.antialias);
        self.set_terrain_skirts(options.terrain_skirts);
        self.set_previews(options.previews);
        self.set_stable_labels(options.stable_labels);
        self.set_tile_lod(options.tile_lod);
        self.set_missing_tiles(options.missing_tiles);
    }
}

//! The style: one document of sources, layers, light, atmosphere,
//! terrain and surface, which the [`v8`](crate::v8) front end builds.

use orb_expr::{Diagnostic, Value};

use crate::{Atmosphere, FontFaces, Layer, Light, Source, Sprite, Terrain, Transition};

/// The style model: what a v8 style parses into.
#[derive(Clone, Debug)]
pub struct Style {
    pub name: Option<String>,
    /// The default camera: `[longitude, latitude]`.
    pub center: Option<[f64; 2]>,
    pub zoom: Option<f64>,
    pub bearing: f64,
    pub pitch: f64,
    /// The default camera's roll about the view axis, degrees.
    pub roll: f64,
    /// How paint changes animate unless a property says otherwise.
    pub transition: Transition,
    pub sprites: Vec<Sprite>,
    /// The glyph PBF URL template, with `{fontstack}` and `{range}`.
    pub glyphs: Option<String>,
    /// Font files for `text-font` names (the root `font-faces`).
    pub font_faces: FontFaces,
    /// v8's `light`.
    pub light: Light,
    /// The root `state`: each global state property's default.
    pub state: Vec<(String, Value<'static>)>,
    /// The atmosphere: v8's `sky` lowered.
    pub atmosphere: Option<Atmosphere>,
    pub terrain: Option<Terrain>,
    /// The low-zoom surface the plane's tiles draw on, from the root
    /// `projection`: a globe, `vertical-perspective`, a world plane, a CRS
    /// with a `transition`, or a zoom interpolation from one to `mercator`;
    /// `None` draws the plane alone.
    pub globe: Option<orb_crs::Globe>,
    /// The plane the style draws on: the root `projection`'s CRS (such as
    /// `EPSG:3413`) when it has no `transition`; `None` for Web Mercator.
    pub projection: Option<orb_crs::Projection>,
    pub sources: Vec<std::sync::Arc<Source>>,
    /// Shared, so a snapshot that changes some layers (an edit, a frame's
    /// transitions) copies only those.
    pub layers: Vec<std::sync::Arc<Layer>>,
    /// What parsed but isn't supported or rendered yet.
    pub warnings: Vec<Diagnostic>,
}
impl Style {
    pub fn source(&self, id: &str) -> Option<&Source> {
        self.sources.iter().find(|s| s.id == id).map(|s| &**s)
    }

    /// Sources' tile matrix sets by source id; the rest use
    /// `WebMercatorQuad`, GeoJSON the plane's own.
    pub fn tile_matrix_sets(&self) -> Vec<(String, orb_crs::TileMatrixSet)> {
        self.sources
            .iter()
            .filter_map(|s| Some((s.id.clone(), s.tile_matrix_set.clone()?)))
            .collect()
    }
}

//! The scene: one document of sources, layers, lights, atmosphere,
//! terrain and globe, which `orb-v8` builds from a v8 style and the scene
//! document that extends it.

use orb_expr::{Diagnostic, Value};

use crate::{Atmosphere, FontFaces, Layer, Lights, Source, Sprite, Terrain, Transition};

/// Which document a style is: v8 as it is, or a style a scene document
/// extends (plan §11.1), whose layers may be the scene's types (`model`)
/// and whose root may declare `models`. v8 documents never mean more than
/// v8 says.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dialect {
    #[default]
    V8,
    Scene,
}

/// The native scene document (plan §11.2): what a v8 style and a scene
/// document that extends it parse into, merged.
#[derive(Clone, Debug)]
pub struct Scene {
    pub dialect: Dialect,
    /// glTF models `model` layers draw, by id: their URLs (scene dialect).
    pub models: Vec<(String, String)>,
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
    /// The lights: v8's `light` lowered (decision 0026).
    pub lights: Lights,
    /// The root `state`: each global state property's default.
    pub state: Vec<(String, Value<'static>)>,
    /// The atmosphere: v8's `sky` lowered (decision 0025).
    pub atmosphere: Option<Atmosphere>,
    pub terrain: Option<Terrain>,
    /// The low-zoom surface the plane's tiles draw on: the scene
    /// document's, else the root `projection`'s (a globe,
    /// `vertical-perspective` or a zoom interpolation from one to
    /// `mercator`); `None` draws the plane alone.
    pub globe: Option<orb_crs::Globe>,
    /// The plane the scene draws on (the scene document's `projection`);
    /// `None` for Web Mercator.
    pub projection: Option<orb_crs::Projection>,
    /// Sources' tile matrix sets by source id (the scene document's); the
    /// rest use `WebMercatorQuad`, GeoJSON the plane's own.
    pub tile_matrix_sets: Vec<(String, orb_crs::TileMatrixSet)>,
    pub sources: Vec<Source>,
    pub layers: Vec<Layer>,
    /// What parsed but isn't supported or rendered yet.
    pub warnings: Vec<Diagnostic>,
}
impl Scene {
    pub fn source(&self, id: &str) -> Option<&Source> {
        self.sources.iter().find(|s| s.id == id)
    }
}

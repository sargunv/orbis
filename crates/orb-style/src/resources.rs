//! What the scene loads beside its sources: sprites, fonts and the
//! terrain's DEM.

/// One sprite sheet. The default namespace exposes unprefixed image names;
/// other namespaces expose `id:name`, following GL JS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sprite {
    pub id: String,
    pub url: String,
}

/// A declared font file.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FontFace {
    pub url: String,
    /// The inclusive code point ranges it may draw (all, without a
    /// `unicode-range`).
    pub ranges: Vec<[u32; 2]>,
}
impl FontFace {
    pub fn covers(&self, code_point: u32) -> bool {
        self.ranges
            .iter()
            .any(|&[start, end]| (start..=end).contains(&code_point))
    }
}
/// The declared files by font name, in the style's order.
pub type FontFaces = Vec<(String, Vec<FontFace>)>;

#[derive(Clone, Debug, PartialEq)]
pub struct Terrain {
    /// The id of a `raster-dem` source.
    pub source: String,
    /// Multiplies elevations; 0 is flat.
    pub exaggeration: f64,
}
impl Terrain {
    pub const DEFAULT_EXAGGERATION: f64 = 1.0;
}

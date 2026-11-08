//! What tile jobs and the renderer agree on: the family a render node
//! draws, how a colour packs into a property row, and the images rows name.

/// The layout family a render node draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    FillExtrusion,
    Fill,
    /// A fill's antialiasing outline: 1 px line segments, drawn with fill
    /// paint (outline colour and opacity).
    FillOutline,
    Line,
    Circle,
    Heatmap,
    /// Image tiles of a `raster` source.
    Raster,
    /// Slopes of a `raster-dem` source's tiles.
    Hillshade,
    /// Elevations of a `raster-dem` source's tiles through a colour ramp.
    ColorRelief,
    /// A symbol layer's icons.
    SymbolIcon,
    /// A symbol layer's text.
    SymbolText,
}

impl Family {
    /// Whether the family draws image tiles from a page.
    pub fn is_image(self) -> bool {
        matches!(self, Self::Raster | Self::Hillshade | Self::ColorRelief)
    }

    /// Whether the family draws symbol quads.
    pub fn is_symbol(self) -> bool {
        matches!(self, Self::SymbolIcon | Self::SymbolText)
    }

    /// Whether terrain drapes the family's layers over its surface rather
    /// than drawing them in 3D or on screen (GL JS's render to texture).
    pub fn is_draped(self) -> bool {
        matches!(
            self,
            Self::Fill
                | Self::FillOutline
                | Self::Line
                | Self::Raster
                | Self::Hillshade
                | Self::ColorRelief
        )
    }
}

/// Packs a premultiplied colour into a property row word (rgba8).
pub fn pack_color(color: [f32; 4]) -> u32 {
    let [r, g, b, a] = color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u32);
    r | g << 8 | b << 16 | a << 24
}

#[test]
fn packs_colors_as_rgba8() {
    assert_eq!(pack_color([1.0, 0.0, 0.5, 1.0]), 0xff80_00ff);
}

/// Deferred image identity, resolved only on the scene side. Generated
/// textures cannot collide with sprite names. Dash values are f32 bits.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ImageKey {
    Name(String),
    Dash { array: Vec<u32>, round: bool },
    Ramp { colors: Vec<u32>, nearest: bool },
}

/// Sprite metadata retained for symbol layout as well as pattern sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageMetadata {
    pub pixel_ratio: f32,
    pub sdf: bool,
    pub stretch_x: Vec<[f32; 2]>,
    pub stretch_y: Vec<[f32; 2]>,
    pub content: Option<[f32; 4]>,
    pub text_fit_width: Option<String>,
    pub text_fit_height: Option<String>,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        Self {
            pixel_ratio: 1.0,
            sdf: false,
            stretch_x: Vec::new(),
            stretch_y: Vec::new(),
            content: None,
            text_fit_width: None,
            text_fit_height: None,
        }
    }
}

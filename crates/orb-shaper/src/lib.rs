//! `shaper.wasm`: the `font-faces` files of a style, drawn
//! with HarfRust and skrifa as natively, so labels lay out identically.
//! Job workers load it the first time a style declares a file, and
//! `orb-web`'s `ShaperFiles` calls it as the map's
//! `orb_host::FontFiles`. Its module is separate from `core.wasm`, so maps
//! without `font-faces` never download it.

#![forbid(unsafe_code)]

use orb_text::font::Files;
use orb_text::sdf::InkBox;
use wasm_bindgen::prelude::*;

/// Open font files by URL.
#[wasm_bindgen]
#[derive(Debug, Default)]
pub struct Shaper {
    files: Files,
}

#[wasm_bindgen]
impl Shaper {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens `url`'s bytes: TrueType, OpenType, WOFF or WOFF2.
    pub fn open(&mut self, url: &str, bytes: &[u8]) -> Result<(), JsError> {
        self.files.open(url, bytes).map_err(|e| JsError::new(&e))
    }

    /// `text` shaped with `url` at 48 px: advance, then the ink's ascent,
    /// descent, left and right.
    pub fn measure(&mut self, url: &str, text: &str) -> Vec<f64> {
        let m = self.files.measure(url, text);
        vec![m.advance, m.ascent, m.descent, m.left, m.right]
    }

    /// The alpha coverage of the ink box, row-major.
    pub fn coverage(
        &mut self,
        url: &str,
        text: &str,
        left: i32,
        top: i32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let ink = InkBox {
            glyph_width: width,
            glyph_height: height,
            glyph_top: top,
            glyph_left: left,
            advance: 0.0,
        };
        self.files.coverage(url, text, &ink)
    }
}

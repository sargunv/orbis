//! Text for map labels: itemization and bidi,
//! grapheme and word segmentation, the glyph chain, advance shaping and
//! line breaking at parity with the pinned GL JS, and glyph SDFs.
//!
//! A job-side core crate: no I/O, threads or GPU. Glyph PBF bytes arrive
//! from the runtime's host `Fetch`; the scene owns the glyph atlas.
//!
//! The flow for one tile: build each feature's [`Label`] (applying
//! [`transform_text`]), collect [`GlyphRequests`], resolve each request
//! through [`provider`] into a [`GlyphMap`], then [`shape_text`] each label.

#![forbid(unsafe_code)]

mod bidi;
#[cfg(feature = "fonts")]
pub mod font;
mod glyph;
mod label;
pub mod pbf;
pub mod sdf;
mod segment;
mod shaping;
#[cfg(test)]
mod tests;
mod transform;
pub mod unicode;

pub use glyph::{
    FontId, FontPolicy, FontStack, Glyph, GlyphId, GlyphKey, GlyphMap, GlyphMetrics, GlyphRequest,
    GlyphRequests, LocalFont, Provider, Variant, glyphs_url, provider,
};
pub use label::{Content, IMAGE_PADDING, Images, InlineImage, Label, Section, VerticalAlign};
pub use segment::{HostWords, WordBreaker};
pub use shaping::{
    Anchor, DEFAULT_OFFSET, GlyphSource, Justify, ONE_EM, PositionedGlyph, PositionedLine,
    ShapeContext, ShapeOptions, ShapedSection, Shaping, WritingMode, shape_text,
};
pub use transform::{TextTransform, transform_text};

#[cfg(all(test, target_arch = "wasm32"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

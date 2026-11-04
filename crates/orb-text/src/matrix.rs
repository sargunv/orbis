//! The right-to-left and complex-script matrix (decision 0009): labels in
//! the scripts that need shaping, drawn from the vendored Noto subsets
//! (`tests/data/fonts`) and laid out in every writing mode. Its digest
//! covers every bit of every glyph's metrics and SDF and of every glyph
//! position, so one value natively and in wasm is the byte-identity check
//! of M2's exit. The harness module exports it for the Chrome run.

use std::sync::Arc;

use crate::font::{Face, local_glyph};
use crate::{
    Anchor, FontStack, Glyph, GlyphKey, GlyphMap, GlyphRequests, Images, Justify, Label,
    ShapeContext, ShapeOptions, TextTransform, Variant, WritingMode, shape_text, transform_text,
};

/// Each font and the labels drawn with it.
const MATRIX: [(&[u8], &[&str]); 7] = [
    (
        include_bytes!("../tests/data/fonts/NotoSansArabic.ttf"),
        &["القاهرة", "الإسكندرية مصر", "مصر ١٢٣"],
    ),
    (
        include_bytes!("../tests/data/fonts/NotoSansHebrew.ttf"),
        &["ירושלים", "תל אביב", "שָׁלוֹם"],
    ),
    (
        include_bytes!("../tests/data/fonts/NotoSansDevanagari.ttf"),
        &["नई दिल्ली", "मुंबई", "क्षत्रिय स्ते"],
    ),
    (
        include_bytes!("../tests/data/fonts/NotoSansThai.ttf"),
        &["กรุงเทพมหานคร", "ประเทศไทย"],
    ),
    (
        include_bytes!("../tests/data/fonts/NotoSansKhmer.ttf"),
        &["ភ្នំពេញ", "ភាសាខ្មែរ"],
    ),
    (
        include_bytes!("../tests/data/fonts/NotoSansMyanmar.ttf"),
        &["ရန်ကုန်", "ဘင်္ဂလားဒေ့ရှ်"],
    ),
    (
        include_bytes!("../tests/data/fonts/NotoSans.ttf"),
        &["Cairo", "New Delhi (12)", "Bangkok-é"],
    ),
];

/// FNV-1a over 64-bit words.
#[derive(Default)]
struct Digest(u64);

impl Digest {
    fn feed(&mut self, bits: u64) {
        if self.0 == 0 {
            self.0 = 0xcbf2_9ce4_8422_2325;
        }
        for byte in bits.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// The matrix's digest.
pub fn digest() -> u64 {
    let mut d = Digest::default();
    let stack = FontStack::from_joined("Noto Sans Regular");
    let no_images: Images<'_> = &|_| None;
    for (bytes, labels) in MATRIX {
        let bytes: crate::font::FontBytes = Arc::new(bytes);
        let face = Face::new(bytes, 0).expect("a vendored font");
        for text in labels {
            let label = Label::from_text(&transform_text(text, TextTransform::None));
            let mut requests = GlyphRequests::default();
            requests.add_label(&label, &stack, false, true);
            let mut glyphs = GlyphMap::default();
            for request in requests {
                let (metrics, sdf) = local_glyph(&face, &request.text);
                for v in [
                    metrics.width,
                    metrics.height,
                    metrics.left,
                    metrics.top,
                    metrics.advance,
                ] {
                    d.feed(v.to_bits());
                }
                d.feed(u64::from(sdf.width) << 32 | u64::from(sdf.height));
                for chunk in sdf.data.chunks(8) {
                    let mut word = [0u8; 8];
                    word[..chunk.len()].copy_from_slice(chunk);
                    d.feed(u64::from_le_bytes(word));
                }
                let glyph = Glyph {
                    key: GlyphKey::Font {
                        font: face.id(),
                        variant: Variant::Default,
                        text: request.text.clone(),
                    },
                    metrics,
                    bitmap: [sdf.width, sdf.height],
                    double_resolution: true,
                };
                glyphs.insert(request, Some(glyph));
            }
            for (mode, allow) in [
                (WritingMode::Horizontal, false),
                (WritingMode::Vertical, true),
            ] {
                for max_width in [f64::INFINITY, 60.0] {
                    let options = ShapeOptions {
                        max_width,
                        line_height: 28.8,
                        anchor: Anchor::Center,
                        justify: Justify::Center,
                        spacing: 0.0,
                        translate: [0.0, 0.0],
                        writing_mode: mode,
                        allow_vertical_placement: allow,
                        layout_text_size: 16.0,
                        layout_text_size_this_zoom: 16.0,
                    };
                    let context = ShapeContext {
                        glyphs: &glyphs,
                        images: &no_images,
                        words: None,
                    };
                    let Some(shaping) = shape_text(&label, &stack, context, &options) else {
                        d.feed(u64::MAX);
                        continue;
                    };
                    for v in [shaping.top, shaping.bottom, shaping.left, shaping.right] {
                        d.feed(v.to_bits());
                    }
                    for line in &shaping.lines {
                        d.feed(line.line_offset.to_bits());
                        for g in &line.glyphs {
                            d.feed(u64::from(g.code_point as u32));
                            d.feed(g.x.to_bits());
                            d.feed(g.y.to_bits());
                            d.feed(g.scale.to_bits());
                            d.feed(u64::from(g.vertical));
                        }
                    }
                }
            }
        }
    }
    d.0
}

#[cfg(test)]
mod tests {
    /// The same value in wasm (the Chrome fixture run compares it).
    #[test]
    fn matrix_digest() {
        assert_eq!(super::digest(), 2_718_566_842_981_457_007);
    }
}

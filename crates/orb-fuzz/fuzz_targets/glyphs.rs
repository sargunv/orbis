//! A glyph range PBF, parsed, and each glyph read as labels read it.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    if let Ok(range) = orb_text::pbf::GlyphRange::parse(data) {
        let stack = orb_text::FontStack::new(&["Noto Sans Regular"]);
        for glyph in range.glyphs() {
            let _ = glyph.bitmap_size();
            let _ = glyph.glyph(&stack);
        }
    }
});

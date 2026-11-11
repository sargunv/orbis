//! A font file (TrueType, OpenType, WOFF or WOFF2) opened as a style's
//! `font-faces` opens one, and a label drawn from it into its SDF.

#![no_main]
#![forbid(unsafe_code)]

use orb_text::sdf::local;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let mut files = orb_text::font::Files::default();
    if files.open("fuzz", data).is_ok() {
        for text in ["Ag", "ά", "שָׁ", "क्षि"] {
            let ink = local::ink_box(text, &files.measure("fuzz", text));
            let _ = local::sdf(&ink, &files.coverage("fuzz", text, &ink));
        }
    }
});

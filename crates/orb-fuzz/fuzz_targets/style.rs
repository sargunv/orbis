//! A style: parsed with every preview, its expressions compiled, and set
//! up for layout as a tile worker does.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|text: &str| {
    let _ = orb_style::v8::parse_with(text, orb_style::Previews::ALL, &[]);
    let _ = orb_runtime::TileWorker::new(text);
});

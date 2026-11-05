//! Shaping conformance with the pinned GL JS: its `symbol-shaping`
//! expectations compared field by field, ports of `shaping.test.ts`, and
//! right-to-left and complex-script cases. Shared by native tests and wasm
//! tests in Chrome.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::*;

const GLYPHS: &str = include_str!("../tests/data/fontstack-glyphs.json");

macro_rules! fixture {
    ($name:literal) => {
        include_str!(concat!(
            "../tests/data/symbol-shaping/text-shaping-",
            $name,
            ".json"
        ))
    };
}

fn test_stack() -> FontStack {
    FontStack::new(&["Test"])
}

/// GL JS's test glyphs, keyed by code point, with each one's atlas rect.
struct Fixture {
    glyphs: GlyphMap,
    rects: HashMap<GlyphId, Value>,
}

impl Fixture {
    fn new() -> Self {
        let json: Value = serde_json::from_str(GLYPHS).unwrap();
        let mut glyphs = GlyphMap::default();
        let mut rects = HashMap::new();
        for (cp, entry) in json.as_object().unwrap() {
            let cp: u32 = cp.parse().unwrap();
            let m = &entry["metrics"];
            let f = |k: &str| m[k].as_f64().unwrap();
            let rect = entry["rect"].clone();
            let id = glyphs.insert(
                GlyphRequest {
                    stack: test_stack(),
                    variant: Variant::Default,
                    text: char::from_u32(cp).unwrap().to_string().into(),
                },
                Some(Glyph {
                    key: GlyphKey::Pbf {
                        stack: test_stack(),
                        code_point: cp,
                    },
                    metrics: GlyphMetrics {
                        width: f("width"),
                        height: f("height"),
                        left: f("left"),
                        top: f("top"),
                        advance: f("advance"),
                    },
                    bitmap: [
                        rect["w"].as_u64().unwrap() as u32,
                        rect["h"].as_u64().unwrap() as u32,
                    ],
                    double_resolution: false,
                }),
            );
            rects.insert(id, rect);
        }
        Self { glyphs, rects }
    }
}

fn image(name: &str) -> Option<InlineImage> {
    let size = match name {
        "square" => [16, 16],
        "tall" => [16, 32],
        "wide" => [32, 16],
        _ => return None,
    };
    Some(InlineImage {
        padded_size: size,
        pixel_ratio: 1.0,
    })
}

const ONE: f64 = ONE_EM;

fn options(max_width: f64, spacing: f64, mode: WritingMode, allow_vertical: bool) -> ShapeOptions {
    ShapeOptions {
        max_width,
        line_height: ONE,
        anchor: Anchor::Center,
        justify: Justify::Center,
        spacing,
        translate: [0.0, 0.0],
        writing_mode: mode,
        allow_vertical_placement: allow_vertical,
        layout_text_size: 16.0,
        layout_text_size_this_zoom: 16.0,
    }
}

fn shape(fixture: &Fixture, label: &Label, options: &ShapeOptions) -> Option<Shaping> {
    let context = ShapeContext {
        glyphs: &fixture.glyphs,
        images: &image,
        words: None,
    };
    shape_text(label, &test_stack(), context, options)
}

/// The shaping in GL JS's JSON form.
fn to_json(fixture: &Fixture, shaping: &Shaping) -> Value {
    let lines: Vec<Value> = shaping
        .lines
        .iter()
        .map(|line| {
            let glyphs: Vec<Value> = line
                .glyphs
                .iter()
                .map(|g| {
                    let (grapheme, image, stack, rect) = match g.source {
                        GlyphSource::Glyph(id) => {
                            let request = fixture.glyphs.request(id);
                            (
                                request.text.to_string(),
                                Value::Null,
                                request.stack.as_str().to_string(),
                                fixture.rects[&id].clone(),
                            )
                        }
                        GlyphSource::Image { section } => {
                            let ShapedSection::Image(name) = &shaping.sections[section] else {
                                panic!("image glyph from a text section");
                            };
                            let size = image(name).unwrap().padded_size;
                            (
                                g.code_point.to_string(),
                                json!(name.as_ref()),
                                String::new(),
                                json!({"x": 0, "y": 0, "w": size[0], "h": size[1]}),
                            )
                        }
                    };
                    json!({
                        "glyph": g.code_point as u32,
                        "grapheme": grapheme,
                        "imageName": image,
                        "x": g.x,
                        "y": g.y,
                        "vertical": g.vertical,
                        "scale": g.scale,
                        "fontStack": stack,
                        "sectionIndex": g.section,
                        "metrics": {
                            "width": g.metrics.width,
                            "height": g.metrics.height,
                            "left": g.metrics.left,
                            "top": g.metrics.top,
                            "advance": g.metrics.advance,
                        },
                        "rect": rect,
                    })
                })
                .collect();
            json!({"positionedGlyphs": glyphs, "lineOffset": line.line_offset})
        })
        .collect();
    json!({
        "positionedLines": lines,
        "text": shaping.text,
        "top": shaping.top,
        "bottom": shaping.bottom,
        "left": shaping.left,
        "right": shaping.right,
        "writingMode": match shaping.writing_mode {
            WritingMode::Horizontal => 1,
            WritingMode::Vertical => 2,
        },
        "iconsInText": shaping.icons_in_text,
        "verticalizable": shaping.verticalizable,
    })
}

/// Compares JSON with numbers as `f64`, reporting the first difference.
fn same(path: &str, a: &Value, b: &Value) -> Result<(), String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            if x.as_f64() == y.as_f64() {
                Ok(())
            } else {
                Err(format!("{path}: {x} != {y}"))
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                same(
                    &format!("{path}.{k}"),
                    x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null),
                )?;
            }
            Ok(())
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Err(format!("{path}: length {} != {}", x.len(), y.len()));
            }
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                same(&format!("{path}[{i}]"), p, q)?;
            }
            Ok(())
        }
        _ if a == b => Ok(()),
        _ => Err(format!("{path}: {a} != {b}")),
    }
}

fn check(
    fixture: &Fixture,
    name: &str,
    expected: &str,
    label: Label,
    options: ShapeOptions,
) -> Result<(), String> {
    let shaping = shape(fixture, &label, &options).ok_or_else(|| format!("{name}: no shaping"))?;
    let expected: Value = serde_json::from_str(expected).unwrap();
    same(name, &to_json(fixture, &shaping), &expected)
}

/// GL JS `test/integration/symbol-shaping/shaping.test.ts`: every
/// expectation file, compared field by field.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn gl_js_symbol_shaping_fixtures() {
    use WritingMode::*;
    let f = Fixture::new();
    let text = Label::from_text;
    let image_section = |name: &str, align| Section::image(name).with_vertical_align(align);
    let text_section = |s: &str| Section::text(s, None, None);
    let cases: Vec<(&str, &str, Label, ShapeOptions)> = vec![
        (
            "null",
            fixture!("null"),
            text("hi\0"),
            options(15.0 * ONE, 0.0, Horizontal, false),
        ),
        (
            "default",
            fixture!("default"),
            text("abcde"),
            options(15.0 * ONE, 0.0, Horizontal, false),
        ),
        (
            "spacing",
            fixture!("spacing"),
            text("abcde"),
            options(15.0 * ONE, 0.125 * ONE, Horizontal, false),
        ),
        (
            "linebreak",
            fixture!("linebreak"),
            text("abcde abcde"),
            options(4.0 * ONE, 0.0, Horizontal, false),
        ),
        (
            "newline",
            fixture!("newline"),
            text("abcde\nabcde"),
            options(15.0 * ONE, 0.0, Horizontal, false),
        ),
        (
            "newlines-in-middle",
            fixture!("newlines-in-middle"),
            text("abcde\n\nabcde"),
            options(15.0 * ONE, 0.0, Horizontal, false),
        ),
        (
            "zero-width-space",
            fixture!("zero-width-space"),
            text("三三\u{200b}三三\u{200b}三三\u{200b}三三三三三三\u{200b}三三"),
            options(5.0 * ONE, 0.0, Horizontal, false),
        ),
        (
            "break-before-bracket",
            fixture!("break-before-bracket"),
            text("oooooooo(aaaaaaaaa)"),
            options(5.0 * ONE, 0.0, Horizontal, false),
        ),
        (
            "images-horizontal",
            fixture!("images-horizontal"),
            Label {
                sections: vec![
                    text_section("Foo"),
                    image_section("square", VerticalAlign::Bottom),
                    image_section("wide", VerticalAlign::Bottom),
                    text_section("\n"),
                    image_section("tall", VerticalAlign::Bottom),
                    image_section("square", VerticalAlign::Bottom),
                    text_section(" bar"),
                ],
            },
            options(5.0 * ONE, 0.0, Horizontal, false),
        ),
        (
            "images-vertical",
            fixture!("images-vertical"),
            Label {
                sections: vec![
                    text_section("三"),
                    image_section("square", VerticalAlign::Bottom),
                    image_section("wide", VerticalAlign::Bottom),
                    text_section("\u{200b}"),
                    image_section("tall", VerticalAlign::Bottom),
                    image_section("square", VerticalAlign::Bottom),
                    text_section("三"),
                ],
            },
            options(5.0 * ONE, 0.0, Vertical, true),
        ),
    ];
    let total = cases.len();
    let failures: Vec<String> = cases
        .into_iter()
        .filter_map(|(name, expected, label, options)| {
            check(&f, name, expected, label, options).err()
        })
        .collect();
    report("GL JS symbol-shaping fixtures", total, &failures);
}

fn report(what: &str, total: usize, failures: &[String]) {
    #[cfg(not(target_arch = "wasm32"))]
    println!("{what}: {}/{total} passed", total - failures.len());
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::console_log!("{what}: {}/{total} passed", total - failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn positioned(shaping: &Shaping) -> Vec<&PositionedGlyph> {
    shaping.lines.iter().flat_map(|l| &l.glyphs).collect()
}

/// The rest of GL JS's `symbol-shaping/shaping.test.ts`.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn gl_js_shaping_cases() {
    use WritingMode::*;
    let f = Fixture::new();
    let h15 = options(15.0 * ONE, 0.0, Horizontal, false);
    let h5 = options(5.0 * ONE, 0.0, Horizontal, false);

    let crlf = shape(&f, &Label::from_text("abcde\r\nabcde"), &h15).unwrap();
    let expected: Value = serde_json::from_str(fixture!("newline")).unwrap();
    same(
        "crlf",
        &to_json(&f, &crlf)["positionedLines"],
        &expected["positionedLines"],
    )
    .unwrap();

    assert_eq!(shape(&f, &Label::from_text(""), &h15), None);
    assert_eq!(shape(&f, &Label::from_text("\0"), &h15), None);

    let padded = shape(&f, &Label::from_text("   foo bar\n"), &h15).unwrap();
    let plain = shape(&f, &Label::from_text("foo bar"), &h15).unwrap();
    assert_eq!(padded.lines, plain.lines);

    let square = shape(
        &f,
        &Label {
            sections: vec![Section::image("square")],
        },
        &h5,
    )
    .unwrap();
    assert_eq!((square.top, square.left), (-12.0, -10.5));

    let brackets = shape(
        &f,
        &Label::from_text("()"),
        &options(5.0 * ONE, 0.0, Vertical, true),
    )
    .unwrap();
    let cps: Vec<u32> = positioned(&brackets)
        .iter()
        .map(|g| g.code_point as u32)
        .collect();
    assert_eq!(cps, [0xfe35, 0xfe36]);

    let astral = shape(&f, &Label::from_text("🗺️🌐"), &h5).unwrap();
    let glyphs: Vec<(u32, f64)> = positioned(&astral)
        .iter()
        .map(|g| (g.code_point as u32, g.x))
        .collect();
    assert_eq!(glyphs, [(0x1f5fa, -36.0), (0xfe0f, -12.0), (0x1f310, 12.0)]);

    let text_align = Label {
        sections: vec![
            Section::text("A", Some(3.0), None),
            Section::text("A", Some(1.0), None).with_vertical_align(VerticalAlign::Top),
            Section::text("A", Some(1.0), None).with_vertical_align(VerticalAlign::Center),
            Section::text("A", Some(1.0), None).with_vertical_align(VerticalAlign::Bottom),
        ],
    };
    let ys: Vec<f64> = positioned(&shape(&f, &text_align, &h5).unwrap())
        .iter()
        .map(|g| g.y)
        .collect();
    assert_eq!(ys, [-36.0, -36.0, -12.0, 12.0]);

    let image_align = Label {
        sections: vec![
            Section::text("A", Some(3.0), None),
            Section::image("square").with_vertical_align(VerticalAlign::Top),
            Section::image("square").with_vertical_align(VerticalAlign::Center),
            Section::image("square").with_vertical_align(VerticalAlign::Bottom),
        ],
    };
    let ys: Vec<f64> = positioned(&shape(&f, &image_align, &h5).unwrap())
        .iter()
        .map(|g| g.y)
        .collect();
    assert_eq!(ys, [-36.0, -36.0, -10.5, 15.0]);
}

/// Glyphs with the stub metrics of GL JS's `shaping.test.ts`, for every
/// cluster of `text` and the vertical forms of its punctuation.
fn stub_glyphs(text: &str, stack: &FontStack) -> GlyphMap {
    let mut map = GlyphMap::default();
    let mut add = |s: &str| {
        insert_stub(&mut map, stack, Variant::Default, s, 1.0);
    };
    for r in graphemes(text) {
        add(&text[r]);
    }
    for c in text.chars() {
        if let Some(v) = vertical_form(c) {
            add(&v.to_string());
        }
    }
    map
}

fn insert_stub(map: &mut GlyphMap, stack: &FontStack, variant: Variant, text: &str, left: f64) {
    let first = text.chars().next().unwrap() as u32;
    map.insert(
        GlyphRequest {
            stack: stack.clone(),
            variant,
            text: text.into(),
        },
        Some(Glyph {
            key: GlyphKey::Pbf {
                stack: stack.clone(),
                code_point: first,
            },
            metrics: GlyphMetrics {
                width: 10.0,
                height: 18.0,
                left,
                top: -8.0,
                advance: 12.0,
            },
            bitmap: [16, 24],
            double_resolution: false,
        }),
    );
}

fn no_images(_: &str) -> Option<InlineImage> {
    None
}

fn line_label(
    label: &Label,
    glyphs: &GlyphMap,
    mode: WritingMode,
    allow_vertical: bool,
) -> Option<Shaping> {
    let options = ShapeOptions {
        max_width: f64::INFINITY,
        line_height: 24.0,
        anchor: Anchor::Center,
        justify: Justify::Center,
        spacing: 0.0,
        translate: [0.0, 0.0],
        writing_mode: mode,
        allow_vertical_placement: allow_vertical,
        layout_text_size: 24.0,
        layout_text_size_this_zoom: 24.0,
    };
    let context = ShapeContext {
        glyphs,
        images: &no_images,
        words: None,
    };
    shape_text(label, &test_stack(), context, &options)
}

fn orientations(glyphs: &GlyphMap, shaping: &Shaping) -> Vec<(String, &'static str)> {
    positioned(shaping)
        .iter()
        .map(|g| {
            let GlyphSource::Glyph(id) = g.source else {
                panic!()
            };
            let text = glyphs.request(id).text.to_string();
            (text, if g.vertical { "upright" } else { "along-line" })
        })
        .collect()
}

/// GL JS `shaping.test.ts`, "shapeText vertical glyph orientation".
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn gl_js_vertical_orientation() {
    use WritingMode::*;
    let u = "upright";
    let a = "along-line";
    type Case<'a> = (&'a str, WritingMode, bool, &'a [(&'a str, &'a str)]);
    let cases: &[Case] = &[
        (
            "반포대로21길",
            Vertical,
            false,
            &[
                ("반", u),
                ("포", u),
                ("대", u),
                ("로", u),
                ("2", u),
                ("1", u),
                ("길", u),
            ],
        ),
        (
            "身什戰33",
            Vertical,
            false,
            &[("身", u), ("什", u), ("戰", u), ("3", u), ("3", u)],
        ),
        (
            "国道1234号",
            Vertical,
            false,
            &[
                ("国", u),
                ("道", u),
                ("1", u),
                ("2", u),
                ("3", u),
                ("4", u),
                ("号", u),
            ],
        ),
        (
            "身什戰 1",
            Vertical,
            false,
            &[("身", u), ("什", u), ("戰", u), (" ", a), ("1", u)],
        ),
        (
            "国道A号",
            Vertical,
            false,
            &[("国", u), ("道", u), ("A", u), ("号", u)],
        ),
        (
            "国道e\u{301}号",
            Vertical,
            false,
            &[("国", u), ("道", u), ("e\u{301}", a), ("号", u)],
        ),
        (
            "国道α号",
            Vertical,
            false,
            &[("国", u), ("道", u), ("α", a), ("号", u)],
        ),
        (
            "国道\u{41C}1号",
            Vertical,
            false,
            &[("国", u), ("道", u), ("\u{41C}", u), ("1", u), ("号", u)],
        ),
        (
            "国道ب号",
            Vertical,
            false,
            &[("国", u), ("道", u), ("ب", a), ("号", u)],
        ),
        (
            "札幌タワー",
            Vertical,
            false,
            &[("札", u), ("幌", u), ("タ", u), ("ワ", u), ("ー", a)],
        ),
        (
            "身什戰1〜2",
            Vertical,
            false,
            &[
                ("身", u),
                ("什", u),
                ("戰", u),
                ("1", u),
                ("〜", a),
                ("2", u),
            ],
        ),
        (
            "two 身什戰",
            Vertical,
            false,
            &[
                ("t", a),
                ("w", a),
                ("o", a),
                (" ", a),
                ("身", u),
                ("什", u),
                ("戰", u),
            ],
        ),
        (
            "身什戰A1",
            Vertical,
            false,
            &[("身", u), ("什", u), ("戰", u), ("A", u), ("1", u)],
        ),
        (
            "銀座Ginza通り",
            Vertical,
            false,
            &[
                ("銀", u),
                ("座", u),
                ("G", a),
                ("i", a),
                ("n", a),
                ("z", a),
                ("a", a),
                ("通", u),
                ("り", u),
            ],
        ),
        (
            "ヴィラ ISHIKAWA",
            Vertical,
            false,
            &[
                ("ヴ", u),
                ("ィ", u),
                ("ラ", u),
                (" ", a),
                ("I", a),
                ("S", a),
                ("H", a),
                ("I", a),
                ("K", a),
                ("A", a),
                ("W", a),
                ("A", a),
            ],
        ),
        (
            "JR山手線",
            Vertical,
            false,
            &[("J", u), ("R", u), ("山", u), ("手", u), ("線", u)],
        ),
        (
            "国道1-2号",
            Vertical,
            false,
            &[
                ("国", u),
                ("道", u),
                ("1", u),
                ("︲", u),
                ("2", u),
                ("号", u),
            ],
        ),
        (
            "国道3.5km",
            Vertical,
            false,
            &[
                ("国", u),
                ("道", u),
                ("3", a),
                (".", a),
                ("5", a),
                ("k", a),
                ("m", a),
            ],
        ),
        (
            "身什戰33",
            Horizontal,
            false,
            &[("身", a), ("什", a), ("戰", a), ("3", a), ("3", a)],
        ),
        (
            "two 身什戰",
            Vertical,
            true,
            &[
                ("t", u),
                ("w", u),
                ("o", u),
                (" ", a),
                ("身", u),
                ("什", u),
                ("戰", u),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for &(text, mode, allow, expected) in cases {
        let glyphs = stub_glyphs(text, &test_stack());
        let shaping = line_label(&Label::from_text(text), &glyphs, mode, allow).unwrap();
        let got = orientations(&glyphs, &shaping);
        let expected: Vec<(String, &str)> =
            expected.iter().map(|&(t, o)| (t.to_string(), o)).collect();
        if got != expected {
            failures.push(format!("{text}: {got:?}"));
        }
    }

    // A missing glyph places nothing, and the rest keep their own
    // orientations.
    let text = "what 国21号";
    let glyphs = stub_glyphs(&text.replace('w', ""), &test_stack());
    let shaping = line_label(&Label::from_text(text), &glyphs, Vertical, false).unwrap();
    let got = orientations(&glyphs, &shaping);
    let expected = [
        ("h", a),
        ("a", a),
        ("t", a),
        (" ", a),
        ("国", u),
        ("2", u),
        ("1", u),
        ("号", u),
    ];
    if got != expected.map(|(t, o)| (t.to_string(), o)) {
        failures.push(format!("{text}: {got:?}"));
    }
    report(
        "GL JS vertical orientation cases",
        cases.len() + 1,
        &failures,
    );
}

/// GL JS `shaping.test.ts`, "font-provided vertical forms".
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn gl_js_font_provided_vertical_forms() {
    use WritingMode::Vertical;
    let stack = test_stack();
    let with_vertical = |text: &str, forms: &[&str]| {
        let mut glyphs = stub_glyphs(text, &stack);
        for form in forms {
            insert_stub(&mut glyphs, &stack, Variant::Vertical, form, 5.0);
        }
        glyphs
    };
    let summary = |glyphs: &GlyphMap, shaping: &Shaping| -> Vec<(String, bool, f64)> {
        positioned(shaping)
            .iter()
            .map(|g| {
                let GlyphSource::Glyph(id) = g.source else {
                    panic!()
                };
                (
                    glyphs.request(id).text.to_string(),
                    g.vertical,
                    g.metrics.left,
                )
            })
            .collect()
    };

    // Each section's font decides between its vertical form and the
    // compatibility character.
    let other = FontStack::new(&["Other"]);
    let mut glyphs = with_vertical("（", &["（"]);
    insert_stub(&mut glyphs, &other, Variant::Default, "（", 1.0);
    insert_stub(&mut glyphs, &other, Variant::Default, "︵", 1.0);
    let label = Label {
        sections: vec![
            Section::text("（", Some(1.0), Some(stack.clone())),
            Section::text("（", Some(1.0), Some(other.clone())),
        ],
    };
    let shaping = line_label(&label, &glyphs, Vertical, true).unwrap();
    let got: Vec<(String, String, f64)> = positioned(&shaping)
        .iter()
        .map(|g| {
            let GlyphSource::Glyph(id) = g.source else {
                panic!()
            };
            let r = glyphs.request(id);
            (
                r.text.to_string(),
                r.stack.as_str().to_string(),
                g.metrics.left,
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("（".to_string(), "Test".to_string(), 5.0),
            ("︵".to_string(), "Other".to_string(), 1.0)
        ]
    );

    let text = "東京AB（CD）";
    let glyphs = with_vertical(text, &["（", "）"]);
    let shaping = line_label(&Label::from_text(text), &glyphs, Vertical, false).unwrap();
    let s = |t: &str, v, l| (t.to_string(), v, l);
    assert_eq!(
        summary(&glyphs, &shaping),
        [
            s("東", true, 1.0),
            s("京", true, 1.0),
            s("A", false, 1.0),
            s("B", false, 1.0),
            s("（", false, 1.0),
            s("C", false, 1.0),
            s("D", false, 1.0),
            s("）", false, 1.0),
        ]
    );

    let text = "東京12（…）ABC";
    let glyphs = with_vertical(text, &["…"]);
    let shaping = line_label(&Label::from_text(text), &glyphs, Vertical, false).unwrap();
    let got: Vec<(String, bool)> = summary(&glyphs, &shaping)
        .into_iter()
        .map(|(t, v, _)| (t, v))
        .collect();
    let expected = [
        ("東", true),
        ("京", true),
        ("1", true),
        ("2", true),
        ("︵", true),
        ("︙", true),
        ("）", false),
        ("A", false),
        ("B", false),
        ("C", false),
    ];
    assert_eq!(got, expected.map(|(t, v)| (t.to_string(), v)));

    let text = "東京 Tokyo … Station";
    let glyphs = with_vertical(text, &["…"]);
    let shaping = line_label(&Label::from_text(text), &glyphs, Vertical, false).unwrap();
    let ellipsis = summary(&glyphs, &shaping)
        .into_iter()
        .find(|g| g.0 == "…")
        .unwrap();
    assert_eq!((ellipsis.1, ellipsis.2), (false, 1.0));

    let text = "東京 AB … CD";
    let glyphs = with_vertical(text, &["…"]);
    let shaping = line_label(&Label::from_text(text), &glyphs, Vertical, false).unwrap();
    let ellipsis = summary(&glyphs, &shaping)
        .into_iter()
        .find(|g| g.0 == "…")
        .unwrap();
    assert_eq!((ellipsis.1, ellipsis.2), (true, 5.0));
}

/// Glyphs of advance 10 for every cluster of `text` and every code point,
/// as a tile asks for them.
fn cluster_glyphs(text: &str, clusters: bool) -> GlyphMap {
    cluster_glyphs_with(text, clusters, |_| 10.0)
}

/// [`cluster_glyphs`] with each glyph's advance chosen by its first code
/// point.
fn cluster_glyphs_with(text: &str, clusters: bool, advance: impl Fn(char) -> f64) -> GlyphMap {
    let stack = test_stack();
    let mut map = GlyphMap::default();
    let mut add = |s: &str| {
        let first = s.chars().next().unwrap() as u32;
        map.insert(
            GlyphRequest {
                stack: stack.clone(),
                variant: Variant::Default,
                text: s.into(),
            },
            Some(Glyph {
                key: GlyphKey::Pbf {
                    stack: stack.clone(),
                    code_point: first,
                },
                metrics: GlyphMetrics {
                    width: 10.0,
                    height: 10.0,
                    left: 0.0,
                    top: -8.0,
                    advance: advance(s.chars().next().unwrap()),
                },
                bitmap: [16, 16],
                double_resolution: false,
            }),
        );
    };
    for r in graphemes(text) {
        if clusters {
            add(&text[r.clone()]);
        }
        for c in text[r].chars() {
            add(&c.to_string());
        }
    }
    map
}

fn drawn(glyphs: &GlyphMap, shaping: &Shaping) -> Vec<String> {
    positioned(shaping)
        .iter()
        .map(|g| match g.source {
            GlyphSource::Glyph(id) => glyphs.request(id).text.to_string(),
            GlyphSource::Image { .. } => "<image>".to_string(),
        })
        .collect()
}

fn horizontal(text: &str, glyphs: &GlyphMap, max_width: f64) -> Shaping {
    let options = ShapeOptions {
        max_width,
        line_height: 24.0,
        anchor: Anchor::Center,
        justify: Justify::Center,
        spacing: 0.0,
        translate: [0.0, 0.0],
        writing_mode: WritingMode::Horizontal,
        allow_vertical_placement: false,
        layout_text_size: 24.0,
        layout_text_size_this_zoom: 24.0,
    };
    let context = ShapeContext {
        glyphs,
        images: &no_images,
        words: None,
    };
    shape_text(&Label::from_text(text), &test_stack(), context, &options).unwrap()
}

/// GL JS `shaping.test.ts` "right-to-left", with the built-in bidi in place
/// of its stub plugin: letters reverse, each keeping its marks in writing
/// order, and only requested clusters are drawn.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn hebrew_clusters_reverse_whole() {
    const SHIN_SHEVA_DOT: &str = "שְׁ";
    const DALET_TSERE: &str = "דֵ";
    const RESH: &str = "ר";
    const VAV_HOLAM: &str = "וֹ";
    const TAV: &str = "ת";
    let text = [SHIN_SHEVA_DOT, DALET_TSERE, RESH, VAV_HOLAM, TAV].concat();

    let glyphs = cluster_glyphs(&text, true);
    let as_clusters = drawn(&glyphs, &horizontal(&text, &glyphs, f64::INFINITY));
    assert_eq!(
        as_clusters,
        [TAV, VAV_HOLAM, RESH, DALET_TSERE, SHIN_SHEVA_DOT]
    );

    let code_points = cluster_glyphs(&text, false);
    let as_code_points = drawn(
        &code_points,
        &horizontal(&text, &code_points, f64::INFINITY),
    );
    assert_eq!(as_code_points.concat(), as_clusters.concat());
    assert_eq!(as_code_points.len(), text.chars().count());
}

/// Right-to-left and complex-script labels, end to end.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn complex_scripts() {
    // Arabic: presentation forms, then visual order (right to left).
    let shaped = apply_arabic_shaping("مرحبا");
    let glyphs = cluster_glyphs(&shaped, true);
    let shaping = horizontal(&shaped, &glyphs, f64::INFINITY);
    assert_eq!(drawn(&glyphs, &shaping).concat(), "ﺎﺒﺣﺮﻣ");
    let xs: Vec<f64> = positioned(&shaping).iter().map(|g| g.x).collect();
    assert_eq!(xs, [-25.0, -15.0, -5.0, 5.0, 15.0]);

    // Hebrew with a number and brackets: digits keep their order, brackets
    // mirror.
    let text = "רחוב (12)";
    let glyphs = cluster_glyphs(text, true);
    assert_eq!(
        drawn(&glyphs, &horizontal(text, &glyphs, f64::INFINITY)).concat(),
        "(12) בוחר"
    );

    // Mixed directions break into lines first, then reorder each line.
    let text = "שלום world עולם";
    let glyphs = cluster_glyphs(text, true);
    let shaping = horizontal(text, &glyphs, 60.0);
    let lines: Vec<String> = shaping
        .lines
        .iter()
        .map(|l| {
            l.glyphs
                .iter()
                .map(|g| match g.source {
                    GlyphSource::Glyph(id) => glyphs.request(id).text.to_string(),
                    GlyphSource::Image { .. } => unreachable!(),
                })
                .collect()
        })
        .collect();
    assert_eq!(lines, ["םולש", "world", "םלוע"]);

    // Devanagari: a cluster with its own glyph is drawn whole; without one,
    // a code point at a time, measured the same way.
    let text = "दिल्ली";
    let whole = cluster_glyphs(text, true);
    assert_eq!(
        drawn(&whole, &horizontal(text, &whole, f64::INFINITY)),
        ["दि", "ल्ली"]
    );
    let parts = cluster_glyphs(text, false);
    let shaping = horizontal(text, &parts, f64::INFINITY);
    assert_eq!(drawn(&parts, &shaping).len(), text.chars().count());
    assert_eq!(
        shaping.right - shaping.left,
        10.0 * text.chars().count() as f64
    );

    // Astral ideographs are one glyph each.
    let text = "𰻞𰻞麵";
    let glyphs = cluster_glyphs(text, true);
    assert_eq!(
        drawn(&glyphs, &horizontal(text, &glyphs, f64::INFINITY)),
        ["𰻞", "𰻞", "麵"]
    );

    // An inline image takes the bidi path, as in GL JS, and keeps its place.
    let glyphs = cluster_glyphs("ab", true);
    let label = Label {
        sections: vec![
            Section::text("a", None, None),
            Section::image("square"),
            Section::text("b", None, None),
        ],
    };
    let square = |name: &str| {
        (name == "square").then_some(InlineImage {
            padded_size: [16, 16],
            pixel_ratio: 1.0,
        })
    };
    let options = ShapeOptions {
        max_width: f64::INFINITY,
        line_height: 24.0,
        anchor: Anchor::Center,
        justify: Justify::Center,
        spacing: 0.0,
        translate: [0.0, 0.0],
        writing_mode: WritingMode::Horizontal,
        allow_vertical_placement: false,
        layout_text_size: 24.0,
        layout_text_size_this_zoom: 24.0,
    };
    let context = ShapeContext {
        glyphs: &glyphs,
        images: &square,
        words: None,
    };
    let shaping = shape_text(&label, &test_stack(), context, &options).unwrap();
    assert_eq!(drawn(&glyphs, &shaping), ["a", "<image>", "b"]);
}

/// Thai has no spaces: lines break only where the dictionary finds words.
#[cfg(feature = "sea-dictionaries")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn thai_breaks_at_dictionary_words() {
    let text = "ราชอาณาจักรไทย";
    let glyphs = cluster_glyphs(text, true);
    let options = ShapeOptions {
        max_width: 60.0,
        line_height: 24.0,
        anchor: Anchor::Center,
        justify: Justify::Center,
        spacing: 0.0,
        translate: [0.0, 0.0],
        writing_mode: WritingMode::Horizontal,
        allow_vertical_placement: false,
        layout_text_size: 24.0,
        layout_text_size_this_zoom: 24.0,
    };
    let lines = |words: Option<&WordBreaker>| -> Vec<String> {
        let context = ShapeContext {
            glyphs: &glyphs,
            images: &no_images,
            words,
        };
        let shaping =
            shape_text(&Label::from_text(text), &test_stack(), context, &options).unwrap();
        shaping
            .lines
            .iter()
            .map(|l| {
                l.glyphs
                    .iter()
                    .map(|g| match g.source {
                        GlyphSource::Glyph(id) => glyphs.request(id).text.to_string(),
                        GlyphSource::Image { .. } => unreachable!(),
                    })
                    .collect()
            })
            .collect()
    };
    assert_eq!(lines(None), [text]);
    // ICU4X's dictionary also finds ราช | อาณาจักร; GL JS's test only asks
    // for the break after อาณาจักร, which the browser's ICU finds too.
    let words = WordBreaker::southeast_asian();
    assert_eq!(lines(Some(&words)), ["ราช", "อาณาจักร", "ไทย"]);
}

/// One digest of every bit of layout over a right-to-left and
/// complex-script matrix. The same constant natively and in wasm is the
/// byte-identity check.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn layout_is_bit_identical_across_targets() {
    let labels = [
        "مرحبا بالعالم 123",
        "רחוב הרצל (12) תל אביב",
        "दिल्ली मेट्रो",
        "ราชอาณาจักรไทย กรุงเทพ",
        "東京都 Tokyo 1-2",
        "𰻞𰻞麵 𪚥",
        "שלום world עולם",
        "ភ្នំពេញ",
    ];
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bits: u64| {
        for byte in bits.to_le_bytes() {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for text in labels {
        let shaped = apply_arabic_shaping(text);
        // Advances vary by code point so the digest sees real arithmetic.
        let glyphs = cluster_glyphs_with(&shaped, true, |c| f64::from(c as u32 % 17) * 0.75 + 3.5);
        for (mode, allow) in [
            (WritingMode::Horizontal, false),
            (WritingMode::Vertical, false),
            (WritingMode::Vertical, true),
        ] {
            for max_width in [f64::INFINITY, 37.5] {
                let options = ShapeOptions {
                    max_width,
                    line_height: 28.8,
                    anchor: Anchor::BottomLeft,
                    justify: Justify::Right,
                    spacing: 1.2,
                    translate: [3.25, -1.5],
                    writing_mode: mode,
                    allow_vertical_placement: allow,
                    layout_text_size: 17.0,
                    layout_text_size_this_zoom: 16.0,
                };
                let context = ShapeContext {
                    glyphs: &glyphs,
                    images: &no_images,
                    words: None,
                };
                let Some(shaping) =
                    shape_text(&Label::from_text(&shaped), &test_stack(), context, &options)
                else {
                    continue;
                };
                for v in [shaping.top, shaping.bottom, shaping.left, shaping.right] {
                    feed(v.to_bits());
                }
                for line in &shaping.lines {
                    feed(line.line_offset.to_bits());
                    for g in &line.glyphs {
                        feed(u64::from(g.code_point as u32));
                        feed(g.x.to_bits());
                        feed(g.y.to_bits());
                        feed(g.scale.to_bits());
                        feed(u64::from(g.vertical));
                    }
                }
            }
        }
    }
    assert_eq!(digest, 11_657_041_922_221_159_172);
}

/// The whole job-side flow on real glyph PBFs: requests, the provider
/// choice, PBF ranges, then shaping an Arabic and Latin label.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn glyph_pbfs_end_to_end() {
    let ranges: HashMap<u32, pbf::GlyphRange> = [
        (
            0,
            &include_bytes!("../tests/data/glyphs/Noto Sans Regular/0-255.pbf")[..],
        ),
        (
            0xFE,
            &include_bytes!("../tests/data/glyphs/Noto Sans Regular/65024-65279.pbf")[..],
        ),
    ]
    .into_iter()
    .map(|(range, bytes)| (range, pbf::GlyphRange::parse(bytes).unwrap()))
    .collect();
    assert_eq!(ranges[&0].glyphs().len(), 223);
    let a = ranges[&0].get('a' as u32).unwrap();
    assert_eq!(
        (a.metrics.advance, a.bitmap_size(), a.bitmap.len()),
        (13.0, [17, 19], 17 * 19)
    );

    let stack = FontStack::new(&["Noto Sans Regular"]);
    let label = Label::from_text(&transform_text("مرحبا abc", TextTransform::None));
    let mut requests = GlyphRequests::default();
    requests.add_label(&label, &stack, false, false);
    let policy = FontPolicy {
        glyphs_url: true,
        local_ideographs: false,
    };
    let mut glyphs = GlyphMap::default();
    for request in requests {
        let glyph = match provider(&request, policy) {
            Provider::Pbf { range } => {
                let cp = request.text.chars().next().unwrap() as u32;
                ranges
                    .get(&range)
                    .and_then(|r| r.get(cp))
                    .map(|g| g.glyph(&stack))
            }
            Provider::Local | Provider::Unavailable => None,
        };
        assert!(glyph.is_some(), "{:?} unresolved", request.text);
        glyphs.insert(request, glyph);
    }

    let options = ShapeOptions {
        max_width: 10.0 * ONE_EM,
        line_height: 1.2 * ONE_EM,
        anchor: Anchor::Center,
        justify: Justify::Center,
        spacing: 0.0,
        translate: [0.0, 0.0],
        writing_mode: WritingMode::Horizontal,
        allow_vertical_placement: false,
        layout_text_size: 16.0,
        layout_text_size_this_zoom: 16.0,
    };
    let context = ShapeContext {
        glyphs: &glyphs,
        images: &no_images,
        words: None,
    };
    let shaping = shape_text(&label, &stack, context, &options).unwrap();
    // Visual order: Latin first, then the Arabic word reversed.
    assert_eq!(drawn(&glyphs, &shaping).concat(), "abc ﺎﺒﺣﺮﻣ");
    let advances: f64 = positioned(&shaping).iter().map(|g| g.metrics.advance).sum();
    assert_eq!(shaping.right - shaping.left, advances);
}

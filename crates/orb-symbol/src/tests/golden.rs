//! The line-by-line leaves against GL JS's own results (`leaves.json`).

use std::sync::LazyLock;

use orb_text::{Shaping, WritingMode};
use serde_json::Value;

use super::{anchor, assert_same, image, line, lines, num, nums};
use crate::anchors::{Anchor, LabelExtent, center_anchor, check_max_angle, line_anchors};
use crate::collision::{CollisionFeature, Shape, collision_feature};
use crate::geometry::{Point, to_i16};
use crate::icon::{IconTextFit, PositionedIcon, apply_text_fit, fit_icon_to_text, shape_icon};
use crate::lines::{Lines, clip_line};
use crate::merge_lines::merge_lines;
use crate::pole::pole_of_inaccessibility;
use crate::quads::icon_quads;
use crate::variable::{TextOffset, variable_offset};

static LEAVES: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../tests/data/leaves.json")).expect("leaves.json parses")
});

fn cases(name: &str) -> &'static [Value] {
    let cases = LEAVES[name].as_array().expect("a case list");
    assert!(!cases.is_empty(), "{name} has cases");
    cases
}

#[track_caller]
fn assert_point(actual: Point, expected: &Value, what: &str) {
    let [x, y] = nums(expected);
    assert_same(actual.x, x, &format!("{what}.x"));
    assert_same(actual.y, y, &format!("{what}.y"));
}

fn anchor_in(v: &Value) -> Anchor {
    Anchor {
        point: Point::new(num(&v[0]), num(&v[1])),
        angle: num(&v[2]),
        segment: v[3].as_u64().map(|s| s as usize),
    }
}

#[track_caller]
fn assert_anchor(actual: &Anchor, expected: &Value, what: &str) {
    let e = anchor_in(expected);
    assert_same(actual.point.x, e.point.x, &format!("{what}.x"));
    assert_same(actual.point.y, e.point.y, &format!("{what}.y"));
    assert_same(actual.angle, e.angle, &format!("{what}.angle"));
    assert_eq!(actual.segment, e.segment, "{what}.segment");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn clip_line_matches_gl_js() {
    for (i, case) in cases("clipLine").iter().enumerate() {
        let input = lines(&case["lines"]);
        let [x1, y1, x2, y2] = nums(&case["box"]);
        let mut out = Lines::default();
        clip_line(input.iter().map(Vec::as_slice), x1, y1, x2, y2, &mut out);
        let expected = lines(&case["out"]);
        assert_eq!(out.len(), expected.len(), "case {i}: line count");
        for (got, want) in out.iter().zip(&expected) {
            assert_eq!(got, &want[..], "case {i}");
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn check_max_angle_matches_gl_js() {
    for (i, case) in cases("checkMaxAngle").iter().enumerate() {
        let got = check_max_angle(
            &line(&case["line"]),
            &anchor_in(&case["anchor"]),
            num(&case["labelLength"]),
            num(&case["windowSize"]),
            num(&case["maxAngle"]),
        );
        assert_eq!(got, case["out"].as_bool().unwrap(), "case {i}");
    }
}

fn extent(case: &Value) -> LabelExtent {
    let width = |v: &Value| (!v.is_null()).then(|| num(&v["right"]) - num(&v["left"]));
    LabelExtent {
        text: width(&case["text"]),
        icon: width(&case["icon"]),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn get_anchors_matches_gl_js() {
    let mut total = 0;
    for (i, case) in cases("getAnchors").iter().enumerate() {
        let mut out = Vec::new();
        line_anchors(
            &line(&case["line"]),
            num(&case["spacing"]),
            num(&case["maxAngle"]),
            extent(case),
            num(&case["glyphSize"]),
            num(&case["boxScale"]),
            num(&case["overscaling"]),
            8192.0,
            &mut out,
        );
        let expected = case["out"].as_array().unwrap();
        assert_eq!(out.len(), expected.len(), "case {i}: anchor count");
        for (k, (got, want)) in out.iter().zip(expected).enumerate() {
            assert_anchor(got, want, &format!("case {i} anchor {k}"));
        }
        total += out.len();
    }
    assert!(total > 200, "the cases place anchors ({total})");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn get_center_anchor_matches_gl_js() {
    let mut placed = 0;
    for (i, case) in cases("getCenterAnchor").iter().enumerate() {
        let got = center_anchor(
            &line(&case["line"]),
            num(&case["maxAngle"]),
            extent(case),
            num(&case["glyphSize"]),
            num(&case["boxScale"]),
        );
        match (&got, case["out"].is_null()) {
            (None, true) => {}
            (Some(a), false) => {
                assert_anchor(a, &case["out"], &format!("case {i}"));
                placed += 1;
            }
            _ => panic!("case {i}: {got:?} != {}", case["out"]),
        }
    }
    assert!(placed > 20, "the cases place center anchors ({placed})");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn merge_lines_matches_gl_js() {
    for (i, case) in cases("mergeLines").iter().enumerate() {
        let features = case["features"].as_array().unwrap();
        let texts: Vec<&str> = features
            .iter()
            .map(|f| f["text"].as_str().unwrap_or(""))
            .collect();
        let firsts: Vec<Vec<Point>> = features.iter().map(|f| line(&f["line"])).collect();
        let first_refs: Vec<&[Point]> = firsts.iter().map(Vec::as_slice).collect();
        let merged = merge_lines(&texts, &first_refs);
        let expected = case["out"].as_array().unwrap();
        assert_eq!(merged.len(), expected.len(), "case {i}: feature count");
        for (got, want) in merged.iter().zip(expected) {
            assert_eq!(
                got.feature as u64,
                want["index"].as_u64().unwrap(),
                "case {i}"
            );
            let line_got = got.joined.as_deref().unwrap_or(first_refs[got.feature]);
            assert_eq!(line_got, &line(&want["line"])[..], "case {i}");
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pole_of_inaccessibility_matches_gl_js() {
    for (i, case) in cases("pole").iter().enumerate() {
        let polygon = lines(&case["polygon"]);
        let rings: Vec<&[Point]> = polygon.iter().map(Vec::as_slice).collect();
        let got = pole_of_inaccessibility(&rings, num(&case["precision"]));
        assert_point(got, &case["out"], &format!("case {i}"));
    }
}

fn positioned<'a>(image: &'a crate::icon::IconImage, v: &Value) -> PositionedIcon<'a> {
    PositionedIcon {
        image,
        top: num(&v["top"]),
        bottom: num(&v["bottom"]),
        left: num(&v["left"]),
        right: num(&v["right"]),
        collision_padding: (!v["collisionPadding"].is_null()).then(|| nums(&v["collisionPadding"])),
    }
}

#[track_caller]
fn assert_icon(got: &PositionedIcon<'_>, expected: &Value, what: &str) {
    for (field, value) in [
        ("top", got.top),
        ("bottom", got.bottom),
        ("left", got.left),
        ("right", got.right),
    ] {
        assert_same(value, num(&expected[field]), &format!("{what}.{field}"));
    }
    match (
        got.collision_padding,
        expected["collisionPadding"].is_null(),
    ) {
        (None, true) => {}
        (Some(p), false) => {
            let want: [f64; 4] = nums(&expected["collisionPadding"]);
            for k in 0..4 {
                assert_same(p[k], want[k], &format!("{what}.collisionPadding[{k}]"));
            }
        }
        _ => panic!("{what}: collision padding {:?}", got.collision_padding),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn shape_icon_matches_gl_js() {
    for (i, case) in cases("shapeIcon").iter().enumerate() {
        let image = image(&case["image"]);
        let got = shape_icon(
            &image,
            nums(&case["offset"]),
            anchor(case["anchor"].as_str().unwrap()),
        );
        assert_icon(&got, &case["out"], &format!("case {i}"));
    }
}

fn shaping_box(v: &Value) -> Shaping {
    Shaping {
        lines: Vec::new(),
        top: num(&v["top"]),
        bottom: num(&v["bottom"]),
        left: num(&v["left"]),
        right: num(&v["right"]),
        writing_mode: WritingMode::Horizontal,
        text: String::new(),
        icons_in_text: false,
        verticalizable: false,
        sections: Vec::new(),
    }
}

fn icon_text_fit(name: &str) -> IconTextFit {
    match name {
        "width" => IconTextFit::Width,
        "height" => IconTextFit::Height,
        "both" => IconTextFit::Both,
        _ => IconTextFit::None,
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fit_icon_to_text_matches_gl_js() {
    for (i, case) in cases("fitIconToText").iter().enumerate() {
        let image = image(&case["image"]);
        let icon = positioned(&image, &case["icon"]);
        let got = fit_icon_to_text(
            &icon,
            &shaping_box(&case["text"]),
            icon_text_fit(case["fit"].as_str().unwrap()),
            nums(&case["padding"]),
            nums(&case["offset"]),
            num(&case["fontScale"]),
        );
        assert_icon(&got, &case["out"], &format!("case {i}"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn apply_text_fit_matches_gl_js() {
    for (i, case) in cases("applyTextFit").iter().enumerate() {
        let image = image(&case["image"]);
        let mut icon = case["icon"].clone();
        icon["collisionPadding"] = Value::Null;
        let got = apply_text_fit(&positioned(&image, &icon));
        let want: [f64; 4] = nums(&case["out"]);
        for (k, v) in [got.x1, got.y1, got.x2, got.y2].into_iter().enumerate() {
            assert_same(v, want[k], &format!("case {i}[{k}]"));
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn icon_quads_match_gl_js() {
    let mut pieces = 0;
    for (i, case) in cases("iconQuads").iter().enumerate() {
        let image = image(&case["image"]);
        let icon = positioned(&image, &case["icon"]);
        let mut quads = Vec::new();
        icon_quads(
            &icon,
            0,
            num(&case["rotate"]),
            case["sdf"].as_bool().unwrap(),
            case["hasIconTextFit"].as_bool().unwrap(),
            &mut quads,
        );
        let expected = case["out"].as_array().unwrap();
        assert_eq!(quads.len(), expected.len(), "case {i}: quad count");
        for (k, (q, e)) in quads.iter().zip(expected).enumerate() {
            let what = format!("case {i} quad {k}");
            assert_point(q.tl, &e["tl"], &format!("{what} tl"));
            assert_point(q.tr, &e["tr"], &format!("{what} tr"));
            assert_point(q.bl, &e["bl"], &format!("{what} bl"));
            assert_point(q.br, &e["br"], &format!("{what} br"));
            let tex: [f64; 4] = nums(&e["tex"]);
            for (j, v) in [q.tex.x, q.tex.y, q.tex.w, q.tex.h].into_iter().enumerate() {
                assert_same(v, tex[j], &format!("{what} tex[{j}]"));
            }
            assert_point(
                q.pixel_offset_tl,
                &e["pixelOffsetTL"],
                &format!("{what} pixelOffsetTL"),
            );
            assert_point(
                q.pixel_offset_br,
                &e["pixelOffsetBR"],
                &format!("{what} pixelOffsetBR"),
            );
            let scale: [f64; 2] = nums(&e["minFontScale"]);
            assert_same(
                q.min_font_scale[0],
                scale[0],
                &format!("{what} minFontScaleX"),
            );
            assert_same(
                q.min_font_scale[1],
                scale[1],
                &format!("{what} minFontScaleY"),
            );
            assert_eq!(q.sdf, e["sdf"].as_bool().unwrap(), "{what} sdf");
        }
        pieces += quads.len();
    }
    assert!(pieces > 100, "nine-patch pieces are covered ({pieces})");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn collision_features_match_gl_js() {
    let (mut boxes, mut circles) = (0, 0);
    for (i, case) in cases("collisionFeature").iter().enumerate() {
        let image = (!case["image"].is_null()).then(|| image(&case["image"]));
        let icon = image
            .as_ref()
            .map(|image| positioned(image, &case["shape"]));
        let shape = match &icon {
            Some(icon) => Shape::icon(icon),
            None => Shape {
                top: num(&case["shape"]["top"]),
                bottom: num(&case["shape"]["bottom"]),
                left: num(&case["shape"]["left"]),
                right: num(&case["shape"]["right"]),
                icon: None,
            },
        };
        let got = collision_feature(
            shape,
            num(&case["boxScale"]),
            nums(&case["padding"]),
            case["alignLine"].as_bool().unwrap(),
            num(&case["rotate"]),
        );
        let out = &case["out"];
        match got {
            CollisionFeature::Box(b) => {
                let want: [f64; 4] = nums(&out["box"]);
                let got = [b.x1, b.y1, b.x2, b.y2].map(to_i16);
                assert_eq!(got, want.map(|v| v as i16), "case {i} box");
                assert!(out["circle"].is_null(), "case {i}");
                boxes += 1;
            }
            CollisionFeature::Circles(d) => {
                assert!(out["box"].is_null(), "case {i}");
                assert_same(d, num(&out["circle"]), &format!("case {i} circle"));
                circles += 1;
            }
            CollisionFeature::None => {
                assert!(out["box"].is_null() && out["circle"].is_null(), "case {i}");
            }
        }
    }
    assert!(
        boxes > 50 && circles > 10,
        "{boxes} boxes, {circles} circles"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn variable_offsets_match_gl_js() {
    for (i, case) in cases("variableOffset").iter().enumerate() {
        let [a, b] = nums(&case["offset"]);
        let offset = if b == f64::INFINITY {
            TextOffset::Radial(a)
        } else {
            TextOffset::Xy([a, b])
        };
        let got = variable_offset(anchor(case["anchor"].as_str().unwrap()), offset);
        let want: [f64; 2] = nums(&case["out"]);
        assert_same(got[0], want[0], &format!("case {i} x"));
        assert_same(got[1], want[1], &format!("case {i} y"));
    }
}

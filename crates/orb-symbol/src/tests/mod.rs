//! Golden vectors from GL JS (`tests/data/README.md`) and unit tests.

mod golden;
mod layouts;
mod unit;

use orb_text::Anchor;
use serde_json::Value;

use crate::geometry::Point;
use crate::icon::{IconImage, TextFit};

/// A golden number: JSON numbers, or `"NaN"`, `"Infinity"`, `"-Infinity"`
/// and `"-0"` for what JSON can't hold.
pub(crate) fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().expect("a number"),
        Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            "-0" => -0.0,
            other => panic!("not a number: {other}"),
        },
        other => panic!("not a number: {other}"),
    }
}

pub(crate) fn nums<const N: usize>(v: &Value) -> [f64; N] {
    let a = v.as_array().expect("an array");
    assert_eq!(a.len(), N, "{v}");
    std::array::from_fn(|i| num(&a[i]))
}

pub(crate) fn point(v: &Value) -> Point {
    let [x, y] = nums(v);
    Point::new(x, y)
}

pub(crate) fn line(v: &Value) -> Vec<Point> {
    v.as_array().expect("a line").iter().map(point).collect()
}

pub(crate) fn lines(v: &Value) -> Vec<Vec<Point>> {
    v.as_array().expect("lines").iter().map(line).collect()
}

pub(crate) fn anchor(name: &str) -> Anchor {
    match name {
        "center" => Anchor::Center,
        "left" => Anchor::Left,
        "right" => Anchor::Right,
        "top" => Anchor::Top,
        "bottom" => Anchor::Bottom,
        "top-left" => Anchor::TopLeft,
        "top-right" => Anchor::TopRight,
        "bottom-left" => Anchor::BottomLeft,
        "bottom-right" => Anchor::BottomRight,
        other => panic!("unknown anchor {other}"),
    }
}

fn text_fit(v: &Value) -> Option<TextFit> {
    match v.as_str()? {
        "stretchOrShrink" => Some(TextFit::StretchOrShrink),
        "stretchOnly" => Some(TextFit::StretchOnly),
        "proportional" => Some(TextFit::Proportional),
        other => panic!("unknown text fit {other}"),
    }
}

fn zones(v: &Value) -> Option<Vec<[f64; 2]>> {
    v.as_array().map(|zones| zones.iter().map(nums).collect())
}

/// An image spec of the generator.
pub(crate) fn image(v: &Value) -> IconImage {
    IconImage {
        size: [num(&v["width"]) as u32, num(&v["height"]) as u32],
        pixel_ratio: num(&v["pixelRatio"]),
        sdf: v["sdf"].as_bool().unwrap_or(false),
        stretch_x: zones(&v["stretchX"]),
        stretch_y: zones(&v["stretchY"]),
        content: (!v["content"].is_null()).then(|| nums(&v["content"])),
        text_fit_width: text_fit(&v["textFitWidth"]),
        text_fit_height: text_fit(&v["textFitHeight"]),
    }
}

/// Exact equality of doubles, NaN included.
#[track_caller]
pub(crate) fn assert_same(actual: f64, expected: f64, what: &str) {
    assert!(
        actual.to_bits() == expected.to_bits() || (actual == expected && actual == 0.0),
        "{what}: {actual:?} != {expected:?}"
    );
}

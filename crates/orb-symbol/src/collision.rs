//! Collision features: GL JS `collision_feature.ts`, line by line.

use crate::geometry::{Point, degrees_to_radians, js_max, js_min};
use crate::icon::{Box2, PositionedIcon, apply_text_fit};

/// The box a shaped label or icon spans around its anchor, in layout units.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape<'a> {
    pub(crate) top: f64,
    pub(crate) bottom: f64,
    pub(crate) left: f64,
    pub(crate) right: f64,
    /// For icons: the icon, whose image may fit its content to text.
    pub(crate) icon: Option<&'a PositionedIcon<'a>>,
}

impl<'a> Shape<'a> {
    pub(crate) fn text(shaping: &orb_text::Shaping) -> Self {
        Self {
            top: shaping.top,
            bottom: shaping.bottom,
            left: shaping.left,
            right: shaping.right,
            icon: None,
        }
    }

    pub(crate) fn icon(icon: &'a PositionedIcon<'a>) -> Self {
        Self {
            top: icon.top,
            bottom: icon.bottom,
            left: icon.left,
            right: icon.right,
            icon: Some(icon),
        }
    }

    fn collision_padding(&self) -> Option<[f64; 4]> {
        self.icon.and_then(|i| i.collision_padding)
    }
}

/// What a label collides as.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CollisionFeature {
    /// A box around the anchor, in tile units: `x1, y1, x2, y2`.
    Box(Box2),
    /// Line labels collide as circles placed along the line at placement;
    /// this is their diameter in layout units.
    Circles(f64),
    /// A line label with no height: nothing collides.
    None,
}

impl CollisionFeature {
    pub(crate) fn circle_diameter(self) -> Option<f64> {
        match self {
            Self::Circles(d) => Some(d),
            _ => None,
        }
    }
}

/// GL JS `new CollisionFeature(...)`. `padding` is CSS order, in tile units;
/// `rotate` is in degrees.
pub(crate) fn collision_feature(
    shape: Shape<'_>,
    box_scale: f64,
    padding: [f64; 4],
    align_line: bool,
    rotate: f64,
) -> CollisionFeature {
    if align_line {
        let mut top = shape.top;
        let mut bottom = shape.bottom;
        if let Some(p) = shape.collision_padding() {
            top -= p[1];
            bottom += p[3];
        }
        let height = bottom - top;
        if height > 0.0 {
            return CollisionFeature::Circles(js_max(10.0, height));
        }
        return CollisionFeature::None;
    }

    let mut b = match shape.icon {
        Some(icon) if icon.image.has_text_fit() => apply_text_fit(icon),
        _ => Box2 {
            x1: shape.left,
            y1: shape.top,
            x2: shape.right,
            y2: shape.bottom,
        },
    };
    b.y1 = b.y1 * box_scale - padding[0];
    b.y2 = b.y2 * box_scale + padding[2];
    b.x1 = b.x1 * box_scale - padding[3];
    b.x2 = b.x2 * box_scale + padding[1];

    if let Some(p) = shape.collision_padding() {
        b.x1 -= p[0] * box_scale;
        b.y1 -= p[1] * box_scale;
        b.x2 += p[2] * box_scale;
        b.y2 += p[3] * box_scale;
    }

    if rotate != 0.0 && !rotate.is_nan() {
        let radians = degrees_to_radians(rotate);
        let tl = Point::new(b.x1, b.y1).rotate(radians);
        let tr = Point::new(b.x2, b.y1).rotate(radians);
        let bl = Point::new(b.x1, b.y2).rotate(radians);
        let br = Point::new(b.x2, b.y2).rotate(radians);
        let min4 = |a: f64, b: f64, c: f64, d: f64| js_min(js_min(js_min(a, b), c), d);
        let max4 = |a: f64, b: f64, c: f64, d: f64| js_max(js_max(js_max(a, b), c), d);
        b.x1 = min4(tl.x, tr.x, bl.x, br.x);
        b.x2 = max4(tl.x, tr.x, bl.x, br.x);
        b.y1 = min4(tl.y, tr.y, bl.y, br.y);
        b.y2 = max4(tl.y, tr.y, bl.y, br.y);
    }
    CollisionFeature::Box(b)
}

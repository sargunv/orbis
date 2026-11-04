//! Text offsets per anchor: GL JS `variable_text_anchor.ts`.

use std::f64::consts::SQRT_2;

use orb_text::{Anchor, ONE_EM};

/// The radial offset is to the edge of the text box, and glyphs "start" at
/// the baseline, assumed `ONE_EM - 17` below the top (GL JS
/// `baselineOffset`).
const BASELINE_OFFSET: f64 = 7.0;

/// A text offset in layout units: `text-offset`, or `text-radial-offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TextOffset {
    Xy([f64; 2]),
    Radial(f64),
}

/// The offset of a label placed at `anchor` (GL JS
/// `evaluateVariableOffset`).
pub(crate) fn variable_offset(anchor: Anchor, offset: TextOffset) -> [f64; 2] {
    use Anchor::*;
    match offset {
        TextOffset::Radial(radial) => {
            let radial = if radial < 0.0 { 0.0 } else { radial };
            let hypotenuse = radial / SQRT_2;
            let y = match anchor {
                TopRight | TopLeft => hypotenuse - BASELINE_OFFSET,
                BottomRight | BottomLeft => -hypotenuse + BASELINE_OFFSET,
                Bottom => -radial + BASELINE_OFFSET,
                Top => radial - BASELINE_OFFSET,
                _ => 0.0,
            };
            let x = match anchor {
                TopRight | BottomRight => -hypotenuse,
                TopLeft | BottomLeft => hypotenuse,
                Left => radial,
                Right => -radial,
                _ => 0.0,
            };
            [x, y]
        }
        TextOffset::Xy([x, y]) => {
            let (x, y) = (x.abs(), y.abs());
            let y = match anchor {
                TopRight | TopLeft | Top => y - BASELINE_OFFSET,
                BottomRight | BottomLeft | Bottom => -y + BASELINE_OFFSET,
                _ => 0.0,
            };
            let x = match anchor {
                TopRight | BottomRight | Right => -x,
                TopLeft | BottomLeft | Left => x,
                _ => 0.0,
            };
            [x, y]
        }
    }
}

/// `text-variable-anchor-offset` in ems, converted to layout units with
/// the baseline shift (the first branch of GL JS
/// `getTextVariableAnchorOffset`).
pub(crate) fn from_anchor_offsets(
    values: &[(Anchor, [f64; 2])],
    out: &mut Vec<(Anchor, [f64; 2])>,
) {
    for &(anchor, [x, y]) in values {
        let mut offset = [x * ONE_EM, y * ONE_EM];
        if matches!(anchor, Anchor::Top | Anchor::TopLeft | Anchor::TopRight) {
            offset[1] -= BASELINE_OFFSET;
        } else if matches!(
            anchor,
            Anchor::Bottom | Anchor::BottomLeft | Anchor::BottomRight
        ) {
            offset[1] += BASELINE_OFFSET;
        }
        out.push((anchor, offset));
    }
}

/// `text-variable-anchor` with one offset for every anchor (the second
/// branch of `getTextVariableAnchorOffset`).
pub(crate) fn from_anchors(
    anchors: &[Anchor],
    offset: TextOffset,
    out: &mut Vec<(Anchor, [f64; 2])>,
) {
    for &anchor in anchors {
        out.push((anchor, variable_offset(anchor, offset)));
    }
}

/// GL JS `TextAnchorEnum`.
pub(crate) fn anchor_code(anchor: Anchor) -> u32 {
    match anchor {
        Anchor::Center => 1,
        Anchor::Left => 2,
        Anchor::Right => 3,
        Anchor::Top => 4,
        Anchor::Bottom => 5,
        Anchor::TopLeft => 6,
        Anchor::TopRight => 7,
        Anchor::BottomLeft => 8,
        Anchor::BottomRight => 9,
    }
}

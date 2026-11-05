//! An image stretched over four corners on a plane, as v8 `image` sources
//! warp theirs (GL JS's `calculateImageWarp`): the perspective view of the
//! image's plane where the corners describe one, blended toward a bilinear
//! rubber sheet as they stop doing so.

use glam::DVec2;

use crate::WorldRect;

/// Foreshortening, the spread of the projective map's denominator over the
/// quad, at which the warp starts blending toward bilinear, and at which it
/// is wholly bilinear.
const PROJECTIVE_FORESHORTENING: f64 = 4.0;
const BILINEAR_FORESHORTENING: f64 = 512.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageQuad {
    /// Top left, top right, bottom right, bottom left, in world units.
    corners: [DVec2; 4],
    /// The projective map's perspective terms, and its blend toward the
    /// bilinear one (1: wholly bilinear).
    warp: [f64; 3],
}

impl ImageQuad {
    pub fn new(corners: [DVec2; 4]) -> Self {
        Self {
            corners,
            warp: warp(corners),
        }
    }

    /// The world position of image position `uv` (fractions, y down).
    pub fn at(&self, uv: DVec2) -> DVec2 {
        let [tl, tr, br, bl] = self.corners;
        let [px, py, blend] = self.warp;
        let bilinear = tl.lerp(tr, uv.x).lerp(bl.lerp(br, uv.x), uv.y);
        if blend >= 1.0 {
            return bilinear;
        }
        let denominator = px * uv.x + py * uv.y + 1.0;
        let across = tr - tl + tr * px;
        let down = bl - tl + bl * py;
        let projective = (across * uv.x + down * uv.y + tl) / denominator;
        projective.lerp(bilinear, blend)
    }

    /// The weight that makes linear interpolation of image positions across
    /// a triangle projective: the reciprocal of the projective map's
    /// denominator at `uv`, blended toward 1 with the warp (GL JS's
    /// `perspectiveRatio`). Image positions interpolate as `uv · w` over
    /// `w`.
    pub fn weight(&self, uv: DVec2) -> f64 {
        let [px, py, blend] = self.warp;
        let denominator = px * uv.x + py * uv.y + 1.0;
        (1.0 / denominator) * (1.0 - blend) + blend
    }

    /// The rectangle holding the image.
    pub fn bounds(&self) -> WorldRect {
        WorldRect::around(self.corners)
    }
}

fn warp([tl, tr, br, bl]: [DVec2; 4]) -> [f64; 3] {
    const BILINEAR: [f64; 3] = [0.0, 0.0, 1.0];
    // A parallelogram's warp is affine either way.
    let scale = (tr - tl)
        .length()
        .max((bl - tl).length())
        .max(f64::MIN_POSITIVE);
    if (tl + br - tr - bl).length() <= scale * 1e-12 {
        return BILINEAR;
    }
    let sum = tl - tr + br - bl;
    let (right, down) = (tr - br, bl - br);
    let determinant = right.x * down.y - down.x * right.y;
    let px = (sum.x * down.y - down.x * sum.y) / determinant;
    let py = (right.x * sum.y - sum.x * right.y) / determinant;
    let denominators = [1.0, 1.0 + px, 1.0 + px + py, 1.0 + py];
    let max = denominators
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min = denominators.iter().copied().fold(f64::INFINITY, f64::min);
    let foreshortening = max / min;
    let blend = ((1.0 - PROJECTIVE_FORESHORTENING / foreshortening)
        / (1.0 - PROJECTIVE_FORESHORTENING / BILINEAR_FORESHORTENING))
        .max(0.0);
    if !(1.0..=BILINEAR_FORESHORTENING).contains(&foreshortening) || blend >= 1.0 {
        return BILINEAR;
    }
    [px, py, blend]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_map_to_corners() {
        let quads = [
            [[0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]],
            [[0.0, 10.0], [10.0, 12.0], [11.0, 0.0], [1.0, -2.0]],
            [[2.0, 10.0], [8.0, 10.0], [10.0, 0.0], [0.0, 0.0]],
        ];
        for corners in quads {
            let corners = corners.map(DVec2::from);
            let quad = ImageQuad::new(corners);
            for (uv, c) in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                .into_iter()
                .zip(corners)
            {
                assert!(
                    quad.at(DVec2::from(uv)).distance(c) < 1e-9,
                    "{corners:?} {uv:?}"
                );
            }
        }
        // A trapezoid is a perspective view: its middle is nearer the far
        // (short) edge than bilinear's.
        let trapezoid = ImageQuad::new(quads[2].map(DVec2::from));
        assert!(trapezoid.at(DVec2::splat(0.5)).y > 5.0);
    }
}

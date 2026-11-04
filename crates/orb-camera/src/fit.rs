//! `cameraForBounds`: the camera that shows a box, after GL JS, and at any
//! pitch.

use glam::DVec2;
use orb_crs::Projection;

use crate::animate::CameraTarget;
use crate::moves::Bounds;
use crate::{CameraState, View, Viewport};

/// Space to leave at the viewport's edges, in logical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Padding {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Padding {
    pub const fn uniform(p: f64) -> Self {
        Self {
            top: p,
            right: p,
            bottom: p,
            left: p,
        }
    }
}

/// GL JS's `CameraForBoundsOptions`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FitOptions {
    pub padding: Padding,
    /// Pixels the box's centre ends up from the padded area's centre.
    pub offset: DVec2,
    /// Never zoom in further than this.
    pub max_zoom: Option<f64>,
    /// The bearing to fit at; the current one if unset.
    pub bearing: Option<f64>,
    /// The pitch to fit at; the current one if unset. GL JS fits as if
    /// looking straight down; a pitched fit here keeps the whole box on
    /// screen.
    pub pitch: Option<f64>,
}

impl CameraState {
    /// The camera that shows `bounds` inside the padded viewport, or `None`
    /// when the padding leaves no room.
    pub fn for_bounds(
        &self,
        bounds: Bounds,
        options: &FitOptions,
        projection: &Projection,
        viewport: Viewport,
    ) -> Option<CameraTarget> {
        let pad = options.padding;
        let available = DVec2::new(
            viewport.width - pad.left - pad.right,
            viewport.height - pad.top - pad.bottom,
        );
        if available.x <= 0.0 || available.y <= 0.0 {
            return None;
        }
        let bearing = options.bearing.unwrap_or(self.bearing);
        let pitch = options.pitch.unwrap_or(self.pitch);
        let max_zoom = options.max_zoom.unwrap_or(f64::INFINITY);

        // Looking straight down (GL JS `cameraForBoxAndBearing`): the box's
        // outline in a y-down frame turned by the bearing.
        let outline = bounds.outline(projection);
        let angle = (-bearing).to_radians();
        let turn = DVec2::new(angle.cos(), angle.sin());
        let mut min = DVec2::splat(f64::INFINITY);
        let mut max = DVec2::splat(f64::NEG_INFINITY);
        for point in &outline {
            let p = DVec2::new(point.x, -point.y).rotate(turn);
            min = min.min(p);
            max = max.max(p);
        }
        let size = max - min;
        if !size.is_finite() {
            return None;
        }
        let px_per_world = (available.x / size.x).min(available.y / size.y);
        let world_per_px = self.world_per_px(projection);
        let zoom = (self.zoom() + (world_per_px * px_per_world).log2()).min(max_zoom);
        let final_world_per_px = world_per_px * (self.zoom() - zoom).exp2();
        let shift = DVec2::new((pad.left - pad.right) / 2.0, (pad.top - pad.bottom) / 2.0).rotate(
            DVec2::new(bearing.to_radians().cos(), bearing.to_radians().sin()),
        ) + options.offset.rotate(DVec2::new(
            bearing.to_radians().cos(),
            bearing.to_radians().sin(),
        ));
        // A rectangle's middle is its diagonal's; a curved outline's is its
        // turned bounds', turned back.
        let middle = if outline.len() == 4 {
            (outline[0] + outline[2]) / 2.0
        } else {
            let m = ((min + max) / 2.0).rotate(DVec2::new(angle.cos(), -angle.sin()));
            DVec2::new(m.x, -m.y)
        };
        let center = middle - DVec2::new(shift.x, -shift.y) * final_world_per_px;

        let mut camera = *self;
        camera.bearing = bearing;
        camera.pitch = pitch;
        camera.set_target_world(projection, center, final_world_per_px);
        if pitch > 0.0 {
            fit_pitched(
                &mut camera,
                &outline,
                options,
                projection,
                viewport,
                available,
            );
            camera.set_zoom(camera.zoom().min(max_zoom));
        }
        Some(CameraTarget {
            center: Some(camera.target),
            zoom: Some(camera.zoom()),
            bearing: Some(bearing),
            pitch: Some(pitch),
            ..CameraTarget::default()
        })
    }
}

/// Refines a top-down fit for a pitched camera: the box's corners project
/// to a trapezoid, so zoom until its screen bounds fill the padded area and
/// pan their middle onto the area's. Converges in a few steps, as the
/// projection is nearly a scale at a fixed pitch.
fn fit_pitched(
    camera: &mut CameraState,
    corners: &[DVec2],
    options: &FitOptions,
    projection: &Projection,
    viewport: Viewport,
    available: DVec2,
) {
    let pad = options.padding;
    let goal =
        DVec2::new(pad.left + available.x / 2.0, pad.top + available.y / 2.0) + options.offset;
    for _ in 0..16 {
        let view = View::new(camera, viewport, projection);
        let Some(points) = corners
            .iter()
            .map(|&c| view.project(c))
            .collect::<Option<Vec<_>>>()
            .filter(|points| points.iter().all(|p| p.y >= view.ground_horizon()))
        else {
            // A corner is past the horizon: back away and try again.
            camera.set_zoom(camera.zoom() - 1.0);
            continue;
        };
        let min = points.iter().fold(DVec2::INFINITY, |a, p| a.min(*p));
        let max = points.iter().fold(DVec2::NEG_INFINITY, |a, p| a.max(*p));
        let scale = (available.x / (max.x - min.x)).min(available.y / (max.y - min.y));
        let middle = (min + max) / 2.0;
        if scale.log2().abs() < 1e-4 && middle.distance(goal) < 0.05 {
            return;
        }
        camera.pan(projection, viewport, middle, goal);
        let center = DVec2::new(viewport.width, viewport.height) / 2.0;
        camera.zoom_around(projection, viewport, center, camera.zoom() + scale.log2());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_crs::TileMatrixSet;
    use orb_geo::Geodetic;

    const VIEWPORT: Viewport = Viewport {
        width: 800.0,
        height: 600.0,
    };

    fn on_screen(camera: &CameraState, target: &CameraTarget, projection: &Projection) -> View {
        let mut fitted = *camera;
        fitted.target = target.center.unwrap();
        fitted.set_zoom(target.zoom.unwrap());
        fitted.bearing = target.bearing.unwrap();
        fitted.pitch = target.pitch.unwrap();
        View::new(&fitted, VIEWPORT, projection)
    }

    #[test]
    fn fits_a_box_inside_the_padding() {
        let projection = *TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection();
        let bounds = Bounds {
            west: 139.6,
            south: 35.6,
            east: 139.9,
            north: 35.8,
        };
        let padding = Padding {
            top: 20.0,
            right: 40.0,
            bottom: 60.0,
            left: 80.0,
        };
        for (bearing, pitch) in [(0.0, 0.0), (30.0, 0.0), (0.0, 50.0), (-70.0, 60.0)] {
            let camera = CameraState::new(Geodetic::new(0.0, 0.0), 2.0);
            let options = FitOptions {
                padding,
                bearing: Some(bearing),
                pitch: Some(pitch),
                ..FitOptions::default()
            };
            let target = camera
                .for_bounds(bounds, &options, &projection, VIEWPORT)
                .unwrap();
            let view = on_screen(&camera, &target, &projection);
            let points: Vec<_> = bounds
                .corners()
                .iter()
                .map(|c| view.project(projection.forward(*c)).unwrap())
                .collect();
            let min = points.iter().fold(DVec2::INFINITY, |a, p| a.min(*p));
            let max = points.iter().fold(DVec2::NEG_INFINITY, |a, p| a.max(*p));
            let tolerance = 0.5;
            assert!(
                min.x >= padding.left - tolerance,
                "{bearing} {pitch}: {min}"
            );
            assert!(min.y >= padding.top - tolerance, "{bearing} {pitch}: {min}");
            assert!(max.x <= VIEWPORT.width - padding.right + tolerance, "{max}");
            assert!(
                max.y <= VIEWPORT.height - padding.bottom + tolerance,
                "{max}"
            );
            // It fills one dimension.
            let filled = (max.x - min.x - (VIEWPORT.width - 120.0))
                .abs()
                .min((max.y - min.y - (VIEWPORT.height - 80.0)).abs());
            assert!(filled < 1.0, "{bearing} {pitch}: {min} {max}");
        }
        let cramped = FitOptions {
            padding: Padding::uniform(400.0),
            ..FitOptions::default()
        };
        let camera = CameraState::new(Geodetic::new(0.0, 0.0), 2.0);
        assert!(
            camera
                .for_bounds(bounds, &cramped, &projection, VIEWPORT)
                .is_none()
        );
    }
}

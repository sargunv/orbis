//! `cameraForBounds`: the camera that shows a box, after GL JS, and at any
//! pitch.

use glam::DVec2;

use crate::animate::CameraTarget;
use crate::moves::Bounds;
use crate::{CameraState, View, Viewport};
use orb_crs::{Surface, WorldRect};

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

    pub(crate) fn lerp(self, to: Self, k: f64) -> Self {
        let at = |a: f64, b: f64| a + (b - a) * k;
        Self {
            top: at(self.top, to.top),
            right: at(self.right, to.right),
            bottom: at(self.bottom, to.bottom),
            left: at(self.left, to.left),
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
    /// The camera that shows `bounds` inside the padded viewport (the
    /// fit's padding inside the camera's own), or `None` when the padding
    /// leaves no room.
    pub fn for_bounds(
        &self,
        bounds: Bounds,
        options: &FitOptions,
        surface: &Surface,
        viewport: Viewport,
    ) -> Option<CameraTarget> {
        let projection = &surface.projection;
        let pad = options.padding;
        let own = self.padding;
        let available = DVec2::new(
            viewport.width - pad.left - pad.right - own.left - own.right,
            viewport.height - pad.top - pad.bottom - own.top - own.bottom,
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
        let turned = WorldRect::around(outline.iter().map(|p| DVec2::new(p.x, -p.y).rotate(turn)));
        let size = turned.size();
        if !size.is_finite() {
            return None;
        }
        let px_per_world = (available.x / size.x).min(available.y / size.y);
        let world_per_px = self.world_per_px(projection);
        let zoom = (self.zoom(projection) + (world_per_px * px_per_world).log2()).min(max_zoom);
        let final_world_per_px = world_per_px * (self.zoom(projection) - zoom).exp2();
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
            let m = turned
                .center()
                .rotate(DVec2::new(angle.cos(), -angle.sin()));
            DVec2::new(m.x, -m.y)
        };
        let center = middle - DVec2::new(shift.x, -shift.y) * final_world_per_px;

        let mut camera = *self;
        camera.bearing = bearing;
        camera.pitch = pitch;
        camera.set_target_world(projection, center, final_world_per_px);
        if pitch > 0.0 {
            fit_pitched(&mut camera, &outline, options, surface, viewport, available);
            camera.set_zoom(projection, camera.zoom(projection).min(max_zoom));
        }
        Some(CameraTarget {
            center: Some(camera.target),
            zoom: Some(camera.zoom(projection)),
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
    surface: &Surface,
    viewport: Viewport,
    available: DVec2,
) {
    let projection = &surface.projection;
    let (pad, own) = (options.padding, camera.padding);
    let goal = DVec2::new(
        pad.left + own.left + available.x / 2.0,
        pad.top + own.top + available.y / 2.0,
    ) + options.offset;
    for _ in 0..16 {
        let view = View::new(camera, viewport, surface);
        let Some(points) = corners
            .iter()
            .map(|&c| view.project(c))
            .collect::<Option<Vec<_>>>()
            .filter(|points| points.iter().all(|p| p.y >= view.ground_horizon()))
        else {
            // A corner is past the horizon: back away and try again.
            camera.set_zoom(projection, camera.zoom(projection) - 1.0);
            continue;
        };
        let rect = WorldRect::around(points.iter().copied());
        let (size, middle) = (rect.size(), rect.center());
        let scale = (available.x / size.x).min(available.y / size.y);
        if scale.log2().abs() < 1e-4 && middle.distance(goal) < 0.05 {
            return;
        }
        camera.pan(surface, viewport, middle, goal);
        let center = camera.center(viewport);
        camera.zoom_around(
            surface,
            viewport,
            center,
            camera.zoom(&surface.projection) + scale.log2(),
        );
    }
}

#[cfg(test)]
mod tests {
    fn mercator() -> orb_crs::Projection {
        *orb_crs::TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection()
    }

    use super::*;
    use orb_crs::Geodetic;
    use orb_crs::TileMatrixSet;

    const VIEWPORT: Viewport = Viewport {
        width: 800.0,
        height: 600.0,
    };

    fn on_screen(camera: &CameraState, target: &CameraTarget, surface: &Surface) -> View {
        let mut fitted = *camera;
        fitted.target = target.center.unwrap();
        fitted.set_zoom(&mercator(), target.zoom.unwrap());
        fitted.bearing = target.bearing.unwrap();
        fitted.pitch = target.pitch.unwrap();
        View::new(&fitted, VIEWPORT, surface)
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
            let camera = CameraState::new(Geodetic::new(0.0, 0.0), 2.0, &mercator());
            let options = FitOptions {
                padding,
                bearing: Some(bearing),
                pitch: Some(pitch),
                ..FitOptions::default()
            };
            let target = camera
                .for_bounds(bounds, &options, &Surface::from(&projection), VIEWPORT)
                .unwrap();
            let view = on_screen(&camera, &target, &Surface::from(&projection));
            let points: Vec<_> = bounds
                .corners()
                .iter()
                .map(|c| view.project(projection.forward(*c)).unwrap())
                .collect();
            let WorldRect { min, max } = WorldRect::around(points.iter().copied());
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
        let camera = CameraState::new(Geodetic::new(0.0, 0.0), 2.0, &mercator());
        assert!(
            camera
                .for_bounds(bounds, &cramped, &Surface::from(&projection), VIEWPORT)
                .is_none()
        );
    }
}

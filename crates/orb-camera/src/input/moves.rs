//! The camera moves a recognizer makes, and how the camera applies them,
//! resolving each against itself then: what a move needs of the camera
//! (where the view's middle is, the ground under a point, the zoom, the
//! momentum a press stopped) is read as it applies, so moves made on one
//! thread apply in order to a camera on another.

use glam::DVec2;

use super::{GestureOptions, Movement, Release};
use crate::animate::Glide;
use crate::{
    AnchoredMomentum, Camera, CameraTarget, EaseOptions, Easing, GestureMomentum, MomentumVelocity,
    PanMomentum,
};

/// One camera move of a recognizer, for [`Camera::apply`].
#[derive(Clone, Debug, PartialEq)]
pub struct Move(pub(super) Step);

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Step {
    /// Input takes the camera, stopping its animations; a gesture's first
    /// press keeps how fast a momentum it stopped still moved, for its
    /// release.
    Hold {
        first: bool,
    },
    /// Input takes the camera unless it holds it already: a scroll.
    Scroll,
    Release,
    /// The platform took the gesture away: the camera's animations stop.
    Cancel,
    /// The ground under `from` moves to `to`.
    Pan {
        from: DVec2,
        to: DVec2,
    },
    ZoomBy {
        around: DVec2,
        levels: f64,
    },
    /// Turns about a point, or the view's middle.
    RotateBy {
        around: Option<DVec2>,
        degrees: f64,
    },
    PitchBy(f64),
    Glide(Box<Glide>),
    /// A double tap's zoom around a point, eased (GL JS, 300 ms).
    TapZoom {
        at: DVec2,
        levels: f64,
    },
    /// Zooms so the box between two corners fills the view, centred on its
    /// middle (GL JS `BoxZoomHandler`); a box without area does nothing.
    ZoomToBox([DVec2; 2]),
    /// A key's eased step (GL JS's `KeyboardHandler`): levels, degrees of
    /// bearing and pitch, and pixels the ground moves.
    Key {
        zoom: f64,
        bearing: f64,
        pitch: f64,
        offset: DVec2,
    },
    /// A gesture's release: its glide and settling.
    Momentum(Box<Plan>),
}

/// A release, resolved into a momentum once the camera applies it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Plan {
    pub release: Release,
    /// Whether it glides on, or only settles.
    pub glide: bool,
    pub options: GestureOptions,
    /// Whether the gesture changed the zoom, bearing and pitch.
    pub changed: [bool; 3],
}

impl Plan {
    /// The momentum the release continues with: each movement's decay,
    /// `caught`, the momentum the gesture's first press stopped, added to
    /// each part going its way, and each part the gesture changed settling
    /// on its snap (Compose's `resolveGestureMomentum`). `None` when
    /// nothing moves or settles.
    fn resolve(&self, caught: Option<MomentumVelocity>) -> Option<GestureMomentum> {
        let (options, release) = (&self.options, self.release);
        let release = if self.glide {
            release
        } else {
            Release::default()
        };
        let boost = caught.unwrap_or_default();
        let fling = |movement: &Movement| movement.momentum.filter(|_| movement.enabled);
        let pan = release
            .pan
            .zip(fling(&options.pan))
            .and_then(|(mut v, fling)| {
                let decay = fling.decay(v.length()).filter(|_| !v.is_nan())?;
                if v.dot(boost.pan) > 0.0 {
                    v += boost.pan;
                }
                // Planned glides plan the boosted speed.
                let decay = fling.decay(v.length()).unwrap_or(decay);
                Some(PanMomentum { velocity: v, decay })
            });
        let part = |v: Option<f64>,
                    boost: f64,
                    movement: &Movement,
                    changed: bool,
                    anchor: Option<DVec2>,
                    make: fn(Option<f64>) -> AnchoredMomentum| {
            let glide = v.zip(fling(movement)).and_then(|(v, fling)| {
                let decay = fling.decay(v.abs())?;
                let v = if v.signum() == boost.signum() {
                    v + boost
                } else {
                    v
                };
                Some((v, fling.decay(v.abs()).unwrap_or(decay)))
            });
            let snap = movement.snap.filter(|_| changed && movement.enabled);
            (glide.is_some() || snap.is_some()).then(|| {
                let base = make(glide.map(|(v, _)| v));
                AnchoredMomentum {
                    decay: glide.map_or(base.decay, |(_, decay)| decay),
                    ..base
                }
                .with_anchor(anchor)
                .with_snap(snap)
            })
        };
        let [zoomed, rotated, pitched] = self.changed;
        let momentum = GestureMomentum {
            pan,
            zoom: part(
                release.zoom,
                boost.zoom,
                &options.zoom,
                zoomed,
                release.zoom_anchor,
                AnchoredMomentum::zoom,
            ),
            bearing: part(
                release.bearing,
                boost.bearing,
                &options.rotate,
                rotated,
                release.bearing_anchor,
                AnchoredMomentum::bearing,
            ),
            pitch: part(
                release.pitch,
                boost.pitch,
                &options.pitch,
                pitched,
                None,
                AnchoredMomentum::pitch,
            ),
        };
        (momentum != GestureMomentum::default()).then_some(momentum)
    }
}

impl Camera {
    /// Applies a recognizer's move at host time `now` (milliseconds), the
    /// time its animations start from. Returns whether input took the
    /// camera from its animations, stopping them: a gesture that begins
    /// or is cancelled, or a step that eases from where the camera is.
    pub fn apply(&mut self, mv: Move, now: f64) -> bool {
        match mv.0 {
            Step::Hold { first } => {
                let caught = self.hold();
                if first {
                    self.caught = caught;
                }
                return true;
            }
            Step::Scroll => {
                if !self.is_held() {
                    self.hold();
                    return true;
                }
            }
            Step::Release => self.release(),
            Step::Cancel => {
                self.stop();
                return true;
            }
            Step::Pan { from, to } => self.pan(from, to),
            Step::ZoomBy { around, levels } => self.zoom_by(around, levels),
            Step::RotateBy { around, degrees } => {
                let around = around.unwrap_or_else(|| self.state().center(self.viewport()));
                self.rotate_by(around, degrees);
            }
            Step::PitchBy(degrees) => self.pitch_by(degrees),
            Step::Glide(glide) => {
                self.glide(*glide, now);
                return true;
            }
            Step::TapZoom { at, levels } => {
                let Some(around) = self.unproject(at) else {
                    return false;
                };
                let target = CameraTarget {
                    zoom: Some(self.zoom() + levels),
                    around: Some(around),
                    ..CameraTarget::default()
                };
                let options = EaseOptions {
                    duration: 300.0,
                    easing: Easing::DEFAULT,
                };
                self.user_ease(&target, options, now);
            }
            Step::ZoomToBox([a, b]) => {
                let size = (b - a).abs();
                let viewport = self.viewport();
                if size.x < 1.0 || size.y < 1.0 {
                    return false;
                }
                let Some(center) = self.unproject((a + b) / 2.0) else {
                    return false;
                };
                let zoom = (viewport.width / size.x)
                    .min(viewport.height / size.y)
                    .log2();
                let target = CameraTarget {
                    center: Some(center),
                    zoom: Some(self.zoom() + zoom),
                    ..CameraTarget::default()
                };
                self.user_ease(&target, EaseOptions::default(), now);
            }
            Step::Key {
                zoom,
                bearing,
                pitch,
                offset,
            } => {
                let state = *self.state();
                let some = |v: f64, at: f64| (v != 0.0).then_some(at + v);
                let target = CameraTarget {
                    center: Some(state.target),
                    zoom: some(zoom, self.zoom()),
                    bearing: some(bearing, state.bearing),
                    pitch: some(pitch, state.pitch),
                    offset,
                    ..CameraTarget::default()
                };
                let options = EaseOptions {
                    duration: 300.0,
                    easing: Easing::EASE_OUT,
                };
                self.user_ease(&target, options, now);
            }
            Step::Momentum(plan) => {
                if let Some(momentum) = plan.resolve(self.caught.take()) {
                    self.start_momentum(&momentum, Some(now));
                }
            }
        }
        false
    }
}

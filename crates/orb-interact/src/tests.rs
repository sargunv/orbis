use glam::DVec2;
use orb_camera::{Camera, CameraEvent, CameraState, Change, Phase, Viewport};
use orb_crs::Geodetic;
use orb_crs::TileMatrixSet;

use super::*;

fn camera() -> Camera {
    let projection = *TileMatrixSet::by_id("WebMercatorQuad")
        .unwrap()
        .projection();
    Camera::new(
        CameraState::new(Geodetic::new(139.7, 35.68), 12.0, &projection),
        projection.into(),
        Viewport {
            width: 800.0,
            height: 600.0,
        },
    )
}

struct Rig {
    camera: Camera,
    input: Interaction,
    now: f64,
}

impl Rig {
    fn new() -> Self {
        Self {
            camera: camera(),
            input: Interaction::default(),
            now: 1000.0,
        }
    }

    fn send(&mut self, input: Input) {
        self.input.handle(&mut self.camera, input, self.now);
    }

    fn down(&mut self, id: u32, kind: PointerKind, x: f64, y: f64) {
        self.send(Input::PointerDown {
            id,
            kind,
            button: Button::Primary,
            position: DVec2::new(x, y),
            modifiers: Modifiers::default(),
        });
    }

    fn to(&mut self, id: u32, x: f64, y: f64) {
        self.send(Input::PointerMove {
            id,
            position: DVec2::new(x, y),
        });
    }

    fn up(&mut self, id: u32, x: f64, y: f64) {
        self.send(Input::PointerUp {
            id,
            position: DVec2::new(x, y),
        });
    }

    /// Runs frames until nothing moves; returns how many.
    fn settle(&mut self) -> u32 {
        let mut frames = 0;
        loop {
            self.now += 16.0;
            let gesture = self.input.frame(&mut self.camera, self.now);
            let moving = self.camera.frame(self.now);
            if !gesture && !moving {
                return frames;
            }
            frames += 1;
            assert!(frames < 1000, "settles");
        }
    }

    fn frame(&mut self) {
        self.now += 16.0;
        self.input.frame(&mut self.camera, self.now);
        self.camera.frame(self.now);
    }
}

#[test]
fn dragging_pans_and_glides_on() {
    let mut rig = Rig::new();
    let ground = rig.camera.unproject(DVec2::new(400.0, 300.0)).unwrap();
    rig.down(1, PointerKind::Mouse, 400.0, 300.0);
    for i in 1..=10 {
        rig.now += 16.0;
        rig.to(1, 400.0 + 10.0 * f64::from(i), 300.0);
        rig.frame();
    }
    // The ground under the pointer follows it.
    let under = rig.camera.project(ground).unwrap();
    assert!(under.distance(DVec2::new(500.0, 300.0)) < 1e-6, "{under}");
    rig.up(1, 500.0, 300.0);
    let frames = rig.settle();
    // GL JS's velocity window: the last 10 px over the 48 ms to release,
    // 62.5 px/s after its linearity, gliding 2.6 px over 83 ms.
    assert!(
        (4..=6).contains(&frames),
        "inertia runs for a while: {frames}"
    );
    let after = rig.camera.project(ground).unwrap();
    assert!(
        (after.x - 502.6).abs() < 0.1,
        "and carries on in the same direction: {after}"
    );
    assert!((after.y - 300.0).abs() < 1e-6);
    let events = rig.camera.take_events();
    let ends = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                CameraEvent::Changed {
                    change: Change::Move,
                    phase: Phase::End,
                    user: true
                }
            )
        })
        .count();
    assert_eq!(ends, 1, "one move, inertia included");
    assert!(rig.input.take_events().is_empty(), "a drag isn't a click");
}

#[test]
fn holding_still_before_release_stops_inertia() {
    let mut rig = Rig::new();
    rig.down(1, PointerKind::Mouse, 400.0, 300.0);
    rig.now += 16.0;
    rig.to(1, 450.0, 300.0);
    rig.now += 300.0;
    rig.up(1, 450.0, 300.0);
    assert_eq!(rig.settle(), 0);
}

#[test]
fn clicks_and_double_clicks() {
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    rig.down(1, PointerKind::Mouse, 200.0, 200.0);
    rig.to(1, 201.0, 200.0);
    rig.up(1, 201.0, 200.0);
    assert_eq!(
        rig.input.take_events(),
        [InteractEvent::Click {
            position: DVec2::new(201.0, 200.0),
            button: Button::Primary
        }]
    );
    rig.now += 100.0;
    let ground = rig.camera.unproject(DVec2::new(201.0, 200.0)).unwrap();
    rig.down(1, PointerKind::Mouse, 201.0, 200.0);
    rig.up(1, 201.0, 200.0);
    assert!(
        rig.input
            .take_events()
            .contains(&InteractEvent::DoubleClick {
                position: DVec2::new(201.0, 200.0)
            })
    );
    rig.settle();
    assert!((rig.camera.zoom() - (zoom + 1.0)).abs() < 1e-9);
    let under = rig.camera.project(ground).unwrap();
    assert!(under.distance(DVec2::new(201.0, 200.0)) < 1e-6);
}

#[test]
fn pinching_zooms_and_twisting_rotates_around_the_fingers() {
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.down(2, PointerKind::Touch, 500.0, 300.0);
    let ground = rig.camera.unproject(DVec2::new(400.0, 300.0)).unwrap();
    // Spread to twice the distance and turn a quarter clockwise.
    for i in 1..=20 {
        let t = f64::from(i) / 20.0;
        let half = 100.0 * (1.0 + t);
        let angle = t * std::f64::consts::FRAC_PI_2;
        let offset = DVec2::new(angle.cos(), angle.sin()) * half;
        rig.now += 16.0;
        rig.to(1, 400.0 - offset.x, 300.0 - offset.y);
        rig.to(2, 400.0 + offset.x, 300.0 + offset.y);
        rig.frame();
    }
    let state = *rig.camera.state();
    // The first 0.1 levels of spread only arm the zoom, as in GL JS.
    assert!(
        (rig.camera.zoom() - (zoom + 0.93)).abs() < 0.02,
        "{}",
        rig.camera.zoom()
    );
    // Clockwise fingers turn the map clockwise: the bearing decreases, by
    // all but the 14 degrees (25 px of arc) that arm the rotation.
    assert!((state.bearing + 76.5).abs() < 1.0, "{}", state.bearing);
    let under = rig.camera.project(ground).unwrap();
    assert!(under.distance(DVec2::new(400.0, 300.0)) < 1.0, "{under}");
}

#[test]
fn two_fingers_moving_up_together_tilt() {
    let mut rig = Rig::new();
    rig.down(1, PointerKind::Touch, 300.0, 400.0);
    rig.down(2, PointerKind::Touch, 500.0, 400.0);
    for i in 1..=10 {
        rig.now += 16.0;
        let y = 400.0 - 10.0 * f64::from(i);
        rig.to(1, 300.0, y);
        rig.to(2, 500.0, y);
    }
    let state = *rig.camera.state();
    assert!(state.pitch > 40.0, "{}", state.pitch);
    assert!((rig.camera.zoom() - 12.0).abs() < 1e-9);
}

#[test]
fn double_tap_zooms_in_and_two_finger_tap_zooms_out() {
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.up(1, 300.0, 300.0);
    rig.now += 150.0;
    rig.down(1, PointerKind::Touch, 305.0, 302.0);
    rig.up(1, 305.0, 302.0);
    rig.settle();
    assert!((rig.camera.zoom() - (zoom + 1.0)).abs() < 1e-9);
    rig.now += 1000.0;
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.down(2, PointerKind::Touch, 400.0, 300.0);
    rig.now += 60.0;
    rig.up(1, 300.0, 300.0);
    rig.up(2, 400.0, 300.0);
    rig.settle();
    assert!((rig.camera.zoom() - zoom).abs() < 1e-9);
}

#[test]
fn tap_and_drag_zooms() {
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.up(1, 300.0, 300.0);
    rig.now += 150.0;
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.now += 16.0;
    rig.to(1, 300.0, 428.0);
    rig.now += 300.0;
    rig.up(1, 300.0, 428.0);
    rig.settle();
    assert!((rig.camera.zoom() - (zoom + 1.0)).abs() < 1e-9);
}

#[test]
fn wheels_zoom_smoothly_and_trackpads_at_once() {
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    let at = DVec2::new(100.0, 100.0);
    let ground = rig.camera.unproject(at).unwrap();
    // One notch of a mouse wheel, toward the user: zoom in.
    rig.send(Input::Wheel {
        position: at,
        delta: DVec2::new(0.0, -100.0 * WHEEL_STEP / 4.0 * 4.0 / 100.0 * 25.0),
        unit: WheelUnit::Pixel,
        modifiers: Modifiers::default(),
    });
    assert_eq!(rig.camera.zoom(), zoom, "eased, not immediate");
    let frames = rig.settle();
    assert!((10..=14).contains(&frames), "{frames}");
    assert!(rig.camera.zoom() > zoom + 0.1);
    let under = rig.camera.project(ground).unwrap();
    assert!(under.distance(at) < 1e-6);
    // A trackpad's small deltas apply at once and end the move later.
    let before = rig.camera.zoom();
    rig.now += 1000.0;
    rig.send(Input::Wheel {
        position: at,
        delta: DVec2::new(0.0, 2.0),
        unit: WheelUnit::Pixel,
        modifiers: Modifiers::default(),
    });
    assert!(rig.camera.zoom() < before);
    rig.camera.take_events();
    rig.frame();
    assert!(!rig.camera.take_events().iter().any(|e| matches!(
        e,
        CameraEvent::Changed {
            phase: Phase::End,
            ..
        }
    )));
    rig.settle();
    assert!(rig.camera.take_events().iter().any(|e| matches!(
        e,
        CameraEvent::Changed {
            change: Change::Move,
            phase: Phase::End,
            ..
        }
    )));
}

#[test]
fn keys_step_the_camera() {
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    rig.send(Input::Key {
        key: Key::ZoomIn,
        modifiers: Modifiers::default(),
    });
    rig.settle();
    assert!((rig.camera.zoom() - (zoom + 1.0)).abs() < 1e-9);
    rig.send(Input::Key {
        key: Key::Right,
        modifiers: Modifiers {
            shift: true,
            ..Modifiers::default()
        },
    });
    rig.settle();
    assert!((rig.camera.state().bearing - 15.0).abs() < 1e-9);
    let ground = rig.camera.unproject(DVec2::new(400.0, 300.0)).unwrap();
    rig.send(Input::Key {
        key: Key::Left,
        modifiers: Modifiers::default(),
    });
    rig.settle();
    let moved = rig.camera.project(ground).unwrap();
    assert!(moved.distance(DVec2::new(500.0, 300.0)) < 1e-6, "{moved}");
}

#[test]
fn input_interrupts_animations() {
    let mut rig = Rig::new();
    let id = rig.camera.ease_to(
        &orb_camera::CameraTarget {
            zoom: Some(4.0),
            ..Default::default()
        },
        orb_camera::EaseOptions::default(),
        rig.now,
    );
    rig.frame();
    rig.down(1, PointerKind::Mouse, 10.0, 10.0);
    assert!(
        rig.camera
            .take_events()
            .contains(&CameraEvent::AnimationEnd {
                id,
                interrupted: true
            })
    );
    assert!(!rig.camera.is_animating());
}

use crate::{Camera, CameraEvent, CameraState, Change, Phase, Viewport};
use glam::DVec2;
use orb_crs::Geodetic;
use orb_crs::TileMatrixSet;

use super::*;
use crate::MomentumSnap;

fn camera() -> Camera {
    let projection = *TileMatrixSet::web_mercator().projection();
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
        self.down_with(id, kind, x, y, Modifiers::default());
    }

    fn down_with(&mut self, id: u32, kind: PointerKind, x: f64, y: f64, modifiers: Modifiers) {
        self.send(Input::PointerDown {
            id,
            kind,
            button: Button::Primary,
            position: DVec2::new(x, y),
            modifiers,
        });
    }

    /// A mouse drag from `x` to the right at `speed` px per 16 ms move,
    /// released at once; returns where the ground under its start ends up
    /// once everything settles, relative to the release point.
    fn fling(&mut self, x: f64, speed: f64) -> f64 {
        let ground = self.camera.unproject(DVec2::new(x, 300.0)).unwrap();
        self.down(1, PointerKind::Mouse, x, 300.0);
        let mut at = x;
        for _ in 0..8 {
            self.now += 16.0;
            at += speed;
            self.to(1, at, 300.0);
        }
        self.up(1, at, 300.0);
        self.settle();
        self.camera.project(ground).unwrap().x - at
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
    // 10 px every 32 ms is 312.5 px/s; mln-ffi's pan friction (half-life
    // 140 ms, stopping at 10 px/s) glides (312.5 − 10) / (ln 2 / 0.14 s)
    // = 61.1 px over ln(31.25) / (ln 2 / 0.14 s) = 695 ms.
    assert!(
        (42..=45).contains(&frames),
        "momentum runs for a while: {frames}"
    );
    let after = rig.camera.project(ground).unwrap();
    let glide = 302.5 / (std::f64::consts::LN_2 / 0.14);
    assert!(
        (after.x - 500.0 - glide).abs() < 0.01,
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
    assert_eq!(ends, 1, "one move, momentum included");
    assert!(rig.input.take_events().is_empty(), "a drag isn't a click");
}

#[test]
fn holding_still_before_release_stops_momentum() {
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
        source: WheelSource::Unknown,
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
        source: WheelSource::Unknown,
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
        &crate::CameraTarget {
            zoom: Some(4.0),
            ..Default::default()
        },
        crate::EaseOptions::default(),
        Some(rig.now),
    );
    rig.frame();
    rig.down(1, PointerKind::Mouse, 10.0, 10.0);
    assert!(
        rig.camera
            .take_events()
            .contains(&CameraEvent::AnimationEnd {
                id,
                interrupted: true,
                velocity: Default::default(),
            })
    );
    assert!(!rig.camera.is_animating());
}

#[test]
fn flinging_a_glide_again_adds_its_speed() {
    let mut rig = Rig::new();
    let alone = rig.fling(100.0, 10.0);
    // 10 px per 16 ms is 625 px/s: (625 − 10) / (ln 2 / 0.14 s) = 124 px.
    assert!(
        (alone - 615.0 / (std::f64::consts::LN_2 / 0.14)).abs() < 0.01,
        "{alone}"
    );

    // Caught 100 ms into a glide and flung again the same way, the second
    // glide starts with what the first had left.
    let mut rig = Rig::new();
    let ground = rig.camera.unproject(DVec2::new(100.0, 300.0)).unwrap();
    rig.down(1, PointerKind::Mouse, 100.0, 300.0);
    for i in 1..=8 {
        rig.now += 16.0;
        rig.to(1, 100.0 + 10.0 * f64::from(i), 300.0);
    }
    rig.up(1, 180.0, 300.0);
    for _ in 0..6 {
        rig.frame();
    }
    let caught = rig.camera.project(ground).unwrap().x - 180.0;
    let boosted = rig.fling(300.0, 10.0);
    assert!(
        boosted > alone * 1.5,
        "{boosted} vs {alone} (caught after {caught})"
    );

    // Flung back the other way, it starts from rest.
    let mut rig = Rig::new();
    rig.fling(100.0, 10.0);
    let back = rig.fling(600.0, -10.0);
    assert!((back + alone).abs() < 0.01, "{back}");
}

#[test]
fn turning_near_north_settles_on_north() {
    let mut rig = Rig::new();
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.down(2, PointerKind::Touch, 500.0, 300.0);
    // Turn 19.5° counterclockwise: 14° arm the rotation, 5.5° turn the map.
    for i in 1..=10 {
        let angle = -(f64::from(i) / 10.0 * 19.5).to_radians();
        let offset = DVec2::new(angle.cos(), angle.sin()) * 100.0;
        rig.now += 16.0;
        rig.to(1, 400.0 - offset.x, 300.0 - offset.y);
        rig.to(2, 400.0 + offset.x, 300.0 + offset.y);
    }
    let turned = rig.camera.state().bearing;
    assert!((3.0..7.0).contains(&turned), "{turned}");
    // Held still, then released: no glide, but the bearing settles.
    rig.now += 300.0;
    rig.up(1, 0.0, 0.0);
    rig.up(2, 0.0, 0.0);
    let frames = rig.settle();
    assert!(rig.camera.state().bearing.abs() < 1e-9);
    assert!((17..=20).contains(&frames), "300 ms: {frames}");

    // Beyond 7°, it stays.
    let mut rig = Rig::new();
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.down(2, PointerKind::Touch, 500.0, 300.0);
    for i in 1..=10 {
        let angle = -(f64::from(i) * 3.0).to_radians();
        let offset = DVec2::new(angle.cos(), angle.sin()) * 100.0;
        rig.now += 16.0;
        rig.to(1, 400.0 - offset.x, 300.0 - offset.y);
        rig.to(2, 400.0 + offset.x, 300.0 + offset.y);
    }
    rig.now += 300.0;
    let turned = rig.camera.state().bearing;
    rig.up(1, 0.0, 0.0);
    rig.up(2, 0.0, 0.0);
    rig.settle();
    assert!(turned > 7.0 && rig.camera.state().bearing == turned);
}

#[test]
fn shift_drag_zooms_to_a_box() {
    let shift = Modifiers {
        shift: true,
        ..Modifiers::default()
    };
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    let middle = rig.camera.unproject(DVec2::new(300.0, 250.0)).unwrap();
    rig.down_with(1, PointerKind::Mouse, 100.0, 100.0, shift);
    rig.now += 16.0;
    rig.to(1, 500.0, 400.0);
    assert_eq!(
        rig.input.box_zoom(),
        Some([DVec2::new(100.0, 100.0), DVec2::new(500.0, 400.0)])
    );
    // The box draws; the map stays.
    assert_eq!(rig.camera.zoom(), zoom);
    rig.up(1, 500.0, 400.0);
    assert_eq!(rig.input.box_zoom(), None);
    rig.settle();
    // 400 × 300 px in 800 × 600: one level in, on the box's middle.
    assert!((rig.camera.zoom() - (zoom + 1.0)).abs() < 1e-9);
    let at = rig.camera.project(middle).unwrap();
    assert!(at.distance(DVec2::new(400.0, 300.0)) < 1e-6, "{at}");
    assert!(rig.input.take_events().is_empty(), "a box isn't a click");

    // Escape drops it.
    rig.down_with(1, PointerKind::Mouse, 100.0, 100.0, shift);
    rig.to(1, 300.0, 300.0);
    rig.send(Input::Key {
        key: Key::Escape,
        modifiers: Modifiers::default(),
    });
    assert_eq!(rig.input.box_zoom(), None);
    rig.up(1, 300.0, 300.0);
    rig.settle();
    assert!((rig.camera.zoom() - (zoom + 1.0)).abs() < 1e-9);
}

#[test]
fn shift_double_click_zooms_out() {
    let shift = Modifiers {
        shift: true,
        ..Modifiers::default()
    };
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    for _ in 0..2 {
        rig.down_with(1, PointerKind::Mouse, 200.0, 200.0, shift);
        rig.up(1, 200.0, 200.0);
        rig.now += 100.0;
    }
    rig.settle();
    assert!((rig.camera.zoom() - (zoom - 1.0)).abs() < 1e-9);
}

impl Rig {
    fn with(options: GestureOptions) -> Self {
        let mut rig = Self::new();
        rig.input.set_options(&mut rig.camera, options);
        rig
    }

    fn pinch(&mut self, scale: f64, rotation: f64, phase: GesturePhase) {
        self.send(Input::Pinch {
            center: DVec2::new(200.0, 200.0),
            scale,
            rotation,
            phase,
        });
    }

    /// A pinch spreading the fingers from 200 to 400 px apart over 20
    /// moves, released at once.
    fn spread(&mut self) {
        self.down(1, PointerKind::Touch, 300.0, 300.0);
        self.down(2, PointerKind::Touch, 500.0, 300.0);
        for i in 1..=20 {
            let half = 100.0 * (1.0 + f64::from(i) / 20.0);
            self.now += 16.0;
            self.to(1, 400.0 - half, 300.0);
            self.to(2, 400.0 + half, 300.0);
        }
        self.up(1, 200.0, 300.0);
        self.up(2, 600.0, 300.0);
    }
}

#[test]
fn a_touch_held_still_is_a_long_click() {
    let mut rig = Rig::new();
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    for _ in 0..20 {
        rig.frame();
    }
    assert!(rig.input.take_events().is_empty(), "320 ms is a press");
    rig.to(1, 302.0, 301.0);
    for _ in 0..6 {
        rig.frame();
    }
    assert_eq!(
        rig.input.take_events(),
        [InteractEvent::LongClick {
            position: DVec2::new(302.0, 301.0)
        }]
    );
    rig.frame();
    rig.up(1, 302.0, 301.0);
    rig.settle();
    assert!(rig.input.take_events().is_empty(), "and not a click too");

    // A press that moves on is a drag; a mouse held still is a press.
    let mut rig = Rig::new();
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.now += 16.0;
    rig.to(1, 320.0, 300.0);
    for _ in 0..40 {
        rig.frame();
    }
    rig.up(1, 320.0, 300.0);
    assert!(rig.input.take_events().is_empty());
    let mut rig = Rig::new();
    rig.down(1, PointerKind::Mouse, 300.0, 300.0);
    for _ in 0..40 {
        rig.frame();
    }
    rig.up(1, 300.0, 300.0);
    assert_eq!(
        rig.input.take_events(),
        [InteractEvent::Click {
            position: DVec2::new(300.0, 300.0)
        }]
    );
}

#[test]
fn a_right_click_is_a_long_click() {
    let mut rig = Rig::new();
    rig.send(Input::PointerDown {
        id: 1,
        kind: PointerKind::Mouse,
        button: Button::Secondary,
        position: DVec2::new(100.0, 120.0),
        modifiers: Modifiers::default(),
    });
    rig.up(1, 100.0, 120.0);
    assert_eq!(
        rig.input.take_events(),
        [InteractEvent::LongClick {
            position: DVec2::new(100.0, 120.0)
        }]
    );
}

#[test]
fn movements_turned_off_stay_put() {
    let mut rig = Rig::with(GestureOptions::none());
    let state = *rig.camera.state();
    rig.fling(100.0, 10.0);
    rig.spread();
    rig.settle();
    let used = rig.input.handle(
        &mut rig.camera,
        Input::Wheel {
            position: DVec2::new(100.0, 100.0),
            delta: DVec2::new(0.0, -100.0),
            unit: WheelUnit::Pixel,
            modifiers: Modifiers::default(),
            source: WheelSource::Unknown,
        },
        rig.now,
    );
    assert!(!used, "the page scrolls instead");
    rig.send(Input::Key {
        key: Key::ZoomIn,
        modifiers: Modifiers::default(),
    });
    rig.settle();
    assert_eq!(*rig.camera.state(), state);

    // Pinching with rotation off zooms and doesn't turn.
    let mut rig = Rig::with(GestureOptions {
        rotate: Movement {
            enabled: false,
            ..GestureOptions::default().rotate
        },
        ..GestureOptions::default()
    });
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.down(2, PointerKind::Touch, 500.0, 300.0);
    for i in 1..=20 {
        let t = f64::from(i) / 20.0;
        let half = 100.0 * (1.0 + t);
        let angle = t * std::f64::consts::FRAC_PI_2;
        let offset = DVec2::new(angle.cos(), angle.sin()) * half;
        rig.now += 16.0;
        rig.to(1, 400.0 - offset.x, 300.0 - offset.y);
        rig.to(2, 400.0 + offset.x, 300.0 + offset.y);
    }
    assert!(rig.camera.zoom() > 12.9);
    assert_eq!(rig.camera.state().bearing, 0.0);
}

#[test]
fn changing_the_options_cancels_a_gesture() {
    let mut rig = Rig::new();
    rig.down(1, PointerKind::Mouse, 100.0, 300.0);
    rig.now += 16.0;
    rig.to(1, 150.0, 300.0);
    assert!(rig.camera.is_held());
    let options = GestureOptions {
        thresholds: Thresholds {
            click_tolerance: 8.0,
            ..Thresholds::default()
        },
        ..GestureOptions::default()
    };
    rig.input.set_options(&mut rig.camera, options);
    assert!(!rig.camera.is_held());
    let state = *rig.camera.state();
    rig.now += 16.0;
    rig.to(1, 200.0, 300.0);
    rig.up(1, 200.0, 300.0);
    assert_eq!(rig.settle(), 0);
    assert_eq!(*rig.camera.state(), state);
    assert!(rig.input.take_events().is_empty());

    // The new tolerance holds: 6 px is still a click.
    rig.down(1, PointerKind::Touch, 100.0, 300.0);
    rig.to(1, 106.0, 300.0);
    rig.up(1, 106.0, 300.0);
    assert_eq!(rig.input.take_events().len(), 1);
    assert_eq!(*rig.camera.state(), state);
}

#[test]
fn momentum_follows_each_movements_fling() {
    // Off: the drag stops where it lets go.
    let mut rig = Rig::with(GestureOptions {
        pan: Movement {
            momentum: None,
            ..GestureOptions::default().pan
        },
        ..GestureOptions::default()
    });
    assert!(rig.fling(100.0, 10.0).abs() < 1e-6);

    // Android's: 625 px/s glides as its `OverScroller` would.
    let friction = Rig::new().fling(100.0, 10.0);
    let mut rig = Rig::with(GestureOptions {
        pan: Movement {
            momentum: Some(Fling::ANDROID),
            ..GestureOptions::default().pan
        },
        ..GestureOptions::default()
    });
    let android = rig.fling(100.0, 10.0);
    let speed = 625.0f64;
    let (i, f, p) = (0.35, 0.015, 9.80665 * 39.37 * 160.0 * 0.84);
    let rate = 0.78f64.ln() / 0.9f64.ln();
    let expected = f * p * (rate / (rate - 1.0) * (i * speed / (f * p)).ln()).exp();
    assert!((android - expected).abs() < 0.01, "{android} vs {expected}");
    // 86 px against friction's 124: Android's slow flings stop sooner,
    // and its fast ones go farther.
    assert!(android < friction, "{android} vs {friction}");
}

#[test]
fn zoom_settles_on_a_whole_level_when_asked() {
    let mut rig = Rig::with(GestureOptions {
        zoom: Movement {
            momentum: None,
            snap: Some(MomentumSnap::ZOOM),
            ..GestureOptions::default().zoom
        },
        ..GestureOptions::default()
    });
    rig.spread();
    rig.settle();
    assert!(
        (rig.camera.zoom() - 13.0).abs() < 1e-9,
        "{}",
        rig.camera.zoom()
    );

    // Without a snap it stays where the pinch left it.
    let mut rig = Rig::with(GestureOptions {
        zoom: Movement {
            momentum: None,
            ..GestureOptions::default().zoom
        },
        ..GestureOptions::default()
    });
    rig.spread();
    rig.settle();
    assert!(
        (rig.camera.zoom() - 12.93).abs() < 0.02,
        "{}",
        rig.camera.zoom()
    );
}

#[test]
fn platform_pinches_zoom_turn_and_glide() {
    let mut rig = Rig::new();
    let ground = rig.camera.unproject(DVec2::new(200.0, 200.0)).unwrap();
    rig.pinch(1.0, 0.0, GesturePhase::Begin);
    assert!(rig.input.is_active() && rig.camera.is_held());
    for _ in 0..10 {
        rig.now += 16.0;
        rig.pinch(2f64.powf(0.05), -2.0, GesturePhase::Update);
    }
    // Half a level in, and 20° counterclockwise: the bearing rises.
    assert!((rig.camera.zoom() - 12.5).abs() < 1e-9);
    assert!((rig.camera.state().bearing - 20.0).abs() < 1e-9);
    let at = rig.camera.project(ground).unwrap();
    assert!(at.distance(DVec2::new(200.0, 200.0)) < 1e-6, "{at}");
    rig.pinch(1.0, 0.0, GesturePhase::End);
    rig.settle();
    // 0.05 levels per 16 ms glides on, as touch pinches do.
    assert!(rig.camera.zoom() > 12.7, "{}", rig.camera.zoom());
    assert!(rig.camera.state().bearing > 25.0);
    assert!(!rig.input.is_active());

    // Cancelled, nothing glides; a touch takes over from one.
    let mut rig = Rig::new();
    for _ in 0..10 {
        rig.now += 16.0;
        rig.pinch(2f64.powf(0.05), 0.0, GesturePhase::Update);
    }
    rig.pinch(1.0, 0.0, GesturePhase::Cancel);
    assert_eq!(rig.settle(), 0);
    assert!((rig.camera.zoom() - 12.5).abs() < 1e-9);
    rig.pinch(2.0, 0.0, GesturePhase::Begin);
    rig.down(1, PointerKind::Touch, 300.0, 300.0);
    rig.pinch(2.0, 0.0, GesturePhase::Update);
    assert!((rig.camera.zoom() - 12.5).abs() < 1e-9);
}

/// A trackpad pinch the platform hands over as a pinch zooms once, even
/// where it also arrives as Ctrl-wheel events, as Chrome and Firefox send
/// a pinch; a Ctrl-wheel outside a pinch still zooms.
#[test]
fn a_platform_pinch_takes_no_ctrl_wheel_beside_it() {
    let mut rig = Rig::new();
    let ctrl_wheel = |rig: &mut Rig| {
        rig.send(Input::Wheel {
            position: DVec2::new(200.0, 200.0),
            delta: DVec2::new(0.0, -10.0),
            unit: WheelUnit::Pixel,
            modifiers: Modifiers {
                ctrl: true,
                ..Modifiers::default()
            },
            source: WheelSource::Unknown,
        });
    };
    rig.pinch(1.0, 0.0, GesturePhase::Begin);
    for _ in 0..10 {
        rig.now += 16.0;
        ctrl_wheel(&mut rig);
        rig.pinch(2f64.powf(0.05), 0.0, GesturePhase::Update);
    }
    assert!((rig.camera.zoom() - 12.5).abs() < 1e-9);
    rig.pinch(1.0, 0.0, GesturePhase::Cancel);
    rig.settle();
    rig.now += 1000.0;
    ctrl_wheel(&mut rig);
    rig.settle();
    assert!(rig.camera.zoom() > 12.5);
}

#[test]
fn changing_the_options_cancels_a_platform_pinch_until_it_ends() {
    let mut rig = Rig::new();
    rig.pinch(1.0, 0.0, GesturePhase::Begin);
    rig.now += 16.0;
    rig.pinch(2.0, 0.0, GesturePhase::Update);
    let options = GestureOptions {
        rotate: Movement {
            enabled: false,
            ..GestureOptions::default().rotate
        },
        ..GestureOptions::default()
    };
    rig.input.set_options(&mut rig.camera, options);
    let (state, zoom) = (*rig.camera.state(), rig.camera.zoom());
    // The rest of the pinch moves nothing and glides nowhere.
    for _ in 0..5 {
        rig.now += 16.0;
        rig.pinch(2.0, 0.0, GesturePhase::Update);
    }
    rig.pinch(1.0, 0.0, GesturePhase::End);
    assert_eq!(rig.settle(), 0);
    assert_eq!(*rig.camera.state(), state);
    assert!(!rig.input.is_active());
    // The next one zooms.
    rig.pinch(1.0, 0.0, GesturePhase::Begin);
    rig.pinch(2.0, 0.0, GesturePhase::Update);
    assert!((rig.camera.zoom() - zoom - 1.0).abs() < 1e-9);
}

#[test]
fn a_known_trackpad_zooms_at_once_from_its_first_event() {
    let mut rig = Rig::new();
    let zoom = rig.camera.zoom();
    // 10 px alone would read as a wheel's first step and ease.
    for source in [WheelSource::Unknown, WheelSource::Trackpad] {
        rig.now += 1000.0;
        let before = rig.camera.zoom();
        rig.send(Input::Wheel {
            position: DVec2::new(100.0, 100.0),
            delta: DVec2::new(0.0, -10.0),
            unit: WheelUnit::Pixel,
            modifiers: Modifiers::default(),
            source,
        });
        let at_once = rig.camera.zoom() != before;
        assert_eq!(at_once, source == WheelSource::Trackpad, "{source:?}");
        rig.settle();
    }
    assert!(rig.camera.zoom() > zoom);
}

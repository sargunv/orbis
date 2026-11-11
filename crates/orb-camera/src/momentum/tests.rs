//! After maplibre-native-ffi's transform tests for its momentum patch.

use super::*;
use crate::{Camera, CameraEvent, CameraTarget};
use orb_crs::{Geodetic, TileMatrixSet};

fn camera(target: CameraTarget) -> Camera {
    let projection = *TileMatrixSet::web_mercator().projection();
    let mut camera = Camera::new(
        CameraState::new(Geodetic::new(20.0, 10.0), 8.0, &projection),
        projection.into(),
        Viewport {
            width: 1000.0,
            height: 1000.0,
        },
    );
    camera.jump_to(&target);
    camera.frame(0.0);
    camera.take_events();
    camera
}

/// The friction decay rate per second for mln-ffi's half-life.
fn rate() -> f64 {
    std::f64::consts::LN_2 / (HALF_LIFE / 1000.0)
}

fn friction(half_life: f64, stop_speed: f64) -> MomentumDecay {
    MomentumDecay {
        model: DecayModel::Friction {
            half_life,
            stop_speed,
        },
        ..MomentumDecay::friction(1.0)
    }
}

#[test]
fn each_part_slows_by_friction() {
    let mut camera = camera(CameraTarget {
        pitch: Some(10.0),
        ..Default::default()
    });
    let momentum = GestureMomentum {
        zoom: Some(AnchoredMomentum {
            decay: friction(250.0, 0.01),
            ..AnchoredMomentum::zoom(Some(1.0))
        }),
        bearing: Some(AnchoredMomentum::bearing(Some(-100.0))),
        pitch: Some(AnchoredMomentum::pitch(Some(20.0))),
        ..Default::default()
    };
    let center = camera.state().target;
    let bearing = camera.state().bearing;
    camera.start_momentum(&momentum, Some(0.0)).unwrap();

    // A speed v that decays at rate k has moved (v / k)(1 - e^(-kt)).
    let zoom_rate = std::f64::consts::LN_2 / 0.25;
    camera.frame(500.0);
    let moved = 1.0 / zoom_rate * -(-zoom_rate * 0.5f64).exp_m1();
    assert!((camera.zoom() - (8.0 + moved)).abs() < 1e-9);

    // Each part stops at its stop speed, so it moves (v − stop) / k.
    assert!(!camera.frame(5000.0));
    assert!((camera.zoom() - (8.0 + 0.99 / zoom_rate)).abs() < 1e-9);
    let turned = wrap_bearing(camera.state().bearing - bearing);
    assert!((turned + 99.0 / rate()).abs() < 1e-9, "{turned}");
    assert!((camera.state().pitch - (10.0 + 19.0 / rate())).abs() < 1e-9);
    assert!((camera.state().target.lon - center.lon).abs() < 1e-9);
    assert!((camera.state().target.lat - center.lat).abs() < 1e-9);
    let ends: Vec<_> = camera
        .take_events()
        .into_iter()
        .filter(|e| matches!(e, CameraEvent::AnimationEnd { .. }))
        .collect();
    assert!(matches!(
        ends[..],
        [CameraEvent::AnimationEnd {
            interrupted: false,
            velocity,
            ..
        }] if velocity == MomentumVelocity::default()
    ));
}

#[test]
fn pans_move_their_distance_on_screen() {
    let mut camera = camera(CameraTarget::default());
    let ground = camera.state().target;
    let momentum = GestureMomentum {
        pan: Some(PanMomentum::new(DVec2::new(1200.0, -500.0))),
        ..Default::default()
    };
    camera.start_momentum(&momentum, Some(0.0)).unwrap();
    camera.frame(100.0);
    let moved = -(-rate() * 0.1f64).exp_m1() / rate();
    let at = camera.project(ground).unwrap();
    assert!((at - DVec2::new(500.0 + 1200.0 * moved, 500.0 - 500.0 * moved)).length() < 1e-6);

    camera.frame(5000.0);
    let distance = (1300.0 - 10.0) / rate();
    let at = camera.project(ground).unwrap();
    let expected = DVec2::new(500.0, 500.0) + DVec2::new(1200.0, -500.0) / 1300.0 * distance;
    assert!((at - expected).length() < 1e-6, "{at} {expected}");
    assert!(!camera.is_animating());
}

#[test]
fn eased_glides_follow_their_curve() {
    let mut camera = camera(CameraTarget::default());
    let momentum = GestureMomentum {
        zoom: Some(AnchoredMomentum {
            decay: MomentumDecay {
                model: DecayModel::Ease {
                    distance: 2.0,
                    duration: 1000.0,
                    easing: Easing::Linear,
                },
                ..MomentumDecay::ZOOM
            },
            ..AnchoredMomentum::zoom(Some(-1.0))
        }),
        ..Default::default()
    };
    camera.start_momentum(&momentum, Some(0.0));
    camera.frame(250.0);
    assert!((camera.zoom() - 7.5).abs() < 1e-6);
    // Cancelled, it reports the curve's slope: 2 levels a second.
    camera.stop();
    let velocity = camera.take_events().into_iter().find_map(|e| match e {
        CameraEvent::AnimationEnd { velocity, .. } => Some(velocity),
        _ => None,
    });
    assert!((velocity.unwrap().zoom + 2.0).abs() < 1e-6);

    camera.start_momentum(&momentum, Some(0.0));
    camera.frame(1000.0);
    assert!((camera.zoom() - 5.5).abs() < 1e-9);
    assert!(!camera.is_animating());
}

#[test]
fn anchors_stay_in_place() {
    let anchor = DVec2::new(800.0, 300.0);
    for part in [ZOOM, BEARING, PITCH] {
        let mut camera = camera(CameraTarget {
            pitch: Some(20.0),
            ..Default::default()
        });
        let ground = camera.unproject(anchor).unwrap();
        let mut momentum = GestureMomentum::default();
        let slot = match part {
            ZOOM => &mut momentum.zoom,
            BEARING => &mut momentum.bearing,
            _ => &mut momentum.pitch,
        };
        *slot = Some(AnchoredMomentum::pitch(Some(20.0)).with_anchor(Some(anchor)));
        camera.start_momentum(&momentum, Some(0.0));
        camera.frame(5000.0);
        let at = camera.project(ground).unwrap();
        assert!((at - anchor).length() < 1e-3, "part {part}: {at}");
    }
}

#[test]
fn distances_are_capped() {
    let mut camera = camera(CameraTarget::default());
    let momentum = GestureMomentum {
        zoom: Some(AnchoredMomentum::zoom(Some(50.0))),
        ..Default::default()
    };
    camera.start_momentum(&momentum, Some(0.0));
    camera.frame(5000.0);
    assert!((camera.zoom() - 10.5).abs() < 1e-9);
}

#[test]
fn a_hold_reports_the_speed_left() {
    let mut camera = camera(CameraTarget::default());
    let momentum = GestureMomentum {
        pan: Some(PanMomentum {
            velocity: DVec2::new(1000.0, 0.0),
            decay: friction(1000.0, 1.0),
        }),
        zoom: Some(AnchoredMomentum {
            decay: friction(1000.0, 0.01),
            ..AnchoredMomentum::zoom(Some(1.0))
        }),
        ..Default::default()
    };
    camera.start_momentum(&momentum, Some(0.0));
    camera.frame(1000.0);
    // After one half-life, each part moves at half its release speed.
    let velocity = camera.hold().unwrap();
    assert!((velocity.pan - DVec2::new(500.0, 0.0)).length() < 1e-9);
    assert!((velocity.zoom - 0.5).abs() < 1e-9);
    assert!(!camera.is_animating());
}

#[test]
fn snaps_settle_after_the_glide() {
    let mut camera = camera(CameraTarget {
        bearing: Some(84.0),
        ..Default::default()
    });
    // Moves (v − 1) / k = 4°, then settles on the nearest of every 90°.
    let momentum = GestureMomentum {
        bearing: Some(
            AnchoredMomentum {
                decay: MomentumDecay::friction(1.0),
                ..AnchoredMomentum::bearing(Some(1.0 + 4.0 * rate()))
            }
            .with_snap(Some(MomentumSnap {
                spacing: 90.0,
                duration: 200.0,
                ..MomentumSnap::BEARING
            })),
        ),
        ..Default::default()
    };
    camera.start_momentum(&momentum, Some(0.0));
    assert!(camera.frame(2000.0));
    assert!((camera.state().bearing - 88.0).abs() < 1e-9);
    assert!(!camera.frame(2200.0));
    assert!((camera.state().bearing - 90.0).abs() < 1e-9);

    // Without motion, instant snaps settle at once: the bearing around the
    // circle, and the zoom on the nearest whole level. A value outside the
    // tolerance stays.
    let instant = |snap: MomentumSnap| {
        Some(MomentumSnap {
            duration: 0.0,
            ..snap
        })
    };
    let snaps = GestureMomentum {
        zoom: Some(AnchoredMomentum::zoom(None).with_snap(instant(MomentumSnap::ZOOM))),
        bearing: Some(AnchoredMomentum::bearing(None).with_snap(instant(MomentumSnap::BEARING))),
        pitch: Some(AnchoredMomentum::pitch(None).with_snap(instant(MomentumSnap::PITCH))),
        ..Default::default()
    };
    camera.jump_to(&CameraTarget {
        bearing: Some(355.0),
        zoom: Some(7.3),
        pitch: Some(30.0),
        ..Default::default()
    });
    assert_eq!(camera.start_momentum(&snaps, Some(0.0)), None);
    assert!(camera.state().bearing.abs() < 1e-9);
    assert!((camera.zoom() - 7.0).abs() < 1e-9);
    assert!((camera.state().pitch - 30.0).abs() < 1e-9);

    // A spacing that doesn't divide the circle still finds the nearest
    // target around it: 270 is 1° from −89.
    camera.jump_to(&CameraTarget {
        bearing: Some(-89.0),
        ..Default::default()
    });
    let uneven = GestureMomentum {
        bearing: Some(
            AnchoredMomentum::bearing(None).with_snap(Some(MomentumSnap {
                spacing: 270.0,
                duration: 0.0,
                ..MomentumSnap::BEARING
            })),
        ),
        ..Default::default()
    };
    camera.start_momentum(&uneven, Some(0.0));
    assert!((camera.state().bearing + 90.0).abs() < 1e-9);
}

#[test]
fn limits_stop_a_part() {
    let mut camera = camera(CameraTarget::default());
    camera.set_constraints(crate::Constraints {
        max_zoom: 8.0,
        ..Default::default()
    });
    let momentum = GestureMomentum {
        zoom: Some(AnchoredMomentum::zoom(Some(1.0))),
        pitch: Some(AnchoredMomentum::pitch(Some(-20.0))),
        ..Default::default()
    };
    camera.start_momentum(&momentum, Some(0.0));
    assert!(!camera.frame(20.0), "both parts are held at once");
    assert!((camera.zoom() - 8.0).abs() < 1e-9);
}

#[test]
fn slow_releases_and_bad_decays_do_not_move() {
    assert_eq!(Plan::new(150.0, &MomentumDecay::PAN), None);
    assert_eq!(Plan::new(5.0, &MomentumDecay::PAN), None);
    assert_eq!(Plan::new(f64::NAN, &MomentumDecay::PAN), None);
    assert_eq!(Plan::new(100.0, &friction(0.0, 1.0)), None);
    assert_eq!(Plan::new(100.0, &friction(100.0, 0.0)), None);
    // A release faster than the cap glides as the cap does.
    let capped = Plan::new(50_000.0, &MomentumDecay::PAN).unwrap();
    let max = Plan::new(10_000.0, &MomentumDecay::PAN).unwrap();
    assert_eq!(capped, max);
    assert!((max.distance - 9990.0 / rate()).abs() < 1e-9);
}

#[test]
fn friction_speed_is_the_slope_of_its_progress() {
    let plan = Plan::new(2000.0, &MomentumDecay::PAN).unwrap();
    for t in [0.0, 0.1, 0.3, 0.6] {
        let h = 1e-6;
        let slope = (plan.progress(t + h) - plan.progress(t)) / h;
        assert!((slope - plan.speed(t)).abs() < 1e-3, "{t}");
    }
    assert!((plan.speed(0.0) * plan.distance - 2000.0).abs() < 1e-6);
}

//! The cull drops only what draws nothing: with it keeping every
//! candidate (`cull.wgsl`'s `keep_all`, as a driver without the cull
//! draws), the Tokyo extract's views draw the same pixels, as does a style
//! at the edge of a margin.

mod common;

use common::{Geodetic, Offscreen, Options, differing_pixels};

/// Frames until the map is idle twice in a row.
fn settle(offscreen: &mut Offscreen) {
    let started = std::time::Instant::now();
    let mut was_idle = false;
    loop {
        let idle = offscreen.frame().idle;
        if idle && was_idle {
            return;
        }
        was_idle = idle;
        assert!(started.elapsed().as_secs() < 60, "timed out settling");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

fn culled_and_kept(at: (f64, f64, f64), edit: Option<&str>) {
    let (zoom, bearing, pitch) = at;
    culled_and_kept_in(
        &Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((Geodetic::new(139.74, 35.685), zoom, bearing, pitch)),
        },
        edit,
    );
}

fn culled_and_kept_in(options: &Options, edit: Option<&str>) {
    let (_, zoom, bearing, pitch) = options.at.unwrap();
    let (mut offscreen, _, _) = Offscreen::load(options).unwrap();
    if let Some(edit) = edit {
        let tx = orb_runtime::Transaction::from_commands(edit).unwrap();
        offscreen.map.edit(&tx).unwrap();
        settle(&mut offscreen);
    }
    let culled = offscreen.pixels();
    offscreen.map.set_keep_all_cull(true);
    settle(&mut offscreen);
    let kept = offscreen.pixels();
    assert_eq!(
        differing_pixels(&culled, &kept),
        0,
        "z{zoom} bearing {bearing} pitch {pitch}: keeping every candidate changed pixels"
    );
}

#[test]
fn keeping_every_candidate_draws_the_same_pixels() {
    culled_and_kept((12.0, 0.0, 0.0), None);
    culled_and_kept((15.0, 30.0, 60.0), None);
}

/// Extrusions, whose cull reaches up by their height.
#[test]
fn keeping_every_extrusion_draws_the_same_pixels() {
    let buildings = r#"[["addLayer", {"id": "building-3d", "type": "fill-extrusion", "source": "openmaptiles", "source-layer": "building", "minzoom": 14, "paint": {"fill-extrusion-color": "hsl(35,8%,85%)", "fill-extrusion-height": ["get", "render_height"], "fill-extrusion-base": ["get", "render_min_height"]}}]]"#;
    culled_and_kept((16.0, -20.0, 60.0), Some(buildings));
}

/// A line's miter reaches past its half width, as far as the extrusion's
/// bytes hold: here into the next tile from a vertex in its buffer more
/// than the half width away, where only that tile draws it.
#[test]
fn keeping_every_line_draws_the_same_miter_across_a_tile_seam() {
    culled_and_kept_in(
        &Options {
            style: Some(format!(
                "file://{}/tests/data/miter-seam.json",
                env!("CARGO_MANIFEST_DIR")
            )),
            encoding: "mvt".into(),
            at: Some((Geodetic::new(0.0, 0.0), 2.0, 0.0, 0.0)),
        },
        None,
    );
}

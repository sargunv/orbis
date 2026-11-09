//! `mlnframes <style> --views views.txt [--filter s] [--frames N]
//! [--warmup N]`: mln-rs's still bench (its PR #27) over a views file's
//! cameras for the style (the style column is the URL's last path
//! segment): per view, a `MapMode::Static` map loads, then each frame
//! renders and reads back with the zoom alternating by 0.001. Prints
//! `frame <name> <median ms> <p90 ms> <frames>`, as `orb-bench --nudge`'s
//! frame wall.

use std::rc::Rc;
use std::time::Instant;

use mln_rs::mln_actor::RunLoop;
use mln_rs::mln_map::{CameraOptions, Map, MapMode, MapOptions, NullMapObserver};
use mln_rs::mln_platform_default::HeadlessFrontend;
use mln_rs::mln_util::geo::LatLng;
use mln_rs::mln_util::size::Size;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let style = args.first().expect("a style URL");
    let name = style.rsplit('/').next().unwrap_or(style);
    let name = name.strip_suffix(".json").unwrap_or(name);
    let views = mln_h2h::flag(&args, "--views", "");
    let filter = mln_h2h::flag(&args, "--filter", "");
    let frames: usize = mln_h2h::flag(&args, "--frames", "100")
        .parse()
        .expect("--frames N");
    let warmup: usize = mln_h2h::flag(&args, "--warmup", "10")
        .parse()
        .expect("--warmup N");
    let _run_loop = RunLoop::new();
    for line in std::fs::read_to_string(&views)
        .expect("--views file")
        .lines()
    {
        let line = line.split('#').next().unwrap_or_default().trim();
        let f: Vec<&str> = line.split_whitespace().collect();
        let [view_style, view, ref numbers @ ..] = f[..] else {
            continue;
        };
        if view_style != name || !view.contains(filter.as_str()) {
            continue;
        }
        let v: Vec<f64> = numbers
            .iter()
            .map(|s| s.parse().expect("a number"))
            .collect();
        let [lon, lat, zoom, bearing, pitch, ratio, w, h] = v[..] else {
            panic!("{views}: want style name lon lat zoom bearing pitch ratio width height");
        };
        let size = Size::new(w as u32, h as u32);
        let mut frontend = HeadlessFrontend::new(size, ratio as f32);
        let mut map = Map::new(
            &frontend,
            Rc::new(NullMapObserver),
            MapOptions::new()
                .with_map_mode(MapMode::Static)
                .with_size(size)
                .with_pixel_ratio(ratio as f32),
        );
        map.get_style_mut().load_url(style);
        let camera = |nudge: f64| {
            CameraOptions::default()
                .with_center(Some(LatLng::new(lat, lon).expect("a latitude")))
                .with_zoom(Some(zoom + nudge))
                .with_bearing(Some(bearing))
                .with_pitch(Some(pitch))
        };
        map.jump_to(&camera(0.0));
        mln_h2h::render(&mut frontend, &mut map);
        let mut times = Vec::with_capacity(frames);
        for i in 0..warmup + frames {
            map.jump_to(&camera(if i % 2 == 1 { 0.001 } else { 0.0 }));
            let started = Instant::now();
            mln_h2h::render(&mut frontend, &mut map);
            if i >= warmup {
                times.push(mln_h2h::ms(started.elapsed()));
            }
        }
        times.sort_by(f64::total_cmp);
        let at = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
        println!("frame {view} {:.3} {:.3} {frames}", at(0.5), at(0.9));
    }
}

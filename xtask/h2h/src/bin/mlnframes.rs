//! `mlnframes <style> --views views.txt [--filter s] [--frames N]
//! [--warmup N] [--backend b]`: mln-rs's still bench (its PR #27) over a
//! views file's cameras for the style (the style column is the URL's last
//! path segment): per view, a `MapMode::Static` map loads, then each frame
//! renders and reads back with the zoom alternating by 0.001. Prints
//! `frame <name> <median ms> <p90 ms> <frames>`, as `orb-bench --nudge`'s
//! frame wall.

use std::time::Instant;

use mln_rs::mln_map::MapMode;
use mln_rs::mln_util::size::Size;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device = mln_h2h::device(&args);
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
        let map = mln_h2h::Still::new(device.clone(), MapMode::Static, size, ratio as f32, style);
        let camera = |nudge: f64| mln_h2h::camera(&[lon, lat, zoom + nudge, bearing, pitch]);
        map.jump(camera(0.0));
        map.render();
        let mut times = Vec::with_capacity(frames);
        for i in 0..warmup + frames {
            map.jump(camera(if i % 2 == 1 { 0.001 } else { 0.0 }));
            let started = Instant::now();
            map.render();
            if i >= warmup {
                times.push(mln_h2h::ms(started.elapsed()));
            }
        }
        times.sort_by(f64::total_cmp);
        let at = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
        println!("frame {view} {:.3} {:.3} {frames}", at(0.5), at(0.9));
    }
}

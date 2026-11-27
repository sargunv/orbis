//! `mlnstill <style> -o out.png --at lon,lat,z[,bearing,pitch] [--size WxH]
//! [--ratio r] [--backend b]`: one still through mln-rs (`MapMode::Static`, an in-memory
//! cache), as `orbis render` draws one. Prints the map's time to its
//! first complete frame as `orbis render` does.

use std::time::Instant;

use mln_rs::mln_map::MapMode;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device = mln_h2h::device(&args);
    let style = args.first().expect("a style URL");
    let out = mln_h2h::flag(&args, "-o", "out.png");
    let at: Vec<f64> = mln_h2h::flag(&args, "--at", "0,0,0")
        .split(',')
        .map(|s| s.parse().expect("--at lon,lat,z[,bearing,pitch]"))
        .collect();
    let size = mln_h2h::size(&args, "512x512");
    let ratio: f32 = mln_h2h::flag(&args, "--ratio", "1")
        .parse()
        .expect("--ratio r");
    let started = Instant::now();
    let map = mln_h2h::Still::new(device, MapMode::Static, size, ratio, style);
    map.jump(mln_h2h::camera(&at));
    let image = map.render();
    let took = started.elapsed();
    std::fs::write(&out, mln_rs::mln_image::encode_png(&image)).expect("writing the PNG");
    eprintln!("mlnstill: {out} in {:.0} ms", mln_h2h::ms(took));
}

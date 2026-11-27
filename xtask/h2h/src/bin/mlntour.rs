//! `mlntour <style> --tour file [--size WxH] [--ratio r] [--backend b]`:
//! mln-rs's stills at each camera of the file's lines (`lon,lat,zoom
//! [,bearing,pitch]`) in turn on one map, as `orbis render --tour` draws
//! them: what a map holds over many views. mln-rs's runtime reports no
//! renderer memory for still maps, so measure the process around it
//! (`/usr/bin/time -l`).

use std::time::Instant;

use mln_rs::mln_map::MapMode;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device = mln_h2h::device(&args);
    let style = args.first().expect("a style URL");
    let tour = std::fs::read_to_string(mln_h2h::flag(&args, "--tour", "")).expect("--tour file");
    let size = mln_h2h::size(&args, "512x512");
    let ratio: f32 = mln_h2h::flag(&args, "--ratio", "1")
        .parse()
        .expect("--ratio r");
    let map = mln_h2h::Still::new(device, MapMode::Static, size, ratio, style);
    for line in tour.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let at: Vec<f64> = (line.split(','))
            .map(|s| s.trim().parse().expect("lon,lat,zoom[,bearing,pitch]"))
            .collect();
        let started = Instant::now();
        map.jump(mln_h2h::camera(&at));
        map.render();
        eprintln!(
            "mlntour: {line} in {:.0} ms",
            mln_h2h::ms(started.elapsed())
        );
    }
}

//! `mlntour <style> --tour file [--size WxH] [--ratio r] [--backend b]`:
//! mln-rs's stills at each camera of the file's lines (`lon,lat,zoom
//! [,bearing,pitch]`) in turn on one map, as `orbis render --tour` draws
//! them: what a map holds over many views. Prints the renderer's own
//! memory counts after each.

use mln_rs::mln_actor::RunLoop;
use mln_rs::mln_headless::HeadlessFrontend;
use mln_rs::mln_map::{CameraOptions, Map, MapMode, MapOptions};
use mln_rs::mln_util::geo::LatLng;
use mln_rs::mln_util::size::Size;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    mln_h2h::backend(&args);
    let style = args.first().expect("a style URL");
    let tour = std::fs::read_to_string(mln_h2h::flag(&args, "--tour", "")).expect("--tour file");
    let wh: Vec<u32> = mln_h2h::flag(&args, "--size", "512x512")
        .split('x')
        .map(|s| s.parse().expect("--size WxH"))
        .collect();
    let ratio: f32 = mln_h2h::flag(&args, "--ratio", "1")
        .parse()
        .expect("--ratio r");
    let size = Size::new(wh[0], wh[1]);
    let _run_loop = RunLoop::new();
    let mut frontend = HeadlessFrontend::new(size, ratio);
    let mut map = Map::new(
        MapOptions::new()
            .with_map_mode(MapMode::Static)
            .with_size(size)
            .with_pixel_ratio(ratio),
    )
    .expect("a map");
    map.style_mut().load_url(style);
    let mib = |b: i64| b as f64 / f64::from(1 << 20);
    for line in tour.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let at: Vec<f64> = (line.split(','))
            .map(|s| s.trim().parse().expect("lon,lat,zoom[,bearing,pitch]"))
            .collect();
        map.jump_to(
            &CameraOptions::default()
                .with_center(Some(LatLng::new(at[1], at[0]).expect("a latitude")))
                .with_zoom(Some(at[2]))
                .with_bearing(at.get(3).copied())
                .with_pitch(at.get(4).copied()),
        );
        mln_h2h::render(&mut frontend, &mut map);
        if let Some(s) = map.get_last_frame_stats() {
            eprintln!(
                "mlntour: {line}: textures {:.1} MiB, vertices {:.1} MiB, indices {:.1} MiB",
                mib(s.mem_textures as i64),
                mib(s.mem_vertex_buffers as i64),
                mib(s.mem_index_buffers as i64),
            );
        }
    }
}

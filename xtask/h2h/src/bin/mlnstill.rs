//! `mlnstill <style> -o out.png --at lon,lat,z[,bearing,pitch] [--size WxH]
//! [--ratio r]`: one still through mln-rs (`MapMode::Static`, an in-memory
//! cache), as `orbis render` draws one. Prints the map's time to its
//! first complete frame as `orbis render` does.

use std::rc::Rc;
use std::time::Instant;

use mln_rs::mln_actor::RunLoop;
use mln_rs::mln_map::{CameraOptions, Map, MapMode, MapOptions, NullMapObserver};
use mln_rs::mln_platform_default::HeadlessFrontend;
use mln_rs::mln_storage::{ClientOptions, ResourceOptions};
use mln_rs::mln_util::geo::LatLng;
use mln_rs::mln_util::size::Size;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let style = args.first().expect("a style URL");
    let out = mln_h2h::flag(&args, "-o", "out.png");
    let at: Vec<f64> = mln_h2h::flag(&args, "--at", "0,0,0")
        .split(',')
        .map(|s| s.parse().expect("--at lon,lat,z[,bearing,pitch]"))
        .collect();
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
    let started = Instant::now();
    let mut map = Map::with_resource_options(
        &frontend,
        Rc::new(NullMapObserver),
        MapOptions::new()
            .with_map_mode(MapMode::Static)
            .with_size(size)
            .with_pixel_ratio(ratio),
        &ResourceOptions::default().with_cache_path(":memory:"),
        &ClientOptions::default(),
    );
    map.get_style_mut().load_url(style);
    map.jump_to(
        &CameraOptions::default()
            .with_center(Some(LatLng::new(at[1], at[0]).expect("a latitude")))
            .with_zoom(Some(at[2]))
            .with_bearing(at.get(3).copied())
            .with_pitch(at.get(4).copied()),
    );
    let image = mln_h2h::render(&mut frontend, &mut map);
    let took = started.elapsed();
    std::fs::write(&out, mln_rs::mln_platform_default_image::encode_png(&image))
        .expect("writing the PNG");
    eprintln!("mlnstill: {out} in {:.0} ms", mln_h2h::ms(took));
}

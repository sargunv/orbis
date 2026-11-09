//! What the harness's binaries share: a still that waits rather than
//! spins, process CPU time, and the camera of a Web Mercator tile.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use mln_rs::mln_actor::RunLoop;
use mln_rs::mln_map::{CameraOptions, Map};
use mln_rs::mln_platform_default::HeadlessFrontend;
use mln_rs::mln_util::geo::LatLng;
use mln_rs::mln_util::image::PremultipliedImage;

/// One still, as `HeadlessFrontend::render` makes it, but waiting on the
/// run loop (`RunLoop::run`, which sleeps until a task or timer is due)
/// instead of spinning on the port's non-blocking `run_once`: C++'s
/// `runOnce` blocks in libuv, and the spin would count as CPU time.
pub fn render(frontend: &mut HeadlessFrontend, map: &mut Map) -> PremultipliedImage {
    let done: Rc<RefCell<Option<Option<String>>>> = Rc::new(RefCell::new(None));
    let d = done.clone();
    map.render_still(Box::new(move |error| {
        *d.borrow_mut() = Some(error.map(|e| e.to_string()));
        RunLoop::get().stop();
    }));
    RunLoop::get().run();
    if let Some(Some(error)) = done.borrow_mut().take() {
        panic!("render: {error}");
    }
    frontend.read_still_image()
}

/// The process's CPU time so far, every thread's.
pub fn cpu() -> Duration {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
    Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
}

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// The camera showing exactly Web Mercator tile `z/x/y` in a `px`-pixel
/// square (512-pixel tiles at their own zoom).
pub fn tile_camera(z: u8, x: u32, y: u32, px: u32) -> CameraOptions {
    let n = f64::from(1u32 << z);
    let lon = (f64::from(x) + 0.5) / n * 360.0 - 180.0;
    let lat = (std::f64::consts::PI * (1.0 - 2.0 * (f64::from(y) + 0.5) / n))
        .sinh()
        .atan()
        .to_degrees();
    let zoom = f64::from(z) + (f64::from(px) / 512.0).log2();
    CameraOptions::default()
        .with_center(Some(LatLng::new(lat, lon).expect("a tile's centre")))
        .with_zoom(Some(zoom))
}

/// A flag's value, or `default`.
pub fn flag(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map_or(default.to_string(), Clone::clone)
}

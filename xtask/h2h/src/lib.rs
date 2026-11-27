//! What the harness's binaries share: a still map on mln-rs's runtime,
//! process CPU time, and the camera of a Web Mercator tile.

use std::time::Duration;

use mln_rs::mln_map::{CameraOptions, CameraUpdate, MapMode};
use mln_rs::mln_util::geo::LatLng;
use mln_rs::mln_util::image::PremultipliedImage;
use mln_rs::mln_util::size::Size;
use mln_rs::{
    AttachOptions, BackendType, Device, DeviceOptions, Map, MapOptions, RenderSession,
    RenderTarget, Runtime, RuntimeOptions,
};

/// `--backend webgpu|metal|vulkan|gl`: the device mln-rs renders with
/// (WebGPU, through wgpu, by default).
pub fn device(args: &[String]) -> Device {
    let backend = match flag(args, "--backend", "webgpu").as_str() {
        "webgpu" => BackendType::WebGPU,
        "metal" => BackendType::Metal,
        "vulkan" => BackendType::Vulkan,
        "gl" => BackendType::OpenGL,
        other => panic!("--backend {other}: want webgpu, metal, vulkan or gl"),
    };
    Device::with_backend(backend, DeviceOptions::default()).expect("a device on the backend")
}

/// A static or tile map of `style` on a runtime of its own (its cache in
/// memory), drawn by an offscreen render session on `device`, as mln-rs's
/// public API renders without a window. Its map options are mln-rs's
/// defaults, which draw CJK from local fonts, as Orbis's do.
pub struct Still {
    // Dropped in this order: the session, then the map, then the runtime,
    // whose drop waits for their threads.
    session: RenderSession,
    map: Map,
    _runtime: Runtime,
}

impl Still {
    pub fn new(device: Device, mode: MapMode, size: Size, ratio: f32, style: &str) -> Still {
        let runtime = Runtime::new(&RuntimeOptions::default()).expect("a runtime");
        let mut options = MapOptions::default();
        options.map.map_mode = mode;
        options.map.size = size;
        options.map.pixel_ratio = ratio;
        let (map, _) = runtime.create_map(options).expect("a map");
        // Commands run in order, so the still after them reports a style
        // or camera that failed.
        drop(map.set_style_url(style).expect("the style's URL"));
        let target = RenderTarget::Offscreen {
            size,
            device: Some(device),
        };
        let (session, _) = map
            .attach(target, AttachOptions::default())
            .expect("an offscreen session");
        Still {
            session,
            map,
            _runtime: runtime,
        }
    }

    /// Moves the camera at once; the next still shows it.
    pub fn jump(&self, camera: CameraOptions) {
        let update = CameraUpdate {
            camera,
            ..CameraUpdate::default()
        };
        drop(self.map.update_camera(update).expect("a camera"));
    }

    /// One still, once everything it needs has loaded, read back from the
    /// GPU.
    pub fn render(&self) -> PremultipliedImage {
        (self
            .map
            .request_still_image()
            .expect("a still image")
            .wait())
        .expect("a still image");
        (self.session.read_texture().expect("a readback").wait()).expect("the still's pixels")
    }
}

/// The camera at `lon,lat,zoom[,bearing,pitch]`.
pub fn camera(at: &[f64]) -> CameraOptions {
    CameraOptions {
        center: Some(LatLng::new(at[1], at[0]).expect("a latitude")),
        zoom: Some(at[2]),
        bearing: at.get(3).copied(),
        pitch: at.get(4).copied(),
        ..CameraOptions::default()
    }
}

/// The process's CPU time so far, every thread's.
#[cfg(unix)]
pub fn cpu() -> Duration {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
    Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
}

/// The process's CPU time so far, every thread's: its kernel and user
/// time, in the 100 ns ticks Windows counts.
#[cfg(windows)]
pub fn cpu() -> Duration {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let mut times = [FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    }; 4];
    let [creation, exit, kernel, user] = &mut times;
    // SAFETY: the current process's pseudo-handle, which needs no closing,
    // and four `FILETIME`s it writes.
    let ok = unsafe { GetProcessTimes(GetCurrentProcess(), creation, exit, kernel, user) };
    assert!(
        ok != 0,
        "GetProcessTimes: {}",
        std::io::Error::last_os_error()
    );
    let ticks = |t: &FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
    Duration::from_nanos((ticks(&times[2]) + ticks(&times[3])) * 100)
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
    camera(&[lon, lat, f64::from(z) + (f64::from(px) / 512.0).log2()])
}

/// A flag's value, or `default`.
pub fn flag(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map_or(default.to_string(), Clone::clone)
}

/// `--size WxH`, or `default`.
pub fn size(args: &[String], default: &str) -> Size {
    let wh: Vec<u32> = flag(args, "--size", default)
        .split('x')
        .map(|s| s.parse().expect("--size WxH"))
        .collect();
    Size::new(wh[0], wh[1])
}

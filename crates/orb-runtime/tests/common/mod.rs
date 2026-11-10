//! What the frame tests share: the Tokyo extract on native services, a
//! map drawing offscreen at 1024×768 at 60 Hz frames, and comparisons.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Instant;

pub(crate) use orb_crs::Geodetic;
use orb_hal::Gpu;
pub(crate) use orb_hal_wgpu::WgpuGpu;
use orb_host::jobs::ThreadPool;
pub(crate) use orb_host_native::{NativeFetch, NativeImageDecode};
pub(crate) use orb_runtime::{
    CameraState, Map, MapEvent, PoolJobs, SceneCapacity, Services, TargetFormat, Viewport,
};

pub(crate) const DEMO_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data");

/// How a test's map starts.
pub(crate) struct Options {
    /// A style URL; the committed Tokyo extract by default.
    pub style: Option<String>,
    /// The extract's tile encoding: `mvt` or `mlt`.
    pub encoding: String,
    /// Target, zoom, bearing and pitch.
    pub at: Option<(Geodetic, f64, f64, f64)>,
}

/// HTTP and file fetches, and tile jobs on a thread pool.
pub(crate) fn native_services(wake: Arc<dyn Fn() + Send + Sync>) -> Services {
    let fetch = NativeFetch::with_cache(None);
    let executor = Arc::new(ThreadPool::new("jobs", ThreadPool::default_size()));
    let jobs = PoolJobs::new(fetch.clone(), Arc::new(NativeImageDecode), executor);
    jobs.set_local_glyphs(orb_host_native::SystemFonts::shared());
    jobs.set_font_files(Arc::new(orb_host_native::FontFileFaces::new()));
    Services {
        clock: Arc::new(orb_host_native::NativeClock::default()),
        image_decode: Arc::new(orb_host_native::NativeImageDecode),
        images: Default::default(),
        jobs,
        fetch,
        wake,
    }
}

/// The committed extract and reduced Liberty style, or an explicit style URL.
pub(crate) fn load_style(map: &mut Map, options: &Options) -> Result<(), String> {
    if let Some(url) = &options.style {
        map.load_style_url(url);
        return Ok(());
    }
    let path = std::path::Path::new(DEMO_DATA)
        .join(format!("tokyo-{}.json", options.encoding))
        .canonicalize()
        .map_err(|e| e.to_string())?;
    map.load_style_url(&format!("file://{}", path.display()));
    Ok(())
}

/// The default surface's projection.
pub(crate) fn web_mercator() -> orb_crs::Projection {
    *orb_crs::TileMatrixSet::by_id("WebMercatorQuad")
        .expect("built-in tile matrix set")
        .projection()
}

/// The camera of `at`, or `None` to use the style's. Its zoom is on
/// the default Web Mercator surface: a style's own plane is known only
/// once it loads.
pub(crate) fn camera(options: &Options) -> Option<CameraState> {
    options
        .at
        .map(|(target, zoom, bearing, pitch)| CameraState {
            bearing,
            pitch,
            ..CameraState::new(target, zoom, &web_mercator())
        })
}

/// A map rendering offscreen at 1024×768, for snapshots and tests.
pub(crate) struct Offscreen {
    pub gpu: WgpuGpu,
    pub target: orb_hal::TextureId,
    pub format: TargetFormat,
    /// The map's renderer, made for its first frame.
    pub renderer: Option<orb_runtime::Renderer>,
    pub viewport: Viewport,
    pub map: Map,
    /// The next frame's time: frames run at 60 Hz.
    pub time: f64,
}

impl Offscreen {
    pub(crate) const WIDTH: u32 = 1024;
    pub(crate) const HEIGHT: u32 = 768;

    /// Loads the style and renders until everything in view has arrived.
    /// Returns the map, the last frame's report and the frame count.
    pub(crate) fn load(options: &Options) -> Result<(Self, orb_runtime::FrameReport, u32), String> {
        Self::load_with(options, SceneCapacity::default())
    }

    /// Like [`load`](Self::load), with a given GPU scene capacity.
    pub(crate) fn load_with(
        options: &Options,
        capacity: SceneCapacity,
    ) -> Result<(Self, orb_runtime::FrameReport, u32), String> {
        let mut this = Self::new(options, capacity)?;
        let started = Instant::now();
        let mut frames = 0;
        // Two idle frames in a row: the second draws what the first's
        // messages installed.
        let mut was_idle = false;
        loop {
            let report = this.frame();
            frames += 1;
            for event in this.map.take_events() {
                if let MapEvent::Warning { message } | MapEvent::Error { message } = event {
                    eprintln!("orb-demo: {message}");
                }
            }
            if report.idle && was_idle {
                return Ok((this, report, frames));
            }
            was_idle = report.idle;
            if started.elapsed().as_secs() > 60 {
                return Err("timed out waiting for tiles".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// The map with its style set, before anything has loaded.
    pub(crate) fn new(options: &Options, capacity: SceneCapacity) -> Result<Self, String> {
        // Tests count frames: with no executor, pipelines compile when
        // asked for.
        let mut gpu = pollster::block_on(WgpuGpu::offscreen())?;
        let format = TargetFormat {
            color: orb_hal::TextureFormat::Rgba8Unorm,
            sample_count: 1,
        };
        let target = gpu.create_texture(&orb_hal::TextureDesc {
            label: "snapshot",
            width: Self::WIDTH,
            height: Self::HEIGHT,
            array_layers: 1,
            dimension: orb_hal::TextureViewDimension::D2,
            format: format.color,
            usage: orb_hal::TextureUsages::RENDER_ATTACHMENT | orb_hal::TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        let services = native_services(Arc::new(|| {}));
        let mut map = Map::new(services, camera(options));
        // Snapshots use the same white backing as the native demo window.
        map.set_clear_color([1.0; 4]);
        map.set_scene_capacity(capacity);
        // Tests compare maps reached by different paths: labels placed by
        // the camera alone, as AGENTS.md's path rule has the path tests.
        map.set_stable_labels(false);
        load_style(&mut map, options)?;
        Ok(Self {
            gpu,
            target,
            format,
            renderer: None,
            viewport: Viewport {
                width: f64::from(Self::WIDTH),
                height: f64::from(Self::HEIGHT),
            },
            map,
            time: 0.0,
        })
    }

    pub(crate) fn frame(&mut self) -> orb_runtime::FrameReport {
        self.frame_at(self.viewport)
    }

    /// The next frame of the camera as it is, in `viewport`: frames run at
    /// 60 Hz.
    pub(crate) fn frame_at(&mut self, viewport: Viewport) -> orb_runtime::FrameReport {
        self.time += 1000.0 / 60.0;
        let time = self.time;
        let map = &mut self.map;
        let (gpu, format) = (&mut self.gpu, self.format);
        let renderer = self
            .renderer
            .get_or_insert_with(|| map.create_renderer(gpu, format));
        map.render(gpu, renderer, self.target, viewport, 1.0, time)
    }

    pub(crate) fn pixels(&self) -> Vec<u8> {
        pollster::block_on(self.gpu.read_texture(self.target))
    }
}

/// Draws the map at `options` until idle: its pixels.
pub(crate) fn snapshot(options: &Options) -> Vec<u8> {
    Offscreen::load(options).unwrap().0.pixels()
}

/// A loop of camera flights: legs of `flyTo` options, as the demos fly
/// them.
pub(crate) struct Flight {
    pub legs: Vec<(orb_camera::CameraTarget, orb_camera::FlyOptions)>,
}

impl Flight {
    pub(crate) fn read(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        let legs = json["legs"]
            .as_array()
            .ok_or_else(|| format!("{path}: no legs"))?
            .iter()
            .map(|leg| {
                let number = |key: &str| leg[key].as_f64();
                let center = leg["center"]
                    .as_array()
                    .and_then(|c| Some(Geodetic::new(c.first()?.as_f64()?, c.get(1)?.as_f64()?)));
                (
                    orb_camera::CameraTarget {
                        center,
                        zoom: number("zoom"),
                        bearing: number("bearing"),
                        pitch: number("pitch"),
                        ..Default::default()
                    },
                    orb_camera::FlyOptions {
                        duration: number("duration"),
                        ..Default::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        if legs.is_empty() {
            return Err(format!("{path}: no legs"));
        }
        Ok(Self { legs })
    }
}

pub(crate) fn write_png(path: &str, pixels: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut encoder = png::Encoder::new(
        std::io::BufWriter::new(file),
        Offscreen::WIDTH,
        Offscreen::HEIGHT,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut w| w.write_image_data(pixels))
        .map_err(|e| format!("{path}: {e}"))
}

/// Jumps the camera to a changed copy of itself.
pub(crate) fn move_camera(map: &mut Map, change: impl FnOnce(&mut CameraState)) {
    let mut camera = *map.camera();
    change(&mut camera);
    map.jump_to_state(camera);
}

/// The warnings and errors since the last call.
pub(crate) fn take_warnings(map: &mut Map) -> Vec<String> {
    map.take_events()
        .into_iter()
        .filter_map(|event| match event {
            MapEvent::Warning { message } | MapEvent::Error { message } => Some(message),
            _ => None,
        })
        .collect()
}

/// Pixels that differ.
pub(crate) fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .filter(|(a, b)| a != b)
        .count()
}

//! The CPU frame driver (WebGL 2's, #140) writes what the GPU's cull
//! writes, word for word: on a native device made to take the CPU driver
//! with its tables in storage buffers, the GPU's cull runs beside it over
//! the same scene and frames into buffers of its own
//! (`Renderer::check_driver`), and every frame both drivers' draw
//! arguments and counters, candidates (what each decided of every
//! chunk in every placement), compacted indices and visible lists must
//! agree. Over the scenes that draw every family, the bench's views, the
//! path tests' gestures, and with the cull keeping every candidate.

mod common;

use std::sync::Arc;

use common::scenes::{Scene, scenes};
use common::{CameraState, Geodetic, Map, TargetFormat, Viewport, WgpuGpu};
use orb_hal::{Gpu, TextureId};
use orb_render::{Compared, Renderer};

const SIZE: (u32, u32) = (512, 384);

/// A map of a scene on a device that takes the CPU driver, its GPU twin
/// running.
struct Checked {
    gpu: WgpuGpu,
    map: Map,
    renderer: Renderer,
    target: TextureId,
    time: f64,
    /// Frames compared, and what the last compared.
    frames: usize,
    last: Compared,
}

impl Checked {
    fn new(scene: &Scene) -> Self {
        let mut gpu = pollster::block_on(WgpuGpu::offscreen()).unwrap();
        gpu.force_cpu_driver(false);
        let format = TargetFormat {
            color: orb_hal::TextureFormat::Rgba8Unorm,
            sample_count: 1,
        };
        let target = gpu.create_texture(&orb_hal::TextureDesc {
            label: "target",
            width: SIZE.0,
            height: SIZE.1,
            array_layers: 1,
            dimension: orb_hal::TextureViewDimension::D2,
            format: format.color,
            usage: orb_hal::TextureUsages::RENDER_ATTACHMENT | orb_hal::TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
        });
        let (zoom, bearing, pitch) = scene.at;
        let camera = common::camera(&common::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((Geodetic::new(139.74, 35.685), zoom, bearing, pitch)),
        });
        let mut map = Map::new(common::native_services(Arc::new(|| {})), camera);
        map.set_stable_labels(false);
        map.set_antialias_3d(scene.antialias);
        map.load_style_url(&scene.url);
        let mut renderer = map.create_renderer(&mut gpu, format);
        assert!(
            renderer.check_driver(&mut gpu, true),
            "the renderer takes the CPU driver, with a twin"
        );
        Self {
            gpu,
            map,
            renderer,
            target,
            time: 0.0,
            frames: 0,
            last: Compared::default(),
        }
    }

    /// One frame, both drivers' words compared after it.
    fn frame(&mut self, what: &str) -> bool {
        self.time += 1000.0 / 60.0;
        let viewport = Viewport {
            width: f64::from(SIZE.0),
            height: f64::from(SIZE.1),
        };
        let report = self.map.render(
            &mut self.gpu,
            &mut self.renderer,
            self.target,
            viewport,
            1.0,
            self.time,
        );
        for warning in common::take_warnings(&mut self.map) {
            eprintln!("{what}: {warning}");
        }
        let check = (self.renderer.driver_check()).expect("the CPU driver, with a twin");
        let gpu = &self.gpu;
        let mut read = |buffer| -> Vec<u32> {
            bytemuck::cast_slice(&pollster::block_on(gpu.read_buffer(buffer))).to_vec()
        };
        match check.compare(&mut read) {
            Ok(compared) => self.last = compared,
            Err(e) => panic!("{what}, frame {}: {e}", self.frames),
        }
        self.frames += 1;
        report.idle
    }

    /// Frames until the map is idle three times in a row, each compared.
    fn settle(&mut self, what: &str) {
        let started = std::time::Instant::now();
        let mut idle = 0;
        while idle < 3 {
            idle = if self.frame(what) { idle + 1 } else { 0 };
            assert!(
                started.elapsed().as_secs() < 90,
                "{what}: timed out settling"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// Moves the camera to `to` over `steps` frames, comparing each, then
    /// settles there.
    fn glide(&mut self, what: &str, to: impl Fn(&mut CameraState, f64), steps: u32) {
        let from = *self.map.camera();
        for step in 1..=steps {
            let mut camera = from;
            to(&mut camera, f64::from(step) / f64::from(steps));
            self.map.jump_to_state(camera);
            self.frame(what);
        }
        self.settle(what);
    }

    /// Settled, then with the cull keeping every candidate, then culling
    /// again.
    fn settle_and_keep_all(&mut self, what: &str) {
        self.settle(what);
        let culled = self.last;
        self.map.set_keep_all_cull(true);
        self.settle(&format!("{what}, keeping every candidate"));
        let kept = self.last;
        assert!(kept.kept >= culled.kept, "{what}: keep-all kept fewer");
        self.map.set_keep_all_cull(false);
        self.settle(&format!("{what}, culling again"));
        eprintln!(
            "{what}: {} candidates; indexed: {} of {} kept ({} keeping all); listed: {} of {} rows",
            culled.candidates,
            culled.kept,
            culled.indexed,
            kept.kept,
            culled.visible_rows,
            kept.visible_rows
        );
    }
}

/// Every scene, settled, culled and kept whole.
#[test]
fn the_cpu_driver_writes_the_gpu_cull_s_words_in_every_scene() {
    let names = [
        "liberty-z12",
        "liberty-z15-pitched",
        "globe",
        "terrain",
        "terrain-drape",
        "heatmap",
        "raster",
        "extrusions",
        "extrusions-lit",
    ];
    for scene in scenes(&names) {
        let mut checked = Checked::new(&scene);
        checked.settle_and_keep_all(scene.name);
        if let Some(edit) = scene.edit {
            let tx = orb_runtime::Transaction::from_commands(edit).unwrap();
            checked.map.edit(&tx).unwrap();
            checked.settle(scene.name);
        }
    }
}

/// The bench's views of Liberty and its 3D buildings (bearing 0), and
/// the path tests' gestures between them: a zoom in with a turn and a
/// tilt, a pan frame by frame, a zoom out.
#[test]
fn the_cpu_driver_writes_the_gpu_cull_s_words_along_gestures() {
    let [liberty, buildings] = <[Scene; 2]>::try_from(scenes(&["liberty-z12", "extrusions-lit"]))
        .unwrap_or_else(|_| unreachable!());
    let mut checked = Checked::new(&liberty);
    checked.settle("z12");
    let zoom_to = |zoom: f64| {
        move |c: &mut CameraState, t: f64, from: f64| {
            c.resolution *= 2f64.powf(-(zoom - from) * t);
        }
    };
    let from = 12.0;
    let z15 = zoom_to(15.0);
    checked.glide("to z15", |c, t| z15(c, t, from), 8);
    checked.settle_and_keep_all("z15");
    checked.glide("to z15 pitch 60", |c, t| c.pitch = 60.0 * t, 6);
    checked.settle_and_keep_all("z15 pitch 60");
    // A pan frame by frame, east by a screen, part of it with layers
    // hidden, whose slots move while their nodes aren't culled.
    let visibility = |value: &str| {
        let layers = ["water", "park", "building", "road_minor"];
        let commands = (layers.iter())
            .map(|l| format!(r#"["setLayoutProperty", "{l}", "visibility", "{value}"]"#))
            .collect::<Vec<_>>();
        orb_runtime::Transaction::from_commands(&format!("[{}]", commands.join(","))).unwrap()
    };
    let target = checked.map.camera().target;
    checked.map.edit(&visibility("none")).unwrap();
    checked.glide(
        "panning, layers hidden",
        |c, t| {
            c.target = Geodetic::new(target.lon + 0.015 * t, target.lat);
        },
        6,
    );
    checked.map.edit(&visibility("visible")).unwrap();
    let target = checked.map.camera().target;
    checked.glide(
        "panning",
        |c, t| {
            c.target = Geodetic::new(target.lon + 0.015 * t, target.lat);
        },
        6,
    );
    checked.glide(
        "to z17 flat, turning",
        |c, t| {
            c.pitch = 60.0 * (1.0 - t);
            c.bearing = 45.0 * t;
            c.resolution *= 2f64.powf(-2.0 * t);
        },
        10,
    );
    checked.settle_and_keep_all("z17");
    checked.glide(
        "zooming out to z12",
        |c, t| c.resolution *= 2f64.powf(5.0 * t),
        10,
    );
    checked.settle("z12 again");

    let mut checked = Checked::new(&buildings);
    checked.glide("to bearing 0", |c, t| c.bearing = -20.0 * (1.0 - t), 4);
    checked.settle_and_keep_all("z16 pitch 60, 3D buildings");
}

//! The native demo: the web demo's map app in a window, drawn through
//! `orb-api` as any host draws it, with egui on the same wgpu device for
//! its controls: the map picker (`web/demo/data/scenes.json`), the map
//! options, zoom and the compass. It moves with mouse, trackpad, touch and
//! keyboard; a click prints the features under it.
//!
//! `cargo run --release -p orb-demo -- [scene key or style URL]
//! [--at lon,lat,zoom[,bearing,pitch]] [--flight] [--cache dir]
//! [--count-frames seconds]`
//!
//! Without a scene it opens the Globe. `--flight` flies the scene's
//! looping flight, if it has one, until the user takes the camera.
//! `--cache <dir>` keeps state there between runs: the HTTP cache and the
//! driver's pipelines. `--count-frames <seconds>` prints how many frames
//! the window drew in that long after the map first went idle, then exits.
//! Still pictures are `orbis render` (`orb-cli`).

#![forbid(unsafe_code)]

mod app;
mod options;
mod scenes;

use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupExisting};
use orb_api::{Runtime, RuntimeOptions};

use crate::app::{Demo, Launch};
use crate::scenes::{Camera, Catalogue};

/// The scene a fresh window opens, as the web demo's.
const DEFAULT_SCENE: &str = "globe";

struct Options {
    scene: Option<String>,
    at: Option<Camera>,
    flight: bool,
    count_frames: Option<f64>,
    cache: Option<String>,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        scene: None,
        at: None,
        flight: false,
        count_frames: None,
        cache: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--at" => {
                let parts: Vec<f64> = value()?
                    .split(',')
                    .map(|p| p.trim().parse::<f64>().map_err(|e| format!("--at: {e}")))
                    .collect::<Result<_, _>>()?;
                let (lon, lat, zoom, bearing, pitch) = match parts[..] {
                    [lon, lat, zoom] => (lon, lat, zoom, 0.0, 0.0),
                    [lon, lat, zoom, bearing, pitch] => (lon, lat, zoom, bearing, pitch),
                    _ => return Err("--at needs lon,lat,zoom[,bearing,pitch]".into()),
                };
                options.at = Some(Camera {
                    center: [lon, lat],
                    zoom,
                    bearing,
                    pitch,
                });
            }
            "--count-frames" => {
                let seconds = value()?;
                options.count_frames = Some(
                    seconds
                        .parse()
                        .map_err(|e| format!("--count-frames: {e}"))?,
                );
            }
            "--flight" => options.flight = true,
            "--cache" => options.cache = Some(value()?),
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}")),
            _ => options.scene = Some(arg),
        }
    }
    Ok(options)
}

fn main() -> eframe::Result {
    let options = match options() {
        Ok(options) => options,
        Err(e) => {
            eprintln!("orb-demo: {e}");
            std::process::exit(2);
        }
    };
    let runtime = Runtime::new(&RuntimeOptions {
        state_dir: options.cache.as_ref().map(Into::into),
        ..RuntimeOptions::default()
    })
    .expect("runtime");
    eprintln!("orb-demo: {}", runtime.stats().adapter);
    // egui draws with the runtime's device, so the map's texture is one
    // it can sample.
    let existing = runtime.with_device(|gpu| {
        let (instance, adapter) = gpu
            .instance_and_adapter()
            .expect("the runtime made its own device");
        WgpuSetupExisting {
            instance: instance.clone(),
            adapter: adapter.clone(),
            device: gpu.device().clone(),
            queue: gpu.queue().clone(),
        }
    });
    let native = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Orbis")
            .with_inner_size([1280.0, 800.0]),
        wgpu_options: WgpuConfiguration {
            wgpu_setup: WgpuSetup::Existing(existing),
            ..WgpuConfiguration::default()
        },
        ..eframe::NativeOptions::default()
    };
    let catalogue = Catalogue::read();
    let scene = catalogue.scene_for(options.scene.as_deref().unwrap_or(DEFAULT_SCENE));
    let launch = Launch {
        scene,
        camera: options.at.map(|at| at.state()),
        flight: options.flight,
        count_frames: options.count_frames,
    };
    eframe::run_native(
        "Orbis",
        native,
        Box::new(move |cc| Ok(Box::new(Demo::new(cc, runtime, launch)))),
    )
}

//! The native demo: the web demo's map app in a window, drawn through
//! `orb-api` as any host draws it, with egui on the same wgpu device for
//! its controls: the map picker (`web/demo/data/scenes.json`), the map
//! options, zoom and the compass. It moves with mouse, trackpad, touch and
//! keyboard; a click prints the features under it.
//!
//! `cargo run --release -p orb-demo -- [scene key or style URL]
//! [--at lon,lat,zoom[,bearing,pitch]] [--flight] [--cache dir]`
//!
//! Without a scene it opens the Globe. `--flight` flies the scene's
//! looping flight, if it has one, until the user takes the camera.
//! `--cache <dir>` keeps state there between runs: the HTTP cache and the
//! driver's pipelines. Still pictures are `orbis render` (`orb-cli`).

#![forbid(unsafe_code)]

mod app;
mod host;
mod options;
mod quake;
mod scenes;

use orb_api::{Runtime, RuntimeOptions};

use crate::app::Launch;
use crate::scenes::{Camera, Catalogue};

/// The scene a fresh window opens, as the web demo's.
const DEFAULT_SCENE: &str = "globe";

struct Options {
    scene: Option<String>,
    at: Option<Camera>,
    flight: bool,
    cache: Option<String>,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        scene: None,
        at: None,
        flight: false,
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
            "--flight" => options.flight = true,
            "--cache" => options.cache = Some(value()?),
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}")),
            _ => options.scene = Some(arg),
        }
    }
    Ok(options)
}

fn main() -> Result<(), winit::error::EventLoopError> {
    let options = match options() {
        Ok(options) => options,
        Err(e) => {
            eprintln!("orb-demo: {e}");
            std::process::exit(2);
        }
    };
    let state_dir = options.cache.map(Into::into);
    let runtime = move || {
        Runtime::new(&RuntimeOptions {
            state_dir,
            ..RuntimeOptions::default()
        })
        .expect("runtime")
    };
    let catalogue = Catalogue::read();
    let scene = catalogue.scene_for(options.scene.as_deref().unwrap_or(DEFAULT_SCENE));
    let launch = Launch {
        scene,
        camera: options.at.map(|at| at.state()),
        flight: options.flight,
    };
    host::run(runtime, launch)
}

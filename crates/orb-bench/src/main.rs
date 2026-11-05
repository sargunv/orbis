//! The native bench: `cargo xtask bench [style URL | alps] [--format mvt|mlt]
//! [--frames N] [--warmup N] [--rounds N] [--size WxH] [--ratio R]
//! [--at lon,lat] [--timing timestamps|work-done]
//! [--experiment frames|gestures] [--paths city|alps]`.
//!
//! Without a style URL it loads the committed Tokyo extract, as the browser
//! bench (`web/demo/bench.html`) does; `alps` is the terrain demo's style
//! (live tiles), at its view and with the gestures bench's Alps paths.
//! Prints the Markdown report.

#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(e) = native::run() {
        eprintln!("orb-bench: {e}");
        std::process::exit(1);
    }
}

/// The browser runs the bench through `orb-web`'s `bench` feature.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::Arc;

    use orb_bench::Settings;
    use orb_host_native::{NativeFetch, NativeImageDecode};
    use orb_jobs::ThreadPool;
    use orb_runtime::{Map, PoolJobs, Services};

    const DEMO_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data");

    pub(super) fn run() -> Result<(), String> {
        let mut settings = Settings::default();
        let file = |name: &str| {
            let path = std::path::Path::new(DEMO_DATA).join(name);
            let path = path.canonicalize().map_err(|e| format!("{name}: {e}"))?;
            Ok::<_, String>(format!("file://{}", path.display()))
        };
        let mut style = file("tokyo-mvt.json")?;
        let mut data = "Tokyo extract (mvt, reduced Liberty)".to_string();
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "alps") {
            // The terrain demo's view and paths, unless the options say otherwise.
            settings.center = orb_crs::Geodetic::new(orb_bench::ALPS.0, orb_bench::ALPS.1);
            settings.paths = orb_bench::GesturePaths::Alps;
        }
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            if let Some(name) = arg.strip_prefix("--") {
                let value = args.next().ok_or(format!("{arg} needs a value"))?;
                if name == "format" {
                    if !matches!(value.as_str(), "mvt" | "mlt") {
                        return Err("--format needs mvt or mlt".into());
                    }
                    style = file(&format!("tokyo-{value}.json"))?;
                    data = format!("Tokyo extract ({value}, reduced Liberty)");
                } else {
                    settings.set(name, &value)?;
                }
            } else if arg == "alps" {
                style = file("alps.json")?;
                data = style.clone();
            } else {
                style = arg;
                data = style.clone();
            }
        }
        let device = pollster::block_on(orb_bench::device(settings.timing))?;
        let about = vec![
            format!("adapter: {}", device.adapter),
            "native, wgpu".to_string(),
            format!("data: {data}"),
        ];
        let new_map = |camera| {
            let mut map = Map::new(services(), Some(camera));
            map.load_style_url(&style);
            Ok(map)
        };
        let report = pollster::block_on(orb_bench::run(
            device,
            &settings,
            about,
            new_map,
            |status| eprintln!("orb-bench: {status}"),
        ))?;
        print!("{}", report.markdown());
        Ok(())
    }

    fn services() -> Services {
        let fetch = NativeFetch::new();
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
            wake: Arc::new(|| {}),
        }
    }
}

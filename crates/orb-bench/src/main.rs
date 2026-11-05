//! The native bench: `cargo xtask bench [style URL | alps] [--format mvt|mlt]
//! [--frames N] [--warmup N] [--rounds N] [--size WxH] [--ratio R]
//! [--at lon,lat] [--timing timestamps|work-done]
//! [--experiment frames|labels|gestures]
//! [--labels on|off] [--paths city|alps]`.
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

    /// The style: an explicit URL, or the Tokyo extract in one encoding.
    enum Style {
        Url(String),
        Tokyo(String),
    }

    pub(super) fn run() -> Result<(), String> {
        let mut settings = Settings::default();
        let mut style = Style::Tokyo("mvt".into());
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "alps") {
            // The Alps' view and paths, unless the options say otherwise.
            settings.center = orb_geo::Geodetic::new(orb_bench::ALPS.0, orb_bench::ALPS.1);
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
                    style = Style::Tokyo(value);
                } else {
                    settings.set(name, &value)?;
                }
            } else if arg == "alps" {
                let path = std::path::Path::new(DEMO_DATA)
                    .join("alps.json")
                    .canonicalize()
                    .map_err(|e| e.to_string())?;
                style = Style::Url(format!("file://{}", path.display()));
            } else {
                style = Style::Url(arg);
            }
        }
        let device = pollster::block_on(orb_bench::device(settings.timing))?;
        let (data, json) = match &style {
            Style::Url(url) => (url.clone(), None),
            Style::Tokyo(encoding) => (
                format!("Tokyo extract ({encoding}, reduced Liberty)"),
                Some(tokyo_style(encoding)?),
            ),
        };
        let about = vec![
            format!("adapter: {}", device.adapter),
            "native, wgpu".to_string(),
            format!("data: {data}"),
        ];
        let new_map = |camera| {
            let mut map = Map::new(services(), Some(camera));
            match (&style, &json) {
                (_, Some(json)) => map.set_style_json(json)?,
                (Style::Url(url), None) => map.load_style_url(url),
                (Style::Tokyo(_), None) => unreachable!("loaded above"),
            }
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

    /// The committed extract's style, its archive, glyphs and sprites
    /// pointed at the files.
    fn tokyo_style(encoding: &str) -> Result<String, String> {
        let path = std::path::Path::new(DEMO_DATA)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let json = std::fs::read_to_string(path.join(format!("tokyo-{encoding}.json")))
            .map_err(|e| e.to_string())?;
        Ok(json
            .replace(
                "pmtiles://./data/",
                &format!("pmtiles://file://{}/", path.display()),
            )
            .replace("\"./data/", &format!("\"file://{}/", path.display())))
    }
}

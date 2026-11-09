//! The bench: `cargo xtask bench [style URL | alps] [--format mvt|mlt]
//! [--experiment frames|gestures|load|tiles] [--frames N] [--warmup N]
//! [--rounds N] [--samples N] [--size WxH] [--ratio R] [--at lon,lat]
//! [--view zoom,pitch,bearing]... [--views file] [--filter s] [--nudge]
//! [--paths city|alps]`.
//!
//! Without a style URL it loads the committed Tokyo extract, as the browser
//! bench (`web/demo/bench.html`) does; `alps` is the terrain demo's style
//! (live tiles), at its view and with the gestures bench's Alps paths.
//! `--views` reads cameras in mln-rs's `views.txt` format, keeping those
//! whose style column is the style URL's last path segment (`liberty` for
//! `…/styles/liberty`). Prints the Markdown report.

#![forbid(unsafe_code)]

use orb_api::{Runtime, RuntimeOptions};
use orb_bench::Settings;

const DEMO_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data");

fn main() {
    if let Err(e) = run() {
        eprintln!("orb-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut settings = Settings::default();
    let data = std::path::Path::new(DEMO_DATA)
        .canonicalize()
        .map_err(|e| format!("{DEMO_DATA}: {e}"))?;
    let file = |name: &str| format!("file://{}", data.join(name).display());
    let mut style = file("tokyo-mvt.json");
    let mut about_data = "Tokyo extract (mvt, reduced Liberty)".to_string();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "alps") {
        // The terrain demo's view and paths, unless the options say otherwise.
        settings.center = orb_api::Geodetic::new(orb_bench::ALPS.0, orb_bench::ALPS.1);
        settings.paths = orb_bench::GesturePaths::Alps;
    }
    let mut views_file = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--nudge" {
            settings.nudge = true;
        } else if let Some(name) = arg.strip_prefix("--") {
            let value = args.next().ok_or(format!("{arg} needs a value"))?;
            match name {
                "format" => {
                    if !matches!(value.as_str(), "mvt" | "mlt") {
                        return Err("--format needs mvt or mlt".into());
                    }
                    style = file(&format!("tokyo-{value}.json"));
                    about_data = format!("Tokyo extract ({value}, reduced Liberty)");
                }
                "views" => views_file = Some(value),
                _ => settings.set(name, &value)?,
            }
        } else if arg == "alps" {
            style = file("alps.json");
            about_data = style.clone();
        } else {
            style = arg;
            about_data = style.clone();
        }
    }
    if let Some(path) = &views_file {
        let name = style_name(&style);
        settings.views = orb_bench::read_views(path, name)?;
        about_data = format!("{about_data}; the {name} views of {path}");
    }
    let runtime = Runtime::new(&RuntimeOptions::default()).map_err(|e| e.to_string())?;
    let about = vec![
        format!("device: {}", runtime.stats().adapter),
        "native, orb-api".to_string(),
        format!("data: {about_data}"),
    ];
    let report = orb_bench::run(&runtime, &settings, about, &style, &data, |status| {
        eprintln!("orb-bench: {status}")
    })?;
    print!("{}", report.markdown());
    Ok(())
}

/// A style URL's last path segment without its extension: `liberty` for
/// `https://tiles.openfreemap.org/styles/liberty`.
fn style_name(url: &str) -> &str {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let last = path.rsplit('/').next().unwrap_or(path);
    last.strip_suffix(".json").unwrap_or(last)
}

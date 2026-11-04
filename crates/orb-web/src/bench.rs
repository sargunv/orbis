//! The frame and paint benches (decisions 0002, 0017) in the browser, for
//! `web/demo/bench.html`. Built into `harness.wasm` with the `harness`
//! feature, never into `core.wasm`. The measurement is `orb-bench`'s, shared with the
//! native tool; this module only wires the page's options, the map's
//! services and progress to it.

use std::sync::Arc;

use orb_bench::Settings;
use orb_host_web::{WebFetch, WebImageDecode};
use orb_jobs::Inline;
use orb_runtime::{Map, PoolJobs, Services};
use wasm_bindgen::prelude::*;

/// Runs the bench on the style at `style_url`, its jobs on this thread.
/// `options` are `orb-bench` settings as name, value pairs (the page's
/// query); `about` lines head the report (adapter, browser, data).
/// `on_progress` hears a line per phase. Resolves to the Markdown report.
#[wasm_bindgen(js_name = runBench)]
pub async fn run_bench(
    style_url: String,
    options: Vec<String>,
    about: Vec<String>,
    on_progress: js_sys::Function,
) -> Result<String, JsError> {
    let mut settings = Settings::default();
    for pair in options.chunks(2) {
        let [name, value] = pair else {
            return Err(JsError::new("options come in name, value pairs"));
        };
        settings.set(name, value).map_err(|e| JsError::new(&e))?;
    }
    let device = orb_bench::device(settings.timing)
        .await
        .map_err(|e| JsError::new(&e))?;
    let new_map = |camera| {
        let fetch = Arc::new(WebFetch);
        let jobs = PoolJobs::new(fetch.clone(), Arc::new(WebImageDecode), Arc::new(Inline));
        jobs.set_local_glyphs(Arc::new(orb_host_web::CanvasGlyphs));
        jobs.set_font_files(Arc::new(orb_host::FontFileFaces::new()));
        let services = Services {
            clock: Arc::new(orb_host_web::WebClock),
            image_decode: Arc::new(orb_host_web::WebImageDecode),
            images: Default::default(),
            jobs,
            fetch,
            wake: Arc::new(|| {}),
        };
        let mut map = Map::new(services, Some(camera));
        map.load_style_url(&style_url);
        Ok(map)
    };
    let progress = |status: &str| {
        let _ = on_progress.call1(&JsValue::NULL, &JsValue::from_str(status));
    };
    let report = orb_bench::run(device, &settings, about, new_map, progress)
        .await
        .map_err(|e| JsError::new(&e))?;
    Ok(report.markdown())
}

//! Offline regions in the browser: each JS context (the page and every
//! worker) fetches through one [`RegionFetch`] over the origin private
//! file system, and the page downloads, lists and deletes regions. A page
//! that changes the regions tells the other contexts on the
//! `orbis-regions` broadcast channel, and each calls `reloadRegions`.
//! Each context loads the archive module its regions' extracts are read
//! and written with through its own loader (`setModuleLoader`), as a job
//! worker loads the bidirectional algorithm's.

use rustc_hash::FxHashMap as HashMap;
use std::cell::RefCell;
use std::sync::Arc;

use crate::host::{OpfsStorage, WebFetch};
use orb_host::{Fetch, Storage};
use orb_runtime::modules::Load;
use orb_runtime::offline::{self, RegionDefinition, RegionEvent, RegionFetch};
use wasm_bindgen::prelude::*;

thread_local! {
    static FETCH: Arc<RegionFetch> =
        RegionFetch::new(Arc::new(WebFetch), Arc::new(OpfsStorage), load());
    /// Loads a feature module into this context: `loader(name)` answers
    /// a promise of it in.
    static LOADER: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
    /// Each download's event callback, by a token its events carry.
    static HANDLERS: RefCell<HashMap<u32, js_sys::Function>> = RefCell::default();
    static NEXT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// What this context's maps and tile jobs fetch through.
pub(crate) fn service() -> Arc<dyn Fetch> {
    FETCH.with(|f| f.clone())
}

/// How this context loads a feature module, by its name (`web/src/
/// builds.ts`'s `instantiate` sets it): `loader(name)` answers a promise
/// that resolves once the module is in this context, or `undefined` if
/// it already is. Loading one twice must be harmless.
#[wasm_bindgen(js_name = setModuleLoader)]
pub fn set_module_loader(loader: js_sys::Function) {
    LOADER.with(|l| *l.borrow_mut() = Some(loader));
}

/// Loads modules into this context through its loader.
pub(crate) fn load() -> Load {
    Arc::new(|modules, done| {
        let loads: Result<js_sys::Array, JsValue> = LOADER.with(|l| {
            let loader = l.borrow().clone().ok_or("no module loader")?;
            modules
                .names()
                .map(|name| loader.call1(&JsValue::NULL, &name.into()))
                .collect()
        });
        wasm_bindgen_futures::spawn_local(async move {
            let loaded = match loads {
                Ok(loads) => wasm_bindgen_futures::JsFuture::from(js_sys::Promise::all(&loads))
                    .await
                    .map(drop),
                Err(e) => Err(e),
            };
            done(loaded.map_err(|e| {
                web_sys::console::warn_1(&e);
                match e.dyn_ref::<js_sys::Error>() {
                    Some(e) => e.message().into(),
                    None => "a module didn't load".into(),
                }
            }));
        });
    })
}

/// Transforms every request this JS context fetches (the page's or a job
/// worker's; GL JS `transformRequest`): `hook(url)` answers `{url,
/// headers}` or a promise of it, `null` to refuse, or `undefined` to leave
/// it. No hook, none.
#[wasm_bindgen(js_name = setRequestHook)]
pub fn set_request_hook(hook: Option<js_sys::Function>) {
    crate::host::set_request_hook(hook);
}

/// What the page fetched ahead of its maps: `take(url)` answers a
/// promise of the response for `url`, once, or `undefined`
/// (`web/src/preload.ts`). The page's whole-body fetches take theirs.
#[wasm_bindgen(js_name = setPreloads)]
pub fn set_preloads(take: Option<js_sys::Function>) {
    crate::host::set_preloads(take);
}

/// Reads the regions again, after another context changed them.
#[wasm_bindgen(js_name = reloadRegions)]
pub fn reload_regions() {
    FETCH.with(|f| f.reload());
}

/// Downloads `style_url`'s map of `bounds` (west, south, east, north)
/// between map zooms `zooms` into the origin private file system.
/// `on_event` gets `{type: "progress" | "complete" | "failed", id, done,
/// total, bytes, failed, message}`. Resolves to the region's id.
#[wasm_bindgen(js_name = downloadRegion)]
pub async fn download_region(
    name: String,
    style_url: String,
    bounds: Vec<f64>,
    zooms: Vec<f64>,
    on_event: js_sys::Function,
) -> Result<u32, JsError> {
    let (Ok(bounds), Ok(zooms)) = (bounds.try_into(), zooms.try_into()) else {
        return Err(JsError::new("bounds are four numbers and zooms two"));
    };
    let token = NEXT.with(|n| {
        n.set(n.get() + 1);
        n.get()
    });
    HANDLERS.with(|h| h.borrow_mut().insert(token, on_event));
    let events: offline::RegionEvents = Arc::new(move |event| {
        // Single-threaded: events arrive on this context's thread.
        let done = !matches!(event, RegionEvent::Progress { .. });
        if matches!(event, RegionEvent::Complete { .. }) {
            reload_regions();
        }
        let handler = HANDLERS.with(|h| {
            let mut h = h.borrow_mut();
            if done {
                h.remove(&token)
            } else {
                h.get(&token).cloned()
            }
        });
        if let Some(handler) = handler {
            let _ = handler.call1(&JsValue::NULL, &event_object(&event));
        }
    });
    let storage: Arc<dyn Storage> = Arc::new(OpfsStorage);
    let (tx, rx) = futures_channel::oneshot::channel();
    offline::download(
        &load(),
        Arc::new(WebFetch),
        storage,
        RegionDefinition {
            name,
            style_url,
            bounds,
            zooms,
            ..RegionDefinition::default()
        },
        events,
        Box::new(move |r| drop(tx.send(r))),
    );
    let download = rx
        .await
        .map_err(|_| JsError::new("download dropped"))?
        .map_err(|e| JsError::new(&e))?;
    Ok(download.id())
}

fn event_object(event: &RegionEvent) -> JsValue {
    let object = js_sys::Object::new();
    let set = |key: &str, value: JsValue| crate::js::set(&object, key, &value);
    match event {
        RegionEvent::Progress {
            id,
            done,
            total,
            bytes,
        } => {
            set("type", "progress".into());
            set("id", (*id).into());
            set("done", (*done as f64).into());
            set("total", (*total as f64).into());
            set("bytes", (*bytes as f64).into());
        }
        RegionEvent::Complete { id, bytes, failed } => {
            set("type", "complete".into());
            set("id", (*id).into());
            set("bytes", (*bytes as f64).into());
            set("failed", (*failed as f64).into());
        }
        RegionEvent::Failed { id, message } => {
            set("type", "failed".into());
            set("id", (*id).into());
            set("message", message.into());
        }
    }
    object.into()
}

/// Deletes a region.
#[wasm_bindgen(js_name = deleteRegion)]
pub async fn delete_region(id: u32) -> Result<(), JsError> {
    let (tx, rx) = futures_channel::oneshot::channel();
    offline::delete(&OpfsStorage, id, Box::new(move |r| drop(tx.send(r))));
    rx.await
        .map_err(|_| JsError::new("delete dropped"))?
        .map_err(|e| JsError::new(&e))?;
    reload_regions();
    Ok(())
}

/// The regions, as JSON: `[{id, name, style, bounds, zooms, complete,
/// bytes}]`.
#[wasm_bindgen(js_name = listRegions)]
pub async fn list_regions() -> Result<String, JsError> {
    let (tx, rx) = futures_channel::oneshot::channel();
    offline::list(Arc::new(OpfsStorage), Box::new(move |r| drop(tx.send(r))));
    let regions = rx
        .await
        .map_err(|_| JsError::new("list dropped"))?
        .map_err(|e| JsError::new(&e))?;
    let json = regions
        .iter()
        .map(|r| {
            let d = &r.definition;
            format!(
                r#"{{"id":{},"name":{:?},"style":{:?},"bounds":{:?},"zooms":{:?},"complete":{},"bytes":{}}}"#,
                r.id, d.name, d.style_url, d.bounds, d.zooms, r.complete, r.bytes
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("[{json}]"))
}

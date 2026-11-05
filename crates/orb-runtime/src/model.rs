//! Model layers (plan §9.4): loading a scene document's glTF models, and
//! a model node's paint.
//!
//! A model layer draws as one node per model its `model-id` can pick, each
//! with the layer's points that pick it (`style`, `tile`). The map loads
//! the document's models through [`Jobs::model`](crate::Jobs::model) and
//! uploads each at its index in the document's models.

use std::sync::{Arc, Mutex};

use orb_3d::gpu::GpuModel;
use orb_3d::load::Load;
use orb_hal::Gpu;
use orb_host::Fetch;
use orb_jobs::Executor;
use orb_render::{Curve, ModelPaint, NodePaint, Renderer};

use crate::style::{NodePlan, PaintProperty, StylePlan};

/// Called once with a loaded model, on any thread.
pub type ModelDone = Box<dyn FnOnce(Result<GpuModel, String>) + Send>;

/// Loads the glTF document at `url` through `fetch`: its external buffers,
/// then its images by URI, which are decoded and packed with the model on
/// `executor`. An image that fails to load or decode leaves its material
/// slots untextured.
pub fn load(fetch: Arc<dyn Fetch>, executor: Arc<dyn Executor>, url: String, done: ModelDone) {
    let document = url.clone();
    let f = Arc::clone(&fetch);
    fetch.get(
        &document,
        Box::new(move |fetched| {
            let load = fetched
                .map_err(|e| format!("model {url}: {e}"))
                .and_then(|bytes| Load::new(&url, &bytes).map_err(|e| format!("model {url}: {e}")));
            match load {
                Ok(load) => step(f, executor, Arc::new(Mutex::new(Some((load, done))))),
                Err(e) => done(Err(e)),
            }
        }),
    );
}

/// A model loading and its done, taken by whichever step finishes it.
type Loading = Arc<Mutex<Option<(Load, ModelDone)>>>;

/// Fetches what the load wants next, or decodes and packs it once it
/// wants nothing.
fn step(fetch: Arc<dyn Fetch>, executor: Arc<dyn Executor>, loading: Loading) {
    let mut guard = loading.lock().expect("model load poisoned");
    let Some((load, _)) = guard.as_mut() else {
        return;
    };
    // The buffers, then (once the model is built from them) its images.
    let mut wanted = load.wanted();
    let mut encoded = None;
    if wanted.is_empty() {
        match load.images() {
            Ok(None) => wanted = load.wanted(),
            Ok(Some(images)) => encoded = Some(images),
            Err(e) => {
                let (_, done) = guard.take().expect("checked above");
                drop(guard);
                done(Err(e));
                return;
            }
        }
    }
    if let Some(encoded) = encoded {
        let (load, done) = guard.take().expect("checked above");
        drop(guard);
        executor.spawn(Box::new(move || {
            let decoded: Vec<_> = encoded
                .iter()
                .map(|bytes| {
                    let image = orb_host::decode_image(bytes.as_deref()?).ok()?;
                    Some((image.width, image.height, image.pixels))
                })
                .collect();
            done(load.finish(&decoded));
        }));
        return;
    }
    drop(guard);
    let remaining = Arc::new(Mutex::new(wanted.len()));
    for url in wanted {
        let (fetch2, executor, loading, remaining) = (
            Arc::clone(&fetch),
            Arc::clone(&executor),
            Arc::clone(&loading),
            Arc::clone(&remaining),
        );
        let target = url.clone();
        fetch.get(
            &target,
            Box::new(move |fetched| {
                if let Some((load, _)) = loading.lock().expect("model load poisoned").as_mut() {
                    // A failed fetch arrives empty: a buffer then fails to
                    // read, an image to decode.
                    load.fetched(&url, fetched.unwrap_or_default());
                }
                let mut left = remaining.lock().expect("model load poisoned");
                *left -= 1;
                if *left == 0 {
                    drop(left);
                    step(fetch2, executor, loading);
                }
            }),
        );
    }
}

/// Writes a model node's paint at `zoom`. `maxima` are its property rows'
/// maxima; `metres_per_px` the camera target's, for the cull's reach.
pub(crate) fn paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (plan, node, index): (&StylePlan, &NodePlan, u32),
    zoom: f64,
    maxima: &[f32],
    (bounds, metres_per_px): (Option<[[f32; 3]; 2]>, f64),
) {
    let l = &plan.style.layers[node.layer];
    let mut curves = [Curve::Step; 12];
    let mut numbers = [0.0f32; 12];
    let mut color = [0.0; 4];
    let mut table_mask = 0;
    for (slot, property) in plan.paint_properties(node).iter().enumerate() {
        if property.per_feature() {
            table_mask |= 1 << slot;
            curves[slot] = property.curve();
            numbers[slot] = maxima.get(slot).copied().unwrap_or(0.0);
        } else {
            match property {
                PaintProperty::Color(_) => color = property.color(zoom),
                p => numbers[slot] = p.number(zoom),
            }
        }
    }
    let triple = |at: usize| [numbers[at], numbers[at + 1], numbers[at + 2]];
    let (scale, translation) = (triple(3), triple(9));
    // The farthest a model reaches from its point, and above the ground.
    let [lo, hi] = bounds.unwrap_or_default();
    let largest = scale.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let radius = lo.iter().chain(&hi).fold(0.0f32, |m, v| m.max(v.abs())) * largest
        + translation.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    // A view's latitudes vary its metres per pixel a little: slack.
    let reach_px = (f64::from(radius) / metres_per_px * 1.1) as f32 + 1.0;
    let height = hi[2].max(lo[2]).max(0.0) * scale[2].abs() + translation[2];
    renderer.set_paint(
        gpu,
        index,
        &NodePaint::Model(ModelPaint {
            color,
            color_mix: numbers[1],
            emissive: numbers[2],
            scale,
            rotation: triple(6),
            translation,
            opacity: l.get(orb_style::MODEL_OPACITY).evaluate(zoom),
            table_mask,
            curves,
            reach_px,
            height,
        }),
    );
}

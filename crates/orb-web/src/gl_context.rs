//! A WebGL 2 canvas's context lost and restored. WebGPU says a device is
//! lost through its callback; a WebGL context says so with events on its
//! canvas, and comes back on the same canvas once the browser restores it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use js_sys::{Function, Reflect};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};

fn call(target: &JsValue, method: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
    let function = Reflect::get(target, &method.into())?.dyn_into::<Function>()?;
    let args: js_sys::Array = args.iter().collect();
    function.apply(target, &args)
}

/// The canvas's context, which wgpu made.
fn of(canvas: &JsValue) -> Result<JsValue, JsValue> {
    call(canvas, "getContext", &["webgl2".into()])
}

fn listen(canvas: &JsValue, event: &str, handler: impl FnMut(JsValue) + 'static) {
    let handler = Closure::<dyn FnMut(JsValue)>::new(handler).into_js_value();
    let options = crate::js::object(&[("once", true.into())]);
    let _ = call(
        canvas,
        "addEventListener",
        &[event.into(), handler, options.into()],
    );
}

/// Sets `lost` and wakes map `map` when the context is lost, keeping it
/// restorable (the event's default would give it up). It listens before
/// the device is asked for, so a context lost while the map starts is
/// heard too.
pub(crate) fn watch(canvas: &JsValue, lost: Arc<AtomicBool>, map: u32) {
    listen(canvas, "webglcontextlost", move |event| {
        let _ = call(&event, "preventDefault", &[]);
        lost.store(true, Ordering::Relaxed);
        crate::map::wake(map);
    });
}

/// Whether the canvas's context is lost now.
pub(crate) fn is_lost(canvas: &JsValue) -> bool {
    of(canvas)
        .and_then(|context| call(&context, "isContextLost", &[]))
        .is_ok_and(|lost| lost.is_truthy())
}

/// Takes flat varyings from a primitive's first vertex, as WebGPU does
/// (`WEBGL_provoking_vertex`), where the browser can, and says whether it
/// did: the terrain's mesh then orders its triangles for it
/// (`Caps::flat_from_last`). Under GL's default, the last vertex, ANGLE's
/// Metal back end (Chrome and Safari on Apple devices) rewrites the
/// indices of every draw of a program with flat varyings in a compute
/// pass first (#140).
pub(crate) fn first_vertex(canvas: &JsValue) -> bool {
    const FIRST_VERTEX_CONVENTION_WEBGL: u32 = 0x8E4D;
    let Ok(extension) = of(canvas)
        .and_then(|context| call(&context, "getExtension", &["WEBGL_provoking_vertex".into()]))
    else {
        return false;
    };
    extension.is_object()
        && call(
            &extension,
            "provokingVertexWEBGL",
            &[FIRST_VERTEX_CONVENTION_WEBGL.into()],
        )
        .is_ok()
}

/// How long a lost context is waited for. Chrome restored one 1005 and
/// 1008 ms after its GPU process crashed (the DevTools protocol's
/// `Browser.crashGpuProcess`), and never one lost again soon after (not
/// within a minute): a browser restores within seconds or not at all.
const RESTORE_MS: i32 = 10_000;

/// Waits until the canvas's context, if lost, is restored, and fails after
/// [`RESTORE_MS`].
pub(crate) async fn restored(canvas: &JsValue) -> Result<(), String> {
    if !is_lost(canvas) {
        return Ok(());
    }
    let restored = js_sys::Promise::new(&mut |resolve, reject| {
        listen(canvas, "webglcontextrestored", move |_| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
        let expire = Closure::once_into_js(move || {
            let _ = reject.call0(&JsValue::UNDEFINED);
        });
        let _ = call(
            &js_sys::global(),
            "setTimeout",
            &[expire, RESTORE_MS.into()],
        );
    });
    match wasm_bindgen_futures::JsFuture::from(restored).await {
        Ok(_) => Ok(()),
        Err(_) => Err(format!(
            "the browser didn't restore the WebGL 2 context within {} s",
            RESTORE_MS / 1000
        )),
    }
}

/// Loses the context on purpose (`WEBGL_lose_context`), as a browser
/// would, and has the browser restore it a moment later.
pub(crate) fn lose(canvas: &JsValue) -> Result<(), JsValue> {
    let extension = call(&of(canvas)?, "getExtension", &["WEBGL_lose_context".into()])?;
    call(&extension, "loseContext", &[])?;
    let restore = Closure::once_into_js(move || {
        let _ = call(&extension, "restoreContext", &[]);
    });
    call(&js_sys::global(), "setTimeout", &[restore, 50.into()])?;
    Ok(())
}

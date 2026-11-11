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
/// restorable (the event's default would give it up).
pub(crate) fn watch(canvas: &JsValue, lost: Arc<AtomicBool>, map: u32) {
    listen(canvas, "webglcontextlost", move |event| {
        let _ = call(&event, "preventDefault", &[]);
        lost.store(true, Ordering::Relaxed);
        crate::map::wake(map);
    });
}

/// Takes flat varyings from a primitive's first vertex, as WebGPU does
/// (`WEBGL_provoking_vertex`), where the browser can. Orbis's flat
/// varyings are the same at every vertex of a primitive, so this changes
/// no pixel; but under GL's default, the last vertex, ANGLE's Metal back
/// end (Chrome and Safari on Apple devices) rewrites the indices of every
/// draw of a program with flat varyings in a compute pass first (#140).
pub(crate) fn first_vertex(canvas: &JsValue) {
    const FIRST_VERTEX_CONVENTION_WEBGL: u32 = 0x8E4D;
    let Ok(extension) = of(canvas)
        .and_then(|context| call(&context, "getExtension", &["WEBGL_provoking_vertex".into()]))
    else {
        return;
    };
    if extension.is_object() {
        let _ = call(
            &extension,
            "provokingVertexWEBGL",
            &[FIRST_VERTEX_CONVENTION_WEBGL.into()],
        );
    }
}

/// Waits until the canvas's context, if lost, is restored.
pub(crate) async fn restored(canvas: &JsValue) -> Result<(), JsValue> {
    let context = of(canvas)?;
    if !call(&context, "isContextLost", &[])?.is_truthy() {
        return Ok(());
    }
    let restored = js_sys::Promise::new(&mut |resolve, _| {
        listen(canvas, "webglcontextrestored", move |_| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
    });
    wasm_bindgen_futures::JsFuture::from(restored).await?;
    Ok(())
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

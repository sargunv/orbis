//! Wall-clock time and waiting on both targets: `Instant` natively, and
//! `performance.now()` in the browser, which coarsens it (Chrome to 100 µs,
//! or 5 µs cross-origin isolated; other browsers to as much as 1 ms).

use std::time::Duration;

/// Time since an arbitrary origin.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn now() -> Duration {
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    ORIGIN.get_or_init(std::time::Instant::now).elapsed()
}

/// Time since the page's time origin.
#[cfg(target_arch = "wasm32")]
pub(crate) fn now() -> Duration {
    let ms = window().performance().expect("performance").now();
    Duration::from_secs_f64(ms / 1000.0)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) async fn sleep(duration: Duration) {
    std::thread::sleep(duration);
}

/// Yields to the browser's event loop for at least `duration`: fetches and
/// WebGPU callbacks run only then.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn sleep(duration: Duration) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        window()
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                &resolve,
                duration.as_millis() as i32,
            )
            .expect("setTimeout");
    });
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("setTimeout resolves");
}

#[cfg(target_arch = "wasm32")]
fn window() -> web_sys::Window {
    web_sys::window().expect("the bench runs on the page's thread")
}

/// The smallest step [`now`] takes, sampled over up to 50 ms.
pub(crate) fn resolution() -> Duration {
    let started = now();
    let mut last = started;
    let mut steps = 0;
    let mut smallest = Duration::MAX;
    while steps < 50 && last - started < Duration::from_millis(50) {
        let t = now();
        if t > last {
            smallest = smallest.min(t - last);
            steps += 1;
            last = t;
        }
    }
    smallest
}

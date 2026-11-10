// The process's log callback. Orbis reports problems as map events; those
// MapLibre Native logs instead (a tile's or a sprite's failure) come here.

use std::cell::Cell;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::base::{Failure, Result, UserData, boundary, sized};
use crate::delivery::deliver;
use crate::sys::*;

struct Handler {
    callback: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        mln_log_severity,
        mln_log_event,
        i64,
        mln_buffer_view,
    ),
    user_data: UserData,
    release_user_data: mln_user_data_release,
    queue: u64,
}

static HANDLER: Mutex<Option<Handler>> = Mutex::new(None);

thread_local! {
    /// Set while this thread runs a log callback or its release. Both may
    /// run under [`HANDLER`]'s lock, as MapLibre Native runs them under its
    /// log mutex, so a call back into the logging API is refused rather
    /// than left to deadlock.
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

fn handler() -> MutexGuard<'static, Option<Handler>> {
    HANDLER.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs a log callback or its release, marking this thread as inside one.
fn in_callback(f: impl FnOnce()) {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_CALLBACK.set(self.0);
        }
    }
    let _reset = Reset(IN_CALLBACK.replace(true));
    f();
}

/// Refuses a logging call made from a log callback or its release.
fn not_reentered() -> Result {
    if IN_CALLBACK.get() {
        return Err(Failure::state(
            "the logging API can't be called from a log callback or its release",
        ));
    }
    Ok(())
}

fn release(old: Option<Handler>) {
    if let Some(Handler {
        user_data,
        release_user_data: Some(release),
        queue,
        ..
    }) = old
    {
        deliver(
            queue,
            Box::new(move || {
                // SAFETY: the registration's release, called once.
                in_callback(|| unsafe { release(user_data.get()) });
            }),
        );
    }
}

pub(crate) fn log(severity: mln_log_severity, message: &str) {
    // A message raised inside a log callback can't reach the callback:
    // this thread may hold its lock.
    let handler = (!IN_CALLBACK.get()).then(handler);
    let Some(h) = handler.as_ref().and_then(|h| h.as_ref()) else {
        if severity >= MLN_LOG_SEVERITY_WARNING {
            eprintln!("orbis: {message}");
        }
        return;
    };
    let (callback, user_data) = (h.callback, h.user_data);
    let mut message = message.to_owned();
    deliver(
        h.queue,
        Box::new(move || {
            // SAFETY: the caller's registration; the message lives until it returns.
            in_callback(|| unsafe {
                callback(
                    user_data.get(),
                    severity,
                    MLN_LOG_EVENT_GENERAL,
                    0,
                    crate::base::lend(&mut message),
                )
            });
        }),
    );
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_log_set_callback(
    handler: *const mln_log_handler,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            not_reentered()?;
            let h = sized(handler, "handler")?;
            let callback = h
                .callback
                .ok_or_else(|| Failure::argument("handler.callback is null"))?;
            crate::delivery::check_queue(h.queue.0, "handler.queue")?;
            let new = Handler {
                callback,
                user_data: UserData(h.user_data),
                release_user_data: h.release_user_data,
                queue: h.queue.0,
            };
            release(self::handler().replace(new));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_log_clear_callback(out_diagnostic: *mut mln_diagnostic) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            not_reentered()?;
            release(handler().take());
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_log_set_async_severity_mask(
    mask: mln_log_severity_flags,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // Every message is delivered from the thread that raises it; there is
    // no synchronous path to choose against.
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            if mask & !MLN_LOG_SEVERITY_FLAG_ALL != 0 {
                return Err(Failure::argument(format!(
                    "the severity mask {mask:#x} has bits that name no severity"
                )));
            }
            Ok(())
        })
    }
}

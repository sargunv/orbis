// The process's log callback. Orbis reports problems as map events; those
// MapLibre Native logs instead (a tile's or a sprite's failure) come here.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use crate::base::{Failure, Result, UserData, boundary, sized};
use crate::delivery::{deliver, deliver_now};
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
    /// The severities that also go to the platform's log.
    platform: mln_log_severity_flags,
}

static HANDLER: Mutex<Option<Handler>> = Mutex::new(None);

/// The severities logged on the log thread rather than the one that logs
/// them.
static ASYNC: AtomicU32 = AtomicU32::new(MLN_LOG_SEVERITY_FLAG_DEFAULT);

/// A message for the log thread.
struct Record {
    severity: mln_log_severity,
    message: String,
}

/// The log thread, which delivers the asynchronous severities in order,
/// as MapLibre Native's logging thread does.
fn log_thread() -> &'static Mutex<Sender<Record>> {
    static THREAD: OnceLock<Mutex<Sender<Record>>> = OnceLock::new();
    THREAD.get_or_init(|| {
        let (send, receive) = channel::<Record>();
        std::thread::Builder::new()
            .name("orbis-log".into())
            .spawn(move || {
                for record in receive {
                    write(record.severity, record.message);
                }
            })
            .expect("the log thread starts");
        Mutex::new(send)
    })
}

fn flag(severity: mln_log_severity) -> mln_log_severity_flags {
    1 << severity
}

/// Checks that `mask` names severities only.
fn check_severities(mask: mln_log_severity_flags, what: &str) -> Result {
    if mask & !MLN_LOG_SEVERITY_FLAG_ALL != 0 {
        return Err(Failure::argument(format!(
            "{what} {mask:#x} has unknown bits: they name no severity"
        )));
    }
    Ok(())
}

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
                crate::base::to_host(|| in_callback(|| unsafe { release(user_data.get()) }));
            }),
        );
    }
}

/// Logs `message`: on this thread, or for the asynchronous severities on
/// the log thread.
pub(crate) fn log(severity: mln_log_severity, message: &str) {
    let message = message.to_owned();
    if ASYNC.load(Ordering::Relaxed) & flag(severity) == 0 {
        return write(severity, message);
    }
    let sent = log_thread()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .send(Record { severity, message });
    // Without the log thread, the message goes out here.
    if let Err(unsent) = sent {
        write(severity, unsent.0.message);
    }
}

/// Hands a message to the callback, or its queue, or the platform's log.
fn write(severity: mln_log_severity, mut message: String) {
    // A message raised inside a log callback can't reach the callback:
    // this thread may hold its lock.
    let handler = (!IN_CALLBACK.get()).then(handler);
    let Some(h) = handler.as_ref().and_then(|h| h.as_ref()) else {
        if severity >= MLN_LOG_SEVERITY_WARNING {
            eprintln!("orbis: {message}");
        }
        return;
    };
    if h.platform & flag(severity) != 0 {
        eprintln!("orbis: {message}");
    }
    let (callback, user_data) = (h.callback, h.user_data);
    // A record reaches a callback without a queue at once, under the
    // handler's lock, as Native calls its observer.
    deliver_now(
        h.queue,
        Box::new(move || {
            // SAFETY: the caller's registration; the message lives until it
            // returns. Native's records carry no code, which it reports as
            // -1.
            crate::base::to_host(|| {
                in_callback(|| unsafe {
                    callback(
                        user_data.get(),
                        severity,
                        MLN_LOG_EVENT_GENERAL,
                        -1,
                        crate::base::lend(&mut message),
                    )
                })
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
            crate::base::watch_exit();
            not_reentered()?;
            let h = sized(handler, "handler")?;
            let callback = h
                .callback
                .ok_or_else(|| Failure::argument("handler.callback is null"))?;
            crate::delivery::check_queue(h.queue.0, "handler.queue")?;
            check_severities(h.platform_severities, "handler.platform_severities")?;
            let new = Handler {
                callback,
                user_data: UserData(h.user_data),
                release_user_data: h.release_user_data,
                queue: h.queue.0,
                platform: h.platform_severities,
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
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            check_severities(mask, "the severity mask")?;
            ASYNC.store(mask, Ordering::Relaxed);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The child's role in [`no_callback_runs_once_the_process_exits`].
    const EXIT_CHILD: &str = "ORB_TEST_EXIT_CHILD";

    unsafe extern "C" fn callback(
        _: *mut std::ffi::c_void,
        _: mln_log_severity,
        _: mln_log_event,
        _: i64,
        message: mln_buffer_view,
    ) {
        // SAFETY: the library's message, alive through the call.
        let message =
            unsafe { std::slice::from_raw_parts(message.data.cast::<u8>(), message.size) };
        eprintln!("callback: {}", String::from_utf8_lossy(message));
    }

    /// Once the process has begun to exit, the library calls none of the
    /// host's functions, whose state may already be gone: a record logged
    /// from an exit handler that runs after the library's never reaches
    /// the log callback. In a child process, which exits.
    #[test]
    fn no_callback_runs_once_the_process_exits() {
        if std::env::var_os(EXIT_CHILD).is_none() {
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "logging::tests::no_callback_runs_once_the_process_exits",
                    "--nocapture",
                ])
                .env(EXIT_CHILD, "1")
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&child.stderr);
            assert!(child.status.success(), "{stderr}");
            assert!(stderr.contains("callback: before exit"), "{stderr}");
            assert!(!stderr.contains("after exit"), "{stderr}");
            return;
        }
        unsafe extern "C" {
            fn atexit(handler: extern "C" fn()) -> std::ffi::c_int;
        }
        extern "C" fn after_the_library() {
            log(MLN_LOG_SEVERITY_ERROR, "after exit");
        }
        // Exit handlers run last registered first, so this one, registered
        // before the log callback is set, runs after the library's.
        // SAFETY: a plain function that lives as long as the process.
        unsafe { atexit(after_the_library) };
        let handler = mln_log_handler {
            size: std::mem::size_of::<mln_log_handler>() as u32,
            callback: Some(callback),
            user_data: std::ptr::null_mut(),
            release_user_data: None,
            queue: mln_delivery_queue(0),
            platform_severities: 0,
        };
        // SAFETY: a valid handler, and no diagnostic.
        let status = unsafe { mln_log_set_callback(&handler, std::ptr::null_mut()) };
        assert_eq!(status, MLN_STATUS_OK);
        log(MLN_LOG_SEVERITY_ERROR, "before exit");
        std::process::exit(0);
    }
}

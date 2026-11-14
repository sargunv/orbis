// MapLibre Native's network status, which is process-wide: every runtime's
// `orb_api::Runtime::set_online` follows it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::base::{Failure, boundary, put};
use crate::runtime::RuntimeHandle;
use crate::sys::*;

static ONLINE: AtomicBool = AtomicBool::new(true);
static RUNTIMES: Mutex<Vec<Weak<RuntimeHandle>>> = Mutex::new(Vec::new());

pub(crate) fn online() -> bool {
    ONLINE.load(Ordering::Acquire)
}

pub(crate) fn register(runtime: &Arc<RuntimeHandle>) {
    let mut runtimes = RUNTIMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    runtimes.retain(|r| r.strong_count() > 0);
    runtimes.push(Arc::downgrade(runtime));
}

pub(crate) fn unregister(runtime: &Arc<RuntimeHandle>) {
    RUNTIMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|r| r.strong_count() > 0 && !std::ptr::eq(r.as_ptr(), Arc::as_ptr(runtime)));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_network_get_status(
    out_status: *mut mln_network_status,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let status = if online() {
                MLN_NETWORK_STATUS_ONLINE
            } else {
                MLN_NETWORK_STATUS_OFFLINE
            };
            put(out_status, status, "out_status")
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_network_set_status(
    status: mln_network_status,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            let online = match status {
                MLN_NETWORK_STATUS_ONLINE => true,
                MLN_NETWORK_STATUS_OFFLINE => false,
                _ => {
                    return Err(Failure::argument(format!(
                        "{status} names no network status"
                    )));
                }
            };
            ONLINE.store(online, Ordering::Release);
            let runtimes: Vec<_> = RUNTIMES
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
                .filter_map(Weak::upgrade)
                .collect();
            for runtime in runtimes {
                let _ = runtime.post(move |state| {
                    state.runtime().set_online(online);
                    if online {
                        for cell in state.maps.values() {
                            cell.lock().map.retry_failed_tiles();
                        }
                    }
                });
            }
            Ok(())
        })
    }
}

/// Gives fetching the app's JVM and `Context` (`orb_host_native::android::init`):
/// HTTPS then verifies servers with Android's verifier (its Kotlin half,
/// `org.rustls:rustls-platform-verifier`, in the app; before, or without
/// it, the system store's certificate files), plain HTTP follows the app's
/// network security config, and `file:///android_asset/` URLs read the
/// app's assets. Elsewhere there is nothing to give.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_android_init(
    jni_env: *mut std::ffi::c_void,
    context: *mut std::ffi::c_void,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if jni_env.is_null() || context.is_null() {
                return Err(Failure::argument("jni_env and context must not be null"));
            }
            #[cfg(target_os = "android")]
            return orb_api::init_android(jni_env, context).map_err(Failure::native);
            #[cfg(not(target_os = "android"))]
            Err(Failure::unsupported(
                "Android initialization is not supported",
            ))
        })
    }
}

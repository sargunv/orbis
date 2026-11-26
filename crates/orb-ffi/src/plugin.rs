//! MapLibre Native's plugin registration (`mln/plugin/plugin_api.h`), which
//! the library exports beside the C API's own functions. Orbis takes no
//! plugins yet.

use std::ffi::c_char;

use crate::sys::*;

/// `mln_plugin_register_v1`: registers nothing, as this library takes no
/// plugin ABI version yet, and writes why into `error_message`,
/// NUL-terminated within `error_message_capacity` bytes, when it is
/// non-null with room.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_plugin_register_v1(
    _descriptor: *const mln_plugin_descriptor_v1,
    error_message: *mut c_char,
    error_message_capacity: usize,
) -> mln_plugin_status {
    const MESSAGE: &[u8] = b"Orbis takes no plugins";
    if !error_message.is_null() && error_message_capacity > 0 {
        let len = MESSAGE.len().min(error_message_capacity - 1);
        // SAFETY: the caller gives `error_message_capacity` writable bytes,
        // and `len + 1` fits in them.
        unsafe {
            std::ptr::copy_nonoverlapping(MESSAGE.as_ptr(), error_message.cast::<u8>(), len);
            *error_message.add(len) = 0;
        }
    }
    MLN_PLUGIN_STATUS_UNSUPPORTED_ABI
}

/// `mln_plugin_get_register_function_v1`: [`mln_plugin_register_v1`], which
/// stays valid for the life of the process, as the header promises.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_plugin_get_register_function_v1() -> mln_plugin_register_function_v1 {
    Some(mln_plugin_register_v1)
}

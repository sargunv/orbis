//! The C API's types and declarations. `generated.rs` is the Rust -sys
//! crate's file at the pinned mln-ffi commit, which mln-ffi's bindgen makes
//! from its headers (and whose layouts its tests check against the C
//! compiler's); `cargo xtask mln-ffi declarations` copies it, and `cargo
//! xtask check` checks it against the pin. C enums are plain integers
//! there, so an unknown value from a caller is never undefined behaviour.

#![allow(
    dead_code,
    missing_debug_implementations,
    non_camel_case_types,
    non_upper_case_globals,
    unreachable_pub,
    clippy::all
)]

// The -sys crate implements its handle trait with this macro; nothing here
// needs it.
macro_rules! native_handles {
    ($($handle:ident),* $(,)?) => {};
}

mod generated;

pub use generated::*;

/// Upstream's `mln/plugin/plugin_api.h` declares the plugin descriptor,
/// which the -sys crate leaves opaque.
#[repr(C)]
pub struct mln_plugin_descriptor_v1 {
    _opaque: [u8; 0],
}

pub type mln_plugin_status = std::ffi::c_uint;

pub type mln_plugin_register_function_v1 = Option<
    unsafe extern "C" fn(
        descriptor: *const mln_plugin_descriptor_v1,
        error_message: *mut std::ffi::c_char,
        error_message_capacity: usize,
    ) -> mln_plugin_status,
>;

/// `Default` as all-zero fields, for plain C data the library fills.
macro_rules! zeroed_default {
    ($($ty:ty),* $(,)?) => {$(
        impl Default for $ty {
            fn default() -> Self {
                // SAFETY: plain C data, for which all zeros is a valid value.
                unsafe { std::mem::zeroed() }
            }
        }
    )*};
}
zeroed_default!(
    mln_map_snapshot,
    mln_map_view,
    mln_camera,
    mln_camera_options,
    mln_render_frame_result,
    mln_render_session_snapshot,
    mln_render_session_capabilities,
    mln_runtime_event,
    mln_texture_readback_result,
);

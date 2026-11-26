//! The C API's types and declarations. `generated.rs` is bindgen's
//! reading of the pinned headers (`cargo xtask mln-ffi declarations`
//! regenerates it, and `cargo xtask check` checks that it names the pin).
//! C enums are plain integers there, so an unknown value from a caller is
//! never undefined behaviour, and each kind of handle is a type of its own.

#![allow(
    dead_code,
    missing_debug_implementations,
    non_camel_case_types,
    non_upper_case_globals,
    unreachable_pub,
    clippy::all
)]

mod generated;

pub use generated::*;

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

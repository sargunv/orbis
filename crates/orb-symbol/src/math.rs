//! The trigonometry layout and placement use: the same code natively and
//! in wasm, so that both lay labels out alike. Natively that is `libm`;
//! wasm's `std` already calls compiler-builtins' copy of it, which a
//! second copy would only duplicate.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use libm::{atan, atan2, cos, hypot, sin, sqrt};

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm::{atan, atan2, cos, hypot, sin, sqrt};

#[cfg(target_arch = "wasm32")]
mod wasm {
    pub(crate) fn atan(x: f64) -> f64 {
        x.atan()
    }

    pub(crate) fn atan2(y: f64, x: f64) -> f64 {
        y.atan2(x)
    }

    pub(crate) fn cos(x: f64) -> f64 {
        x.cos()
    }

    pub(crate) fn hypot(x: f64, y: f64) -> f64 {
        x.hypot(y)
    }

    pub(crate) fn sin(x: f64) -> f64 {
        x.sin()
    }

    pub(crate) fn sqrt(x: f64) -> f64 {
        x.sqrt()
    }
}

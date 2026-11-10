//! The web backend as a cfg: `webgpu` (browser WebGPU, core.wasm) or
//! `webgl` (WebGL 2, core-gl.wasm), from the feature that picks it. Code
//! for the web either way stays under `target_arch = "wasm32"`.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(webgpu, webgl)");
    if std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|a| a == "wasm32") {
        // Both, or neither, is a compile error in lib.rs.
        if std::env::var_os("CARGO_FEATURE_WEBGPU").is_some() {
            println!("cargo::rustc-cfg=webgpu");
        }
        if std::env::var_os("CARGO_FEATURE_GL").is_some() {
            println!("cargo::rustc-cfg=webgl");
        }
    }
}

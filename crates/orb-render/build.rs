//! `cpu_driven`: the CPU frame driver and the tables' texture backing,
//! for devices without compute-driven draws (WebGL 2, #140). Natively
//! always, for the tests that hold it to the GPU driver; on the web only
//! in the WebGL 2 build (`cpu-driven`), so `core.wasm` carries none of it.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cpu_driven)");
    let wasm = std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "wasm32");
    if !wasm || std::env::var_os("CARGO_FEATURE_CPU_DRIVEN").is_some() {
        println!("cargo::rustc-cfg=cpu_driven");
    }
}

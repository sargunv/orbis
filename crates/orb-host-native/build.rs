//! `cfg(ureq_transport)`: HTTP goes through ureq, everywhere but Android
//! and there with the `standalone` feature.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ureq_transport)");
    let android = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "android");
    if !android || std::env::var_os("CARGO_FEATURE_STANDALONE").is_some() {
        println!("cargo::rustc-cfg=ureq_transport");
    }
}

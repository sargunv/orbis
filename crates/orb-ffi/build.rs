//! Names the shared library as mln-ffi's bindings look for it.

fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match os.as_str() {
        "macos" | "ios" => println!(
            "cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libmaplibre-native-c.dylib"
        ),
        "linux" | "android" => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libmaplibre-native-c.so")
        }
        _ => {}
    }
}

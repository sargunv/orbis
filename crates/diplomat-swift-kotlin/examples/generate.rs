//! `cargo run -p diplomat-swift-kotlin --example generate -- <lib.rs> <out dir> <package> <library>`
//! writes Kotlin and Swift bindings for a bridge, for a look at the output.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [entry, out, package, library] = &args[..] else {
        eprintln!("usage: generate <lib.rs> <out dir> <package> <library>");
        std::process::exit(2);
    };
    let entry = std::path::Path::new(entry);
    let kotlin = diplomat_swift_kotlin::kotlin(
        entry,
        &diplomat_swift_kotlin::KotlinConfig {
            package: package.clone(),
            library: library.clone(),
            native_object: "Native".into(),
            c_headers: vec!["bridge.h".into()],
        },
    );
    let swift = diplomat_swift_kotlin::swift(
        entry,
        &diplomat_swift_kotlin::SwiftConfig {
            c_module: format!("{library}C"),
        },
    );
    for files in [kotlin, swift] {
        match files {
            Ok(files) => {
                for (path, text) in files {
                    let path = std::path::Path::new(out).join(path);
                    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                    std::fs::write(path, text).unwrap();
                }
            }
            Err(errors) => {
                for e in errors {
                    eprintln!("error: {e}");
                }
                std::process::exit(1);
            }
        }
    }
}

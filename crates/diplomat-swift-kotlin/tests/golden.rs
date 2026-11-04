//! The fixture's bindings match `tests/golden`. `UPDATE_GOLDEN=1 cargo test
//! -p diplomat-swift-kotlin --test golden` rewrites them.

mod support;

use std::path::Path;

#[test]
fn fixture_bindings_match_the_golden_files() {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let files = support::generate();
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        let _ = std::fs::remove_dir_all(&golden);
        for (path, text) in &files {
            let path = golden.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        return;
    }
    let mut on_disk = Vec::new();
    collect(&golden, &golden, &mut on_disk);
    on_disk.sort();
    let generated: Vec<&String> = files.keys().collect();
    assert_eq!(
        on_disk.iter().collect::<Vec<_>>(),
        generated,
        "golden file set differs; UPDATE_GOLDEN=1 rewrites it"
    );
    for (path, text) in &files {
        let expected = std::fs::read_to_string(golden.join(path)).unwrap();
        assert!(
            &expected == text,
            "{path} differs from its golden file; UPDATE_GOLDEN=1 rewrites it"
        );
    }
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(root, &path, out);
        } else {
            out.push(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

#[test]
fn unsupported_shapes_are_errors() {
    let dir = std::env::temp_dir().join(format!("dsk-unsupported-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let entry = dir.join("lib.rs");
    std::fs::write(
        &entry,
        r#"
#[diplomat::bridge]
pub mod ffi {
    #[diplomat::opaque]
    pub struct Thing;
    impl Thing {
        #[diplomat::attr(contract, sometimes)]
        pub fn odd(&self) -> u8 { 0 }
        pub fn many(&self, f: impl Fn(u32) -> u32) {}
    }
}
"#,
    )
    .unwrap();
    let errors = diplomat_swift_kotlin::kotlin(&entry, &support::kotlin_config()).unwrap_err();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("unknown contract `sometimes`")),
        "{errors:?}"
    );
}

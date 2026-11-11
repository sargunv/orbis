//! What the integration tests share: the demo's Tokyo extract and drawing
//! until idle.

// Each test binary uses some of these.
#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Duration;

use orb_api::{Map, Target};

/// The demo's data directory: the Tokyo extract, its glyphs and sprites.
pub(crate) fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/demo/data")
        .canonicalize()
        .unwrap()
}

/// The demo's Tokyo extract with Liberty's style, as a URL its relative
/// URLs resolve against.
pub(crate) fn tokyo_url() -> String {
    format!("file://{}", data().join("tokyo-mvt.json").display())
}

/// The Tokyo extract's style as text, its relative URLs made absolute: for
/// tests that change it before setting it.
pub(crate) fn tokyo_json() -> String {
    let data = data();
    std::fs::read_to_string(data.join("tokyo-mvt.json"))
        .unwrap()
        .replace(
            "pmtiles://./",
            &format!("pmtiles://file://{}/", data.display()),
        )
        .replace("\"./", &format!("\"file://{}/", data.display()))
}

/// Draws until the map is loaded and idle.
pub(crate) fn settle(map: &mut Map, target: &mut Target) {
    map.render_until_idle(target, Duration::from_secs(60))
        .expect("the map idles");
}

/// A directory of a test's own under the system's temporary one, removed
/// with what it holds when it drops, as a failing test unwinds too.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(name: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("orb-{name}-{}-{n}", std::process::id()));
        // What a killed run with this process id left.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl std::ops::Deref for Scratch {
    type Target = std::path::Path;
    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl AsRef<std::path::Path> for Scratch {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

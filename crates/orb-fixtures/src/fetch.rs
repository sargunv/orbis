//! Fixture URLs: GL JS's test server's assets, or MapLibre Native's
//! render-test cache, as files.

use std::path::{Component, Path, PathBuf};

/// Where a suite's URLs come from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Assets {
    /// GL JS's test server: `local://` and its origin name the assets;
    /// other URLs go to the network.
    GlJs,
    /// Native's runner, offline: every request comes from its cache, whose
    /// `http://<path>` rows (the runner's spelling of `local://` and GL JS's
    /// origin) are at `<path>` and other schemes' under the scheme's name
    /// (`fixtures/native/SOURCE.md`). Nothing goes to the network.
    Native,
}

/// The URL to fetch for `url`: `local://` and the test server's origin
/// name the assets in `assets`, including URLs inside TileJSON; `None`
/// refuses paths that leave them.
pub(crate) fn resolve(assets: &Path, url: &str, kind: Assets) -> Option<String> {
    // GL JS's test server serves the assets at this origin; TileJSON in
    // the assets names it.
    let local = url
        .strip_prefix("local://")
        .or_else(|| url.strip_prefix("http://localhost:2900/"));
    let (assets, relative) = match (local, kind) {
        (Some(relative), _) => (assets.to_path_buf(), relative),
        (None, Assets::GlJs) => return Some(url.into()),
        (None, Assets::Native) => {
            if url.starts_with("data:") {
                return Some(url.into());
            }
            let (scheme, rest) = url.split_once("://")?;
            let dir: PathBuf = match scheme {
                "http" => assets.to_path_buf(),
                "https" | "mapbox" => assets.join(scheme),
                _ => return None,
            };
            (dir, rest)
        }
    };
    // HTTP's file server ignores query/fragment and decodes percent escapes.
    let relative = relative.split(['?', '#']).next().unwrap_or_default();
    let mut decoded = Vec::new();
    let mut bytes = relative.bytes();
    while let Some(byte) = bytes.next() {
        decoded.push(if byte == b'%' {
            let hi = bytes.next().and_then(|c| (c as char).to_digit(16));
            let lo = bytes.next().and_then(|c| (c as char).to_digit(16));
            match (hi, lo) {
                (Some(hi), Some(lo)) => (hi * 16 + lo) as u8,
                _ => return None,
            }
        } else {
            byte
        });
    }
    let decoded = String::from_utf8(decoded).ok()?;
    let path = Path::new(&decoded);
    if path
        .components()
        .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
        || decoded.contains('\0')
    {
        return None;
    }
    Some(format!("file://{}", assets.join(path).display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_assets_queries_and_tilejson_templates() {
        let assets = Path::new("/fixtures/assets");
        assert_eq!(
            resolve(assets, "local://tiles/0-0-0.mvt?cache=1", Assets::GlJs).unwrap(),
            "file:///fixtures/assets/tiles/0-0-0.mvt"
        );
        assert_eq!(
            resolve(assets, "local://glyphs/Open%20Sans/0-255.pbf", Assets::GlJs).unwrap(),
            "file:///fixtures/assets/glyphs/Open Sans/0-255.pbf"
        );
        assert_eq!(
            resolve(assets, "https://example.org/a", Assets::GlJs).unwrap(),
            "https://example.org/a"
        );
        assert_eq!(
            resolve(assets, "https://example.org/a%2cb", Assets::Native).unwrap(),
            "file:///fixtures/assets/https/example.org/a,b"
        );
        assert_eq!(
            resolve(assets, "http://tiles/1.png", Assets::Native).unwrap(),
            "file:///fixtures/assets/tiles/1.png"
        );
        assert!(resolve(assets, "file:///etc/passwd", Assets::Native).is_none());
        for url in [
            "local://../secret",
            "local://%2e%2e/secret",
            "local:///absolute",
            "local://bad%0",
        ] {
            assert!(resolve(assets, url, Assets::GlJs).is_none(), "{url}");
            assert!(resolve(assets, url, Assets::Native).is_none(), "{url}");
        }
    }
}

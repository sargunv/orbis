//! Fixture URLs: GL JS's test server's assets, as files.

use std::path::{Component, Path};

/// The URL to fetch for `url`: `local://` and the test server's origin
/// name the assets in `assets`, including URLs inside TileJSON; `None`
/// refuses paths that leave them.
pub(crate) fn resolve(assets: &Path, url: &str) -> Option<String> {
    // GL JS's test server serves the assets at this origin; TileJSON in
    // the assets names it.
    let Some(relative) = url
        .strip_prefix("local://")
        .or_else(|| url.strip_prefix("http://localhost:2900/"))
    else {
        return Some(url.into());
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
            resolve(assets, "local://tiles/0-0-0.mvt?cache=1").unwrap(),
            "file:///fixtures/assets/tiles/0-0-0.mvt"
        );
        assert_eq!(
            resolve(assets, "local://glyphs/Open%20Sans/0-255.pbf").unwrap(),
            "file:///fixtures/assets/glyphs/Open Sans/0-255.pbf"
        );
        assert_eq!(
            resolve(assets, "https://example.org/a").unwrap(),
            "https://example.org/a"
        );
        for url in [
            "local://../secret",
            "local://%2e%2e/secret",
            "local:///absolute",
            "local://bad%0",
        ] {
            assert!(resolve(assets, url).is_none(), "{url}");
        }
    }
}

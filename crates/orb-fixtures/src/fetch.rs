//! Resolve local:// at the Fetch boundary, including URLs inside TileJSON.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use orb_host::{Fetch, FetchDone, FetchError};
use orb_host_native::NativeFetch;

#[derive(Debug)]
pub(crate) struct FixtureFetch {
    assets: PathBuf,
    native: Arc<NativeFetch>,
}

impl FixtureFetch {
    pub(crate) fn new(assets: PathBuf) -> Self {
        Self {
            assets,
            native: NativeFetch::new(),
        }
    }

    fn resolve(&self, url: &str) -> Result<String, FetchError> {
        // GL JS's test server serves the assets at this origin; TileJSON in
        // the assets names it.
        let Some(relative) = url
            .strip_prefix("local://")
            .or_else(|| url.strip_prefix("http://localhost:2900/"))
        else {
            return Ok(url.into());
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
                    _ => return Err(FetchError::UnsupportedUrl(url.into())),
                }
            } else {
                byte
            });
        }
        let decoded =
            String::from_utf8(decoded).map_err(|_| FetchError::UnsupportedUrl(url.into()))?;
        let path = Path::new(&decoded);
        if path
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
            || decoded.contains('\0')
        {
            return Err(FetchError::UnsupportedUrl(url.into()));
        }
        Ok(format!("file://{}", self.assets.join(path).display()))
    }
}

impl Fetch for FixtureFetch {
    fn get_range(&self, url: &str, offset: u64, length: u64, done: FetchDone) {
        match self.resolve(url) {
            Ok(url) => self.native.get_range(&url, offset, length, done),
            Err(error) => done(Err(error)),
        }
    }

    fn get(&self, url: &str, done: FetchDone) {
        match self.resolve(url) {
            Ok(url) => self.native.get(&url, done),
            Err(error) => done(Err(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_local_urls_returned_inside_tilejson() {
        let dir = std::env::temp_dir().join(format!("orb-fixtures-fetch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("source.json"),
            br#"{"tiles":["local://0-0-0.mvt"]}"#,
        )
        .unwrap();
        std::fs::write(dir.join("0-0-0.mvt"), b"tile bytes").unwrap();
        let fetch = FixtureFetch::new(dir.clone());
        let get = |url: &str| {
            let (tx, rx) = std::sync::mpsc::channel();
            fetch.get(url, Box::new(move |r| tx.send(r).unwrap()));
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap()
                .unwrap()
        };
        let source: serde_json::Value =
            serde_json::from_slice(&get("local://source.json")).unwrap();
        assert_eq!(get(source["tiles"][0].as_str().unwrap()), b"tile bytes");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn local_assets_queries_and_tilejson_templates() {
        let fetch = FixtureFetch::new(PathBuf::from("/fixtures/assets"));
        assert_eq!(
            fetch.resolve("local://tiles/0-0-0.mvt?cache=1").unwrap(),
            "file:///fixtures/assets/tiles/0-0-0.mvt"
        );
        assert_eq!(
            fetch
                .resolve("local://glyphs/Open%20Sans/0-255.pbf")
                .unwrap(),
            "file:///fixtures/assets/glyphs/Open Sans/0-255.pbf"
        );
        assert_eq!(
            fetch.resolve("https://example.org/a").unwrap(),
            "https://example.org/a"
        );
        for url in [
            "local://../secret",
            "local://%2e%2e/secret",
            "local:///absolute",
            "local://bad%0",
        ] {
            assert!(fetch.resolve(url).is_err(), "{url}");
        }
    }
}

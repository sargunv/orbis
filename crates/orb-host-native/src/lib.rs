//! Host services on native platforms.

#![forbid(unsafe_code)]

mod fonts;

use std::sync::Arc;

pub use fonts::SystemFonts;
pub use orb_host::FontFileFaces;

use orb_host::{Fetch, FetchDone, FetchError, ImageDecode, ImageDone};
use orb_jobs::{Executor, ThreadPool};

/// Fetches `http(s)://` URLs with ureq and `file://` URLs from disk, on a
/// small pool of I/O threads.
#[derive(Debug)]
pub struct NativeFetch {
    agent: ureq::Agent,
    pool: ThreadPool,
}

/// Largest response body accepted.
const MAX_BODY: u64 = 64 << 20;

impl NativeFetch {
    pub fn new() -> Arc<Self> {
        let agent = ureq::Agent::config_builder()
            .user_agent(concat!("orbis/", env!("CARGO_PKG_VERSION")))
            .build()
            .new_agent();
        Arc::new(Self {
            agent,
            pool: ThreadPool::new("fetch", 8),
        })
    }

    fn get_blocking(
        agent: &ureq::Agent,
        url: &str,
        range: Option<(u64, u64)>,
    ) -> Result<Vec<u8>, FetchError> {
        if let Some(path) = url.strip_prefix("file://") {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(path).map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => FetchError::NotFound,
                _ => FetchError::Network(e.to_string()),
            })?;
            let mut bytes = Vec::new();
            let limit = if let Some((offset, length)) = range {
                orb_host::range_end(offset, length)?;
                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| FetchError::Network(e.to_string()))?;
                length
            } else {
                MAX_BODY
            };
            file.take(limit + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| FetchError::Network(e.to_string()))?;
            if range.is_some() {
                bytes.truncate(limit as usize);
            }
            if (range.is_some() && bytes.len() as u64 != limit) || bytes.len() as u64 > limit {
                return Err(FetchError::Range("short or oversized file".into()));
            }
            return Ok(bytes);
        }
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(FetchError::UnsupportedUrl(url.to_string()));
        }
        let mut request = agent.get(encode_url(url).as_ref());
        if let Some((offset, length)) = range {
            let end = orb_host::range_end(offset, length)?;
            request = request
                .header("Range", format!("bytes={offset}-{end}"))
                .header("Accept-Encoding", "identity");
        }
        let mut response = request.call().map_err(|e| match e {
            ureq::Error::StatusCode(404) => FetchError::NotFound,
            ureq::Error::StatusCode(code) => FetchError::Status(code),
            e => FetchError::Network(e.to_string()),
        })?;
        if response.status() == 204 {
            return Err(FetchError::NotFound);
        }
        if let Some((offset, length)) = range {
            orb_host::check_range(
                response.status().as_u16(),
                response
                    .headers()
                    .get("content-range")
                    .and_then(|h| h.to_str().ok()),
                offset,
                length,
            )?;
        }
        let bytes = response
            .body_mut()
            .with_config()
            .limit(range.map_or(MAX_BODY, |(_, length)| length + 1))
            .read_to_vec()
            .map_err(|e| FetchError::Network(e.to_string()))?;
        if range.is_some_and(|(_, length)| bytes.len() as u64 != length) {
            return Err(FetchError::Range("short range body".into()));
        }
        Ok(bytes)
    }
}

impl Fetch for NativeFetch {
    fn get(&self, url: &str, done: FetchDone) {
        let agent = self.agent.clone();
        let url = url.to_string();
        self.pool.spawn(Box::new(move || {
            done(Self::get_blocking(&agent, &url, None))
        }));
    }
    fn get_range(&self, url: &str, offset: u64, length: u64, done: FetchDone) {
        let agent = self.agent.clone();
        let url = url.to_string();
        self.pool.spawn(Box::new(move || {
            done(Self::get_blocking(&agent, &url, Some((offset, length))))
        }));
    }
}

/// Decodes images with the core's decoder ([`orb_host::decode_image`]),
/// JPEG included here, on the caller's thread (tile and sprite decodes
/// already run on job threads).
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeImageDecode;

impl ImageDecode for NativeImageDecode {
    fn decode(&self, bytes: Vec<u8>, done: ImageDone) {
        done(orb_host::decode_image(&bytes));
    }
}

/// Percent-encodes what a URI may not contain, as a browser's URL parser
/// does before fetching: styles write glyph URLs with spaces in font names
/// (`.../fonts/Noto Sans Regular/0-255.pbf`).
fn encode_url(url: &str) -> std::borrow::Cow<'_, str> {
    let invalid = |b: u8| {
        b <= b' '
            || b >= 0x7f
            || matches!(
                b,
                b'"' | b'<' | b'>' | b'\\' | b'^' | b'`' | b'{' | b'|' | b'}'
            )
    };
    if !url.bytes().any(invalid) {
        return url.into();
    }
    let mut out = String::with_capacity(url.len() + 16);
    for b in url.bytes() {
        if invalid(b) {
            out.push_str(&format!("%{b:02X}"));
        } else {
            out.push(b as char);
        }
    }
    out.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_are_encoded_as_browsers_do() {
        assert_eq!(
            encode_url("https://a.org/fonts/Noto Sans Regular,Ünï/0-255.pbf"),
            "https://a.org/fonts/Noto%20Sans%20Regular,%C3%9Cn%C3%AF/0-255.pbf"
        );
        let plain = "https://a.org/t/1/2/3.pbf?k=v%20";
        assert!(matches!(encode_url(plain), std::borrow::Cow::Borrowed(p) if p == plain));
    }

    #[test]
    fn decodes_png_to_straight_rgba() {
        // A 1×1 PNG with one half-transparent red pixel.
        use image::ImageEncoder;
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&[255, 0, 0, 128], 1, 1, image::ExtendedColorType::Rgba8)
            .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        NativeImageDecode.decode(png, Box::new(move |r| tx.send(r).unwrap()));
        let image = rx.recv().unwrap().unwrap();
        assert_eq!(
            (image.width, image.height, image.pixels),
            (1, 1, vec![255, 0, 0, 128])
        );
        let (tx, rx) = std::sync::mpsc::channel();
        NativeImageDecode.decode(
            b"not an image".to_vec(),
            Box::new(move |r| tx.send(r).unwrap()),
        );
        assert!(rx.recv().unwrap().is_err());
    }
    use std::sync::mpsc::channel;

    #[test]
    fn reads_files_and_reports_missing_ones() {
        let fetch = NativeFetch::new();
        let path = std::env::temp_dir().join(format!("orb-host-native-{}", std::process::id()));
        std::fs::write(&path, b"hello").unwrap();
        let (tx, rx) = channel();
        for url in [
            format!("file://{}", path.display()),
            "file:///nonexistent/orbis".into(),
            "ftp://x".into(),
        ] {
            let tx = tx.clone();
            fetch.get(&url.clone(), Box::new(move |r| tx.send((url, r)).unwrap()));
        }
        let mut results: Vec<_> = (0..3).map(|_| rx.recv().unwrap()).collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        std::fs::remove_file(&path).unwrap();
        assert_eq!(results[0].1, Err(FetchError::NotFound));
        assert_eq!(results[1].1, Ok(b"hello".to_vec()));
        assert!(matches!(results[2].1, Err(FetchError::UnsupportedUrl(_))));
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn response(status: &str, headers: &str, body: &[u8]) -> Result<Vec<u8>, FetchError> {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/archive", server.local_addr().unwrap());
        let reply = format!("HTTP/1.1 {status}\r\n{headers}\r\nConnection: close\r\n\r\n");
        let body = body.to_vec();
        let thread = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut request = [0; 2048];
            let n = stream.read(&mut request).unwrap();
            assert!(
                String::from_utf8_lossy(&request[..n])
                    .to_lowercase()
                    .contains("range: bytes=2-4")
            );
            stream.write_all(reply.as_bytes()).unwrap();
            let _ = stream.write_all(&body);
        });
        let result =
            NativeFetch::get_blocking(&ureq::Agent::new_with_defaults(), &url, Some((2, 3)));
        thread.join().unwrap();
        result
    }

    #[test]
    fn range_semantics_and_bounded_bodies() {
        assert_eq!(
            response(
                "206 Partial Content",
                "Content-Range: bytes 2-4/6\r\nContent-Length: 3",
                b"cde"
            )
            .unwrap(),
            b"cde"
        );
        // Advertise an 83 GB response without sending any body. It must be
        // rejected on headers, rather than attempting to read it.
        assert!(matches!(
            response("200 OK", "Content-Length: 83000000000", b""),
            Err(FetchError::Range(_))
        ));
        assert!(
            response(
                "206 Partial Content",
                "Content-Range: bytes 0-2/6\r\nContent-Length: 3",
                b"abc"
            )
            .is_err()
        );
        assert!(response("206 Partial Content", "Content-Length: 3", b"abc").is_err());
        assert!(response("206 Partial Content", "Content-Range: bytes 2-4/6", b"cdef").is_err());
        assert!(response("206 Partial Content", "Content-Range: bytes 2-4/6", b"cd").is_err());
    }

    #[test]
    fn file_ranges_and_invalid_lengths() {
        let path = std::env::temp_dir().join(format!("orb-range-{}", std::process::id()));
        std::fs::write(&path, b"abcdef").unwrap();
        let url = format!("file://{}", path.display());
        let agent = ureq::Agent::new_with_defaults();
        assert_eq!(
            NativeFetch::get_blocking(&agent, &url, Some((2, 3))).unwrap(),
            b"cde"
        );
        for range in [(4, 3), (0, 0), (0, MAX_BODY + 1), (u64::MAX, 2)] {
            assert!(NativeFetch::get_blocking(&agent, &url, Some(range)).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }
}

/// A monotonic animation clock, local to a runtime.
#[derive(Debug)]
pub struct NativeClock(std::time::Instant);

impl Default for NativeClock {
    fn default() -> Self {
        Self(std::time::Instant::now())
    }
}

impl orb_host::Clock for NativeClock {
    fn now(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1000.0
    }
}

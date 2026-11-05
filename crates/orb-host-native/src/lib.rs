//! Host services on native platforms.

#![forbid(unsafe_code)]

mod cache;
mod fonts;
mod storage;

use std::sync::Arc;

pub use cache::{CacheStats, HttpCache};
pub use fonts::SystemFonts;
pub use orb_host::FontFileFaces;
pub use storage::{FileStorage, wait};

use orb_host::{Fetch, FetchDone, FetchError, ImageDecode, ImageDone, Request, RequestTransform};
use orb_jobs::{Executor, ThreadPool};

/// Fetches `http(s)://` URLs with ureq and `file://` URLs from disk, on a
/// small pool of I/O threads, through an [`HttpCache`] when it has one.
pub struct NativeFetch {
    agent: ureq::Agent,
    pool: ThreadPool,
    cache: Option<Arc<HttpCache>>,
    transform: std::sync::RwLock<Option<Arc<RequestTransform>>>,
    /// Off, requests answer from the cache alone.
    online: Arc<std::sync::atomic::AtomicBool>,
}

impl std::fmt::Debug for NativeFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeFetch")
            .field("cache", &self.cache)
            .field("online", &self.online)
            .finish_non_exhaustive()
    }
}

/// Largest response body accepted.
const MAX_BODY: u64 = 64 << 20;

/// What the network answered.
enum Response {
    Body {
        bytes: Vec<u8>,
        /// `None` when the response may not be stored.
        fresh_until: Option<u64>,
        etag: Option<String>,
        last_modified: Option<String>,
    },
    NotModified {
        fresh_until: Option<u64>,
    },
}

impl NativeFetch {
    pub fn new() -> Arc<Self> {
        Self::with_cache(None)
    }

    /// Fetches through `cache`: fresh entries without the network, stale
    /// ones revalidated, and either when the network fails.
    pub fn with_cache(cache: Option<Arc<HttpCache>>) -> Arc<Self> {
        let agent = ureq::Agent::config_builder()
            .user_agent(concat!("orbis/", env!("CARGO_PKG_VERSION")))
            .build()
            .new_agent();
        Arc::new(Self {
            agent,
            pool: ThreadPool::new("fetch", 8),
            cache,
            transform: Default::default(),
            online: Arc::new(true.into()),
        })
    }

    /// Rewrites every request from here on: its URL, before the cache, and
    /// headers to send (`None` for none).
    pub fn set_transform(&self, transform: Option<Arc<RequestTransform>>) {
        *self.transform.write().expect("transform poisoned") = transform;
    }

    /// Off, nothing goes to the network: requests answer from the cache,
    /// stale or not, and fail without it. `file://` URLs still read.
    pub fn set_online(&self, online: bool) {
        self.online
            .store(online, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn online(&self) -> bool {
        self.online.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The cache fetches go through.
    pub fn cache(&self) -> Option<&Arc<HttpCache>> {
        self.cache.as_ref()
    }

    fn get_blocking(
        agent: &ureq::Agent,
        cache: Option<&HttpCache>,
        request: &Request,
        range: Option<(u64, u64)>,
        online: bool,
    ) -> Result<Vec<u8>, FetchError> {
        let url = request.url.as_str();
        if let Some(path) = url.strip_prefix("file://") {
            return read_file(path, range);
        }
        if !orb_host::cache::cacheable(url) {
            return Err(FetchError::UnsupportedUrl(url.to_string()));
        }
        let http = |validators| http(agent, request, range, validators);
        if !online {
            let cached = cache.and_then(|c| c.lookup(&orb_host::cache::key(url, range)));
            return cached
                .map(|c| c.bytes)
                .ok_or_else(|| FetchError::Network("offline".into()));
        }
        let Some(cache) = cache else {
            return match http(None)? {
                Response::Body { bytes, .. } => Ok(bytes),
                Response::NotModified { .. } => Err(FetchError::Status(304)),
            };
        };
        let key = orb_host::cache::key(url, range);
        let cached = cache.lookup(&key);
        if let Some(cached) = cached.as_ref().filter(|c| c.fresh) {
            return Ok(cached.bytes.clone());
        }
        let validators = cached
            .as_ref()
            .map(|c| (c.etag.as_deref(), c.last_modified.as_deref()));
        match (http(validators), cached) {
            (
                Ok(Response::Body {
                    bytes,
                    fresh_until,
                    etag,
                    last_modified,
                }),
                _,
            ) => {
                match fresh_until {
                    Some(until) => cache.store(
                        &key,
                        &bytes,
                        until,
                        etag.as_deref(),
                        last_modified.as_deref(),
                    ),
                    None => cache.remove(&key),
                }
                Ok(bytes)
            }
            (Ok(Response::NotModified { fresh_until }), Some(cached)) => {
                cache.refresh(&key, fresh_until.unwrap_or(0));
                Ok(cached.bytes)
            }
            (Ok(Response::NotModified { .. }), None) => Err(FetchError::Status(304)),
            // Offline, or the server failing: the stale copy.
            (Err(FetchError::Network(_) | FetchError::Status(500..)), Some(cached)) => {
                Ok(cached.bytes)
            }
            (Err(error), _) => {
                cache.remove(&key);
                Err(error)
            }
        }
    }
}

fn read_file(path: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>, FetchError> {
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
    Ok(bytes)
}

/// One HTTP GET, conditional on `validators` (ETag, Last-Modified).
fn http(
    agent: &ureq::Agent,
    req: &Request,
    range: Option<(u64, u64)>,
    validators: Option<(Option<&str>, Option<&str>)>,
) -> Result<Response, FetchError> {
    let mut request = agent.get(encode_url(&req.url).as_ref());
    for (name, value) in &req.headers {
        request = request.header(name, value);
    }
    if let Some((offset, length)) = range {
        let end = orb_host::range_end(offset, length)?;
        request = request
            .header("Range", format!("bytes={offset}-{end}"))
            .header("Accept-Encoding", "identity");
    }
    if let Some((etag, last_modified)) = validators {
        if let Some(etag) = etag {
            request = request.header("If-None-Match", etag);
        }
        if let Some(last_modified) = last_modified {
            request = request.header("If-Modified-Since", last_modified);
        }
    }
    let mut response = request.call().map_err(|e| match e {
        ureq::Error::StatusCode(404) => FetchError::NotFound,
        ureq::Error::StatusCode(code) => FetchError::Status(code),
        e => FetchError::Network(e.to_string()),
    })?;
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|h| h.to_str().ok())
            .map(str::to_owned)
    };
    let (cache_control, expires, date) =
        (header("cache-control"), header("expires"), header("date"));
    let (etag, last_modified) = (header("etag"), header("last-modified"));
    let fresh_until = orb_host::cache::fresh_until(
        orb_host::cache::Headers {
            cache_control: cache_control.as_deref(),
            expires: expires.as_deref(),
            date: date.as_deref(),
            last_modified: last_modified.as_deref(),
        },
        cache::now(),
    );
    match response.status().as_u16() {
        // No content: a valid, empty tile.
        204 => {
            return Ok(Response::Body {
                bytes: Vec::new(),
                fresh_until,
                etag,
                last_modified,
            });
        }
        304 => return Ok(Response::NotModified { fresh_until }),
        _ => {}
    }
    if let Some((offset, length)) = range {
        orb_host::check_range(
            response.status().as_u16(),
            header("content-range").as_deref(),
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
    Ok(Response::Body {
        bytes,
        fresh_until,
        etag,
        last_modified,
    })
}

impl Fetch for NativeFetch {
    fn get(&self, url: &str, done: FetchDone) {
        self.get_range_opt(url, None, done);
    }
    fn get_range(&self, url: &str, offset: u64, length: u64, done: FetchDone) {
        self.get_range_opt(url, Some((offset, length)), done);
    }
}

impl NativeFetch {
    fn get_range_opt(&self, url: &str, range: Option<(u64, u64)>, done: FetchDone) {
        let transform = self.transform.read().expect("transform poisoned").clone();
        let request = match transform {
            Some(transform) => match transform(url) {
                Some(request) => request,
                None => return done(Err(FetchError::UnsupportedUrl(format!("{url}: refused")))),
            },
            None => Request {
                url: url.to_string(),
                headers: Vec::new(),
            },
        };
        let (agent, cache, online) = (self.agent.clone(), self.cache.clone(), self.online());
        self.pool.spawn(Box::new(move || {
            done(Self::get_blocking(
                &agent,
                cache.as_deref(),
                &request,
                range,
                online,
            ))
        }));
    }
}

/// Writes `data` to a file of its own beside `path`, then renames it over
/// `path`: readers see the old file or the new one whole, and writers in
/// other runtimes or processes never share the temporary file.
pub fn write_atomically(path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(format!(".{}.{n}.partial", std::process::id()));
    let partial = path.with_file_name(name);
    let written = std::fs::write(&partial, data).and_then(|()| std::fs::rename(&partial, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    written
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

/// A request for `url` alone.
#[cfg(test)]
fn plain(url: &str) -> Request {
    Request {
        url: url.into(),
        headers: Vec::new(),
    }
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
        let result = NativeFetch::get_blocking(
            &ureq::Agent::new_with_defaults(),
            None,
            &plain(&url),
            Some((2, 3)),
            true,
        );
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
            NativeFetch::get_blocking(&agent, None, &plain(&url), Some((2, 3)), true).unwrap(),
            b"cde"
        );
        for range in [(4, 3), (0, 0), (0, MAX_BODY + 1), (u64::MAX, 2)] {
            assert!(
                NativeFetch::get_blocking(&agent, None, &plain(&url), Some(range), true).is_err()
            );
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

#[cfg(test)]
mod write_tests {
    #[test]
    fn concurrent_writers_publish_whole_files() {
        let dir = std::env::temp_dir().join(format!("orb-host-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache");
        let writers: Vec<_> = (0..8u8)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        super::write_atomically(&path, &vec![i; 64 << 10]).unwrap();
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        let data = std::fs::read(&path).unwrap();
        assert_eq!(data.len(), 64 << 10);
        assert!(data.iter().all(|&b| b == data[0]), "bytes of two writers");
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "partial files left"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Serves `replies` to one connection each, and returns the requests.
    fn serve(replies: Vec<String>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/tile", server.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            replies
                .into_iter()
                .map(|reply| {
                    let (mut stream, _) = server.accept().unwrap();
                    let mut request = [0; 2048];
                    let n = stream.read(&mut request).unwrap();
                    stream.write_all(reply.as_bytes()).unwrap();
                    String::from_utf8_lossy(&request[..n]).to_lowercase()
                })
                .collect()
        });
        (url, thread)
    }

    #[test]
    fn revalidates_stale_entries_and_answers_offline() {
        let dir = std::env::temp_dir().join(format!("orb-fetch-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = HttpCache::open(Arc::new(FileStorage::new(&dir)), 1 << 20).unwrap();
        let agent = ureq::Agent::new_with_defaults();
        let close = "Connection: close\r\n";
        let (url, server) = serve(vec![
            format!(
                "HTTP/1.1 200 OK\r\n{close}Cache-Control: no-cache\r\nETag: \"v1\"\r\nContent-Length: 4\r\n\r\ntile"
            ),
            format!("HTTP/1.1 304 Not Modified\r\n{close}Cache-Control: max-age=60\r\n\r\n"),
        ]);
        let get = || NativeFetch::get_blocking(&agent, Some(&cache), &plain(&url), None, true);
        assert_eq!(get().unwrap(), b"tile");
        assert_eq!(get().unwrap(), b"tile");
        // Fresh after the 304: no request.
        assert_eq!(get().unwrap(), b"tile");
        let requests = server.join().unwrap();
        assert!(
            requests[1].contains("if-none-match: \"v1\""),
            "{}",
            requests[1]
        );
        // Stale and the server gone: the stale copy.
        cache.refresh(&orb_host::cache::key(&url, None), 0);
        assert_eq!(get().unwrap(), b"tile");
        drop(cache);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn transforms_rewrite_and_sign_requests_and_offline_answers_from_the_cache() {
        let dir = std::env::temp_dir().join(format!("orb-fetch-hooks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = HttpCache::open(Arc::new(FileStorage::new(&dir)), 1 << 20).unwrap();
        let fetch = NativeFetch::with_cache(Some(cache));
        let (url, server) = serve(vec![
            "HTTP/1.1 200 OK\r\nConnection: close\r\nCache-Control: max-age=0\r\nContent-Length: 4\r\n\r\ntile"
                .into(),
        ]);
        let base = url.trim_end_matches("tile").to_string();
        fetch.set_transform(Some(Arc::new(move |url: &str| {
            let path = url.strip_prefix("tiles://")?;
            Some(Request {
                url: format!("{base}{path}"),
                headers: vec![("Authorization".into(), "Bearer token".into())],
            })
        })));
        let get = |url: &str| {
            let (send, receive) = std::sync::mpsc::channel();
            fetch.get(url, Box::new(move |r| send.send(r).unwrap()));
            receive.recv().unwrap()
        };
        assert_eq!(get("tiles://tile").unwrap(), b"tile");
        let requests = server.join().unwrap();
        assert!(requests[0].starts_with("get /tile "), "{}", requests[0]);
        assert!(
            requests[0].contains("authorization: bearer token"),
            "{}",
            requests[0]
        );
        assert!(matches!(
            get("https://elsewhere/"),
            Err(FetchError::UnsupportedUrl(_))
        ));
        // Offline: the cached copy, stale as it is, and nothing else.
        fetch.set_online(false);
        assert_eq!(get("tiles://tile").unwrap(), b"tile");
        assert_eq!(
            get("tiles://other"),
            Err(FetchError::Network("offline".into()))
        );
        drop(fetch);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

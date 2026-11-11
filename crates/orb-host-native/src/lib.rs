//! Host services on native platforms.

#![deny(unsafe_code)]

mod cache;
mod connect;
mod fonts;
mod storage;

use std::sync::Arc;

pub use cache::{CacheStats, HttpCache};
pub use fonts::SystemFonts;
pub use orb_host::FontFileFaces;
pub use storage::{FileStorage, wait};

use orb_host::jobs::{Executor, ThreadPool};
use orb_host::{
    Abort, DecodedImage, Fetch, FetchDone, FetchError, ImageDecode, ImageDone, NetworkAnswer,
    NetworkHook, NetworkRequest, Prior, Request, RequestTransform, ResourceKind,
};

/// Fetches `http(s)://` URLs with ureq and `file://` URLs from disk, on a
/// small pool of I/O threads, through an [`HttpCache`] when it has one. A
/// [`NetworkHook`] can serve requests where the network would. An aborted
/// request still waiting for a thread never starts, and one reading its
/// body stops reading and drops its connection.
pub struct NativeFetch {
    agent: ureq::Agent,
    pool: Arc<ThreadPool>,
    cache: Option<Arc<HttpCache>>,
    transform: std::sync::RwLock<Option<Arc<RequestTransform>>>,
    hook: std::sync::RwLock<Option<Arc<dyn NetworkHook>>>,
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

/// Requests in flight at once, each on a thread and connection of its own.
/// Cold loads take as long with 16 (and longer with 4).
const FETCH_THREADS: usize = 8;

/// A request's cache key, when it has one, and the stale copy the
/// network's answer replaces or confirms.
type Stale = (Option<String>, Option<cache::Cached>);

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
        Arc::new(Self {
            // Every thread's connection stays open for the next request.
            agent: connect::agent(FETCH_THREADS),
            pool: Arc::new(ThreadPool::new("fetch", FETCH_THREADS)),
            cache,
            transform: Default::default(),
            hook: Default::default(),
            online: Arc::new(true.into()),
        })
    }

    /// Rewrites every request from here on: its URL, before the cache, and
    /// headers to send (`None` for none).
    pub fn set_transform(&self, transform: Option<Arc<RequestTransform>>) {
        *self.transform.write().expect("transform poisoned") = transform;
    }

    /// Serves the requests the cache can't answer through `hook` from here
    /// on, whatever their URL's scheme, or through the network again
    /// (`None`).
    pub fn set_network_hook(&self, hook: Option<Arc<dyn NetworkHook>>) {
        *self.hook.write().expect("hook poisoned") = hook;
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

    /// One request through the cache and the network, on this thread.
    fn get_blocking(
        agent: &ureq::Agent,
        cache: Option<&HttpCache>,
        request: &Request,
        range: Option<(u64, u64)>,
        online: bool,
        abort: &Abort,
    ) -> Result<Vec<u8>, FetchError> {
        let (key, cached) = match Self::prepare(cache, &request.url, range, online, false) {
            Ok(answer) => return answer,
            Err(network) => network,
        };
        let validators = cached
            .as_ref()
            .map(|c| (c.etag.as_deref(), c.last_modified.as_deref()));
        let response = http(agent, request, range, validators, abort);
        Self::finish(cache, key, cached, response)
    }

    /// What answers a request without the network: a file, or the cache
    /// when it holds a fresh copy or the network is off. Otherwise the
    /// cache key and the stale copy the network's answer goes with.
    fn prepare(
        cache: Option<&HttpCache>,
        url: &str,
        range: Option<(u64, u64)>,
        online: bool,
        hooked: bool,
    ) -> Result<Result<Vec<u8>, FetchError>, Stale> {
        if let Some(path) = url.strip_prefix("file://") {
            return Ok(read_file(path, range));
        }
        let cacheable = orb_host::cache::cacheable(url);
        if !cacheable && !hooked {
            return Ok(Err(FetchError::UnsupportedUrl(url.to_string())));
        }
        let key = cacheable.then(|| orb_host::cache::key(url, range));
        let cached = cache.zip(key.as_deref()).and_then(|(c, k)| c.lookup(k));
        if !online {
            return Ok(cached
                .map(|c| c.bytes)
                .ok_or_else(|| FetchError::Network("offline".into())));
        }
        if let Some(cached) = cached.as_ref().filter(|c| c.fresh) {
            return Ok(Ok(cached.bytes.clone()));
        }
        Err((key, cached))
    }

    /// The network's answer, kept in the cache as it says, or the stale
    /// copy when the network fails.
    fn finish(
        cache: Option<&HttpCache>,
        key: Option<String>,
        cached: Option<cache::Cached>,
        response: Result<Response, FetchError>,
    ) -> Result<Vec<u8>, FetchError> {
        let Some((cache, key)) = cache.zip(key) else {
            return match (response?, cached) {
                (Response::Body { bytes, .. }, _) => Ok(bytes),
                (Response::NotModified { .. }, Some(cached)) => Ok(cached.bytes),
                (Response::NotModified { .. }, None) => Err(FetchError::Status(304)),
            };
        };
        match (response, cached) {
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
            // Given up: the entry stays as it was.
            (Err(FetchError::Aborted), _) => Err(FetchError::Aborted),
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

/// The file a `file://` URL's path names, its `%XX` escapes decoded.
fn file_path(url_path: &str) -> std::path::PathBuf {
    let bytes = percent_decode(url_path);
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        std::ffi::OsStr::from_bytes(&bytes).into()
    }
    #[cfg(not(unix))]
    {
        String::from_utf8_lossy(&bytes).into_owned().into()
    }
}

/// `text` with each `%` and two hex digits decoded to their byte, as the
/// WHATWG URL standard decodes: a `%` without two digits stays as it is.
fn percent_decode(text: &str) -> std::borrow::Cow<'_, [u8]> {
    let bytes = text.as_bytes();
    if !bytes.contains(&b'%') {
        return bytes.into();
    }
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let escaped = (bytes[i] == b'%')
            .then(|| Some((hex(*bytes.get(i + 1)?)?, hex(*bytes.get(i + 2)?)?)))
            .flatten();
        match escaped {
            Some((high, low)) => {
                out.push(high << 4 | low);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    out.into()
}

/// Reads a `file://` URL's path, `range` of it when given.
fn read_file(url_path: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>, FetchError> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(file_path(url_path)).map_err(|e| match e.kind() {
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

/// Redirects a request carrying the host's headers follows: ureq's default.
const MAX_REDIRECTS: usize = 10;

/// One HTTP GET, conditional on `validators` (ETag, Last-Modified). An
/// abort stops it between the body's reads.
///
/// A request carrying the host's headers follows its redirects itself and
/// drops those headers once a redirect leaves the URL's origin: ureq strips
/// only `Authorization` and `Cookie`, so a key in a header of the host's
/// own (`X-Api-Key`) would reach whatever origin a redirect names.
fn http(
    agent: &ureq::Agent,
    req: &Request,
    range: Option<(u64, u64)>,
    validators: Option<(Option<&str>, Option<&str>)>,
    abort: &Abort,
) -> Result<Response, FetchError> {
    let manual = !req.headers.is_empty();
    let mut url = encode_url(&req.url).into_owned();
    let mut headers = req.headers.as_slice();
    let mut redirects = 0;
    let mut response = loop {
        let mut request = agent.get(&url);
        if manual {
            request = request.config().max_redirects(0).build();
        }
        for (name, value) in headers {
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
        let response = request.call().map_err(|e| match e {
            ureq::Error::StatusCode(404) => FetchError::NotFound,
            ureq::Error::StatusCode(code) => FetchError::Status(code),
            e => FetchError::Network(e.to_string()),
        })?;
        let status = response.status().as_u16();
        if !manual || !matches!(status, 301 | 302 | 303 | 307 | 308) {
            break response;
        }
        let location = response
            .headers()
            .get("location")
            .and_then(|h| h.to_str().ok())
            .ok_or(FetchError::Status(status))?;
        redirects += 1;
        if redirects > MAX_REDIRECTS {
            return Err(FetchError::Network("too many redirects".into()));
        }
        let next = encode_url(&redirect_target(&url, location)).into_owned();
        if !origin(&next).is_some_and(|next| origin(&url) == Some(next)) {
            headers = &[];
        }
        url = next;
    };
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
    let bytes = read_body(response.body_mut(), range, abort)?;
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

/// A response's body, at most a range's `length` or [`MAX_BODY`], read a
/// chunk at a time until it ends or `abort` fires.
fn read_body(
    body: &mut ureq::Body,
    range: Option<(u64, u64)>,
    abort: &Abort,
) -> Result<Vec<u8>, FetchError> {
    use std::io::Read;
    let limit = range.map_or(MAX_BODY, |(_, length)| length + 1);
    let capacity = range.map_or_else(
        || body.content_length().unwrap_or(0).min(limit),
        |(_, length)| length,
    );
    let mut reader = body.with_config().limit(limit).reader();
    let mut bytes = Vec::with_capacity(capacity as usize);
    loop {
        if abort.is_aborted() {
            return Err(FetchError::Aborted);
        }
        // Straight into the body, 64 KiB at most between the checks.
        match (&mut reader).take(64 << 10).read_to_end(&mut bytes) {
            Ok(0) => return Ok(bytes),
            Ok(_) => {}
            Err(e) => return Err(FetchError::Network(e.to_string())),
        }
    }
}

/// `location` resolved against `base`, the URL that answered with it.
fn redirect_target(base: &str, location: &str) -> String {
    if location.contains("://") {
        return location.to_string();
    }
    let scheme_end = base.find("://").map_or(0, |i| i + 3);
    if let Some(authority) = location.strip_prefix("//") {
        return format!("{}{authority}", &base[..scheme_end]);
    }
    let path = base.split(['?', '#']).next().unwrap_or(base);
    let origin_end = path[scheme_end..]
        .find('/')
        .map_or(path.len(), |i| scheme_end + i);
    if location.starts_with('/') {
        return format!("{}{location}", &path[..origin_end]);
    }
    // Beside the base's last segment, `./` and `../` left to the server.
    let dir_end = path[origin_end..]
        .rfind('/')
        .map_or(origin_end, |i| origin_end + i);
    format!("{}/{location}", &path[..dir_end])
}

/// A URL's scheme, host and port, the port spelled out where the scheme
/// implies it.
fn origin(url: &str) -> Option<(String, String, u16)> {
    let uri: ureq::http::Uri = url.parse().ok()?;
    let scheme = uri.scheme_str()?.to_ascii_lowercase();
    let port = uri.port_u16().or(match scheme.as_str() {
        "http" => Some(80),
        "https" => Some(443),
        _ => None,
    })?;
    Some((scheme, uri.host()?.to_ascii_lowercase(), port))
}

impl Fetch for NativeFetch {
    fn get(&self, kind: ResourceKind, url: &str, abort: &Abort, done: FetchDone) {
        self.get_range_opt(kind, url, None, abort, done);
    }
    fn get_range(
        &self,
        kind: ResourceKind,
        url: &str,
        offset: u64,
        length: u64,
        abort: &Abort,
        done: FetchDone,
    ) {
        self.get_range_opt(kind, url, Some((offset, length)), abort, done);
    }
}

impl NativeFetch {
    fn get_range_opt(
        &self,
        kind: ResourceKind,
        url: &str,
        range: Option<(u64, u64)>,
        abort: &Abort,
        done: FetchDone,
    ) {
        let done = abort.answer(done);
        if abort.is_aborted() {
            return;
        }
        let abort = abort.clone();
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
        let hook = self.hook.read().expect("hook poisoned").clone();
        let (agent, cache, online) = (self.agent.clone(), self.cache.clone(), self.online());
        let (pool, url) = (self.pool.clone(), url.to_string());
        self.pool.spawn(Box::new(move || {
            // Given up while it waited for a thread: `done` has its answer.
            if abort.is_aborted() {
                return;
            }
            let Some(hook) = hook else {
                return done(Self::get_blocking(
                    &agent,
                    cache.as_deref(),
                    &request,
                    range,
                    online,
                    &abort,
                ));
            };
            let (key, cached) =
                match Self::prepare(cache.as_deref(), &request.url, range, online, true) {
                    Ok(answer) => return done(answer),
                    Err(network) => network,
                };
            let prior = cached.as_ref().map(|c| Prior {
                etag: c.etag.clone(),
                modified: c
                    .last_modified
                    .as_deref()
                    .and_then(orb_host::cache::http_date),
                bytes: c.bytes.clone(),
            });
            let network = NetworkRequest {
                kind,
                url,
                request,
                range,
                prior,
                abort: abort.clone(),
            };
            hook.request(
                network,
                Box::new(move |answer| {
                    let response = match answer {
                        NetworkAnswer::Body {
                            bytes,
                            fresh_until,
                            etag,
                            modified,
                        } => Ok(Response::Body {
                            bytes,
                            fresh_until,
                            etag,
                            last_modified: modified.map(orb_host::cache::format_http_date),
                        }),
                        NetworkAnswer::NotModified { fresh_until } => {
                            Ok(Response::NotModified { fresh_until })
                        }
                        NetworkAnswer::Failed(error) => Err(error),
                        // The host's network, on its own threads.
                        NetworkAnswer::Network(request) => {
                            return pool.clone().spawn(Box::new(move || {
                                if abort.is_aborted() {
                                    return;
                                }
                                let response =
                                    if let Some(path) = request.url.strip_prefix("file://") {
                                        read_file(path, range).map(|bytes| Response::Body {
                                            bytes,
                                            fresh_until: None,
                                            etag: None,
                                            last_modified: None,
                                        })
                                    } else if orb_host::cache::cacheable(&request.url) {
                                        let validators = cached.as_ref().map(|c| {
                                            (c.etag.as_deref(), c.last_modified.as_deref())
                                        });
                                        http(&agent, &request, range, validators, &abort)
                                    } else {
                                        Err(FetchError::UnsupportedUrl(request.url.clone()))
                                    };
                                done(Self::finish(cache.as_deref(), key, cached, response));
                            }));
                        }
                    };
                    done(Self::finish(cache.as_deref(), key, cached, response));
                }),
            );
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

/// Decodes PNG, JPEG and WebP images with `image`, exactly (DEM tiles are
/// data), on the caller's thread (tile and sprite decodes already run on
/// job threads).
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeImageDecode;

impl ImageDecode for NativeImageDecode {
    fn decode(&self, bytes: Vec<u8>, done: ImageDone) {
        done(
            image::load_from_memory(&bytes)
                .map(|image| {
                    let rgba = image.into_rgba8();
                    DecodedImage {
                        width: rgba.width(),
                        height: rgba.height(),
                        pixels: rgba.into_raw(),
                    }
                })
                .map_err(|e| e.to_string()),
        );
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

/// A directory of a test's own under the system's temporary one, removed
/// with what it holds when it drops, as a failing test unwinds too.
#[cfg(test)]
pub(crate) struct Scratch(std::path::PathBuf);

#[cfg(test)]
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

#[cfg(test)]
impl std::ops::Deref for Scratch {
    type Target = std::path::Path;
    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

#[cfg(test)]
impl AsRef<std::path::Path> for Scratch {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

#[cfg(test)]
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc::channel;

    /// A request for `url` alone.
    fn plain(url: &str) -> Request {
        Request {
            url: url.into(),
            headers: Vec::new(),
        }
    }

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

    #[test]
    fn reads_files_and_reports_missing_ones() {
        let fetch = NativeFetch::new();
        let dir = Scratch::new("host-native");
        let path = dir.join("hello");
        std::fs::write(&path, b"hello").unwrap();
        let (tx, rx) = channel();
        for url in [
            format!("file://{}", path.display()),
            "file:///nonexistent/orbis".into(),
            "ftp://x".into(),
        ] {
            let tx = tx.clone();
            fetch.get(
                ResourceKind::Unknown,
                &url.clone(),
                &Abort::NEVER,
                Box::new(move |r| tx.send((url, r)).unwrap()),
            );
        }
        let mut results: Vec<_> = (0..3).map(|_| rx.recv().unwrap()).collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(results[0].1, Err(FetchError::NotFound));
        assert_eq!(results[1].1, Ok(b"hello".to_vec()));
        assert!(matches!(results[2].1, Err(FetchError::UnsupportedUrl(_))));
    }

    #[test]
    fn file_urls_are_percent_decoded() {
        assert_eq!(&*percent_decode("a%20b%2fc%zz%4"), b"a b/c%zz%4");
        let dir = Scratch::new("file-url");
        std::fs::write(dir.join("a b%.json"), b"{}").unwrap();
        let agent = ureq::Agent::new_with_defaults();
        let get = |name: &str| {
            let url = format!("file://{}/{name}", dir.display());
            NativeFetch::get_blocking(&agent, None, &plain(&url), None, true, &Abort::NEVER)
        };
        assert_eq!(get("a%20b%25.json").unwrap(), b"{}");
        assert_eq!(get("a%20b%.json").unwrap(), b"{}");
        assert_eq!(get("a b%.json").unwrap(), b"{}");
    }

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
            &Abort::NEVER,
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
        let dir = Scratch::new("range");
        let path = dir.join("abcdef");
        std::fs::write(&path, b"abcdef").unwrap();
        let url = format!("file://{}", path.display());
        let agent = ureq::Agent::new_with_defaults();
        assert_eq!(
            NativeFetch::get_blocking(
                &agent,
                None,
                &plain(&url),
                Some((2, 3)),
                true,
                &Abort::NEVER
            )
            .unwrap(),
            b"cde"
        );
        for range in [(4, 3), (0, 0), (0, MAX_BODY + 1), (u64::MAX, 2)] {
            assert!(
                NativeFetch::get_blocking(
                    &agent,
                    None,
                    &plain(&url),
                    Some(range),
                    true,
                    &Abort::NEVER
                )
                .is_err()
            );
        }
    }

    #[test]
    fn concurrent_writers_publish_whole_files() {
        let dir = Scratch::new("host-write");
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
    }

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
        let dir = Scratch::new("fetch-cache");
        let cache = HttpCache::open(Arc::new(FileStorage::new(&dir)), 1 << 20).unwrap();
        let agent = ureq::Agent::new_with_defaults();
        let close = "Connection: close\r\n";
        let (url, server) = serve(vec![
            format!(
                "HTTP/1.1 200 OK\r\n{close}Cache-Control: no-cache\r\nETag: \"v1\"\r\nContent-Length: 4\r\n\r\ntile"
            ),
            format!("HTTP/1.1 304 Not Modified\r\n{close}Cache-Control: max-age=60\r\n\r\n"),
        ]);
        let get = || {
            NativeFetch::get_blocking(
                &agent,
                Some(&cache),
                &plain(&url),
                None,
                true,
                &Abort::NEVER,
            )
        };
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
    }
    #[test]
    fn host_headers_follow_redirects_within_their_origin_alone() {
        let (other, elsewhere) = serve(vec![
            "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 4\r\n\r\ntile".into(),
        ]);
        let close = "Connection: close\r\n";
        let (url, origin) = serve(vec![
            format!("HTTP/1.1 302 Found\r\n{close}Location: /moved\r\nContent-Length: 0\r\n\r\n"),
            format!(
                "HTTP/1.1 307 Temporary Redirect\r\n{close}Location: {other}\r\nContent-Length: 0\r\n\r\n"
            ),
        ]);
        let request = Request {
            url,
            headers: vec![("X-Api-Key".into(), "secret".into())],
        };
        let agent = ureq::Agent::new_with_defaults();
        assert_eq!(
            NativeFetch::get_blocking(&agent, None, &request, None, true, &Abort::NEVER).unwrap(),
            b"tile"
        );
        let (origin, elsewhere) = (origin.join().unwrap(), elsewhere.join().unwrap());
        assert!(origin[0].contains("x-api-key: secret"), "{}", origin[0]);
        assert!(origin[1].starts_with("get /moved "), "{}", origin[1]);
        assert!(origin[1].contains("x-api-key: secret"), "{}", origin[1]);
        assert!(!elsewhere[0].contains("x-api-key"), "{}", elsewhere[0]);
    }

    #[test]
    fn redirects_resolve_against_the_url_that_sent_them() {
        let base = "http://a.org:8080/dir/tile.pbf?k=v";
        for (location, target) in [
            ("https://b.org/x", "https://b.org/x"),
            ("//b.org/x", "http://b.org/x"),
            ("/x?y", "http://a.org:8080/x?y"),
            ("x", "http://a.org:8080/dir/x"),
        ] {
            assert_eq!(redirect_target(base, location), target);
        }
        assert_eq!(redirect_target("http://a.org", "x"), "http://a.org/x");
        assert_eq!(origin("HTTP://A.org/x"), origin("http://a.org:80/y"));
        assert_ne!(origin("http://a.org/"), origin("https://a.org/"));
    }

    #[test]
    fn transforms_rewrite_and_sign_requests_and_offline_answers_from_the_cache() {
        let dir = Scratch::new("fetch-hooks");
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
            fetch.get(
                ResourceKind::Unknown,
                url,
                &Abort::NEVER,
                Box::new(move |r| send.send(r).unwrap()),
            );
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
    }

    /// An abort answers at once, and a body being read stops: the server
    /// sees the connection close long before it sends the whole body.
    #[test]
    fn an_abort_stops_reading_the_body() {
        const LENGTH: usize = 64 << 20;
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/big", server.local_addr().unwrap());
        let (started, started_rx) = channel();
        let thread = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut request = [0; 2048];
            let _ = stream.read(&mut request).unwrap();
            let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {LENGTH}\r\n\r\n");
            stream.write_all(head.as_bytes()).unwrap();
            let chunk = vec![0; 64 << 10];
            let mut sent = 0;
            while sent < LENGTH {
                if stream.write_all(&chunk).is_err() {
                    break;
                }
                sent += chunk.len();
                if sent == chunk.len() {
                    started.send(()).unwrap();
                }
            }
            sent
        });
        let fetch = NativeFetch::new();
        let abort = Abort::new();
        let (tx, rx) = channel();
        fetch.get(
            ResourceKind::Tile,
            &url,
            &abort,
            Box::new(move |r| tx.send(r).unwrap()),
        );
        started_rx.recv().unwrap();
        abort.abort();
        assert_eq!(rx.recv().unwrap(), Err(FetchError::Aborted));
        let sent = thread.join().unwrap();
        assert!(sent < LENGTH, "sent the whole body");
        // Given up before it starts, a request never reaches the network.
        let (tx, rx) = channel();
        let abort = Abort::new();
        abort.abort();
        fetch.get(
            ResourceKind::Tile,
            "http://127.0.0.1:9/never",
            &abort,
            Box::new(move |r| tx.send(r).unwrap()),
        );
        assert_eq!(rx.recv().unwrap(), Err(FetchError::Aborted));
    }
}

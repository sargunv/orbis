//! Host services on native platforms.

#![deny(unsafe_code)]

#[cfg(target_os = "android")]
pub mod android;
mod cache;
#[cfg(ureq_transport)]
mod connect;
mod fonts;
mod http;
mod storage;
#[cfg(target_os = "android")]
mod url_connection;

use std::sync::Arc;

pub use cache::{CacheStats, HttpCache};
pub use fonts::SystemFonts;
pub use orb_host::FontFileFaces;
pub use storage::{FileStorage, wait};

use http::Freshness;
use orb_host::jobs::{Executor, ThreadPool};
use orb_host::{
    Abort, DecodedImage, Fetch, FetchDone, FetchError, Fetched, ImageDecode, ImageDone,
    NetworkAnswer, NetworkHook, NetworkRequest, Prior, Request, RequestTransform, ResourceKind,
};

/// Fetches `http(s)://` URLs with ureq (on Android through the platform's
/// `HttpURLConnection`) and `file://` URLs from disk, on a small pool of
/// I/O threads, through an [`HttpCache`] when it has one. A [`NetworkHook`]
/// can serve requests where the network would. An aborted request still
/// waiting for a thread never starts, and one reading its body stops
/// reading and drops its connection.
pub struct NativeFetch {
    /// HTTPS verifying servers as the platform does, made with the first
    /// request that goes through ureq.
    #[cfg(ureq_transport)]
    agent: std::sync::OnceLock<ureq::Agent>,
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

/// Requests other than tile fetches in flight beside them: glyph
/// ranges, sprites, documents.
const OTHER_FETCHES: usize = 4;

/// Requests in flight at once, each on a thread and connection of its
/// own: every tile fetch a map's turns start ([`orb_host::tile_fetches`])
/// and others beside them. Fewer would queue tile fetches here again, in
/// the order they started rather than the map's, and glyph ranges behind
/// them: over a link answering 150 ms late, 8 threads behind 16 turns
/// took the view after a globe spin three times as long to come in.
fn fetch_threads() -> usize {
    orb_host::tile_fetches() + OTHER_FETCHES
}

/// A request's cache key, when it has one, and the stale copy the
/// network's answer replaces or confirms.
type Stale = (Option<String>, Option<cache::Cached>);

/// How a request may be answered: the bytes it wants, whether the
/// network may answer it, and whether a stale cached copy may before the
/// network revalidates it ([`Fetch::get_or_stale`]).
#[derive(Clone, Copy, Debug)]
struct Ask {
    range: Option<(u64, u64)>,
    online: bool,
    stale: bool,
}

/// What the network answered.
enum Response {
    Body {
        bytes: Vec<u8>,
        /// `None` when the response may not be stored.
        freshness: Option<Freshness>,
        etag: Option<String>,
        last_modified: Option<String>,
    },
    NotModified {
        freshness: Option<Freshness>,
        /// It said nothing of `Cache-Control`: the stored response's
        /// directives stand (RFC 9111 §4.3.4).
        directives_kept: bool,
    },
}

impl NativeFetch {
    pub fn new() -> Arc<Self> {
        Self::with_cache(None)
    }

    /// Fetches through `cache`: fresh entries without the network, stale
    /// ones revalidated, and either when the network fails. Through
    /// [`Fetch::get_or_stale`], a stale entry answers at once, unless its
    /// response said it must revalidate first.
    pub fn with_cache(cache: Option<Arc<HttpCache>>) -> Arc<Self> {
        Arc::new(Self {
            #[cfg(ureq_transport)]
            agent: std::sync::OnceLock::new(),
            pool: Arc::new(ThreadPool::new("fetch", fetch_threads())),
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

    /// What HTTP goes through now: on Android `HttpURLConnection`, but
    /// ureq in a `standalone` build before [`android::init`] gave a JVM.
    fn transport(&self) -> Transport {
        #[cfg(all(target_os = "android", ureq_transport))]
        if android::java_vm().is_some() {
            return Transport::UrlConnection;
        }
        #[cfg(ureq_transport)]
        // Every thread's connection stays open for the next request.
        return Transport::Ureq(
            self.agent
                .get_or_init(|| connect::agent(fetch_threads()))
                .clone(),
        );
        #[cfg(not(ureq_transport))]
        Transport::UrlConnection
    }

    /// The cache fetches go through.
    pub fn cache(&self) -> Option<&Arc<HttpCache>> {
        self.cache.as_ref()
    }

    /// What answers a request without the network: a file, or the cache
    /// when it holds a fresh copy, a stale one the request takes, or the
    /// network is off. Otherwise the cache key and the stale copy the
    /// network's answer goes with: none for a URL the cache doesn't keep,
    /// which a [`NetworkHook`] may serve and [`network`] refuses.
    fn prepare(
        cache: Option<&HttpCache>,
        url: &str,
        Ask {
            range,
            online,
            stale,
        }: Ask,
    ) -> Result<Result<Fetched, FetchError>, Stale> {
        if let Some(path) = url.strip_prefix("file://") {
            return Ok(read_file(path, range).map(Fetched::from));
        }
        let key = http::cacheable(url).then(|| orb_host::cache::key(url, range));
        let cached = cache.zip(key.as_deref()).and_then(|(c, k)| c.lookup(k));
        let cached = match cached {
            Some(cached) if !online || cached.fresh => return Ok(Ok(cached.fetched())),
            Some(cached) if stale => match cached.stale() {
                Ok(fetched) => return Ok(Ok(fetched)),
                Err(cached) => Some(cached),
            },
            None if !online => return Ok(Err(FetchError::Network("offline".into()))),
            cached => cached,
        };
        Err((key, cached))
    }

    /// The network's answer, kept in the cache as it says, or the stale
    /// copy when the network fails.
    fn finish(
        cache: Option<&HttpCache>,
        key: Option<String>,
        cached: Option<cache::Cached>,
        response: Result<Response, FetchError>,
    ) -> Result<Fetched, FetchError> {
        // A lifetime already over is none: the response came stale. One
        // the cache guessed keeps it fresh there, but isn't the map's.
        let now = cache::now();
        let fresh = |freshness: Option<Freshness>| {
            freshness
                .filter(|f| f.explicit && f.until > now)
                .map(|f| f.until)
        };
        let Some((cache, key)) = cache.zip(key) else {
            return match (response?, cached) {
                (
                    Response::Body {
                        bytes, freshness, ..
                    },
                    _,
                ) => Ok(Fetched {
                    bytes,
                    fresh_until: fresh(freshness),
                    stale: false,
                }),
                (Response::NotModified { freshness, .. }, Some(cached)) => Ok(Fetched {
                    bytes: cached.bytes,
                    fresh_until: fresh(freshness),
                    stale: false,
                }),
                (Response::NotModified { .. }, None) => Err(FetchError::Status(304)),
            };
        };
        match (response, cached) {
            (
                Ok(Response::Body {
                    bytes,
                    freshness,
                    etag,
                    last_modified,
                }),
                _,
            ) => {
                match freshness {
                    Some(freshness) => cache.store(
                        &key,
                        &bytes,
                        freshness,
                        etag.as_deref(),
                        last_modified.as_deref(),
                    ),
                    None => cache.remove(&key),
                }
                Ok(Fetched {
                    bytes,
                    fresh_until: fresh(freshness),
                    stale: false,
                })
            }
            (
                Ok(Response::NotModified {
                    freshness,
                    directives_kept,
                }),
                Some(cached),
            ) => {
                // Without directives, those the stale copy came with stand.
                let freshness = freshness.map(|f| Freshness {
                    must_revalidate: match directives_kept {
                        true => cached.freshness.must_revalidate,
                        false => f.must_revalidate,
                    },
                    ..f
                });
                // Unstorable, it leaves the copy stale, its directives kept.
                let stale = Freshness {
                    until: 0,
                    ..cached.freshness
                };
                cache.refresh(&key, freshness.unwrap_or(stale));
                Ok(Fetched {
                    bytes: cached.bytes,
                    fresh_until: fresh(freshness),
                    stale: false,
                })
            }
            (Ok(Response::NotModified { .. }), None) => Err(FetchError::Status(304)),
            // Given up: the entry stays as it was.
            (Err(FetchError::Aborted), _) => Err(FetchError::Aborted),
            // Offline, or the server failing or rate-limiting: the stale
            // copy.
            (Err(error), Some(cached)) if error.transient() => Ok(Fetched {
                bytes: cached.bytes,
                fresh_until: None,
                stale: true,
            }),
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

/// Reads a `file://` URL's path (what follows `file://`), `range` of it
/// when given, on the caller's thread: what [`NativeFetch`] does with such
/// a URL on its I/O threads. On Android a query or fragment ends the path,
/// and `/android_asset/` names the app's assets once [`android::init`]
/// gave their manager.
pub fn read_file(url_path: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>, FetchError> {
    #[cfg(target_os = "android")]
    let url_path = url_path.split(['?', '#']).next().unwrap_or_default();
    let path = file_path(url_path);
    #[cfg(target_os = "android")]
    if let Some(read) = android::asset_name(&path).and_then(|name| android::read_asset(name, range))
    {
        return read;
    }
    let file = std::fs::File::open(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => FetchError::NotFound,
        _ => FetchError::Network(e.to_string()),
    })?;
    read_from(file, range)
}

/// `range` of what `file` reads when given, else all of it up to
/// [`MAX_BODY`].
fn read_from(
    mut file: impl std::io::Read + std::io::Seek,
    range: Option<(u64, u64)>,
) -> Result<Vec<u8>, FetchError> {
    use std::io::{Read, SeekFrom};
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

/// What the network answers `request`, `range` of it when given: a
/// `file://` URL's file (what a [`NetworkHook`] may hand back), or an HTTP
/// GET, conditional on the stale copy `cached` when there is one. Other
/// schemes are refused.
fn network(
    transport: &Transport,
    request: &Request,
    range: Option<(u64, u64)>,
    cached: Option<&cache::Cached>,
    abort: &Abort,
) -> Result<Response, FetchError> {
    if let Some(path) = request.url.strip_prefix("file://") {
        return read_file(path, range).map(|bytes| Response::Body {
            bytes,
            freshness: None,
            etag: None,
            last_modified: None,
        });
    }
    if !http::cacheable(&request.url) {
        return Err(FetchError::UnsupportedUrl(request.url.clone()));
    }
    let validators = cached.map(|c| (c.etag.as_deref(), c.last_modified.as_deref()));
    http(transport, request, range, validators, abort)
}

/// What sends a GET: ureq, or on Android the platform's
/// `HttpURLConnection`.
#[derive(Clone)]
enum Transport {
    #[cfg(ureq_transport)]
    Ureq(ureq::Agent),
    #[cfg(target_os = "android")]
    UrlConnection,
}

impl Transport {
    /// GETs `url` with `headers`, following its redirects where `follow`
    /// says and the transport can.
    fn get(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        follow: bool,
        abort: &Abort,
    ) -> Result<Box<dyn Reply>, FetchError> {
        match self {
            #[cfg(ureq_transport)]
            Self::Ureq(agent) => {
                let _ = abort;
                connect::get(agent, url, headers, follow)
            }
            #[cfg(target_os = "android")]
            Self::UrlConnection => {
                let _ = follow;
                url_connection::get(url, headers, abort)
            }
        }
    }

    /// Whether it can follow redirects itself: `HttpURLConnection` won't
    /// from `http` to `https`, so [`http`] follows them for it.
    fn follows_redirects(&self) -> bool {
        match self {
            #[cfg(ureq_transport)]
            Self::Ureq(_) => true,
            #[cfg(target_os = "android")]
            Self::UrlConnection => false,
        }
    }
}

/// A transport's response to one GET, its body unread until asked for.
trait Reply {
    fn status(&self) -> u16;
    /// The first value of the header `name`, in any case.
    fn header(&self, name: &str) -> Option<String>;
    /// The body, at most `limit` bytes ([`read_body`]).
    fn body(self: Box<Self>, limit: u64, abort: &Abort) -> Result<Vec<u8>, FetchError>;
}

/// The user agent requests send.
const USER_AGENT: &str = concat!("orbis/", env!("CARGO_PKG_VERSION"));

/// Redirects a request followed by [`http`] follows: ureq's default.
const MAX_REDIRECTS: usize = 10;

/// One HTTP GET, conditional on `validators` (ETag, Last-Modified). An
/// abort stops it between the body's reads.
///
/// A request carrying the host's headers follows its redirects here and
/// drops those headers once a redirect leaves the URL's origin: ureq strips
/// only `Authorization` and `Cookie`, so a key in a header of the host's
/// own (`X-Api-Key`) would reach whatever origin a redirect names. So does
/// every request through a transport that can't follow them all.
fn http(
    transport: &Transport,
    req: &Request,
    range: Option<(u64, u64)>,
    validators: Option<(Option<&str>, Option<&str>)>,
    abort: &Abort,
) -> Result<Response, FetchError> {
    let follow = req.headers.is_empty() && transport.follows_redirects();
    let range_header = range
        .map(|(offset, length)| {
            orb_host::range_end(offset, length).map(|end| format!("bytes={offset}-{end}"))
        })
        .transpose()?;
    let mut url = encode_url(&req.url).into_owned();
    let mut host_headers = req.headers.as_slice();
    let mut redirects = 0;
    let reply = loop {
        let mut headers: Vec<(&str, &str)> = host_headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        if let Some(range) = &range_header {
            headers.extend([("Range", range.as_str()), ("Accept-Encoding", "identity")]);
        }
        if let Some((etag, last_modified)) = validators {
            headers.extend(etag.map(|etag| ("If-None-Match", etag)));
            headers.extend(last_modified.map(|date| ("If-Modified-Since", date)));
        }
        let reply = transport.get(&url, &headers, follow, abort)?;
        let status = reply.status();
        if follow || !matches!(status, 301 | 302 | 303 | 307 | 308) {
            break reply;
        }
        let location = reply.header("location").ok_or(FetchError::Status(status))?;
        redirects += 1;
        if redirects > MAX_REDIRECTS {
            return Err(FetchError::Network("too many redirects".into()));
        }
        let next = encode_url(&redirect_target(&url, &location)).into_owned();
        if !origin(&next).is_some_and(|next| origin(&url) == Some(next)) {
            host_headers = &[];
        }
        url = next;
    };
    let status = reply.status();
    let (cache_control, expires, date) = (
        reply.header("cache-control"),
        reply.header("expires"),
        reply.header("date"),
    );
    let (etag, last_modified) = (reply.header("etag"), reply.header("last-modified"));
    let headers = http::Headers {
        cache_control: cache_control.as_deref(),
        expires: expires.as_deref(),
        date: date.as_deref(),
        last_modified: last_modified.as_deref(),
    };
    let freshness = http::freshness(headers, cache::now());
    match status {
        200..=203 | 205..=299 => {}
        // No content: a valid, empty tile.
        204 => {
            return Ok(Response::Body {
                bytes: Vec::new(),
                freshness,
                etag,
                last_modified,
            });
        }
        304 => {
            return Ok(Response::NotModified {
                freshness,
                directives_kept: cache_control.is_none(),
            });
        }
        status => {
            let now = cache::now();
            let wait = reply.header("retry-after").and_then(|h| {
                orb_host::parse_retry_after(&h, |date| {
                    let at = http::http_date(date, now)?;
                    Some(std::time::Duration::from_secs(at.saturating_sub(now)))
                })
            });
            return Err(FetchError::of_status(status, wait));
        }
    }
    if let Some((offset, length)) = range {
        orb_host::check_range(
            status,
            reply.header("content-range").as_deref(),
            offset,
            length,
        )?;
    }
    let bytes = reply.body(range.map_or(MAX_BODY, |(_, length)| length), abort)?;
    if range.is_some_and(|(_, length)| bytes.len() as u64 != length) {
        return Err(FetchError::Range("short range body".into()));
    }
    Ok(Response::Body {
        bytes,
        freshness,
        etag,
        last_modified,
    })
}

/// What `body` reads, at most `limit` bytes (a `length` it says beyond
/// that fails before any read), a chunk at a time until it ends or `abort`
/// fires.
fn read_body(
    mut body: impl std::io::Read,
    length: Option<u64>,
    limit: u64,
    abort: &Abort,
) -> Result<Vec<u8>, FetchError> {
    use std::io::Read;
    let oversized = || FetchError::Range("oversized body".into());
    if length.is_some_and(|length| length > limit) {
        return Err(oversized());
    }
    let mut bytes = Vec::with_capacity(length.unwrap_or(0) as usize);
    loop {
        if abort.is_aborted() {
            return Err(FetchError::Aborted);
        }
        // Straight into the body, 64 KiB at most between the checks, and
        // one byte past the limit at most.
        let room = (limit + 1 - bytes.len() as u64).min(64 << 10);
        match (&mut body).take(room).read_to_end(&mut bytes) {
            Ok(0) => return Ok(bytes),
            Ok(_) if bytes.len() as u64 > limit => return Err(oversized()),
            Ok(_) => {}
            // An abort can end a read in flight.
            Err(_) if abort.is_aborted() => return Err(FetchError::Aborted),
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
    let (scheme, rest) = url.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    let (host, port) = match authority.strip_prefix('[') {
        Some(v6) => {
            let (host, rest) = v6.split_once(']')?;
            (host, rest.strip_prefix(':'))
        }
        None => match authority.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        },
    };
    let port = match port.filter(|p| !p.is_empty()) {
        Some(port) => port.parse().ok()?,
        None => match scheme.as_str() {
            "http" => 80,
            "https" => 443,
            _ => return None,
        },
    };
    (!host.is_empty()).then(|| (scheme, host.to_ascii_lowercase(), port))
}

impl Fetch for NativeFetch {
    fn get(&self, kind: ResourceKind, url: &str, abort: &Abort, done: FetchDone) {
        self.fetch(kind, url, (None, false), abort, done);
    }
    fn get_or_stale(&self, kind: ResourceKind, url: &str, abort: &Abort, done: FetchDone) {
        self.fetch(kind, url, (None, true), abort, done);
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
        self.fetch(kind, url, (Some((offset, length)), false), abort, done);
    }
}

impl NativeFetch {
    /// Fetches `url` (the range given of it), taking a stale cached copy
    /// if `stale` says.
    fn fetch(
        &self,
        kind: ResourceKind,
        url: &str,
        (range, stale): (Option<(u64, u64)>, bool),
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
        let (transport, cache) = (self.transport(), self.cache.clone());
        let ask = Ask {
            range,
            online: self.online(),
            stale,
        };
        let (pool, url) = (self.pool.clone(), url.to_string());
        self.pool.spawn(Box::new(move || {
            // Given up while it waited for a thread: `done` has its answer.
            if abort.is_aborted() {
                return;
            }
            let (key, cached) = match Self::prepare(cache.as_deref(), &request.url, ask) {
                Ok(answer) => return done(answer),
                Err(stale) => stale,
            };
            let Some(hook) = hook else {
                let response = network(&transport, &request, range, cached.as_ref(), &abort);
                return done(Self::finish(cache.as_deref(), key, cached, response));
            };
            let prior = cached.as_ref().map(|c| Prior {
                etag: c.etag.clone(),
                modified: c
                    .last_modified
                    .as_deref()
                    .and_then(|date| http::http_date(date, cache::now())),
                bytes: c.bytes.clone(),
            });
            let asked = NetworkRequest {
                kind,
                url,
                request,
                range,
                prior,
                abort: abort.clone(),
            };
            hook.request(
                asked,
                Box::new(move |answer| {
                    let response = match answer {
                        NetworkAnswer::Body {
                            bytes,
                            fresh_until,
                            must_revalidate,
                            etag,
                            modified,
                        } => Ok(Response::Body {
                            bytes,
                            freshness: hooked(fresh_until, must_revalidate),
                            etag,
                            last_modified: modified.map(http::format_http_date),
                        }),
                        NetworkAnswer::NotModified {
                            fresh_until,
                            must_revalidate,
                        } => Ok(Response::NotModified {
                            freshness: hooked(fresh_until, must_revalidate),
                            directives_kept: false,
                        }),
                        NetworkAnswer::Failed(error) => Err(error),
                        // The host's network, on its own threads.
                        NetworkAnswer::Network(request) => {
                            return pool.clone().spawn(Box::new(move || {
                                if abort.is_aborted() {
                                    return;
                                }
                                let response =
                                    network(&transport, &request, range, cached.as_ref(), &abort);
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

/// The lifetime a [`NetworkHook`]'s answer gives, which the host gave.
fn hooked(fresh_until: Option<u64>, must_revalidate: bool) -> Option<Freshness> {
    fresh_until.map(|until| Freshness {
        until,
        explicit: true,
        must_revalidate,
    })
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

    fn unix_ms(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
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

// Through ureq, which Android has only in a `standalone` build.
#[cfg(all(test, ureq_transport))]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc::channel;

    /// What a fetch `start` begins answers.
    fn wait(start: impl FnOnce(FetchDone)) -> Result<Fetched, FetchError> {
        let (send, receive) = channel();
        start(Box::new(move |r| send.send(r).unwrap()));
        receive.recv().unwrap()
    }

    /// What `fetch` answers for `url`.
    fn get(fetch: &NativeFetch, url: &str) -> Result<Fetched, FetchError> {
        wait(|done| fetch.get(ResourceKind::Unknown, url, &Abort::NEVER, done))
    }

    /// What `fetch` answers for `length` bytes of `url` from `offset`.
    fn get_range(
        fetch: &NativeFetch,
        url: &str,
        (offset, length): (u64, u64),
    ) -> Result<Fetched, FetchError> {
        wait(|done| {
            fetch.get_range(
                ResourceKind::Unknown,
                url,
                offset,
                length,
                &Abort::NEVER,
                done,
            )
        })
    }

    /// A fetch through a cache of its own in `dir`.
    fn cached(dir: &Scratch) -> (Arc<NativeFetch>, Arc<HttpCache>) {
        let cache = HttpCache::open(Arc::new(FileStorage::new(dir)), 1 << 20).unwrap();
        (NativeFetch::with_cache(Some(cache.clone())), cache)
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
        assert_eq!(results[1].1, Ok(b"hello".to_vec().into()));
        assert!(matches!(results[2].1, Err(FetchError::UnsupportedUrl(_))));
    }

    #[test]
    fn file_urls_are_percent_decoded() {
        assert_eq!(&*percent_decode("a%20b%2fc%zz%4"), b"a b/c%zz%4");
        let dir = Scratch::new("file-url");
        std::fs::write(dir.join("a b%.json"), b"{}").unwrap();
        let fetch = NativeFetch::new();
        let get = |name: &str| get(&fetch, &format!("file://{}/{name}", dir.display()));
        assert_eq!(get("a%20b%25.json").unwrap().bytes, b"{}");
        assert_eq!(get("a%20b%.json").unwrap().bytes, b"{}");
        assert_eq!(get("a b%.json").unwrap().bytes, b"{}");
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
        let result = get_range(&NativeFetch::new(), &url, (2, 3));
        thread.join().unwrap();
        result.map(|fetched| fetched.bytes)
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
        let fetch = NativeFetch::new();
        assert_eq!(get_range(&fetch, &url, (2, 3)).unwrap().bytes, b"cde");
        for range in [(4, 3), (0, 0), (0, MAX_BODY + 1), (u64::MAX, 2)] {
            assert!(get_range(&fetch, &url, range).is_err());
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
        let (fetch, cache) = cached(&dir);
        let close = "Connection: close\r\n";
        let (url, server) = serve(vec![
            format!(
                "HTTP/1.1 200 OK\r\n{close}Cache-Control: no-cache\r\nETag: \"v1\"\r\nContent-Length: 4\r\n\r\ntile"
            ),
            format!("HTTP/1.1 304 Not Modified\r\n{close}Cache-Control: max-age=60\r\n\r\n"),
        ]);
        let get = || get(&fetch, &url);
        // `no-cache`: it came stale, with no lifetime.
        assert_eq!(get().unwrap(), b"tile".to_vec().into());
        let renewed = get().unwrap();
        assert_eq!(renewed.bytes, b"tile");
        let until = renewed.fresh_until.expect("the 304's lifetime");
        assert!((59..=61).contains(&(until - cache::now())), "{until}");
        // Fresh after the 304: no request, and the same lifetime.
        assert_eq!(get().unwrap().fresh_until, Some(until));
        let requests = server.join().unwrap();
        assert!(
            requests[1].contains("if-none-match: \"v1\""),
            "{}",
            requests[1]
        );
        // Stale and the server gone: the stale copy, with no lifetime.
        cache.refresh(&orb_host::cache::key(&url, None), Freshness::default());
        let fallback = get().unwrap();
        assert_eq!(
            (fallback.bytes, fallback.fresh_until),
            (b"tile".to_vec(), None)
        );
        assert!(fallback.stale, "no server vouched for it");
    }

    #[test]
    fn a_stale_entry_answers_at_once_where_asked_unless_it_must_revalidate() {
        let dir = Scratch::new("fetch-stale");
        let (fetch, _) = cached(&dir);
        let close = "Connection: close\r\n";
        let get = |url: &str, stale| match stale {
            true => wait(|done| fetch.get_or_stale(ResourceKind::Tile, url, &Abort::NEVER, done)),
            false => get(&fetch, url),
        };
        let (url, server) = serve(vec![
            format!(
                "HTTP/1.1 200 OK\r\n{close}Cache-Control: max-age=0\r\nETag: \"v1\"\r\nContent-Length: 4\r\n\r\ntile"
            ),
            format!("HTTP/1.1 304 Not Modified\r\n{close}Cache-Control: max-age=60\r\n\r\n"),
        ]);
        assert_eq!(get(&url, true).unwrap(), b"tile".to_vec().into());
        // Stale: at once, with a lifetime already over, and no request.
        let stale = get(&url, true).unwrap();
        assert_eq!(stale.bytes, b"tile");
        assert!(stale.fresh_until.is_some_and(|until| until <= cache::now()));
        // What revalidates it asks the server, with its validator.
        let renewed = get(&url, false).unwrap();
        assert!(
            renewed
                .fresh_until
                .is_some_and(|until| until > cache::now())
        );
        let requests = server.join().unwrap();
        assert!(
            requests[1].contains("if-none-match: \"v1\""),
            "{}",
            requests[1]
        );

        // One that must revalidate waits for the server each time, though
        // a 304 that says nothing of it renewed it.
        let (url, server) = serve(vec![
            format!(
                "HTTP/1.1 200 OK\r\n{close}Cache-Control: max-age=0, must-revalidate\r\nETag: \"v1\"\r\nContent-Length: 4\r\n\r\ntile"
            ),
            format!("HTTP/1.1 304 Not Modified\r\n{close}\r\n"),
            format!(
                "HTTP/1.1 200 OK\r\n{close}Cache-Control: max-age=60\r\nContent-Length: 4\r\n\r\nnext"
            ),
        ]);
        assert_eq!(get(&url, true).unwrap().bytes, b"tile");
        assert_eq!(get(&url, true).unwrap().bytes, b"tile");
        assert_eq!(get(&url, true).unwrap().bytes, b"next");
        assert_eq!(server.join().unwrap().len(), 3);
    }

    #[test]
    fn rate_limited_requests_answer_with_the_stale_copy() {
        let dir = Scratch::new("fetch-limited");
        let (fetch, _) = cached(&dir);
        let close = "Connection: close\r\n";
        let (url, server) = serve(vec![
            format!(
                "HTTP/1.1 200 OK\r\n{close}Cache-Control: no-cache\r\nContent-Length: 4\r\n\r\ntile"
            ),
            format!("HTTP/1.1 429 Too Many Requests\r\n{close}Retry-After: 60\r\n\r\n"),
            format!("HTTP/1.1 408 Request Timeout\r\n{close}\r\n"),
        ]);
        for _ in 0..3 {
            assert_eq!(get(&fetch, &url).unwrap().bytes, b"tile");
        }
        server.join().unwrap();
    }

    #[test]
    fn failed_responses_say_when_to_ask_again() {
        use std::time::Duration;
        let close = "Connection: close\r\n";
        let in_an_hour = http::format_http_date(cache::now() + 3600);
        let (url, server) = serve(vec![
            format!("HTTP/1.1 429 Too Many Requests\r\n{close}Retry-After: 120\r\n\r\n"),
            format!("HTTP/1.1 429 Too Many Requests\r\n{close}Retry-After: {in_an_hour}\r\n\r\n"),
            format!("HTTP/1.1 429 Too Many Requests\r\n{close}Retry-After: soon\r\n\r\n"),
            format!("HTTP/1.1 503 Service Unavailable\r\n{close}Retry-After: 30\r\n\r\n"),
            format!("HTTP/1.1 503 Service Unavailable\r\n{close}Content-Length: 4\r\n\r\ntile"),
            format!("HTTP/1.1 408 Request Timeout\r\n{close}Content-Length: 4\r\n\r\ntile"),
            format!("HTTP/1.1 404 Not Found\r\n{close}Content-Length: 4\r\n\r\ntile"),
            format!("HTTP/1.1 403 Forbidden\r\n{close}Content-Length: 4\r\n\r\ntile"),
        ]);
        let fetch = NativeFetch::new();
        let get = || get(&fetch, &url).unwrap_err();
        assert_eq!(get().retry_after(), Some(Duration::from_secs(120)));
        let hour = get().retry_after().unwrap().as_secs();
        assert!((3598..=3600).contains(&hour), "{hour} s");
        // Unreadable: rate-limited for the default wait.
        assert_eq!(get().retry_after(), Some(orb_host::RATE_LIMIT_WAIT));
        assert_eq!(get().retry_after(), Some(Duration::from_secs(30)));
        // Failures' bodies are never read as the resource.
        assert_eq!(get(), FetchError::Status(503));
        assert_eq!(get(), FetchError::Status(408));
        assert_eq!(get(), FetchError::NotFound);
        assert_eq!(get(), FetchError::Status(403));
        server.join().unwrap();
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
        let fetch = NativeFetch::new();
        fetch.set_transform(Some(Arc::new(|url: &str| {
            Some(Request {
                url: url.into(),
                headers: vec![("X-Api-Key".into(), "secret".into())],
            })
        })));
        assert_eq!(get(&fetch, &url).unwrap().bytes, b"tile");
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
        assert_eq!(
            origin("http://user:pass@[::1]:8080?q"),
            Some(("http".into(), "::1".into(), 8080))
        );
        assert_eq!(origin("https://A.org#f"), origin("https://a.org:443/"));
    }

    #[test]
    fn transforms_rewrite_and_sign_requests_and_offline_answers_from_the_cache() {
        let dir = Scratch::new("fetch-hooks");
        let (fetch, _) = cached(&dir);
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
        let get = |url: &str| get(&fetch, url);
        assert_eq!(get("tiles://tile").unwrap().bytes, b"tile");
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
        assert_eq!(get("tiles://tile").unwrap().bytes, b"tile");
        assert_eq!(
            get("tiles://other"),
            Err(FetchError::Network("offline".into()))
        );
    }

    /// Hands every request back to the network, a `mem://` one as the file
    /// it names, noting the cached copy's validator it was asked with.
    struct Passing(std::sync::Mutex<Vec<Option<String>>>, String);

    impl NetworkHook for Passing {
        fn request(&self, asked: NetworkRequest, done: orb_host::NetworkDone) {
            let etag = asked.prior.and_then(|prior| prior.etag);
            self.0.lock().unwrap().push(etag);
            let mut request = asked.request;
            if let Some(name) = request.url.strip_prefix("mem://") {
                request.url = format!("file://{}/{name}", self.1);
            }
            done(NetworkAnswer::Network(request));
        }
    }

    #[test]
    fn a_hook_handing_requests_back_fetches_as_the_network_would() {
        let files = Scratch::new("fetch-passing-files");
        std::fs::write(files.join("style.json"), b"{}").unwrap();
        let dir = Scratch::new("fetch-passing");
        let (fetch, _) = cached(&dir);
        let hook = Arc::new(Passing(Default::default(), files.display().to_string()));
        fetch.set_network_hook(Some(hook.clone()));
        let close = "Connection: close\r\n";
        let (url, server) = serve(vec![
            format!(
                "HTTP/1.1 200 OK\r\n{close}Cache-Control: no-cache\r\nETag: \"v1\"\r\nContent-Length: 4\r\n\r\ntile"
            ),
            format!("HTTP/1.1 304 Not Modified\r\n{close}Cache-Control: max-age=60\r\n\r\n"),
        ]);
        assert_eq!(get(&fetch, &url).unwrap().bytes, b"tile");
        // Revalidated with the stale copy's validator, then fresh.
        assert!(get(&fetch, &url).unwrap().fresh_until.is_some());
        assert!(get(&fetch, &url).unwrap().fresh_until.is_some());
        let requests = server.join().unwrap();
        assert!(
            requests[1].contains("if-none-match: \"v1\""),
            "{}",
            requests[1]
        );
        assert_eq!(get(&fetch, "mem://style.json").unwrap().bytes, b"{}");
        assert!(matches!(
            get(&fetch, "mem://missing"),
            Err(FetchError::NotFound)
        ));
        assert!(matches!(
            get(&fetch, "ftp://x"),
            Err(FetchError::UnsupportedUrl(_))
        ));
        let asked = hook.0.lock().unwrap().clone();
        assert_eq!(asked[..2], [None, Some("\"v1\"".into())]);
        assert_eq!(asked.len(), 5, "the fresh copy answered without the hook");
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

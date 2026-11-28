//! Host service traits: what the core asks of its platform, the job
//! executors that run its work, and how every platform answers a lost GPU
//! ([`loss`]). Implementations live in `orb-host-native` and, for the
//! browser, `orb-web`.

#![forbid(unsafe_code)]

pub mod abort;
pub mod cache;
pub mod http;
pub mod jobs;
pub mod loss;

use std::fmt;
use std::time::Duration;

pub use abort::Abort;

/// Called once with the outcome of a fetch, on any thread.
pub type FetchDone = Box<dyn FnOnce(Result<Fetched, FetchError>) + Send + 'static>;

/// What a fetch brought.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fetched {
    pub bytes: Vec<u8>,
    pub lifetime: Lifetime,
}

impl From<Vec<u8>> for Fetched {
    /// Bytes with no lifetime.
    fn from(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            lifetime: Lifetime::Unknown,
        }
    }
}

/// How long what a fetch brought stays fresh, which a map's reloads of
/// what it draws from it follow (GL JS's reload timer, MapLibre Native's
/// revalidation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Lifetime {
    /// Fresh until then (Unix seconds), as its response's caching headers
    /// said: a map loads what it draws from it again then.
    Fresh { until: u64 },
    /// Nothing gave it a lifetime that lasts (`no-store`, `no-cache`,
    /// `max-age=0`, a file, no headers): what it shows stays until it
    /// loads again for another reason.
    #[default]
    Unknown,
    /// A cached copy past its lifetime, standing in without the server's
    /// word on it (offline, a failure): a map revalidating what it draws
    /// tries again later, as after a failure.
    Stale,
    /// A cached copy that expired then (Unix seconds), answered before the
    /// server revalidates it ([`Fetch::get_or_stale`]): a map loads it
    /// again at once.
    Expired { at: u64 },
}

impl Lifetime {
    /// A fresh response's lifetime, from when its headers said it stays
    /// fresh until (`None`: they gave none), at `now` (Unix seconds).
    pub fn given(until: Option<u64>, now: u64) -> Self {
        match until {
            Some(until) if until > now => Self::Fresh { until },
            _ => Self::Unknown,
        }
    }
}

/// What a fetched resource is for. Hosts that serve requests themselves
/// (maplibre-native-ffi's resource providers, and the rules and transforms
/// that pick requests by kind) tell them apart by it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    #[default]
    Unknown,
    /// A style document.
    Style,
    /// A source's TileJSON or GeoJSON data, or what of a tile archive
    /// finds its tiles: its header, directories and pages.
    Source,
    /// A tile, in the format given.
    Tile(TileFormat),
    Glyphs,
    SpriteImage,
    SpriteJson,
    /// An image source's image.
    Image,
}

impl ResourceKind {
    /// The `Accept` header a request for this kind sends when its host's
    /// headers have none: its formats first, then anything else at
    /// q=0.1, so a server that negotiates sends what Orbis reads best and
    /// one that offers none of them still answers. Every value is
    /// CORS-safelisted, so a browser asks no preflight for it.
    pub fn accept(self) -> Option<&'static str> {
        match self {
            Self::Tile(TileFormat::Mvt) => {
                Some("application/vnd.mapbox-vector-tile, application/x-protobuf;q=0.9, */*;q=0.1")
            }
            Self::Tile(TileFormat::Mlt) => Some("application/vnd.maplibre-tile, */*;q=0.1"),
            Self::Tile(TileFormat::Image) | Self::SpriteImage | Self::Image => {
                Some("image/webp, image/png, image/jpeg, */*;q=0.1")
            }
            Self::Tile(TileFormat::Archive)
            | Self::Unknown
            | Self::Style
            | Self::Source
            | Self::Glyphs
            | Self::SpriteJson => None,
        }
    }
}

/// What a tile fetch brings back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileFormat {
    /// A Mapbox Vector Tile.
    Mvt,
    /// A MapLibre Tile.
    Mlt,
    /// A raster or raster-dem tile: PNG, WebP or JPEG.
    Image,
    /// Bytes of a tile archive (PMTiles, MBTiles) that hold tiles, read by
    /// range: no format of their own to ask for.
    Archive,
}

/// Loads resources by URL. Each load ends when `abort` fires, if it
/// hasn't: one waiting to start never does, one in flight stops as far as
/// the host can stop it, and `done` gets [`FetchError::Aborted`] at once
/// ([`Abort::answer`] does this part).
pub trait Fetch: Send + Sync {
    /// Starts loading `url`, a `kind` of resource; `done` gets the body.
    fn get(&self, kind: ResourceKind, url: &str, abort: &Abort, done: FetchDone);
    /// As [`get`](Self::get), but a cached copy past its lifetime answers
    /// at once rather than wait for the server to revalidate it, unless
    /// its response forbade that (`must-revalidate`, `no-cache`): it comes
    /// [`Lifetime::Expired`], and the caller gets it again to revalidate
    /// it, as a map reloads a tile whose data expired (MapLibre Native's
    /// cache answers so). A host without such a cache answers as `get`
    /// does.
    fn get_or_stale(&self, kind: ResourceKind, url: &str, abort: &Abort, done: FetchDone) {
        self.get(kind, url, abort, done);
    }
    /// Reads exactly `length` bytes at `offset` (at most 64 MiB). HTTP must
    /// answer 206 with a matching Content-Range. Reject 200 before reading
    /// its body, and bound streaming reads even without Content-Length.
    fn get_range(
        &self,
        kind: ResourceKind,
        url: &str,
        offset: u64,
        length: u64,
        abort: &Abort,
        done: FetchDone,
    );
    /// Reads `read` of `url`, as [`get`](Self::get),
    /// [`get_or_stale`](Self::get_or_stale) or
    /// [`get_range`](Self::get_range) would.
    fn read(&self, kind: ResourceKind, url: &str, read: Read, abort: &Abort, done: FetchDone) {
        match read {
            Read::Whole { stale: false } => self.get(kind, url, abort, done),
            Read::Whole { stale: true } => self.get_or_stale(kind, url, abort, done),
            Read::Range(offset, length) => self.get_range(kind, url, offset, length, abort, done),
        }
    }
}

/// What a fetch reads of its URL.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Read {
    /// All of it, a stale cached copy too where `stale` says
    /// ([`Fetch::get_or_stale`]).
    Whole { stale: bool },
    /// `length` bytes at `offset` ([`Fetch::get_range`]).
    Range(u64, u64),
}

impl Read {
    /// The bytes read, `(offset, length)`, of a range.
    pub fn range(self) -> Option<(u64, u64)> {
        match self {
            Self::Whole { .. } => None,
            Self::Range(offset, length) => Some((offset, length)),
        }
    }
}

/// Tile fetches a map keeps in flight at once, by default: they take
/// turns in the map's fetch order, so that loads given up wait where
/// they can be dropped and loads wait in the view's order rather than
/// the network's. MapLibre Native keeps 20. More hold up the view where
/// many tiles load at once, even over HTTP/2: a globe wheel-zoomed in
/// fetches more, aborts more and fills the view's centre later after
/// each move with 64 in flight than with 16 (`pnpm session --fetches`).
pub const TILE_FETCHES: usize = 16;

/// [`TILE_FETCHES`], or what `ORB_FETCHES` sets: a stand-in for a host
/// with fewer connections, or more, for measuring how loads take turns.
pub fn tile_fetches() -> usize {
    std::env::var("ORB_FETCHES")
        .ok()
        .and_then(|n| n.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(TILE_FETCHES)
}

/// A request a [`NetworkHook`] can serve where the network would: one the
/// HTTP cache had no fresh copy of.
#[derive(Clone, Debug)]
pub struct NetworkRequest {
    pub kind: ResourceKind,
    /// The URL as the style names it, before any [`RequestTransform`].
    pub url: String,
    /// What the host's network would send: the URL after the transform,
    /// with its headers.
    pub request: Request,
    /// The bytes wanted, `(offset, length)`.
    pub range: Option<(u64, u64)>,
    /// The cached copy a new answer would replace, which a
    /// [`NetworkAnswer::NotModified`] keeps.
    pub prior: Option<Prior>,
    /// Fires when the requester gives the request up: the hook can stop
    /// serving it, and its answer goes nowhere.
    pub abort: Abort,
}

/// A cached copy of a resource.
#[derive(Clone, Debug, Default)]
pub struct Prior {
    pub etag: Option<String>,
    /// When it was last modified, Unix seconds.
    pub modified: Option<u64>,
    pub bytes: Vec<u8>,
}

/// How a [`NetworkHook`] answers a request.
#[derive(Clone, Debug)]
pub enum NetworkAnswer {
    /// The resource. `fresh_until` (Unix seconds) says how long the cache
    /// may answer with it; `None` keeps it out of the cache. With
    /// `must_revalidate`, a copy past it never answers before the server
    /// revalidates it ([`Fetch::get_or_stale`]).
    Body {
        bytes: Vec<u8>,
        fresh_until: Option<u64>,
        must_revalidate: bool,
        etag: Option<String>,
        /// Unix seconds.
        modified: Option<u64>,
    },
    /// The request's [`Prior`] copy is current.
    NotModified {
        fresh_until: Option<u64>,
        must_revalidate: bool,
    },
    Failed(FetchError),
    /// The host's own network loads it, as `request`.
    Network(Request),
}

/// Called once with a [`NetworkHook`]'s answer, on any thread.
pub type NetworkDone = Box<dyn FnOnce(NetworkAnswer) + Send + 'static>;

/// Serves a host's network requests in its place, as maplibre-native-ffi's
/// resource providers and asynchronous request transforms do: it sits
/// behind the HTTP cache, where the network does, and sees every URL
/// scheme, so it can serve ones the host can't load.
pub trait NetworkHook: Send + Sync {
    /// Starts serving `request`; `done` gets the answer, now or later.
    fn request(&self, request: NetworkRequest, done: NetworkDone);
}

/// What a [`RequestTransform`] makes of a URL: the URL to fetch, and
/// headers to send with it (an auth token, say).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Request {
    pub url: String,
    pub headers: Vec<(String, String)>,
}

/// Rewrites every request a host's fetch makes (GL JS `transformRequest`):
/// the request to send, or `None` to refuse it.
pub type RequestTransform = dyn Fn(&str) -> Option<Request> + Send + Sync;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchError {
    /// The resource doesn't exist (HTTP 404, a missing file). HTTP 204 is
    /// a valid empty body.
    NotFound,
    /// Invalid range, oversized body, or a server ignoring Range.
    Range(String),
    /// Any other HTTP status.
    Status(u16),
    /// `error`, and the server asked not to be asked again for `after`:
    /// HTTP 429, or a 5xx with a `Retry-After` ([`FetchError::of_status`]).
    RetryAfter {
        error: Box<FetchError>,
        after: Duration,
    },
    /// The request failed before a response.
    Network(String),
    /// The URL scheme isn't supported by this host.
    UnsupportedUrl(String),
    /// The requester gave the load up ([`Abort`]).
    Aborted,
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range(message) => write!(f, "range error: {message}"),
            Self::NotFound => f.write_str("not found"),
            Self::Status(code) => write!(f, "HTTP {code}"),
            Self::Network(message) => write!(f, "network error: {message}"),
            Self::UnsupportedUrl(url) => write!(f, "unsupported URL: {url}"),
            Self::Aborted => f.write_str("aborted"),
            Self::RetryAfter { error, after } => {
                write!(f, "{error}, retry in {} s", after.as_secs_f64().ceil())
            }
        }
    }
}

impl std::error::Error for FetchError {}

/// How long a server that rate-limits without saying for how long is
/// left alone: MapLibre Native's wait.
pub const RATE_LIMIT_WAIT: Duration = Duration::from_secs(5);

impl FetchError {
    /// What an HTTP response of `status` that brought no resource comes
    /// to, given the wait its `Retry-After` header asked for when a host
    /// could read one ([`http::retry_after`]): 404 is [`NotFound`](Self::NotFound);
    /// 429 waits as asked, else [`RATE_LIMIT_WAIT`]; a 5xx waits as asked
    /// (a 503 says for how long the server is out of service).
    pub fn of_status(status: u16, retry_after: Option<Duration>) -> Self {
        let after = match (status, retry_after) {
            (404, _) => return Self::NotFound,
            (429, after) => after.unwrap_or(RATE_LIMIT_WAIT),
            (500..=599, Some(after)) => after,
            _ => return Self::Status(status),
        };
        Self::RetryAfter {
            error: Box::new(Self::Status(status)),
            after,
        }
    }

    /// How long the server asked not to be asked again, if it did.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RetryAfter { after, .. } => Some(*after),
            _ => None,
        }
    }

    /// Whether the failure may pass by itself (the network, a server
    /// failing, timing out or rate-limiting), so that a stale copy is
    /// better than none.
    pub fn transient(&self) -> bool {
        matches!(
            self,
            Self::Network(_) | Self::Status(408 | 429 | 500..) | Self::RetryAfter { .. }
        )
    }
}

/// `url` without its credentials, query and fragment, for messages: a
/// host's logs mustn't carry its users' tokens.
pub fn redacted_url(url: &str) -> String {
    let url = url.split(['?', '#']).next().unwrap_or_default();
    match url.split_once("://") {
        Some((scheme, rest)) => {
            let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
            let host = authority.rsplit('@').next().unwrap_or_default();
            format!("{scheme}://{host}{path}")
        }
        None => url.to_string(),
    }
}

/// Validates a finite byte range before starting I/O.
pub fn range_end(offset: u64, length: u64) -> Result<u64, FetchError> {
    if length == 0 || length > 64 << 20 {
        return Err(FetchError::Range("length must be 1..=64 MiB".into()));
    }
    offset
        .checked_add(length - 1)
        .ok_or_else(|| FetchError::Range("offset overflow".into()))
}

/// Checks response headers before any range body is read.
pub fn check_range(
    status: u16,
    content_range: Option<&str>,
    offset: u64,
    length: u64,
) -> Result<(), FetchError> {
    let end = range_end(offset, length)?;
    if status != 206 {
        return Err(FetchError::Range(format!("expected 206, got {status}")));
    }
    let valid = content_range
        .and_then(|s| s.strip_prefix("bytes "))
        .and_then(|s| s.split_once('/'))
        .and_then(|(span, total)| Some((span.split_once('-')?, total)))
        .is_some_and(|((start, stop), total)| {
            start.parse::<u64>() == Ok(offset)
                && stop.parse::<u64>() == Ok(end)
                && (total == "*" || total.parse::<u64>().is_ok_and(|n| n > end))
        });
    if !valid {
        return Err(FetchError::Range("mismatched Content-Range".into()));
    }
    Ok(())
}

/// Called once with a [`Storage`] operation's outcome, on any thread.
pub type StorageDone<T> = Box<dyn FnOnce(Result<T, String>) + Send + 'static>;

/// Keeps bytes by key between runs: the HTTP cache's entries and offline
/// regions. Files in a directory natively, the origin private
/// file system on the web. Keys are `/`-separated relative paths.
pub trait Storage: Send + Sync + fmt::Debug {
    /// Reads `key` whole, or `length` bytes at `offset` of it; `None` when
    /// it doesn't exist.
    fn read(&self, key: &str, range: Option<(u64, u64)>, done: StorageDone<Option<Vec<u8>>>);
    /// Writes `bytes` to `key`, replacing it whole (readers see the old
    /// bytes or the new), or with `append` after what it holds.
    fn write(&self, key: &str, bytes: Vec<u8>, append: bool, done: StorageDone<()>);
    /// Removes `key` and every key under `key/`; absent keys are no error.
    fn remove(&self, key: &str, done: StorageDone<()>);
    /// The names directly under `prefix/` (`""` for the top), keys and
    /// prefixes alike, in no particular order.
    fn list(&self, prefix: &str, done: StorageDone<Vec<String>>);
}

/// An image decoded to straight (not premultiplied) RGBA8, rows top to
/// bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Called once with the outcome of a decode, on any thread.
pub type ImageDone = Box<dyn FnOnce(Result<DecodedImage, String>) + Send + 'static>;

/// Decodes PNG, JPEG and WebP images: the browser's decoder on the web,
/// `image` natively.
pub trait ImageDecode: Send + Sync {
    fn decode(&self, bytes: Vec<u8>, done: ImageDone);
}

/// The host's time.
pub trait Clock: Send + Sync {
    /// Monotonic time in milliseconds, for animations and fades.
    fn now(&self) -> f64;
    /// Milliseconds since the Unix epoch, for what HTTP dates: when the
    /// data a map draws expires ([`Lifetime`]).
    fn unix_ms(&self) -> f64;
}

/// A font as the parts of a CSS font shorthand, drawn at 48 px (GL JS's
/// TinySDF size for local glyphs).
#[derive(Clone, Copy, Debug)]
pub struct LocalFace<'a> {
    /// A CSS font-family list, ending in a generic family.
    pub family: &'a str,
    /// A CSS weight (`"600"`), or `None` for normal.
    pub weight: Option<&'a str>,
    /// `normal`, `italic` or `oblique`.
    pub style: &'a str,
}

/// What canvas `measureText` reports, and the ink box TinySDF draws a
/// text into: `orb-text`'s, which the glyph jobs take as they are.
pub use orb_text::sdf::InkBox;
pub use orb_text::sdf::local::TextMetrics;

/// Draws glyphs with fonts only the host can read (the "local" provider): the browser's canvas on the web, system fonts natively.
pub trait LocalGlyphs: Send + Sync + std::fmt::Debug {
    /// Canvas `measureText` of `text` in `font`.
    fn measure(&self, font: &LocalFace<'_>, text: &str) -> TextMetrics;
    /// The alpha coverage of `ink`'s box, row-major, with `text` drawn as
    /// TinySDF draws it: its origin at `(-glyph_left, glyph_top)` of the
    /// box.
    fn coverage(&self, font: &LocalFace<'_>, text: &str, ink: &InkBox) -> Vec<u8>;
}

/// Draws glyphs from font files a style declares (`font-faces`), from the
/// same bytes on every platform so labels lay out identically:
/// HarfRust shapes and outlines, at regular weight. Natively `orb-text` links in; on
/// the web a lazily loaded `shaper.wasm` holds it. Files are keyed by URL.
pub trait FontFiles: Send + Sync + std::fmt::Debug {
    /// Calls `ready` once files can be opened: at once, unless the engine
    /// loads on first use (`shaper.wasm`).
    fn prepare(&self, ready: Box<dyn FnOnce(Result<(), String>)>) {
        ready(Ok(()));
    }
    /// Opens the file `url`'s bytes; an error if they aren't a font this
    /// can read.
    fn open(&self, url: &str, bytes: &[u8]) -> Result<(), String>;
    /// Whether `url` is open here: a raster job for it needs no fetch.
    fn is_open(&self, url: &str) -> bool;
    /// `text` shaped with the open file `url` and measured as
    /// [`LocalGlyphs::measure`] measures it, at 48 px.
    fn measure(&self, url: &str, text: &str) -> TextMetrics;
    /// The coverage of `ink`'s box, as [`LocalGlyphs::coverage`].
    fn coverage(&self, url: &str, text: &str, ink: &InkBox) -> Vec<u8>;
}

#[cfg(feature = "fonts")]
/// `font-faces` files, drawn by `orb-text` as `shaper.wasm` draws them on
/// the web.
#[derive(Debug, Default)]
pub struct FontFileFaces {
    files: std::sync::Mutex<orb_text::font::Files>,
}

#[cfg(feature = "fonts")]
impl FontFileFaces {
    pub fn new() -> Self {
        Self::default()
    }

    fn files(&self) -> std::sync::MutexGuard<'_, orb_text::font::Files> {
        self.files.lock().expect("font files poisoned")
    }
}

#[cfg(feature = "fonts")]
impl FontFiles for FontFileFaces {
    fn open(&self, url: &str, bytes: &[u8]) -> Result<(), String> {
        self.files().open(url, bytes)
    }

    fn is_open(&self, url: &str) -> bool {
        self.files().is_open(url)
    }

    fn measure(&self, url: &str, text: &str) -> TextMetrics {
        self.files().measure(url, text)
    }

    fn coverage(&self, url: &str, text: &str, ink: &InkBox) -> Vec<u8> {
        self.files().coverage(url, text, ink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_headers_need_no_cors_preflight() {
        // Fetch's CORS-safelisted request-header rules for `Accept`.
        let unsafe_byte =
            |b: u8| (b < 0x20 && b != b'\t') || b == 0x7f || br#""():<>?@[\]{}"#.contains(&b);
        let formats = [
            TileFormat::Mvt,
            TileFormat::Mlt,
            TileFormat::Image,
            TileFormat::Archive,
        ];
        let kinds = formats.map(ResourceKind::Tile).into_iter().chain([
            ResourceKind::Unknown,
            ResourceKind::Style,
            ResourceKind::Source,
            ResourceKind::Glyphs,
            ResourceKind::SpriteImage,
            ResourceKind::SpriteJson,
            ResourceKind::Image,
        ]);
        for accept in kinds.filter_map(ResourceKind::accept) {
            assert!(accept.len() <= 128, "{accept}");
            assert!(!accept.bytes().any(unsafe_byte), "{accept}");
        }
    }

    #[test]
    fn statuses_ask_to_wait_as_rfc_9110_and_native_do() {
        let wait = |status, header| FetchError::of_status(status, header).retry_after();
        let minute = Some(Duration::from_secs(60));
        assert_eq!(FetchError::of_status(404, minute), FetchError::NotFound);
        assert_eq!(wait(429, minute), minute);
        assert_eq!(wait(429, None), Some(RATE_LIMIT_WAIT));
        assert_eq!(wait(503, minute), minute);
        assert_eq!(FetchError::of_status(503, None), FetchError::Status(503));
        assert_eq!(FetchError::of_status(408, minute), FetchError::Status(408));
        assert!(FetchError::of_status(429, None).transient());
        assert!(FetchError::Status(408).transient());
        assert!(!FetchError::Status(403).transient());
    }
}

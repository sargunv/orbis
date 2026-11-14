//! Host service traits: what the core asks of its platform, and the job
//! executors that run its work. Implementations live in `orb-host-native`
//! and, for the browser, `orb-web`.

#![forbid(unsafe_code)]

pub mod abort;
pub mod cache;
pub mod jobs;

use std::fmt;

pub use abort::Abort;

/// Called once with the outcome of a fetch, on any thread.
pub type FetchDone = Box<dyn FnOnce(Result<Vec<u8>, FetchError>) + Send + 'static>;

/// What a fetched resource is for. Hosts that serve requests themselves
/// (maplibre-native-ffi's resource providers, and the rules and transforms
/// that pick requests by kind) tell them apart by it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    #[default]
    Unknown,
    /// A style document.
    Style,
    /// A source's TileJSON or GeoJSON data, or a tile archive's header and
    /// directories.
    Source,
    Tile,
    Glyphs,
    SpriteImage,
    SpriteJson,
    /// An image or video source's image, or a font face file.
    Image,
}

/// Loads resources by URL. Each load ends when `abort` fires, if it
/// hasn't: one waiting to start never does, one in flight stops as far as
/// the host can stop it, and `done` gets [`FetchError::Aborted`] at once
/// ([`Abort::answer`] does this part).
pub trait Fetch: Send + Sync {
    /// Starts loading `url`, a `kind` of resource; `done` gets the body.
    fn get(&self, kind: ResourceKind, url: &str, abort: &Abort, done: FetchDone);
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
    /// may answer with it; `None` keeps it out of the cache.
    Body {
        bytes: Vec<u8>,
        fresh_until: Option<u64>,
        etag: Option<String>,
        /// Unix seconds.
        modified: Option<u64>,
    },
    /// The request's [`Prior`] copy is current.
    NotModified {
        fresh_until: Option<u64>,
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
        }
    }
}

impl std::error::Error for FetchError {}

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

/// Monotonic time in milliseconds, for zoom cross-fades.
pub trait Clock: Send + Sync {
    fn now(&self) -> f64;
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
/// HarfRust shapes, skrifa outlines. Natively `orb-text` links in; on
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

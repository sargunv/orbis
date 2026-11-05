//! Host service traits (plan §5.1): what the core asks of its platform.
//! Implementations live in `orb-host-native` and, for the browser,
//! `orb-host-web`.

#![forbid(unsafe_code)]

use std::fmt;

/// Called once with the outcome of a fetch, on any thread.
pub type FetchDone = Box<dyn FnOnce(Result<Vec<u8>, FetchError>) + Send + 'static>;

/// Loads resources by URL.
pub trait Fetch: Send + Sync {
    /// Starts loading `url`; `done` gets the body.
    fn get(&self, url: &str, done: FetchDone);
    /// Reads exactly `length` bytes at `offset` (at most 64 MiB). HTTP must
    /// answer 206 with a matching Content-Range. Reject 200 before reading
    /// its body, and bound streaming reads even without Content-Length.
    fn get_range(&self, url: &str, offset: u64, length: u64, done: FetchDone);
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchError {
    /// The resource doesn't exist (HTTP 404 or 204, a missing file).
    NotFound,
    /// Invalid range, oversized body, or a server ignoring Range.
    Range(String),
    /// Any other HTTP status.
    Status(u16),
    /// The request failed before a response.
    Network(String),
    /// The URL scheme isn't supported by this host.
    UnsupportedUrl(String),
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range(message) => write!(f, "range error: {message}"),
            Self::NotFound => f.write_str("not found"),
            Self::Status(code) => write!(f, "HTTP {code}"),
            Self::Network(message) => write!(f, "network error: {message}"),
            Self::UnsupportedUrl(url) => write!(f, "unsupported URL: {url}"),
        }
    }
}

impl std::error::Error for FetchError {}

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

/// An image decoded to straight (not premultiplied) RGBA8, rows top to
/// bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Decodes a PNG or WebP image (and JPEG where a native host enables it)
/// with the core's own decoder, exactly: DEM
/// tiles are data, and a browser's readback of an image it decoded can
/// differ by a bit here and there (privacy features perturb it), which a
/// Terrarium DEM turns into spikes of 256 m.
#[cfg(feature = "decode")]
pub fn decode_image(bytes: &[u8]) -> Result<DecodedImage, String> {
    let rgba = image::load_from_memory(bytes)
        .map_err(|e| e.to_string())?
        .into_rgba8();
    Ok(DecodedImage {
        width: rgba.width(),
        height: rgba.height(),
        pixels: rgba.into_raw(),
    })
}

/// Called once with the outcome of a decode, on any thread.
pub type ImageDone = Box<dyn FnOnce(Result<DecodedImage, String>) + Send + 'static>;

/// Decodes PNG, JPEG and WebP images (plan §10.1: the platform's decoder,
/// not one bundled in core).
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

/// What canvas `measureText` reports, in pixels: the advance and the ink's
/// `actualBoundingBox{Ascent,Descent,Left,Right}`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TextMetrics {
    pub advance: f64,
    pub ascent: f64,
    pub descent: f64,
    pub left: f64,
    pub right: f64,
}

/// A text's ink box relative to its pen origin, whole pixels, as TinySDF
/// sizes it: the text is drawn with its origin at `(-left, top)` of the box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InkRect {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
}

/// Draws glyphs with fonts only the host can read (decision 0009's "local"
/// provider): the browser's canvas on the web, system fonts natively.
pub trait LocalGlyphs: Send + Sync + std::fmt::Debug {
    /// Canvas `measureText` of `text` in `font`.
    fn measure(&self, font: &LocalFace<'_>, text: &str) -> TextMetrics;
    /// The alpha coverage of `ink`'s box, row-major, with `text` drawn as
    /// TinySDF draws it.
    fn coverage(&self, font: &LocalFace<'_>, text: &str, ink: InkRect) -> Vec<u8>;
}

/// Draws glyphs from font files a style declares (`font-faces`), from the
/// same bytes on every platform so labels lay out identically (decision
/// 0009): HarfRust shapes, skrifa outlines. Natively `orb-text` links in; on
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
    /// `text` shaped with the open file `url` and measured as
    /// [`LocalGlyphs::measure`] measures it, at 48 px.
    fn measure(&self, url: &str, text: &str) -> TextMetrics;
    /// The coverage of `ink`'s box, as [`LocalGlyphs::coverage`].
    fn coverage(&self, url: &str, text: &str, ink: InkRect) -> Vec<u8>;
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

    fn measure(&self, url: &str, text: &str) -> TextMetrics {
        let m = self.files().measure(url, text);
        TextMetrics {
            advance: m.advance,
            ascent: m.ascent,
            descent: m.descent,
            left: m.left,
            right: m.right,
        }
    }

    fn coverage(&self, url: &str, text: &str, ink: InkRect) -> Vec<u8> {
        let ink = orb_text::sdf::InkBox {
            glyph_width: ink.width,
            glyph_height: ink.height,
            glyph_top: ink.top,
            glyph_left: ink.left,
            advance: 0.0,
        };
        self.files().coverage(url, text, &ink)
    }
}

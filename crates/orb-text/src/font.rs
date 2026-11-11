//! Glyphs drawn from font bytes: HarfRust shapes a cluster,
//! skrifa outlines it, and its coverage goes through TinySDF as GL JS's
//! local path draws a canvas glyph. Declared `font-faces` files and, on
//! native hosts, system fonts draw this way; every platform computes the
//! same metrics and bitmaps from the same bytes.

use std::sync::Arc;

use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, MetadataProvider};

use crate::glyph::FontId;
pub use crate::sdf::local::TextMetrics;
use crate::sdf::{InkBox, local};

/// Font file bytes, shared without copying (a memory-mapped system font).
pub type FontBytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// A font face from file bytes (TrueType or OpenType outlines).
#[derive(Clone)]
pub struct Face {
    bytes: FontBytes,
    index: u32,
    shaping: harfrust::Font,
    id: FontId,
}

impl std::fmt::Debug for Face {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Face")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Face {
    /// The face at `index` of `bytes` (a collection's member, or 0),
    /// identified by a hash of its bytes.
    pub fn new(bytes: FontBytes, index: u32) -> Option<Self> {
        let id = FontId(hash((*bytes).as_ref(), index));
        Self::identified(bytes, index, id)
    }

    /// The face at `index` of `bytes`, with an identity the caller vouches
    /// for (a system font's path), so large files are not hashed.
    pub fn identified(bytes: FontBytes, index: u32, id: FontId) -> Option<Self> {
        FontRef::from_index((*bytes).as_ref(), index).ok()?;
        let shaping = harfrust::Font::new(Arc::clone(&bytes), index)?;
        Some(Self {
            id,
            bytes,
            index,
            shaping,
        })
    }

    /// Identifies the face by its bytes, so every worker agrees on it.
    pub fn id(&self) -> FontId {
        self.id
    }

    fn outlines(&self) -> FontRef<'_> {
        FontRef::from_index((*self.bytes).as_ref(), self.index).expect("checked at construction")
    }

    /// Whether the face maps `c` to a glyph.
    pub fn covers(&self, c: char) -> bool {
        self.outlines().charmap().map(c).is_some()
    }
}

/// FNV-1a over the bytes and face index.
fn hash(bytes: &[u8], index: u32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes.iter().chain(&index.to_le_bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// A text shaped and rasterized at one size, with the baseline origin at
/// `origin` of its coverage.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawn {
    pub metrics: TextMetrics,
    /// Coverage, row-major, `width × height`, tight around the ink.
    width: u32,
    height: u32,
    alpha: Vec<u8>,
}

struct Pen {
    commands: Vec<zeno::Command>,
    x: f32,
    y: f32,
}

impl Pen {
    fn at(&self, x: f32, y: f32) -> zeno::Point {
        // Font units point up; coverage rows go down.
        [self.x + x, self.y - y].into()
    }
}

impl OutlinePen for Pen {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.at(x, y);
        self.commands.push(zeno::Command::MoveTo(p));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.at(x, y);
        self.commands.push(zeno::Command::LineTo(p));
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (c, p) = (self.at(cx, cy), self.at(x, y));
        self.commands.push(zeno::Command::QuadTo(c, p));
    }
    fn curve_to(&mut self, ax: f32, ay: f32, bx: f32, by: f32, x: f32, y: f32) {
        let (a, b, p) = (self.at(ax, ay), self.at(bx, by), self.at(x, y));
        self.commands.push(zeno::Command::CurveTo(a, b, p));
    }
    fn close(&mut self) {
        self.commands.push(zeno::Command::Close);
    }
}

/// Shapes `text` with `face` (HarfRust, its default features, the script
/// and direction it guesses) and rasterizes it at `size` pixels, as a canvas
/// would fill it at an integer origin.
pub fn draw(face: &Face, text: &str, size: f32) -> Drawn {
    let shaper = harfrust::ShaperFont::new(&face.shaping);
    let mut buffer = harfrust::Buffer::new();
    buffer.push_str(text);
    buffer.guess_segment_properties();
    // A malformed font draws nothing, as a canvas without the glyph would.
    let shaped = harfrust::shape(&shaper, &mut buffer, harfrust::ShapeOptions::new()).is_ok();
    let scale = size / f32::from(face.shaping.units_per_em());
    let fref = face.outlines();
    let outlines = fref.outline_glyphs();
    let mut pen = Pen {
        commands: Vec::new(),
        x: 0.0,
        y: 0.0,
    };
    let mut advance = 0i32;
    if shaped {
        for (info, pos) in buffer.glyph_infos().iter().zip(buffer.glyph_positions()) {
            pen.x = (advance + pos.x_offset) as f32 * scale;
            pen.y = -(pos.y_offset as f32) * scale;
            if let Some(glyph) = outlines.get(skrifa::GlyphId::new(info.glyph_id)) {
                let settings = DrawSettings::unhinted(Size::new(size), LocationRef::default());
                let _ = glyph.draw(settings, &mut pen);
            }
            advance += pos.x_advance;
        }
    }
    let advance = f64::from(advance as f32 * scale);
    // Nothing drawn: the advance alone.
    let blank = || Drawn {
        metrics: TextMetrics {
            advance,
            ..TextMetrics::default()
        },
        width: 0,
        height: 0,
        alpha: Vec::new(),
    };
    if pen.commands.is_empty() {
        return blank();
    }
    let (mask, placement) = zeno::Mask::new(&pen.commands[..])
        .origin(zeno::Origin::TopLeft)
        .render();
    // Trim to the ink, as canvas metrics bound what is drawn.
    let (w, h) = (placement.width as usize, placement.height as usize);
    let row_inked = |y: usize| mask[y * w..(y + 1) * w].iter().any(|&a| a > 0);
    let col_inked = |x: usize| (0..h).any(|y| mask[y * w + x] > 0);
    let (Some(top), Some(left)) = (
        (0..h).find(|&y| row_inked(y)),
        (0..w).find(|&x| col_inked(x)),
    ) else {
        return blank();
    };
    let bottom = (0..h).rev().find(|&y| row_inked(y)).expect("an inked row") + 1;
    let right = (0..w)
        .rev()
        .find(|&x| col_inked(x))
        .expect("an inked column")
        + 1;
    let (width, height) = (right - left, bottom - top);
    let mut alpha = Vec::with_capacity(width * height);
    for y in top..bottom {
        alpha.extend_from_slice(&mask[y * w + left..y * w + right]);
    }
    // Pixel edges relative to the baseline origin.
    let x0 = placement.left + left as i32;
    let y0 = placement.top + top as i32;
    Drawn {
        metrics: TextMetrics {
            advance,
            ascent: f64::from(-y0),
            descent: f64::from(y0 + height as i32),
            left: f64::from(-x0),
            right: f64::from(x0 + width as i32),
        },
        width: width as u32,
        height: height as u32,
        alpha,
    }
}

impl Drawn {
    /// The coverage of `ink`'s box, as TinySDF reads it back from its
    /// canvas: the glyph drawn at `(-glyph_left, glyph_top)`, clipped.
    pub fn ink_alpha(&self, ink: &InkBox) -> Vec<u8> {
        let (bw, bh) = (ink.glyph_width as usize, ink.glyph_height as usize);
        let mut out = vec![0; bw * bh];
        // The coverage's top-left pixel, in box pixels.
        let ox = (-self.metrics.left) as i64 - i64::from(ink.glyph_left);
        let oy = i64::from(ink.glyph_top) - self.metrics.ascent as i64;
        for y in 0..i64::from(self.height) {
            let by = y + oy;
            if by < 0 || by >= bh as i64 {
                continue;
            }
            for x in 0..i64::from(self.width) {
                let bx = x + ox;
                if bx < 0 || bx >= bw as i64 {
                    continue;
                }
                out[by as usize * bw + bx as usize] =
                    self.alpha[(y as u64 * u64::from(self.width) + x as u64) as usize];
            }
        }
        out
    }
}

/// A font file's sfnt bytes: TrueType and OpenType as they are, WOFF and
/// WOFF2 decompressed (GL JS hands a `font-faces` file to the browser,
/// which reads all four).
pub(crate) fn sfnt(bytes: &[u8]) -> Result<Vec<u8>, String> {
    match bytes.get(..4) {
        Some(b"wOF2") => wuff::decompress_woff2(bytes).map_err(|e| format!("WOFF2: {e:?}")),
        Some(b"wOFF") => {
            check_woff1(bytes)?;
            wuff::decompress_woff1(bytes).map_err(|e| format!("WOFF: {e:?}"))
        }
        _ => Ok(bytes.to_vec()),
    }
}

/// Deflate's largest ratio of output to input, from the zlib manual.
const MAX_DEFLATE_RATIO: u64 = 1032;
/// The most a WOFF file may unpack to, several times any real font.
const MAX_SFNT: u64 = 128 << 20;

/// Checks a WOFF 1.0 file's table directory before wuff unpacks it. wuff
/// sizes a compressed table's buffer by the `origLength` it declares and
/// lets tables share bytes, so a 100-byte file could ask for 4 GiB (a trap
/// on wasm32) and a small one repeat a table thousands of times. Here the
/// tables must lie in the file without overlapping, none may claim to
/// inflate past deflate's 1032:1, and all together to no more than
/// [`MAX_SFNT`].
fn check_woff1(bytes: &[u8]) -> Result<(), String> {
    let word = |at: usize| {
        let b = bytes.get(at..at + 4)?;
        Some(u64::from(u32::from_be_bytes([b[0], b[1], b[2], b[3]])))
    };
    let n = bytes
        .get(12..14)
        .map_or(0, |b| usize::from(u16::from_be_bytes([b[0], b[1]])));
    let mut tables = (0..n)
        .map(|i| {
            let at = 44 + i * 20;
            Some((word(at + 4)?, word(at + 8)?, word(at + 12)?))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or("WOFF: truncated table directory")?;
    tables.sort_unstable();
    let (mut end, mut sfnt) = (44 + 20 * n as u64, 12 + 16 * n as u64);
    for (offset, compressed, original) in tables {
        if offset < end || offset + compressed > bytes.len() as u64 {
            return Err("WOFF: a table overlaps another or the file's end".into());
        }
        if compressed > original || original > compressed * MAX_DEFLATE_RATIO {
            return Err(format!(
                "WOFF: a {compressed}-byte table claims {original} bytes"
            ));
        }
        end = offset + compressed;
        sfnt += original.next_multiple_of(4);
    }
    if sfnt > MAX_SFNT {
        return Err(format!("WOFF: {sfnt} bytes unpacked, past {MAX_SFNT}"));
    }
    Ok(())
}

/// Open font files by URL, drawn as GL JS's local path draws glyphs
/// (TinySDF at 2× over the cluster's coverage), for hosts that draw
/// `font-faces` glyphs: [`measure`](Self::measure) then
/// [`coverage`](Self::coverage) of the same text draws it once.
#[derive(Debug, Default)]
pub struct Files {
    faces: std::collections::HashMap<String, Face>,
    last: Option<(String, String, Drawn)>,
}

impl Files {
    /// Opens `url`'s bytes (TrueType, OpenType, WOFF or WOFF2).
    pub fn open(&mut self, url: &str, bytes: &[u8]) -> Result<(), String> {
        let sfnt = sfnt(bytes)?;
        let face = Face::new(Arc::new(sfnt), 0).ok_or("not a font file")?;
        self.faces.insert(url.to_string(), face);
        Ok(())
    }

    /// Whether `url` is open.
    pub fn is_open(&self, url: &str) -> bool {
        self.faces.contains_key(url)
    }

    fn drawn(&mut self, url: &str, text: &str) -> Option<&Drawn> {
        let cached = self
            .last
            .as_ref()
            .is_some_and(|(u, t, _)| u == url && t == text);
        if !cached {
            let drawn = draw(self.faces.get(url)?, text, local::FONT_SIZE as f32);
            self.last = Some((url.to_string(), text.to_string(), drawn));
        }
        self.last.as_ref().map(|(_, _, d)| d)
    }

    /// `text` shaped and measured with the file `url`, as a canvas measures
    /// it; nothing for a file that isn't open.
    pub fn measure(&mut self, url: &str, text: &str) -> TextMetrics {
        self.drawn(url, text).map(|d| d.metrics).unwrap_or_default()
    }

    /// The coverage of `ink`'s box, as TinySDF reads it from its canvas.
    pub fn coverage(&mut self, url: &str, text: &str, ink: &InkBox) -> Vec<u8> {
        match self.drawn(url, text) {
            Some(drawn) => drawn.ink_alpha(ink),
            None => vec![0; (ink.glyph_width * ink.glyph_height) as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GL JS's `font-faces` fixture declares this WOFF2 file.
    #[test]
    fn opens_woff2_files() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/fonts/noto-sans-hebrew-latin-400-normal.woff2"
        );
        let bytes = std::fs::read(path).expect("the test font");
        let mut files = Files::default();
        files.open("latin", &bytes).expect("a WOFF2 font");
        let m = files.measure("latin", "á");
        assert!(m.advance > 20.0 && m.ascent > 30.0, "{m:?}");
        assert!(files.open("bad", b"<html>").is_err());
    }

    /// `sfnt` as a WOFF 1.0 file, each table zlib-compressed when that
    /// makes it smaller.
    fn woff1(sfnt: &[u8]) -> Vec<u8> {
        let u16_at = |at: usize| usize::from(u16::from_be_bytes([sfnt[at], sfnt[at + 1]]));
        let n = u16_at(4);
        let mut directory = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            let entry = &sfnt[12 + i * 16..28 + i * 16];
            let field = |j: usize| u32::from_be_bytes(entry[j..j + 4].try_into().unwrap());
            let (offset, length) = (field(8) as usize, field(12) as usize);
            let table = &sfnt[offset..offset + length];
            let deflated = miniz_oxide::deflate::compress_to_vec_zlib(table, 6);
            let stored = if deflated.len() < table.len() {
                &deflated[..]
            } else {
                table
            };
            let at = 44 + n * 20 + data.len();
            directory.extend_from_slice(&entry[..4]);
            for word in [at as u32, stored.len() as u32, length as u32, field(4)] {
                directory.extend_from_slice(&word.to_be_bytes());
            }
            data.extend_from_slice(stored);
            data.resize(data.len().next_multiple_of(4), 0);
        }
        let mut woff = b"wOFF".to_vec();
        woff.extend_from_slice(&sfnt[..4]);
        woff.extend_from_slice(&((44 + directory.len() + data.len()) as u32).to_be_bytes());
        woff.extend_from_slice(&(n as u16).to_be_bytes());
        woff.resize(44, 0);
        woff.extend(directory);
        woff.extend(data);
        woff
    }

    #[test]
    fn opens_woff1_files() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/fonts/NotoSans.ttf");
        let ttf = std::fs::read(path).expect("the test font");
        let woff = woff1(&ttf);
        assert!(woff.len() < ttf.len(), "{} of {}", woff.len(), ttf.len());
        let mut files = Files::default();
        files.open("ttf", &ttf).expect("a TrueType font");
        files.open("woff", &woff).expect("a WOFF font");
        let m = files.measure("woff", "Ag");
        assert_eq!(m, files.measure("ttf", "Ag"));
        assert!(m.advance > 50.0 && m.ascent > 20.0, "{m:?}");
    }

    /// A WOFF header and table directory of `(offset, compLength,
    /// origLength)` entries, zeros to `len` bytes.
    fn forged(tables: &[(u32, u32, u32)], len: usize) -> Vec<u8> {
        let mut woff = b"wOFF\0\x01\0\0".to_vec();
        woff.extend_from_slice(&(len as u32).to_be_bytes());
        woff.extend_from_slice(&(tables.len() as u16).to_be_bytes());
        woff.resize(44, 0);
        for (tag, &(offset, compressed, original)) in tables.iter().enumerate() {
            for word in [tag as u32, offset, compressed, original, 0] {
                woff.extend_from_slice(&word.to_be_bytes());
            }
        }
        woff.resize(len, 0);
        woff
    }

    /// wuff reserves a compressed table's `origLength` before inflating
    /// it, and copies tables that share bytes once per entry.
    #[test]
    fn woff1_tables_cannot_claim_more_than_their_bytes_make() {
        // 72 bytes whose one 8-byte table claims to inflate to 4 GiB.
        let error = sfnt(&forged(&[(64, 8, 0xffff_fff0)], 72)).expect_err("a forged WOFF");
        assert!(error.contains("claims 4294967280 bytes"), "{error}");
        // Two entries for one table's bytes.
        let error = sfnt(&forged(&[(84, 8, 8), (84, 8, 8)], 92)).expect_err("a forged WOFF");
        assert!(error.contains("overlaps"), "{error}");
        // Tables each within deflate's ratio, together past the cap.
        let tables: Vec<_> = (0..256).map(|i| (5164 + i * 1024, 1024, 1 << 20)).collect();
        let error = sfnt(&forged(&tables, 5164 + 256 * 1024)).expect_err("a forged WOFF");
        assert!(error.contains("past"), "{error}");
    }
}

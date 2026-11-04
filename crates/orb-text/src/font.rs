//! Glyphs drawn from font bytes (decision 0009): HarfRust shapes a cluster,
//! skrifa outlines it, and its coverage goes through TinySDF as GL JS's
//! local path draws a canvas glyph. Declared `font-faces` files and, on
//! native hosts, system fonts draw this way; every platform computes the
//! same metrics and bitmaps from the same bytes.

use std::sync::Arc;

use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, MetadataProvider};

use crate::glyph::{FontId, GlyphMetrics};
pub use crate::sdf::local::TextMetrics;
use crate::sdf::{InkBox, Sdf, local};

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

/// Whether the face at `index` of `bytes` maps `c` to a glyph, without
/// loading it.
pub fn covers(bytes: &[u8], index: u32, c: char) -> bool {
    FontRef::from_index(bytes, index).is_ok_and(|f| f.charmap().map(c).is_some())
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
    if pen.commands.is_empty() {
        return Drawn {
            metrics: TextMetrics {
                advance,
                ascent: 0.0,
                descent: 0.0,
                left: 0.0,
                right: 0.0,
            },
            width: 0,
            height: 0,
            alpha: Vec::new(),
        };
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
        return Drawn {
            metrics: TextMetrics {
                advance,
                ascent: 0.0,
                descent: 0.0,
                left: 0.0,
                right: 0.0,
            },
            width: 0,
            height: 0,
            alpha: Vec::new(),
        };
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
        for y in 0..self.height as i64 {
            let by = y + oy;
            if by < 0 || by >= bh as i64 {
                continue;
            }
            for x in 0..self.width as i64 {
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

/// A glyph as GL JS's local path makes it, from font bytes: TinySDF at 2×
/// over the cluster's coverage (`GlyphManager._createGlyph`).
pub fn local_glyph(face: &Face, text: &str) -> (GlyphMetrics, Sdf) {
    let drawn = draw(face, text, local::FONT_SIZE as f32);
    let ink = local::ink_box(text, &drawn.metrics);
    let sdf = local::sdf(&ink, &drawn.ink_alpha(&ink));
    let (metrics, _) = local::metrics(text, &ink);
    (metrics, sdf)
}

/// A font file's sfnt bytes: TrueType and OpenType as they are, WOFF and
/// WOFF2 decompressed (GL JS hands a `font-faces` file to the browser,
/// which reads all four).
pub fn sfnt(bytes: &[u8]) -> Result<Vec<u8>, String> {
    match bytes.get(..4) {
        Some(b"wOF2") => wuff::decompress_woff2(bytes).map_err(|e| format!("WOFF2: {e:?}")),
        Some(b"wOFF") => wuff::decompress_woff1(bytes).map_err(|e| format!("WOFF: {e:?}")),
        _ => Ok(bytes.to_vec()),
    }
}

/// Open font files by URL, drawing as [`local_glyph`] does, for hosts that
/// draw `font-faces` glyphs: [`measure`](Self::measure) then
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
}

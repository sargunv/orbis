//! v8 glyph PBF ranges (GL JS `parse_glyph_pbf.ts`): SDF glyphs at 24 px
//! with a 3 px border, 256 code points per range.

use std::fmt;

use crate::glyph::{FontStack, Glyph, GlyphKey, GlyphMetrics};

/// The SDF border around a PBF glyph's bitmap (GL JS `GLYPH_PBF_BORDER`).
pub const BORDER: f64 = BORDER_PX as f64;
const BORDER_PX: u32 = 3;

/// One glyph of a range.
#[derive(Clone, Debug, PartialEq)]
pub struct PbfGlyph {
    pub code_point: u32,
    pub metrics: GlyphMetrics,
    /// `(width + 6) × (height + 6)` SDF texels, row-major; empty when the
    /// file carries no bitmap (a space).
    pub bitmap: Vec<u8>,
}

impl PbfGlyph {
    /// The bitmap's size including the border, as GL JS sizes its image.
    pub fn bitmap_size(&self) -> [u32; 2] {
        [
            self.metrics.width as u32 + 2 * BORDER_PX,
            self.metrics.height as u32 + 2 * BORDER_PX,
        ]
    }

    /// The glyph as shaping and the atlas know it.
    pub fn glyph(&self, stack: &FontStack) -> Glyph {
        Glyph {
            key: GlyphKey::Pbf {
                stack: stack.clone(),
                code_point: self.code_point,
            },
            metrics: self.metrics,
            bitmap: self.bitmap_size(),
            double_resolution: false,
        }
    }
}

/// A parsed range, by code point.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlyphRange {
    glyphs: Vec<PbfGlyph>,
}

impl GlyphRange {
    /// Parses a range file. Glyphs from every font stack in it are kept, as
    /// GL JS keeps them; a later duplicate code point wins. A glyph whose
    /// metrics are out of range is left out, as MapLibre Native leaves it
    /// out: none is that big, and its bitmap's size would overflow.
    pub fn parse(bytes: &[u8]) -> Result<Self, PbfError> {
        let mut glyphs = Vec::new();
        let mut file = Reader::new(bytes);
        while let Some((tag, value)) = file.field()? {
            if let (1, Value::Bytes(stack)) = (tag, value) {
                let mut stack = Reader::new(stack);
                while let Some((tag, value)) = stack.field()? {
                    if let (3, Value::Bytes(glyph)) = (tag, value) {
                        glyphs.extend(parse_glyph(glyph)?);
                    }
                }
            }
        }
        glyphs.reverse();
        glyphs.sort_by_key(|g| g.code_point);
        glyphs.dedup_by_key(|g| g.code_point);
        Ok(Self { glyphs })
    }

    pub fn get(&self, code_point: u32) -> Option<&PbfGlyph> {
        self.glyphs
            .binary_search_by_key(&code_point, |g| g.code_point)
            .ok()
            .map(|i| &self.glyphs[i])
    }

    pub fn glyphs(&self) -> &[PbfGlyph] {
        &self.glyphs
    }

    /// Drops the bitmaps, keeping what shaping needs: a job worker that
    /// doesn't rasterize this range keeps only its metrics.
    pub fn without_bitmaps(mut self) -> Self {
        for glyph in &mut self.glyphs {
            glyph.bitmap = Vec::new();
        }
        self
    }
}

/// A glyph, or `None` where its code point or metrics are out of the
/// range MapLibre Native's `parseGlyphPBF` keeps: sizes and advances
/// under 256, offsets from −128 up to 128.
fn parse_glyph(bytes: &[u8]) -> Result<Option<PbfGlyph>, PbfError> {
    let mut glyph = PbfGlyph {
        code_point: 0,
        metrics: GlyphMetrics::default(),
        bitmap: Vec::new(),
    };
    // The largest size or advance, and the code point.
    let (mut largest, mut code_point) = (0, 0);
    let mut reader = Reader::new(bytes);
    while let Some((tag, value)) = reader.field()? {
        match (tag, value) {
            (1, Value::Varint(v)) => code_point = v,
            (2, Value::Bytes(b)) => glyph.bitmap = b.to_vec(),
            (3, Value::Varint(v)) => (glyph.metrics.width, largest) = (v as f64, largest.max(v)),
            (4, Value::Varint(v)) => (glyph.metrics.height, largest) = (v as f64, largest.max(v)),
            (5, Value::Varint(v)) => glyph.metrics.left = zigzag(v),
            (6, Value::Varint(v)) => glyph.metrics.top = zigzag(v),
            (7, Value::Varint(v)) => (glyph.metrics.advance, largest) = (v as f64, largest.max(v)),
            _ => {}
        }
    }
    let offsets = -128.0..128.0;
    let GlyphMetrics { left, top, .. } = glyph.metrics;
    match u32::try_from(code_point) {
        Ok(c) if c <= u32::from(char::MAX) => glyph.code_point = c,
        _ => return Ok(None),
    }
    if largest >= 256 || !offsets.contains(&left) || !offsets.contains(&top) {
        return Ok(None);
    }
    let [w, h] = glyph.bitmap_size();
    if !glyph.bitmap.is_empty() && glyph.bitmap.len() != (w * h) as usize {
        return Err(PbfError::BitmapSize(glyph.code_point));
    }
    Ok(Some(glyph))
}

fn zigzag(v: u64) -> f64 {
    let v = v as u32;
    f64::from(((v >> 1) as i32) ^ -((v & 1) as i32))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PbfError {
    Truncated,
    WireType(u8),
    /// A glyph's bitmap doesn't match its size.
    BitmapSize(u32),
}

impl fmt::Display for PbfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("glyph PBF is truncated"),
            Self::WireType(t) => write!(f, "glyph PBF has unsupported wire type {t}"),
            Self::BitmapSize(cp) => write!(f, "glyph U+{cp:04X} has a bitmap of the wrong size"),
        }
    }
}

impl std::error::Error for PbfError {}

enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Other,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn varint(&mut self) -> Result<u64, PbfError> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self.bytes.get(self.at).ok_or(PbfError::Truncated)?;
            self.at += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(PbfError::Truncated)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], PbfError> {
        let end = self.at.checked_add(n).ok_or(PbfError::Truncated)?;
        let bytes = self.bytes.get(self.at..end).ok_or(PbfError::Truncated)?;
        self.at = end;
        Ok(bytes)
    }

    fn field(&mut self) -> Result<Option<(u64, Value<'a>)>, PbfError> {
        if self.at >= self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Value::Other
            }
            2 => {
                let n = usize::try_from(self.varint()?).map_err(|_| PbfError::Truncated)?;
                Value::Bytes(self.take(n)?)
            }
            5 => {
                self.take(4)?;
                Value::Other
            }
            t => return Err(PbfError::WireType(t as u8)),
        };
        Ok(Some((key >> 3, value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    fn bytes_field(tag: u64, body: &[u8], out: &mut Vec<u8>) {
        varint(tag << 3 | 2, out);
        varint(body.len() as u64, out);
        out.extend_from_slice(body);
    }

    fn varint_field(tag: u64, v: u64, out: &mut Vec<u8>) {
        varint(tag << 3, out);
        varint(v, out);
    }

    fn glyph(cp: u32, w: u32, h: u32, left: i32, top: i32, advance: u32) -> Vec<u8> {
        let mut g = Vec::new();
        varint_field(1, cp.into(), &mut g);
        if w > 0 {
            bytes_field(2, &vec![7; ((w + 6) * (h + 6)) as usize], &mut g);
        }
        varint_field(3, w.into(), &mut g);
        varint_field(4, h.into(), &mut g);
        varint_field(5, u64::from(((left << 1) ^ (left >> 31)) as u32), &mut g);
        varint_field(6, u64::from(((top << 1) ^ (top >> 31)) as u32), &mut g);
        varint_field(7, advance.into(), &mut g);
        g
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn parses_stacks_and_glyphs() {
        let mut stack = Vec::new();
        bytes_field(1, b"Test", &mut stack);
        bytes_field(2, b"0-255", &mut stack);
        bytes_field(3, &glyph(0x61, 11, 15, 1, -12, 13), &mut stack);
        bytes_field(3, &glyph(0x20, 0, 0, 0, -26, 6), &mut stack);
        let mut file = Vec::new();
        bytes_field(1, &stack, &mut file);

        let range = GlyphRange::parse(&file).unwrap();
        assert_eq!(range.glyphs().len(), 2);
        let a = range.get(0x61).unwrap();
        assert_eq!(
            a.metrics,
            GlyphMetrics {
                width: 11.0,
                height: 15.0,
                left: 1.0,
                top: -12.0,
                advance: 13.0
            }
        );
        assert_eq!((a.bitmap_size(), a.bitmap.len()), ([17, 21], 17 * 21));
        let space = range.get(0x20).unwrap();
        assert_eq!((space.bitmap_size(), space.bitmap.len()), ([6, 6], 0));
        assert!(
            range
                .clone()
                .without_bitmaps()
                .get(0x61)
                .unwrap()
                .bitmap
                .is_empty()
        );

        assert_eq!(
            GlyphRange::parse(&file[..file.len() - 1]),
            Err(PbfError::Truncated)
        );
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn leaves_out_glyphs_out_of_range() {
        let mut stack = Vec::new();
        bytes_field(3, &glyph(0x61, 0, 0, 0, 0, 255), &mut stack);
        bytes_field(3, &glyph(0x62, 0, 0, -128, 127, 6), &mut stack);
        // Too wide, too far left, an advance too long.
        bytes_field(3, &glyph(0x63, 256, 0, 0, 0, 6), &mut stack);
        bytes_field(3, &glyph(0x64, 0, 0, -129, 0, 6), &mut stack);
        bytes_field(3, &glyph(0x65, 0, 0, 0, 0, 256), &mut stack);
        // Sizes whose bitmap's would overflow, and no code point.
        let mut huge = Vec::new();
        varint_field(3, u64::from(u32::MAX), &mut huge);
        varint_field(4, u64::from(u32::MAX), &mut huge);
        bytes_field(3, &huge, &mut stack);
        bytes_field(3, &glyph(0x11_0000, 0, 0, 0, 0, 6), &mut stack);
        let mut file = Vec::new();
        bytes_field(1, &stack, &mut file);
        let range = GlyphRange::parse(&file).unwrap();
        let kept: Vec<_> = range.glyphs().iter().map(|g| g.code_point).collect();
        assert_eq!(kept, [0x61, 0x62]);
    }
}

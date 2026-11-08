//! Signed distance fields from glyph coverage: an exact port of TinySDF
//! 2.3.0's `draw` (the version GL JS pins), minus the canvas. The platform
//! that draws a glyph supplies its coverage and measurements; everything from
//! there on is the same arithmetic in the same order, so an SDF matches
//! TinySDF's for the same coverage, on every platform.

use crate::glyph::GlyphMetrics;

const INF: f64 = 1e20;

/// TinySDF's gamma-corrected, signed squared distances for alpha 0..=254,
/// `d * |d|` with `d = 0.5 - (a / 255) ^ (1 / 2.2)`, as V8 computes them
/// (`Math.pow` differs by an ulp between libms; these are its exact bits).
/// Alpha 255 is `-INF`.
#[rustfmt::skip]
static ALPHA: [u64; 255] = [
    0x3fd0000000000000, 0x3fc684e0da5d19ab, 0x3fc36deb3458161f, 0x3fc143d249a4c426,
    0x3fbf218b2e26ce0a, 0x3fbc507dc4f68ba5, 0x3fb9e798cb68fd9d, 0x3fb7cc8e44490a46,
    0x3fb5ee6c812ce6f0, 0x3fb4418c44b7dfc6, 0x3fb2bd847fbba5ff, 0x3fb15c08a6afdfe6,
    0x3fb0183cfa9bd102, 0x3faddc95f6527849, 0x3fabb635f5c7ffda, 0x3fa9b848993e7370,
    0x3fa7de978defbcb6, 0x3fa625976654dcdc, 0x3fa48a4438a4e2d0, 0x3fa30a071981286c,
    0x3fa1a2a1e27489a2, 0x3fa0521f88ec0561, 0x3f9e2d8fa6cc7dd3, 0x3f9bde2b2a0d6c6b,
    0x3f99b35da96b91e3, 0x3f97aabcddd1b209, 0x3f95c21dbd19cf1f, 0x3f93f78bdce96bef,
    0x3f9249424787f341, 0x3f90b5a579bc86a3, 0x3f8e767ca244fdc2, 0x3f8bb16b7d592474,
    0x3f8919a234bb390c, 0x3f86acddf8bb925f, 0x3f8469090a26f979, 0x3f824c36046161db,
    0x3f80549bc40659df, 0x3f7d0123a3ff2ee5, 0x3f799d1a7e810bf9, 0x3f767a3bc17a6a31,
    0x3f7395d7bbe4e8af, 0x3f70ed6b0d200f0c, 0x3f6cfd3596a0c7c0, 0x3f688e62291a66c9,
    0x3f648a33fa312970, 0x3f60ecc19dd16cd2, 0x3f5b64b641f5940a, 0x3f55af0b29a51f61,
    0x3f50b1ee10a1d360, 0x3f48ce4b8f008ccb, 0x3f41919a6be91f06, 0x3f37453566232a54,
    0x3f2bdabe6e2e3740, 0x3f1c1e1cf217f346, 0x3f0403f57ad9f4b4, 0x3ed1809758f15856,
    0xbed1aad66da42e88, 0xbf038f07645e22b3, 0xbf1adf2e8d8050dd, 0xbf2a1292cb8cd8b8,
    0xbf3557e2765a7b66, 0xbf3f95bf49f4ee99, 0xbf45dacf90cdbacc, 0xbf4cd567aadb61a2,
    0xbf525a48d0ba6815, 0xbf56b93b329f8c09, 0xbf5b84bd56723f2b, 0xbf605d0eeda44aeb,
    0xbf632b6310866ffb, 0xbf662c1c6fca44ee, 0xbf695e086ad0cee3, 0xbf6cbfffc6f41bca,
    0xbf7028730e178d78, 0xbf7207d4a590e97d, 0xbf73fda07ddda061, 0xbf76095711f12d3a,
    0xbf782a7d4ac3220c, 0xbf7a609c49c56f36, 0xbf7cab413689581d, 0xbf7f09fd0f58ea2b,
    0xbf80be323e47d211, 0xbf820107d340aae6, 0xbf834d4d06e11f45, 0xbf84a2d132d3819b,
    0xbf860165386b3125, 0xbf8768db6fe4bf89, 0xbf88d907988d2db1, 0xbf8a51bec9c14524,
    0xbf8bd2d764b736ce, 0xbf8d5c290705af4d, 0xbf8eed8c7ddc8315, 0xbf90436ddcf1f93e,
    0xbf9113f8e1dc28d8, 0xbf91e855587c4c7d, 0xbf92c071ccef8804, 0xbf939c3d47adf0da,
    0xbf947ba748d4fc8b, 0xbf955e9fc3ab865e, 0xbf9645171a5c1c27, 0xbf972efe19e27bcf,
    0xbf981c45f6295f07, 0xbf990ce04655e309, 0xbf9a00bf013e07b7, 0xbf9af7d47a07eaae,
    0xbf9bf2135cef8958, 0xbf9cef6eac30f876, 0xbf9defd9bd1521a2, 0xbf9ef348351f3513,
    0xbf9ff9ae07591ade, 0xbfa0817fb8dea508, 0xbfa107987d5e41c5, 0xbfa18f1bb76f0346,
    0xbfa21803ef3c754c, 0xbfa2a24bcdc2ae66, 0xbfa32dee1bc24995, 0xbfa3bae5c0bf2c71,
    0xbfa4492dc2099392, 0xbfa4d8c141d0e681, 0xbfa5699b7e3fdbf0, 0xbfa5fbb7d0a17d08,
    0xbfa68f11ac8e9d8a, 0xbfa723a49f2363a4, 0xbfa7b96c4e3c807d, 0xbfa8506477bbbf34,
    0xbfa8e888f0d39510, 0xbfa981d5a55961c6, 0xbfaa1c46971e13d7, 0xbfaab7d7dd4ce7e5,
    0xbfab5485a3cffed2, 0xbfabf24c2aba889f, 0xbfac9127c5b84703, 0xbfad3114db822c34,
    0xbfadd20fe557de80, 0xbfae74156e7deb89, 0xbfaf172213c07874, 0xbfafbb3282fa3fc4,
    0xbfb03021bd4fd737, 0xbfb08328e4a6fab5, 0xbfb0d6ad26aef30d, 0xbfb12aacfa3d3921,
    0xbfb17f26dd9d5525, 0xbfb1d419565f8ac6, 0xbfb22982f12921af, 0xbfb27f62418639d1,
    0xbfb2d5b5e1bd1c4c, 0xbfb32c7c72a2f9cf, 0xbfb383b49b7208bd, 0xbfb3db5d09a0f546,
    0xbfb4337470bb96cd, 0xbfb48bf98a3cde2d, 0xbfb4e4eb1569f147, 0xbfb53e47d72e6864,
    0xbfb5980e99f9a2dd, 0xbfb5f23e2d9d28bc, 0xbfb64cd5672c0f4b, 0xbfb6a7d320db5771,
    0xbfb7033639e33cc5, 0xbfb75efd96616c64, 0xbfb7bb281f3c1bdc, 0xbfb817b4c205f7c1,
    0xbfb874a270e2e258, 0xbfb8d1f0226d7b8c, 0xbfb92f9cd19d6a80, 0xbfb98da77dae6288,
    0xbfb9ec0f2a07dd06, 0xbfba4ad2de2581c7, 0xbfbaa9f1a5803827, 0xbfbb096a8f77db50,
    0xbfbb693caf3d8bdb, 0xbfbbc9671bbe99fa, 0xbfbc29e8ef9002ac, 0xbfbc8ac148da7b80,
    0xbfbcebef49470822, 0xbfbd4d7215ec14df, 0xbfbdaf48d73b128d, 0xbfbe1172b8ee8ed5,
    0xbfbe73eee9f8c590, 0xbfbed6bc9c72a6f5, 0xbfbf39db058b4f19, 0xbfbf9d495d77eb05,
    0xbfc000836fb20420, 0xbfc0328964b12585, 0xbfc064b62e2ec447, 0xbfc097096e0520d6,
    0xbfc0c982c76c3dcb, 0xbfc0fc21def2f984, 0xbfc12ee65a785405, 0xbfc161cfe124df63,
    0xbfc194de1b6459c3, 0xbfc1c810b2df6f78, 0xbfc1fb675275a43d, 0xbfc22ee1a6376249,
    0xbfc2627f5b602e12, 0xbfc296402050fdd8, 0xbfc2ca23a48ab3cb, 0xbfc2fe2998a8b9df,
    0xbfc33251ae5bbe3e, 0xbfc3669b98648f8e, 0xbfc39b070a8f1801, 0xbfc3cf93b9ad765f,
    0xbfc404415b933431, 0xbfc4390fa710984a, 0xbfc46dfe53ee14ad, 0xbfc4a30d1ae7cf61,
    0xbfc4d83bb5a9453b, 0xbfc50d89dec905ce, 0xbfc542f751c48846, 0xbfc57883cafc17e2,
    0xbfc5ae2f07aed7e6, 0xbfc5e3f8c5f6de34, 0xbfc619e0c4c563d6, 0xbfc64fe6c3df0b2e,
    0xbfc6860a83d83ad3, 0xbfc6bc4bc6118ceb, 0xbfc6f2aa4cb4525a, 0xbfc72925daaf290a,
    0xbfc75fbe33b2a52e, 0xbfc796731c2e0cbe, 0xbfc7cd44594c24b8, 0xbfc80431b0f00fce,
    0xbfc83b3ae9b23df3, 0xbfc8725fcadd6c79, 0xbfc8a9a01c6bb625, 0xbfc8e0fba703b2f9,
    0xbfc9187233f5a752, 0xbfc950038d38c1bc, 0xbfc987af7d686787, 0xbfc9bf75cfc18f40,
    0xbfc9f75650202931, 0xbfca2f50cafc9537, 0xbfca67650d6925bd, 0xbfca9f92e50faf93,
    0xbfcad7da202f263a, 0xbfcb103a8d994462, 0xbfcb48b3fcb0405f, 0xbfcb81463d648c26,
    0xbfcbb9f12032a0b1, 0xbfcbf2b47620d468, 0xbfcc2b9010bd3c6f, 0xbfcc6483c21b986b,
    0xbfcc9d8f5cd34897, 0xbfccd6b2b3fd4dfd, 0xbfcd0fed9b325471, 0xbfcd493fe688c628,
    0xbfcd82a96a92e8bb, 0xbfcdbc29fc5d033a, 0xbfcdf5c1716b8d5e, 0xbfce2f6f9fb96759,
    0xbfce69345db61a2f, 0xbfcea30f82442086, 0xbfcedd00e4b73794, 0xbfcf17085cd2b7f8,
    0xbfcf5125c2c7f676, 0xbfcf8b58ef34ac58, 0xbfcfc5a1bb216734,
];

fn alpha_distance(a: u8) -> f64 {
    match ALPHA.get(usize::from(a)) {
        Some(&bits) => f64::from_bits(bits),
        None => -INF,
    }
}

/// A glyph's ink box as TinySDF measures it from a text engine's metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InkBox {
    pub glyph_width: u32,
    pub glyph_height: u32,
    /// Ascent above the baseline, whole pixels.
    pub glyph_top: i32,
    /// Ink left edge relative to the origin, whole pixels.
    pub glyph_left: i32,
    /// Horizontal advance, unrounded.
    pub advance: f64,
}

impl InkBox {
    /// TinySDF's box from canvas `TextMetrics` (`width`,
    /// `actualBoundingBox{Ascent,Descent,Left,Right}`), clipped to
    /// `max_glyph_dim`.
    pub(crate) fn from_text_metrics(
        advance: f64,
        ascent: f64,
        descent: f64,
        left: f64,
        right: f64,
        max_glyph_dim: u32,
    ) -> Self {
        let glyph_top = ascent.ceil();
        let glyph_left = (-left).floor();
        let max = f64::from(max_glyph_dim);
        let glyph_width = (right.ceil() - glyph_left).min(max).max(0.0);
        let glyph_height = (glyph_top + descent.ceil()).min(max).max(0.0);
        Self {
            glyph_width: glyph_width as u32,
            glyph_height: glyph_height as u32,
            glyph_top: glyph_top as i32,
            glyph_left: glyph_left as i32,
            advance,
        }
    }
}

/// An SDF bitmap, row-major, one byte per texel.
#[derive(Clone, Debug, PartialEq)]
pub struct Sdf {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// TinySDF's `draw` from the glyph's coverage: `alpha` holds
/// `glyph_width × glyph_height` alpha bytes of the ink box, drawn with the
/// box's top-left at the pen offset TinySDF uses.
pub(crate) fn tiny_sdf(ink: &InkBox, alpha: &[u8], buffer: u32, radius: f64, cutoff: f64) -> Sdf {
    let (gw, gh) = (ink.glyph_width as usize, ink.glyph_height as usize);
    let buffer = buffer as usize;
    let width = gw + 2 * buffer;
    let height = gh + 2 * buffer;
    let len = width * height;
    let mut sdf = Sdf {
        width: width as u32,
        height: height as u32,
        data: vec![0; len],
    };
    if gw == 0 || gh == 0 {
        return sdf;
    }
    assert_eq!(alpha.len(), gw * gh, "coverage must cover the ink box");

    let mut outer = vec![INF; len];
    let mut inner = vec![0.0; len];
    for y in 0..gh {
        for x in 0..gw {
            let a = alpha[y * gw + x];
            if a == 0 {
                continue;
            }
            let j = (y + buffer) * width + buffer + x;
            let t = alpha_distance(a);
            outer[j] = js_max0(t);
            inner[j] = js_max0(-t);
        }
    }

    let side = width.max(height) + 1;
    let mut scratch = Scratch {
        f: vec![0.0; side],
        z: vec![0.0; side + 1],
        v: vec![0; side],
    };
    edt(&mut outer, 0, 0, width, height, width, &mut scratch, buffer);
    let pad = buffer.min(1);
    edt(
        &mut inner,
        buffer - pad,
        buffer - pad,
        gw + 2 * pad,
        gh + 2 * pad,
        width,
        &mut scratch,
        0,
    );

    let scale = 255.0 / radius;
    let base = 255.0 * (1.0 - cutoff);
    for i in 0..len {
        let d = outer[i].sqrt() - inner[i].sqrt();
        sdf.data[i] = to_uint8_clamped(base - scale * d);
    }
    sdf
}

/// `Math.max(0, t)`.
fn js_max0(t: f64) -> f64 {
    if t.is_nan() {
        f64::NAN
    } else if t > 0.0 {
        t
    } else {
        0.0
    }
}

/// A `Uint8ClampedArray` store: clamp, then round half to even.
fn to_uint8_clamped(x: f64) -> u8 {
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    if x >= 255.0 {
        return 255;
    }
    let f = x.floor();
    let rounded = if x - f > 0.5 || (x - f == 0.5 && f % 2.0 != 0.0) {
        f + 1.0
    } else {
        f
    };
    rounded as u8
}

struct Scratch {
    f: Vec<f64>,
    z: Vec<f64>,
    v: Vec<usize>,
}

/// 2D squared Euclidean distance transform (Felzenszwalb and Huttenlocher),
/// skipping `skip_cols` seedless columns at each side, as TinySDF does.
#[allow(clippy::too_many_arguments)]
fn edt(
    data: &mut [f64],
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    grid: usize,
    s: &mut Scratch,
    skip_cols: usize,
) {
    for x in x0 + skip_cols..(x0 + width).saturating_sub(skip_cols) {
        edt1d(data, y0 * grid + x, grid, height, s);
    }
    for y in y0..y0 + height {
        edt1d(data, y * grid + x0, 1, width, s);
    }
}

fn edt1d(grid: &mut [f64], offset: usize, stride: usize, length: usize, s: &mut Scratch) {
    let Scratch { f, z, v } = s;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    f[0] = grid[offset];
    let mut k: isize = 0;
    for q in 1..length {
        let fq = grid[offset + q * stride];
        f[q] = fq;
        let q2 = (q * q) as f64;
        let mut sv;
        loop {
            let r = v[k as usize];
            sv = (fq - f[r] + q2 - (r * r) as f64) / (q as f64 - r as f64) / 2.0;
            if sv <= z[k as usize] {
                k -= 1;
                if k > -1 {
                    continue;
                }
            }
            break;
        }
        k += 1;
        v[k as usize] = q;
        z[k as usize] = sv;
        z[k as usize + 1] = INF;
    }
    let mut k = 0;
    for q in 0..length {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let r = v[k];
        let qr = q as f64 - r as f64;
        grid[offset + q * stride] = f[r] + qr * qr;
    }
}

/// How GL JS draws local glyphs: TinySDF at 2× (48 px), buffer 6, radius
/// 16, cutoff 0.25 (`GlyphManager._createTinySDF`).
pub mod local {
    use super::*;

    pub const FONT_SIZE: f64 = 48.0;
    pub(crate) const BUFFER: u32 = 6;
    pub const RADIUS: f64 = 16.0;
    pub(crate) const CUTOFF: f64 = 0.25;
    const TEXTURE_SCALE: f64 = 2.0;
    const TOP_ADJUSTMENT: f64 = 27.5;
    const LEFT_ADJUSTMENT: f64 = 0.5;

    /// The largest ink box TinySDF keeps: 66 px for a code point, 120 px
    /// for a cluster, whose canvas is three ems wide.
    pub(crate) fn max_glyph_dim(cluster: bool) -> u32 {
        if cluster { 120 } else { 66 }
    }

    /// `x || fallback`.
    fn or(x: f64, fallback: f64) -> f64 {
        if x == 0.0 || x.is_nan() { fallback } else { x }
    }

    /// What canvas `measureText` reports for a text, in pixels at
    /// [`FONT_SIZE`]: the advance and the ink's
    /// `actualBoundingBox{Ascent,Descent,Left,Right}`.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct TextMetrics {
        pub advance: f64,
        pub ascent: f64,
        pub descent: f64,
        pub left: f64,
        pub right: f64,
    }

    /// The ink box TinySDF draws `text` into from its measurements.
    pub fn ink_box(text: &str, m: &TextMetrics) -> InkBox {
        InkBox::from_text_metrics(
            m.advance,
            m.ascent,
            m.descent,
            m.left,
            m.right,
            max_glyph_dim(crate::segment::is_cluster(text)),
        )
    }

    /// TinySDF's SDF of `alpha`, the coverage of `ink`'s box.
    pub fn sdf(ink: &InkBox, alpha: &[u8]) -> Sdf {
        tiny_sdf(ink, alpha, BUFFER, RADIUS, CUTOFF)
    }

    /// A drawn glyph's layout metrics and bitmap size (GL JS
    /// `_createGlyph`). Invisible format characters advance nothing.
    pub fn metrics(text: &str, ink: &InkBox) -> (GlyphMetrics, [u32; 2]) {
        let control = !text.is_empty() && text.chars().all(crate::unicode::is_format);
        let metrics = GlyphMetrics {
            width: if control {
                0.0
            } else {
                or(f64::from(ink.glyph_width) / TEXTURE_SCALE, 24.0)
            },
            height: or(f64::from(ink.glyph_height) / TEXTURE_SCALE, 24.0),
            left: or(
                f64::from(ink.glyph_left) / TEXTURE_SCALE + LEFT_ADJUSTMENT,
                0.0,
            ),
            top: or(
                f64::from(ink.glyph_top) / TEXTURE_SCALE - TOP_ADJUSTMENT,
                -8.0,
            ),
            advance: if control {
                0.0
            } else {
                or(ink.advance / TEXTURE_SCALE, 24.0)
            },
        };
        let width = ink.glyph_width + 2 * BUFFER;
        let height = ink.glyph_height + 2 * BUFFER;
        (metrics, [width, height])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn gamma_table_is_tinysdf() {
        // Spot checks against `d * |d|`, `d = 0.5 - (i / 255) ^ (1 / 2.2)`.
        assert_eq!(alpha_distance(0), 0.25);
        assert_eq!(alpha_distance(255), -INF);
        assert_eq!(alpha_distance(128), -0.053379226919785375);
        for a in 1..255u8 {
            assert!(alpha_distance(a) < alpha_distance(a - 1));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn sdf_of_a_square() {
        let ink = InkBox {
            glyph_width: 8,
            glyph_height: 8,
            glyph_top: 8,
            glyph_left: 0,
            advance: 10.0,
        };
        let sdf = tiny_sdf(&ink, &[255; 64], 3, 8.0, 0.25);
        assert_eq!((sdf.width, sdf.height), (14, 14));
        let at = |x: usize, y: usize| sdf.data[y * 14 + x];
        // Inside saturates; the corner is √18 texels out, still within the
        // radius: 191.25 − 31.875 × 4.243 = 56.0. The field is symmetric
        // about the square's centre.
        assert_eq!(at(7, 7), 255);
        assert_eq!(at(0, 0), 56);
        for y in 0..14 {
            for x in 0..14 {
                assert_eq!(at(x, y), at(13 - x, y));
                assert_eq!(at(x, y), at(x, 13 - y));
            }
        }
        // One texel inside the edge: base + 255 / radius × 1 = 223.125.
        assert_eq!(at(3, 7), 223);
        // One texel outside: base − 255 / 8 = 159.375.
        assert_eq!(at(2, 7), 159);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn clamped_rounding_is_half_to_even() {
        assert_eq!(to_uint8_clamped(0.5), 0);
        assert_eq!(to_uint8_clamped(1.5), 2);
        assert_eq!(to_uint8_clamped(2.5), 2);
        assert_eq!(to_uint8_clamped(254.6), 255);
        assert_eq!(to_uint8_clamped(-3.0), 0);
        assert_eq!(to_uint8_clamped(f64::NAN), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn local_glyph_metrics() {
        let ink =
            InkBox::from_text_metrics(48.0, 41.2, 6.1, -1.5, 45.3, local::max_glyph_dim(false));
        assert_eq!(
            (
                ink.glyph_top,
                ink.glyph_left,
                ink.glyph_width,
                ink.glyph_height
            ),
            (42, 1, 45, 49)
        );
        let (metrics, bitmap) = local::metrics("東", &ink);
        assert_eq!(
            metrics,
            GlyphMetrics {
                width: 22.5,
                height: 24.5,
                left: 1.0,
                top: -6.5,
                advance: 24.0
            }
        );
        assert_eq!(bitmap, [57, 61]);
        let (metrics, _) = local::metrics("\u{200d}", &ink);
        assert_eq!((metrics.width, metrics.advance), (0.0, 0.0));
    }
}

//! The pixelmatch MapLibre Native's render-test runner links:
//! [pixelmatch-cpp](https://github.com/mapbox/pixelmatch-cpp) at
//! `61f433cb` (ISC, Copyright Mapbox; see LICENSE-pixelmatch-cpp), the
//! submodule of the pinned commit's `vendor/maplibre-native-base`. It
//! predates pixelmatch's OKLab metric: YIQ distance, alpha blended with
//! white, the 2009 anti-aliasing detector, `includeAA` false. Native calls
//! it on premultiplied pixels with threshold 0.1285. The same `f64` math in
//! the same order, and the same truncating conversions to bytes.

/// C++ `blend`: a channel blended with white, the `double` truncated to a
/// byte.
fn blend(c: u8, a: f64) -> u8 {
    (255.0 + f64::from(i32::from(c) - 255) * a) as u8
}

fn rgb2y(r: u8, g: u8, b: u8) -> f64 {
    f64::from(r) * 0.29889531 + f64::from(g) * 0.58662247 + f64::from(b) * 0.11448223
}

fn rgb2i(r: u8, g: u8, b: u8) -> f64 {
    f64::from(r) * 0.59597799 - f64::from(g) * 0.27417610 - f64::from(b) * 0.32180189
}

fn rgb2q(r: u8, g: u8, b: u8) -> f64 {
    f64::from(r) * 0.21147017 - f64::from(g) * 0.52261711 + f64::from(b) * 0.31114694
}

/// The YIQ difference of pixel `k` of `a` and pixel `m` of `b` (byte
/// offsets); with `y_only`, the signed brightness difference.
fn color_delta(a: &[u8], b: &[u8], k: usize, m: usize, y_only: bool) -> f64 {
    let a1 = f64::from(a[k + 3]) / 255.0;
    let a2 = f64::from(b[m + 3]) / 255.0;
    let (r1, g1, b1) = (blend(a[k], a1), blend(a[k + 1], a1), blend(a[k + 2], a1));
    let (r2, g2, b2) = (blend(b[m], a2), blend(b[m + 1], a2), blend(b[m + 2], a2));
    let y = rgb2y(r1, g1, b1) - rgb2y(r2, g2, b2);
    if y_only {
        return y;
    }
    let i = rgb2i(r1, g1, b1) - rgb2i(r2, g2, b2);
    let q = rgb2q(r1, g1, b1) - rgb2q(r2, g2, b2);
    0.5053 * y * y + 0.299 * i * i + 0.1957 * q * q
}

fn gray_pixel(img: &[u8], i: usize) -> f64 {
    let a = f64::from(img[i + 3]) / 255.0;
    rgb2y(blend(img[i], a), blend(img[i + 1], a), blend(img[i + 2], a))
}

/// Whether pixel `(x1, y1)` of `img` is likely anti-aliasing (Vysniauskas,
/// 2009). `other` is `None` in the recursive calls, as C++'s `nullptr`.
fn antialiased(
    img: &[u8],
    x1: usize,
    y1: usize,
    width: usize,
    height: usize,
    other: Option<&[u8]>,
) -> bool {
    let x0 = x1.saturating_sub(1);
    let y0 = y1.saturating_sub(1);
    let x2 = (x1 + 1).min(width - 1);
    let y2 = (y1 + 1).min(height - 1);
    let pos = (y1 * width + x1) * 4;
    let (mut zeroes, mut positives, mut negatives) = (0u32, 0u32, 0u32);
    let (mut min, mut max) = (0.0, 0.0);
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (0, 0, 0, 0);
    for x in x0..=x2 {
        for y in y0..=y2 {
            if x == x1 && y == y1 {
                continue;
            }
            let delta = color_delta(img, img, pos, (y * width + x) * 4, true);
            if delta == 0.0 {
                zeroes += 1;
            } else if delta < 0.0 {
                negatives += 1;
            } else if delta > 0.0 {
                positives += 1;
            }
            if zeroes > 2 {
                return false;
            }
            if other.is_none() {
                continue;
            }
            if delta < min {
                min = delta;
                (min_x, min_y) = (x, y);
            }
            if delta > max {
                max = delta;
                (max_x, max_y) = (x, y);
            }
        }
    }
    let Some(other) = other else {
        return true;
    };
    if negatives == 0 || positives == 0 {
        return false;
    }
    (!antialiased(img, min_x, min_y, width, height, None)
        && !antialiased(other, min_x, min_y, width, height, None))
        || (!antialiased(img, max_x, max_y, width, height, None)
            && !antialiased(other, max_x, max_y, width, height, None))
}

/// `mapbox::pixelmatch(a, b, width, height, output, threshold)`: the
/// mismatched pixel count and the diff image.
pub(crate) fn compare(
    a: &[u8],
    b: &[u8],
    width: usize,
    height: usize,
    threshold: f64,
) -> (usize, Vec<u8>) {
    assert_eq!(a.len(), width * height * 4);
    assert_eq!(a.len(), b.len());
    let max_delta = 35215.0 * threshold * threshold;
    let mut output = vec![0; a.len()];
    let mut diff = 0;
    for y in 0..height {
        for x in 0..width {
            let pos = (y * width + x) * 4;
            let color = if color_delta(a, b, pos, pos, false) > max_delta {
                if antialiased(a, x, y, width, height, Some(b))
                    || antialiased(b, x, y, width, height, Some(a))
                {
                    [255, 255, 0, 255]
                } else {
                    diff += 1;
                    [255, 0, 0, 255]
                }
            } else {
                // C++ passes the `double` to `blend(uint8_t, double)`.
                let gray = blend(gray_pixel(a, pos) as u8, 0.1);
                [gray, gray, gray, 255]
            };
            output[pos..pos + 4].copy_from_slice(&color);
        }
    }
    (diff, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_real_differences_and_ignores_antialiasing() {
        let white = [255u8; 4];
        let black = [0, 0, 0, 255];
        // A black square on white.
        let mut square = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                let inside = (2..6).contains(&x) && (2..6).contains(&y);
                square.extend(if inside { black } else { white });
            }
        }
        let (count, diff) = compare(&square, &square, 8, 8, 0.1285);
        assert_eq!(count, 0);
        assert_eq!(diff[..4], [255; 4]);
        // A different colour everywhere: every pixel counts.
        let red = [255, 0, 0, 255].repeat(64);
        let blue = [0, 0, 255, 255].repeat(64);
        assert_eq!(compare(&red, &blue, 8, 8, 0.1285).0, 64);
        // Premultiplied transparency blends with white: transparent black
        // matches white.
        let clear = [0u8; 4].repeat(64);
        let opaque_white = white.repeat(64);
        assert_eq!(compare(&clear, &opaque_white, 8, 8, 0.1285).0, 0);
    }
}

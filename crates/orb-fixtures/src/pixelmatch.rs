//! Pixelmatch 8.0.0 (ISC, Copyright Mapbox; see LICENSE-pixelmatch).
//!
//! OKLab HyAB with toe-corrected lightness, the upstream lookup tables,
//! checkerboard alpha compositing and includeAA=false. These are the defaults
//! used by the pinned GL JS harness; it supplies the per-test threshold.

use std::sync::OnceLock;

/// Returns the mismatched pixel count and an opaque RGBA diff image.
pub(crate) fn compare(
    a: &[u8],
    b: &[u8],
    width: usize,
    height: usize,
    threshold: f64,
) -> (usize, Vec<u8>) {
    assert_eq!(a.len(), width * height * 4);
    assert_eq!(a.len(), b.len());
    let mut output = vec![0; a.len()];
    let mut count = 0;
    let tables = TABLES.get_or_init(Tables::new);
    for y in 0..height {
        for x in 0..width {
            let p = (y * width + x) * 4;
            let color = if a[p..p + 4] != b[p..p + 4] && tables.exceeds(a, b, p, threshold) {
                if antialiased(a, b, x, y, width, height) || antialiased(b, a, x, y, width, height)
                {
                    [255, 255, 0, 255]
                } else {
                    count += 1;
                    [255, 0, 0, 255]
                }
            } else {
                // JS writes to a Uint8Array: fractional bytes truncate.
                let gray =
                    (255.0 + (luma(rgb(a, p)) - 255.0) * 0.1 * f64::from(a[p + 3]) / 255.0) as u8;
                [gray, gray, gray, 255]
            };
            output[p..p + 4].copy_from_slice(&color);
        }
    }
    (count, output)
}

fn rgb(image: &[u8], p: usize) -> [f64; 3] {
    std::array::from_fn(|i| f64::from(image[p + i]))
}

fn luma([r, g, b]: [f64; 3]) -> f64 {
    r * 0.29889531 + g * 0.58662247 + b * 0.11448223
}

struct Tables {
    linear: [f64; 257],
    cube_root: [f64; 4098],
}
static TABLES: OnceLock<Tables> = OnceLock::new();

impl Tables {
    fn new() -> Self {
        Self {
            linear: std::array::from_fn(|i| {
                let c = i.min(255) as f64 / 255.0;
                if c <= 0.04045 {
                    c / 12.92
                } else {
                    ((c + 0.055) / 1.055).powf(2.4)
                }
            }),
            cube_root: std::array::from_fn(|i| (i as f64 / 4096.0).cbrt()),
        }
    }
    fn linear(&self, x: f64) -> f64 {
        let i = x as usize;
        self.linear[i] + (self.linear[i + 1] - self.linear[i]) * (x - i as f64)
    }
    fn cbrt(&self, x: f64) -> f64 {
        let t = x * 4096.0;
        let i = t as usize;
        if i < 8 {
            x.cbrt()
        } else {
            self.cube_root[i] + (self.cube_root[i + 1] - self.cube_root[i]) * (t - i as f64)
        }
    }
    // Return cube-rooted LMS and toe-corrected OKLab lightness. The JS
    // opaque RGB cache is only an optimization; both paths use these tables.
    fn color(&self, image: &[u8], p: usize, background: [f64; 3]) -> [f64; 4] {
        let alpha = f64::from(image[p + 3]);
        let rgb: [f64; 3] = std::array::from_fn(|c| {
            if alpha == 255.0 {
                self.linear[usize::from(image[p + c])]
            } else {
                self.linear(
                    (f64::from(image[p + c]) * alpha + background[c] * (255.0 - alpha)) / 255.0,
                )
            }
        });
        let [r, g, b] = rgb;
        let l = self.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
        let m = self.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
        let s = self.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
        let lightness = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
        let k3 = (1.0 + 0.206) / (1.0 + 0.03);
        let x = k3 * lightness - 0.206;
        let lr = 0.5 * (x + (x * x + 4.0 * 0.03 * k3 * lightness).sqrt());
        [l, m, s, lr]
    }
    fn exceeds(&self, a: &[u8], b: &[u8], p: usize, threshold: f64) -> bool {
        let hash = (p as u32 >> 2).wrapping_mul(0x9e3779b1);
        let bg = [
            48.0 + 159.0 * f64::from(hash >> 31),
            48.0 + 159.0 * f64::from((hash >> 30) & 1),
            48.0 + 159.0 * f64::from((hash >> 29) & 1),
        ];
        let c1 = self.color(a, p, bg);
        let c2 = self.color(b, p, bg);
        let [dl, dm, ds, dlr] = std::array::from_fn(|i| c1[i] - c2[i]);
        let rest = threshold - dlr.abs();
        if rest < 0.0 {
            return true;
        }
        let da = 1.9779984951 * dl - 2.4285922050 * dm + 0.4505937099 * ds;
        let db = 0.0259040371 * dl + 0.7827717662 * dm - 0.8086757660 * ds;
        da * da + db * db > rest * rest
    }
}

// The AA detector deliberately uses gamma-space Rec.601 luma over white.
// When premultiplied luma cancels, an alpha difference gives ramp direction.
fn brightness_delta(a: &[u8], p: usize, q: usize) -> f64 {
    if a[p..p + 4] == a[q..q + 4] {
        return 0.0;
    }
    let a1 = f64::from(a[p + 3]);
    let a2 = f64::from(a[q + 3]);
    let da = a1 - a2;
    let [r1, g1, b1] = rgb(a, p);
    let [r2, g2, b2] = rgb(a, q);
    if a1 < 255.0 || a2 < 255.0 {
        let d = luma([
            (r1 * a1 - r2 * a2 - 255.0 * da) / 255.0,
            (g1 * a1 - g2 * a2 - 255.0 * da) / 255.0,
            (b1 * a1 - b2 * a2 - 255.0 * da) / 255.0,
        ]);
        if d == 0.0 && da != 0.0 { da / 2.0 } else { d }
    } else {
        luma([r1 - r2, g1 - g2, b1 - b2])
    }
}

fn bounds(x: usize, y: usize, width: usize, height: usize) -> (usize, usize, usize, usize) {
    (
        x.saturating_sub(1),
        y.saturating_sub(1),
        (x + 1).min(width - 1),
        (y + 1).min(height - 1),
    )
}

fn antialiased(a: &[u8], b: &[u8], x: usize, y: usize, width: usize, height: usize) -> bool {
    let (x0, y0, x2, y2) = bounds(x, y, width, height);
    let mut zeroes = usize::from(x == x0 || x == x2 || y == y0 || y == y2);
    let (mut min, mut max) = (0.0, 0.0);
    let (mut darkest, mut brightest) = ((x, y), (x, y));
    let p = (y * width + x) * 4;
    for nx in x0..=x2 {
        for ny in y0..=y2 {
            if (nx, ny) == (x, y) {
                continue;
            }
            let d = brightness_delta(a, p, (ny * width + nx) * 4);
            if d == 0.0 {
                zeroes += 1;
                if zeroes > 2 {
                    return false;
                }
            } else if d < min {
                min = d;
                darkest = (nx, ny);
            } else if d > max {
                max = d;
                brightest = (nx, ny);
            }
        }
    }
    if min == 0.0 || max == 0.0 {
        return false;
    }
    [darkest, brightest]
        .into_iter()
        .any(|(nx, ny)| siblings(a, nx, ny, width, height) && siblings(b, nx, ny, width, height))
}

fn siblings(image: &[u8], x: usize, y: usize, width: usize, height: usize) -> bool {
    let (x0, y0, x2, y2) = bounds(x, y, width, height);
    let mut zeroes = usize::from(x == x0 || x == x2 || y == y0 || y == y2);
    let p = (y * width + x) * 4;
    for nx in x0..=x2 {
        for ny in y0..=y2 {
            if (nx, ny) == (x, y) {
                continue;
            }
            let q = (ny * width + nx) * 4;
            if image[p..p + 4] == image[q..q + 4] {
                zeroes += 1;
            }
            if zeroes > 2 {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts and full diff-image hashes generated independently by the
    /// unmodified upstream pixelmatch 8.0.0 JavaScript implementation.
    #[test]
    fn matches_upstream_on_seeded_rgba_images() {
        let goldens = [
            (1, 1, [(1, 3331205267), (1, 3331205267), (0, 1859787237)]),
            (2, 2, [(4, 180144701), (2, 1157789889), (0, 1307571777)]),
            (9, 7, [(62, 3357250438), (48, 884636493), (4, 2507070432)]),
            (
                11,
                11,
                [(120, 2021498162), (91, 3826322217), (7, 1340911110)],
            ),
        ];
        for (width, height, expected) in goldens {
            let mut seed = 123456789u32;
            let mut next = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 24) as u8
            };
            let mut a = vec![0; width * height * 4];
            let mut b = a.clone();
            for i in 0..a.len() {
                a[i] = next();
                b[i] = if i % 5 == 0 { a[i] } else { next() };
            }
            for (threshold, (expected_count, expected_hash)) in
                [0.01, 0.1285, 0.5].into_iter().zip(expected)
            {
                let (count, diff) = compare(&a, &b, width, height, threshold);
                let hash = diff.iter().fold(2166136261u32, |h, b| {
                    (h ^ u32::from(*b)).wrapping_mul(16777619)
                });
                assert_eq!(
                    (count, hash),
                    (expected_count, expected_hash),
                    "{width}x{height}, {threshold}"
                );
            }
        }
    }

    #[test]
    fn transparent_colors_match_but_alpha_is_visible_on_checkerboard() {
        assert_eq!(compare(&[0, 0, 0, 0], &[255, 0, 0, 0], 1, 1, 0.0).0, 0);
        // Both are white over white, but the checkerboard exposes the alpha.
        assert_eq!(compare(&[255, 255, 255, 0], &[255; 4], 1, 1, 0.1285).0, 1);
        assert_eq!(compare(&[0, 0, 0, 255], &[255; 4], 1, 1, 0.1285).0, 1);
    }

    #[test]
    fn threshold_is_oklab_hyab() {
        let t = Tables::new();
        let black = [0, 0, 0, 255];
        let white = [255; 4];
        assert!(t.exceeds(&black, &white, 0, 0.99));
        assert!(!t.exceeds(&black, &white, 0, 1.01));
        // Dark values use exact cbrt rather than the inaccurate first LUT bin.
        assert!((t.cbrt(0.0001) - 0.0001f64.cbrt()).abs() < 1e-15);
    }

    #[test]
    fn excludes_antialiasing_at_an_edge() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..5 {
            for x in 0..5 {
                let c = [0, 128, 255, 255, 255][x];
                a.extend_from_slice(&[c, c, c, 255]);
                let c = if x < 2 { 0 } else { 255 };
                b.extend_from_slice(&[c, c, c, 255]);
            }
        }
        let (count, diff) = compare(&a, &b, 5, 5, 0.1);
        assert_eq!(count, 0);
        assert_eq!(&diff[4..8], &[255, 255, 0, 255]);
    }
}

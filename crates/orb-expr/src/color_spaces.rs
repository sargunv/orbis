//! Colour spaces for interpolation, ported from style-spec: D50 Lab and
//! HCL, and CSS Color 4's Oklab and Oklch.
use crate::{Color, ir::ColorSpace};

/// sRGB's transfer function, gamma-encoded to linear.
fn linear(x: f64) -> f64 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn lab(c: Color) -> [f64; 4] {
    let [r, g, b, a] = c.straight().map(f64::from);
    let f = |t: f64| {
        if t > (6.0f64 / 29.0).powi(3) {
            t.powf(1.0 / 3.0)
        } else {
            t / (3.0 * (6.0f64 / 29.0).powi(2)) + 4.0 / 29.0
        }
    };
    let (r, g, b) = (linear(r), linear(g), linear(b));
    let y = f(0.2225045 * r + 0.7168786 * g + 0.0606169 * b);
    let (x, z) = if r == g && g == b {
        (y, y)
    } else {
        (
            f((0.4360747 * r + 0.3850649 * g + 0.1430804 * b) / 0.96422),
            f((0.0139322 * r + 0.0971045 * g + 0.7141733 * b) / 0.82521),
        )
    };
    [
        (116.0 * y - 16.0).max(0.0),
        500.0 * (x - y),
        200.0 * (y - z),
        a,
    ]
}
fn rgb([l, a, b, alpha]: [f64; 4]) -> Color {
    let y = (l + 16.0) / 116.0;
    let x = if a.is_nan() { y } else { y + a / 500.0 };
    let z = if b.is_nan() { y } else { y - b / 200.0 };
    let f = |t: f64| {
        if t > 6.0 / 29.0 {
            t * t * t
        } else {
            3.0 * (6.0f64 / 29.0).powi(2) * (t - 4.0 / 29.0)
        }
    };
    let (x, y, z) = (0.96422 * f(x), f(y), 0.82521 * f(z));
    let gamma = |x: f64| {
        (if x <= 0.00304 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        })
        .clamp(0.0, 1.0)
    };
    Color::new(
        gamma(3.1338561 * x - 1.6168667 * y - 0.4906146 * z),
        gamma(-0.9787684 * x + 1.9161415 * y + 0.033454 * z),
        gamma(0.0719453 * x - 0.2289914 * y + 1.4052427 * z),
        alpha,
    )
}
type Matrix = [[f64; 3]; 3];

fn multiply(m: &Matrix, [x, y, z]: [f64; 3]) -> [f64; 3] {
    m.map(|row| row[0] * x + row[1] * y + row[2] * z)
}

/// `a·b`, at compile time.
const fn product(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = [[0.0; 3]; 3];
    let mut i = 0;
    while i < 9 {
        let (r, c) = (i / 3, i % 3);
        m[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        i += 1;
    }
    m
}

// CSS Color 4's conversion matrices, through XYZ (D65) to LMS, folded.
const LINEAR_SRGB_TO_LMS: Matrix = product(
    &[
        [0.819022437996703, 0.3619062600528904, -0.1288737815209879],
        [0.0329836539323885, 0.9292868615863434, 0.0361446663506424],
        [0.0481771893596242, 0.2642395317527308, 0.6335478284694309],
    ],
    &[
        [506752.0 / 1228815.0, 87881.0 / 245763.0, 12673.0 / 70218.0],
        [87098.0 / 409605.0, 175762.0 / 245763.0, 12673.0 / 175545.0],
        [7918.0 / 409605.0, 87881.0 / 737289.0, 1001167.0 / 1053270.0],
    ],
);
const LMS_TO_LINEAR_SRGB: Matrix = product(
    &[
        [12831.0 / 3959.0, -329.0 / 214.0, -1974.0 / 3959.0],
        [
            -851781.0 / 878810.0,
            1648619.0 / 878810.0,
            36519.0 / 878810.0,
        ],
        [705.0 / 12673.0, -2585.0 / 12673.0, 705.0 / 667.0],
    ],
    &[
        [1.2268798758459243, -0.5578149944602171, 0.2813910456659647],
        [-0.0405757452148008, 1.112286803280317, -0.0717110580655164],
        [-0.0763729366746601, -0.4214933324022432, 1.5869240198367816],
    ],
);
const LMS_TO_OKLAB: Matrix = [
    [0.210454268309314, 0.7936177747023054, -0.0040720430116193],
    [1.9779985324311684, -2.42859224204858, 0.450593709617411],
    [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
];
const OKLAB_TO_LMS: Matrix = [
    [1.0, 0.3963377773761749, 0.2158037573099136],
    [1.0, -0.1055613458156586, -0.0638541728258133],
    [1.0, -0.0894841775298119, -1.2914855480194092],
];

/// Straight `[l, a, b, alpha]` in Oklab.
fn oklab(c: Color) -> [f64; 4] {
    let [r, g, b, alpha] = c.straight().map(f64::from);
    let lms = multiply(&LINEAR_SRGB_TO_LMS, [linear(r), linear(g), linear(b)]);
    // An sRGB colour's LMS components are never negative, where `powf` and
    // `cbrt` agree (and `cbrt` would cost the web 1 KiB).
    let [l, a, b] = multiply(&LMS_TO_OKLAB, lms.map(|x| x.powf(1.0 / 3.0)));
    [l, a, b, alpha]
}

/// `[l, c, h, alpha]` in Oklch, the hue in degrees, NaN where it is
/// powerless (a chroma of at most 0.000004, as CSS Color 4 has it).
fn oklch(c: Color) -> [f64; 4] {
    let [l, a, b, alpha] = oklab(c);
    let c = (a * a + b * b).sqrt();
    if c <= 0.000004 {
        [l, 0.0, f64::NAN, alpha]
    } else {
        [l, c, b.atan2(a).to_degrees().rem_euclid(360.0), alpha]
    }
}

/// Past this chroma the clipped colour stops changing; capping it keeps the
/// conversion finite.
const MAX_CHROMA: f64 = 1e6;

/// An Oklch colour (hue in degrees, NaN as 0) in sRGB, clipped channel by
/// channel as browsers draw a colour outside it.
pub(crate) fn oklch_to_rgb([l, c, h, alpha]: [f64; 4]) -> [f64; 4] {
    let h = if h.is_nan() { 0.0 } else { h.to_radians() };
    let c = c.min(MAX_CHROMA);
    let lms = multiply(&OKLAB_TO_LMS, [l, c * h.cos(), c * h.sin()]);
    let rgb = multiply(&LMS_TO_LINEAR_SRGB, lms.map(|x| x * x * x));
    let [r, g, b] = rgb.map(|x| {
        let abs = x.abs();
        let encoded = if abs > 0.0031308 {
            x.signum() * (1.055 * abs.powf(1.0 / 2.4) - 0.055)
        } else {
            12.92 * x
        };
        encoded.clamp(0.0, 1.0)
    });
    [r, g, b, alpha]
}

/// An Oklab colour in sRGB, clipped as [`oklch_to_rgb`] clips (an axis so
/// long that its square overflows has a chroma past the cap either way).
pub(crate) fn oklab_to_rgb([l, a, b, alpha]: [f64; 4]) -> [f64; 4] {
    let c = (a * a + b * b).sqrt();
    oklch_to_rgb([l, c, b.atan2(a).to_degrees(), alpha])
}

/// Oklab or Oklch by CSS Color 4's rules: components premultiplied by
/// alpha, and in Oklch the hue the shorter way round, a grey stop taking
/// the other stop's.
fn interpolate_ok(a: Color, b: Color, t: f64, space: ColorSpace) -> Color {
    let lerp = |x: f64, y: f64| x + t * (y - x);
    let (mut a, mut b) = if space == ColorSpace::Oklch {
        (oklch(a), oklch(b))
    } else {
        (oklab(a), oklab(b))
    };
    // Lightness and chroma, or all three Oklab axes, carry alpha.
    let premultiplied = if space == ColorSpace::Oklch { 2 } else { 3 };
    if space == ColorSpace::Oklch {
        if a[2].is_nan() {
            a[2] = b[2];
        } else if b[2].is_nan() {
            b[2] = a[2];
        }
        if b[2] - a[2] > 180.0 {
            a[2] += 360.0;
        } else if b[2] - a[2] < -180.0 {
            b[2] += 360.0;
        }
    }
    let alpha = lerp(a[3], b[3]);
    let divisor = if alpha == 0.0 { 1.0 } else { alpha };
    let mut v = [0.0, 0.0, 0.0, alpha];
    for i in 0..3 {
        v[i] = if i < premultiplied {
            lerp(a[i] * a[3], b[i] * b[3]) / divisor
        } else {
            lerp(a[i], b[i])
        };
    }
    let [r, g, b, alpha] = if space == ColorSpace::Oklch {
        oklch_to_rgb(v)
    } else {
        oklab_to_rgb(v)
    };
    Color::new(r, g, b, alpha)
}

pub(crate) fn interpolate(a: Color, b: Color, t: f64, space: ColorSpace) -> Color {
    if matches!(space, ColorSpace::Oklch | ColorSpace::Oklab) {
        return interpolate_ok(a, b, t, space);
    }
    let mut a = lab(a);
    let mut b = lab(b);
    if space == ColorSpace::Hcl {
        let hcl = |[l, a, b, alpha]: [f64; 4]| {
            let c = (a * a + b * b).sqrt();
            let h = if (c * 10000.0).round() == 0.0 {
                f64::NAN
            } else {
                b.atan2(a).to_degrees().rem_euclid(360.0)
            };
            [h, c, l, alpha]
        };
        a = hcl(a);
        b = hcl(b);
        if a[0].is_nan() {
            a[0] = b[0];
        } else if b[0].is_nan() {
            b[0] = a[0];
        }
        let delta = b[0] - a[0];
        if delta > 180.0 {
            b[0] -= 360.0;
        } else if delta < -180.0 {
            b[0] += 360.0;
        }
    }
    let v = std::array::from_fn(|i| a[i] + t * (b[i] - a[i]));
    if space == ColorSpace::Hcl {
        let [h, c, l, alpha] = v;
        let h = if h.is_nan() { 0.0 } else { h.to_radians() };
        rgb([l, h.cos() * c, h.sin() * c, alpha])
    } else {
        rgb(v)
    }
}

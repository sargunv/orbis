//! D50 Lab/HCL conversion and interpolation, ported from style-spec.
use crate::{Color, ir::ColorSpace};
fn lab(c: Color) -> [f64; 4] {
    let [r, g, b, a] = c.straight().map(f64::from);
    let linear = |x: f64| {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
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
pub(crate) fn interpolate(a: Color, b: Color, t: f64, space: ColorSpace) -> Color {
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

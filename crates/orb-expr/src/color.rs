//! Colours: CSS colour strings to RGBA in the style's sRGB space, stored as
//! GL JS stores them (plan §11.1: defaults match GL JS).

use std::fmt;

use crate::js;

/// An sRGB colour with alpha.
///
/// Like GL JS's `Color`, it keeps both the premultiplied components the GPU
/// wants and the straight components that interpolation and `to-rgba` read.
/// The two differ beyond rounding only for a fully transparent colour, whose
/// straight components survive premultiplication so that it interpolates
/// toward its own hue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    premultiplied: [f32; 4],
    straight: [f32; 4],
}

impl Color {
    pub const TRANSPARENT: Self = Self {
        premultiplied: [0.0; 4],
        straight: [0.0; 4],
    };
    pub const BLACK: Self = Self {
        premultiplied: [0.0, 0.0, 0.0, 1.0],
        straight: [0.0, 0.0, 0.0, 1.0],
    };
    pub const WHITE: Self = Self {
        premultiplied: [1.0; 4],
        straight: [1.0; 4],
    };

    /// A colour from straight (not premultiplied) components in `0..=1`.
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self {
            premultiplied: [(r * a) as f32, (g * a) as f32, (b * a) as f32, a as f32],
            straight: [r as f32, g as f32, b as f32, a as f32],
        }
    }

    /// A colour from premultiplied components, as `to-color` builds one from
    /// an array.
    pub(crate) fn from_premultiplied(r: f64, g: f64, b: f64, a: f64) -> Self {
        let f = if a == 0.0 { f64::INFINITY } else { a };
        Self {
            premultiplied: [r as f32, g as f32, b as f32, a as f32],
            straight: [(r / f) as f32, (g / f) as f32, (b / f) as f32, a as f32],
        }
    }

    /// `self` to `other` by `t`, in premultiplied components as the style
    /// spec interpolates colours.
    pub fn mix(&self, other: &Self, t: f64) -> Self {
        let [a, b] = [self.premultiplied, other.premultiplied];
        let at = |i: usize| f64::from(a[i]) + (f64::from(b[i]) - f64::from(a[i])) * t;
        Self::from_premultiplied(at(0), at(1), at(2), at(3))
    }

    /// Parses a CSS colour: a keyword (including `transparent`), hex with 3,
    /// 4, 6 or 8 digits, `rgb()`/`rgba()` or `hsl()`/`hsla()` in comma or
    /// space syntax. `None` if the string is not a colour.
    pub fn parse(input: &str) -> Option<Self> {
        let [r, g, b, a] = parse_css(input)?;
        Some(Self::new(r, g, b, a))
    }

    /// Premultiplied `[r, g, b, a]` in `0..=1`.
    pub fn premultiplied(&self) -> [f32; 4] {
        self.premultiplied
    }

    /// Straight (not premultiplied) `[r, g, b, a]` in `0..=1`.
    pub fn straight(&self) -> [f32; 4] {
        self.straight
    }

    /// Linear interpolation of the straight components, as GL JS's
    /// `interpolate` does for colours in `rgb` space.
    pub(crate) fn interpolate(from: Self, to: Self, t: f64) -> Self {
        let [r, g, b, a] = std::array::from_fn(|i| {
            let (x, y) = (f64::from(from.straight[i]), f64::from(to.straight[i]));
            x + t * (y - x)
        });
        Self::new(r, g, b, a)
    }
}

/// `rgba(r,g,b,a)` with 0–255 channels, as GL JS's `Color.toString()`.
impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [r, g, b, a] = self.straight;
        let channel = |c: f32| (f64::from(c) * 255.0 + 0.5).floor();
        write!(f, "rgba({},{},{},", channel(r), channel(g), channel(b))?;
        js::write_number(f, f64::from(a))?;
        f.write_str(")")
    }
}

/// `Math.min(Math.max(min, n), max)`, which keeps NaN.
fn clamp(n: f64, min: f64, max: f64) -> f64 {
    if n.is_nan() { n } else { n.max(min).min(max) }
}

fn parse_css(input: &str) -> Option<[f64; 4]> {
    let input = input.trim().to_lowercase();
    if input == "transparent" {
        return Some([0.0; 4]);
    }
    if let Ok(i) = NAMED.binary_search_by(|(name, _)| (*name).cmp(input.as_str())) {
        let [r, g, b] = NAMED[i].1;
        return Some([
            f64::from(r) / 255.0,
            f64::from(g) / 255.0,
            f64::from(b) / 255.0,
            1.0,
        ]);
    }
    if let Some(hex) = input.strip_prefix('#') {
        return parse_hex(hex);
    }
    if let Some(args) = input
        .strip_prefix("rgba(")
        .or_else(|| input.strip_prefix("rgb("))
    {
        return parse_rgb(args);
    }
    if let Some(args) = input
        .strip_prefix("hsla(")
        .or_else(|| input.strip_prefix("hsl("))
    {
        return parse_hsl(args);
    }
    None
}

fn parse_hex(hex: &str) -> Option<[f64; 4]> {
    if !matches!(hex.len(), 3 | 4 | 6 | 8) || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let step = if hex.len() < 6 { 1 } else { 2 };
    let channel = |i: usize| -> f64 {
        let Some(digits) = hex.get(i * step..(i + 1) * step) else {
            return 1.0;
        };
        let v = u8::from_str_radix(digits, 16).expect("hex digits");
        let v = if step == 1 { v * 17 } else { v };
        f64::from(v) / 255.0
    };
    Some([channel(0), channel(1), channel(2), channel(3)])
}

/// The argument list of `rgb()`/`hsl()`, tokenised as GL JS's regular
/// expressions read it: three values separated by whitespace or commas, then
/// an optional alpha after `,` or `/`, then `)`.
struct Args<'a> {
    values: [(&'a str, bool); 3],
    alpha: Option<(&'a str, bool)>,
}

/// A run of `[0-9e.+-]`, as the regular expressions' numeric class.
fn numeric(s: &str) -> Option<(&str, &str)> {
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, 'e' | '.' | '+' | '-')))
        .unwrap_or(s.len());
    (end > 0).then(|| s.split_at(end))
}

fn is_regex_space(c: char) -> bool {
    // JavaScript's `\s`.
    c.is_whitespace() || c == '\u{feff}'
}

fn parse_args(s: &str, hue: bool) -> Option<Args<'_>> {
    let mut rest = s.trim_start_matches(is_regex_space);
    let mut values = [("", false); 3];
    let mut separators = [false; 2];
    for (i, value) in values.iter_mut().enumerate() {
        let (number, tail) = numeric(rest)?;
        rest = tail;
        let mut percent = false;
        if i == 0 && hue {
            rest = rest.strip_prefix("deg").unwrap_or(rest);
        } else if let Some(tail) = rest.strip_prefix('%') {
            rest = tail;
            percent = true;
        }
        *value = (number, percent);
        if i < 2 {
            let trimmed = rest.trim_start_matches(is_regex_space);
            let had_space = trimmed.len() < rest.len();
            if let Some(tail) = trimmed.strip_prefix(',') {
                separators[i] = true;
                rest = tail.trim_start_matches(is_regex_space);
            } else if had_space {
                rest = trimmed;
            } else {
                return None;
            }
        }
    }
    let trimmed = rest.trim_start_matches(is_regex_space);
    let mut alpha = None;
    let mut alpha_comma = None;
    if let Some(tail) = trimmed.strip_prefix([',', '/']) {
        alpha_comma = Some(trimmed.starts_with(','));
        let (number, tail) = numeric(tail.trim_start_matches(is_regex_space))?;
        let (percent, tail) = match tail.strip_prefix('%') {
            Some(tail) => (true, tail),
            None => (false, tail),
        };
        alpha = Some((number, percent));
        rest = tail.trim_start_matches(is_regex_space);
    } else {
        rest = trimmed;
    }
    if rest != ")" {
        return None;
    }
    // Commas everywhere or nowhere; `/` only goes with the space syntax.
    match (separators, alpha_comma) {
        ([false, false], None | Some(false)) | ([true, true], None | Some(true)) => {
            Some(Args { values, alpha })
        }
        _ => None,
    }
}

fn number(s: &str) -> f64 {
    // The token is drawn from `[0-9e.+-]`, where JavaScript's unary `+` and
    // Rust's float grammar agree.
    s.parse().unwrap_or(f64::NAN)
}

fn alpha(a: Option<(&str, bool)>) -> f64 {
    match a {
        None => 1.0,
        Some((n, percent)) => {
            let n = number(n);
            clamp(if percent { n / 100.0 } else { n }, 0.0, 1.0)
        }
    }
}

fn parse_rgb(s: &str) -> Option<[f64; 4]> {
    let args = parse_args(s, false)?;
    let max = match args.values.map(|(_, p)| p) {
        [true, true, true] => 100.0,
        [false, false, false] => 255.0,
        _ => return None,
    };
    let [r, g, b] = args.values.map(|(n, _)| clamp(number(n) / max, 0.0, 1.0));
    let rgba = [r, g, b, alpha(args.alpha)];
    (!rgba.iter().any(|v| v.is_nan())).then_some(rgba)
}

fn parse_hsl(s: &str) -> Option<[f64; 4]> {
    let args = parse_args(s, true)?;
    let [(h, _), (sat, sp), (l, lp)] = args.values;
    if !sp || !lp {
        return None;
    }
    let h = number(h);
    let s = clamp(number(sat), 0.0, 100.0);
    let l = clamp(number(l), 0.0, 100.0);
    let a = alpha(args.alpha);
    if [h, s, l, a].iter().any(|v| v.is_nan()) {
        return None;
    }
    Some(hsl_to_rgb(h, s, l, a))
}

fn hsl_to_rgb(h: f64, s: f64, l: f64, a: f64) -> [f64; 4] {
    let mut h = h % 360.0;
    if h < 0.0 {
        h += 360.0;
    }
    let (s, l) = (s / 100.0, l / 100.0);
    let f = |n: f64| {
        let k = (n + h / 30.0) % 12.0;
        let a = s * l.min(1.0 - l);
        l - a * (-1.0f64).max((k - 3.0).min(9.0 - k).min(1.0))
    };
    [f(0.0), f(8.0), f(4.0), a]
}

/// CSS Color 4 named colours, sorted by name.
const NAMED: &[(&str, [u8; 3])] = &[
    ("aliceblue", [240, 248, 255]),
    ("antiquewhite", [250, 235, 215]),
    ("aqua", [0, 255, 255]),
    ("aquamarine", [127, 255, 212]),
    ("azure", [240, 255, 255]),
    ("beige", [245, 245, 220]),
    ("bisque", [255, 228, 196]),
    ("black", [0, 0, 0]),
    ("blanchedalmond", [255, 235, 205]),
    ("blue", [0, 0, 255]),
    ("blueviolet", [138, 43, 226]),
    ("brown", [165, 42, 42]),
    ("burlywood", [222, 184, 135]),
    ("cadetblue", [95, 158, 160]),
    ("chartreuse", [127, 255, 0]),
    ("chocolate", [210, 105, 30]),
    ("coral", [255, 127, 80]),
    ("cornflowerblue", [100, 149, 237]),
    ("cornsilk", [255, 248, 220]),
    ("crimson", [220, 20, 60]),
    ("cyan", [0, 255, 255]),
    ("darkblue", [0, 0, 139]),
    ("darkcyan", [0, 139, 139]),
    ("darkgoldenrod", [184, 134, 11]),
    ("darkgray", [169, 169, 169]),
    ("darkgreen", [0, 100, 0]),
    ("darkgrey", [169, 169, 169]),
    ("darkkhaki", [189, 183, 107]),
    ("darkmagenta", [139, 0, 139]),
    ("darkolivegreen", [85, 107, 47]),
    ("darkorange", [255, 140, 0]),
    ("darkorchid", [153, 50, 204]),
    ("darkred", [139, 0, 0]),
    ("darksalmon", [233, 150, 122]),
    ("darkseagreen", [143, 188, 143]),
    ("darkslateblue", [72, 61, 139]),
    ("darkslategray", [47, 79, 79]),
    ("darkslategrey", [47, 79, 79]),
    ("darkturquoise", [0, 206, 209]),
    ("darkviolet", [148, 0, 211]),
    ("deeppink", [255, 20, 147]),
    ("deepskyblue", [0, 191, 255]),
    ("dimgray", [105, 105, 105]),
    ("dimgrey", [105, 105, 105]),
    ("dodgerblue", [30, 144, 255]),
    ("firebrick", [178, 34, 34]),
    ("floralwhite", [255, 250, 240]),
    ("forestgreen", [34, 139, 34]),
    ("fuchsia", [255, 0, 255]),
    ("gainsboro", [220, 220, 220]),
    ("ghostwhite", [248, 248, 255]),
    ("gold", [255, 215, 0]),
    ("goldenrod", [218, 165, 32]),
    ("gray", [128, 128, 128]),
    ("green", [0, 128, 0]),
    ("greenyellow", [173, 255, 47]),
    ("grey", [128, 128, 128]),
    ("honeydew", [240, 255, 240]),
    ("hotpink", [255, 105, 180]),
    ("indianred", [205, 92, 92]),
    ("indigo", [75, 0, 130]),
    ("ivory", [255, 255, 240]),
    ("khaki", [240, 230, 140]),
    ("lavender", [230, 230, 250]),
    ("lavenderblush", [255, 240, 245]),
    ("lawngreen", [124, 252, 0]),
    ("lemonchiffon", [255, 250, 205]),
    ("lightblue", [173, 216, 230]),
    ("lightcoral", [240, 128, 128]),
    ("lightcyan", [224, 255, 255]),
    ("lightgoldenrodyellow", [250, 250, 210]),
    ("lightgray", [211, 211, 211]),
    ("lightgreen", [144, 238, 144]),
    ("lightgrey", [211, 211, 211]),
    ("lightpink", [255, 182, 193]),
    ("lightsalmon", [255, 160, 122]),
    ("lightseagreen", [32, 178, 170]),
    ("lightskyblue", [135, 206, 250]),
    ("lightslategray", [119, 136, 153]),
    ("lightslategrey", [119, 136, 153]),
    ("lightsteelblue", [176, 196, 222]),
    ("lightyellow", [255, 255, 224]),
    ("lime", [0, 255, 0]),
    ("limegreen", [50, 205, 50]),
    ("linen", [250, 240, 230]),
    ("magenta", [255, 0, 255]),
    ("maroon", [128, 0, 0]),
    ("mediumaquamarine", [102, 205, 170]),
    ("mediumblue", [0, 0, 205]),
    ("mediumorchid", [186, 85, 211]),
    ("mediumpurple", [147, 112, 219]),
    ("mediumseagreen", [60, 179, 113]),
    ("mediumslateblue", [123, 104, 238]),
    ("mediumspringgreen", [0, 250, 154]),
    ("mediumturquoise", [72, 209, 204]),
    ("mediumvioletred", [199, 21, 133]),
    ("midnightblue", [25, 25, 112]),
    ("mintcream", [245, 255, 250]),
    ("mistyrose", [255, 228, 225]),
    ("moccasin", [255, 228, 181]),
    ("navajowhite", [255, 222, 173]),
    ("navy", [0, 0, 128]),
    ("oldlace", [253, 245, 230]),
    ("olive", [128, 128, 0]),
    ("olivedrab", [107, 142, 35]),
    ("orange", [255, 165, 0]),
    ("orangered", [255, 69, 0]),
    ("orchid", [218, 112, 214]),
    ("palegoldenrod", [238, 232, 170]),
    ("palegreen", [152, 251, 152]),
    ("paleturquoise", [175, 238, 238]),
    ("palevioletred", [219, 112, 147]),
    ("papayawhip", [255, 239, 213]),
    ("peachpuff", [255, 218, 185]),
    ("peru", [205, 133, 63]),
    ("pink", [255, 192, 203]),
    ("plum", [221, 160, 221]),
    ("powderblue", [176, 224, 230]),
    ("purple", [128, 0, 128]),
    ("rebeccapurple", [102, 51, 153]),
    ("red", [255, 0, 0]),
    ("rosybrown", [188, 143, 143]),
    ("royalblue", [65, 105, 225]),
    ("saddlebrown", [139, 69, 19]),
    ("salmon", [250, 128, 114]),
    ("sandybrown", [244, 164, 96]),
    ("seagreen", [46, 139, 87]),
    ("seashell", [255, 245, 238]),
    ("sienna", [160, 82, 45]),
    ("silver", [192, 192, 192]),
    ("skyblue", [135, 206, 235]),
    ("slateblue", [106, 90, 205]),
    ("slategray", [112, 128, 144]),
    ("slategrey", [112, 128, 144]),
    ("snow", [255, 250, 250]),
    ("springgreen", [0, 255, 127]),
    ("steelblue", [70, 130, 180]),
    ("tan", [210, 180, 140]),
    ("teal", [0, 128, 128]),
    ("thistle", [216, 191, 216]),
    ("tomato", [255, 99, 71]),
    ("turquoise", [64, 224, 208]),
    ("violet", [238, 130, 238]),
    ("wheat", [245, 222, 179]),
    ("white", [255, 255, 255]),
    ("whitesmoke", [245, 245, 245]),
    ("yellow", [255, 255, 0]),
    ("yellowgreen", [154, 205, 50]),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(s: &str) -> Option<[f64; 4]> {
        parse_css(s).map(|c| c.map(|v| (v * 1000.0).round() / 1000.0))
    }

    #[test]
    fn parses_css_colours() {
        assert_eq!(rgba("red"), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(rgba(" Transparent "), Some([0.0; 4]));
        assert_eq!(rgba("#f0c"), Some([1.0, 0.0, 0.8, 1.0]));
        assert_eq!(rgba("#f0c8"), Some([1.0, 0.0, 0.8, 0.533]));
        assert_eq!(rgba("#ff00cc80"), Some([1.0, 0.0, 0.8, 0.502]));
        assert_eq!(rgba("rgb(255, 0, 51)"), Some([1.0, 0.0, 0.2, 1.0]));
        assert_eq!(rgba("rgba(255,0,51,0.5)"), Some([1.0, 0.0, 0.2, 0.5]));
        assert_eq!(rgba("rgb(255 0 51 / 50%)"), Some([1.0, 0.0, 0.2, 0.5]));
        assert_eq!(rgba("rgb(100% 0% 20%)"), Some([1.0, 0.0, 0.2, 1.0]));
        assert_eq!(rgba("rgb(300, -5, 0)"), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(rgba("hsl(120, 100%, 50%)"), Some([0.0, 1.0, 0.0, 1.0]));
        assert_eq!(rgba("hsla(120,100%,25%,0.84)"), Some([0.0, 0.5, 0.0, 0.84]));
        assert_eq!(
            rgba("hsl(480deg 100% 50% / .5)"),
            Some([0.0, 1.0, 0.0, 0.5])
        );
    }

    #[test]
    fn rejects_malformed_colours() {
        for bad in [
            "",
            "#ff",
            "#fffff",
            "#ggg",
            "rgb(1,2)",
            "rgb(1 2 3 4)",
            "rgb(1, 2 3)",
            "rgb(1 2 3, 0.5)",
            "rgb(1, 2, 3 / 0.5)",
            "rgb(1% 2 3)",
            "rgb(1..2 3 4)",
            "hsl(1 2 3)",
            "hsl(1, 2%, 3%",
            "rgb(1 2 3))",
            "redd",
        ] {
            assert_eq!(parse_css(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn keeps_hue_of_transparent_colours() {
        let c = Color::parse("rgba(255, 0, 0, 0)").unwrap();
        assert_eq!(c.premultiplied(), [0.0; 4]);
        assert_eq!(c.straight(), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(c.to_string(), "rgba(255,0,0,0)");
    }

    #[test]
    fn named_colours_are_sorted() {
        assert!(NAMED.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(NAMED.len(), 148);
    }
}

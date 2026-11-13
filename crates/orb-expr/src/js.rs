//! The few JavaScript semantics expressions inherit from GL JS: number to
//! string, string to number, `JSON.stringify` escaping, and UTF-16 string
//! order.

use std::cmp::Ordering;
use std::fmt;

/// `Number.prototype.toString()`.
pub(crate) fn write_number(f: &mut impl fmt::Write, x: f64) -> fmt::Result {
    if x.is_nan() {
        return f.write_str("NaN");
    }
    if x == 0.0 {
        return f.write_str("0");
    }
    if x.is_infinite() {
        return f.write_str(if x > 0.0 { "Infinity" } else { "-Infinity" });
    }
    if x < 0.0 {
        f.write_str("-")?;
    }
    // Rust's `{:e}` prints the shortest round-tripping digits, as
    // ECMAScript's Number::toString requires; only the layout differs.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exponent.parse::<i32>().expect("exponent") + 1;
    if k <= n && n <= 21 {
        f.write_str(&digits)?;
        for _ in 0..n - k {
            f.write_str("0")?;
        }
        Ok(())
    } else if 0 < n && n <= 21 {
        let (int, frac) = digits.split_at(n as usize);
        write!(f, "{int}.{frac}")
    } else if -6 < n && n <= 0 {
        f.write_str("0.")?;
        for _ in 0..-n {
            f.write_str("0")?;
        }
        f.write_str(&digits)
    } else {
        let sign = if n > 0 { '+' } else { '-' };
        let (first, rest) = digits.split_at(1);
        if rest.is_empty() {
            write!(f, "{first}e{sign}{}", (n - 1).abs())
        } else {
            write!(f, "{first}.{rest}e{sign}{}", (n - 1).abs())
        }
    }
}

pub(crate) fn number_to_string(x: f64) -> String {
    let mut s = String::new();
    write_number(&mut s, x).expect("write to String");
    s
}

/// A number as `JSON.stringify` writes it: non-finite numbers become `null`.
pub(crate) fn write_json_number(f: &mut impl fmt::Write, x: f64) -> fmt::Result {
    if x.is_finite() {
        write_number(f, x)
    } else {
        f.write_str("null")
    }
}

/// A string as `JSON.stringify` writes it.
pub(crate) fn write_json_string(f: &mut impl fmt::Write, s: &str) -> fmt::Result {
    f.write_char('"')?;
    for c in s.chars() {
        match c {
            '"' => f.write_str("\\\"")?,
            '\\' => f.write_str("\\\\")?,
            '\u{8}' => f.write_str("\\b")?,
            '\u{c}' => f.write_str("\\f")?,
            '\n' => f.write_str("\\n")?,
            '\r' => f.write_str("\\r")?,
            '\t' => f.write_str("\\t")?,
            c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
            c => f.write_char(c)?,
        }
    }
    f.write_char('"')
}

fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
}

/// `Number(string)`: NaN when the string is not a numeric literal.
pub(crate) fn string_to_number(s: &str) -> f64 {
    let s = s.trim_matches(is_js_whitespace);
    if s.is_empty() {
        return 0.0;
    }
    let radix = |prefix: [&str; 2], radix| {
        prefix.iter().find_map(|p| s.strip_prefix(p)).map(|digits| {
            if digits.is_empty() {
                return f64::NAN;
            }
            digits
                .chars()
                .try_fold(0.0, |acc, c| {
                    c.to_digit(radix)
                        .map(|d| acc * f64::from(radix) + f64::from(d))
                })
                .unwrap_or(f64::NAN)
        })
    };
    if let Some(v) = radix(["0x", "0X"], 16)
        .or_else(|| radix(["0o", "0O"], 8))
        .or_else(|| radix(["0b", "0B"], 2))
    {
        return v;
    }
    let unsigned = s.strip_prefix(['+', '-']).unwrap_or(s);
    if unsigned == "Infinity" {
        return if s.starts_with('-') {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if !is_decimal_literal(unsigned) {
        return f64::NAN;
    }
    crate::decimal::parse(s).unwrap_or(f64::NAN)
}

/// `digits [. digits?] | . digits`, then an optional exponent.
fn is_decimal_literal(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    let digits = |i: &mut usize| {
        let start = *i;
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
        }
        *i - start
    };
    let int = digits(&mut i);
    let mut frac = 0;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        frac = digits(&mut i);
    }
    if int == 0 && frac == 0 {
        return false;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return false;
        }
    }
    i == b.len()
}

/// JavaScript string order: by UTF-16 code unit.
pub(crate) fn compare_strings(a: &str, b: &str) -> Ordering {
    if a.is_ascii() && b.is_ascii() {
        return a.cmp(b);
    }
    a.encode_utf16().cmp(b.encode_utf16())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_print_like_javascript() {
        for (x, s) in [
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (-1.5, "-1.5"),
            (0.1 + 0.2, "0.30000000000000004"),
            (123456789.0, "123456789"),
            (1e21, "1e+21"),
            (1.5e21, "1.5e+21"),
            (1e20, "100000000000000000000"),
            (0.000001, "0.000001"),
            (1e-7, "1e-7"),
            (1.25e-7, "1.25e-7"),
            (f64::NAN, "NaN"),
            (f64::NEG_INFINITY, "-Infinity"),
        ] {
            assert_eq!(number_to_string(x), s);
        }
    }

    #[test]
    fn strings_convert_like_javascript() {
        for (s, x) in [
            ("", 0.0),
            ("  12 ", 12.0),
            ("1e3", 1000.0),
            ("-.5", -0.5),
            ("5.", 5.0),
            ("0x1F", 31.0),
            ("-Infinity", f64::NEG_INFINITY),
        ] {
            assert_eq!(string_to_number(s), x, "{s:?}");
        }
        for s in [
            "abc",
            "1e",
            ".",
            "inf",
            "nan",
            "0x",
            "-0x1",
            "1 2",
            "Infinityx",
        ] {
            assert!(string_to_number(s).is_nan(), "{s:?}");
        }
    }

    #[test]
    fn strings_order_by_utf16() {
        // U+FF5E sorts after U+1F600 by code point but before it in UTF-16.
        assert_eq!(compare_strings("\u{ff5e}", "\u{1f600}"), Ordering::Greater);
        assert_eq!(compare_strings("a", "b"), Ordering::Less);
    }
}

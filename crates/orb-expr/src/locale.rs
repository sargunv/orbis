//! Locale-sensitive operations: collation, resolved locales and number
//! formatting. In a browser they are the host's ([`HostLocale`], the
//! browser's `Intl`, as GL JS asks it). Elsewhere they are built in and
//! deterministic: Latin search collation (with German and Swedish
//! tailoring), decimal grouping, common currency symbols and units, not
//! the whole Unicode collation algorithm, CLDR locale data, numbering
//! systems or every ECMA-402 unit and currency.
use std::cmp::Ordering;
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq)]
pub struct Collator {
    pub case_sensitive: bool,
    pub diacritic_sensitive: bool,
    pub locale: String,
}

/// `number-format`'s options.
#[derive(Clone, Copy, Debug, Default)]
pub struct NumberFormat<'a> {
    pub locale: Option<&'a str>,
    pub currency: Option<&'a str>,
    pub unit: Option<&'a str>,
    pub min_fraction_digits: Option<f64>,
    pub max_fraction_digits: Option<f64>,
}

/// A host's locale services: the browser's `Intl`.
pub trait HostLocale: Send + Sync {
    /// The locale a collator for `requested` (or the host's own) resolves
    /// to.
    fn resolved_locale(&self, requested: Option<&str>) -> String;
    /// Orders two strings as `collator` does.
    fn compare(&self, lhs: &str, rhs: &str, collator: &Collator) -> Ordering;
    /// `number-format`, or the error the host's formatter gives.
    fn format_number(&self, number: f64, format: &NumberFormat<'_>) -> Result<String, String>;
}

static HOST: OnceLock<Box<dyn HostLocale>> = OnceLock::new();

/// Answers locale expressions with `host` in this process (or wasm
/// instance) from now on. Only the first call counts.
pub fn set_host_locale(host: Box<dyn HostLocale>) {
    let _ = HOST.set(host);
}

/// Orders two strings as `collator` does.
pub(crate) fn compare(lhs: &str, rhs: &str, c: &Collator) -> Ordering {
    match HOST.get() {
        Some(host) => host.compare(lhs, rhs, c),
        None => builtin::compare(lhs, rhs, c),
    }
}

/// The locale a collator resolves `requested` to.
pub(crate) fn resolved_locale(requested: Option<&str>) -> String {
    match HOST.get() {
        Some(host) => host.resolved_locale(requested),
        None => builtin::resolved_locale(requested),
    }
}

/// `number-format`.
pub(crate) fn format_number(number: f64, o: NumberFormat<'_>) -> Result<String, String> {
    match HOST.get() {
        Some(host) => host.format_number(number, &o),
        None => builtin::format_number(number, o),
    }
}

/// Without a host in a browser: code points in order, and numbers as
/// JavaScript prints them, rather than the built-in tables the browser
/// would carry.
#[cfg(all(target_arch = "wasm32", not(test)))]
mod builtin {
    use super::{Collator, NumberFormat, Ordering};

    pub(super) fn compare(lhs: &str, rhs: &str, _: &Collator) -> Ordering {
        lhs.cmp(rhs)
    }

    pub(super) fn resolved_locale(requested: Option<&str>) -> String {
        requested.unwrap_or("en-US").into()
    }

    pub(super) fn format_number(number: f64, _: NumberFormat<'_>) -> Result<String, String> {
        Ok(crate::js::number_to_string(number))
    }
}

// Built in natively, and in wasm tests, which have no browser host.
#[cfg(any(test, not(target_arch = "wasm32")))]
mod builtin {
    use super::{Collator, NumberFormat, Ordering};

    fn latin(c: char) -> char {
        match c {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
            'ç' | 'ć' | 'č' => 'c',
            'ď' | 'đ' => 'd',
            'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ė' | 'ę' | 'ě' => 'e',
            'ì' | 'í' | 'î' | 'ï' | 'ī' | 'į' => 'i',
            'ñ' | 'ń' | 'ň' => 'n',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ő' => 'o',
            'ù' | 'ú' | 'û' | 'ü' | 'ū' | 'ů' | 'ű' | 'ų' => 'u',
            'ý' | 'ÿ' => 'y',
            'ś' | 'š' | 'ş' => 's',
            'ž' | 'ź' | 'ż' => 'z',
            'ł' => 'l',
            'ř' => 'r',
            'ť' => 't',
            _ => c,
        }
    }

    pub(super) fn compare(lhs: &str, rhs: &str, c: &Collator) -> Ordering {
        let key = |s: &str| {
            let mut primary = String::new();
            let mut accents = Vec::new();
            let mut cases = Vec::new();
            for original in s.chars() {
                for ch in original.to_lowercase() {
                    if ('\u{300}'..='\u{36f}').contains(&ch) {
                        if c.diacritic_sensitive {
                            accents.push(ch as u32);
                        }
                        continue;
                    }
                    let base = latin(ch);
                    if c.locale.starts_with("de") && matches!(ch, 'ä' | 'ö' | 'ü') {
                        primary.push(base);
                        primary.push('e');
                        if c.case_sensitive {
                            cases.push(original.is_uppercase());
                        }
                    } else if c.locale.starts_with("sv") && matches!(ch, 'å' | 'ä' | 'ö') {
                        primary.push(match ch {
                            'å' => '{',
                            'ä' => '|',
                            _ => '}',
                        });
                    } else {
                        primary.push(base);
                    }
                    if c.diacritic_sensitive {
                        accents.push(if ch == base { 0 } else { ch as u32 });
                    }
                    if c.case_sensitive {
                        cases.push(original.is_uppercase());
                    }
                }
            }
            (primary, accents, cases)
        };
        key(lhs).cmp(&key(rhs))
    }

    pub(super) fn resolved_locale(requested: Option<&str>) -> String {
        let s = requested.unwrap_or("en-US");
        let language = s.split('-').next().unwrap_or("en");
        if matches!(
            language,
            "en" | "de"
                | "sv"
                | "fr"
                | "es"
                | "it"
                | "ja"
                | "zh"
                | "tr"
                | "pt"
                | "ru"
                | "ar"
                | "ko"
                | "nl"
                | "da"
        ) {
            s.into()
        } else {
            "en-US".into()
        }
    }

    pub(super) fn format_number(number: f64, o: NumberFormat<'_>) -> Result<String, String> {
        let currency_digits = if matches!(o.currency, Some("JPY" | "KRW")) {
            0
        } else {
            2
        };
        let default_min = if o.currency.is_some() {
            currency_digits
        } else {
            0
        };
        let min = o.min_fraction_digits.unwrap_or(f64::from(default_min));
        let max = o.max_fraction_digits.unwrap_or(if o.currency.is_some() {
            f64::from(currency_digits).max(min)
        } else {
            3.0f64.max(min)
        });
        if !(0.0..=100.0).contains(&min) || !(0.0..=100.0).contains(&max) || min > max {
            return Err("minimumFractionDigits value is out of range.".into());
        }
        let min = min as usize;
        let max = max as usize;
        // Intl formats the shortest decimal representation of a double. Avoid
        // exposing binary tails when padding beyond its significant digits.
        let shortest = crate::js::number_to_string(number.abs());
        let raw = if let Some((_, fraction)) = shortest.split_once('.') {
            if !shortest.contains('e') && fraction.len() <= max {
                shortest
            } else {
                format!("{:.*}", max, number.abs())
            }
        } else if shortest.contains('e') {
            format!("{:.*}", max, number.abs())
        } else {
            shortest
        };
        let (integer, fraction) = raw.split_once('.').unwrap_or((&raw, ""));
        let mut fraction = fraction.trim_end_matches('0').to_string();
        while fraction.len() < min {
            fraction.push('0');
        }
        let european = o
            .locale
            .is_some_and(|s| s.starts_with("de") || s.starts_with("fr"));
        let separator = if o.locale.is_some_and(|s| s.starts_with("fr")) {
            '\u{202f}'
        } else if european {
            '.'
        } else {
            ','
        };
        let mut grouped = String::new();
        for (i, ch) in integer.chars().enumerate() {
            if i > 0 && (integer.len() - i).is_multiple_of(3) {
                grouped.push(separator);
            }
            grouped.push(ch);
        }
        if !fraction.is_empty() {
            grouped.push(if european { ',' } else { '.' });
            grouped.push_str(&fraction);
        }
        if let Some(currency) = o.currency {
            let symbol = match currency {
                "USD" => "$",
                "EUR" => "€",
                "JPY" => "¥",
                "GBP" => "£",
                "KRW" => "₩",
                "CNY" => "CN¥",
                _ => currency,
            };
            grouped = if european {
                format!("{grouped}\u{a0}{symbol}")
            } else {
                format!("{symbol}{grouped}")
            };
        }
        if let Some(unit) = o.unit {
            let suffix = match unit {
                "meter" => " m",
                "kilometer" => " km",
                "celsius" => "°C",
                "fahrenheit" => "°F",
                "kilobyte" => " kB",
                "byte" => " byte",
                "percent" => "%",
                "second" => " sec",
                "liter" => " L",
                _ => {
                    return Err(format!(
                        "Invalid unit argument for Intl.NumberFormat() '{unit}'"
                    ));
                }
            };
            grouped.push_str(suffix);
        }
        if number.is_sign_negative() {
            grouped.insert(0, '-');
        }
        Ok(grouped)
    }
}

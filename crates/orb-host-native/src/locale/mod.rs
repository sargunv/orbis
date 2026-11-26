//! Locale expressions (`collator`, `resolved-locale`, `number-format`)
//! answered by the platform's own locale library, as GL JS gets them from
//! the browser's `Intl`, with the system's CLDR data rather than any of
//! our own. A host calls [`install`] once at start; where the platform has
//! no backend, or its library or a function in it is missing, orb-expr's
//! built-in tables answer as before. Tests and fixtures never install it,
//! so they stay deterministic.
//!
//! This module holds everything that doesn't depend on the platform: the
//! language tag parser, the default locale rule, which Unicode extensions
//! a collator honors, `Intl.NumberFormat`'s option rules and error
//! messages, the per-thread caches and the fallback to the built-in
//! tables. A platform's backend is only the thin binding to its library,
//! one file each, chosen by `cfg` in `platform` below:
//!
//! - `apple.rs`: the system ICU, `libicucore`, through `icu4c.rs`.
//! - `linux.rs`: the system ICU, `libicuuc` and `libicui18n`, through
//!   `icu4c.rs`.
//! - `android.rs`: the framework's ICU, `android.icu`, through JNI.
//!
//! A backend file provides `open() -> Option<Box<dyn Backend>>`, which
//! loads its library and resolves every function it calls at run time
//! (`dlopen` and `dlsym`, `LoadLibrary` and `GetProcAddress`, JNI
//! lookups), answering `None` when anything is missing, so the binary
//! links nothing new and still runs where the library isn't. An ICU4C
//! platform (Apple's `libicucore`, Linux's `libicuuc` and `libicui18n`,
//! Windows' `icu.dll`) hands `icu4c::Icu::load` a resolver for ICU's
//! plain function names, adding its build's version suffix if it has
//! one, and its user's locale, and is done; another implements
//! [`Backend`], whose methods state the contract. A new backend widens
//! the `cfg`s here, including the tests' `system` module, which then
//! checks it against the platform's library.

// A platform without a backend yet uses nothing but `install`.
#![cfg_attr(
    not(any(target_vendor = "apple", target_os = "linux", target_os = "android")),
    allow(dead_code, reason = "no backend on this platform yet")
)]

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use orb_expr::{BuiltinLocale, Collator, HostLocale, NumberFormat};

#[cfg(target_os = "android")]
mod android;
#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(any(target_vendor = "apple", target_os = "linux"))]
mod icu4c;
#[cfg(target_os = "linux")]
mod linux;

/// This platform's backend.
mod platform {
    #[cfg(target_os = "android")]
    pub(super) use super::android::open;
    #[cfg(target_vendor = "apple")]
    pub(super) use super::apple::open;
    #[cfg(target_os = "linux")]
    pub(super) use super::linux::open;

    #[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "android")))]
    pub(super) fn open() -> Option<Box<dyn super::Backend>> {
        None
    }
}

/// Whether a host asked for the platform's library, and whether it's in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Install {
    Off,
    Asked,
    Done,
}

static INSTALL: Mutex<Install> = Mutex::new(Install::Off);

/// Answers locale expressions in this process with the platform's locale
/// library from now on, where it has one. Call it as a host starts, before
/// any style parses: parsing folds constant expressions. On Android the
/// library is the app's JVM's, so it comes in once
/// [`init`](crate::android::init) hands that over, if that is later. Calls
/// after it's in do nothing.
pub fn install() {
    let mut install = INSTALL.lock().unwrap_or_else(|e| e.into_inner());
    if *install == Install::Off {
        *install = Install::Asked;
    }
    open(&mut install);
}

/// Brings the platform's library in if a host asked for it and it wasn't
/// there to open: on Android, once the app's JVM is.
#[cfg(target_os = "android")]
pub(crate) fn platform_ready() {
    open(&mut INSTALL.lock().unwrap_or_else(|e| e.into_inner()));
}

fn open(install: &mut Install) {
    if *install == Install::Asked
        && let Some(backend) = platform::open()
    {
        orb_expr::set_host_locale(Box::new(SystemLocale::new(backend)));
        *install = Install::Done;
    }
}

/// What a collator tells apart beyond base letters, as `Intl.Collator`'s
/// `sensitivity`. A backend maps it to ICU's strengths: base letters alone
/// at primary strength, then diacritics at secondary, case alone as
/// primary with the case level on, both at tertiary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sensitivity {
    pub(crate) case: bool,
    pub(crate) diacritics: bool,
}

/// What `number-format` formats a number as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Style<'a> {
    Decimal,
    /// An amount of a currency, by its upper-case ISO 4217 code.
    Currency(&'a str),
    /// A measure in one of `Intl`'s sanctioned units, or a unit per
    /// another.
    Unit {
        unit: &'a str,
        per: Option<&'a str>,
    },
}

/// A number format, its options settled by `Intl.NumberFormat`'s rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NumberRequest<'a> {
    /// A BCP 47 tag with data on this platform, with a `-u-nu-` numbering
    /// system if one was asked for.
    pub(crate) locale: &'a str,
    pub(crate) style: Style<'a>,
    pub(crate) min_fraction_digits: u8,
    pub(crate) max_fraction_digits: u8,
}

impl NumberRequest<'_> {
    /// The ICU number skeleton of the request, which ICU 62 (macOS 10.14,
    /// iOS 12) and later take: units as `measure-unit/<type>-<unit>`,
    /// rounding half away from zero as `Intl` does, the locale's grouping,
    /// and the short unit and symbol currency widths, `Intl`'s defaults.
    pub(crate) fn skeleton(&self) -> String {
        let mut skeleton = match self.style {
            Style::Decimal => String::new(),
            Style::Currency(code) => format!("currency/{code} "),
            Style::Unit { unit, per } => {
                // ICU 62 has `percent` only as this stem, not as a unit.
                let mut stem = if unit == "percent" {
                    "percent ".to_owned()
                } else {
                    format!(
                        "measure-unit/{}-{unit} ",
                        unit_type(unit).unwrap_or_default()
                    )
                };
                if let Some(per) = per {
                    stem += &format!(
                        "per-measure-unit/{}-{per} ",
                        unit_type(per).unwrap_or_default()
                    );
                }
                stem
            }
        };
        let (min, max) = (self.min_fraction_digits, self.max_fraction_digits);
        if max == 0 {
            skeleton += "precision-integer";
        } else {
            skeleton.push('.');
            skeleton += &"0".repeat(min.into());
            skeleton += &"#".repeat((max - min).into());
        }
        skeleton + " rounding-mode-half-up"
    }
}

/// A platform's locale library. Every method may be called from any
/// thread; the collators and formatters it opens stay on the thread that
/// opened them. Tags in and out are BCP 47 (`de-DE-u-co-phonebk`), which
/// a backend converts to its library's identifiers itself.
pub(crate) trait Backend: Send + Sync {
    /// The user's locale, as a BCP 47 tag or an ICU identifier
    /// (`en_US@rg=gbzzzz`); its keywords and extensions are dropped. `None`
    /// takes `en-US`.
    fn default_locale(&self) -> Option<String>;
    /// The languages (`de`, `sv`) the library has locale data for. Asked
    /// once.
    fn languages(&self) -> Vec<String>;
    /// Whether the library has the collation type `collation`, as BCP 47
    /// names it (`phonebk`, `trad`), for `locale`.
    fn has_collation(&self, locale: &str, collation: &str) -> bool;
    /// A collator for `locale` at `sensitivity`, honoring the tag's `co`,
    /// `kf` and `kn` extensions, with normalization on as `Intl`'s, or
    /// `None` when the library can't open one (orb-expr's built-in
    /// collation answers instead).
    fn collator(&self, locale: &str, sensitivity: Sensitivity) -> Option<Box<dyn SystemCollator>>;
    /// The ISO 4217 fraction digits of a currency code (three upper-case
    /// ASCII letters), if the library knows it.
    fn currency_digits(&self, currency: &str) -> Option<u8>;
    /// A formatter for `request` ([`NumberRequest::skeleton`] where the
    /// library takes skeletons), or the library's error.
    fn number_format(
        &self,
        request: &NumberRequest<'_>,
    ) -> Result<Box<dyn SystemNumberFormat>, String>;
}

/// An open collator.
pub(crate) trait SystemCollator {
    fn compare(&self, lhs: &str, rhs: &str) -> Ordering;
}

/// An open number formatter.
pub(crate) trait SystemNumberFormat {
    fn format(&self, number: f64) -> Result<String, String>;
}

/// A BCP 47 language tag, parsed as far as locale expressions need.
#[derive(Debug, PartialEq, Eq)]
struct LanguageTag {
    language: String,
    /// The language, script and region, in canonical case: `zh-Hant-TW`.
    base: String,
    /// The `-u-` extension's keywords and their types, in lower case; an
    /// empty type is `true`.
    keywords: Vec<(String, String)>,
}

impl LanguageTag {
    /// Parses a well-formed tag, or `None`. `und` counts as none, and
    /// variants and private use are dropped.
    fn parse(tag: &str) -> Option<Self> {
        let alpha = |s: &str| s.bytes().all(|b| b.is_ascii_alphabetic());
        let alnum = |s: &str| s.bytes().all(|b| b.is_ascii_alphanumeric());
        let mut subtags = tag.split('-').peekable();
        let language = subtags.next()?.to_ascii_lowercase();
        if !alpha(&language) || !matches!(language.len(), 2..=3 | 5..=8) || language == "und" {
            return None;
        }
        let mut base = language.clone();
        // Extended language subtags.
        while subtags.peek().is_some_and(|s| s.len() == 3 && alpha(s)) {
            subtags.next();
        }
        if let Some(script) = subtags.next_if(|s| s.len() == 4 && alpha(s)) {
            base.push('-');
            base += &script[..1].to_ascii_uppercase();
            base += &script[1..].to_ascii_lowercase();
        }
        if let Some(region) = subtags.next_if(|s| {
            (s.len() == 2 && alpha(s)) || (s.len() == 3 && s.bytes().all(|b| b.is_ascii_digit()))
        }) {
            base.push('-');
            base += &region.to_ascii_uppercase();
        }
        while subtags
            .next_if(|s| {
                alnum(s)
                    && (matches!(s.len(), 5..=8)
                        || (s.len() == 4 && s.as_bytes()[0].is_ascii_digit()))
            })
            .is_some()
        {}
        let mut keywords: Vec<(String, String)> = Vec::new();
        while let Some(singleton) = subtags.next() {
            if singleton.len() != 1 || !alnum(singleton) {
                return None;
            }
            let singleton = singleton.to_ascii_lowercase();
            if singleton == "x" {
                // Private use runs to the end.
                return subtags
                    .all(|s| (1..=8).contains(&s.len()) && alnum(s))
                    .then_some(Self {
                        language,
                        base,
                        keywords,
                    });
            }
            let mut any = false;
            while let Some(subtag) = subtags.next_if(|s| s.len() >= 2) {
                if subtag.len() > 8 || !alnum(subtag) {
                    return None;
                }
                any = true;
                if singleton != "u" {
                    continue;
                }
                let subtag = subtag.to_ascii_lowercase();
                if subtag.len() == 2 {
                    if !subtag.as_bytes()[1].is_ascii_alphabetic() {
                        return None;
                    }
                    keywords.push((subtag, String::new()));
                } else if let Some((_, value)) = keywords.last_mut() {
                    if !value.is_empty() {
                        value.push('-');
                    }
                    *value += &subtag;
                }
                // Attributes before the first key don't matter here.
            }
            if !any {
                return None;
            }
        }
        Some(Self {
            language,
            base,
            keywords,
        })
    }

    /// The type of the `-u-` keyword `key`, with `true` for an empty one.
    fn keyword(&self, key: &str) -> Option<&str> {
        let (_, value) = self.keywords.iter().find(|(k, _)| k == key)?;
        Some(if value.is_empty() { "true" } else { value })
    }
}

/// The units `Intl.NumberFormat` takes, with the type ICU's skeletons
/// qualify each with.
fn unit_type(unit: &str) -> Option<&'static str> {
    Some(match unit {
        "acre" | "hectare" => "area",
        "bit" | "byte" | "gigabit" | "gigabyte" | "kilobit" | "kilobyte" | "megabit"
        | "megabyte" | "petabyte" | "terabit" | "terabyte" => "digital",
        "celsius" | "fahrenheit" => "temperature",
        "centimeter" | "foot" | "inch" | "kilometer" | "meter" | "mile" | "mile-scandinavian"
        | "millimeter" | "yard" => "length",
        "day" | "hour" | "microsecond" | "millisecond" | "minute" | "month" | "nanosecond"
        | "second" | "week" | "year" => "duration",
        "degree" => "angle",
        "fluid-ounce" | "gallon" | "liter" | "milliliter" => "volume",
        "gram" | "kilogram" | "ounce" | "pound" | "stone" => "mass",
        "percent" => "concentr",
        _ => return None,
    })
}

/// `number-format`'s style, checked as `Intl.NumberFormat` checks it.
fn style<'a>(currency: &'a mut String, o: &NumberFormat<'a>) -> Result<Style<'a>, String> {
    if let Some(code) = o.currency {
        if code.len() != 3 || !code.bytes().all(|b| b.is_ascii_alphabetic()) {
            return Err(format!("Invalid currency code : {code}"));
        }
        *currency = code.to_ascii_uppercase();
        let currency: &'a String = currency;
        return Ok(Style::Currency(currency));
    }
    let Some(unit) = o.unit else {
        return Ok(Style::Decimal);
    };
    let invalid = || format!("Invalid unit argument for Intl.NumberFormat() '{unit}'");
    match unit.split_once("-per-") {
        Some((unit, per)) if unit_type(unit).is_some() && unit_type(per).is_some() => {
            Ok(Style::Unit {
                unit,
                per: Some(per),
            })
        }
        None if unit_type(unit).is_some() => Ok(Style::Unit { unit, per: None }),
        _ => Err(invalid()),
    }
}

/// The fraction digits `Intl.NumberFormat` settles on: the currency's
/// (or 0 and 3) by default, an omitted maximum at least the minimum and
/// an omitted minimum at most the maximum.
fn fraction_digits(o: &NumberFormat<'_>, currency_digits: Option<u8>) -> Result<(u8, u8), String> {
    let digits = |value: Option<f64>, name: &str| {
        value
            .map(|v| {
                if (0.0..=100.0).contains(&v) {
                    Ok(v as u8)
                } else {
                    Err(format!("{name} value is out of range."))
                }
            })
            .transpose()
    };
    let min = digits(o.min_fraction_digits, "minimumFractionDigits")?;
    let max = digits(o.max_fraction_digits, "maximumFractionDigits")?;
    let (default_min, default_max) = currency_digits.map_or((0, 3), |d| (d, d));
    Ok(match (min, max) {
        (None, None) => (default_min, default_max),
        (Some(min), None) => (min, default_max.max(min)),
        (None, Some(max)) => (default_min.min(max), max),
        (Some(min), Some(max)) if min <= max => (min, max),
        _ => return Err("maximumFractionDigits value is out of range.".into()),
    })
}

/// `number-format`'s options as the formatter cache's key.
type FormatKey = ([Option<String>; 3], [Option<u64>; 2]);

/// A collator's locale and sensitivity, and the collator, if it opened.
type CachedCollator = (String, Sensitivity, Option<Box<dyn SystemCollator>>);

// The last collator and formatter made on each thread: an expression asks
// for one per feature, and opening one costs far more than using it.
thread_local! {
    static COLLATOR: RefCell<Option<CachedCollator>> = const { RefCell::new(None) };
    static FORMAT: RefCell<Option<(FormatKey, Box<dyn SystemNumberFormat>)>> =
        const { RefCell::new(None) };
}

/// The locale services of the platform's library.
struct SystemLocale {
    backend: Box<dyn Backend>,
    languages: OnceLock<HashSet<String>>,
    default: OnceLock<String>,
}

impl SystemLocale {
    fn new(backend: Box<dyn Backend>) -> Self {
        Self {
            backend,
            languages: OnceLock::new(),
            default: OnceLock::new(),
        }
    }

    /// A tag whose language has data here, else `None`.
    fn supported(&self, tag: &str) -> Option<LanguageTag> {
        let languages = self
            .languages
            .get_or_init(|| self.backend.languages().into_iter().collect());
        LanguageTag::parse(tag).filter(|t| languages.contains(&t.language))
    }

    /// The default locale: the user's, without its extensions or
    /// keywords, or `en-US`.
    fn default_locale(&self) -> &str {
        self.default.get_or_init(|| {
            self.backend
                .default_locale()
                .and_then(|id| {
                    self.supported(&id.split('@').next().unwrap_or_default().replace('_', "-"))
                })
                .map_or_else(|| "en-US".into(), |t| t.base)
        })
    }
}

impl HostLocale for SystemLocale {
    /// The language the collator compares in, or the default locale for
    /// an empty, malformed or unsupported tag, with the `co`, `kf` and
    /// `kn` extensions it honors, as `Intl.Collator`'s resolved locale.
    fn resolved_locale(&self, requested: Option<&str>) -> String {
        let Some(tag) = requested.and_then(|r| self.supported(r)) else {
            return self.default_locale().into();
        };
        let mut extensions = String::new();
        if let Some(co) = tag.keyword("co")
            && !matches!(co, "standard" | "search")
            && self.backend.has_collation(&tag.base, co)
        {
            extensions += &format!("-co-{co}");
        }
        if let Some(kf @ ("upper" | "lower" | "false")) = tag.keyword("kf") {
            extensions += &format!("-kf-{kf}");
        }
        match tag.keyword("kn") {
            Some("true") => extensions += "-kn",
            Some("false") => extensions += "-kn-false",
            _ => {}
        }
        if extensions.is_empty() {
            tag.base
        } else {
            format!("{}-u{extensions}", tag.base)
        }
    }

    fn compare(&self, lhs: &str, rhs: &str, c: &Collator) -> Ordering {
        let sensitivity = Sensitivity {
            case: c.case_sensitive,
            diacritics: c.diacritic_sensitive,
        };
        COLLATOR.with_borrow_mut(|cached| {
            if !cached
                .as_ref()
                .is_some_and(|(l, s, _)| *l == c.locale && *s == sensitivity)
            {
                *cached = Some((
                    c.locale.clone(),
                    sensitivity,
                    self.backend.collator(&c.locale, sensitivity),
                ));
            }
            match cached
                .as_ref()
                .and_then(|(_, _, collator)| collator.as_ref())
            {
                Some(collator) => collator.compare(lhs, rhs),
                None => BuiltinLocale.compare(lhs, rhs, c),
            }
        })
    }

    fn format_number(&self, number: f64, o: &NumberFormat<'_>) -> Result<String, String> {
        let key: FormatKey = (
            [o.locale, o.currency, o.unit].map(|s| s.map(str::to_owned)),
            [o.min_fraction_digits, o.max_fraction_digits].map(|d| d.map(f64::to_bits)),
        );
        FORMAT.with_borrow_mut(|cached| {
            if let Some((k, format)) = cached
                && *k == key
            {
                return format.format(number);
            }
            let mut currency = String::new();
            let style = style(&mut currency, o)?;
            let currency_digits = match style {
                Style::Currency(code) => Some(self.backend.currency_digits(code).unwrap_or(2)),
                _ => None,
            };
            let (min, max) = fraction_digits(o, currency_digits)?;
            let locale = match o.locale.and_then(|l| self.supported(l)) {
                Some(tag) => match tag.keyword("nu") {
                    Some(nu) if nu != "true" => format!("{}-u-nu-{nu}", tag.base),
                    _ => tag.base,
                },
                None => self.default_locale().into(),
            };
            let request = NumberRequest {
                locale: &locale,
                style,
                min_fraction_digits: min,
                max_fraction_digits: max,
            };
            let format = self.backend.number_format(&request)?;
            let formatted = format.format(number);
            *cached = Some((key, format));
            formatted
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_parse_to_their_base_and_unicode_keywords() {
        let tag = LanguageTag::parse("DE-latn-de-1996-u-co-PHONEBK-kn-x-private").unwrap();
        assert_eq!(tag.language, "de");
        assert_eq!(tag.base, "de-Latn-DE");
        assert_eq!(tag.keyword("co"), Some("phonebk"));
        assert_eq!(tag.keyword("kn"), Some("true"));
        assert_eq!(LanguageTag::parse("zh-Hant-TW").unwrap().base, "zh-Hant-TW");
        assert_eq!(LanguageTag::parse("es-419").unwrap().base, "es-419");
        for malformed in [
            "", "e", "en_US", "en-", "en--US", "und", "en-u", "en-u-k1", "123",
        ] {
            assert_eq!(LanguageTag::parse(malformed), None, "{malformed}");
        }
    }

    #[test]
    fn fraction_digits_follow_intl() {
        let o = |min, max| NumberFormat {
            min_fraction_digits: min,
            max_fraction_digits: max,
            ..NumberFormat::default()
        };
        assert_eq!(fraction_digits(&o(None, None), None), Ok((0, 3)));
        assert_eq!(fraction_digits(&o(Some(5.0), None), None), Ok((5, 5)));
        assert_eq!(fraction_digits(&o(None, Some(1.0)), Some(2)), Ok((1, 1)));
        assert_eq!(fraction_digits(&o(None, None), Some(0)), Ok((0, 0)));
        assert!(fraction_digits(&o(Some(4.0), Some(2.0)), None).is_err());
        assert!(fraction_digits(&o(Some(101.0), None), None).is_err());
    }

    #[test]
    fn skeletons_spell_units_and_precision() {
        let request = |style| NumberRequest {
            locale: "en",
            style,
            min_fraction_digits: 1,
            max_fraction_digits: 3,
        };
        assert_eq!(
            request(Style::Unit {
                unit: "kilometer",
                per: Some("hour")
            })
            .skeleton(),
            "measure-unit/length-kilometer per-measure-unit/duration-hour .0## rounding-mode-half-up"
        );
        assert_eq!(
            NumberRequest {
                max_fraction_digits: 0,
                min_fraction_digits: 0,
                ..request(Style::Currency("JPY"))
            }
            .skeleton(),
            "currency/JPY precision-integer rounding-mode-half-up"
        );
    }

    /// The platform's library, on platforms with a backend a test binary
    /// can open: Android's needs an app's JVM.
    #[cfg(any(target_vendor = "apple", target_os = "linux"))]
    mod system {
        use super::super::*;

        fn system() -> SystemLocale {
            SystemLocale::new(platform::open().expect("the platform's locale library"))
        }

        fn collator(locale: &str, case: bool, diacritics: bool) -> Collator {
            Collator {
                case_sensitive: case,
                diacritic_sensitive: diacritics,
                locale: system().resolved_locale(Some(locale)),
            }
        }

        fn format(
            number: f64,
            locale: &str,
            f: impl FnOnce(&mut NumberFormat<'_>),
        ) -> Result<String, String> {
            let mut o = NumberFormat {
                locale: Some(locale),
                ..NumberFormat::default()
            };
            f(&mut o);
            system().format_number(number, &o)
        }

        #[test]
        fn collation_follows_the_locale_and_its_extensions() {
            let s = system();
            let order = |a, b, c: &Collator| s.compare(a, b, c);
            assert_eq!(
                order("ä", "z", &collator("de-DE", false, false)),
                Ordering::Less
            );
            assert_eq!(
                order("ä", "z", &collator("sv-SE", false, false)),
                Ordering::Greater
            );
            assert_eq!(
                order("ä", "ae", &collator("de-DE-u-co-phonebk", false, false)),
                Ordering::Equal
            );
            assert_eq!(
                order("2", "10", &collator("en-US-u-kn", false, false)),
                Ordering::Less
            );
            assert_eq!(
                order("2", "10", &collator("en-US", false, false)),
                Ordering::Greater
            );
            assert_eq!(
                order("A", "a", &collator("en-US", false, true)),
                Ordering::Equal
            );
            assert_ne!(
                order("A", "a", &collator("en-US", true, false)),
                Ordering::Equal
            );
            assert_eq!(
                order("é", "e", &collator("en-US", true, false)),
                Ordering::Equal
            );
            assert_ne!(
                order("é", "e", &collator("en-US", false, true)),
                Ordering::Equal
            );
        }

        #[test]
        fn resolved_locales_keep_the_extensions_they_honor() {
            let s = system();
            assert_eq!(s.resolved_locale(Some("sv-SE")), "sv-SE");
            assert_eq!(
                s.resolved_locale(Some("de-DE-u-kn-co-phonebk")),
                "de-DE-u-co-phonebk-kn"
            );
            assert_eq!(s.resolved_locale(Some("en-US-u-co-phonebk")), "en-US");
            let default = s.resolved_locale(None);
            assert!(
                LanguageTag::parse(&default).is_some_and(|t| t.keywords.is_empty()),
                "{default}"
            );
            for unsupported in ["xx-ZZ", "", "en_US"] {
                assert_eq!(
                    s.resolved_locale(Some(unsupported)),
                    default,
                    "{unsupported}"
                );
            }
        }

        #[test]
        fn numbers_format_with_the_locale_currency_and_unit() {
            assert_eq!(format(1234.5, "de-DE", |_| {}).unwrap(), "1.234,5");
            // Apple's data has the half-width yen sign, as Safari shows;
            // CLDR's has the full-width one.
            let yen = format(1234.6, "ja-JP", |o| o.currency = Some("JPY")).unwrap();
            assert!(matches!(yen.as_str(), "¥1,235" | "￥1,235"), "{yen}");
            assert_eq!(
                format(1234.5, "de-DE", |o| o.currency = Some("eur")).unwrap(),
                "1.234,50\u{a0}€"
            );
            assert_eq!(
                format(-12.3, "en-US", |o| o.currency = Some("USD")).unwrap(),
                "-$12.30"
            );
            assert_eq!(
                format(12.3, "en-US", |o| {
                    o.currency = Some("USD");
                    o.max_fraction_digits = Some(0.0);
                })
                .unwrap(),
                "$12"
            );
            assert_eq!(
                format(2.5, "en-US", |o| o.max_fraction_digits = Some(0.0)).unwrap(),
                "3"
            );
            assert_eq!(
                format(21.5, "en-US", |o| o.unit = Some("celsius")).unwrap(),
                "21.5°C"
            );
            assert_eq!(
                format(80.0, "en-US", |o| o.unit = Some("kilometer-per-hour")).unwrap(),
                "80 km/h"
            );
            assert_eq!(
                format(12.0, "en-US", |o| o.unit = Some("percent")).unwrap(),
                "12%"
            );
        }

        #[test]
        fn bad_currencies_and_units_fail() {
            assert_eq!(
                format(1.0, "en-US", |o| o.currency = Some("US$")),
                Err("Invalid currency code : US$".into())
            );
            assert!(format(1.0, "en-US", |o| o.unit = Some("furlong")).is_err());
            assert!(format(1.0, "en-US", |o| o.unit = Some("meter-per-furlong")).is_err());
        }
    }
}

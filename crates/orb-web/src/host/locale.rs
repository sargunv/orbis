//! Locale expressions through the browser's `Intl`, as GL JS evaluates
//! them: every locale, collation, currency and unit the browser knows,
//! where the built-in tables know a few.

use std::cell::RefCell;
use std::cmp::Ordering;

use js_sys::{Array, Function, Object, Reflect};
use orb_expr::{Collator, NumberFormat};
use wasm_bindgen::{JsCast, JsValue};

pub(crate) struct BrowserLocale;

/// `number-format`'s options, as the cache's key: the locale, currency and
/// unit, and the fraction digits.
type FormatKey = ([Option<String>; 3], [Option<f64>; 2]);

// The last of each `Intl` object made: an expression asks for one per
// feature, and making one costs a hundred times what using it does (about
// 20 µs against 0.2 µs in V8).
thread_local! {
    /// The last requested locale and what it resolved to.
    static RESOLVED: RefCell<Option<(Option<String>, String)>> = const { RefCell::new(None) };
    /// The last collator's locale and sensitivity, and its `compare`.
    static COLLATOR: RefCell<Option<(String, &'static str, Function)>> = const { RefCell::new(None) };
    /// The last number format's options, and its `format`.
    static FORMAT: RefCell<Option<(FormatKey, Function)>> = const { RefCell::new(None) };
}

/// `new Intl[class](locale ?? [], options)`, or the message it throws.
fn construct(
    class: &str,
    locale: Option<&str>,
    options: &[(&str, JsValue)],
) -> Result<JsValue, String> {
    let message = |e: JsValue| {
        Reflect::get(&e, &"message".into())
            .ok()
            .and_then(|m| m.as_string())
            .unwrap_or_default()
    };
    let class = Reflect::get(&js_sys::global(), &"Intl".into())
        .and_then(|intl| Reflect::get(&intl, &class.into()))
        .map_err(message)?;
    let object = Object::new();
    for (key, value) in options.iter().filter(|(_, v)| !v.is_undefined()) {
        Reflect::set(&object, &(*key).into(), value).map_err(message)?;
    }
    let locales = locale.map_or_else(|| Array::new().into(), JsValue::from);
    Reflect::construct(
        class.unchecked_ref::<Function>(),
        &Array::of2(&locales, &object),
    )
    .map_err(message)
}

/// `instance[method]`, a function bound to it as `Intl`'s getters are.
fn method(instance: &JsValue, name: &str) -> Option<Function> {
    Reflect::get(instance, &name.into()).ok()?.dyn_into().ok()
}

impl orb_expr::HostLocale for BrowserLocale {
    fn resolved_locale(&self, requested: Option<&str>) -> String {
        RESOLVED.with_borrow_mut(|cached| {
            if let Some((r, resolved)) = cached
                && r.as_deref() == requested
            {
                return resolved.clone();
            }
            let resolved = construct("Collator", requested, &[])
                .ok()
                .and_then(|c| method(&c, "resolvedOptions")?.call0(&c).ok())
                .and_then(|o| Reflect::get(&o, &"locale".into()).ok()?.as_string())
                .unwrap_or_else(|| "en-US".into());
            *cached = Some((requested.map(str::to_owned), resolved.clone()));
            resolved
        })
    }

    fn compare(&self, lhs: &str, rhs: &str, c: &Collator) -> Ordering {
        let sensitivity = match (c.case_sensitive, c.diacritic_sensitive) {
            (true, true) => "variant",
            (true, false) => "case",
            (false, true) => "accent",
            (false, false) => "base",
        };
        COLLATOR.with_borrow_mut(|cached| {
            if !cached
                .as_ref()
                .is_some_and(|(l, s, _)| *l == c.locale && *s == sensitivity)
            {
                let options = [
                    ("sensitivity", sensitivity.into()),
                    ("usage", "search".into()),
                ];
                *cached = construct("Collator", Some(&c.locale), &options)
                    .ok()
                    .and_then(|collator| method(&collator, "compare"))
                    .map(|compare| (c.locale.clone(), sensitivity, compare));
            }
            let order = cached
                .as_ref()
                .and_then(|(_, _, compare)| {
                    compare
                        .call2(&JsValue::UNDEFINED, &lhs.into(), &rhs.into())
                        .ok()?
                        .as_f64()
                })
                .unwrap_or(0.0);
            order.partial_cmp(&0.0).unwrap_or(Ordering::Equal)
        })
    }

    fn format_number(&self, number: f64, o: &NumberFormat<'_>) -> Result<String, String> {
        let strings = [o.locale, o.currency, o.unit];
        let digits = [o.min_fraction_digits, o.max_fraction_digits];
        let format = FORMAT.with_borrow_mut(|cached| {
            if let Some(((s, d), format)) = cached
                && s.iter().map(Option::as_deref).eq(strings)
                && *d == digits
            {
                return Ok(format.clone());
            }
            let style = if o.currency.is_some() {
                "currency"
            } else if o.unit.is_some() {
                "unit"
            } else {
                "decimal"
            };
            let format = construct(
                "NumberFormat",
                o.locale,
                &[
                    ("style", style.into()),
                    ("currency", o.currency.into()),
                    ("unit", o.unit.into()),
                    ("minimumFractionDigits", o.min_fraction_digits.into()),
                    ("maximumFractionDigits", o.max_fraction_digits.into()),
                ],
            )?;
            let format = method(&format, "format").ok_or_else(String::new)?;
            *cached = Some((
                (strings.map(|s| s.map(str::to_owned)), digits),
                format.clone(),
            ));
            Ok::<_, String>(format)
        })?;
        format
            .call1(&JsValue::UNDEFINED, &number.into())
            .ok()
            .and_then(|s| s.as_string())
            .ok_or_else(String::new)
    }
}

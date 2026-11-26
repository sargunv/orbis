//! Android: the framework's ICU, `android.icu` (API level 24), through
//! JNI in the app's JVM, which [`init`](crate::android::init) hands over;
//! Android's NDK has no ICU4C API to call instead. The user's locale is
//! `ULocale.getDefault()`, the app's.
//!
//! Numbers format through `NumberFormatter.forSkeleton` from API level 30,
//! with the skeletons Apple's ICU4C takes. Before it, `NumberFormat`
//! formats them, wrapped in a `MeasureFormat` of the short width for a
//! unit: the same data, but a unit per another needs API level 26
//! (`formatMeasurePerUnit`), and a unit is there only from the level that
//! added its `MeasureUnit` (`mile-scandinavian` 26, `percent` per a unit
//! and `petabyte` 30); one missing fails as Intl fails an unknown unit.
//! The built-in tables would do worse: they know nine units and six
//! currency symbols.
//!
//! Each thread that evaluates an expression attaches to the JVM once and
//! stays attached; collators and formatters are global references, used
//! only on the thread that made them.

use std::cmp::Ordering;

use jni::errors::Error as JniError;
use jni::objects::{Global, JObject, JObjectArray, JString, JValue};
use jni::strings::{JNIStr, JNIString};
use jni::{Env, JavaVM, jni_sig, jni_str};

use super::{Backend, NumberRequest, Sensitivity, Style, SystemCollator, SystemNumberFormat};

/// The first API level with `android.icu`.
const ICU_LEVEL: i32 = 24;
/// The first API level with `MeasureFormat.formatMeasurePerUnit`.
const PER_UNIT_LEVEL: i32 = 26;
/// The first API level with `android.icu.number.NumberFormatter`.
const NUMBER_FORMATTER_LEVEL: i32 = 30;

// `android.icu.text.Collator`'s strengths and decomposition mode, and
// `android.icu.math.BigDecimal.ROUND_HALF_UP`: half away from zero, as
// `Intl` rounds.
const PRIMARY: i32 = 0;
const SECONDARY: i32 = 1;
const TERTIARY: i32 = 2;
const CANONICAL_DECOMPOSITION: i32 = 17;
const ROUND_HALF_UP: i32 = 4;

/// The backend, once [`init`](crate::android::init) gave the app's JVM:
/// before it, `None`, and `locale::install` tries again from `init`.
pub(in crate::locale) fn open() -> Option<Box<dyn Backend>> {
    let vm = crate::android::java_vm()?;
    let level = java(vm, |env| {
        Ok(env
            .get_static_field(
                jni_str!("android/os/Build$VERSION"),
                jni_str!("SDK_INT"),
                jni_sig!("I"),
            )?
            .i()?)
    })
    .ok()?;
    (level >= ICU_LEVEL).then(|| Box::new(AndroidIcu { vm, level }) as Box<dyn Backend>)
}

/// Why a call into `android.icu` failed.
enum Failure {
    Jni(JniError),
    /// What the system's ICU lacks, said as `Intl` would.
    Unsupported(String),
}

impl From<JniError> for Failure {
    fn from(error: JniError) -> Self {
        Self::Jni(error)
    }
}

impl Failure {
    fn into_message(self) -> String {
        match self {
            Self::Jni(JniError::CaughtJavaException { name, msg, .. }) => format!("{name}: {msg}"),
            Self::Jni(error) => error.to_string(),
            Self::Unsupported(message) => message,
        }
    }
}

/// Runs `f` on this thread attached to the JVM (attaching it for good if
/// it isn't), in a local reference frame of its own. A Java exception `f`
/// leaves is caught and fails it.
fn java<T>(vm: &JavaVM, f: impl FnOnce(&mut Env<'_>) -> Result<T, Failure>) -> Result<T, String> {
    vm.attach_current_thread(f).map_err(Failure::into_message)
}

/// The text of a Java string.
fn string(env: &Env<'_>, object: JObject<'_>) -> Result<String, JniError> {
    env.cast_local::<JString>(object)?.try_to_string(env)
}

/// `ULocale.forLanguageTag(tag)`.
fn locale<'local>(env: &mut Env<'local>, tag: &str) -> Result<JObject<'local>, JniError> {
    let tag = env.new_string(tag)?;
    env.call_static_method(
        jni_str!("android/icu/util/ULocale"),
        jni_str!("forLanguageTag"),
        jni_sig!("(Ljava/lang/String;)Landroid/icu/util/ULocale;"),
        &[JValue::from(&tag)],
    )?
    .l()
}

/// `Currency.getInstance(code)`.
fn currency<'local>(env: &mut Env<'local>, code: &str) -> Result<JObject<'local>, JniError> {
    let code = env.new_string(code)?;
    env.call_static_method(
        jni_str!("android/icu/util/Currency"),
        jni_str!("getInstance"),
        jni_sig!("(Ljava/lang/String;)Landroid/icu/util/Currency;"),
        &[JValue::from(&code)],
    )?
    .l()
}

/// The `MeasureUnit` constant of a unit `Intl` names (`FLUID_OUNCE` for
/// `fluid-ounce`), if this API level has it.
/// `whole` is the unit `number-format` was asked for, which a missing one
/// fails naming.
fn measure_unit<'local>(
    env: &mut Env<'local>,
    unit: &str,
    whole: &str,
) -> Result<JObject<'local>, Failure> {
    let name = JNIString::new(unit.to_ascii_uppercase().replace('-', "_"));
    let class = jni_str!("android/icu/util/MeasureUnit");
    // Durations from a second to a year are declared as `TimeUnit`s.
    let field = if matches!(
        unit,
        "second" | "minute" | "hour" | "day" | "week" | "month" | "year"
    ) {
        env.get_static_field(class, &name, jni_sig!("Landroid/icu/util/TimeUnit;"))
    } else {
        env.get_static_field(class, &name, jni_sig!("Landroid/icu/util/MeasureUnit;"))
    };
    field.and_then(|f| f.l()).map_err(|_| {
        env.exception_clear();
        Failure::Unsupported(format!(
            "Invalid unit argument for Intl.NumberFormat() '{whole}'"
        ))
    })
}

/// `android.icu` in the app's JVM.
struct AndroidIcu {
    vm: &'static JavaVM,
    /// `Build.VERSION.SDK_INT`.
    level: i32,
}

impl Backend for AndroidIcu {
    fn default_locale(&self) -> Option<String> {
        java(self.vm, |env| {
            let locale = env
                .call_static_method(
                    jni_str!("android/icu/util/ULocale"),
                    jni_str!("getDefault"),
                    jni_sig!("()Landroid/icu/util/ULocale;"),
                    &[],
                )?
                .l()?;
            let tag = env
                .call_method(
                    &locale,
                    jni_str!("toLanguageTag"),
                    jni_sig!("()Ljava/lang/String;"),
                    &[],
                )?
                .l()?;
            Ok(string(env, tag)?)
        })
        .ok()
    }

    fn languages(&self) -> Vec<String> {
        java(self.vm, |env| {
            let locales = env
                .call_static_method(
                    jni_str!("android/icu/util/ULocale"),
                    jni_str!("getAvailableLocales"),
                    jni_sig!("()[Landroid/icu/util/ULocale;"),
                    &[],
                )?
                .l()?;
            let locales = env.cast_local::<JObjectArray>(locales)?;
            let mut languages = Vec::new();
            for i in 0..locales.len(env)? {
                languages.push(env.with_local_frame(4, |env| -> Result<_, JniError> {
                    let locale = locales.get_element(env, i)?;
                    let language = env
                        .call_method(
                            &locale,
                            jni_str!("getLanguage"),
                            jni_sig!("()Ljava/lang/String;"),
                            &[],
                        )?
                        .l()?;
                    string(env, language)
                })?);
            }
            Ok(languages)
        })
        .unwrap_or_default()
    }

    fn has_collation(&self, locale: &str, collation: &str) -> bool {
        java(self.vm, |env| {
            let keyword = env.new_string("collation")?;
            let bcp47 = env.new_string(collation)?;
            // ICU lists types by their legacy names: `phonebook` for
            // `phonebk`.
            let legacy = env
                .call_static_method(
                    jni_str!("android/icu/util/ULocale"),
                    jni_str!("toLegacyType"),
                    jni_sig!("(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;"),
                    &[JValue::from(&keyword), JValue::from(&bcp47)],
                )?
                .l()?;
            let legacy = if legacy.is_null() {
                collation.to_owned()
            } else {
                string(env, legacy)?
            };
            let locale = self::locale(env, locale)?;
            let types = env
                .call_static_method(
                    jni_str!("android/icu/text/Collator"),
                    jni_str!("getKeywordValuesForLocale"),
                    jni_sig!("(Ljava/lang/String;Landroid/icu/util/ULocale;Z)[Ljava/lang/String;"),
                    &[
                        JValue::from(&keyword),
                        JValue::from(&locale),
                        JValue::Bool(false),
                    ],
                )?
                .l()?;
            let types = env.cast_local::<JObjectArray<JString>>(types)?;
            for i in 0..types.len(env)? {
                let found = env.with_local_frame(2, |env| -> Result<_, JniError> {
                    Ok(types.get_element(env, i)?.try_to_string(env)? == legacy)
                })?;
                if found {
                    return Ok(true);
                }
            }
            Ok(false)
        })
        .unwrap_or(false)
    }

    fn collator(&self, locale: &str, sensitivity: Sensitivity) -> Option<Box<dyn SystemCollator>> {
        java(self.vm, |env| {
            // The tag's `co`, `kf` and `kn` become the collator's settings.
            let locale = self::locale(env, locale)?;
            let collator = env
                .call_static_method(
                    jni_str!("android/icu/text/Collator"),
                    jni_str!("getInstance"),
                    jni_sig!("(Landroid/icu/util/ULocale;)Landroid/icu/text/Collator;"),
                    &[JValue::from(&locale)],
                )?
                .l()?;
            let strength = match (sensitivity.case, sensitivity.diacritics) {
                (true, true) => TERTIARY,
                (false, true) => SECONDARY,
                _ => PRIMARY,
            };
            env.call_method(
                &collator,
                jni_str!("setStrength"),
                jni_sig!("(I)V"),
                &[JValue::Int(strength)],
            )?;
            env.call_method(
                &collator,
                jni_str!("setDecomposition"),
                jni_sig!("(I)V"),
                &[JValue::Int(CANONICAL_DECOMPOSITION)],
            )?;
            if sensitivity.case
                && !sensitivity.diacritics
                && env.is_instance_of(&collator, jni_str!("android/icu/text/RuleBasedCollator"))?
            {
                env.call_method(
                    &collator,
                    jni_str!("setCaseLevel"),
                    jni_sig!("(Z)V"),
                    &[JValue::Bool(true)],
                )?;
            }
            Ok(Box::new(JavaCollator {
                vm: self.vm,
                collator: env.new_global_ref(&collator)?,
            }) as Box<dyn SystemCollator>)
        })
        .ok()
    }

    fn currency_digits(&self, currency: &str) -> Option<u8> {
        java(self.vm, |env| {
            let currency = self::currency(env, currency)?;
            let digits = env
                .call_method(
                    &currency,
                    jni_str!("getDefaultFractionDigits"),
                    jni_sig!("()I"),
                    &[],
                )?
                .i()?;
            Ok(u8::try_from(digits).ok())
        })
        .ok()
        .flatten()
    }

    fn number_format(
        &self,
        request: &NumberRequest<'_>,
    ) -> Result<Box<dyn SystemNumberFormat>, String> {
        let formatter = java(self.vm, |env| {
            if self.level >= NUMBER_FORMATTER_LEVEL {
                number_formatter(env, request)
            } else {
                number_format(env, request, self.level)
            }
        })?;
        Ok(Box::new(JavaFormatter {
            vm: self.vm,
            formatter,
        }))
    }
}

/// A `LocalizedNumberFormatter` for the request's skeleton.
fn number_formatter(env: &mut Env<'_>, request: &NumberRequest<'_>) -> Result<Formatter, Failure> {
    let skeleton = env.new_string(request.skeleton())?;
    let unlocalized = env
        .call_static_method(
            jni_str!("android/icu/number/NumberFormatter"),
            jni_str!("forSkeleton"),
            jni_sig!("(Ljava/lang/String;)Landroid/icu/number/UnlocalizedNumberFormatter;"),
            &[JValue::from(&skeleton)],
        )?
        .l()?;
    let locale = locale(env, request.locale)?;
    let formatter = env
        .call_method(
            &unlocalized,
            jni_str!("locale"),
            jni_sig!("(Landroid/icu/util/ULocale;)Landroid/icu/number/LocalizedNumberFormatter;"),
            &[JValue::from(&locale)],
        )?
        .l()?;
    Ok(Formatter::Skeleton(env.new_global_ref(&formatter)?))
}

/// A `NumberFormat` in the decimal, currency or percent style, or a
/// `MeasureFormat` of the short width over a decimal one, with the
/// request's fraction digits and rounding.
fn number_format(
    env: &mut Env<'_>,
    request: &NumberRequest<'_>,
    level: i32,
) -> Result<Formatter, Failure> {
    let locale = locale(env, request.locale)?;
    fn instance<'local>(
        env: &mut Env<'local>,
        name: &JNIStr,
        locale: &JObject<'_>,
    ) -> Result<JObject<'local>, JniError> {
        env.call_static_method(
            jni_str!("android/icu/text/NumberFormat"),
            name,
            jni_sig!("(Landroid/icu/util/ULocale;)Landroid/icu/text/NumberFormat;"),
            &[JValue::from(locale)],
        )?
        .l()
    }
    let mut units = None;
    let format = match request.style {
        Style::Decimal => instance(env, jni_str!("getInstance"), &locale)?,
        Style::Currency(code) => {
            let format = instance(env, jni_str!("getCurrencyInstance"), &locale)?;
            let currency = currency(env, code)?;
            env.call_method(
                &format,
                jni_str!("setCurrency"),
                jni_sig!("(Landroid/icu/util/Currency;)V"),
                &[JValue::from(&currency)],
            )?;
            format
        }
        // A percent is already a percentage, as in `Intl`'s unit style.
        Style::Unit {
            unit: "percent",
            per: None,
        } => {
            let format = instance(env, jni_str!("getPercentInstance"), &locale)?;
            if env.is_instance_of(&format, jni_str!("android/icu/text/DecimalFormat"))? {
                env.call_method(
                    &format,
                    jni_str!("setMultiplier"),
                    jni_sig!("(I)V"),
                    &[JValue::Int(1)],
                )?;
            }
            format
        }
        Style::Unit { unit, per } => {
            let whole = match per {
                Some(per) => format!("{unit}-per-{per}"),
                None => unit.to_owned(),
            };
            if per.is_some() && level < PER_UNIT_LEVEL {
                return Err(Failure::Unsupported(format!(
                    "Invalid unit argument for Intl.NumberFormat() '{whole}'"
                )));
            }
            let unit = measure_unit(env, unit, &whole)?;
            let per = per.map(|per| measure_unit(env, per, &whole)).transpose()?;
            units = Some((
                env.new_global_ref(&unit)?,
                per.map(|per| env.new_global_ref(&per)).transpose()?,
            ));
            instance(env, jni_str!("getInstance"), &locale)?
        }
    };
    // After `setCurrency`, which sets the currency's digits.
    for (setter, value) in [
        (
            jni_str!("setMaximumFractionDigits"),
            request.max_fraction_digits,
        ),
        (
            jni_str!("setMinimumFractionDigits"),
            request.min_fraction_digits,
        ),
    ] {
        env.call_method(
            &format,
            setter,
            jni_sig!("(I)V"),
            &[JValue::Int(value.into())],
        )?;
    }
    env.call_method(
        &format,
        jni_str!("setRoundingMode"),
        jni_sig!("(I)V"),
        &[JValue::Int(ROUND_HALF_UP)],
    )?;
    let Some((unit, per)) = units else {
        return Ok(Formatter::Number(env.new_global_ref(&format)?));
    };
    let short = env
        .get_static_field(
            jni_str!("android/icu/text/MeasureFormat$FormatWidth"),
            jni_str!("SHORT"),
            jni_sig!("Landroid/icu/text/MeasureFormat$FormatWidth;"),
        )?
        .l()?;
    let format = env
        .call_static_method(
            jni_str!("android/icu/text/MeasureFormat"),
            jni_str!("getInstance"),
            jni_sig!(
                "(Landroid/icu/util/ULocale;Landroid/icu/text/MeasureFormat$FormatWidth;Landroid/icu/text/NumberFormat;)Landroid/icu/text/MeasureFormat;"
            ),
            &[
                JValue::from(&locale),
                JValue::from(&short),
                JValue::from(&format),
            ],
        )?
        .l()?;
    Ok(Formatter::Measure {
        format: env.new_global_ref(&format)?,
        unit,
        per,
    })
}

/// An `android.icu.text.Collator`.
struct JavaCollator {
    vm: &'static JavaVM,
    collator: Global<JObject<'static>>,
}

impl SystemCollator for JavaCollator {
    fn compare(&self, lhs: &str, rhs: &str) -> Ordering {
        // Only JNI failing, as out of memory, fails this.
        java(self.vm, |env| {
            let (lhs, rhs) = (env.new_string(lhs)?, env.new_string(rhs)?);
            let order = env
                .call_method(
                    &self.collator,
                    jni_str!("compare"),
                    jni_sig!("(Ljava/lang/String;Ljava/lang/String;)I"),
                    &[JValue::from(&lhs), JValue::from(&rhs)],
                )?
                .i()?;
            Ok(order.cmp(&0))
        })
        .unwrap_or(Ordering::Equal)
    }
}

/// A number formatter of one API or another.
enum Formatter {
    /// A `LocalizedNumberFormatter` (API level 30).
    Skeleton(Global<JObject<'static>>),
    /// A `NumberFormat`.
    Number(Global<JObject<'static>>),
    /// A `MeasureFormat` and the `MeasureUnit`s it formats in: the unit,
    /// and the one it's per.
    Measure {
        format: Global<JObject<'static>>,
        unit: Global<JObject<'static>>,
        per: Option<Global<JObject<'static>>>,
    },
}

struct JavaFormatter {
    vm: &'static JavaVM,
    formatter: Formatter,
}

impl SystemNumberFormat for JavaFormatter {
    fn format(&self, number: f64) -> Result<String, String> {
        java(self.vm, |env| {
            let formatted = match &self.formatter {
                Formatter::Skeleton(formatter) => {
                    let formatted = env
                        .call_method(
                            formatter,
                            jni_str!("format"),
                            jni_sig!("(D)Landroid/icu/number/FormattedNumber;"),
                            &[JValue::Double(number)],
                        )?
                        .l()?;
                    env.call_method(
                        &formatted,
                        jni_str!("toString"),
                        jni_sig!("()Ljava/lang/String;"),
                        &[],
                    )?
                    .l()?
                }
                Formatter::Number(format) => env
                    .call_method(
                        format,
                        jni_str!("format"),
                        jni_sig!("(D)Ljava/lang/String;"),
                        &[JValue::Double(number)],
                    )?
                    .l()?,
                Formatter::Measure { format, unit, per } => {
                    let amount = env
                        .call_static_method(
                            jni_str!("java/lang/Double"),
                            jni_str!("valueOf"),
                            jni_sig!("(D)Ljava/lang/Double;"),
                            &[JValue::Double(number)],
                        )?
                        .l()?;
                    let unit = env.new_local_ref(unit)?;
                    let measure = env.new_object(
                        jni_str!("android/icu/util/Measure"),
                        jni_sig!("(Ljava/lang/Number;Landroid/icu/util/MeasureUnit;)V"),
                        &[JValue::from(&amount), JValue::from(&unit)],
                    )?;
                    match per {
                        None => env
                            .call_method(
                                format,
                                jni_str!("format"),
                                jni_sig!("(Ljava/lang/Object;)Ljava/lang/String;"),
                                &[JValue::from(&measure)],
                            )?
                            .l()?,
                        Some(per) => {
                            let per = env.new_local_ref(per)?;
                            let builder = env.new_object(
                                jni_str!("java/lang/StringBuilder"),
                                jni_sig!("()V"),
                                &[],
                            )?;
                            let position = env.new_object(
                                jni_str!("java/text/FieldPosition"),
                                jni_sig!("(I)V"),
                                &[JValue::Int(0)],
                            )?;
                            let builder = env
                                .call_method(
                                    format,
                                    jni_str!("formatMeasurePerUnit"),
                                    jni_sig!(
                                        "(Landroid/icu/util/Measure;Landroid/icu/util/MeasureUnit;Ljava/lang/StringBuilder;Ljava/text/FieldPosition;)Ljava/lang/StringBuilder;"
                                    ),
                                    &[
                                        JValue::from(&measure),
                                        JValue::from(&per),
                                        JValue::from(&builder),
                                        JValue::from(&position),
                                    ],
                                )?
                                .l()?;
                            env.call_method(
                                &builder,
                                jni_str!("toString"),
                                jni_sig!("()Ljava/lang/String;"),
                                &[],
                            )?
                            .l()?
                        }
                    }
                }
            };
            Ok(string(env, formatted)?)
        })
    }
}

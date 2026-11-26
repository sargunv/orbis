//! A [`Backend`] over ICU4C's C API, for platforms whose system ICU is
//! ICU4C: Apple's `libicucore`, Linux's `libicuuc` and `libicui18n`,
//! Windows' `icu.dll`. A platform hands [`Icu::load`] a resolver for the
//! functions' plain names (it adds a version suffix such as `_74` if its
//! build has them) and its user's locale. ICU 62 or later, for number
//! skeletons with `measure-unit/` units.

#![allow(unsafe_code, reason = "ICU's C API, resolved at run time")]

use std::cmp::Ordering;
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr::NonNull;

use super::{Backend, NumberRequest, Sensitivity, SystemCollator, SystemNumberFormat};

/// An ICU status: positive for an error, negative for a warning.
type UErrorCode = i32;
const U_ZERO_ERROR: UErrorCode = 0;
const U_BUFFER_OVERFLOW_ERROR: UErrorCode = 15;

// `UColAttribute` and `UColAttributeValue`.
const UCOL_CASE_LEVEL: i32 = 3;
const UCOL_NORMALIZATION_MODE: i32 = 4;
const UCOL_STRENGTH: i32 = 5;
const UCOL_PRIMARY: i32 = 0;
const UCOL_SECONDARY: i32 = 1;
const UCOL_TERTIARY: i32 = 2;
const UCOL_ON: i32 = 17;

/// Declares ICU's functions as pointers resolved when ICU loads.
macro_rules! functions {
    ($($name:ident($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[expect(non_snake_case, reason = "ICU's names")]
        struct Functions {
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Functions {
            fn load(resolve: &mut dyn FnMut(&str) -> *mut c_void) -> Option<Self> {
                Some(Self {
                    $($name: {
                        let symbol = NonNull::new(resolve(stringify!($name)))?;
                        // SAFETY: the library exports the function under
                        // this name, with ICU's signature, declared here.
                        unsafe {
                            std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) $(-> $ret)?>(
                                symbol.as_ptr(),
                            )
                        }
                    },)*
                })
            }
        }
    };
}

functions! {
    uloc_countAvailable() -> i32;
    uloc_getAvailable(i32) -> *const c_char;
    uloc_forLanguageTag(*const c_char, *mut c_char, i32, *mut i32, *mut UErrorCode) -> i32;
    uloc_toLegacyType(*const c_char, *const c_char) -> *const c_char;
    uenum_next(*mut c_void, *mut i32, *mut UErrorCode) -> *const c_char;
    uenum_close(*mut c_void);
    ucol_open(*const c_char, *mut UErrorCode) -> *mut c_void;
    ucol_close(*mut c_void);
    ucol_setAttribute(*mut c_void, i32, i32, *mut UErrorCode);
    ucol_strcollUTF8(*mut c_void, *const c_char, i32, *const c_char, i32, *mut UErrorCode) -> i32;
    ucol_getKeywordValuesForLocale(*const c_char, *const c_char, u8, *mut UErrorCode) -> *mut c_void;
    ucurr_getDefaultFractionDigits(*const u16, *mut UErrorCode) -> i32;
    unumf_openForSkeletonAndLocale(*const u16, i32, *const c_char, *mut UErrorCode) -> *mut c_void;
    unumf_close(*mut c_void);
    unumf_openResult(*mut UErrorCode) -> *mut c_void;
    unumf_closeResult(*mut c_void);
    unumf_formatDouble(*const c_void, f64, *mut c_void, *mut UErrorCode);
    unumf_resultToString(*const c_void, *mut u16, i32, *mut UErrorCode) -> i32;
}

/// The system's ICU4C.
pub(super) struct Icu {
    icu: &'static Functions,
    default_locale: Option<String>,
}

impl Icu {
    /// ICU through `resolve`, or `None` when it misses a function.
    /// `default_locale` is the user's (see [`Backend::default_locale`]).
    pub(super) fn load(
        mut resolve: impl FnMut(&str) -> *mut c_void,
        default_locale: Option<String>,
    ) -> Option<Self> {
        let icu = Functions::load(&mut resolve)?;
        Some(Self {
            // Loaded once a process, and used until it exits.
            icu: Box::leak(Box::new(icu)),
            default_locale,
        })
    }

    /// The ICU locale identifier of a BCP 47 tag.
    fn locale_id(&self, tag: &str) -> Option<CString> {
        let tag = CString::new(tag).ok()?;
        let mut buffer = vec![0u8; 160];
        loop {
            let mut status = U_ZERO_ERROR;
            let capacity = i32::try_from(buffer.len()).ok()?;
            // SAFETY: the tag is NUL-terminated and the buffer holds
            // `capacity` bytes.
            let len = unsafe {
                (self.icu.uloc_forLanguageTag)(
                    tag.as_ptr(),
                    buffer.as_mut_ptr().cast(),
                    capacity,
                    std::ptr::null_mut(),
                    &raw mut status,
                )
            };
            let len = usize::try_from(len).ok()?;
            if status == U_BUFFER_OVERFLOW_ERROR || len >= buffer.len() {
                buffer.resize(len + 1, 0);
                continue;
            }
            if status > 0 {
                return None;
            }
            buffer.truncate(len);
            return CString::new(buffer).ok();
        }
    }
}

/// Fails on an ICU error, saying what failed.
fn check(status: UErrorCode, what: &str) -> Result<(), String> {
    if status > 0 {
        Err(format!("{what} failed with ICU error {status}"))
    } else {
        Ok(())
    }
}

impl Backend for Icu {
    fn default_locale(&self) -> Option<String> {
        self.default_locale.clone()
    }

    fn languages(&self) -> Vec<String> {
        // SAFETY: plain calls; ICU owns the strings it returns.
        let count = unsafe { (self.icu.uloc_countAvailable)() };
        (0..count)
            .filter_map(|i| {
                let id = unsafe { (self.icu.uloc_getAvailable)(i) };
                if id.is_null() {
                    return None;
                }
                // SAFETY: ICU returns a NUL-terminated identifier.
                let id = unsafe { CStr::from_ptr(id) }.to_str().ok()?;
                Some(id.split(['_', '@']).next()?.to_owned())
            })
            .collect()
    }

    fn has_collation(&self, locale: &str, collation: &str) -> bool {
        let (Some(locale), Ok(collation)) = (self.locale_id(locale), CString::new(collation))
        else {
            return false;
        };
        // SAFETY: the strings are NUL-terminated; ICU returns its own
        // static string or null.
        let legacy =
            unsafe { (self.icu.uloc_toLegacyType)(c"collation".as_ptr(), collation.as_ptr()) };
        let legacy = if legacy.is_null() {
            collation.as_c_str()
        } else {
            // SAFETY: as above.
            unsafe { CStr::from_ptr(legacy) }
        };
        let mut status = U_ZERO_ERROR;
        // SAFETY: the strings are NUL-terminated.
        let values = unsafe {
            (self.icu.ucol_getKeywordValuesForLocale)(
                c"collation".as_ptr(),
                locale.as_ptr(),
                0,
                &raw mut status,
            )
        };
        if values.is_null() || status > 0 {
            return false;
        }
        let mut found = false;
        loop {
            let mut status = U_ZERO_ERROR;
            // SAFETY: the enumeration is open until closed below.
            let value =
                unsafe { (self.icu.uenum_next)(values, std::ptr::null_mut(), &raw mut status) };
            if value.is_null() || status > 0 {
                break;
            }
            // SAFETY: ICU returns a NUL-terminated string.
            if unsafe { CStr::from_ptr(value) } == legacy {
                found = true;
                break;
            }
        }
        // SAFETY: open, and closed only here.
        unsafe { (self.icu.uenum_close)(values) };
        found
    }

    fn collator(&self, locale: &str, sensitivity: Sensitivity) -> Option<Box<dyn SystemCollator>> {
        let locale = self.locale_id(locale)?;
        let mut status = U_ZERO_ERROR;
        // SAFETY: the locale is NUL-terminated.
        let collator =
            NonNull::new(unsafe { (self.icu.ucol_open)(locale.as_ptr(), &raw mut status) })?;
        let collator = IcuCollator {
            icu: self.icu,
            collator,
        };
        if status > 0 {
            return None;
        }
        let strength = match (sensitivity.case, sensitivity.diacritics) {
            (true, true) => UCOL_TERTIARY,
            (false, true) => UCOL_SECONDARY,
            _ => UCOL_PRIMARY,
        };
        let mut attributes = vec![
            (UCOL_STRENGTH, strength),
            (UCOL_NORMALIZATION_MODE, UCOL_ON),
        ];
        if sensitivity.case && !sensitivity.diacritics {
            attributes.push((UCOL_CASE_LEVEL, UCOL_ON));
        }
        for (attribute, value) in attributes {
            // SAFETY: the collator is open.
            unsafe {
                (self.icu.ucol_setAttribute)(
                    collator.collator.as_ptr(),
                    attribute,
                    value,
                    &raw mut status,
                )
            };
        }
        (status <= 0).then(|| Box::new(collator) as Box<dyn SystemCollator>)
    }

    fn currency_digits(&self, currency: &str) -> Option<u8> {
        let code: Vec<u16> = currency.encode_utf16().chain([0]).collect();
        let mut status = U_ZERO_ERROR;
        // SAFETY: the code is NUL-terminated UTF-16.
        let digits =
            unsafe { (self.icu.ucurr_getDefaultFractionDigits)(code.as_ptr(), &raw mut status) };
        if status > 0 {
            None
        } else {
            u8::try_from(digits).ok()
        }
    }

    fn number_format(
        &self,
        request: &NumberRequest<'_>,
    ) -> Result<Box<dyn SystemNumberFormat>, String> {
        let locale = self
            .locale_id(request.locale)
            .ok_or_else(|| format!("Incorrect locale information provided: {}", request.locale))?;
        let skeleton: Vec<u16> = request.skeleton().encode_utf16().collect();
        let len = i32::try_from(skeleton.len()).map_err(|e| e.to_string())?;
        let mut status = U_ZERO_ERROR;
        // SAFETY: the skeleton is `len` UTF-16 units and the locale
        // NUL-terminated.
        let formatter = unsafe {
            (self.icu.unumf_openForSkeletonAndLocale)(
                skeleton.as_ptr(),
                len,
                locale.as_ptr(),
                &raw mut status,
            )
        };
        let formatter = NonNull::new(formatter).map(|formatter| IcuFormatter {
            icu: self.icu,
            formatter,
            result: None,
        });
        check(status, "Opening a number formatter")?;
        let mut formatter = formatter.ok_or("ICU opened no number formatter")?;
        // SAFETY: the status is a live local.
        formatter.result = NonNull::new(unsafe { (self.icu.unumf_openResult)(&raw mut status) });
        check(status, "Opening a formatted number")?;
        if formatter.result.is_none() {
            return Err("ICU opened no formatted number".into());
        }
        Ok(Box::new(formatter))
    }
}

/// An open `UCollator`.
struct IcuCollator {
    icu: &'static Functions,
    collator: NonNull<c_void>,
}

impl Drop for IcuCollator {
    fn drop(&mut self) {
        // SAFETY: open, and closed only here.
        unsafe { (self.icu.ucol_close)(self.collator.as_ptr()) }
    }
}

impl SystemCollator for IcuCollator {
    fn compare(&self, lhs: &str, rhs: &str) -> Ordering {
        let len = |s: &str| i32::try_from(s.len()).unwrap_or(i32::MAX);
        let mut status = U_ZERO_ERROR;
        // SAFETY: the collator is open, and the strings are valid UTF-8
        // of the lengths given.
        let order = unsafe {
            (self.icu.ucol_strcollUTF8)(
                self.collator.as_ptr(),
                lhs.as_ptr().cast(),
                len(lhs),
                rhs.as_ptr().cast(),
                len(rhs),
                &raw mut status,
            )
        };
        order.cmp(&0)
    }
}

/// An open `UNumberFormatter` and the `UFormattedNumber` it formats
/// into.
struct IcuFormatter {
    icu: &'static Functions,
    formatter: NonNull<c_void>,
    result: Option<NonNull<c_void>>,
}

impl Drop for IcuFormatter {
    fn drop(&mut self) {
        // SAFETY: open, and closed only here.
        unsafe {
            if let Some(result) = self.result {
                (self.icu.unumf_closeResult)(result.as_ptr());
            }
            (self.icu.unumf_close)(self.formatter.as_ptr());
        }
    }
}

impl SystemNumberFormat for IcuFormatter {
    fn format(&self, number: f64) -> Result<String, String> {
        let result = self.result.ok_or("ICU opened no formatted number")?;
        let mut status = U_ZERO_ERROR;
        // SAFETY: the formatter and the result are open.
        unsafe {
            (self.icu.unumf_formatDouble)(
                self.formatter.as_ptr(),
                number,
                result.as_ptr(),
                &raw mut status,
            )
        };
        check(status, "Formatting a number")?;
        let mut buffer = vec![0u16; 32];
        loop {
            let capacity = i32::try_from(buffer.len()).map_err(|e| e.to_string())?;
            // SAFETY: the result is open, and the buffer holds `capacity`
            // units.
            let len = unsafe {
                (self.icu.unumf_resultToString)(
                    result.as_ptr(),
                    buffer.as_mut_ptr(),
                    capacity,
                    &raw mut status,
                )
            };
            let len = usize::try_from(len).unwrap_or_default();
            if status == U_BUFFER_OVERFLOW_ERROR {
                status = U_ZERO_ERROR;
                buffer.resize(len, 0);
                continue;
            }
            check(status, "Reading a formatted number")?;
            return Ok(String::from_utf16_lossy(&buffer[..len]));
        }
    }
}

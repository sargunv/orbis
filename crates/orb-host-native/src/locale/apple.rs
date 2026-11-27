//! macOS and iOS: the system ICU, `libicucore`, which every Apple OS
//! ships (without headers) in the shared cache, built without ICU's
//! version suffixes; the user's locale is Foundation's current locale,
//! as Safari's `Intl` takes it.

#![allow(unsafe_code, reason = "dlopen and dlsym")]

use std::ffi::{CString, c_void};

use objc2_core_foundation::CFLocale;

use super::Backend;
use super::icu4c::Icu;

pub(in crate::locale) fn open() -> Option<Box<dyn Backend>> {
    // SAFETY: a path, and flags dlopen takes; the library stays loaded.
    let library = unsafe {
        libc::dlopen(
            c"/usr/lib/libicucore.A.dylib".as_ptr(),
            libc::RTLD_LAZY | libc::RTLD_LOCAL,
        )
    };
    if library.is_null() {
        return None;
    }
    let resolve = |name: &str| -> *mut c_void {
        match CString::new(name) {
            // SAFETY: the library is open, and the name NUL-terminated.
            Ok(name) => unsafe { libc::dlsym(library, name.as_ptr()) },
            Err(_) => std::ptr::null_mut(),
        }
    };
    Some(Box::new(Icu::load(resolve, Some(current_locale()))?))
}

/// The user's current locale's identifier, `en_US` or `de_DE@rg=chzzzz`.
fn current_locale() -> String {
    CFLocale::current()
        .and_then(|locale| locale.identifier())
        .map(|identifier| identifier.to_string())
        .unwrap_or_default()
}

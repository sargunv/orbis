//! macOS and iOS: the system ICU, `libicucore`, which every Apple OS
//! ships (without headers) in the shared cache, built without ICU's
//! version suffixes; the user's locale is Foundation's current locale,
//! as Safari's `Intl` takes it.

#![allow(unsafe_code, reason = "dlopen, dlsym and Core Foundation")]

use std::ffi::{CString, c_void};

use core_foundation::base::TCFType;
use core_foundation::string::CFString;
use core_foundation_sys::locale::{CFLocaleCopyCurrent, CFLocaleGetIdentifier};

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
    // SAFETY: CFLocaleCopyCurrent returns a locale we own, whose
    // identifier we only borrow while it lives.
    unsafe {
        let locale = CFLocaleCopyCurrent();
        let identifier = CFString::wrap_under_get_rule(CFLocaleGetIdentifier(locale)).to_string();
        core_foundation_sys::base::CFRelease(locale.cast());
        identifier
    }
}

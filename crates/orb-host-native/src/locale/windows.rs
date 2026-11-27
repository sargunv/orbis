//! Windows: the system ICU, `icu.dll` in System32 (Windows 10 1903 and
//! later), which exports ICU's C API without version suffixes; the user's
//! locale is the one their regional format names (`GetUserDefaultLocaleName`).

#![allow(
    unsafe_code,
    reason = "LoadLibraryExW, GetProcAddress and the user's locale"
)]

use std::ffi::{CString, c_char, c_void};

use super::Backend;
use super::icu4c::Icu;

/// `LOAD_LIBRARY_SEARCH_SYSTEM32`: System32 alone, never the application's
/// or the current directory, so no other `icu.dll` can stand in.
const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;
/// `LOCALE_NAME_MAX_LENGTH`, in UTF-16 units with the NUL.
const LOCALE_NAME_MAX_LENGTH: usize = 85;

// Kernel32, which every Windows process has loaded, imported as std imports
// it (no import library): nothing new is linked.
#[link(name = "kernel32", kind = "raw-dylib")]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn GetUserDefaultLocaleName(name: *mut u16, len: i32) -> i32;
}

pub(in crate::locale) fn open() -> Option<Box<dyn Backend>> {
    let name: Vec<u16> = "icu.dll".encode_utf16().chain([0]).collect();
    // SAFETY: a NUL-terminated name, no file handle, and a flag the call
    // takes; the library stays loaded for the process.
    let library = unsafe {
        LoadLibraryExW(
            name.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    if library.is_null() {
        return None;
    }
    let resolve = |name: &str| -> *mut c_void {
        match CString::new(name) {
            // SAFETY: the library is loaded, and the name NUL-terminated.
            Ok(name) => unsafe { GetProcAddress(library, name.as_ptr()) },
            Err(_) => std::ptr::null_mut(),
        }
    };
    Some(Box::new(Icu::load(resolve, user_locale())?))
}

/// The user's locale's name, a BCP 47 tag (`de-CH`) or one with a sort
/// order after an underscore (`de-DE_phoneb`), which the seam drops.
pub(crate) fn user_locale() -> Option<String> {
    let mut name = [0u16; LOCALE_NAME_MAX_LENGTH];
    // SAFETY: the buffer holds `LOCALE_NAME_MAX_LENGTH` units.
    let len = unsafe { GetUserDefaultLocaleName(name.as_mut_ptr(), name.len() as i32) };
    // The length counts the NUL; 0 is a failure.
    let len = usize::try_from(len).ok()?.checked_sub(1)?;
    let name = String::from_utf16(name.get(..len)?).ok()?;
    (!name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::super::LanguageTag;

    #[test]
    fn the_users_locale_is_a_language_tag() {
        let locale = super::user_locale().expect("the user's locale");
        assert!(
            LanguageTag::parse(&locale.replace('_', "-")).is_some(),
            "{locale}"
        );
    }
}

//! Linux: the system ICU, `libicuuc` and `libicui18n`, whose functions
//! carry the library's major version as a suffix (`ucol_open_77`) unless
//! the distribution built it without; the user's locale is the POSIX
//! environment's, as browsers on Linux take it.

#![allow(unsafe_code, reason = "dlopen and dlsym")]

use std::ffi::{CString, c_void};
use std::ptr::NonNull;

use super::Backend;
use super::icu4c::Icu;

/// The ICU major versions probed where only versioned sonames are
/// installed, newest first: 62 (2018) has the number skeletons we need.
const VERSIONS: std::ops::RangeInclusive<u32> = 62..=150;

pub(in crate::locale) fn open() -> Option<Box<dyn Backend>> {
    let (uc, i18n, suffix) = libraries()?;
    let resolve = |name: &str| -> *mut c_void {
        // The collator and number functions are i18n's, uloc and uenum uc's.
        [i18n, uc]
            .into_iter()
            .find_map(|library| symbol(library, &format!("{name}{suffix}")))
            .map_or(std::ptr::null_mut(), NonNull::as_ptr)
    };
    Some(Box::new(Icu::load(resolve, user_locale())?))
}

/// The two libraries, kept loaded for the process, and their functions'
/// suffix: the unversioned names a development package installs, else the
/// newest versioned sonames present.
fn libraries() -> Option<(NonNull<c_void>, NonNull<c_void>, String)> {
    if let Some(uc) = library("libicuuc.so") {
        if let Some(suffix) = VERSIONS.rev().find_map(|v| suffix(uc, v))
            && let Some(i18n) = library("libicui18n.so")
        {
            return Some((uc, i18n, suffix));
        }
        close(uc);
    }
    VERSIONS.rev().find_map(|version| {
        let uc = library(&format!("libicuuc.so.{version}"))?;
        let opened = suffix(uc, version)
            .and_then(|suffix| Some((uc, library(&format!("libicui18n.so.{version}"))?, suffix)));
        if opened.is_none() {
            close(uc);
        }
        opened
    })
}

/// The suffix of ICU `version`'s functions in `uc`: `_77`, or none in a
/// build without renaming.
fn suffix(uc: NonNull<c_void>, version: u32) -> Option<String> {
    let versioned = format!("_{version}");
    if symbol(uc, &format!("u_getVersion{versioned}")).is_some() {
        Some(versioned)
    } else {
        symbol(uc, "u_getVersion").map(|_| String::new())
    }
}

fn library(name: &str) -> Option<NonNull<c_void>> {
    let name = CString::new(name).ok()?;
    // SAFETY: a NUL-terminated name, and flags dlopen takes.
    NonNull::new(unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) })
}

fn close(library: NonNull<c_void>) {
    // SAFETY: opened by `library`, and nothing resolved from it is kept.
    unsafe { libc::dlclose(library.as_ptr()) };
}

fn symbol(library: NonNull<c_void>, name: &str) -> Option<NonNull<c_void>> {
    let name = CString::new(name).ok()?;
    // SAFETY: the library is open, and the name NUL-terminated.
    NonNull::new(unsafe { libc::dlsym(library.as_ptr(), name.as_ptr()) })
}

/// The user's locale for messages, by POSIX's precedence (`LC_ALL`, then
/// `LC_MESSAGES`, then `LANG`) without its encoding or modifier:
/// `de_DE.UTF-8@euro` is `de_DE`. `C` and `POSIX` name no language.
pub(crate) fn user_locale() -> Option<String> {
    let value = ["LC_ALL", "LC_MESSAGES", "LANG"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))?;
    locale_of(&value)
}

fn locale_of(value: &str) -> Option<String> {
    let locale = value.split(['.', '@']).next()?;
    (!locale.is_empty() && locale != "C" && locale != "POSIX").then(|| locale.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posix_locales_lose_their_encoding_and_modifier() {
        assert_eq!(locale_of("de_DE.UTF-8@euro").as_deref(), Some("de_DE"));
        assert_eq!(locale_of("sr_RS@latin").as_deref(), Some("sr_RS"));
        assert_eq!(locale_of("ja_JP").as_deref(), Some("ja_JP"));
        for none in ["C", "C.UTF-8", "POSIX", ".UTF-8"] {
            assert_eq!(locale_of(none), None, "{none}");
        }
    }
}

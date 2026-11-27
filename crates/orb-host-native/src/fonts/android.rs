//! Android's fallback fonts: the system's fallback chain for the app's
//! languages (its `fonts.xml`, through minikin), asked of the NDK's
//! `AFontMatcher` (API level 29), the fonts Android's own text falls back
//! to, so ideographs take the forms of the language the user reads them
//! in and every script finds the font the system draws it with.
//!
//! The matcher is resolved at run time, so the library still loads from
//! API level 24, and asked only once [`init`](crate::android::init) gave
//! the app's JVM: minikin's font list is the one the app's `Typeface`
//! registers, and a process no app started (a test pushed with adb) has
//! none, where matching dereferences an empty map. Until then, and where
//! the matcher is missing, the fixed list in `fonts.rs` stands in.

#![allow(unsafe_code, reason = "the NDK's font matcher is the FFI here")]

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use jni::errors::Error as JniError;
use jni::objects::JString;
use jni::{Env, JavaVM, jni_sig, jni_str};
use ndk_sys::{AFont, AFontMatcher};

/// Per generic family, code point, weight and italic, the face found.
type Found = HashMap<(&'static str, char, u16, bool), Option<fontdb::ID>>;

/// The faces Android falls back to for the app's languages, found in the
/// database or loaded into it.
#[derive(Default)]
pub(super) struct Fallbacks {
    /// The matcher, once tried: `Some(None)` where the system lacks it.
    matcher: Option<Option<Matcher>>,
    found: Found,
}

impl Fallbacks {
    /// The face Android draws `c` in for the CSS family list `families`
    /// at CSS weight `weight`, when `draws` says the database's face of it
    /// draws `c`: its generic family's, as the families before it were the
    /// database's to find. `None` before the app gave its JVM, and where
    /// no font of the system's chain has `c` (the matcher then names one
    /// that draws it as tofu).
    pub(super) fn face(
        &mut self,
        db: &mut fontdb::Database,
        families: &str,
        c: char,
        weight: u16,
        italic: bool,
        mut draws: impl FnMut(&mut fontdb::Database, fontdb::ID) -> bool,
    ) -> Option<fontdb::ID> {
        let key = (generic(families), c, weight, italic);
        if let Some(&id) = self.found.get(&key) {
            return id;
        }
        let matcher = self.matcher()?;
        let id = matcher
            .font(key.0, c, weight, italic)
            .and_then(|(path, index)| resolve(db, &path, index))
            .filter(|&id| draws(db, id));
        self.found.insert(key, id);
        id
    }

    /// The matcher, made once the app gave its JVM, for its languages.
    fn matcher(&mut self) -> Option<&mut Matcher> {
        if self.matcher.is_none() {
            let vm = crate::android::java_vm()?;
            self.matcher = Some(Matcher::new(&app_languages(vm)));
        }
        self.matcher.as_mut().and_then(Option::as_mut)
    }
}

/// The family of minikin's that the CSS family list `families` falls back
/// through: its first generic family (each family's fallback chain ends
/// in the same fonts, but serif's takes serif forms of CJK first), else
/// sans-serif. Names minikin lacks would be sans-serif too.
fn generic(families: &str) -> &'static str {
    families
        .split(',')
        .find_map(|name| match name.trim() {
            "serif" => Some("serif"),
            "monospace" => Some("monospace"),
            "cursive" => Some("cursive"),
            "sans-serif" => Some("sans-serif"),
            _ => None,
        })
        .unwrap_or("sans-serif")
}

/// The face of the file at `path` that is its collection's `index`, loaded
/// into the database if its scan missed the file (a font updated outside
/// `/system/fonts`), nameless then so that no family query matches it and
/// a face queried by name doesn't depend on what fell back before. The
/// scan holds a symlinked file under its target's path.
fn resolve(db: &mut fontdb::Database, path: &Path, index: u32) -> Option<fontdb::ID> {
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let find = |db: &fontdb::Database| {
        db.faces()
            .find(|face| {
                face.index == index
                    && matches!(&face.source,
                        fontdb::Source::File(p) | fontdb::Source::SharedFile(p, _)
                            if p == path || *p == target)
            })
            .map(|face| face.id)
    };
    if let Some(id) = find(db) {
        return Some(id);
    }
    for id in db.load_font_source(fontdb::Source::File(target.clone())) {
        let Some(mut info) = db.face(id).cloned() else {
            continue;
        };
        db.remove_face(id);
        info.families.clear();
        db.push_face_info(info);
    }
    find(db)
}

/// The app's languages, most preferred first, as comma-separated BCP 47
/// tags (`LocaleList.getDefault()`); none if Java fails, leaving no
/// exception pending on the attached thread.
fn app_languages(vm: &JavaVM) -> String {
    vm.attach_current_thread(|env| -> Result<String, JniError> {
        let tags = language_tags(env);
        if env.exception_check() {
            env.exception_clear();
        }
        tags
    })
    .unwrap_or_default()
}

/// `LocaleList.getDefault().toLanguageTags()`.
fn language_tags(env: &mut Env<'_>) -> Result<String, JniError> {
    let list = env
        .call_static_method(
            jni_str!("android/os/LocaleList"),
            jni_str!("getDefault"),
            jni_sig!("()Landroid/os/LocaleList;"),
            &[],
        )?
        .l()?;
    let tags = env
        .call_method(
            &list,
            jni_str!("toLanguageTags"),
            jni_sig!("()Ljava/lang/String;"),
            &[],
        )?
        .l()?;
    env.cast_local::<JString>(tags)?.try_to_string(env)
}

/// Declares the NDK's font functions as pointers resolved from
/// `libandroid.so` at run time.
macro_rules! functions {
    ($($name:ident($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[expect(non_snake_case, reason = "the NDK's names")]
        struct Functions {
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Functions {
            /// The functions, or `None` below API level 29.
            fn load() -> Option<Self> {
                // SAFETY: a NUL-terminated name, and flags dlopen takes.
                // The library, which the process has loaded already,
                // stays loaded.
                let library = NonNull::new(unsafe {
                    libc::dlopen(c"libandroid.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL)
                })?;
                Some(Self {
                    $($name: {
                        let name = concat!(stringify!($name), "\0");
                        // SAFETY: the library is open, and the name
                        // NUL-terminated.
                        let symbol = NonNull::new(unsafe {
                            libc::dlsym(library.as_ptr(), name.as_ptr().cast())
                        })?;
                        // SAFETY: the library exports the function under
                        // this name, with the NDK's signature, declared
                        // here.
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
    AFontMatcher_create() -> *mut AFontMatcher;
    AFontMatcher_destroy(*mut AFontMatcher);
    AFontMatcher_setStyle(*mut AFontMatcher, u16, bool);
    AFontMatcher_setLocales(*mut AFontMatcher, *const c_char);
    AFontMatcher_match(*const AFontMatcher, *const c_char, *const u16, u32, *mut u32) -> *mut AFont;
    AFont_getFontFilePath(*const AFont) -> *const c_char;
    AFont_getCollectionIndex(*const AFont) -> usize;
    AFont_close(*mut AFont);
}

/// An `AFontMatcher` for some languages.
struct Matcher {
    api: Functions,
    matcher: NonNull<AFontMatcher>,
}

// SAFETY: the matcher is plain data the NDK reads only in these calls,
// from whichever thread holds the state, behind a mutex.
unsafe impl Send for Matcher {}

impl Matcher {
    /// A matcher for `languages`, comma-separated BCP 47 tags; `None`
    /// below API level 29.
    fn new(languages: &str) -> Option<Self> {
        let api = Functions::load()?;
        // SAFETY: a function of the NDK's.
        let matcher = NonNull::new(unsafe { (api.AFontMatcher_create)() })?;
        if let Ok(languages) = CString::new(languages)
            && !languages.is_empty()
        {
            // SAFETY: a live matcher and a NUL-terminated list.
            unsafe { (api.AFontMatcher_setLocales)(matcher.as_ptr(), languages.as_ptr()) };
        }
        Some(Self { api, matcher })
    }

    /// The file and collection index of the font minikin draws `c` in for
    /// its family `family` at `weight`, italic or not. Where no font has
    /// `c`, one that doesn't.
    fn font(
        &mut self,
        family: &'static str,
        c: char,
        weight: u16,
        italic: bool,
    ) -> Option<(PathBuf, u32)> {
        let family = CString::new(family).ok()?;
        let mut units = [0u16; 2];
        let text = c.encode_utf16(&mut units);
        let api = &self.api;
        // SAFETY: a live matcher, a NUL-terminated family and `text`'s
        // units; the font is closed once its path is copied.
        unsafe {
            (api.AFontMatcher_setStyle)(self.matcher.as_ptr(), weight, italic);
            let font = (api.AFontMatcher_match)(
                self.matcher.as_ptr(),
                family.as_ptr(),
                text.as_ptr(),
                text.len() as u32,
                std::ptr::null_mut(),
            );
            let font = NonNull::new(font)?;
            let path = (api.AFont_getFontFilePath)(font.as_ptr());
            let path = (!path.is_null())
                .then(|| PathBuf::from(CStr::from_ptr(path).to_string_lossy().into_owned()));
            let index = (api.AFont_getCollectionIndex)(font.as_ptr());
            (api.AFont_close)(font.as_ptr());
            Some((path?, u32::try_from(index).ok()?))
        }
    }
}

impl Drop for Matcher {
    fn drop(&mut self) {
        // SAFETY: made by `AFontMatcher_create` and not destroyed before.
        unsafe { (self.api.AFontMatcher_destroy)(self.matcher.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::generic;

    #[test]
    fn falls_back_through_the_lists_generic_family() {
        assert_eq!(
            generic("Open Sans Semibold,Arial Unicode MS Bold,sans-serif"),
            "sans-serif"
        );
        assert_eq!(generic("Noto Serif Regular, serif"), "serif");
        assert_eq!(generic("Fira Mono,monospace,serif"), "monospace");
        assert_eq!(generic("Open Sans"), "sans-serif");
    }
}

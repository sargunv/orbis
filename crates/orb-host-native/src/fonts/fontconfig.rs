//! Linux's fallback fonts: the first font of fontconfig's order for a
//! code point in the CSS family list's generic family and the user's
//! language that has it, as Chrome on Linux falls back, so ideographs take the forms of the language the
//! user reads (a Chinese user gets Noto Sans CJK SC where the fixed list
//! gave everyone JP), a serif stack falls back to the serif fonts the
//! configuration names, and every script finds the font the desktop draws
//! it with rather than the first installed face that maps it.
//!
//! `libfontconfig` is loaded at run time, so the binary links nothing new
//! and still runs where it isn't installed; there the fixed list in
//! `fonts.rs` stands in.

#![allow(unsafe_code, reason = "fontconfig is the FFI here")]

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

/// Per generic family, code point, weight and italic, the face found.
type Found = HashMap<(&'static str, char, u16, bool), Option<fontdb::ID>>;

/// The faces fontconfig falls back to for the user's language, found in
/// the database or loaded into it.
#[derive(Default)]
pub(super) struct Fallbacks {
    /// Fontconfig, once tried: `Some(None)` where the system lacks it.
    fontconfig: Option<Option<Fontconfig>>,
    /// The language a test sets, else the user's.
    language: Option<String>,
    /// Per file and collection index, its face in the database.
    resolved: HashMap<(PathBuf, u32), Option<fontdb::ID>>,
    found: Found,
}

impl Fallbacks {
    /// For `language`, a BCP 47 or POSIX tag, not the user's.
    #[cfg(test)]
    pub(super) fn with_language(language: &str) -> Self {
        Self {
            language: Some(language.to_string()),
            ..Self::default()
        }
    }

    /// Whether the system has a fontconfig this can use.
    #[cfg(test)]
    pub(super) fn available() -> bool {
        Functions::load().is_some()
    }

    /// The face a browser on Linux draws `c` in for the CSS family list
    /// `families` at CSS weight `weight`: the first font of fontconfig's
    /// order for it in the list's generic family that has it and whose
    /// face in the database `draws` says draws it, as the member of its
    /// family CSS font matching picks for the weight. The names before the
    /// generic family were the database's to find, and the system falls
    /// back per generic family (Chrome asks fontconfig with none, so
    /// sans-serif). `None` where fontconfig is missing or no font has `c`.
    pub(super) fn face(
        &mut self,
        db: &mut fontdb::Database,
        families: &str,
        c: char,
        weight: u16,
        italic: bool,
        mut draws: impl FnMut(&mut fontdb::Database, fontdb::ID) -> bool,
    ) -> Option<fontdb::ID> {
        let key = (super::generic(families), c, weight, italic);
        if let Some(&id) = self.found.get(&key) {
            return id;
        }
        let Self {
            fontconfig,
            language,
            resolved,
            found,
        } = self;
        let fontconfig = fontconfig
            .get_or_insert_with(|| {
                let language = language.take().or_else(crate::locale::user_locale);
                Fontconfig::new(language.as_deref())
            })
            .as_mut()?;
        let id = fontconfig.fonts(key.0, c).iter().find_map(|(path, index)| {
            let id = (*resolved.entry((path.clone(), *index)).or_insert_with(|| {
                // The scan holds a symlinked file under its target's path.
                let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
                super::resolve(db, &target, *index, |p| p == path || p == target)
            }))?;
            member(db, id, weight, italic)
                .filter(|&m| m != id && draws(db, m))
                .or_else(|| draws(db, id).then_some(id))
        });
        found.insert(key, id);
        id
    }
}

/// The member of `id`'s family that CSS font matching picks at `weight`,
/// italic or not, as the family list's own faces are picked. Fontconfig
/// is asked at the normal weight, as Chrome asks it: given the weight, it
/// would trade the family for it, falling back from a sans-serif stack at
/// 600 to a serif font's semibold. A variable face is its own family's only member,
/// drawn at the weight later.
fn member(db: &fontdb::Database, id: fontdb::ID, weight: u16, italic: bool) -> Option<fontdb::ID> {
    let face = db.face(id)?;
    let (family, _) = face.families.first()?;
    db.query(&fontdb::Query {
        families: &[fontdb::Family::Name(family)],
        weight: fontdb::Weight(weight),
        stretch: face.stretch,
        style: if italic {
            fontdb::Style::Italic
        } else {
            fontdb::Style::Normal
        },
    })
}

/// A language tag as fontconfig takes one: its language and region,
/// without a script or anything after (`zh-Hant-TW` is `zh-TW`), which
/// fontconfig then normalizes to one of its languages.
fn shape(tag: &str) -> Option<String> {
    let mut subtags = tag.split(['-', '_']);
    let language = subtags
        .next()
        .filter(|l| (2..=3).contains(&l.len()) && l.bytes().all(|b| b.is_ascii_alphabetic()))?;
    let region = subtags.find(|r| r.len() == 2 && r.bytes().all(|b| b.is_ascii_alphabetic()));
    Some(match region {
        Some(region) => format!("{language}-{region}"),
        None => language.to_string(),
    })
}

/// Fontconfig's types, which this holds only by pointer.
#[repr(C)]
struct FcConfig {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FcPattern {
    _opaque: [u8; 0],
}
#[repr(C)]
struct FcCharSet {
    _opaque: [u8; 0],
}
/// A list of fonts, which fontconfig's header declares.
#[repr(C)]
struct FcFontSet {
    nfont: c_int,
    sfont: c_int,
    fonts: *mut *mut FcPattern,
}

/// `FcResultMatch`, `FcMatchPattern` and `FcSetSystem`, from the header.
const RESULT_MATCH: c_int = 0;
const MATCH_PATTERN: c_int = 0;
const SET_SYSTEM: c_int = 0;

// Fontconfig's functions, all in its library since 2.11.
functions! {
    [c"libfontconfig.so.1", c"libfontconfig.so"]
    FcInitLoadConfigAndFonts() -> *mut FcConfig;
    FcConfigDestroy(*mut FcConfig);
    FcConfigGetFonts(*mut FcConfig, c_int) -> *mut FcFontSet;
    FcConfigSubstitute(*mut FcConfig, *mut FcPattern, c_int) -> c_int;
    FcDefaultSubstitute(*mut FcPattern);
    FcFontSort(*mut FcConfig, *mut FcPattern, c_int, *mut *mut FcCharSet, *mut c_int) -> *mut FcFontSet;
    FcFontSetDestroy(*mut FcFontSet);
    FcPatternCreate() -> *mut FcPattern;
    FcPatternDuplicate(*const FcPattern) -> *mut FcPattern;
    FcPatternDestroy(*mut FcPattern);
    FcPatternAddString(*mut FcPattern, *const c_char, *const u8) -> c_int;
    FcPatternAddCharSet(*mut FcPattern, *const c_char, *const FcCharSet) -> c_int;
    FcPatternGetString(*const FcPattern, *const c_char, c_int, *mut *mut u8) -> c_int;
    FcPatternGetInteger(*const FcPattern, *const c_char, c_int, *mut c_int) -> c_int;
    FcPatternGetCharSet(*const FcPattern, *const c_char, c_int, *mut *mut FcCharSet) -> c_int;
    FcCharSetCreate() -> *mut FcCharSet;
    FcCharSetDestroy(*mut FcCharSet);
    FcCharSetAddChar(*mut FcCharSet, u32) -> c_int;
    FcCharSetHasChar(*const FcCharSet, u32) -> c_int;
    FcLangNormalize(*const u8) -> *mut u8;
    FcStrFree(*mut u8);
}

/// Which of the installed fonts have a code point, a bit each.
type Coverage = Vec<u64>;

/// A generic family's pattern, substituted, and for each coverage asked
/// the files and collection indices of the fonts that have its code
/// points, in fontconfig's order.
struct List {
    pattern: Option<NonNull<FcPattern>>,
    sorted: HashMap<Coverage, Vec<(PathBuf, u32)>>,
}

/// Fontconfig's configuration and fonts, for a language.
///
/// The fonts are sorted per code point, as Chrome and Skia sort them. A
/// sort keeps the language's preference only for the first font in its
/// order that supports the language, so one sort for a family list would
/// rank every script by a Latin font, and a match (`FcFontMatch`), which
/// has no such step, prefers any font that has the code point and
/// supports the user's language, such as a coding font's Georgian over a
/// Georgian font's. A sort depends on the code point only through which
/// fonts have it, so each coverage is sorted once: most ideographs share
/// one, and a sort takes half a millisecond.
struct Fontconfig {
    api: Functions,
    config: NonNull<FcConfig>,
    /// The language, as fontconfig names it, or none for its default.
    language: Option<CString>,
    /// The installed fonts' code points, which the configuration holds.
    charsets: Vec<NonNull<FcCharSet>>,
    /// Per generic family, its pattern and sorts.
    lists: HashMap<&'static str, List>,
}

// SAFETY: the configuration and patterns are fontconfig's data, read only
// in these calls, from whichever thread holds the state, behind a mutex.
unsafe impl Send for Fontconfig {}

impl Fontconfig {
    /// The system's configuration, for `language` (fontconfig's default
    /// where none); `None` where fontconfig is missing.
    fn new(language: Option<&str>) -> Option<Self> {
        let api = Functions::load()?;
        // SAFETY: a function of fontconfig's.
        let config = NonNull::new(unsafe { (api.FcInitLoadConfigAndFonts)() })?;
        let language = language
            .and_then(shape)
            .and_then(|tag| CString::new(tag).ok())
            .and_then(|tag| {
                // SAFETY: a NUL-terminated tag; the normalized copy is
                // fontconfig's to free, once copied.
                unsafe {
                    let normal = NonNull::new((api.FcLangNormalize)(tag.as_ptr().cast()))?;
                    let owned = CStr::from_ptr(normal.as_ptr().cast()).to_owned();
                    (api.FcStrFree)(normal.as_ptr());
                    Some(owned)
                }
            });
        // SAFETY: a live configuration, whose fonts live as long as it.
        let charsets = unsafe {
            fonts((api.FcConfigGetFonts)(config.as_ptr(), SET_SYSTEM))
                .iter()
                .filter_map(|&font| {
                    let mut charset = std::ptr::null_mut();
                    let got = (api.FcPatternGetCharSet)(font, c"charset".as_ptr(), 0, &mut charset);
                    NonNull::new(charset).filter(|_| got == RESULT_MATCH)
                })
                .collect()
        };
        Some(Self {
            api,
            config,
            language,
            charsets,
            lists: HashMap::new(),
        })
    }

    /// The files and collection indices of the fonts that have `c`, in
    /// the order fontconfig falls back through them for the generic
    /// family `family`.
    fn fonts(&mut self, family: &'static str, c: char) -> &[(PathBuf, u32)] {
        let api = &self.api;
        let coverage: Coverage = self
            .charsets
            .chunks(64)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0, |bits, (i, charset)| {
                    // SAFETY: the configuration holds the charset.
                    let has = unsafe { (api.FcCharSetHasChar)(charset.as_ptr(), u32::from(c)) };
                    bits | (u64::from(has != 0) << i)
                })
            })
            .collect();
        if coverage.iter().all(|&bits| bits == 0) {
            return &[];
        }
        let (config, language) = (self.config, self.language.as_deref());
        let list = self.lists.entry(family).or_insert_with(|| List {
            // SAFETY: a live configuration.
            pattern: unsafe { pattern(api, config, family, language) },
            sorted: HashMap::new(),
        });
        let Some(pattern) = list.pattern else {
            return &[];
        };
        list.sorted
            .entry(coverage)
            // SAFETY: a live configuration and substituted pattern.
            .or_insert_with(|| unsafe { sorted(api, config, pattern, c) })
    }
}

/// The pattern a browser asks fontconfig for `family` in `language`,
/// substituted by the configuration (its aliases, the generic families
/// and the languages' preferred fonts) and its defaults.
///
/// # Safety
///
/// `config` is live.
unsafe fn pattern(
    api: &Functions,
    config: NonNull<FcConfig>,
    family: &str,
    language: Option<&CStr>,
) -> Option<NonNull<FcPattern>> {
    // SAFETY: a live configuration and pattern, and NUL-terminated
    // strings, which the pattern copies.
    unsafe {
        let pattern = NonNull::new((api.FcPatternCreate)())?;
        if let Ok(family) = CString::new(family) {
            (api.FcPatternAddString)(pattern.as_ptr(), c"family".as_ptr(), family.as_ptr().cast());
        }
        if let Some(language) = language {
            (api.FcPatternAddString)(pattern.as_ptr(), c"lang".as_ptr(), language.as_ptr().cast());
        }
        (api.FcConfigSubstitute)(config.as_ptr(), pattern.as_ptr(), MATCH_PATTERN);
        (api.FcDefaultSubstitute)(pattern.as_ptr());
        Some(pattern)
    }
}

/// The files and collection indices of the fonts that have `c`, in the
/// order fontconfig sorts them for it with `pattern`.
///
/// # Safety
///
/// `config` and `pattern` are live.
unsafe fn sorted(
    api: &Functions,
    config: NonNull<FcConfig>,
    pattern: NonNull<FcPattern>,
    c: char,
) -> Vec<(PathBuf, u32)> {
    // SAFETY: live objects, each destroyed once, and the sorted fonts'
    // strings read while their list lives.
    unsafe {
        let Some(asked) = NonNull::new((api.FcPatternDuplicate)(pattern.as_ptr())) else {
            return Vec::new();
        };
        let charset = (api.FcCharSetCreate)();
        if !charset.is_null() {
            (api.FcCharSetAddChar)(charset, u32::from(c));
            (api.FcPatternAddCharSet)(asked.as_ptr(), c"charset".as_ptr(), charset);
            (api.FcCharSetDestroy)(charset);
        }
        let mut result = 0;
        let set = (api.FcFontSort)(
            config.as_ptr(),
            asked.as_ptr(),
            0,
            std::ptr::null_mut(),
            &mut result,
        );
        (api.FcPatternDestroy)(asked.as_ptr());
        let found = fonts(set)
            .iter()
            .filter_map(|&font| read(api, font, c))
            .collect();
        if !set.is_null() {
            (api.FcFontSetDestroy)(set);
        }
        found
    }
}

/// The patterns of a font list, none for a null one.
///
/// # Safety
///
/// `set` is null or a live list, which outlives the slice.
unsafe fn fonts<'a>(set: *const FcFontSet) -> &'a [*mut FcPattern] {
    // SAFETY: a live list's `nfont` patterns.
    unsafe {
        match set.as_ref() {
            Some(set) if set.nfont > 0 && !set.fonts.is_null() => {
                std::slice::from_raw_parts(set.fonts, set.nfont as usize)
            }
            _ => &[],
        }
    }
}

/// The file and collection index of the font `font`, if it has `c`.
///
/// # Safety
///
/// `font` is a live pattern.
unsafe fn read(api: &Functions, font: *const FcPattern, c: char) -> Option<(PathBuf, u32)> {
    let mut file = std::ptr::null_mut();
    let mut index = 0;
    let mut charset = std::ptr::null_mut();
    // SAFETY: a live pattern and the objects' names; the strings are the
    // pattern's, copied.
    unsafe {
        let found = (api.FcPatternGetString)(font, c"file".as_ptr(), 0, &mut file) == RESULT_MATCH
            && (api.FcPatternGetInteger)(font, c"index".as_ptr(), 0, &mut index) == RESULT_MATCH
            && (api.FcPatternGetCharSet)(font, c"charset".as_ptr(), 0, &mut charset)
                == RESULT_MATCH;
        if !found || file.is_null() || charset.is_null() {
            return None;
        }
        if (api.FcCharSetHasChar)(charset, u32::from(c)) == 0 {
            return None;
        }
        let path = Path::new(std::ffi::OsStr::from_bytes(
            CStr::from_ptr(file.cast()).to_bytes(),
        ));
        // A variable font's named instance is its face's index with the
        // instance above 16 bits; the database holds the face.
        Some((path.to_path_buf(), u32::try_from(index).ok()? & 0xFFFF))
    }
}

impl Drop for Fontconfig {
    fn drop(&mut self) {
        // SAFETY: patterns made by `pattern` and a configuration made by
        // `FcInitLoadConfigAndFonts`, none destroyed before.
        unsafe {
            for pattern in self.lists.drain().filter_map(|(_, list)| list.pattern) {
                (self.api.FcPatternDestroy)(pattern.as_ptr());
            }
            (self.api.FcConfigDestroy)(self.config.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::shape;

    #[test]
    fn tags_keep_their_language_and_region() {
        assert_eq!(shape("zh-CN").as_deref(), Some("zh-CN"));
        assert_eq!(shape("ja_JP").as_deref(), Some("ja-JP"));
        assert_eq!(shape("zh-Hant-TW").as_deref(), Some("zh-TW"));
        assert_eq!(shape("sr-Latn").as_deref(), Some("sr"));
        assert_eq!(shape("es-419").as_deref(), Some("es"));
        assert_eq!(shape("ko").as_deref(), Some("ko"));
        assert_eq!(shape(""), None);
    }
}

//! Windows' fallback fonts: DirectWrite's system font fallback
//! (`IDWriteFactory2::GetSystemFontFallback`, Windows 8.1) for the user's
//! locale, the fonts Windows' own text falls back to, so ideographs take
//! the forms of the language the user reads (Microsoft YaHei for Chinese,
//! Yu Gothic for Japanese, Malgun Gothic for Korean) and every script
//! finds the font the system draws it with. Where any call fails, the
//! fixed list in `fonts.rs` stands in.

#![allow(unsafe_code, reason = "DirectWrite is the FFI here")]

use std::collections::HashMap;
use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt;
use std::path::PathBuf;

use ::windows::Win32::Graphics::DirectWrite::{
    DWRITE_FACTORY_TYPE_SHARED, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_ITALIC,
    DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_WEIGHT, DWRITE_READING_DIRECTION,
    DWRITE_READING_DIRECTION_LEFT_TO_RIGHT, DWriteCreateFactory, IDWriteFactory2,
    IDWriteFontCollection, IDWriteFontFallback, IDWriteLocalFontFileLoader,
    IDWriteNumberSubstitution, IDWriteTextAnalysisSource, IDWriteTextAnalysisSource_Impl,
};
use ::windows::core::{Interface, OutRef, PCWSTR, Result, implement};

/// Per base family, code point, weight and italic, the face found.
type Found = HashMap<(Option<String>, char, u16, bool), Option<fontdb::ID>>;

/// The faces Windows falls back to for the user's locale, found in the
/// database or loaded into it.
#[derive(Default)]
pub(super) struct Fallbacks {
    /// The system's fallback, once tried: `Some(None)` where DirectWrite
    /// lacks it.
    system: Option<Option<System>>,
    found: Found,
}

impl Fallbacks {
    /// For `locale`, not the user's.
    #[cfg(test)]
    pub(super) fn with_locale(locale: &str) -> Self {
        Self {
            system: Some(System::new(locale)),
            ..Self::default()
        }
    }

    /// The face Windows draws `c` in for the CSS family list `families`
    /// at CSS weight `weight`, when `draws` says the database's face of it
    /// draws `c`. `None` where no font of the system's fallback has `c`,
    /// or DirectWrite fails.
    pub(super) fn face(
        &mut self,
        db: &mut fontdb::Database,
        families: &str,
        c: char,
        weight: u16,
        italic: bool,
        mut draws: impl FnMut(&mut fontdb::Database, fontdb::ID) -> bool,
    ) -> Option<fontdb::ID> {
        let family = base_family(families);
        let key = (family.map(str::to_string), c, weight, italic);
        if let Some(&id) = self.found.get(&key) {
            return id;
        }
        let system = self
            .system
            .get_or_insert_with(|| {
                let locale = crate::locale::user_locale().unwrap_or_default();
                // A sort order after an underscore isn't a language's.
                System::new(locale.split('_').next().unwrap_or_default())
            })
            .as_ref()?;
        let id = system
            .font(family, c, weight, italic)
            .and_then(|(path, index)| {
                super::resolve(db, &path, index, |p| {
                    p.as_os_str().eq_ignore_ascii_case(path.as_os_str())
                })
            })
            .filter(|&id| draws(db, id));
        self.found.insert(key, id);
        id
    }
}

/// The family DirectWrite maps from: the list's first that isn't generic,
/// as DirectWrite knows no generic families.
fn base_family(families: &str) -> Option<&str> {
    families.split(',').map(str::trim).find(|name| {
        !matches!(
            *name,
            "sans-serif" | "serif" | "monospace" | "cursive" | "fantasy" | ""
        )
    })
}

/// DirectWrite's system font fallback for one locale.
struct System {
    fallback: IDWriteFontFallback,
    collection: IDWriteFontCollection,
    /// The locale's name, NUL-terminated.
    locale: Vec<u16>,
}

// SAFETY: objects of DirectWrite's shared factory are free-threaded, and
// these are called only from whichever thread holds the state, behind a
// mutex.
unsafe impl Send for System {}

impl System {
    /// The fallback for `locale`, a BCP 47 tag; `None` before Windows 8.1.
    fn new(locale: &str) -> Option<Self> {
        // SAFETY: plain calls of DirectWrite's, their out-parameter live.
        unsafe {
            let factory: IDWriteFactory2 = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).ok()?;
            let fallback = factory.GetSystemFontFallback().ok()?;
            let mut collection = None;
            factory
                .GetSystemFontCollection(&mut collection, false)
                .ok()?;
            Some(Self {
                fallback,
                collection: collection?,
                locale: locale.encode_utf16().chain([0]).collect(),
            })
        }
    }

    /// The file and collection index of the font DirectWrite draws `c` in
    /// for the family `family` at `weight`, italic or not.
    fn font(
        &self,
        family: Option<&str>,
        c: char,
        weight: u16,
        italic: bool,
    ) -> Option<(PathBuf, u32)> {
        let mut units = [0u16; 2];
        let text = c.encode_utf16(&mut units).to_vec();
        let length = text.len() as u32;
        let source: IDWriteTextAnalysisSource = Source {
            text,
            locale: self.locale.clone(),
        }
        .into();
        let family: Option<Vec<u16>> = family.map(|f| f.encode_utf16().chain([0]).collect());
        let family = family
            .as_ref()
            .map_or(PCWSTR::null(), |f| PCWSTR(f.as_ptr()));
        let style = if italic {
            DWRITE_FONT_STYLE_ITALIC
        } else {
            DWRITE_FONT_STYLE_NORMAL
        };
        let (mut mapped, mut font, mut scale) = (0, None, 1.0);
        // SAFETY: a live source over `length` units, a NUL-terminated
        // family or none, and live out-parameters; the reference key is
        // read while its file lives, into a buffer of the length asked.
        unsafe {
            self.fallback
                .MapCharacters(
                    &source,
                    0,
                    length,
                    &self.collection,
                    family,
                    DWRITE_FONT_WEIGHT(i32::from(weight)),
                    style,
                    DWRITE_FONT_STRETCH_NORMAL,
                    &mut mapped,
                    &mut font,
                    &mut scale,
                )
                .ok()?;
            let face = font?.CreateFontFace().ok()?;
            let mut count = 1;
            let mut files = [None];
            face.GetFiles(&mut count, Some(files.as_mut_ptr())).ok()?;
            let [Some(file)] = files else { return None };
            let loader: IDWriteLocalFontFileLoader = file.GetLoader().ok()?.cast().ok()?;
            let (mut key, mut size) = (std::ptr::null_mut(), 0);
            file.GetReferenceKey(&mut key, &mut size).ok()?;
            let length = loader.GetFilePathLengthFromKey(key, size).ok()?;
            let mut path = vec![0u16; length as usize + 1];
            loader.GetFilePathFromKey(key, size, &mut path).ok()?;
            path.truncate(length as usize);
            Some((PathBuf::from(OsString::from_wide(&path)), face.GetIndex()))
        }
    }
}

/// The text DirectWrite maps: a code point's units, in one locale, left
/// to right.
#[implement(IDWriteTextAnalysisSource)]
struct Source {
    text: Vec<u16>,
    /// NUL-terminated.
    locale: Vec<u16>,
}

impl Source {
    /// The units from `position` on, and how many.
    fn rest(&self, position: u32) -> &[u16] {
        self.text.get(position as usize..).unwrap_or_default()
    }
}

impl IDWriteTextAnalysisSource_Impl for Source_Impl {
    fn GetTextAtPosition(
        &self,
        position: u32,
        text: *mut *mut u16,
        length: *mut u32,
    ) -> Result<()> {
        let rest = self.rest(position);
        // SAFETY: DirectWrite's out-parameters; the units live as long as
        // the source.
        unsafe {
            *text = if rest.is_empty() {
                std::ptr::null_mut()
            } else {
                rest.as_ptr().cast_mut()
            };
            *length = rest.len() as u32;
        }
        Ok(())
    }

    fn GetTextBeforePosition(
        &self,
        position: u32,
        text: *mut *mut u16,
        length: *mut u32,
    ) -> Result<()> {
        let before = self.text.get(..position as usize).unwrap_or_default();
        // SAFETY: as in `GetTextAtPosition`.
        unsafe {
            *text = if before.is_empty() {
                std::ptr::null_mut()
            } else {
                before.as_ptr().cast_mut()
            };
            *length = before.len() as u32;
        }
        Ok(())
    }

    fn GetParagraphReadingDirection(&self) -> DWRITE_READING_DIRECTION {
        DWRITE_READING_DIRECTION_LEFT_TO_RIGHT
    }

    fn GetLocaleName(&self, position: u32, length: *mut u32, locale: *mut *mut u16) -> Result<()> {
        // SAFETY: as in `GetTextAtPosition`.
        unsafe {
            *length = self.rest(position).len() as u32;
            *locale = self.locale.as_ptr().cast_mut();
        }
        Ok(())
    }

    fn GetNumberSubstitution(
        &self,
        position: u32,
        length: *mut u32,
        substitution: OutRef<IDWriteNumberSubstitution>,
    ) -> Result<()> {
        // SAFETY: as in `GetTextAtPosition`.
        unsafe { *length = self.rest(position).len() as u32 };
        substitution.write(None)
    }
}

#[cfg(test)]
mod tests {
    use super::base_family;

    #[test]
    fn maps_from_the_lists_first_named_family() {
        assert_eq!(
            base_family("Open Sans Semibold,Arial Unicode MS Bold,sans-serif"),
            Some("Open Sans Semibold")
        );
        assert_eq!(base_family("serif, Noto Serif"), Some("Noto Serif"));
        assert_eq!(base_family("sans-serif"), None);
    }
}

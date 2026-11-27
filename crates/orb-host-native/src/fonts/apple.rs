//! Apple platforms' fallback fonts: a CSS family list's first font that
//! CoreText has, then its default cascade list for the user's languages
//! (`CTFontCopyDefaultCascadeListForLanguages`), the fonts Safari and
//! Apple's own text fall back to, so ideographs take the forms of the
//! language the user reads them in. An iOS app can read none of the
//! system's font files, and some fonts have no file a process can name,
//! so a font without one is rebuilt from its tables, which CoreText hands
//! out without copying.

#![allow(unsafe_code, reason = "CoreText is the FFI here")]

use std::collections::HashMap;
use std::path::PathBuf;

use objc2_core_foundation::{CFArray, CFData, CFDictionary, CFLocale, CFRetained, CFString, CFURL};
use objc2_core_text::{
    CTFont, CTFontDescriptor, CTFontSymbolicTraits, CTFontTableOptions, kCTFontFamilyNameAttribute,
    kCTFontURLAttribute,
};

/// A CoreText font.
#[derive(Clone)]
struct Font {
    font: CFRetained<CTFont>,
    /// Its PostScript name.
    name: String,
}

// SAFETY: CoreText's fonts are immutable and may be used from any thread
// ("Core Text Programming Guide", thread safety); the state that holds
// them is behind a mutex.
unsafe impl Send for Font {}

impl Font {
    fn new(font: CFRetained<CTFont>) -> Self {
        // SAFETY: a valid font.
        let name = unsafe { font.post_script_name() }.to_string();
        Self { font, name }
    }

    /// Whether it maps `c` to a glyph. Its character set would say too,
    /// but CoreText takes most of a second to make a CJK font's.
    fn covers(&self, c: char) -> bool {
        // objc2-core-text binds it only with CoreGraphics' bindings, for
        // the glyph type.
        unsafe extern "C" {
            fn CTFontGetGlyphsForCharacters(
                font: &CTFont,
                characters: *const u16,
                glyphs: *mut u16,
                count: isize,
            ) -> bool;
        }
        let mut units = [0u16; 2];
        let len = c.encode_utf16(&mut units).len();
        let mut glyphs = [0u16; 2];
        // SAFETY: a valid font, and both buffers hold `len` code units.
        unsafe {
            CTFontGetGlyphsForCharacters(
                &self.font,
                units.as_ptr(),
                glyphs.as_mut_ptr(),
                len as isize,
            )
        }
    }

    /// Its family's name.
    fn family(&self) -> String {
        // SAFETY: a valid font.
        unsafe { self.font.family_name() }.to_string()
    }

    /// Its weight class (`OS/2`'s), 400 without one.
    fn weight(&self) -> u16 {
        self.table(*b"OS/2")
            // SAFETY: the table lives through this call, and nothing
            // mutates it.
            .and_then(|os2| {
                unsafe { os2.as_bytes_unchecked() }
                    .get(4..6)
                    .map(|w| [w[0], w[1]])
            })
            .map_or(400, u16::from_be_bytes)
    }

    /// Whether it is italic.
    fn italic(&self) -> bool {
        // SAFETY: a valid font.
        let traits = unsafe { self.font.symbolic_traits() };
        traits.contains(CTFontSymbolicTraits::TraitItalic)
    }

    /// The file it is in, where the process may name one.
    fn path(&self) -> Option<PathBuf> {
        // SAFETY: a valid font and attribute name.
        let url = unsafe { self.font.attribute(kCTFontURLAttribute) }?;
        url.downcast_ref::<CFURL>()?.to_file_path()
    }

    /// The tags of its tables.
    fn tags(&self) -> Vec<[u8; 4]> {
        // SAFETY: a valid font.
        let Some(tags) = (unsafe { self.font.available_tables(CTFontTableOptions::NoOptions) })
        else {
            return Vec::new();
        };
        // SAFETY: the array holds `CTFontTableTag`s as its values, not
        // objects, and each index is within it.
        (0..tags.len())
            .map(|i| (unsafe { tags.value_at_index(i as isize) } as usize as u32).to_be_bytes())
            .collect()
    }

    /// Whether it has outlines `orb-text` draws: `glyf`, `CFF ` or `CFF2`.
    /// Some system fonts have only Apple's `hvgl` outlines (recent
    /// macOS's Chinese fonts), whose glyphs would come out blank.
    fn draws(&self) -> bool {
        self.tags()
            .iter()
            .any(|tag| matches!(tag, b"glyf" | b"CFF " | b"CFF2"))
    }

    /// Its table `tag`, mapped, not copied.
    fn table(&self, tag: [u8; 4]) -> Option<CFRetained<CFData>> {
        // SAFETY: a valid font and tag.
        unsafe {
            self.font
                .table(u32::from_be_bytes(tag), CTFontTableOptions::NoOptions)
        }
    }

    /// A font file of all its tables: the one copy made of it.
    fn file(&self) -> Vec<u8> {
        let tables = self
            .tags()
            .into_iter()
            .filter_map(|tag| Some((tag, self.table(tag)?)))
            .collect::<Vec<_>>();
        sfnt(
            tables
                .iter()
                .map(|(tag, data)| {
                    // SAFETY: the data lives while `tables` does, and
                    // nothing mutates it.
                    (*tag, unsafe { data.as_bytes_unchecked() })
                })
                .collect(),
        )
    }
}

/// Per code point, weight and italic, the face found.
type Found = HashMap<(char, u16, bool), Option<fontdb::ID>>;

/// The faces CoreText falls back to for the user's languages, found in
/// the database or rebuilt into it.
#[derive(Default)]
pub(super) struct Fallbacks {
    cascade: Cascade,
    /// Per CSS family list, the faces found.
    found: HashMap<String, Found>,
    /// Per `head` table, the face of a font rebuilt from its tables, so
    /// that the weights of a variable font share one copy.
    rebuilt: HashMap<Vec<u8>, Option<fontdb::ID>>,
}

impl Fallbacks {
    /// For `languages`, not the user's.
    #[cfg(test)]
    pub(super) fn with_languages(languages: Vec<String>) -> Self {
        Self {
            cascade: Cascade {
                languages: Some(languages),
                ..Cascade::default()
            },
            ..Self::default()
        }
    }

    /// The face Safari draws `c` in for the CSS family list `families`
    /// at CSS weight `weight`: the first font of the family list's
    /// cascade that has it and that `opens` opens.
    pub(super) fn face(
        &mut self,
        db: &mut fontdb::Database,
        families: &str,
        c: char,
        weight: u16,
        italic: bool,
        mut opens: impl FnMut(&mut fontdb::Database, fontdb::ID) -> bool,
    ) -> Option<fontdb::ID> {
        let key = (c, weight, italic);
        if let Some(&id) = self.found.get(families).and_then(|found| found.get(&key)) {
            return id;
        }
        let id = self
            .cascade
            .fonts(families, c, weight, italic)
            .iter()
            .find_map(|font| self.resolve(db, font).filter(|&id| opens(db, id)));
        let found = self.found.entry(families.to_string()).or_default();
        found.insert(key, id);
        id
    }

    /// `font`'s face in the database: the installed file's, mapped, where
    /// the process can name it, else its tables rebuilt into a font.
    fn resolve(&mut self, db: &mut fontdb::Database, font: &Font) -> Option<fontdb::ID> {
        if let Some(path) = font.path() {
            let in_file = |face: &&fontdb::FaceInfo| match &face.source {
                fontdb::Source::File(p) | fontdb::Source::SharedFile(p, _) => *p == path,
                fontdb::Source::Binary(_) => false,
            };
            let faces: Vec<_> = db.faces().filter(in_file).collect();
            // A variable font's PostScript name changes with the weight
            // CoreText instanced it at; its file's only face is it.
            let face = match faces[..] {
                [face] => Some(face),
                _ => faces.into_iter().find(|f| f.post_script_name == font.name),
            };
            if let Some(face) = face {
                return Some(face.id);
            }
        }
        let head = font.table(*b"head")?;
        // SAFETY: the table lives through this call, and nothing mutates
        // it.
        let head = unsafe { head.as_bytes_unchecked() }.to_vec();
        *self.rebuilt.entry(head).or_insert_with(|| {
            let file = std::sync::Arc::new(font.file());
            let id = *db.load_font_source(fontdb::Source::Binary(file)).first()?;
            // Nameless, so that no family query matches it: reached only
            // from here, a face queried by name doesn't depend on which
            // fallbacks were rebuilt before.
            let mut info = db.face(id)?.clone();
            db.remove_face(id);
            info.families.clear();
            Some(db.push_face_info(info))
        })
    }
}

/// The user's preferred languages, most preferred first, as BCP 47 tags.
fn preferred_languages() -> Vec<String> {
    let Some(languages) = CFLocale::preferred_languages() else {
        return Vec::new();
    };
    // SAFETY: the array holds strings.
    let languages: &CFArray<CFString> = unsafe { languages.cast_unchecked() };
    languages.iter().map(|tag| tag.to_string()).collect()
}

/// CSS family lists' fonts, then their cascade lists' fonts that
/// `orb-text` can draw, for some languages, each with the other members
/// of its family.
#[derive(Default)]
struct Cascade {
    /// The languages, the user's unless a test sets them, read once a
    /// text first falls back.
    languages: Option<Vec<String>>,
    /// Per CSS family list, its fonts: a cascade list takes milliseconds
    /// to make.
    lists: HashMap<String, Vec<Font>>,
    /// Per family name, its members that `orb-text` can draw.
    families: HashMap<String, Vec<Member>>,
}

impl Cascade {
    /// The fonts for the CSS family list `families` that have `c`, in the
    /// order Safari tries them, each the member of its family it draws at
    /// CSS weight `weight` and in italic or not.
    fn fonts(&mut self, families: &str, c: char, weight: u16, italic: bool) -> Vec<Font> {
        let languages = self.languages.get_or_insert_with(preferred_languages);
        if !self.lists.contains_key(families) {
            let fonts = list(primary(families), languages);
            self.lists.insert(families.to_string(), fonts);
        }
        self.lists[families]
            .iter()
            .filter(|font| font.covers(c))
            .map(|font| {
                let members = self
                    .families
                    .entry(font.family())
                    .or_insert_with_key(|family| members(family));
                members
                    .iter()
                    .filter(|m| m.font.covers(c))
                    .min_by_key(|m| (m.italic != italic, css_weight_rank(weight, m.weight)))
                    .map_or(font, |m| &m.font)
                    .clone()
            })
            .collect()
    }
}

/// A member of a family, with what choosing among them reads, read once.
struct Member {
    font: Font,
    weight: u16,
    italic: bool,
}

/// How well a face of weight `actual` answers CSS weight `desired`, best
/// lowest: CSS Fonts' font matching, which a canvas follows. Weights
/// from the desired one to 500 come first for 400 to 500, then lighter,
/// then heavier; lighter first below 400, heavier first above 500.
fn css_weight_rank(desired: u16, actual: u16) -> (u8, u16) {
    let tier = match desired {
        ..400 if actual <= desired => 0,
        ..400 => 2,
        400..=500 if (desired..=500).contains(&actual) => 0,
        400..=500 if actual < desired => 1,
        400..=500 => 2,
        _ if actual >= desired => 0,
        _ => 1,
    };
    (tier, desired.abs_diff(actual))
}

/// The first font of the CSS family list `families` that CoreText has,
/// with the generic families as Safari has them; Helvetica if none.
fn primary(families: &str) -> CFRetained<CTFont> {
    let named = |name: &str| {
        let font = with_name(name);
        // SAFETY: a valid font.
        let names = unsafe {
            [
                font.family_name(),
                font.full_name(),
                font.post_script_name(),
            ]
        };
        names
            .iter()
            .any(|n| n.to_string().eq_ignore_ascii_case(name))
            .then_some(font)
    };
    families
        .split(',')
        .map(|name| match name.trim() {
            "sans-serif" => "Helvetica",
            "serif" => "Times",
            "monospace" => "Courier",
            name => name,
        })
        .find_map(named)
        .unwrap_or_else(|| with_name("Helvetica"))
}

/// The font CoreText has by the name `name`, or another for a name it
/// lacks.
fn with_name(name: &str) -> CFRetained<CTFont> {
    // SAFETY: a valid name and a null matrix.
    unsafe { CTFont::with_name(&CFString::from_str(name), 0.0, std::ptr::null()) }
}

/// `primary`, then the fonts of its cascade list for `languages`, that
/// `orb-text` can draw.
fn list(primary: CFRetained<CTFont>, languages: &[String]) -> Vec<Font> {
    let languages: Vec<CFRetained<CFString>> = languages
        .iter()
        .map(|tag| CFString::from_str(tag))
        .collect();
    let languages = CFArray::from_retained_objects(&languages);
    // SAFETY: a valid font and languages; the cascade list holds font
    // descriptors.
    let cascade = unsafe {
        primary
            .default_cascade_list_for_languages(Some(languages.as_ref()))
            .map(|list| CFRetained::cast_unchecked::<CFArray<CTFontDescriptor>>(list))
    };
    std::iter::once(Font::new(primary))
        .chain(cascade.iter().flat_map(|list| fonts(list)))
        .filter(Font::draws)
        .collect()
}

/// The members of the family `family` that `orb-text` can draw, which
/// CoreText finds by name even for most hidden fonts; none for some (the
/// system font's, whose weights are one variable font's instances).
fn members(family: &str) -> Vec<Member> {
    // SAFETY: CoreText's attribute name, which lives for the process.
    let key = unsafe { kCTFontFamilyNameAttribute };
    let name = CFString::from_str(family);
    let attributes = CFDictionary::from_slices(&[key], &[&*name]);
    // SAFETY: valid attributes; the matches are font descriptors.
    unsafe {
        let descriptor = CTFontDescriptor::with_attributes(attributes.as_ref());
        descriptor
            .matching_font_descriptors(None)
            .map(|list| {
                fonts(&CFRetained::cast_unchecked::<CFArray<CTFontDescriptor>>(
                    list,
                ))
            })
            .unwrap_or_default()
            .into_iter()
            .filter(Font::draws)
            .map(|font| Member {
                weight: font.weight(),
                italic: font.italic(),
                font,
            })
            .collect()
    }
}

/// The fonts of `descriptors`.
fn fonts(descriptors: &CFArray<CTFontDescriptor>) -> Vec<Font> {
    descriptors
        .iter()
        // SAFETY: a valid descriptor and a null matrix.
        .map(|d| Font::new(unsafe { CTFont::with_font_descriptor(&d, 0.0, std::ptr::null()) }))
        .collect()
}

/// A font file of `tables`, in tag order as the sfnt format asks.
fn sfnt(mut tables: Vec<([u8; 4], &[u8])>) -> Vec<u8> {
    tables.sort_by_key(|(tag, _)| *tag);
    let count = tables.len() as u16;
    let cff = tables
        .iter()
        .any(|(tag, _)| tag == b"CFF " || tag == b"CFF2");
    // searchRange, entrySelector and rangeShift: the largest power of two
    // at most `count`, for a binary search of the records.
    let entry_selector = count.checked_ilog2().unwrap_or(0) as u16;
    let search_range = (1u16 << entry_selector) * 16;
    let header = 12 + 16 * tables.len();
    let body: usize = tables
        .iter()
        .map(|(_, d)| d.len().next_multiple_of(4))
        .sum();
    let mut out = Vec::with_capacity(header + body);
    out.extend_from_slice(if cff { b"OTTO" } else { &[0, 1, 0, 0] });
    for word in [
        count,
        search_range,
        entry_selector,
        (count * 16).saturating_sub(search_range),
    ] {
        out.extend_from_slice(&word.to_be_bytes());
    }
    let mut offset = header;
    for (tag, data) in &tables {
        out.extend_from_slice(tag);
        out.extend_from_slice(&checksum(data).to_be_bytes());
        out.extend_from_slice(&(offset as u32).to_be_bytes());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        offset += data.len().next_multiple_of(4);
    }
    for (_, data) in &tables {
        out.extend_from_slice(data);
        out.resize(out.len().next_multiple_of(4), 0);
    }
    out
}

/// The sum of a table's big-endian 32-bit words, the last zero-padded.
fn checksum(data: &[u8]) -> u32 {
    data.chunks(4).fold(0u32, |sum, chunk| {
        let mut word = [0; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        sum.wrapping_add(u32::from_be_bytes(word))
    })
}

#[cfg(test)]
mod tests {
    use super::css_weight_rank;

    /// The weight a browser draws from a family of these weights.
    fn chosen(desired: u16, weights: &[u16]) -> u16 {
        *weights
            .iter()
            .min_by_key(|&&w| css_weight_rank(desired, w))
            .unwrap()
    }

    #[test]
    fn weights_match_as_css_matches_them() {
        let family = [300, 400, 600, 800];
        assert_eq!(chosen(400, &family), 400);
        assert_eq!(chosen(500, &family), 400, "lighter before heavier");
        assert_eq!(chosen(450, &[300, 500, 600]), 500, "up to 500 first");
        assert_eq!(chosen(600, &family), 600);
        assert_eq!(chosen(700, &family), 800, "heavier first above 500");
        assert_eq!(chosen(700, &[300, 400, 600]), 600);
        assert_eq!(chosen(200, &family), 300, "heavier when none is lighter");
        assert_eq!(chosen(350, &family), 300, "lighter first below 400");
    }
}

//! Local glyphs from system fonts: fontdb finds a face for
//! the CSS family list GL JS would hand a canvas, falling back per code
//! point as a browser does, and `orb-text` draws it as TinySDF draws a
//! canvas glyph.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use orb_host::{LocalFace, LocalGlyphs, TextMetrics};
use orb_text::FontId;
use orb_text::font::{Drawn, Face, draw};
use orb_text::sdf::InkBox;

/// The system's fonts, loaded on first use.
#[derive(Debug, Default)]
pub struct SystemFonts {
    state: OnceLock<Mutex<State>>,
}

/// Families a browser falls back to for scripts the requested ones lack,
/// tried before every installed face.
#[cfg(target_os = "macos")]
const FALLBACKS: &[&str] = &[
    "Hiragino Sans",
    "PingFang SC",
    "Apple SD Gothic Neo",
    "Kohinoor Devanagari",
    "Thonburi",
    "Geeza Pro",
    "Arial Hebrew",
    "Khmer Sangam MN",
    "Myanmar Sangam MN",
];
/// Windows's own faces for those scripts, installed with every language:
/// without them every CJK code point would scan the installed faces.
#[cfg(windows)]
const FALLBACKS: &[&str] = &[
    "Yu Gothic UI",
    "Microsoft YaHei UI",
    "Malgun Gothic",
    "Nirmala UI",
    "Leelawadee UI",
    "Segoe UI",
    "Khmer UI",
    "Myanmar Text",
];
#[cfg(not(any(target_os = "macos", windows)))]
const FALLBACKS: &[&str] = &[
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "Noto Sans Devanagari",
    "Noto Sans Thai",
    "Noto Sans Arabic",
    "Noto Sans Hebrew",
    "Noto Sans Khmer",
    "Noto Sans Myanmar",
];

struct State {
    db: fontdb::Database,
    faces: HashMap<fontdb::ID, Option<Arc<Face>>>,
    /// Per code point, the face a fallback search found.
    fallback: HashMap<char, Option<fontdb::ID>>,
    /// Per family, weight and style, the face the database matched: each
    /// query scans every installed face.
    queries: HashMap<(String, u16, fontdb::Style), Option<fontdb::ID>>,
    /// The last text drawn, which `coverage` reads after `measure`.
    last: Option<(String, String, Drawn)>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("faces", &self.db.len())
            .finish_non_exhaustive()
    }
}

impl State {
    fn new() -> Self {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        // fontdb doesn't know Android's font directory.
        #[cfg(target_os = "android")]
        db.load_fonts_dir("/system/fonts");
        // What `sans-serif` is in a browser on this platform.
        #[cfg(target_os = "macos")]
        db.set_sans_serif_family("Helvetica");
        #[cfg(target_os = "android")]
        db.set_sans_serif_family("Roboto");
        Self {
            db,
            faces: HashMap::new(),
            fallback: HashMap::new(),
            queries: HashMap::new(),
            last: None,
        }
    }

    fn face(&mut self, id: fontdb::ID) -> Option<Arc<Face>> {
        if let Some(face) = self.faces.get(&id) {
            return face.clone();
        }
        // A system face is identified by its place in the database.
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::hash::DefaultHasher::new();
            id.hash(&mut h);
            h.finish()
        };
        // Mapped, not read: CJK fallbacks are files of 20 to 60 MB, of
        // which a map touches the few tables and glyphs it draws. The
        // faces of one collection share its mapping.
        // SAFETY: the mapping is of an installed font file, which nothing
        // writes while it is installed (the system's are read-only), as
        // browsers map them. A file changed under the map would show the
        // change, not fault, unless truncated.
        #[allow(unsafe_code)]
        let shared = unsafe { self.db.make_shared_face_data(id) };
        let face = shared
            .and_then(|(bytes, index)| Face::identified(bytes, index, FontId(key)))
            .map(Arc::new);
        self.faces.insert(id, face.clone());
        face
    }

    /// The face for `text` in `font`: the first family that has its first
    /// code point, else a fallback face that does.
    fn choose(&mut self, font: &LocalFace<'_>, text: &str) -> Option<Arc<Face>> {
        let first = text.chars().next()?;
        let weight = font
            .weight
            .and_then(|w| w.parse().ok())
            .map_or(fontdb::Weight::NORMAL, fontdb::Weight);
        let style = match font.style {
            "italic" => fontdb::Style::Italic,
            "oblique" => fontdb::Style::Oblique,
            _ => fontdb::Style::Normal,
        };
        let names: Vec<String> = font
            .family
            .split(',')
            .map(|f| f.trim().to_string())
            .collect();
        for name in names
            .iter()
            .map(String::as_str)
            .chain(FALLBACKS.iter().copied())
        {
            let family = match name {
                "sans-serif" => fontdb::Family::SansSerif,
                "serif" => fontdb::Family::Serif,
                "monospace" => fontdb::Family::Monospace,
                "cursive" => fontdb::Family::Cursive,
                "fantasy" => fontdb::Family::Fantasy,
                name => fontdb::Family::Name(name),
            };
            let query = fontdb::Query {
                families: &[family],
                weight,
                stretch: fontdb::Stretch::Normal,
                style,
            };
            let key = (name.to_string(), weight.0, style);
            let db = &self.db;
            let matched = *self.queries.entry(key).or_insert_with(|| db.query(&query));
            if let Some(id) = matched
                && let Some(face) = self.face(id)
                && face.covers(first)
            {
                return Some(face);
            }
        }
        let id = match self.fallback.get(&first) {
            Some(id) => *id,
            None => {
                // The first installed face that has it, in the database's
                // order whatever was drawn before. Each face is mapped
                // once and kept: opening every file again for each code
                // point cost Windows seconds per CJK view (#141).
                let ids: Vec<fontdb::ID> = self.db.faces().map(|f| f.id).collect();
                let id = ids
                    .into_iter()
                    .find(|&id| self.face(id).is_some_and(|f| f.covers(first)));
                self.fallback.insert(first, id);
                id
            }
        };
        self.face(id?)
    }

    fn drawn(&mut self, font: &LocalFace<'_>, text: &str) -> Option<&Drawn> {
        let key = format!("{} {:?} {}", font.style, font.weight, font.family);
        let cached = self
            .last
            .as_ref()
            .is_some_and(|(k, t, _)| *k == key && t == text);
        if !cached {
            let face = self.choose(font, text)?;
            let drawn = draw(&face, text, 48.0);
            self.last = Some((key, text.to_string(), drawn));
        }
        self.last.as_ref().map(|(_, _, d)| d)
    }
}

impl SystemFonts {
    pub fn new() -> Self {
        Self::default()
    }

    /// One database of the system's fonts for the process.
    pub fn shared() -> Arc<Self> {
        static SHARED: OnceLock<Arc<SystemFonts>> = OnceLock::new();
        Arc::clone(SHARED.get_or_init(|| Arc::new(Self::new())))
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .get_or_init(|| Mutex::new(State::new()))
            .lock()
            .expect("system fonts poisoned")
    }
}

impl LocalGlyphs for SystemFonts {
    fn measure(&self, font: &LocalFace<'_>, text: &str) -> TextMetrics {
        let mut state = self.state();
        let Some(drawn) = state.drawn(font, text) else {
            return TextMetrics::default();
        };
        drawn.metrics
    }

    fn coverage(&self, font: &LocalFace<'_>, text: &str, ink: &InkBox) -> Vec<u8> {
        let mut state = self.state();
        match state.drawn(font, text) {
            Some(drawn) => drawn.ink_alpha(ink),
            None => vec![0; (ink.glyph_width * ink.glyph_height) as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_latin_and_complex_scripts_from_system_fonts() {
        let fonts = SystemFonts::new();
        let face = LocalFace {
            family: "Open Sans Semibold,Arial Unicode MS Bold,sans-serif",
            weight: Some("600"),
            style: "normal",
        };
        for text in ["A", "क्ष", "東"] {
            let m = fonts.measure(&face, text);
            assert!(m.advance > 10.0 && m.ascent > 10.0, "{text}: {m:?}");
            let ink = orb_text::sdf::local::ink_box(text, &m);
            let alpha = fonts.coverage(&face, text, &ink);
            assert!(alpha.contains(&255), "{text} has solid ink");
        }
    }
}

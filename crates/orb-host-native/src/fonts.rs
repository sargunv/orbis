//! Local glyphs from system fonts: fontdb finds a face for
//! the CSS family list GL JS would hand a canvas, falling back per code
//! point as a browser does (on Apple platforms to the fonts CoreText
//! falls back to for the user's languages, on Android to its fallback
//! chain for the app's), and `orb-text` draws it as TinySDF draws a
//! canvas glyph.

#[cfg(target_os = "android")]
mod android;
#[cfg(target_vendor = "apple")]
mod apple;

/// The fonts the system itself falls back to.
#[cfg(target_os = "android")]
use android::Fallbacks;
#[cfg(target_vendor = "apple")]
use apple::Fallbacks;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use orb_host::{LocalFace, LocalGlyphs, TextMetrics};
use orb_text::font::{Drawn, Face, draw};
use orb_text::sdf::InkBox;

/// The system's fonts, loaded on first use.
#[derive(Debug, Default)]
pub struct SystemFonts {
    state: OnceLock<Mutex<State>>,
}

/// Families a browser falls back to for scripts the requested ones lack,
/// tried before every installed face; Apple platforms ask CoreText
/// instead, and Android its font matcher, which takes these only before
/// the app gave its JVM or below API level 29. On Windows its own faces
/// for those scripts, installed with every language: without them every
/// CJK code point would scan the installed faces.
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
#[cfg(not(any(target_vendor = "apple", windows)))]
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
    /// Each face opened, at its default instance.
    faces: HashMap<fontdb::ID, Option<Face>>,
    /// Each face at a weight it was drawn at: a variable font's instance
    /// there, which keeps what it derives from the location, or a static
    /// face as it is.
    weighted: HashMap<(fontdb::ID, u16), Face>,
    /// Per code point, the face a search of every installed face found.
    fallback: HashMap<char, Option<fontdb::ID>>,
    /// The fonts the system falls back to for the user's languages.
    #[cfg(any(target_vendor = "apple", target_os = "android"))]
    system: Fallbacks,
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
            weighted: HashMap::new(),
            fallback: HashMap::new(),
            #[cfg(any(target_vendor = "apple", target_os = "android"))]
            system: Fallbacks::default(),
            queries: HashMap::new(),
            last: None,
        }
    }

    /// The face `id` at its default instance, opened once.
    fn face(&mut self, id: fontdb::ID) -> Option<&Face> {
        open(&mut self.faces, &mut self.db, id)
    }

    /// The face for `text` in `font`, at its weight: the first family that
    /// has its first code point, else a fallback face that does.
    fn choose(&mut self, font: &LocalFace<'_>, text: &str) -> Option<Face> {
        let weight = font
            .weight
            .and_then(|w| w.parse().ok())
            .map_or(fontdb::Weight::NORMAL, fontdb::Weight);
        let id = self.choose_id(font, weight, text)?;
        if let Some(face) = self.weighted.get(&(id, weight.0)) {
            return Some(face.clone());
        }
        let face = self.face(id)?.at_weight(f32::from(weight.0));
        self.weighted.insert((id, weight.0), face.clone());
        Some(face)
    }

    /// The database's face for `text` in `font`, as [`choose`](Self::choose)
    /// picks it: the family list's, then the system's fallback, then a
    /// fixed list's, then the first installed face's.
    fn choose_id(
        &mut self,
        font: &LocalFace<'_>,
        weight: fontdb::Weight,
        text: &str,
    ) -> Option<fontdb::ID> {
        let first = text.chars().next()?;
        let style = match font.style {
            "italic" => fontdb::Style::Italic,
            "oblique" => fontdb::Style::Oblique,
            _ => fontdb::Style::Normal,
        };
        for name in font.family.split(',') {
            if let Some(id) = self.named(name.trim(), weight, style, first) {
                return Some(id);
            }
        }
        #[cfg(any(target_vendor = "apple", target_os = "android"))]
        {
            let italic = style != fontdb::Style::Normal;
            let faces = &mut self.faces;
            let draws = |db: &mut fontdb::Database, id| {
                open(faces, db, id).is_some_and(|f| f.covers(first))
            };
            if let Some(id) =
                (self.system).face(&mut self.db, font.family, first, weight.0, italic, draws)
            {
                return Some(id);
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        for name in FALLBACKS {
            if let Some(id) = self.named(name, weight, style, first) {
                return Some(id);
            }
        }
        match self.fallback.get(&first) {
            Some(id) => *id,
            None => {
                // The first installed face that has it, in the database's
                // order whatever was drawn before, so not the fallbacks
                // rebuilt in memory. Each face is mapped once and kept:
                // opening every file again for each code point cost
                // Windows seconds per CJK view (#141).
                let ids: Vec<fontdb::ID> = self
                    .db
                    .faces()
                    .filter(|f| !matches!(f.source, fontdb::Source::Binary(_)))
                    .map(|f| f.id)
                    .collect();
                let id = ids
                    .into_iter()
                    .find(|&id| self.face(id).is_some_and(|f| f.covers(first)));
                self.fallback.insert(first, id);
                id
            }
        }
    }

    /// The database's face of the CSS family `name` at `weight` and
    /// `style`, if it has `first`.
    fn named(
        &mut self,
        name: &str,
        weight: fontdb::Weight,
        style: fontdb::Style,
        first: char,
    ) -> Option<fontdb::ID> {
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
        matched.filter(|&id| self.face(id).is_some_and(|face| face.covers(first)))
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

/// The face `id` of `db` at its default instance, opened once into
/// `faces`.
fn open<'a>(
    faces: &'a mut HashMap<fontdb::ID, Option<Face>>,
    db: &mut fontdb::Database,
    id: fontdb::ID,
) -> Option<&'a Face> {
    faces
        .entry(id)
        .or_insert_with(|| {
            // Mapped, not read: CJK fallbacks are files of 20 to 60 MB, of
            // which a map touches the few tables and glyphs it draws. The
            // faces of one collection share its mapping.
            // SAFETY: the mapping is of an installed font file, which
            // nothing writes while it is installed (the system's are
            // read-only), as browsers map them. A file changed under the
            // map would show the change, not fault, unless truncated. A
            // face rebuilt in memory is shared, not mapped.
            #[allow(unsafe_code)]
            let shared = unsafe { db.make_shared_face_data(id) };
            shared.and_then(|(bytes, index)| Face::new(bytes, index))
        })
        .as_ref()
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

    /// The face `state` draws `text` in, in the default sans-serif at
    /// `weight`: its PostScript name and the ink it draws.
    #[cfg(target_vendor = "apple")]
    fn drawn_in(state: &mut State, text: &str, weight: &str) -> (String, u64) {
        let font = LocalFace {
            family: "sans-serif",
            weight: Some(weight),
            style: "normal",
        };
        let id = state
            .choose_id(&font, fontdb::Weight(weight.parse().unwrap()), text)
            .unwrap_or_else(|| panic!("no face for {text}"));
        let name = state.db.face(id).unwrap().post_script_name.clone();
        let face = state.choose(&font, text).unwrap();
        let ink = coverage(&face, text).iter().map(|&a| u64::from(a)).sum();
        (name, ink)
    }

    /// `text`'s coverage in `face`, as a glyph job reads it.
    #[cfg(target_vendor = "apple")]
    fn coverage(face: &Face, text: &str) -> Vec<u8> {
        let drawn = draw(face, text, 48.0);
        drawn.ink_alpha(&orb_text::sdf::local::ink_box(text, &drawn.metrics))
    }

    /// Han ideographs take the forms of the language the user reads: the
    /// font CoreText falls back to for it. (Chinese can't be checked
    /// this way: recent macOS's Chinese fonts have only `hvgl` outlines,
    /// so Chinese falls through to the Japanese font.)
    #[cfg(target_vendor = "apple")]
    #[test]
    fn ideographs_take_the_users_language_and_the_weight_asked() {
        let mut faces = Vec::new();
        for language in ["ja-JP", "ko-KR"] {
            let mut state = State::new();
            state.system = apple::Fallbacks::with_languages(vec![language.to_string()]);
            let (regular, regular_ink) = drawn_in(&mut state, "直", "400");
            let (bold, bold_ink) = drawn_in(&mut state, "直", "700");
            assert!(regular_ink > 0, "{language}: {regular} draws the ideograph");
            assert!(
                bold_ink > regular_ink,
                "{language}: {bold} draws bolder than {regular}"
            );
            faces.push(regular);
        }
        assert_ne!(faces[0], faces[1], "Japanese and Korean forms differ");
    }

    /// A fallback font the database lacks (every system font on iOS,
    /// whose files an app can't read) is rebuilt from its tables into one
    /// that draws.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn fallback_fonts_without_files_are_rebuilt_from_their_tables() {
        let mut db = fontdb::Database::new();
        let mut faces = HashMap::new();
        let mut fallbacks = apple::Fallbacks::with_languages(vec!["ja".to_string()]);
        for c in ['直', 'A', 'ก'] {
            let id = fallbacks
                .face(&mut db, "sans-serif", c, 400, false, |db, id| {
                    open(&mut faces, db, id).is_some()
                })
                .unwrap_or_else(|| panic!("a face for {c}"));
            let face = open(&mut faces, &mut db, id).unwrap();
            assert!(face.covers(c), "{c}");
            assert!(
                coverage(face, &c.to_string()).contains(&255),
                "{c} has solid ink"
            );
        }
    }
}

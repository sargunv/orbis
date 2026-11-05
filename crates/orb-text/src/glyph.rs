//! Glyph identity, the glyphs a tile asks for, and the provider each comes
//! from (decision 0009).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use crate::label::{Label, SectionKind, Tagged};
use crate::segment::is_cluster;
use crate::transform::vertical_form;
use crate::{shaping, unicode};

/// A `text-font` stack: font names joined by `,`, as GL JS keys glyphs and
/// fills a glyphs URL's `{fontstack}`.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct FontStack(Arc<str>);

impl FontStack {
    pub fn new<S: AsRef<str>>(fonts: &[S]) -> Self {
        let joined: Vec<&str> = fonts.iter().map(AsRef::as_ref).collect();
        Self(joined.join(",").into())
    }

    /// A stack already joined by `,`.
    pub fn from_joined(stack: &str) -> Self {
        Self(stack.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for FontStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

/// Which form of a glyph: as written, or its vertical alternate.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Variant {
    Default,
    Vertical,
}

/// A font file loaded by the runtime: a hash of its bytes and face index,
/// so every worker loading the same file agrees on it.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct FontId(pub u64);

/// What a glyph is, across tiles, workers and the atlas.
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum GlyphKey {
    /// A glyph from a v8 `glyphs` PBF range: one code point of a stack.
    Pbf { stack: FontStack, code_point: u32 },
    /// A cluster drawn from a font file (a declared `font-faces` file, or
    /// a host's system font): one atlas glyph per cluster, as GL JS's local
    /// path draws one, shaped inside by HarfRust.
    Font {
        font: FontId,
        variant: Variant,
        text: Box<str>,
    },
    /// A cluster drawn by a platform text engine from a font Orbis can't
    /// read (the browser's system fonts).
    Cluster {
        stack: FontStack,
        variant: Variant,
        text: Box<str>,
    },
}

/// Glyph placement in 24 px em units (GL JS `GlyphMetrics`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlyphMetrics {
    pub width: f64,
    pub height: f64,
    pub left: f64,
    pub top: f64,
    pub advance: f64,
}

/// A resolved glyph: its identity, metrics and SDF bitmap size.
#[derive(Clone, Debug, PartialEq)]
pub struct Glyph {
    pub key: GlyphKey,
    pub metrics: GlyphMetrics,
    /// The SDF bitmap's size in texels, including its border.
    pub bitmap: [u32; 2],
    /// Rasterized at 48 px rather than 24 (GL JS `isDoubleResolution`).
    pub double_resolution: bool,
}

/// A glyph a tile needs: one cluster of one stack, in one variant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GlyphRequest {
    pub stack: FontStack,
    pub variant: Variant,
    pub text: Box<str>,
}

/// The glyphs a tile needs, deduplicated, in first-seen order.
#[derive(Debug, Default)]
pub struct GlyphRequests {
    requests: Vec<GlyphRequest>,
    seen: HashSet<GlyphRequest>,
}

impl GlyphRequests {
    /// Adds a label's glyphs, as GL JS's `calculateGlyphDependencies`: each
    /// cluster, each of its code points (drawn one at a time where the
    /// cluster has no glyph), and, when vertical forms may be used, vertical
    /// alternates and the vertical punctuation substitutes.
    pub fn add_label(
        &mut self,
        label: &Label,
        default_stack: &FontStack,
        needs_vertical_forms: bool,
        allow_vertical_placement: bool,
    ) {
        let tagged = Tagged::from_label(label, default_stack, false, None);
        for i in 0..tagged.len() {
            let SectionKind::Text(stack) = &tagged.section(i).kind else {
                continue;
            };
            let grapheme = tagged.grapheme(i);
            let first = grapheme.chars().next().expect("clusters are not empty");
            let vertical = needs_vertical_forms
                && shaping::may_use_vertical_glyph(first, allow_vertical_placement);
            let cluster = is_cluster(grapheme);
            if cluster {
                self.add(stack, Variant::Default, grapheme);
            }
            if vertical {
                self.add(stack, Variant::Vertical, grapheme);
            }
            for c in grapheme.chars() {
                let mut buf = [0; 4];
                self.add(stack, Variant::Default, c.encode_utf8(&mut buf));
                if !needs_vertical_forms {
                    continue;
                }
                if vertical && cluster {
                    self.add(stack, Variant::Vertical, c.encode_utf8(&mut buf));
                }
                if let Some(v) = vertical_form(c) {
                    self.add(stack, Variant::Default, v.encode_utf8(&mut buf));
                }
            }
        }
    }

    fn add(&mut self, stack: &FontStack, variant: Variant, text: &str) {
        let request = GlyphRequest {
            stack: stack.clone(),
            variant,
            text: text.into(),
        };
        if self.seen.insert(request.clone()) {
            self.requests.push(request);
        }
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GlyphRequest> {
        self.requests.iter()
    }
}

impl IntoIterator for GlyphRequests {
    type Item = GlyphRequest;
    type IntoIter = std::vec::IntoIter<GlyphRequest>;

    fn into_iter(self) -> Self::IntoIter {
        self.requests.into_iter()
    }
}

/// Index of a glyph in a [`GlyphMap`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlyphId(pub u32);

/// The glyphs available to shaping: each requested cluster resolved to a
/// glyph or to nothing (GL JS's `GlyphMap` and `GlyphPositions` in one).
#[derive(Debug, Default)]
pub struct GlyphMap {
    entries: Vec<(GlyphRequest, Option<Glyph>)>,
    index: HashMap<FontStack, [HashMap<Box<str>, GlyphId>; 2]>,
}

impl GlyphMap {
    /// Records what `request` resolved to; `None` means unavailable.
    pub fn insert(&mut self, request: GlyphRequest, glyph: Option<Glyph>) -> GlyphId {
        let by_variant = self.index.entry(request.stack.clone()).or_default();
        let ids = &mut by_variant[request.variant as usize];
        if let Some(&id) = ids.get(&request.text) {
            self.entries[id.0 as usize].1 = glyph;
            return id;
        }
        let id = GlyphId(u32::try_from(self.entries.len()).expect("glyph count fits u32"));
        ids.insert(request.text.clone(), id);
        self.entries.push((request, glyph));
        id
    }

    /// The glyph for `text`, if one was resolved.
    pub fn get(
        &self,
        stack: &FontStack,
        variant: Variant,
        text: &str,
    ) -> Option<(GlyphId, &Glyph)> {
        let id = *self.index.get(stack)?[variant as usize].get(text)?;
        let glyph = self.entries[id.0 as usize].1.as_ref()?;
        Some((id, glyph))
    }

    pub(crate) fn has(&self, stack: &FontStack, variant: Variant, text: &str) -> bool {
        self.get(stack, variant, text).is_some()
    }

    /// How many requests the map holds: ids are `0..len`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// What `id` was requested as.
    pub fn request(&self, id: GlyphId) -> &GlyphRequest {
        &self.entries[id.0 as usize].0
    }

    /// The glyph `id` resolved to.
    pub fn glyph(&self, id: GlyphId) -> Option<&Glyph> {
        self.entries[id.0 as usize].1.as_ref()
    }
}

/// How a style's glyphs are provided (map and style options).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FontPolicy {
    /// The style has a v8 `glyphs` URL.
    pub glyphs_url: bool,
    /// The map sets `localIdeographFontFamily`.
    pub local_ideographs: bool,
}

/// The font a local glyph is drawn with, as GL JS's `_getTinySDF` and
/// `_createTinySDF` choose it: the stack's fonts then `sans-serif`, or the
/// map's `localIdeographFontFamily` for ideographs of the default stack,
/// with weight and style sniffed from the first font's name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LocalFont {
    /// A CSS font-family list.
    pub family: String,
    /// A CSS weight (`"600"`), or the normal weight.
    pub weight: Option<String>,
    /// `normal`, `italic` or `oblique`.
    pub style: &'static str,
}

/// The v8 `text-font` default, GL JS's `defaultStack`.
const DEFAULT_STACK: &str = "Open Sans Regular,Arial Unicode MS Regular";

impl LocalFont {
    /// The font for `text` of `stack`; `ideographs` is the map's
    /// `localIdeographFontFamily`, if set.
    pub fn for_text(stack: &FontStack, text: &str, ideographs: Option<&str>) -> Self {
        let first = text.chars().next().unwrap_or('\0');
        let family = match ideographs {
            Some(family)
                if stack.as_str() == DEFAULT_STACK
                    && !family.is_empty()
                    && unicode::uses_local_ideograph_font(first) =>
            {
                family
            }
            _ => stack.as_str(),
        };
        let mut families: Vec<&str> = family.split(',').collect();
        families.push("sans-serif");
        // GL JS's `/[-\w]+/.test(name)` matches every name with a word
        // character, so names are never quoted in practice.
        let css = families.join(",");
        let first = families[0];
        Self {
            family: css,
            weight: font_weight(first),
            style: font_style(first),
        }
    }

    /// The canvas font shorthand TinySDF sets, at its 48 px.
    pub fn css(&self) -> String {
        format!(
            "{} {} {}px {}",
            self.style,
            self.weight.as_deref().unwrap_or("normal"),
            crate::sdf::local::FONT_SIZE,
            self.family
        )
    }
}

/// GL JS `_fontStyle`.
fn font_style(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    if lower.contains("italic") {
        "italic"
    } else if lower.contains("oblique") {
        "oblique"
    } else {
        "normal"
    }
}

/// GL JS `_fontWeight`: the last of its names (in its table's order) that
/// the font name has as a whole word.
fn font_weight(name: &str) -> Option<String> {
    const WEIGHTS: [(&str, u32); 17] = [
        ("thin", 100),
        ("hairline", 100),
        ("extra light", 200),
        ("ultra light", 200),
        ("light", 300),
        ("normal", 400),
        ("regular", 400),
        ("medium", 500),
        ("semibold", 600),
        ("demibold", 600),
        ("bold", 700),
        ("extra bold", 800),
        ("ultra bold", 800),
        ("black", 900),
        ("heavy", 900),
        ("extra black", 950),
        ("ultra black", 950),
    ];
    let lower = name.to_ascii_lowercase();
    let word = |w: &str| {
        lower.match_indices(w).any(|(at, _)| {
            let before = lower[..at].chars().next_back();
            let after = lower[at + w.len()..].chars().next();
            // JavaScript's `\b`: between `\w` (ASCII word) and not.
            let boundary =
                |c: Option<char>| c.is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
            boundary(before) && boundary(after)
        })
    };
    WEIGHTS
        .iter()
        .rev()
        .find(|(w, _)| word(w))
        .map(|(_, weight)| weight.to_string())
}

/// Where a requested glyph comes from (GL JS `GlyphManager`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    /// PBF range `code_point / 256` of the stack; local if that fails.
    Pbf { range: u32 },
    /// Drawn from a local font: host system fonts, or the browser's.
    Local,
    /// Never available: vertical alternates need declared font files.
    Unavailable,
}

/// Chooses the provider for a request, as GL JS's
/// `_getAndCacheGlyphsPromise` does for a style without `font-faces`.
pub fn provider(request: &GlyphRequest, policy: FontPolicy) -> Provider {
    if request.variant != Variant::Default {
        return Provider::Unavailable;
    }
    let first = request.text.chars().next().unwrap_or('\0');
    if !policy.glyphs_url
        || is_cluster(&request.text)
        || (policy.local_ideographs && unicode::uses_local_ideograph_font(first))
        || unicode::is_complex_shaping(first)
    {
        return Provider::Local;
    }
    Provider::Pbf {
        range: first as u32 / 256,
    }
}

/// The URL of a stack's PBF range, filling the first `{fontstack}` and
/// `{range}` of a v8 `glyphs` template as GL JS does.
pub fn glyphs_url(template: &str, stack: &FontStack, range: u32) -> String {
    let start = range * 256;
    template
        .replacen("{fontstack}", stack.as_str(), 1)
        .replacen("{range}", &format!("{start}-{}", start + 255), 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_fonts_follow_gl_js() {
        let semibold = LocalFont::for_text(
            &FontStack::new(&["Open Sans Semibold", "Arial Unicode MS Bold"]),
            "ก",
            Some("sans-serif"),
        );
        // The last weight name the first font has wins (GL JS's order).
        assert_eq!(
            semibold.css(),
            "normal 600 48px Open Sans Semibold,Arial Unicode MS Bold,sans-serif"
        );
        let default = FontStack::from_joined(DEFAULT_STACK);
        // Ideographs of the default stack draw with the map's family.
        assert_eq!(
            LocalFont::for_text(&default, "東", Some("Hiragino Sans")).css(),
            "normal normal 48px Hiragino Sans,sans-serif"
        );
        assert_eq!(
            LocalFont::for_text(&default, "क", Some("Hiragino Sans")).css(),
            "normal 400 48px Open Sans Regular,Arial Unicode MS Regular,sans-serif"
        );
        let italic = LocalFont::for_text(&FontStack::from_joined("Noto Sans Italic"), "a", None);
        assert_eq!(italic.style, "italic");
        assert_eq!(font_weight("Extra Bold Condensed").as_deref(), Some("800"));
        assert_eq!(font_weight("Boldface"), None);
    }
    use crate::label::Section;

    fn requested(requests: &GlyphRequests) -> Vec<(Variant, &str)> {
        requests.iter().map(|r| (r.variant, &*r.text)).collect()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn requests_clusters_their_code_points_and_vertical_forms() {
        let stack = FontStack::new(&["Test"]);
        let mut requests = GlyphRequests::default();
        requests.add_label(&Label::from_text("e\u{301}a"), &stack, false, false);
        use Variant::*;
        assert_eq!(
            requested(&requests),
            [
                (Default, "e\u{301}"),
                (Default, "e"),
                (Default, "\u{301}"),
                (Default, "a")
            ]
        );

        let mut requests = GlyphRequests::default();
        requests.add_label(&Label::from_text("東(京"), &stack, true, false);
        assert_eq!(
            requested(&requests),
            [
                (Vertical, "東"),
                (Default, "東"),
                (Vertical, "("),
                (Default, "("),
                (Default, "︵"),
                (Vertical, "京"),
                (Default, "京"),
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn a_cluster_spanning_sections_belongs_to_its_first_section() {
        let a = FontStack::new(&["A"]);
        let b = FontStack::new(&["B"]);
        let label = Label {
            sections: vec![
                Section::text("e", None, Some(a.clone())),
                Section::text("\u{301}x", None, Some(b.clone())),
            ],
        };
        let mut requests = GlyphRequests::default();
        requests.add_label(&label, &a, false, false);
        let stacks: Vec<(&str, &str)> = requests
            .iter()
            .map(|r| (r.stack.as_str(), &*r.text))
            .collect();
        assert_eq!(
            stacks,
            [("A", "e\u{301}"), ("A", "e"), ("A", "\u{301}"), ("B", "x")]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn chooses_providers_as_gl_js() {
        let stack = FontStack::new(&["Open Sans Semibold", "Arial Unicode MS Bold"]);
        let request = |text: &str, variant| GlyphRequest {
            stack: stack.clone(),
            variant,
            text: text.into(),
        };
        let pbf = FontPolicy {
            glyphs_url: true,
            local_ideographs: false,
        };
        let local_cjk = FontPolicy {
            local_ideographs: true,
            ..pbf
        };
        assert_eq!(
            provider(&request("a", Variant::Default), pbf),
            Provider::Pbf { range: 0 }
        );
        assert_eq!(
            provider(&request("東", Variant::Default), pbf),
            Provider::Pbf { range: 0x67 }
        );
        assert_eq!(
            provider(&request("東", Variant::Default), local_cjk),
            Provider::Local
        );
        assert_eq!(
            provider(&request("e\u{301}", Variant::Default), pbf),
            Provider::Local
        );
        assert_eq!(
            provider(&request("क", Variant::Default), pbf),
            Provider::Local
        );
        assert_eq!(
            provider(&request("a", Variant::Default), FontPolicy::default()),
            Provider::Local
        );
        assert_eq!(
            provider(&request("(", Variant::Vertical), pbf),
            Provider::Unavailable
        );
        assert_eq!(
            provider(&request("\u{1F5FA}", Variant::Default), pbf),
            Provider::Pbf { range: 0x1F5 }
        );
        assert_eq!(
            glyphs_url("local://glyphs/{fontstack}/{range}.pbf", &stack, 0x67),
            "local://glyphs/Open Sans Semibold,Arial Unicode MS Bold/26368-26623.pbf"
        );
    }
}

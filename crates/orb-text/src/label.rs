//! Label text and its sections, and the cluster-indexed string layout works
//! on (GL JS `TaggedString`, including `determineLineBreaks`).

use std::ops::Range;

use crate::glyph::{FontStack, GlyphMap, Variant};
use crate::segment::{WordBreaker, graphemes};
use crate::shaping::ONE_EM;
use crate::transform::verticalize_punctuation;
use crate::unicode;

/// A label: the evaluated `text-field`, as sections of text or images.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Label {
    pub sections: Vec<Section>,
}

impl Label {
    /// One section of plain text.
    pub fn from_text(text: &str) -> Self {
        Self {
            sections: vec![Section::text(text, None, None)],
        }
    }

    /// The text of every text section, joined (GL JS `Formatted.toString`).
    pub fn text(&self) -> String {
        self.sections
            .iter()
            .filter_map(|s| match &s.content {
                Content::Text(t) => Some(t.as_str()),
                Content::Image(_) => None,
            })
            .collect()
    }

    /// Whether shaping it runs the bidirectional algorithm: a text section
    /// holds a right-to-left character, or an image section takes a private
    /// use character, which counts as one. The web loads that code apart
    /// (orb-runtime's `Modules::BIDI`).
    pub fn needs_bidi(&self) -> bool {
        self.sections.iter().any(|s| match &s.content {
            Content::Text(text) => unicode::contains_rtl(text),
            Content::Image(name) => !name.is_empty(),
        })
    }
}

/// One section of a `format` expression or plain text.
#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    pub content: Content,
    /// `font-scale`; unset or 0 means 1.
    pub scale: Option<f64>,
    /// `text-font` override; unset means the layer's stack.
    pub font_stack: Option<FontStack>,
    pub vertical_align: VerticalAlign,
}

impl Section {
    pub fn text(text: &str, scale: Option<f64>, font_stack: Option<FontStack>) -> Self {
        Self {
            content: Content::Text(text.into()),
            scale,
            font_stack,
            vertical_align: VerticalAlign::Bottom,
        }
    }

    pub fn image(name: &str) -> Self {
        Self {
            content: Content::Image(name.into()),
            scale: None,
            font_stack: None,
            vertical_align: VerticalAlign::Bottom,
        }
    }

    #[must_use]
    pub fn with_vertical_align(mut self, align: VerticalAlign) -> Self {
        self.vertical_align = align;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    Text(String),
    /// An image, by name.
    Image(String),
}

/// `format`'s `vertical-align`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VerticalAlign {
    #[default]
    Bottom,
    Center,
    Top,
}

impl VerticalAlign {
    pub(crate) fn factor(self) -> f64 {
        match self {
            Self::Top => 0.0,
            Self::Center => 0.5,
            Self::Bottom => 1.0,
        }
    }
}

/// An image placed in text: its padded atlas size and pixel ratio, as
/// GL JS's `ImagePosition` carries them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InlineImage {
    /// Size in pixels including the one-pixel padding on each side.
    pub padded_size: [u32; 2],
    pub pixel_ratio: f64,
}

/// GL JS `IMAGE_PADDING`.
pub const IMAGE_PADDING: f64 = 1.0;

impl InlineImage {
    /// Size in layout pixels (GL JS `displaySize`).
    pub fn display_size(&self) -> [f64; 2] {
        [
            (f64::from(self.padded_size[0]) - IMAGE_PADDING * 2.0) / self.pixel_ratio,
            (f64::from(self.padded_size[1]) - IMAGE_PADDING * 2.0) / self.pixel_ratio,
        ]
    }
}

/// Looks inline images up by name.
pub type Images<'a> = &'a dyn Fn(&str) -> Option<InlineImage>;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SectionKind {
    Text(FontStack),
    Image(Box<str>),
}

/// A section as layout sees it (GL JS `SectionOptions`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SectionOptions {
    pub(crate) scale: f64,
    pub(crate) vertical_align: VerticalAlign,
    pub(crate) kind: SectionKind,
}

const PUA_BEGIN: u32 = 0xE000;
const PUA_END: u32 = 0xF8FF;

/// Text indexed by grapheme cluster, each cluster tagged with its section.
#[derive(Clone, Debug)]
pub(crate) struct Tagged {
    pub(crate) text: String,
    graphemes: Vec<Range<usize>>,
    pub(crate) sections: Vec<SectionOptions>,
    pub(crate) section_index: Vec<usize>,
    image_section_id: Option<u32>,
}

impl Tagged {
    pub(crate) fn new(
        text: String,
        sections: Vec<SectionOptions>,
        section_index: Vec<usize>,
    ) -> Self {
        let graphemes = graphemes(&text);
        Self {
            text,
            graphemes,
            sections,
            section_index,
            image_section_id: None,
        }
    }

    /// GL JS `TaggedString.fromFeature`.
    pub(crate) fn from_label(
        label: &Label,
        default_stack: &FontStack,
        vertical: bool,
        glyphs: Option<&GlyphMap>,
    ) -> Self {
        let mut tagged = Self::new(String::new(), Vec::new(), Vec::new());
        for section in &label.sections {
            match &section.content {
                Content::Text(text) => {
                    tagged.sections.push(SectionOptions {
                        scale: section
                            .scale
                            .filter(|&s| s != 0.0 && !s.is_nan())
                            .unwrap_or(1.0),
                        vertical_align: section.vertical_align,
                        kind: SectionKind::Text(
                            section
                                .font_stack
                                .clone()
                                .unwrap_or_else(|| default_stack.clone()),
                        ),
                    });
                    tagged.append(text, tagged.sections.len() - 1);
                }
                Content::Image(name) => {
                    if name.is_empty() {
                        continue;
                    }
                    let Some(code) = tagged.next_image_code() else {
                        continue;
                    };
                    tagged.sections.push(SectionOptions {
                        scale: 1.0,
                        vertical_align: section.vertical_align,
                        kind: SectionKind::Image(name.as_str().into()),
                    });
                    let c = char::from_u32(code).expect("private use is a character");
                    tagged.append(c.encode_utf8(&mut [0; 4]), tagged.sections.len() - 1);
                }
            }
        }
        if vertical {
            tagged.verticalize_punctuation(glyphs, None);
        }
        tagged
    }

    fn next_image_code(&mut self) -> Option<u32> {
        let next = match self.image_section_id {
            None => PUA_BEGIN,
            Some(id) if id >= PUA_END => return None,
            Some(id) => id + 1,
        };
        self.image_section_id = Some(next);
        Some(next)
    }

    /// Appends a section's text. A cluster belongs to the section of its
    /// first character, so text that only extends the last cluster adds no
    /// cluster of its own.
    fn append(&mut self, text: &str, section: usize) {
        let tail = self.graphemes.pop();
        let tail_start = tail.as_ref().map_or(self.text.len(), |t| t.start);
        self.text.push_str(text);
        let joined = graphemes(&self.text[tail_start..]);
        let added = joined.len() - usize::from(tail.is_some());
        self.graphemes.extend(
            joined
                .into_iter()
                .map(|r| r.start + tail_start..r.end + tail_start),
        );
        self.section_index
            .extend(std::iter::repeat_n(section, added));
    }

    pub(crate) fn len(&self) -> usize {
        self.graphemes.len()
    }

    pub(crate) fn grapheme(&self, i: usize) -> &str {
        &self.text[self.graphemes[i].clone()]
    }

    pub(crate) fn grapheme_range(&self, i: usize) -> Range<usize> {
        self.graphemes[i].clone()
    }

    pub(crate) fn first_char(&self, i: usize) -> char {
        self.grapheme(i)
            .chars()
            .next()
            .expect("clusters are not empty")
    }

    pub(crate) fn section(&self, i: usize) -> &SectionOptions {
        &self.sections[self.section_index[i]]
    }

    /// Replaces the text with these clusters, without segmenting it again.
    pub(crate) fn set_graphemes(&mut self, clusters: &[String]) {
        self.text.clear();
        self.graphemes.clear();
        for cluster in clusters {
            let start = self.text.len();
            self.text.push_str(cluster);
            self.graphemes.push(start..self.text.len());
        }
    }

    pub(crate) fn clusters(&self) -> Vec<String> {
        (0..self.len())
            .map(|i| self.grapheme(i).to_string())
            .collect()
    }

    /// Uses vertical punctuation where a font's vertical form is missing or
    /// unused, keeping cluster boundaries and sections (GL JS
    /// `TaggedString.verticalizePunctuation`).
    pub(crate) fn verticalize_punctuation(
        &mut self,
        glyphs: Option<&GlyphMap>,
        verticals: Option<&[bool]>,
    ) {
        let replacements = verticalize_punctuation(&self.text);
        let mut offset = 0;
        let clusters: Vec<String> = (0..self.len())
            .map(|i| {
                let grapheme = self.grapheme(i);
                let n = grapheme.chars().count();
                let replacement: String = replacements[offset..offset + n].iter().collect();
                offset += n;
                let keep = verticals.is_none_or(|v| v[i])
                    && match &self.section(i).kind {
                        SectionKind::Text(stack) => {
                            glyphs.is_some_and(|g| g.has(stack, Variant::Vertical, grapheme))
                        }
                        SectionKind::Image(_) => false,
                    };
                if keep {
                    grapheme.to_string()
                } else {
                    replacement
                }
            })
            .collect();
        self.set_graphemes(&clusters);
    }

    /// Drops whitespace clusters at both ends (GL JS `trim`), then segments
    /// the remaining text afresh, as GL JS does.
    pub(crate) fn trim(&mut self) {
        let is_space = |i: usize| self.grapheme(i).chars().all(unicode::is_whitespace);
        let mut start = 0;
        while start < self.len() && is_space(start) {
            start += 1;
        }
        let mut end = self.len();
        while end > start && is_space(end - 1) {
            end -= 1;
        }
        let range = if start < end {
            self.graphemes[start].start..self.graphemes[end - 1].end
        } else {
            0..0
        };
        self.text = self.text[range].to_string();
        self.section_index = self.section_index[start..end].to_vec();
        self.graphemes = graphemes(&self.text);
    }

    pub(crate) fn substring(&self, start: usize, end: usize) -> Tagged {
        let range = if start < end {
            self.graphemes[start].start..self.graphemes[end - 1].end
        } else {
            0..0
        };
        Tagged::new(
            self.text[range].to_string(),
            self.sections.clone(),
            self.section_index[start..end].to_vec(),
        )
    }

    /// The byte offset of cluster `i` (GL JS `toCodeUnitIndex`).
    pub(crate) fn byte_index(&self, i: usize) -> usize {
        self.graphemes.get(i).map_or(self.text.len(), |r| r.start)
    }

    pub(crate) fn max_scale(&self) -> f64 {
        self.section_index
            .iter()
            .fold(0.0, |max, &i| js_max(max, self.sections[i].scale))
    }

    pub(crate) fn max_image_size(&self, images: Images<'_>) -> [f64; 2] {
        let mut max = [0.0, 0.0];
        for i in 0..self.len() {
            if let SectionKind::Image(name) = &self.section(i).kind
                && let Some(image) = images(name)
            {
                let size = image.display_size();
                max = [js_max(max[0], size[0]), js_max(max[1], size[1])];
            }
        }
        max
    }

    fn advance(
        &self,
        i: usize,
        glyphs: &GlyphMap,
        images: Images<'_>,
        spacing: f64,
        layout_text_size: f64,
    ) -> f64 {
        let section = self.section(i);
        let grapheme = self.grapheme(i);
        match &section.kind {
            SectionKind::Text(stack) => {
                if let Some((_, glyph)) = glyphs.get(stack, Variant::Default, grapheme) {
                    return glyph.metrics.advance * section.scale + spacing;
                }
                let mut advance = 0.0;
                for c in grapheme.chars() {
                    if let Some((_, glyph)) =
                        glyphs.get(stack, Variant::Default, c.encode_utf8(&mut [0; 4]))
                    {
                        advance += glyph.metrics.advance * section.scale + spacing;
                    }
                }
                advance
            }
            SectionKind::Image(name) => match images(name) {
                Some(image) => {
                    image.display_size()[0] * section.scale * ONE_EM / layout_text_size + spacing
                }
                None => 0.0,
            },
        }
    }

    /// The cluster indices to break at for lines of about `max_width`
    /// (GL JS `determineLineBreaks`).
    pub(crate) fn line_breaks(
        &self,
        spacing: f64,
        max_width: f64,
        glyphs: &GlyphMap,
        images: Images<'_>,
        layout_text_size: f64,
        words: Option<&WordBreaker>,
    ) -> Vec<usize> {
        // Each cluster's advance, looked up once: on the stack for most
        // labels.
        let (mut short, mut long) = ([0.0; 64], Vec::new());
        let advances = if self.len() <= short.len() {
            &mut short[..self.len()]
        } else {
            long.resize(self.len(), 0.0);
            &mut long[..]
        };
        for (i, advance) in advances.iter_mut().enumerate() {
            *advance = self.advance(i, glyphs, images, spacing, layout_text_size);
        }
        let advances = &*advances;
        let total_width = advances.iter().fold(0.0, |total, a| total + a);
        if spacing >= 0.0
            && total_width > 0.0
            && total_width <= max_width
            && !self.text.contains(['\n', '\r'])
        {
            return vec![self.len()];
        }

        let target_width = total_width / js_max(1.0, (total_width / max_width).ceil());
        let zero_width_spaces = self.text.contains('\u{200b}');
        let mut breaks: Vec<Break> = Vec::new();
        let mut word_starts: Option<Vec<usize>> = None;
        let mut x = 0.0;

        for (i, &advance) in advances.iter().enumerate() {
            let c = self.first_char(i);
            if i > 0 {
                let previous = self.first_char(i - 1);
                let ideographic = unicode::allows_ideographic_breaking(previous);
                let wordless = unicode::is_written_without_spaces(previous)
                    && unicode::is_written_without_spaces(c);
                let mut at_word = || {
                    let starts = word_starts.get_or_insert_with(|| {
                        words.map_or_else(Vec::new, |w| w.word_starts(&self.text))
                    });
                    starts.binary_search(&self.graphemes[i].start).is_ok()
                };
                if breakable(previous)
                    || ideographic
                    || matches!(self.section(i - 1).kind, SectionKind::Image(_))
                    || (i + 1 < self.len() && c == '(')
                    || (wordless && at_word())
                {
                    let penalty = penalty(previous, c, ideographic && zero_width_spaces);
                    let b = evaluate_break(i, x, target_width, &breaks, penalty, false);
                    breaks.push(b);
                }
            }
            if !unicode::is_whitespace(c) {
                x += advance;
            }
        }

        let last = evaluate_break(self.len(), x, target_width, &breaks, 0.0, true);
        let mut indices = vec![last.index];
        let mut prior = last.prior;
        while let Some(p) = prior {
            indices.push(breaks[p].index);
            prior = breaks[p].prior;
        }
        indices.reverse();
        indices
    }
}

/// GL JS `breakable`: characters a line may end on.
fn breakable(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | ' '
            | '&'
            | ')'
            | '+'
            | '-'
            | '/'
            | '\u{ad}'
            | '\u{b7}'
            | '\u{200b}'
            | '\u{2010}'
            | '\u{2013}'
            | '\u{2027}'
    )
}

struct Break {
    index: usize,
    x: f64,
    prior: Option<usize>,
    badness: f64,
}

fn badness(line_width: f64, target_width: f64, penalty: f64, last: bool) -> f64 {
    let d = line_width - target_width;
    let raggedness = d * d;
    if last {
        if line_width < target_width {
            raggedness / 2.0
        } else {
            raggedness * 2.0
        }
    } else {
        raggedness + penalty.abs() * penalty
    }
}

fn penalty(c: char, next: char, penalizable_ideographic: bool) -> f64 {
    let mut penalty = 0.0;
    if c == '\n' || c == '\r' {
        penalty -= 10000.0;
    }
    if penalizable_ideographic {
        penalty += 150.0;
    }
    if c == '(' || c == '\u{ff08}' {
        penalty += 50.0;
    }
    if next == ')' || next == '\u{ff09}' {
        penalty += 50.0;
    }
    penalty
}

fn evaluate_break(
    index: usize,
    x: f64,
    target_width: f64,
    breaks: &[Break],
    penalty: f64,
    last: bool,
) -> Break {
    let mut prior = None;
    let mut best = badness(x, target_width, penalty, last);
    for (i, b) in breaks.iter().enumerate() {
        let candidate = badness(x - b.x, target_width, penalty, last) + b.badness;
        if candidate <= best {
            prior = Some(i);
            best = candidate;
        }
    }
    Break {
        index,
        x,
        prior,
        badness: best,
    }
}

/// JavaScript's `Math.max` of two numbers: NaN if either is, and +0 over
/// -0 (Rust's `max` returns the other for NaN and either zero), for the
/// ports of GL JS that compute what it computes.
pub fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b || (a == b && b.is_sign_negative()) {
        a
    } else {
        b
    }
}

/// JavaScript's `Math.min` of two numbers: NaN if either is, and -0 under
/// +0.
pub fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b || (a == b && a.is_sign_negative()) {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glyph::{Glyph, GlyphKey, GlyphMetrics, GlyphRequest};

    fn stack() -> FontStack {
        FontStack::new(&["Test"])
    }

    fn tagged(text: &str) -> Tagged {
        let n = graphemes(text).len();
        let section = SectionOptions {
            scale: 1.0,
            vertical_align: VerticalAlign::Bottom,
            kind: SectionKind::Text(stack()),
        };
        Tagged::new(text.into(), vec![section], vec![0; n])
    }

    fn no_images(_: &str) -> Option<InlineImage> {
        None
    }

    /// GL JS `tagged_string.test.ts`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn tagged_string() {
        let label = Label {
            sections: vec![
                Section::text("a", Some(1.0), None),
                Section::text("\u{301}", Some(0.5), None),
                Section::text("b", Some(2.0), None),
            ],
        };
        let t = Tagged::from_label(&label, &stack(), false, None);
        assert_eq!(t.clusters(), ["a\u{301}", "b"]);
        assert_eq!(t.section_index, [0, 2]);

        assert_eq!(tagged("茹𦨭").len(), 2);

        let burmese = FontStack::new(&["Burmese"]);
        let other = FontStack::new(&["Other"]);
        let label = Label {
            sections: vec![
                Section::text("လား", Some(1.0), Some(burmese.clone())),
                Section::text("、a\u{301}", Some(2.0), Some(other.clone())),
            ],
        };
        let mut t = Tagged::from_label(&label, &stack(), false, None);
        let mut map = GlyphMap::default();
        let glyph = |stack: &FontStack, cp: u32| Glyph {
            key: GlyphKey::Pbf {
                stack: stack.clone(),
                code_point: cp,
            },
            metrics: GlyphMetrics::default(),
            bitmap: [0, 0],
            double_resolution: false,
        };
        let vertical = |stack: &FontStack, text: &str| GlyphRequest {
            stack: stack.clone(),
            variant: Variant::Vertical,
            text: text.into(),
        };
        map.insert(vertical(&burmese, "လ"), Some(glyph(&burmese, 0x101C)));
        map.insert(vertical(&other, "、"), Some(glyph(&other, 0x3001)));
        t.verticalize_punctuation(Some(&map), None);
        assert_eq!(t.text, "လား、a\u{301}");
        assert_eq!(t.clusters(), ["လ", "ာ", "း", "、", "a\u{301}"]);
        assert_eq!(t.section_index, [0, 0, 0, 1, 1]);

        assert!(
            tagged("三三\u{200b}三三\u{200b}三三\u{200b}三三三三三三\u{200b}三三")
                .text
                .contains('\u{200b}')
        );

        let mut t = tagged("  \t    \u{b} ");
        t.trim();
        assert_eq!(t.text, "");
        assert!(t.section_index.is_empty());
        let mut t = tagged(" 茹𦨭 ");
        t.trim();
        assert_eq!((t.text.as_str(), t.section_index.len()), ("茹𦨭", 2));
        let mut t = tagged("\r\nabc\r\n");
        t.trim();
        assert_eq!(t.text, "abc");
        assert_eq!(t.section_index.len(), t.len());

        let t = tagged("𰻞𰻞麵𪚥𪚥");
        for (start, end, text) in [
            (0, 1, "𰻞"),
            (0, 2, "𰻞𰻞"),
            (1, 2, "𰻞"),
            (1, 3, "𰻞麵"),
            (2, 5, "麵𪚥𪚥"),
        ] {
            let s = t.substring(start, end);
            assert_eq!(s.text, text);
            assert_eq!(s.section_index, vec![0; end - start]);
        }
        let offsets: Vec<usize> = (0..=5).map(|i| t.byte_index(i)).collect();
        assert_eq!(offsets, [0, 4, 8, 11, 15, 19]);
    }

    /// GL JS `tagged_string.test.ts` (`determineLineBreaks`).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn line_breaks() {
        let metrics = GlyphMetrics {
            width: 22.0,
            height: 18.0,
            left: 0.0,
            top: -8.0,
            advance: 22.0,
        };
        let mut map = GlyphMap::default();
        for (c, advance) in [
            ('a', 22.0),
            ('b', 22.0),
            ('c', 22.0),
            (' ', 7.0),
            ('\u{9EB5}', 22.0),
            ('\u{30EDE}', 22.0),
        ] {
            map.insert(
                GlyphRequest {
                    stack: stack(),
                    variant: Variant::Default,
                    text: c.to_string().into(),
                },
                Some(Glyph {
                    key: GlyphKey::Pbf {
                        stack: stack(),
                        code_point: c as u32,
                    },
                    metrics: GlyphMetrics { advance, ..metrics },
                    bitmap: [32, 32],
                    double_resolution: false,
                }),
            );
        }
        let breaks = |text: &str, spacing: f64| {
            tagged(text).line_breaks(spacing, 300.0, &map, &no_images, 30.0, None)
        };
        assert_eq!(breaks("abc", 0.0), [3]);
        assert_eq!(breaks("𰻞𰻞麵", 0.0), [3]);
        assert_eq!(breaks("a b", -14.0), [2, 3]);
    }
}

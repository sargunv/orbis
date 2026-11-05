//! Advance shaping and line layout (GL JS `shaping.ts` `shapeText`).
//!
//! Positions are the same `f64` values GL JS computes; nothing here uses
//! platform maths, so native and wasm agree bit for bit.

use crate::bidi::reorder_lines;
use crate::glyph::{FontStack, GlyphId, GlyphMap, GlyphMetrics, Variant};
use crate::label::{IMAGE_PADDING, Images, Label, SectionKind, Tagged, js_max};
use crate::pbf::BORDER;
use crate::segment::{WordBreaker, graphemes, is_cluster};
use crate::transform::vertical_form;
use crate::unicode;

/// One em in layout units: the size glyphs are laid out at (GL JS `ONE_EM`).
pub const ONE_EM: f64 = 24.0;

/// GL JS `SHAPING_DEFAULT_OFFSET`.
pub const DEFAULT_OFFSET: f64 = -17.0;

/// `text-anchor`, `icon-anchor`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Anchor {
    #[default]
    Center,
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Anchor {
    /// Horizontal and vertical alignment, 0 to 1 (GL JS
    /// `getAnchorAlignment`).
    pub fn alignment(self) -> (f64, f64) {
        use Anchor::*;
        let horizontal = match self {
            Right | TopRight | BottomRight => 1.0,
            Left | TopLeft | BottomLeft => 0.0,
            _ => 0.5,
        };
        let vertical = match self {
            Bottom | BottomRight | BottomLeft => 1.0,
            Top | TopRight | TopLeft => 0.0,
            _ => 0.5,
        };
        (horizontal, vertical)
    }
}

/// `text-justify` (after resolving `auto`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Justify {
    Left,
    #[default]
    Center,
    Right,
}

impl Justify {
    fn factor(self) -> f64 {
        match self {
            Self::Left => 0.0,
            Self::Center => 0.5,
            Self::Right => 1.0,
        }
    }
}

/// The direction lines run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WritingMode {
    #[default]
    Horizontal,
    Vertical,
}

/// `shapeText`'s parameters, in layout units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeOptions {
    /// `text-max-width` × [`ONE_EM`] for point labels; infinite along lines.
    pub max_width: f64,
    /// `text-line-height` × [`ONE_EM`].
    pub line_height: f64,
    pub anchor: Anchor,
    pub justify: Justify,
    /// `text-letter-spacing` × [`ONE_EM`], or 0 for cursive text.
    pub spacing: f64,
    /// `text-offset` or the radial offset, in layout units.
    pub translate: [f64; 2],
    pub writing_mode: WritingMode,
    /// `text-writing-mode` includes `vertical`.
    pub allow_vertical_placement: bool,
    /// `text-size` at the tile's layout zoom + 1.
    pub layout_text_size: f64,
    /// `text-size` at the tile's layout zoom.
    pub layout_text_size_this_zoom: f64,
}

/// What a positioned glyph draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlyphSource {
    Glyph(GlyphId),
    /// The inline image of this label section.
    Image {
        section: usize,
    },
}

/// A glyph placed relative to the label's anchor (GL JS `PositionedGlyph`).
#[derive(Clone, Debug, PartialEq)]
pub struct PositionedGlyph {
    /// The first code point of what is drawn.
    pub code_point: char,
    pub source: GlyphSource,
    pub x: f64,
    pub y: f64,
    /// Upright in a vertical line.
    pub vertical: bool,
    pub scale: f64,
    /// Index of the label section, counting only sections layout kept.
    pub section: usize,
    pub metrics: GlyphMetrics,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PositionedLine {
    pub glyphs: Vec<PositionedGlyph>,
    pub line_offset: f64,
}

/// A shaped label (GL JS `Shaping`).
#[derive(Clone, Debug, PartialEq)]
pub struct Shaping {
    pub lines: Vec<PositionedLine>,
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
    pub writing_mode: WritingMode,
    /// The logical text, with vertical punctuation applied.
    pub text: String,
    pub icons_in_text: bool,
    pub verticalizable: bool,
    /// The label's sections as layout kept them, which
    /// [`PositionedGlyph::section`] indexes.
    pub sections: Vec<ShapedSection>,
}

/// A label section as shaped: its font stack or image name.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapedSection {
    Text(FontStack),
    Image(Box<str>),
}

/// Inputs shared by every label of a tile.
#[derive(Clone, Copy)]
pub struct ShapeContext<'a> {
    pub glyphs: &'a GlyphMap,
    pub images: Images<'a>,
    /// Southeast Asian word breaks, if loaded.
    pub words: Option<&'a WordBreaker>,
}

impl std::fmt::Debug for ShapeContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeContext").finish_non_exhaustive()
    }
}

/// Shapes a label into positioned lines (GL JS `shapeText`). `None` when no
/// glyph is placed.
pub fn shape_text(
    label: &Label,
    default_stack: &FontStack,
    context: ShapeContext<'_>,
    options: &ShapeOptions,
) -> Option<Shaping> {
    let ShapeContext {
        glyphs,
        images,
        words,
    } = context;
    let vertical = options.writing_mode == WritingMode::Vertical;
    let logical = Tagged::from_label(label, default_stack, vertical, Some(glyphs));
    let breaks = logical.line_breaks(
        options.spacing,
        options.max_width,
        glyphs,
        images,
        options.layout_text_size,
        words,
    );
    let lines = if unicode::contains_rtl(&logical.text) {
        bidi_lines(&logical, &breaks)
    } else {
        break_lines(&logical, &breaks)
    };

    let mut shaping = Shaping {
        lines: Vec::new(),
        top: options.translate[1],
        bottom: options.translate[1],
        left: options.translate[0],
        right: options.translate[0],
        writing_mode: options.writing_mode,
        text: logical.text.clone(),
        icons_in_text: false,
        verticalizable: false,
        sections: logical
            .sections
            .iter()
            .map(|s| match &s.kind {
                SectionKind::Text(stack) => ShapedSection::Text(stack.clone()),
                SectionKind::Image(name) => ShapedSection::Image(name.clone()),
            })
            .collect(),
    };
    shape_lines(&mut shaping, glyphs, images, lines, options);
    if shaping.lines.iter().all(|l| l.glyphs.is_empty()) {
        return None;
    }
    Some(shaping)
}

fn break_lines(input: &Tagged, breaks: &[usize]) -> Vec<Tagged> {
    let mut lines = Vec::new();
    let mut start = 0;
    for &b in breaks {
        lines.push(input.substring(start, b));
        start = b;
    }
    if start < input.len() {
        lines.push(input.substring(start, input.len()));
    }
    lines
}

/// Reorders each line visually (GL JS `bidiLines`, without the deprecated
/// plugin): sections follow their characters, and each cluster takes the
/// section of its first character.
fn bidi_lines(logical: &Tagged, breaks: &[usize]) -> Vec<Tagged> {
    let byte_breaks: Vec<usize> = breaks.iter().map(|&i| logical.byte_index(i)).collect();
    let mut section_of_byte = vec![0; logical.text.len()];
    for i in 0..logical.len() {
        section_of_byte[logical.grapheme_range(i)].fill(logical.section_index[i]);
    }
    let single = logical.sections.len() == 1;
    reorder_lines(&logical.text, &byte_breaks)
        .into_iter()
        .map(|line| {
            let char_sections: Vec<usize> = if single {
                vec![0; line.sources.len()]
            } else {
                line.sources
                    .iter()
                    .map(|&b| section_of_byte.get(b).copied().unwrap_or(0))
                    .collect()
            };
            let mut section_index = Vec::new();
            let mut at = 0;
            for r in graphemes(&line.text) {
                section_index.push(char_sections.get(at).copied().unwrap_or(0));
                at += line.text[r].chars().count();
            }
            Tagged::new(line.text, logical.sections.clone(), section_index)
        })
        .collect()
}

/// Whether a glyph may be drawn upright in a vertical line, by its first
/// code point (GL JS `isLineVertical`).
fn is_line_vertical(mode: WritingMode, allow_vertical_placement: bool, c: char) -> bool {
    !(mode == WritingMode::Horizontal
        || (!allow_vertical_placement && !unicode::has_upright_vertical_orientation(c))
        || (allow_vertical_placement && (unicode::is_whitespace(c) || unicode::is_arabic(c))))
}

/// Uppercase runs longer than this are words that read better along the
/// line; shorter ones are codes (“JR”, “A1”).
const MAX_UPRIGHT_LETTER_RUN: usize = 3;

fn run_is_upright(run: &[char]) -> bool {
    let number = run.iter().any(|&c| unicode::is_decimal_digit(c))
        && run
            .iter()
            .all(|&c| unicode::is_decimal_digit(c) || unicode::is_symbol_or_punctuation(c));
    let code = run.len() <= MAX_UPRIGHT_LETTER_RUN
        && run
            .iter()
            .all(|&c| unicode::is_uppercase_letter(c) || unicode::is_decimal_digit(c));
    number || code
}

fn is_upright_in_run(c: char) -> bool {
    (unicode::is_decimal_digit(c) || unicode::is_uppercase_letter(c)) && !unicode::is_arabic(c)
}

/// Whether a cluster starting with `c` may need a vertical glyph (GL JS
/// `mayUseVerticalGlyph`).
pub fn may_use_vertical_glyph(c: char, allow_vertical_placement: bool) -> bool {
    if allow_vertical_placement {
        return is_line_vertical(WritingMode::Vertical, true, c);
    }
    !unicode::has_rotated_vertical_orientation(c)
        || is_upright_in_run(c)
        || unicode::is_symbol_or_punctuation(c)
}

/// Marks punctuation between upright characters upright, substituting its
/// compatibility form where the font has no vertical alternate.
fn verticalize_surrounded_punctuation(
    clusters: &mut [String],
    verticals: &mut [bool],
    vertical_forms: &[bool],
) -> bool {
    let mut replaced = false;
    let n = clusters.len();
    for i in 0..n {
        if verticals[i] {
            continue;
        }
        let mut chars = clusters[i].chars();
        let (Some(c), None) = (chars.next(), chars.next()) else {
            continue;
        };
        let Some(v) = vertical_form(c) else {
            continue;
        };
        if (i == 0 || verticals[i - 1]) && (i == n - 1 || verticals[i + 1]) {
            verticals[i] = true;
            if vertical_forms[i] {
                continue;
            }
            clusters[i] = v.to_string();
            replaced = true;
        }
    }
    replaced
}

fn has_sideways_text_beside(cps: &[char], mut index: isize, step: isize) -> bool {
    let at = |i: isize| (i >= 0 && (i as usize) < cps.len()).then(|| cps[i as usize]);
    while at(index).is_some_and(unicode::is_whitespace) {
        index += step;
    }
    let mut run = Vec::new();
    while let Some(c) = at(index) {
        if unicode::has_upright_vertical_orientation(c) || unicode::is_whitespace(c) {
            break;
        }
        run.push(c);
        index += step;
    }
    !run_is_upright(&run)
        && !run.iter().all(|&c| unicode::is_symbol_or_punctuation(c))
        && run
            .iter()
            .any(|&c| unicode::has_rotated_vertical_orientation(c))
}

/// An upright flag per cluster of a vertical line label, applying vertical
/// punctuation to `line` (GL JS `determineLineVerticals`).
fn determine_line_verticals(line: &mut Tagged, glyphs: &GlyphMap) -> Vec<bool> {
    let mut clusters = line.clusters();
    let cps: Vec<char> = (0..line.len()).map(|i| line.first_char(i)).collect();
    let vertical_forms: Vec<bool> = (0..line.len())
        .map(|i| match &line.section(i).kind {
            SectionKind::Text(stack) => glyphs.has(stack, Variant::Vertical, &clusters[i]),
            SectionKind::Image(_) => false,
        })
        .collect();
    let mut verticals: Vec<bool> = cps
        .iter()
        .map(|&c| unicode::has_upright_vertical_orientation(c))
        .collect();
    let is_image = |i: usize| matches!(line.section(i).kind, SectionKind::Image(_));

    let mut start = 0;
    while start < cps.len() {
        let is_run = |i: usize, verticals: &[bool]| {
            !verticals[i] && !unicode::is_whitespace(cps[i]) && !is_image(i)
        };
        if !is_run(start, &verticals) {
            start += 1;
            continue;
        }
        let mut end = start;
        while end + 1 < cps.len() && is_run(end + 1, &verticals) {
            end += 1;
        }
        let run = &cps[start..=end];
        let upright = run_is_upright(run);
        let punctuation_only = run.iter().all(|&c| unicode::is_symbol_or_punctuation(c));
        let beside_sideways = punctuation_only
            && (has_sideways_text_beside(&cps, start as isize - 1, -1)
                || has_sideways_text_beside(&cps, end as isize + 1, 1));
        let sideways = !upright
            && run
                .iter()
                .any(|&c| unicode::has_rotated_vertical_orientation(c));
        for i in start..=end {
            verticals[i] = (upright && is_upright_in_run(cps[i]))
                || (vertical_forms[i]
                    && !beside_sideways
                    && (punctuation_only
                        || (!sideways && !unicode::has_rotated_vertical_orientation(cps[i]))));
        }
        start = end + 1;
    }

    if vertical_forms
        .iter()
        .zip(&verticals)
        .any(|(&form, &v)| form && !v)
    {
        let previous = line.text.clone();
        line.verticalize_punctuation(Some(glyphs), Some(&verticals));
        if line.text != previous {
            return determine_line_verticals(line, glyphs);
        }
    }

    if verticalize_surrounded_punctuation(&mut clusters, &mut verticals, &vertical_forms) {
        line.set_graphemes(&clusters);
    }
    verticals
}

struct SectionAttributes {
    metrics: GlyphMetrics,
    baseline_offset: f64,
}

fn shape_lines(
    shaping: &mut Shaping,
    glyphs: &GlyphMap,
    images: Images<'_>,
    mut lines: Vec<Tagged>,
    options: &ShapeOptions,
) {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut max_line_length: f64 = 0.0;
    let mut max_line_height: f64 = 0.0;
    let justify = options.justify.factor();
    let size_factor = ONE_EM / options.layout_text_size_this_zoom;
    let spacing = options.spacing;
    let line_count = lines.len();

    for line in &mut lines {
        line.trim();
        let line_max_scale = line.max_scale();
        let mut positioned = PositionedLine::default();
        let mut image_offset = 0.0;

        if line.len() == 0 {
            y += options.line_height;
            shaping.lines.push(positioned);
            continue;
        }

        let max_glyph_size = line_max_scale * ONE_EM;
        let [max_image_width, max_image_height] = line.max_image_size(images);
        let content_height = js_max(max_glyph_size, max_image_height * size_factor);
        let content_width = js_max(max_glyph_size, max_image_width * size_factor);

        let verticals = (options.writing_mode == WritingMode::Vertical
            && !options.allow_vertical_placement)
            .then(|| determine_line_verticals(line, glyphs));

        for i in 0..line.len() {
            let grapheme = line.grapheme(i).to_string();
            let first = line.first_char(i);
            let vertical = verticals.as_ref().map_or_else(
                || {
                    is_line_vertical(
                        options.writing_mode,
                        options.allow_vertical_placement,
                        first,
                    )
                },
                |v| v[i],
            );
            let section_index = line.section_index[i];
            let section = &line.sections[section_index];
            let keys: Vec<String> = match &section.kind {
                SectionKind::Text(stack)
                    if is_cluster(&grapheme) && !glyphs.has(stack, Variant::Default, &grapheme) =>
                {
                    grapheme.chars().map(String::from).collect()
                }
                _ => vec![grapheme],
            };

            for key in keys {
                let section = &mut line.sections[section_index];
                let (source, attributes) = match &section.kind {
                    SectionKind::Text(stack) => {
                        let variant = if vertical && glyphs.has(stack, Variant::Vertical, &key) {
                            Variant::Vertical
                        } else {
                            Variant::Default
                        };
                        let Some((id, glyph)) = glyphs.get(stack, variant, &key) else {
                            continue;
                        };
                        let baseline_offset = if vertical {
                            content_width - section.scale * ONE_EM
                        } else {
                            (content_height - section.scale * ONE_EM)
                                * section.vertical_align.factor()
                        };
                        let attributes = SectionAttributes {
                            metrics: glyph.metrics,
                            baseline_offset,
                        };
                        (GlyphSource::Glyph(id), attributes)
                    }
                    SectionKind::Image(name) => {
                        shaping.icons_in_text = true;
                        section.scale *= size_factor;
                        let Some(image) = images(name) else {
                            continue;
                        };
                        let [w, h] = image.display_size();
                        let metrics = GlyphMetrics {
                            width: w,
                            height: h,
                            left: IMAGE_PADDING,
                            top: -BORDER,
                            advance: if vertical { h } else { w },
                        };
                        let baseline_offset = if vertical {
                            content_width - h * section.scale
                        } else {
                            (content_height - h * section.scale) * section.vertical_align.factor()
                        };
                        let offset = (if vertical { w } else { h }) * section.scale
                            - ONE_EM * line_max_scale;
                        let attributes = SectionAttributes {
                            metrics,
                            baseline_offset,
                        };
                        image_offset = js_max(image_offset, offset);
                        (
                            GlyphSource::Image {
                                section: section_index,
                            },
                            attributes,
                        )
                    }
                };
                let is_image = matches!(source, GlyphSource::Image { .. });
                let glyph = PositionedGlyph {
                    code_point: key.chars().next().unwrap_or('\0'),
                    source,
                    x,
                    y: y + DEFAULT_OFFSET + attributes.baseline_offset,
                    vertical,
                    scale: section.scale,
                    section: section_index,
                    metrics: attributes.metrics,
                };
                if !vertical {
                    x += glyph.metrics.advance * section.scale + spacing;
                } else {
                    shaping.verticalizable = true;
                    let advance = if is_image {
                        glyph.metrics.advance
                    } else {
                        ONE_EM
                    };
                    x += advance * section.scale + spacing;
                }
                positioned.glyphs.push(glyph);
            }
        }

        if !positioned.glyphs.is_empty() {
            let line_length = x - spacing;
            max_line_length = js_max(line_length, max_line_length);
            justify_line(&mut positioned.glyphs, justify);
        }

        x = 0.0;
        let max_line_offset = (line_max_scale - 1.0) * ONE_EM;
        positioned.line_offset = js_max(image_offset, max_line_offset);
        let line_height = options.line_height * line_max_scale + image_offset;
        y += line_height;
        max_line_height = js_max(line_height, max_line_height);
        shaping.lines.push(positioned);
    }

    let (horizontal_align, vertical_align) = options.anchor.alignment();
    align(
        &mut shaping.lines,
        justify,
        horizontal_align,
        vertical_align,
        max_line_length,
        max_line_height,
        options.line_height,
        y,
        line_count,
    );
    shaping.top += -vertical_align * y;
    shaping.bottom = shaping.top + y;
    shaping.left += -horizontal_align * max_line_length;
    shaping.right = shaping.left + max_line_length;
}

fn justify_line(glyphs: &mut [PositionedGlyph], justify: f64) {
    if justify == 0.0 {
        return;
    }
    let last = glyphs.last().expect("line has glyphs");
    let indent = (last.x + last.metrics.advance * last.scale) * justify;
    for glyph in glyphs {
        glyph.x -= indent;
    }
}

#[allow(clippy::too_many_arguments)]
fn align(
    lines: &mut [PositionedLine],
    justify: f64,
    horizontal_align: f64,
    vertical_align: f64,
    max_line_length: f64,
    max_line_height: f64,
    line_height: f64,
    block_height: f64,
    line_count: usize,
) {
    let shift_x = (justify - horizontal_align) * max_line_length;
    let shift_y = if max_line_height != line_height {
        -block_height * vertical_align - DEFAULT_OFFSET
    } else {
        -vertical_align * line_count as f64 * line_height + 0.5 * line_height
    };
    for glyph in lines.iter_mut().flat_map(|l| &mut l.glyphs) {
        glyph.x += shift_x;
        glyph.y += shift_y;
    }
}

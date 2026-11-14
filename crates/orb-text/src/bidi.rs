//! Visual reordering of bidirectional text (GL JS `bidi.ts`).
//!
//! `unicode-bidi` resolves embedding levels and paragraphs (rules P–I, GL JS
//! uses `bidi-js`); the line rules are GL JS's: L1 per line, L2 over whole
//! grapheme clusters (which is how GL JS satisfies L3), mirroring of single
//! characters at odd levels, and bidi controls dropped once they've acted.

use std::ops::Range;
use std::sync::OnceLock;

use unicode_bidi::{BidiClass, BidiDataSource, BidiInfo};

use crate::segment::graphemes;
use crate::unicode;

/// `unicode-bidi`'s classes in the order `cargo xtask unicode` numbers
/// them (`xtask/src/unicode.rs`'s `BIDI_CLASSES`).
const BIDI_CLASSES: [BidiClass; 23] = {
    use BidiClass::*;
    [
        AL, AN, B, BN, CS, EN, ES, ET, FSI, L, LRE, LRI, LRO, NSM, ON, PDF, PDI, R, RLE, RLI, RLO,
        S, WS,
    ]
};

/// The algorithm's tables, [`unicode::BIDI_DATA`] unpacked: private to
/// this module, all of whose code the web's split moves into
/// core-bidi.wasm (xtask's `split.rs`), and whose loader hands the module's
/// copy of the data over ([`set_data`]) before that code can run. So no
/// code reads them before they are in, whatever calls into this module.
struct Tables {
    /// `Bidi_Mirroring_Glyph` pairs.
    mirror: Vec<(u32, u32)>,
    /// `Bidi_Class` runs, and each run's index in [`BIDI_CLASSES`].
    runs: Vec<(u32, u32)>,
    classes: Box<[u8]>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

impl Tables {
    fn unpack(data: &[u8]) -> Self {
        assert_eq!(data.len(), unicode::BIDI_DATA.len(), "this build's data");
        let (mirror, rest) = data.split_at(unicode::MIRROR_LEN);
        let (runs, classes) = rest.split_at(unicode::BIDI_SPANS_LEN);
        Self {
            mirror: unicode::unpack(mirror),
            runs: unicode::unpack(runs),
            classes: classes.into(),
        }
    }
}

/// Gives the algorithm its tables, [`unicode::BIDI_DATA`]: on the web,
/// orb-web's `setModuleData` as core-bidi.wasm comes in, which stays in
/// core.wasm (`split.rs`'s `except`). Native builds have them anyway. Only
/// the first call counts.
pub fn set_data(data: &[u8]) {
    TABLES.get_or_init(|| Tables::unpack(data));
}

fn tables() -> &'static Tables {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    return TABLES.get_or_init(|| Tables::unpack(&unicode::BIDI_DATA));
    #[cfg(all(not(test), target_arch = "wasm32"))]
    TABLES
        .get()
        .expect("core-bidi.wasm's data comes in before its code runs")
}

/// `Bidi_Class` from the UCD the other character tables come from
/// ([`Tables`]), rather than `unicode-bidi`'s own table.
struct Classes;

impl BidiDataSource for Classes {
    fn bidi_class(&self, c: char) -> BidiClass {
        let tables = tables();
        unicode::find(&tables.runs, c).map_or(BidiClass::L, |i| {
            BIDI_CLASSES[usize::from(tables.classes[i])]
        })
    }
}

/// The `Bidi_Mirroring_Glyph` of `c`.
fn mirror(c: char) -> Option<char> {
    let mirror = &tables().mirror;
    (mirror.binary_search_by_key(&(c as u32), |&(from, _)| from))
        .ok()
        .and_then(|i| char::from_u32(mirror[i].1))
}

/// A line of a label in visual order.
#[derive(Debug, PartialEq)]
pub(crate) struct ReorderedLine {
    pub(crate) text: String,
    /// For each character of `text`, the byte offset in the logical text of
    /// the cluster it came from.
    pub(crate) sources: Vec<usize>,
}

struct Cluster {
    range: Range<usize>,
    level: u8,
}

/// Breaks `text` at `breaks` (byte offsets) and at every paragraph start,
/// and puts each line into visual order.
pub(crate) fn reorder_lines(text: &str, breaks: &[usize]) -> Vec<ReorderedLine> {
    let info = BidiInfo::new_with_data_source(&Classes, text, None);
    let mut bounds: Vec<usize> = breaks
        .iter()
        .copied()
        .chain(info.paragraphs.iter().map(|p| p.range.start))
        .filter(|&b| b > 0 && b < text.len())
        .collect();
    bounds.push(0);
    bounds.push(text.len());
    bounds.sort_unstable();
    bounds.dedup();
    bounds
        .windows(2)
        .map(|w| {
            let paragraph_level = info
                .paragraphs
                .iter()
                .find(|p| p.range.start <= w[0] && w[0] < p.range.end)
                .map_or(0, |p| p.level.number());
            reorder_line(text, &info, paragraph_level, w[0]..w[1])
        })
        .collect()
}

fn reorder_line(
    text: &str,
    info: &BidiInfo<'_>,
    paragraph_level: u8,
    line: Range<usize>,
) -> ReorderedLine {
    let mut clusters: Vec<Cluster> = graphemes(&text[line.clone()])
        .into_iter()
        .map(|r| {
            let range = r.start + line.start..r.end + line.start;
            let level = info.levels[range.start].number();
            Cluster { range, level }
        })
        .collect();
    reset_trailing_neutrals(text, &mut clusters, paragraph_level);
    reorder(&mut clusters, paragraph_level);

    let mut out = ReorderedLine {
        text: String::with_capacity(line.len()),
        sources: Vec::new(),
    };
    for cluster in &clusters {
        let s = &text[cluster.range.clone()];
        if s.chars().any(unicode::is_bidi_control) {
            continue;
        }
        let mirrored = (cluster.level % 2 == 1)
            .then(|| single_char(s).and_then(mirror))
            .flatten();
        match mirrored {
            Some(m) => {
                out.text.push(m);
                out.sources.push(cluster.range.start);
            }
            None => {
                out.text.push_str(s);
                out.sources.extend(s.chars().map(|_| cluster.range.start));
            }
        }
    }
    out
}

fn single_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    let c = chars.next()?;
    chars.next().is_none().then_some(c)
}

/// Rule L1: whitespace and isolates at the end of a line, and segment or
/// paragraph separators with the whitespace before them, take the
/// paragraph's level.
fn reset_trailing_neutrals(text: &str, clusters: &mut [Cluster], paragraph_level: u8) {
    let mut trailing = true;
    for cluster in clusters.iter_mut().rev() {
        let first = text[cluster.range.clone()].chars().next();
        let class = first.map(|c| Classes.bidi_class(c));
        if matches!(class, Some(BidiClass::S | BidiClass::B)) {
            cluster.level = paragraph_level;
            trailing = true;
        } else if trailing
            && matches!(
                class,
                Some(
                    BidiClass::WS
                        | BidiClass::FSI
                        | BidiClass::LRI
                        | BidiClass::RLI
                        | BidiClass::PDI
                )
            )
        {
            cluster.level = paragraph_level;
        } else {
            trailing = false;
        }
    }
}

/// Rule L2: from the highest level down to the lowest odd one, reverse every
/// run at that level or above.
fn reorder(clusters: &mut [Cluster], paragraph_level: u8) {
    let mut highest = paragraph_level;
    let mut lowest_odd = u8::MAX;
    for c in clusters.iter() {
        highest = highest.max(c.level);
        lowest_odd = lowest_odd.min(c.level | 1);
    }
    let mut level = highest;
    while level >= lowest_odd {
        let mut start = 0;
        while start < clusters.len() {
            if clusters[start].level < level {
                start += 1;
                continue;
            }
            let mut end = start;
            while end + 1 < clusters.len() && clusters[end + 1].level >= level {
                end += 1;
            }
            clusters[start..=end].reverse();
            start = end + 1;
        }
        if level == 0 {
            break;
        }
        level -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str, breaks_in_chars: &[usize]) -> Vec<String> {
        let breaks: Vec<usize> = breaks_in_chars
            .iter()
            .map(|&n| text.char_indices().nth(n).map_or(text.len(), |(i, _)| i))
            .collect();
        reorder_lines(text, &breaks)
            .into_iter()
            .map(|l| l.text)
            .collect()
    }

    /// GL JS `bidi.test.ts` (`processBidirectionalText`).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn process_bidirectional_text() {
        assert_eq!(lines("hello world", &[]), ["hello world"]);
        assert_eq!(lines("שלום", &[]), ["םולש"]);
        assert_eq!(lines("hello שלום world", &[]), ["hello םולש world"]);
        assert_eq!(lines("שלום hello עולם", &[]), ["םלוע hello םולש"]);
        assert_eq!(lines("שלום 123", &[]), ["123 םולש"]);
        assert_eq!(lines("שלום (עולם)", &[]), ["(םלוע) םולש"]);
        assert_eq!(lines("שלום עולם טוב", &[5, 10]), [" םולש", " םלוע", "בוט"]);
        assert_eq!(lines("שלום ", &[]), [" םולש"]);
        assert_eq!(lines("hello\u{200f}world", &[]), ["helloworld"]);
        assert_eq!(lines("שלום 😀 עולם", &[]), ["םלוע 😀 םולש"]);
    }

    /// GL JS `bidi.test.ts` (`processStyledBidirectionalText`): each
    /// reordered character keeps where it came from.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn sources() {
        let text = "ab שלום";
        let [line] = &reorder_lines(text, &[])[..] else {
            panic!()
        };
        assert_eq!(line.text, "ab םולש");
        let chars: Vec<usize> = line
            .sources
            .iter()
            .map(|&b| text[..b].chars().count())
            .collect();
        assert_eq!(chars, [0, 1, 2, 6, 5, 4, 3]);

        let text = "ab\u{200f}cd";
        let [line] = &reorder_lines(text, &[])[..] else {
            panic!()
        };
        assert_eq!(line.text, "abcd");
        let chars: Vec<usize> = line
            .sources
            .iter()
            .map(|&b| text[..b].chars().count())
            .collect();
        assert_eq!(chars, [0, 1, 3, 4]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn keeps_marks_with_their_letters() {
        let text = "שְׁדֵרוֹת";
        assert_eq!(lines(text, &[]), ["תוֹרדֵשְׁ"]);
    }
}

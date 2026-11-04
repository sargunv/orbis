//! Visual reordering of bidirectional text (GL JS `bidi.ts`).
//!
//! `unicode-bidi` resolves embedding levels and paragraphs (rules P–I, GL JS
//! uses `bidi-js`); the line rules are GL JS's: L1 per line, L2 over whole
//! grapheme clusters (which is how GL JS satisfies L3), mirroring of single
//! characters at odd levels, and bidi controls dropped once they've acted.

use std::ops::Range;

use unicode_bidi::{BidiClass, BidiInfo, bidi_class};

use crate::segment::graphemes;
use crate::unicode;

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
    let info = BidiInfo::new(text, None);
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
            .then(|| single_char(s).and_then(unicode::mirror))
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
        let class = first.map(bidi_class);
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

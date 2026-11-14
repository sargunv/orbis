//! Character properties text layout asks about, as GL JS asks them
//! (`script_detection.ts` and the generated `unicode_properties.g.ts` of the
//! pinned GL JS). The tables come from UCD 17.0.0.

use std::sync::OnceLock;

/// Pairs of code points (ranges, or a character and its mirror), packed
/// as the build compiles them: per pair a varint of the first's distance
/// from the pair before's first, then one of the second's zigzagged
/// distance from the first. A lookup unpacks them the first time; packed,
/// they are a quarter of their size as `u32` pairs, and the wasm's data
/// is that much smaller.
pub(crate) struct Pairs {
    packed: &'static [u8],
    unpacked: OnceLock<Vec<(u32, u32)>>,
}

/// The pairs given, packed at compile time ([`Pairs`]).
macro_rules! pairs {
    ($($pairs:tt)*) => {{
        const PAIRS: &[(u32, u32)] = &[$($pairs)*];
        const PACKED: [u8; super::packed_len(PAIRS)] = super::pack(PAIRS);
        super::Pairs {
            packed: &PACKED,
            unpacked: std::sync::OnceLock::new(),
        }
    }};
}

/// Runs of code points with a value (a `Bidi_Class`): the runs packed as
/// [`Pairs`], their values a byte apiece.
pub(crate) struct Classes {
    runs: Pairs,
    values: &'static [u8],
}

/// The runs given, packed at compile time ([`Classes`]).
macro_rules! classes {
    ($($runs:tt)*) => {{
        const RUNS: &[(u32, u32, u8)] = &[$($runs)*];
        const SPANS: [(u32, u32); RUNS.len()] = super::spans(RUNS);
        const PACKED: [u8; super::packed_len(&SPANS)] = super::pack(&SPANS);
        const VALUES: [u8; RUNS.len()] = super::values(RUNS);
        super::Classes {
            runs: super::Pairs {
                packed: &PACKED,
                unpacked: std::sync::OnceLock::new(),
            },
            values: &VALUES,
        }
    }};
}

#[rustfmt::skip]
mod tables;

use tables::*;

/// Pair `i`'s two varints' values ([`Pairs`]).
#[allow(clippy::cast_lossless)]
const fn deltas(pairs: &[(u32, u32)], i: usize) -> (u64, u64) {
    let (a, b) = pairs[i];
    let before = if i == 0 { 0 } else { pairs[i - 1].0 };
    assert!(a >= before, "pairs are in order of their first code point");
    let d = b as i64 - a as i64;
    ((a - before) as u64, ((d << 1) ^ (d >> 63)) as u64)
}

const fn varint_len(mut v: u64) -> usize {
    let mut len = 1;
    while v >= 0x80 {
        v >>= 7;
        len += 1;
    }
    len
}

const fn packed_len(pairs: &[(u32, u32)]) -> usize {
    let mut len = 0;
    let mut i = 0;
    while i < pairs.len() {
        let (a, b) = deltas(pairs, i);
        len += varint_len(a) + varint_len(b);
        i += 1;
    }
    len
}

const fn pack<const N: usize>(pairs: &[(u32, u32)]) -> [u8; N] {
    let mut out = [0; N];
    let mut at = 0;
    let mut i = 0;
    while i < pairs.len() {
        let (a, b) = deltas(pairs, i);
        let values = [a, b];
        let mut k = 0;
        while k < 2 {
            let mut v = values[k];
            while v >= 0x80 {
                out[at] = (v as u8) | 0x80;
                v >>= 7;
                at += 1;
            }
            out[at] = v as u8;
            at += 1;
            k += 1;
        }
        i += 1;
    }
    out
}

const fn spans<const N: usize>(runs: &[(u32, u32, u8)]) -> [(u32, u32); N] {
    let mut out = [(0, 0); N];
    let mut i = 0;
    while i < N {
        out[i] = (runs[i].0, runs[i].1);
        i += 1;
    }
    out
}

const fn values<const N: usize>(runs: &[(u32, u32, u8)]) -> [u8; N] {
    let mut out = [0; N];
    let mut i = 0;
    while i < N {
        out[i] = runs[i].2;
        i += 1;
    }
    out
}

impl Pairs {
    fn get(&self) -> &[(u32, u32)] {
        self.unpacked.get_or_init(|| {
            let mut bytes = self.packed.iter();
            let mut varint = || {
                let mut value = 0u64;
                let mut shift = 0;
                for &byte in bytes.by_ref() {
                    value |= u64::from(byte & 0x7f) << shift;
                    if byte < 0x80 {
                        return Some(value);
                    }
                    shift += 7;
                }
                None
            };
            let mut pairs = Vec::new();
            let mut first = 0u32;
            while let (Some(a), Some(b)) = (varint(), varint()) {
                first += a as u32;
                let d = (b >> 1) as i64 ^ -((b & 1) as i64);
                pairs.push((first, (i64::from(first) + d) as u32));
            }
            pairs
        })
    }
}

/// The index of the range of `ranges` that holds `c`, if one does.
fn find(ranges: &Pairs, c: char) -> Option<usize> {
    let cp = c as u32;
    ranges
        .get()
        .binary_search_by(|&(start, end)| {
            if end < cp {
                std::cmp::Ordering::Less
            } else if start > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .ok()
}

fn contains(ranges: &Pairs, c: char) -> bool {
    find(ranges, c).is_some()
}

/// The `Bidi_Class` of `c`, by its index in `bidi`'s `BIDI_CLASSES`;
/// `None` for `L`.
pub(crate) fn bidi_class(c: char) -> Option<u8> {
    find(&BIDI_CLASS.runs, c).map(|i| BIDI_CLASS.values[i])
}

/// JavaScript's `\s`.
pub fn is_whitespace(c: char) -> bool {
    contains(&JS_WHITESPACE, c)
}

/// `\p{P}` or `\p{S}`.
pub(crate) fn is_symbol_or_punctuation(c: char) -> bool {
    contains(&PUNCTUATION_OR_SYMBOL, c)
}

/// `\p{Nd}`: a decimal digit of any script.
pub(crate) fn is_decimal_digit(c: char) -> bool {
    contains(&DECIMAL_DIGIT, c)
}

/// `\p{Lu}`: an uppercase letter of any script.
pub(crate) fn is_uppercase_letter(c: char) -> bool {
    contains(&UPPERCASE_LETTER, c)
}

/// `\p{Cf}`: an invisible formatting character.
pub(crate) fn is_format(c: char) -> bool {
    contains(&FORMAT, c)
}

/// In a script written right to left. Like GL JS, this includes every code
/// point without a script (unassigned and private use).
pub(crate) fn is_rtl(c: char) -> bool {
    contains(&RTL_SCRIPT, c)
}

/// In a script whose letters join, so they can't be spaced apart.
pub(crate) fn is_cursive(c: char) -> bool {
    contains(&CURSIVE_SCRIPT, c)
}

/// GL JS `charInComplexShapingScript`: in the Arabic script. Vertical text
/// never sets these upright.
pub(crate) fn is_arabic(c: char) -> bool {
    contains(&ARABIC_SCRIPT, c)
}

/// In a block of a script whose letters combine into clusters (Devanagari,
/// Thai, Khmer, …). GL JS draws these locally rather than from glyph PBFs.
pub fn is_complex_shaping(c: char) -> bool {
    contains(&COMPLEX_SHAPING_BLOCK, c)
}

/// Drawn with `localIdeographFontFamily` when a map sets one.
pub fn uses_local_ideograph_font(c: char) -> bool {
    contains(&LOCAL_IDEOGRAPH, c)
}

/// A line may break after this character.
pub(crate) fn allows_ideographic_breaking(c: char) -> bool {
    contains(&IDEOGRAPHIC_BREAKING, c)
}

/// Drawn upright in vertical text in its own right.
pub fn has_upright_vertical_orientation(c: char) -> bool {
    contains(&UPRIGHT_VERTICAL, c)
}

/// Drawn upright in vertical text only beside upright characters.
pub(crate) fn has_neutral_vertical_orientation(c: char) -> bool {
    contains(&NEUTRAL_VERTICAL, c)
}

/// Drawn rotated in vertical text, and rotates neutral neighbours with it.
pub(crate) fn has_rotated_vertical_orientation(c: char) -> bool {
    !(has_upright_vertical_orientation(c) || has_neutral_vertical_orientation(c))
}

/// In a script that doesn't space its words, so lines break only where a
/// word segmenter finds words.
pub fn is_written_without_spaces(c: char) -> bool {
    contains(&WRITTEN_WITHOUT_SPACES, c)
}

/// Can take part in a grapheme cluster of several code points.
pub(crate) fn can_form_cluster(c: char) -> bool {
    contains(&GRAPHEME_CAPABLE, c)
}

/// Joins the cluster it ends to the next one (a virama or ZWJ).
pub(crate) fn joins_following(c: char) -> bool {
    contains(&JOINS_FOLLOWING, c)
}

/// `\p{Mc}`: a spacing mark.
pub(crate) fn is_spacing_mark(c: char) -> bool {
    contains(&SPACING_MARK, c)
}

/// A character that steers bidi without being drawn.
pub(crate) fn is_bidi_control(c: char) -> bool {
    contains(&BIDI_CONTROL, c)
}

/// The `Bidi_Mirroring_Glyph` of `c`.
pub(crate) fn mirror(c: char) -> Option<char> {
    let cp = c as u32;
    let mirror = MIRROR.get();
    mirror
        .binary_search_by_key(&cp, |&(from, _)| from)
        .ok()
        .and_then(|i| char::from_u32(mirror[i].1))
}

/// Arabic joining type: `R L D C T`, or `U` for anything else.
pub(crate) fn joining_type(c: char) -> u8 {
    let cp = c as u32;
    match JOINING_TYPE.binary_search_by(|&(start, end, _)| {
        if end < cp {
            std::cmp::Ordering::Less
        } else if start > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => JOINING_TYPE[i].2,
        Err(_) => b'U',
    }
}

/// The presentation forms of an Arabic letter: `[isolated, final, initial,
/// medial]`, 0 where the letter has no such form.
pub(crate) fn presentation_forms(c: char) -> Option<[u32; 4]> {
    let cp = c as u32;
    PRESENTATION_FORMS
        .binary_search_by_key(&cp, |&(base, _)| base)
        .ok()
        .map(|i| PRESENTATION_FORMS[i].1)
}

/// The lam-alef ligature for lam followed by `alef`: `[isolated, final]`.
pub(crate) fn lam_alef(alef: char) -> Option<[u32; 2]> {
    let cp = alef as u32;
    LAM_ALEF
        .binary_search_by_key(&cp, |&(a, _)| a)
        .ok()
        .map(|i| LAM_ALEF[i].1)
}

/// Whether every character allows letter spacing (no cursive script).
pub fn allows_letter_spacing(text: &str) -> bool {
    text.chars().all(|c| !is_cursive(c))
}

/// Whether any character is upright in vertical text, so the text may be
/// written vertically.
pub fn allows_vertical_writing_mode(text: &str) -> bool {
    text.chars().any(has_upright_vertical_orientation)
}

/// Whether any character is in a right-to-left script.
pub(crate) fn contains_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_pairs_unpack_as_written() {
        let written = [
            (0x29, 0x28),
            (0x41, 0x5A),
            (0x41, 0x5A),
            (0x10FFFD, 0x10FFFF),
        ];
        let pairs: Pairs = pairs![
            (0x29, 0x28),
            (0x41, 0x5A),
            (0x41, 0x5A),
            (0x10FFFD, 0x10FFFF)
        ];
        assert_eq!(pairs.get(), written);
    }

    fn c(cp: u32) -> char {
        char::from_u32(cp).unwrap()
    }

    /// GL JS `script_detection.test.ts`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn script_detection() {
        for w in [' ', '\t', '\u{b}', '\u{c}', '\r', '\n'] {
            assert!(is_whitespace(w), "{w:?}");
        }
        let letter_spacing = |ch: char| !is_cursive(ch);
        assert!(letter_spacing('A'));
        for ch in ['۳', 'ݣ', 'ࢳ', '࢐', 'ﰤ', 'ﺽ'] {
            assert!(!letter_spacing(ch), "{ch}");
            assert!(is_arabic(ch), "{ch}");
            assert!(is_rtl(ch), "{ch}");
        }
        for cp in [
            0x1bc00, 0x10f70, 0x1820, 0x0710, 0x1e900, 0x07ca, 0x0840, 0x10d00,
        ] {
            assert!(!letter_spacing(c(cp)), "{cp:x}");
        }
        for ch in ['ה', 'ޘ', 'द'] {
            assert!(letter_spacing(ch), "{ch}");
        }
        assert!(!is_arabic('3'));
        assert!(!is_rtl('3'));
        for ch in ['ה', 'ﬡ', 'ޘ', c(0x10d40), c(0x10940), c(0x10f70)] {
            assert!(is_rtl(ch), "{ch}");
        }
        for ch in [c(0x105c0), c(0x13000), 'a', 'Я', '三'] {
            assert!(!is_rtl(ch), "{ch}");
        }
        // The GL JS quirk reproduced: private use counts as RTL.
        assert!(is_rtl('\u{E000}'));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn derived_properties() {
        assert!(allows_ideographic_breaking('三') && !allows_ideographic_breaking('a'));
        assert!(has_upright_vertical_orientation('三') && has_upright_vertical_orientation('ア'));
        assert!(!has_upright_vertical_orientation('ー') && !has_upright_vertical_orientation('a'));
        assert!(has_neutral_vertical_orientation('½') && has_rotated_vertical_orientation('a'));
        assert!(is_complex_shaping('क') && is_complex_shaping('ก') && !is_complex_shaping('a'));
        assert!(is_written_without_spaces('ก') && !is_written_without_spaces('三'));
        assert!(is_whitespace('\u{3000}') && is_whitespace('\u{FEFF}') && !is_whitespace('\u{85}'));
        assert_eq!(mirror('('), Some(')'));
        assert_eq!(mirror('a'), None);
        assert_eq!(
            (joining_type('ب'), joining_type('ا'), joining_type('a')),
            (b'D', b'R', b'U')
        );
    }
}

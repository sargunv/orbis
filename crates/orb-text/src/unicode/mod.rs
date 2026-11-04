//! Character properties text layout asks about, as GL JS asks them
//! (`script_detection.ts` and the generated `unicode_properties.g.ts` of the
//! pinned GL JS). The tables come from UCD 17.0.0 (decision 0009).

#[rustfmt::skip]
mod tables;

use tables::*;

fn contains(ranges: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    ranges
        .binary_search_by(|&(start, end)| {
            if end < cp {
                std::cmp::Ordering::Less
            } else if start > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// JavaScript's `\s`.
pub fn is_whitespace(c: char) -> bool {
    contains(JS_WHITESPACE, c)
}

/// `\p{P}` or `\p{S}`.
pub fn is_symbol_or_punctuation(c: char) -> bool {
    contains(PUNCTUATION_OR_SYMBOL, c)
}

/// `\p{Nd}`: a decimal digit of any script.
pub fn is_decimal_digit(c: char) -> bool {
    contains(DECIMAL_DIGIT, c)
}

/// `\p{Lu}`: an uppercase letter of any script.
pub fn is_uppercase_letter(c: char) -> bool {
    contains(UPPERCASE_LETTER, c)
}

/// `\p{Cf}`: an invisible formatting character.
pub fn is_format(c: char) -> bool {
    contains(FORMAT, c)
}

/// In a script written right to left. Like GL JS, this includes every code
/// point without a script (unassigned and private use; decision 0009).
pub fn is_rtl(c: char) -> bool {
    contains(RTL_SCRIPT, c)
}

/// In a script whose letters join, so they can't be spaced apart.
pub fn is_cursive(c: char) -> bool {
    contains(CURSIVE_SCRIPT, c)
}

/// GL JS `charInComplexShapingScript`: in the Arabic script. Vertical text
/// never sets these upright.
pub fn is_arabic(c: char) -> bool {
    contains(ARABIC_SCRIPT, c)
}

/// In a block of a script whose letters combine into clusters (Devanagari,
/// Thai, Khmer, …). GL JS draws these locally rather than from glyph PBFs.
pub fn is_complex_shaping(c: char) -> bool {
    contains(COMPLEX_SHAPING_BLOCK, c)
}

/// Drawn with `localIdeographFontFamily` when a map sets one.
pub fn uses_local_ideograph_font(c: char) -> bool {
    contains(LOCAL_IDEOGRAPH, c)
}

/// A line may break after this character.
pub fn allows_ideographic_breaking(c: char) -> bool {
    contains(IDEOGRAPHIC_BREAKING, c)
}

/// Drawn upright in vertical text in its own right.
pub fn has_upright_vertical_orientation(c: char) -> bool {
    contains(UPRIGHT_VERTICAL, c)
}

/// Drawn upright in vertical text only beside upright characters.
pub fn has_neutral_vertical_orientation(c: char) -> bool {
    contains(NEUTRAL_VERTICAL, c)
}

/// Drawn rotated in vertical text, and rotates neutral neighbours with it.
pub fn has_rotated_vertical_orientation(c: char) -> bool {
    !(has_upright_vertical_orientation(c) || has_neutral_vertical_orientation(c))
}

/// In a script that doesn't space its words, so lines break only where a
/// word segmenter finds words.
pub fn is_written_without_spaces(c: char) -> bool {
    contains(WRITTEN_WITHOUT_SPACES, c)
}

/// Can take part in a grapheme cluster of several code points.
pub(crate) fn can_form_cluster(c: char) -> bool {
    contains(GRAPHEME_CAPABLE, c)
}

/// Joins the cluster it ends to the next one (a virama or ZWJ).
pub(crate) fn joins_following(c: char) -> bool {
    contains(JOINS_FOLLOWING, c)
}

/// `\p{Mc}`: a spacing mark.
pub(crate) fn is_spacing_mark(c: char) -> bool {
    contains(SPACING_MARK, c)
}

/// A character that steers bidi without being drawn.
pub(crate) fn is_bidi_control(c: char) -> bool {
    contains(BIDI_CONTROL, c)
}

/// The `Bidi_Mirroring_Glyph` of `c`.
pub(crate) fn mirror(c: char) -> Option<char> {
    let cp = c as u32;
    MIRROR
        .binary_search_by_key(&cp, |&(from, _)| from)
        .ok()
        .and_then(|i| char::from_u32(MIRROR[i].1))
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
pub fn contains_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // The GL JS quirk reproduced (decision 0009): private use counts as RTL.
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

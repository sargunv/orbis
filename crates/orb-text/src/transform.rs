//! Text rewrites before layout: `text-transform`, Arabic presentation forms
//! (GL JS `transform_text.ts`, `arabic_shaping.ts`) and vertical punctuation
//! (`verticalize_punctuation.ts`).

use crate::unicode;

/// `text-transform`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextTransform {
    #[default]
    None,
    Uppercase,
    Lowercase,
}

/// Applies `text-transform` and Arabic shaping to one section's text, as
/// GL JS's `transformText` does to each section.
pub fn transform_text(text: &str, transform: TextTransform) -> String {
    let cased = match transform {
        TextTransform::None => None,
        TextTransform::Uppercase => Some(text.to_uppercase()),
        TextTransform::Lowercase => Some(text.to_lowercase()),
    };
    apply_arabic_shaping(cased.as_deref().unwrap_or(text))
}

const LAM: char = '\u{644}';

#[derive(Clone, Copy)]
enum Form {
    Isolated = 0,
    Final = 1,
    Initial = 2,
    Medial = 3,
}

#[derive(Clone, Copy)]
struct Shaped {
    c: char,
    joining: u8,
    form: Option<Form>,
}

fn arabic_block(c: char) -> bool {
    matches!(c as u32,
        0x0600..=0x06FF | 0x0750..=0x077F | 0x0870..=0x089F | 0x08A0..=0x08FF
        | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF)
}

fn joins_forward(t: u8) -> bool {
    matches!(t, b'D' | b'L' | b'C')
}

fn joins_backward(t: u8) -> bool {
    matches!(t, b'D' | b'R' | b'C')
}

fn form_for(t: u8, before: bool, after: bool) -> Option<Form> {
    match t {
        b'D' => Some(match (before, after) {
            (true, true) => Form::Medial,
            (true, false) => Form::Final,
            (false, true) => Form::Initial,
            (false, false) => Form::Isolated,
        }),
        b'R' => Some(if before { Form::Final } else { Form::Isolated }),
        b'L' => Some(if after { Form::Initial } else { Form::Isolated }),
        _ => None,
    }
}

/// Rewrites Arabic letters as the Presentation Forms-B code point of the
/// shape each takes in its word, joining lam-alef. Text without Arabic is
/// returned unchanged.
pub fn apply_arabic_shaping(text: &str) -> String {
    if !text.chars().any(arabic_block) {
        return text.to_string();
    }
    let mut chars: Vec<Shaped> = text
        .chars()
        .map(|c| Shaped {
            c,
            joining: unicode::joining_type(c),
            form: None,
        })
        .collect();

    // Letter forms, looking through transparent marks.
    let visible: Vec<usize> = (0..chars.len())
        .filter(|&i| chars[i].joining != b'T')
        .collect();
    for (k, &i) in visible.iter().enumerate() {
        let before = k > 0 && joins_forward(chars[visible[k - 1]].joining);
        let after = k + 1 < visible.len() && joins_backward(chars[visible[k + 1]].joining);
        chars[i].form = form_for(chars[i].joining, before, after);
    }

    // Mark forms follow whether their letter joins forwards.
    let mut letter_joins_forward = false;
    for ch in &mut chars {
        if ch.joining == b'T' {
            ch.form = Some(if letter_joins_forward {
                Form::Medial
            } else {
                Form::Isolated
            });
        } else {
            letter_joins_forward = matches!(ch.form, Some(Form::Initial | Form::Medial));
        }
    }

    // Lam-alef ligatures; marks between the two letters follow.
    let mut out: Vec<Shaped> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].c != LAM {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let mut next = i + 1;
        while next < chars.len() && chars[next].joining == b'T' {
            next += 1;
        }
        let ligature = chars.get(next).and_then(|alef| unicode::lam_alef(alef.c));
        let Some(ligature) = ligature else {
            out.push(chars[i]);
            i += 1;
            continue;
        };
        let joined_before = matches!(chars[i].form, Some(Form::Medial | Form::Final));
        let c = ligature[usize::from(joined_before)];
        out.push(Shaped {
            c: char::from_u32(c).expect("ligature is a character"),
            joining: b'R',
            form: Some(Form::Isolated),
        });
        out.extend_from_slice(&chars[i + 1..next]);
        i = next + 1;
    }

    out.iter()
        .map(|ch| {
            let Some(form) = ch.form else { return ch.c };
            let Some(forms) = unicode::presentation_forms(ch.c) else {
                return ch.c;
            };
            let cp = match forms[form as usize] {
                0 => forms[Form::Isolated as usize],
                cp => cp,
            };
            if cp == 0 {
                ch.c
            } else {
                char::from_u32(cp).expect("presentation form is a character")
            }
        })
        .collect()
}

/// The vertical presentation form of a punctuation mark (GL JS
/// `verticalizedCharacterMap`).
pub fn vertical_form(c: char) -> Option<char> {
    Some(match c {
        '!' => '︕',
        '#' => '＃',
        '$' => '＄',
        '%' => '％',
        '&' => '＆',
        '(' => '︵',
        ')' => '︶',
        '*' => '＊',
        '+' => '＋',
        ',' => '︐',
        '-' => '︲',
        '.' => '・',
        '/' => '／',
        ':' => '︓',
        ';' => '︔',
        '<' => '︿',
        '=' => '＝',
        '>' => '﹀',
        '?' => '︖',
        '@' => '＠',
        '[' => '﹇',
        '\\' => '＼',
        ']' => '﹈',
        '^' => '＾',
        '_' => '︳',
        '`' => '｀',
        '{' => '︷',
        '|' => '―',
        '}' => '︸',
        '~' => '～',
        '¢' => '￠',
        '£' => '￡',
        '¥' => '￥',
        '¦' => '￤',
        '¬' => '￢',
        '¯' => '￣',
        '–' => '︲',
        '—' => '︱',
        '‘' => '﹃',
        '’' => '﹄',
        '“' => '﹁',
        '”' => '﹂',
        '…' => '︙',
        '⋯' => '︙',
        '‧' => '・',
        '₩' => '￦',
        '、' => '︑',
        '。' => '︒',
        '〈' => '︿',
        '〉' => '﹀',
        '《' => '︽',
        '》' => '︾',
        '「' => '﹁',
        '」' => '﹂',
        '『' => '﹃',
        '』' => '﹄',
        '【' => '︻',
        '】' => '︼',
        '〔' => '︹',
        '〕' => '︺',
        '〖' => '︗',
        '〗' => '︘',
        '！' => '︕',
        '（' => '︵',
        '）' => '︶',
        '，' => '︐',
        '－' => '︲',
        '．' => '・',
        '：' => '︓',
        '；' => '︔',
        '＜' => '︿',
        '＞' => '﹀',
        '？' => '︖',
        '［' => '﹇',
        '］' => '﹈',
        '＿' => '︳',
        '｛' => '︷',
        '｜' => '―',
        '｝' => '︸',
        '｟' => '︵',
        '｠' => '︶',
        '｡' => '︒',
        '｢' => '﹁',
        '｣' => '﹂',
        _ => return None,
    })
}

/// Replaces punctuation with its vertical form wherever neither neighbour is
/// a rotated character (GL JS `verticalizePunctuation`). One character in,
/// one out, so positions are preserved.
pub fn verticalize_punctuation(text: &str) -> Vec<char> {
    let chars: Vec<char> = text.chars().collect();
    let replaceable_neighbour = |c: Option<&char>| {
        c.is_none_or(|&c| {
            !unicode::has_rotated_vertical_orientation(c) || vertical_form(c).is_some()
        })
    };
    (0..chars.len())
        .map(|i| {
            let can_replace = replaceable_neighbour(chars.get(i + 1))
                && replaceable_neighbour(i.checked_sub(1).map(|p| &chars[p]));
            match vertical_form(chars[i]) {
                Some(v) if can_replace => v,
                _ => chars[i],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code_points(text: &str) -> Vec<String> {
        text.chars().map(|c| format!("{:04X}", c as u32)).collect()
    }

    const BEH: &str = "ب";
    const LAM: &str = "ل";
    const ALEF: &str = "ا";
    const HAMZA: &str = "ء";
    const FATHA: &str = "\u{64E}";
    const DAMMATAN: &str = "\u{64C}";
    const ZWNJ: &str = "\u{200C}";

    /// GL JS `arabic_shaping.test.ts`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn arabic_shaping() {
        assert_eq!(apply_arabic_shaping("hello world"), "hello world");
        assert_eq!(apply_arabic_shaping("שלום"), "שלום");
        assert_eq!(apply_arabic_shaping(""), "");
        assert_eq!(code_points(&apply_arabic_shaping(BEH)), ["FE8F"]);
        let s = |parts: &[&str]| apply_arabic_shaping(&parts.concat());
        assert_eq!(code_points(&s(&[BEH, BEH, BEH])), ["FE91", "FE92", "FE90"]);
        assert_eq!(code_points(&s(&[BEH, ALEF, BEH])), ["FE91", "FE8E", "FE8F"]);
        assert_eq!(
            code_points(&s(&[BEH, HAMZA, BEH])),
            ["FE8F", "0621", "FE8F"]
        );
        assert_eq!(code_points(&s(&[LAM, ALEF])), ["FEFB"]);
        assert_eq!(code_points(&s(&[BEH, LAM, ALEF])), ["FE91", "FEFC"]);
        assert_eq!(
            code_points(&s(&[BEH, FATHA, BEH])),
            ["FE91", "FE77", "FE90"]
        );
        assert_eq!(code_points(&s(&[BEH, FATHA])), ["FE8F", "FE76"]);
        assert_eq!(
            code_points(&s(&[BEH, DAMMATAN, BEH])),
            ["FE91", "FE72", "FE90"]
        );
        assert_eq!(code_points(&s(&[BEH, ZWNJ, BEH])), ["FE8F", "200C", "FE8F"]);
        assert_eq!(apply_arabic_shaping("hello مرحبا 123"), "hello ﻣﺮﺣﺒﺎ 123");
    }

    /// GL JS `verticalize_punctuation.test.ts`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn vertical_punctuation() {
        let v = |s: &str| verticalize_punctuation(s).into_iter().collect::<String>();
        assert_eq!(v("ABC123"), "ABC123");
        assert_eq!(v("!?"), "︕︖");
        assert_eq!(v("(…)"), "︵︙︶");
        assert_eq!(v("（⋯）"), "︵︙︶");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn transforms() {
        assert_eq!(
            transform_text("straße", TextTransform::Uppercase),
            "STRASSE"
        );
        assert_eq!(
            transform_text("\u{39F}\u{394}\u{39F}\u{3A3}", TextTransform::Lowercase),
            "\u{3BF}\u{3B4}\u{3BF}\u{3C2}"
        );
        assert_eq!(
            code_points(&transform_text(BEH, TextTransform::Uppercase)),
            ["FE8F"]
        );
    }
}

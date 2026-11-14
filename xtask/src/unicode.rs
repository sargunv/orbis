//! `cargo xtask unicode`: writes `orb-text`'s character tables from the
//! Unicode Character Database.
//!
//! The sets reproduce `build/generate-unicode-data.ts` of the pinned GL JS
//! (`COMMIT` in `fixtures.rs`), which derives them from UCD 17.0.0, plus the
//! JavaScript regular-expression classes its layout code tests characters
//! with. Only explicit UCD entries are read (no `@missing` defaults), as the
//! `@unicode/unicode-17.0.0` package GL JS reads does.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

const VERSION: &str = "17.0.0";
const OUT: &str = "crates/orb-text/src/unicode/tables.rs";
const MAX: usize = 0x11_0000;

/// `Bidi_Class` values in the order orb-text's `BIDI_CLASSES` lists
/// `unicode-bidi`'s.
const BIDI_CLASSES: [&str; 23] = [
    "AL", "AN", "B", "BN", "CS", "EN", "ES", "ET", "FSI", "L", "LRE", "LRI", "LRO", "NSM", "ON",
    "PDF", "PDI", "R", "RLE", "RLI", "RLO", "S", "WS",
];

/// `Bidi_Class` long names, which `@missing` lines use.
const BIDI_ALIASES: [(&str, &str); 23] = [
    ("AL", "Arabic_Letter"),
    ("AN", "Arabic_Number"),
    ("B", "Paragraph_Separator"),
    ("BN", "Boundary_Neutral"),
    ("CS", "Common_Separator"),
    ("EN", "European_Number"),
    ("ES", "European_Separator"),
    ("ET", "European_Terminator"),
    ("FSI", "First_Strong_Isolate"),
    ("L", "Left_To_Right"),
    ("LRE", "Left_To_Right_Embedding"),
    ("LRI", "Left_To_Right_Isolate"),
    ("LRO", "Left_To_Right_Override"),
    ("NSM", "Nonspacing_Mark"),
    ("ON", "Other_Neutral"),
    ("PDF", "Pop_Directional_Format"),
    ("PDI", "Pop_Directional_Isolate"),
    ("R", "Right_To_Left"),
    ("RLE", "Right_To_Left_Embedding"),
    ("RLI", "Right_To_Left_Isolate"),
    ("RLO", "Right_To_Left_Override"),
    ("S", "Segment_Separator"),
    ("WS", "White_Space"),
];

/// A set of code points.
#[derive(Clone)]
struct Set(Vec<bool>);

impl Set {
    fn new() -> Self {
        Self(vec![false; MAX])
    }

    fn add(&mut self, start: u32, end: u32) {
        for cp in start..=end {
            self.0[cp as usize] = true;
        }
    }

    fn remove(&mut self, start: u32, end: u32) {
        for cp in start..=end {
            self.0[cp as usize] = false;
        }
    }

    fn union(&mut self, other: &Set) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= *b;
        }
    }

    fn subtract(&mut self, other: &Set) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a &= !*b;
        }
    }

    fn has(&self, cp: u32) -> bool {
        self.0[cp as usize]
    }

    fn any(&self, f: impl Fn(u32) -> bool) -> bool {
        (0..MAX as u32).any(|cp| self.has(cp) && f(cp))
    }

    fn ranges(&self) -> Vec<(u32, u32)> {
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for cp in 0..MAX as u32 {
            if !self.has(cp) {
                continue;
            }
            match ranges.last_mut() {
                Some(last) if last.1 + 1 == cp => last.1 = cp,
                _ => ranges.push((cp, cp)),
            }
        }
        ranges
    }
}

/// One UCD file's `code points ; value` entries.
struct Property(Vec<(u32, u32, String)>);

impl Property {
    fn parse(text: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split(';').map(str::trim);
            let range = fields.next().ok_or("missing range")?;
            let value = fields.next().ok_or_else(|| format!("no value: {line}"))?;
            let (start, end) = parse_range(range)?;
            entries.push((start, end, value.to_string()));
        }
        Ok(Self(entries))
    }

    /// The `# @missing: range; value` default lines, in file order.
    fn parse_missing(text: &str) -> Result<Self, String> {
        let lines: String = text
            .lines()
            .filter_map(|line| line.strip_prefix("# @missing:"))
            .map(|line| format!("{line}\n"))
            .collect();
        Self::parse(&lines)
    }

    /// The code points whose value is `value` (compared loosely, as UCD
    /// aliases are: case, spaces, hyphens and underscores ignored).
    fn set(&self, value: &str) -> Set {
        let wanted = loose(value);
        let mut set = Set::new();
        for (start, end, v) in &self.0 {
            if loose(v) == wanted {
                set.add(*start, *end);
            }
        }
        set
    }

    fn values(&self) -> BTreeSet<String> {
        self.0.iter().map(|(_, _, v)| v.clone()).collect()
    }

    fn map(&self) -> BTreeMap<u32, String> {
        let mut map = BTreeMap::new();
        for (start, end, v) in &self.0 {
            for cp in *start..=*end {
                map.insert(cp, v.clone());
            }
        }
        map
    }
}

fn parse_range(range: &str) -> Result<(u32, u32), String> {
    let hex = |s: &str| u32::from_str_radix(s, 16).map_err(|e| format!("{s}: {e}"));
    match range.split_once("..") {
        Some((a, b)) => Ok((hex(a)?, hex(b)?)),
        None => Ok((hex(range)?, hex(range)?)),
    }
}

fn loose(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, ' ' | '-' | '_'))
        .flat_map(char::to_lowercase)
        .collect()
}

struct Ucd {
    blocks: Property,
    scripts: Property,
    general_category: Property,
    bidi_class: Property,
    /// `DerivedBidiClass.txt`'s `@missing` defaults.
    bidi_missing: Property,
    joining_type: Property,
    vertical_orientation: Property,
    indic_syllabic: Property,
    grapheme_break: Property,
    line_break: Property,
    prop_list: Property,
    unicode_data: String,
    bidi_mirroring: Property,
}

impl Ucd {
    fn fetch() -> Result<Self, String> {
        let get = |path: &str| -> Result<String, String> {
            let url = format!("https://www.unicode.org/Public/{VERSION}/ucd/{path}");
            eprintln!("fetching {url}");
            ureq::get(&url)
                .call()
                .map_err(|e| format!("{url}: {e}"))?
                .body_mut()
                .with_config()
                .limit(16 << 20)
                .read_to_string()
                .map_err(|e| format!("{url}: {e}"))
        };
        let prop = |path: &str| Property::parse(&get(path)?);
        let bidi = get("extracted/DerivedBidiClass.txt")?;
        Ok(Self {
            blocks: prop("Blocks.txt")?,
            scripts: prop("Scripts.txt")?,
            general_category: prop("extracted/DerivedGeneralCategory.txt")?,
            bidi_class: Property::parse(&bidi)?,
            bidi_missing: Property::parse_missing(&bidi)?,
            joining_type: prop("extracted/DerivedJoiningType.txt")?,
            vertical_orientation: prop("VerticalOrientation.txt")?,
            indic_syllabic: prop("IndicSyllabicCategory.txt")?,
            grapheme_break: prop("auxiliary/GraphemeBreakProperty.txt")?,
            line_break: prop("LineBreak.txt")?,
            prop_list: prop("PropList.txt")?,
            unicode_data: get("UnicodeData.txt")?,
            bidi_mirroring: prop("BidiMirroring.txt")?,
        })
    }

    fn blocks(&self, names: &[&str]) -> Result<Set, String> {
        let mut set = Set::new();
        for name in names {
            let block = self.blocks.set(name);
            if !block.any(|_| true) {
                return Err(format!("no block named {name}"));
            }
            set.union(&block);
        }
        Ok(set)
    }

    fn scripts(&self, names: &[&str]) -> Result<Set, String> {
        let mut set = Set::new();
        for name in names {
            let script = self.scripts.set(name);
            if !script.any(|_| true) {
                return Err(format!("no script named {name}"));
            }
            set.union(&script);
        }
        Ok(set)
    }

    /// General categories; a one-letter name is the major class (`P` is
    /// every `P*`), as in regular expressions.
    fn categories(&self, names: &[&str]) -> Set {
        let mut set = Set::new();
        for (start, end, value) in &self.general_category.0 {
            let matches = names.iter().any(|name| {
                if name.len() == 1 {
                    value.starts_with(name)
                } else {
                    value == name
                }
            });
            if matches {
                set.add(*start, *end);
            }
        }
        set
    }

    /// Every script except `skip`, as sets.
    fn each_script<'a>(&'a self, skip: &'a [&str]) -> impl Iterator<Item = Set> + 'a {
        self.scripts
            .values()
            .into_iter()
            .filter(move |s| !skip.contains(&s.as_str()))
            .map(|s| self.scripts.set(&s))
    }

    /// The blocks, as sets.
    fn each_block(&self) -> impl Iterator<Item = Set> + '_ {
        self.blocks
            .values()
            .into_iter()
            .map(|b| self.blocks.set(&b))
    }

    /// `isInCursiveScript`: scripts with a joining letter, plus Duployan.
    fn cursive(&self) -> Result<Set, String> {
        let mut joins = Set::new();
        for t in ["D", "L", "R", "C"] {
            joins.union(&self.joining_type.set(t));
        }
        let mut set = self.scripts(&["Duployan"])?;
        for script in self.each_script(&["Common", "Inherited"]) {
            if script.any(|cp| joins.has(cp)) {
                set.union(&script);
            }
        }
        Ok(set)
    }

    /// `isInComplexShapingScript`: blocks of the scripts with a combining
    /// dependent vowel sign.
    fn complex_shaping(&self) -> Set {
        let marks = self.categories(&["Mn", "Mc"]);
        let mut vowels = self.indic_syllabic.set("Vowel_Dependent");
        for cp in 0..MAX as u32 {
            if vowels.has(cp) && !marks.has(cp) {
                vowels.0[cp as usize] = false;
            }
        }
        let mut letters = Set::new();
        for script in self.each_script(&["Common", "Inherited"]) {
            if script.any(|cp| vowels.has(cp)) {
                letters.union(&script);
            }
        }
        let mut set = Set::new();
        for block in self.each_block() {
            if block.any(|cp| letters.has(cp)) {
                set.union(&block);
            }
        }
        set
    }

    /// `isInRTLScript`: scripts with a right-to-left letter, Common excepted.
    ///
    /// The npm package GL JS reads lists Unknown (every code point without a
    /// script: unassigned, private use, noncharacters) as a script, and its
    /// `Bidi_Class` includes the `@missing` right-to-left defaults of
    /// unassigned code points in the Hebrew and Arabic blocks. So Unknown
    /// counts as a right-to-left script, and GL JS lays out any label holding
    /// a private-use character (as every inline image is) through bidi.
    fn rtl(&self) -> Set {
        let mut rtl = self.bidi_class.set("R");
        rtl.union(&self.bidi_class.set("AL"));
        let mut set = Set::new();
        for script in self.each_script(&["Common"]) {
            if script.any(|cp| rtl.has(cp)) {
                set.union(&script);
            }
        }
        let mut unknown = Set::new();
        unknown.add(0, MAX as u32 - 1);
        unknown.remove(0xD800, 0xDFFF);
        for (start, end, _) in &self.scripts.0 {
            unknown.remove(*start, *end);
        }
        let mut defaults = self.bidi_missing.set("Right_To_Left");
        defaults.union(&self.bidi_missing.set("Arabic_Letter"));
        defaults.union(&rtl);
        if unknown.any(|cp| defaults.has(cp)) {
            set.union(&unknown);
        }
        set
    }

    fn local_ideograph(&self) -> Result<Set, String> {
        let mut set = self.blocks(&[
            "CJK Compatibility Forms",
            "CJK Compatibility",
            "CJK Radicals Supplement",
            "CJK Strokes",
            "CJK Unified Ideographs",
            "Enclosed CJK Letters And Months",
            "Enclosed Ideographic Supplement",
            "Halfwidth And Fullwidth Forms",
            "Hangul Syllables",
            "Hiragana",
            "Ideographic Symbols And Punctuation",
            "Kana Extended-A",
            "Kana Extended-B",
            "Kana Supplement",
            "Kangxi Radicals",
            "Katakana",
            "Katakana Phonetic Extensions",
            "CJK Symbols And Punctuation",
            "Small Kana Extension",
            "Vertical Forms",
        ])?;
        set.union(&self.scripts(&[
            "Bopomofo",
            "Han",
            "Hangul",
            "Hiragana",
            "Katakana",
            "Khitan Small Script",
            "Nushu",
            "Tangut",
            "Yi",
        ])?);
        set.union(&self.prop_list.set("Ideographic"));
        Ok(set)
    }

    fn ideographic_breaking(&self) -> Result<Set, String> {
        let mut set = self.blocks(&[
            "CJK Compatibility Forms",
            "CJK Compatibility",
            "CJK Radicals Supplement",
            "CJK Strokes",
            "CJK Symbols And Punctuation",
            "Enclosed CJK Letters And Months",
            "Enclosed Ideographic Supplement",
            "Halfwidth And Fullwidth Forms",
            "Ideographic Description Characters",
            "Ideographic Symbols And Punctuation",
            "Kana Extended-A",
            "Kana Extended-B",
            "Kana Supplement",
            "Kangxi Radicals",
            "Katakana Phonetic Extensions",
            "Small Kana Extension",
            "Vertical Forms",
        ])?;
        set.union(&self.scripts(&[
            "Bopomofo",
            "Han",
            "Hiragana",
            "Katakana",
            "Khitan Small Script",
            "Nushu",
            "Tangut",
            "Yi",
        ])?);
        Ok(set)
    }

    fn upright_vertical(&self) -> Result<Set, String> {
        let mut set = self.vertical_orientation.set("U");
        set.union(&self.vertical_orientation.set("Tu"));
        set.subtract(&self.neutral_vertical()?);
        set.union(&self.blocks(&[
            "CJK Compatibility Forms",
            "CJK Symbols And Punctuation",
            "Halfwidth And Fullwidth Forms",
            "Katakana",
            "Small Form Variants",
        ])?);
        set.add(0x02EA, 0x02EB);
        set.remove(0xFE49, 0xFE4F);
        set.remove(0x3008, 0x3011);
        set.remove(0x3014, 0x301F);
        set.remove(0x3030, 0x3030);
        set.remove(0x30FC, 0x30FC);
        set.remove(0xFF08, 0xFF09);
        set.remove(0xFF0D, 0xFF0D);
        set.remove(0xFF1A, 0xFF1E);
        set.remove(0xFF3B, 0xFF3B);
        set.remove(0xFF3D, 0xFF3D);
        set.remove(0xFF3F, 0xFF3F);
        set.remove(0xFF5B, 0xFFDF);
        set.remove(0xFFE3, 0xFFE3);
        set.remove(0xFFE8, 0xFFEF);
        set.remove(0xFE58, 0xFE5E);
        set.remove(0xFE63, 0xFE66);
        Ok(set)
    }

    fn neutral_vertical(&self) -> Result<Set, String> {
        let mut set = self.blocks(&[
            "CJK Compatibility Forms",
            "CJK Symbols And Punctuation",
            "Control Pictures",
            "Enclosed Alphanumerics",
            "Geometric Shapes",
            "Halfwidth And Fullwidth Forms",
            "Katakana",
            "Letterlike Symbols",
            "Miscellaneous Symbols",
            "Number Forms",
            "Optical Character Recognition",
            "Private Use Area",
            "Small Form Variants",
            "Supplementary Private Use Area-A",
            "Supplementary Private Use Area-B",
        ])?;
        for cp in [
            0x00A7, 0x00A9, 0x00AE, 0x00B1, 0x00BC, 0x00BD, 0x00BE, 0x00D7, 0x00F7, 0x2016, 0x2020,
            0x2021, 0x2030, 0x2031, 0x203B, 0x203C, 0x2042, 0x2047, 0x2048, 0x2049, 0x2051,
        ] {
            set.add(cp, cp);
        }
        for (a, b) in [
            (0x2300, 0x2307),
            (0x230C, 0x231F),
            (0x2324, 0x2328),
            (0x232B, 0x232B),
            (0x237D, 0x239A),
            (0x23BE, 0x23CD),
            (0x23CF, 0x23CF),
            (0x23D1, 0x23DB),
            (0x23E2, 0x23FF),
        ] {
            set.add(a, b);
        }
        set.remove(0x2423, 0x2423);
        set.remove(0x261A, 0x261F);
        set.add(0x2B12, 0x2B2F);
        set.add(0x2B50, 0x2B59);
        set.add(0x2BB8, 0x2BEB);
        set.add(0x221E, 0x221E);
        set.add(0x2234, 0x2235);
        set.add(0x2700, 0x2767);
        set.add(0x2776, 0x2793);
        set.add(0xFFFC, 0xFFFD);
        Ok(set)
    }

    fn grapheme_capable(&self) -> Set {
        let mut set = Set::new();
        for value in [
            "CR",
            "Extend",
            "L",
            "LV",
            "LVT",
            "Prepend",
            "Regional_Indicator",
            "SpacingMark",
            "T",
            "V",
            "ZWJ",
        ] {
            set.union(&self.grapheme_break.set(value));
        }
        set
    }

    fn written_without_spaces(&self) -> Result<Set, String> {
        let mut set = self.line_break.set("SA");
        set.union(&self.scripts(&["Balinese", "Javanese", "Tibetan"])?);
        Ok(set)
    }

    fn joins_following(&self) -> Set {
        let mut set = self.indic_syllabic.set("Invisible_Stacker");
        set.add(0x200D, 0x200D);
        set
    }

    /// JavaScript's `\s`: Zs, the other white space, and line terminators.
    fn js_whitespace(&self) -> Set {
        let mut set = self.general_category.set("Zs");
        for cp in [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0xFEFF, 0x2028, 0x2029] {
            set.add(cp, cp);
        }
        set
    }

    /// Arabic joining types (`R L D C T`), as `arabic_shaping.ts` reads them.
    fn arabic_joining(&self) -> Vec<(u32, u32, u8)> {
        const ARABIC: [(u32, u32); 7] = [
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x0870, 0x089F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
            (0x200C, 0x200D),
        ];
        let arabic = |cp: u32| ARABIC.iter().any(|&(a, b)| (a..=b).contains(&cp));
        let mut ranges: Vec<(u32, u32, u8)> = Vec::new();
        for (cp, value) in self.joining_type.map() {
            let t = match value.as_str() {
                "R" => b'R',
                "L" => b'L',
                "D" => b'D',
                "C" => b'C',
                "T" => b'T',
                _ => continue,
            };
            if !arabic(cp) {
                continue;
            }
            match ranges.last_mut() {
                Some(last) if last.1 + 1 == cp && last.2 == t => last.1 = cp,
                _ => ranges.push((cp, cp, t)),
            }
        }
        ranges
    }

    /// Presentation forms by base letter `[isolated, final, initial, medial]`,
    /// and the lam-alef ligatures by alef `[isolated, final]`.
    #[allow(clippy::type_complexity)]
    fn presentation_forms(&self) -> Result<(Vec<(u32, [u32; 4])>, Vec<(u32, [u32; 2])>), String> {
        let mut forms: BTreeMap<u32, [u32; 4]> = BTreeMap::new();
        let mut ligatures: BTreeMap<u32, [u32; 2]> = BTreeMap::new();
        for line in self.unicode_data.lines() {
            let fields: Vec<&str> = line.split(';').collect();
            let (Some(cp), Some(decomposition)) = (fields.first(), fields.get(5)) else {
                continue;
            };
            let cp = u32::from_str_radix(cp, 16).map_err(|e| e.to_string())?;
            if !(0xFB50..=0xFEFF).contains(&cp) {
                continue;
            }
            let Some((tag, rest)) = decomposition.split_once('>') else {
                continue;
            };
            let shape = match tag {
                "<isolated" => 0,
                "<final" => 1,
                "<initial" => 2,
                "<medial" => 3,
                _ => continue,
            };
            let bases: Vec<u32> = rest
                .split_whitespace()
                .map(|h| u32::from_str_radix(h, 16).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?;
            let carried = bases.len() == 2 && matches!(bases[0], 0x0020 | 0x0640);
            if bases.len() == 1 || carried {
                let base = if carried { bases[1] } else { bases[0] };
                forms.entry(base).or_default()[shape] = cp;
            } else if (0xFEF5..=0xFEFC).contains(&cp) && shape < 2 {
                if bases[0] != 0x0644 || bases.len() != 2 {
                    return Err(format!("unexpected lam-alef decomposition for {cp:04X}"));
                }
                ligatures.entry(bases[1]).or_default()[shape] = cp;
            }
        }
        Ok((forms.into_iter().collect(), ligatures.into_iter().collect()))
    }

    /// Each code point's `Bidi_Class`, `@missing` defaults then explicit
    /// entries, as runs of one class: each with its index in
    /// [`BIDI_CLASSES`], leaving out `L`, which every other code point is.
    fn bidi_classes(&self) -> Result<Vec<(u32, u32, u8)>, String> {
        let index = |name: &str| {
            let short = BIDI_ALIASES
                .iter()
                .find(|(_, long)| loose(long) == loose(name))
                .map_or(name, |(short, _)| short);
            BIDI_CLASSES
                .iter()
                .position(|class| *class == short)
                .map(|i| i as u8)
                .ok_or_else(|| format!("unknown Bidi_Class {name}"))
        };
        let left = index("L")?;
        let mut classes = vec![left; MAX];
        for (start, end, value) in self.bidi_missing.0.iter().chain(&self.bidi_class.0) {
            let class = index(value)?;
            classes[*start as usize..=*end as usize].fill(class);
        }
        let mut runs: Vec<(u32, u32, u8)> = Vec::new();
        for (cp, &class) in classes.iter().enumerate() {
            let cp = cp as u32;
            match runs.last_mut() {
                Some(run) if run.2 == class && run.1 + 1 == cp => run.1 = cp,
                _ if class == left => {}
                _ => runs.push((cp, cp, class)),
            }
        }
        Ok(runs)
    }

    fn mirrors(&self) -> Result<Vec<(u32, u32)>, String> {
        self.bidi_mirroring
            .0
            .iter()
            .map(|(cp, _, mirror)| {
                let mirror = u32::from_str_radix(mirror, 16).map_err(|e| e.to_string())?;
                Ok((*cp, mirror))
            })
            .collect()
    }
}

pub(crate) fn generate() -> Result<(), String> {
    let ucd = Ucd::fetch()?;
    let mut out = String::new();
    writeln!(
        out,
        "//! Character tables from UCD {VERSION}, written by `cargo xtask unicode`\n\
         //! (`xtask/src/unicode.rs`). Do not edit.\n"
    )
    .unwrap();

    let sets: [(&str, &str, Set); 18] = [
        (
            "CURSIVE_SCRIPT",
            "Scripts whose letters join (GL JS `codePointIsInCursiveScript`).",
            ucd.cursive()?,
        ),
        (
            "RTL_SCRIPT",
            "Scripts written right to left (GL JS `codePointIsInRTLScript`).",
            ucd.rtl(),
        ),
        (
            "COMPLEX_SHAPING_BLOCK",
            "Blocks of scripts whose letters combine into clusters (GL JS\n/// `codePointIsInComplexShapingScript`).",
            ucd.complex_shaping(),
        ),
        (
            "LOCAL_IDEOGRAPH",
            "Characters `localIdeographFontFamily` draws (GL JS\n/// `codePointUsesLocalIdeographFontFamily`).",
            ucd.local_ideograph()?,
        ),
        (
            "IDEOGRAPHIC_BREAKING",
            "Characters a line may break after (GL JS\n/// `codePointAllowsIdeographicBreaking`).",
            ucd.ideographic_breaking()?,
        ),
        (
            "UPRIGHT_VERTICAL",
            "Characters upright in vertical text (GL JS\n/// `codePointHasUprightVerticalOrientation`).",
            ucd.upright_vertical()?,
        ),
        (
            "NEUTRAL_VERTICAL",
            "Characters upright only beside upright ones (GL JS\n/// `codePointHasNeutralVerticalOrientation`).",
            ucd.neutral_vertical()?,
        ),
        (
            "GRAPHEME_CAPABLE",
            "Characters that can join a multi-code-point grapheme cluster (GL JS\n/// `textCanContainGraphemeClusters`).",
            ucd.grapheme_capable(),
        ),
        (
            "WRITTEN_WITHOUT_SPACES",
            "Scripts that don't space their words (GL JS\n/// `codePointIsWrittenWithoutSpaces`).",
            ucd.written_without_spaces()?,
        ),
        (
            "JOINS_FOLLOWING",
            "Characters that join their cluster to the next (GL JS\n/// `canCombineGraphemes`).",
            ucd.joins_following(),
        ),
        ("JS_WHITESPACE", "JavaScript `\\s`.", ucd.js_whitespace()),
        (
            "PUNCTUATION_OR_SYMBOL",
            "`\\p{P}` or `\\p{S}`.",
            ucd.categories(&["P", "S"]),
        ),
        ("DECIMAL_DIGIT", "`\\p{Nd}`.", ucd.categories(&["Nd"])),
        ("UPPERCASE_LETTER", "`\\p{Lu}`.", ucd.categories(&["Lu"])),
        ("SPACING_MARK", "`\\p{Mc}`.", ucd.categories(&["Mc"])),
        ("FORMAT", "`\\p{Cf}`.", ucd.categories(&["Cf"])),
        (
            "ARABIC_SCRIPT",
            "`\\p{sc=Arab}`.",
            ucd.scripts(&["Arabic"])?,
        ),
        (
            "BIDI_CONTROL",
            "Characters GL JS drops after reordering (`bidi.ts` `BIDI_CONTROLS`).",
            {
                let mut set = Set::new();
                for (a, b) in [
                    (0x061C, 0x061C),
                    (0x200E, 0x200F),
                    (0x202A, 0x202E),
                    (0x2066, 0x2069),
                ] {
                    set.add(a, b);
                }
                set
            },
        ),
    ];
    for (name, doc, set) in &sets {
        writeln!(out, "/// {doc}").unwrap();
        writeln!(
            out,
            "#[rustfmt::skip]\npub(crate) static {name}: super::Pairs = pairs!["
        )
        .unwrap();
        write_rows(
            &mut out,
            set.ranges()
                .iter()
                .map(|(a, b)| format!("(0x{a:X}, 0x{b:X})")),
        );
        writeln!(out, "];\n").unwrap();
    }

    writeln!(
        out,
        "/// Arabic joining types (`R L D C T`) in the Arabic blocks; others are `U`."
    )
    .unwrap();
    writeln!(
        out,
        "#[rustfmt::skip]\npub(crate) static JOINING_TYPE: &[(u32, u32, u8)] = &["
    )
    .unwrap();
    write_rows(
        &mut out,
        ucd.arabic_joining()
            .iter()
            .map(|(a, b, t)| format!("(0x{a:X}, 0x{b:X}, b'{}')", *t as char)),
    );
    writeln!(out, "];\n").unwrap();

    let (forms, ligatures) = ucd.presentation_forms()?;
    writeln!(
        out,
        "/// Presentation forms by letter: `[isolated, final, initial, medial]`, 0 if none."
    )
    .unwrap();
    writeln!(
        out,
        "#[rustfmt::skip]\npub(crate) static PRESENTATION_FORMS: &[(u32, [u32; 4])] = &["
    )
    .unwrap();
    write_rows(
        &mut out,
        forms.iter().map(|(base, [i, f, n, m])| {
            format!("(0x{base:X}, [0x{i:X}, 0x{f:X}, 0x{n:X}, 0x{m:X}])")
        }),
    );
    writeln!(out, "];\n").unwrap();
    writeln!(
        out,
        "/// Lam-alef ligatures by the alef: `[isolated, final]`."
    )
    .unwrap();
    writeln!(
        out,
        "#[rustfmt::skip]\npub(crate) static LAM_ALEF: &[(u32, [u32; 2])] = &["
    )
    .unwrap();
    write_rows(
        &mut out,
        ligatures
            .iter()
            .map(|(alef, [i, f])| format!("(0x{alef:X}, [0x{i:X}, 0x{f:X}])")),
    );
    writeln!(out, "];\n").unwrap();

    writeln!(out, "/// `Bidi_Mirroring_Glyph` pairs.").unwrap();
    writeln!(
        out,
        "#[rustfmt::skip]\npub(crate) static MIRROR: super::Pairs = pairs!["
    )
    .unwrap();
    write_rows(
        &mut out,
        ucd.mirrors()?
            .iter()
            .map(|(a, b)| format!("(0x{a:X}, 0x{b:X})")),
    );
    writeln!(out, "];\n").unwrap();

    writeln!(
        out,
        "/// `Bidi_Class` runs, by index in `BIDI_CLASSES`; `L` elsewhere."
    )
    .unwrap();
    writeln!(
        out,
        "#[rustfmt::skip]\npub(crate) static BIDI_CLASS: super::Classes = classes!["
    )
    .unwrap();
    write_rows(
        &mut out,
        ucd.bidi_classes()?
            .iter()
            .map(|(a, b, class)| format!("(0x{a:X}, 0x{b:X}, {class})")),
    );
    writeln!(out, "];").unwrap();

    std::fs::write(OUT, &out).map_err(|e| format!("{OUT}: {e}"))?;
    crate::run(Command::new("rustfmt").args(["--edition", "2024", OUT]))?;
    let size = std::fs::metadata(Path::new(OUT))
        .map_err(|e| e.to_string())?
        .len();
    eprintln!("wrote {OUT} ({size} bytes)");
    Ok(())
}

/// Writes items several to a line, so rustfmt leaves the table compact.
fn write_rows(out: &mut String, items: impl Iterator<Item = String>) {
    let mut line = String::from("   ");
    for item in items {
        if line.len() + item.len() + 2 > 98 {
            writeln!(out, "{line}").unwrap();
            line = String::from("   ");
        }
        write!(line, " {item},").unwrap();
    }
    if !line.trim().is_empty() {
        writeln!(out, "{line}").unwrap();
    }
}

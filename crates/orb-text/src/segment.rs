//! Grapheme clusters and word boundaries (GL JS `graphemes.ts`).
//!
//! Clusters come from ICU4X's grapheme segmenter, corrected as GL JS
//! corrects `Intl.Segmenter`: a cluster ending in a virama or ZWJ joins the
//! next one, and so does one followed by a spacing mark. Word boundaries are
//! only asked for inside scripts written without spaces, from ICU4X's
//! dictionary segmenter with the Southeast Asian dictionaries, or from the
//! host's (the browser's `Intl.Segmenter`, which GL JS asks).

use std::ops::Range;

use icu_segmenter::GraphemeClusterSegmenter;

use crate::unicode;

/// Splits `text` into grapheme clusters, as byte ranges.
pub(crate) fn graphemes(text: &str) -> Vec<Range<usize>> {
    let mut clusters: Vec<Range<usize>> = Vec::new();
    if !text.chars().any(unicode::can_form_cluster) {
        // No character can join another: one cluster per code point.
        clusters.extend(text.char_indices().map(|(i, c)| i..i + c.len_utf8()));
        return clusters;
    }
    let mut start = 0;
    for end in GraphemeClusterSegmenter::new().segment_str(text).skip(1) {
        let segment = start..end;
        start = end;
        if let Some(last) = clusters.last_mut()
            && can_combine(&text[last.clone()], &text[segment.clone()])
        {
            last.end = segment.end;
            continue;
        }
        clusters.push(segment);
    }
    clusters
}

/// GL JS `canCombineGraphemes`.
fn can_combine(former: &str, latter: &str) -> bool {
    former
        .chars()
        .next_back()
        .is_some_and(unicode::joins_following)
        || latter.chars().next().is_some_and(unicode::is_spacing_mark)
}

/// Whether a cluster is more than one code point, and so must be drawn as a
/// whole (GL JS `isCluster`).
pub(crate) fn is_cluster(grapheme: &str) -> bool {
    grapheme.chars().nth(1).is_some()
}

/// Finds words in scripts written without spaces (Thai, Lao, Khmer,
/// Burmese), which offer no other line breaks.
///
/// From [`WordBreaker::southeast_asian`]'s compiled dictionaries (the
/// `sea-dictionaries` feature), or the host's segmenter
/// ([`WordBreaker::from_host`]). Without one, no break is offered inside
/// such text, as GL JS behaves without `Intl.Segmenter`.
pub struct WordBreaker {
    breaks: Breaks,
}

/// A segmenter's word starts in a text, as byte offsets, its end included.
pub type HostWords = dyn Fn(&str) -> Vec<usize> + Send + Sync;

enum Breaks {
    #[cfg(feature = "sea-dictionaries")]
    Dictionaries(Box<icu_segmenter::WordSegmenter>),
    Host(Box<HostWords>),
}

impl std::fmt::Debug for WordBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.breaks {
            #[cfg(feature = "sea-dictionaries")]
            Breaks::Dictionaries(_) => "WordBreaker(dictionaries)",
            Breaks::Host(_) => "WordBreaker(host)",
        })
    }
}

impl WordBreaker {
    /// The four Southeast Asian dictionaries ICU4X ships (≈1 MB), without
    /// its CJK dictionary, which GL JS never consults.
    #[cfg(feature = "sea-dictionaries")]
    pub fn southeast_asian() -> Self {
        use icu_segmenter::options::WordBreakOptions;
        use icu_segmenter::provider::Baked;
        let mut segmenter = icu_segmenter::WordSegmenter::try_new_for_non_complex_scripts_unstable(
            &Baked,
            WordBreakOptions::default(),
        )
        .expect("compiled word data");
        segmenter
            .load_dictionary_unstable(&sea::Dictionaries)
            .expect("compiled dictionary data");
        Self {
            breaks: Breaks::Dictionaries(Box::new(segmenter)),
        }
    }

    /// The host's segmenter: `words(text)` answers the byte offsets at
    /// which a word begins, and the text's end.
    pub fn from_host(words: Box<HostWords>) -> Self {
        Self {
            breaks: Breaks::Host(words),
        }
    }

    /// The byte offsets at which a word begins.
    pub(crate) fn word_starts(&self, text: &str) -> Vec<usize> {
        match &self.breaks {
            #[cfg(feature = "sea-dictionaries")]
            Breaks::Dictionaries(segmenter) => segmenter.as_borrowed().segment_str(text).collect(),
            Breaks::Host(words) => words(text),
        }
    }
}

#[cfg(feature = "sea-dictionaries")]
mod sea {
    use icu_provider::prelude::*;
    use icu_segmenter::provider::{
        Baked, SegmenterDictionaryAutoV2, SegmenterDictionaryExtendedV2,
    };

    /// Serves ICU4X's Southeast Asian dictionaries and no CJK dictionary.
    pub(super) struct Dictionaries;

    impl DataProvider<SegmenterDictionaryExtendedV2> for Dictionaries {
        fn load(
            &self,
            request: DataRequest,
        ) -> Result<DataResponse<SegmenterDictionaryExtendedV2>, DataError> {
            Baked.load(request)
        }
    }

    impl DataProvider<SegmenterDictionaryAutoV2> for Dictionaries {
        fn load(
            &self,
            request: DataRequest,
        ) -> Result<DataResponse<SegmenterDictionaryAutoV2>, DataError> {
            Err(DataErrorKind::IdentifierNotFound
                .with_req(SegmenterDictionaryAutoV2::INFO, request))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<&str> {
        graphemes(text).into_iter().map(|r| &text[r]).collect()
    }

    /// GL JS `graphemes.test.ts`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn clusters() {
        assert_eq!(split("Tel Aviv"), ["T", "e", "l", " ", "A", "v", "i", "v"]);
        assert_eq!(split("e\u{301}"), ["e\u{301}"]);
        assert_eq!(split("שְׁ"), ["שְׁ"]);
        assert_eq!(split("שְׁדֵ"), ["שְׁ", "דֵ"]);
        assert_eq!(split("दिल्ली"), ["दि", "ल्ली"]);
        assert_eq!(split("ភ្នំ"), ["ភ្នំ"]);
        assert_eq!(split("ဘင်္ဂလားဒေ့ရှ်"), ["ဘ", "င်္ဂ", "လား", "ဒေ့", "ရှ်"]);
        assert_eq!(split("\u{30EDE}\u{30EDE}"), ["\u{30EDE}", "\u{30EDE}"]);
        assert_eq!(split("\r\nab"), ["\r\n", "a", "b"]);
        assert!(split("").is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn cluster_detection() {
        for single in ["a", " ", "麵", "\u{30EDE}"] {
            assert!(!is_cluster(single), "{single}");
        }
        for cluster in ["e\u{301}", "שְׁ", "\u{30EDE}\u{301}"] {
            assert!(is_cluster(cluster), "{cluster}");
        }
    }

    /// GL JS `wordBoundaries` tests, in byte offsets.
    #[cfg(feature = "sea-dictionaries")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn word_boundaries() {
        let words = WordBreaker::southeast_asian();
        assert_eq!(words.word_starts("Tel Aviv"), [0, 3, 4, 8]);
        let thai = "ราชอาณาจักรไทย";
        assert!(words.word_starts(thai).contains(&"ราชอาณาจักร".len()));
        assert!(words.word_starts("ab\ncd").contains(&3));
        assert!(words.word_starts("ab\r\ncd").contains(&4));
        assert!(
            words
                .word_starts("三三\u{200b}三三")
                .contains(&"三三\u{200b}".len())
        );
    }
}

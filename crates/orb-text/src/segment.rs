//! Grapheme clusters and word boundaries (GL JS `graphemes.ts`).
//!
//! Clusters come from ICU4X's grapheme segmenter, corrected as GL JS
//! corrects `Intl.Segmenter`: a cluster ending in a virama or ZWJ joins the
//! next one, and so does one followed by a spacing mark. Word boundaries are
//! only asked for inside scripts written without spaces, from ICU4X's
//! dictionary segmenter with the Southeast Asian dictionaries.

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
/// Built from data: [`WordBreaker::southeast_asian`] with the
/// `sea-dictionaries` feature (compiled in). Without one, no break is offered
/// inside such text, as GL JS behaves without `Intl.Segmenter`.
#[derive(Debug)]
pub struct WordBreaker {
    segmenter: icu_segmenter::WordSegmenter,
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
        Self { segmenter }
    }

    /// The same dictionaries from `sea_dictionary_bytes`' container, as
    /// the web loads them.
    #[cfg(feature = "sea-buffer")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        use icu_segmenter::options::WordBreakOptions;
        use icu_segmenter::provider::Baked;
        let dictionaries = buffer::Dictionaries::parse(bytes)?;
        let mut segmenter = icu_segmenter::WordSegmenter::try_new_for_non_complex_scripts_unstable(
            &Baked,
            WordBreakOptions::default(),
        )
        .map_err(|e| e.to_string())?;
        segmenter
            .load_dictionary_with_buffer_provider(&dictionaries)
            .map_err(|e| e.to_string())?;
        Ok(Self { segmenter })
    }

    /// The byte offsets at which a word begins.
    pub(crate) fn word_starts(&self, text: &str) -> Vec<usize> {
        self.segmenter.as_borrowed().segment_str(text).collect()
    }
}

/// The four dictionary attributes ICU4X's Southeast Asian segmenters ask
/// for.
#[cfg(any(feature = "sea-buffer", feature = "sea-export"))]
const SEA_DICTIONARIES: [&str; 4] = ["burmesedict", "khmerdict", "laodict", "thaidict"];

/// A container of the compiled Southeast Asian dictionaries for
/// [`WordBreaker::from_bytes`]: `ORBSEA1`, then per dictionary its
/// attribute (a byte of length, then UTF-8) and its postcard bytes (four
/// bytes of little-endian length, then the data).
#[cfg(feature = "sea-export")]
pub fn sea_dictionary_bytes() -> Vec<u8> {
    use icu_provider::prelude::*;
    use icu_segmenter::provider::{Baked, SegmenterDictionaryExtendedV2};
    let mut out = b"ORBSEA1".to_vec();
    for attribute in SEA_DICTIONARIES {
        let request = DataRequest {
            id: DataIdentifierBorrowed::for_marker_attributes(
                DataMarkerAttributes::from_str_or_panic(attribute),
            ),
            ..Default::default()
        };
        let response: DataResponse<SegmenterDictionaryExtendedV2> =
            Baked.load(request).expect("compiled dictionary");
        let data = postcard::to_allocvec(response.payload.get()).expect("serializable dictionary");
        out.push(attribute.len() as u8);
        out.extend(attribute.as_bytes());
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
    }
    out
}

#[cfg(feature = "sea-buffer")]
mod buffer {
    use icu_provider::buf::BufferFormat;
    use icu_provider::prelude::*;
    use icu_segmenter::provider::SegmenterDictionaryExtendedV2;

    /// Serves the container's dictionaries as postcard buffers.
    pub(super) struct Dictionaries(Vec<(String, Box<[u8]>)>);

    impl Dictionaries {
        pub(super) fn parse(mut bytes: &[u8]) -> Result<Self, String> {
            let bad = || "not a Southeast Asian dictionary container".to_string();
            bytes = bytes.strip_prefix(b"ORBSEA1").ok_or_else(bad)?;
            let mut entries = Vec::new();
            while let Some((&len, rest)) = bytes.split_first() {
                let (name, rest) = rest.split_at_checked(usize::from(len)).ok_or_else(bad)?;
                let (size, rest) = rest.split_at_checked(4).ok_or_else(bad)?;
                let size = u32::from_le_bytes(size.try_into().expect("four bytes")) as usize;
                let (data, rest) = rest.split_at_checked(size).ok_or_else(bad)?;
                let name = std::str::from_utf8(name).map_err(|_| bad())?;
                entries.push((name.to_string(), data.into()));
                bytes = rest;
            }
            if !super::SEA_DICTIONARIES
                .iter()
                .all(|a| entries.iter().any(|(n, _)| n == a))
            {
                return Err(bad());
            }
            Ok(Self(entries))
        }
    }

    impl DynamicDataProvider<BufferMarker> for Dictionaries {
        fn load_data(
            &self,
            marker: DataMarkerInfo,
            request: DataRequest,
        ) -> Result<DataResponse<BufferMarker>, DataError> {
            // The CJ dictionary, and anything else, is not served.
            let attribute = request.id.marker_attributes.as_str();
            let data = (marker.id == SegmenterDictionaryExtendedV2::INFO.id)
                .then(|| self.0.iter().find(|(name, _)| name == attribute))
                .flatten()
                .ok_or_else(|| DataErrorKind::IdentifierNotFound.with_req(marker, request))?;
            let mut metadata = DataResponseMetadata::default();
            metadata.buffer_format = Some(BufferFormat::Postcard1);
            Ok(DataResponse {
                metadata,
                payload: DataPayload::from_owned_buffer(data.1.clone()),
            })
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

    /// The web's container breaks words as the compiled dictionaries do.
    #[cfg(all(feature = "sea-export", feature = "sea-buffer"))]
    #[test]
    fn dictionary_bytes_break_as_compiled_data() {
        let compiled = WordBreaker::southeast_asian();
        let loaded = WordBreaker::from_bytes(&sea_dictionary_bytes()).unwrap();
        for text in ["ราชอาณาจักรไทย", "ພາສາລາວ", "ភាសាខ្មែរ", "မြန်မာဘာသာ"]
        {
            assert_eq!(
                loaded.word_starts(text),
                compiled.word_starts(text),
                "{text}"
            );
        }
        assert!(loaded.word_starts("ราชอาณาจักรไทย").len() > 2);
        assert!(WordBreaker::from_bytes(b"nonsense").is_err());
    }

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

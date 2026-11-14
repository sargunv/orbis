//! Glyph metrics on the job side: the PBF ranges a worker
//! has fetched, and the resolution of a tile's glyph requests through them.
//!
//! A range is keyed by its URL. Workers keep metrics only; the scene's
//! glyph atlas asks a range's owner worker for bitmaps separately, with a
//! [`GlyphRaster`] job.

use rustc_hash::FxHashMap as HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use orb_text::pbf::GlyphRange;
use orb_text::sdf::InkBox;
use orb_text::sdf::local::{self, TextMetrics};
use orb_text::{
    FontPolicy, FontStack, Glyph, GlyphKey, GlyphMap, GlyphMetrics, GlyphRequests, LocalFont,
    Provider,
};

pub use orb_host::{FontFiles, LocalGlyphs};

fn face(font: &LocalFont) -> orb_host::LocalFace<'_> {
    orb_host::LocalFace {
        family: &font.family,
        weight: font.weight.as_deref(),
        style: font.style,
    }
}

fn measure(local: &dyn LocalGlyphs, font: &LocalFont, text: &str) -> TextMetrics {
    local.measure(&face(font), text)
}

fn coverage(local: &dyn LocalGlyphs, font: &LocalFont, text: &str, ink: &InkBox) -> Vec<u8> {
    local.coverage(&face(font), text, ink)
}

/// A local glyph's layout metrics and bitmap size.
type LocalMetrics = (GlyphMetrics, [u32; 2]);

fn file_metrics(files: &dyn FontFiles, url: &str, text: &str) -> LocalMetrics {
    let m = files.measure(url, text);
    local::metrics(text, &local::ink_box(text, &m))
}

/// The font a declared file draws: its URL's hash, the same everywhere.
pub(crate) fn file_id(url: &str) -> orb_text::FontId {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in url.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    orb_text::FontId(h)
}

/// A local glyph's layout metrics and bitmap size.
fn local_metrics(local: &dyn LocalGlyphs, font: &LocalFont, text: &str) -> LocalMetrics {
    let ink = local::ink_box(text, &measure(local, font, text));
    local::metrics(text, &ink)
}

/// The PBF ranges one worker has fetched, shared by its threads and its
/// style's revisions.
#[derive(Debug, Default)]
pub(crate) struct GlyphCatalog {
    ranges: Mutex<HashMap<String, RangeState>>,
    /// The host's local fonts, if it has any, and what they measured.
    local: Mutex<Option<Arc<dyn LocalGlyphs>>>,
    measured: Mutex<HashMap<(LocalFont, Box<str>), LocalMetrics>>,
    warned_local: AtomicBool,
    /// The host's `font-faces` drawing, if it has one, the declared files
    /// fetched (`true` once open, `false` if they failed), and what they
    /// measured.
    files: Mutex<Option<Arc<dyn FontFiles>>>,
    fonts: Mutex<HashMap<String, bool>>,
    file_measured: Mutex<HashMap<(String, Box<str>), LocalMetrics>>,
    /// Ranges that failed, for the next tile build to report.
    warnings: Mutex<Vec<String>>,
}

#[derive(Debug, Clone)]
enum RangeState {
    Ready(Arc<GlyphRange>),
    /// The range failed to load or parse: its glyphs fall back to local
    /// fonts, as in GL JS, until [`GlyphCatalog::retry_failed`] lets it be
    /// fetched again.
    Failed,
}

/// The font files text may draw with: the style's, by name, and the
/// map's fallbacks, which every stack tries after its own names.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fonts<'a> {
    pub(crate) declared: &'a orb_style::FontFaces,
    pub(crate) fallback: &'a [orb_style::FontFace],
}

impl<'a> Fonts<'a> {
    /// No font files.
    #[cfg(test)]
    pub(crate) const NONE: Fonts<'static> = Fonts {
        declared: &Vec::new(),
        fallback: &[],
    };

    /// Every file, declared ones first.
    pub(crate) fn files(self) -> impl Iterator<Item = &'a orb_style::FontFace> {
        (self.declared.iter())
            .flat_map(|(_, files)| files)
            .chain(self.fallback)
    }

    fn is_empty(self) -> bool {
        self.declared.is_empty() && self.fallback.is_empty()
    }
}

/// What resolving a tile's requests found.
#[derive(Debug, Default)]
pub(crate) struct Resolved {
    pub(crate) glyphs: GlyphMap,
    /// Glyphs that need a local font, which the host doesn't provide.
    pub(crate) local: usize,
    /// Some glyph's range failed: it draws locally, or not at all, until
    /// a retry fetches the range.
    pub(crate) failed: bool,
}

impl GlyphCatalog {
    /// Draws local glyphs with the host's fonts from now on.
    pub(crate) fn set_local(&self, local: Arc<dyn LocalGlyphs>) {
        *self.local.lock().expect("glyph catalog poisoned") = Some(local);
    }

    /// Draws declared font files with the host's engine from now on.
    pub(crate) fn set_files(&self, files: Arc<dyn FontFiles>) {
        *self.files.lock().expect("glyph catalog poisoned") = Some(files);
    }

    /// Opens a fetched font file, or records its failure with GL JS's
    /// warning: its code points fall to the next file, the glyphs URL or
    /// local fonts.
    pub(crate) fn insert_font(&self, url: &str, bytes: Result<&[u8], String>) {
        let files = self.files.lock().expect("glyph catalog poisoned").clone();
        let opened = match files {
            Some(files) => bytes.and_then(|b| files.open(url, b)),
            None => Err("this host draws no font files".into()),
        };
        if let Err(e) = &opened {
            self.warn(format!("Ignoring the font face at {url}: {e}"));
        }
        self.fonts
            .lock()
            .expect("glyph catalog poisoned")
            .insert(url.to_string(), opened.is_ok());
    }

    /// The declared file that draws `request`, GL JS `getFontFamily`: each
    /// name of the stack in turn, then the map's fallbacks, each of their
    /// files covering the first code point, skipping files that failed.
    /// `Err` names a file not fetched yet.
    fn font_file<'a>(
        &self,
        faces: Fonts<'a>,
        request: &orb_text::GlyphRequest,
    ) -> Result<Option<&'a str>, &'a str> {
        if faces.is_empty() || request.variant != orb_text::Variant::Default {
            return Ok(None);
        }
        let code_point = request.text.chars().next().map_or(0, u32::from);
        let fonts = self.fonts.lock().expect("glyph catalog poisoned");
        let named = request.stack.as_str().split(',').filter_map(|name| {
            let declared = faces.declared.iter().find(|(n, _)| n == name.trim());
            declared.map(|(_, files)| files.as_slice())
        });
        for files in named.chain([faces.fallback]) {
            for face in files.iter().filter(|f| f.covers(code_point)) {
                match fonts.get(&face.url) {
                    Some(true) => return Ok(Some(&face.url)),
                    Some(false) => {}
                    None => return Err(&face.url),
                }
            }
        }
        Ok(None)
    }

    /// Whether to warn that local glyphs aren't drawn: once per worker.
    pub(crate) fn warn_local(&self) -> bool {
        !self.warned_local.swap(true, Ordering::Relaxed)
    }

    /// Records a fetched range, or its failure, which the next build
    /// reports.
    pub(crate) fn insert(&self, url: &str, bytes: Result<&[u8], String>) {
        let local = self.local.lock().expect("glyph catalog poisoned").is_some();
        let (state, warning) =
            match bytes.and_then(|b| GlyphRange::parse(b).map_err(|e| e.to_string())) {
                Ok(range) => (RangeState::Ready(Arc::new(range.without_bitmaps())), None),
                // GL JS warns alike, and draws the range's glyphs locally.
                Err(e) => (
                    RangeState::Failed,
                    Some(if local {
                        format!("glyphs {url}: {e}; its glyphs are drawn with local fonts instead")
                    } else {
                        format!("glyphs {url}: {e}; its glyphs are not drawn")
                    }),
                ),
            };
        self.ranges
            .lock()
            .expect("glyph catalog poisoned")
            .insert(url.to_string(), state);
        if let Some(warning) = warning {
            self.warnings
                .lock()
                .expect("glyph warnings poisoned")
                .push(warning);
        }
    }

    /// Forgets the ranges that failed: tiles that need one fetch it again
    /// ([`missing`](Self::missing)) rather than keep drawing its glyphs
    /// locally.
    pub(crate) fn retry_failed(&self) {
        (self.ranges.lock().expect("glyph catalog poisoned"))
            .retain(|_, state| !matches!(state, RangeState::Failed));
    }

    /// Reports a text-stage warning with the glyph warnings.
    pub(crate) fn warn(&self, warning: String) {
        self.warnings
            .lock()
            .expect("glyph warnings poisoned")
            .push(warning);
    }

    /// Warnings since the last call.
    pub(crate) fn take_warnings(&self) -> Vec<String> {
        std::mem::take(&mut self.warnings.lock().expect("glyph warnings poisoned"))
    }

    /// Whether `url` (a range or a font file) was fetched, or failed.
    pub(crate) fn has(&self, url: &str) -> bool {
        self.ranges
            .lock()
            .expect("glyph catalog poisoned")
            .contains_key(url)
            || self
                .fonts
                .lock()
                .expect("glyph catalog poisoned")
                .contains_key(url)
    }

    /// The host's font file engine, if it has one.
    pub(crate) fn files(&self) -> Option<Arc<dyn FontFiles>> {
        self.files.lock().expect("glyph catalog poisoned").clone()
    }

    /// The range URLs `requests` needs that this worker hasn't fetched, in
    /// first-requested order.
    pub(crate) fn missing(
        &self,
        requests: &GlyphRequests,
        template: Option<&str>,
        policy: FontPolicy,
        faces: Fonts<'_>,
    ) -> Vec<String> {
        let mut missing: Vec<String> = Vec::new();
        let mut pbf = Vec::new();
        for request in requests.iter() {
            match self.font_file(faces, request) {
                Err(url) => {
                    if !missing.iter().any(|m| m == url) {
                        missing.push(url.to_string());
                    }
                }
                Ok(Some(_)) => {}
                Ok(None) => pbf.push(request),
            }
        }
        let Some(template) = template else {
            return missing;
        };
        let ranges = self.ranges.lock().expect("glyph catalog poisoned");
        for request in pbf {
            if let Provider::Pbf { range } = orb_text::provider(request, policy) {
                let url = orb_text::glyphs_url(template, &request.stack, range);
                if !ranges.contains_key(&url) && !missing.contains(&url) {
                    missing.push(url);
                }
            }
        }
        missing
    }

    /// Resolves every request: PBF glyphs from fetched ranges, everything a
    /// local font would draw as unavailable. Ranges must have been fetched
    /// (see [`missing`](Self::missing)); a range still missing resolves as
    /// failed.
    /// `ideographs` is the map's `localIdeographFontFamily`.
    pub(crate) fn resolve(
        &self,
        requests: GlyphRequests,
        template: Option<&str>,
        policy: FontPolicy,
        ideographs: Option<&str>,
        faces: Fonts<'_>,
    ) -> Resolved {
        let files = self.files.lock().expect("glyph catalog poisoned").clone();
        let mut file_measured = self.file_measured.lock().expect("glyph catalog poisoned");
        let mut resolved = Resolved::default();
        let ranges = self.ranges.lock().expect("glyph catalog poisoned");
        let local = self.local.lock().expect("glyph catalog poisoned").clone();
        let mut measured = self.measured.lock().expect("glyph catalog poisoned");
        let mut draw_locally = |request: &orb_text::GlyphRequest, resolved: &mut Resolved| {
            let Some(local) = &local else {
                resolved.local += 1;
                return None;
            };
            let font = LocalFont::for_text(&request.stack, &request.text, ideographs);
            let (metrics, bitmap) = *measured
                .entry((font.clone(), request.text.clone()))
                .or_insert_with(|| local_metrics(local.as_ref(), &font, &request.text));
            Some(Glyph {
                key: GlyphKey::Cluster {
                    stack: request.stack.clone(),
                    variant: request.variant,
                    text: request.text.clone(),
                },
                metrics,
                bitmap,
                double_resolution: true,
            })
        };
        for request in requests {
            // Declared files first, as GL JS asks its font face manager.
            if let (Ok(Some(url)), Some(files)) = (self.font_file(faces, &request), &files) {
                let (metrics, bitmap) = *file_measured
                    .entry((url.to_string(), request.text.clone()))
                    .or_insert_with(|| file_metrics(files.as_ref(), url, &request.text));
                let glyph = Glyph {
                    key: GlyphKey::Font {
                        font: file_id(url),
                        variant: request.variant,
                        text: request.text.clone(),
                    },
                    metrics,
                    bitmap,
                    double_resolution: true,
                };
                resolved.glyphs.insert(request, Some(glyph));
                continue;
            }
            let glyph = match orb_text::provider(&request, policy) {
                Provider::Pbf { range } => {
                    let url =
                        orb_text::glyphs_url(template.unwrap_or_default(), &request.stack, range);
                    match ranges.get(&url) {
                        Some(RangeState::Ready(range)) => {
                            let code_point = request.text.chars().next().map_or(0, u32::from);
                            range.get(code_point).map(|g| g.glyph(&request.stack))
                        }
                        // A failed range draws locally, as in GL JS.
                        Some(RangeState::Failed) | None => {
                            resolved.failed = true;
                            draw_locally(&request, &mut resolved)
                        }
                    }
                }
                Provider::Local => draw_locally(&request, &mut resolved),
                Provider::Unavailable => None,
            };
            resolved.glyphs.insert(request, glyph);
        }
        resolved
    }
}

/// A raster job: bitmaps of some glyphs of one unit, a
/// glyph PBF range or a font stack's local glyphs. The scene asks each
/// unit's owner worker, once per glyph.
#[derive(Clone, Debug, PartialEq, Eq, orb_wire::Wire)]
pub enum GlyphRaster {
    Pbf {
        url: String,
        code_points: Vec<u32>,
    },
    /// Clusters (or single code points) of `stack`, drawn with the host's
    /// fonts; `ideographs` is the map's `localIdeographFontFamily`.
    Local {
        stack: String,
        texts: Vec<String>,
        ideographs: Option<String>,
    },
    /// Clusters (or single code points) drawn from a declared font file.
    File {
        url: String,
        texts: Vec<String>,
    },
}

/// One glyph's SDF bitmap, border included; `index` is its position in
/// the job's code points or texts.
#[derive(Clone, Debug, PartialEq, Eq, orb_wire::Wire)]
pub struct GlyphBitmap {
    pub index: u32,
    pub size: [u32; 2],
    pub sdf: Vec<u8>,
}

/// Called once with a raster job's bitmaps, on any thread. A code point the
/// range lacks, or that has no bitmap (a space), is left out.
pub type RasterDone = Box<dyn FnOnce(Result<Vec<GlyphBitmap>, String>) + Send + 'static>;

impl GlyphRaster {
    /// The worker that owns the unit, of `workers`.
    pub fn owner(&self, workers: u32) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut h = std::hash::DefaultHasher::new();
        match self {
            Self::Pbf { url, .. } => url.hash(&mut h),
            Self::Local { stack, .. } => stack.hash(&mut h),
            Self::File { url, .. } => url.hash(&mut h),
        }
        (h.finish() % u64::from(workers.max(1))) as u32
    }

    /// The requested bitmaps of a fetched range (a PBF job).
    pub fn run(&self, bytes: &[u8]) -> Result<Vec<GlyphBitmap>, String> {
        let Self::Pbf { url, code_points } = self else {
            return Err("a local raster job has no range".into());
        };
        let range = GlyphRange::parse(bytes).map_err(|e| format!("glyphs {url}: {e}"))?;
        Ok(code_points
            .iter()
            .enumerate()
            .filter_map(|(i, &cp)| Some((i, range.get(cp)?)))
            .filter(|(_, g)| !g.bitmap.is_empty())
            .map(|(i, g)| GlyphBitmap {
                index: i as u32,
                size: g.bitmap_size(),
                sdf: g.bitmap.clone(),
            })
            .collect())
    }

    /// The requested bitmaps drawn with the host's fonts (a local job):
    /// TinySDF over the host's coverage.
    pub fn draw(&self, local: &dyn LocalGlyphs) -> Result<Vec<GlyphBitmap>, String> {
        let Self::Local {
            stack,
            texts,
            ideographs,
        } = self
        else {
            return Err("a PBF raster job draws from its range".into());
        };
        let stack = FontStack::from_joined(stack);
        Ok(texts
            .iter()
            .enumerate()
            .filter_map(|(i, text)| {
                let font = LocalFont::for_text(&stack, text, ideographs.as_deref());
                let ink = local::ink_box(text, &measure(local, &font, text));
                if ink.glyph_width == 0 || ink.glyph_height == 0 {
                    return None;
                }
                let sdf = local::sdf(&ink, &coverage(local, &font, text, &ink));
                Some(GlyphBitmap {
                    index: i as u32,
                    size: [sdf.width, sdf.height],
                    sdf: sdf.data,
                })
            })
            .collect())
    }

    /// The requested bitmaps drawn from a declared font file, opening it
    /// from `bytes` unless `opened`.
    pub(crate) fn draw_file(
        &self,
        files: &dyn FontFiles,
        bytes: Option<&[u8]>,
    ) -> Result<Vec<GlyphBitmap>, String> {
        let Self::File { url, texts } = self else {
            return Err("only a font file job draws from a file".into());
        };
        if let Some(bytes) = bytes {
            files
                .open(url, bytes)
                .map_err(|e| format!("font face {url}: {e}"))?;
        }
        Ok(texts
            .iter()
            .enumerate()
            .filter_map(|(i, text)| {
                let ink = local::ink_box(text, &files.measure(url, text));
                if ink.glyph_width == 0 || ink.glyph_height == 0 {
                    return None;
                }
                let sdf = local::sdf(&ink, &files.coverage(url, text, &ink));
                Some(GlyphBitmap {
                    index: i as u32,
                    size: [sdf.width, sdf.height],
                    sdf: sdf.data,
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_text::{FontStack, Label};

    const TEMPLATE: &str = "local://glyphs/{fontstack}/{range}.pbf";

    fn requests(text: &str) -> GlyphRequests {
        let mut requests = GlyphRequests::default();
        requests.add_label(
            &Label::from_text(text),
            &FontStack::new(&["Noto Sans Regular"]),
            false,
            false,
        );
        requests
    }

    fn range_bytes() -> Vec<u8> {
        let path = format!(
            "{}/glyphs/Noto Sans Regular/0-255.pbf",
            crate::testing::ASSETS
        );
        std::fs::read(path).expect("the committed glyph range")
    }

    #[test]
    fn resolves_fetched_ranges_and_reports_missing_ones() {
        let catalog = GlyphCatalog::default();
        let policy = FontPolicy {
            glyphs_url: true,
            local_ideographs: false,
        };
        let r = requests("Ab東");
        let missing = catalog.missing(&r, Some(TEMPLATE), policy, Fonts::NONE);
        assert_eq!(
            missing,
            [
                "local://glyphs/Noto Sans Regular/0-255.pbf",
                "local://glyphs/Noto Sans Regular/26368-26623.pbf",
            ]
        );
        catalog.insert(&missing[0], Ok(&range_bytes()));
        assert!(catalog.take_warnings().is_empty());
        catalog.insert(&missing[1], Err("not found".into()));
        assert_eq!(catalog.take_warnings().len(), 1);
        assert!(
            catalog
                .missing(&r, Some(TEMPLATE), policy, Fonts::NONE)
                .is_empty()
        );
        let resolved = catalog.resolve(r, Some(TEMPLATE), policy, None, Fonts::NONE);
        let stack = FontStack::new(&["Noto Sans Regular"]);
        let (_, a) = resolved
            .glyphs
            .get(&stack, orb_text::Variant::Default, "A")
            .expect("A is in range 0");
        assert!(a.metrics.advance > 0.0);
        assert!(
            resolved
                .glyphs
                .get(&stack, orb_text::Variant::Default, "東")
                .is_none()
        );
        assert_eq!(
            resolved.local, 1,
            "a failed range falls back to local fonts"
        );
        assert!(resolved.failed);
        // A retry lets the failed range be fetched again, and only it.
        catalog.retry_failed();
        let r = requests("Ab東");
        assert_eq!(
            catalog.missing(&r, Some(TEMPLATE), policy, Fonts::NONE),
            [missing[1].clone()]
        );
        assert!(!catalog.has(&missing[1]) && catalog.has(&missing[0]));
        catalog.insert(&missing[1], Ok(&range_bytes()));
        let resolved = catalog.resolve(r, Some(TEMPLATE), policy, None, Fonts::NONE);
        assert!(!resolved.failed && resolved.local == 0);
    }

    #[test]
    fn rasters_requested_code_points_and_packs_them() {
        let raster = GlyphRaster::Pbf {
            url: "test://0-255.pbf".into(),
            code_points: vec![u32::from('A'), u32::from(' '), u32::from('é')],
        };
        let bitmaps = raster.run(&range_bytes()).unwrap();
        // A space has no bitmap.
        assert_eq!(bitmaps.len(), 2);
        let a = &bitmaps[0];
        assert_eq!(a.sdf.len() as u32, a.size[0] * a.size[1]);
        assert_eq!(
            orb_wire::decode::<Vec<GlyphBitmap>>(&orb_wire::encode(&bitmaps)).unwrap(),
            bitmaps
        );
    }

    #[test]
    fn a_style_without_glyphs_draws_everything_locally() {
        let catalog = GlyphCatalog::default();
        let policy = FontPolicy::default();
        let r = requests("ab");
        assert!(catalog.missing(&r, None, policy, Fonts::NONE).is_empty());
        assert_eq!(catalog.resolve(r, None, policy, None, Fonts::NONE).local, 2);
    }

    #[derive(Debug)]
    struct FakeFiles;

    impl FontFiles for FakeFiles {
        fn is_open(&self, _: &str) -> bool {
            true
        }
        fn open(&self, _: &str, bytes: &[u8]) -> Result<(), String> {
            if bytes == b"font" {
                Ok(())
            } else {
                Err("not a font file".into())
            }
        }
        fn measure(&self, _: &str, _: &str) -> orb_host::TextMetrics {
            orb_host::TextMetrics {
                advance: 20.0,
                ascent: 30.0,
                descent: 5.0,
                left: 0.0,
                right: 18.0,
            }
        }
        fn coverage(&self, _: &str, _: &str, ink: &InkBox) -> Vec<u8> {
            vec![255; (ink.glyph_width * ink.glyph_height) as usize]
        }
    }

    /// A declared file draws the code points it covers, ahead of the
    /// glyphs URL; one that fails falls through to the next provider.
    #[test]
    fn declared_font_files_come_first() {
        let catalog = GlyphCatalog::default();
        catalog.set_files(Arc::new(FakeFiles));
        let policy = FontPolicy {
            glyphs_url: true,
            local_ideographs: false,
        };
        let faces: orb_style::FontFaces = vec![(
            "Noto Sans Regular".into(),
            vec![orb_style::FontFace {
                url: "test://latin.woff2".into(),
                ranges: vec![[0x41, 0x5A]],
            }],
        )];
        let fonts = Fonts {
            declared: &faces,
            fallback: &[],
        };
        let r = requests("Ab");
        // "A" needs the file first; "b" (outside its range) the PBF range.
        assert_eq!(
            catalog.missing(&r, Some(TEMPLATE), policy, fonts),
            vec![
                "test://latin.woff2".to_string(),
                TEMPLATE
                    .replace("{fontstack}", "Noto Sans Regular")
                    .replace("{range}", "0-255"),
            ]
        );
        catalog.insert_font("test://latin.woff2", Ok(b"font"));
        catalog.insert(
            &TEMPLATE
                .replace("{fontstack}", "Noto Sans Regular")
                .replace("{range}", "0-255"),
            Ok(&range_bytes()),
        );
        let resolved = catalog.resolve(r, Some(TEMPLATE), policy, None, fonts);
        let stack = FontStack::new(&["Noto Sans Regular"]);
        let key = |text: &str| {
            resolved
                .glyphs
                .get(&stack, orb_text::Variant::Default, text)
                .map(|(_, g)| g.key.clone())
        };
        assert!(
            matches!(key("A"), Some(GlyphKey::Font { font, .. }) if font == file_id("test://latin.woff2"))
        );
        assert!(matches!(key("b"), Some(GlyphKey::Pbf { .. })));

        let failed = GlyphCatalog::default();
        failed.set_files(Arc::new(FakeFiles));
        failed.insert_font("test://latin.woff2", Ok(b"html"));
        assert_eq!(failed.take_warnings().len(), 1);
        let missing = failed.missing(&requests("A"), Some(TEMPLATE), policy, fonts);
        assert!(missing.iter().all(|m| m.ends_with(".pbf")), "{missing:?}");
    }

    /// A map's fallback files draw what a stack's own names don't cover,
    /// whatever the stack.
    #[test]
    fn fallback_font_files_follow_the_stacks_own() {
        let catalog = GlyphCatalog::default();
        catalog.set_files(Arc::new(FakeFiles));
        let policy = FontPolicy {
            glyphs_url: true,
            local_ideographs: false,
        };
        let face = |url: &str, ranges| orb_style::FontFace {
            url: url.into(),
            ranges,
        };
        let declared: orb_style::FontFaces = vec![(
            "Noto Sans Regular".into(),
            vec![face("test://a.woff2", vec![[0x41, 0x41]])],
        )];
        let fallback = [face("test://any.ttf", vec![[0x41, 0x42]])];
        let fonts = Fonts {
            declared: &declared,
            fallback: &fallback,
        };
        // "A" takes the stack's own file, "B" the fallback; "c" neither.
        let missing = catalog.missing(&requests("ABc"), Some(TEMPLATE), policy, fonts);
        assert_eq!(missing[..2], ["test://a.woff2", "test://any.ttf"]);
        assert!(missing[2].ends_with("0-255.pbf"), "{missing:?}");
    }
}

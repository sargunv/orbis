//! Tile jobs (plan §7.3): what a map asks of its job executor, the pure
//! [`TileWorker`] that does the work, and the packed form results take
//! across web workers.
//!
//! A map hands its jobs each style revision as a [`StyleUpdate`] (the
//! layers and sources new in it), then requests tiles. Native hosts and the
//! browser's main-thread mode use [`PoolJobs`], which runs the work on an
//! [`Executor`]; the browser's worker mode runs a `TileWorker` in each job
//! worker and ships results back packed (see [`pack`]).
//!
//! Workers keep each tile's decoded [`TileFrame`](orb_tile::TileFrame)
//! (plan §10.4: the owner job worker keeps them), so a request for some of
//! a tile's nodes, or only their property rows, lays them out again
//! without fetching or decoding.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use orb_crs::{TileIndex, TileMatrixSet};
use orb_expr::json::{self, Json, JsonValue, Member};
use orb_hal::{HostBytesId, HostImageId};
use orb_host::{DecodedImage, Fetch, FetchError, ImageDecode};
use orb_jobs::Executor;
use orb_layout::{Chunk, TileBox};
use orb_style::CostModel;
use orb_tile::TileFrame;
use orb_v8::{Encoding, GeoJsonData, SourceKind, Style, Value};

use crate::glyphs::{GlyphCatalog, GlyphRaster, LocalGlyphs, RasterDone};
use crate::style::{ImageKind, StylePlan};
use crate::symbol::{ResultSymbols, SpriteIcons};
use crate::tile::{self, ImageRef, Select, TileBuild};
use crate::{TileLoader, TileLocation};

/// Decoded tiles a worker keeps per source, least recently used first out.
const FRAME_CACHE: usize = 256;

/// One source's data in a worker, shared by its style revisions.
#[derive(Debug, Default)]
struct SourceData {
    geojson: Mutex<Option<Result<orb_tile::geojson::Index, String>>>,
    frames: Mutex<Frames>,
}

#[derive(Debug, Default)]
struct Frames {
    tiles: HashMap<TileIndex, (Arc<TileFrame>, u64)>,
    clock: u64,
}

impl Frames {
    fn get(&mut self, tile: TileIndex) -> Option<Arc<TileFrame>> {
        self.clock += 1;
        let clock = self.clock;
        self.tiles.get_mut(&tile).map(|(frame, used)| {
            *used = clock;
            Arc::clone(frame)
        })
    }

    fn insert(&mut self, tile: TileIndex, frame: Arc<TileFrame>) {
        self.clock += 1;
        if self.tiles.len() >= FRAME_CACHE && !self.tiles.contains_key(&tile) {
            let oldest = self.tiles.iter().min_by_key(|(_, (_, used))| *used);
            if let Some((&oldest, _)) = oldest {
                self.tiles.remove(&oldest);
            }
        }
        self.tiles.insert(tile, (frame, self.clock));
    }
}

/// What a worker keeps across a style's revisions.
#[derive(Debug, Default)]
struct WorkerData {
    /// Per source key, kept while the source is.
    sources: Mutex<HashMap<u32, Arc<SourceData>>>,
    /// Glyph metrics of the PBF ranges fetched so far.
    glyphs: GlyphCatalog,
    /// Sprite metadata for icon layout.
    icons: Mutex<Arc<SpriteIcons>>,
    /// Southeast Asian word data, and the URL it is fetched from.
    words: Mutex<Words>,
    word_url: Mutex<Option<String>>,
}

/// The data labels in scripts written without spaces (Thai, Lao, Khmer,
/// Burmese) break lines with (decision 0009).
#[derive(Clone, Debug)]
enum Words {
    Ready(Arc<orb_text::WordBreaker>),
    /// Fetched from this URL before text that needs it is laid out.
    Fetch(String),
    /// Such text offers no breaks, as in GL JS without `Intl.Segmenter`.
    Unavailable,
}

impl Default for Words {
    /// Native builds compile the data in; the web fetches it lazily.
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            static COMPILED: std::sync::OnceLock<Arc<orb_text::WordBreaker>> =
                std::sync::OnceLock::new();
            Self::Ready(Arc::clone(
                COMPILED.get_or_init(|| Arc::new(orb_text::WordBreaker::southeast_asian())),
            ))
        }
        #[cfg(target_arch = "wasm32")]
        Self::Unavailable
    }
}

/// Lays out tiles for one revision of a style. Pure, so it runs on any job
/// thread or worker.
#[derive(Debug)]
pub struct TileWorker {
    plan: StylePlan,
    data: Arc<WorkerData>,
    /// The map's `localIdeographFontFamily`.
    ideographs: Option<String>,
    /// The plane GeoJSON tiles in; `None` for Web Mercator.
    plane: Option<TileMatrixSet>,
}

/// A tile job's outcome: the laid-out tile, or what its labels need first.
#[derive(Debug)]
pub enum Built {
    Tile(TileBuild),
    /// URLs (glyph PBF ranges, word data) to fetch and add with
    /// [`TileWorker::add_resource`]; then build again.
    Fetch(Vec<String>),
}

impl TileWorker {
    /// Parses a v8 style that will not be edited. Returns the style's
    /// errors joined by newlines.
    pub fn new(style_json: &str) -> Result<Self, String> {
        let style = orb_v8::parse(style_json).map_err(|d| d.to_string())?;
        let layers = (1..=style.layers.len() as u32).collect();
        let sources = (1..=style.sources.len() as u32).collect();
        Ok(Self::with_style(
            None,
            Arc::new(style),
            layers,
            sources,
            0,
            CostModel::DEFAULT,
            None,
            None,
        ))
    }

    /// A revision of a parsed style, keeping `previous`'s data for the
    /// sources both have.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_style(
        previous: Option<&TileWorker>,
        style: Arc<Style>,
        layer_keys: Vec<u32>,
        source_keys: Vec<u32>,
        revision: u64,
        cost: CostModel,
        ideographs: Option<String>,
        plane: Option<TileMatrixSet>,
    ) -> Self {
        let plan = StylePlan::new(style, layer_keys, source_keys, revision, cost);
        let data = match previous {
            Some(previous) => {
                let data = Arc::clone(&previous.data);
                data.sources
                    .lock()
                    .expect("worker data poisoned")
                    .retain(|key, _| plan.source(*key).is_some());
                data
            }
            None => Arc::default(),
        };
        Self {
            plan,
            data,
            ideographs,
            plane,
        }
    }

    /// Adds a resource a build asked for (or its failure) for later
    /// builds, which report a failure.
    pub fn add_resource(&self, url: &str, bytes: Result<&[u8], String>) {
        let mut words = self.data.words.lock().expect("words poisoned");
        let word_url = self.data.word_url.lock().expect("words poisoned");
        if word_url.as_deref() == Some(url) {
            // Tiles that waited on the same fetch each bring it back.
            if !matches!(*words, Words::Fetch(_)) {
                return;
            }
            *words = match bytes.and_then(orb_text::WordBreaker::from_bytes) {
                Ok(breaker) => Words::Ready(Arc::new(breaker)),
                Err(e) => {
                    self.data.glyphs.warn(format!(
                        "word data {url}: {e}; labels in Thai, Lao, Khmer or Burmese do not wrap"
                    ));
                    Words::Unavailable
                }
            };
            return;
        }
        drop(words);
        if self.declares_font(url) {
            self.data.glyphs.insert_font(url, bytes);
            return;
        }
        self.data.glyphs.insert(url, bytes);
    }

    /// Fetches the Southeast Asian word data from `url` when text first
    /// needs it (the web, which doesn't compile it in), unless it is here.
    pub fn set_word_data_url(&self, url: &str) {
        let mut words = self.data.words.lock().expect("words poisoned");
        if matches!(*words, Words::Unavailable) {
            *words = Words::Fetch(url.to_string());
            *self.data.word_url.lock().expect("words poisoned") = Some(url.to_string());
        }
    }

    /// The word data, once loaded, to hand to workers of other styles.
    pub fn word_breaker(&self) -> Option<Arc<orb_text::WordBreaker>> {
        match &*self.data.words.lock().expect("words poisoned") {
            Words::Ready(breaker) => Some(Arc::clone(breaker)),
            _ => None,
        }
    }

    /// Uses word data another worker loaded.
    pub fn set_word_breaker(&self, breaker: Arc<orb_text::WordBreaker>) {
        *self.data.words.lock().expect("words poisoned") = Words::Ready(breaker);
    }

    /// Draws local glyphs with the host's fonts, for this revision and
    /// later ones.
    pub fn set_local_glyphs(&self, local: Arc<dyn LocalGlyphs>) {
        self.data.glyphs.set_local(local);
    }

    /// Whether the style declares `url` as a font file (`font-faces`).
    pub fn declares_font(&self, url: &str) -> bool {
        let declared = self.plan.style.font_faces.iter();
        declared.flat_map(|(_, files)| files).any(|f| f.url == url)
    }

    /// Draws declared font files (`font-faces`) with the host's engine, for
    /// this revision and later ones.
    pub fn set_font_files(&self, files: Arc<dyn crate::glyphs::FontFiles>) {
        self.data.glyphs.set_files(files);
    }

    /// Sets the sprite metadata icons are laid out with, for this revision
    /// and later ones.
    pub fn set_icons(&self, icons: Arc<SpriteIcons>) {
        *self.data.icons.lock().expect("icons poisoned") = icons;
    }

    /// Applies an update from the map: the first update has every layer
    /// and source, later ones those new in their revision.
    pub fn update(previous: Option<&TileWorker>, update: &StyleUpdate) -> Result<Self, String> {
        let old = previous.map(|p| &p.plan);
        let mut sources = Vec::with_capacity(update.sources.len());
        for key in &update.sources {
            let source = match update.source_json.iter().find(|(k, ..)| k == key) {
                Some((_, id, json)) => orb_v8::parse_source(id, json).map_err(|d| d.to_string())?.0,
                None => old
                    .and_then(|plan| {
                        let at = plan.source_keys.iter().position(|k| k == key)?;
                        Some(plan.style.sources[at].clone())
                    })
                    .ok_or_else(|| format!("style update without source {key}"))?,
            };
            sources.push(source);
        }
        let mut layers = Vec::with_capacity(update.layers.len());
        for (index, key) in update.layers.iter().enumerate() {
            let layer = match update.layer_json.iter().find(|(k, _)| k == key) {
                Some((_, json)) => {
                    orb_v8::parse_layer(json, index, &sources, &update.state)
                        .map_err(|d| d.to_string())?
                        .0
                }
                None => old
                    .and_then(|plan| {
                        let at = plan.layer_keys.iter().position(|k| k == key)?;
                        Some(plan.style.layers[at].clone())
                    })
                    .ok_or_else(|| format!("style update without layer {key}"))?,
            };
            layers.push(layer);
        }
        let style = Style {
            name: None,
            center: None,
            zoom: None,
            bearing: 0.0,
            pitch: 0.0,
            sprites: Vec::new(),
            glyphs: update.glyphs.clone(),
            font_faces: update.font_faces.clone(),
            light: Default::default(),
            sky: None,
            terrain: None,
            state: Vec::new(),
            sources,
            layers,
            warnings: Vec::new(),
        };
        Ok(Self::with_style(
            previous,
            Arc::new(style),
            update.layers.clone(),
            update.sources.clone(),
            update.revision,
            update.cost,
            update.local_ideographs.clone(),
            update.plane.clone(),
        ))
    }

    pub(crate) fn plan(&self) -> &StylePlan {
        &self.plan
    }

    /// The style revision this worker lays out.
    pub fn revision(&self) -> u64 {
        self.plan.revision
    }

    fn data(&self, source: u32) -> Arc<SourceData> {
        let mut data = self.data.sources.lock().expect("worker data poisoned");
        Arc::clone(data.entry(source).or_default())
    }

    /// Whether `source` (a source key) has image tiles to decode rather
    /// than lay out, and of which kind: any of the style's sources, drawn
    /// by layers or by the terrain alone.
    pub fn image_kind(&self, source: u32) -> Option<ImageKind> {
        let at = self.plan.source_keys.iter().position(|&k| k == source)?;
        match self.plan.style.sources[at].kind {
            orb_v8::SourceKind::Raster(_) => Some(ImageKind::Raster),
            orb_v8::SourceKind::RasterDem(_) => Some(ImageKind::Dem),
            _ => None,
        }
    }

    fn geojson_source(&self, source: u32) -> Option<&orb_v8::GeoJsonSource> {
        match &self.plan.source_def(self.plan.source(source)?).kind {
            SourceKind::GeoJson(source) => Some(source),
            _ => None,
        }
    }

    /// The whole GeoJSON document URL, if this is a URL-backed GeoJSON
    /// source. Job hosts deduplicate that fetch per source, and treat a
    /// missing document as a source error rather than a missing vector tile.
    pub fn data_url(&self, source: u32) -> Option<&str> {
        match &self.geojson_source(source)?.data {
            GeoJsonData::Url(url) => Some(url),
            GeoJsonData::Inline(_) => None,
        }
    }

    /// Whether `request` can build without fetching bytes: its tile is
    /// cached, or its source is inline or already indexed GeoJSON. Job
    /// hosts use this to avoid fetching again; `build` ignores bytes then.
    pub fn has_data(&self, request: &TileRequest) -> bool {
        if self.plan.source(request.source).is_none() {
            return false;
        }
        let data = self.data(request.source);
        if data
            .frames
            .lock()
            .expect("frames poisoned")
            .tiles
            .contains_key(&request.tile)
        {
            return true;
        }
        self.geojson_source(request.source).is_some_and(|s| {
            matches!(s.data, GeoJsonData::Inline(_))
                || data
                    .geojson
                    .lock()
                    .expect("GeoJSON index poisoned")
                    .is_some()
        })
    }

    /// Builds `request`'s tile and lays out its nodes, from the cached tile
    /// if there is one. Vector bytes are of the ancestor at the request's
    /// data zoom, in `encoding`; GeoJSON bytes are the complete URL
    /// document on the first call, and are ignored for inline data or once
    /// the source's index has been initialized.
    pub fn build(
        &self,
        bytes: &[u8],
        encoding: Encoding,
        request: &TileRequest,
    ) -> Result<Built, String> {
        let index = self
            .plan
            .source(request.source)
            .ok_or_else(|| format!("source {} was removed", request.source))?;
        let mut warnings = Vec::new();
        let frame = self.frame(bytes, encoding, request)?;
        let select = Select {
            nodes: request.nodes.as_deref(),
            rows_only: request.rows_only,
        };
        let icons = Arc::clone(&self.data.icons.lock().expect("icons poisoned"));
        let words = self.data.words.lock().expect("words poisoned").clone();
        let symbols = tile::Symbols {
            glyphs: &self.data.glyphs,
            icons: &icons,
            ideographs: self.ideographs.as_deref(),
            words: match &words {
                Words::Ready(breaker) => tile::WordData::Ready(breaker),
                Words::Fetch(url) => tile::WordData::Fetch(url),
                Words::Unavailable => tile::WordData::Unavailable,
            },
        };
        match tile::build(&frame, &self.plan, index, request.zoom, select, symbols) {
            Ok(mut build) => {
                warnings.extend(self.data.glyphs.take_warnings());
                build.warnings.splice(0..0, warnings);
                Ok(Built::Tile(build))
            }
            Err(missing) => Ok(Built::Fetch(missing)),
        }
    }

    /// The tile's decoded frame: cached, or decoded from `bytes` and cached.
    pub(crate) fn frame(
        &self,
        bytes: &[u8],
        encoding: Encoding,
        request: &TileRequest,
    ) -> Result<Arc<TileFrame>, String> {
        let TileRequest {
            source,
            tile,
            data_zoom,
            ..
        } = *request;
        if data_zoom > tile.z {
            return Err("tile data must come from an ancestor".into());
        }
        let index = self
            .plan
            .source(source)
            .ok_or_else(|| format!("source {source} was removed"))?;
        let data = self.data(source);
        let cached = data.frames.lock().expect("frames poisoned").get(tile);
        Ok(match cached {
            Some(frame) => frame,
            None => {
                let frame = Arc::new(self.decode(bytes, encoding, index, &data, tile, data_zoom)?);
                data.frames
                    .lock()
                    .expect("frames poisoned")
                    .insert(tile, Arc::clone(&frame));
                frame
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        bytes: &[u8],
        encoding: Encoding,
        index: usize,
        data: &SourceData,
        tile: TileIndex,
        data_zoom: u8,
    ) -> Result<TileFrame, String> {
        let frame = if let SourceKind::GeoJson(geojson) = &self.plan.source_def(index).kind {
            let mut cached = data.geojson.lock().expect("GeoJSON index poisoned");
            let index = cached.get_or_insert_with(|| {
                let features = match &geojson.data {
                    GeoJsonData::Inline(features) => features.clone(),
                    GeoJsonData::Url(_) => orb_v8::parse_geojson(
                        std::str::from_utf8(bytes).map_err(|e| e.to_string())?,
                    )?,
                };
                let options = orb_tile::geojson::Options {
                    plane: self.plane.clone().map(Arc::new),
                    ..geojson.options.clone()
                };
                orb_tile::geojson::Index::new(features, options)
            });
            index
                .as_mut()
                .map_err(|e| e.clone())?
                .tile(tile.z, tile.col as i32, tile.row)
                .unwrap_or_default()
        } else if bytes.is_empty() {
            // An archive's left-out tile.
            TileFrame::default()
        } else {
            match encoding {
                Encoding::Mvt => orb_tile::mvt::decode(bytes),
                Encoding::Mlt => orb_tile::mlt::decode(bytes),
            }
            .map_err(|e| e.to_string())?
        };
        Ok(if data_zoom < tile.z {
            let depth = tile.z - data_zoom;
            let mask = (1u32 << depth) - 1;
            frame.overzoom(depth, tile.col & mask, tile.row & mask)
        } else {
            frame
        })
    }
}

#[cfg(test)]
impl Built {
    /// The tile, for tests whose styles need no glyphs.
    pub(crate) fn tile(self) -> TileBuild {
        match self {
            Self::Tile(build) => build,
            Self::Fetch(urls) => panic!("the tile needs resources: {urls:?}"),
        }
    }
}

/// A style revision for the jobs: the order of its layer and source keys,
/// the layers and sources new or changed in it as v8 JSON, and the global
/// state its properties resolve.
#[derive(Clone, Debug, Default)]
pub struct StyleUpdate {
    pub revision: u64,
    pub layers: Vec<u32>,
    pub sources: Vec<u32>,
    pub layer_json: Vec<(u32, Arc<Json>)>,
    /// Key, id and definition.
    pub source_json: Vec<(u32, String, Arc<Json>)>,
    pub state: Vec<(String, Value<'static>)>,
    /// The style's glyph PBF URL template.
    pub glyphs: Option<String>,
    /// The style's declared font files.
    pub font_faces: orb_v8::FontFaces,
    /// The map's `localIdeographFontFamily`: ideographs draw with it.
    pub local_ideographs: Option<String>,
    /// Which paint the GPU interprets, as the map's plan decides it.
    pub cost: CostModel,
    /// The plane GeoJSON sources tile in: the surface's one-tile set, or
    /// `None` for Web Mercator.
    pub plane: Option<TileMatrixSet>,
}

impl StyleUpdate {
    /// The update as one JSON document, for a job worker.
    pub fn to_json(&self) -> String {
        let synthetic = |value| Json::new(value, Default::default());
        let member = |key: &str, value: Json| Member {
            key: key.into(),
            key_span: Default::default(),
            value,
        };
        let keys = |keys: &[u32]| {
            synthetic(JsonValue::Array(
                keys.iter()
                    .map(|&k| synthetic(JsonValue::Number(f64::from(k))))
                    .collect(),
            ))
        };
        let state = self
            .state
            .iter()
            .map(|(k, v)| {
                let value = json::parse(&v.to_json_string()).expect("values write JSON");
                member(k, value)
            })
            .collect();
        let layers = self
            .layer_json
            .iter()
            .map(|(k, json)| {
                synthetic(JsonValue::Array(vec![
                    synthetic(JsonValue::Number(f64::from(*k))),
                    (**json).clone(),
                ]))
            })
            .collect();
        let sources = self
            .source_json
            .iter()
            .map(|(k, id, json)| {
                synthetic(JsonValue::Array(vec![
                    synthetic(JsonValue::Number(f64::from(*k))),
                    synthetic(JsonValue::String(id.clone())),
                    (**json).clone(),
                ]))
            })
            .collect();
        synthetic(JsonValue::Object(vec![
            member(
                "revision",
                synthetic(JsonValue::Number(self.revision as f64)),
            ),
            member("layers", keys(&self.layers)),
            member("sources", keys(&self.sources)),
            member("layerJson", synthetic(JsonValue::Array(layers))),
            member("sourceJson", synthetic(JsonValue::Array(sources))),
            member("state", synthetic(JsonValue::Object(state))),
            member(
                "glyphs",
                synthetic(match &self.glyphs {
                    Some(url) => JsonValue::String(url.clone()),
                    None => JsonValue::Null,
                }),
            ),
            member(
                "fontFaces",
                synthetic(JsonValue::Array(
                    self.font_faces
                        .iter()
                        .map(|(name, files)| {
                            let files = files
                                .iter()
                                .map(|f| {
                                    let ranges = f
                                        .ranges
                                        .iter()
                                        .flatten()
                                        .map(|&v| synthetic(JsonValue::Number(f64::from(v))))
                                        .collect();
                                    synthetic(JsonValue::Array(vec![
                                        synthetic(JsonValue::String(f.url.clone())),
                                        synthetic(JsonValue::Array(ranges)),
                                    ]))
                                })
                                .collect();
                            synthetic(JsonValue::Array(vec![
                                synthetic(JsonValue::String(name.clone())),
                                synthetic(JsonValue::Array(files)),
                            ]))
                        })
                        .collect(),
                )),
            ),
            member(
                "localIdeographFontFamily",
                synthetic(match &self.local_ideographs {
                    Some(family) => JsonValue::String(family.clone()),
                    None => JsonValue::Null,
                }),
            ),
            member(
                "maxOps",
                synthetic(JsonValue::Number(self.cost.max_ops as f64)),
            ),
            member(
                "plane",
                synthetic(match &self.plane {
                    None => JsonValue::Null,
                    Some(tms) => JsonValue::String(crate::projection_to_json(tms.projection())),
                }),
            ),
        ]))
        .to_string()
    }

    /// Reads [`to_json`](Self::to_json)'s document.
    pub fn parse(text: &str) -> Result<Self, String> {
        let bad = || "malformed style update".to_string();
        let doc = json::parse(text).map_err(|e| e.to_string())?;
        let keys = |key: &str| -> Result<Vec<u32>, String> {
            doc.get(key)
                .and_then(Json::as_array)
                .ok_or_else(bad)?
                .iter()
                .map(|k| k.as_f64().map(|k| k as u32).ok_or_else(bad))
                .collect()
        };
        let entries = |key: &str| doc.get(key).and_then(Json::as_array).ok_or_else(bad);
        let layer_json = entries("layerJson")?
            .iter()
            .map(|e| match e.as_array() {
                Some([k, json]) => Ok((k.as_f64().ok_or_else(bad)? as u32, Arc::new(json.clone()))),
                _ => Err(bad()),
            })
            .collect::<Result<_, String>>()?;
        let source_json = entries("sourceJson")?
            .iter()
            .map(|e| match e.as_array() {
                Some([k, id, json]) => Ok((
                    k.as_f64().ok_or_else(bad)? as u32,
                    id.as_str().ok_or_else(bad)?.to_string(),
                    Arc::new(json.clone()),
                )),
                _ => Err(bad()),
            })
            .collect::<Result<_, String>>()?;
        let state = doc
            .get("state")
            .and_then(Json::as_object)
            .ok_or_else(bad)?
            .iter()
            .map(|m| (m.key.clone(), Value::from_json(&m.value)))
            .collect();
        let font_faces = entries("fontFaces")?
            .iter()
            .map(|e| match e.as_array() {
                Some([name, files]) => Ok((
                    name.as_str().ok_or_else(bad)?.to_string(),
                    files
                        .as_array()
                        .ok_or_else(bad)?
                        .iter()
                        .map(|f| match f.as_array() {
                            Some([url, ranges]) => {
                                let bounds = ranges
                                    .as_array()
                                    .ok_or_else(bad)?
                                    .iter()
                                    .map(|v| v.as_f64().map(|v| v as u32).ok_or_else(bad))
                                    .collect::<Result<Vec<u32>, String>>()?;
                                Ok(orb_v8::FontFace {
                                    url: url.as_str().ok_or_else(bad)?.to_string(),
                                    ranges: bounds
                                        .chunks(2)
                                        .map(|r| [r[0], r[r.len() - 1]])
                                        .collect(),
                                })
                            }
                            _ => Err(bad()),
                        })
                        .collect::<Result<_, String>>()?,
                )),
                _ => Err(bad()),
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            revision: doc.get("revision").and_then(Json::as_f64).ok_or_else(bad)? as u64,
            layers: keys("layers")?,
            sources: keys("sources")?,
            layer_json,
            source_json,
            state,
            glyphs: doc.get("glyphs").and_then(Json::as_str).map(str::to_string),
            font_faces,
            local_ideographs: doc
                .get("localIdeographFontFamily")
                .and_then(Json::as_str)
                .map(str::to_string),
            cost: CostModel {
                max_ops: doc.get("maxOps").and_then(Json::as_f64).ok_or_else(bad)? as usize,
            },
            plane: match doc.get("plane").and_then(Json::as_str) {
                Some(plane) => {
                    let projection = crate::projection_from_json(plane)?;
                    Some(TileMatrixSet::square(projection, projection.world_bounds()))
                }
                None => None,
            },
        })
    }
}

/// A tile for a job to load and lay out.
#[derive(Clone, Debug, PartialEq)]
pub struct TileRequest {
    /// Which style to lay out with, as given to [`Jobs::set_style`].
    pub style: u64,
    /// The source's key.
    pub source: u32,
    pub tile: TileIndex,
    /// The style zoom the tile lays out at: its own zoom in
    /// `WebMercatorQuad`, from its ground resolution in other sets.
    pub zoom: f64,
    /// Zoom of the ancestor whose bytes are fetched.
    pub data_zoom: u8,
    pub location: TileLocation,
    pub encoding: Encoding,
    /// Lay out only these nodes (by key); `None` for all of the source's.
    pub nodes: Option<Vec<u32>>,
    /// Only property rows: a column job.
    pub rows_only: bool,
    /// A terrain DEM tile: also decode its elevations with this unpack
    /// vector, for the CPU ([`TileResult::dem`]).
    pub dem_unpack: Option<DemUnpack>,
}

/// GL JS's unpack vector of a DEM encoding: metres are
/// `r·x + g·y + b·z − w` over the 0–255 channels. Requests compare it
/// bit for bit.
#[derive(Clone, Copy, Debug)]
pub struct DemUnpack(pub [f64; 4]);

impl PartialEq for DemUnpack {
    fn eq(&self, other: &Self) -> bool {
        self.0.map(f64::to_bits) == other.0.map(f64::to_bits)
    }
}

impl Eq for DemUnpack {}

impl TileRequest {
    /// The ancestor whose bytes the host fetches (including archive reads).
    pub fn data_tile(&self) -> TileIndex {
        let depth = self.tile.z - self.data_zoom;
        TileIndex {
            z: self.data_zoom,
            col: self.tile.col >> depth,
            row: self.tile.row >> depth,
        }
    }
}

/// Called once with a tile's result, on any thread.
pub type TileDone = Box<dyn FnOnce(Result<TileResult, String>) + Send + 'static>;

/// Runs tile jobs for a map.
pub trait Jobs: Send + Sync {
    /// Makes a revision of style `id` available to later requests. The
    /// first update of an id has every layer and source. The `worker` is
    /// the revision built in this process; job hosts in other workers apply
    /// `update` instead.
    fn set_style(&self, id: u64, update: &StyleUpdate, worker: &Arc<TileWorker>);
    /// Sets the sprite metadata style `id`'s symbol layers lay icons out
    /// with, before the tiles that need them are requested. `worker` is
    /// the style's latest revision in this process.
    fn set_icons(&self, id: u64, icons: &Arc<SpriteIcons>, worker: &Arc<TileWorker>);
    /// Loads and lays out a tile. A tile that doesn't exist yields an empty
    /// result.
    fn load(&self, request: TileRequest, done: TileDone);
    /// Rasterizes glyphs for the atlas: one deduplicated job per unit, run
    /// by the unit's owner worker (decision 0009).
    fn raster(&self, request: GlyphRaster, done: RasterDone);
    /// Reads features of a tile for a query, in the worker that loaded it
    /// (plan §13), from its cached frame or from the tile fetched again.
    fn features(&self, request: crate::FeatureRequest, done: crate::FeaturesDone);
}

type PendingSources = HashMap<(u64, u32), Vec<(TileRequest, TileDone)>>;

/// Glyph range fetches in flight, by style and URL: what to run when each
/// arrives.
type GlyphWaiters = HashMap<(u64, String), Vec<Box<dyn FnOnce() + Send>>>;

/// Jobs in this process: fetch through the host, then decode and lay out on
/// an executor.
pub struct PoolJobs {
    fetch: Arc<dyn Fetch>,
    decode: Arc<dyn ImageDecode>,
    loader: Arc<TileLoader>,
    executor: Arc<dyn Executor>,
    styles: Mutex<Vec<(u64, Arc<TileWorker>)>>,
    // In-flight source fetches, shared by all tile requests for that source.
    geojson_loads: Arc<Mutex<PendingSources>>,
    glyphs: Arc<GlyphLoads>,
    /// The host's local fonts, if any.
    local: Mutex<Option<Arc<dyn LocalGlyphs>>>,
    files: Mutex<Option<Arc<dyn crate::glyphs::FontFiles>>>,
}

impl std::fmt::Debug for PoolJobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolJobs").finish_non_exhaustive()
    }
}

/// Fetches glyph ranges for tiles that wait on them, once per range.
struct GlyphLoads {
    fetch: Arc<dyn Fetch>,
    executor: Arc<dyn Executor>,
    waiting: Mutex<GlyphWaiters>,
}

impl GlyphLoads {
    /// Runs `ready` once `worker` has `url`, fetching it unless that's
    /// already under way.
    fn wait(
        self: &Arc<Self>,
        worker: &Arc<TileWorker>,
        style: u64,
        url: String,
        ready: Box<dyn FnOnce() + Send>,
    ) {
        let key = (style, url);
        let mut waiting = self.waiting.lock().expect("glyph loads poisoned");
        if let Some(waiters) = waiting.get_mut(&key) {
            waiters.push(ready);
            return;
        }
        // A fetch that finished after the build asked would be fetched again.
        if worker.data.glyphs.has(&key.1) {
            drop(waiting);
            ready();
            return;
        }
        waiting.insert(key.clone(), vec![ready]);
        drop(waiting);
        let (this, worker) = (Arc::clone(self), Arc::clone(worker));
        let url = key.1.clone();
        self.fetch.get(
            &key.1.clone(),
            Box::new(move |fetched| {
                let files = worker
                    .data
                    .glyphs
                    .files()
                    .filter(|_| worker.declares_font(&url));
                let finish = move |prepared: Result<(), String>| {
                    let bytes = prepared.and(fetched.as_deref().map_err(ToString::to_string));
                    worker.add_resource(&url, bytes);
                    let waiters = this
                        .waiting
                        .lock()
                        .expect("glyph loads poisoned")
                        .remove(&key)
                        .unwrap_or_default();
                    for ready in waiters {
                        ready();
                    }
                };
                // A font file opens once the engine is in.
                match files {
                    Some(files) => files.prepare(Box::new(finish)),
                    None => finish(Ok(())),
                }
            }),
        );
    }

    /// Builds a tile on this thread, fetching the glyph ranges its labels
    /// need first and building again once they are all in.
    fn complete(
        self: &Arc<Self>,
        worker: Arc<TileWorker>,
        bytes: Arc<Vec<u8>>,
        encoding: Encoding,
        request: TileRequest,
        done: TileDone,
    ) {
        let urls = match worker.build(&bytes, encoding, &request) {
            Ok(Built::Tile(build)) => return done(Ok(build.into())),
            Err(e) => return done(Err(e)),
            Ok(Built::Fetch(urls)) => urls,
        };
        let remaining = Arc::new(std::sync::atomic::AtomicUsize::new(urls.len()));
        let retry = Arc::new(Mutex::new(Some((bytes, request, done))));
        let style = retry
            .lock()
            .expect("retry poisoned")
            .as_ref()
            .expect("set")
            .1
            .style;
        for url in urls {
            let (this, worker2) = (Arc::clone(self), Arc::clone(&worker));
            let (remaining, retry) = (Arc::clone(&remaining), Arc::clone(&retry));
            let ready = Box::new(move || {
                if remaining.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) != 1 {
                    return;
                }
                let (bytes, request, done) = retry
                    .lock()
                    .expect("retry poisoned")
                    .take()
                    .expect("retried once");
                let executor = Arc::clone(&this.executor);
                executor.spawn(Box::new(move || {
                    this.complete(worker2, bytes, encoding, request, done);
                }));
            });
            self.wait(&worker, style, url, ready);
        }
    }
}

impl PoolJobs {
    pub fn new(
        fetch: Arc<dyn Fetch>,
        decode: Arc<dyn ImageDecode>,
        executor: Arc<dyn Executor>,
    ) -> Arc<Self> {
        Arc::new(Self {
            loader: TileLoader::new(fetch.clone()),
            glyphs: Arc::new(GlyphLoads {
                fetch: fetch.clone(),
                executor: executor.clone(),
                waiting: Mutex::default(),
            }),
            fetch,
            decode,
            executor,
            styles: Mutex::new(Vec::new()),
            geojson_loads: Arc::default(),
            local: Mutex::new(None),
            files: Mutex::new(None),
        })
    }

    /// Draws local glyphs (decision 0009) with the host's fonts: system
    /// fonts natively, the canvas on the web's main thread.
    pub fn set_local_glyphs(&self, local: Arc<dyn LocalGlyphs>) {
        *self.local.lock().expect("local glyphs poisoned") = Some(local);
    }

    /// Draws declared font files with the host's engine: `orb-text`
    /// natively, `shaper.wasm` in the web's job workers.
    pub fn set_font_files(&self, files: Arc<dyn crate::glyphs::FontFiles>) {
        *self.files.lock().expect("font files poisoned") = Some(files);
    }
}

impl Jobs for PoolJobs {
    fn set_style(&self, id: u64, _: &StyleUpdate, worker: &Arc<TileWorker>) {
        let mut styles = self.styles.lock().expect("styles poisoned");
        // Requests for the previous style may still be in flight; requests
        // for an older revision of this one use the newest.
        styles.retain(|(i, _)| *i + 1 >= id && *i != id);
        styles.push((id, Arc::clone(worker)));
        if let Some(local) = &*self.local.lock().expect("local glyphs poisoned") {
            worker.set_local_glyphs(Arc::clone(local));
        }
        if let Some(files) = &*self.files.lock().expect("font files poisoned") {
            worker.set_font_files(Arc::clone(files));
        }
    }

    fn set_icons(&self, _: u64, icons: &Arc<SpriteIcons>, worker: &Arc<TileWorker>) {
        // Revisions share their worker data, so this reaches every one.
        worker.set_icons(Arc::clone(icons));
    }

    fn raster(&self, request: GlyphRaster, done: RasterDone) {
        let executor = Arc::clone(&self.executor);
        match &request {
            GlyphRaster::Pbf { url, .. } => {
                let url = url.clone();
                self.fetch.get(
                    &url.clone(),
                    Box::new(move |fetched| match fetched {
                        Ok(bytes) => executor.spawn(Box::new(move || done(request.run(&bytes)))),
                        Err(e) => done(Err(format!("glyphs {url}: {e}"))),
                    }),
                );
            }
            GlyphRaster::Local { .. } => {
                match self.local.lock().expect("local glyphs poisoned").clone() {
                    Some(local) => {
                        executor.spawn(Box::new(move || done(request.draw(local.as_ref()))));
                    }
                    None => done(Err("this host draws no local glyphs".into())),
                }
            }
            // The workers share the engine, which opened the file when a
            // tile resolved its glyphs.
            GlyphRaster::File { .. } => {
                match self.files.lock().expect("font files poisoned").clone() {
                    Some(files) => executor.spawn(Box::new(move || {
                        done(request.draw_file(files.as_ref(), None))
                    })),
                    None => done(Err("this host draws no font files".into())),
                }
            }
        }
    }

    fn features(&self, request: crate::FeatureRequest, done: crate::FeaturesDone) {
        self.query(request, done);
    }

    fn load(&self, request: TileRequest, done: TileDone) {
        let worker = self
            .styles
            .lock()
            .expect("styles poisoned")
            .iter()
            .find(|(id, _)| *id == request.style)
            .map(|(_, w)| Arc::clone(w));
        let Some(worker) = worker else {
            done(Err(format!("style {} was replaced", request.style)));
            return;
        };
        let glyphs = Arc::clone(&self.glyphs);
        if worker.has_data(&request) {
            self.executor.spawn(Box::new(move || {
                let encoding = request.encoding;
                glyphs.complete(worker, Arc::default(), encoding, request, done);
            }));
            return;
        }
        if worker.geojson_source(request.source).is_some() {
            let key = (request.style, request.source);
            let mut loads = self.geojson_loads.lock().expect("GeoJSON loads poisoned");
            if let Some(waiting) = loads.get_mut(&key) {
                waiting.push((request, done));
                return;
            }
            // Recheck while holding the fetch registry lock: a previous load
            // may have initialized the index between the checks.
            if worker.has_data(&request) {
                drop(loads);
                self.executor.spawn(Box::new(move || {
                    let encoding = request.encoding;
                    glyphs.complete(worker, Arc::default(), encoding, request, done);
                }));
                return;
            }
            let TileLocation::Url(url) = &request.location else {
                done(Err("GeoJSON source requires a document URL".into()));
                return;
            };
            let url = url.clone();
            loads.insert(key, vec![(request, done)]);
            drop(loads);
            let loads = Arc::clone(&self.geojson_loads);
            let executor = Arc::clone(&self.executor);
            self.fetch.get(
                &url,
                Box::new(move |fetched| {
                    executor.spawn(Box::new(move || {
                        // Initialize the index before removing the in-flight
                        // entry, without holding the registry during CPU work.
                        let (request, done) = loads
                            .lock()
                            .expect("GeoJSON loads poisoned")
                            .get_mut(&key)
                            .expect("registered fetch")
                            .pop()
                            .expect("first request");
                        let fetched = fetched.map(Arc::new).map_err(|e| e.to_string());
                        let build = |request: TileRequest, done: TileDone, bytes| match &fetched {
                            Ok(_) => glyphs.complete(
                                Arc::clone(&worker),
                                bytes,
                                request.encoding,
                                request,
                                done,
                            ),
                            Err(e) => done(Err(e.clone())),
                        };
                        build(request, done, fetched.clone().unwrap_or_default());
                        let waiting = loads
                            .lock()
                            .expect("GeoJSON loads poisoned")
                            .remove(&key)
                            .expect("registered fetch");
                        // The index is built: later requests ignore bytes.
                        for (request, done) in waiting {
                            build(request, done, Arc::default());
                        }
                    }));
                }),
            );
            return;
        }
        let executor = Arc::clone(&self.executor);
        if let Some(kind) = worker.image_kind(request.source) {
            let decode = Arc::clone(&self.decode);
            let dem_unpack = request.dem_unpack;
            self.loader.load(
                request.location,
                request.tile,
                request.encoding,
                Box::new(move |fetched| match fetched {
                    Ok((bytes, _)) if bytes.is_empty() => done(Ok(TileResult::default())),
                    Ok((bytes, _)) => executor.spawn(Box::new(move || {
                        decode.decode(
                            bytes,
                            Box::new(move |decoded| {
                                done(decoded.and_then(|i| image_result(kind, i, dem_unpack)))
                            }),
                        );
                    })),
                    Err(e) => done(Err(e).or_else(missing)),
                }),
            );
            return;
        }
        let data_tile = request.data_tile();
        self.loader.load(
            request.location.clone(),
            data_tile,
            request.encoding,
            Box::new(move |fetched| {
                executor.spawn(Box::new(move || match fetched {
                    Ok((bytes, encoding)) => {
                        glyphs.complete(worker, Arc::new(bytes), encoding, request, done)
                    }
                    Err(e) => done(Err(e).or_else(missing)),
                }));
            }),
        );
    }
}

impl PoolJobs {
    fn worker(&self, style: u64) -> Option<Arc<TileWorker>> {
        self.styles
            .lock()
            .expect("styles poisoned")
            .iter()
            .find(|(id, _)| *id == style)
            .map(|(_, w)| Arc::clone(w))
    }

    /// [`Jobs::features`]: from the cached frame, or the tile fetched again.
    fn query(&self, request: crate::FeatureRequest, done: crate::FeaturesDone) {
        let Some(worker) = self.worker(request.tile.style) else {
            done(Err(format!("style {} was replaced", request.tile.style)));
            return;
        };
        let executor = Arc::clone(&self.executor);
        if worker.has_data(&request.tile) {
            executor.spawn(Box::new(move || {
                done(worker.features(&[], request.tile.encoding, &request));
            }));
            return;
        }
        let tile = &request.tile;
        self.loader.load(
            tile.location.clone(),
            tile.data_tile(),
            tile.encoding,
            Box::new(move |fetched| {
                executor.spawn(Box::new(move || match fetched {
                    Ok((bytes, encoding)) => done(worker.features(&bytes, encoding, &request)),
                    Err(FetchError::NotFound) => done(Ok(Vec::new())),
                    Err(e) => done(Err(e.to_string())),
                }));
            }),
        );
    }
}

/// A tile the source doesn't have (HTTP 404) is [`TileResult::missing`];
/// other fetch errors are errors.
fn missing(error: FetchError) -> Result<TileResult, String> {
    match error {
        FetchError::NotFound => Ok(TileResult {
            missing: true,
            ..TileResult::default()
        }),
        e => Err(e.to_string()),
    }
}

/// A decoded image tile as a result: raster colours premultiplied, as GL
/// JS uploads them; DEM texels as decoded, with their elevations when
/// terrain asked for them.
pub(crate) fn image_result(
    kind: ImageKind,
    mut image: DecodedImage,
    dem_unpack: Option<DemUnpack>,
) -> Result<TileResult, String> {
    if kind == ImageKind::Raster {
        premultiply(&mut image.pixels);
    }
    let dem = match dem_unpack {
        Some(DemUnpack(unpack)) if kind == ImageKind::Dem => {
            if image.width != image.height {
                return Err("DEM tiles must be square".into());
            }
            Some(Arc::new(orb_tile::DemGrid::from_rgba(
                image.width,
                &image.pixels,
                unpack,
            )?))
        }
        _ => None,
    };
    Ok(TileResult {
        dem,
        image: Some(TileImage {
            width: image.width,
            height: image.height,
            pixels: ImagePixels::Core(image.pixels),
        }),
        ..TileResult::default()
    })
}

/// GL JS's `premultiplyAlpha`, in place on RGBA8 texels.
fn premultiply(pixels: &mut [u8]) {
    for texel in pixels.as_chunks_mut::<4>().0 {
        let a = u32::from(texel[3]);
        for c in &mut texel[..3] {
            *c = ((u32::from(*c) * a + 127) / 255) as u8;
        }
    }
}

/// A finished tile, ready to upload: in core memory, or in host-held bytes.
#[derive(Debug, Default)]
pub struct TileResult {
    pub warnings: Vec<String>,
    /// The style revision the tile was laid out with.
    pub revision: u64,
    /// Keys of the nodes laid out; those without a mesh have no features.
    pub nodes: Vec<u32>,
    /// Meshes carry property rows only (a column job).
    pub rows_only: bool,
    /// Feature ids the meshes' state ids index.
    pub ids: Vec<String>,
    pub meshes: Vec<ResultMesh>,
    /// Per symbol layer laid out, its bucket. Its icon and text nodes'
    /// meshes carry property rows only.
    pub symbols: Vec<ResultSymbols>,
    /// Holds every [`Stream::Host`] of the meshes and symbols; released
    /// after upload.
    pub host_bytes: Option<HostBytesId>,
    /// An image source's tile.
    pub image: Option<TileImage>,
    /// A terrain DEM tile's elevations.
    pub dem: Option<Arc<orb_tile::DemGrid>>,
    /// The tileset doesn't have the tile; nearby zooms stand in.
    pub missing: bool,
}

/// An image tile's texels, rows top to bottom.
#[derive(Debug)]
pub struct TileImage {
    pub width: u32,
    pub height: u32,
    pub pixels: ImagePixels,
}

#[derive(Debug)]
pub enum ImagePixels {
    /// RGBA8 in core memory, ready to store: premultiplied for rasters.
    Core(Vec<u8>),
    /// A host-held image, straight alpha; released after upload.
    Host(HostImageId),
}

/// One render node's share of a finished tile.
#[derive(Debug)]
pub struct ResultMesh {
    pub hits: Vec<crate::HitFeature>,
    /// The node's key.
    pub node: u32,
    pub extent: u32,
    pub chunks: Vec<Chunk>,
    pub vertices: Stream,
    pub indices: Stream,
    pub props: Stream,
    /// Words per feature of the property rows.
    pub row_words: u32,
    /// Per paint property, the largest number in the property rows; then
    /// the input columns' ranges (see `NodeMesh`).
    pub maxima: Vec<f32>,
    pub images: Vec<ImageRef>,
    /// Per feature slot, an index into [`TileResult::ids`] (or `u32::MAX`),
    /// for nodes that read feature state.
    pub state_ids: Option<Vec<u32>>,
}

/// A stream of u32 words.
#[derive(Debug, PartialEq, Eq)]
pub enum Stream {
    Core(Vec<u32>),
    /// `len` words at byte `offset` of the result's host bytes.
    Host {
        offset: u64,
        len: u32,
    },
}

impl From<TileBuild> for TileResult {
    fn from(build: TileBuild) -> Self {
        Self {
            meshes: build
                .meshes
                .into_iter()
                .map(|m| ResultMesh {
                    node: m.node,
                    hits: m.hits,
                    extent: m.extent,
                    chunks: m.mesh.chunks,
                    vertices: Stream::Core(m.mesh.vertices),
                    indices: Stream::Core(m.mesh.indices),
                    props: Stream::Core(m.props),
                    row_words: m.row_words,
                    maxima: m.maxima,
                    images: m.images,
                    state_ids: m.state_ids,
                })
                .collect(),
            symbols: build
                .symbols
                .iter()
                .map(|(layer, s)| ResultSymbols::new(*layer, s))
                .collect(),
            revision: build.revision,
            nodes: build.nodes,
            rows_only: build.rows_only,
            ids: build.ids,
            host_bytes: None,
            warnings: build.warnings,
            image: None,
            dem: None,
            missing: false,
        }
    }
}

/// Packs a build into one buffer for another worker: a header of u32 words
/// (its own length first; the revision as two words, the rows-only flag,
/// the laid-out node keys and the feature id strings; then per mesh: node,
/// extent, chunk count, stream lengths, row words, images, state ids, maxima and
/// chunks), then each mesh's vertices, indices and property rows. After the
/// mesh header, diagnostics are a count followed by each UTF-8 byte length
/// and its bytes packed into padded words. Little-endian.
pub fn pack(build: &TileBuild) -> Vec<u8> {
    let mut header: Vec<u32> = vec![
        0,
        build.revision as u32,
        (build.revision >> 32) as u32,
        u32::from(build.rows_only),
        build.nodes.len() as u32,
    ];
    header.extend(&build.nodes);
    header.push(build.ids.len() as u32);
    for id in &build.ids {
        write_text(&mut header, id);
    }
    header.push(build.meshes.len() as u32);
    for m in &build.meshes {
        header.push(m.hits.len() as u32);
        for h in &m.hits {
            header.push(u32::from(h.id.is_some()));
            if let Some(id) = &h.id {
                write_text(&mut header, id);
            }
            header.extend([
                h.feature,
                h.kind,
                h.extent,
                h.radius.to_bits(),
                h.stroke.to_bits(),
                h.height.to_bits(),
                h.base.to_bits(),
                h.parts.len() as u32,
            ]);
            for part in &h.parts {
                header.push(part.len() as u32);
                for ring in part {
                    header.push(ring.len() as u32);
                    for point in ring {
                        header.extend(point.map(|v| v as u32));
                    }
                }
            }
        }
        header.extend([
            m.node,
            m.extent,
            m.mesh.chunks.len() as u32,
            m.mesh.vertices.len() as u32,
            m.mesh.indices.len() as u32,
            m.props.len() as u32,
            m.row_words,
            m.maxima.len() as u32,
        ]);
        header.push(m.images.len() as u32);
        for image in &m.images {
            header.push(image.word);
            match &image.key {
                orb_render::ImageKey::Name(name) => {
                    header.push(0);
                    write_text(&mut header, name);
                }
                orb_render::ImageKey::Dash { array, round } => {
                    header.extend([1, u32::from(*round), array.len() as u32]);
                    header.extend(array);
                }
                orb_render::ImageKey::Ramp { colors, nearest } => {
                    header.extend([2, u32::from(*nearest), colors.len() as u32]);
                    header.extend(colors);
                }
            }
        }
        match &m.state_ids {
            None => header.push(u32::MAX),
            Some(ids) => {
                header.push(ids.len() as u32);
                header.extend(ids);
            }
        }
        header.extend(m.maxima.iter().map(|v| v.to_bits()));
        for c in &m.mesh.chunks {
            let pack = |p: [i16; 2]| u32::from(p[0] as u16) | u32::from(p[1] as u16) << 16;
            header.extend([
                c.first_index,
                c.index_count,
                pack(c.bounds.min),
                pack(c.bounds.max),
                u32::from(c.sort_key.is_some()),
                c.sort_key.unwrap_or(0.0).to_bits(),
            ]);
        }
    }
    let symbols: Vec<ResultSymbols> = build
        .symbols
        .iter()
        .map(|(layer, s)| ResultSymbols::new(*layer, s))
        .collect();
    header.push(symbols.len() as u32);
    for s in &symbols {
        crate::symbol::write_header(&mut header, s);
    }
    header.push(build.warnings.len() as u32);
    for warning in &build.warnings {
        write_text(&mut header, warning);
    }
    header[0] = header.len() as u32;
    let mut words = header;
    for m in &build.meshes {
        words.extend_from_slice(&m.mesh.vertices);
        words.extend_from_slice(&m.mesh.indices);
        words.extend_from_slice(&m.props);
    }
    for s in &symbols {
        for table in &s.tables {
            let Stream::Core(table) = table else {
                unreachable!("built tables are in core memory")
            };
            words.extend_from_slice(table);
        }
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub(crate) fn write_text(words: &mut Vec<u32>, text: &str) {
    words.push(text.len() as u32);
    for chunk in text.as_bytes().chunks(4) {
        let mut bytes = [0; 4];
        bytes[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(bytes));
    }
}

pub(crate) fn read_text(next: &mut dyn FnMut() -> Result<u32, String>) -> Result<String, String> {
    let len = next()? as usize;
    let mut bytes = Vec::new();
    for _ in 0..len.div_ceil(4) {
        bytes.extend_from_slice(&next()?.to_le_bytes());
    }
    bytes.truncate(len);
    String::from_utf8(bytes).map_err(|_| "malformed packed string".into())
}

/// Reads a packed tile's header (see [`pack`]): its meshes with
/// [`Stream::Host`] streams into the packed buffer, registered as `bytes`.
/// `header` holds at least the header's words; the payload need not be in
/// core memory.
pub fn unpack(header: &[u32], bytes: HostBytesId) -> Result<TileResult, String> {
    let bad = || "malformed packed tile".to_string();
    let length = *header.first().ok_or_else(bad)? as usize;
    let header = header.get(..length).ok_or_else(bad)?;
    let mut words = header.iter().copied();
    let mut next = || words.next().ok_or_else(bad);
    let header_len = next()?;
    let revision = u64::from(next()?) | u64::from(next()?) << 32;
    let rows_only = next()? != 0;
    let nodes = (0..next()?).map(|_| next()).collect::<Result<_, _>>()?;
    let ids = (0..next()?)
        .map(|_| read_text(&mut next))
        .collect::<Result<_, _>>()?;
    let count = next()?;
    let mut offset = u64::from(header_len) * 4;
    let mut meshes = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let mut hits = Vec::new();
        for _ in 0..next()? {
            let id = if next()? != 0 {
                Some(read_text(&mut next)?)
            } else {
                None
            };
            let feature = next()?;
            let kind = next()?;
            let extent = next()?;
            let radius = f32::from_bits(next()?);
            let stroke = f32::from_bits(next()?);
            let height = f32::from_bits(next()?);
            let base = f32::from_bits(next()?);
            let mut parts = Vec::new();
            for _ in 0..next()? {
                let mut rings = Vec::new();
                for _ in 0..next()? {
                    let mut points = Vec::new();
                    for _ in 0..next()? {
                        points.push([next()? as i32, next()? as i32]);
                    }
                    rings.push(points);
                }
                parts.push(rings);
            }
            hits.push(crate::HitFeature {
                id,
                feature,
                kind,
                extent,
                radius,
                stroke,
                height,
                base,
                parts,
            });
        }
        let (node, extent, chunks, vertices, indices, props, row_words, maxima) = (
            next()?,
            next()?,
            next()?,
            next()?,
            next()?,
            next()?,
            next()?,
            next()?,
        );
        let images = (0..next()?)
            .map(|_| {
                Ok(ImageRef {
                    word: next()?,
                    key: match next()? {
                        0 => orb_render::ImageKey::Name(read_text(&mut next)?),
                        tag @ (1 | 2) => {
                            let flag = next()? != 0;
                            let data = (0..next()?)
                                .map(|_| next())
                                .collect::<Result<Vec<_>, _>>()?;
                            if tag == 1 {
                                orb_render::ImageKey::Dash {
                                    array: data,
                                    round: flag,
                                }
                            } else {
                                orb_render::ImageKey::Ramp {
                                    colors: data,
                                    nearest: flag,
                                }
                            }
                        }
                        _ => return Err(bad()),
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let state_ids = match next()? {
            u32::MAX => None,
            n => Some((0..n).map(|_| next()).collect::<Result<Vec<_>, _>>()?),
        };
        let maxima = (0..maxima)
            .map(|_| next().map(f32::from_bits))
            .collect::<Result<_, _>>()?;
        let unpack = |w: u32| [w as u16 as i16, (w >> 16) as u16 as i16];
        let chunks = (0..chunks)
            .map(|_| {
                Ok(Chunk {
                    first_index: next()?,
                    index_count: next()?,
                    bounds: TileBox {
                        min: unpack(next()?),
                        max: unpack(next()?),
                    },
                    sort_key: {
                        let present = next()? != 0;
                        let value = f32::from_bits(next()?);
                        present.then_some(value)
                    },
                })
            })
            .collect::<Result<_, String>>()?;
        let mut stream = |len: u32| {
            let s = Stream::Host { offset, len };
            offset += u64::from(len) * 4;
            s
        };
        meshes.push(ResultMesh {
            hits,
            node,
            extent,
            chunks,
            vertices: stream(vertices),
            indices: stream(indices),
            props: stream(props),
            row_words,
            maxima,
            images,
            state_ids,
        });
    }
    let symbols = (0..next()?)
        .map(|_| crate::symbol::read_header(&mut next, &mut offset))
        .collect::<Result<Vec<_>, _>>()?;
    let warning_count = next()?;
    let mut warnings = Vec::new();
    for _ in 0..warning_count {
        warnings.push(read_text(&mut next)?);
    }
    if words.next().is_some() {
        return Err(bad());
    }
    Ok(TileResult {
        revision,
        nodes,
        rows_only,
        ids,
        meshes,
        symbols,
        host_bytes: Some(bytes),
        warnings,
        image: None,
        dem: None,
        missing: false,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tile::NodeMesh;
    use orb_layout::Mesh;

    fn request(tile: TileIndex) -> TileRequest {
        TileRequest {
            style: 1,
            source: 1,
            tile,
            zoom: f64::from(tile.z),
            data_zoom: tile.z,
            location: TileLocation::Url("test://data.json".into()),
            encoding: Encoding::Mvt,
            nodes: None,
            rows_only: false,
            dem_unpack: None,
        }
    }

    const INLINE: &str = r#"{"version":8,"sources":{"geo":{"type":"geojson","maxzoom":2,"data":{"type":"Feature","properties":{"color":"red"},"geometry":{"type":"Polygon","coordinates":[[[-20,-20],[20,-20],[20,20],[-20,20],[-20,-20]]]}}}},"layers":[{"id":"fill","type":"fill","source":"geo","paint":{"fill-color":["get","color"]}},{"id":"line","type":"line","source":"geo"}]}"#;

    /// Thai labels wait for the Southeast Asian word data where the host
    /// loads it lazily; natively it is compiled in.
    #[test]
    fn thai_labels_wait_for_lazy_word_data() {
        const THAI: &str = r#"{"version":8,"glyphs":"test://{fontstack}/{range}.pbf",
            "sources":{"geo":{"type":"geojson","data":{"type":"Feature",
                "properties":{"name":"ราชอาณาจักรไทย"},"geometry":{"type":"Point","coordinates":[0,0]}}}},
            "layers":[{"id":"t","type":"symbol","source":"geo","layout":{"text-field":"{name}"}}]}"#;
        let tile = TileIndex::new(0, 0, 0);
        let compiled = TileWorker::new(THAI).unwrap();
        assert!(compiled.word_breaker().is_some());
        assert!(matches!(
            compiled.build(&[], Encoding::Mvt, &request(tile)).unwrap(),
            Built::Tile(_)
        ));

        let lazy = TileWorker::new(THAI).unwrap();
        *lazy.data.words.lock().unwrap() = Words::Unavailable;
        lazy.set_word_data_url("test://seasia.bin");
        let Built::Fetch(urls) = lazy.build(&[], Encoding::Mvt, &request(tile)).unwrap() else {
            panic!("Thai text waits for word data")
        };
        assert_eq!(urls, ["test://seasia.bin"]);
        lazy.add_resource(&urls[0], Ok(&orb_text::sea_dictionary_bytes()));
        assert!(lazy.word_breaker().is_some());
        assert!(matches!(
            lazy.build(&[], Encoding::Mvt, &request(tile)).unwrap(),
            Built::Tile(_)
        ));
    }

    #[test]
    fn lays_out_inline_geojson_and_reuses_index() {
        let worker = TileWorker::new(INLINE).unwrap();
        let tile = TileIndex::new(0, 0, 0);
        assert!(worker.has_data(&request(tile)));
        let build = worker
            .build(&[], Encoding::Mvt, &request(tile))
            .unwrap()
            .tile();
        assert_eq!(build.meshes.len(), 3);
        assert!(
            build
                .meshes
                .iter()
                .all(|m| m.extent == 8192 && !m.mesh.is_empty())
        );
        assert_eq!(
            &build.meshes[0].props[..2],
            &[
                orb_render::pack_color([1.0, 0.0, 0.0, 1.0]),
                orb_render::pack_color([1.0, 0.0, 0.0, 1.0])
            ]
        );
        // Subsequent bytes are ignored: the per-source index is authoritative.
        assert_eq!(
            pack(&build),
            pack(
                &worker
                    .build(b"invalid", Encoding::Mlt, &request(tile))
                    .unwrap()
                    .tile()
            )
        );
        // A column job re-evaluates the fill's rows from the cached tile.
        let rows = worker
            .build(
                &[],
                Encoding::Mvt,
                &TileRequest {
                    nodes: Some(vec![2]),
                    rows_only: true,
                    ..request(tile)
                },
            )
            .unwrap()
            .tile();
        assert_eq!(rows.nodes, [2]);
        assert_eq!(rows.meshes.len(), 1);
        assert!(rows.meshes[0].mesh.is_empty());
        assert_eq!(rows.meshes[0].props, build.meshes[0].props);
    }

    /// Tests load no image sources.
    pub(crate) struct NoImages;

    impl ImageDecode for NoImages {
        fn decode(&self, _: Vec<u8>, _: orb_host::ImageDone) {
            unreachable!("no image sources");
        }
    }

    #[test]
    fn premultiplies_like_gl_js() {
        let mut texels = [255, 128, 3, 128, 10, 20, 30, 0, 9, 8, 7, 255];
        premultiply(&mut texels);
        // Math.round(c * a / 255).
        assert_eq!(texels, [128, 64, 2, 128, 0, 0, 0, 0, 9, 8, 7, 255]);
    }

    #[derive(Default)]
    struct DeferredFetch {
        callbacks: Mutex<Vec<orb_host::FetchDone>>,
    }
    impl Fetch for DeferredFetch {
        fn get_range(&self, _: &str, _: u64, _: u64, _: orb_host::FetchDone) {
            panic!("GeoJSON must fetch its whole document");
        }

        fn get(&self, url: &str, done: orb_host::FetchDone) {
            assert_eq!(url, "test://data.json");
            self.callbacks.lock().unwrap().push(done);
        }
    }

    #[test]
    fn url_data_is_fetched_once_for_concurrent_tiles() {
        let inline = orb_expr::json::parse(INLINE).unwrap();
        let data = inline
            .get("sources")
            .unwrap()
            .get("geo")
            .unwrap()
            .get("data")
            .unwrap()
            .to_string();
        let style = INLINE.replace(&data, "\"test://data.json\"");
        let worker = Arc::new(TileWorker::new(&style).unwrap());
        assert!(!worker.has_data(&request(TileIndex::new(1, 0, 0))));
        let fetch = Arc::new(DeferredFetch::default());
        let jobs = PoolJobs::new(
            fetch.clone(),
            Arc::new(NoImages),
            Arc::new(orb_jobs::Inline),
        );
        jobs.set_style(1, &StyleUpdate::default(), &worker);
        let (tx, rx) = std::sync::mpsc::channel();
        for col in 0..2 {
            let tx = tx.clone();
            jobs.load(
                request(TileIndex::new(1, col, 0)),
                Box::new(move |r| tx.send(r).unwrap()),
            );
        }
        let callback = fetch.callbacks.lock().unwrap().pop().unwrap();
        assert!(fetch.callbacks.lock().unwrap().is_empty());
        callback(Ok(data.into_bytes()));
        assert!(worker.has_data(&request(TileIndex::new(0, 0, 0))));
        assert_eq!(
            rx.try_iter()
                .filter(|r| r.as_ref().is_ok_and(|t| !t.meshes.is_empty()))
                .count(),
            2
        );
        let tx = tx.clone();
        jobs.load(
            request(TileIndex::new(0, 0, 0)),
            Box::new(move |r| tx.send(r).unwrap()),
        );
        assert!(rx.try_recv().unwrap().is_ok());
        assert!(fetch.callbacks.lock().unwrap().is_empty());
    }

    fn polar_plane() -> TileMatrixSet {
        let projection = orb_crs::Projection::from_descriptor("EPSG:3413").unwrap();
        TileMatrixSet::square(
            projection,
            orb_crs::WorldRect {
                min: glam::DVec2::splat(-4_194_304.0),
                max: glam::DVec2::splat(4_194_304.0),
            },
        )
    }

    #[test]
    fn geojson_tiles_on_the_surface_plane() {
        let style = r#"{"version":8,"sources":{"geo":{"type":"geojson","data":{"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[0,90]}},
            {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[0,-60]}}]}}},
            "layers":[{"id":"c","type":"circle","source":"geo"}]}"#;
        let plain = TileWorker::new(style).unwrap();
        let worker = TileWorker {
            plane: Some(polar_plane()),
            ..plain
        };
        let tile = request(TileIndex::new(0, 0, 0));
        let frame = worker
            .decode(&[], Encoding::Mvt, 0, &worker.data(1), tile.tile, 0)
            .unwrap();
        let layer = frame.layer(orb_tile::geojson::LAYER_NAME).unwrap();
        // The pole is the plane's centre; the southern point is outside the
        // projection's domain.
        assert_eq!(layer.len(), 1);
        let part = layer.geometry(0).parts().next().unwrap();
        assert_eq!(part.rings().next().unwrap(), [[4096, 4096]]);
    }

    #[test]
    fn geojson_errors_are_terminal_and_clusters_skipped() {
        let worker =
            TileWorker::new(&INLINE.replace("\"data\":{", "\"cluster\":true,\"data\":{")).unwrap();
        assert!(worker.plan.nodes.is_empty());
        // Updates round-trip through JSON for job workers.
        let update = StyleUpdate {
            revision: 3,
            layers: vec![1],
            sources: vec![1],
            layer_json: vec![(
                1,
                Arc::new(
                    orb_expr::json::parse(r#"{"id":"l","type":"line","source":"geo"}"#).unwrap(),
                ),
            )],
            source_json: vec![(
                1,
                "geo".into(),
                Arc::new(
                    orb_expr::json::parse(r#"{"type":"geojson","data":"test://data.json"}"#)
                        .unwrap(),
                ),
            )],
            state: vec![("k".into(), Value::Number(2.0))],
            glyphs: Some("test://glyphs/{fontstack}/{range}.pbf".into()),
            font_faces: vec![(
                "Noto Sans".into(),
                vec![orb_v8::FontFace {
                    url: "test://noto.woff2".into(),
                    ranges: vec![[0, 0x24F], [0x600, 0x6FF]],
                }],
            )],
            local_ideographs: Some("Hiragino Sans".into()),
            cost: CostModel { max_ops: 7 },
            plane: Some(polar_plane()),
        };
        let parsed = StyleUpdate::parse(&update.to_json()).unwrap();
        assert_eq!(parsed.to_json(), update.to_json());
        assert_eq!(parsed.plane, update.plane);
        let revised = TileWorker::update(None, &parsed).unwrap();
        assert_eq!(revised.revision(), 3);
        assert_eq!(revised.plan.nodes.len(), 1);
        assert!(!worker.plan.style.warnings.is_empty());
        let data = orb_expr::json::parse(INLINE)
            .unwrap()
            .get("sources")
            .unwrap()
            .get("geo")
            .unwrap()
            .get("data")
            .unwrap()
            .to_string();
        let worker = TileWorker::new(&INLINE.replace(&data, "\"test://data.json\"")).unwrap();
        let tile = request(TileIndex::new(0, 0, 0));
        assert!(worker.build(b"invalid", Encoding::Mvt, &tile).is_err());
        assert!(worker.has_data(&tile));
        assert!(worker.build(b"{}", Encoding::Mvt, &tile).is_err());
    }

    /// Tiles that waited on one word data fetch each hand it back; the
    /// second is not a glyph range.
    #[test]
    fn word_data_arriving_twice_is_not_a_glyph_range() {
        let worker = TileWorker::new(INLINE).unwrap();
        *worker.data.words.lock().unwrap() = Words::Unavailable;
        worker.set_word_data_url("test://seasia.bin");
        worker.add_resource("test://seasia.bin", Err("offline".into()));
        worker.add_resource("test://seasia.bin", Err("offline".into()));
        let warnings = worker.data.glyphs.take_warnings();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].starts_with("word data"), "{warnings:?}");
    }

    #[test]
    fn url_geojson_keeps_nested_properties() {
        let source = orb_expr::json::parse(INLINE).unwrap();
        let data = source
            .get("sources")
            .unwrap()
            .get("geo")
            .unwrap()
            .get("data")
            .unwrap()
            .to_string();
        let style = INLINE.replace(&data, "\"test://data.json\"");
        let data = data.replace(
            "\"color\":\"red\"",
            "\"color\":\"red\",\"nested\":{\"a\":[1]}",
        );
        let worker = TileWorker::new(&style).unwrap();
        let tile = request(TileIndex::new(0, 0, 0));
        let build = worker
            .build(data.as_bytes(), Encoding::Mvt, &tile)
            .unwrap()
            .tile();
        assert!(build.warnings.is_empty(), "{:?}", build.warnings);
    }

    #[test]
    fn packs_and_unpacks() {
        let build = TileBuild {
            symbols: Vec::new(),
            revision: 1 << 33 | 5,
            nodes: vec![3, 4, 9],
            rows_only: false,
            ids: vec!["a".into(), "1.5".into()],
            warnings: vec![
                "source geo: nested object and array property values become strings (é)".into(),
            ],
            meshes: vec![
                NodeMesh {
                    hits: vec![crate::HitFeature {
                        id: Some("promoted é".into()),
                        feature: 7,
                        kind: 0,
                        extent: 4096,
                        radius: 2.0,
                        stroke: 1.0,
                        height: 40.0,
                        base: 2.0,
                        parts: vec![vec![vec![[-3, 5], [30, 50], [30, 5]]]],
                    }],
                    node: 3,
                    extent: 4096,
                    mesh: Mesh {
                        vertices: vec![1, 2, 3, 4],
                        indices: vec![0, 1, 0],
                        chunks: vec![Chunk {
                            first_index: 0,
                            index_count: 3,
                            sort_key: Some(2.5),
                            bounds: TileBox {
                                min: [-5, 0],
                                max: [10, 4096],
                            },
                        }],
                    },
                    props: vec![9; 12],
                    row_words: 6,
                    maxima: vec![0.0, 2.5],
                    images: vec![
                        ImageRef {
                            word: 0,
                            key: orb_render::ImageKey::Name("extra:pattern-é".into()),
                        },
                        ImageRef {
                            word: 3,
                            key: orb_render::ImageKey::Dash {
                                array: vec![1.0f32.to_bits(), 2.0f32.to_bits()],
                                round: true,
                            },
                        },
                        ImageRef {
                            word: 4,
                            key: orb_render::ImageKey::Ramp {
                                colors: vec![0xff0000ff; 256],
                                nearest: true,
                            },
                        },
                    ],
                    state_ids: Some(vec![1, crate::tile::NO_ID, 0]),
                },
                NodeMesh {
                    hits: Vec::new(),
                    node: 4,
                    extent: 8192,
                    mesh: Mesh::default(),
                    props: vec![],
                    row_words: 0,
                    maxima: vec![],
                    images: vec![],
                    state_ids: None,
                },
            ],
        };
        let bytes = pack(&build);
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let result = unpack(&words, HostBytesId(7)).unwrap();
        assert_eq!(result.host_bytes, Some(HostBytesId(7)));
        assert_eq!(result.meshes[0].hits, build.meshes[0].hits);
        assert_eq!(result.warnings, build.warnings);
        assert_eq!(
            (result.revision, &result.nodes, &result.ids),
            (build.revision, &build.nodes, &build.ids)
        );
        assert_eq!(result.meshes[0].state_ids, build.meshes[0].state_ids);
        assert_eq!(result.meshes[1].state_ids, None);
        let m = &result.meshes[0];
        assert_eq!(
            (m.node, m.extent, m.row_words, m.maxima.clone()),
            (3, 4096, 6, vec![0.0, 2.5])
        );
        assert_eq!(m.chunks, build.meshes[0].mesh.chunks);
        assert_eq!(m.images, build.meshes[0].images);
        let Stream::Host { offset, len } = m.vertices else {
            panic!("host stream")
        };
        let at = offset as usize / 4;
        assert_eq!(&words[at..at + len as usize], [1, 2, 3, 4]);
        let Stream::Host { offset, len } = m.props else {
            panic!("host stream")
        };
        assert_eq!(&words[offset as usize / 4..][..len as usize], [9; 12]);
        assert_eq!(result.meshes[1].node, 4);
        assert!(unpack(&words[..3], HostBytesId(0)).is_err());
    }

    use crate::testing::{ASSETS, SYMBOL_TILE, SYMBOLS};

    /// Serves the committed test data for `local://` URLs, counting gets.
    /// Glyph ranges are held until released.
    #[derive(Default)]
    struct AssetFetch {
        gets: Mutex<Vec<String>>,
        held: Mutex<Vec<(String, orb_host::FetchDone)>>,
    }

    impl AssetFetch {
        fn release(&self) {
            let held = std::mem::take(&mut *self.held.lock().unwrap());
            for (path, done) in held {
                done(std::fs::read(path).map_err(|_| FetchError::NotFound));
            }
        }
    }

    impl Fetch for AssetFetch {
        fn get_range(&self, _: &str, _: u64, _: u64, _: orb_host::FetchDone) {
            unreachable!("no archives")
        }

        fn get(&self, url: &str, done: orb_host::FetchDone) {
            self.gets.lock().unwrap().push(url.to_string());
            let path = format!("{ASSETS}/{}", url.strip_prefix("local://").unwrap());
            if url.contains("/glyphs/") {
                self.held.lock().unwrap().push((path, done));
            } else {
                done(std::fs::read(path).map_err(|_| FetchError::NotFound));
            }
        }
    }

    fn symbol_request() -> TileRequest {
        let (z, x, y) = SYMBOL_TILE;
        TileRequest {
            location: TileLocation::Url(format!("local://tiles/{z}-{x}-{y}.mvt")),
            ..request(TileIndex::new(z, x, y))
        }
    }

    #[test]
    fn symbol_tiles_wait_for_glyphs_and_pack() {
        let worker = TileWorker::new(SYMBOLS).unwrap();
        let (z, x, y) = SYMBOL_TILE;
        let bytes = std::fs::read(format!("{ASSETS}/tiles/{z}-{x}-{y}.mvt")).unwrap();
        let request = symbol_request();
        let Built::Fetch(urls) = worker.build(&bytes, Encoding::Mvt, &request).unwrap() else {
            panic!("labels need glyph ranges first")
        };
        assert_eq!(urls, ["local://glyphs/Noto Sans Regular/0-255.pbf"]);
        let path = format!("{ASSETS}/{}", urls[0].strip_prefix("local://").unwrap());
        worker.add_resource(&urls[0], Ok(&std::fs::read(path).unwrap()));
        // The decoded tile is cached; the second build needs no bytes.
        let build = worker.build(&[], Encoding::Mvt, &request).unwrap().tile();
        let (layer, symbols) = &build.symbols[0];
        assert_eq!(*layer, 1);
        let b = &symbols.bucket;
        assert!(!b.instances.is_empty() && b.icon.quads.is_empty());
        assert!(
            b.text
                .quads
                .iter()
                .all(|q| symbols.glyphs[q.texture as usize].is_some())
        );
        // The text node carries per-feature colour rows; the icon node has
        // no icons, so no mesh.
        let nodes: Vec<u32> = build.meshes.iter().map(|m| m.node).collect();
        assert_eq!(nodes, [3]);
        assert_eq!(
            build.meshes[0].props.len() % orb_render::SYMBOL_ROW_WORDS,
            0
        );
        assert_eq!(symbols.instance_rows.len(), b.instances.len());

        let packed = pack(&build);
        let words: Vec<u32> = packed
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let result = unpack(&words, HostBytesId(3)).unwrap();
        let core = TileResult::from(build);
        let (unpacked, original) = (&result.symbols[0], &core.symbols[0]);
        assert_eq!(unpacked.layer, original.layer);
        assert_eq!(unpacked.meta, original.meta);
        assert_eq!(unpacked.glyphs, original.glyphs);
        assert_eq!(unpacked.images, original.images);
        for (host, core) in unpacked.tables.iter().zip(&original.tables) {
            let (Stream::Host { offset, len }, Stream::Core(core)) = (host, core) else {
                panic!("host and core streams")
            };
            assert_eq!(&words[*offset as usize / 4..][..*len as usize], &core[..]);
        }
    }

    #[test]
    fn concurrent_tiles_fetch_each_glyph_range_once() {
        let fetch = Arc::new(AssetFetch::default());
        let jobs = PoolJobs::new(
            fetch.clone(),
            Arc::new(NoImages),
            Arc::new(orb_jobs::Inline),
        );
        let worker = Arc::new(TileWorker::new(SYMBOLS).unwrap());
        jobs.set_style(1, &StyleUpdate::default(), &worker);
        let (tx, rx) = std::sync::mpsc::channel();
        for _ in 0..2 {
            let tx = tx.clone();
            jobs.load(symbol_request(), Box::new(move |r| tx.send(r).unwrap()));
        }
        // Both tiles wait on the one fetch of their range.
        assert_eq!(rx.try_iter().count(), 0);
        fetch.release();
        let results: Vec<_> = rx.try_iter().collect();
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|r| r.as_ref().is_ok_and(|t| t.symbols.len() == 1))
        );
        let gets = fetch.gets.lock().unwrap();
        let glyph_gets = gets.iter().filter(|u| u.contains("/glyphs/")).count();
        assert_eq!(glyph_gets, 1, "{gets:?}");
    }

    #[test]
    fn sprite_icons_cross_workers() {
        let mut icons = SpriteIcons {
            pixel_ratio: 2.0,
            ..SpriteIcons::default()
        };
        icons.add(
            "extra:pill",
            [40, 20],
            &orb_render::ImageMetadata {
                pixel_ratio: 2.0,
                sdf: true,
                stretch_x: vec![[4.0, 36.0]],
                stretch_y: Vec::new(),
                content: Some([4.0, 2.0, 36.0, 18.0]),
                text_fit_width: Some("proportional".into()),
                text_fit_height: None,
            },
        );
        icons.add("dot", [8, 8], &orb_render::ImageMetadata::default());
        assert_eq!(SpriteIcons::parse(&icons.to_json()).unwrap(), icons);
    }
}

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
use orb_expr::json::Json;
use orb_hal::{HostBytesId, HostImageId};
use orb_host::{DecodedImage, Fetch, ImageDecode};
use orb_jobs::Executor;
use orb_layout::Chunk;
use orb_style::{Encoding, GeoJsonData, Scene, SourceKind, Value};
use orb_tile::TileFrame;

use crate::TileLocation;
use crate::glyphs::{GlyphCatalog, GlyphRaster, LocalGlyphs, RasterDone};
use crate::style::{ImageKind, StylePlan};
use crate::symbol::{ResultSymbols, SpriteIcons};
use crate::tile::{self, ImageRef, Select, TileBuild};

/// Decoded tiles a worker keeps per source, least recently used first out.
const FRAME_CACHE: usize = 256;

/// One source's data in a worker, shared by its style revisions.
#[derive(Debug, Default)]
struct SourceData {
    geojson: Mutex<Option<Result<crate::geojson::Index, String>>>,
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

/// A source's data revisions a worker keeps, oldest first: the style
/// revision each began at, and its index and decoded tiles. Queries read
/// the revision a drawn tile was laid out from (#89).
type SourceRevisions = Vec<(u64, Arc<SourceData>)>;

/// How many data revisions of a source a worker keeps: the drawn tiles'
/// and those coming.
const DATA_REVISIONS: usize = 3;

/// What a worker keeps across a style's revisions.
#[derive(Debug, Default)]
pub(crate) struct WorkerData {
    /// Per source key, kept while the source is.
    sources: Mutex<HashMap<u32, SourceRevisions>>,
    /// Glyph metrics of the PBF ranges fetched so far.
    pub(crate) glyphs: GlyphCatalog,
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
    pub(crate) data: Arc<WorkerData>,
    /// The map's `localIdeographFontFamily`.
    ideographs: Option<String>,
    /// The plane GeoJSON tiles in, and sources' tile matrix sets.
    surface: crate::SurfaceTiles,
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
            &[],
            0,
            None,
            crate::SurfaceTiles::default(),
        ))
    }

    /// A revision of a parsed style, keeping `previous`'s data for the
    /// sources both have.
    #[allow(clippy::too_many_arguments)]
    /// A revision of a style, keeping `previous`'s data for the cached tile
    /// frames but for the sources in `new_data` (GeoJSON data replaced),
    /// whose index and frames are made again from the new data.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_style(
        previous: Option<&TileWorker>,
        style: Arc<Scene>,
        layer_keys: Vec<u32>,
        source_keys: Vec<u32>,
        new_data: &[u32],
        revision: u64,
        ideographs: Option<String>,
        surface: crate::SurfaceTiles,
    ) -> Self {
        let plan = StylePlan::new(style, layer_keys, source_keys, revision);
        let data = match previous {
            Some(previous) => {
                let data = Arc::clone(&previous.data);
                let mut sources = data.sources.lock().expect("worker data poisoned");
                sources.retain(|key, _| plan.source(*key).is_some());
                for key in new_data {
                    let revisions = sources.entry(*key).or_default();
                    revisions.push((revision, Arc::default()));
                    if revisions.len() > DATA_REVISIONS {
                        revisions.remove(0);
                    }
                }
                drop(sources);
                data
            }
            None => Arc::default(),
        };
        Self {
            plan,
            data,
            ideographs,
            surface,
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
                Some((_, id, json)) => {
                    Arc::new(orb_v8::parse_source(id, json).map_err(|d| d.to_string())?.0)
                }
                None => old
                    .and_then(|plan| {
                        let at = plan.source_keys.iter().position(|k| k == key)?;
                        Some(plan.style.sources[at].clone())
                    })
                    .ok_or_else(|| format!("style update without source {key}"))?,
            };
            sources.push(source);
        }
        let dialect = match update.models {
            Some(_) => orb_style::Dialect::Scene,
            None => orb_style::Dialect::V8,
        };
        let mut layers = Vec::with_capacity(update.layers.len());
        for (index, key) in update.layers.iter().enumerate() {
            let layer = match update.layer_json.iter().find(|(k, _)| k == key) {
                Some((_, json)) => Arc::new(
                    orb_v8::parse_layer(json, index, &sources, &update.state, dialect)
                        .map_err(|d| d.to_string())?
                        .0,
                ),
                None => old
                    .and_then(|plan| {
                        let at = plan.layer_keys.iter().position(|k| k == key)?;
                        Some(plan.style.layers[at].clone())
                    })
                    .ok_or_else(|| format!("style update without layer {key}"))?,
            };
            layers.push(layer);
        }
        let style = Scene {
            dialect,
            models: update
                .models
                .iter()
                .flatten()
                .map(|id| (id.clone(), String::new()))
                .collect(),
            name: None,
            center: None,
            zoom: None,
            bearing: 0.0,
            pitch: 0.0,
            roll: 0.0,
            transition: Default::default(),
            sprites: Vec::new(),
            glyphs: update.glyphs.clone(),
            font_faces: update.font_faces.clone(),
            lights: orb_v8::default_lights(),
            atmosphere: None,
            terrain: None,
            globe: None,
            projection: None,
            tile_matrix_sets: Vec::new(),
            symbol_occlusion: false,
            state: Vec::new(),
            sources,
            layers,
            warnings: Vec::new(),
        };
        // A source sent again has new data (an added one has none yet).
        let new_data: Vec<u32> = update.source_json.iter().map(|(k, ..)| *k).collect();
        Ok(Self::with_style(
            previous,
            Arc::new(style),
            update.layers.clone(),
            update.sources.clone(),
            &new_data,
            update.revision,
            update.local_ideographs.clone(),
            update.surface.clone(),
        ))
    }

    pub(crate) fn plan(&self) -> &StylePlan {
        &self.plan
    }

    /// The plane GeoJSON tiles in and sources' tile matrix sets.
    /// Layout grid units (8192 a tile side) per ground metre at a tile's
    /// centre, in its source's tile matrix set: GL JS's
    /// `getTileUnitsForMeters`.
    fn units_per_metre(&self, source: usize, tile: TileIndex) -> f64 {
        let id = &self.plan.style.sources[source].id;
        let web_mercator;
        let tms = match self.surface.sources.iter().find(|(s, _)| s == id) {
            Some((_, tms)) => tms,
            None => {
                web_mercator = TileMatrixSet::by_id("WebMercatorQuad").expect("registered");
                &web_mercator
            }
        };
        let projection = tms.projection();
        let center = projection.inverse(tms.tile_bounds(tile).center());
        8192.0 * projection.scale_factor(center) / tms.tile_span(tile.z)
    }

    /// Where `within` and `distance` find a tile of `source`: the tile
    /// itself on Web Mercator's grid, the one they measure in; nowhere on
    /// other grids.
    pub(crate) fn canonical(
        &self,
        source: usize,
        tile: TileIndex,
    ) -> Option<orb_expr::CanonicalTile> {
        let id = &self.plan.style.sources[source].id;
        let mercator = match self.surface.sources.iter().find(|(s, _)| s == id) {
            Some((_, tms)) => tms.id() == "WebMercatorQuad",
            None => true,
        };
        mercator.then_some(orb_expr::CanonicalTile {
            z: tile.z,
            x: tile.col,
            y: tile.row,
        })
    }

    pub(crate) fn surface(&self) -> &crate::SurfaceTiles {
        &self.surface
    }

    /// The style revision this worker lays out.
    pub fn revision(&self) -> u64 {
        self.plan.revision
    }

    /// A source's newest data.
    fn data(&self, source: u32) -> Arc<SourceData> {
        self.data_at(source, None)
    }

    /// A source's data as of style revision `revision`: the newest that
    /// began by then, or the newest of all.
    fn data_at(&self, source: u32, revision: Option<u64>) -> Arc<SourceData> {
        let mut data = self.data.sources.lock().expect("worker data poisoned");
        let revisions = data.entry(source).or_default();
        if revisions.is_empty() {
            revisions.push((0, Arc::default()));
        }
        let at = revision
            .and_then(|r| revisions.iter().rposition(|(began, _)| *began <= r))
            .unwrap_or(revisions.len() - 1);
        Arc::clone(&revisions[at].1)
    }

    /// Whether `source` (a source key) has image tiles to decode rather
    /// than lay out, and of which kind: any of the style's sources, drawn
    /// by layers or by the terrain alone.
    pub fn image_kind(&self, source: u32) -> Option<ImageKind> {
        let at = self.plan.source_keys.iter().position(|&k| k == source)?;
        match self.plan.style.sources[at].kind {
            orb_style::SourceKind::Raster(_) | orb_style::SourceKind::Image(_) => {
                Some(ImageKind::Raster)
            }
            orb_style::SourceKind::RasterDem(_) => Some(ImageKind::Dem),
            _ => None,
        }
    }

    fn geojson_source(&self, source: u32) -> Option<&orb_style::GeoJsonSource> {
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
        // A warped tile's geometry follows the warp's curve; the cached
        // frame keeps the tile's own, for queries.
        let frame = match request.bend {
            Some(bend) if bend.cells > 1 => Arc::new(frame.subdivided(bend.cells)),
            _ => frame,
        };
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
        let bend = request.bend;
        match tile::build(
            &frame,
            &self.plan,
            index,
            (
                request.zoom,
                self.units_per_metre(index, request.tile),
                self.canonical(index, request.tile),
            ),
            bend,
            select,
            symbols,
        ) {
            Ok(mut build) => {
                warnings.extend(self.data.glyphs.take_warnings());
                build.warnings.splice(0..0, warnings);
                Ok(Built::Tile(build))
            }
            Err(missing) => Ok(Built::Fetch(missing)),
        }
    }

    /// A GeoJSON source's index, from its inline data or its document's
    /// `bytes`.
    fn geojson_index(
        &self,
        bytes: &[u8],
        geojson: &orb_style::GeoJsonSource,
    ) -> Result<crate::geojson::Index, String> {
        let features = match &geojson.data {
            GeoJsonData::Inline(features) => features.clone(),
            GeoJsonData::Url(_) => {
                orb_v8::parse_geojson(std::str::from_utf8(bytes).map_err(|e| e.to_string())?)?
            }
        };
        let plane = self.surface.plane.clone().map(Arc::new);
        crate::geojson::Index::new(features, geojson, plane)
    }

    /// Answers a cluster query from the source's index, built from its
    /// document's `bytes` when this worker has none yet.
    pub fn cluster(&self, bytes: &[u8], request: &crate::ClusterRequest) -> Result<String, String> {
        let source = request.tile.source;
        let geojson = self
            .geojson_source(source)
            .ok_or("the source isn't a GeoJSON source")?;
        let data = self.data(source);
        let mut cached = data.geojson.lock().expect("GeoJSON index poisoned");
        cached
            .get_or_insert_with(|| self.geojson_index(bytes, geojson))
            .as_ref()
            .map_err(Clone::clone)?
            .answer(&request.query)
    }

    /// The tile's decoded frame: cached, or decoded from `bytes` and cached.
    pub(crate) fn frame(
        &self,
        bytes: &[u8],
        encoding: Encoding,
        request: &TileRequest,
    ) -> Result<Arc<TileFrame>, String> {
        self.frame_at(bytes, encoding, request, None)
    }

    /// [`frame`](Self::frame) from the source's data as of style revision
    /// `revision` (the newest when `None`).
    pub(crate) fn frame_at(
        &self,
        bytes: &[u8],
        encoding: Encoding,
        request: &TileRequest,
        revision: Option<u64>,
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
        let data = self.data_at(source, revision);
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
            cached
                .get_or_insert_with(|| self.geojson_index(bytes, geojson))
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
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
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
    pub font_faces: orb_style::FontFaces,
    /// The map's `localIdeographFontFamily`: ideographs draw with it.
    pub local_ideographs: Option<String>,
    /// The plane GeoJSON sources tile in and the tile matrix sets of
    /// sources not in `WebMercatorQuad`.
    pub surface: crate::SurfaceTiles,
    /// A style a scene document extends (its dialect) and its models' ids;
    /// `None` for a v8 style.
    pub models: Option<Vec<String>>,
}

impl StyleUpdate {
    /// The update's dialect and model ids from `doc`.
    pub(crate) fn models_of(style: &Scene) -> Option<Vec<String>> {
        (style.dialect == orb_style::Dialect::Scene)
            .then(|| style.models.iter().map(|(id, _)| id.clone()).collect())
    }
}

/// A tile for a job to load and lay out.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TileRequest {
    /// Which style to lay out with, as given to [`Jobs::set_style`].
    pub style: u64,
    /// The source's key.
    pub source: u32,
    pub tile: TileIndex,
    /// The style zoom the tile lays out at: its own zoom in
    /// `WebMercatorQuad`, from its ground resolution in other sets.
    pub zoom: f64,
    /// How geometry bends: cut along a grid of cells per side (the tile
    /// draws through a curved warp or on a globe), with fills on the
    /// world's edge extended to the poles.
    pub bend: Option<orb_layout::fill::Bend>,
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
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
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

/// A request to a job host in another worker, as [`crate::wire`] carries it.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum JobRequest {
    /// [`Jobs::set_style`].
    Style {
        style: u64,
        update: Box<StyleUpdate>,
    },
    /// [`Jobs::set_icons`].
    Icons {
        style: u64,
        icons: SpriteIcons,
    },
    Load {
        id: u32,
        request: TileRequest,
    },
    Features {
        id: u32,
        request: crate::FeatureRequest,
    },
    Cluster {
        id: u32,
        request: crate::ClusterRequest,
    },
    Raster {
        id: u32,
        request: GlyphRaster,
    },
    /// [`Jobs::model`].
    Model {
        id: u32,
        url: String,
    },
}

/// A job host's answer to the request with `id`. Its large parts travel
/// beside it, transferred: a packed tile ([`pack`]) or a DEM tile's RGBA
/// pixels as the reply's buffer, an image tile as its bitmap.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum JobReply {
    /// The packed tile is the buffer.
    Tile {
        id: u32,
    },
    /// An image tile: the bitmap.
    Image {
        id: u32,
    },
    /// A terrain DEM tile: its pixels are the buffer.
    Dem {
        id: u32,
        width: u32,
        elevations: Vec<f32>,
    },
    /// The source doesn't have the tile.
    Missing {
        id: u32,
    },
    Features {
        id: u32,
        features: Vec<crate::TileFeature>,
    },
    /// A cluster query's answer, as JSON.
    Answer {
        id: u32,
        json: String,
    },
    Glyphs {
        id: u32,
        bitmaps: Vec<crate::GlyphBitmap>,
    },
    /// A model: its words ([`GpuModel::to_words`]) are the buffer.
    ///
    /// [`GpuModel::to_words`]: orb_3d::gpu::GpuModel::to_words
    Model {
        id: u32,
    },
    Error {
        id: u32,
        message: String,
    },
}

impl JobReply {
    pub fn id(&self) -> u32 {
        match self {
            Self::Tile { id }
            | Self::Image { id }
            | Self::Dem { id, .. }
            | Self::Missing { id }
            | Self::Features { id, .. }
            | Self::Answer { id, .. }
            | Self::Glyphs { id, .. }
            | Self::Model { id }
            | Self::Error { id, .. } => *id,
        }
    }
}

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
    /// Loads the glTF model at `url`, packed for the GPU
    /// ([`model::load`](crate::model::load)).
    fn model(&self, url: String, done: crate::model::ModelDone);
    /// Answers a cluster query, in the worker that loads the request's
    /// tile: it has the source's data, or fetches it.
    fn cluster(&self, request: crate::ClusterRequest, done: crate::QueryDone);
}

/// Jobs in this process: a [`JobHost`](crate::JobHost) on an executor,
/// decoding image tiles with the host's decoder.
pub struct PoolJobs {
    host: Arc<crate::JobHost>,
    decode: Arc<dyn ImageDecode>,
    executor: Arc<dyn Executor>,
}

impl std::fmt::Debug for PoolJobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolJobs").finish_non_exhaustive()
    }
}

impl PoolJobs {
    pub fn new(
        fetch: Arc<dyn Fetch>,
        decode: Arc<dyn ImageDecode>,
        executor: Arc<dyn Executor>,
    ) -> Arc<Self> {
        Arc::new(Self {
            host: crate::JobHost::new(fetch, Arc::clone(&executor)),
            decode,
            executor,
        })
    }

    /// Draws local glyphs (decision 0009) with the host's fonts.
    pub fn set_local_glyphs(&self, local: Arc<dyn LocalGlyphs>) {
        self.host.set_local_glyphs(local);
    }

    /// Draws declared font files with the host's engine.
    pub fn set_font_files(&self, files: Arc<dyn crate::glyphs::FontFiles>) {
        self.host.set_font_files(files);
    }
}

impl Jobs for PoolJobs {
    fn set_style(&self, id: u64, _: &StyleUpdate, worker: &Arc<TileWorker>) {
        self.host.set_style(id, Arc::clone(worker));
    }

    fn set_icons(&self, _: u64, icons: &Arc<SpriteIcons>, worker: &Arc<TileWorker>) {
        // Revisions share their worker data, so this reaches every one.
        worker.set_icons(Arc::clone(icons));
    }

    fn raster(&self, request: GlyphRaster, done: RasterDone) {
        self.host.raster(request, done);
    }

    fn features(&self, request: crate::FeatureRequest, done: crate::FeaturesDone) {
        self.host.features(request, done);
    }

    fn model(&self, url: String, done: crate::model::ModelDone) {
        self.host.model(url, done);
    }

    fn cluster(&self, request: crate::ClusterRequest, done: crate::QueryDone) {
        self.host.cluster(request, done);
    }

    fn load(&self, request: TileRequest, done: TileDone) {
        let (decode, executor) = (Arc::clone(&self.decode), Arc::clone(&self.executor));
        let dem_unpack = request.dem_unpack;
        self.host.load(
            request,
            Box::new(move |loaded| match loaded {
                Ok(crate::Loaded::Tile(build)) => done(Ok(build.into())),
                Ok(crate::Loaded::Missing) => done(Ok(TileResult {
                    missing: true,
                    ..TileResult::default()
                })),
                // DEM tiles are data: the core decodes them exactly.
                Ok(crate::Loaded::Image { bytes, kind }) if kind == ImageKind::Dem => executor
                    .spawn(Box::new(move || {
                        done(
                            orb_host::decode_image(&bytes)
                                .and_then(|i| image_result(kind, i, dem_unpack)),
                        );
                    })),
                Ok(crate::Loaded::Image { bytes, kind }) => executor.spawn(Box::new(move || {
                    decode.decode(
                        bytes,
                        Box::new(move |decoded| {
                            done(decoded.and_then(|i| image_result(kind, i, dem_unpack)))
                        }),
                    );
                })),
                Err(e) => done(Err(e)),
            }),
        );
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
pub(crate) fn premultiply(pixels: &mut [u8]) {
    for texel in pixels.as_chunks_mut::<4>().0 {
        let a = u32::from(texel[3]);
        for c in &mut texel[..3] {
            *c = ((u32::from(*c) * a + 127) / 255) as u8;
        }
    }
}

/// A finished tile, ready to upload: in core memory, or in host-held bytes.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TileResult {
    pub warnings: Vec<String>,
    /// Images the tile's symbol layers name that the style doesn't have.
    pub missing_images: Vec<String>,
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
    #[serde(skip)]
    pub host_bytes: Option<HostBytesId>,
    /// An image source's tile.
    #[serde(skip)]
    pub image: Option<TileImage>,
    /// A terrain DEM tile's elevations.
    #[serde(skip)]
    pub dem: Option<Arc<orb_tile::DemGrid>>,
    /// The tileset doesn't have the tile; nearby zooms stand in.
    #[serde(skip)]
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
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ResultMesh {
    pub hits: std::sync::Arc<crate::MeshHits>,
    /// The node's key.
    pub node: u32,
    pub extent: u32,
    pub chunks: Vec<Chunk>,
    pub vertices: Stream,
    pub indices: Stream,
    pub props: Stream,
    /// Words per feature of the property rows.
    pub row_words: u32,
    /// Per paint property, the largest number in the property rows.
    pub maxima: Vec<f32>,
    pub images: Vec<ImageRef>,
    /// Per feature slot, an index into [`TileResult::ids`] (or `u32::MAX`),
    /// for nodes that read feature state.
    pub state_ids: Option<Vec<u32>>,
    /// Per feature slot, what paint reading both state and data reads of
    /// the feature (see `NodeMesh`).
    pub data: Vec<orb_expr::FeatureData>,
}

/// A stream of u32 words. Packed for another worker, a stream's header
/// holds its length; its words follow the header raw ([`pack`]).
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
                    hits: std::sync::Arc::new(m.hits),
                    extent: m.extent,
                    chunks: m.mesh.chunks,
                    vertices: Stream::Core(m.mesh.vertices),
                    indices: Stream::Core(m.mesh.indices),
                    props: Stream::Core(m.props),
                    row_words: m.row_words,
                    maxima: m.maxima,
                    images: m.images,
                    state_ids: m.state_ids,
                    data: m.data,
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
            missing_images: build.missing_images,
            image: None,
            dem: None,
            missing: false,
        }
    }
}

impl Stream {
    fn len(&self) -> u32 {
        match self {
            Self::Core(words) => words.len() as u32,
            Self::Host { len, .. } => *len,
        }
    }
}

impl serde::Serialize for Stream {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.len())
    }
}

/// A packed stream: its offset is set once the whole header is read.
impl<'de> serde::Deserialize<'de> for Stream {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let len = u32::deserialize(deserializer)?;
        Ok(Self::Host { offset: 0, len })
    }
}

impl TileResult {
    /// The streams in packed order: each mesh's vertices, indices and
    /// rows, then each symbol layer's tables.
    fn streams_mut(&mut self) -> impl Iterator<Item = &mut Stream> {
        self.meshes
            .iter_mut()
            .flat_map(|m| [&mut m.vertices, &mut m.indices, &mut m.props])
            .chain(self.symbols.iter_mut().flat_map(|s| s.tables.iter_mut()))
    }
}

/// Packs a build into one buffer for another worker: the header's length
/// in bytes (a little-endian u32), the header (the result without its
/// streams' words, [`crate::wire`]), padding to a word, then every stream's
/// words in [`TileResult::streams_mut`]'s order, little-endian.
pub fn pack(build: TileBuild) -> Vec<u8> {
    let mut result = TileResult::from(build);
    let header = crate::wire::encode(&result);
    let mut packed = Vec::with_capacity(header.len() + 8);
    packed.extend((header.len() as u32).to_le_bytes());
    packed.extend(&header);
    packed.resize(packed.len().next_multiple_of(4), 0);
    for stream in result.streams_mut() {
        let Stream::Core(words) = stream else {
            unreachable!("built streams are in core memory")
        };
        packed.extend(words.iter().flat_map(|w| w.to_le_bytes()));
    }
    packed
}

/// The header of a packed tile (see [`pack`]): the bytes after its length.
pub fn packed_header(packed: &[u8]) -> Result<&[u8], String> {
    let bad = || "malformed packed tile".to_string();
    let len = packed.get(..4).ok_or_else(bad)?;
    let len = u32::from_le_bytes(len.try_into().expect("four bytes")) as usize;
    packed.get(4..4 + len).ok_or_else(bad)
}

/// Reads a packed tile's `header` (see [`packed_header`]): its streams are
/// [`Stream::Host`] at their offsets in the packed buffer, registered as
/// `bytes`; the payload need not be in core memory.
pub fn unpack(header: &[u8], bytes: HostBytesId) -> Result<TileResult, String> {
    let mut result: TileResult = crate::wire::decode(header)?;
    let mut offset = (4 + header.len()).next_multiple_of(4) as u64;
    for stream in result.streams_mut() {
        if let Stream::Host { offset: at, len } = stream {
            *at = offset;
            offset += u64::from(*len) * 4;
        }
    }
    result.host_bytes = Some(bytes);
    Ok(result)
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
            bend: None,
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
        let again = || {
            worker
                .build(b"invalid", Encoding::Mlt, &request(tile))
                .unwrap()
                .tile()
        };
        assert_eq!(pack(again()), pack(again()));
        assert_eq!(build.meshes.len(), again().meshes.len());
        // A column job re-evaluates the fill's rows from the cached tile.
        let rows = worker
            .build(
                &[],
                Encoding::Mvt,
                &TileRequest {
                    nodes: Some(vec![crate::style::node_key(1, 0)]),
                    rows_only: true,
                    ..request(tile)
                },
            )
            .unwrap()
            .tile();
        assert_eq!(rows.nodes, [crate::style::node_key(1, 0)]);
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
            surface: crate::SurfaceTiles {
                plane: Some(polar_plane()),
                sources: Vec::new(),
            },
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
    fn geojson_errors_are_terminal_and_clusters_draw() {
        let worker =
            TileWorker::new(&INLINE.replace("\"data\":{", "\"cluster\":true,\"data\":{")).unwrap();
        assert!(!worker.plan.nodes.is_empty());
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
                vec![orb_style::FontFace {
                    url: "test://noto.woff2".into(),
                    ranges: vec![[0, 0x24F], [0x600, 0x6FF]],
                }],
            )],
            local_ideographs: Some("Hiragino Sans".into()),
            surface: crate::SurfaceTiles {
                plane: Some(polar_plane()),
                sources: vec![("geo".into(), TileMatrixSet::by_id("NZTM2000Quad").unwrap())],
            },
            models: Some(vec!["tower".into()]),
        };
        let bytes = crate::wire::encode(&update);
        let parsed: StyleUpdate = crate::wire::decode(&bytes).unwrap();
        assert_eq!(parsed.models, update.models);
        assert_eq!(crate::wire::encode(&parsed), bytes);
        assert_eq!(parsed.surface.plane, update.surface.plane);
        let tile = TileIndex::new(4, 7, 9);
        assert_eq!(
            parsed.surface.sources[0].1.tile_bounds(tile),
            update.surface.sources[0].1.tile_bounds(tile)
        );
        let revised = TileWorker::update(None, &parsed).unwrap();
        assert_eq!(revised.revision(), 3);
        assert_eq!(revised.plan.nodes.len(), 1);
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
        let make = || TileBuild {
            symbols: Vec::new(),
            revision: 1 << 33 | 5,
            nodes: vec![3, 4, 9],
            rows_only: false,
            ids: vec!["a".into(), "1.5".into()],
            warnings: vec![
                "source geo: nested object and array property values become strings (é)".into(),
            ],
            missing_images: vec!["marker".into(), "ü".into()],
            meshes: vec![
                NodeMesh {
                    hits: crate::MeshHits::new(&[crate::HitFeature {
                        id: Some("promoted é".into()),
                        feature: 7,
                        kind: 0,
                        extent: 4096,
                        radius: 2.0,
                        stroke: 1.0,
                        height: 40.0,
                        base: 2.0,
                        parts: vec![vec![vec![[-3, 5], [30, 50], [30, 5]]]],
                    }]),
                    node: 3,
                    extent: 4096,
                    mesh: Mesh {
                        vertices: vec![1, 2, 3, 4],
                        indices: vec![0, 1, 0],
                        chunks: vec![Chunk {
                            first_index: 0,
                            index_count: 3,
                            sort_key: Some(2.5),
                            bounds: orb_layout::TileBox {
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
                    data: vec![
                        orb_expr::FeatureData {
                            id: Value::Number(7.0),
                            geometry_type: Some(orb_tile::GeometryType::LineString),
                            properties: vec![("class".into(), Value::String("major".into()))],
                        };
                        3
                    ],
                },
                NodeMesh {
                    hits: crate::MeshHits::default(),
                    node: 4,
                    extent: 8192,
                    mesh: Mesh::default(),
                    props: vec![],
                    row_words: 0,
                    maxima: vec![],
                    images: vec![],
                    state_ids: None,
                    data: vec![],
                },
            ],
        };
        let build = make();
        let bytes = pack(make());
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let result = unpack(packed_header(&bytes).unwrap(), HostBytesId(7)).unwrap();
        assert_eq!(result.host_bytes, Some(HostBytesId(7)));
        assert_eq!(*result.meshes[0].hits, build.meshes[0].hits);
        assert_eq!(result.warnings, build.warnings);
        assert_eq!(result.missing_images, build.missing_images);
        assert_eq!(
            (result.revision, &result.nodes, &result.ids),
            (build.revision, &build.nodes, &build.ids)
        );
        assert_eq!(result.meshes[0].state_ids, build.meshes[0].state_ids);
        assert_eq!(result.meshes[0].data, build.meshes[0].data);
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
        assert!(packed_header(&bytes[..3]).is_err());
        assert!(unpack(&bytes[4..8], HostBytesId(0)).is_err());
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
                done(std::fs::read(path).map_err(|_| orb_host::FetchError::NotFound));
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
                done(std::fs::read(path).map_err(|_| orb_host::FetchError::NotFound));
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
        assert_eq!(nodes, [crate::style::node_key(1, 1)]);
        assert_eq!(
            build.meshes[0].props.len() % orb_render::SYMBOL_ROW_WORDS,
            0
        );
        assert_eq!(symbols.instance_rows.len(), 2 * b.instances.len());

        let core = TileResult::from(build);
        let packed = pack(worker.build(&[], Encoding::Mvt, &request).unwrap().tile());
        let words: Vec<u32> = packed
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let result = unpack(packed_header(&packed).unwrap(), HostBytesId(3)).unwrap();
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
        let mut icons = SpriteIcons::default();
        icons.pixel_ratio = 2.0;
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
        assert_eq!(
            crate::wire::decode::<SpriteIcons>(&crate::wire::encode(&icons)).unwrap(),
            icons
        );
    }
}

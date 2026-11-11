//! The tile worker: lays out tiles for one revision of a
//! style, pure, on any job thread or web worker. Workers keep each tile's
//! decoded [`TileFrame`] (the owner job worker keeps them), so
//! a request for some of a tile's nodes, or only their property rows, lays
//! them out again without fetching or decoding. This side of the jobs names
//! no renderer or GPU type.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use orb_crs::{TileIndex, TileMatrixSet};
use orb_expr::json::Json;
use orb_style::{Encoding, GeoJsonData, SourceKind, Style, Value};
use orb_tile::TileFrame;

use crate::TileLocation;
use crate::glyphs::{GlyphCatalog, LocalGlyphs};
use crate::style::{ImageKind, StylePlan};
use crate::symbol::SpriteIcons;
use crate::tile::{self, Select, TileBuild};

/// Bytes of decoded tiles a worker keeps, least recently used first out.
const FRAME_BUDGET: usize = 64 << 20;

/// One source's data in a worker, shared by its style revisions.
#[derive(Debug, Default)]
struct SourceData {
    geojson: Mutex<Option<Result<crate::geojson::Index, String>>>,
}

/// A cached frame's key: its source, the style revision its source data
/// began at, and its tile.
type FrameKey = (u32, u64, TileIndex);

#[derive(Debug, Default)]
struct Frames {
    /// Each frame, its bytes and when it was last used.
    tiles: HashMap<FrameKey, (Arc<TileFrame>, usize, u64)>,
    bytes: usize,
    clock: u64,
}

impl Frames {
    fn get(&mut self, key: FrameKey) -> Option<Arc<TileFrame>> {
        self.clock += 1;
        let clock = self.clock;
        self.tiles.get_mut(&key).map(|(frame, _, used)| {
            *used = clock;
            Arc::clone(frame)
        })
    }

    fn insert(&mut self, key: FrameKey, frame: Arc<TileFrame>) {
        self.clock += 1;
        let bytes = frame.bytes();
        self.bytes += bytes;
        if let Some((_, old, _)) = self.tiles.insert(key, (frame, bytes, self.clock)) {
            self.bytes -= old;
        }
        while self.bytes > FRAME_BUDGET && self.tiles.len() > 1 {
            let (&oldest, _) = self
                .tiles
                .iter()
                .min_by_key(|(_, (.., used))| *used)
                .expect("a frame");
            self.remove(oldest);
        }
    }

    fn remove(&mut self, key: FrameKey) {
        if let Some((_, bytes, _)) = self.tiles.remove(&key) {
            self.bytes -= bytes;
        }
    }
}

/// A source's data revisions a worker keeps, oldest first: the style
/// revision each began at, and its index and decoded tiles. Queries read
/// the revision a drawn tile was laid out from.
type SourceRevisions = Vec<(u64, Arc<SourceData>)>;

/// How many data revisions of a source a worker keeps: the drawn tiles'
/// and those coming.
const DATA_REVISIONS: usize = 3;

/// What a worker keeps across a style's revisions.
#[derive(Debug, Default)]
pub(crate) struct WorkerData {
    /// Per source key, kept while the source is.
    sources: Mutex<HashMap<u32, SourceRevisions>>,
    /// Decoded tiles of every source.
    frames: Mutex<Frames>,
    /// Glyph metrics of the PBF ranges fetched so far.
    pub(crate) glyphs: GlyphCatalog,
    /// Sprite metadata for icon layout, and the layouts waiting for a
    /// later set.
    icons: Mutex<IconState>,
    /// How labels in scripts written without spaces break.
    words: Mutex<Words>,
}

#[derive(Default)]
struct IconState {
    icons: Arc<SpriteIcons>,
    /// By the set (`SpriteIcons::epoch`) each waits for.
    waiting: Vec<(u32, Box<dyn FnOnce() + Send>)>,
}

impl std::fmt::Debug for IconState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IconState")
            .field("icons", &self.icons)
            .field("waiting", &self.waiting.len())
            .finish()
    }
}

/// How labels in scripts written without spaces (Thai, Lao, Khmer,
/// Burmese) break lines: natively with the compiled dictionaries; on the
/// web the job hosts set the browser's segmenter
/// ([`JobHost::set_word_breaker`](crate::JobHost::set_word_breaker)). Without
/// one, such text offers no breaks, as in GL JS without `Intl.Segmenter`.
#[derive(Clone, Debug)]
#[cfg_attr(target_arch = "wasm32", derive(Default))]
struct Words(Option<Arc<orb_text::WordBreaker>>);

#[cfg(not(target_arch = "wasm32"))]
impl Default for Words {
    fn default() -> Self {
        static COMPILED: std::sync::OnceLock<Arc<orb_text::WordBreaker>> =
            std::sync::OnceLock::new();
        Self(Some(Arc::clone(COMPILED.get_or_init(|| {
            Arc::new(orb_text::WordBreaker::southeast_asian())
        }))))
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
    surface: crate::surface::SurfaceTiles,
}

/// A tile job's outcome: the laid-out tile, or what its labels need first.
#[derive(Debug)]
pub enum Built {
    Tile(TileBuild),
    /// URLs (glyph PBF ranges, word data) to fetch and add with
    /// `TileWorker::add_resource`; then build again.
    Fetch(Vec<String>),
}

impl TileWorker {
    /// Parses a v8 style that will not be edited. Returns the style's
    /// errors joined by newlines.
    pub fn new(style_json: &str) -> Result<Self, String> {
        let style = orb_style::v8::parse(style_json).map_err(|d| d.to_string())?;
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
            crate::surface::SurfaceTiles::default(),
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
        style: Arc<Style>,
        layer_keys: Vec<u32>,
        source_keys: Vec<u32>,
        new_data: &[u32],
        revision: u64,
        ideographs: Option<String>,
        surface: crate::surface::SurfaceTiles,
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
    pub(crate) fn add_resource(&self, url: &str, bytes: Result<&[u8], String>) {
        if self.declares_font(url) {
            self.data.glyphs.insert_font(url, bytes);
            return;
        }
        self.data.glyphs.insert(url, bytes);
    }

    /// Breaks words in scripts written without spaces with `breaker`, for
    /// this revision and later ones.
    pub(crate) fn set_word_breaker(&self, breaker: Arc<orb_text::WordBreaker>) {
        *self.data.words.lock().expect("words poisoned") = Words(Some(breaker));
    }

    /// Draws local glyphs with the host's fonts, for this revision and
    /// later ones.
    pub fn set_local_glyphs(&self, local: Arc<dyn LocalGlyphs>) {
        self.data.glyphs.set_local(local);
    }

    /// Whether the style declares `url` as a font file (`font-faces`).
    pub(crate) fn declares_font(&self, url: &str) -> bool {
        let declared = self.plan.style.font_faces.iter();
        declared.flat_map(|(_, files)| files).any(|f| f.url == url)
    }

    /// Whether `url`, fetched for a tile's text, is a glyph range: not a
    /// declared font file.
    pub(crate) fn is_glyph_range(&self, url: &str) -> bool {
        !self.declares_font(url)
    }

    /// Draws declared font files (`font-faces`) with the host's engine, for
    /// this revision and later ones.
    pub fn set_font_files(&self, files: Arc<dyn crate::glyphs::FontFiles>) {
        self.data.glyphs.set_files(files);
    }

    /// Sets the sprite metadata icons are laid out with, for this revision
    /// and later ones, and runs the layouts that waited for it.
    pub fn set_icons(&self, icons: Arc<SpriteIcons>) {
        let ready = {
            let mut state = self.data.icons.lock().expect("icons poisoned");
            let epoch = icons.epoch;
            state.icons = icons;
            let (ready, waiting) = std::mem::take(&mut state.waiting)
                .into_iter()
                .partition::<Vec<_>, _>(|(wants, _)| *wants <= epoch);
            state.waiting = waiting;
            ready
        };
        for (_, ready) in ready {
            ready();
        }
    }

    /// Whether a layout must wait for a later set of icons than this
    /// worker has: the set `epoch` names ([`TileRequest::icons`]).
    pub(crate) fn awaits_icons(&self, epoch: u32) -> bool {
        self.data.icons.lock().expect("icons poisoned").icons.epoch < epoch
    }

    /// The sprite icons the worker lays out with.
    pub(crate) fn icons(&self) -> Arc<SpriteIcons> {
        Arc::clone(&self.data.icons.lock().expect("icons poisoned").icons)
    }

    /// Runs `ready` once this worker has icons set `epoch`: now, or when
    /// [`set_icons`](Self::set_icons) brings it.
    pub(crate) fn with_icons(&self, epoch: u32, ready: Box<dyn FnOnce() + Send>) {
        {
            let mut state = self.data.icons.lock().expect("icons poisoned");
            if state.icons.epoch < epoch {
                state.waiting.push((epoch, ready));
                return;
            }
        }
        ready();
    }

    /// Applies an update from the map: the first update has every layer
    /// and source, later ones those new in their revision.
    pub fn update(previous: Option<&TileWorker>, update: &StyleUpdate) -> Result<Self, String> {
        let old = previous.map(|p| &p.plan);
        let mut sources = Vec::with_capacity(update.sources.len());
        for key in &update.sources {
            let previous = || {
                old.and_then(|plan| {
                    let at = plan.source_keys.iter().position(|k| k == key)?;
                    Some(&plan.style.sources[at])
                })
            };
            let diffs = update.source_diffs.iter().find(|(k, _)| k == key);
            let source = match update.source_json.iter().find(|(k, ..)| k == key) {
                Some((_, id, json)) => Arc::new(
                    orb_style::v8::parse_source(id, json, update.previews)
                        .map_err(|d| d.to_string())?
                        .0,
                ),
                None if let Some((_, diffs)) = diffs => {
                    let source =
                        previous().ok_or_else(|| format!("a diff of unknown source {key}"))?;
                    Arc::new(orb_style::v8::apply_diffs(source, diffs)?)
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
        let mut layers = Vec::with_capacity(update.layers.len());
        for (index, key) in update.layers.iter().enumerate() {
            let layer = match update.layer_json.iter().find(|(k, _)| k == key) {
                Some((_, json)) => Arc::new(
                    orb_style::v8::parse_layer(
                        json,
                        index,
                        &sources,
                        &update.state,
                        update.previews,
                    )
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
        let style = Style {
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
            light: orb_style::v8::default_light(),
            atmosphere: None,
            terrain: None,
            globe: None,
            projection: None,
            state: Vec::new(),
            previews: update.previews,
            color_space: Default::default(),
            sources,
            layers,
            warnings: Vec::new(),
        };
        // A source sent again or diffed has new data (an added one has
        // none yet).
        let new_data: Vec<u32> = update
            .source_json
            .iter()
            .map(|(k, ..)| *k)
            .chain(update.source_diffs.iter().map(|(k, _)| *k))
            .collect();
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

    /// The glyph ranges the labels shown at `zoom` will most likely ask
    /// for first: Basic Latin's range of each font stack their text is set
    /// in, where that doesn't depend on the feature, and that this worker
    /// hasn't fetched.
    pub(crate) fn first_glyphs(&self, zoom: f64) -> Vec<String> {
        use orb_expr::Kind;
        let style = &self.plan.style;
        let template = style.glyphs.as_deref();
        let policy = orb_text::FontPolicy {
            glyphs_url: template.is_some(),
            local_ideographs: false,
        };
        let mut requests = orb_text::GlyphRequests::default();
        for node in &self.plan.nodes {
            let layer = &style.layers[node.layer];
            if node.family != orb_layout::Family::SymbolIcon || !layer.is_visible_at(zoom) {
                continue;
            }
            let text = layer.get(orb_style::TEXT_FIELD);
            let font = layer.get(orb_style::TEXT_FONT);
            let has_text = text
                .constant()
                .is_none_or(|s| s.iter().any(|s| !s.text.is_empty()));
            if !has_text || !matches!(font.kind(), Kind::Constant | Kind::Camera) {
                continue;
            }
            let stack = font.evaluate(zoom);
            if !stack.is_empty() {
                let label = orb_text::Label::from_text("a");
                requests.add_label(&label, &orb_text::FontStack::new(&stack), false, false);
            }
        }
        let faces = &style.font_faces;
        let missing = self.data.glyphs.missing(&requests, template, policy, faces);
        missing
            .into_iter()
            .filter(|url| self.is_glyph_range(url))
            .collect()
    }

    /// Layout grid units (8192 a tile side) per ground metre at a tile's
    /// centre, in its source's tile matrix set: GL JS's
    /// `getTileUnitsForMeters`.
    fn units_per_metre(&self, source: usize, tile: TileIndex) -> f64 {
        let tms = self.surface.source_tms(&self.plan.style.sources[source].id);
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
        let tms = self.surface.source_tms(&self.plan.style.sources[source].id);
        (tms.id() == "WebMercatorQuad").then_some(orb_expr::CanonicalTile {
            z: tile.z,
            x: tile.col,
            y: tile.row,
        })
    }

    pub(crate) fn surface(&self) -> &crate::surface::SurfaceTiles {
        &self.surface
    }

    /// A tile of source `key` on the surface's plane: its warp's grid
    /// there, as the map draws it (`crate::image::warp_grid`). `None`
    /// if no part of it reaches the plane.
    pub(crate) fn warp_grid(&self, key: u32, tile: TileIndex) -> Option<orb_crs::WarpGrid> {
        let at = self.plan.source_keys.iter().position(|&k| k == key)?;
        let tms = (self.surface).source_tms(&self.plan.style.sources[at].id);
        let plane = self.surface.plane.as_ref();
        let plane = *plane.unwrap_or(TileMatrixSet::web_mercator()).projection();
        crate::image::warp_grid(&orb_crs::Warp::new(tms.clone(), plane), tile)
    }

    /// The style revision this worker lays out.
    pub fn revision(&self) -> u64 {
        self.plan.revision
    }

    /// A source's newest data.
    fn data(&self, source: u32) -> Arc<SourceData> {
        self.data_at(source, None).1
    }

    /// A source's data as of style revision `revision`: the newest that
    /// began by then, or the newest of all, with the revision it began at.
    fn data_at(&self, source: u32, revision: Option<u64>) -> (u64, Arc<SourceData>) {
        let mut data = self.data.sources.lock().expect("worker data poisoned");
        let revisions = data.entry(source).or_default();
        if revisions.is_empty() {
            revisions.push((0, Arc::default()));
        }
        let at = revision
            .and_then(|r| revisions.iter().rposition(|(began, _)| *began <= r))
            .unwrap_or(revisions.len() - 1);
        let (began, data) = &revisions[at];
        (*began, Arc::clone(data))
    }

    /// Whether `source` (a source key) has image tiles to decode rather
    /// than lay out, and of which kind: any of the style's sources, drawn
    /// by layers or by the terrain alone.
    pub(crate) fn image_kind(&self, source: u32) -> Option<ImageKind> {
        let at = self.plan.source_keys.iter().position(|&k| k == source)?;
        match self.plan.style.sources[at].kind {
            orb_style::SourceKind::Raster(_) | orb_style::SourceKind::Image(_) => {
                Some(ImageKind::Raster)
            }
            orb_style::SourceKind::RasterDem(_) => Some(ImageKind::Dem),
            _ => None,
        }
    }

    /// What a fetch of `source`'s data is (a source key): an image
    /// source's image, a GeoJSON document, or a tile.
    pub(crate) fn resource_kind(&self, source: u32) -> orb_host::ResourceKind {
        let at = self.plan.source_keys.iter().position(|&k| k == source);
        match at.map(|at| &self.plan.style.sources[at].kind) {
            Some(orb_style::SourceKind::Image(_)) => orb_host::ResourceKind::Image,
            Some(orb_style::SourceKind::GeoJson(_)) => orb_host::ResourceKind::Source,
            _ => orb_host::ResourceKind::Tile,
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
    pub(crate) fn has_data(&self, request: &TileRequest) -> bool {
        // An app's tile comes with the request.
        if self.plan.source(request.source).is_none()
            || matches!(request.location, crate::TileLocation::Host(_))
        {
            return false;
        }
        let (began, data) = self.data_at(request.source, None);
        let key = (request.source, began, request.tile);
        // An overzoomed tile slices its ancestor's frame, decoded for a
        // sibling.
        let whole = (request.source, began, request.data_tile());
        let frames = self.data.frames.lock().expect("frames poisoned");
        if frames.tiles.contains_key(&key) || frames.tiles.contains_key(&whole) {
            return true;
        }
        drop(frames);
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
            Some(bend) if bend.cells.iter().any(|&c| c > 1) => {
                Arc::new(frame.subdivided(bend.cells))
            }
            _ => frame,
        };
        let select = Select {
            nodes: request.nodes.as_deref(),
            rows_only: request.rows_only,
            neighbour: request.neighbour,
        };
        let icons = self.icons();
        let words = self.data.words.lock().expect("words poisoned").clone();
        let symbols = tile::Symbols {
            glyphs: &self.data.glyphs,
            icons: &icons,
            ideographs: self.ideographs.as_deref(),
            words: words.0.as_deref(),
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
            GeoJsonData::Url(_) => std::str::from_utf8(bytes)
                .map_err(|e| e.to_string())
                .and_then(orb_style::v8::parse_geojson)
                .map_err(|e| format!("{e}; {}", orb_style::v8::EMPTY_SOURCE))?,
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
        let (began, data) = self.data_at(source, revision);
        // An app's tile: decoded from its answer, which a later one
        // replaces, rather than kept.
        if let crate::TileLocation::Host(answer) = &request.location {
            let data_tile = request.data_tile();
            let frame = match answer {
                orb_source::endpoint::HostTile::GeoJson(text) => {
                    Arc::new(crate::geojson::host_frame(text, data_tile)?)
                }
                _ => self.decode(bytes, encoding, index, &data, data_tile, data_tile.z)?,
            };
            return Ok(overzoom(frame, tile, data_zoom));
        }
        let key = (source, began, tile);
        let frames = &self.data.frames;
        let cached = frames.lock().expect("frames poisoned").get(key);
        if let Some(frame) = cached {
            return Ok(frame);
        }
        let geojson = matches!(self.plan.source_def(index).kind, SourceKind::GeoJson(_));
        let frame = if data_zoom < tile.z {
            // An ancestor's overzoomed tiles slice its one decode.
            let whole = request.data_tile();
            let whole_key = (source, began, whole);
            let cached = frames.lock().expect("frames poisoned").get(whole_key);
            let whole_frame = match cached {
                Some(frame) => frame,
                None => {
                    let frame = self.decode(bytes, encoding, index, &data, whole, data_zoom)?;
                    let mut frames = frames.lock().expect("frames poisoned");
                    frames.insert(whole_key, Arc::clone(&frame));
                    frame
                }
            };
            Arc::new(slice(&whole_frame, tile, data_zoom, geojson))
        } else {
            self.decode(bytes, encoding, index, &data, tile, data_zoom)?
        };
        let mut frames = frames.lock().expect("frames poisoned");
        frames.insert(key, Arc::clone(&frame));
        Ok(frame)
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
    ) -> Result<Arc<TileFrame>, String> {
        let depth = tile.z - data_zoom;
        let frame = if let SourceKind::GeoJson(geojson) = &self.plan.source_def(index).kind {
            let mut cached = data.geojson.lock().expect("GeoJSON index poisoned");
            let fresh = cached.is_none();
            match cached.get_or_insert_with(|| self.geojson_index(bytes, geojson)) {
                Ok(index) => index
                    .tile(data_zoom, (tile.col >> depth) as i32, tile.row >> depth)
                    .unwrap_or_default(),
                // A source whose index can't be built (a document that
                // isn't GeoJSON) fails the tile that tried, which reports
                // why, and is empty after, as in MapLibre Native, rather
                // than failing every tile and every retry with the same
                // error.
                Err(e) if fresh => return Err(e.clone()),
                Err(_) => Arc::default(),
            }
        } else if bytes.is_empty() {
            // An archive's left-out tile.
            Arc::default()
        } else {
            match encoding {
                Encoding::Mvt => orb_tile::mvt::decode(bytes),
                Encoding::Mlt => orb_tile::mlt::decode(bytes),
            }
            .map(Arc::new)
            .map_err(|e| e.to_string())?
        };
        Ok(overzoom(frame, tile, data_zoom))
    }
}

/// The zoom GL JS stops slicing an overzoomed vector tile at, laying its
/// descendants out from that zoom's slice, overscaled: four zooms
/// (`zoomLevelsToOverscale`) below its default `maxZoom`, 22.
const VECTOR_LABEL_ZOOM: u8 = 18;

/// `tile`'s slice of `frame`, its ancestor's at `data_zoom`, with its labels
/// laid out where GL JS lays them out: in the tile it overscales, the
/// source's last tile or its slice at [`VECTOR_LABEL_ZOOM`]. (MapLibre
/// Native overscales the last tile at any depth, so its anchors round to
/// that tile's units, which grow coarse far below it.)
fn overzoom(frame: Arc<TileFrame>, tile: TileIndex, data_zoom: u8) -> Arc<TileFrame> {
    if data_zoom >= tile.z {
        return frame;
    }
    Arc::new(slice(&frame, tile, data_zoom, false))
}

/// [`overzoom`] of an ancestor's frame that stays whole. A GeoJSON
/// source's labels lay out in its data tile at any depth, as GL JS and
/// MapLibre Native overscale it whole.
fn slice(frame: &Arc<TileFrame>, tile: TileIndex, data_zoom: u8, geojson: bool) -> TileFrame {
    let depth = tile.z - data_zoom;
    let mask = (1u32 << depth) - 1;
    let labels_zoom = if geojson {
        data_zoom
    } else {
        data_zoom.max(VECTOR_LABEL_ZOOM).min(tile.z)
    };
    frame.overzoom(
        depth,
        tile.col & mask,
        tile.row & mask,
        labels_zoom - data_zoom,
    )
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
    /// GeoJSON sources whose data changed by diffs alone, with the diffs:
    /// job hosts apply them to the data they hold.
    pub source_diffs: Vec<(u32, Vec<Arc<Json>>)>,
    pub state: Vec<(String, Value<'static>)>,
    /// The previews the style is parsed with.
    pub previews: orb_style::Previews,
    /// The style's glyph PBF URL template.
    pub glyphs: Option<String>,
    /// The style's declared font files.
    pub font_faces: orb_style::FontFaces,
    /// The map's `localIdeographFontFamily`: ideographs draw with it.
    pub local_ideographs: Option<String>,
    /// The plane GeoJSON sources tile in and the tile matrix sets of
    /// sources not in `WebMercatorQuad`.
    pub surface: crate::surface::SurfaceTiles,
}

/// A tile for a job to load and lay out.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TileRequest {
    /// Which style to lay out with, as given to [`Jobs::set_style`](crate::Jobs::set_style).
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
    /// An image tile drawn on the surface's plane, not its own: the job
    /// also makes its warp's grid there
    /// ([`TileResult::warp_grid`](crate::TileResult::warp_grid)).
    pub warp: bool,
    /// Zoom of the ancestor whose bytes are fetched.
    pub(crate) data_zoom: u8,
    pub location: TileLocation,
    pub encoding: Encoding,
    /// Lay out only these nodes (by key); `None` for all of the source's.
    pub nodes: Option<Vec<u32>>,
    /// Only property rows: a column job.
    pub(crate) rows_only: bool,
    /// A neighbour of the tile Tile mode draws: only what reaches into the
    /// tiles beside it lays out, the nodes whose content crosses tile
    /// edges and the features at its edges of those whose features reach
    /// past them ([`Family::crosses_tiles`], [`Family::reaches_edges`]).
    ///
    /// [`Family::crosses_tiles`]: orb_layout::Family::crosses_tiles
    /// [`Family::reaches_edges`]: orb_layout::Family::reaches_edges
    pub(crate) neighbour: bool,
    /// A terrain DEM tile: also decode its elevations with this unpack
    /// vector, for the CPU ([`TileResult::dem`](crate::TileResult::dem)).
    pub dem_unpack: Option<DemUnpack>,
    /// The set of sprite icons (`SpriteIcons::epoch`) the layout waits
    /// for, fetched and decoded meanwhile: a style's sprites still loading
    /// hold symbol layout, not the tile's request. 0 waits for none.
    pub(crate) icons: u32,
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
    /// The whole of `tile`, every node of source `source` of style `style`
    /// laid out at the tile's own zoom from its own bytes at `location`:
    /// no bend, warp, DEM or icons to wait for. Requests that differ
    /// update it.
    pub fn whole(
        style: u64,
        source: u32,
        tile: TileIndex,
        location: TileLocation,
        encoding: Encoding,
    ) -> Self {
        Self {
            style,
            source,
            tile,
            zoom: f64::from(tile.z),
            bend: None,
            warp: false,
            data_zoom: tile.z,
            location,
            encoding,
            nodes: None,
            rows_only: false,
            neighbour: false,
            dem_unpack: None,
            icons: 0,
        }
    }

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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::jobs::*;
    use orb_host::Fetch;

    fn request(tile: TileIndex) -> TileRequest {
        let location = TileLocation::Url("test://data.json".into());
        TileRequest::whole(1, 1, tile, location, Encoding::Mvt)
    }

    const INLINE: &str = r#"{"version":8,"sources":{"geo":{"type":"geojson","maxzoom":2,"data":{"type":"Feature","properties":{"color":"red"},"geometry":{"type":"Polygon","coordinates":[[[-20,-20],[20,-20],[20,20],[-20,20],[-20,-20]]]}}}},"layers":[{"id":"fill","type":"fill","source":"geo","paint":{"fill-color":["get","color"]}},{"id":"line","type":"line","source":"geo"}]}"#;

    /// Thai labels break where the word breaker says: the compiled
    /// dictionaries natively, or the host's segmenter.
    #[test]
    fn thai_labels_break_with_the_host_segmenter() {
        const THAI: &str = r#"{"version":8,"glyphs":"test://{fontstack}/{range}.pbf",
            "sources":{"geo":{"type":"geojson","data":{"type":"Feature",
                "properties":{"name":"ราชอาณาจักรไทย"},"geometry":{"type":"Point","coordinates":[0,0]}}}},
            "layers":[{"id":"t","type":"symbol","source":"geo","layout":{"text-field":"{name}"}}]}"#;
        let tile = TileIndex::new(0, 0, 0);
        let compiled = TileWorker::new(THAI).unwrap();
        assert!(matches!(
            compiled.build(&[], Encoding::Mvt, &request(tile)).unwrap(),
            Built::Tile(_)
        ));

        let host = TileWorker::new(THAI).unwrap();
        let asked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&asked);
        host.set_word_breaker(Arc::new(orb_text::WordBreaker::from_host(Box::new(
            move |text: &str| {
                counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                vec![0, text.len()]
            },
        ))));
        assert!(matches!(
            host.build(&[], Encoding::Mvt, &request(tile)).unwrap(),
            Built::Tile(_)
        ));
        assert!(asked.load(std::sync::atomic::Ordering::Relaxed) > 0);
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
                orb_layout::pack_color([1.0, 0.0, 0.0, 1.0]),
                orb_layout::pack_color([1.0, 0.0, 0.0, 1.0])
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
                    neighbour: false,
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

    #[derive(Default)]
    struct DeferredFetch {
        callbacks: Mutex<Vec<orb_host::FetchDone>>,
    }
    impl Fetch for DeferredFetch {
        fn get_range(
            &self,
            _: orb_host::ResourceKind,
            _: &str,
            _: u64,
            _: u64,
            _: orb_host::FetchDone,
        ) {
            panic!("GeoJSON must fetch its whole document");
        }

        fn get(&self, _: orb_host::ResourceKind, url: &str, done: orb_host::FetchDone) {
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
            Arc::new(crate::testing::NoDecode),
            Arc::new(orb_host::jobs::Inline),
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
            surface: crate::surface::SurfaceTiles {
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

    /// A job makes a warped raster tile's grid on the surface's plane as
    /// the map would, in this process and in a job worker that got the
    /// surface over the wire; the grid crosses back in the reply.
    #[test]
    fn jobs_make_the_maps_warp_grids() {
        let style = r#"{"version":8,"sources":{"r":{"type":"raster","tiles":["x/{z}/{x}/{y}.png"]}},
            "layers":[{"id":"r","type":"raster","source":"r"}]}"#;
        let surface = crate::surface::SurfaceTiles {
            plane: Some(polar_plane()),
            sources: Vec::new(),
        };
        let here = TileWorker {
            surface: surface.clone(),
            ..TileWorker::new(style).unwrap()
        };
        let there = TileWorker {
            surface: crate::wire::decode(&crate::wire::encode(&surface)).unwrap(),
            ..TileWorker::new(style).unwrap()
        };
        let mercator = TileMatrixSet::web_mercator();
        let warp = orb_crs::Warp::new(mercator.clone(), *polar_plane().projection());
        for tile in [TileIndex::new(1, 1, 0), TileIndex::new(4, 9, 3)] {
            let grid = crate::image::warp_grid(&warp, tile);
            assert!(grid.is_some());
            assert_eq!(here.warp_grid(1, tile), grid);
            assert_eq!(there.warp_grid(1, tile), grid);
            let reply = JobReply::Image {
                id: 7,
                warp_grid: grid.clone(),
            };
            let back: JobReply = crate::wire::decode(&crate::wire::encode(&reply)).unwrap();
            assert!(matches!(back, JobReply::Image { warp_grid, .. } if warp_grid == grid));
        }
    }

    #[test]
    fn geojson_errors_empty_the_source_and_clusters_draw() {
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
            source_diffs: Vec::new(),
            source_json: vec![(
                1,
                "geo".into(),
                Arc::new(
                    orb_expr::json::parse(r#"{"type":"geojson","data":"test://data.json"}"#)
                        .unwrap(),
                ),
            )],
            state: vec![("k".into(), Value::Number(2.0))],
            previews: orb_style::Previews::ALL,
            glyphs: Some("test://glyphs/{fontstack}/{range}.pbf".into()),
            font_faces: vec![(
                "Noto Sans".into(),
                vec![orb_style::FontFace {
                    url: "test://noto.woff2".into(),
                    ranges: vec![[0, 0x24F], [0x600, 0x6FF]],
                }],
            )],
            local_ideographs: Some("Hiragino Sans".into()),
            surface: crate::surface::SurfaceTiles {
                plane: Some(polar_plane()),
                sources: vec![("geo".into(), TileMatrixSet::by_id("NZTM2000Quad").unwrap())],
            },
        };
        let bytes = crate::wire::encode(&update);
        let parsed: StyleUpdate = crate::wire::decode(&bytes).unwrap();
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
        let error = worker
            .build(b"invalid", Encoding::Mvt, &tile)
            .err()
            .unwrap();
        assert!(error.ends_with(orb_style::v8::EMPTY_SOURCE), "{error}");
        assert!(worker.has_data(&tile));
        // The tile that read the document reported it; the source is
        // empty after, for its retry and every other tile.
        for tile in [tile, request(TileIndex::new(1, 1, 0))] {
            let Ok(Built::Tile(build)) = worker.build(b"{}", Encoding::Mvt, &tile) else {
                panic!("an empty tile");
            };
            assert!(build.meshes.iter().all(|m| m.mesh.is_empty()));
        }
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
        fn get_range(
            &self,
            _: orb_host::ResourceKind,
            _: &str,
            _: u64,
            _: u64,
            _: orb_host::FetchDone,
        ) {
            unreachable!("no archives")
        }

        fn get(&self, _: orb_host::ResourceKind, url: &str, done: orb_host::FetchDone) {
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
            build.meshes[0].props.len() % orb_shaders::layout::SYMBOL_ROW_WORDS,
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
        let result = crate::jobs::tests::unpacked(&packed);
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
            Arc::new(crate::testing::NoDecode),
            Arc::new(orb_host::jobs::Inline),
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

    /// A range fetched ahead of the tiles serves their layout and the
    /// atlas's bitmaps: one fetch in all.
    #[test]
    fn prefetched_ranges_serve_layout_and_bitmaps() {
        let fetch = Arc::new(AssetFetch::default());
        let jobs = PoolJobs::new(
            fetch.clone(),
            Arc::new(crate::testing::NoDecode),
            Arc::new(orb_host::jobs::Inline),
        );
        let worker = Arc::new(TileWorker::new(SYMBOLS).unwrap());
        jobs.set_style(1, &StyleUpdate::default(), &worker);
        let first = worker.first_glyphs(symbol_request().zoom);
        assert_eq!(first, ["local://glyphs/Noto Sans Regular/0-255.pbf"]);
        jobs.prefetch_glyphs(1, first.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        jobs.load(symbol_request(), Box::new(move |r| tx.send(r).unwrap()));
        fetch.release();
        let tile = rx.try_iter().next().expect("laid out").unwrap();
        assert_eq!(tile.symbols.len(), 1);
        let (tx, rx) = std::sync::mpsc::channel();
        let raster = crate::GlyphRaster::Pbf {
            url: first[0].clone(),
            code_points: vec!['a'.into()],
        };
        jobs.raster(raster, Box::new(move |r| tx.send(r).unwrap()));
        let bitmaps = rx.try_iter().next().expect("drawn").unwrap();
        assert_eq!(bitmaps.len(), 1);
        let gets = fetch.gets.lock().unwrap();
        let glyph_gets = gets.iter().filter(|u| u.contains("/glyphs/")).count();
        assert_eq!(glyph_gets, 1, "{gets:?}");
    }

    /// Overzoomed tiles of one ancestor share its fetch and its decode.
    #[test]
    fn overzoomed_siblings_share_their_ancestor() {
        let fetch = Arc::new(AssetFetch::default());
        let jobs = PoolJobs::new(
            fetch.clone(),
            Arc::new(crate::testing::NoDecode),
            Arc::new(orb_host::jobs::Inline),
        );
        let worker = Arc::new(TileWorker::new(SYMBOLS).unwrap());
        jobs.set_style(1, &StyleUpdate::default(), &worker);
        jobs.prefetch_glyphs(1, worker.first_glyphs(symbol_request().zoom));
        fetch.release();
        let (z, x, y) = SYMBOL_TILE;
        let (tx, rx) = std::sync::mpsc::channel();
        for col in [x * 2, x * 2 + 1] {
            let tile = TileIndex::new(z + 1, col, y * 2);
            let request = TileRequest {
                tile,
                zoom: f64::from(z + 1),
                data_zoom: z,
                ..symbol_request()
            };
            let tx = tx.clone();
            jobs.load(request, Box::new(move |r| tx.send(r).unwrap()));
        }
        let built: Vec<_> = rx.try_iter().collect();
        assert_eq!(built.len(), 2);
        assert!(built.iter().all(Result::is_ok));
        let gets = fetch.gets.lock().unwrap();
        let tiles = gets.iter().filter(|u| u.contains("/tiles/")).count();
        assert_eq!(tiles, 1, "{gets:?}");
    }

    /// Holds every tile fetch until the test answers it.
    #[derive(Default)]
    struct HeldTiles(Mutex<Vec<orb_host::FetchDone>>);

    impl Fetch for HeldTiles {
        fn get_range(
            &self,
            _: orb_host::ResourceKind,
            _: &str,
            _: u64,
            _: u64,
            _: orb_host::FetchDone,
        ) {
            unreachable!("no archives")
        }

        fn get(&self, _: orb_host::ResourceKind, _: &str, done: orb_host::FetchDone) {
            self.0.lock().unwrap().push(done);
        }
    }

    /// Tile fetches take turns: those past [`crate::job_host::FETCHES`]
    /// wait, and one given up while it waits never starts and answers
    /// cancelled at once. One already fetching completes as usual, and
    /// each that ends starts the next.
    #[test]
    fn loads_given_up_before_their_turn_never_fetch() {
        let fetch = Arc::new(HeldTiles::default());
        let host = crate::JobHost::new(fetch.clone(), Arc::new(orb_host::jobs::Inline));
        host.set_style(1, Arc::new(TileWorker::new(SYMBOLS).unwrap()));
        let n = crate::job_host::FETCHES as u32;
        let (tx, rx) = std::sync::mpsc::channel();
        for col in 0..n + 2 {
            let tx = tx.clone();
            let tile = TileIndex::new(14, col, 0);
            let request = TileRequest {
                location: TileLocation::Url(format!("local://{col}")),
                ..request(tile)
            };
            host.load(request, Box::new(move |r| tx.send((col, r)).unwrap()));
        }
        assert_eq!(fetch.0.lock().unwrap().len(), n as usize);
        // The first waits its turn: cancelled. The second fetches already.
        host.cancel(1, 1, TileIndex::new(14, n, 0));
        host.cancel(1, 1, TileIndex::new(14, 0, 0));
        let answered: Vec<_> = rx.try_iter().collect();
        assert!(
            matches!(answered[..], [(col, Ok(crate::Loaded::Cancelled))] if col == n),
            "{answered:?}"
        );
        // A fetch ends: the last waiting one starts.
        let first = fetch.0.lock().unwrap().remove(0);
        first(Err(orb_host::FetchError::NotFound));
        assert!(matches!(
            rx.try_iter().next(),
            Some((0, Ok(crate::Loaded::Missing)))
        ));
        assert_eq!(fetch.0.lock().unwrap().len(), n as usize);
    }

    /// A glyph range another context fetches for the job hosts: a tile
    /// that needs it waits for it rather than fetch it again.
    #[test]
    fn expected_ranges_arrive_from_elsewhere() {
        let fetch = Arc::new(AssetFetch::default());
        let host = crate::JobHost::new(fetch.clone(), Arc::new(orb_host::jobs::Inline));
        let worker = Arc::new(TileWorker::new(SYMBOLS).unwrap());
        host.set_style(1, Arc::clone(&worker));
        let first = worker.first_glyphs(symbol_request().zoom);
        host.expect_glyphs(1, first.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        host.load(symbol_request(), Box::new(move |r| tx.send(r).unwrap()));
        assert_eq!(rx.try_iter().count(), 0, "layout waits for the range");
        let path = format!("{ASSETS}/{}", first[0].strip_prefix("local://").unwrap());
        host.glyph_range(1, first[0].clone(), Ok(std::fs::read(path).unwrap()));
        let Some(Ok(crate::Loaded::Tile(tile))) = rx.try_iter().next() else {
            panic!("laid out")
        };
        assert_eq!(tile.symbols.len(), 1);
        let gets = fetch.gets.lock().unwrap();
        assert!(!gets.iter().any(|u| u.contains("/glyphs/")), "{gets:?}");
    }

    /// A tile whose bytes arrive after its style was replaced isn't built:
    /// the map drops a replaced style's results.
    #[test]
    fn loads_of_a_replaced_style_are_not_built() {
        struct Held(Mutex<Vec<orb_host::FetchDone>>);
        impl Fetch for Held {
            fn get(&self, _: orb_host::ResourceKind, _: &str, done: orb_host::FetchDone) {
                self.0.lock().unwrap().push(done);
            }
            fn get_range(
                &self,
                _: orb_host::ResourceKind,
                _: &str,
                _: u64,
                _: u64,
                _: orb_host::FetchDone,
            ) {
                unreachable!("no archives")
            }
        }
        let fetch = Arc::new(Held(Mutex::default()));
        let host = crate::JobHost::new(fetch.clone(), Arc::new(orb_host::jobs::Inline));
        let worker = Arc::new(TileWorker::new(SYMBOLS).unwrap());
        host.set_style(1, Arc::clone(&worker));
        let (tx, rx) = std::sync::mpsc::channel();
        host.load(symbol_request(), Box::new(move |r| tx.send(r).unwrap()));
        host.set_style(2, worker);
        let (z, x, y) = SYMBOL_TILE;
        let bytes = std::fs::read(format!("{ASSETS}/tiles/{z}-{x}-{y}.mvt")).unwrap();
        for done in std::mem::take(&mut *fetch.0.lock().unwrap()) {
            done(Ok(bytes.clone()));
        }
        let Some(Err(e)) = rx.try_iter().next() else {
            panic!("answered without a build")
        };
        assert_eq!(e, "style 1 was replaced");
    }

    /// A tile asked for while the sprites load is fetched at once, and laid
    /// out once the icons it waits for arrive.
    #[test]
    fn tiles_load_while_icons_do_and_lay_out_after() {
        let fetch = Arc::new(AssetFetch::default());
        let jobs = PoolJobs::new(
            fetch.clone(),
            Arc::new(crate::testing::NoDecode),
            Arc::new(orb_host::jobs::Inline),
        );
        let worker = Arc::new(TileWorker::new(SYMBOLS).unwrap());
        jobs.set_style(1, &StyleUpdate::default(), &worker);
        jobs.prefetch_glyphs(1, worker.first_glyphs(symbol_request().zoom));
        fetch.release();
        let (tx, rx) = std::sync::mpsc::channel();
        let request = TileRequest {
            icons: 1,
            ..symbol_request()
        };
        jobs.load(request, Box::new(move |r| tx.send(r).unwrap()));
        let gets = fetch.gets.lock().unwrap().clone();
        assert!(gets.iter().any(|u| u.contains("/tiles/")), "{gets:?}");
        assert_eq!(rx.try_iter().count(), 0, "layout waits for the icons");
        let mut icons = SpriteIcons::default();
        icons.epoch = 1;
        jobs.set_icons(1, &Arc::new(icons), &worker);
        let tile = rx.try_iter().next().expect("laid out").unwrap();
        assert_eq!(tile.symbols.len(), 1);
    }

    #[test]
    fn sprite_icons_cross_workers() {
        let mut icons = SpriteIcons::default();
        icons.pixel_ratio = 2.0;
        icons.add(
            "extra:pill",
            [40, 20],
            &orb_layout::ImageMetadata {
                pixel_ratio: 2.0,
                sdf: true,
                stretch_x: vec![[4.0, 36.0]],
                stretch_y: Vec::new(),
                content: Some([4.0, 2.0, 36.0, 18.0]),
                text_fit_width: Some("proportional".into()),
                text_fit_height: None,
            },
        );
        icons.add("dot", [8, 8], &orb_layout::ImageMetadata::default());
        assert_eq!(
            crate::wire::decode::<SpriteIcons>(&crate::wire::encode(&icons)).unwrap(),
            icons
        );
    }
}

//! Tile jobs: what a map asks of its job executor, and the
//! results it uploads. A map hands its jobs each style revision as a
//! [`StyleUpdate`] (the layers and sources new in it), then requests tiles.
//! Natively [`PoolJobs`] runs a [`TileWorker`](crate::TileWorker) on an
//! [`Executor`]; the web's job workers each run one and ship results back
//! packed (see [`pack`]).

use std::sync::Arc;

use orb_hal::{HostBytesId, HostImageId};
use orb_host::jobs::Executor;
use orb_host::{DecodedImage, Fetch, ImageDecode};
use orb_layout::Chunk;

use crate::glyphs::{GlyphRaster, LocalGlyphs, RasterDone};
use crate::style::ImageKind;
use crate::symbol::{ResultSymbols, SpriteIcons};
use crate::tile::{ImageRef, TileBuild};
use crate::worker::{DemUnpack, StyleUpdate, TileRequest, TileWorker};

/// Called once with what a tile's load came to, on any thread.
pub type TileDone = Box<dyn FnOnce(Result<TileOutcome, String>) + Send + 'static>;

/// What a tile's load came to, unless it failed.
#[derive(Debug)]
#[allow(clippy::large_enum_variant, reason = "moved once, in a boxed message")]
pub enum TileOutcome {
    /// Laid out, or its image decoded: ready to upload.
    Built(TileResult),
    /// The tileset doesn't have the tile; nearby zooms stand in. A worker
    /// that set it aside (it took a worker down) says why in `warnings`.
    Missing { warnings: Vec<String> },
    /// The map gave the load up before its bytes came
    /// ([`Jobs::cancel`]): nothing was loaded.
    Cancelled,
}

#[cfg(test)]
impl TileOutcome {
    /// The built tile, which a test expects.
    pub(crate) fn built(self) -> TileResult {
        match self {
            Self::Built(build) => build,
            other => panic!("not built: {other:?}"),
        }
    }
}

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
    /// [`Jobs::prefetch_glyphs`]: glyph ranges the sender fetches once
    /// for every job host and hands in with [`GlyphRange`](Self::GlyphRange)
    /// ([`JobHost::expect_glyphs`](crate::JobHost::expect_glyphs)).
    Glyphs {
        style: u64,
        urls: Vec<String>,
    },
    /// One of them, fetched or failed.
    GlyphRange {
        style: u64,
        url: String,
        bytes: Result<Vec<u8>, String>,
    },
    /// [`Jobs::retry_glyphs`].
    RetryGlyphs,
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
    /// [`Jobs::add_archive_head`].
    ArchiveHead {
        url: String,
        head: Vec<u8>,
    },
    /// [`Jobs::cancel`].
    Cancel {
        style: u64,
        source: u32,
        tile: orb_crs::TileIndex,
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
    /// An image tile: the bitmap, and the warp grid its request asked
    /// for.
    Image {
        id: u32,
        warp_grid: Option<orb_crs::WarpGrid>,
    },
    /// A terrain DEM tile: the buffer holds its texels, then its
    /// elevations as little-endian `f32`s.
    Dem {
        id: u32,
        width: u32,
    },
    /// The source doesn't have the tile.
    Missing {
        id: u32,
    },
    /// The load was given up before its fetch started.
    Cancelled {
        id: u32,
    },
    /// A load's fetch ended (its reply comes later): the page's turn
    /// ([`Turns`](crate::Turns)) passes to the next.
    Fetched {
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
    Error {
        id: u32,
        message: String,
    },
}

impl JobReply {
    pub fn id(&self) -> u32 {
        match self {
            Self::Tile { id }
            | Self::Image { id, .. }
            | Self::Dem { id, .. }
            | Self::Missing { id }
            | Self::Cancelled { id }
            | Self::Fetched { id }
            | Self::Features { id, .. }
            | Self::Answer { id, .. }
            | Self::Glyphs { id, .. }
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
    /// The tile fetches these jobs started, aborted once started and
    /// completed so far.
    fn fetch_counts(&self) -> crate::FetchCounts {
        crate::FetchCounts::default()
    }
    /// Ranks the loads waiting for their fetches again: each that `rank`
    /// ranks takes that rank ([`Turns::reorder`](crate::Turns::reorder)).
    fn reorder(&self, _rank: &dyn Fn(crate::LoadKey) -> Option<u64>) {}
    /// Gives up the load of `tile` of style `style`'s source `source`: one
    /// whose fetch hasn't started never does and answers
    /// [`TileResult::cancelled`]; others complete as usual.
    fn cancel(&self, _style: u64, _source: u32, _tile: orb_crs::TileIndex) {}
    /// Lets go of what the jobs keep of `tile` of style `style`'s source
    /// `source` (its decoded frame), so that a later load fetches it again:
    /// a map without a tile cache keeps nothing of tiles out of view.
    fn forget(&self, _style: u64, _source: u32, _tile: orb_crs::TileIndex) {}
    /// Starts fetching glyph ranges style `id`'s tiles are about to need,
    /// so that their layout finds them in.
    fn prefetch_glyphs(&self, _id: u64, _urls: Vec<String>) {}
    /// Lets the glyph ranges that failed be fetched again by the tiles
    /// that need them, in every job host.
    fn retry_glyphs(&self) {}
    /// PMTiles archive `url` begins with `head`, which the map read for its
    /// header: loads of its tiles start from these bytes rather than read
    /// them again.
    fn add_archive_head(&self, url: &str, head: &Arc<Vec<u8>>);
    /// Rasterizes glyphs for the atlas: one deduplicated job per unit, run
    /// by the unit's owner worker.
    fn raster(&self, request: GlyphRaster, done: RasterDone);
    /// Reads features of a tile for a query, in the worker that loaded it,
    /// from its cached frame or from the tile fetched again.
    fn features(&self, request: crate::FeatureRequest, done: crate::FeaturesDone);
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

    /// Draws local glyphs with the host's fonts.
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

    fn prefetch_glyphs(&self, id: u64, urls: Vec<String>) {
        self.host.prefetch_glyphs(id, urls);
    }

    fn retry_glyphs(&self) {
        self.host.retry_glyphs();
    }

    fn add_archive_head(&self, url: &str, head: &Arc<Vec<u8>>) {
        self.host.add_archive_head(url, head);
    }

    fn features(&self, request: crate::FeatureRequest, done: crate::FeaturesDone) {
        self.host.features(request, done);
    }

    fn cluster(&self, request: crate::ClusterRequest, done: crate::QueryDone) {
        self.host.cluster(request, done);
    }

    fn cancel(&self, style: u64, source: u32, tile: orb_crs::TileIndex) {
        self.host.cancel(style, source, tile);
    }

    fn fetch_counts(&self) -> crate::FetchCounts {
        self.host.fetch_counts()
    }

    fn reorder(&self, rank: &dyn Fn(crate::LoadKey) -> Option<u64>) {
        self.host.reorder(rank);
    }

    fn forget(&self, style: u64, source: u32, tile: orb_crs::TileIndex) {
        self.host.forget(style, source, tile);
    }

    fn load(&self, request: TileRequest, done: TileDone) {
        let (decode, executor) = (Arc::clone(&self.decode), Arc::clone(&self.executor));
        let (dem_unpack, rank) = (request.dem_unpack, request.rank);
        self.host.load(
            request,
            None,
            Box::new(move |loaded| match loaded {
                Ok(crate::Loaded::Tile(build)) => done(Ok(TileOutcome::Built(build.into()))),
                Ok(crate::Loaded::Missing) => done(Ok(TileOutcome::Missing {
                    warnings: Vec::new(),
                })),
                Ok(crate::Loaded::Cancelled) => done(Ok(TileOutcome::Cancelled)),
                Ok(crate::Loaded::Image {
                    bytes,
                    kind,
                    warp_grid,
                }) => executor.spawn_ranked(
                    rank,
                    Box::new(move || {
                        decode.decode(
                            bytes,
                            Box::new(move |decoded| {
                                done(decoded.and_then(|i| {
                                    let result = image_result(kind, i, dem_unpack)?;
                                    Ok(TileOutcome::Built(TileResult {
                                        warp_grid,
                                        ..result
                                    }))
                                }))
                            }),
                        );
                    }),
                ),
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

/// A drawn picture's premultiplied RGBA8 as straight alpha, in place, for
/// PNGs and `ImageData`. Light added over transparency (the globe's
/// atmosphere in space, blended additively as in GL JS) leaves more
/// colour than alpha, which straight alpha can't hold: its alpha rises to
/// its brightest channel, so it keeps its colour and shows exactly as the
/// canvas does over black. Dividing by the old alpha would clamp the halo
/// to white.
pub fn unpremultiply(pixels: &mut [u8]) {
    for texel in pixels.as_chunks_mut::<4>().0 {
        let a = texel.iter().copied().max().expect("four bytes");
        texel[3] = a;
        let a = u32::from(a);
        if a != 0 && a != 255 {
            for c in &mut texel[..3] {
                *c = ((u32::from(*c) * 255 + a / 2) / a) as u8;
            }
        }
    }
}

/// A finished tile, ready to upload: its words laid out once in its
/// [`Payload`], in core memory or in host-held bytes.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TileResult {
    pub warnings: Vec<String>,
    /// Images the tile's symbol layers name that the style doesn't have.
    pub missing_images: Vec<String>,
    /// Some label drew a glyph from a range that failed.
    pub glyphs_failed: bool,
    /// Symbol nodes, by key, left out while their glyphs load: the map
    /// lays them out again once the tile is in. Labels never hold up a
    /// tile's other layers.
    pub labels_later: Vec<u32>,
    /// The style revision the tile was laid out with.
    pub revision: u64,
    /// Keys of the nodes laid out; those without a mesh have no features.
    pub nodes: Vec<u32>,
    /// Meshes carry property rows only (a column job).
    pub rows_only: bool,
    /// Laid out as a neighbour of the tile Tile mode draws: only what
    /// reaches into the tiles beside it.
    pub neighbour: bool,
    /// Feature ids the meshes' state ids index.
    pub ids: Vec<String>,
    pub meshes: Vec<ResultMesh>,
    /// Per source layer laid out, the geometry its meshes' hits point into.
    pub hit_geometry: Vec<Arc<crate::query::HitGeometry>>,
    /// Per symbol layer laid out, its bucket. Its icon and text nodes'
    /// meshes carry property rows only.
    pub symbols: Vec<ResultSymbols>,
    /// Every stream's words; host bytes are released after upload.
    #[serde(skip)]
    pub payload: Payload,
    /// An image source's tile.
    #[serde(skip)]
    pub image: Option<TileImage>,
    /// An image tile drawn on another plane: its warp's grid there
    /// ([`TileRequest::warp`](crate::TileRequest::warp)).
    #[serde(skip)]
    pub warp_grid: Option<Arc<orb_crs::WarpGrid>>,
    /// A terrain DEM tile's elevations.
    #[serde(skip)]
    pub dem: Option<Arc<orb_tile::DemGrid>>,
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
    /// Texels at byte `offset` of host-held bytes, ready to store (a DEM
    /// tile's, decoded in a job worker); released after upload.
    Bytes { bytes: HostBytesId, offset: u64 },
}

impl ImagePixels {
    /// The pixels for the renderer, a host image premultiplied on upload
    /// if `premultiply`.
    pub(crate) fn upload(&self, premultiply: bool) -> orb_render::Pixels<'_> {
        match self {
            Self::Core(bytes) => orb_render::Pixels::Core(bytes),
            &Self::Host(image) => orb_render::Pixels::Host { image, premultiply },
            &Self::Bytes { bytes, offset } => orb_render::Pixels::HostBytes { bytes, offset },
        }
    }
}

/// One render node's share of a finished tile.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ResultMesh {
    pub hits: Arc<crate::query::MeshHits>,
    /// The node's key.
    pub node: u32,
    pub extent: u32,
    pub chunks: Vec<Chunk>,
    pub vertices: Stream,
    pub indices: Stream,
    pub props: Stream,
    /// The earlier mesh of the result whose vertices, and whose indices,
    /// equal its own, which it leaves empty ([`TileResult::vertices`]).
    pub shares: [Option<u32>; 2],
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

/// A tile's words, laid out once in the order they upload: every mesh's
/// vertices, then every mesh's indices, then the scene's meshes' property
/// rows and the symbol nodes' after them, then each symbol layer's tables
/// as placement reads them. Each pool's words are contiguous, so a tile
/// uploads with one write per pool. It moves rather than copies: a `Vec`
/// natively, and on the web the buffer a job worker transferred.
#[derive(Debug, Default)]
pub enum Payload {
    /// A result with no words (an image tile's).
    #[default]
    None,
    Core(Vec<u32>),
    /// `words` words at byte `base` of host-held bytes.
    Host {
        bytes: HostBytesId,
        base: u64,
        words: u32,
    },
}

impl Payload {
    /// All its words, for an upload.
    pub(crate) fn words(&self) -> orb_render::Words<'_> {
        match *self {
            Self::None => orb_render::Words::Core(&[]),
            Self::Core(ref words) => orb_render::Words::Core(words),
            Self::Host { bytes, base, words } => orb_render::Words::Host {
                bytes,
                offset: base,
                len: words,
            },
        }
    }

    /// The host bytes it is in, which go once it's uploaded.
    pub(crate) fn host_bytes(&self) -> Option<HostBytesId> {
        match *self {
            Self::Host { bytes, .. } => Some(bytes),
            _ => None,
        }
    }
}

/// A span of a tile's [`Payload`]: `len` words from word `at`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Stream {
    pub at: u32,
    pub len: u32,
}

impl Stream {
    /// Its length in words.
    pub(crate) fn len(self) -> u32 {
        self.len
    }

    /// Its words in the payload.
    pub(crate) fn range(self) -> std::ops::Range<u32> {
        self.at..self.at + self.len
    }

    /// Its words for an upload, in `payload`.
    pub(crate) fn words(self, payload: &Payload) -> orb_render::Words<'_> {
        payload.words().slice(self.range())
    }

    /// Its words in core memory, read back from the host's bytes when the
    /// payload is there.
    pub(crate) fn core_words<'a>(
        self,
        gpu: &dyn orb_hal::Gpu,
        payload: &'a Payload,
    ) -> std::borrow::Cow<'a, [u32]> {
        match self.words(payload) {
            orb_render::Words::Core(words) => std::borrow::Cow::Borrowed(words),
            orb_render::Words::Host { bytes, offset, len } => {
                let bytes = gpu.read_host_bytes(bytes, offset..offset + u64::from(len) * 4);
                // Host bytes come unaligned (copied from a JS buffer).
                std::borrow::Cow::Owned(bytemuck::pod_collect_to_vec(&bytes))
            }
        }
    }
}

/// Lays a tile's streams out one after another in its payload.
#[derive(Debug, Default)]
pub(crate) struct Arena {
    words: Vec<u32>,
}

impl Arena {
    /// Words, as they are.
    pub(crate) fn push(&mut self, words: &[u32]) -> Stream {
        let at = self.words.len() as u32;
        self.words.extend_from_slice(words);
        Stream {
            at,
            len: words.len() as u32,
        }
    }

    /// Plain values: their bytes, zero-padded to a word.
    pub(crate) fn push_pod<T: bytemuck::NoUninit>(&mut self, rows: &[T]) -> Stream {
        if let Ok(words) = bytemuck::try_cast_slice::<T, u32>(rows) {
            return self.push(words);
        }
        let bytes: &[u8] = bytemuck::cast_slice(rows);
        let at = self.words.len();
        self.words.resize(at + bytes.len().div_ceil(4), 0);
        bytemuck::cast_slice_mut::<u32, u8>(&mut self.words[at..])[..bytes.len()]
            .copy_from_slice(bytes);
        Stream {
            at: at as u32,
            len: (self.words.len() - at) as u32,
        }
    }
}

impl From<TileBuild> for TileResult {
    fn from(build: TileBuild) -> Self {
        let hit_geometry: Vec<_> = build.hits.into_iter().map(|(_, g)| Arc::new(g)).collect();
        let shares = share_geometry(&build.meshes);
        let symbol_nodes: Vec<u32> = (build.symbols.iter())
            .flat_map(|(layer, _)| [0, 1].map(|n| crate::style::node_key(*layer, n)))
            .collect();
        let mut arena = Arena::default();
        let mesh_words: usize = (build.meshes.iter())
            .map(|m| m.mesh.vertices.len() + m.mesh.indices.len() + m.props.len())
            .sum();
        arena.words.reserve(mesh_words);
        let mut streams = vec![[Stream::default(); 3]; build.meshes.len()];
        for pool in 0..2 {
            for (i, m) in build.meshes.iter().enumerate() {
                if shares[i][pool].is_none() {
                    streams[i][pool] = arena.push([&m.mesh.vertices, &m.mesh.indices][pool]);
                }
            }
        }
        // The scene's rows together, then those that go up with labels.
        for symbol in [false, true] {
            for (i, m) in build.meshes.iter().enumerate() {
                if symbol_nodes.contains(&m.node) == symbol {
                    streams[i][2] = arena.push(&m.props);
                }
            }
        }
        let symbols = (build.symbols.iter())
            .map(|(layer, s)| ResultSymbols::new(*layer, s, &mut arena))
            .collect();
        let mut result = Self {
            meshes: (build.meshes.into_iter().zip(streams).zip(shares))
                .map(|((m, [vertices, indices, props]), shares)| ResultMesh {
                    node: m.node,
                    hits: Arc::new(m.hits),
                    extent: m.extent,
                    chunks: m.mesh.chunks,
                    vertices,
                    indices,
                    props,
                    shares,
                    row_words: m.row_words,
                    maxima: m.maxima,
                    images: m.images,
                    state_ids: m.state_ids,
                    data: m.data,
                })
                .collect(),
            symbols,
            revision: build.revision,
            nodes: build.nodes,
            rows_only: build.rows_only,
            neighbour: build.neighbour,
            ids: build.ids,
            payload: Payload::Core(arena.words),
            warnings: build.warnings,
            missing_images: build.missing_images,
            glyphs_failed: build.glyphs_failed,
            labels_later: build.labels_later,
            image: None,
            warp_grid: None,
            dem: None,
            hit_geometry,
        };
        result.link_hits();
        result
    }
}

/// Per mesh, the earlier mesh whose vertices, and whose indices, equal its
/// own, so that it lays none of them out itself. A road's casing and its
/// fill lay out the same lines, and a fill and its outline the same
/// vertices: as GL JS and Native group layers that lay out alike into one
/// bucket, the words cross to the map and upload once.
fn share_geometry(meshes: &[crate::tile::NodeMesh]) -> Vec<[Option<u32>; 2]> {
    fn streams(m: &crate::tile::NodeMesh) -> [&Vec<u32>; 2] {
        [&m.mesh.vertices, &m.mesh.indices]
    }
    let mut shares = vec![[None; 2]; meshes.len()];
    for (i, mesh) in meshes.iter().enumerate() {
        for (s, words) in streams(mesh).into_iter().enumerate() {
            if words.is_empty() {
                continue;
            }
            // The first that has them, which shares nothing of them.
            let same = (meshes[..i].iter().enumerate())
                .find(|&(e, earlier)| shares[e][s].is_none() && streams(earlier)[s] == words);
            shares[i][s] = same.map(|(of, _)| of as u32);
        }
    }
    shares
}

impl TileResult {
    /// A mesh's vertices: its own, or those of the mesh it shares them
    /// with.
    pub(crate) fn vertices(&self, mesh: &ResultMesh) -> Stream {
        mesh.shares[0].map_or(mesh.vertices, |of| self.meshes[of as usize].vertices)
    }

    /// A mesh's indices: its own, or those of the mesh it shares them
    /// with.
    pub(crate) fn indices(&self, mesh: &ResultMesh) -> Stream {
        mesh.shares[1].map_or(mesh.indices, |of| self.meshes[of as usize].indices)
    }

    /// Points its meshes' hits at its hit geometry.
    fn link_hits(&mut self) {
        for mesh in &mut self.meshes {
            Arc::get_mut(&mut mesh.hits)
                .expect("a result's hits are its own")
                .link(&self.hit_geometry);
        }
    }

    /// Whether every stream lies in a payload of `words` words, and every
    /// share names an earlier mesh that has its own: what a packed tile
    /// from another worker must hold to.
    fn holds_together(&self, words: u32) -> bool {
        let inside = |s: &Stream| s.at.checked_add(s.len).is_some_and(|end| end <= words);
        let meshes = self.meshes.iter().enumerate().all(|(i, m)| {
            let shares = (0..2).all(|s| {
                m.shares[s].is_none_or(|of| {
                    (of as usize) < i && self.meshes[of as usize].shares[s].is_none()
                })
            });
            shares && [m.vertices, m.indices, m.props].iter().all(inside)
        });
        meshes && (self.symbols.iter()).all(|s| s.tables.iter().all(inside))
    }
}

/// A tile packed for another worker: `head` holds the header's length in
/// bytes (a little-endian u32), the header (the result without its
/// payload, [`crate::wire`]) and padding to a word; the payload's words
/// follow it as they are.
#[derive(Debug)]
pub struct Packed {
    pub head: Vec<u8>,
    pub words: Vec<u32>,
}

// Packed words are the payload's own bytes: every target Orbis ships on
// is little-endian.
const _: () = assert!(cfg!(target_endian = "little"));

impl Packed {
    /// Its bytes in one buffer, as they travel.
    pub fn to_bytes(&self) -> Vec<u8> {
        [&self.head[..], self.payload_bytes()].concat()
    }

    /// The payload's bytes, which follow the head.
    pub fn payload_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }

    /// Its length in bytes.
    pub fn len(&self) -> usize {
        self.head.len() + self.words.len() * 4
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Packs a build for another worker ([`Packed`]): its payload moves in as
/// it is.
pub fn pack(build: TileBuild) -> Packed {
    let mut result = TileResult::from(build);
    let Payload::Core(words) = std::mem::take(&mut result.payload) else {
        unreachable!("built payloads are in core memory")
    };
    let header = crate::wire::encode(&result);
    let mut head = Vec::with_capacity((header.len() + 4).next_multiple_of(4));
    head.extend((header.len() as u32).to_le_bytes());
    head.extend(&header);
    head.resize(head.len().next_multiple_of(4), 0);
    Packed { head, words }
}

/// The header of a packed tile (see [`pack`]): the bytes after its length.
pub fn packed_header(packed: &[u8]) -> Result<&[u8], String> {
    let bad = || "malformed packed tile".to_string();
    let len = packed.get(..4).ok_or_else(bad)?;
    let len = u32::from_le_bytes(len.try_into().expect("four bytes")) as usize;
    packed.get(4..4 + len).ok_or_else(bad)
}

/// Reads a packed tile's `header` (see [`packed_header`]): its payload
/// stays in the packed buffer, `len` bytes registered as `bytes`, and need
/// not be in core memory.
pub fn unpack(header: &[u8], bytes: HostBytesId, len: u64) -> Result<TileResult, String> {
    let mut result: TileResult = crate::wire::decode(header)?;
    let base = (4 + header.len()).next_multiple_of(4) as u64;
    let words = (len.checked_sub(base)).and_then(|b| u32::try_from(b / 4).ok());
    let Some(words) = words.filter(|&w| result.holds_together(w)) else {
        return Err("malformed packed tile".into());
    };
    result.payload = Payload::Host { bytes, base, words };
    result.link_hits();
    Ok(result)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tile::NodeMesh;
    use orb_layout::Mesh;

    /// A packed tile read back, its payload in host bytes.
    pub(crate) fn unpacked(packed: &[u8]) -> TileResult {
        let header = packed_header(packed).unwrap();
        unpack(header, HostBytesId(3), packed.len() as u64).unwrap()
    }

    #[test]
    fn premultiplies_like_gl_js() {
        let mut texels = [255, 128, 3, 128, 10, 20, 30, 0, 9, 8, 7, 255];
        premultiply(&mut texels);
        // Math.round(c * a / 255).
        assert_eq!(texels, [128, 64, 2, 128, 0, 0, 0, 0, 9, 8, 7, 255]);
    }

    /// Ordinary pixels divide by their alpha; the atmosphere's halo in
    /// space (colour above alpha) keeps its colour instead of turning
    /// white.
    #[test]
    fn unpremultiplies_added_light_without_whitening_it() {
        let mut texels = [64, 32, 0, 128, 55, 80, 100, 13, 7, 10, 13, 0, 9, 8, 7, 255];
        unpremultiply(&mut texels);
        assert_eq!(
            texels,
            [
                128, 64, 0, 128, 140, 204, 255, 100, 137, 196, 255, 13, 9, 8, 7, 255
            ]
        );
    }

    fn mesh(node: u32, vertices: Vec<u32>, props: Vec<u32>) -> NodeMesh {
        NodeMesh {
            hits: crate::query::MeshHits::default(),
            node,
            extent: 4096,
            mesh: Mesh {
                indices: vec![node; 2],
                vertices,
                ..Mesh::default()
            },
            props,
            row_words: 6,
            maxima: vec![2.5],
            images: Vec::new(),
            state_ids: Some(vec![1, crate::tile::NO_ID]),
            data: Vec::new(),
        }
    }

    /// A packed tile unpacks as it was, a mesh's words that equal an
    /// earlier one's shared rather than sent twice.
    #[test]
    fn packs_and_unpacks() {
        let packed = pack(TileBuild {
            revision: 1 << 33 | 5,
            nodes: vec![3, 4, 5],
            warnings: vec!["é".into()],
            meshes: vec![
                mesh(3, vec![1, 2, 3], vec![9; 12]),
                mesh(4, vec![], vec![]),
                mesh(5, vec![1, 2, 3], vec![8; 12]),
            ],
            ..TileBuild::default()
        });
        let bytes = packed.to_bytes();
        assert_eq!(bytes.len(), packed.len());
        let words: Vec<u32> = bytemuck::pod_collect_to_vec(&bytes);
        let result = unpacked(&bytes);
        let Payload::Host {
            bytes: id, base, ..
        } = result.payload
        else {
            panic!("a host payload")
        };
        assert_eq!((id, base), (HostBytesId(3), packed.head.len() as u64));
        assert_eq!(
            (result.revision, &result.warnings[..]),
            (1 << 33 | 5, &["é".to_string()][..])
        );
        assert_eq!(
            result.meshes[1].state_ids,
            Some(vec![1, crate::tile::NO_ID])
        );
        let stream = |s: Stream| &words[base as usize / 4..][s.at as usize..][..s.len as usize];
        assert_eq!(stream(result.meshes[0].vertices), [1, 2, 3]);
        assert_eq!(stream(result.meshes[0].props), [9; 12]);
        let shared = &result.meshes[2];
        assert_eq!(shared.shares, [Some(0), None]);
        assert!(stream(shared.vertices).is_empty());
        assert_eq!(stream(result.vertices(shared)), [1, 2, 3]);
        assert_eq!(stream(shared.props), [8; 12]);
        assert!(packed_header(&bytes[..3]).is_err());
        assert!(unpack(&bytes[4..8], HostBytesId(0), 8).is_err());
    }

    /// A packed tile whose streams reach past its payload, or that shares
    /// a later mesh's words, is malformed, not a panic.
    #[test]
    fn a_packed_tile_must_hold_together() {
        let packed = pack(TileBuild {
            meshes: vec![mesh(3, vec![1, 2], vec![9; 6]), mesh(4, vec![1, 2], vec![])],
            ..TileBuild::default()
        });
        let header = packed_header(&packed.head).unwrap();
        let len = packed.len() as u64;
        assert!(unpack(header, HostBytesId(0), len).is_ok());
        assert!(
            unpack(header, HostBytesId(0), len - 4).is_err(),
            "cut short"
        );
        let mut result = unpack(header, HostBytesId(0), len).unwrap();
        result.payload = Payload::None;
        result.meshes.swap(0, 1);
        let header = crate::wire::encode(&result);
        let len = ((4 + header.len()).next_multiple_of(4) + packed.words.len() * 4) as u64;
        assert!(
            unpack(&header, HostBytesId(0), len).is_err(),
            "a later share"
        );
    }
}

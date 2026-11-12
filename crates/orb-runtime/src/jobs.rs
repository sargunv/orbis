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
    /// The load was given up before its fetch started.
    Cancelled {
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

    fn forget(&self, style: u64, source: u32, tile: orb_crs::TileIndex) {
        self.host.forget(style, source, tile);
    }

    fn load(&self, request: TileRequest, done: TileDone) {
        let (decode, executor) = (Arc::clone(&self.decode), Arc::clone(&self.executor));
        let (dem_unpack, rank) = (request.dem_unpack, request.rank);
        self.host.load(
            request,
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

/// A finished tile, ready to upload: in core memory, or in host-held bytes.
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
    /// Holds every [`Stream::Host`] of the meshes and symbols; released
    /// after upload.
    #[serde(skip)]
    pub host_bytes: Option<HostBytesId>,
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
}

impl ImagePixels {
    /// The pixels for the renderer, a host image premultiplied on upload
    /// if `premultiply`.
    pub(crate) fn upload(&self, premultiply: bool) -> orb_render::Pixels<'_> {
        match self {
            Self::Core(bytes) => orb_render::Pixels::Core(bytes),
            &Self::Host(image) => orb_render::Pixels::Host { image, premultiply },
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

impl Stream {
    /// Its length in words.
    pub(crate) fn len(&self) -> u32 {
        match self {
            Self::Core(words) => words.len() as u32,
            Self::Host { len, .. } => *len,
        }
    }

    /// Its words for an upload, in core memory or in the result's host
    /// `bytes`.
    pub(crate) fn words(&self, bytes: Option<HostBytesId>) -> orb_render::Words<'_> {
        match self {
            Self::Core(words) => orb_render::Words::Core(words),
            Self::Host { offset, len } => orb_render::Words::Host {
                bytes: bytes.expect("host streams come with host bytes"),
                offset: *offset,
                len: *len,
            },
        }
    }

    /// Its words in core memory, read back from the host's `bytes` when
    /// they are there.
    pub(crate) fn core_words(
        &self,
        gpu: &dyn orb_hal::Gpu,
        bytes: Option<HostBytesId>,
    ) -> std::borrow::Cow<'_, [u32]> {
        match self {
            Self::Core(words) => std::borrow::Cow::Borrowed(words),
            Self::Host { offset, len } => {
                let bytes = gpu.read_host_bytes(
                    bytes.expect("host streams come with host bytes"),
                    *offset..offset + u64::from(*len) * 4,
                );
                let words = bytes.as_chunks::<4>().0.iter();
                std::borrow::Cow::Owned(words.map(|w| u32::from_le_bytes(*w)).collect())
            }
        }
    }
}

impl From<TileBuild> for TileResult {
    fn from(build: TileBuild) -> Self {
        let hit_geometry: Vec<_> = build.hits.into_iter().map(|(_, g)| Arc::new(g)).collect();
        let mut result = Self {
            meshes: build
                .meshes
                .into_iter()
                .map(|m| ResultMesh {
                    node: m.node,
                    hits: Arc::new(m.hits),
                    extent: m.extent,
                    chunks: m.mesh.chunks,
                    vertices: Stream::Core(m.mesh.vertices),
                    indices: Stream::Core(m.mesh.indices),
                    props: Stream::Core(m.props),
                    shares: [None; 2],
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
            neighbour: build.neighbour,
            ids: build.ids,
            host_bytes: None,
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
        result.share_geometry();
        result
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
    /// A mesh's vertices: its own, or those of the mesh it shares them
    /// with.
    pub(crate) fn vertices<'a>(&'a self, mesh: &'a ResultMesh) -> &'a Stream {
        mesh.shares[0].map_or(&mesh.vertices, |of| &self.meshes[of as usize].vertices)
    }

    /// A mesh's indices: its own, or those of the mesh it shares them
    /// with.
    pub(crate) fn indices<'a>(&'a self, mesh: &'a ResultMesh) -> &'a Stream {
        mesh.shares[1].map_or(&mesh.indices, |of| &self.meshes[of as usize].indices)
    }

    /// Points each mesh's vertices and indices that equal an earlier
    /// mesh's at that mesh, leaving its own empty. A road's casing and
    /// its fill lay out the same lines, and a fill and its outline the
    /// same vertices: as GL JS and Native group layers that lay out alike
    /// into one bucket, the words pack, cross to the map and upload once.
    fn share_geometry(&mut self) {
        for i in 1..self.meshes.len() {
            let (earlier, rest) = self.meshes.split_at_mut(i);
            let mesh = &mut rest[0];
            let streams = [&mut mesh.vertices, &mut mesh.indices];
            for (s, stream) in streams.into_iter().enumerate() {
                let Stream::Core(words) = stream else {
                    continue;
                };
                if words.is_empty() {
                    continue;
                }
                let same = earlier.iter().position(|e| {
                    let theirs = [&e.vertices, &e.indices][s];
                    matches!(theirs, Stream::Core(w) if w == words)
                });
                if let Some(of) = same {
                    *stream = Stream::Core(Vec::new());
                    mesh.shares[s] = Some(of as u32);
                }
            }
        }
    }

    /// Points its meshes' hits at its hit geometry.
    fn link_hits(&mut self) {
        for mesh in &mut self.meshes {
            Arc::get_mut(&mut mesh.hits)
                .expect("a result's hits are its own")
                .link(&self.hit_geometry);
        }
    }

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
/// words in `TileResult::streams_mut`'s order, little-endian.
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
    result.link_hits();
    Ok(result)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tile::NodeMesh;
    use orb_layout::Mesh;

    /// A packed tile read back, its streams in host bytes.
    pub(crate) fn unpacked(packed: &[u8]) -> TileResult {
        unpack(packed_header(packed).unwrap(), HostBytesId(3)).unwrap()
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

    /// The header is postcard; the streams follow it raw, where the
    /// result's host bytes say, a mesh's words that equal an earlier
    /// one's once.
    #[test]
    fn packs_and_unpacks() {
        let mesh = |node, vertices: Vec<u32>, props: Vec<u32>| NodeMesh {
            hits: crate::query::MeshHits::default(),
            node,
            extent: 4096,
            mesh: Mesh {
                vertices,
                ..Mesh::default()
            },
            props,
            row_words: 6,
            maxima: vec![2.5],
            images: Vec::new(),
            state_ids: Some(vec![1, crate::tile::NO_ID]),
            data: Vec::new(),
        };
        let bytes = pack(TileBuild {
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
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let result = unpack(packed_header(&bytes).unwrap(), HostBytesId(7)).unwrap();
        assert_eq!(result.host_bytes, Some(HostBytesId(7)));
        assert_eq!(
            (result.revision, &result.warnings[..]),
            (1 << 33 | 5, &["é".to_string()][..])
        );
        assert_eq!(
            result.meshes[1].state_ids,
            Some(vec![1, crate::tile::NO_ID])
        );
        let stream = |s: &Stream| match *s {
            Stream::Host { offset, len } => &words[offset as usize / 4..][..len as usize],
            Stream::Core(_) => panic!("a host stream"),
        };
        assert_eq!(stream(&result.meshes[0].vertices), [1, 2, 3]);
        assert_eq!(stream(&result.meshes[0].props), [9; 12]);
        let shared = &result.meshes[2];
        assert_eq!(shared.shares, [Some(0), None]);
        assert!(stream(&shared.vertices).is_empty());
        assert_eq!(stream(result.vertices(shared)), [1, 2, 3]);
        assert_eq!(stream(&shared.props), [8; 12]);
        assert!(packed_header(&bytes[..3]).is_err());
        assert!(unpack(&bytes[4..8], HostBytesId(0)).is_err());
    }
}

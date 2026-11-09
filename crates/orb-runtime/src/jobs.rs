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
    /// [`Jobs::add_archive_head`].
    ArchiveHead {
        url: String,
        head: Vec<u8>,
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
    /// Starts fetching glyph ranges style `id`'s tiles are about to need,
    /// so that their layout finds them in.
    fn prefetch_glyphs(&self, _id: u64, _urls: Vec<String>) {}
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

    fn add_archive_head(&self, url: &str, head: &Arc<Vec<u8>>) {
        self.host.add_archive_head(url, head);
    }

    fn features(&self, request: crate::FeatureRequest, done: crate::FeaturesDone) {
        self.host.features(request, done);
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
    /// Per source layer laid out, the geometry its meshes' hits point into.
    pub hit_geometry: Vec<Arc<crate::HitGeometry>>,
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
    pub hits: Arc<crate::MeshHits>,
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
            hit_geometry,
        };
        result.link_hits();
        result
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

    /// The header is postcard; the streams follow it raw, where the
    /// result's host bytes say.
    #[test]
    fn packs_and_unpacks() {
        let mesh = |node, vertices: Vec<u32>, props: Vec<u32>| NodeMesh {
            hits: crate::MeshHits::default(),
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
            nodes: vec![3, 4],
            warnings: vec!["é".into()],
            meshes: vec![mesh(3, vec![1, 2, 3], vec![9; 12]), mesh(4, vec![], vec![])],
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
        assert!(packed_header(&bytes[..3]).is_err());
        assert!(unpack(&bytes[4..8], HostBytesId(0)).is_err());
    }
}

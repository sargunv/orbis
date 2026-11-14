//! Sources.

use crate::property::{DemEncoding, Encoding, Scheme};

/// A style source.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub id: String,
    pub kind: SourceKind,
    /// The tile matrix set its tiles are in (its `dev.sargunv.orbis:tileMatrixSet`); `None`
    /// for `WebMercatorQuad`, and for GeoJSON, which tiles on the map's
    /// plane.
    pub tile_matrix_set: Option<orb_crs::TileMatrixSet>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum SourceKind {
    Vector(VectorSource),
    GeoJson(GeoJsonSource),
    Raster(RasterSource),
    RasterDem(RasterDemSource),
    /// An `image` or a `video` source.
    Image(ImageSource),
}
/// An `image`, `video` or `canvas` source: one picture on four geographic
/// corners.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageSource {
    pub content: ImageContent,
    /// Longitude and latitude of the picture's top left, top right, bottom
    /// right and bottom left corners.
    pub coordinates: [[f64; 2]; 4],
}
/// What an [`ImageSource`] shows.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageContent {
    /// An `image` source's image URL.
    Image(String),
    /// A `video` source's URLs, in the order a host tries them (formats
    /// differ in what each platform plays).
    Video(Vec<String>),
    /// A `canvas` source's canvas: the host hands in its pixels.
    Canvas(String),
}
/// A `geojson` source, ready for the tile worker to index.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoJsonSource {
    pub data: GeoJsonData,
    pub options: orb_tile::geojson::Options,
    /// `cluster: true`: its points cluster.
    pub cluster: Option<Cluster>,
}

/// How a `geojson` source's points cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    pub options: orb_tile::geojson::ClusterOptions,
    /// `clusterProperties`: each property's name, the expression a point
    /// contributes (`map`), and the one folding two contributions, which
    /// reads `["accumulated"]` and the other's value as `["get", name]`
    /// (`reduce`).
    pub properties: Vec<ClusterProperty>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterProperty {
    pub name: String,
    pub map: orb_expr::json::Json,
    pub reduce: orb_expr::json::Json,
}
#[derive(Clone, Debug, PartialEq)]
pub enum GeoJsonData {
    Inline(Vec<orb_tile::geojson::Feature>),
    Url(String),
}
/// The keys every tiled source shares.
///
/// Keys the style leaves out come from the TileJSON at `url` if there is one
/// (the style's own keys take precedence, as in GL JS), and otherwise from
/// the spec defaults: [`DEFAULT_MINZOOM`](Self::DEFAULT_MINZOOM) and so on.
#[derive(Clone, Debug, PartialEq)]
pub struct TileSet {
    /// A TileJSON URL, or a `pmtiles://` archive.
    pub url: Option<String>,
    /// Tile URL templates.
    pub tiles: Option<Vec<String>>,
    pub minzoom: Option<f64>,
    pub maxzoom: Option<f64>,
    pub scheme: Option<Scheme>,
    /// `[west, south, east, north]` in degrees.
    pub bounds: Option<[f64; 4]>,
    pub attribution: Option<String>,
}
/// The tile URL template of a vector source the app feeds: the map asks
/// the app for each tile instead of fetching it.
pub const HOST_TILES: &str = "orbis-host:";

impl TileSet {
    pub const DEFAULT_MINZOOM: f64 = 0.0;
    pub const DEFAULT_MAXZOOM: f64 = 22.0;
    pub const DEFAULT_SCHEME: Scheme = Scheme::Xyz;

    /// Whether the app feeds the tiles ([`HOST_TILES`]). Its tiles may hold
    /// GeoJSON, which a layer naming no source layer reads, as it reads a
    /// GeoJSON source's.
    pub fn host(&self) -> bool {
        self.url.is_none()
            && self
                .tiles
                .as_ref()
                .is_some_and(|t| t.iter().any(|t| t.starts_with(HOST_TILES)))
    }
}
/// A `vector` source.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorSource {
    pub tileset: TileSet,
    pub encoding: Option<Encoding>,
    /// Feature ids from a property instead of the tile's ids: one key for
    /// every source layer, or one per named layer.
    pub promote_id: Option<PromoteId>,
    /// `sparse`: an empty response (HTTP 204, or a tile an archive leaves
    /// out) means the tileset has no tile there at that zoom, rather than
    /// nothing there, so the tile's nearest ancestor draws in its place.
    pub sparse: bool,
}
/// A vector source's `promoteId`.
#[derive(Clone, Debug, PartialEq)]
pub enum PromoteId {
    All(String),
    PerLayer(Vec<(String, String)>),
}
impl PromoteId {
    /// The property whose value is the id of features of `layer`.
    pub fn key(&self, layer: &str) -> Option<&str> {
        match self {
            Self::All(key) => Some(key),
            Self::PerLayer(keys) => keys.iter().find(|(l, _)| l == layer).map(|(_, k)| &**k),
        }
    }
}
impl VectorSource {
    pub const DEFAULT_ENCODING: Encoding = Encoding::Mvt;
}
/// A `raster` source of image tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterSource {
    pub tileset: TileSet,
    /// Logical pixels a tile covers; `None` defers to the TileJSON.
    pub tile_size: Option<f64>,
}
impl RasterSource {
    pub const DEFAULT_TILE_SIZE: f64 = 512.0;
}
/// A `raster-dem` source: image tiles whose colours encode elevations.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterDemSource {
    pub raster: RasterSource,
    /// `None` defers to the TileJSON, then to `mapbox`.
    pub encoding: Option<DemEncoding>,
    /// `redFactor`, `greenFactor`, `blueFactor` and `baseShift`, for the
    /// `custom` encoding.
    pub factors: [f64; 4],
}
impl RasterDemSource {
    pub const DEFAULT_ENCODING: DemEncoding = DemEncoding::Mapbox;
}
impl DemEncoding {
    /// GL JS's unpack vector: an elevation in metres is
    /// `r·x + g·y + b·z − w` over the 0–255 channels.
    pub fn unpack(self, custom: [f64; 4]) -> [f64; 4] {
        match self {
            Self::Mapbox => [6553.6, 25.6, 0.1, 10000.0],
            Self::Terrarium => [256.0, 1.0, 1.0 / 256.0, 32768.0],
            Self::Custom => custom,
        }
    }
}

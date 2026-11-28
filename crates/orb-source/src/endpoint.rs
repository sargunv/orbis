//! Where a source's tiles come from, and the facts its TileJSON gives.

use orb_crs::{TileIndex, TileMatrixSet};
use orb_expr::json::{self, Json, JsonValue, Member};
use orb_style::{
    DemEncoding, Encoding, GeoJsonData, GeoJsonSource, RasterSource, Scheme, SourceKind, TileSet,
    VectorSource,
};

use crate::archive::Archive;

/// Vector tiles are laid out for this many logical pixels per side, as GL
/// JS's default `tileSize`.
const VECTOR_TILE_PX: f64 = 512.0;

pub use orb_style::HOST_TILES;

/// Where a tile's bytes are.
#[derive(Clone, Debug, PartialEq, Eq, orb_wire::Wire)]
pub enum TileLocation {
    Url(String),
    Archive(Archive),
    /// The app's answer for a tile of a source it feeds, or
    /// [`HostTile::Empty`] for a source with nothing to fetch.
    Host(HostTile),
}

/// What the app hands in for a tile of a source it feeds.
#[derive(Clone, Debug, PartialEq, Eq, orb_wire::Wire)]
pub enum HostTile {
    /// The tile encoded as `encoding` says (MVT or MLT), or as the
    /// source's tiles are.
    Encoded {
        bytes: Vec<u8>,
        encoding: Option<Encoding>,
    },
    /// Its source layers as GeoJSON: a JSON object of layer names and
    /// GeoJSON documents in longitude and latitude, which the map tiles.
    GeoJson(String),
    /// The tile has nothing in it (also every tile of a source with
    /// nothing to fetch).
    Empty,
}

/// What a TileJSON document, or an archive's header or metadata, says of
/// a tiled source; a source's own keys take precedence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileJson {
    pub tiles: Option<Vec<String>>,
    pub minzoom: Option<f64>,
    pub maxzoom: Option<f64>,
    pub bounds: Option<[f64; 4]>,
    pub scheme: Option<Scheme>,
    pub tile_size: Option<f64>,
    /// A vector source's tile encoding.
    pub encoding: Option<Encoding>,
    /// A `raster-dem` source's encoding.
    pub dem_encoding: Option<DemEncoding>,
}

impl TileJson {
    /// Reads a TileJSON document.
    pub fn parse(text: &str) -> Result<Self, String> {
        let doc = json::parse(text).map_err(|e| format!("bad TileJSON: {e}"))?;
        let number = |key: &str| doc.get(key).and_then(Json::as_f64);
        let text = |key: &str| doc.get(key).and_then(Json::as_str);
        let encoding = text("encoding");
        Ok(Self {
            tiles: doc.get("tiles").and_then(Json::as_array).map(|tiles| {
                (tiles.iter())
                    .filter_map(|u| u.as_str().map(str::to_string))
                    .collect()
            }),
            minzoom: number("minzoom"),
            maxzoom: number("maxzoom"),
            bounds: doc.get("bounds").and_then(Json::as_array).and_then(|b| {
                let b: Option<Vec<f64>> = b.iter().take(4).map(Json::as_f64).collect();
                b?.try_into().ok()
            }),
            scheme: text("scheme").and_then(<Scheme as orb_style::Enum>::from_name),
            tile_size: number("tileSize"),
            encoding: encoding.and_then(<Encoding as orb_style::Enum>::from_name),
            dem_encoding: encoding.and_then(<DemEncoding as orb_style::Enum>::from_name),
        })
    }
}

/// Where a source's tiles come from.
#[derive(Clone, Debug, PartialEq)]
pub enum Tiles {
    /// URL templates, spread over its tiles.
    Templates(Vec<String>),
    Archive(Archive),
    /// The app ([`HOST_TILES`]).
    Host,
    /// A whole GeoJSON document at a URL, or inline (`None`).
    Document(Option<String>),
    /// An `image` source's image, or nothing to fetch: no URL, or a
    /// video's or a canvas's frames, which come from its host.
    Image(Option<String>),
}

/// Where a source's tile jobs get their data, and what its tiles are.
#[derive(Clone, Debug, PartialEq)]
pub struct TileEndpoint {
    pub tiles: Tiles,
    pub bounds: Option<[f64; 4]>,
    pub scheme: Scheme,
    pub minzoom: u8,
    pub maxzoom: u8,
    pub encoding: Encoding,
    /// The tile size the view is covered with, in logical pixels.
    pub cover_px: f64,
    /// A `raster-dem` source's unpack vector.
    pub dem_unpack: Option<[f64; 4]>,
    /// The tile matrix set its tiles are in.
    pub tms: TileMatrixSet,
    /// An `image` source's quad on the surface's plane: its one tile draws
    /// the image warped onto it.
    pub quad: Option<orb_crs::ImageQuad>,
}

impl TileEndpoint {
    /// The endpoint of a tiled source, with the style's keys taking
    /// precedence over its TileJSON's, as in GL JS.
    pub fn resolve(
        source: &SourceKind,
        tilejson: Option<&TileJson>,
        tms: TileMatrixSet,
    ) -> Result<Self, String> {
        let doc = tilejson.cloned().unwrap_or_default();
        let (tileset, raster) = match source {
            SourceKind::Vector(v) => (&v.tileset, None),
            SourceKind::Raster(r) => (&r.tileset, Some(r)),
            SourceKind::RasterDem(d) => (&d.raster.tileset, Some(&d.raster)),
            _ => unreachable!("tiled sources only"),
        };
        let archive = (tileset.url.as_deref())
            .filter(|_| tileset.tiles.is_none())
            .and_then(Archive::of);
        let tiles = match (archive, tileset.tiles.clone().or(doc.tiles)) {
            (Some(archive), _) => Tiles::Archive(archive),
            (None, Some(t)) if t.iter().any(|t| t.starts_with(HOST_TILES)) => Tiles::Host,
            (None, Some(t)) if !t.is_empty() => Tiles::Templates(t),
            (None, _) => return Err("source has no tile URLs".to_string()),
        };
        let zoom = |style: Option<f64>, tilejson: Option<f64>, default: f64| {
            (style.or(tilejson).unwrap_or(default))
                .clamp(f64::from(tms.min_zoom()), f64::from(tms.max_zoom())) as u8
        };
        // Raster sources round the covering zoom, as GL JS's `roundZoom`
        // does: a tile is refined once it would span √2 times its size.
        let cover_px = match raster {
            None => VECTOR_TILE_PX,
            Some(r) => {
                (r.tile_size.or(doc.tile_size)).unwrap_or(RasterSource::DEFAULT_TILE_SIZE)
                    / std::f64::consts::SQRT_2
            }
        };
        let dem_unpack = match source {
            SourceKind::RasterDem(dem) => Some(
                (dem.encoding.or(doc.dem_encoding))
                    .unwrap_or(orb_style::RasterDemSource::DEFAULT_ENCODING)
                    .unpack(dem.factors),
            ),
            _ => None,
        };
        let encoding = match source {
            SourceKind::Vector(v) => v.encoding.or(doc.encoding),
            _ => None,
        };
        Ok(Self {
            tiles,
            bounds: tileset.bounds.or(doc.bounds),
            scheme: (tileset.scheme.or(doc.scheme)).unwrap_or(TileSet::DEFAULT_SCHEME),
            minzoom: zoom(tileset.minzoom, doc.minzoom, TileSet::DEFAULT_MINZOOM),
            maxzoom: zoom(tileset.maxzoom, doc.maxzoom, TileSet::DEFAULT_MAXZOOM),
            encoding: encoding.unwrap_or(VectorSource::DEFAULT_ENCODING),
            cover_px,
            dem_unpack,
            tms,
            quad: None,
        })
    }

    /// A GeoJSON source's: the worker tiles its one document.
    pub fn geojson(source: &GeoJsonSource, tms: TileMatrixSet) -> Self {
        let url = match &source.data {
            GeoJsonData::Url(url) => Some(url.clone()),
            GeoJsonData::Inline(_) => None,
        };
        Self {
            tiles: Tiles::Document(url),
            bounds: None,
            scheme: Scheme::Xyz,
            minzoom: 0,
            maxzoom: source.options.max_zoom,
            encoding: Encoding::Mvt,
            cover_px: VECTOR_TILE_PX,
            dem_unpack: None,
            tms,
            quad: None,
        }
    }

    /// An `image` source on `projection`'s plane: one tile, its image, in a
    /// one-tile set around its quad.
    pub fn image(source: &orb_style::ImageSource, projection: &orb_crs::Projection) -> Self {
        let corners = source
            .coordinates
            .map(|[lon, lat]| projection.forward(orb_crs::Geodetic::new(lon, lat)));
        let quad = orb_crs::ImageQuad::new(corners);
        let bounds = quad.bounds();
        let half = bounds.size().max_element().max(1e-9) / 2.0;
        let square = orb_crs::WorldRect {
            min: bounds.center() - half,
            max: bounds.center() + half,
        };
        let url = match &source.content {
            orb_style::ImageContent::Image(url) => url.clone(),
            orb_style::ImageContent::Video(_) | orb_style::ImageContent::Canvas(_) => None,
        };
        Self {
            tiles: Tiles::Image(url),
            bounds: None,
            scheme: Scheme::Xyz,
            minzoom: 0,
            maxzoom: 0,
            encoding: Encoding::Mvt,
            cover_px: RasterSource::DEFAULT_TILE_SIZE / std::f64::consts::SQRT_2,
            dem_unpack: None,
            tms: TileMatrixSet::square(*projection, square),
            quad: Some(quad),
        }
    }

    pub fn data_tile(&self, mut tile: TileIndex) -> TileIndex {
        while tile.z > self.maxzoom {
            tile = tile.parent().expect("ancestor");
        }
        tile
    }

    /// Whether the app feeds the source's tiles ([`HOST_TILES`]).
    pub fn host(&self) -> bool {
        matches!(self.tiles, Tiles::Host)
    }

    /// Where `tile`'s bytes are: for the app's tiles, the answer it has
    /// yet to give, nothing.
    pub fn location(&self, tile: TileIndex) -> TileLocation {
        match &self.tiles {
            Tiles::Templates(_) => TileLocation::Url(self.url(tile).expect("templated")),
            Tiles::Archive(archive) => TileLocation::Archive(archive.clone()),
            Tiles::Document(Some(url)) | Tiles::Image(Some(url)) => TileLocation::Url(url.clone()),
            Tiles::Host | Tiles::Document(None) | Tiles::Image(None) => {
                TileLocation::Host(HostTile::Empty)
            }
        }
    }

    pub fn contains(&self, tile: TileIndex) -> bool {
        if !self.tms.contains(tile) {
            return false;
        }
        let Some([west, south, east, north]) = self.bounds else {
            return true;
        };
        let [w, s, e, n] = self.tms.geodetic_bounds(tile);
        // Bounds crossing the antimeridian have east past 180.
        let overlaps_lon = |shift: f64| w + shift < east && e + shift > west;
        (overlaps_lon(0.0) || overlaps_lon(360.0) || overlaps_lon(-360.0)) && s < north && n > south
    }

    /// The tiles a map draws of the area `bounds` (west, south, east,
    /// north, in degrees) between map zooms `zooms`, as a view's cover
    /// picks them: an offline region's tiles of this source.
    pub fn region_tiles(&self, bounds: [f64; 4], zooms: [f64; 2]) -> Vec<TileIndex> {
        let offset = (VECTOR_TILE_PX / self.cover_px).log2();
        let zoom = |z: f64| ((z + offset).floor().max(0.0) as u8).clamp(self.minzoom, self.maxzoom);
        let [west, south, east, north] = bounds;
        let corner = |lon: f64, lat: f64| {
            let lat = lat.clamp(
                -orb_crs::MERCATOR_MAX_LATITUDE,
                orb_crs::MERCATOR_MAX_LATITUDE,
            );
            self.tms
                .projection()
                .forward(orb_crs::Geodetic::new(lon, lat))
        };
        let (a, b) = (corner(west, south), corner(east, north));
        let rect = orb_crs::WorldRect {
            min: a.min(b),
            max: a.max(b),
        };
        (zoom(zooms[0])..=zoom(zooms[1]))
            .flat_map(|z| self.tms.tiles_in(rect, z))
            .filter(|&tile| self.contains(tile))
            .collect()
    }

    /// The URL of a templated source's tile, spreading tiles over its
    /// templates, at a pixel ratio of 1: the map never asks for `@2x`.
    pub fn url(&self, tile: TileIndex) -> Option<String> {
        let Tiles::Templates(templates) = &self.tiles else {
            return None;
        };
        let template = &templates[(tile.col as usize + tile.row as usize) % templates.len()];
        Some(expand(template, &self.tms, self.scheme, tile, 1.0))
    }
}

/// `template` filled in for `tile` of `tms` in `scheme`, as GL JS and
/// MapLibre Native fill tile URLs: `{z}`, `{x}`, `{y}`, OGC's
/// `{tileMatrix}`, `{tileCol}` and `{tileRow}`, `{bbox}` in the set's
/// units (and `{bbox-epsg-3857}` on Web Mercator), `{quadkey}`,
/// `{prefix}`, and `{ratio}`, `@2x` above a pixel ratio of 1.
pub fn expand(
    template: &str,
    tms: &TileMatrixSet,
    scheme: Scheme,
    tile: TileIndex,
    ratio: f32,
) -> String {
    let (col, native) = (tms.native_col(tile), tms.native_row(tile));
    let row = match scheme {
        Scheme::Xyz => native,
        Scheme::Tms => tms.matrix_size(tile.z).1 - 1 - native,
    };
    let mut url = template
        .replace("{z}", &tile.z.to_string())
        .replace("{x}", &col.to_string())
        .replace("{y}", &row.to_string())
        .replace("{tileMatrix}", &tile.z.to_string())
        .replace("{tileCol}", &col.to_string())
        .replace("{tileRow}", &native.to_string())
        .replace("{ratio}", if ratio > 1.0 { "@2x" } else { "" })
        .replace("{prefix}", &format!("{:x}{:x}", col % 16, native % 16));
    if url.contains("{quadkey}") {
        let quadkey: String = (1..=tile.z)
            .rev()
            .map(|i| {
                let mask = 1 << (i - 1);
                char::from(b'0' + u8::from(col & mask != 0) + 2 * u8::from(native & mask != 0))
            })
            .collect();
        url = url.replace("{quadkey}", &quadkey);
    }
    if url.contains("{bbox") {
        let b = tms.tile_bounds(tile);
        let bbox = format!("{},{},{},{}", b.min.x, b.min.y, b.max.x, b.max.y);
        if tms.projection().is_web_mercator() {
            url = url.replace("{bbox-epsg-3857}", &bbox);
        }
        url = url.replace("{bbox}", &bbox);
    }
    url
}

/// `relative` resolved against `base`, as a browser resolves a link:
/// absolute URLs stay (and [`HOST_TILES`], which names the app), `/path`
/// keeps the origin, others replace `base`'s last segment with `./` and
/// `../` applied.
pub(crate) fn join_url(base: &str, relative: &str) -> String {
    if relative.contains("://") || relative.starts_with("data:") || relative.starts_with(HOST_TILES)
    {
        return relative.to_string();
    }
    let base = base.split(['?', '#']).next().unwrap_or(base);
    let origin_end = base
        .find("://")
        .map(|i| base[i + 3..].find('/').map_or(base.len(), |j| i + 3 + j))
        .unwrap_or(0);
    let (origin, path) = base.split_at(origin_end);
    if let Some(absolute) = relative.strip_prefix('/') {
        return format!("{origin}/{absolute}");
    }
    let mut segments: Vec<&str> = path.split('/').collect();
    segments.pop();
    for part in relative.split('/') {
        match part {
            "." => {}
            ".." => {
                if segments.len() > 1 {
                    segments.pop();
                }
            }
            part => segments.push(part),
        }
    }
    format!("{origin}{}", segments.join("/"))
}

/// Resolves a style's relative URLs against `base`, the URL it came from,
/// as a browser resolves a style's links: sources' `url`, `urls`, `tiles`
/// and GeoJSON `data`, `glyphs`, `sprite` and `font-faces` (a
/// `pmtiles://` or `mbtiles://` URL keeps its prefix, and an `mbtiles://`
/// path names a file).
pub fn resolve_style_urls(doc: &mut Json, base: &str) {
    // A URL, or each URL of an array of them.
    fn resolve(value: Option<&mut Json>, base: &str) {
        match value.map(|v| &mut v.value) {
            Some(JsonValue::Array(items)) => {
                items.iter_mut().for_each(|v| join(&mut v.value, base))
            }
            Some(value) => join(value, base),
            None => {}
        }
    }
    fn join(value: &mut JsonValue, base: &str) {
        let JsonValue::String(url) = value else {
            return;
        };
        *url = match Archive::of(url) {
            Some(Archive::Pmtiles(rest)) => format!("pmtiles://{}", join_url(base, &rest)),
            // A path names a file, as MapLibre Native reads `mbtiles:///`.
            Some(Archive::Mbtiles(rest)) if rest.starts_with("file://") => return,
            Some(Archive::Mbtiles(rest)) => format!("mbtiles://{}", join_url(base, &rest)),
            None => join_url(base, url),
        };
    }
    fn members(value: Option<&mut Json>) -> &mut [Member] {
        match value.map(|v| &mut v.value) {
            Some(JsonValue::Object(members)) => members,
            _ => &mut [],
        }
    }
    resolve(doc.get_mut("glyphs"), base);
    match doc.get_mut("sprite") {
        Some(Json {
            value: JsonValue::Array(sprites),
            ..
        }) => sprites
            .iter_mut()
            .for_each(|s| resolve(s.get_mut("url"), base)),
        sprite => resolve(sprite, base),
    }
    for source in members(doc.get_mut("sources")) {
        for key in ["url", "urls", "tiles", "data"] {
            resolve(source.value.get_mut(key), base);
        }
    }
    // A font face is its URL or an object with one; a name declares one or
    // an array of them.
    fn face(face: &mut Json, base: &str) {
        match face.value {
            JsonValue::String(_) => join(&mut face.value, base),
            _ => resolve(face.get_mut("url"), base),
        }
    }
    for faces in members(doc.get_mut("font-faces")) {
        match &mut faces.value.value {
            JsonValue::Array(faces) => faces.iter_mut().for_each(|f| face(f, base)),
            _ => face(&mut faces.value, base),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mercator() -> TileMatrixSet {
        TileMatrixSet::web_mercator().clone()
    }

    #[test]
    fn resolves_a_styles_relative_urls() {
        let style = r#"{"version": 8, "glyphs": "./glyphs/{fontstack}/{range}.pbf",
            "sprite": [{"id": "a", "url": "sprites/a"}],
            "sources": {"t": {"type": "vector", "url": "pmtiles://./t.pmtiles"},
                "g": {"type": "geojson", "data": "../g.json"},
                "r": {"type": "raster", "tiles": ["https://x.org/{z}.png", "r/{z}.png"]},
                "h": {"type": "raster-dem", "tiles": ["orbis-host:"]},
                "i": {"type": "geojson", "data": {"type": "FeatureCollection", "features": []}}},
            "font-faces": {"A": "fonts/a.ttf", "B": ["fonts/b.ttf", {"url": "fonts/c.ttf"}],
                "C": {"url": "fonts/d.ttf"}},
            "layers": []}"#;
        let mut doc = json::parse(style).unwrap();
        resolve_style_urls(&mut doc, "file:///d/styles/s.json");
        let at = |path: &[&str]| {
            let mut v = &doc;
            for key in path {
                v = v.get(key).unwrap();
            }
            v.as_str().unwrap().to_string()
        };
        assert_eq!(
            at(&["glyphs"]),
            "file:///d/styles/glyphs/{fontstack}/{range}.pbf"
        );
        let host = doc
            .get("sources")
            .and_then(|s| s.get("h"))
            .and_then(|h| h.get("tiles"));
        assert_eq!(
            host.unwrap().as_array().unwrap()[0].as_str(),
            Some(HOST_TILES)
        );
        assert_eq!(
            doc.get("sprite").unwrap().as_array().unwrap()[0]
                .get("url")
                .unwrap()
                .as_str(),
            Some("file:///d/styles/sprites/a")
        );
        assert_eq!(
            at(&["sources", "t", "url"]),
            "pmtiles://file:///d/styles/t.pmtiles"
        );
        assert_eq!(at(&["sources", "g", "data"]), "file:///d/g.json");
        let tiles = doc
            .get("sources")
            .unwrap()
            .get("r")
            .unwrap()
            .get("tiles")
            .unwrap();
        let tiles: Vec<_> = tiles
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        assert_eq!(
            tiles,
            ["https://x.org/{z}.png", "file:///d/styles/r/{z}.png"]
        );
        let faces = doc.get("font-faces").unwrap();
        assert_eq!(at(&["font-faces", "A"]), "file:///d/styles/fonts/a.ttf");
        let b = faces.get("B").unwrap().as_array().unwrap();
        assert_eq!(b[0].as_str(), Some("file:///d/styles/fonts/b.ttf"));
        assert_eq!(
            b[1].get("url").unwrap().as_str(),
            Some("file:///d/styles/fonts/c.ttf")
        );
        assert_eq!(
            at(&["font-faces", "C", "url"]),
            "file:///d/styles/fonts/d.ttf"
        );
    }

    #[test]
    fn joins_relative_urls() {
        let base = "https://x.org/a/b/scene.json?k=1";
        assert_eq!(join_url(base, "style.json"), "https://x.org/a/b/style.json");
        assert_eq!(
            join_url(base, "./s/style.json"),
            "https://x.org/a/b/s/style.json"
        );
        assert_eq!(
            join_url(base, "../style.json"),
            "https://x.org/a/style.json"
        );
        assert_eq!(join_url(base, "/style.json"), "https://x.org/style.json");
        assert_eq!(join_url(base, "file:///s.json"), "file:///s.json");
        assert_eq!(
            join_url("file:///d/scene.json", "s.json"),
            "file:///d/s.json"
        );
    }

    #[test]
    fn matrix_set_tokens_and_bounds() {
        let nztm = TileMatrixSet::by_id("NZTM2000Quad").unwrap();
        let source = vector(
            r#"{"type": "raster", "tiles": ["t/{tileMatrix}/{tileCol}/{tileRow}?b={bbox}"], "bounds": [166, -48, 179, -34]}"#,
        );
        let endpoint = TileEndpoint::resolve(&source, None, nztm.clone()).unwrap();
        let tile = TileIndex::new(1, 1, 0);
        let b = nztm.tile_bounds(tile);
        assert_eq!(
            endpoint.url(tile).unwrap(),
            format!("t/1/1/0?b={},{},{},{}", b.min.x, b.min.y, b.max.x, b.max.y)
        );
        // New Zealand is at the grid's centre: the middle four tiles at
        // zoom 2.
        assert!(endpoint.contains(TileIndex::new(2, 1, 1)));
        assert!(endpoint.contains(TileIndex::new(2, 2, 2)));
        assert!(!endpoint.contains(TileIndex::new(2, 0, 0)));
        assert!(!endpoint.contains(TileIndex::new(2, 3, 3)));
        assert!(!endpoint.contains(TileIndex::new(2, 4, 0)));
    }

    fn vector(style: &str) -> SourceKind {
        let style = orb_style::v8::parse(&format!(
            r#"{{"version": 8, "sources": {{"s": {style}}}, "layers": []}}"#
        ))
        .unwrap();
        style.sources[0].kind.clone()
    }

    #[test]
    fn raster_sources_round_their_cover_and_unpack_dems() {
        let source =
            vector(r#"{"type": "raster", "tiles": ["t/{z}/{x}/{y}.png"], "tileSize": 256}"#);
        let endpoint = TileEndpoint::resolve(&source, None, mercator()).unwrap();
        assert_eq!(endpoint.cover_px, 256.0 / std::f64::consts::SQRT_2);
        assert_eq!(endpoint.dem_unpack, None);
        let source = vector(r#"{"type": "raster-dem", "url": "https://x/dem.json"}"#);
        let tilejson =
            r#"{"tiles": ["t/{z}/{x}/{y}.png"], "tileSize": 256, "encoding": "terrarium"}"#;
        let tilejson = TileJson::parse(tilejson).unwrap();
        let endpoint = TileEndpoint::resolve(&source, Some(&tilejson), mercator()).unwrap();
        assert_eq!(endpoint.cover_px, 256.0 / std::f64::consts::SQRT_2);
        assert_eq!(
            endpoint.dem_unpack,
            Some([256.0, 1.0, 1.0 / 256.0, 32768.0])
        );
    }

    #[test]
    fn tilejson_bounds_are_their_first_four_numbers() {
        let bounds = |b: &str| {
            TileJson::parse(&format!(r#"{{"bounds": {b}}}"#))
                .unwrap()
                .bounds
        };
        assert_eq!(bounds("[1, 2, 3, 4, 5]"), Some([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(bounds(r#"[1, "a", 2, 3, 4]"#), None);
        assert_eq!(bounds("[1, 2, 3]"), None);
    }

    #[test]
    fn mbtiles_urls_name_files_or_urls() {
        assert_eq!(
            Archive::of("mbtiles:///data/a.mbtiles"),
            Some(Archive::Mbtiles("file:///data/a.mbtiles".into()))
        );
        assert_eq!(
            Archive::of("mbtiles://https://x/a.mbtiles"),
            Some(Archive::Mbtiles("https://x/a.mbtiles".into()))
        );
        assert_eq!(Archive::of("https://x/a.json"), None);
        let mut doc = json::parse(
            r#"{"sources": {"a": {"url": "mbtiles://./a.mbtiles"}, "b": {"url": "mbtiles:///b.mbtiles"}}}"#,
        )
        .unwrap();
        resolve_style_urls(&mut doc, "https://x/styles/s.json");
        let url = |id| {
            let source = doc.get("sources").unwrap().get(id).unwrap();
            source.get("url").unwrap().as_str().unwrap().to_string()
        };
        assert_eq!(url("a"), "mbtiles://https://x/styles/a.mbtiles");
        assert_eq!(url("b"), "mbtiles:///b.mbtiles");
        let source = vector(r#"{"type":"vector","url":"mbtiles:///b.mbtiles"}"#);
        let endpoint = TileEndpoint::resolve(&source, None, mercator()).unwrap();
        assert_eq!(
            endpoint.location(TileIndex::new(2, 1, 1)),
            TileLocation::Archive(Archive::Mbtiles("file:///b.mbtiles".into()))
        );
    }

    #[test]
    fn archive_header_metadata_and_style_precedence() {
        let mut header = [0; 127];
        header[..8].copy_from_slice(b"PMTiles\x03");
        for (at, value) in [(8, 127u64), (16, 1), (24, 128), (32, 1), (56, 129), (64, 1)] {
            header[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        header[97..102].copy_from_slice(&[1, 1, 6, 10, 14]);
        for (i, value) in [1397250000i32, 356770000, 1397550000, 356970000]
            .into_iter()
            .enumerate()
        {
            header[102 + i * 4..106 + i * 4].copy_from_slice(&value.to_le_bytes());
        }
        let metadata = crate::pmtiles::Header::parse(&header).unwrap().tilejson();
        let source =
            vector(r#"{"type":"vector","url":"pmtiles://https://x/tokyo.pmtiles","maxzoom":12}"#);
        let endpoint = TileEndpoint::resolve(&source, Some(&metadata), mercator()).unwrap();
        assert_eq!(
            (endpoint.minzoom, endpoint.maxzoom, endpoint.encoding),
            (10, 12, Encoding::Mlt)
        );
        assert_eq!(endpoint.bounds, Some([139.725, 35.677, 139.755, 35.697]));
        assert_eq!(
            endpoint.location(TileIndex::new(10, 909, 403)),
            TileLocation::Archive(Archive::Pmtiles("https://x/tokyo.pmtiles".into()))
        );
        assert!(endpoint.contains(TileIndex::new(10, 909, 403)));
        assert!(!endpoint.contains(TileIndex::new(10, 0, 0)));
        let source = vector(
            r#"{"type":"vector","url":"pmtiles://https://x/a","tiles":["https://x/{z}/{x}/{y}.pbf"],"bounds":[-180,-85,180,85],"encoding":"mvt"}"#,
        );
        let endpoint = TileEndpoint::resolve(&source, Some(&metadata), mercator()).unwrap();
        assert_eq!(endpoint.encoding, Encoding::Mvt);
        assert!(endpoint.contains(TileIndex::new(10, 0, 1)));
        assert!(matches!(endpoint.tiles, Tiles::Templates(_)));
    }

    #[test]
    fn region_tiles_stop_at_the_source_maxzoom() {
        let source = orb_style::v8::parse_source(
            "s",
            &json::parse(r#"{"type": "raster", "tiles": ["https://x/{z}/{x}/{y}.png"], "tileSize": 256, "maxzoom": 3}"#).unwrap(),
            orb_style::Previews::NONE,
        )
        .unwrap()
        .0;
        let endpoint = TileEndpoint::resolve(&source.kind, None, mercator()).unwrap();
        // 256-pixel rasters draw from a zoom deeper than the map's, to
        // the source's maxzoom.
        let tiles = endpoint.region_tiles([0.1, 0.1, 10.0, 10.0], [0.0, 5.0]);
        let zooms: Vec<u8> = tiles.iter().map(|t| t.z).collect();
        assert_eq!(zooms, [1, 2, 3]);
        assert_eq!(tiles[0], TileIndex::new(1, 1, 0));
    }

    #[test]
    fn style_keys_override_tilejson() {
        let source = vector(r#"{"type": "vector", "url": "https://x/tiles.json", "maxzoom": 12}"#);
        let tilejson = r#"{"tiles": ["https://x/{z}/{x}/{y}.pbf"], "minzoom": 2, "maxzoom": 14}"#;
        let tilejson = TileJson::parse(tilejson).unwrap();
        let endpoint = TileEndpoint::resolve(&source, Some(&tilejson), mercator()).unwrap();
        assert_eq!((endpoint.minzoom, endpoint.maxzoom), (2, 12));
        assert_eq!(
            endpoint.url(TileIndex::new(3, 1, 2)).unwrap(),
            "https://x/3/1/2.pbf"
        );
    }

    #[test]
    fn tms_rows_count_from_the_bottom() {
        let source = vector(r#"{"type": "vector", "tiles": ["t/{z}/{x}/{y}"], "scheme": "tms"}"#);
        let endpoint = TileEndpoint::resolve(&source, None, mercator()).unwrap();
        assert_eq!(endpoint.url(TileIndex::new(2, 1, 0)).unwrap(), "t/2/1/3");
    }

    /// Bing's `{quadkey}` and `{prefix}`, `{ratio}` and WMS's
    /// `{bbox-epsg-3857}`, as GL JS and Native fill them.
    #[test]
    fn templates_fill_every_token() {
        let tile = TileIndex::new(3, 5, 2);
        let url = |template, ratio| expand(template, &mercator(), Scheme::Xyz, tile, ratio);
        assert_eq!(
            url("https://{prefix}.x/{z}/{x}/{y}{ratio}.png?q={quadkey}", 2.0),
            "https://52.x/3/5/2@2x.png?q=121"
        );
        assert_eq!(url("{x}{ratio}", 1.0), "5");
        let b = mercator().tile_bounds(tile);
        assert_eq!(
            url("?bbox={bbox-epsg-3857}", 1.0),
            format!("?bbox={},{},{},{}", b.min.x, b.min.y, b.max.x, b.max.y)
        );
    }

    #[test]
    fn needs_tile_urls() {
        let source = vector(r#"{"type": "vector", "url": "https://x/tiles.json"}"#);
        let tilejson = TileJson::parse("{}").unwrap();
        assert!(TileEndpoint::resolve(&source, Some(&tilejson), mercator()).is_err());
    }
}

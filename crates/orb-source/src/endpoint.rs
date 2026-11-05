//! Tiled sources' TileJSON/tile URLs and GeoJSON source-data endpoints.

use orb_crs::{TileIndex, TileMatrixSet};
use orb_expr::json;
use orb_v8::{
    DemEncoding, Encoding, GeoJsonData, GeoJsonSource, RasterSource, Scheme, SourceKind, TileSet,
    VectorSource,
};

/// Vector tiles are laid out for this many logical pixels per side, as GL
/// JS's default `tileSize`.
const VECTOR_TILE_PX: f64 = 512.0;

/// Where a tile's bytes are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileLocation {
    Url(String),
    /// The archive URL without the `pmtiles://` protocol prefix.
    Pmtiles(String),
}

/// Where a source's tile jobs get their data (merged TileJSON for tiled
/// sources).
#[derive(Clone, Debug, PartialEq)]
pub struct TileEndpoint {
    archive: Option<String>,
    pub bounds: Option<[f64; 4]>,
    pub templates: Vec<String>,
    /// A whole GeoJSON document; an empty URL denotes inline source data.
    data_url: Option<String>,
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
        tilejson: Option<&str>,
        tms: TileMatrixSet,
    ) -> Result<Self, String> {
        let doc = tilejson
            .map(|text| json::parse(text).map_err(|e| format!("bad TileJSON: {e}")))
            .transpose()?;
        let number = |key: &str| {
            doc.as_ref()
                .and_then(|d| d.get(key))
                .and_then(|v| v.as_f64())
        };
        let text = |key: &str| {
            doc.as_ref()
                .and_then(|d| d.get(key))
                .and_then(|v| v.as_str())
        };
        let (tileset, raster) = match source {
            SourceKind::Vector(v) => (&v.tileset, None),
            SourceKind::Raster(r) => (&r.tileset, Some(r)),
            SourceKind::RasterDem(d) => (&d.raster.tileset, Some(&d.raster)),
            _ => unreachable!("tiled sources only"),
        };
        let templates = match &tileset.tiles {
            Some(tiles) => tiles.clone(),
            None => doc
                .as_ref()
                .and_then(|d| d.get("tiles"))
                .and_then(|t| t.as_array())
                .map(|t| {
                    t.iter()
                        .filter_map(|u| u.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
        };
        let archive = tileset.url.as_ref().and_then(|url| {
            if tileset.tiles.is_none() {
                url.strip_prefix("pmtiles://").map(str::to_string)
            } else {
                None
            }
        });
        if templates.is_empty() && archive.is_none() {
            return Err("source has no tile URLs".to_string());
        }
        let scheme = tileset.scheme.unwrap_or(match text("scheme") {
            Some("tms") => Scheme::Tms,
            _ => TileSet::DEFAULT_SCHEME,
        });
        let zoom = |style: Option<f64>, key: &str, default: f64| {
            style
                .or_else(|| number(key))
                .unwrap_or(default)
                .clamp(f64::from(tms.min_zoom()), f64::from(tms.max_zoom())) as u8
        };
        let bounds = tileset.bounds.or_else(|| {
            let values = doc.as_ref()?.get("bounds")?.as_array()?;
            Some([
                values.first()?.as_f64()?,
                values.get(1)?.as_f64()?,
                values.get(2)?.as_f64()?,
                values.get(3)?.as_f64()?,
            ])
        });
        // Raster sources round the covering zoom, as GL JS's `roundZoom`
        // does: a tile is refined once it would span √2 times its size.
        let cover_px = match raster {
            None => VECTOR_TILE_PX,
            Some(r) => {
                r.tile_size
                    .or_else(|| number("tileSize"))
                    .unwrap_or(RasterSource::DEFAULT_TILE_SIZE)
                    / std::f64::consts::SQRT_2
            }
        };
        let dem_unpack = match source {
            SourceKind::RasterDem(dem) => {
                let encoding = dem.encoding.unwrap_or(
                    match text("encoding").and_then(<DemEncoding as orb_v8::Enum>::from_name) {
                        Some(encoding) => encoding,
                        None => orb_v8::RasterDemSource::DEFAULT_ENCODING,
                    },
                );
                Some(encoding.unpack(dem.factors))
            }
            _ => None,
        };
        let encoding = match source {
            SourceKind::Vector(v) => v.encoding.unwrap_or(match text("encoding") {
                Some("mlt") => Encoding::Mlt,
                _ => VectorSource::DEFAULT_ENCODING,
            }),
            _ => VectorSource::DEFAULT_ENCODING,
        };
        Ok(Self {
            archive,
            bounds,
            templates,
            data_url: None,
            scheme,
            minzoom: zoom(tileset.minzoom, "minzoom", TileSet::DEFAULT_MINZOOM),
            maxzoom: zoom(tileset.maxzoom, "maxzoom", TileSet::DEFAULT_MAXZOOM),
            encoding,
            cover_px,
            dem_unpack,
            tms,
            quad: None,
        })
    }

    /// GeoJSON uses one source URL (or no URL for inline data); the tile
    /// worker supplies geometry rather than decoding a vector tile response.
    pub fn geojson(source: &GeoJsonSource, tms: TileMatrixSet) -> Self {
        Self {
            archive: None,
            bounds: None,
            templates: Vec::new(),
            data_url: Some(match &source.data {
                GeoJsonData::Url(url) => url.clone(),
                GeoJsonData::Inline(_) => String::new(),
            }),
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

    /// An `image` source on `projection`'s plane: one tile, the image at
    /// its URL, in a one-tile set around its quad.
    pub fn image(source: &orb_v8::ImageSource, projection: &orb_crs::Projection) -> Self {
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
        Self {
            archive: None,
            bounds: None,
            templates: vec![source.url.clone()],
            data_url: None,
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

    pub fn cover_zoom(&self) -> u8 {
        if self.data_url.is_some() {
            self.maxzoom
        } else {
            24
        }
    }

    pub fn data_tile(&self, mut tile: TileIndex) -> TileIndex {
        while tile.z > self.maxzoom {
            tile = tile.parent().expect("ancestor");
        }
        tile
    }

    pub fn location(&self, tile: TileIndex) -> TileLocation {
        match &self.archive {
            Some(url) => TileLocation::Pmtiles(url.clone()),
            None => TileLocation::Url(self.url(tile)),
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
            let lat = lat.clamp(-85.051_128_78, 85.051_128_78);
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

    /// The URL of a tile, spreading tiles over the templates.
    pub fn url(&self, tile: TileIndex) -> String {
        if let Some(url) = &self.data_url {
            return url.clone();
        }
        let native = self.tms.native_row(tile);
        let row = match self.scheme {
            Scheme::Xyz => native,
            Scheme::Tms => self.tms.matrix_size(tile.z).1 - 1 - native,
        };
        let template =
            &self.templates[(tile.col as usize + tile.row as usize) % self.templates.len()];
        let mut url = template
            .replace("{z}", &tile.z.to_string())
            .replace("{x}", &tile.col.to_string())
            .replace("{y}", &row.to_string())
            .replace("{tileMatrix}", &tile.z.to_string())
            .replace("{tileCol}", &tile.col.to_string())
            .replace("{tileRow}", &native.to_string());
        if url.contains("{bbox}") {
            let b = self.tms.tile_bounds(tile);
            url = url.replace(
                "{bbox}",
                &format!("{},{},{},{}", b.min.x, b.min.y, b.max.x, b.max.y),
            );
        }
        url
    }
}

/// `relative` resolved against `base`, as a browser resolves a link:
/// absolute URLs stay, `/path` keeps the origin, others replace `base`'s
/// last segment with `./` and `../` applied.
pub fn join_url(base: &str, relative: &str) -> String {
    if relative.contains("://") || relative.starts_with("data:") {
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

/// Header metadata has the same precedence as TileJSON keys.
pub fn archive_tilejson(bytes: &[u8]) -> Result<String, String> {
    let header = crate::pmtiles::Header::parse(bytes)?;
    let [west, south, east, north] = header.bounds;
    let encoding = match header.tile_type {
        crate::pmtiles::TileType::Mvt => r#","encoding":"mvt""#,
        crate::pmtiles::TileType::Mlt => r#","encoding":"mlt""#,
        _ => "",
    };
    Ok(format!(
        r#"{{"minzoom":{},"maxzoom":{},"bounds":[{west},{south},{east},{north}]{encoding}}}"#,
        header.minzoom, header.maxzoom
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mercator() -> TileMatrixSet {
        TileMatrixSet::by_id("WebMercatorQuad").unwrap()
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
            endpoint.url(tile),
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
        let style = orb_v8::parse(&format!(
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
        let endpoint = TileEndpoint::resolve(&source, Some(tilejson), mercator()).unwrap();
        assert_eq!(endpoint.cover_px, 256.0 / std::f64::consts::SQRT_2);
        assert_eq!(
            endpoint.dem_unpack,
            Some([256.0, 1.0, 1.0 / 256.0, 32768.0])
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
        let metadata = archive_tilejson(&header).unwrap();
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
            TileLocation::Pmtiles("https://x/tokyo.pmtiles".into())
        );
        assert!(endpoint.contains(TileIndex::new(10, 909, 403)));
        assert!(!endpoint.contains(TileIndex::new(10, 0, 0)));
        let source = vector(
            r#"{"type":"vector","url":"pmtiles://https://x/a","tiles":["https://x/{z}/{x}/{y}.pbf"],"bounds":[-180,-85,180,85],"encoding":"mvt"}"#,
        );
        let endpoint = TileEndpoint::resolve(&source, Some(&metadata), mercator()).unwrap();
        assert_eq!(endpoint.encoding, Encoding::Mvt);
        assert!(endpoint.contains(TileIndex::new(10, 0, 1)));
        assert!(matches!(
            endpoint.location(TileIndex::new(10, 909, 403)),
            TileLocation::Url(_)
        ));
    }

    #[test]
    fn region_tiles_follow_the_cover_zoom() {
        let source = orb_v8::parse_source(
            "s",
            &json::parse(r#"{"type": "raster", "tiles": ["https://x/{z}/{x}/{y}.png"], "tileSize": 256, "maxzoom": 3}"#).unwrap(),
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
        let endpoint = TileEndpoint::resolve(&source, Some(tilejson), mercator()).unwrap();
        assert_eq!((endpoint.minzoom, endpoint.maxzoom), (2, 12));
        assert_eq!(endpoint.url(TileIndex::new(3, 1, 2)), "https://x/3/1/2.pbf");
    }

    #[test]
    fn tms_rows_count_from_the_bottom() {
        let source = vector(r#"{"type": "vector", "tiles": ["t/{z}/{x}/{y}"], "scheme": "tms"}"#);
        let endpoint = TileEndpoint::resolve(&source, None, mercator()).unwrap();
        assert_eq!(endpoint.url(TileIndex::new(2, 1, 0)), "t/2/1/3");
    }

    #[test]
    fn needs_tile_urls() {
        let source = vector(r#"{"type": "vector", "url": "https://x/tiles.json"}"#);
        assert!(TileEndpoint::resolve(&source, Some("{}"), mercator()).is_err());
    }
}

//! The surface a map draws on, from its scene document, and the planner
//! (plan §9.2) that decides how each source's tiles reach it.

use orb_crs::{Descriptor, Projection, TileMatrixSet, WorldRect};

use crate::style::ImageKind;
use orb_expr::json::{self, Json, JsonValue, Member};

/// A plane, the tile matrix sets of the sources on it, and the one-tile
/// quadtree over it that backgrounds and GeoJSON tile in.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Surface {
    /// The plane's projection; its world bounds are the plane's extent.
    projection: Projection,
    /// Backgrounds and GeoJSON sources tile in this set.
    tms: TileMatrixSet,
    /// Sources' tile matrix sets by source id, from the scene document.
    sources: Vec<(String, TileMatrixSet)>,
    /// The style's globe (v8 `projection`), which the plane's tiles wrap
    /// at low zoom.
    globe: Option<orb_crs::Globe>,
}

/// What job workers need of the surface: the plane GeoJSON tiles in, and
/// the tile matrix sets of sources not in `WebMercatorQuad`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfaceTiles {
    /// The surface's one-tile set; `None` for Web Mercator.
    pub plane: Option<TileMatrixSet>,
    /// By source id.
    pub sources: Vec<(String, TileMatrixSet)>,
}

impl SurfaceTiles {
    /// A source's set: its own, the plane for GeoJSON, or `None` for
    /// `WebMercatorQuad`.
    pub fn source(&self, id: &str, geojson: bool) -> Option<&TileMatrixSet> {
        match self.sources.iter().find(|(source, _)| source == id) {
            Some((_, tms)) => Some(tms),
            None if geojson => self.plane.as_ref(),
            None => None,
        }
    }
}

/// How a source's tiles map onto the surface.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Plan {
    /// The source tiles the surface's own plane: one affine row per tile.
    Affine,
    /// Tiles on another plane: rasters as an error-bounded mesh per tile
    /// (`RasterWarp`), vectors subdivided and drawn through quadratic
    /// patches (`GpuPerVertex`).
    Warp(Box<orb_crs::Warp>),
}

impl Default for Surface {
    /// Web Mercator, every source in `WebMercatorQuad`.
    fn default() -> Self {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").expect("built-in tile matrix set");
        Self {
            projection: *tms.projection(),
            tms,
            sources: Vec::new(),
            globe: None,
        }
    }
}

impl Surface {
    /// A surface on `projection` (Web Mercator if `None`) with the scene
    /// document's tile matrix sets. The plane's extent is that of the first
    /// source set on it, or else the projection's own.
    pub(crate) fn new(
        projection: Option<Projection>,
        sources: Vec<(String, TileMatrixSet)>,
    ) -> Self {
        let mercator = Self::default();
        let projection = projection.unwrap_or(mercator.projection);
        let tms = if projection.same_plane(&mercator.projection) {
            mercator.tms
        } else {
            let extent = sources
                .iter()
                .find(|(_, tms)| tms.projection().same_plane(&projection))
                .map_or(projection.world_bounds(), |(_, tms)| {
                    tms.projection().world_bounds()
                });
            TileMatrixSet::square(projection, extent)
        };
        Self {
            projection: *tms.projection(),
            tms,
            sources,
            globe: None,
        }
    }

    pub(crate) fn projection(&self) -> &Projection {
        &self.projection
    }

    /// What camera views draw on: the plane, or the globe around it.
    pub(crate) fn view_surface(&self) -> orb_crs::Surface {
        orb_crs::Surface {
            projection: self.projection,
            globe: self.globe,
        }
    }

    pub(crate) fn globe(&self) -> Option<orb_crs::Globe> {
        self.globe
    }

    /// Wraps the plane's tiles around a globe at low zoom, or not.
    pub(crate) fn set_globe(&mut self, globe: Option<orb_crs::Globe>) {
        self.globe = globe;
    }

    /// The tile matrix set backgrounds and GeoJSON use.
    pub(crate) fn tms(&self) -> &TileMatrixSet {
        &self.tms
    }

    /// The plane GeoJSON tiles in, for the jobs: `None` for Web Mercator.
    pub(crate) fn plane(&self) -> Option<TileMatrixSet> {
        let mercator = Self::default();
        (!self.projection.same_plane(&mercator.projection)).then(|| self.tms.clone())
    }

    /// The style zoom a tile lays out at (decision 0022): the view's style
    /// zoom where the tile spans 512 pixels. A tile on the surface's plane
    /// has its set's; a warped tile's comes from its footprint's scale on
    /// the surface.
    pub(crate) fn tile_zoom(
        &self,
        tms: &TileMatrixSet,
        tile: orb_crs::TileIndex,
        footprint: Option<orb_crs::Footprint>,
    ) -> f64 {
        match footprint {
            None => tms.style_zoom(tile),
            Some(f) => self
                .projection
                .zoom_at_scale(f.span / 512.0)
                .unwrap_or_else(|| {
                    // Web Mercator's style zoom of a scale, at any latitude.
                    (2.0 * std::f64::consts::PI
                        * orb_geo::wgs84::SEMI_MAJOR_M
                        * self
                            .projection
                            .scale_factor(orb_geo::Geodetic::new(0.0, 0.0))
                        / f.span)
                        .log2()
                }),
        }
    }

    /// What job workers need of the surface.
    pub(crate) fn tiles(&self) -> SurfaceTiles {
        SurfaceTiles {
            plane: self.plane(),
            sources: self.sources.clone(),
        }
    }

    /// The x period of a wrapping plane; 0 if it doesn't wrap.
    pub(crate) fn period(&self) -> f64 {
        self.projection.wrap_period().unwrap_or(0.0)
    }

    /// The surface with another projection, keeping the sources' sets and
    /// the globe.
    pub(crate) fn with_projection(&self, projection: Projection) -> Self {
        Self {
            globe: self.globe,
            ..Self::new(Some(projection), self.sources.clone())
        }
    }

    /// A source's tile matrix set: the scene document's, else the
    /// surface's own for GeoJSON (which is geodetic and tiles anywhere)
    /// and `WebMercatorQuad` for the rest, as v8 means them.
    pub(crate) fn source_tms(&self, id: &str, geojson: bool) -> TileMatrixSet {
        match self.sources.iter().find(|(source, _)| source == id) {
            Some((_, tms)) => tms.clone(),
            None if geojson => self.tms.clone(),
            None => Self::default().tms,
        }
    }

    /// How a source in `tms` reaches the surface, or why it can't yet:
    /// `image` is the kind of an image source.
    pub(crate) fn plan(
        &self,
        tms: &TileMatrixSet,
        image: Option<ImageKind>,
    ) -> Result<Plan, String> {
        if tms.projection().same_plane(&self.projection) {
            return Ok(Plan::Affine);
        }
        if self.globe.is_some() {
            return Err(format!(
                "tile matrix set `{}` is not on the surface's plane; a globe draws only its plane's sources yet",
                tms.id()
            ));
        }
        match image {
            Some(ImageKind::Dem) => Err(format!(
                "tile matrix set `{}` is not on the surface's plane; DEM sources can't be warped yet",
                tms.id()
            )),
            _ => Ok(Plan::Warp(Box::new(orb_crs::Warp::new(
                tms.clone(),
                self.projection,
            )))),
        }
    }
}

/// A projection and its world extent as JSON, for another thread or worker
/// (the web ui role's camera, job workers' GeoJSON plane):
/// `{"projection": {"type": …, <params>}, "extent": [x0, y0, x1, y1]}`.
pub fn projection_to_json(projection: &Projection) -> String {
    surface_to_json(&orb_crs::Surface::plane(*projection))
}

/// A surface as JSON, for the web ui role's camera: its projection as in
/// [`projection_to_json`], and `"globe": [from, to]` (the zooms it hands
/// over between) or `"globe": true` (a globe at every zoom) if it has one.
pub fn surface_to_json(surface: &orb_crs::Surface) -> String {
    let projection = &surface.projection;
    let synthetic = |value| Json::new(value, Default::default());
    let member = |key: &str, value: Json| Member {
        key: key.into(),
        key_span: Default::default(),
        value,
    };
    let descriptor = projection.descriptor();
    let mut fields = vec![member(
        "type",
        synthetic(JsonValue::String(descriptor.kind)),
    )];
    fields.extend(
        descriptor
            .params
            .into_iter()
            .map(|(name, value)| member(&name, synthetic(JsonValue::Number(value)))),
    );
    let b = projection.world_bounds();
    let extent = [b.min.x, b.min.y, b.max.x, b.max.y].map(|v| synthetic(JsonValue::Number(v)));
    let mut members = vec![
        member("projection", synthetic(JsonValue::Object(fields))),
        member("extent", synthetic(JsonValue::Array(extent.to_vec()))),
    ];
    if let Some(globe) = surface.globe {
        let value = match globe.handover {
            Some(zooms) => {
                JsonValue::Array(zooms.map(|z| synthetic(JsonValue::Number(z))).to_vec())
            }
            None => JsonValue::Bool(true),
        };
        members.push(member("globe", synthetic(value)));
    }
    synthetic(JsonValue::Object(members)).to_string()
}

/// A tile matrix set as an OGC 2D-TMS document, for another thread or
/// worker; `None` for a set whose projection has no CRS code.
pub fn tms_to_json(tms: &TileMatrixSet) -> Option<String> {
    let def = tms.definition()?;
    let synthetic = |value| Json::new(value, Default::default());
    let member = |key: &str, value: Json| Member {
        key: key.into(),
        key_span: Default::default(),
        value,
    };
    let number = |v: f64| synthetic(JsonValue::Number(v));
    let text = |v: &str| synthetic(JsonValue::String(v.to_string()));
    let matrices = def
        .matrices
        .iter()
        .map(|m| {
            synthetic(JsonValue::Object(vec![
                member("id", text(&m.id)),
                member("scaleDenominator", number(m.scale_denominator)),
                member(
                    "cellSize",
                    number(m.cell_size.expect("definitions give cells")),
                ),
                member(
                    "pointOfOrigin",
                    synthetic(JsonValue::Array(m.point_of_origin.map(number).to_vec())),
                ),
                member(
                    "cornerOfOrigin",
                    text(if m.bottom_left {
                        "bottomLeft"
                    } else {
                        "topLeft"
                    }),
                ),
                member("tileWidth", number(f64::from(m.tile_width))),
                member("tileHeight", number(f64::from(m.tile_height))),
                member("matrixWidth", number(f64::from(m.matrix_width))),
                member("matrixHeight", number(f64::from(m.matrix_height))),
            ]))
        })
        .collect();
    Some(
        synthetic(JsonValue::Object(vec![
            member("id", text(&def.id)),
            member("crs", text(&def.crs)),
            member(
                "orderedAxes",
                synthetic(JsonValue::Array(vec![text("E"), text("N")])),
            ),
            member("tileMatrices", synthetic(JsonValue::Array(matrices))),
        ]))
        .to_string(),
    )
}

/// Reads [`tms_to_json`]'s document (or any OGC 2D-TMS document).
pub fn tms_from_json(text: &str) -> Result<TileMatrixSet, String> {
    let doc = json::parse(text).map_err(|e| e.to_string())?;
    let def = orb_v8::tms_definition(&doc)?;
    TileMatrixSet::from_definition(&def).map_err(|e| e.to_string())
}

/// Reads [`projection_to_json`]'s document.
pub fn projection_from_json(text: &str) -> Result<Projection, String> {
    surface_from_json(text).map(|surface| surface.projection)
}

/// A surface from [`surface_to_json`].
pub fn surface_from_json(text: &str) -> Result<orb_crs::Surface, String> {
    let bad = || "malformed projection".to_string();
    let doc = json::parse(text).map_err(|e| e.to_string())?;
    let fields = doc
        .get("projection")
        .and_then(Json::as_object)
        .ok_or_else(bad)?;
    let mut descriptor = Descriptor::default();
    for m in fields {
        match (m.key.as_str(), &m.value.value) {
            ("type", JsonValue::String(kind)) => descriptor.kind = kind.clone(),
            (name, JsonValue::Number(value)) => descriptor.params.push((name.to_string(), *value)),
            _ => return Err(bad()),
        }
    }
    let extent = doc
        .get("extent")
        .and_then(Json::as_array)
        .filter(|e| e.len() == 4)
        .ok_or_else(bad)?
        .iter()
        .map(|v| v.as_f64().ok_or_else(bad))
        .collect::<Result<Vec<_>, String>>()?;
    let projection = Projection::new(&descriptor).map_err(|e| e.to_string())?;
    let globe = match doc.get("globe").map(|g| &g.value) {
        None => None,
        Some(JsonValue::Bool(true)) => Some(orb_crs::Globe { handover: None }),
        Some(JsonValue::Array(zooms)) if zooms.len() == 2 => Some(orb_crs::Globe {
            handover: Some([
                zooms[0].as_f64().ok_or_else(bad)?,
                zooms[1].as_f64().ok_or_else(bad)?,
            ]),
        }),
        Some(_) => return Err(bad()),
    };
    Ok(orb_crs::Surface {
        projection: projection.with_extent(WorldRect {
            min: glam::DVec2::new(extent[0], extent[1]),
            max: glam::DVec2::new(extent[2], extent[3]),
        }),
        globe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projections_cross_as_json() {
        let nztm = TileMatrixSet::by_id("NZTM2000Quad").unwrap();
        let surface = Surface::new(
            Some(*nztm.projection()),
            vec![("linz".into(), nztm.clone())],
        );
        // The plane's extent is the source set's.
        assert_eq!(surface.projection(), nztm.projection());
        assert_eq!(
            surface.tms().tile_bounds(orb_crs::TileIndex::new(0, 0, 0)),
            nztm.tile_bounds(orb_crs::TileIndex::new(0, 0, 0))
        );
        let json = projection_to_json(surface.projection());
        assert_eq!(&projection_from_json(&json).unwrap(), surface.projection());
        for globe in [
            None,
            Some(orb_crs::Globe::V8),
            Some(orb_crs::Globe { handover: None }),
        ] {
            let mercator = orb_crs::Surface {
                globe,
                ..Surface::default().view_surface()
            };
            assert_eq!(surface_from_json(&surface_to_json(&mercator)), Ok(mercator));
        }
        assert_eq!(surface.plan(&nztm, None), Ok(Plan::Affine));
        let mercator = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        assert!(matches!(surface.plan(&mercator, None), Ok(Plan::Warp(_))));
        assert!(surface.plan(&mercator, Some(ImageKind::Dem)).is_err());
        assert!(matches!(
            surface.plan(&mercator, Some(ImageKind::Raster)),
            Ok(Plan::Warp(_))
        ));
        assert_eq!(surface.source_tms("other", true), *surface.tms());
        assert_eq!(surface.plane().as_ref(), Some(surface.tms()));
        assert_eq!(Surface::default().plane(), None);
        let back = tms_from_json(&tms_to_json(&nztm).unwrap()).unwrap();
        let tile = orb_crs::TileIndex::new(6, 31, 33);
        assert_eq!(back.tile_bounds(tile), nztm.tile_bounds(tile));
        assert_eq!(surface.tiles().source("linz", false), Some(&nztm));
    }
}

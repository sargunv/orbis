//! The surface a map draws on, from its style's `projection` and sources'
//! tile matrix sets, and the planner
//! that decides how each source's tiles reach it.

use orb_crs::{Descriptor, Projection, TileMatrixSet, TileMatrixSetDefinition, WorldRect};

use crate::style::ImageKind;

/// A plane, the tile matrix sets of the sources on it, and the one-tile
/// quadtree over it that backgrounds and GeoJSON tile in.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Surface {
    /// The plane's projection; its world bounds are the plane's extent.
    projection: Projection,
    /// Backgrounds and GeoJSON sources tile in this set.
    tms: TileMatrixSet,
    /// Sources' tile matrix sets by source id.
    sources: Vec<(String, TileMatrixSet)>,
    /// The low-zoom surface the plane's tiles draw on.
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

/// [`SurfaceTiles`] between workers: the plane's projection and extent,
/// and the sources' tile matrix sets by their definitions.
#[derive(orb_wire::Wire)]
struct SurfaceText {
    plane: Option<(Descriptor, [f64; 4])>,
    sources: Vec<(String, TileMatrixSetDefinition)>,
}

impl From<SurfaceTiles> for SurfaceText {
    fn from(tiles: SurfaceTiles) -> Self {
        Self {
            plane: tiles.plane.map(|tms| {
                let b = tms.projection().world_bounds();
                (
                    tms.projection().descriptor(),
                    [b.min.x, b.min.y, b.max.x, b.max.y],
                )
            }),
            sources: tiles
                .sources
                .iter()
                .filter_map(|(id, tms)| Some((id.clone(), tms.definition()?)))
                .collect(),
        }
    }
}

impl TryFrom<SurfaceText> for SurfaceTiles {
    type Error = String;

    fn try_from(text: SurfaceText) -> Result<Self, String> {
        let plane = |(descriptor, [x0, y0, x1, y1]): (Descriptor, [f64; 4])| {
            let projection = Projection::new(&descriptor).map_err(|e| e.to_string())?;
            let projection = projection.with_extent(WorldRect {
                min: glam::DVec2::new(x0, y0),
                max: glam::DVec2::new(x1, y1),
            });
            Ok::<_, String>(TileMatrixSet::square(projection, projection.world_bounds()))
        };
        Ok(Self {
            plane: text.plane.map(plane).transpose()?,
            sources: text
                .sources
                .iter()
                .map(|(id, def)| {
                    let tms = TileMatrixSet::from_definition(def).map_err(|e| e.to_string())?;
                    Ok((id.clone(), tms))
                })
                .collect::<Result<_, String>>()?,
        })
    }
}

/// Between workers as [`SurfaceText`].
impl orb_wire::Wire for SurfaceTiles {
    fn put(&self, out: &mut Vec<u8>) {
        SurfaceText::from(self.clone()).put(out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        SurfaceText::take(input)?.try_into().ok()
    }
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

    /// A tiled source's set: its own, or `WebMercatorQuad`.
    pub fn source_tms(&self, id: &str) -> &TileMatrixSet {
        self.source(id, false)
            .unwrap_or_else(|| TileMatrixSet::web_mercator())
    }
}

impl Default for Surface {
    /// Web Mercator, every source in `WebMercatorQuad`.
    fn default() -> Self {
        let tms = TileMatrixSet::web_mercator().clone();
        Self {
            projection: *tms.projection(),
            tms,
            sources: Vec::new(),
            globe: None,
        }
    }
}

impl Surface {
    /// A style's surface: its plane, its sources' tile matrix sets and
    /// its low-zoom surface.
    pub(crate) fn of(style: &orb_style::Style) -> Self {
        Self::new(style.projection, style.tile_matrix_sets(), style.globe)
    }

    /// A surface on `projection` (Web Mercator if `None`) with the sources'
    /// tile matrix sets and the low-zoom surface. The plane's extent
    /// is that of the first source set on it, or else the projection's own.
    fn new(
        projection: Option<Projection>,
        sources: Vec<(String, TileMatrixSet)>,
        globe: Option<orb_crs::Globe>,
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
            globe,
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

    /// The tile matrix set backgrounds and GeoJSON use.
    pub(crate) fn tms(&self) -> &TileMatrixSet {
        &self.tms
    }

    /// The plane GeoJSON tiles in, for the jobs: `None` for Web Mercator.
    pub(crate) fn plane(&self) -> Option<TileMatrixSet> {
        let mercator = Self::default();
        (!self.projection.same_plane(&mercator.projection)).then(|| self.tms.clone())
    }

    /// The style zoom a tile lays out at: the view's style
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
                        * orb_crs::wgs84::SEMI_MAJOR_M
                        * self
                            .projection
                            .scale_factor(orb_crs::Geodetic::new(0.0, 0.0))
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

    /// A source's tile matrix set: its `dev.sargunv.orbis:tileMatrixSet`, else the
    /// surface's own for GeoJSON (which is geodetic and tiles anywhere)
    /// and `WebMercatorQuad` for the rest, as v8 means them.
    pub(crate) fn source_tms(&self, id: &str, geojson: bool) -> TileMatrixSet {
        match self.sources.iter().find(|(source, _)| source == id) {
            Some((_, tms)) => tms.clone(),
            None if geojson => self.tms.clone(),
            None => Self::default().tms,
        }
    }

    /// Whether a source in `tms` reaches the surface through a warp, or why
    /// it can't yet: `image` is the kind of an image source. A source on
    /// the surface's own plane draws one affine row per tile; on another,
    /// rasters draw as an error-bounded mesh per tile (`RasterWarp`) and
    /// vectors subdivided through quadratic patches (`GpuPerVertex`).
    pub(crate) fn plan(
        &self,
        tms: &TileMatrixSet,
        image: Option<ImageKind>,
    ) -> Result<bool, String> {
        if tms.projection().same_plane(&self.projection) {
            return Ok(false);
        }
        match image {
            Some(ImageKind::Dem) => Err(format!(
                "tile matrix set `{}` is not on the surface's plane; DEM sources can't be warped yet",
                tms.id()
            )),
            _ => Ok(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surfaces_plan_and_cross_workers() {
        let nztm = TileMatrixSet::by_id("NZTM2000Quad").unwrap();
        let surface = Surface::new(
            Some(*nztm.projection()),
            vec![("linz".into(), nztm.clone())],
            None,
        );
        // The plane's extent is the source set's.
        assert_eq!(surface.projection(), nztm.projection());
        assert_eq!(
            surface.tms().tile_bounds(orb_crs::TileIndex::new(0, 0, 0)),
            nztm.tile_bounds(orb_crs::TileIndex::new(0, 0, 0))
        );
        // A globe draws sources on other planes too: on the sphere, and
        // through their planar warp as it hands over.
        let globe = Surface::new(None, Vec::new(), Some(orb_crs::Globe::V8));
        let crs84 = TileMatrixSet::by_id("WorldCRS84Quad").unwrap();
        assert!(matches!(
            globe.plan(&crs84, Some(ImageKind::Raster)),
            Ok(true)
        ));
        assert_eq!(surface.plan(&nztm, None), Ok(false));
        let mercator = TileMatrixSet::web_mercator();
        assert!(matches!(surface.plan(mercator, None), Ok(true)));
        assert!(surface.plan(mercator, Some(ImageKind::Dem)).is_err());
        assert!(matches!(
            surface.plan(mercator, Some(ImageKind::Raster)),
            Ok(true)
        ));
        assert_eq!(surface.source_tms("other", true), *surface.tms());
        assert_eq!(surface.plane().as_ref(), Some(surface.tms()));
        assert_eq!(Surface::default().plane(), None);
        // The job workers' share crosses as postcard.
        let tiles: SurfaceTiles = orb_wire::decode(&orb_wire::encode(&surface.tiles())).unwrap();
        let tile = orb_crs::TileIndex::new(6, 31, 33);
        let linz = tiles.source("linz", false).unwrap();
        assert_eq!(linz.tile_bounds(tile), nztm.tile_bounds(tile));
        assert_eq!(tiles.plane.unwrap().projection(), surface.projection());
    }
}

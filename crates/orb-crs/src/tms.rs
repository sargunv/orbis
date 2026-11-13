use std::fmt;

use glam::{DAffine2, DMat2, DVec2};

use crate::{Descriptor, Projection, WorldRect};

/// A tile in a tile matrix set: zoom (the matrix identifier), column (left
/// to right) and row (top to bottom).
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct TileIndex {
    pub z: u8,
    pub col: u32,
    pub row: u32,
}

/// `z/col/row`, as messages name a tile.
impl std::fmt::Display for TileIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.z, self.col, self.row)
    }
}

impl TileIndex {
    pub const fn new(z: u8, col: u32, row: u32) -> Self {
        Self { z, col, row }
    }

    /// The four children of a quad tile, in row-major order.
    pub fn children(self) -> [Self; 4] {
        let (z, c, r) = (self.z + 1, self.col * 2, self.row * 2);
        [
            Self::new(z, c, r),
            Self::new(z, c + 1, r),
            Self::new(z, c, r + 1),
            Self::new(z, c + 1, r + 1),
        ]
    }

    /// The parent of a quad tile, or `None` at zoom 0.
    pub fn parent(self) -> Option<Self> {
        (self.z > 0).then(|| Self::new(self.z - 1, self.col / 2, self.row / 2))
    }

    /// The quad tile at zoom `z` holding this one, itself at its own zoom,
    /// or `None` below it.
    pub fn ancestor(self, z: u8) -> Option<Self> {
        let depth = self.z.checked_sub(z)?;
        let shift = |v: u32| v.checked_shr(u32::from(depth)).unwrap_or(0);
        Some(Self::new(z, shift(self.col), shift(self.row)))
    }

    /// Whether this quad tile is `outer` or lies inside it.
    pub fn is_within(self, outer: Self) -> bool {
        self.ancestor(outer.z) == Some(outer)
    }
}

/// A tiling of a projection's plane (OGC 2D TileMatrixSet) whose matrices
/// form a quadtree: each matrix halves the previous one's cell size on the
/// grid of the first one's corner, so every tile has a parent and up to
/// four children. Matrix identifiers are consecutive zooms from
/// [`min_zoom`](Self::min_zoom); the first matrix may have several columns
/// and rows (`WorldCRS84Quad` is 2×1). Later matrices may hold only part
/// of their quadtree level, as NASA GIBS's geographic sets do (2×1, 3×2,
/// 5×3, 10×5…, the tiles that reach the world): the set's extent is then
/// the area every matrix covers, and tiles reaching past it draw only
/// what lies inside.
///
/// Tile indices count columns and rows on the quadtree's grid from the
/// first matrix's top-left corner, whatever the set's own corner of
/// origin and a matrix's first tile ([`native_col`](Self::native_col),
/// [`native_row`](Self::native_row)).
#[derive(Clone, Debug, PartialEq)]
pub struct TileMatrixSet {
    id: String,
    projection: Projection,
    /// World position of the top-left corner of every matrix.
    origin: DVec2,
    /// Width and height of one tile of the first matrix in world units.
    span: f64,
    min_zoom: u8,
    max_zoom: u8,
    /// Pixels per tile side.
    tile_px: u32,
    /// Rows are counted from the bottom in the set's own indices (OGC
    /// `cornerOfOrigin: bottomLeft`).
    bottom_origin: bool,
    /// Each matrix from `min_zoom` on, where it lies on its quadtree level.
    matrices: Vec<Matrix>,
}

/// Where a matrix lies on its quadtree level: its first column and row,
/// and its columns and rows.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Matrix {
    col: u32,
    row: u32,
    cols: u32,
    rows: u32,
}

impl Matrix {
    /// The matrix `levels` deeper on the same area.
    fn deeper(self, levels: u32) -> Self {
        let shift = |v: u32| (u64::from(v) << levels.min(32)).min(u64::from(u32::MAX)) as u32;
        Self {
            col: shift(self.col),
            row: shift(self.row),
            cols: shift(self.cols),
            rows: shift(self.rows),
        }
    }
}

/// One matrix of a tile matrix set definition, as an OGC document writes
/// it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TileMatrix {
    pub id: String,
    pub scale_denominator: f64,
    /// World units per pixel; derived from the scale denominator if absent.
    pub cell_size: Option<f64>,
    /// `pointOfOrigin` (`topLeftCorner` in TMS 1.0), in the order of
    /// [`TileMatrixSetDefinition::northing_first`].
    pub point_of_origin: [f64; 2],
    pub bottom_left: bool,
    pub tile_width: u32,
    pub tile_height: u32,
    pub matrix_width: u32,
    pub matrix_height: u32,
    /// Rows whose tiles coalesce several columns (`variableMatrixWidths`).
    pub variable_widths: bool,
}

/// A tile matrix set as an OGC 2D-TMS document defines it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TileMatrixSetDefinition {
    pub id: String,
    /// The CRS URI or `EPSG:n` code.
    pub crs: String,
    /// Whether points list northing first (`orderedAxes`); the CRS's own
    /// axis order if absent.
    pub northing_first: Option<bool>,
    pub matrices: Vec<TileMatrix>,
}

/// A tile matrix set Orbis can't use: unknown, malformed or not a quadtree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownTileMatrixSet(pub String);

impl fmt::Display for UnknownTileMatrixSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnknownTileMatrixSet {}

/// OGC's standardized rendering pixel size, in metres.
const PIXEL_M: f64 = 0.000_28;

/// A registered set: CRS, top-left origin (easting, northing), first
/// matrix's cell size, columns and rows, zoom range and tile pixels.
struct Builtin {
    crs: &'static str,
    origin: [f64; 2],
    cell: f64,
    size: (u32, u32),
    zooms: (u8, u8),
    tile_px: u32,
}

/// OGC's UTM grids (`UTM31WGS84Quad`) share one layout across zones.
const UTM_QUAD: Builtin = Builtin {
    crs: "",
    origin: [-9_501_965.729_312_76, 20_003_931.458_625_5],
    cell: 78_140.357_260_255_9,
    size: (1, 2),
    zooms: (1, 24),
    tile_px: 256,
};

fn builtin(id: &str) -> Option<Builtin> {
    let ups = Builtin {
        crs: "",
        origin: [-14_440_759.350_252, 18_440_759.350_252],
        cell: 128_443.432_4,
        size: (1, 1),
        zooms: (0, 24),
        tile_px: 256,
    };
    Some(match id {
        "WorldCRS84Quad" => Builtin {
            crs: "OGC:CRS84",
            origin: [-180.0, 90.0],
            cell: 0.703_125,
            size: (2, 1),
            zooms: (0, 23),
            tile_px: 256,
        },
        // LINZ's NZTM2000 Map Tile Grid V2.
        "NZTM2000Quad" => Builtin {
            crs: "EPSG:2193",
            origin: [-3_260_586.728_4, 10_438_190.165_2],
            cell: 139_770_566.007_179 * PIXEL_M,
            size: (1, 1),
            zooms: (0, 21),
            tile_px: 256,
        },
        "EuropeanETRS89_LAEAQuad" => Builtin {
            crs: "EPSG:3035",
            origin: [2_000_000.0, 5_500_000.0],
            cell: 17_578.125,
            size: (1, 1),
            zooms: (0, 15),
            tile_px: 256,
        },
        "UPSArcticWGS84Quad" => Builtin {
            crs: "EPSG:5041",
            ..ups
        },
        "UPSAntarcticWGS84Quad" => Builtin {
            crs: "EPSG:5042",
            ..ups
        },
        _ => return None,
    })
}

impl TileMatrixSet {
    /// `WebMercatorQuad`, made once.
    pub fn web_mercator() -> &'static Self {
        static SET: std::sync::LazyLock<TileMatrixSet> = std::sync::LazyLock::new(|| {
            TileMatrixSet::by_id("WebMercatorQuad").expect("built-in tile matrix set")
        });
        &SET
    }

    /// A registered tile matrix set by its OGC identifier.
    pub fn by_id(id: &str) -> Result<Self, UnknownTileMatrixSet> {
        if id == "WebMercatorQuad" {
            let projection =
                Projection::from_descriptor("web-mercator").expect("built-in projection");
            let bounds = projection.world_bounds();
            return Ok(Self::quad(
                id,
                projection,
                DVec2::new(bounds.min.x, bounds.max.y),
                bounds.size().x,
                (1, 1),
                (0, 30),
                256,
                false,
            ));
        }
        let utm = id
            .strip_prefix("UTM")
            .and_then(|rest| rest.strip_suffix("WGS84Quad"))
            .and_then(|zone| zone.parse::<u32>().ok())
            .filter(|zone| (1..=60).contains(zone));
        let (crs, b) = match (utm, builtin(id)) {
            (Some(zone), _) => (format!("EPSG:{}", 32600 + zone), UTM_QUAD),
            (None, Some(b)) => (b.crs.to_string(), b),
            (None, None) => {
                return Err(UnknownTileMatrixSet(format!(
                    "unknown tile matrix set `{id}`"
                )));
            }
        };
        let projection = Projection::new(&Descriptor::named(&crs)).expect("built-in projection");
        Ok(Self::quad(
            id,
            projection,
            DVec2::from(b.origin),
            b.cell * f64::from(b.tile_px),
            b.size,
            b.zooms,
            b.tile_px,
            false,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn quad(
        id: &str,
        projection: Projection,
        origin: DVec2,
        span: f64,
        size: (u32, u32),
        (min_zoom, max_zoom): (u8, u8),
        tile_px: u32,
        bottom_origin: bool,
    ) -> Self {
        let extent = WorldRect {
            min: DVec2::new(origin.x, origin.y - span * f64::from(size.1)),
            max: DVec2::new(origin.x + span * f64::from(size.0), origin.y),
        };
        let first = Matrix {
            col: 0,
            row: 0,
            cols: size.0,
            rows: size.1,
        };
        Self {
            id: id.to_string(),
            projection: projection.with_extent(extent),
            origin,
            span,
            min_zoom,
            max_zoom,
            tile_px,
            bottom_origin,
            matrices: (0..=u32::from(max_zoom - min_zoom))
                .map(|k| first.deeper(k))
                .collect(),
        }
    }

    /// A tile matrix set from its OGC definition. Its matrices must form a
    /// quadtree with consecutive numeric identifiers, each on the grid of
    /// the first and within the children of the one before; variable-width
    /// rows and other layouts are reported as errors.
    pub fn from_definition(def: &TileMatrixSetDefinition) -> Result<Self, UnknownTileMatrixSet> {
        let fail = |message: String| Err(UnknownTileMatrixSet(format!("{}: {message}", def.id)));
        let projection = match Projection::from_crs_uri(&def.crs) {
            Ok(p) => p,
            Err(e) => return fail(e.to_string()),
        };
        let Some(first) = def.matrices.first() else {
            return fail("no tile matrices".into());
        };
        let northing_first = def.northing_first.unwrap_or(projection.northing_first());
        let unit_m = projection.metres_per_unit();
        let cell = |m: &TileMatrix| {
            m.cell_size
                .unwrap_or(m.scale_denominator * PIXEL_M / unit_m)
        };
        let origin = |m: &TileMatrix| {
            let [a, b] = m.point_of_origin;
            if northing_first {
                DVec2::new(b, a)
            } else {
                DVec2::new(a, b)
            }
        };
        let Ok(min_zoom) = first.id.parse::<u8>() else {
            return fail(format!("matrix `{}` is not a zoom number", first.id));
        };
        if min_zoom > 30 {
            return fail(format!("matrix `{}` is past zoom 30", first.id));
        }
        if first.tile_width != first.tile_height || first.tile_width == 0 {
            return fail("tiles are not square".into());
        }
        let cell0 = cell(first);
        // A matrix's top-left corner.
        let corner = |m: &TileMatrix| {
            let o = origin(m);
            if m.bottom_left {
                DVec2::new(
                    o.x,
                    o.y + cell(m) * f64::from(m.tile_height) * f64::from(m.matrix_height),
                )
            } else {
                o
            }
        };
        let top_left = corner(first);
        let span = cell0 * f64::from(first.tile_width);
        let matrices = &def.matrices[..def.matrices.len().min(31 - usize::from(min_zoom))];
        let mut placed: Vec<Matrix> = Vec::with_capacity(matrices.len());
        let mut extent: Option<WorldRect> = None;
        for (k, m) in matrices.iter().enumerate() {
            let zoom = u32::from(min_zoom) + k as u32;
            if m.id.parse::<u32>().ok() != Some(zoom) {
                return fail(format!("matrix `{}` should be zoom {zoom}", m.id));
            }
            if m.variable_widths {
                return fail("variable matrix widths (coalesced rows) are not supported".into());
            }
            let scale = f64::from(1u32 << k);
            let expected = cell0 / scale;
            if (cell(m) - expected).abs() > 1e-6 * expected
                || m.tile_width != first.tile_width
                || m.tile_height != first.tile_height
                || m.bottom_left != first.bottom_left
            {
                return fail(format!(
                    "matrix `{}` doesn't halve the one before: not a quadtree",
                    m.id
                ));
            }
            // Its first tile on the quadtree's grid at this level.
            let tile = span / scale;
            let offset = (corner(m) - top_left) * DVec2::new(1.0, -1.0) / tile;
            let (rounded, cols, rows) = (offset.round(), m.matrix_width, m.matrix_height);
            if (offset - rounded).abs().max_element() > 1e-6 || rounded.min_element() < 0.0 {
                return fail(format!(
                    "matrix `{}` isn't on the first one's tile grid",
                    m.id
                ));
            }
            if cols == 0 || rows == 0 {
                return fail(format!("matrix `{}` has no tiles", m.id));
            }
            let matrix = Matrix {
                col: rounded.x as u32,
                row: rounded.y as u32,
                cols,
                rows,
            };
            // Every tile needs its parent.
            if let Some(&up) = placed.last() {
                let up = up.deeper(1);
                let end = |first: u32, count: u32| u64::from(first) + u64::from(count);
                if matrix.col < up.col
                    || matrix.row < up.row
                    || end(matrix.col, cols) > end(up.col, up.cols)
                    || end(matrix.row, rows) > end(up.row, up.rows)
                {
                    return fail(format!(
                        "matrix `{}` reaches past the one before's children",
                        m.id
                    ));
                }
            }
            placed.push(matrix);
            let min = DVec2::new(
                top_left.x + f64::from(matrix.col) * tile,
                top_left.y - f64::from(matrix.row + rows) * tile,
            );
            let area = WorldRect {
                min,
                max: min + DVec2::new(f64::from(cols), f64::from(rows)) * tile,
            };
            extent = Some(match extent {
                Some(e) => WorldRect {
                    min: e.min.max(area.min),
                    max: e.max.min(area.max),
                },
                None => area,
            });
        }
        let max_zoom = (u32::from(min_zoom) + placed.len() as u32 - 1) as u8;
        let mut set = Self::quad(
            &def.id,
            projection,
            top_left,
            span,
            (first.matrix_width, first.matrix_height),
            (min_zoom, max_zoom),
            first.tile_width,
            first.bottom_left,
        );
        set.projection = set.projection.with_extent(extent.expect("a first matrix"));
        set.matrices = placed;
        Ok(set)
    }

    /// A one-tile quadtree over the square around `rect`, for a plane that
    /// has no tile matrix set of its own: 512-pixel tiles to zoom 30.
    pub fn square(projection: Projection, rect: WorldRect) -> Self {
        let size = rect.size();
        let span = size.max_element();
        let origin = if size.x == size.y {
            DVec2::new(rect.min.x, rect.max.y)
        } else {
            let center = rect.center();
            DVec2::new(center.x - span / 2.0, center.y + span / 2.0)
        };
        Self::quad(
            "square",
            projection,
            origin,
            span,
            (1, 1),
            (0, 30),
            512,
            false,
        )
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// The set as an OGC definition, which
    /// [`from_definition`](Self::from_definition) reads back; `None` for a set whose
    /// projection has no CRS code (a parametric descriptor's).
    pub fn definition(&self) -> Option<TileMatrixSetDefinition> {
        let kind = self.projection.descriptor().kind;
        let crs = match kind.as_str() {
            "geographic" => "OGC:CRS84".to_string(),
            kind if kind.starts_with("EPSG:") => kind.to_string(),
            _ => return None,
        };
        let unit_m = self.projection.metres_per_unit();
        let matrices = (self.min_zoom..=self.max_zoom)
            .map(|z| {
                let m = self.matrix(z).expect("a matrix");
                let (cols, rows) = (m.cols, m.rows);
                let span = self.tile_span(z);
                let cell = span / f64::from(self.tile_px);
                let x = self.origin.x + f64::from(m.col) * span;
                let origin = if self.bottom_origin {
                    [x, self.origin.y - span * f64::from(m.row + rows)]
                } else {
                    [x, self.origin.y - span * f64::from(m.row)]
                };
                TileMatrix {
                    id: z.to_string(),
                    scale_denominator: cell * unit_m / PIXEL_M,
                    cell_size: Some(cell),
                    point_of_origin: origin,
                    bottom_left: self.bottom_origin,
                    tile_width: self.tile_px,
                    tile_height: self.tile_px,
                    matrix_width: cols,
                    matrix_height: rows,
                    variable_widths: false,
                }
            })
            .collect();
        Some(TileMatrixSetDefinition {
            id: self.id.clone(),
            crs,
            northing_first: Some(false),
            matrices,
        })
    }

    /// The set's projection, whose world bounds are the set's extent.
    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    /// The first matrix's zoom.
    pub fn min_zoom(&self) -> u8 {
        self.min_zoom
    }

    pub fn max_zoom(&self) -> u8 {
        self.max_zoom
    }

    /// World width and height of one tile at zoom `z`.
    pub fn tile_span(&self, z: u8) -> f64 {
        self.span / 2f64.powi(i32::from(z) - i32::from(self.min_zoom))
    }

    /// The matrix at zoom `z`; past the last, the last one's area.
    fn matrix(&self, z: u8) -> Option<Matrix> {
        let k = usize::from(z.checked_sub(self.min_zoom)?);
        let last = self.matrices.len() - 1;
        Some(self.matrices[k.min(last)].deeper(k.saturating_sub(last) as u32))
    }

    /// Columns and rows of the matrix at zoom `z`.
    pub fn matrix_size(&self, z: u8) -> (u32, u32) {
        self.matrix(z).map_or((0, 0), |m| (m.cols, m.rows))
    }

    /// The first column and row of the matrix at zoom `z` on its
    /// quadtree level: zero but for a matrix that holds only part of it
    /// away from the set's corner.
    pub fn matrix_start(&self, z: u8) -> (u32, u32) {
        self.matrix(z).map_or((0, 0), |m| (m.col, m.row))
    }

    /// Whether a tile lies inside its matrix.
    pub fn contains(&self, tile: TileIndex) -> bool {
        let within = |v: u32, first: u32, count: u32| {
            (u64::from(first)..u64::from(first) + u64::from(count)).contains(&u64::from(v))
        };
        self.matrix(tile.z)
            .is_some_and(|m| within(tile.col, m.col, m.cols) && within(tile.row, m.row, m.rows))
    }

    /// The column a tile has in its matrix's own indices.
    pub fn native_col(&self, tile: TileIndex) -> u32 {
        tile.col.wrapping_sub(self.matrix_start(tile.z).0)
    }

    /// The row a tile has in its matrix's own indices: counted from the
    /// bottom for a bottom-left corner of origin.
    pub fn native_row(&self, tile: TileIndex) -> u32 {
        let (first, rows) = (self.matrix_start(tile.z).1, self.matrix_size(tile.z).1);
        if self.bottom_origin {
            (first + rows).wrapping_sub(1).wrapping_sub(tile.row)
        } else {
            tile.row.wrapping_sub(first)
        }
    }

    /// The tiles of the first matrix.
    pub fn roots(&self) -> impl Iterator<Item = TileIndex> + '_ {
        let Matrix { cols, rows, .. } = self.matrices[0];
        let z = self.min_zoom;
        (0..rows).flat_map(move |row| (0..cols).map(move |col| TileIndex::new(z, col, row)))
    }

    /// The area every level's tiles cover without a gap, when each matrix
    /// holds its whole quadtree level under the first's tiles: a cover of
    /// a region inside it leaves no hole. `None` for a set whose matrices
    /// hold only part of a level.
    pub fn full_area(&self) -> Option<WorldRect> {
        let first = self.matrices[0];
        let whole = (first.col, first.row) == (0, 0)
            && (self.matrices.iter().enumerate()).all(|(k, &m)| m == first.deeper(k as u32));
        whole.then(|| {
            let areas = self.roots().map(|t| self.tile_area(t));
            areas.reduce(|a, b| WorldRect {
                min: a.min.min(b.min),
                max: a.max.max(b.max),
            })
        })?
    }

    /// The world rectangle a tile covers.
    pub fn tile_bounds(&self, tile: TileIndex) -> WorldRect {
        let span = self.tile_span(tile.z);
        let min_x = self.origin.x + f64::from(tile.col) * span;
        let max_y = self.origin.y - f64::from(tile.row) * span;
        WorldRect {
            min: DVec2::new(min_x, max_y - span),
            max: DVec2::new(min_x + span, max_y),
        }
    }

    /// The part of a tile inside the set's extent: all of it, but for tiles
    /// of a partial matrix that reach past the area every matrix covers.
    pub fn tile_area(&self, tile: TileIndex) -> WorldRect {
        let b = self.tile_bounds(tile);
        let e = self.projection.world_bounds();
        WorldRect {
            min: b.min.max(e.min),
            max: b.max.min(e.max),
        }
    }

    /// A point of a tile, in fractions of its side from its top-left
    /// corner (y down), in world space; held to the set's extent, so a
    /// tile reaching past it maps its outside onto the extent's edge.
    pub fn tile_point(&self, tile: TileIndex, uv: DVec2) -> DVec2 {
        let b = self.tile_bounds(tile);
        let world = DVec2::new(b.min.x + uv.x * b.size().x, b.max.y - uv.y * b.size().y);
        self.projection.world_bounds().clamp(world)
    }

    /// The affine map from a tile's data units, `extent` per side with y down
    /// from the top-left corner, into world space.
    pub fn data_to_world(&self, tile: TileIndex, extent: u32) -> DAffine2 {
        let bounds = self.tile_bounds(tile);
        let scale = self.tile_span(tile.z) / f64::from(extent);
        DAffine2::from_mat2_translation(
            DMat2::from_diagonal(DVec2::new(scale, -scale)),
            DVec2::new(bounds.min.x, bounds.max.y),
        )
    }

    /// A world position as a fraction of the first matrix's tile span from
    /// the set's top-left corner, x right and y down.
    pub fn unit(&self, world: DVec2) -> DVec2 {
        DVec2::new(world.x - self.origin.x, self.origin.y - world.y) / self.span
    }

    /// The style zoom a tile lays out at on its own plane: the zoom of a
    /// view in which it spans 512 pixels. On a plane with a
    /// zoom reference it is the plane's; in Web Mercator, ground resolution
    /// at the tile's centre in GL JS's convention, which is the tile's own
    /// zoom in `WebMercatorQuad`.
    pub fn style_zoom(&self, tile: TileIndex) -> f64 {
        let zoom = match self
            .projection
            .zoom_at_scale(self.tile_span(tile.z) / 512.0)
        {
            Some(zoom) => zoom,
            None => {
                let center = self.projection.inverse(self.tile_bounds(tile).center());
                let ground_span = self.tile_span(tile.z) / self.projection.scale_factor(center);
                let parallel = 2.0
                    * std::f64::consts::PI
                    * crate::wgs84::SEMI_MAJOR_M
                    * crate::style_parallel(center.lat);
                (parallel / ground_span).log2()
            }
        };
        // Snap the round-off of planes whose zooms are whole.
        if (zoom - zoom.round()).abs() < 1e-9 {
            zoom.round()
        } else {
            zoom
        }
    }

    /// A tile's longitude and latitude extent, `[west, south, east, north]`
    /// in degrees, from points along the edges of its area; a tile around
    /// a pole spans every longitude.
    pub fn geodetic_bounds(&self, tile: TileIndex) -> [f64; 4] {
        const STEPS: u32 = 8;
        let rect = self.tile_area(tile);
        let mut b = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        let first = self.projection.inverse(rect.min);
        let mut previous = first.lon;
        let mut unwrapped = first.lon;
        for i in 0..4 * STEPS {
            let t = f64::from(i % STEPS) / f64::from(STEPS);
            let p = match i / STEPS {
                0 => DVec2::new(rect.min.x + t * rect.size().x, rect.min.y),
                1 => DVec2::new(rect.max.x, rect.min.y + t * rect.size().y),
                2 => DVec2::new(rect.max.x - t * rect.size().x, rect.max.y),
                _ => DVec2::new(rect.min.x, rect.max.y - t * rect.size().y),
            };
            let g = self.projection.inverse(p);
            // Follow longitude continuously around the edge.
            unwrapped += (g.lon - previous + 540.0).rem_euclid(360.0) - 180.0;
            previous = g.lon;
            b = [
                b[0].min(unwrapped),
                b[1].min(g.lat),
                b[2].max(unwrapped),
                b[3].max(g.lat),
            ];
        }
        let encircles = (unwrapped - first.lon).abs() > 180.0;
        for pole in [90.0, -90.0] {
            let p = crate::Geodetic::new(0.0, pole);
            // Planes that clamp the pole (Web Mercator) don't contain it.
            let w = self.projection.forward(p);
            let inside = self.projection.contains(p)
                && (self.projection.inverse(w).lat - pole).abs() < 1e-6
                && w.cmpge(rect.min).all()
                && w.cmple(rect.max).all();
            if inside || (encircles && (pole > 0.0) == (b[3] > 0.0)) {
                b[1] = b[1].min(pole);
                b[3] = b[3].max(pole);
            }
        }
        if encircles || b[2] - b[0] >= 360.0 {
            b[0] = -180.0;
            b[2] = 180.0;
        }
        b
    }

    /// The tiles at zoom `z` that a world rectangle touches.
    pub fn tiles_in(&self, rect: WorldRect, z: u8) -> impl Iterator<Item = TileIndex> + use<> {
        let span = self.tile_span(z);
        let m = self.matrix(z).unwrap_or(Matrix {
            col: 0,
            row: 0,
            cols: 0,
            rows: 0,
        });
        let index = |v: f64, first: u32, count: u32| {
            v.floor()
                .clamp(f64::from(first), f64::from(first + count.max(1) - 1)) as u32
        };
        let col = |x: f64| index((x - self.origin.x) / span, m.col, m.cols);
        let row = |y: f64| index((self.origin.y - y) / span, m.row, m.rows);
        let (c0, c1) = (col(rect.min.x), col(rect.max.x));
        let (r0, r1) = (row(rect.max.y), row(rect.min.y));
        (r0..=r1).flat_map(move |r| (c0..=c1).map(move |c| TileIndex::new(z, c, r)))
    }
}

/// The centre of a tile's area (inside its set's extent) in fractions of
/// the tile's side, y down.
pub(crate) fn area_center(tms: &TileMatrixSet, tile: TileIndex) -> DVec2 {
    let (b, c) = (tms.tile_bounds(tile), tms.tile_area(tile).center());
    DVec2::new(c.x - b.min.x, b.max.y - c.y) / b.size()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_mercator_quad_bounds() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let world = tms.projection().world_bounds();
        assert_eq!(tms.tile_bounds(TileIndex::new(0, 0, 0)), world);
        let ne = tms.tile_bounds(TileIndex::new(1, 1, 0));
        assert_eq!(ne.min, DVec2::ZERO);
        assert_eq!(ne.max, world.max);
    }

    #[test]
    fn data_to_world_maps_corners() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let tile = TileIndex::new(3, 5, 2);
        let bounds = tms.tile_bounds(tile);
        let m = tms.data_to_world(tile, 4096);
        assert_eq!(
            m.transform_point2(DVec2::ZERO),
            DVec2::new(bounds.min.x, bounds.max.y)
        );
        let br = m.transform_point2(DVec2::splat(4096.0));
        assert!(
            (br - DVec2::new(bounds.max.x, bounds.min.y))
                .abs()
                .max_element()
                < 1e-6
        );
    }

    #[test]
    fn parent_and_children_agree() {
        let t = TileIndex::new(4, 7, 9);
        assert!(t.children().iter().all(|c| c.parent() == Some(t)));
        assert_eq!(TileIndex::new(0, 0, 0).parent(), None);
        let deep = TileIndex::new(9, 0b1_0110_1011, 0b0_1101_0010);
        assert_eq!(deep.ancestor(9), Some(deep));
        assert_eq!(
            deep.ancestor(5),
            Some(TileIndex::new(5, 0b1_0110, 0b0_1101))
        );
        assert_eq!(deep.ancestor(10), None);
        assert!(deep.is_within(TileIndex::new(5, 0b1_0110, 0b0_1101)));
        assert!(!deep.is_within(TileIndex::new(5, 0b1_0111, 0b0_1101)));
        assert!(!TileIndex::new(5, 0b1_0110, 0b0_1101).is_within(deep));
    }

    fn matrix(id: u32, sd: f64, origin: [f64; 2], size: (u32, u32)) -> TileMatrix {
        TileMatrix {
            id: id.to_string(),
            scale_denominator: sd,
            cell_size: None,
            point_of_origin: origin,
            bottom_left: false,
            tile_width: 256,
            tile_height: 256,
            matrix_width: size.0,
            matrix_height: size.1,
            variable_widths: false,
        }
    }

    #[test]
    fn registry_traps() {
        // WorldCRS84Quad is two tiles wide at zoom 0, in degrees.
        let crs84 = TileMatrixSet::by_id("WorldCRS84Quad").unwrap();
        assert_eq!(crs84.matrix_size(0), (2, 1));
        assert_eq!(
            crs84.tile_bounds(TileIndex::new(0, 1, 0)).max,
            DVec2::new(180.0, 90.0)
        );
        // EuropeanETRS89_LAEAQuad lists northing first; its origin is the
        // north-west corner at easting 2,000 km, northing 5,500 km.
        let laea = TileMatrixSet::by_id("EuropeanETRS89_LAEAQuad").unwrap();
        let root = laea.tile_bounds(TileIndex::new(0, 0, 0));
        assert_eq!((root.min.x, root.max.y), (2_000_000.0, 5_500_000.0));
        // UTM grids start at zoom 1 with one column and two rows.
        let utm = TileMatrixSet::by_id("UTM31WGS84Quad").unwrap();
        assert_eq!((utm.min_zoom(), utm.matrix_size(1)), (1, (1, 2)));
        assert_eq!(utm.roots().count(), 2);
        assert!(TileMatrixSet::by_id("UTM61WGS84Quad").is_err());
        // NZTM2000Quad's 10,018 km square spans 256 pixels at zoom 0.
        let nztm = TileMatrixSet::by_id("NZTM2000Quad").unwrap();
        assert!((nztm.tile_span(0) - 10_018_754.171_394_6).abs() < 1e-3);
        assert!(nztm.projection().world_bounds().size().x > 1e7);
    }

    #[test]
    fn definitions_follow_axis_order_and_corner() {
        // NZTM2000Quad as LINZ writes it: northing first, from EPSG:2193.
        let nztm = TileMatrixSetDefinition {
            id: "NZTM2000Quad".into(),
            crs: "https://www.opengis.net/def/crs/EPSG/0/2193".into(),
            northing_first: None,
            matrices: (0..22)
                .map(|z| {
                    matrix(
                        z,
                        139_770_566.007_179 / f64::from(1u32 << z),
                        [10_438_190.165_2, -3_260_586.728_4],
                        (1 << z, 1 << z),
                    )
                })
                .collect(),
        };
        let parsed = TileMatrixSet::from_definition(&nztm).unwrap();
        let builtin = TileMatrixSet::by_id("NZTM2000Quad").unwrap();
        assert_eq!(
            parsed.tile_bounds(TileIndex::new(3, 2, 5)),
            builtin.tile_bounds(TileIndex::new(3, 2, 5))
        );
        assert!((parsed.tile_span(4) - builtin.tile_span(4)).abs() < 1e-6);

        // A bottom-left corner counts rows from the bottom.
        let mut flipped = nztm.clone();
        for m in &mut flipped.matrices {
            m.bottom_left = true;
            m.point_of_origin = [10_438_190.165_2 - 10_018_754.171_394_6, -3_260_586.728_4];
        }
        let flipped = TileMatrixSet::from_definition(&flipped).unwrap();
        let tile = TileIndex::new(2, 1, 0);
        assert_eq!(flipped.native_row(tile), 3);
        assert!(
            (flipped.tile_bounds(tile).max - builtin.tile_bounds(tile).max)
                .abs()
                .max_element()
                < 1e-6
        );

        // A set's definition reads back as the same set.
        let back = TileMatrixSet::from_definition(&parsed.definition().unwrap()).unwrap();
        assert_eq!(
            back.tile_bounds(TileIndex::new(5, 7, 9)),
            parsed.tile_bounds(TileIndex::new(5, 7, 9))
        );
        assert_eq!(
            (back.min_zoom(), back.max_zoom()),
            (parsed.min_zoom(), parsed.max_zoom())
        );
        let utm = TileMatrixSet::by_id("UTM33WGS84Quad").unwrap();
        let back = TileMatrixSet::from_definition(&utm.definition().unwrap()).unwrap();
        assert_eq!(
            back.tile_bounds(TileIndex::new(3, 1, 5)),
            utm.tile_bounds(TileIndex::new(3, 1, 5))
        );

        // Coalesced rows and scales that don't halve are refused.
        let mut gnosis = nztm.clone();
        gnosis.matrices[1].variable_widths = true;
        assert!(TileMatrixSet::from_definition(&gnosis).is_err());
        let mut lcc = nztm.clone();
        lcc.matrices[2].scale_denominator *= 1.1;
        assert!(TileMatrixSet::from_definition(&lcc).is_err());
        let mut unknown = nztm;
        unknown.crs = "EPSG:31467".into();
        assert!(TileMatrixSet::from_definition(&unknown).is_err());
    }

    #[test]
    fn style_zooms_and_geodetic_bounds() {
        let mercator = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        for tile in [TileIndex::new(0, 0, 0), TileIndex::new(14, 14552, 6451)] {
            assert_eq!(mercator.style_zoom(tile), f64::from(tile.z));
        }
        let b = mercator.geodetic_bounds(TileIndex::new(1, 1, 0));
        assert!((b[0] - 0.0).abs() < 1e-9 && (b[2] - 180.0).abs() < 1e-9);
        assert!((b[1] - 0.0).abs() < 1e-9 && (b[3] - crate::MERCATOR_MAX_LATITUDE).abs() < 1e-9);
        let nztm = TileMatrixSet::by_id("NZTM2000Quad").unwrap();
        let p = nztm
            .projection()
            .forward(crate::Geodetic::new(174.78, -41.29));
        let tile = nztm
            .tiles_in(WorldRect { min: p, max: p }, 8)
            .next()
            .unwrap();
        // Its tiles are a quarter of Mercator's, measured at its natural
        // origin on the equator with its 0.9996 scale: two zooms deeper
        // than their matrix, everywhere on the plane.
        let expected = 10.0 + 0.9996f64.log2();
        assert!((nztm.style_zoom(tile) - expected).abs() < 1e-6);
        assert_eq!(
            nztm.style_zoom(TileIndex::new(8, 0, 0)),
            nztm.style_zoom(tile)
        );
        // A polar tile around the pole spans every longitude.
        let ups = TileMatrixSet::by_id("UPSArcticWGS84Quad").unwrap();
        let pole = ups.projection().forward(crate::Geodetic::new(0.0, 90.0));
        let tile = ups
            .tiles_in(
                WorldRect {
                    min: pole,
                    max: pole,
                },
                3,
            )
            .next()
            .unwrap();
        let b = ups.geodetic_bounds(tile);
        assert_eq!((b[0], b[2], b[3]), (-180.0, 180.0, 90.0));
    }

    /// NASA GIBS's geographic 500 m set, as its capabilities write it: 288°
    /// tiles at level 0, halving, each matrix the tiles that reach the
    /// world.
    fn gibs() -> TileMatrixSetDefinition {
        let sizes = [
            (2, 1),
            (3, 2),
            (5, 3),
            (10, 5),
            (20, 10),
            (40, 20),
            (80, 40),
            (160, 80),
        ];
        TileMatrixSetDefinition {
            id: "500m".into(),
            crs: "urn:ogc:def:crs:OGC:1.3:CRS84".into(),
            northing_first: None,
            matrices: (0..8)
                .map(|z| TileMatrix {
                    tile_width: 512,
                    tile_height: 512,
                    ..matrix(
                        z,
                        223_632_905.611_487_1 / f64::from(1u32 << z),
                        [-180.0, 90.0],
                        sizes[z as usize],
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn partial_matrices_keep_to_the_world() {
        let tms = TileMatrixSet::from_definition(&gibs()).unwrap();
        assert!((tms.tile_span(0) - 288.0).abs() < 1e-9);
        assert_eq!(tms.matrix_size(1), (3, 2));
        assert!(tms.contains(TileIndex::new(1, 2, 1)));
        assert!(!tms.contains(TileIndex::new(1, 3, 0)));
        assert!(!tms.contains(TileIndex::new(2, 4, 3)));
        // The extent is the world, which every matrix covers.
        let world = tms.projection().world_bounds();
        assert!((world.min - DVec2::new(-180.0, -90.0)).abs().max_element() < 1e-9);
        assert!((world.max - DVec2::new(180.0, 90.0)).abs().max_element() < 1e-9);
        // A tile past the antimeridian and the south pole draws its part.
        let area = tms.tile_area(TileIndex::new(0, 1, 0));
        assert!((area.min - DVec2::new(108.0, -90.0)).abs().max_element() < 1e-9);
        assert!((area.max - DVec2::new(180.0, 90.0)).abs().max_element() < 1e-9);
        let corner = tms.tile_point(TileIndex::new(0, 1, 0), DVec2::ONE);
        assert!((corner - DVec2::new(180.0, -90.0)).abs().max_element() < 1e-9);
        let b = tms.geodetic_bounds(TileIndex::new(1, 2, 1));
        assert!(
            (b[1] + 90.0).abs() < 1e-9 && (b[2] - 180.0).abs() < 1e-9,
            "{b:?}"
        );
        // Covers visit only the tiles the matrices hold, children of the
        // edge's tiles too.
        let cover = tms.cover(0..=0, 2, |_, _| crate::CoverDecision::Refine);
        assert_eq!(cover.len(), 15);
        assert!(cover.iter().all(|t| tms.contains(t.index)));
        // Its definition reads back as the same set.
        let back = TileMatrixSet::from_definition(&tms.definition().unwrap()).unwrap();
        assert_eq!(back, tms);
    }

    #[test]
    fn partial_matrices_off_the_corner() {
        // A matrix holding the middle of its level, in a set counting rows
        // from the bottom: its tiles keep their quadtree indices, and URLs
        // take its own.
        let mut def = gibs();
        def.matrices.truncate(3);
        for m in &mut def.matrices {
            m.bottom_left = true;
        }
        def.matrices[0].point_of_origin = [-180.0, 90.0 - 288.0];
        def.matrices[1].point_of_origin = [-36.0, 90.0 - 288.0];
        def.matrices[1].matrix_width = 2;
        def.matrices[2].point_of_origin = [-36.0, 90.0 - 216.0];
        def.matrices[2].matrix_width = 4;
        let tms = TileMatrixSet::from_definition(&def).unwrap();
        assert_eq!(tms.matrix_start(1), (1, 0));
        assert!(!tms.contains(TileIndex::new(1, 0, 0)));
        let tile = TileIndex::new(2, 3, 0);
        assert!(tms.contains(tile));
        assert_eq!((tms.native_col(tile), tms.native_row(tile)), (1, 2));
        let world = tms.projection().world_bounds();
        assert!((world.min - DVec2::new(-36.0, -126.0)).abs().max_element() < 1e-9);
        assert!((world.max - DVec2::new(252.0, 90.0)).abs().max_element() < 1e-9);
        let back = TileMatrixSet::from_definition(&tms.definition().unwrap()).unwrap();
        assert_eq!(back, tms);
        // Matrices off the first one's grid, or past the one before's
        // children, are refused.
        let mut off = def.clone();
        off.matrices[2].point_of_origin[0] += 10.0;
        assert!(TileMatrixSet::from_definition(&off).is_err());
        let mut orphan = def.clone();
        orphan.matrices[2].point_of_origin[0] = -180.0;
        assert!(TileMatrixSet::from_definition(&orphan).is_err());
        // So are empty matrices, overflowing ones and zooms past 30.
        let mut empty = def.clone();
        empty.matrices[0].matrix_height = 0;
        assert!(TileMatrixSet::from_definition(&empty).is_err());
        let mut huge = def.clone();
        huge.matrices[2].matrix_width = u32::MAX;
        assert!(TileMatrixSet::from_definition(&huge).is_err());
        let mut deep = def;
        for (k, m) in deep.matrices.iter_mut().enumerate() {
            m.id = (40 + k).to_string();
        }
        assert!(TileMatrixSet::from_definition(&deep).is_err());
    }

    #[test]
    fn tiles_in_a_rect() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let rect = tms.tile_bounds(TileIndex::new(2, 1, 1));
        let inner = WorldRect {
            min: rect.min + 1.0,
            max: rect.max - 1.0,
        };
        let tiles: Vec<_> = tms.tiles_in(inner, 3).collect();
        assert_eq!(tiles, TileIndex::new(2, 1, 1).children());
    }
}

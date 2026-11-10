//! Geometry predicates and CheapRuler distances from the pinned style-spec.
//! Stored feature coordinates are tile-local; literal geometry is longitude /
//! latitude. Conversion matches the reference, including integer quantization.
use crate::{CanonicalTile, FeatureGeometry, json::Json};
use orb_tile::GeometryType;
type Point = [f64; 2];
type Polygon = Vec<Vec<Point>>;
#[derive(Clone, Debug)]
pub(crate) enum Geometry {
    Point(Point),
    Line(Vec<Point>),
    Polygon(Polygon),
}

pub(crate) fn parse_geojson(json: &Json) -> Option<Vec<Geometry>> {
    fn point(j: &Json) -> Option<Point> {
        let a = j.as_array()?;
        Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
    }
    fn line(j: &Json) -> Option<Vec<Point>> {
        j.as_array()?.iter().map(point).collect()
    }
    fn polygon(j: &Json) -> Option<Polygon> {
        j.as_array()?.iter().map(line).collect()
    }
    let ty = json.get("type")?.as_str()?;
    if ty == "Feature" {
        return parse_geojson(json.get("geometry")?);
    }
    if ty == "FeatureCollection" {
        return json
            .get("features")?
            .as_array()?
            .iter()
            .map(parse_geojson)
            .collect::<Option<Vec<_>>>()
            .map(|g| g.into_iter().flatten().collect());
    }
    let c = json.get("coordinates")?;
    Some(match ty {
        "Point" => vec![Geometry::Point(point(c)?)],
        "MultiPoint" => line(c)?.into_iter().map(Geometry::Point).collect(),
        "LineString" => vec![Geometry::Line(line(c)?)],
        "MultiLineString" => polygon(c)?.into_iter().map(Geometry::Line).collect(),
        "Polygon" => vec![Geometry::Polygon(polygon(c)?)],
        "MultiPolygon" => c
            .as_array()?
            .iter()
            .map(|j| polygon(j).map(Geometry::Polygon))
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    })
}
fn world(p: Point, z: u8) -> Point {
    let scale = 8192.0 * 2.0f64.powi(z as i32);
    let x = (180.0 + p[0]) / 360.0;
    let y = (180.0
        - 180.0 / std::f64::consts::PI
            * (std::f64::consts::FRAC_PI_4 + p[1] * std::f64::consts::PI / 360.0)
                .tan()
                .ln())
        / 360.0;
    [(x * scale + 0.5).floor(), (y * scale + 0.5).floor()]
}
fn lng_lat(p: Point, tile: CanonicalTile, extent: f64) -> Point {
    let scale = 2.0f64.powi(tile.z as i32);
    let x = (p[0] / extent + tile.x as f64) / scale;
    let y = (p[1] / extent + tile.y as f64) / scale;
    [
        x * 360.0 - 180.0,
        360.0 / std::f64::consts::PI
            * ((180.0 - y * 360.0) * std::f64::consts::PI / 180.0)
                .exp()
                .atan()
            - 90.0,
    ]
}
/// A GeoJSON geometry quantized into `tile`'s local integer coordinates,
/// as the reference runner does before it evaluates, and as a tile holds
/// a feature's geometry.
pub(crate) fn fixture_geometry(json: &Json, tile: CanonicalTile) -> Option<FeatureGeometry> {
    let geometries = parse_geojson(json)?;
    let kind = match json.get("type")?.as_str()? {
        "Point" | "MultiPoint" => GeometryType::Point,
        "LineString" | "MultiLineString" => GeometryType::LineString,
        _ => GeometryType::Polygon,
    };
    let mut parts = Vec::new();
    for g in geometries {
        match g {
            Geometry::Point(p) => parts.push(vec![p]),
            Geometry::Line(l) => parts.push(l),
            Geometry::Polygon(p) => parts.extend(p),
        }
    }
    for p in parts.iter_mut().flatten() {
        let w = world(*p, tile.z);
        *p = [w[0] - tile.x as f64 * 8192.0, w[1] - tile.y as f64 * 8192.0];
    }
    Some(FeatureGeometry {
        kind,
        parts,
        tile,
        extent: 8192.0,
    })
}
fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}
fn intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
    let p = [b[0] - a[0], b[1] - a[1]];
    let q = [d[0] - c[0], d[1] - c[1]];
    if q[0] * p[1] - q[1] * p[0] == 0.0 {
        return false;
    }
    cross(a, b, c) * cross(a, b, d) <= 0.0 && cross(c, d, a) * cross(c, d, b) <= 0.0
}
fn contains(p: Point, rings: &[Vec<Point>], boundary: bool) -> bool {
    let mut inside = false;
    for ring in rings {
        for edge in ring.windows(2) {
            let (a, b) = (edge[0], edge[1]);
            if cross(p, a, b) == 0.0
                && (p[0] - a[0]) * (p[0] - b[0]) <= 0.0
                && (p[1] - a[1]) * (p[1] - b[1]) <= 0.0
            {
                return boundary;
            }
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
            {
                inside = !inside;
            }
        }
    }
    inside
}
fn bbox(points: impl Iterator<Item = Point>) -> [f64; 4] {
    points.fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |b, p| {
            [
                b[0].min(p[0]),
                b[1].min(p[1]),
                b[2].max(p[0]),
                b[3].max(p[1]),
            ]
        },
    )
}
pub(crate) fn within(feature: &FeatureGeometry, geometries: &[Geometry]) -> bool {
    if feature.kind == GeometryType::Polygon {
        return false;
    }
    let polygons = geometries
        .iter()
        .filter_map(|g| {
            if let Geometry::Polygon(p) = g {
                Some(
                    p.iter()
                        .map(|r| {
                            r.iter()
                                .map(|&p| world(p, feature.tile.z))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Polygon>(),
                )
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let poly_bbox = bbox(polygons.iter().flatten().flatten().copied());
    let mut parts = feature
        .parts
        .iter()
        .map(|part| {
            part.iter()
                .map(|p| {
                    [
                        p[0] * 8192.0 / feature.extent + feature.tile.x as f64 * 8192.0,
                        p[1] * 8192.0 / feature.extent + feature.tile.y as f64 * 8192.0,
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let world_size = 2.0f64.powi(feature.tile.z as i32) * 8192.0;
    let b = bbox(parts.iter().flatten().copied());
    if feature.kind == GeometryType::Point || b[2] - b[0] <= world_size / 2.0 {
        for p in parts.iter_mut().flatten() {
            if p[0] < poly_bbox[0] || p[0] > poly_bbox[2] {
                let mut shift = if p[0] - poly_bbox[0] > world_size / 2.0 {
                    -world_size
                } else if poly_bbox[0] - p[0] > world_size / 2.0 {
                    world_size
                } else {
                    0.0
                };
                if shift == 0.0 {
                    shift = if p[0] - poly_bbox[2] > world_size / 2.0 {
                        -world_size
                    } else if poly_bbox[2] - p[0] > world_size / 2.0 {
                        world_size
                    } else {
                        0.0
                    };
                }
                p[0] += shift;
            }
        }
    }
    let b = bbox(parts.iter().flatten().copied());
    if b[0] <= poly_bbox[0] || b[1] <= poly_bbox[1] || b[2] >= poly_bbox[2] || b[3] >= poly_bbox[3]
    {
        return false;
    }
    if feature.kind == GeometryType::Point {
        return parts
            .iter()
            .flatten()
            .all(|&p| polygons.iter().any(|poly| contains(p, poly, false)));
    }
    parts.iter().all(|line| {
        polygons.iter().any(|poly| {
            line.iter().all(|&p| contains(p, poly, false))
                && !line.windows(2).any(|seg| {
                    poly.iter().any(|ring| {
                        ring.windows(2)
                            .any(|edge| intersect(seg[0], seg[1], edge[0], edge[1]))
                    })
                })
        })
    })
}
struct Ruler {
    kx: f64,
    ky: f64,
}
fn wrap(degrees: f64) -> f64 {
    if (-180.0..=180.0).contains(&degrees) {
        degrees
    } else {
        (degrees + 180.0).rem_euclid(360.0) - 180.0
    }
}
impl Ruler {
    fn new(lat: f64) -> Self {
        let flattening = 1.0 / 298.257223563;
        let e2 = flattening * (2.0 - flattening);
        let m = std::f64::consts::PI / 180.0 * 6378.137 * 1000.0;
        let cos = lat.to_radians().cos();
        let w2 = 1.0 / (1.0 - e2 * (1.0 - cos * cos));
        let w = w2.sqrt();
        Self {
            kx: m * w * cos,
            ky: m * w * w2 * (1.0 - e2),
        }
    }
    fn distance(&self, a: Point, b: Point) -> f64 {
        let dx = wrap(a[0] - b[0]) * self.kx;
        let dy = (a[1] - b[1]) * self.ky;
        (dx * dx + dy * dy).sqrt()
    }
    fn segment(&self, p: Point, a: Point, b: Point) -> f64 {
        let dx = wrap(b[0] - a[0]) * self.kx;
        let dy = (b[1] - a[1]) * self.ky;
        let t = if dx != 0.0 || dy != 0.0 {
            (wrap(p[0] - a[0]) * self.kx * dx + (p[1] - a[1]) * self.ky * dy) / (dx * dx + dy * dy)
        } else {
            0.0
        };
        let nearest = if t > 1.0 {
            b
        } else if t > 0.0 {
            [a[0] + dx / self.kx * t, a[1] + dy / self.ky * t]
        } else {
            a
        };
        self.distance(p, nearest)
    }
    fn segments(&self, a: Point, b: Point, c: Point, d: Point) -> f64 {
        if intersect(a, b, c, d) {
            0.0
        } else {
            self.segment(a, c, d)
                .min(self.segment(b, c, d))
                .min(self.segment(c, a, b))
                .min(self.segment(d, a, b))
        }
    }
}
fn points(g: &Geometry) -> Vec<Point> {
    match g {
        Geometry::Point(p) => vec![*p],
        Geometry::Line(l) => l.clone(),
        Geometry::Polygon(p) => p.iter().flatten().copied().collect(),
    }
}
fn edges(g: &Geometry) -> Vec<(Point, Point)> {
    let mut out = Vec::new();
    let mut line = |l: &[Point], closed: bool| {
        out.extend(l.windows(2).map(|w| (w[0], w[1])));
        if closed && let (Some(&a), Some(&b)) = (l.first(), l.last()) {
            out.push((b, a));
        }
    };
    match g {
        Geometry::Point(_) => {}
        Geometry::Line(l) => line(l, false),
        Geometry::Polygon(p) => p.iter().for_each(|r| line(r, true)),
    }
    out
}
fn pair_distance(a: &Geometry, b: &Geometry, r: &Ruler) -> f64 {
    let ap = points(a);
    let bp = points(b);
    if let Geometry::Polygon(poly) = a
        && bp.iter().any(|&p| contains(p, poly, true))
    {
        return 0.0;
    }
    if let Geometry::Polygon(poly) = b
        && ap.iter().any(|&p| contains(p, poly, true))
    {
        return 0.0;
    }
    let ae = edges(a);
    let be = edges(b);
    let mut distance = f64::INFINITY;
    if ae.is_empty() && be.is_empty() {
        for &p in &ap {
            for &q in &bp {
                distance = distance.min(r.distance(p, q));
            }
        }
    } else if ae.is_empty() {
        for &p in &ap {
            for &(a, b) in &be {
                distance = distance.min(r.segment(p, a, b));
            }
        }
    } else if be.is_empty() {
        for &p in &bp {
            for &(a, b) in &ae {
                distance = distance.min(r.segment(p, a, b));
            }
        }
    } else {
        for &(a, b) in &ae {
            for &(c, d) in &be {
                distance = distance.min(r.segments(a, b, c, d));
            }
        }
    }
    distance
}
pub(crate) fn distance(feature: &FeatureGeometry, geometries: &[Geometry]) -> f64 {
    let parts = feature
        .parts
        .iter()
        .map(|r| {
            r.iter()
                .map(|&p| lng_lat(p, feature.tile, feature.extent))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let Some(&first) = parts.iter().flatten().next() else {
        return f64::NAN;
    };
    let ruler = Ruler::new(first[1]);
    let mut input = Vec::new();
    match feature.kind {
        GeometryType::Point => input.extend(parts.into_iter().flatten().map(Geometry::Point)),
        GeometryType::LineString => {
            input.push(Geometry::Line(parts.into_iter().flatten().collect()))
        }
        GeometryType::Polygon => {
            let mut outer_sign = None;
            let mut polygon = Vec::new();
            for ring in parts {
                let area = ring
                    .iter()
                    .zip(ring.iter().cycle().skip(1))
                    .take(ring.len())
                    .map(|(a, b)| (a[0] - b[0]) * (b[1] + a[1]))
                    .sum::<f64>();
                if area == 0.0 {
                    continue;
                }
                let sign = area < 0.0;
                let outer = *outer_sign.get_or_insert(sign);
                if sign == outer && !polygon.is_empty() {
                    input.push(Geometry::Polygon(std::mem::take(&mut polygon)));
                }
                polygon.push(ring);
            }
            if !polygon.is_empty() {
                input.push(Geometry::Polygon(polygon));
            }
        }
    }
    let mut distance = f64::INFINITY;
    for a in &input {
        for b in geometries {
            distance = distance.min(pair_distance(a, b, &ruler));
        }
    }
    distance
}

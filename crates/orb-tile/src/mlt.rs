//! MapLibre Tile decoding using `mlt-core`'s stable tag-0x01 format.

use mlt_core::geo_types::{Geometry, LineString, Polygon};
use mlt_core::{Decoder, LendingIterator, ParsedLayer, Parser, PropValueRef};

use crate::decode::{finish_layer, signed_area};
use crate::{DecodeError, GeometryType, LayerBuilder, TileFrame, Value};

/// Decode an uncompressed MLT tile into the same conventions as [`crate::mvt`].
///
/// Uses `mlt-core`'s default 64 MiB parser and decoder budgets. Its lending
/// iterator reuses a geometry object and property cursors; coordinates are
/// copied into the builder, and present strings into owned property values.
/// Unknown layer tags are skipped, as prescribed by MLT. Experimental tag-0x02
/// layers are not enabled.
///
/// Both decoders sort property columns by name. Exact cross-format equality
/// requires encodings that preserve feature order and property types. In
/// particular, `mlt-core`'s MVT converter stringifies mixed scalar columns and
/// its encoder may narrow positive signed integer columns to unsigned ones;
/// a decoder cannot recover the original types from those encodings.
pub fn decode(bytes: &[u8]) -> Result<TileFrame, DecodeError> {
    let layers = Parser::default().parse_layers(bytes)?;
    let mut decoder = Decoder::default();
    let mut tile = TileFrame::default();
    for layer in layers {
        let ParsedLayer::Tag01(layer) = layer.decode_all(&mut decoder)? else {
            continue;
        };
        let mut builder = LayerBuilder::new(layer.name(), layer.extent().get());
        let names: Vec<_> = layer.iterate_prop_names().map(|n| n.to_string()).collect();
        let mut features = layer.iter_features();
        while let Some(feature) = features.next() {
            let feature = feature?;
            let kind = match feature.geometry() {
                Geometry::Point(_) | Geometry::MultiPoint(_) => GeometryType::Point,
                Geometry::LineString(_) | Geometry::MultiLineString(_) => GeometryType::LineString,
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => GeometryType::Polygon,
                _ => return Err(DecodeError::InvalidGeometry("unsupported MLT geometry")),
            };
            builder.begin_feature(feature.id(), kind);
            if !geometry(feature.geometry(), &mut builder)? {
                builder.abandon_feature();
                continue;
            }
            for (name, value) in names.iter().zip(feature.iter_all_properties()) {
                if let Some(value) = value {
                    builder.set_property(name, property(value));
                }
            }
        }
        tile.layers.push(finish_layer(builder));
    }
    Ok(tile)
}

fn geometry(geometry: &Geometry<i32>, builder: &mut LayerBuilder) -> Result<bool, DecodeError> {
    match geometry {
        Geometry::Point(point) => {
            builder.begin_part();
            builder.begin_ring();
            builder.push_vertex([point.x(), point.y()]);
            Ok(true)
        }
        Geometry::MultiPoint(points) => {
            if points.0.is_empty() {
                return Ok(false);
            }
            builder.begin_part();
            builder.begin_ring();
            for point in points {
                builder.push_vertex([point.x(), point.y()]);
            }
            Ok(true)
        }
        Geometry::LineString(line) => line_string(line, builder),
        Geometry::MultiLineString(lines) => {
            let mut nonempty = false;
            for line in lines {
                nonempty |= line_string(line, builder)?;
            }
            Ok(nonempty)
        }
        Geometry::Polygon(polygon) => Ok(polygon_geometry(polygon, builder)),
        Geometry::MultiPolygon(polygons) => {
            let mut nonempty = false;
            for polygon in polygons {
                nonempty |= polygon_geometry(polygon, builder);
            }
            Ok(nonempty)
        }
        _ => Err(DecodeError::InvalidGeometry("unsupported MLT geometry")),
    }
}

fn line_string(line: &LineString<i32>, builder: &mut LayerBuilder) -> Result<bool, DecodeError> {
    if line.0.is_empty() {
        return Ok(false);
    }
    if line.0.len() < 2 {
        return Err(DecodeError::InvalidGeometry(
            "line has fewer than two vertices",
        ));
    }
    builder.begin_part();
    builder.begin_ring();
    for point in &line.0 {
        builder.push_vertex([point.x, point.y]);
    }
    Ok(true)
}

fn polygon_geometry(polygon: &Polygon<i32>, builder: &mut LayerBuilder) -> bool {
    if !polygon_ring(polygon.exterior(), true, builder) {
        return false;
    }
    for hole in polygon.interiors() {
        polygon_ring(hole, false, builder);
    }
    true
}

fn polygon_ring(line: &LineString<i32>, exterior: bool, builder: &mut LayerBuilder) -> bool {
    let mut points = line.0.as_slice();
    while points.len() > 1 && points.first() == points.last() {
        points = &points[..points.len() - 1];
    }
    let area = signed_area(points.iter().map(|p| [p.x, p.y]));
    if area == 0 {
        return false;
    }
    if exterior {
        builder.begin_part();
    }
    builder.begin_ring();
    // MLT carries topology explicitly and need not use MVT winding. Reverse
    // when necessary, keeping the first vertex as the start of the ring.
    if (area > 0) == exterior {
        for point in points {
            builder.push_vertex([point.x, point.y]);
        }
    } else {
        builder.push_vertex([points[0].x, points[0].y]);
        for point in points[1..].iter().rev() {
            builder.push_vertex([point.x, point.y]);
        }
    }
    true
}

fn property(value: PropValueRef<'_>) -> Value {
    match value {
        PropValueRef::Bool(v) => Value::Bool(v),
        PropValueRef::I8(v) => Value::I64(i64::from(v)),
        PropValueRef::I32(v) => Value::I64(i64::from(v)),
        PropValueRef::I64(v) => Value::I64(v),
        PropValueRef::U8(v) => Value::U64(u64::from(v)),
        PropValueRef::U32(v) => Value::U64(u64::from(v)),
        PropValueRef::U64(v) => Value::U64(v),
        PropValueRef::F32(v) => Value::F64(f64::from(v)),
        PropValueRef::F64(v) => Value::F64(v),
        PropValueRef::Str(v) => Value::Str(v.to_owned()),
    }
}

//! Mapbox Vector Tile 2.1 decoding from uncompressed protobuf bytes.

use std::borrow::Cow;

use crate::builder::CodedColumns;
use crate::decode::{finish_layer_with, signed_area};
use crate::{DecodeError, GeometryType, Layer, LayerBuilder, TileFrame, Value};

/// Decode an uncompressed MVT tile, dropping unknown and empty geometries.
///
/// Layer dictionaries and feature headers are read before their payloads:
/// protobuf permits these fields after the features/geometry that need them.
/// Payloads are decoded once directly into the builder, with one reusable
/// polygon ring buffer and no intermediate feature objects; tags stay
/// indices into the layer's tables until its columns are typed.
/// Property columns are sorted by name, matching [`crate::mlt::decode`].
pub fn decode(bytes: &[u8]) -> Result<TileFrame, DecodeError> {
    // Every vertex/part/feature consumes input bytes, so this also bounds the
    // builder's u32 offsets without adding checks to its public hot path.
    if bytes.len() > u32::MAX as usize {
        return Err(DecodeError::TileTooLarge);
    }
    let mut tile = TileFrame::default();
    let mut reader = Reader::new(bytes);
    while let Some((field, wire)) = reader.field()? {
        if field == 3 {
            tile.layers.push(layer(reader.message(wire)?)?);
        } else {
            reader.skip(field, wire)?;
        }
    }
    Ok(tile)
}

fn layer(bytes: &[u8]) -> Result<Layer, DecodeError> {
    let mut reader = Reader::new(bytes);
    let (mut name, mut version) = (None, None);
    let mut extent = 4096;
    let mut keys = Vec::new();
    let mut values = Vec::new();
    while let Some((field, wire)) = reader.field()? {
        match field {
            1 => name = Some(string(reader.message(wire)?)),
            3 => keys.push(string(reader.message(wire)?)),
            4 => values.push(value(reader.message(wire)?)?),
            5 => extent = reader.uint32(wire)?,
            15 => version = Some(reader.uint32(wire)?),
            _ => reader.skip(field, wire)?,
        }
    }
    // The spec's default is 1. Version 1 has the same encoding; version 2
    // only made ring winding normative, and rings are classified by signed
    // area either way, as GL JS's vector-tile-js does.
    let version = version.unwrap_or(1);
    if !(1..=2).contains(&version) {
        return Err(DecodeError::UnsupportedVersion(version));
    }
    if extent == 0 {
        return Err(DecodeError::InvalidValue("zero layer extent"));
    }
    let mut builder =
        LayerBuilder::new(name.ok_or(DecodeError::MissingField("layer.name"))?, extent);
    let mut columns = CodedColumns::new(&keys, &values);
    let mut ring = Vec::new();
    let mut reader = Reader::new(bytes);
    while let Some((field, wire)) = reader.field()? {
        if field == 2 {
            let tables = (keys.len(), values.len());
            feature(
                reader.message(wire)?,
                tables,
                &mut builder,
                &mut columns,
                &mut ring,
            )?;
        } else {
            reader.skip(field, wire)?;
        }
    }
    let columns = columns.finish(builder.features());
    Ok(finish_layer_with(builder, columns))
}

/// Decodes one feature into `builder`, its tags into `columns`; `tables`
/// are the layer's key and value counts.
fn feature(
    bytes: &[u8],
    (keys, values): (usize, usize),
    builder: &mut LayerBuilder,
    columns: &mut CodedColumns<'_>,
    ring: &mut Vec<[i32; 2]>,
) -> Result<(), DecodeError> {
    let mut reader = Reader::new(bytes);
    let mut id = None;
    let mut kind = 0;
    while let Some((field, wire)) = reader.field()? {
        match field {
            1 => id = Some(reader.uint(wire)?),
            3 => kind = reader.uint32(wire)?,
            _ => reader.skip(field, wire)?,
        }
    }
    let kind = match kind {
        1 => GeometryType::Point,
        2 => GeometryType::LineString,
        3 => GeometryType::Polygon,
        _ => return Ok(()),
    };
    builder.begin_feature(id, kind);
    let nonempty = geometry(Repeated::new(bytes, 4), kind, builder, ring)?;
    let feature = (builder.features() - 1) as u32;
    let mut tags = Repeated::new(bytes, 2);
    while let Some(key) = tags.next()? {
        let Some(index) = tags.next()? else {
            return Err(DecodeError::InvalidValue("odd property tag count"));
        };
        if key as usize >= keys {
            return Err(DecodeError::InvalidTagIndex {
                table: "key",
                index: key,
            });
        }
        if index as usize >= values {
            return Err(DecodeError::InvalidTagIndex {
                table: "value",
                index,
            });
        }
        if nonempty {
            columns.set(feature, key, index);
        }
    }
    if !nonempty {
        builder.abandon_feature();
    }
    Ok(())
}

fn geometry(
    mut commands: Repeated<'_>,
    kind: GeometryType,
    builder: &mut LayerBuilder,
    ring: &mut Vec<[i32; 2]>,
) -> Result<bool, DecodeError> {
    let mut cursor = [0i32; 2];
    let mut open = false;
    let mut vertices = 0;
    let mut nonempty = false;
    let mut exterior = false;
    ring.clear();
    while let Some(command) = commands.next()? {
        let count = command >> 3;
        if count == 0 {
            return Err(DecodeError::InvalidGeometry("zero command count"));
        }
        match command & 7 {
            1 => {
                if kind != GeometryType::Point && count != 1 {
                    return Err(DecodeError::InvalidGeometry("MoveTo count must be one"));
                }
                if kind == GeometryType::Polygon && open {
                    return Err(DecodeError::InvalidGeometry("unclosed polygon ring"));
                }
                if kind == GeometryType::LineString && open && vertices < 2 {
                    return Err(DecodeError::InvalidGeometry(
                        "line has fewer than two vertices",
                    ));
                }
                if kind != GeometryType::Polygon && (kind != GeometryType::Point || !nonempty) {
                    builder.begin_part();
                    builder.begin_ring();
                }
                open = true;
                vertices = 0;
                for _ in 0..count {
                    advance(&mut commands, &mut cursor)?;
                    if kind == GeometryType::Polygon {
                        ring.push(cursor);
                    } else {
                        builder.push_vertex(cursor);
                        nonempty = true;
                    }
                    vertices += 1;
                }
            }
            2 => {
                if !open || kind == GeometryType::Point {
                    return Err(DecodeError::InvalidGeometry("unexpected LineTo"));
                }
                for _ in 0..count {
                    advance(&mut commands, &mut cursor)?;
                    if kind == GeometryType::Polygon {
                        ring.push(cursor);
                    } else {
                        builder.push_vertex(cursor);
                    }
                    vertices += 1;
                }
            }
            7 => {
                if kind != GeometryType::Polygon || !open || count != 1 || vertices < 3 {
                    return Err(DecodeError::InvalidGeometry("unexpected ClosePath"));
                }
                // Tolerate an explicit closing vertex but store closure implicitly.
                while ring.len() > 1 && ring.first() == ring.last() {
                    ring.pop();
                }
                let area = signed_area(ring.iter().copied());
                if area != 0 {
                    if area > 0 {
                        builder.begin_part();
                        exterior = true;
                    } else if !exterior {
                        return Err(DecodeError::InvalidGeometry("hole precedes exterior"));
                    }
                    builder.begin_ring();
                    for &point in ring.iter() {
                        builder.push_vertex(point);
                    }
                    nonempty = true;
                }
                ring.clear();
                open = false;
            }
            _ => return Err(DecodeError::InvalidGeometry("unknown command")),
        }
    }
    if kind == GeometryType::Polygon && open {
        return Err(DecodeError::InvalidGeometry("unclosed polygon ring"));
    }
    if kind == GeometryType::LineString && open && vertices < 2 {
        return Err(DecodeError::InvalidGeometry(
            "line has fewer than two vertices",
        ));
    }
    Ok(nonempty)
}

// The decoder's hot paths build their errors only on failure: `ok_or`
// would build and drop one for every vertex and every byte read, and
// `DecodeError` has a variant with a destructor.
fn advance(commands: &mut Repeated<'_>, cursor: &mut [i32; 2]) -> Result<(), DecodeError> {
    for coordinate in cursor {
        let Some(delta) = commands.next()? else {
            return Err(DecodeError::UnexpectedEnd);
        };
        let delta = ((delta >> 1) as i32) ^ -((delta & 1) as i32);
        let Some(moved) = coordinate.checked_add(delta) else {
            return Err(DecodeError::CoordinateOverflow);
        };
        *coordinate = moved;
    }
    Ok(())
}

fn value(bytes: &[u8]) -> Result<Value, DecodeError> {
    let mut reader = Reader::new(bytes);
    let mut value = None;
    while let Some((field, wire)) = reader.field()? {
        let item = match field {
            1 => Value::Str(string(reader.message(wire)?).into_owned()),
            2 => {
                reader.expect_wire(wire, 5)?;
                Value::F64(f64::from(f32::from_le_bytes(reader.fixed()?)))
            }
            3 => {
                reader.expect_wire(wire, 1)?;
                Value::F64(f64::from_le_bytes(reader.fixed()?))
            }
            4 => Value::I64(reader.uint(wire)? as i64),
            5 => Value::U64(reader.uint(wire)?),
            6 => {
                let n = reader.uint(wire)?;
                Value::I64(((n >> 1) as i64) ^ -((n & 1) as i64))
            }
            7 => Value::Bool(reader.uint(wire)? != 0),
            _ => {
                reader.skip(field, wire)?;
                continue;
            }
        };
        if value.replace(item).is_some() {
            return Err(DecodeError::InvalidValue("multiple fields in MVT Value"));
        }
    }
    value.ok_or(DecodeError::InvalidValue("empty MVT Value"))
}

/// A tile string, invalid UTF-8 replaced with U+FFFD as GL JS's pbf
/// decodes it: a stray byte in one name shouldn't lose the tile.
fn string(bytes: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(bytes)
}

/// Borrowed protobuf reader. All lengths, wire types and varint widths are
/// checked before slicing; unknown fields (including groups) are skipped.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], DecodeError> {
        let Some((head, tail)) = self.bytes.split_at_checked(len) else {
            return Err(DecodeError::UnexpectedEnd);
        };
        self.bytes = tail;
        Ok(head)
    }

    fn varint(&mut self) -> Result<u64, DecodeError> {
        let mut value = 0;
        for shift in (0..70).step_by(7) {
            let byte = self.take(1)?[0];
            if shift == 63 && byte > 1 {
                return Err(DecodeError::InvalidVarint);
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(DecodeError::InvalidVarint)
    }

    fn field(&mut self) -> Result<Option<(u32, u8)>, DecodeError> {
        if self.bytes.is_empty() {
            return Ok(None);
        }
        let tag = self.varint()?;
        let field = tag >> 3;
        // Field 0, which protobuf reserves, reads as an unknown field to
        // skip, as GL JS's `pbf` reads it: tiles padded with zero bytes
        // decode.
        if field > 0x1fff_ffff {
            return Err(DecodeError::InvalidProtobuf("invalid field number"));
        }
        Ok(Some((field as u32, (tag & 7) as u8)))
    }

    fn expect_wire(&self, wire: u8, expected: u8) -> Result<(), DecodeError> {
        if wire == expected {
            Ok(())
        } else {
            Err(DecodeError::InvalidProtobuf("unexpected wire type"))
        }
    }

    fn uint(&mut self, wire: u8) -> Result<u64, DecodeError> {
        self.expect_wire(wire, 0)?;
        self.varint()
    }

    fn uint32(&mut self, wire: u8) -> Result<u32, DecodeError> {
        u32::try_from(self.uint(wire)?).map_err(|_| DecodeError::InvalidProtobuf("uint32 overflow"))
    }

    fn message(&mut self, wire: u8) -> Result<&'a [u8], DecodeError> {
        self.expect_wire(wire, 2)?;
        let len = usize::try_from(self.varint()?).map_err(|_| DecodeError::TileTooLarge)?;
        self.take(len)
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let mut bytes = [0; N];
        bytes.copy_from_slice(self.take(N)?);
        Ok(bytes)
    }

    fn skip(&mut self, field: u32, wire: u8) -> Result<(), DecodeError> {
        // Iterative group skipping avoids recursion on untrusted input.
        let mut groups = Vec::new();
        let (mut field, mut wire) = (field, wire);
        loop {
            match wire {
                0 => {
                    self.varint()?;
                }
                1 => {
                    self.take(8)?;
                }
                2 => {
                    self.message(wire)?;
                }
                3 => groups.push(field),
                4 if groups.pop() == Some(field) => {}
                5 => {
                    self.take(4)?;
                }
                _ => return Err(DecodeError::InvalidProtobuf("invalid wire type or group")),
            }
            if groups.is_empty() {
                return Ok(());
            }
            let Some(next) = self.field()? else {
                return Err(DecodeError::UnexpectedEnd);
            };
            (field, wire) = next;
        }
    }
}

/// Iterates repeated uint32 fields, concatenating packed and unpacked segments
/// without allocating, even when a segment boundary splits a geometry command.
struct Repeated<'a> {
    reader: Reader<'a>,
    packed: Reader<'a>,
    field: u32,
}

impl<'a> Repeated<'a> {
    fn new(bytes: &'a [u8], field: u32) -> Self {
        Self {
            reader: Reader::new(bytes),
            packed: Reader::new(&[]),
            field,
        }
    }

    fn next(&mut self) -> Result<Option<u32>, DecodeError> {
        loop {
            if !self.packed.bytes.is_empty() {
                return self.packed.uint32(0).map(Some);
            }
            let Some((field, wire)) = self.reader.field()? else {
                return Ok(None);
            };
            if field != self.field {
                self.reader.skip(field, wire)?;
            } else if wire == 2 {
                self.packed = Reader::new(self.reader.message(wire)?);
            } else {
                return self.reader.uint32(wire).map(Some);
            }
        }
    }
}

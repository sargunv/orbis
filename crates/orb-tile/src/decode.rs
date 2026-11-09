use std::{error::Error, fmt};

/// A malformed or unsupported vector tile.
#[derive(Debug)]
pub enum DecodeError {
    UnexpectedEnd,
    InvalidVarint,
    InvalidProtobuf(&'static str),
    InvalidUtf8(std::str::Utf8Error),
    MissingField(&'static str),
    UnsupportedVersion(u32),
    InvalidValue(&'static str),
    InvalidTagIndex { table: &'static str, index: u32 },
    InvalidGeometry(&'static str),
    CoordinateOverflow,
    TileTooLarge,
    Mlt(mlt_core::MltError),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd => f.write_str("unexpected end of tile"),
            Self::InvalidVarint => f.write_str("invalid protobuf varint"),
            Self::InvalidProtobuf(reason) => write!(f, "invalid protobuf: {reason}"),
            Self::InvalidUtf8(error) => write!(f, "invalid tile string: {error}"),
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported MVT version: {version}"),
            Self::InvalidValue(reason) => write!(f, "invalid tile value: {reason}"),
            Self::InvalidTagIndex { table, index } => {
                write!(f, "property {table} index {index} is out of bounds")
            }
            Self::InvalidGeometry(reason) => write!(f, "invalid geometry: {reason}"),
            Self::CoordinateOverflow => f.write_str("tile coordinate exceeds i32"),
            Self::TileTooLarge => f.write_str("tile exceeds u32 offsets"),
            Self::Mlt(error) => write!(f, "invalid MLT: {error}"),
        }
    }
}

impl Error for DecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidUtf8(error) => Some(error),
            Self::Mlt(error) => Some(error),
            _ => None,
        }
    }
}

impl From<mlt_core::MltError> for DecodeError {
    fn from(error: mlt_core::MltError) -> Self {
        Self::Mlt(error)
    }
}

// Twice the signed area. i128 also covers full-range i32 coordinates without
// overflow when summing a ring with up to u32::MAX vertices.
pub(crate) fn signed_area(points: impl Iterator<Item = [i32; 2]>) -> i128 {
    let mut points = points;
    let Some(first) = points.next() else {
        return 0;
    };
    let mut previous = first;
    let mut area = 0;
    for point in points.chain(std::iter::once(first)) {
        area += i128::from(previous[0]) * i128::from(point[1])
            - i128::from(point[0]) * i128::from(previous[1]);
        previous = point;
    }
    area
}

/// Finishes a decoded layer with `columns` besides the builder's.
pub(crate) fn finish_layer_with(
    builder: crate::LayerBuilder,
    columns: Vec<(String, crate::Column)>,
) -> crate::Layer {
    let mut layer = builder.finish_with(columns);
    // Property order on the wire is not semantic. MLT shared dictionaries may
    // regroup columns; use one stable order for equal TileFrames across formats.
    layer.properties.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    layer
}

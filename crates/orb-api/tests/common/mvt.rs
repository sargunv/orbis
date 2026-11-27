//! Mapbox Vector Tiles built by hand, for tests that serve their own.

/// A feature of a tile's layer.
pub(crate) struct Feature<'a> {
    /// The geometry type: 1 point, 2 line, 3 polygon.
    pub(crate) kind: u64,
    /// Its geometry's commands, as [`path`] writes them.
    pub(crate) geometry: Vec<u64>,
    /// Its properties: pairs of indices into the layer's keys and values.
    pub(crate) tags: &'a [u64],
}

/// A vector tile of one layer, `name`, extent 4096, of `features` with
/// `keys` and string `values`.
pub(crate) fn tile(name: &str, features: &[Feature], keys: &[&str], values: &[&str]) -> Vec<u8> {
    let mut layer = Vec::new();
    uint(15, 2, &mut layer);
    message(1, name.as_bytes(), &mut layer);
    for feature in features {
        let mut out = Vec::new();
        if !feature.tags.is_empty() {
            message(2, &packed(feature.tags), &mut out);
        }
        uint(3, feature.kind, &mut out);
        message(4, &packed(&feature.geometry), &mut out);
        message(2, &out, &mut layer);
    }
    for key in keys {
        message(3, key.as_bytes(), &mut layer);
    }
    for value in values {
        let mut out = Vec::new();
        message(1, value.as_bytes(), &mut out);
        message(4, &out, &mut layer);
    }
    uint(5, 4096, &mut layer);
    let mut tile = Vec::new();
    message(3, &layer, &mut tile);
    tile
}

/// A line's or, `closed`, a ring's geometry commands through `points` in
/// tile coordinates: a MoveTo, a LineTo through the rest, a ClosePath.
pub(crate) fn path(points: &[[i64; 2]], closed: bool) -> Vec<u64> {
    let zigzag = |n: i64| ((n << 1) ^ (n >> 63)) as u64;
    let [x, y] = points[0];
    let mut commands = vec![9, zigzag(x), zigzag(y), 2 | (points.len() as u64 - 1) << 3];
    for pair in points.windows(2) {
        commands.push(zigzag(pair[1][0] - pair[0][0]));
        commands.push(zigzag(pair[1][1] - pair[0][1]));
    }
    if closed {
        commands.push(15);
    }
    commands
}

fn packed(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    values.iter().for_each(|&v| varint(v, &mut out));
    out
}

fn varint(mut n: u64, out: &mut Vec<u8>) {
    while n >= 128 {
        out.push(n as u8 | 128);
        n >>= 7;
    }
    out.push(n as u8);
}

/// A length-delimited protobuf field.
fn message(field: u64, bytes: &[u8], out: &mut Vec<u8>) {
    varint(field << 3 | 2, out);
    varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

fn uint(field: u64, n: u64, out: &mut Vec<u8>) {
    varint(field << 3, out);
    varint(n, out);
}

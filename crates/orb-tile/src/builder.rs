use std::collections::HashMap;

use crate::{Bitmap, Column, ColumnData, GeometryType, Layer, Value};

/// Builds a [`Layer`] one feature at a time.
///
/// Call [`begin_feature`](Self::begin_feature), then for each part
/// [`begin_part`](Self::begin_part), for each ring
/// [`begin_ring`](Self::begin_ring) and its vertices. Properties may be set
/// any time while the feature is current. Column types are inferred when the
/// layer is finished: a key whose values share one type gets that type,
/// numbers of different types become `F64`, anything else is `Mixed`.
#[derive(Debug)]
pub struct LayerBuilder {
    name: String,
    extent: u32,
    /// Per column, its values by feature, in feature order.
    ids: Vec<(u32, Value)>,
    types: Vec<GeometryType>,
    feature_parts: Vec<u32>,
    part_rings: Vec<u32>,
    ring_vertices: Vec<u32>,
    vertices: Vec<[i32; 2]>,
    keys: HashMap<String, usize>,
    columns: Vec<(String, Vec<(u32, Value)>)>,
    clip: Option<i32>,
}

impl LayerBuilder {
    pub fn new(name: impl Into<String>, extent: u32) -> Self {
        Self {
            name: name.into(),
            extent,
            ids: Vec::new(),
            types: Vec::new(),
            feature_parts: Vec::new(),
            part_rings: Vec::new(),
            ring_vertices: Vec::new(),
            vertices: Vec::new(),
            keys: HashMap::new(),
            columns: Vec::new(),
            clip: None,
        }
    }

    pub fn begin_feature(&mut self, id: Option<u64>, geometry_type: GeometryType) {
        self.types.push(geometry_type);
        self.feature_parts.push(len_u32(self.part_rings.len()));
        if let Some(id) = id {
            self.set_id(Value::U64(id));
        }
    }

    /// Sets the current feature's id, including string ids from GeoJSON.
    pub fn set_id(&mut self, id: Value) {
        let feature = self.feature().expect("set_id outside a feature");
        set(&mut self.ids, feature, id);
    }

    fn feature(&self) -> Option<u32> {
        self.types.len().checked_sub(1).map(len_u32)
    }

    pub fn begin_part(&mut self) {
        assert!(!self.types.is_empty(), "begin_part outside a feature");
        self.part_rings.push(len_u32(self.ring_vertices.len()));
    }

    pub fn begin_ring(&mut self) {
        assert!(!self.part_rings.is_empty(), "begin_ring outside a part");
        self.ring_vertices.push(len_u32(self.vertices.len()));
    }

    pub fn push_vertex(&mut self, vertex: [i32; 2]) {
        debug_assert!(!self.ring_vertices.is_empty(), "push_vertex outside a ring");
        self.vertices.push(vertex);
    }

    /// Sets a property of the current feature, replacing any earlier value
    /// for the same key.
    pub fn set_property(&mut self, key: &str, value: Value) {
        let feature = self.feature().expect("set_property outside a feature");
        let index = match self.keys.get(key) {
            Some(&index) => index,
            None => {
                self.keys.insert(key.to_string(), self.columns.len());
                self.columns.push((key.to_string(), Vec::new()));
                self.columns.len() - 1
            }
        };
        set(&mut self.columns[index].1, feature, value);
    }

    /// Discards the current feature and everything added to it.
    pub(crate) fn abandon_feature(&mut self) {
        let Some(part_start) = self.feature_parts.pop() else {
            return;
        };
        let feature = self.feature().expect("a feature");
        self.types.pop();
        let ring_start = self.part_rings.get(part_start as usize).copied();
        self.part_rings.truncate(part_start as usize);
        if let Some(ring_start) = ring_start {
            let vertex_start = self.ring_vertices.get(ring_start as usize).copied();
            self.ring_vertices.truncate(ring_start as usize);
            if let Some(vertex_start) = vertex_start {
                self.vertices.truncate(vertex_start as usize);
            }
        }
        for values in
            std::iter::once(&mut self.ids).chain(self.columns.iter_mut().map(|c| &mut c.1))
        {
            if values.last().is_some_and(|(f, _)| *f == feature) {
                values.pop();
            }
        }
    }

    /// The layer's geometry was clipped `buffer` data units past the tile
    /// square ([`Layer::clip_buffer`]).
    pub(crate) fn clipped_at(&mut self, buffer: i32) -> &mut Self {
        self.clip = Some(buffer);
        self
    }

    pub fn finish(mut self) -> Layer {
        let len = self.types.len();
        self.feature_parts.push(len_u32(self.part_rings.len()));
        self.part_rings.push(len_u32(self.ring_vertices.len()));
        self.ring_vertices.push(len_u32(self.vertices.len()));

        let ids = typed_column(len, self.ids);
        let properties = self
            .columns
            .into_iter()
            .map(|(key, values)| (key, typed_column(len, values)))
            .collect();

        Layer {
            name: self.name,
            extent: self.extent,
            ids,
            types: self.types,
            feature_parts: self.feature_parts,
            part_rings: self.part_rings,
            ring_vertices: self.ring_vertices,
            vertices: self.vertices,
            properties,
            clip: self.clip,
        }
    }
}

fn len_u32(len: usize) -> u32 {
    u32::try_from(len).expect("layer exceeds u32 offsets")
}

/// Sets `feature`'s value in `values`, replacing one set before.
fn set(values: &mut Vec<(u32, Value)>, feature: u32, value: Value) {
    match values.last_mut() {
        Some((f, v)) if *f == feature => *v = value,
        _ => values.push((feature, value)),
    }
}

fn typed_column(len: usize, values: Vec<(u32, Value)>) -> Column {
    #[derive(PartialEq)]
    enum Kind {
        Bool,
        I64,
        U64,
        F64,
        Number,
        Str,
        Mixed,
    }
    let mut kind = None;
    for (_, value) in &values {
        let k = match value {
            Value::Bool(_) => Kind::Bool,
            Value::I64(_) => Kind::I64,
            Value::U64(_) => Kind::U64,
            Value::F64(_) => Kind::F64,
            Value::Str(_) => Kind::Str,
            Value::Json(_) => Kind::Mixed,
        };
        kind = Some(match kind {
            None => k,
            Some(prev) if prev == k => prev,
            Some(Kind::I64 | Kind::U64 | Kind::F64 | Kind::Number)
                if matches!(k, Kind::I64 | Kind::U64 | Kind::F64) =>
            {
                Kind::Number
            }
            Some(_) => Kind::Mixed,
        });
    }

    let mut present = Bitmap::new();
    let mut next = values.iter().map(|(f, _)| *f as usize).peekable();
    for feature in 0..len {
        present.push(next.next_if_eq(&feature).is_some());
    }
    let values = values.into_iter().map(|(_, v)| v);
    let data = match kind.unwrap_or(Kind::Bool) {
        Kind::Bool => ColumnData::Bool(values.map(|v| v == Value::Bool(true)).collect()),
        Kind::I64 => ColumnData::I64(
            values
                .map(|v| match v {
                    Value::I64(v) => v,
                    _ => unreachable!("an i64 column"),
                })
                .collect(),
        ),
        Kind::U64 => ColumnData::U64(
            values
                .map(|v| match v {
                    Value::U64(v) => v,
                    _ => unreachable!("a u64 column"),
                })
                .collect(),
        ),
        Kind::F64 | Kind::Number => ColumnData::F64(
            values
                .map(|v| match v {
                    Value::I64(v) => v as f64,
                    Value::U64(v) => v as f64,
                    Value::F64(v) => v,
                    _ => unreachable!("a number column"),
                })
                .collect(),
        ),
        Kind::Str => {
            let mut dictionary = Vec::new();
            let mut lookup: HashMap<String, u32> = HashMap::new();
            let codes = values
                .map(|v| match v {
                    Value::Str(s) => *lookup.entry(s).or_insert_with_key(|s| {
                        dictionary.push(s.clone());
                        len_u32(dictionary.len() - 1)
                    }),
                    _ => unreachable!("a string column"),
                })
                .collect();
            ColumnData::Str { dictionary, codes }
        }
        Kind::Mixed => ColumnData::Mixed(values.collect()),
    };
    Column::new(present, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ValueRef;

    fn sample() -> Layer {
        let mut b = LayerBuilder::new("roads", 4096);
        b.begin_feature(Some(7), GeometryType::LineString);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([0, 0]);
        b.push_vertex([10, 0]);
        b.set_property("class", Value::Str("primary".into()));
        b.set_property("lanes", Value::I64(2));

        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for v in [[0, 0], [10, 0], [10, 10]] {
            b.push_vertex(v);
        }
        b.begin_ring();
        for v in [[2, 2], [2, 4], [4, 4]] {
            b.push_vertex(v);
        }
        b.set_property("class", Value::Str("park".into()));
        b.set_property("lanes", Value::F64(0.5));

        b.begin_feature(Some(9), GeometryType::Point);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([5, 5]);
        b.set_property("class", Value::Str("primary".into()));
        b.set_property("flag", Value::Bool(true));
        b.finish()
    }

    #[test]
    fn geometry_offsets() {
        let layer = sample();
        assert_eq!(layer.len(), 3);
        let rings: Vec<Vec<&[[i32; 2]]>> = layer
            .geometry(1)
            .parts()
            .map(|p| p.rings().collect())
            .collect();
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 2);
        assert_eq!(rings[0][1][1], [2, 4]);
        let point: Vec<_> = layer
            .geometry(2)
            .parts()
            .flat_map(|p| p.rings().collect::<Vec<_>>())
            .collect();
        assert_eq!(point, vec![&[[5, 5]][..]]);
    }

    #[test]
    fn column_types() {
        let layer = sample();
        let class = layer.property("class").unwrap();
        assert!(
            matches!(class.data(), ColumnData::Str { dictionary, .. } if dictionary.len() == 2)
        );
        assert_eq!(class.get(2), Some(ValueRef::Str("primary")));
        let lanes = layer.property("lanes").unwrap();
        assert!(matches!(lanes.data(), ColumnData::F64(_)));
        assert_eq!(lanes.get(0), Some(ValueRef::F64(2.0)));
        assert_eq!(lanes.get(2), None);
        let flag = layer.property("flag").unwrap();
        assert_eq!(flag.get(0), None);
        assert_eq!(flag.get(2), Some(ValueRef::Bool(true)));
        assert_eq!(layer.ids().get(1), None);
        assert_eq!(layer.ids().get(2), Some(ValueRef::U64(9)));
    }

    #[test]
    fn abandon_feature_rolls_back() {
        let mut b = LayerBuilder::new("x", 4096);
        b.begin_feature(Some(1), GeometryType::Point);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([1, 1]);
        b.begin_feature(Some(2), GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([2, 2]);
        b.set_property("k", Value::Bool(true));
        b.abandon_feature();
        let layer = b.finish();
        assert_eq!(layer.len(), 1);
        assert_eq!(layer.vertices, &[[1, 1]]);
        assert_eq!(layer.property("k").unwrap().get(0), None);
    }
}

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
    ids: Vec<Option<Value>>,
    types: Vec<GeometryType>,
    feature_parts: Vec<u32>,
    part_rings: Vec<u32>,
    ring_vertices: Vec<u32>,
    vertices: Vec<[i32; 2]>,
    keys: HashMap<String, usize>,
    columns: Vec<(String, Vec<Option<Value>>)>,
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
        }
    }

    pub fn begin_feature(&mut self, id: Option<u64>, geometry_type: GeometryType) {
        self.ids.push(id.map(Value::U64));
        self.types.push(geometry_type);
        self.feature_parts.push(len_u32(self.part_rings.len()));
    }

    /// Sets the current feature's id, including string ids from GeoJSON.
    pub fn set_id(&mut self, id: Value) {
        *self.ids.last_mut().expect("set_id outside a feature") = Some(id);
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
        let feature = self
            .types
            .len()
            .checked_sub(1)
            .expect("set_property outside a feature");
        let index = match self.keys.get(key) {
            Some(&index) => index,
            None => {
                self.keys.insert(key.to_string(), self.columns.len());
                self.columns.push((key.to_string(), Vec::new()));
                self.columns.len() - 1
            }
        };
        let values = &mut self.columns[index].1;
        values.resize(feature + 1, None);
        values[feature] = Some(value);
    }

    /// Discards the current feature and everything added to it.
    pub fn abandon_feature(&mut self) {
        let Some(part_start) = self.feature_parts.pop() else {
            return;
        };
        let feature = self.types.len() - 1;
        self.ids.pop();
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
        for (_, values) in &mut self.columns {
            values.truncate(feature);
        }
    }

    pub fn finish(mut self) -> Layer {
        let len = self.types.len();
        self.feature_parts.push(len_u32(self.part_rings.len()));
        self.part_rings.push(len_u32(self.ring_vertices.len()));
        self.ring_vertices.push(len_u32(self.vertices.len()));

        let ids = typed_column(self.ids);
        let properties = self
            .columns
            .into_iter()
            .map(|(key, mut values)| {
                values.resize(len, None);
                (key, typed_column(values))
            })
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
        }
    }
}

fn len_u32(len: usize) -> u32 {
    u32::try_from(len).expect("layer exceeds u32 offsets")
}

fn typed_column(values: Vec<Option<Value>>) -> Column {
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
    for value in values.iter().flatten() {
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
    for value in &values {
        present.push(value.is_some());
    }
    let data = match kind.unwrap_or(Kind::Bool) {
        Kind::Bool => ColumnData::Bool(
            values
                .into_iter()
                .map(|v| matches!(v, Some(Value::Bool(true))))
                .collect(),
        ),
        Kind::I64 => ColumnData::I64(
            values
                .into_iter()
                .map(|v| match v {
                    Some(Value::I64(v)) => v,
                    _ => 0,
                })
                .collect(),
        ),
        Kind::U64 => ColumnData::U64(
            values
                .into_iter()
                .map(|v| match v {
                    Some(Value::U64(v)) => v,
                    _ => 0,
                })
                .collect(),
        ),
        Kind::F64 | Kind::Number => ColumnData::F64(
            values
                .into_iter()
                .map(|v| match v {
                    Some(Value::I64(v)) => v as f64,
                    Some(Value::U64(v)) => v as f64,
                    Some(Value::F64(v)) => v,
                    _ => 0.0,
                })
                .collect(),
        ),
        Kind::Str => {
            let mut dictionary = Vec::new();
            let mut lookup: HashMap<String, u32> = HashMap::new();
            let codes = values
                .into_iter()
                .map(|v| match v {
                    Some(Value::Str(s)) => *lookup.entry(s).or_insert_with_key(|s| {
                        dictionary.push(s.clone());
                        len_u32(dictionary.len() - 1)
                    }),
                    _ => 0,
                })
                .collect();
            ColumnData::Str { dictionary, codes }
        }
        Kind::Mixed => ColumnData::Mixed(
            values
                .into_iter()
                .map(|v| v.unwrap_or(Value::Bool(false)))
                .collect(),
        ),
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
        assert_eq!(layer.vertices(), &[[1, 1]]);
        assert_eq!(layer.property("k").unwrap().get(0), None);
    }
}

use rustc_hash::FxHashMap;

use crate::{Bitmap, Column, ColumnData, GeometryType, Layer, Value, ValueRef};

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
    keys: FxHashMap<String, usize>,
    columns: Vec<(String, Vec<(u32, Value)>)>,
    clip: Option<i32>,
    whole_reach: Option<i32>,
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
            keys: FxHashMap::default(),
            columns: Vec::new(),
            clip: None,
            whole_reach: None,
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

    /// Makes room for `vertices` more.
    pub fn reserve(&mut self, vertices: usize) {
        self.vertices.reserve(vertices);
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

    /// Polygons keep whole as far past the square as in an overzoom
    /// slice: `reach` ([`Layer::whole_reach`]).
    pub(crate) fn whole_within(&mut self, reach: i32) -> &mut Self {
        self.whole_reach = Some(reach);
        self
    }

    /// Clipped as `layer` was, and keeping polygons whole as far past the
    /// square: for a layer of `layer`'s features or their parts.
    pub fn clipped_like(&mut self, layer: &Layer) -> &mut Self {
        self.clip = Some(layer.buffer);
        self.whole_reach = layer.whole_reach;
        self
    }

    /// Puts a layer wider than `max` units a side in a grid of `extent`,
    /// its vertices rounded as GL JS's `loadGeometry` rounds them.
    pub(crate) fn fit_extent(&mut self, max: u32, extent: u32) {
        if self.extent <= max {
            return;
        }
        let scale = f64::from(extent) / f64::from(self.extent);
        for v in &mut self.vertices {
            *v = v.map(|c| crate::js_round(f64::from(c) * scale) as i32);
        }
        self.extent = extent;
    }

    /// The features begun so far.
    pub(crate) fn features(&self) -> usize {
        self.types.len()
    }

    pub fn finish(self) -> Layer {
        self.finish_with(Vec::new())
    }

    /// Finishes a layer of features of `from`, its `rows`, one per feature
    /// begun: their ids and properties are those rows' of `from`, and none
    /// set here.
    pub fn finish_selecting(self, from: &Layer, rows: &[u32]) -> Layer {
        assert_eq!(rows.len(), self.types.len(), "one row per feature");
        let mut layer = self.finish();
        layer.ids = from.ids.select(rows);
        layer.properties = (from.properties.iter())
            .map(|(key, column)| (key, column.select(rows)))
            .filter(|(_, column)| !column.is_absent())
            .map(|(key, column)| (key.clone(), column))
            .collect();
        layer
    }

    /// Finishes the layer with `coded` columns besides those set through
    /// [`set_property`](Self::set_property) ([`CodedColumns`]).
    pub(crate) fn finish_with(mut self, coded: Vec<(String, Column)>) -> Layer {
        let len = self.types.len();
        self.feature_parts.push(len_u32(self.part_rings.len()));
        self.part_rings.push(len_u32(self.ring_vertices.len()));
        self.ring_vertices.push(len_u32(self.vertices.len()));

        let ids = typed_column(len, &borrowed(&self.ids));
        let properties = (self.columns.into_iter())
            .map(|(key, values)| (key, typed_column(len, &borrowed(&values))))
            .chain(coded)
            .collect();
        let buffer = (self.clip).unwrap_or_else(|| crate::reach(&self.vertices, self.extent));

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
            buffer,
            whole_reach: self.whole_reach,
            labels: None,
            data_features: None,
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

/// A column's type: one all its values share, numbers of different types
/// as `Number` (`F64`), anything else `Mixed`.
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

fn kind<'a>(values: impl Iterator<Item = ValueRef<'a>>) -> Kind {
    let mut kind = None;
    for value in values {
        let k = match value {
            ValueRef::Bool(_) => Kind::Bool,
            ValueRef::I64(_) => Kind::I64,
            ValueRef::U64(_) => Kind::U64,
            ValueRef::F64(_) => Kind::F64,
            ValueRef::Str(_) => Kind::Str,
            ValueRef::Json(_) => Kind::Mixed,
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
    kind.unwrap_or(Kind::Bool)
}

/// Which of `len` features have a value, from those that do, ascending.
fn presence(len: usize, features: impl Iterator<Item = u32>) -> Bitmap {
    Bitmap::from_rows(len, features.map(|f| f as usize))
}

fn number(value: ValueRef<'_>) -> f64 {
    match value {
        ValueRef::I64(v) => v as f64,
        ValueRef::U64(v) => v as f64,
        ValueRef::F64(v) => v,
        _ => unreachable!("a number column"),
    }
}

/// A column of `len` rows from its present rows' values, by row, typed
/// as [`LayerBuilder`] says. Equal strings share a dictionary code.
pub(crate) fn typed_column(len: usize, values: &[(u32, ValueRef<'_>)]) -> Column {
    let present = presence(len, values.iter().map(|(f, _)| *f));
    let values = values.iter().map(|&(_, v)| v);
    let data = match kind(values.clone()) {
        Kind::Bool => ColumnData::Bool(values.map(|v| v == ValueRef::Bool(true)).collect()),
        Kind::I64 => ColumnData::I64(
            values
                .map(|v| match v {
                    ValueRef::I64(v) => v,
                    _ => unreachable!("an i64 column"),
                })
                .collect(),
        ),
        Kind::U64 => ColumnData::U64(
            values
                .map(|v| match v {
                    ValueRef::U64(v) => v,
                    _ => unreachable!("a u64 column"),
                })
                .collect(),
        ),
        Kind::F64 | Kind::Number => ColumnData::F64(values.map(number).collect()),
        Kind::Str => {
            let mut dictionary = Vec::new();
            let mut lookup: FxHashMap<&str, u32> = FxHashMap::default();
            let codes = values
                .map(|v| match v {
                    ValueRef::Str(s) => *lookup.entry(s).or_insert_with(|| {
                        dictionary.push(s.to_owned());
                        len_u32(dictionary.len() - 1)
                    }),
                    _ => unreachable!("a string column"),
                })
                .collect();
            ColumnData::Str { dictionary, codes }
        }
        Kind::Mixed => ColumnData::Mixed(values.map(ValueRef::to_owned).collect()),
    };
    Column::new(present, data)
}

/// `values` borrowed, for [`typed_column`].
fn borrowed(values: &[(u32, Value)]) -> Vec<(u32, ValueRef<'_>)> {
    values.iter().map(|(f, v)| (*f, v.as_ref())).collect()
}

/// Property columns whose values index a table of values, as an MVT
/// layer's tags index its keys and values: filled feature by feature
/// without hashing keys or copying values, and typed once the layer is
/// done, as [`LayerBuilder::set_property`]'s columns are.
pub(crate) struct CodedColumns<'a> {
    keys: &'a [&'a str],
    values: &'a [Value],
    /// Per key, the first key equal to it: a layer may name a key twice.
    key_of: Vec<u32>,
    /// Per key, its values: (feature, value index), by feature.
    columns: Vec<Vec<(u32, u32)>>,
}

impl<'a> CodedColumns<'a> {
    pub(crate) fn new(keys: &'a [&'a str], values: &'a [Value]) -> Self {
        let mut first: FxHashMap<&str, u32> = FxHashMap::default();
        let key_of = (keys.iter().enumerate())
            .map(|(i, k)| *first.entry(k).or_insert(len_u32(i)))
            .collect();
        Self {
            keys,
            values,
            key_of,
            columns: vec![Vec::new(); keys.len()],
        }
    }

    /// Sets `feature`'s value of key `key` to value `value`, replacing one
    /// set before. Features come in order.
    pub(crate) fn set(&mut self, feature: u32, key: u32, value: u32) {
        let column = &mut self.columns[self.key_of[key as usize] as usize];
        match column.last_mut() {
            Some((f, v)) if *f == feature => *v = value,
            _ => column.push((feature, value)),
        }
    }

    /// The columns, for `len` features.
    pub(crate) fn finish(self, len: usize) -> Vec<(String, Column)> {
        let values = self.values;
        // Per value, its code in the column being typed. A string's code is
        // its value's in the table: encoders write each value once, and a
        // table holding one twice gives it two codes, which only keeps
        // filters from sharing an answer between them (`Column::code`).
        let mut codes = vec![u32::MAX; values.len()];
        (self.columns.into_iter().enumerate())
            .filter(|(_, column)| !column.is_empty())
            .map(|(key, column)| {
                let row = |&(_, v): &(u32, u32)| &values[v as usize];
                let present = presence(len, column.iter().map(|(f, _)| *f));
                let data = match kind(column.iter().map(|e| row(e).as_ref())) {
                    Kind::Bool => ColumnData::Bool(
                        column
                            .iter()
                            .map(|e| *row(e) == Value::Bool(true))
                            .collect(),
                    ),
                    Kind::I64 => ColumnData::I64(
                        (column.iter().map(row))
                            .map(|v| match v {
                                Value::I64(v) => *v,
                                _ => unreachable!("an i64 column"),
                            })
                            .collect(),
                    ),
                    Kind::U64 => ColumnData::U64(
                        (column.iter().map(row))
                            .map(|v| match v {
                                Value::U64(v) => *v,
                                _ => unreachable!("a u64 column"),
                            })
                            .collect(),
                    ),
                    Kind::F64 | Kind::Number => {
                        ColumnData::F64(column.iter().map(|e| number(row(e).as_ref())).collect())
                    }
                    Kind::Str => {
                        let mut dictionary = Vec::new();
                        let column_codes = (column.iter())
                            .map(|&(_, v)| {
                                let v = v as usize;
                                if codes[v] == u32::MAX {
                                    codes[v] = len_u32(dictionary.len());
                                    let Value::Str(s) = &values[v] else {
                                        unreachable!("a string column")
                                    };
                                    dictionary.push(s.clone());
                                }
                                codes[v]
                            })
                            .collect();
                        for &(_, v) in &column {
                            codes[v as usize] = u32::MAX;
                        }
                        ColumnData::Str {
                            dictionary,
                            codes: column_codes,
                        }
                    }
                    Kind::Mixed => {
                        ColumnData::Mixed(column.iter().map(|e| row(e).clone()).collect())
                    }
                };
                (self.keys[key].to_string(), Column::new(present, data))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn geometry_slices_parts() {
        let mut b = LayerBuilder::new("x", 4096);
        b.begin_feature(None, GeometryType::Polygon);
        for i in 0..5 {
            b.begin_part();
            b.begin_ring();
            for v in [[i, 0], [i + 1, 0], [i + 1, 1]] {
                b.push_vertex(v);
            }
        }
        let layer = b.finish();
        let firsts: Vec<i32> = (layer.geometry(0).slice(1..4).parts())
            .map(|p| p.rings().next().unwrap()[0][0])
            .collect();
        assert_eq!(firsts, [1, 2, 3]);
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

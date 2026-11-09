/// A value for some of a layer's features: a bitmap of which have one, and
/// their values in feature order. Most keys of a layer are on few of its
/// features, so absent rows take one bit.
#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    present: Bitmap,
    /// Per bitmap word, the values before it.
    ranks: Vec<u32>,
    data: ColumnData,
}

/// A column's present values, typed for the whole column.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnData {
    Bool(Vec<bool>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    F64(Vec<f64>),
    /// Strings as codes into a dictionary of distinct values.
    Str {
        dictionary: Vec<String>,
        codes: Vec<u32>,
    },
    /// Values of more than one type.
    Mixed(Vec<Value>),
}

/// An owned property value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Str(String),
    /// A nested array or object (a GeoJSON property), as JSON text.
    Json(String),
}

/// A borrowed property value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValueRef<'a> {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Str(&'a str),
    /// A nested array or object, as JSON text.
    Json(&'a str),
}

impl Value {
    pub fn as_ref(&self) -> ValueRef<'_> {
        match self {
            Self::Bool(v) => ValueRef::Bool(*v),
            Self::I64(v) => ValueRef::I64(*v),
            Self::U64(v) => ValueRef::U64(*v),
            Self::F64(v) => ValueRef::F64(*v),
            Self::Str(v) => ValueRef::Str(v),
            Self::Json(v) => ValueRef::Json(v),
        }
    }
}

impl ValueRef<'_> {
    pub fn to_owned(self) -> Value {
        match self {
            Self::Bool(v) => Value::Bool(v),
            Self::I64(v) => Value::I64(v),
            Self::U64(v) => Value::U64(v),
            Self::F64(v) => Value::F64(v),
            Self::Str(v) => Value::Str(v.into()),
            Self::Json(v) => Value::Json(v.into()),
        }
    }
}

impl Column {
    /// A column from its presence bitmap and its present values.
    pub fn new(present: Bitmap, data: ColumnData) -> Self {
        let mut ranks = Vec::with_capacity(present.words.len());
        let mut count = 0;
        for word in &present.words {
            ranks.push(count);
            count += word.count_ones();
        }
        let column = Self {
            present,
            ranks,
            data,
        };
        assert_eq!(count as usize, column.data_len(), "column length mismatch");
        column
    }

    pub fn len(&self) -> usize {
        self.present.len()
    }

    pub fn is_empty(&self) -> bool {
        self.present.len() == 0
    }

    pub fn data(&self) -> &ColumnData {
        &self.data
    }

    /// The value at `row`, or `None` if the feature has none.
    pub fn get(&self, row: usize) -> Option<ValueRef<'_>> {
        let row = self.rank(row)?;
        Some(match &self.data {
            ColumnData::Bool(v) => ValueRef::Bool(v[row]),
            ColumnData::I64(v) => ValueRef::I64(v[row]),
            ColumnData::U64(v) => ValueRef::U64(v[row]),
            ColumnData::F64(v) => ValueRef::F64(v[row]),
            ColumnData::Str { dictionary, codes } => {
                ValueRef::Str(&dictionary[codes[row] as usize])
            }
            ColumnData::Mixed(v) => v[row].as_ref(),
        })
    }

    /// A number standing for the value at `row`, or `None` if the feature
    /// has none: rows whose codes are equal have equal values. A string's
    /// is its dictionary code; a column of mixed types gives each row its
    /// own.
    pub fn code(&self, row: usize) -> Option<u64> {
        let row = self.rank(row)?;
        Some(match &self.data {
            ColumnData::Bool(v) => u64::from(v[row]),
            ColumnData::I64(v) => v[row] as u64,
            ColumnData::U64(v) => v[row],
            ColumnData::F64(v) => v[row].to_bits(),
            ColumnData::Str { codes, .. } => u64::from(codes[row]),
            ColumnData::Mixed(_) => row as u64,
        })
    }

    /// The column of rows `rows` (in any order, repeats allowed), as a
    /// layer of those features has it: a string column keeps only the
    /// dictionary entries they use.
    pub fn select(&self, rows: &[u32]) -> Self {
        let ranks: Vec<Option<usize>> = rows.iter().map(|&r| self.rank(r as usize)).collect();
        let present = Bitmap::from_rows(rows.len(), (ranks.iter().enumerate()).filter_map(|(i, r)| r.map(|_| i)));
        let kept = || ranks.iter().flatten().copied();
        let data = match &self.data {
            ColumnData::Bool(v) => ColumnData::Bool(kept().map(|i| v[i]).collect()),
            ColumnData::I64(v) => ColumnData::I64(kept().map(|i| v[i]).collect()),
            ColumnData::U64(v) => ColumnData::U64(kept().map(|i| v[i]).collect()),
            ColumnData::F64(v) => ColumnData::F64(kept().map(|i| v[i]).collect()),
            ColumnData::Str { dictionary, codes } => {
                let mut remap = vec![u32::MAX; dictionary.len()];
                let mut kept_dictionary = Vec::new();
                let codes = kept()
                    .map(|i| {
                        let code = &mut remap[codes[i] as usize];
                        if *code == u32::MAX {
                            *code = kept_dictionary.len() as u32;
                            kept_dictionary.push(dictionary[codes[i] as usize].clone());
                        }
                        *code
                    })
                    .collect();
                ColumnData::Str {
                    dictionary: kept_dictionary,
                    codes,
                }
            }
            ColumnData::Mixed(v) => ColumnData::Mixed(kept().map(|i| v[i].clone()).collect()),
        };
        Self::new(present, data)
    }

    /// Whether no row has a value.
    pub(crate) fn is_absent(&self) -> bool {
        self.data_len() == 0
    }

    /// Where `row`'s value is among the present ones, if it has one.
    fn rank(&self, row: usize) -> Option<usize> {
        if !self.present.get(row) {
            return None;
        }
        let below = self.present.words[row / 64] & ((1 << (row % 64)) - 1);
        Some((self.ranks[row / 64] + below.count_ones()) as usize)
    }

    /// Heap bytes the column holds.
    pub fn bytes(&self) -> usize {
        let text = |s: &String| size_of::<String>() + s.len();
        self.present.words.len() * 12
            + match &self.data {
                ColumnData::Bool(v) => v.len(),
                ColumnData::I64(v) => v.len() * 8,
                ColumnData::U64(v) => v.len() * 8,
                ColumnData::F64(v) => v.len() * 8,
                ColumnData::Str { dictionary, codes } => {
                    codes.len() * 4 + dictionary.iter().map(text).sum::<usize>()
                }
                ColumnData::Mixed(v) => v
                    .iter()
                    .map(|v| match v {
                        Value::Str(s) | Value::Json(s) => text(s),
                        _ => size_of::<Value>(),
                    })
                    .sum(),
            }
    }

    fn data_len(&self) -> usize {
        match &self.data {
            ColumnData::Bool(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::U64(v) => v.len(),
            ColumnData::F64(v) => v.len(),
            ColumnData::Str { codes, .. } => codes.len(),
            ColumnData::Mixed(v) => v.len(),
        }
    }
}

/// A packed bit vector.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bitmap {
    words: Vec<u64>,
    len: usize,
}

impl Bitmap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `len` bits, set at `rows`, ascending.
    pub(crate) fn from_rows(len: usize, rows: impl Iterator<Item = usize>) -> Self {
        let mut words = vec![0u64; len.div_ceil(64)];
        for row in rows {
            words[row / 64] |= 1 << (row % 64);
        }
        Self { words, len }
    }

    pub fn push(&mut self, bit: bool) {
        if self.len.is_multiple_of(64) {
            self.words.push(0);
        }
        if bit {
            self.words[self.len / 64] |= 1 << (self.len % 64);
        }
        self.len += 1;
    }

    pub fn get(&self, i: usize) -> bool {
        assert!(
            i < self.len,
            "bit {i} out of range for bitmap of {}",
            self.len
        );
        self.words[i / 64] & (1 << (i % 64)) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitmap_push_and_get() {
        let mut b = Bitmap::new();
        for i in 0..130 {
            b.push(i % 3 == 0);
        }
        assert_eq!(b.len(), 130);
        assert!((0..130).all(|i| b.get(i) == (i % 3 == 0)));
        let rows = Bitmap::from_rows(130, (0..130).filter(|i| i % 3 == 0));
        assert_eq!(rows, b);
    }

    #[test]
    fn select_keeps_rows_and_the_strings_they_use() {
        // 100 rows, every third with a value, cycling through 3 strings.
        let present = Bitmap::from_rows(100, (0..100).step_by(3));
        let dictionary = ["a", "b", "c"].map(String::from).to_vec();
        let codes = (0..34).map(|i| i % 3).collect();
        let column = Column::new(present, ColumnData::Str { dictionary, codes });
        // Rows 3 ("b") twice, 4 (none), 99 ("a"), 70 (none).
        let selected = column.select(&[3, 3, 4, 99, 70]);
        assert_eq!(selected.len(), 5);
        let values: Vec<_> = (0..5).map(|r| selected.get(r)).collect();
        let (b, a) = (Some(ValueRef::Str("b")), Some(ValueRef::Str("a")));
        assert_eq!(values, [b, b, None, a, None]);
        let ColumnData::Str { dictionary, .. } = selected.data() else {
            panic!("a string column")
        };
        assert_eq!(dictionary, &["b", "a"]);
        assert!(column.select(&[1, 2]).is_absent());
    }
}

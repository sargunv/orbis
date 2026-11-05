/// One value per feature, with a bitmap of which features have one.
#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    present: Bitmap,
    data: ColumnData,
}

/// A column's values, typed for the whole column. Rows that are absent hold
/// an unspecified placeholder.
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
    /// A column from its data and presence bitmap, which must be the same
    /// length.
    pub fn new(present: Bitmap, data: ColumnData) -> Self {
        let column = Self { present, data };
        assert_eq!(
            column.present.len(),
            column.data_len(),
            "column length mismatch"
        );
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
        if !self.present.get(row) {
            return None;
        }
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
    }
}

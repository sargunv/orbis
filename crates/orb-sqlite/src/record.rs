//! Varints and records: how SQLite encodes a row or an index entry.

use std::cmp::Ordering;

use crate::{Error, Result};

/// A value in a record.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    /// UTF-8 text, as the file holds it (not checked).
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

impl Value {
    pub fn as_integer(&self) -> Option<i64> {
        match *self {
            Value::Integer(i) => Some(i),
            _ => None,
        }
    }

    /// An integer or a real, as a real.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::Integer(i) => Some(i as f64),
            Value::Real(r) => Some(r),
            _ => None,
        }
    }

    /// Text, if it is valid UTF-8.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(text) => std::str::from_utf8(text).ok(),
            _ => None,
        }
    }

    /// The bytes of a blob or of text.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Blob(bytes) | Value::Text(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Takes the bytes of a blob or of text, leaving null.
    pub fn take_bytes(&mut self) -> Option<Vec<u8>> {
        match std::mem::replace(self, Value::Null) {
            Value::Blob(bytes) | Value::Text(bytes) => Some(bytes),
            other => {
                *self = other;
                None
            }
        }
    }

    fn class(&self) -> u8 {
        match self {
            Value::Null => 0,
            Value::Integer(_) | Value::Real(_) => 1,
            Value::Text(_) => 2,
            Value::Blob(_) => 3,
        }
    }

    /// SQLite's order of values with the binary collation: null, then
    /// numbers by value, then text and blobs by their bytes.
    pub fn compare(&self, other: &Value) -> Ordering {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
            (Value::Integer(a), Value::Real(b)) => compare_int_real(*a, *b),
            (Value::Real(a), Value::Integer(b)) => compare_int_real(*b, *a).reverse(),
            (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Value::Text(a), Value::Text(b)) | (Value::Blob(a), Value::Blob(b)) => a.cmp(b),
            _ => self.class().cmp(&other.class()),
        }
    }
}

fn compare_int_real(int: i64, real: f64) -> Ordering {
    if real.is_nan() {
        // SQLite stores NaN as null; a NaN read is less than everything.
        return Ordering::Greater;
    }
    // Exact for every integer an f64 holds, and in order past them.
    match (int as f64).partial_cmp(&real) {
        Some(Ordering::Equal) => {
            if real >= 9.223_372_036_854_776e18 {
                Ordering::Less
            } else {
                int.cmp(&(real as i64))
            }
        }
        Some(order) => order,
        None => Ordering::Equal,
    }
}

/// Reads a varint at `at`, moving `at` past it.
pub(crate) fn varint(bytes: &[u8], at: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for i in 0..9 {
        let byte = *bytes
            .get(*at)
            .ok_or_else(|| Error::corrupt("a varint past its page"))?;
        *at += 1;
        if i == 8 {
            return Ok((value << 8) | u64::from(byte));
        }
        value = (value << 7) | u64::from(byte & 0x7f);
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    unreachable!("the ninth byte returns")
}

/// The values of a record: its header's serial types, then their bodies.
/// Values past `limit` are left out.
pub(crate) fn decode(payload: &[u8], limit: usize) -> Result<Vec<Value>> {
    let mut at = 0;
    let header_len = varint(payload, &mut at)?;
    let header_len = usize::try_from(header_len)
        .ok()
        .filter(|&len| len >= at && len <= payload.len())
        .ok_or_else(|| Error::corrupt("a record's header past its end"))?;
    let mut body = header_len;
    let mut values = Vec::new();
    while at < header_len && values.len() < limit {
        let serial = varint(&payload[..header_len], &mut at)?;
        let size = match serial {
            0 | 8 | 9 => 0,
            1..=4 => serial as usize,
            5 => 6,
            6 | 7 => 8,
            10 | 11 => return Err(Error::corrupt("a reserved serial type")),
            n => usize::try_from((n - 12) / 2).map_err(|_| Error::corrupt("an oversized value"))?,
        };
        let bytes = body
            .checked_add(size)
            .and_then(|end| payload.get(body..end))
            .ok_or_else(|| Error::corrupt("a value past its record"))?;
        body += size;
        let int = || {
            let mut value = i64::from(bytes[0] as i8);
            for &b in &bytes[1..] {
                value = (value << 8) | i64::from(b);
            }
            value
        };
        values.push(match serial {
            0 => Value::Null,
            1..=6 => Value::Integer(int()),
            7 => Value::Real(f64::from_bits(u64::from_be_bytes(
                bytes.try_into().expect("eight bytes"),
            ))),
            8 => Value::Integer(0),
            9 => Value::Integer(1),
            n if n % 2 == 0 => Value::Blob(bytes.to_vec()),
            _ => Value::Text(bytes.to_vec()),
        });
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_read_as_sqlite_writes_them() {
        let read = |bytes: &[u8]| {
            let mut at = 0;
            (varint(bytes, &mut at).unwrap(), at)
        };
        assert_eq!(read(&[0x05]), (5, 1));
        assert_eq!(read(&[0x81, 0x00]), (128, 2));
        assert_eq!(read(&[0xff; 9]), (u64::MAX, 9));
        assert!(varint(&[0x80], &mut 0).is_err());
    }

    #[test]
    fn records_decode_every_serial_type() {
        // Header: its length, then null, a 1-byte int, a 2-byte int, a
        // real, 0, 1, a 2-byte blob and 1-byte text.
        let payload = [
            9, 0, 1, 2, 7, 8, 9, 16, 15,   // header
            0xff, // -1
            0x01, 0x00, // 256
            0x3f, 0xf0, 0, 0, 0, 0, 0, 0, // 1.0
            0xab, 0xcd, b'a',
        ];
        assert_eq!(
            decode(&payload, usize::MAX).unwrap(),
            [
                Value::Null,
                Value::Integer(-1),
                Value::Integer(256),
                Value::Real(1.0),
                Value::Integer(0),
                Value::Integer(1),
                Value::Blob(vec![0xab, 0xcd]),
                Value::Text(b"a".to_vec()),
            ]
        );
        assert_eq!(decode(&payload, 2).unwrap().len(), 2);
        assert!(decode(&payload[..12], usize::MAX).is_err());
    }

    #[test]
    fn values_order_as_sqlite_orders_them() {
        use Value::*;
        let ordered = [
            Null,
            Integer(-3),
            Real(-2.5),
            Integer(1),
            Real(1.5),
            Text(b"a".to_vec()),
            Text(b"b".to_vec()),
            Blob(vec![0]),
        ];
        for (i, a) in ordered.iter().enumerate() {
            for (j, b) in ordered.iter().enumerate() {
                assert_eq!(a.compare(b), i.cmp(&j), "{a:?} against {b:?}");
            }
        }
        assert_eq!(Integer(2).compare(&Real(2.0)), Ordering::Equal);
    }
}

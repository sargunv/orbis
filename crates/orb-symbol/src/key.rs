//! Stable symbol keys `(source-layer, id | promoteId, text)`.
//!
//! GL JS keys a symbol by `murmur3(text)`, mixed with the promoted id when
//! the source has one, truncated to 16 bits. Orbis keeps 64 bits and the
//! source layer. The hash is FNV-1a over a tagged encoding: the same on
//! every platform and worker.

use orb_tile::Value;

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

struct Fnv(u64);

impl Fnv {
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(PRIME);
        }
    }

    /// A length-prefixed field, so fields never run into each other.
    fn field(&mut self, tag: u8, bytes: &[u8]) {
        self.bytes(&[tag]);
        self.bytes(&(bytes.len() as u64).to_le_bytes());
        self.bytes(bytes);
    }
}

/// The key of a symbol with label `text` (empty for icons alone), of a
/// feature of `source_layer` with stable `id`, if the source has one.
pub(crate) fn stable_key(source_layer: &str, id: Option<&Value>, text: &str) -> u64 {
    let mut h = Fnv(OFFSET);
    h.field(0, source_layer.as_bytes());
    match id {
        None => h.field(1, &[]),
        Some(Value::Str(s)) => h.field(2, s.as_bytes()),
        Some(Value::Json(s)) => h.field(8, s.as_bytes()),
        Some(Value::Bool(b)) => h.field(3, &[u8::from(*b)]),
        Some(&Value::I64(v)) => h.field(4, &v.to_le_bytes()),
        // Numbers equal as JavaScript numbers get equal keys.
        Some(&Value::U64(v)) => match i64::try_from(v) {
            Ok(v) => h.field(4, &v.to_le_bytes()),
            Err(_) => h.field(5, &v.to_le_bytes()),
        },
        Some(&Value::F64(v)) => {
            if v.fract() == 0.0 && v >= -(2f64.powi(63)) && v < 2f64.powi(63) {
                h.field(4, &(v as i64).to_le_bytes());
            } else if v.fract() == 0.0 && v >= 0.0 && v < 2f64.powi(64) {
                h.field(5, &(v as u64).to_le_bytes());
            } else {
                h.field(6, &v.to_bits().to_le_bytes());
            }
        }
    }
    h.field(7, text.as_bytes());
    h.0
}

//! The one codec for messages between a map's workers on the web, which
//! share no memory: style updates, sprite icons, tile requests and
//! replies, placement messages and glyph bitmaps. Mesh and table streams
//! travel beside a message, raw, so they can be transferred.
//!
//! Both ends are the same build, so a message is only its values, in the
//! order its type declares them ([`Wire`], derived): integers as LEB128
//! varints (signed ones zigzagged), floats as their little-endian bytes,
//! lengths before sequences, a tag byte before an `Option`, an enum's
//! variant index before its fields. Bytes that don't decode are a bug in
//! the build, not input to recover from: [`decode`] says only that they
//! don't.
//!
//! It replaced serde with postcard, whose generic visitors were 79 KiB of
//! core.wasm (20 KiB brotli).

#![forbid(unsafe_code)]

extern crate self as orb_wire;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{BuildHasher, Hash};
use std::sync::Arc;

pub use orb_wire_derive::Wire;

/// A value a message carries.
pub trait Wire: Sized {
    /// Appends its bytes to `out`.
    fn put(&self, out: &mut Vec<u8>);
    /// Reads one from the front of `input`, advancing past it.
    fn take(input: &mut &[u8]) -> Option<Self>;
}

/// A message's bytes.
pub fn encode<T: Wire>(message: &T) -> Vec<u8> {
    let mut out = Vec::new();
    message.put(&mut out);
    out
}

/// A message from its bytes, all of them.
pub fn decode<T: Wire>(mut bytes: &[u8]) -> Result<T, String> {
    T::take(&mut bytes)
        .filter(|_| bytes.is_empty())
        .ok_or_else(|| "malformed message".to_string())
}

/// The first `n` bytes of `input`, advancing past them.
fn bytes<'a>(input: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, rest) = input.split_at_checked(n)?;
    *input = rest;
    Some(head)
}

fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn take_varint(input: &mut &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let (&byte, rest) = input.split_first()?;
        *input = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Some(value);
        }
    }
    None
}

/// A sequence's length, no more than the bytes left could hold, so that a
/// length never reserves more than its message.
fn take_len(input: &mut &[u8]) -> Option<usize> {
    usize::try_from(take_varint(input)?)
        .ok()
        .filter(|&len| len <= input.len())
}

impl Wire for u8 {
    fn put(&self, out: &mut Vec<u8>) {
        out.push(*self);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let (&byte, rest) = input.split_first()?;
        *input = rest;
        Some(byte)
    }
}

impl Wire for i8 {
    fn put(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        u8::take(input).map(|b| b as i8)
    }
}

impl Wire for bool {
    fn put(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        match u8::take(input)? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

// `as` widens: no `From` takes `usize` to `u64`, nor `isize` to `i64`.
macro_rules! unsigned {
    ($($t:ty),*) => {$(
        impl Wire for $t {
            #[allow(clippy::cast_lossless)]
            fn put(&self, out: &mut Vec<u8>) {
                put_varint(*self as u64, out);
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                <$t>::try_from(take_varint(input)?).ok()
            }
        }
    )*};
}
unsigned!(u16, u32, u64, usize);

macro_rules! signed {
    ($($t:ty),*) => {$(
        impl Wire for $t {
            #[allow(clippy::cast_lossless)]
            fn put(&self, out: &mut Vec<u8>) {
                let value = *self as i64;
                put_varint(((value << 1) ^ (value >> 63)) as u64, out);
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                let zigzag = take_varint(input)?;
                let value = (zigzag >> 1) as i64 ^ -((zigzag & 1) as i64);
                <$t>::try_from(value).ok()
            }
        }
    )*};
}
signed!(i16, i32, i64, isize);

macro_rules! float {
    ($($t:ty),*) => {$(
        impl Wire for $t {
            fn put(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                let bytes = bytes(input, size_of::<$t>())?;
                Some(<$t>::from_le_bytes(bytes.try_into().ok()?))
            }
        }
    )*};
}
float!(f32, f64);

impl Wire for char {
    fn put(&self, out: &mut Vec<u8>) {
        u32::from(*self).put(out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        char::from_u32(u32::take(input)?)
    }
}

impl Wire for () {
    fn put(&self, _: &mut Vec<u8>) {}
    fn take(_: &mut &[u8]) -> Option<Self> {
        Some(())
    }
}

/// Puts `text`, as a `String` would.
pub fn put_str(text: &str, out: &mut Vec<u8>) {
    text.len().put(out);
    out.extend_from_slice(text.as_bytes());
}

/// Takes a `String`'s text, borrowed from `input`.
pub fn take_str<'a>(input: &mut &'a [u8]) -> Option<&'a str> {
    let len = take_len(input)?;
    std::str::from_utf8(bytes(input, len)?).ok()
}

impl Wire for String {
    fn put(&self, out: &mut Vec<u8>) {
        put_str(self, out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        take_str(input).map(String::from)
    }
}

impl Wire for Box<str> {
    fn put(&self, out: &mut Vec<u8>) {
        put_str(self, out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        take_str(input).map(Box::from)
    }
}

impl Wire for Arc<str> {
    fn put(&self, out: &mut Vec<u8>) {
        put_str(self, out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        take_str(input).map(Arc::from)
    }
}

impl<T: Wire> Wire for Box<T> {
    fn put(&self, out: &mut Vec<u8>) {
        (**self).put(out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        T::take(input).map(Box::new)
    }
}

impl<T: Wire> Wire for Arc<T> {
    fn put(&self, out: &mut Vec<u8>) {
        (**self).put(out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        T::take(input).map(Arc::new)
    }
}

impl<T: Wire> Wire for Option<T> {
    fn put(&self, out: &mut Vec<u8>) {
        match self {
            None => out.push(0),
            Some(value) => {
                out.push(1);
                value.put(out);
            }
        }
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        match u8::take(input)? {
            0 => Some(None),
            1 => T::take(input).map(Some),
            _ => None,
        }
    }
}

/// Puts a sequence of `len` items.
fn put_seq<'a, T: Wire + 'a>(
    len: usize,
    items: impl IntoIterator<Item = &'a T>,
    out: &mut Vec<u8>,
) {
    len.put(out);
    for item in items {
        item.put(out);
    }
}

impl<T: Wire> Wire for Vec<T> {
    fn put(&self, out: &mut Vec<u8>) {
        put_seq(self.len(), self, out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let len = take_len(input)?;
        let mut items = Vec::with_capacity(len);
        for _ in 0..len {
            items.push(T::take(input)?);
        }
        Some(items)
    }
}

impl<T: Wire> Wire for Box<[T]> {
    fn put(&self, out: &mut Vec<u8>) {
        put_seq(self.len(), self.iter(), out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        Vec::take(input).map(Vec::into_boxed_slice)
    }
}

impl<T: Wire> Wire for Arc<[T]> {
    fn put(&self, out: &mut Vec<u8>) {
        put_seq(self.len(), self.iter(), out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        Vec::take(input).map(Arc::from)
    }
}

impl<T: Wire, const N: usize> Wire for [T; N] {
    fn put(&self, out: &mut Vec<u8>) {
        for item in self {
            item.put(out);
        }
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let mut failed = false;
        let items: [Option<T>; N] = std::array::from_fn(|_| {
            let item = T::take(input).filter(|_| !failed);
            failed |= item.is_none();
            item
        });
        if failed {
            return None;
        }
        Some(items.map(|item| item.expect("every item taken")))
    }
}

impl<K: Wire + Eq + Hash, V: Wire, S: BuildHasher + Default> Wire for HashMap<K, V, S> {
    fn put(&self, out: &mut Vec<u8>) {
        self.len().put(out);
        for (key, value) in self {
            key.put(out);
            value.put(out);
        }
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let len = take_len(input)?;
        let mut map = HashMap::with_capacity_and_hasher(len, S::default());
        for _ in 0..len {
            map.insert(K::take(input)?, V::take(input)?);
        }
        Some(map)
    }
}

impl<T: Wire + Eq + Hash, S: BuildHasher + Default> Wire for HashSet<T, S> {
    fn put(&self, out: &mut Vec<u8>) {
        put_seq(self.len(), self, out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let len = take_len(input)?;
        let mut set = HashSet::with_capacity_and_hasher(len, S::default());
        for _ in 0..len {
            set.insert(T::take(input)?);
        }
        Some(set)
    }
}

impl<K: Wire + Ord, V: Wire> Wire for BTreeMap<K, V> {
    fn put(&self, out: &mut Vec<u8>) {
        self.len().put(out);
        for (key, value) in self {
            key.put(out);
            value.put(out);
        }
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let len = take_len(input)?;
        let mut map = BTreeMap::new();
        for _ in 0..len {
            map.insert(K::take(input)?, V::take(input)?);
        }
        Some(map)
    }
}

impl<T: Wire, E: Wire> Wire for Result<T, E> {
    fn put(&self, out: &mut Vec<u8>) {
        match self {
            Ok(value) => {
                out.push(0);
                value.put(out);
            }
            Err(error) => {
                out.push(1);
                error.put(out);
            }
        }
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        match u8::take(input)? {
            0 => T::take(input).map(Ok),
            1 => E::take(input).map(Err),
            _ => None,
        }
    }
}

impl<T: Wire> Wire for std::ops::Range<T> {
    fn put(&self, out: &mut Vec<u8>) {
        self.start.put(out);
        self.end.put(out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        Some(T::take(input)?..T::take(input)?)
    }
}

macro_rules! tuple {
    ($($name:ident)+) => {
        impl<$($name: Wire),+> Wire for ($($name,)+) {
            #[allow(non_snake_case)]
            fn put(&self, out: &mut Vec<u8>) {
                let ($($name,)+) = self;
                $($name.put(out);)+
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                Some(($($name::take(input)?,)+))
            }
        }
    };
}
tuple!(A);
tuple!(A B);
tuple!(A B C);
tuple!(A B C D);
tuple!(A B C D E);
tuple!(A B C D E F);

macro_rules! glam {
    ($($t:ty: $array:ty, $to:ident, $from:ident;)*) => {$(
        impl Wire for $t {
            fn put(&self, out: &mut Vec<u8>) {
                self.$to().put(out);
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                <$array>::take(input).map(|array| Self::$from(&array))
            }
        }
    )*};
}
glam! {
    glam::DVec2: [f64; 2], to_array, from_slice;
    glam::DVec3: [f64; 3], to_array, from_slice;
    glam::DMat4: [f64; 16], to_cols_array, from_cols_slice;
    glam::DAffine2: [f64; 6], to_cols_array, from_cols_slice;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Default, Wire)]
    struct Named {
        a: u32,
        b: Option<String>,
        #[wire(skip)]
        c: Vec<u8>,
        d: [i16; 3],
    }

    #[derive(Debug, PartialEq, Wire)]
    struct Tuple(f64, bool);

    #[derive(Debug, PartialEq, Wire)]
    struct Generic<V> {
        items: Vec<V>,
    }

    #[derive(Debug, PartialEq, Wire)]
    enum Shape {
        Unit,
        Pair(i64, char),
        Named { at: glam::DVec2, tuple: Tuple },
    }

    fn round_trip<T: Wire + PartialEq + std::fmt::Debug>(value: T) {
        assert_eq!(decode::<T>(&encode(&value)).unwrap(), value);
    }

    #[test]
    fn values_round_trip() {
        round_trip(Named {
            a: 300,
            b: Some("é".into()),
            c: Vec::new(),
            d: [-1, 0, i16::MAX],
        });
        round_trip(Generic {
            items: vec![Shape::Unit, Shape::Pair(i64::MIN, '∑')],
        });
        round_trip(Shape::Named {
            at: glam::DVec2::new(1.5, -0.0),
            tuple: Tuple(f64::MAX, true),
        });
        round_trip((u64::MAX, -5i32, usize::MAX, 0u16));
        round_trip(HashMap::<u32, Vec<u8>>::from([
            (1, vec![2, 3]),
            (4, vec![]),
        ]));
        round_trip(glam::DMat4::from_cols_array(&std::array::from_fn(|i| {
            i as f64
        })));
    }

    #[test]
    fn a_skipped_field_arrives_as_its_default() {
        let sent = Named {
            c: vec![1, 2, 3],
            ..Named::default()
        };
        assert_eq!(decode::<Named>(&encode(&sent)).unwrap().c, Vec::<u8>::new());
    }

    #[test]
    fn integers_are_varints() {
        assert_eq!(encode(&127u32), [127]);
        assert_eq!(encode(&128u32), [0x80, 1]);
        assert_eq!(encode(&-1i32), [1]);
        assert_eq!(encode(&1i32), [2]);
    }

    #[test]
    fn bytes_that_dont_decode_say_so() {
        // Too short, too long, a bad tag, a length past the end, an
        // integer that doesn't fit.
        assert!(decode::<u64>(&[0x80]).is_err());
        assert!(decode::<u8>(&[1, 2]).is_err());
        assert!(decode::<Option<u8>>(&[2, 0]).is_err());
        assert!(decode::<Shape>(&[3]).is_err());
        assert!(decode::<Vec<u8>>(&[5, 1]).is_err());
        assert!(decode::<u16>(&encode(&70_000u32)).is_err());
        assert!(decode::<String>(&[1, 0xff]).is_err());
    }
}

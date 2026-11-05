//! Gzip (RFC 1952) bodies: one member, inflated by `miniz_oxide` and checked
//! against its CRC-32 and length. Like browsers' `DecompressionStream`, bytes
//! after the member are an error.

/// The inflated body of the single gzip member `bytes`, at most `limit`
/// bytes long.
pub(crate) fn decode(bytes: &[u8], limit: usize) -> Result<Vec<u8>, &'static str> {
    const FHCRC: u8 = 1 << 1;
    const FEXTRA: u8 = 1 << 2;
    const FNAME: u8 = 1 << 3;
    const FCOMMENT: u8 = 1 << 4;
    const TRAILER: usize = 8;

    let [0x1f, 0x8b, 8, flags, _, _, _, _, _, _, rest @ ..] = bytes else {
        return Err("not a deflate gzip member");
    };
    if flags & !(FHCRC | FEXTRA | FNAME | FCOMMENT | 1) != 0 {
        return Err("reserved gzip flags");
    }
    let mut rest = rest;
    if flags & FEXTRA != 0 {
        let [lo, hi, tail @ ..] = rest else {
            return Err("truncated gzip header");
        };
        let len = u16::from_le_bytes([*lo, *hi]) as usize;
        rest = tail.get(len..).ok_or("truncated gzip header")?;
    }
    for flag in [FNAME, FCOMMENT] {
        if flags & flag != 0 {
            let end = rest
                .iter()
                .position(|&b| b == 0)
                .ok_or("truncated gzip header")?;
            rest = &rest[end + 1..];
        }
    }
    if flags & FHCRC != 0 {
        rest = rest.get(2..).ok_or("truncated gzip header")?;
    }
    let split = rest
        .len()
        .checked_sub(TRAILER)
        .ok_or("truncated gzip member")?;
    let (deflate, trailer) = rest.split_at(split);
    let body = miniz_oxide::inflate::decompress_to_vec_with_limit(deflate, limit)
        .map_err(|_| "invalid or oversized gzip body")?;
    let crc = u32::from_le_bytes(trailer[..4].try_into().expect("four bytes"));
    let len = u32::from_le_bytes(trailer[4..].try_into().expect("four bytes"));
    if crc32(&body) != crc || body.len() as u32 != len {
        return Err("gzip body fails its CRC-32 or length check");
    }
    Ok(body)
}

/// CRC-32 (ISO-HDLC, gzip's), a byte at a time.
/// `bytes` as one gzip member.
pub(crate) fn encode(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255];
    out.extend(miniz_oxide::deflate::compress_to_vec(bytes, 6));
    out.extend(crc32(bytes).to_le_bytes());
    out.extend((bytes.len() as u32).to_le_bytes());
    out
}

fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };
    !bytes.iter().fold(!0u32, |c, &b| {
        TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8)
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn gzip(body: &[u8], builder: flate2::GzBuilder) -> Vec<u8> {
        let mut encoder = builder.write(Vec::new(), flate2::Compression::default());
        encoder.write_all(body).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn decodes_members_with_optional_header_fields() {
        let body: Vec<u8> = (0..10_000u32)
            .flat_map(|i| (i % 251).to_le_bytes())
            .collect();
        for builder in [
            flate2::GzBuilder::new(),
            flate2::GzBuilder::new()
                .filename("tile.mvt")
                .comment("c")
                .extra(vec![1, 2, 3]),
        ] {
            assert_eq!(decode(&gzip(&body, builder), body.len()).unwrap(), body);
        }
        assert_eq!(
            decode(&gzip(b"", flate2::GzBuilder::new()), 0).unwrap(),
            b""
        );
    }

    #[test]
    fn rejects_corrupt_oversized_and_trailing_input() {
        let body = b"orbis orbis orbis orbis".repeat(40);
        let good = gzip(&body, flate2::GzBuilder::new());
        assert!(decode(&good, body.len() - 1).is_err());
        let mut crc = good.clone();
        let at = crc.len() - 8;
        crc[at] ^= 1;
        assert!(decode(&crc, body.len()).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(decode(&trailing, body.len()).is_err());
        for len in 0..good.len() {
            assert!(decode(&good[..len], body.len()).is_err());
        }
    }

    #[test]
    fn encodes_what_it_decodes() {
        let body = b"the quick brown fox jumps over the lazy dog".repeat(20);
        assert_eq!(decode(&encode(&body), body.len()).unwrap(), body);
    }

    #[test]
    fn crc32_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }
}

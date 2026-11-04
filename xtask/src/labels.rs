//! `cargo xtask demo-labels`: the Tokyo demo's labels, offline. Liberty's
//! symbol layers go into the committed styles; its sprite and the glyphs the
//! extract's names use (a subset of each glyph range) go beside them.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

use orb_crs::TileIndex;
use orb_source::pmtiles::Reader;
use orb_text::TextTransform;
use serde_json::value::RawValue;

const OUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/demo/data");
const LIBERTY: &str = "https://tiles.openfreemap.org/styles/liberty";
const FONTS: &str = "https://tiles.openfreemap.org/fonts";
const SPRITE: &str = "https://tiles.openfreemap.org/sprites/ofm_f384/ofm";
/// The demo style's URLs; hosts resolve `./data/` as they do for tiles.
const GLYPHS_URL: &str = "./data/glyphs/{fontstack}/{range}.pbf";
const SPRITE_URL: &str = "./data/sprites/ofm";

pub(crate) fn generate() -> Result<(), String> {
    let agent = ureq::Agent::new_with_defaults();
    let get = |url: &str| -> Result<Vec<u8>, String> {
        agent
            .get(url)
            .call()
            .map_err(|e| format!("{url}: {e}"))?
            .body_mut()
            .with_config()
            .limit(16 << 20)
            .read_to_vec()
            .map_err(|e| format!("{url}: {e}"))
    };

    let liberty = String::from_utf8(get(LIBERTY)?).map_err(|e| e.to_string())?;
    let liberty = layers(&liberty)?;

    // Per font stack, the source layers it labels.
    let mut fonts: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for raw in &liberty {
        let layer = parse(raw)?;
        if let Some(stack) = layer.get("layout").and_then(|l| l.get("text-font")) {
            let stack = stack
                .as_array()
                .ok_or("text-font is a literal array")?
                .iter()
                .map(|f| f.as_str().unwrap_or_default())
                .collect::<Vec<_>>()
                .join(",");
            let source_layer = layer["source-layer"].as_str().unwrap_or_default();
            fonts.entry(stack).or_default().insert(source_layer.into());
        }
    }

    for encoding in ["mvt", "mlt"] {
        let path = format!("{OUT}/tokyo-{encoding}.json");
        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        std::fs::write(&path, splice(&text, &liberty)?).map_err(|e| e.to_string())?;
    }

    // Glyphs: per font and touched range, the range's glyphs its layers'
    // names use.
    let by_layer = code_points()?;
    let mut bytes = 0;
    let mut count = 0;
    for (stack, source_layers) in &fonts {
        let mut code_points: BTreeSet<u32> = (0x20..0x7f).collect();
        for layer in source_layers {
            code_points.extend(by_layer.get(layer).into_iter().flatten());
        }
        count += code_points.len();
        let dir = format!("{OUT}/glyphs/{stack}");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let ranges: BTreeSet<u32> = code_points.iter().map(|c| c / 256).collect();
        for range in ranges {
            let name = format!("{}-{}", range * 256, range * 256 + 255);
            let url = format!("{FONTS}/{}/{name}.pbf", stack.replace(' ', "%20"));
            let subset = subset_glyphs(&get(&url)?, &code_points)?;
            bytes += subset.len();
            std::fs::write(format!("{dir}/{name}.pbf"), subset).map_err(|e| e.to_string())?;
        }
    }

    let sprites = format!("{OUT}/sprites");
    std::fs::create_dir_all(&sprites).map_err(|e| e.to_string())?;
    for file in ["ofm.json", "ofm.png", "ofm@2x.json", "ofm@2x.png"] {
        let body = get(&format!("{}/{file}", SPRITE.rsplit_once('/').unwrap().0))?;
        bytes += body.len();
        std::fs::write(format!("{sprites}/{file}"), body).map_err(|e| e.to_string())?;
    }
    eprintln!(
        "demo labels: {count} code points in {} fonts, {bytes} bytes of glyphs and sprites",
        fonts.len()
    );
    Ok(())
}

/// The committed style with Liberty's symbol layers, each after the layer
/// Liberty draws before it, and the glyph and sprite URLs.
fn splice(style: &str, liberty: &[&RawValue]) -> Result<String, String> {
    let top: HashMap<&str, &RawValue> = serde_json::from_str(style).map_err(|e| e.to_string())?;
    let field = |key: &str| top.get(key).copied().ok_or(format!("style has no {key}"));
    let id = |raw: &RawValue| -> Result<(String, String), String> {
        let v = parse(raw)?;
        Ok((
            v["id"].as_str().unwrap_or_default().to_string(),
            v["type"].as_str().unwrap_or_default().to_string(),
        ))
    };
    // The committed layers without symbols, then Liberty's symbols where
    // Liberty has them.
    let mut kept = Vec::new();
    for raw in &layers(style)? {
        let (id, kind) = id(raw)?;
        if kind != "symbol" {
            kept.push((id, raw.get().to_string()));
        }
    }
    let mut out_layers: Vec<String> = Vec::new();
    let mut next = kept.iter().peekable();
    for raw in liberty {
        let (id, kind) = id(raw)?;
        if kind == "symbol" {
            out_layers.push(pretty(raw.get(), 2));
        } else if next.peek().is_some_and(|(k, _)| *k == id) {
            out_layers.push(next.next().expect("peeked").1.clone());
        }
    }
    if let Some((id, _)) = next.next() {
        return Err(format!("layer {id} is not in Liberty's order"));
    }
    let mut out = String::from("{\n  \"version\": 8,\n");
    writeln!(out, "  \"glyphs\": \"{GLYPHS_URL}\",").unwrap();
    writeln!(out, "  \"sprite\": \"{SPRITE_URL}\",").unwrap();
    writeln!(out, "  \"sources\": {},", field("sources")?.get()).unwrap();
    writeln!(
        out,
        "  \"layers\": [\n    {}\n  ],",
        out_layers.join(",\n    ")
    )
    .unwrap();
    writeln!(out, "  \"center\": {},", field("center")?.get()).unwrap();
    writeln!(out, "  \"zoom\": {}\n}}", field("zoom")?.get()).unwrap();
    Ok(out)
}

/// A style's layers, as written.
fn layers(style: &str) -> Result<Vec<&RawValue>, String> {
    let top: HashMap<&str, &RawValue> = serde_json::from_str(style).map_err(|e| e.to_string())?;
    let layers = top.get("layers").ok_or("style has no layers")?;
    serde_json::from_str(layers.get()).map_err(|e| e.to_string())
}

fn parse(raw: &RawValue) -> Result<serde_json::Value, String> {
    serde_json::from_str(raw.get()).map_err(|e| e.to_string())
}

/// JSON pretty-printed as the committed styles are: two spaces, every
/// non-empty array and object expanded, `indent` levels deep. Keys keep
/// their order.
fn pretty(json: &str, indent: usize) -> String {
    let newline = |out: &mut String, depth: usize| {
        out.push('\n');
        out.push_str(&"  ".repeat(depth));
    };
    let mut out = String::new();
    let mut depth = indent;
    // Whitespace outside strings only separates tokens: drop it.
    let mut source = json.chars().peekable();
    while let Some(c) = source.next() {
        match c {
            '"' => {
                out.push(c);
                let mut escaped = false;
                for c in source.by_ref() {
                    out.push(c);
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == '"' {
                        break;
                    }
                }
            }
            c if c.is_whitespace() => {}
            '{' | '[' => {
                out.push(c);
                while source.next_if(|c| c.is_whitespace()).is_some() {}
                if let Some(close) = source.next_if(|&n| n == '}' || n == ']') {
                    out.push(close);
                } else {
                    depth += 1;
                    newline(&mut out, depth);
                }
            }
            '}' | ']' => {
                depth -= 1;
                newline(&mut out, depth);
                out.push(c);
            }
            ',' => {
                out.push(c);
                newline(&mut out, depth);
            }
            ':' => out.push_str(": "),
            _ => out.push(c),
        }
    }
    out
}

/// Per source layer, the code points of the extract's names and refs that
/// glyph PBFs draw, with their upper case (Liberty upper-cases some
/// labels). Complex scripts, and ideographs under the default
/// `localIdeographFontFamily`, draw from local fonts.
fn code_points() -> Result<HashMap<String, BTreeSet<u32>>, String> {
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{OUT}/manifest.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let archive = std::fs::read(format!("{OUT}/tokyo-mvt.pmtiles")).map_err(|e| e.to_string())?;
    let mut reader = Reader::default();
    let mut by_layer: HashMap<String, BTreeSet<u32>> = HashMap::new();
    for coordinate in manifest["tiles"].as_array().ok_or("manifest tiles")? {
        let tile = TileIndex::new(
            coordinate[0].as_u64().unwrap() as u8,
            coordinate[1].as_u64().unwrap() as u32,
            coordinate[2].as_u64().unwrap() as u32,
        );
        let bytes = crate::pmtiles::local_tile(&mut reader, &archive, tile)?;
        let frame = orb_tile::mvt::decode(&bytes).map_err(|e| e.to_string())?;
        for layer in &frame.layers {
            let points = by_layer.entry(layer.name().to_string()).or_default();
            for (key, column) in layer.properties() {
                if !(key.starts_with("name") || key == "ref") {
                    continue;
                }
                for row in 0..column.len() {
                    if let Some(orb_tile::ValueRef::Str(s)) = column.get(row) {
                        // As layout transforms text (Arabic shaping too).
                        for transform in [TextTransform::None, TextTransform::Uppercase] {
                            let text = orb_text::transform_text(s, transform);
                            // Glyph ranges stop at U+FFFF.
                            points.extend(
                                text.chars()
                                    .filter(|&c| {
                                        !orb_text::unicode::is_complex_shaping(c)
                                            && !orb_text::unicode::uses_local_ideograph_font(c)
                                    })
                                    .map(|c| c as u32)
                                    .filter(|&c| c <= 0xffff),
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(by_layer)
}

/// A glyph PBF keeping only the glyphs of `keep`: `glyphs { stacks {
/// name = 1; range = 2; glyphs = 3 } }`, each glyph's id field 1.
fn subset_glyphs(pbf: &[u8], keep: &BTreeSet<u32>) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for (field, value) in fields(pbf)? {
        let Field::Bytes(stack) = value else {
            return Err("glyph PBF: unexpected field".into());
        };
        if field != 1 {
            continue;
        }
        let mut kept = Vec::new();
        for (field, value) in fields(stack)? {
            let Field::Bytes(bytes) = value else {
                return Err("glyph PBF: unexpected stack field".into());
            };
            let keep_it = field != 3
                || fields(bytes)?.iter().any(|(f, v)| {
                    *f == 1 && matches!(v, Field::Varint(id) if keep.contains(&(*id as u32)))
                });
            if keep_it {
                put(&mut kept, field, bytes);
            }
        }
        put(&mut out, 1, &kept);
    }
    Ok(out)
}

enum Field<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

fn fields(mut bytes: &[u8]) -> Result<Vec<(u64, Field<'_>)>, String> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        let key = varint(&mut bytes)?;
        let value = match key & 7 {
            0 => Field::Varint(varint(&mut bytes)?),
            2 => {
                let len = varint(&mut bytes)? as usize;
                if len > bytes.len() {
                    return Err("glyph PBF: truncated".into());
                }
                let (value, rest) = bytes.split_at(len);
                bytes = rest;
                Field::Bytes(value)
            }
            wire => return Err(format!("glyph PBF: wire type {wire}")),
        };
        out.push((key >> 3, value));
    }
    Ok(out)
}

fn varint(bytes: &mut &[u8]) -> Result<u64, String> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let (&byte, rest) = bytes.split_first().ok_or("glyph PBF: truncated varint")?;
        *bytes = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("glyph PBF: varint too long".into())
}

fn put(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    for mut n in [field << 3 | 2, bytes.len() as u64] {
        while n >= 0x80 {
            out.push((n & 0x7f) as u8 | 0x80);
            n >>= 7;
        }
        out.push(n as u8);
    }
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty_matches_the_committed_layout() {
        assert_eq!(
            pretty(r#"{"id":"a b","f":["==",["get","x"],1],"e":{}}"#, 1),
            "{\n    \"id\": \"a b\",\n    \"f\": [\n      \"==\",\n      [\n        \"get\",\n        \"x\"\n      ],\n      1\n    ],\n    \"e\": {}\n  }"
        );
    }

    #[test]
    fn subsets_keep_wanted_glyphs() {
        let glyph = |id: u8| {
            let mut g = vec![0x08, id];
            g.extend([0x10, 3]);
            g
        };
        let mut stack = Vec::new();
        put(&mut stack, 1, b"Font");
        put(&mut stack, 2, b"0-255");
        put(&mut stack, 3, &glyph(65));
        put(&mut stack, 3, &glyph(66));
        let mut pbf = Vec::new();
        put(&mut pbf, 1, &stack);
        let subset = subset_glyphs(&pbf, &BTreeSet::from([66])).unwrap();
        let mut want = Vec::new();
        put(&mut want, 1, b"Font");
        put(&mut want, 2, b"0-255");
        put(&mut want, 3, &glyph(66));
        let mut expected = Vec::new();
        put(&mut expected, 1, &want);
        assert_eq!(subset, expected);
    }
}

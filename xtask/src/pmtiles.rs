//! Reproducible Tokyo demo extracts from OpenFreeMap's OpenMapTiles planet.

use std::io::Write;
use std::time::{Duration, Instant};

use orb_crs::TileIndex;
use orb_source::pmtiles::{HEADER_LEN, Lookup, Range, Reader, tile_id};

const SOURCE: &str = "https://btrfs.openfreemap.com/areas/planet/20261101_080013_pt/tiles.pmtiles";
const OUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/demo/data");
const BOUNDS: [f64; 4] = [139.725, 35.677, 139.755, 35.697];
/// Every zoom, so the extract draws from the whole world down to the area.
const MIN_ZOOM: u8 = 0;
const MAX_ZOOM: u8 = 14;

pub(crate) fn extract() -> Result<(), String> {
    std::fs::create_dir_all(OUT).map_err(|e| e.to_string())?;
    let agent = ureq::Agent::new_with_defaults();
    let mut reader = Reader::default();
    let mut tiles = Vec::new();
    let mut requests = 0;
    let mut downloaded = 0;
    let mut coordinates = Vec::new();
    let [west, south, east, north] = BOUNDS;
    for z in MIN_ZOOM..=MAX_ZOOM {
        let n = (1u64 << z) as f64;
        let x = |lon: f64| ((lon + 180.0) / 360.0 * n).floor() as u32;
        let y = |lat: f64| {
            ((1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor() as u32
        };
        for col in x(west)..=x(east) {
            for row in y(north)..=y(south) {
                coordinates.push(TileIndex::new(z, col, row));
            }
        }
    }
    coordinates.sort_by_key(|t| tile_id(*t).unwrap());
    for tile in &coordinates {
        loop {
            match reader.lookup(*tile)? {
                Lookup::Missing => break,
                Lookup::Read(range) => {
                    let bytes = remote(&agent, range)?;
                    requests += 1;
                    downloaded += bytes.len();
                    reader.insert(range, &bytes)?;
                }
                Lookup::Tile {
                    range, compression, ..
                } => {
                    let bytes = remote(&agent, range)?;
                    requests += 1;
                    downloaded += bytes.len();
                    tiles.push((tile_id(*tile)?, compression.decode(&bytes)?));
                    eprintln!(
                        "extract: {}/{}, {downloaded} bytes in {requests} ranges",
                        tiles.len(),
                        coordinates.len()
                    );
                    break;
                }
            }
        }
    }
    write_archive(&tiles, 1, &format!("{OUT}/tokyo-mvt.pmtiles"))?;
    let manifest = serde_json::json!({"source": SOURCE, "bounds": BOUNDS,
        "tiles": coordinates.iter().map(|t| [u32::from(t.z), t.col, t.row]).collect::<Vec<_>>(),
        "attribution": "© OpenStreetMap contributors (ODbL 1.0), © OpenMapTiles; OpenFreeMap"});
    std::fs::write(
        format!("{OUT}/manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    transcode()
}

pub(crate) fn transcode() -> Result<(), String> {
    use mlt_core::encoder::{
        Codecs, Encoder, EncoderConfig, Network, Profile, SortStrategy, StagedLayer,
        StagedProperty, Tessellation, stage_tile,
    };
    use orb_tile::{ColumnData, ValueRef};
    let archive = std::fs::read(format!("{OUT}/tokyo-mvt.pmtiles")).map_err(|e| e.to_string())?;
    let header = orb_source::pmtiles::Header::parse(&archive[..127])?;
    let entries = orb_source::pmtiles::directory(&header.internal_compression.decode(
        &archive[header.root.offset as usize..(header.root.offset + header.root.length) as usize],
    )?)?;
    let config = EncoderConfig::from_profile(Profile::Native(Network::Cellular4G))
        .with_tessellation(Tessellation::Off)
        .with_id_sort(false)
        .with_spatial_morton_sort(false)
        .with_spatial_hilbert_sort(false);
    let mut tiles = Vec::new();
    for entry in entries {
        let offset = header.tiles_offset + entry.offset;
        let bytes = header
            .tile_compression
            .decode(&archive[offset as usize..(offset + entry.length) as usize])?;
        let frame = orb_tile::mvt::decode(&bytes).map_err(|e| e.to_string())?;
        let layers = mlt_core::mvt::mvt_to_tile_layers(&bytes).map_err(|e| e.to_string())?;
        let mut encoded = Vec::new();
        for (layer, original) in layers.iter().zip(&frame.layers) {
            let staged = stage_tile(layer, SortStrategy::Unsorted, false, false);
            let mut properties = Vec::new();
            for (name, column) in original.properties() {
                let values = (0..column.len()).map(|row| column.get(row));
                let property = match column.data() {
                    ColumnData::Bool(_) => StagedProperty::opt_bool(
                        name,
                        values.map(|v| {
                            v.map(|v| match v {
                                ValueRef::Bool(v) => v,
                                _ => unreachable!(),
                            })
                        }),
                    ),
                    ColumnData::I64(_) => StagedProperty::opt_i64(
                        name,
                        values.map(|v| {
                            v.map(|v| match v {
                                ValueRef::I64(v) => v,
                                _ => unreachable!(),
                            })
                        }),
                    ),
                    ColumnData::U64(_) => StagedProperty::opt_u64(
                        name,
                        values.map(|v| {
                            v.map(|v| match v {
                                ValueRef::U64(v) => v,
                                _ => unreachable!(),
                            })
                        }),
                    ),
                    ColumnData::F64(_) => StagedProperty::opt_f64(
                        name,
                        values.map(|v| {
                            v.map(|v| match v {
                                ValueRef::F64(v) => v,
                                _ => unreachable!(),
                            })
                        }),
                    ),
                    ColumnData::Str { .. } => StagedProperty::opt_str(
                        name,
                        values.map(|v| {
                            v.map(|v| match v {
                                ValueRef::Str(v) => v,
                                _ => unreachable!(),
                            })
                        }),
                    ),
                    ColumnData::Mixed(_) => {
                        return Err(format!(
                            "MLT v1 cannot preserve mixed column {}.{name}",
                            original.name()
                        ));
                    }
                };
                properties.push(property);
            }
            let staged = StagedLayer::new(
                original.name(),
                original.extent(),
                staged.id().clone(),
                staged.geometry().clone(),
                properties,
            )
            .map_err(|e| e.to_string())?;
            let encoder = staged
                .encode_into(Encoder::new(config), &mut Codecs::default())
                .map_err(|e| e.to_string())?;
            encoded.extend(encoder.into_layer_bytes().map_err(|e| e.to_string())?);
        }
        tiles.push((entry.id, encoded));
    }
    write_archive(&tiles, 6, &format!("{OUT}/tokyo-mlt.pmtiles"))?;
    verify()
}

fn remote(agent: &ureq::Agent, range: Range) -> Result<Vec<u8>, String> {
    let end = orb_host::range_end(range.offset, range.length).map_err(|e| e.to_string())?;
    let mut response = agent
        .get(SOURCE)
        .header("Range", format!("bytes={}-{end}", range.offset))
        .header("Accept-Encoding", "identity")
        .call()
        .map_err(|e| e.to_string())?;
    orb_host::check_range(
        response.status().as_u16(),
        response
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok()),
        range.offset,
        range.length,
    )
    .map_err(|e| e.to_string())?;
    response
        .body_mut()
        .with_config()
        .limit(range.length + 1)
        .read_to_vec()
        .map_err(|e| e.to_string())
}

fn varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 128 {
        out.push(value as u8 | 128);
        value >>= 7;
    }
    out.push(value as u8);
}

fn gzip(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut writer = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    writer.write_all(bytes).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())
}

fn write_archive(tiles: &[(u64, Vec<u8>)], kind: u8, path: &str) -> Result<(), String> {
    let bodies = tiles
        .iter()
        .map(|(_, bytes)| gzip(bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let mut dir = Vec::new();
    varint(tiles.len() as u64, &mut dir);
    let mut previous = 0;
    for (id, _) in tiles {
        varint(id - previous, &mut dir);
        previous = *id;
    }
    for _ in tiles {
        varint(1, &mut dir);
    }
    for bytes in &bodies {
        varint(bytes.len() as u64, &mut dir);
    }
    for i in 0..tiles.len() {
        varint(u64::from(i == 0), &mut dir);
    }
    let dir = gzip(&dir)?;
    if dir.len() + HEADER_LEN as usize > 16384 {
        return Err("extract root exceeds first 16 KiB".into());
    }
    let metadata = gzip(r#"{"name":"Tokyo demo","attribution":"© OpenStreetMap contributors, © OpenMapTiles; OpenFreeMap","description":"OpenFreeMap planet extract; ODbL 1.0"}"#.as_bytes())?;
    let mut header = vec![0u8; HEADER_LEN as usize];
    header[..8].copy_from_slice(b"PMTiles\x03");
    let root = HEADER_LEN;
    let meta = root + dir.len() as u64;
    let data = meta + metadata.len() as u64;
    let total: u64 = bodies.iter().map(|b| b.len() as u64).sum();
    for (offset, value) in [
        (8, root),
        (16, dir.len() as u64),
        (24, meta),
        (32, metadata.len() as u64),
        (40, data),
        (48, 0),
        (56, data),
        (64, total),
        (72, tiles.len() as u64),
        (80, tiles.len() as u64),
        (88, tiles.len() as u64),
    ] {
        header[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    header[96..102].copy_from_slice(&[1, 2, 2, kind, MIN_ZOOM, MAX_ZOOM]);
    for (i, bound) in BOUNDS.into_iter().enumerate() {
        header[102 + i * 4..106 + i * 4]
            .copy_from_slice(&((bound * 1e7).round() as i32).to_le_bytes());
    }
    header[118] = 14;
    header[119..123].copy_from_slice(&(1397350000i32).to_le_bytes());
    header[123..127].copy_from_slice(&(356850000i32).to_le_bytes());
    let mut out = header;
    out.extend(dir);
    out.extend(metadata);
    for bytes in bodies {
        out.extend(bytes);
    }
    if out.len() > 10_000_000 {
        return Err("extract exceeds 10 MB".into());
    }
    std::fs::write(path, &out).map_err(|e| e.to_string())?;
    eprintln!("{path}: {} bytes, {} tiles", out.len(), tiles.len());
    Ok(())
}

pub(crate) fn verify() -> Result<(), String> {
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{OUT}/manifest.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mvt = std::fs::read(format!("{OUT}/tokyo-mvt.pmtiles")).map_err(|e| e.to_string())?;
    let mlt = std::fs::read(format!("{OUT}/tokyo-mlt.pmtiles")).map_err(|e| e.to_string())?;
    let (mut a, mut b) = (Reader::default(), Reader::default());
    let (mut time_a, mut time_b) = (Duration::ZERO, Duration::ZERO);
    let mut count = 0;
    for coordinate in manifest["tiles"].as_array().ok_or("manifest tiles")? {
        let tile = TileIndex::new(
            coordinate[0].as_u64().unwrap() as u8,
            coordinate[1].as_u64().unwrap() as u32,
            coordinate[2].as_u64().unwrap() as u32,
        );
        let x = local_tile(&mut a, &mvt, tile)?;
        let y = local_tile(&mut b, &mlt, tile)?;
        let start = Instant::now();
        let x = orb_tile::mvt::decode(&x).map_err(|e| e.to_string())?;
        time_a += start.elapsed();
        let start = Instant::now();
        let y = orb_tile::mlt::decode(&y).map_err(|e| e.to_string())?;
        time_b += start.elapsed();
        if x != y {
            for (x, y) in x.layers.iter().zip(&y.layers) {
                if x != y {
                    eprintln!(
                        "mismatch {tile:?} layer {}: features {} vs {}, vertices {} vs {}",
                        x.name(),
                        x.len(),
                        y.len(),
                        x.vertices().len(),
                        y.vertices().len()
                    );
                    for ((key, a), (_, b)) in x.properties().zip(y.properties()) {
                        if a != b {
                            let row = (0..a.len().min(b.len()))
                                .find(|&i| a.get(i) != b.get(i))
                                .unwrap_or(0);
                            eprintln!(
                                "property {key} row {row}: {:?} vs {:?}",
                                a.get(row),
                                b.get(row)
                            );
                        }
                    }
                    break;
                }
            }
            return Err(format!("column mismatch at {tile:?}"));
        }
        count += 1;
    }
    eprintln!(
        "{count} tiles: identical columns; decode MVT {:.3} ms, MLT {:.3} ms; archives {} vs {} bytes",
        time_a.as_secs_f64() * 1000.0,
        time_b.as_secs_f64() * 1000.0,
        mvt.len(),
        mlt.len()
    );
    Ok(())
}

pub(crate) fn local_tile(
    reader: &mut Reader,
    archive: &[u8],
    tile: TileIndex,
) -> Result<Vec<u8>, String> {
    loop {
        match reader.lookup(tile)? {
            Lookup::Read(range) => reader.insert(
                range,
                &archive[range.offset as usize..(range.offset + range.length) as usize],
            )?,
            Lookup::Tile {
                range, compression, ..
            } => {
                return compression.decode(
                    &archive[range.offset as usize..(range.offset + range.length) as usize],
                );
            }
            Lookup::Missing => return Err(format!("missing extract tile {tile:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn extract_columns_match() {
        super::verify().unwrap();
    }
}

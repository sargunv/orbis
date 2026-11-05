//! Reproducible Tokyo demo extracts from OpenFreeMap's OpenMapTiles planet.

use std::io::Write;

use orb_crs::TileIndex;
use orb_source::pmtiles::{Compression, Contents, Entry, Lookup, Range, Reader, TileType, tile_id};

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
    write_archive(&tiles, TileType::Mvt, &format!("{OUT}/tokyo-mvt.pmtiles"))?;
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

fn transcode() -> Result<(), String> {
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
    write_archive(&tiles, TileType::Mlt, &format!("{OUT}/tokyo-mlt.pmtiles"))
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

fn gzip(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut writer = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    writer.write_all(bytes).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())
}

fn write_archive(tiles: &[(u64, Vec<u8>)], kind: TileType, path: &str) -> Result<(), String> {
    let mut entries = Vec::new();
    let mut data = Vec::new();
    for (id, bytes) in tiles {
        let body = gzip(bytes)?;
        entries.push(Entry {
            id: *id,
            offset: data.len() as u64,
            length: body.len() as u64,
            run: 1,
        });
        data.extend(body);
    }
    let contents = Contents {
        tile_type: kind,
        tile_compression: Compression::Gzip,
        minzoom: MIN_ZOOM,
        maxzoom: MAX_ZOOM,
        bounds: BOUNDS,
        metadata: r#"{"name":"Tokyo demo","attribution":"© OpenStreetMap contributors, © OpenMapTiles; OpenFreeMap","description":"OpenFreeMap planet extract; ODbL 1.0"}"#.into(),
    };
    let mut out = orb_source::pmtiles::write_head(&mut entries, &contents)?;
    out.extend(data);
    if out.len() > 10_000_000 {
        return Err("extract exceeds 10 MB".into());
    }
    std::fs::write(path, &out).map_err(|e| e.to_string())?;
    eprintln!("{path}: {} bytes, {} tiles", out.len(), tiles.len());
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
    use super::*;

    /// The MLT archive decodes to the same columns as the MVT one.
    #[test]
    fn extract_columns_match() {
        let read = |name: &str| std::fs::read(format!("{OUT}/{name}")).unwrap();
        let manifest: serde_json::Value = serde_json::from_slice(&read("manifest.json")).unwrap();
        let (mvt, mlt) = (read("tokyo-mvt.pmtiles"), read("tokyo-mlt.pmtiles"));
        let (mut a, mut b) = (Reader::default(), Reader::default());
        for c in manifest["tiles"].as_array().unwrap() {
            let at = |i: usize| c[i].as_u64().unwrap() as u32;
            let tile = TileIndex::new(at(0) as u8, at(1), at(2));
            let x = orb_tile::mvt::decode(&local_tile(&mut a, &mvt, tile).unwrap()).unwrap();
            let y = orb_tile::mlt::decode(&local_tile(&mut b, &mlt, tile).unwrap()).unwrap();
            for (x, y) in x.layers.iter().zip(&y.layers) {
                assert!(x == y, "{tile:?} layer {} differs", x.name());
            }
            assert_eq!(x.layers.len(), y.layers.len(), "{tile:?}");
        }
    }
}

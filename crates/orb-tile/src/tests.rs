use crate::{ColumnData, DecodeError, GeometryType, TileFrame, ValueRef, mlt, mvt};
use mlt_core::encoder::{EncoderConfig, Network, Profile, Tessellation};

fn varint(mut n: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while n >= 128 {
        bytes.push(n as u8 | 128);
        n >>= 7;
    }
    bytes.push(n as u8);
    bytes
}

fn uint(field: u32, n: u64) -> Vec<u8> {
    [varint(u64::from(field) << 3), varint(n)].concat()
}

fn message(field: u32, bytes: &[u8]) -> Vec<u8> {
    [
        varint((u64::from(field) << 3) | 2),
        varint(bytes.len() as u64),
        bytes.to_vec(),
    ]
    .concat()
}

fn packed(field: u32, words: &[u32]) -> Vec<u8> {
    message(
        field,
        &words
            .iter()
            .flat_map(|&n| varint(u64::from(n)))
            .collect::<Vec<_>>(),
    )
}

fn feature(id: Option<u64>, kind: u32, geometry: &[u32], tags: &[u32]) -> Vec<u8> {
    let mut bytes = packed(4, geometry);
    // Put the header after its payload to exercise arbitrary field order.
    bytes.extend(packed(2, tags));
    bytes.extend(uint(3, u64::from(kind)));
    if let Some(id) = id {
        bytes.extend(uint(1, id));
    }
    bytes
}

fn layer(features: &[Vec<u8>], keys: &[&str], values: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes: Vec<_> = features.iter().flat_map(|f| message(2, f)).collect();
    for key in keys {
        bytes.extend(message(3, key.as_bytes()));
    }
    for value in values {
        bytes.extend(message(4, value));
    }
    bytes.extend(message(1, b"test"));
    bytes.extend(uint(15, 2));
    bytes
}

fn tile(features: &[Vec<u8>], keys: &[&str], values: &[Vec<u8>]) -> Vec<u8> {
    message(3, &layer(features, keys, values))
}

fn zigzag(n: i32) -> u32 {
    ((n as u32) << 1) ^ ((n >> 31) as u32)
}

fn paths(paths: &[&[[i32; 2]]], close: bool) -> Vec<u32> {
    let mut cursor = [0, 0];
    let mut words = Vec::new();
    for path in paths {
        words.push(9);
        for (i, point) in path.iter().enumerate() {
            if i == 1 {
                words.push((((path.len() - 1) as u32) << 3) | 2);
            }
            words.extend([zigzag(point[0] - cursor[0]), zigzag(point[1] - cursor[1])]);
            cursor = *point;
        }
        if close {
            words.push(15);
        }
    }
    words
}

fn rings(frame: &TileFrame, feature: usize) -> Vec<Vec<Vec<[i32; 2]>>> {
    frame.layers[0]
        .geometry(feature)
        .parts()
        .map(|p| p.rings().map(<[_]>::to_vec).collect())
        .collect()
}

fn geometry_fixture() -> Vec<u8> {
    let square = [[0, 0], [20, 0], [20, 20], [0, 20]];
    let hole = [[2, 2], [2, 4], [4, 4], [4, 2]];
    let second_hole = [[6, 6], [6, 8], [8, 8], [8, 6]];
    let island = [[30, 30], [40, 30], [30, 40]];
    tile(
        &[
            feature(Some(0), 1, &[9, 10, 12], &[]),
            feature(None, 1, &[25, 2, 4, 6, 8, 3, 5], &[]),
            feature(
                Some(u64::MAX),
                2,
                &paths(&[&[[-2, 3], [4, 5], [8, -9]]], false),
                &[],
            ),
            feature(
                None,
                2,
                &paths(&[&[[1, 2], [3, 4]], &[[10, 20], [30, 40]]], false),
                &[],
            ),
            feature(None, 3, &paths(&[&square, &hole, &second_hole], true), &[]),
            feature(Some(9), 3, &paths(&[&square, &hole, &island], true), &[]),
        ],
        &[],
        &[],
    )
}

/// Zero bytes after a tile read as fields numbered 0 and are skipped, as
/// GL JS's `pbf` skips them (`filter/modulo`'s tile is padded so).
#[test]
fn zero_padding_is_skipped() {
    let mut padded = geometry_fixture();
    padded.extend([0; 16]);
    assert_eq!(
        mvt::decode(&padded).unwrap().layers[0].geometry_types(),
        mvt::decode(&geometry_fixture()).unwrap().layers[0].geometry_types()
    );
}

#[test]
fn all_geometry_types_and_ids() {
    let frame = mvt::decode(&geometry_fixture()).unwrap();
    let layer = &frame.layers[0];
    assert_eq!(layer.name(), "test");
    assert_eq!(layer.extent(), 4096);
    assert_eq!(
        layer.geometry_types(),
        &[
            GeometryType::Point,
            GeometryType::Point,
            GeometryType::LineString,
            GeometryType::LineString,
            GeometryType::Polygon,
            GeometryType::Polygon,
        ]
    );
    assert_eq!(layer.ids().get(0), Some(ValueRef::U64(0)));
    assert_eq!(layer.ids().get(1), None);
    assert_eq!(layer.ids().get(2), Some(ValueRef::U64(u64::MAX)));
    assert_eq!(rings(&frame, 0), vec![vec![vec![[5, 6]]]]);
    assert_eq!(rings(&frame, 1), vec![vec![vec![[1, 2], [4, 6], [2, 3]]]]);
    assert_eq!(rings(&frame, 2), vec![vec![vec![[-2, 3], [4, 5], [8, -9]]]]);
    assert_eq!(
        rings(&frame, 3),
        vec![vec![vec![[1, 2], [3, 4]]], vec![vec![[10, 20], [30, 40]]]]
    );
    assert_eq!(rings(&frame, 4)[0].len(), 3);
    assert_eq!(rings(&frame, 5).len(), 2);
    assert_eq!(rings(&frame, 5)[1][0], [[30, 30], [40, 30], [30, 40]]);
    // ClosePath never repeats the first vertex in the stored ring.
    assert_eq!(rings(&frame, 4)[0][0].len(), 4);
    assert_eq!(layer.vertices.len(), 34);
}

fn fixed_value(field: u32, wire: u8, bytes: &[u8]) -> Vec<u8> {
    [
        varint((u64::from(field) << 3) | u64::from(wire)),
        bytes.to_vec(),
    ]
    .concat()
}

#[test]
fn every_mvt_value_type_and_sparse_mixed_columns() {
    let values = vec![
        message(1, "héllo".as_bytes()),
        fixed_value(2, 5, &1.25f32.to_le_bytes()),
        fixed_value(3, 1, &2.5f64.to_le_bytes()),
        uint(4, (-7i64) as u64),
        uint(5, u64::MAX),
        uint(6, 17),
        uint(7, 1),
        uint(7, 0),
        uint(6, u64::MAX),
    ];
    let keys = [
        "string", "float", "double", "int", "uint", "sint", "bool", "mixed", "number", "min",
    ];
    let frame = mvt::decode(&tile(
        &[
            feature(
                None,
                1,
                &[9, 0, 0],
                &[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 0, 8, 3, 9, 8],
            ),
            feature(Some(42), 1, &[9, 2, 4], &[6, 7, 7, 6, 8, 2]),
            feature(None, 1, &[9, 4, 6], &[]),
        ],
        &keys,
        &values,
    ))
    .unwrap();
    let layer = &frame.layers[0];
    for (key, expected) in [
        ("string", ValueRef::Str("héllo")),
        ("float", ValueRef::F64(1.25)),
        ("double", ValueRef::F64(2.5)),
        ("int", ValueRef::I64(-7)),
        ("uint", ValueRef::U64(u64::MAX)),
        ("sint", ValueRef::I64(-9)),
        ("bool", ValueRef::Bool(true)),
        ("min", ValueRef::I64(i64::MIN)),
    ] {
        assert_eq!(layer.property(key).unwrap().get(0), Some(expected));
        assert_eq!(layer.property(key).unwrap().get(2), None);
    }
    assert!(matches!(
        layer.property("mixed").unwrap().data(),
        ColumnData::Mixed(_)
    ));
    assert_eq!(
        layer.property("mixed").unwrap().get(1),
        Some(ValueRef::Bool(true))
    );
    assert_eq!(
        layer.property("bool").unwrap().get(1),
        Some(ValueRef::Bool(false))
    );
    assert_eq!(
        layer.property("number").unwrap().get(0),
        Some(ValueRef::F64(-7.0))
    );
    assert_eq!(
        layer.property("number").unwrap().get(1),
        Some(ValueRef::F64(2.5))
    );
}

#[test]
fn duplicate_tags_last_value_wins() {
    let frame = mvt::decode(&tile(
        &[feature(None, 1, &[9, 0, 0], &[0, 0, 0, 1])],
        &["k"],
        &[uint(5, 1), uint(5, 2)],
    ))
    .unwrap();
    assert_eq!(
        frame.layers[0].property("k").unwrap().get(0),
        Some(ValueRef::U64(2))
    );
}

/// A layer may name a key twice and hold a value twice: the key is one
/// column, and each of the equal strings reads back alike.
#[test]
fn repeated_keys_share_a_column() {
    let s = |v: &str| message(1, v.as_bytes());
    let frame = mvt::decode(&tile(
        &[
            feature(None, 1, &[9, 0, 0], &[0, 0]),
            feature(None, 1, &[9, 2, 2], &[1, 1]),
            feature(None, 1, &[9, 4, 4], &[1, 2]),
        ],
        &["k", "k"],
        &[s("a"), s("a"), s("b")],
    ))
    .unwrap();
    let layer = &frame.layers[0];
    assert_eq!(layer.properties().count(), 1);
    let column = layer.property("k").unwrap();
    assert!(matches!(column.data(), ColumnData::Str { .. }));
    assert_eq!(column.get(0), Some(ValueRef::Str("a")));
    assert_eq!(column.get(1), Some(ValueRef::Str("a")));
    assert_eq!(column.get(2), Some(ValueRef::Str("b")));
}

#[test]
fn empty_unknown_and_zero_area_features_are_dropped() {
    let zero = paths(&[&[[0, 0], [1, 1], [2, 2]]], true);
    let mut polygon = paths(&[&[[0, 0], [10, 0], [0, 10]]], true);
    // Cursor is [0,10], so this zero-area ring also checks cursor continuity.
    polygon.extend([9, 0, 19, 18, 2, 2, 2, 2, 15]);
    let frame = mvt::decode(&tile(
        &[
            feature(None, 3, &zero, &[0, 0]),
            feature(Some(1), 1, &[], &[]),
            feature(None, 99, &[123], &[0]),
            feature(None, 0, &[], &[]),
            feature(Some(7), 3, &polygon, &[]),
            feature(None, 1, &[9, 2, 4], &[]),
        ],
        &["dropped"],
        &[uint(7, 1)],
    ))
    .unwrap();
    assert_eq!(frame.layers[0].len(), 2);
    assert_eq!(frame.layers[0].properties().count(), 0);
    assert_eq!(rings(&frame, 0), vec![vec![vec![[0, 0], [10, 0], [0, 10]]]]);
    assert_eq!(frame.layers[0].ids().get(0), Some(ValueRef::U64(7)));
    assert_eq!(rings(&frame, 1), vec![vec![vec![[1, 2]]]]);
}

#[test]
fn explicit_ring_closure_is_removed() {
    let geometry = paths(&[&[[0, 0], [10, 0], [0, 10], [0, 0]]], true);
    let frame = mvt::decode(&tile(&[feature(None, 3, &geometry, &[])], &[], &[])).unwrap();
    assert_eq!(rings(&frame, 0), vec![vec![vec![[0, 0], [10, 0], [0, 10]]]]);
}

#[test]
fn packed_unpacked_and_split_repeated_fields() {
    let mut f = uint(4, 9);
    f.extend(packed(4, &[2])); // split one MoveTo across three segments
    f.extend(uint(4, 4));
    f.extend(uint(2, 0));
    f.extend(packed(2, &[]));
    f.extend(packed(2, &[0]));
    f.extend(uint(3, 1));
    let frame = mvt::decode(&tile(&[f], &["k"], &[uint(7, 1)])).unwrap();
    assert_eq!(rings(&frame, 0), vec![vec![vec![[1, 2]]]]);
    assert_eq!(
        frame.layers[0].property("k").unwrap().get(0),
        Some(ValueRef::Bool(true))
    );
}

#[test]
fn repeated_point_moves_share_one_part() {
    let frame = mvt::decode(&tile(
        &[feature(None, 1, &[9, 2, 4, 9, 4, 6], &[])],
        &[],
        &[],
    ))
    .unwrap();
    assert_eq!(rings(&frame, 0), vec![vec![vec![[1, 2], [3, 5]]]]);
}

#[test]
fn unknown_protobuf_fields_and_groups_are_skipped() {
    let mut unknown = uint(100, u64::MAX);
    unknown.extend(message(101, b"ignored"));
    unknown.extend(fixed_value(102, 1, &[0; 8]));
    unknown.extend(fixed_value(103, 5, &[0; 4]));
    unknown.extend(varint((104 << 3) | 3));
    unknown.extend(varint((105 << 3) | 3));
    unknown.extend(uint(1, 3));
    unknown.extend(varint((105 << 3) | 4));
    unknown.extend(varint((104 << 3) | 4));
    let mut f = feature(None, 1, &[9, 0, 0], &[0, 0]);
    f.extend(&unknown);
    let mut value = uint(7, 1);
    value.extend(&unknown);
    let mut l = layer(&[f], &["k"], &[value]);
    l.extend(&unknown);
    l.extend(uint(5, 512));
    let mut bytes = message(3, &l);
    bytes.extend(unknown);
    let frame = mvt::decode(&bytes).unwrap();
    assert_eq!(frame.layers[0].extent(), 512);
    assert_eq!(
        frame.layers[0].property("k").unwrap().get(0),
        Some(ValueRef::Bool(true))
    );
}

#[test]
fn multiple_layers_and_empty_tiles() {
    assert_eq!(mvt::decode(&[]).unwrap(), TileFrame::default());
    assert_eq!(mlt::decode(&[]).unwrap(), TileFrame::default());
    let mut bytes = tile(&[], &[], &[]);
    bytes.extend(geometry_fixture());
    let frame = mvt::decode(&bytes).unwrap();
    assert!(frame.layers[0].is_empty());
    assert_eq!(frame.layers[1].len(), 6);
}

#[test]
fn malformed_protobuf_and_metadata_return_errors() {
    for bytes in [vec![0], vec![0xff; 10], vec![0x1a, 100], vec![0x1f]].into_iter() {
        assert!(mvt::decode(&bytes).is_err(), "{bytes:?}");
    }
    for l in [
        vec![],
        uint(15, 2),
        [message(1, b"x"), uint(15, 3)].concat(),
        [message(1, b"x"), uint(15, 2), uint(5, 0)].concat(),
        [layer(&[], &[], &[]), uint(5, 1 << 32)].concat(),
        [layer(&[], &[], &[]), varint((100 << 3) | 3)].concat(),
        [layer(&[], &[], &[]), varint((100 << 3) | 4)].concat(),
        [
            layer(&[], &[], &[]),
            varint((100 << 3) | 3),
            varint((101 << 3) | 4),
        ]
        .concat(),
    ] {
        assert!(mvt::decode(&message(3, &l)).is_err(), "{l:?}");
    }
    assert!(matches!(
        mvt::decode(&message(3, &uint(15, 2))),
        Err(DecodeError::MissingField("layer.name"))
    ));
}

#[test]
fn layers_past_the_largest_extent_are_put_in_8192() {
    let ring: &[[i32; 2]] = &[[0, 0], [1 << 30, 0], [3, (1 << 30) + (1 << 17)]];
    let polygon = feature(None, 3, &paths(&[ring], true), &[]);
    let wide = [layer(&[polygon], &[], &[]), uint(5, 1 << 31)].concat();
    let frame = mvt::decode(&message(3, &wide)).unwrap();
    assert_eq!(frame.layers[0].extent(), 8192);
    // GL JS rounds half up.
    assert_eq!(
        rings(&frame, 0),
        vec![vec![vec![[0, 0], [4096, 0], [0, 4097]]]]
    );
    let polygon = feature(None, 3, &paths(&[ring], true), &[]);
    let kept = [layer(&[polygon], &[], &[]), uint(5, 16384)].concat();
    let frame = mvt::decode(&message(3, &kept)).unwrap();
    assert_eq!(frame.layers[0].extent(), 16384);
}

/// Invalid UTF-8 in a layer's name, keys or string values reads as
/// U+FFFD, as GL JS's pbf decodes it, rather than failing the tile
/// (Native's InvalidUTF8InTile test loads such a map).
#[test]
fn invalid_utf8_strings_are_replaced() {
    let bytes = message(
        3,
        &[
            message(2, &feature(None, 1, &[9, 0, 0], &[0, 0])),
            message(3, b"k\xff"),
            message(4, &message(1, b"caf\xe9")),
            message(1, b"r\xffoad"),
            uint(15, 2),
        ]
        .concat(),
    );
    let frame = mvt::decode(&bytes).unwrap();
    let layer = &frame.layers[0];
    assert_eq!(layer.name(), "r\u{fffd}oad");
    let column = layer.property("k\u{fffd}").unwrap();
    assert_eq!(column.get(0), Some(ValueRef::Str("caf\u{fffd}")));
}

#[test]
fn malformed_values_and_tags_return_errors() {
    for v in [
        vec![],
        vec![0x15, 0],
        vec![0x19, 0],
        [uint(7, 1), uint(5, 2)].concat(),
        message(7, &[1]),
    ] {
        assert!(mvt::decode(&tile(&[], &[], &[v])).is_err());
    }
    for tags in [&[0][..], &[1, 0], &[0, 1]] {
        for geometry in [&[9, 0, 0][..], &[]] {
            assert!(
                mvt::decode(&tile(
                    &[feature(None, 1, geometry, tags)],
                    &["k"],
                    &[uint(7, 1)]
                ))
                .is_err()
            );
        }
    }
    let mut f = feature(None, 1, &[9, 0, 0], &[]);
    f.extend(message(2, &[128]));
    assert!(mvt::decode(&tile(&[f], &[], &[])).is_err());
}

#[test]
fn malformed_commands_and_coordinate_overflow_return_errors() {
    for (kind, geometry) in [
        (1, vec![0]),
        (1, vec![8]),
        (1, vec![11]),
        (1, vec![9]),
        (1, vec![9, 0]),
        (1, vec![9, 0, 0, 10, 0, 0]),
        (1, vec![15]),
        (2, vec![17, 0, 0, 0, 0]),
        (2, vec![10, 0, 0]),
        (2, vec![9, 0, 0]),
        (2, vec![9, 0, 0, 9, 0, 0]),
        (2, vec![9, 0, 0, 10, 0]),
        (2, vec![9, 0, 0, 15]),
        (3, vec![9, 0, 0, 18, 20, 0, 0, 20]), // missing ClosePath
        (3, vec![9, 0, 0, 10, 20, 0, 15]),    // two vertices
        (3, vec![9, 0, 0, 18, 20, 0, 0, 20, 23]), // ClosePath count 2
        (3, paths(&[&[[0, 0], [0, 10], [10, 10]]], true)), // orphan hole
        (1, vec![17, zigzag(i32::MAX), 0, 2, 0]),
        (1, vec![17, zigzag(i32::MIN), 0, 1, 0]),
    ] {
        assert!(
            mvt::decode(&tile(&[feature(None, kind, &geometry, &[])], &[], &[])).is_err(),
            "{kind}: {geometry:?}"
        );
    }
}

fn config(tessellation: Tessellation) -> EncoderConfig {
    EncoderConfig::from_profile(Profile::Native(Network::Cellular4G))
        .with_tessellation(tessellation)
        .with_id_sort(false)
        .with_spatial_morton_sort(false)
        .with_spatial_hilbert_sort(false)
}

fn to_mlt(bytes: &[u8], tessellation: Tessellation) -> Vec<u8> {
    mlt_core::mvt::mvt_to_tile_layers(bytes)
        .unwrap()
        .iter()
        .flat_map(|l| l.encode(config(tessellation)).unwrap())
        .collect()
}

#[test]
fn mvt_and_mlt_geometry_frames_are_equal() {
    let bytes = geometry_fixture();
    for tessellation in [Tessellation::Off, Tessellation::Always] {
        let encoded = to_mlt(&bytes, tessellation);
        assert_eq!(mvt::decode(&bytes).unwrap(), mlt::decode(&encoded).unwrap());
    }
}

fn property_fixture() -> Vec<u8> {
    let values = vec![
        message(1, b"road"),
        message(1, b"rue"),
        fixed_value(2, 5, &1.25f32.to_le_bytes()),
        fixed_value(3, 1, &2.5f64.to_le_bytes()),
        uint(4, (-7i64) as u64),
        uint(5, u64::MAX),
        uint(6, 17),
        uint(7, 1),
        uint(7, 0),
    ];
    let keys = [
        "name:en", "name:fr", "float", "double", "int", "uint", "sint", "bool", "number",
    ];
    tile(
        &[
            feature(
                Some(7),
                1,
                &[9, 0, 0],
                &[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 4],
            ),
            feature(None, 1, &[9, 2, 4], &[0, 1, 7, 8, 8, 3]),
            feature(Some(9), 1, &[9, 4, 6], &[1, 0]),
        ],
        &keys,
        &values,
    )
}

#[test]
fn mvt_and_mlt_property_frames_are_equal() {
    let bytes = property_fixture();
    let encoded = to_mlt(&bytes, Tessellation::Off);
    assert_eq!(mvt::decode(&bytes).unwrap(), mlt::decode(&encoded).unwrap());
}

#[test]
fn mvt_and_mlt_multilayer_frames_are_equal() {
    let bytes = [geometry_fixture(), property_fixture()].concat();
    let encoded = to_mlt(&bytes, Tessellation::Off);
    println!(
        "two-layer fixture: {} MVT bytes, {} MLT bytes",
        bytes.len(),
        encoded.len()
    );
    assert_eq!(mvt::decode(&bytes).unwrap(), mlt::decode(&encoded).unwrap());
}

#[test]
fn upstream_conversion_type_losses_are_explicit() {
    // These are changes made by the converter/encoder, rather than information
    // available to the MLT decoder. Preserve the actual wire types in each.
    let bytes = tile(
        &[
            feature(None, 1, &[9, 0, 0], &[0, 0, 1, 1]),
            feature(None, 1, &[9, 2, 4], &[0, 2, 1, 1]),
        ],
        &["mixed", "positive"],
        &[message(1, b"text"), uint(4, 7), uint(7, 1)],
    );
    let original = mvt::decode(&bytes).unwrap();
    let converted = mlt::decode(&to_mlt(&bytes, Tessellation::Off)).unwrap();
    assert!(matches!(
        original.layers[0].property("mixed").unwrap().data(),
        ColumnData::Mixed(_)
    ));
    assert!(matches!(
        converted.layers[0].property("mixed").unwrap().data(),
        ColumnData::Str { .. }
    ));
    assert_eq!(
        original.layers[0].property("positive").unwrap().get(0),
        Some(ValueRef::I64(7))
    );
    assert_eq!(
        converted.layers[0].property("positive").unwrap().get(0),
        Some(ValueRef::U64(7))
    );
}

#[test]
fn mvt_and_mlt_null_first_property_frames_are_equal() {
    let bytes = tile(
        &[
            feature(None, 1, &[9, 0, 0], &[]),
            feature(Some(u64::MAX), 1, &[9, 2, 4], &[0, 0, 1, 1, 2, 2]),
        ],
        &["signed", "unsigned", "string"],
        &[
            uint(4, i64::MIN as u64),
            uint(5, u64::MAX),
            message(1, b"later"),
        ],
    );
    assert_eq!(
        mvt::decode(&bytes).unwrap(),
        mlt::decode(&to_mlt(&bytes, Tessellation::Off)).unwrap()
    );
}

/// Invalid UTF-8 in an MLT string column reads as U+FFFD, as in MVT,
/// rather than failing the tile (mlt-core from Orbis's fork), whether the
/// column stores each feature's string or a dictionary of them.
#[test]
fn mlt_invalid_utf8_strings_are_replaced() {
    // The tile's strings with a stand-in byte, then the invalid one.
    let invalid = |bytes: &mut Vec<u8>| {
        for (from, to) in [(&b"caf~"[..], &b"caf\xe9"[..]), (b"r~ue", b"r\xffue")] {
            let at = bytes.windows(4).position(|w| w == from).unwrap();
            bytes[at..at + 4].copy_from_slice(to);
            assert!(!bytes.windows(4).any(|w| w == from));
        }
    };
    for count in [2, 12] {
        let features: Vec<_> = (0..count)
            .map(|i| feature(None, 1, &[9, 2 * i, 0], &[0, i % 2]))
            .collect();
        let mut mvt_bytes = tile(
            &features,
            &["name"],
            &[message(1, b"caf~"), message(1, b"r~ue")],
        );
        let mut mlt_bytes: Vec<u8> = mlt_core::mvt::mvt_to_tile_layers(&mvt_bytes)
            .unwrap()
            .iter()
            .flat_map(|l| {
                l.encode(config(Tessellation::Off).with_fsst(false))
                    .unwrap()
            })
            .collect();
        invalid(&mut mvt_bytes);
        invalid(&mut mlt_bytes);
        let frame = mlt::decode(&mlt_bytes).unwrap();
        assert_eq!(frame, mvt::decode(&mvt_bytes).unwrap(), "{count} features");
        let column = frame.layers[0].property("name").unwrap();
        assert_eq!(column.get(0), Some(ValueRef::Str("caf\u{fffd}")));
        assert_eq!(column.get(1), Some(ValueRef::Str("r\u{fffd}ue")));
    }
}

#[test]
fn mlt_winding_and_implicit_closure() {
    use mlt_core::geo_types::{Geometry, LineString, Polygon};
    let mut layer = mlt_core::TileLayer::builder("test", 4096).unwrap();
    layer
        .feature(Geometry::Polygon(Polygon::new(
            LineString::from(vec![(0, 0), (0, 10), (10, 0)]),
            vec![LineString::from(vec![(1, 1), (2, 1), (1, 2)])],
        )))
        .finish()
        .unwrap();
    let frame = mlt::decode(&layer.finish().encode(config(Tessellation::Off)).unwrap()).unwrap();
    assert_eq!(
        rings(&frame, 0),
        vec![vec![
            vec![[0, 0], [10, 0], [0, 10]],
            vec![[1, 1], [1, 2], [2, 1]]
        ]]
    );
}

#[test]
fn mlt_skips_unknown_layer_tags() {
    assert_eq!(mlt::decode(&[3, 100, 7, 8]).unwrap(), TileFrame::default());
}

#[test]
fn decoder_errors_implement_std_error() {
    fn check(error: &dyn std::error::Error) {
        assert!(!error.to_string().is_empty());
    }
    check(&mvt::decode(&[0]).unwrap_err());
    check(&mlt::decode(&[255]).unwrap_err());
}

#[test]
fn truncated_and_garbage_inputs_never_panic() {
    let mvt_bytes = [geometry_fixture(), property_fixture()].concat();
    let mlt_bytes = to_mlt(&mvt_bytes, Tessellation::Off);
    // Every prefix includes truncations inside lengths, varints and payloads.
    for bytes in [&mvt_bytes, &mlt_bytes] {
        for end in 0..bytes.len() {
            let prefix = &bytes[..end];
            assert!(
                std::panic::catch_unwind(|| {
                    let _ = mvt::decode(prefix);
                })
                .is_ok(),
                "MVT prefix {end}"
            );
            assert!(
                std::panic::catch_unwind(|| {
                    let _ = mlt::decode(prefix);
                })
                .is_ok(),
                "MLT prefix {end}"
            );
        }
    }
    let mut seed = 0xdead_beef_u32;
    for len in 0..128 {
        for _ in 0..16 {
            let bytes: Vec<_> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed as u8
                })
                .collect();
            assert!(
                std::panic::catch_unwind(|| {
                    let _ = mvt::decode(&bytes);
                })
                .is_ok(),
                "MVT {bytes:?}"
            );
            assert!(
                std::panic::catch_unwind(|| {
                    let _ = mlt::decode(&bytes);
                })
                .is_ok(),
                "MLT {bytes:?}"
            );
        }
    }
}

#[test]
fn full_range_coordinates_do_not_overflow_polygon_area() {
    let points = [
        [i32::MIN, i32::MIN],
        [-1, i32::MIN],
        [i32::MAX - 1, i32::MIN],
        [i32::MAX, i32::MIN],
        [i32::MAX, -1],
        [i32::MAX, i32::MAX - 1],
        [i32::MAX, i32::MAX],
        [-1, i32::MAX],
        [i32::MIN, i32::MAX],
        [i32::MIN, -1],
        [i32::MIN, i32::MIN + 1],
    ];
    let frame = mvt::decode(&tile(
        &[feature(None, 3, &paths(&[&points], true), &[])],
        &[],
        &[],
    ))
    .unwrap();
    assert_eq!(rings(&frame, 0), vec![vec![points.to_vec()]]);
    assert!(crate::decode::signed_area(points.into_iter()) > i128::from(i64::MAX));
}

#[test]
fn decodes_mvt_version_1_and_rejects_unknown_versions() {
    let point = feature(Some(1), 1, &[9, zigzag(5), zigzag(7)], &[]);
    let with_version = |version: Option<u64>| {
        let mut bytes = message(2, &point);
        bytes.extend(message(1, b"test"));
        if let Some(version) = version {
            bytes.extend(uint(15, version));
        }
        message(3, &bytes)
    };
    // Version 1 shares version 2's encoding, and a missing version means 1.
    for version in [Some(1), None, Some(2)] {
        let frame = mvt::decode(&with_version(version)).unwrap();
        assert_eq!(frame.layers[0].vertices, &[[5, 7]], "{version:?}");
    }
    assert!(matches!(
        mvt::decode(&with_version(Some(3))),
        Err(DecodeError::UnsupportedVersion(3))
    ));
}

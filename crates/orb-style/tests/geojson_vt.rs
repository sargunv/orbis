//! Golden vectors from unmodified @maplibre/geojson-vt 6.3.0 JS (ISC/MIT;
//! orb-tile/src/geojson/LICENSE). Input covers world wrapping, polygons with
//! holes, multi-geometries, simplification, clipping, and line metrics.
use orb_expr::json::{self, Json};
use orb_tile::{
    GeometryType, ValueRef,
    geojson::{Index, Options},
};

#[test]
fn matches_geojson_vt_6_3_0() {
    let features = orb_style::v8::parse_geojson(include_str!("geojson/input.json")).unwrap();
    let mut index = Index::new(
        features,
        Options {
            max_zoom: 4,
            line_metrics: true,
            ..Options::default()
        },
    )
    .unwrap();
    let expected = json::parse(include_str!("geojson/expected.json")).unwrap();
    for expected in expected.as_array().unwrap() {
        let num = |v: &Json| v.as_f64().unwrap();
        let z = num(expected.get("z").unwrap()) as u8;
        let x = num(expected.get("x").unwrap()) as i32;
        let y = num(expected.get("y").unwrap()) as u32;
        let frame = index.tile(z, x, y).unwrap_or_default();
        let actual = &frame.layers[0];
        let features = expected.get("features").unwrap().as_array().unwrap();
        assert_eq!(actual.len(), features.len(), "{z}/{x}/{y}");
        for (i, f) in features.iter().enumerate() {
            assert_eq!(
                actual.ids().get(i),
                f.get("id").map(|v| ValueRef::F64(num(v)))
            );
            for tag in f.get("tags").unwrap().as_object().unwrap() {
                let expected = match &tag.value.value {
                    json::JsonValue::String(s) => ValueRef::Str(s),
                    json::JsonValue::Number(n) => ValueRef::F64(*n),
                    other => panic!("unexpected {other:?}"),
                };
                let actual = actual.property(&tag.key).unwrap().get(i).unwrap();
                match (actual, expected) {
                    (ValueRef::F64(a), ValueRef::F64(b)) => assert!(
                        (a - b).abs() < 1e-12,
                        "{z}/{x}/{y}, {i}, {}: {a} != {b}",
                        tag.key
                    ),
                    _ => assert_eq!(actual, expected),
                }
            }
            let geometry = f.get("geometry").unwrap().as_array().unwrap();
            let mut expected: Vec<Vec<[i32; 2]>> = if actual.geometry_type(i) == GeometryType::Point
            {
                vec![
                    geometry
                        .iter()
                        .map(|p| {
                            let p = p.as_array().unwrap();
                            [num(&p[0]) as i32, num(&p[1]) as i32]
                        })
                        .collect(),
                ]
            } else {
                geometry
                    .iter()
                    .map(|r| {
                        r.as_array()
                            .unwrap()
                            .iter()
                            .map(|p| {
                                let p = p.as_array().unwrap();
                                [num(&p[0]) as i32, num(&p[1]) as i32]
                            })
                            .collect()
                    })
                    .collect()
            };
            if actual.geometry_type(i) == GeometryType::Polygon {
                for ring in &mut expected {
                    ring.dedup();
                    if ring.len() > 1 && ring.first() == ring.last() {
                        ring.pop();
                    }
                }
                expected.retain(|r| r.len() >= 3);
            }
            let actual: Vec<Vec<[i32; 2]>> = actual
                .geometry(i)
                .parts()
                .flat_map(|p| p.rings().map(|r| r.to_vec()).collect::<Vec<_>>())
                .collect();
            assert_eq!(actual, expected, "{z}/{x}/{y}, feature {i}");
        }
    }
}

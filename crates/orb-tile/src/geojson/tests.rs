use super::*;
use crate::{GeometryType, ValueRef};

fn feature(geometry: Geometry) -> Feature {
    Feature {
        id: None,
        properties: vec![],
        geometry: Some(geometry),
    }
}
fn options() -> Options {
    Options {
        buffer: 0.0,
        tolerance: 0.0,
        ..Options::default()
    }
}
fn frame(features: Vec<Feature>, options: Options, z: u8, x: i32, y: u32) -> TileFrame {
    Index::new(features, options)
        .unwrap()
        .tile(z, x, y)
        .unwrap()
}

#[test]
fn points_and_multi_geometries() {
    let f = frame(
        vec![
            feature(Geometry::Points(vec![[0.0, 0.0], [90.0, 0.0]])),
            feature(Geometry::Lines(vec![
                vec![[-90.0, 0.0], [0.0, 0.0]],
                vec![[0.0, 0.0], [90.0, 0.0]],
            ])),
            feature(Geometry::Polygons(vec![vec![vec![
                [-10.0, 10.0],
                [10.0, 10.0],
                [10.0, -10.0],
                [-10.0, -10.0],
                [-10.0, 10.0],
            ]]])),
        ],
        options(),
        0,
        0,
        0,
    );
    let l = f.layer(LAYER_NAME).unwrap();
    assert_eq!(l.len(), 3);
    assert_eq!(l.geometry_type(0), GeometryType::Point);
    assert_eq!(
        l.geometry(0)
            .parts()
            .next()
            .unwrap()
            .rings()
            .next()
            .unwrap(),
        &[[4096, 4096], [6144, 4096]]
    );
    assert_eq!(l.geometry(1).parts().len(), 2);
    assert_eq!(l.geometry(2).parts().len(), 1);
}

#[test]
fn simplification_preserves_endpoints_and_maxzoom_detail() {
    let f = feature(Geometry::Lines(vec![vec![
        [-90.0, 0.0],
        [-45.0, 0.00001],
        [0.0, 0.0],
        [45.0, 20.0],
        [90.0, 0.0],
    ]]));
    let mut i = Index::new(
        vec![f],
        Options {
            max_zoom: 2,
            tolerance: 6.0,
            buffer: 0.0,
            ..Options::default()
        },
    )
    .unwrap();
    let root = i.tile(0, 0, 0).unwrap();
    assert_eq!(root.layers[0].vertices().len(), 4);
    assert_eq!(root.layers[0].vertices().first(), Some(&[2048, 4096]));
    let detail = i.tile(2, 1, 1).unwrap();
    assert!(detail.layers[0].vertices().len() >= 2);
}

#[test]
fn clipping_splits_lines_and_tracks_metrics() {
    let line = feature(Geometry::Lines(vec![vec![
        [-90.0, 40.0],
        [90.0, 40.0],
        [-90.0, 20.0],
    ]]));
    let mut o = options();
    o.line_metrics = true;
    let f = frame(vec![line], o, 1, 0, 0);
    let l = &f.layers[0];
    assert_eq!(l.len(), 2);
    for p in l.vertices() {
        assert!(p[0] >= 0 && p[0] <= 8192);
    }
    assert_eq!(
        l.property("geojsonvt_clip_start").unwrap().get(0),
        Some(ValueRef::F64(0.0))
    );
    let Some(ValueRef::F64(end)) = l.property("geojsonvt_clip_end").unwrap().get(0) else {
        panic!()
    };
    assert!(end > 0.0 && end < 0.5);
}

#[test]
fn buffer_keeps_neighbors_and_clips_at_buffer_edge() {
    let make = || {
        vec![feature(Geometry::Lines(vec![vec![
            [-90.0, 40.0],
            [90.0, 40.0],
        ]]))]
    };
    let no = frame(make(), options(), 1, 0, 0);
    assert_eq!(no.layers[0].vertices().last().unwrap()[0], 8192);
    let buffered = frame(
        make(),
        Options {
            buffer: 128.0,
            ..options()
        },
        1,
        0,
        0,
    );
    assert_eq!(buffered.layers[0].vertices().last().unwrap()[0], 8320);
    let point = || vec![feature(Geometry::Points(vec![[1.0, 40.0]]))];
    assert_eq!(frame(point(), options(), 1, 0, 0).layers[0].len(), 0);
    assert_eq!(
        frame(
            point(),
            Options {
                buffer: 128.0,
                ..options()
            },
            1,
            0,
            0
        )
        .layers[0]
            .len(),
        1
    );
}

#[test]
fn wraps_dateline_and_tile_x() {
    let make = || vec![feature(Geometry::Points(vec![[-179.0, 40.0]]))];
    let mut i = Index::new(
        make(),
        Options {
            buffer: 128.0,
            ..options()
        },
    )
    .unwrap();
    let west = i.tile(1, 0, 0).unwrap();
    let east = i.tile(1, 1, 0).unwrap();
    assert_eq!(west.layers[0].len(), 1);
    assert_eq!(east.layers[0].len(), 1);
    assert!(east.layers[0].vertices()[0][0] > 8192);
    assert_eq!(east, i.tile(1, -1, 0).unwrap());
}

#[test]
fn polygon_winding_and_implicit_closure() {
    let outer = vec![
        [-20.0, -20.0],
        [20.0, -20.0],
        [20.0, 20.0],
        [-20.0, 20.0],
        [-20.0, -20.0],
    ];
    let hole = vec![
        [-5.0, -5.0],
        [5.0, -5.0],
        [5.0, 5.0],
        [-5.0, 5.0],
        [-5.0, -5.0],
    ];
    let f = frame(
        vec![feature(Geometry::Polygons(vec![vec![outer, hole]]))],
        options(),
        0,
        0,
        0,
    );
    let rings: Vec<_> = f.layers[0]
        .geometry(0)
        .parts()
        .next()
        .unwrap()
        .rings()
        .collect();
    assert_eq!(rings.len(), 2);
    let area = |r: &[[i32; 2]]| {
        r.iter()
            .zip(r.iter().cycle().skip(1))
            .map(|(a, b)| i64::from(a[0]) * i64::from(b[1]) - i64::from(b[0]) * i64::from(a[1]))
            .sum::<i64>()
    };
    assert!(area(rings[0]) > 0);
    assert!(area(rings[1]) < 0);
    assert_eq!(rings[0].len(), 4);
    let clipped = frame(
        vec![feature(Geometry::Polygons(vec![vec![vec![
            [-20.0, 20.0],
            [20.0, 20.0],
            [20.0, -20.0],
            [-20.0, -20.0],
            [-20.0, 20.0],
        ]]]))],
        options(),
        1,
        0,
        0,
    );
    assert!(
        clipped.layers[0]
            .vertices()
            .iter()
            .all(|p| p[0] <= 8192 && p[1] <= 8192)
    );
}

#[test]
fn generated_ids_include_null_features_and_expand_collections() {
    let features = vec![
        Feature {
            id: None,
            properties: vec![],
            geometry: None,
        },
        feature(Geometry::Collection(vec![
            Geometry::Points(vec![[0.0, 0.0]]),
            Geometry::Lines(vec![vec![[0.0, 0.0], [10.0, 0.0]]]),
        ])),
    ];
    let f = frame(
        features,
        Options {
            generate_id: true,
            ..options()
        },
        0,
        0,
        0,
    );
    assert_eq!(f.layers[0].ids().get(0), Some(ValueRef::U64(1)));
    assert_eq!(f.layers[0].ids().get(1), Some(ValueRef::U64(1)));
}

#[test]
fn promoted_string_ids_and_index_options() {
    let mut f = feature(Geometry::Points(vec![[0.0, 0.0]]));
    f.properties
        .push(("key".into(), Value::Str("hello".into())));
    let features = vec![f];
    let a = frame(
        features.clone(),
        Options {
            promote_id: Some("key".into()),
            index_max_zoom: 0,
            ..options()
        },
        1,
        1,
        1,
    );
    let b = frame(
        features,
        Options {
            promote_id: Some("key".into()),
            index_max_zoom: 4,
            index_max_points: 0,
            ..options()
        },
        1,
        1,
        1,
    );
    assert_eq!(a, b);
    assert_eq!(a.layers[0].ids().get(0), Some(ValueRef::Str("hello")));
    assert!(
        Index::new(
            vec![],
            Options {
                generate_id: true,
                promote_id: Some("id".into()),
                ..options()
            }
        )
        .is_err()
    );
}

/// Polar data on a polar plane: a cap written the way world datasets write
/// Antarctica (along the parallel, then down the antimeridian to the pole
/// and back) tiles as a disc around the pole, not a sliver or a band.
#[test]
fn polar_caps_tile_as_discs_on_polar_planes() {
    for (id, lat) in [
        ("UPSAntarcticWGS84Quad", -70.0),
        ("UPSArcticWGS84Quad", 70.0),
    ] {
        let tms = std::sync::Arc::new(orb_crs::TileMatrixSet::by_id(id).unwrap());
        let pole_lat = 90f64.copysign(lat);
        let mut ring: Vec<[f64; 2]> = (0..=8)
            .map(|i| [-180.0 + 45.0 * f64::from(i), lat])
            .collect();
        ring.extend([[180.0, pole_lat], [-180.0, pole_lat], [-180.0, lat]]);
        if lat < 0.0 {
            ring.reverse();
        }
        let f = frame(
            vec![feature(Geometry::Polygons(vec![vec![ring]]))],
            Options {
                plane: Some(tms.clone()),
                ..options()
            },
            0,
            0,
            0,
        );
        let l = f.layer(LAYER_NAME).unwrap();
        assert_eq!(l.len(), 1, "{id}");
        let geometry = l.geometry(0);
        let parts: Vec<_> = geometry.parts().collect();
        assert_eq!(parts.len(), 1, "{id}");
        let rings: Vec<_> = parts[0].rings().collect();
        assert_eq!(rings.len(), 1, "{id}");
        let ring = rings[0];

        let unit = |lon: f64, lat: f64| {
            tms.unit(tms.projection().forward(orb_geo::Geodetic::new(lon, lat))) * 8192.0
        };
        let pole = unit(0.0, pole_lat);
        let radius = unit(0.0, lat).distance(pole);
        assert!(radius < 4096.0, "{id}: the cap fits the tile");
        // Every vertex lies within the parallel's circle (the run down the
        // antimeridian and back is a spike of no area)...
        for &[x, y] in ring {
            let r = (f64::from(x) - pole.x).hypot(f64::from(y) - pole.y);
            assert!(r < radius + 1.5, "{id}: vertex {x},{y} at {r}");
        }
        // ...and the ring encloses the disc once.
        let area = (0..ring.len())
            .map(|i| {
                let [ax, ay] = ring[i].map(f64::from);
                let [bx, by] = ring[(i + 1) % ring.len()].map(f64::from);
                (ax * by - bx * ay) / 2.0
            })
            .sum::<f64>()
            .abs();
        let disc = std::f64::consts::PI * radius * radius;
        assert!(
            (area / disc - 1.0).abs() < 1e-3,
            "{id}: area {area} vs {disc}"
        );
    }
}

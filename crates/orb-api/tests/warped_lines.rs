//! Lines drawn through an anisotropic warp keep their width across and
//! along: geographic tiles (`WorldCRS84Quad`) on Web Mercator stretch y by
//! sec(lat), 6× at 81°N, and a meridian and a parallel there draw as wide
//! as at the equator.

use orb_api::{Map, MapOptions, Previews, Runtime, RuntimeOptions};

fn varint(mut n: u64, out: &mut Vec<u8>) {
    while n >= 128 {
        out.push(n as u8 | 128);
        n >>= 7;
    }
    out.push(n as u8);
}

/// A length-delimited protobuf field.
fn message(field: u64, bytes: &[u8], out: &mut Vec<u8>) {
    varint(field << 3 | 2, out);
    varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

fn uint(field: u64, n: u64, out: &mut Vec<u8>) {
    varint(field << 3, out);
    varint(n, out);
}

/// A vector tile whose layer `l` holds two-point lines.
fn tile(lines: &[[[i64; 2]; 2]]) -> Vec<u8> {
    let zigzag = |n: i64| ((n << 1) ^ (n >> 63)) as u64;
    let mut layer = Vec::new();
    uint(15, 2, &mut layer);
    message(1, b"l", &mut layer);
    for [a, b] in lines {
        let mut geometry = Vec::new();
        for n in [
            9,
            zigzag(a[0]),
            zigzag(a[1]),
            10,
            zigzag(b[0] - a[0]),
            zigzag(b[1] - a[1]),
        ] {
            varint(n, &mut geometry);
        }
        let mut feature = Vec::new();
        uint(3, 2, &mut feature);
        message(4, &geometry, &mut feature);
        message(2, &feature, &mut layer);
    }
    uint(5, 4096, &mut layer);
    let mut tile = Vec::new();
    message(3, &layer, &mut tile);
    tile
}

const SIZE: u32 = 512;
const LON: f64 = 30.5;
const CENTER: [f64; 2] = [LON, 55.0];

/// A point's pixel at zoom 1 (1024-pixel world) about `CENTER`.
fn pixel([lon, lat]: [f64; 2]) -> [f64; 2] {
    let y = |lat: f64| {
        (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0)
            .tan()
            .ln()
    };
    let world = 1024.0;
    [
        f64::from(SIZE) / 2.0 + (lon - CENTER[0]) / 360.0 * world,
        f64::from(SIZE) / 2.0 - (y(lat) - y(CENTER[1])) / std::f64::consts::TAU * world,
    ]
}

#[test]
fn lines_through_a_stretching_warp_keep_their_width() {
    // Zoom 0 of WorldCRS84Quad: two 180° tiles. A meridian in the eastern
    // one, and parallels near 81°N and the equator across both.
    let unit = |degrees: f64| (degrees / 180.0 * 4096.0).round() as i64;
    let x = unit(LON);
    let (north, equator) = (unit(90.0 - 81.25), unit(90.0 - 0.25));
    let lat = |y: i64| 90.0 - y as f64 / 4096.0 * 180.0;
    let parallels = [
        [[-64, north], [4160, north]],
        [[-64, equator], [4160, equator]],
    ];
    let dir = std::env::temp_dir().join(format!("orb-warped-lines-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("0/0")).unwrap();
    std::fs::write(dir.join("0/0/0.pbf"), tile(&parallels)).unwrap();
    let mut east = parallels.to_vec();
    east.push([[x, -64], [x, 4160]]);
    std::fs::write(dir.join("0/0/1.pbf"), tile(&east)).unwrap();

    let style = format!(
        r##"{{"version": 8, "center": [{}, {}], "zoom": 1,
            "sources": {{"g": {{"type": "vector", "maxzoom": 0, "tileMatrixSet": "WorldCRS84Quad",
              "tiles": ["file://{}/{{tileMatrix}}/{{tileRow}}/{{tileCol}}.pbf"]}}}},
            "layers": [
              {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "l", "type": "line", "source": "g", "source-layer": "l",
                "paint": {{"line-color": "#000000", "line-width": 2}}}}]}}"##,
        CENTER[0],
        CENTER[1],
        dir.display()
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut map = Map::new(
        &runtime,
        MapOptions {
            previews: Previews::ALL,
            ..MapOptions::default()
        },
    );
    map.set_viewport(f64::from(SIZE), f64::from(SIZE));
    map.set_style_json(&style).unwrap();
    let image = map.snapshot(SIZE, SIZE, 1.0).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let ink = |x: i64, y: i64| {
        let i = ((y * i64::from(SIZE) + x) * 4) as usize;
        f64::from(255 - image.pixels[i]) / 255.0
    };

    // The meridian's ink per row, across a band of rows (well clear of the
    // parallels), and a parallel's per column, clear of the meridian.
    let meridian = |from: f64, to: f64| {
        let [mx, _] = pixel([LON, 0.0]);
        let (top, bottom) = (pixel([LON, to])[1] as i64, pixel([LON, from])[1] as i64);
        let rows = top..=bottom;
        let total: f64 = rows
            .clone()
            .flat_map(|y| (mx as i64 - 12..=mx as i64 + 12).map(move |x| (x, y)))
            .map(|(x, y)| ink(x, y))
            .sum();
        total / rows.count() as f64
    };
    let parallel = |y: i64| {
        let [_, py] = pixel([0.0, lat(y)]);
        let columns = 40..140;
        let total: f64 = columns
            .clone()
            .flat_map(|x| (py as i64 - 12..=py as i64 + 12).map(move |y| (x, y)))
            .map(|(x, y)| ink(x, y))
            .sum();
        total / columns.count() as f64
    };
    let widths = [
        ("meridian at 79–80.5°N", meridian(79.0, 80.5)),
        ("meridian at 4–12°S", meridian(-12.0, -4.0)),
        ("parallel at 81.25°N", parallel(north)),
        ("parallel at 0.25°N", parallel(equator)),
    ];
    for (line, width) in widths {
        assert!(
            (width - 2.0).abs() < 0.15,
            "{line}: {width:.2} px, not 2: {widths:?}"
        );
    }
}

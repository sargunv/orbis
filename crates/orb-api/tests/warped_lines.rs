//! Lines drawn through an anisotropic warp keep their width across and
//! along: geographic tiles (`WorldCRS84Quad`) on Web Mercator stretch y by
//! sec(lat), 6× at 81°N, and a meridian and a parallel there draw as wide
//! as at the equator, and their joins meet as the same line's from Web
//! Mercator tiles. Web Mercator's whole-world tile draws on a geographic
//! plane.

mod common;

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

/// A vector tile whose layer `l` holds these lines.
fn tile(lines: &[&[[i64; 2]]]) -> Vec<u8> {
    let zigzag = |n: i64| ((n << 1) ^ (n >> 63)) as u64;
    let mut layer = Vec::new();
    uint(15, 2, &mut layer);
    message(1, b"l", &mut layer);
    for line in lines {
        let mut geometry = Vec::new();
        varint(9, &mut geometry);
        varint(zigzag(line[0][0]), &mut geometry);
        varint(zigzag(line[0][1]), &mut geometry);
        varint(2 | (line.len() as u64 - 1) << 3, &mut geometry);
        for pair in line.windows(2) {
            varint(zigzag(pair[1][0] - pair[0][0]), &mut geometry);
            varint(zigzag(pair[1][1] - pair[0][1]), &mut geometry);
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

/// A `size`-pixel square snapshot of `style`, with every preview on.
fn snapshot(style: &str, size: u32) -> Vec<u8> {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut map = Map::new(
        &runtime,
        MapOptions {
            previews: Previews::ALL,
            ..MapOptions::default()
        },
    );
    map.set_viewport(f64::from(size), f64::from(size));
    map.set_style_json(style).unwrap();
    map.snapshot(size, size, 1.0).unwrap().pixels
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
    let dir = common::Scratch::new("warped-lines");
    std::fs::create_dir_all(dir.join("0/0")).unwrap();
    let parallels: Vec<&[[i64; 2]]> = parallels.iter().map(|line| &line[..]).collect();
    std::fs::write(dir.join("0/0/0.pbf"), tile(&parallels)).unwrap();
    let meridian = [[x, -64], [x, 4160]];
    let mut east = parallels.clone();
    east.push(&meridian);
    std::fs::write(dir.join("0/0/1.pbf"), tile(&east)).unwrap();

    let style = format!(
        r##"{{"version": 8, "center": [{}, {}], "zoom": 1,
            "sources": {{"g": {{"type": "vector", "maxzoom": 0, "dev.sargunv.orbis:tileMatrixSet": "WorldCRS84Quad",
              "tiles": ["file://{}/{{tileMatrix}}/{{tileRow}}/{{tileCol}}.pbf"]}}}},
            "layers": [
              {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "l", "type": "line", "source": "g", "source-layer": "l",
                "paint": {{"line-color": "#000000", "line-width": 2}}}}]}}"##,
        CENTER[0],
        CENTER[1],
        dir.display()
    );
    let pixels = snapshot(&style, SIZE);
    let ink = |x: i64, y: i64| {
        let i = ((y * i64::from(SIZE) + x) * 4) as usize;
        f64::from(255 - pixels[i]) / 255.0
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

#[test]
fn joins_through_a_stretching_warp_meet_on_screen() {
    // A staircase near 81°N, where WorldCRS84Quad stretches y 6×, drawn
    // from geographic tiles and from a Web Mercator tile, whose plane is
    // the map's. Its corners lie on both tiles' grids in x and on the
    // geographic one in y, a third of a pixel from the Mercator one's.
    let (lons, lats) = ([228, 284, 341, 398, 455], [228, 210, 193, 176]);
    let corners: Vec<(i64, i64)> = (0..4)
        .flat_map(|i| [(lons[i], lats[i]), (lons[i + 1], lats[i])])
        .collect();
    let geographic: Vec<[i64; 2]> = corners.iter().map(|&(x, y)| [2 * x, y]).collect();
    let mercator: Vec<[i64; 2]> = corners
        .iter()
        .map(|&(x, y)| {
            // In tile 3/4/0.
            let lat = (90.0 - y as f64 * 180.0 / 4096.0).to_radians();
            let y = (std::f64::consts::FRAC_PI_4 + lat / 2.0).tan().ln();
            let row = (1.0 - y / std::f64::consts::PI) * 4.0;
            [8 * x, (row * 4096.0).round() as i64]
        })
        .collect();
    let dir = common::Scratch::new("warped-joins");
    std::fs::create_dir_all(dir.join("g/0/0")).unwrap();
    std::fs::write(dir.join("g/0/0/1.pbf"), tile(&[&geographic])).unwrap();
    std::fs::create_dir_all(dir.join("m/3/4")).unwrap();
    std::fs::write(dir.join("m/3/4/0.pbf"), tile(&[&mercator])).unwrap();

    let draw = |source: &str, line: &str| {
        snapshot(
            &format!(
                r##"{{"version": 8, "center": [30, 81.2], "zoom": 3.5,
                "sources": {{"s": {source}}},
                "layers": [
                  {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
                  {{"id": "l", "type": "line", "source": "s", "source-layer": "l", {line}}}]}}"##
            ),
            SIZE,
        )
    };
    let geographic = format!(
        r#"{{"type": "vector", "maxzoom": 0, "dev.sargunv.orbis:tileMatrixSet": "WorldCRS84Quad",
            "tiles": ["file://{}/g/{{tileMatrix}}/{{tileRow}}/{{tileCol}}.pbf"]}}"#,
        dir.display()
    );
    let mercator = format!(
        r#"{{"type": "vector", "maxzoom": 3, "tiles": ["file://{}/m/{{z}}/{{x}}/{{y}}.pbf"]}}"#,
        dir.display()
    );
    // Through the bisector's frame alone, the joins missed by 5.7%, 3.5%
    // and 18.9% of the line's ink.
    let cases = [
        ("miter", r#""paint": {"line-width": 10}"#, 0.015),
        (
            "bevel",
            r#""layout": {"line-join": "bevel"}, "paint": {"line-width": 10}"#,
            0.015,
        ),
        (
            "offset",
            r#""paint": {"line-width": 4, "line-offset": 6}"#,
            0.04,
        ),
    ];
    let mut misses = Vec::new();
    for (name, line, limit) in cases {
        let (warped, flat) = (draw(&geographic, line), draw(&mercator, line));
        let ink = |pixels: &[u8], i: usize| f64::from(255 - pixels[i]) / 255.0;
        let (mut miss, mut total) = (0.0, 0.0);
        for i in (0..warped.len()).step_by(4) {
            miss += (ink(&warped, i) - ink(&flat, i)).abs();
            total += ink(&flat, i);
        }
        misses.push((name, miss / total, limit));
    }
    for &(name, miss, limit) in &misses {
        assert!(
            miss < limit,
            "{name}: {:.1}% of ink apart: {misses:?}",
            100.0 * miss
        );
    }
}

/// Web Mercator's z0 tile spans the whole world: on a geographic plane,
/// which wraps at ±180°, its two edges lie half a period from its centre.
/// Taking them into one world copy folded the right edge onto the left,
/// so it had no patches to draw through and views at zoom 1 and below,
/// which cover it, drew nothing.
#[test]
fn whole_world_mercator_tile_draws_on_a_geographic_plane() {
    let dir = common::Scratch::new("warped-world");
    std::fs::create_dir_all(dir.join("0/0")).unwrap();
    // The equator and the prime meridian, end to end.
    let cross = tile(&[&[[0, 2048], [4096, 2048]], &[[2048, 0], [2048, 4096]]]);
    std::fs::write(dir.join("0/0/0.pbf"), cross).unwrap();
    let center = [40.0, 0.0];
    let draw = |zoom: f64| {
        let style = format!(
            r##"{{"version": 8, "projection": {{"type": "EPSG:4326"}},
                "center": [{}, {}], "zoom": {zoom},
                "sources": {{"m": {{"type": "vector", "maxzoom": 0,
                  "tiles": ["file://{}/{{z}}/{{x}}/{{y}}.pbf"]}}}},
                "layers": [
                  {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
                  {{"id": "l", "type": "line", "source": "m", "source-layer": "l",
                    "paint": {{"line-color": "#000000", "line-width": 2}}}}]}}"##,
            center[0],
            center[1],
            dir.display()
        );
        snapshot(&style, SIZE)
    };
    let mut misses = Vec::new();
    for zoom in [0.0, 0.5, 1.0, 1.5] {
        let pixels = draw(zoom);
        let size = i64::from(SIZE);
        let inked = |x: i64, y: i64| {
            (0..size).contains(&x)
                && (0..size).contains(&y)
                && pixels[((y * size + x) * 4) as usize] < 128
        };
        // A position's pixel: the plane's degrees are square, 360° to 512
        // pixels at zoom 0.
        let per_px = 360.0 / (512.0 * zoom.exp2());
        let half = f64::from(SIZE) / 2.0;
        let px = |lon: f64, lat: f64| {
            (
                (half + (lon - center[0]) / per_px).round() as i64,
                (half - (lat - center[1]) / per_px).round() as i64,
            )
        };
        let near = |x: i64, y: i64| (-2..=2).any(|d| inked(x + d, y) || inked(x, y + d));
        // Samples along both lines in view, and those that drew.
        let (mut seen, mut drawn) = (0, 0);
        for k in -40..=40 {
            let along = f64::from(k) * 2.0;
            for (x, y) in [px(0.0, along), px(along * 2.0, 0.0)] {
                if (4..size - 4).contains(&x) && (4..size - 4).contains(&y) {
                    seen += 1;
                    drawn += i32::from(near(x, y));
                }
            }
        }
        if drawn < seen {
            misses.push((zoom, drawn, seen));
        }
    }
    assert!(misses.is_empty(), "(zoom, drawn, in view): {misses:?}");
}

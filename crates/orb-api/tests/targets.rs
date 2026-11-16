//! Render targets (the `dev.sargunv.orbis:targets` preview): what layers
//! draw into a target, what image-process layers make of it, and the fills
//! drawn through it.

mod common;

use orb_api::{CameraTarget, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

fn options() -> MapOptions {
    MapOptions {
        previews: orb_api::Previews::ALL,
        ..MapOptions::default()
    }
}

/// What a map of `style` draws once idle at `center` and `zoom`.
fn draw(
    runtime: &Runtime,
    target: &mut Target,
    style: &str,
    center: Geodetic,
    zoom: f64,
) -> Vec<u8> {
    let mut map = Map::new(runtime, options());
    map.set_style_json(style).unwrap();
    map.jump_to(&CameraTarget {
        center: Some(center),
        zoom: Some(zoom),
        ..CameraTarget::default()
    });
    common::settle(&mut map, target);
    target.read_pixels().unwrap()
}

/// A style over a white background with a triangle as GeoJSON, `targets`
/// declared and `layers` after the background.
fn triangle(targets: &str, layers: &str) -> String {
    format!(
        r##"{{"version": 8,
            "dev.sargunv.orbis:targets": [{targets}],
            "sources": {{"shape": {{"type": "geojson", "data": {{"type": "Polygon",
              "coordinates": [[[-1, -1], [1, -1], [0, 1], [-1, -1]]]}}}}}},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {layers}
            ]}}"##
    )
}

/// The largest difference of any channel between two pictures.
fn largest_difference(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// A fill drawn through a target's coverage draws what the layers drawn
/// into it cover, in its own colour, as the layer itself would draw it; an
/// image-process layer copying the target changes nothing, and one
/// inverting it covers the rest.
#[test]
fn a_fill_through_a_target_covers_what_was_drawn_into_it() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    let mut draw = |style: String| draw(&runtime, &mut target, &style, at.0, at.1);
    let shape = |color: &str, extra: &str| {
        format!(
            r#"{{"id": "shape", "type": "fill", "source": "shape", {extra}
                "paint": {{"fill-color": "{color}"}}}}"#
        )
    };
    let direct = draw(triangle("", &shape("#d02060", "")));
    let red = |id: &str, target: &str| {
        format!(
            r##"{{"id": "{id}", "type": "fill", "dev.sargunv.orbis:source-target": "{target}",
                "paint": {{"fill-color": "#d02060"}}}}"##
        )
    };
    let into = r#""dev.sargunv.orbis:destination-target": "drawn","#;
    let through = draw(triangle(
        r#"{"id": "drawn"}"#,
        &format!("{}, {}", shape("#123456", into), red("through", "drawn")),
    ));
    assert!(
        largest_difference(&direct, &through) <= 2,
        "through a target, the fill differs from the layer drawn directly"
    );
    let process = |ops: &str| {
        format!(
            r#"{{"id": "process", "type": "dev.sargunv.orbis.image-process",
                "dev.sargunv.orbis:source-target": "drawn",
                "dev.sargunv.orbis:destination-target": "processed",
                "dev.sargunv.orbis:operations": [{ops}]}}"#
        )
    };
    let targets = r#"{"id": "drawn"}, {"id": "processed"}"#;
    let copied = draw(triangle(
        targets,
        &format!(
            "{}, {}, {}",
            shape("#123456", into),
            process(""),
            red("through", "processed")
        ),
    ));
    assert!(
        largest_difference(&direct, &copied) <= 2,
        "a copy of the target differs from it"
    );
    let inverted = draw(triangle(
        targets,
        &format!(
            "{}, {}, {}",
            shape("#123456", into),
            process(r#"{"type": "invert"}"#),
            red("through", "processed")
        ),
    ));
    // Where the triangle covers a pixel wholly or not at all, the inverse
    // covers it not at all or wholly.
    let (mut inside, mut outside) = (0, 0);
    for (d, i) in direct.chunks(4).zip(inverted.chunks(4)) {
        if d[..3] == [255; 3] {
            outside += 1;
            assert_eq!(&i[..3], &[0xd0, 0x20, 0x60], "outside the triangle");
        } else if d[..3] == [0xd0, 0x20, 0x60] {
            inside += 1;
            assert_eq!(&i[..3], &[255; 3], "inside the triangle");
        }
    }
    let pixels = (WIDTH * HEIGHT) as usize;
    assert!(
        inside > pixels / 8 && outside > pixels / 8,
        "{inside} in, {outside} out"
    );
}

/// An image-process layer's noise is fixed to the map: a view panned by
/// whole pixels draws the same picture moved by them, but within a blur of
/// the view's edges, which see nothing beyond them.
#[test]
fn noise_moves_with_the_map() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let tiles = format!(
        "pmtiles://file://{}/tokyo-mvt.pmtiles",
        common::data().display()
    );
    let source = format!(r#"{{"type": "vector", "url": "{tiles}"}}"#);
    let style = format!(
        r##"{{"version": 8,
            "dev.sargunv.orbis:targets": [{{"id": "paint"}}, {{"id": "land"}}],
            "sources": {{"openmaptiles": {source}}},
            "layers": [
              {{"id": "paint", "type": "background", "dev.sargunv.orbis:destination-target": "paint",
                "paint": {{"background-color": "#ffff00"}}}},
              {{"id": "roads", "type": "line", "source": "openmaptiles", "source-layer": "transportation",
                "dev.sargunv.orbis:destination-target": "paint",
                "paint": {{"line-color": "#ffffff", "line-width": 4}}}},
              {{"id": "mask", "type": "dev.sargunv.orbis.image-process",
                "dev.sargunv.orbis:source-target": "paint", "dev.sargunv.orbis:destination-target": "land",
                "dev.sargunv.orbis:operations": [
                  {{"type": "select", "value": "#ffff00"}}, {{"type": "blur", "size": 2}},
                  {{"type": "noise", "size": 6, "amount": 0.4}}, {{"type": "threshold"}}]}},
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "land", "type": "fill", "dev.sargunv.orbis:source-target": "land",
                "paint": {{"fill-color": "#d02060"}}}}
            ]}}"##
    );
    let zoom: f64 = 15.3;
    let center = Geodetic::new(139.74, 35.687);
    // 40 pixels east at this zoom.
    let shift = 40;
    let degrees = f64::from(shift) * 360.0 / (512.0 * zoom.exp2());
    let east = Geodetic::new(center.lon + degrees, center.lat);
    let a = draw(&runtime, &mut target, &style, center, zoom);
    let b = draw(&runtime, &mut target, &style, east, zoom);
    let margin = 16;
    let (mut red, mut white, mut off) = (0, 0, Vec::new());
    for y in margin..HEIGHT - margin {
        for x in margin + shift..WIDTH - margin {
            let i = ((y * WIDTH + x) * 4) as usize;
            let j = ((y * WIDTH + x - shift) * 4) as usize;
            let (pa, pb) = (&a[i..i + 3], &b[j..j + 3]);
            red += usize::from(pa == [0xd0, 0x20, 0x60]);
            white += usize::from(pa == [255; 3]);
            if largest_difference(pa, pb) > 3 {
                off.push(format!("({x}, {y}): {pa:?} then {pb:?}"));
            }
        }
    }
    assert!(
        off.is_empty(),
        "{} pixels moved otherwise: {:?}",
        off.len(),
        &off[..off.len().min(8)]
    );
    assert!(
        red > 1000 && white > 1000,
        "{red} land pixels, {white} road pixels"
    );
}

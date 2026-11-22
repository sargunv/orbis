//! Render targets (the `dev.sargunv.orbis:targets` preview): what layers
//! draw into a target, what image-process layers make of it, and the fills
//! drawn through it.

mod common;

use orb_api::{
    CameraTarget, Constraints, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target,
};

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

#[test]
fn noise_respects_its_anchor_across_zoom() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    // The centre pixel and every other pixel at the higher zoom sample
    // exactly the same locations as pixels in the lower zoom view.
    const SIDE: u32 = 129;
    const MID: u32 = SIDE / 2;
    let mut target = Target::texture(&runtime, SIDE, SIDE, 1.0);
    let style = |anchor: &str, reference: &str| {
        format!(
            r##"{{"version": 8,
                "dev.sargunv.orbis:targets": [{{"id": "half"}}],
                "sources": {{}},
                "layers": [
                    {{"id": "half", "type": "background", "dev.sargunv.orbis:destination-target": "half",
                      "paint": {{"background-color": "#000000", "background-opacity": 0.5}}}},
                    {{"id": "paper", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
                    {{"id": "grain", "type": "dev.sargunv.orbis.image-process",
                      "dev.sargunv.orbis:source-target": "half", "dev.sargunv.orbis:operations": [
                        {{"type": "noise", "size": 12, "amount": 0.5,
                          "dev.sargunv.orbis:anchor": "{anchor}"{reference}}}
                      ]}}
                ]}}"##
        )
    };
    let thames = Geodetic::new(-0.1122, 51.5095);
    let map = style("map", r#", "dev.sargunv.orbis:reference-zoom": 20"#);
    let near = draw(&runtime, &mut target, &map, thames, 20.0);
    let closer = draw(&runtime, &mut target, &map, thames, 21.0);
    let pixel = |im: &[u8], x, y| im[((y * SIDE + x) * 4) as usize];
    assert_ne!(near, closer, "map grain did not grow with the map");
    for y in 0..=MID {
        for x in 0..=MID {
            let a = pixel(&near, x + MID / 2, y + MID / 2);
            let b = pixel(&closer, x * 2, y * 2);
            assert!(a.abs_diff(b) <= 1, "grain moved on the map at ({x}, {y})");
        }
    }

    let viewport = style("viewport", "");
    let london = draw(&runtime, &mut target, &viewport, thames, 14.2);
    let tokyo = Geodetic::new(139.74, 35.687);
    let tokyo = draw(&runtime, &mut target, &viewport, tokyo, 20.2);
    assert_ne!(pixel(&london, 0, 0), pixel(&london, 5, 5));
    assert_eq!(london, tokyo, "viewport grain moved with the map");
}

/// What an image-process layer's operations make of a triangle drawn into
/// a target, drawn over the map: moved by an offset; nothing, multiplied by
/// its own inverse; its colour from a ramp; grown by a dilation and shrunk
/// by an erosion.
#[test]
fn operations_move_cut_colour_and_grow() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    let mut draw = |ops: &str| {
        let layers = format!(
            r##"{{"id": "shape", "type": "fill", "source": "shape",
                 "dev.sargunv.orbis:destination-target": "drawn", "paint": {{"fill-color": "#d02060"}}}},
               {{"id": "inverse", "type": "dev.sargunv.orbis.image-process",
                 "dev.sargunv.orbis:source-target": "drawn", "dev.sargunv.orbis:destination-target": "inverse",
                 "dev.sargunv.orbis:operations": [{{"type": "invert"}}]}},
               {{"id": "process", "type": "dev.sargunv.orbis.image-process",
                 "dev.sargunv.orbis:source-target": "drawn", "dev.sargunv.orbis:operations": [{ops}]}}"##
        );
        let style = triangle(r#"{"id": "drawn"}, {"id": "inverse"}"#, &layers);
        draw(&runtime, &mut target, &style, at.0, at.1)
    };
    let plain = draw("");
    let red = |p: &[u8]| p[..3] == [0xd0, 0x20, 0x60];
    let pixel = |image: &[u8], x: u32, y: u32| -> [u8; 4] {
        let i = ((y * WIDTH + x) * 4) as usize;
        image[i..i + 4].try_into().unwrap()
    };
    let covered =
        |image: &[u8], test: &dyn Fn(&[u8]) -> bool| image.chunks(4).filter(|p| test(p)).count();
    let inside = covered(&plain, &red);
    assert!(inside > 2000, "{inside} pixels inside the triangle");

    // Moved 12 right and 6 down.
    let moved = draw(r#"{"type": "offset", "translate": [12, 6]}"#);
    let mut off = 0;
    for y in 8..HEIGHT - 8 {
        for x in 16..WIDTH - 16 {
            off += usize::from(
                largest_difference(&pixel(&moved, x, y), &pixel(&plain, x - 12, y - 6)) > 3,
            );
        }
    }
    assert_eq!(off, 0, "pixels not moved with the offset");

    // Nothing is both the triangle and its inverse but its edge's pixels,
    // a quarter covered at most (half times half).
    let cut = draw(r#"{"type": "combine", "target": "inverse", "mode": "multiply"}"#);
    let quarter = (255 - 0x20) / 4 + 2;
    let most = cut.chunks(4).map(|p| 255 - p[1]).max().unwrap_or(0);
    assert!(
        most <= quarter,
        "the triangle times its inverse covers {most} of 255"
    );

    // A ramp from its coverage: wholly covered becomes green.
    let ramped = draw(r##"{"type": "ramp", "stops": [[0, "rgba(0, 0, 0, 0)"], [1, "#00ff00"]]}"##);
    assert_eq!(covered(&ramped, &|p| p[..3] == [0, 255, 0]), inside);

    // Grown by 4 pixels, then shrunk by 4 from the original.
    let grown = covered(&draw(r#"{"type": "dilate", "size": 4}"#), &red);
    let shrunk = covered(&draw(r#"{"type": "erode", "size": 4}"#), &red);
    assert!(
        grown > inside + 400 && shrunk + 400 < inside,
        "{shrunk} < {inside} < {grown}"
    );
}

/// A fill drawn through a target in a blend mode blends where the target
/// covers and leaves the map beneath elsewhere: multiplied over a
/// coloured ground, its colour times the ground's inside the triangle.
#[test]
fn a_fill_through_a_target_blends_in_its_mode() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let layers = r##"{"id": "shape", "type": "fill", "source": "shape",
          "dev.sargunv.orbis:destination-target": "drawn", "paint": {"fill-color": "#fff"}},
        {"id": "tint", "type": "background", "paint": {"background-color": "#80c0ff"}},
        {"id": "through", "type": "fill", "dev.sargunv.orbis:source-target": "drawn",
          "paint": {"fill-color": "#ff8000", "fill-layer-blend": "multiply"}}"##;
    let style = triangle(r#"{"id": "drawn"}"#, layers);
    let image = draw(&runtime, &mut target, &style, Geodetic::new(0.0, 0.0), 6.0);
    let multiplied = [0x80, 0x60, 0x00];
    let (mut inside, mut outside) = (0, 0);
    for p in image.chunks(4) {
        if largest_difference(&p[..3], &multiplied) <= 2 {
            inside += 1;
        } else if p[..3] == [0x80, 0xc0, 0xff] {
            outside += 1;
        }
    }
    let pixels = (WIDTH * HEIGHT) as usize;
    assert!(
        inside > pixels / 8 && outside > pixels / 8,
        "{inside} multiplied, {outside} as the ground"
    );
    assert!(
        inside + outside > pixels * 9 / 10,
        "{} pixels neither",
        pixels - inside - outside
    );
}

/// A style over a black ground with a white triangle drawn into target
/// `drawn` and an image-process layer of `ops` reading it, drawn over the
/// map.
fn processed_triangle(ops: &str) -> String {
    let layers = format!(
        r##"{{"id": "dark", "type": "background", "paint": {{"background-color": "#000000"}}}},
           {{"id": "shape", "type": "fill", "source": "shape",
             "dev.sargunv.orbis:destination-target": "drawn", "paint": {{"fill-color": "#ffffff"}}}},
           {{"id": "process", "type": "dev.sargunv.orbis.image-process",
             "dev.sargunv.orbis:source-target": "drawn", "dev.sargunv.orbis:operations": [{ops}]}}"##
    );
    triangle(r#"{"id": "drawn"}"#, &layers)
}

/// [`processed_triangle`] with the image-process layer reading a target
/// nothing writes: what its operations make of nothing.
fn over_nothing(ops: &str) -> String {
    processed_triangle(ops)
        .replace(r#"{"id": "drawn"}"#, r#"{"id": "drawn"}, {"id": "empty"}"#)
        .replace(
            r#""dev.sargunv.orbis:source-target": "drawn""#,
            r#""dev.sargunv.orbis:source-target": "empty""#,
        )
}

/// Channel `channel` of the pixel at (`x`, `y`).
fn channel(image: &[u8], x: u32, y: u32, channel: usize) -> u8 {
    image[((y * WIDTH + x) * 4) as usize + channel]
}

/// A split moves each colour channel by its own offset: red right, blue
/// left, green not at all. Radial, the moves shrink towards the view's
/// centre.
#[test]
fn a_split_moves_each_channel_apart() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    let mut draw = |ops: &str| draw(&runtime, &mut target, &processed_triangle(ops), at.0, at.1);
    let plain = draw("");
    let split = draw(r#"{"type": "split", "red": [10, 0], "blue": [-10, 0]}"#);
    let mut off = 0;
    for y in 4..HEIGHT - 4 {
        for x in 12..WIDTH - 12 {
            let expected = [
                channel(&plain, x - 10, y, 0),
                channel(&plain, x, y, 1),
                channel(&plain, x + 10, y, 2),
            ];
            let found = [0, 1, 2].map(|c| channel(&split, x, y, c));
            off += usize::from(largest_difference(&expected, &found) > 2);
        }
    }
    assert_eq!(off, 0, "pixels whose channels didn't move by their offsets");
    // Where the triangle's left edge crosses the middle row, by its red.
    let edge = |image: &[u8]| (0..WIDTH).find(|&x| channel(image, x, HEIGHT / 2, 0) > 127);
    let radial = draw(r#"{"type": "split", "red": [16, 0], "radial": true}"#);
    let (plain_edge, radial_edge) = (edge(&plain).unwrap(), edge(&radial).unwrap());
    // The edge is about a quarter of the way from the centre to a corner.
    let distance =
        f64::from(WIDTH / 2 - plain_edge) / (f64::from(WIDTH) / 2.0).hypot(f64::from(HEIGHT) / 2.0);
    let moved = f64::from(radial_edge - plain_edge);
    assert!(
        (moved - 16.0 * distance).abs() <= 1.5,
        "the edge {distance:.2} of the way out moved {moved} pixels of 16"
    );
}

/// A linear gradient runs from none at one edge of the view to whole at
/// the other, along its angle; a radial one from none at the centre to
/// whole at the corners. It needs nothing drawn beneath it: it reads a
/// target nothing writes.
#[test]
fn a_gradient_runs_across_the_view() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    let mut draw = |ops: &str| draw(&runtime, &mut target, &over_nothing(ops), at.0, at.1);
    let (last, middle) = (WIDTH - 1, WIDTH / 2);
    let near = |found: u8, expected: u8| found.abs_diff(expected) <= 2;
    let down = draw(r#"{"type": "gradient"}"#);
    for x in [0, middle, last] {
        let column: Vec<u8> = (0..HEIGHT).map(|y| channel(&down, x, y, 1)).collect();
        assert!(
            near(column[0], 0) && near(column[HEIGHT as usize - 1], 255),
            "column {x} runs {} to {}",
            column[0],
            column[HEIGHT as usize - 1]
        );
        assert!(
            near(column[middle as usize], 128),
            "{}",
            column[middle as usize]
        );
        assert!(
            column.windows(2).all(|w| w[0] <= w[1]),
            "column {x} isn't monotonic"
        );
    }
    // Turned a quarter clockwise, it runs right to left.
    let across = draw(r#"{"type": "gradient", "kind": "linear", "angle": 90}"#);
    let (right, left) = (
        channel(&across, last, middle, 1),
        channel(&across, 0, middle, 1),
    );
    assert!(
        near(right, 0) && near(left, 255),
        "{right} at the right, {left} at the left"
    );
    let radial = draw(r#"{"type": "gradient", "kind": "radial"}"#);
    let centre = channel(&radial, middle, middle, 1);
    let corners =
        [(0, 0), (last, 0), (0, last), (last, last)].map(|(x, y)| channel(&radial, x, y, 1));
    assert!(
        near(centre, 0) && corners.iter().all(|&c| near(c, 255)),
        "{centre} at the centre, {corners:?} at the corners"
    );
}

/// Bands over a gradient make as many stripes as they count, each the
/// width they give.
#[test]
fn bands_make_stripes() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    let mut draw = |ops: &str| draw(&runtime, &mut target, &processed_triangle(ops), at.0, at.1);
    for (count, width) in [(4, 0.5), (8, 0.25)] {
        let image = draw(&format!(
            r#"{{"type": "gradient"}}, {{"type": "bands", "count": {count}, "width": {width}}}"#
        ));
        let column: Vec<bool> = (0..HEIGHT)
            .map(|y| channel(&image, WIDTH / 2, y, 1) > 127)
            .collect();
        let stripes = column.windows(2).filter(|w| w[1] && !w[0]).count() + usize::from(column[0]);
        let covered = column.iter().filter(|&&c| c).count() as f64 / f64::from(HEIGHT);
        assert_eq!(stripes, count, "{count} bands {width} wide");
        assert!(
            (covered - width).abs() < 0.03,
            "{count} bands {width} wide cover {covered:.3} of the view"
        );
    }
}

/// Noise with a speed changes with time alone, so the map keeps drawing
/// while it shows, at rest but for that, which a wait for the map to
/// settle takes as settled, and stops once it doesn't; still noise idles.
#[test]
fn moving_noise_keeps_the_map_drawing() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let style = |speed: f64| {
        processed_triangle(&format!(
            r#"{{"type": "blur", "size": 8}}, {{"type": "noise", "size": 8, "amount": 0.5, "speed": {speed}}},
                {{"type": "threshold"}}"#
        ))
    };
    let map_of = |style: &str| {
        let mut map = Map::new(&runtime, options());
        map.set_style_json(style).unwrap();
        map.jump_to(&CameraTarget {
            center: Some(Geodetic::new(0.0, 0.0)),
            zoom: Some(6.0),
            ..CameraTarget::default()
        });
        map
    };
    // What a map draws at `time`, and whether it was idle.
    let frame = |map: &mut Map, target: &mut Target, time: f64| {
        let report = map.render(target, time).unwrap().expect("a frame");
        (target.read_pixels().unwrap(), report)
    };
    let mut map = map_of(&style(40.0));
    let mut time = 0.0;
    // Frames until everything has loaded, a few in a row.
    let mut loaded = 0;
    while loaded < 3 {
        let (_, report) = frame(&mut map, &mut target, time);
        assert!(time < 60_000.0, "the map didn't load");
        loaded = if report.loaded { loaded + 1 } else { 0 };
        std::thread::sleep(std::time::Duration::from_millis(16));
        time += 16.0;
    }
    let (a, report) = frame(&mut map, &mut target, time);
    assert!(!report.idle, "a map with moving noise idles");
    assert!(
        report.animating,
        "a loaded map with moving noise isn't animating"
    );
    let settled = map.render_until_idle(&mut target, std::time::Duration::from_secs(60));
    assert!(settled.unwrap().animating);
    let (b, _) = frame(&mut map, &mut target, time + 500.0);
    let (again, _) = frame(&mut map, &mut target, time);
    assert_eq!(a, again, "the same time draws differently");
    let changed = a.chunks(4).zip(b.chunks(4)).filter(|(p, q)| p != q).count();
    assert!(
        changed > (WIDTH * HEIGHT) as usize / 50,
        "half a second later, {changed} pixels changed"
    );
    // Hidden, the noise no longer keeps the map drawing.
    let hide = orb_api::Transaction::from_commands(
        r#"[["setLayoutProperty", "process", "visibility", "none"]]"#,
    )
    .unwrap();
    map.edit(&hide).unwrap();
    let report = (map.render_until_idle(&mut target, std::time::Duration::from_secs(60))).unwrap();
    assert!(report.idle && !report.animating);
    // Still noise idles, and draws the same at any time.
    let mut still = map_of(&style(0.0));
    common::settle(&mut still, &mut target);
    let (a, report) = frame(&mut still, &mut target, 1000.0);
    assert!(report.idle);
    let (b, _) = frame(&mut still, &mut target, 5000.0);
    assert_eq!(a, b, "still noise changed with time");
}

/// A map of `style` at (0, 0), zoom 6 and `camera`'s bearing and pitch,
/// drawn once idle: the same map each call, moved.
fn view(map: &mut Map, target: &mut Target, (bearing, pitch): (f64, f64)) -> Vec<u8> {
    map.jump_to(&CameraTarget {
        center: Some(Geodetic::new(0.0, 0.0)),
        zoom: Some(6.0),
        bearing: Some(bearing),
        pitch: Some(pitch),
        ..CameraTarget::default()
    });
    common::settle(map, target);
    target.read_pixels().unwrap()
}

/// An operation's values may follow the camera, and change as it moves
/// with the zoom kept: a linear gradient whose angle is the bearing's
/// negative turns with the map, its none on the north side wherever that
/// is; a colour that follows the pitch changes as the map tilts.
#[test]
fn operations_follow_the_camera() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let style = over_nothing(
        r##"{"type": "gradient", "angle": ["*", -1, ["bearing"]]},
            {"type": "color", "value": ["interpolate", ["linear"], ["pitch"], 0, "#00ff00", 60, "#ff00ff"]}"##,
    );
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    let (last, middle) = (WIDTH - 1, WIDTH / 2);
    let near = |found: u8, expected: u8| found.abs_diff(expected) <= 2;
    let north_up = view(&mut map, &mut target, (0.0, 0.0));
    let (top, bottom) = (
        channel(&north_up, middle, 0, 1),
        channel(&north_up, middle, last, 1),
    );
    assert!(
        near(top, 0) && near(bottom, 255),
        "north up: {top} at the top, {bottom} at the bottom"
    );
    // East up: north is on the left.
    let east_up = view(&mut map, &mut target, (90.0, 0.0));
    let (left, right) = (
        channel(&east_up, 0, middle, 1),
        channel(&east_up, last, middle, 1),
    );
    assert!(
        near(left, 0) && near(right, 255),
        "east up: {left} at the left, {right} at the right"
    );
    // Tilted, magenta.
    let tilted = view(&mut map, &mut target, (0.0, 60.0));
    let bottom: Vec<u8> = (0..3).map(|c| channel(&tilted, middle, last, c)).collect();
    assert!(
        near(bottom[0], 255) && near(bottom[1], 0) && near(bottom[2], 255),
        "tilted, the bottom is {bottom:?}"
    );
}

/// A ramp's stops may follow the zoom: over a gradient down the view, a
/// hard step from red to green at a coverage that follows the zoom falls
/// a quarter of the way down at one zoom and three quarters at another.
#[test]
fn ramp_stops_follow_the_zoom() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let style = over_nothing(
        r##"{"type": "gradient"},
            {"type": "ramp", "stops": [[0, "#ff0000"],
              [["interpolate", ["linear"], ["zoom"], 5, 0.25, 7, 0.75], "#ff0000"],
              [["interpolate", ["linear"], ["zoom"], 5, 0.25, 7, 0.75], "#00ff00"],
              [1, "#00ff00"]]}"##,
    );
    for (zoom, row) in [(5.0, HEIGHT / 4), (7.0, HEIGHT * 3 / 4)] {
        let image = draw(&runtime, &mut target, &style, Geodetic::new(0.0, 0.0), zoom);
        let green = (0..HEIGHT).find(|&y| channel(&image, WIDTH / 2, y, 1) > 127);
        assert!(
            green.is_some_and(|y| y.abs_diff(row) <= 1),
            "at zoom {zoom}, green from row {green:?}, not {row}"
        );
    }
}

/// A gradient may be placed and sized in pixels: a radial one about a
/// centre off the view's, whole at its radius; a linear one through a
/// centre, none to whole over its length; and by default a radial one
/// off centre whole at the view's farthest corner from it.
#[test]
fn gradients_are_placed_and_sized_in_pixels() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    let mut draw = |ops: &str| draw(&runtime, &mut target, &over_nothing(ops), at.0, at.1);
    let half = [f64::from(WIDTH) / 2.0, f64::from(HEIGHT) / 2.0];
    // Each pixel against what it should cover, at its centre.
    let check = |image: &[u8], what: &str, coverage: &dyn Fn(f64, f64) -> f64| {
        let mut off = Vec::new();
        for y in (0..HEIGHT).step_by(5) {
            for x in (0..WIDTH).step_by(5) {
                let (px, py) = (f64::from(x) + 0.5 - half[0], f64::from(y) + 0.5 - half[1]);
                let expected = (coverage(px, py).clamp(0.0, 1.0) * 255.0).round() as u8;
                let found = channel(image, x, y, 1);
                if found.abs_diff(expected) > 2 {
                    off.push(format!("({x}, {y}): {found}, not {expected}"));
                }
            }
        }
        assert!(off.is_empty(), "{what}: {:?}", &off[..off.len().min(8)]);
    };
    let radial =
        draw(r#"{"type": "gradient", "kind": "radial", "center": [40, -20], "radius": 50}"#);
    check(&radial, "radial", &|x, y| (x - 40.0).hypot(y + 20.0) / 50.0);
    let linear = draw(r#"{"type": "gradient", "center": [0, 32], "length": 64}"#);
    check(&linear, "linear", &|_, y| (y - 32.0) / 64.0 + 0.5);
    let farthest = draw(r#"{"type": "gradient", "kind": "radial", "center": [64, 0]}"#);
    let corner = (half[0] + 64.0).hypot(half[1]);
    check(&farthest, "radial to the corners", &|x, y| {
        (x - 64.0).hypot(y) / corner
    });
}

/// The complementary error function, to seven places (Abramowitz and
/// Stegun 7.1.26).
fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erfc = poly * (-x * x).exp();
    if x < 0.0 { 2.0 - erfc } else { erfc }
}

/// A blur of a straight hard edge is the Gaussian's profile across it, as
/// smooth at every width: a wide one shows no steps or ripples. A wide
/// high-pass keeps what the blur lacks, inside the edge.
#[test]
fn blurs_are_smooth_at_every_width() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    // White over the view's left half, the edge down its middle.
    let mut draw = |ops: &str| {
        let style = processed_triangle(ops).replace(
            "[[[-1, -1], [1, -1], [0, 1], [-1, -1]]]",
            "[[[-30, -30], [0, -30], [0, 30], [-30, 30], [-30, -30]]]",
        );
        draw(&runtime, &mut target, &style, at.0, at.1)
    };
    // The most the middle row strays from `coverage` at each pixel's
    // distance right of the edge, at its centre, and where.
    let stray = |image: &[u8], coverage: &dyn Fn(f64) -> f64| {
        (0..WIDTH)
            .map(|x| {
                let d = f64::from(x) + 0.5 - f64::from(WIDTH) / 2.0;
                let expected = (coverage(d) * 255.0).round() as u8;
                (channel(image, x, HEIGHT / 2, 1).abs_diff(expected), x)
            })
            .max()
            .unwrap_or_default()
    };
    for sigma in [8.0, 24.0, 40.0, 64.0] {
        let image = draw(&format!(r#"{{"type": "blur", "size": {sigma}}}"#));
        let (most, x) = stray(&image, &|d| 0.5 * erfc(d / (sigma * 2f64.sqrt())));
        assert!(most <= 3, "a blur of {sigma} is {most} off at x = {x}");
    }
    let sigma = 40.0;
    let image = draw(&format!(r#"{{"type": "high-pass", "size": {sigma}}}"#));
    let (most, x) = stray(&image, &|d| {
        if d < 0.0 {
            1.0 - 0.5 * erfc(d / (sigma * 2f64.sqrt()))
        } else {
            0.0
        }
    });
    assert!(most <= 3, "a high-pass of {sigma} is {most} off at x = {x}");
}

/// The distance, in standard deviations, past a blurred straight edge at
/// which its Gaussian profile falls to `coverage`.
fn profile_distance(coverage: f64) -> f64 {
    let (mut near, mut far) = (0.0, 10.0);
    for _ in 0..60 {
        let mid = 0.5 * (near + far);
        if 0.5 * erfc(mid / 2f64.sqrt()) > coverage {
            near = mid;
        } else {
            far = mid;
        }
    }
    0.5 * (near + far)
}

/// A high-precision target resolves a contour a few thousandths of
/// coverage wide far out in a wide blur, where an 8-bit one rounds it
/// away: the band where a blur of a straight edge lies between two cuts
/// falls where the Gaussian's profile puts it, and as wide.
#[test]
fn a_high_precision_target_resolves_fine_contours() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    let (sigma, low, high) = (40.0, 0.0045, 0.006);
    // White left of a straight edge 100 pixels left of the view's centre,
    // on a pixel's boundary; drawn without antialiasing, each pixel whole
    // or empty.
    let edge = 100.0 * 360.0 / (512.0 * 6f64.exp2());
    let edge_x = f64::from(WIDTH) / 2.0 - 100.0;
    // With `twice`, a second process draws nothing into `depth`, which is then
    // a target of its own that both composite into.
    let style = |precision: &str, twice: bool| {
        let nothing = match twice {
            true => {
                r#"{"id": "nothing", "type": "dev.sargunv.orbis.image-process",
                "dev.sargunv.orbis:source-target": "empty", "dev.sargunv.orbis:destination-target": "depth"},"#
            }
            false => "",
        };
        format!(
            r##"{{"version": 8,
            "dev.sargunv.orbis:targets": [{{"id": "land"}},
              {{"id": "depth", "precision": "{precision}"}}, {{"id": "deep"}}, {{"id": "empty"}}],
            "sources": {{"shape": {{"type": "geojson", "data": {{"type": "Polygon",
              "coordinates": [[[-30, -30], [-{edge}, -30], [-{edge}, 30], [-30, 30], [-30, -30]]]}}}}}},
            "layers": [
              {{"id": "dark", "type": "background", "paint": {{"background-color": "#000000"}}}},
              {{"id": "land", "type": "fill", "source": "shape",
                "dev.sargunv.orbis:destination-target": "land",
                "paint": {{"fill-color": "#ffffff", "fill-antialias": false}}}},
              {{"id": "field", "type": "dev.sargunv.orbis.image-process",
                "dev.sargunv.orbis:source-target": "land", "dev.sargunv.orbis:destination-target": "depth",
                "dev.sargunv.orbis:operations": [{{"type": "blur", "size": {sigma}}}]}},
              {nothing}
              {{"id": "deep", "type": "dev.sargunv.orbis.image-process",
                "dev.sargunv.orbis:source-target": "depth", "dev.sargunv.orbis:destination-target": "deep",
                "dev.sargunv.orbis:operations": [{{"type": "threshold", "cutoff": {high_cut}}}]}},
              {{"id": "contour", "type": "dev.sargunv.orbis.image-process",
                "dev.sargunv.orbis:source-target": "depth",
                "dev.sargunv.orbis:operations": [{{"type": "threshold", "cutoff": {low_cut}}},
                  {{"type": "combine", "target": "deep", "mode": "subtract"}}]}}
            ]}}"##,
            high_cut = high * 255.0,
            low_cut = low * 255.0,
        )
    };
    // The contour's width (its coverage summed along the middle row) and
    // its centre, in pixels from the view's left edge.
    let measure = |image: &[u8]| {
        let (mut width, mut moment) = (0.0, 0.0);
        for x in 0..WIDTH {
            let coverage = f64::from(channel(image, x, HEIGHT / 2, 1)) / 255.0;
            width += coverage;
            moment += coverage * (f64::from(x) + 0.5);
        }
        (width, moment / width.max(f64::MIN_POSITIVE))
    };
    let expected_width = sigma * (profile_distance(low) - profile_distance(high));
    let expected_centre = edge_x + sigma * 0.5 * (profile_distance(low) + profile_distance(high));
    // Each antialiased edge, sampled at pixel centres, strays by a tenth of
    // a pixel or more with where it falls between them, and the two add.
    for twice in [false, true] {
        let image = draw(&runtime, &mut target, &style("high", twice), at.0, at.1);
        let (width, centre) = measure(&image);
        assert!(
            (width - expected_width).abs() < 0.3 && (centre - expected_centre).abs() < 0.3,
            "high precision (twice: {twice}): {width:.2} pixels wide at {centre:.2}, \
             not {expected_width:.2} at {expected_centre:.2}"
        );
        // In 8 bits the cuts fall between the same two steps of coverage.
        let image = draw(&runtime, &mut target, &style("standard", twice), at.0, at.1);
        let (width, centre) = measure(&image);
        assert!(
            (width - expected_width).abs() > 1.0 || (centre - expected_centre).abs() > 1.0,
            "in 8 bits too (twice: {twice}): {width:.2} pixels wide at {centre:.2}"
        );
    }
}

/// A fill drawn through a target in a colour covers what the target
/// covers in any blend mode and at any layer opacity, the sky too: steeply
/// tilted, through a target covering the whole view, it blends with the
/// sky above the horizon as with the ground below it.
#[test]
fn a_blended_fill_through_a_target_covers_the_sky() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let style = |through: &str| {
        format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 6, "pitch": 75,
              "sky": {{"sky-color": "#203060", "horizon-color": "#203060",
                "fog-color": "#203060", "sky-horizon-blend": 0, "horizon-fog-blend": 0,
                "fog-ground-blend": 1}},
              "dev.sargunv.orbis:targets": [{{"id": "blank"}}, {{"id": "glow"}}],
              "sources": {{}},
              "layers": [
                {{"id": "ground", "type": "background", "paint": {{"background-color": "#204020"}}}},
                {{"id": "cover", "type": "dev.sargunv.orbis.image-process",
                  "dev.sargunv.orbis:source-target": "blank",
                  "dev.sargunv.orbis:operations": [{{"type": "invert"}}],
                  "dev.sargunv.orbis:destination-target": "glow"}}{through}
              ]}}"##
        )
    };
    let mut draw = |style: String| {
        let mut map = Map::new(&runtime, options());
        // The style's pitch of 75 degrees reaches past the default 60.
        map.set_constraints(Constraints {
            max_pitch: 85.0,
            ..Constraints::default()
        });
        map.set_style_json(&style).unwrap();
        common::settle(&mut map, &mut target);
        target.read_pixels().unwrap()
    };
    let plain = draw(style(""));
    let (sky, ground) = ((WIDTH / 2, 2), (WIDTH / 2, HEIGHT - 2));
    let pixel = |image: &[u8], (x, y): (u32, u32)| [0, 1, 2].map(|c| channel(image, x, y, c));
    assert_ne!(pixel(&plain, sky), pixel(&plain, ground), "no sky shows");
    let colour = [0x40, 0x60, 0x80].map(f64::from);
    // What each mode makes of a channel over one beneath it, both 0–255.
    type Mix = fn(f64, f64) -> f64;
    let modes: [(&str, Mix); 3] = [
        ("normal", |_, s| s),
        ("plus-lighter", |b, s| (b + s).min(255.0)),
        ("multiply", |b, s| b * s / 255.0),
    ];
    for (blend, mix) in modes {
        for opacity in [1.0, 0.5] {
            let image = draw(style(&format!(
                r##", {{"id": "through", "type": "fill", "dev.sargunv.orbis:source-target": "glow",
                    "paint": {{"fill-color": "#406080", "fill-layer-blend": "{blend}",
                      "fill-layer-opacity": {opacity}}}}}"##
            )));
            for at in [sky, ground] {
                let beneath = pixel(&plain, at);
                let expected = [0, 1, 2].map(|c| {
                    let b = f64::from(beneath[c]);
                    (b + (mix(b, colour[c]) - b) * opacity).round() as u8
                });
                let drawn = pixel(&image, at);
                assert!(
                    largest_difference(&drawn, &expected) <= 2,
                    "{blend} at {opacity}, at {at:?} over {beneath:?}: {drawn:?}, not {expected:?}"
                );
            }
        }
    }
}

/// Image-process layers drawing into the map and fills drawn through
/// targets in a colour, in a row, draw as they do apart (each a layer
/// between them, which draws nothing), a row longer than one draw takes
/// too.
#[test]
fn composites_in_a_row_draw_as_they_do_apart() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = (Geodetic::new(0.0, 0.0), 6.0);
    let process = |id: usize, ops: &str| {
        format!(
            r#"{{"id": "process-{id}", "type": "dev.sargunv.orbis.image-process",
                "dev.sargunv.orbis:source-target": "drawn", "dev.sargunv.orbis:operations": [{ops}]}}"#
        )
    };
    let kinds = |id: usize| match id % 4 {
        0 => process(
            id,
            r##"{"type": "color", "value": "#d02060"}, {"type": "opacity", "value": 0.6}"##,
        ),
        1 => process(
            id,
            r##"{"type": "blur", "size": 6}, {"type": "color", "value": "#2040ff"}, {"type": "opacity", "value": 0.5}"##,
        ),
        2 => format!(
            r##"{{"id": "through-{id}", "type": "fill", "dev.sargunv.orbis:source-target": "drawn",
                "paint": {{"fill-color": "#20c040", "fill-opacity": 0.4}}}}"##
        ),
        _ => process(
            id,
            r#"{"type": "invert"}, {"type": "opacity", "value": 0.3}"#,
        ),
    };
    let style = |count: usize, apart: bool| {
        let mut layers = vec![
            r##"{"id": "shape", "type": "fill", "source": "shape",
                "dev.sargunv.orbis:destination-target": "drawn", "paint": {"fill-color": "#fff"}}"##
                .to_string(),
        ];
        for id in 0..count {
            layers.push(kinds(id));
            if apart {
                layers.push(format!(
                    r#"{{"id": "between-{id}", "type": "line", "source": "shape", "filter": false}}"#
                ));
            }
        }
        triangle(r#"{"id": "drawn"}"#, &layers.join(", "))
    };
    let plain = draw(&runtime, &mut target, &style(0, false), at.0, at.1);
    for count in [3, 11] {
        let row = draw(&runtime, &mut target, &style(count, false), at.0, at.1);
        let apart = draw(&runtime, &mut target, &style(count, true), at.0, at.1);
        assert!(
            largest_difference(&row, &plain) > 64,
            "{count} composites drew little"
        );
        let difference = largest_difference(&row, &apart);
        assert!(
            difference <= 3,
            "{count} composites in a row differ from them apart by {difference}"
        );
    }
}

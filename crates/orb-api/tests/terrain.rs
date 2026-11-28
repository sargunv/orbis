//! What a map at rest draws on terrain, each test a still view: draped
//! layers keep a flat map's widths and reach, buildings stand on the
//! slopes under them and hide what drapes above them, hillshades blend
//! into the ground and fade into the fog, fog fades through the globe's
//! handover, the cover reaches the ground near the eye, a resized map
//! draws its next frame, and a view at rest redoes nothing. That a moved
//! map draws what a fresh one does is `path.rs`'s.

mod common;

use common::dem::{alps, alps_dem, dem, hills};
use common::{At, STATION, camera, extruded, fresh_at, options, tilted, tokyo};
use orb_api::{Geodetic, Map, Runtime, RuntimeOptions, Target};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 384;

/// A line draped on flat terrain is as wide as on a flat map at any
/// zoom, in screen pixels as the spec has it, with a zoom-dependent width
/// too. Drape slots drawn at their tile's whole zoom magnified it by up to
/// twice between zoom steps (#151).
#[test]
fn a_draped_line_is_as_wide_as_a_flat_one() {
    let (lon, lat) = (139.76, 35.68);
    let (flat, _dem) = dem("flat", (139.2, 35.2, 140.3, 36.2), 11, |_, _| 0.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    // The dark rows down the middle column: the line's width there.
    let width = |pixels: &[u8]| {
        (0..HEIGHT)
            .filter(|&y| pixels[((y * WIDTH + WIDTH / 2) * 4) as usize] < 128)
            .count()
    };
    let mut widths = |zoom: f64| {
        [false, true].map(|terrain| {
            let style = format!(
                r##"{{"version": 8,
                "sources": {{{flat}
                  "line": {{"type": "geojson", "data": {{"type": "LineString",
                    "coordinates": [[{w}, {lat}], [{e}, {lat}]]}}}}
                }},
                {}
                "layers": [
                  {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
                  {{"id": "line", "type": "line", "source": "line", "paint": {{"line-color": "#000000",
                      "line-width": ["interpolate", ["linear"], ["zoom"], 12, 6, 14, 14]}}}}
                ]}}"##,
                match terrain {
                    true => r#""terrain": {"source": "flat", "exaggeration": 1},"#,
                    false => "",
                },
                w = lon - 0.2,
                e = lon + 0.2,
            );
            let mut map = Map::new(&runtime, options());
            map.set_style_json(&style).unwrap();
            map.jump_to(&camera((Geodetic::new(lon, lat), zoom, 0.0, 0.0)));
            common::settle(&mut map, &mut target);
            width(&target.read_pixels().unwrap())
        })
    };
    for zoom in [12.0, 12.3, 12.6, 12.9] {
        let [flat, draped] = widths(zoom);
        // Each is the zoom's width, give or take an edge's rounding.
        let expected = 6.0 + 4.0 * (zoom - 12.0);
        assert!(
            (flat as f64 - expected).abs() <= 1.0,
            "z{zoom}: flat {flat}"
        );
        assert!(
            flat.abs_diff(draped) <= 1,
            "z{zoom}: draped {draped} px against flat {flat} px"
        );
    }
}

/// A line keeps the spec's width at a zoom below its tiles' wherever on
/// the globe it is, draped on terrain or not. A drape slot is flat Web
/// Mercator drawn at one zoom, but on the globe a texel shows larger the
/// nearer the equator it is: slots sized their pixels for the camera's
/// latitude, so with the camera over 70°N a border at 15°N drew nearly
/// three times as wide (the Globe demo's boundaries, zoomed out over
/// Canada). Here a meridian's two segments, around 70°N and 15°N, lie as
/// far north and south of a camera over 42.5°N, so the sphere's
/// perspective narrows both alike, and under the camera each keeps its
/// width east to west, which the sphere doesn't foreshorten: drawn right,
/// they are as wide as each other, draped or not. On Web Mercator, at the
/// equator where it holds the camera, both are the spec's width, in a
/// view 256 px square at z−1, a whole zoom where its height fills the
/// world. At a fractional zoom a drape's texels are filtered down by a
/// scale that isn't a power of two, and a draped line's ink rounds by
/// where its edges fall among them, by up to 5% as it moves a fraction
/// of a pixel: the 512×384 view holds Mercator at z−0.415, where this
/// meridian, on the terrain tiles' seam, measures 2.10 px on one GPU and
/// 2.09 px on another.
#[test]
fn a_line_is_as_wide_at_every_latitude_below_its_tiles_zoom() {
    let (flat, _dem) = dem("flat", (-179.9, -85.0, 179.9, 85.0), 1, |_, _| 0.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    const LINE_WIDTH: f64 = 2.0;
    // The meridian's width around 70°N and around 15°N: the blue ink of
    // the rows of each of its two segments, less two at each end.
    let widths = |projection: &str, terrain: bool| -> [f64; 2] {
        let (width, height, zoom) = match projection {
            "globe" => (WIDTH, HEIGHT, -0.5),
            _ => (256, 256, -1.0),
        };
        let mut target = Target::texture(&runtime, width, height, 1.0);
        let style = format!(
            r##"{{"version": 8, "projection": {{"type": "{projection}"}},
            "sources": {{{flat}
              "line": {{"type": "geojson", "data": {{"type": "MultiLineString",
                "coordinates": [[[0, 5], [0, 25]], [[0, 60], [0, 80]]]}}}}
            }},
            {}
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "line", "type": "line", "source": "line",
                  "paint": {{"line-color": "#0000ff", "line-width": {LINE_WIDTH}}}}}
            ]}}"##,
            match terrain {
                true => r#""terrain": {"source": "flat", "exaggeration": 1},"#,
                false => "",
            },
        );
        let mut map = Map::new(&runtime, options());
        map.set_style_json(&style).unwrap();
        map.jump_to(&camera((Geodetic::new(0.0, 42.5), zoom, 0.0, 0.0)));
        common::settle(&mut map, &mut target);
        assert!(
            (map.zoom() - zoom).abs() < 1e-9,
            "{projection}: z{}",
            map.zoom()
        );
        let pixels = target.read_pixels().unwrap();
        let ink = (0..height).map(|y| -> f64 {
            (0..width)
                .map(|x| {
                    let i = ((y * width + x) * 4) as usize;
                    (f64::from(pixels[i + 2]) - f64::from(pixels[i])).max(0.0) / 255.0
                })
                .sum()
        });
        let mut segments: Vec<Vec<f64>> = Vec::new();
        let mut inked = false;
        for w in ink {
            if w >= 0.5 && !inked {
                segments.push(Vec::new());
            }
            inked = w >= 0.5;
            if inked {
                segments.last_mut().unwrap().push(w);
            }
        }
        assert_eq!(
            segments.len(),
            2,
            "{projection}, terrain {terrain}: {segments:?}"
        );
        [0, 1].map(|i| {
            let inner = &segments[i][2..segments[i].len() - 2];
            inner.iter().sum::<f64>() / inner.len() as f64
        })
    };
    for projection in ["mercator", "globe"] {
        for terrain in [false, true] {
            let [north, south] = widths(projection, terrain);
            let case = format!("{projection}, terrain {terrain}");
            assert!(
                (north - south).abs() < 0.1,
                "{case}: {north:.2} px wide around 70°N, {south:.2} px around 15°N"
            );
            // The globe's perspective narrows both a little, and its
            // drape is filtered down by a scale that isn't a power of two.
            let tolerance = match projection {
                "globe" => 0.2,
                _ => 0.05,
            };
            for width in [north, south] {
                assert!(
                    (width - LINE_WIDTH).abs() < tolerance,
                    "{case}: {width:.2} px wide, not {LINE_WIDTH}"
                );
            }
        }
    }
}

/// A fill draped on terrain and translated far, in a pitched view where
/// near terrain tiles are of zooms past the camera's, draws whole: a slot
/// takes the rows of drawn tiles as far around it as the translation
/// reaches at the camera's zoom, which the slot is drawn at. Reckoned at
/// each terrain tile's own zoom, the reach fell short in near slots, and
/// bands past a drawn tile's edge went blank. (A flat map clips what a
/// tile translates to the tile, as GL JS's stencil does.)
#[test]
fn a_draped_translate_reaches_as_far_as_the_camera_s_pixels() {
    let (lon, lat) = (139.76, 35.68);
    let (flat, _dem) = dem("flat", (139.2, 35.2, 140.3, 36.2), 11, |_, _| 0.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let style = format!(
        r##"{{"version": 8,
        "sources": {{{flat}
          "area": {{"type": "geojson", "data": {{"type": "Polygon",
            "coordinates": [[[{w}, {s}], [{e}, {s}], [{e}, {n}], [{w}, {n}], [{w}, {s}]]]}}}}
        }},
        "terrain": {{"source": "flat", "exaggeration": 1}},
        "layers": [
          {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "area", "type": "fill", "source": "area",
            "paint": {{"fill-color": "#ff0000", "fill-translate": [160, 0]}}}}
        ]}}"##,
        w = lon - 0.5,
        e = lon - 0.01,
        s = lat - 0.3,
        n = lat + 0.3,
    );
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    map.jump_to(&camera((Geodetic::new(lon, lat), 12.3, 0.0, 60.0)));
    common::settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    let red = |x: u32, y: u32| {
        let p = &pixels[((y * WIDTH + x) * 4) as usize..][..4];
        p[0] > 200 && p[1] < 80 && p[2] < 80
    };
    // Each row: red from the left edge to the fill's translated edge, and
    // none past the first pixel that isn't (give or take its edge's).
    for y in 0..HEIGHT {
        let end = (0..WIDTH).find(|&x| !red(x, y)).unwrap_or(WIDTH);
        assert!(end > WIDTH / 4, "row {y}: red to {end} only");
        let past = (end + 2..WIDTH).filter(|&x| red(x, y)).count();
        assert_eq!(past, 0, "row {y}: a gap at {end}, red again past it");
    }
}

/// A map over terrain resized between frames draws its next frame at
/// the new size: the terrain's depth for what it hides, its layers, the
/// extrusions' depth and targets are remade at the new size, and what the
/// frame binds is the new ones, not the destroyed (a submit that used one
/// drew nothing, a black map in browsers).
#[test]
fn a_resized_terrain_map_draws_its_next_frame() {
    let (hills, _dem) = hills();
    let style = extruded(tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},"#,
        &hills,
    ));
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    map.jump_to(&camera((STATION, 15.0, 20.0, 50.0)));
    common::settle(&mut map, &mut target);
    for (width, height) in [
        (WIDTH + 96, HEIGHT - 64),
        (WIDTH / 2, HEIGHT),
        (WIDTH, HEIGHT),
    ] {
        target.resize(width, height, 1.0);
        map.render(&mut target, 0.0).unwrap();
        let pixels = target.read_pixels().unwrap();
        let colours: std::collections::HashSet<&[u8]> = pixels.chunks(4).collect();
        assert!(
            colours.len() > 64,
            "the frame after a resize to {width}×{height} drew {} colours",
            colours.len()
        );
    }
}

/// Tokyo's buildings (unless `buildings` is false) standing on low
/// rolling ground under a sky, over a plain ground and under the
/// comma-separated `above`, and nothing else: a frame of them pitched far
/// enough for fog, which begins near (fog reaches from sea level's depth
/// at the target, so the ground stays low).
fn foggy_extrusions_under(buildings: bool, above: &str) -> Vec<u8> {
    let (lowland, _dem) = dem("lowland", (139.2, 35.2, 140.3, 36.2), 11, |lon, lat| {
        20.0 + 15.0 * (lon * 170.0).sin() * (lat * 210.0).cos()
    });
    let style = tokyo(
        r##""terrain": {"source": "lowland"},
            "sky": {"sky-color": "#4f8fd9", "horizon-color": "#d6e8fb", "fog-color": "#c8d8f0",
                    "fog-ground-blend": 0.2},"##,
        &lowland,
    );
    let mut layers = vec![
        r##"{"id": "ground", "type": "background", "paint": {"background-color": "#e8e4dc"}}"##,
    ];
    if buildings {
        layers.push(
            r##"{"id": "building-3d", "type": "fill-extrusion", "source": "openmaptiles",
                 "source-layer": "building", "minzoom": 14,
                 "paint": {"fill-extrusion-color": "#d9d0c9",
                           "fill-extrusion-height": ["get", "render_height"]}}"##,
        );
    }
    layers.extend(Some(above).filter(|a| !a.is_empty()));
    let at = style.find("\"layers\": [").unwrap();
    let style = format!("{}\"layers\": [{}]}}", &style[..at], layers.join(","));
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    fresh_at(&runtime, &mut target, &style, (STATION, 15.5, 30.0, 80.0)).0
}

/// A draped layer over a fill-extrusion layer on terrain that draws
/// nothing changes nothing: its terrain layer fogs only what it draws, so
/// the fog doesn't land on the ground twice, nor on the buildings in
/// front (the demo's hover highlights, inserted before the first label,
/// hazed the Globe entry's buildings toward the horizon).
#[test]
fn a_draped_layer_that_draws_nothing_over_extrusions_changes_nothing() {
    let plain = foggy_extrusions_under(true, "");
    let highlight = r##"{"id": "highlight", "type": "fill", "source": "openmaptiles",
        "source-layer": "building",
        "paint": {"fill-color": "#e5007e",
                  "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.3, 0]}}"##;
    common::assert_same(
        "empty-drape-over-3d",
        WIDTH,
        (foggy_extrusions_under(true, highlight), plain),
    );
}

/// Buildings hide a draped layer above their layer where they stand in
/// front of it, as GL JS draws such a layer's terrain over their depth:
/// an opaque background above them covers the ground and leaves the
/// buildings as they were.
#[test]
fn buildings_hide_the_draped_layers_above_them() {
    let cover =
        r##"{"id": "cover", "type": "background", "paint": {"background-color": "#ff00ff"}}"##;
    let bare = foggy_extrusions_under(false, "");
    let plain = foggy_extrusions_under(true, "");
    let covered = foggy_extrusions_under(true, cover);
    let px = |frame: &[u8], i: usize| frame[i * 4..i * 4 + 4].to_vec();
    let total = plain.len() / 4;
    let building: Vec<bool> = (0..total).map(|i| px(&plain, i) != px(&bare, i)).collect();
    // Their edges blend with what lies behind them, which the cover
    // changes: only pixels inside a building count.
    let w = WIDTH as usize;
    let buildings: Vec<usize> = (w..total - w)
        .filter(|&i| [i, i - 1, i + 1, i - w, i + w].iter().all(|&j| building[j]))
        .collect();
    let kept = (buildings.iter())
        .filter(|&&i| px(&covered, i) == px(&plain, i))
        .count();
    let ground = (0..total)
        .filter(|&i| px(&covered, i) == [255, 0, 255, 255])
        .count();
    assert!(
        buildings.len() > total / 20,
        "buildings drew {} of {total} pixels",
        buildings.len()
    );
    assert!(
        ground > total / 20,
        "the cover drew {ground} of {total} pixels"
    );
    assert!(
        kept * 20 > buildings.len() * 19,
        "buildings kept {kept} of their {} pixels",
        buildings.len()
    );
}

/// Houses on a hillside each stand on the ground under them: a row of
/// them climbs a slope 300 m high across a few overzoom slices, each house
/// a block of two (OpenMapTiles merges a block's buildings into one
/// multipolygon) in a hue of its own, and every one shows, looked at from
/// downhill and from uphill. Houses raised by one elevation per tile sink
/// into the hill above that point.
#[test]
fn houses_on_a_slope_stand_on_the_ground() {
    const HOUSES: usize = 12;
    let (lat, west) = (35.68, 139.745);
    // A plane rising east, 3,000 m a tenth of a degree: a third.
    let (slope, _dem) = dem("slope", (139.6, 35.6, 139.9, 35.76), 12, |lon, _| {
        (lon - 139.6) * 30_000.0
    });
    let square = |w: f64, s: f64| {
        let (e, n) = (w + 0.00013, s + 0.0003);
        format!("[[[{w}, {s}], [{e}, {s}], [{e}, {n}], [{w}, {n}], [{w}, {s}]]]")
    };
    let houses: Vec<String> = (0..HOUSES)
        .map(|i| {
            let w = west + 0.001 * i as f64;
            format!(
                r#"{{"type": "Feature", "properties": {{"hue": {}}}, "geometry":
                    {{"type": "MultiPolygon", "coordinates": [{}, {}]}}}}"#,
                30 * i,
                square(w, lat),
                square(w + 0.00017, lat),
            )
        })
        .collect();
    let style = format!(
        r##"{{"version": 8,
            "sources": {{{slope}
              "houses": {{"type": "geojson", "maxzoom": 12, "data":
                {{"type": "FeatureCollection", "features": [{}]}}}}}},
            "terrain": {{"source": "slope"}},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "houses", "type": "fill-extrusion", "source": "houses",
                "paint": {{"fill-extrusion-height": 15,
                           "fill-extrusion-color": ["to-color", ["concat",
                             "hsl(", ["to-string", ["get", "hue"]], ", 100%, 50%)"]]}}}}]}}"##,
        houses.join(", "),
    );
    // How many pixels show each house's hue: saturated pixels within 10°
    // of it, whatever the light makes of its faces.
    let shown = |pixels: &[u8]| {
        let mut count = [0u32; HOUSES];
        for p in pixels.chunks(4) {
            let [r, g, b] = [p[0], p[1], p[2]].map(|c| f64::from(c) / 255.0);
            let (hi, lo) = (r.max(g).max(b), r.min(g).min(b));
            if hi - lo < 0.3 {
                continue;
            }
            let d = hi - lo;
            let hue = 60.0
                * if hi == r {
                    ((g - b) / d).rem_euclid(6.0)
                } else if hi == g {
                    (b - r) / d + 2.0
                } else {
                    (r - g) / d + 4.0
                };
            let i = (hue / 30.0).round() as usize % HOUSES;
            if (hue - 30.0 * (hue / 30.0).round()).abs() < 10.0 {
                count[i] += 1;
            }
        }
        count
    };
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let middle = west + 0.0005 * (HOUSES - 1) as f64;
    for (name, at) in [
        (
            "downhill",
            (Geodetic::new(middle, lat - 0.003), 15.2, 0.0, 60.0),
        ),
        (
            "uphill",
            (Geodetic::new(middle, lat + 0.003), 15.2, 180.0, 60.0),
        ),
    ] {
        let (pixels, _) = fresh_at(&runtime, &mut target, &style, at);
        let count = shown(&pixels);
        if count.iter().any(|&n| n < 20) {
            let path = std::env::temp_dir().join(format!(
                "orb-terrain-slope-{name}-{}.png",
                std::process::id()
            ));
            common::write_png(&path, WIDTH, HEIGHT, &pixels);
            panic!(
                "{name}: pixels per house {count:?}, some sunk or missing: {}",
                path.display()
            );
        }
    }
}

/// Terrain fog fades in through a globe's handover to Web Mercator rather
/// than snapping on where it ends (#154): across zoom 12, where the
/// handover finishes, the picture changes no more than over the same
/// zoom step on the plane past it.
#[test]
fn terrain_fog_fades_through_the_globe_handover() {
    let (alps, _dem) = alps();
    let globe = alps.replacen(
        "{\"version\": 8,",
        "{\"version\": 8, \"projection\": {\"type\": \"globe\"},",
        1,
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut at = |zoom: f64| {
        let camera = (Geodetic::new(7.85, 45.95), zoom, 60.0, 75.0);
        fresh_at(&runtime, &mut target, &globe, camera).0
    };
    let [before, after, past] = [11.97, 12.0, 12.03].map(&mut at);
    // The picture's mean colour, which the fog lightens: a step's zoom
    // shifts every pixel, but barely moves it.
    let mean = |pixels: &[u8]| {
        let mut sum = [0.0; 3];
        for p in pixels.chunks(4) {
            for (s, &c) in sum.iter_mut().zip(p) {
                *s += f64::from(c);
            }
        }
        sum.map(|s| s / (pixels.len() / 4) as f64)
    };
    let change = |a: &[u8], b: &[u8]| {
        let (a, b) = (mean(a), mean(b));
        (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max)
    };
    let (handover, plane) = (change(&before, &after), change(&after, &past));
    assert!(
        handover < 2.0 * plane + 0.2,
        "across the handover's end the mean colour changes by {handover:.2} a channel, \
         over the same step on the plane by {plane:.2}"
    );
}

/// A view at rest over the Alps stays put: frames after idle cover the
/// same terrain tiles and read the same DEM tiles, so they pad no DEM
/// borders, write no rows, keep the cull's candidates and dispatch the
/// same cull every frame (at most one, and none once the drape is drawn
/// and nothing else has candidates to cull), from the first frame after
/// idle on. Here the DEM tile under one
/// cover tile culls that tile, which must not let go of the DEM tile and
/// bring the cover tile back on the next frame.
#[test]
fn an_alpine_view_at_rest_redoes_nothing() {
    let (alps, _dem) = alps();
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&alps).unwrap();
    map.jump_to(&camera((Geodetic::new(7.95, 46.0), 11.2, 0.0, 30.0)));
    common::settle(&mut map, &mut target);
    let mut dispatches = None;
    for frame in 1..=4 {
        let report = map.render(&mut target, 0.0).unwrap().unwrap();
        assert!(report.idle(), "frame {frame}");
        assert_eq!(report.frame.prepared, 0, "frame {frame}");
        assert_eq!(report.frame.placements_written, 0, "frame {frame}");
        assert!(report.frame.dispatches <= 1, "frame {frame}");
        let first = *dispatches.get_or_insert(report.frame.dispatches);
        assert_eq!(report.frame.dispatches, first, "frame {frame}");
        assert!(!report.frame.candidates_rebuilt, "frame {frame}");
    }
}

/// A hillshade multiplied over the ground (style-spec#1661's
/// `dev.sargunv.orbis:hillshade-layer-blend`) shades it by its own colour:
/// each pixel is the ground's colour times the hillshade drawn normally
/// over white, W3C multiply's identity, on a flat map, the globe and
/// terrain (lit by the terrain pass, viewport-anchored, which blends it
/// into the drape beneath, or draped, map-anchored, a terrain layer
/// composited apart), on the globe's terrain too, and over another lit
/// hillshade (whose terrain layer it then blends into): each pixel is the
/// map beneath it times the hillshade drawn normally over white. Where the
/// ground doesn't show (the sky, space), the map is as without the
/// hillshade.
#[test]
fn a_multiplied_hillshade_shades_the_ground_by_its_colour() {
    let (dem, _dir) = alps_dem();
    let dem = dem.trim_end_matches(',');
    let at = Geodetic::new(7.9, 46.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let terrain = r#""terrain": {"source": "alps", "exaggeration": 1.2},"#;
    let views: [(&str, &str, At, &str, bool); 6] = [
        ("flat", "", (at, 10.0, 0.0, 0.0), "viewport", false),
        (
            "globe",
            r#""projection": {"type": "globe"},"#,
            (at, 5.0, 0.0, 0.0),
            "viewport",
            false,
        ),
        (
            "terrain, lit",
            terrain,
            (at, 10.5, 20.0, 55.0),
            "viewport",
            false,
        ),
        (
            "terrain, draped",
            terrain,
            (at, 10.5, 20.0, 55.0),
            "map",
            false,
        ),
        (
            "globe terrain",
            r#""projection": {"type": "globe"}, "terrain": {"source": "alps", "exaggeration": 1.2},"#,
            (at, 8.5, 0.0, 45.0),
            "viewport",
            false,
        ),
        (
            "terrain, over a hillshade",
            terrain,
            (at, 10.5, 20.0, 55.0),
            "viewport",
            true,
        ),
    ];
    for (name, extra, view, anchor, under) in views {
        // `under`: another hillshade, lit from the north-east, beneath it.
        let mut draw = |ground: &str, hillshade: Option<&str>, under: bool| {
            let (visibility, blend) = match hillshade {
                Some(blend) => ("visible", blend),
                None => ("none", "normal"),
            };
            let beneath = if under { "visible" } else { "none" };
            let style = format!(
                r##"{{"version": 8, {extra}
                    "sources": {{{dem}}},
                    "layers": [
                      {{"id": "ground", "type": "background", "paint": {{"background-color": "{ground}"}}}},
                      {{"id": "under", "type": "hillshade", "source": "alps",
                        "layout": {{"visibility": "{beneath}"}},
                        "paint": {{"hillshade-illumination-direction": 45,
                          "hillshade-shadow-color": "#204080"}}}},
                      {{"id": "relief", "type": "hillshade", "source": "alps",
                        "layout": {{"visibility": "{visibility}"}},
                        "paint": {{"hillshade-illumination-anchor": "{anchor}",
                          "hillshade-shadow-color": "#3a2a5a", "hillshade-accent-color": "#806a90",
                          "dev.sargunv.orbis:hillshade-layer-blend": "{blend}"}}}}
                    ]}}"##
            );
            let mut map = Map::new(&runtime, options());
            map.set_style_json(&style).unwrap();
            map.jump_to(&camera(view));
            common::settle(&mut map, &mut target);
            target.read_pixels().unwrap()
        };
        let white = draw("#ffffff", None, false);
        let normal = draw("#ffffff", Some("normal"), false);
        let beneath = draw("#66aa33", None, under);
        let multiplied = draw("#66aa33", Some("multiply"), under);
        let (mut ground, mut shaded, mut off) = (0, 0, Vec::new());
        for i in (0..white.len()).step_by(4) {
            let px = |p: &[u8]| [p[i], p[i + 1], p[i + 2]];
            let (got, at) = (px(&multiplied), (i / 4) as u32);
            let (x, y) = (at % WIDTH, at / WIDTH);
            let expected = if px(&white) == [255; 3] {
                ground += 1;
                shaded += usize::from(px(&normal) != [255; 3]);
                let (n, b) = (px(&normal), px(&beneath));
                std::array::from_fn(|c| (f32::from(b[c]) * f32::from(n[c]) / 255.0).round() as u8)
            } else {
                px(&beneath)
            };
            if !got.iter().zip(expected).all(|(&g, e)| g.abs_diff(e) <= 3) {
                off.push(format!("({x}, {y}) is {got:?}, expected {expected:?}"));
            }
        }
        let pixels = white.len() / 4;
        assert!(off.is_empty(), "{name}: {off:#?}");
        assert!(ground > pixels / 4, "{name}: {ground} ground pixels");
        assert!(shaded > ground / 4, "{name}: {shaded} of {ground} shaded");
    }
}

/// A blended hillshade on terrain fades into the fog rather than taking
/// its colour: the ground it blends with is fogged already, so fogged as
/// its own content it would multiply the far ground by the fog's colour
/// again. Toward the horizon, where the fog is thick, the map is as
/// without the hillshade, whether the terrain pass lights it (the fog
/// falls on what it blended into) or it drapes, a terrain layer
/// composited apart (which fades).
#[test]
fn a_blended_hillshade_fades_into_the_fog() {
    // Ridges reaching past the horizon.
    let (dem, _dir) = dem("ridges", (0.0, 40.0, 16.0, 56.0), 7, |lon, lat| {
        1500.0 + 1200.0 * (lon * 9.0).sin() * (lat * 7.0).cos()
    });
    let dem = dem.trim_end_matches(',');
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    for anchor in ["viewport", "map"] {
        let mut draw = |ground: &str, visibility: &str| {
            let style = format!(
                r##"{{"version": 8,
                "terrain": {{"source": "ridges", "exaggeration": 1.2}},
                "sky": {{"sky-color": "#4f8fd9", "horizon-color": "#8899aa", "fog-color": "#8899aa",
                         "fog-ground-blend": 0.2, "horizon-fog-blend": 0.9}},
                "sources": {{{dem}}},
                "layers": [
                  {{"id": "ground", "type": "background", "paint": {{"background-color": "{ground}"}}}},
                  {{"id": "relief", "type": "hillshade", "source": "ridges",
                    "layout": {{"visibility": "{visibility}"}},
                    "paint": {{"hillshade-shadow-color": "#3a2a5a",
                      "hillshade-illumination-anchor": "{anchor}",
                      "dev.sargunv.orbis:hillshade-layer-blend": "multiply"}}}}
                ]}}"##
            );
            let mut map = Map::new(&runtime, options());
            map.set_constraints(tilted());
            map.set_style_json(&style).unwrap();
            map.jump_to(&camera((Geodetic::new(8.0, 43.0), 7.5, 0.0, 75.0)));
            common::settle(&mut map, &mut target);
            target.read_pixels().unwrap()
        };
        let bare = draw("#66aa33", "none");
        let red = draw("#ff0000", "none");
        let shaded = draw("#66aa33", "visible");
        let row = |y: u32| (y * WIDTH * 4) as usize..((y + 1) * WIDTH * 4) as usize;
        // The horizon: the first row mostly ground, which the background's
        // colour shows through.
        let horizon = (0..HEIGHT)
            .find(|&y| {
                let (a, b) = (&bare[row(y)], &red[row(y)]);
                a.chunks(4).zip(b.chunks(4)).filter(|(a, b)| a != b).count() > WIDTH as usize / 2
            })
            .expect("ground in view");
        let shading = |rows: std::ops::Range<u32>| {
            let n = rows.len() as f64 * f64::from(WIDTH);
            let sum: u64 = (rows.flat_map(&row))
                .map(|i| u64::from(bare[i].abs_diff(shaded[i])))
                .sum();
            sum as f64 / n
        };
        let band = HEIGHT / 20;
        let (far, near) = (
            shading(horizon..horizon + band),
            shading(HEIGHT - band..HEIGHT),
        );
        assert!(horizon < HEIGHT / 2, "the horizon at row {horizon}");
        assert!(
            near > 4.0,
            "{anchor}: the hillshade shades the near ground by {near}"
        );
        assert!(
            far < near / 2.0,
            "{anchor}: the hillshade shades the far ground by {far}, the near by {near}"
        );
    }
}

/// A plateau 2,400 m up rising 6 km south of a valley at 300 m, its top
/// red over green ground, seen north at zoom 11.5 and a pitch of 65°: the
/// view's bottom edge falls on the plateau, about 7.3 km south, though at
/// the valley's height it would fall short of it. The cover bounds tiles
/// by the DEM tiles their zoom reads, and near the eye a parent's coarser
/// DEM tile may not have loaded while its children's have: bounded below
/// the plateau as if it stood at the target's height, the parent was
/// culled, with it the children, and nothing asked for the DEM tile that
/// would bring them back. The tiles beyond showed their skirts through
/// the hole, as streaks along the bottom edge.
#[test]
fn ground_near_the_eye_shows_before_its_coarser_dem_tiles_load() {
    let (lon, lat) = (7.9, 46.0);
    let edge = lat - 6.0 / 111.0;
    let (source, _dem) = dem("plateau", (7.6, 45.7, 8.2, 46.3), 12, |_, la| {
        if la < edge { 2400.0 } else { 300.0 }
    });
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    for projection in ["mercator", "globe"] {
        let style = format!(
            r##"{{"version": 8, "projection": {{"type": "{projection}"}},
              "sources": {{{source}
                "top": {{"type": "geojson", "data": {{"type": "Polygon", "coordinates":
                  [[[7.6, 45.7], [8.2, 45.7], [8.2, {edge}], [7.6, {edge}], [7.6, 45.7]]]}}}}}},
              "terrain": {{"source": "plateau", "exaggeration": 1}},
              "layers": [
                {{"id": "ground", "type": "background", "paint": {{"background-color": "#00ff00"}}}},
                {{"id": "top", "type": "fill", "source": "top",
                  "paint": {{"fill-color": "#ff0000", "fill-antialias": false}}}}
              ]}}"##
        );
        let at = (Geodetic::new(lon, lat), 11.5, 0.0, 65.0);
        let (pixels, _) = fresh_at(&runtime, &mut target, &style, at);
        for x in (0..8).map(|i| WIDTH * (2 * i + 1) / 16) {
            let k = (((HEIGHT - 2) * WIDTH + x) * 4) as usize;
            assert_eq!(
                pixels[k..k + 3],
                [255, 0, 0],
                "{projection}: the bottom edge at x {x} misses the plateau"
            );
        }
    }
}

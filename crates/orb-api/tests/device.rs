//! Device robustness: a lost device is replaced with the map drawing the
//! same pixels again, losses in a row back off and then stop, and a driver
//! that hangs compiling is reported.

mod common;

use std::time::{Duration, Instant};

use orb_api::{Error, Map, MapOptions, Runtime, RuntimeOptions, Target};

const STYLE: &str = r##"{"version":8,"sources":{"box":{"type":"geojson","data":{"type":"Polygon",
  "coordinates":[[[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]]}}},
  "layers":[{"id":"bg","type":"background","paint":{"background-color":"#ff0000"}},
  {"id":"box","type":"fill","source":"box","paint":{"fill-color":"#0000ff"}}]}"##;

const SIZE: u32 = 64;

/// Draws until idle; the time it took.
fn settle(map: &mut Map, target: &mut Target) -> Result<Duration, Error> {
    let started = Instant::now();
    map.render_until_idle(target, Duration::from_secs(30))?;
    Ok(started.elapsed())
}

/// The child's role in [`a_hung_compile_is_reported`].
const HANG_CHILD: &str = "ORB_TEST_HANG_CHILD";

/// A driver that hangs in its compiler. Compiles run off the drawing
/// thread, so frames go on; a map waiting only for the compile asks for
/// none, and the watchdog wakes it when the compile runs past its limit.
/// The next frame finds the compile hung and the device lost, and the
/// frame after fails naming it rather than make a device that would hang
/// the same way. In a child process: the fault is set from the environment.
#[test]
fn a_hung_compile_is_reported() {
    if std::env::var_os(HANG_CHILD).is_some() {
        let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
        let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(STYLE).unwrap();
        let started = Instant::now();
        let (mut frames, mut idle) = (0, 0);
        let lost = loop {
            assert!(started.elapsed().as_secs() < 20, "the watchdog never fired");
            // As a host draws: only the frames the map asks for.
            if !map.needs_frame() {
                idle += 1;
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            let report = map.render(&mut target, 0.0).unwrap();
            if runtime.lost() {
                break started.elapsed();
            }
            assert!(report.is_none_or(|r| !r.idle), "nothing compiles");
            frames += 1;
        };
        let hung = runtime
            .stats()
            .hung_compile
            .expect("the watchdog names the compile");
        eprintln!(
            "{frames} frames drawn and {idle} polls idle while {hung} compiled; \
             lost after {lost:?}"
        );
        // A compile that blocked the drawing thread would hold the first
        // frame until the watchdog fired.
        assert!(frames > 0, "frames go on while a compile hangs");
        assert!(idle > 0, "the map waits for the watchdog's wake");
        let error = map.render(&mut target, 0.0).unwrap_err();
        assert!(error.message().contains(&hung), "{error}");
        return;
    }
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "a_hung_compile_is_reported", "--nocapture"])
        .env(HANG_CHILD, "1")
        .env("ORB_FAULT_INJECTION", "hang")
        .env("ORB_COMPILE_WATCHDOG_MS", "300")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&child.stderr);
    assert!(child.status.success(), "{stderr}");
}

#[test]
fn a_lost_device_recovers() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    // What the app added at run time: a feature's state and an image.
    map.set_style_json(
        r##"{"version":8,"sources":{"boxes":{"type":"geojson","data":{"type":"FeatureCollection",
          "features":[{"type":"Feature","id":1,"properties":{},"geometry":{"type":"Polygon",
            "coordinates":[[[-40,-10],[-10,-10],[-10,10],[-40,10],[-40,-10]]]}},
          {"type":"Feature","id":2,"properties":{},"geometry":{"type":"Polygon",
            "coordinates":[[[10,-10],[40,-10],[40,10],[10,10],[10,-10]]]}}]}}},
          "layers":[{"id":"bg","type":"background","paint":{"background-color":"#ff0000"}},
          {"id":"state","type":"fill","source":"boxes","filter":["==",["id"],1],"paint":{"fill-color":
            ["case",["boolean",["feature-state","on"],false],"#00ff00","#0000ff"]}},
          {"id":"image","type":"fill","source":"boxes","filter":["==",["id"],2],"paint":{"fill-pattern":"dot"}}]}"##,
    )
    .unwrap();
    settle(&mut map, &mut target).unwrap();
    let tx = orb_api::Transaction::from_commands(
        r#"[["setFeatureState", {"source": "boxes", "id": 1}, {"on": true}]]"#,
    )
    .unwrap();
    map.edit(&tx).unwrap();
    let yellow = [255, 255, 0, 255].repeat(64);
    let dot = orb_api::StyleImage::with_options(8, 8, yellow, r#"{"pixelRatio": 1}"#).unwrap();
    map.add_image("dot", dot).unwrap();
    settle(&mut map, &mut target).unwrap();
    let before = target.read_pixels().unwrap();
    let at = |pixels: &[u8], x: u32| {
        let i = ((SIZE / 2 * SIZE + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    assert_eq!(
        at(&before, SIZE / 2 - 18),
        [0, 255, 0],
        "the feature's state"
    );
    assert_eq!(at(&before, SIZE / 2 + 18), [255, 255, 0], "the added image");

    runtime.lose_device();
    let lost = Instant::now();
    let drawn = settle(&mut map, &mut target).unwrap();
    eprintln!(
        "recovered in {:.1} ms ({:.1} ms drawing)",
        lost.elapsed().as_secs_f64() * 1000.0,
        drawn.as_secs_f64() * 1000.0
    );
    let after = target.read_pixels().unwrap();
    assert_eq!(
        at(&after, SIZE / 2 - 18),
        [0, 255, 0],
        "the feature's state, again"
    );
    assert_eq!(
        at(&after, SIZE / 2 + 18),
        [255, 255, 0],
        "the added image, again"
    );
    assert!(after == before, "the same picture again");
}

/// A map at rest when another replaces the lost device drew on the old
/// one, which went with its picture: it asks for a frame on the new one,
/// which draws it again.
#[test]
fn a_map_at_rest_draws_again_on_a_replaced_device() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut maps = [0, 1].map(|_| {
        let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(STYLE).unwrap();
        settle(&mut map, &mut target).unwrap();
        // And its upkeep, as a host drawing only what's new leaves none.
        while map.needs_frame() {
            map.render(&mut target, 0.0).unwrap();
        }
        (map, target)
    });
    let before = maps[1].1.read_pixels().unwrap();
    runtime.lose_device();
    let [(first, first_target), (second, second_target)] = &mut maps;
    settle(first, first_target).unwrap();
    assert!(second.has_news(), "the replaced device is news");
    settle(second, second_target).unwrap();
    assert!(
        second_target.read_pixels().unwrap() == before,
        "the same picture again"
    );
}

/// A driver that loses every device it gives (Adreno X1's Vulkan, #141):
/// each loss of a run is answered later than the one before, and past the
/// backoff's last the runtime stops, saying why once as an error event
/// and failing every frame after, rather than reset the map forever.
#[test]
fn losses_in_a_row_back_off_then_stop() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(STYLE).unwrap();
    settle(&mut map, &mut target).unwrap();
    let mut waits = Vec::new();
    for _ in 0..3 {
        runtime.lose_device();
        let lost = Instant::now();
        settle(&mut map, &mut target).unwrap();
        waits.push(lost.elapsed());
    }
    eprintln!("recovered after {waits:?}");
    assert!(waits[1] >= Duration::from_millis(250), "{waits:?}");
    assert!(waits[2] >= Duration::from_secs(1), "{waits:?}");
    runtime.lose_device();
    let error = map.render(&mut target, 0.0).unwrap_err();
    assert!(error.message().contains("lost 4 times"), "{error}");
    assert!(!map.has_news(), "a stopped runtime's loss is news");
    let events: Vec<_> = std::iter::from_fn(|| map.poll_event()).collect();
    let reported = events
        .iter()
        .filter(|e| {
            matches!(e, orb_api::Event::Map(orb_api::MapEvent::Error { message })
                if *message == error.message())
        })
        .count();
    assert_eq!(reported, 1, "{events:?}");
    assert_eq!(map.render(&mut target, 0.0).unwrap_err(), error);
    assert!(
        std::iter::from_fn(|| map.poll_event())
            .all(|e| !matches!(e, orb_api::Event::Map(orb_api::MapEvent::Error { .. }))),
        "once"
    );
}

/// The child's role in [`a_drape_pipeline_arriving_last_still_idles`].
const DRAPE_CHILD: &str = "ORB_TEST_DRAPE_CHILD";

/// A draped layer whose pipeline compiles after everything else has
/// loaded (the last compile on a slow phone): its arrival draws the
/// drape's slots again from the next frame, so the frame it arrives in
/// isn't complete. With nothing left compiling, no compile will wake the
/// map: it asks for that next frame itself, and goes idle after it,
/// rather than wait forever. In a child process: the fault is set from
/// the environment.
#[test]
fn a_drape_pipeline_arriving_last_still_idles() {
    if std::env::var_os(DRAPE_CHILD).is_some() {
        // One flat terrarium tile.
        let dir = common::Scratch::new("api-flat-dem");
        std::fs::create_dir_all(dir.join("0/0")).unwrap();
        let file = std::fs::File::create(dir.join("0/0/0.png")).unwrap();
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 256, 256);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let flat = [128, 0, 0, 255].repeat(256 * 256);
        (encoder.write_header().unwrap())
            .write_image_data(&flat)
            .unwrap();
        let style = format!(
            r##"{{"version":8,"zoom":1,"terrain":{{"source":"dem"}},
              "sources":{{"dem":{{"type":"raster-dem","encoding":"terrarium","tileSize":256,
                "maxzoom":0,"tiles":["file://{}/{{z}}/{{x}}/{{y}}.png"]}},
              "box":{{"type":"geojson","data":{{"type":"Polygon",
                "coordinates":[[[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]]}}}}}},
              "layers":[{{"id":"bg","type":"background","paint":{{"background-color":"#ff0000"}}}},
              {{"id":"box","type":"fill","source":"box","paint":{{"fill-color":"#0000ff"}}}}]}}"##,
            dir.display()
        );
        let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
        let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&style).unwrap();
        let started = Instant::now();
        let mut frames = 0;
        let idled = loop {
            if started.elapsed() > Duration::from_secs(15) {
                break false;
            }
            // As a host draws: only the frames the map asks for.
            if !map.needs_frame() {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            let report = map.render(&mut target, 0.0).unwrap();
            frames += 1;
            if report.is_some_and(|r| r.idle) {
                break true;
            }
        };
        assert!(
            idled,
            "the map stopped asking for frames short of idle, after {frames}"
        );
        let pixels = target.read_pixels().unwrap();
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        assert_eq!(pixels[i..i + 3], [0, 0, 255], "the draped fill");
        return;
    }
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "a_drape_pipeline_arriving_last_still_idles",
            "--nocapture",
        ])
        .env(DRAPE_CHILD, "1")
        .env("ORB_FAULT_INJECTION", "slow:fill:1500")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&child.stderr);
    assert!(child.status.success(), "{stderr}");
}

/// The child's role in [`hidden_layers_compile_ahead_of_need`].
const SET_CHILD: &str = "ORB_TEST_PIPELINE_SET_CHILD";

/// A layer a zoom shows later has its pipelines, the style's pipeline
/// set's, compiled from the load, in the background: the view's load
/// isn't held up by them, the rest isn't broken by them (no frame asked
/// for while they compile), and the zoom that shows the layer waits for
/// nothing. Zoomed in while its compile still runs, the map
/// waits for it and is woken when it ends. In a child process: compiles
/// of the layer's pipeline are slowed from the environment.
#[test]
fn hidden_layers_compile_ahead_of_need() {
    // Long against a load, even on a loaded machine (a full test run,
    // other builds), so the load rests well before it ends.
    const COMPILE: Duration = Duration::from_secs(3);
    if std::env::var_os(SET_CHILD).is_none() {
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "hidden_layers_compile_ahead_of_need",
                "--nocapture",
            ])
            .env(SET_CHILD, "1")
            .env(
                "ORB_FAULT_INJECTION",
                format!("slow:circle:{}", COMPILE.as_millis()),
            )
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&child.stderr);
        assert!(child.status.success(), "{stderr}");
        return;
    }
    let style = r##"{"version":8,"zoom":1,"sources":{"point":{"type":"geojson","data":
      {"type":"Point","coordinates":[0,0]}}},
      "layers":[{"id":"bg","type":"background","paint":{"background-color":"#ff0000"}},
      {"id":"dot","type":"circle","source":"point","minzoom":4,
        "paint":{"circle-radius":20,"circle-color":"#00ff00"}}]}"##;
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    // Draws the frames the map asks for until idle: whether any waited
    // for a compile.
    let rest = |map: &mut Map, target: &mut Target| {
        let started = Instant::now();
        let mut compiling = false;
        loop {
            assert!(started.elapsed() < Duration::from_secs(15), "never idle");
            if !map.needs_frame() {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            if let Some(report) = map.render(target, 0.0).unwrap() {
                compiling |= report.frame.pipelines_pending > 0;
                if report.idle {
                    return compiling;
                }
            }
        }
    };
    let center = |target: &Target| {
        let pixels = target.read_pixels().unwrap();
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    let zoom_in = |map: &mut Map| {
        map.jump_to(&orb_api::CameraTarget {
            zoom: Some(5.0),
            ..Default::default()
        });
    };

    // Compiled from the load, ahead of the zoom, which waits for none of
    // them.
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).unwrap();
    let loading = Instant::now();
    rest(&mut map, &mut target);
    assert!(
        loading.elapsed() < COMPILE,
        "the load waited for the circle"
    );
    // Long enough for the slowed compile to end, its own compile time
    // included, where a frame would collect it.
    let rested = Instant::now();
    while rested.elapsed() < COMPILE * 2 {
        assert!(!map.needs_frame(), "a background compile woke the map");
        std::thread::sleep(Duration::from_millis(10));
    }
    zoom_in(&mut map);
    assert!(
        !rest(&mut map, &mut target),
        "the zoom waited for a compile"
    );
    assert_eq!(center(&target), [0, 255, 0], "the circle");

    // Shown while it compiles: waited for, and woken for.
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).unwrap();
    rest(&mut map, &mut target);
    zoom_in(&mut map);
    rest(&mut map, &mut target);
    assert_eq!(center(&target), [0, 255, 0], "the circle, once compiled");
}

/// A map that never rests (here an animation that outlasts the test)
/// still compiles the pipelines no view has needed yet: they wait for the
/// user's hand to be off the map, not for the map to be idle.
#[test]
fn a_map_that_never_rests_compiles_its_whole_set() {
    let style = r##"{"version":8,"zoom":1,"sources":{"point":{"type":"geojson","data":
      {"type":"Point","coordinates":[0,0]}}},
      "layers":[{"id":"bg","type":"background","paint":{"background-color":"#ff0000"}},
      {"id":"dot","type":"circle","source":"point","minzoom":4,
        "paint":{"circle-radius":20,"circle-color":"#00ff00"}}]}"##;
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).unwrap();
    map.ease_to(
        &orb_api::CameraTarget {
            bearing: Some(180.0),
            ..Default::default()
        },
        &orb_api::EaseOptions {
            duration: 600_000.0,
            ..Default::default()
        },
    );
    let started = Instant::now();
    loop {
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(20),
            "the circle's pipeline never compiled"
        );
        let report = map
            .render(&mut target, elapsed.as_secs_f64() * 1000.0)
            .unwrap();
        assert!(!report.is_some_and(|r| r.idle), "the map rested");
        let counters = runtime.stats().counters;
        // The background's and the cull's, then the circle's.
        if counters.pipelines_requested > 1
            && counters.pipelines_compiled == counters.pipelines_requested
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(16));
    }
}

/// The child's role in [`a_device_whose_compiles_hold_presents_shows_its_whole_set_compiled`].
const HOLD_CHILD: &str = "ORB_TEST_HOLD_CHILD";

/// On a device whose presents wait behind compiles (Firefox's WebGPU,
/// stood in for here), the style's whole set compiles at once, a layer a
/// zoom shows later's too, and the map draws nothing until it has: its
/// first frame shows its view whole, and the zoom that shows the layer
/// compiles nothing. In a child process: the device and the layer's slowed
/// compile are set from the environment.
#[test]
fn a_device_whose_compiles_hold_presents_shows_its_whole_set_compiled() {
    if std::env::var_os(HOLD_CHILD).is_none() {
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "a_device_whose_compiles_hold_presents_shows_its_whole_set_compiled",
                "--nocapture",
            ])
            .env(HOLD_CHILD, "1")
            .env("ORB_COMPILES_HOLD_PRESENTS", "1")
            .env("ORB_FAULT_INJECTION", "slow:circle:500")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&child.stderr);
        assert!(child.status.success(), "{stderr}");
        return;
    }
    let style = r##"{"version":8,"zoom":1,"sources":{"point":{"type":"geojson","data":
      {"type":"Point","coordinates":[0,0]}}},
      "layers":[{"id":"bg","type":"background","paint":{"background-color":"#ff0000"}},
      {"id":"dot","type":"circle","source":"point","minzoom":4,
        "paint":{"circle-radius":20,"circle-color":"#00ff00"}}]}"##;
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let center = |target: &Target| {
        let pixels = target.read_pixels().unwrap();
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).unwrap();
    let started = Instant::now();
    let mut held = 0;
    loop {
        assert!(started.elapsed() < Duration::from_secs(15), "never idle");
        if !map.needs_frame() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }
        let Some(report) = map.render(&mut target, 0.0).unwrap() else {
            continue;
        };
        let counters = runtime.stats().counters;
        if report.compiling {
            held += 1;
            assert_eq!(center(&target), [0, 0, 0], "a frame drew while compiling");
            continue;
        }
        assert_eq!(
            counters.pipelines_compiled, counters.pipelines_requested,
            "a frame drew before the whole set compiled"
        );
        if report.idle {
            break;
        }
    }
    assert!(held > 0, "no frame waited for the slowed compile");
    assert_eq!(center(&target), [255, 0, 0], "the background");
    let asked = runtime.stats().counters.pipelines_requested;
    map.jump_to(&orb_api::CameraTarget {
        zoom: Some(5.0),
        ..Default::default()
    });
    map.render_until_idle(&mut target, Duration::from_secs(15))
        .unwrap();
    assert_eq!(center(&target), [0, 255, 0], "the circle");
    assert_eq!(
        runtime.stats().counters.pipelines_requested,
        asked,
        "the zoom asked for a pipeline"
    );
}

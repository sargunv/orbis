//! Device robustness: a lost device is replaced with the map drawing the
//! same pixels again, and a driver that hangs compiling is reported.

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

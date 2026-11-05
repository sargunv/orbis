//! Device robustness (plan §8.8): the crash sentinel demotes a device whose
//! driver crashed making pipelines, and a lost device is replaced with the
//! map drawing the same pixels again.
//!
//! Compat needs a libEGL: the platform's, or `ORB_EGL_LIBRARY` (ANGLE's on
//! macOS). Without one, demotion is reported but the map stays on Modern.

use std::time::{Duration, Instant};

use orb_api::{Backend, Error, Map, MapOptions, Runtime, RuntimeOptions, Target, Tier};

const STYLE: &str = r##"{"version":8,"sources":{"box":{"type":"geojson","data":{"type":"Polygon",
  "coordinates":[[[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]]}}},
  "layers":[{"id":"bg","type":"background","paint":{"background-color":"#ff0000"}},
  {"id":"box","type":"fill","source":"box","paint":{"fill-color":"#0000ff"}}]}"##;

const SIZE: u32 = 64;

/// The child's role in [`sentinel_demotes_after_a_crash_in_pipelines`].
const CHILD: &str = "ORB_TEST_SENTINEL_CHILD";

fn options(state_dir: Option<std::path::PathBuf>) -> RuntimeOptions {
    RuntimeOptions {
        state_dir,
        egl_library: std::env::var("ORB_EGL_LIBRARY").ok(),
        ..RuntimeOptions::default()
    }
}

/// Draws until idle; the time it took.
fn settle(map: &mut Map, target: &mut Target) -> Result<Duration, Error> {
    let started = Instant::now();
    loop {
        let report = map.render(target)?;
        if report.is_some_and(|r| r.idle) && !map.needs_frame() {
            return Ok(started.elapsed());
        }
        assert!(started.elapsed().as_secs() < 30, "the map never idled");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn first_frame(runtime: &Runtime) -> Vec<u8> {
    let mut target = Target::texture(runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(runtime, MapOptions::default());
    map.set_style_json(STYLE).unwrap();
    settle(&mut map, &mut target).unwrap();
    target.read_pixels().unwrap()
}

#[test]
fn sentinel_demotes_after_a_crash_in_pipelines() {
    if let Some(dir) = std::env::var_os(CHILD) {
        // The child: dies making its first pipelines, as a driver whose
        // compiler crashes does.
        let runtime = Runtime::new(&options(Some(dir.into()))).unwrap();
        first_frame(&runtime);
        unreachable!("the fault injection aborts");
    }
    let dir = std::env::temp_dir().join(format!("orb-sentinel-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // A clean run arms and clears the sentinel.
    let runtime = Runtime::new(&options(Some(dir.clone()))).unwrap();
    let modern = runtime.tier() == Tier::Modern;
    first_frame(&runtime);
    drop(runtime);
    if !modern {
        return; // No Modern adapter here: nothing to demote.
    }
    let clean = Runtime::new(&options(Some(dir.clone()))).unwrap();
    assert_eq!(clean.tier(), Tier::Modern, "{}", clean.selection());
    drop(clean);

    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "sentinel_demotes_after_a_crash_in_pipelines",
            "--nocapture",
        ])
        .env(CHILD, &dir)
        .env("ORB_FAULT_INJECTION", "pipelines")
        .output()
        .unwrap();
    assert!(!child.status.success(), "the child should have crashed");

    let demoted = Runtime::new(&options(Some(dir.clone()))).unwrap();
    let selection = demoted.selection();
    assert!(selection.contains("crash sentinel"), "{selection}");
    if demoted.tier() == Tier::Compat {
        // Compat draws the same map.
        let pixels = first_frame(&demoted);
        assert_eq!(&pixels[..4], &[255, 0, 0, 255]);
        let centre = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        assert_eq!(&pixels[centre..centre + 4], &[0, 0, 255, 255]);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

fn recovers(backend: Backend) {
    let runtime = match Runtime::new(&RuntimeOptions {
        backend,
        ..options(None)
    }) {
        Ok(runtime) => runtime,
        Err(e) if backend == Backend::Compat => return eprintln!("no Compat device: {e}"),
        Err(e) => panic!("{e}"),
    };
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(STYLE).unwrap();
    settle(&mut map, &mut target).unwrap();
    let before = target.read_pixels().unwrap();

    runtime.lose_device();
    let lost = Instant::now();
    assert!(matches!(map.render(&mut target), Err(Error::DeviceLost)));
    assert!(runtime.lost());
    runtime.recover().unwrap();
    let tier = runtime.tier();
    let drawn = settle(&mut map, &mut target).unwrap();
    eprintln!(
        "{backend:?} → {tier:?}: recovered in {:.1} ms ({:.1} ms drawing)",
        lost.elapsed().as_secs_f64() * 1000.0,
        drawn.as_secs_f64() * 1000.0
    );
    assert_eq!(target.read_pixels().unwrap(), before);
}

#[test]
fn a_lost_modern_device_recovers() {
    recovers(Backend::Modern);
}

#[test]
fn a_lost_compat_device_recovers() {
    recovers(Backend::Compat);
}

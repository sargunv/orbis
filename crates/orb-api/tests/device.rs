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
    map.render_until_idle(target, Duration::from_secs(30))?;
    Ok(started.elapsed())
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
    let modern = runtime.stats().tier == Tier::Modern;
    first_frame(&runtime);
    drop(runtime);
    if !modern {
        return; // No Modern adapter here: nothing to demote.
    }
    let clean = Runtime::new(&options(Some(dir.clone()))).unwrap();
    assert_eq!(
        clean.stats().tier,
        Tier::Modern,
        "{}",
        clean.stats().selection
    );
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
    let selection = demoted.stats().selection;
    assert!(selection.contains("crash sentinel"), "{selection}");
    if demoted.stats().tier == Tier::Compat {
        // Compat draws the same map.
        let pixels = first_frame(&demoted);
        assert_eq!(&pixels[..4], &[255, 0, 0, 255]);
        let centre = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        assert_eq!(&pixels[centre..centre + 4], &[0, 0, 255, 255]);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The child's role in [`a_hung_compile_demotes_the_device`].
const HANG_CHILD: &str = "ORB_TEST_HANG_CHILD";

/// Plan §8.8 3: a driver that hangs in its compiler, rather than crashing,
/// demotes the device too. Compiles run off the drawing thread, so frames go
/// on; a map waiting only for the compile asks for none, and the watchdog
/// wakes it when the compile runs past its limit. The next frame finds the
/// compile hung, the device is lost, the sentinel stays armed, and the
/// device made next, now and at the next start, is Compat. In a child
/// process: the fault is set from the environment.
#[test]
fn a_hung_compile_demotes_the_device() {
    if let Some(dir) = std::env::var_os(HANG_CHILD) {
        let runtime = Runtime::new(&options(Some(dir.into()))).unwrap();
        if runtime.stats().tier != Tier::Modern {
            return; // No Modern adapter here: nothing to demote.
        }
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
            match map.render(&mut target, 0.0) {
                Err(Error::DeviceLost) => break started.elapsed(),
                Err(e) => panic!("{e}"),
                Ok(report) => assert!(report.is_none_or(|r| !r.idle), "nothing compiles"),
            }
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
        runtime.recover().unwrap();
        let selection = runtime.stats().selection;
        assert!(selection.contains("crash sentinel"), "{selection}");
        return;
    }
    let dir = std::env::temp_dir().join(format!("orb-hang-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "a_hung_compile_demotes_the_device",
            "--nocapture",
        ])
        .env(HANG_CHILD, &dir)
        .env("ORB_FAULT_INJECTION", "hang")
        .env("ORB_COMPILE_WATCHDOG_MS", "300")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&child.stderr);
    assert!(child.status.success(), "{stderr}");
    if !dir.join("gpu-sentinel").exists() {
        return; // The child had no Modern adapter.
    }
    // The next start demotes too.
    let demoted = Runtime::new(&options(Some(dir.clone()))).unwrap();
    let selection = demoted.stats().selection;
    assert!(selection.contains("crash sentinel"), "{selection}");
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
    assert!(matches!(
        map.render(&mut target, 0.0),
        Err(Error::DeviceLost)
    ));
    assert!(runtime.lost());
    runtime.recover().unwrap();
    let tier = runtime.stats().tier;
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

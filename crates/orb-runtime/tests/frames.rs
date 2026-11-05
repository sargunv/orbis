//! Frames of the Tokyo extract and the demo styles, drawn offscreen: what
//! camera-only frames upload, that views reached by moving draw what fresh
//! views draw, eviction and memory, idle, and labels.

mod common;

mod tests {
    fn tokyo(zoom: f64) -> crate::common::Options {
        crate::common::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((orb_crs::Geodetic::new(139.74, 35.685), zoom, 0.0, 0.0)),
        }
    }

    /// Every frame is right for its camera over terrain (decision 0019):
    /// a frame of a rotation over the Alps that never stops, once its
    /// tiles have loaded, draws what a fresh map at rest at its camera
    /// draws. Drape slots never depend on the camera, nor wait for it to
    /// stop. Labels, which fade in after a move, are hidden. Needs the
    /// network:
    /// `cargo test --release -p orb-demo -- --ignored rotating`.
    #[test]
    #[ignore = "fetches tiles over the network"]
    fn rotating_over_the_alps_matches_a_fresh_view() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        let at = |bearing| crate::common::Options {
            style: Some(format!("file://{root}/web/demo/data/alps.json")),
            at: Some((orb_crs::Geodetic::new(7.7491, 46.0207), 11.6, bearing, 72.0)),
            ..tokyo(11.6)
        };
        let hide_labels = |offscreen: &mut crate::common::Offscreen| {
            let commands = r#"[["setLayoutProperty", "peaks", "visibility", "none"],
                ["setLayoutProperty", "places", "visibility", "none"]]"#;
            let edit = orb_runtime::Transaction::from_commands(commands).unwrap();
            offscreen.map.edit(&edit).unwrap();
            settle(offscreen);
        };
        let set_bearing = |offscreen: &mut crate::common::Offscreen, bearing: f64| {
            crate::common::move_camera(&mut offscreen.map, |c| c.bearing = bearing);
        };
        let (mut moved, _, _) = crate::common::Offscreen::load(&at(-72.0)).unwrap();
        hide_labels(&mut moved);
        // Load what the arrival shows, then turn back without waiting.
        set_bearing(&mut moved, -32.0);
        settle(&mut moved);
        set_bearing(&mut moved, -72.0);
        settle(&mut moved);
        for step in 1..10 {
            set_bearing(&mut moved, -72.0 + 4.0 * f64::from(step));
            moved.frame();
        }
        // The camera never stops: it turns half a degree and back each
        // frame until a frame at the bearing has everything loaded.
        for frame in 0.. {
            set_bearing(&mut moved, -32.0);
            let report = moved.frame();
            if report.tiles_loading == 0 && report.loaded {
                break;
            }
            assert!(frame < 2000, "tiles did not load: {report:?}");
            set_bearing(&mut moved, -31.5);
            moved.frame();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let pixels = moved.pixels();
        let (mut fresh, _, _) = crate::common::Offscreen::load(&at(-32.0)).unwrap();
        hide_labels(&mut fresh);
        fresh.map.jump_to_state(*moved.map.camera());
        settle(&mut fresh);
        let reference = fresh.pixels();
        let differing = crate::common::differing_pixels(&pixels, &reference);
        eprintln!("rotate: {differing} differing pixels");
        if differing > 0 {
            let dir = tempfile_dir("orb-demo-rotate");
            for (name, image) in [("moved.png", &pixels), ("fresh.png", &reference)] {
                crate::common::write_png(dir.join(name).to_str().unwrap(), image).unwrap();
            }
            eprintln!("rotate: frames in {}", dir.display());
        }
        // DEM borders come from the drawn neighbours alone (#84), but a
        // few ridge pixels, up to a dozen, still differ now and then; a
        // slot drawn for another camera would cover thousands.
        assert!(
            differing <= 16,
            "{differing} mid-rotation pixels differ from a fresh view"
        );
    }

    /// A new directory for a test's frames.
    fn tempfile_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    /// The Alps flight through a camera store, as the demos
    /// fly them from the moment the style loads, with tiles loading in wall
    /// time: how far the eye and its near plane stay above the map's
    /// full-resolution terrain, taking the terrain at each revision. Needs
    /// the network: `cargo test
    /// --release -p orb-demo -- --ignored flight`.
    #[test]
    #[ignore = "fetches tiles over the network"]
    fn the_alps_flight_stays_above_the_terrain() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        let options = crate::common::Options {
            style: Some(format!("file://{root}/web/demo/data/alps.json")),
            at: None,
            ..tokyo(0.0)
        };
        let flight =
            crate::common::Flight::read(&format!("{root}/web/demo/data/alps-flight.json")).unwrap();
        {
            let mut offscreen =
                crate::common::Offscreen::new(&options, orb_runtime::SceneCapacity::default())
                    .unwrap();
            // The style's camera.
            while offscreen.map.elevation_revision() == 0 {
                offscreen.frame();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let (viewport, surface) = (offscreen.viewport, offscreen.map.surface());
            let (mut lowest, mut under, mut frames) = (f64::INFINITY, 0, 0);
            let (mut last, mut steps) = (None, Vec::new());
            for (target, fly) in &flight.legs {
                offscreen.map.fly_to(target, fly);
                while offscreen.map.is_moving() {
                    offscreen.frame_at(viewport);
                    let camera = *offscreen.map.camera();
                    let Some(full) = offscreen.map.elevation() else {
                        continue;
                    };
                    let view = orb_camera::View::new(&camera, viewport, &surface);
                    let Some(clearance) = view.clearance(&*full) else {
                        continue;
                    };
                    frames += 1;
                    lowest = lowest.min(clearance);
                    under += usize::from(clearance < 0.0);
                    // The camera's elevation step per frame, for jumps: the
                    // first DEM data under it moves it once.
                    let elevation = camera.elevation;
                    if let Some(previous) = last.replace(elevation) {
                        steps.push((elevation - previous).abs());
                    }
                }
            }
            steps.sort_by(f64::total_cmp);
            println!(
                "{frames} frames, lowest clearance {lowest:.1} m, {under} frames cut into the terrain; elevation steps per frame: p99 {:.1} m, largest {:.1} m, second {:.1} m",
                steps[steps.len() * 99 / 100],
                steps[steps.len() - 1],
                steps[steps.len() - 2],
            );
            assert_eq!(under, 0);
        }
    }

    /// Zooms by `delta` levels around a pixel, as the gestures do.
    fn zoom_by(
        map: &mut orb_runtime::Map,
        viewport: orb_runtime::Viewport,
        at: glam::DVec2,
        delta: f64,
    ) {
        let surface = map.surface();
        let zoom = map.zoom() + delta;
        crate::common::move_camera(map, |c| c.zoom_around(&surface, viewport, at, zoom));
    }

    /// Pans the ground under `from` to `to`, as the gestures do.
    fn pan(
        map: &mut orb_runtime::Map,
        viewport: orb_runtime::Viewport,
        from: glam::DVec2,
        to: glam::DVec2,
    ) {
        let surface = map.surface();
        crate::common::move_camera(map, |c| c.pan(&surface, viewport, from, to));
    }

    /// Renders until two frames in a row are idle: the second draws what
    /// the first's messages installed, as the fixture runner's last frame
    /// does.
    fn settle(offscreen: &mut crate::common::Offscreen) -> orb_runtime::FrameReport {
        let started = std::time::Instant::now();
        let mut was_idle = false;
        loop {
            let report = offscreen.frame();
            if report.idle && was_idle {
                assert!(crate::common::take_warnings(&mut offscreen.map).is_empty());
                return report;
            }
            was_idle = report.idle;
            assert!(started.elapsed().as_secs() < 60, "tiles did not settle");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// A fling past zoom 0 lands a hair below it; the map must still draw
    /// there, as GL JS's does.
    #[test]
    fn draws_just_below_zoom_zero() {
        let options = crate::common::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((orb_crs::Geodetic::new(139.74, 35.685), 3.0, 0.0, 0.0)),
        };
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
        let projection = offscreen.map.projection();
        crate::common::move_camera(&mut offscreen.map, |c| c.set_zoom(&projection, -1e-9));
        let mut report = offscreen.frame();
        while !report.idle {
            std::thread::sleep(std::time::Duration::from_millis(5));
            report = offscreen.frame();
        }
        report = offscreen.frame();
        assert!(report.frame.candidates > 0, "{report:?}");
        assert_ne!(
            &offscreen.pixels()[..4],
            [0, 0, 0, 0],
            "the background draws"
        );
    }

    /// Tiles never draw with some of their layers missing: when the scene
    /// is full, uploads evict unused cached tiles, so a small scene that has
    /// zoomed through several levels draws what an unbounded one does.
    #[test]
    fn a_full_scene_evicts_cached_tiles_instead_of_dropping_layers() {
        let at = |zoom| crate::common::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((orb_crs::Geodetic::new(139.74, 35.685), zoom, 0.0, 0.0)),
        };
        let (_, report, _) = crate::common::Offscreen::load(&at(12.0)).unwrap();
        let used = report.scene;
        // Labels take more property words at the start than at 12.
        let (_, start, _) = crate::common::Offscreen::load(&at(15.0)).unwrap();
        // Room for the views on the way, with their fallbacks, but not for
        // every level's tiles at once: vertex words run out.
        let capacity = crate::common::SceneCapacity {
            vertex_words: used.vertex_words * 7 / 4,
            index_words: used.index_words * 2,
            prop_words: (used.prop_words.max(start.scene.prop_words) * 2).max(1024),
            ..crate::common::SceneCapacity::fixed()
        };
        let zoom_out = |capacity| {
            let (mut offscreen, _, _) =
                crate::common::Offscreen::load_with(&at(15.0), capacity).unwrap();
            let viewport = offscreen.viewport;
            let center = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
            let mut report = offscreen.frame();
            for _ in 0..6 {
                zoom_by(&mut offscreen.map, viewport, center, -0.5);
                report = offscreen.frame();
                while !report.idle {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    report = offscreen.frame();
                }
                report = offscreen.frame();
                let warnings = crate::common::take_warnings(&mut offscreen.map);
                assert!(
                    !warnings.iter().any(|w| w.contains("scene memory")),
                    "{warnings:?}"
                );
            }
            (offscreen, report)
        };
        let (unbounded, report) = zoom_out(crate::common::SceneCapacity::default());
        assert!(
            report.scene.vertex_words > capacity.vertex_words,
            "the path must need eviction to fit: {} cached, {} budget",
            report.scene.vertex_words,
            capacity.vertex_words
        );
        let reference = unbounded.pixels();
        let (zoomed, _) = zoom_out(capacity);
        let pixels = zoomed.pixels();
        let differing = crate::common::differing_pixels(&pixels, &reference);
        assert_eq!(differing, 0, "pixels differ from an unbounded scene");
    }

    /// A tile the scene can't hold even with nothing else to wait for is
    /// an error event and a warning, never a silent hole: its coarser
    /// stand-in draws instead.
    #[test]
    fn tiles_that_cannot_fit_are_errors() {
        let capacity = crate::common::SceneCapacity {
            vertex_words: 2048,
            ..crate::common::SceneCapacity::fixed()
        };
        let mut offscreen = crate::common::Offscreen::new(&tokyo(15.0), capacity).unwrap();
        let started = std::time::Instant::now();
        let mut errors = Vec::new();
        loop {
            let report = offscreen.frame();
            for event in offscreen.map.take_events() {
                if let orb_runtime::MapEvent::Error { message } = event {
                    errors.push(message);
                }
            }
            if report.idle {
                break;
            }
            assert!(started.elapsed().as_secs() < 60, "did not settle");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            errors
                .iter()
                .any(|e| e.contains("doesn't fit in scene memory")),
            "{errors:?}"
        );
    }

    /// Raster tiles make room as vector tiles do (decision 0032): with the
    /// scene full of cached vector tiles the view no longer uses, arriving
    /// raster tiles evict them rather than being dropped, which on the
    /// globe left Liberty's ocean (its shaded-relief raster) with
    /// tile-shaped gaps.
    #[test]
    fn raster_tiles_evict_unused_tiles_for_room() {
        let dir = std::env::temp_dir().join(format!("orbis-raster-room-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tile = dir.join("tile.png");
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 256, 256);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&[64u8, 128, 255, 255].repeat(256 * 256))
                .unwrap();
        }
        std::fs::write(&tile, png).unwrap();
        // A polygon that only low zooms draw; imagery only from zoom 5.
        let ring: Vec<String> = (0..=360)
            .map(|i| {
                let a = f64::from(i).to_radians();
                format!("[{:.4},{:.4}]", 60.0 * a.cos(), 45.0 * a.sin())
            })
            .collect();
        let style = format!(
            r#"{{"version":8,"sources":{{
                "shapes":{{"type":"geojson","data":{{"type":"Polygon","coordinates":[[{ring}]]}}}},
                "imagery":{{"type":"raster","tiles":["file://{tile}"],"tileSize":256}}}},
              "layers":[{{"id":"fill","type":"fill","source":"shapes","maxzoom":4.5}},
                {{"id":"raster","type":"raster","source":"imagery","minzoom":5}}]}}"#,
            ring = ring.join(","),
            tile = tile.display(),
        );
        let path = dir.join("style.json");
        std::fs::write(&path, style).unwrap();
        let options = |lon: f64, zoom: f64| crate::common::Options {
            style: Some(format!("file://{}", path.display())),
            at: Some((orb_crs::Geodetic::new(lon, 0.0), zoom, 0.0, 0.0)),
            ..tokyo(zoom)
        };
        let settle = |offscreen: &mut crate::common::Offscreen| {
            let started = std::time::Instant::now();
            let mut warnings = Vec::new();
            loop {
                let report = offscreen.frame();
                warnings.extend(crate::common::take_warnings(&mut offscreen.map));
                if report.idle {
                    return (report, warnings);
                }
                assert!(started.elapsed().as_secs() < 60, "did not settle");
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        };
        // The scene holds exactly the low-zoom view's vector tiles.
        let (_, low, _) = crate::common::Offscreen::load(&options(0.0, 3.0)).unwrap();
        let capacity = crate::common::SceneCapacity {
            vertex_words: low.scene.vertex_words,
            index_words: low.scene.index_words,
            ..crate::common::SceneCapacity::fixed()
        };
        let (mut offscreen, _, _) =
            crate::common::Offscreen::load_with(&options(0.0, 3.0), capacity).unwrap();
        let (_, warnings) = settle(&mut offscreen);
        assert!(warnings.is_empty(), "{warnings:?}");
        offscreen.map.jump_to_state(orb_runtime::CameraState::new(
            orb_crs::Geodetic::new(150.0, 0.0),
            6.0,
            &offscreen.map.projection(),
        ));
        let (report, warnings) = settle(&mut offscreen);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(report.tiles_drawn > 0);
    }

    /// Plan §8: tiles out of view stay cached by their memory, within a
    /// quarter of the budget, the least recently used giving way first.
    #[test]
    fn tile_cache_stays_within_its_memory_budget() {
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&tokyo(14.0)).unwrap();
        offscreen.map.set_memory_budget(16 << 20);
        let viewport = offscreen.viewport;
        let center = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        let mut most = 0;
        for step in 0..8 {
            let angle = f64::from(step) * 0.8;
            pan(
                &mut offscreen.map,
                viewport,
                center,
                center + glam::DVec2::new(angle.cos(), angle.sin()) * 600.0,
            );
            let report = settle(&mut offscreen);
            let memory = report.memory;
            assert_eq!(memory.cache_budget, 4 << 20);
            assert!(memory.cached <= memory.cache_budget, "{memory:?}");
            most = most.max(memory.cached_tiles);
        }
        assert!(most > 0, "tiles out of view are kept");
    }

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct HeldExecutor(Mutex<Vec<orb_jobs::Job>>);

    impl orb_jobs::Executor for HeldExecutor {
        fn spawn(&self, job: orb_jobs::Job) {
            self.0.lock().unwrap().push(job);
        }
    }

    /// Plan §7.5: once idle, a map, its camera and its gestures ask for no
    /// frames: no wake-ups, no animation, nothing that redraws.
    #[test]
    fn idle_maps_ask_for_no_frames() {
        let wakes = Arc::new(AtomicUsize::new(0));
        let wake_count = wakes.clone();
        let services = crate::common::native_services(Arc::new(move || {
            wake_count.fetch_add(1, Ordering::SeqCst);
        }));
        let mut offscreen = crate::common::Offscreen::load(&tokyo(15.0)).unwrap().0;
        offscreen.map = crate::common::Map::new(services, crate::common::camera(&tokyo(15.0)));
        offscreen.renderer = None;
        crate::common::load_style(&mut offscreen.map, &tokyo(15.0)).unwrap();
        let started = std::time::Instant::now();
        let mut events = Vec::new();
        let mut was_idle = false;
        loop {
            let report = offscreen.frame();
            events.extend(offscreen.map.take_events());
            if report.idle && was_idle {
                break;
            }
            was_idle = report.idle;
            assert!(started.elapsed().as_secs() < 60, "tiles did not settle");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(
            events
                .iter()
                .filter(|e| **e == orb_runtime::MapEvent::Idle)
                .count(),
            1
        );
        let before = wakes.load(Ordering::SeqCst);
        let started = std::time::Instant::now();
        while started.elapsed() < std::time::Duration::from_secs(2) {
            assert!(!offscreen.map.is_moving());
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
        assert_eq!(
            wakes.load(Ordering::SeqCst),
            before,
            "no wake-ups while idle"
        );
        // The next frame, were one drawn, is still idle and reports nothing.
        assert!(offscreen.frame().idle);
        assert!(offscreen.map.take_events().is_empty());
    }

    #[test]
    fn delayed_results_wake_and_publish_without_camera_input() {
        let options = crate::common::Options {
            style: None,
            encoding: "mvt".into(),
            at: None,
        };
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
        let executor = Arc::new(HeldExecutor::default());
        let wakes = Arc::new(AtomicUsize::new(0));
        let wake_count = wakes.clone();
        let fetch = crate::common::NativeFetch::new();
        let services = crate::common::Services {
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(crate::common::NativeImageDecode),
            images: Default::default(),
            jobs: crate::common::PoolJobs::new(
                fetch.clone(),
                Arc::new(crate::common::NativeImageDecode),
                executor.clone(),
            ),
            fetch,
            wake: Arc::new(move || {
                wake_count.fetch_add(1, Ordering::SeqCst);
            }),
        };
        let camera = crate::common::CameraState::new(
            orb_crs::Geodetic::new(0.0, 0.0),
            1.5,
            &crate::common::web_mercator(),
        );
        offscreen.map = crate::common::Map::new(services, Some(camera));
        offscreen.renderer = None;
        offscreen
            .map
            .set_style_json(
                r##"{
            "version":8,
            "sources":{"test":{"type":"geojson","data":{
                "type":"Polygon","coordinates":[[[-40,-40],[40,-40],[40,40],[-40,40],[-40,-40]]]
            }}},
            "layers":[{"id":"fill","type":"fill","source":"test","paint":{"fill-color":"#ff0000"}}]
        }"##,
            )
            .unwrap();
        let first = offscreen.frame();
        assert!(!first.idle);
        assert_eq!(first.tiles_drawn, 0);
        assert!(first.tiles_loading > 0);
        let waiting = offscreen.frame();
        assert_eq!(waiting.tiles_loading, first.tiles_loading);
        let jobs = std::mem::take(&mut *executor.0.lock().unwrap());
        assert_eq!(jobs.len(), first.tiles_loading as usize);
        for job in jobs.into_iter().rev() {
            job();
        }
        assert_eq!(wakes.load(Ordering::SeqCst), first.tiles_loading as usize);
        // Each frame applies at least one arrived tile (plan §7.4's budget).
        let published = (0..first.tiles_loading)
            .map(|_| offscreen.frame())
            .find(|report| report.idle)
            .expect("one frame per arrived tile at most");
        assert_eq!(published.tiles_loading, 0);
        assert!(published.tiles_drawn > 0);
        assert_eq!(
            offscreen.map.zoom(),
            camera.zoom(&crate::common::web_mercator())
        );
        assert_eq!(offscreen.pixels(), {
            offscreen.frame();
            offscreen.pixels()
        });
        assert!(crate::common::take_warnings(&mut offscreen.map).is_empty());
    }

    #[test]
    fn camera_only_frames_touch_only_placements() {
        for patterned in [false, true] {
            let options = crate::common::Options {
                style: None,
                encoding: "mvt".into(),
                at: None,
            };
            let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
            if patterned {
                let sprite = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../web/demo/data/sprites/ofm");
                offscreen.map.set_style_json(&format!(r#"{{"version":8,"sprite":"file://{}","sources":{{}},"layers":[{{"id":"pattern","type":"background","paint":{{"background-pattern":"wetland"}}}}]}}"#, sprite.display())).unwrap();
                let start = std::time::Instant::now();
                while !offscreen.frame().idle {
                    assert!(start.elapsed().as_secs() < 10);
                    std::thread::yield_now();
                }
                offscreen.frame();
            }
            let viewport = offscreen.viewport;
            pan(
                &mut offscreen.map,
                viewport,
                glam::DVec2::new(512.0, 384.0),
                glam::DVec2::new(520.0, 380.0),
            );
            let report = offscreen.frame();
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            assert!(report.tiles_drawn > 0);
            assert_eq!(report.frame.placements_written, 0);
            assert_eq!(report.tiles_loading, 0);
        }
    }

    #[test]
    fn sky_keeps_camera_only_uploads_to_placements() {
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&tokyo(14.0)).unwrap();
        offscreen.map.set_style_json(r##"{"version":8,"sources":{},"sky":{},"layers":[{"id":"background","type":"background","paint":{"background-color":"#ccaa83"}}]}"##).unwrap();
        offscreen.frame();
        offscreen.frame();
        for pitch in [60.0, 80.0, 85.0] {
            crate::common::move_camera(&mut offscreen.map, |c| c.pitch = pitch);
            let report = offscreen.frame();
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            assert_eq!(report.frame.draws, 2);
            assert!(report.idle);
        }
    }

    #[test]
    fn extrusions_keep_camera_only_uploads_to_placements() {
        let options = crate::common::Options {
            style: Some(format!(
                "file://{}/tests/data/extrusion.json",
                env!("CARGO_MANIFEST_DIR")
            )),
            encoding: "mvt".into(),
            at: Some((orb_crs::Geodetic::new(0.0, 0.0), 19.0, -20.0, 60.0)),
        };
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
        for (bearing, pitch) in [(31.0, 61.0), (10.0, 40.0)] {
            crate::common::move_camera(&mut offscreen.map, |c| {
                c.bearing = bearing;
                c.pitch = pitch;
            });
            let report = offscreen.frame();
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            assert!(report.frame.placements_written < report.tiles_drawn);
            assert!(report.tiles_drawn > 0);
        }
        // Switching 1×/4× rebuilds dependent targets and bundles once; the
        // next camera-only frame still replays them without paint writes.
        for enabled in [false, true] {
            offscreen.map.set_antialias_3d(enabled);
            let changed = offscreen.frame();
            assert!(changed.frame.bundle_recorded);
            let replay = offscreen.frame();
            assert!(!replay.frame.bundle_recorded);
            assert_eq!(replay.frame.paints_written, 0);
            assert_eq!(replay.frame.dispatches, 1);
        }
        assert!(
            !crate::common::take_warnings(&mut offscreen.map)
                .iter()
                .any(|w| w.contains("fill-extrusion"))
        );
    }

    /// A scene's lights (decision 0026) shade an extrusion's roof in the
    /// response v8's light uses, and emissive strength shows its colour.
    #[test]
    fn scene_lights_shade_extrusions() {
        let options = crate::common::Options {
            style: Some(format!(
                "file://{}/tests/data/extrusion.json",
                env!("CARGO_MANIFEST_DIR")
            )),
            encoding: "mvt".into(),
            at: Some((orb_crs::Geodetic::new(-0.00015, -0.00025), 19.0, 0.0, 0.0)),
        };
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
        let style = r##"{"version": 8,
            "sources": {"b": {"type": "geojson", "data": {"type": "Polygon",
                "coordinates": [[[-0.0003, -0.0004], [-0.0003, -0.0001], [0, -0.0001], [0, -0.0004], [-0.0003, -0.0004]]]}}},
            "layers": [
                {"id": "ground", "type": "background", "paint": {"background-color": "#000000"}},
                {"id": "block", "type": "fill-extrusion", "source": "b",
                 "paint": {"fill-extrusion-height": 10, "fill-extrusion-color": "#808080"}}]}"##;
        let mut roof = |direction: &str, emissive: f64| {
            let scene = format!(
                r#"{{"style": {style}, "lights": [
                    {{"type": "ambient", "properties": {{"intensity": 0.5}}}},
                    {{"type": "directional", "properties": {{"direction": {direction}, "intensity": 0.5}}}}],
                    "layers": {{"block": {{"emissive-strength": {emissive}}}}}}}"#
            );
            offscreen.map.set_style_json(&scene).unwrap();
            let projection = offscreen.map.projection();
            offscreen.map.jump_to_state(orb_runtime::CameraState::new(
                orb_crs::Geodetic::new(-0.00015, -0.00025),
                19.0,
                &projection,
            ));
            settle(&mut offscreen);
            let pixels = offscreen.pixels();
            let at = (crate::common::Offscreen::HEIGHT as usize / 2
                * crate::common::Offscreen::WIDTH as usize
                + crate::common::Offscreen::WIDTH as usize / 2)
                * 4;
            pixels[at]
        };
        // GL JS's flat response (decision 0026). Overhead the roof gets
        // max(0.5 + 0.5 − its luminance, 1) of its colour lifted by 0.03.
        assert!(roof("[0, 0]", 0.0).abs_diff(136) <= 1);
        // At the horizon only the ambient level, 0.5.
        assert!(roof("[90, 90]", 0.0).abs_diff(68) <= 1);
        // Fully emissive: its own colour.
        assert!(roof("[90, 90]", 1.0).abs_diff(128) <= 1);
    }

    #[test]
    fn point_and_line_families_keep_camera_only_uploads_to_placements() {
        for style in [
            r#"{
            "version":8,
            "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
            "layers":[{"id":"heat","type":"heatmap","source":"s"},{"id":"circle","type":"circle","source":"s"}]
        }"#,
            r#"{
            "version":8,
            "sources":{"s":{"type":"geojson","lineMetrics":true,"data":{"type":"LineString","coordinates":[[-30,0],[30,0]]}}},
            "layers":[
                {"id":"dash","type":"line","source":"s","paint":{"line-width":8,"line-dasharray":[2,1]}},
                {"id":"gradient","type":"line","source":"s","paint":{"line-width":5,"line-gradient":["interpolate",["linear"],["line-progress"],0,"red",1,"blue"],"line-dasharray":[1,1]}}
            ]
        }"#,
        ] {
            let options = crate::common::Options {
                style: None,
                encoding: "mvt".into(),
                at: None,
            };
            let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
            let fetch = crate::common::NativeFetch::new();
            let services = crate::common::Services {
                clock: Arc::new(orb_host_native::NativeClock::default()),
                image_decode: Arc::new(crate::common::NativeImageDecode),
                images: Default::default(),
                jobs: crate::common::PoolJobs::new(
                    fetch.clone(),
                    Arc::new(crate::common::NativeImageDecode),
                    Arc::new(orb_jobs::Inline),
                ),
                fetch,
                wake: Arc::new(|| {}),
            };
            let camera = crate::common::CameraState::new(
                orb_crs::Geodetic::new(0.0, 0.0),
                1.5,
                &crate::common::web_mercator(),
            );
            offscreen.map = crate::common::Map::new(services, Some(camera));
            offscreen.renderer = None;
            offscreen.map.set_style_json(style).unwrap();
            // Inline jobs publish their completion on the following frame.
            let mut loaded = false;
            for _ in 0..8 {
                let report = offscreen.frame();
                if report.idle && report.tiles_drawn > 0 {
                    loaded = true;
                    break;
                }
            }
            assert!(loaded);
            pan(
                &mut offscreen.map,
                offscreen.viewport,
                glam::DVec2::new(512.0, 384.0),
                glam::DVec2::new(520.0, 380.0),
            );
            let report = offscreen.frame();
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            assert_eq!(report.frame.placements_written, 0);
            assert_eq!(report.tiles_loading, 0);
            assert!(crate::common::take_warnings(&mut offscreen.map).is_empty());
        }
    }

    /// A style with a symbol layer loads, places and goes idle. Without a
    /// glyphs URL its text needs local fonts, which warn.
    /// Plan §13: a placement commit's semantic snapshot reads the labels it
    /// shows.
    #[test]
    fn semantic_snapshots_read_shown_labels() {
        let options = crate::common::Options {
            style: None,
            encoding: "mvt".into(),
            at: None,
        };
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
        let fetch = crate::common::NativeFetch::new();
        let jobs = crate::common::PoolJobs::new(
            fetch.clone(),
            Arc::new(crate::common::NativeImageDecode),
            Arc::new(orb_jobs::Inline),
        );
        jobs.set_local_glyphs(orb_host_native::SystemFonts::shared());
        let services = crate::common::Services {
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(crate::common::NativeImageDecode),
            images: Default::default(),
            jobs,
            fetch,
            wake: Arc::new(|| {}),
        };
        let camera = crate::common::CameraState::new(
            orb_crs::Geodetic::new(0.0, 0.0),
            1.5,
            &crate::common::web_mercator(),
        );
        offscreen.map = crate::common::Map::new(services, Some(camera));
        offscreen.renderer = None;
        offscreen
            .map
            .set_style_json(
                r#"{"version":8,
                "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
                "layers":[{"id":"label","type":"symbol","source":"s","layout":{"text-field":"Null Island"}}]}"#,
            )
            .unwrap();
        assert!(
            offscreen.map.semantic_snapshot().is_none(),
            "nothing placed yet"
        );
        // The commit comes from the placement thread.
        let started = std::time::Instant::now();
        let mut snapshot = None;
        while snapshot.is_none() && started.elapsed() < std::time::Duration::from_secs(10) {
            let report = offscreen.frame();
            if report.idle && report.tiles_drawn > 0 {
                snapshot = offscreen.map.semantic_snapshot();
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let snapshot = snapshot.expect("labels were placed");
        let node = &snapshot.nodes[snapshot.reading_order[0] as usize];
        assert_eq!(
            (node.text.as_str(), node.role, node.layer.as_str()),
            ("Null Island", orb_a11y::Role::Label, "label")
        );
        assert!(node.distance < 1.0, "{node:?}");
        let [x0, y0, x1, y1] = node.bounds;
        let center = (
            f64::from(crate::common::Offscreen::WIDTH) / 2.0,
            f64::from(crate::common::Offscreen::HEIGHT) / 2.0,
        );
        assert!(
            x0 < center.0 && center.0 < x1 && y0 < center.1 && center.1 < y1,
            "{node:?}"
        );
        assert_eq!(snapshot.summary.highlights, vec!["Null Island"]);
    }

    #[test]
    fn styles_with_symbol_layers_render() {
        let options = crate::common::Options {
            style: None,
            encoding: "mvt".into(),
            at: None,
        };
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
        let fetch = crate::common::NativeFetch::new();
        let services = crate::common::Services {
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(crate::common::NativeImageDecode),
            images: Default::default(),
            jobs: crate::common::PoolJobs::new(
                fetch.clone(),
                Arc::new(crate::common::NativeImageDecode),
                Arc::new(orb_jobs::Inline),
            ),
            fetch,
            wake: Arc::new(|| {}),
        };
        let camera = crate::common::CameraState::new(
            orb_crs::Geodetic::new(0.0, 0.0),
            1.5,
            &crate::common::web_mercator(),
        );
        offscreen.map = crate::common::Map::new(services, Some(camera));
        offscreen.renderer = None;
        offscreen
            .map
            .set_style_json(
                r#"{"version":8,
                "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
                "layers":[
                    {"id":"circle","type":"circle","source":"s"},
                    {"id":"label","type":"symbol","source":"s","layout":{"text-field":"Null Island"}}
                ]}"#,
            )
            .unwrap();
        let mut loaded = false;
        for _ in 0..8 {
            let report = offscreen.frame();
            if report.idle && report.tiles_drawn > 0 {
                loaded = true;
                break;
            }
        }
        assert!(loaded);
        let warnings = crate::common::take_warnings(&mut offscreen.map);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("glyphs that need local fonts")),
            "{warnings:?}"
        );
    }

    #[test]
    fn mvt_and_mlt_extract_pixels_match() {
        let mut images = Vec::new();
        for encoding in ["mvt", "mlt"] {
            let options = crate::common::Options {
                style: None,
                encoding: encoding.into(),
                at: None,
            };
            images.push(crate::common::snapshot(&options));
        }
        let differing = crate::common::differing_pixels(&images[0], &images[1]);
        assert_eq!(differing, 0, "MVT/MLT differing pixels");
    }
}

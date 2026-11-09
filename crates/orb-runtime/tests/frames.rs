//! Frames of the Tokyo extract and the demo styles, drawn offscreen: what
//! camera-only frames upload, eviction and memory, idle, and labels.
//! Moved views against fresh ones are `orb-api`'s `path` tests.

mod common;

mod tests {
    fn tokyo(zoom: f64) -> crate::common::Options {
        crate::common::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((orb_crs::Geodetic::new(139.74, 35.685), zoom, 0.0, 0.0)),
        }
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

    /// Raster tiles make room as vector tiles do: with the
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
        // A polygon that only low zooms draw, and imagery.
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
                {{"id":"raster","type":"raster","source":"imagery"}}]}}"#,
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
        // The scene holds exactly the low-zoom view's tiles.
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

    /// A source whose layers all hide at the camera's zoom loads and draws
    /// nothing there, as in GL JS.
    #[test]
    fn sources_load_nothing_where_their_layers_hide() {
        let dir = std::env::temp_dir().join(format!("orbis-hidden-{}", std::process::id()));
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
        let path = dir.join("style.json");
        std::fs::write(
            &path,
            format!(
                r#"{{"version":8,"sources":{{
                    "imagery":{{"type":"raster","tiles":["file://{}"],"tileSize":256}}}},
                  "layers":[{{"id":"raster","type":"raster","source":"imagery","maxzoom":7}}]}}"#,
                tile.display()
            ),
        )
        .unwrap();
        let drawn = |zoom: f64| {
            let options = crate::common::Options {
                style: Some(format!("file://{}", path.display())),
                at: Some((orb_crs::Geodetic::new(0.0, 0.0), zoom, 0.0, 0.0)),
                ..tokyo(zoom)
            };
            let (_, report, _) = crate::common::Offscreen::load(&options).unwrap();
            (report.tiles_drawn, report.scene.meshes)
        };
        let (shown, hidden) = (drawn(6.0), drawn(10.0));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(shown.0 > 0 && shown.1 > 0, "{shown:?}");
        assert_eq!(hidden, (0, 0));
    }

    /// Tiles out of view stay cached by their memory, within a
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
    struct HeldExecutor(Mutex<Vec<orb_host::jobs::Job>>);

    impl orb_host::jobs::Executor for HeldExecutor {
        fn spawn(&self, job: orb_host::jobs::Job) {
            self.0.lock().unwrap().push(job);
        }
    }

    /// Once idle, a map, its camera and its gestures ask for no
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
        // Each frame applies at least one arrived tile.
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

    /// A layer that drew, hidden by its zoom while the pipelines it draws
    /// with change (here the 3D sample count), holds no frame back while
    /// it hides, and draws what a fresh map does when a zoom shows it
    /// again.
    #[test]
    fn a_layer_shown_again_after_its_pipelines_changed_draws() {
        let style = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/extrusion.json"
        ))
        .unwrap()
        .replacen(
            r#""source": "blocks","#,
            r#""source": "blocks", "minzoom": 18,"#,
            1,
        );
        let at = (orb_crs::Geodetic::new(0.0, 0.0), 19.0, -20.0, 60.0);
        let load = |antialias: bool| {
            let options = crate::common::Options {
                style: None,
                encoding: "mvt".into(),
                at: Some(at),
            };
            let mut offscreen =
                crate::common::Offscreen::new(&options, orb_runtime::SceneCapacity::default())
                    .unwrap();
            offscreen.map.set_style_json(&style).unwrap();
            offscreen.map.set_antialias_3d(antialias);
            settle(&mut offscreen);
            offscreen
        };
        let mut moved = load(true);
        let centre = glam::DVec2::new(512.0, 384.0);
        zoom_by(&mut moved.map, moved.viewport, centre, -3.0);
        settle(&mut moved);
        moved.map.set_antialias_3d(false);
        settle(&mut moved);
        zoom_by(&mut moved.map, moved.viewport, centre, 3.0);
        settle(&mut moved);
        let fresh = load(false);
        assert_eq!(
            crate::common::differing_pixels(&moved.pixels(), &fresh.pixels()),
            0
        );
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
                    Arc::new(orb_host::jobs::Inline),
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

    /// A placement commit from the placement thread shows a symbol
    /// layer's label, drawn with local fonts.
    #[test]
    fn placement_commits_show_labels() {
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
            Arc::new(orb_host::jobs::Inline),
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
            offscreen.map.shown_labels().is_empty(),
            "nothing placed yet"
        );
        // The commit comes from the placement thread.
        let started = std::time::Instant::now();
        let mut shown = 0.0;
        while shown < 1.0 && started.elapsed() < std::time::Duration::from_secs(10) {
            let report = offscreen.frame();
            if report.idle && report.tiles_drawn > 0 {
                shown = offscreen
                    .map
                    .shown_labels()
                    .get("Null Island")
                    .copied()
                    .unwrap_or(0.0);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(shown, 1.0, "the label was placed and faded in");
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
                Arc::new(orb_host::jobs::Inline),
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

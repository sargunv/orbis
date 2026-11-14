//! Frames of the Tokyo extract and the demo styles, drawn offscreen: what
//! camera-only frames upload, eviction and memory, idle, and labels.
//! Moved views against fresh ones are `orb-api`'s `path` tests.

mod common;

mod tests {
    /// Whether frames draw on the GPU frame driver, whose bundles draw
    /// the counts the cull writes, so that a camera move replays them.
    /// The CPU driver's (`ORB_DRIVER=cpu`, `cpu-tex`, `webgl2`: WebGL 2's,
    /// which draws nothing indirectly) hold their draws' counts and record
    /// again when a move changes what the cull keeps.
    fn gpu_driven() -> bool {
        std::env::var("ORB_DRIVER").map_or(true, |driver| driver.is_empty() || driver == "gpu")
    }

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
    /// the network: `cargo test --release -p orb-runtime --test frames
    /// -- --ignored flight`.
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
                    let Some(orb_crs::Metres(clearance)) = view.clearance(&*full) else {
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

    /// No frame writes more of tiles' words than its budget's bytes: a
    /// tile larger than that (the extract's take 0.3–3 MiB, their labels
    /// under 256 KiB) goes up over frames and shows whole once all of it
    /// is in, so the map at rest draws what one that takes each tile whole
    /// does.
    #[test]
    fn tiles_larger_than_a_frame_s_bytes_go_up_in_pieces() {
        let cap = 256 << 10;
        let tight = orb_runtime::Budget {
            ms: 1000.0,
            bytes: cap,
            ..orb_runtime::Budget::DESKTOP
        };
        let capacity = crate::common::SceneCapacity::default();
        let mut pieces = crate::common::Offscreen::new(&tokyo(14.0), capacity).unwrap();
        pieces.map.set_frame_budget(tight);
        let started = std::time::Instant::now();
        let (mut most, mut was_idle) = (0, false);
        loop {
            let report = pieces.frame();

            most = most.max(report.tile_bytes);
            if report.idle && was_idle {
                break;
            }
            was_idle = report.idle;
            assert!(started.elapsed().as_secs() < 60, "did not settle");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(most <= cap, "a frame uploaded {most} bytes of tiles");
        let (whole, _, _) = crate::common::Offscreen::load(&tokyo(14.0)).unwrap();
        let differing = crate::common::differing_pixels(&pieces.pixels(), &whole.pixels());
        assert_eq!(differing, 0, "pixels differ from tiles taken whole");
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
            "the path must need eviction to fit: {} cached, {} held",
            report.scene.vertex_words,
            capacity.vertex_words
        );
        let reference = unbounded.pixels();
        let (zoomed, _) = zoom_out(capacity);
        let pixels = zoomed.pixels();
        let differing = crate::common::differing_pixels(&pixels, &reference);
        assert_eq!(differing, 0, "pixels differ from an unbounded scene");
    }

    /// Nothing in the renderer knows about gestures: a map whose pointer is
    /// held down and still while its view loads loads as one let go, frame
    /// for frame, its memory growing as the view needs (the vertex and
    /// index pools a block at a time, copying nothing), and ends drawing
    /// what the other does.
    #[test]
    fn a_map_held_still_loads_as_one_let_go() {
        use orb_camera::input::{Button, Input, Modifiers, PointerKind};
        let at = crate::common::Options {
            at: Some((orb_crs::Geodetic::new(139.765, 35.68), 14.5, -12.0, 60.0)),
            ..tokyo(14.5)
        };
        let small = crate::common::SceneCapacity::growable();
        let mut held = crate::common::Offscreen::inline(&at, small).unwrap();
        let mut free = crate::common::Offscreen::inline(&at, small).unwrap();
        let viewport = held.viewport;
        let time = held.time;
        held.map.input(
            Input::PointerDown {
                id: 1,
                kind: PointerKind::Mouse,
                button: Button::Primary,
                position: glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0),
                modifiers: Modifiers::default(),
            },
            time,
        );
        // Frames of the two until each has loaded what its view shows:
        // their loads land alike, by the next frame.
        let (mut loaded, mut copied) = ([None; 2], 0);
        for frame in 0u32..1000 {
            for (i, map) in [&mut held, &mut free].into_iter().enumerate() {
                let report = map.frame();
                if i == 0 {
                    assert!(report.input, "the pointer holds the camera");
                    copied += report.resized.bytes_copied;
                }
                if report.loaded && report.tiles_loading == 0 {
                    loaded[i].get_or_insert(frame);
                }
            }
            if loaded.iter().all(Option::is_some) {
                break;
            }
        }
        let [held_at, free_at] = loaded.map(|f| f.expect("loaded"));
        assert_eq!(held_at, free_at, "frames to load, held and let go");
        let memory = held.frame().memory.scene;
        assert!(
            memory.vertices.allocated > u64::from(small.vertex_words) * 4,
            "the view grew the pools: {memory:#?}"
        );
        // Blocks copy nothing as the view fills them: what moves as it
        // grows (the property rows, the chunk table, the block of what
        // draws as points) copies no more than the rows and tables hold.
        // The CPU driver keeps the row tables in one data texture (WebGL
        // 2 binds few textures), which a block's new node rows move
        // whole.
        assert!(
            !gpu_driven() || copied <= memory.props.allocated + memory.tables,
            "{copied} bytes copied growing the scene: {memory:#?}"
        );
        crate::common::take_warnings(&mut held.map);
        crate::common::take_warnings(&mut free.map);
        let differing = crate::common::differing_pixels(&held.pixels(), &free.pixels());
        assert_eq!(differing, 0, "pixels differ from the map let go");
    }

    /// Coarser tiles left standing in by a zoom don't crowd out the tiles
    /// that replace them: a scene with room for the view, but not for the
    /// view and the levels zoomed through, ends where a fresh map does.
    /// Tiles that found no room were once marked missing, and their
    /// stand-ins drew, without the layers of the finer zoom, for good.
    #[test]
    fn stand_ins_make_room_for_the_tiles_they_stand_in_for() {
        let at = |zoom, pitch| crate::common::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((orb_crs::Geodetic::new(139.765, 35.68), zoom, -12.0, pitch)),
        };
        let (fresh, report, _) = crate::common::Offscreen::load(&at(15.0, 45.0)).unwrap();
        let used = report.scene;
        let reference = fresh.pixels();
        let capacity = crate::common::SceneCapacity {
            vertex_words: used.vertex_words * 5 / 4,
            index_words: used.index_words * 2,
            prop_words: (used.prop_words * 2).max(1024),
            ..crate::common::SceneCapacity::fixed()
        };
        let zoom_in = |capacity| {
            let (mut offscreen, _, _) =
                crate::common::Offscreen::load_with(&at(13.0, 0.0), capacity).unwrap();
            let mut most = 0;
            for step in 1..=4 {
                let t = f64::from(step) / 4.0;
                crate::common::move_camera(&mut offscreen.map, |c| {
                    *c = crate::common::camera(&at(13.0 + 2.0 * t, 45.0 * t)).unwrap();
                });
                let mut report = offscreen.frame();
                while !report.idle {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    report = offscreen.frame();
                }
                report = offscreen.frame();
                most = most.max(report.scene.vertex_words);
                crate::common::take_warnings(&mut offscreen.map);
            }
            (offscreen, most)
        };
        let (_, most) = zoom_in(crate::common::SceneCapacity::default());
        assert!(
            most > capacity.vertex_words,
            "the path must need its stand-ins gone to fit: {most} words, {} room",
            capacity.vertex_words
        );
        let (zoomed, _) = zoom_in(capacity);
        let differing = crate::common::differing_pixels(&zoomed.pixels(), &reference);
        assert_eq!(differing, 0, "pixels differ from a fresh map");
    }

    /// A tile the scene can't hold even with nothing else to wait for is
    /// an error event, never a silent hole: what's loaded stands in, and
    /// the tile is tried again later.
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
        let dir = crate::common::Scratch::new("raster-room");
        let tile = solid_tile(&dir);
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
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(report.tiles_drawn > 0);
    }

    /// A source whose layers all hide at the camera's zoom loads and draws
    /// nothing there, as in GL JS.
    #[test]
    fn sources_load_nothing_where_their_layers_hide() {
        let dir = crate::common::Scratch::new("hidden");
        let tile = solid_tile(&dir);
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
        assert!(shown.0 > 0 && shown.1 > 0, "{shown:?}");
        assert_eq!(hidden, (0, 0));
    }

    /// Raster tile pages give back what a view of many tiles grew them
    /// to once no tile holds their layers, as the scene's pools do: every
    /// chunk no tile holds goes whole, nothing moving, and the next view's
    /// tiles fill the page from its first chunk, as a fresh map's do. (Where
    /// tiles still hold layers, a page keeps the chunks they lie in, which
    /// may be more than a fresh map would need for them.)
    #[test]
    fn tile_pages_shrink_with_their_tiles() {
        let dir = crate::common::Scratch::new("pages");
        let tile = solid_tile(&dir);
        let path = dir.join("style.json");
        std::fs::write(
            &path,
            format!(
                r#"{{"version":8,"sources":{{
                    "imagery":{{"type":"raster","tiles":["file://{}"],"tileSize":256}}}},
                  "layers":[{{"id":"raster","type":"raster","source":"imagery","maxzoom":5}}]}}"#,
                tile.display()
            ),
        )
        .unwrap();
        let options = |zoom: f64, pitch: f64| crate::common::Options {
            style: Some(format!("file://{}", path.display())),
            at: Some((orb_crs::Geodetic::new(0.0, 0.0), zoom, 0.0, pitch)),
            ..tokyo(zoom)
        };
        let pages = |offscreen: &mut crate::common::Offscreen| {
            // The cache keeps nothing out of view.
            offscreen.map.release_cached_tiles();
            settle(offscreen).memory.tile_textures
        };
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&options(4.5, 60.0)).unwrap();
        let many = pages(&mut offscreen);
        let at = |offscreen: &mut crate::common::Offscreen, zoom: f64| {
            offscreen.map.jump_to_state(orb_runtime::CameraState::new(
                orb_crs::Geodetic::new(0.0, 0.0),
                zoom,
                &offscreen.map.projection(),
            ));
            pages(offscreen)
        };
        // Past the layer's maxzoom no tile holds a layer; back at zoom 1,
        // sixteen do.
        let none = at(&mut offscreen, 6.0);
        let few = at(&mut offscreen, 1.0);
        let (mut fresh, _, _) = crate::common::Offscreen::load(&options(1.0, 0.0)).unwrap();
        let fresh = pages(&mut fresh);
        assert!(many > 4 * fresh, "{many} bytes of pages at the tilted view");
        assert_eq!(none, 0, "pages of {many} bytes held {none} with no tile");
        assert_eq!(few, fresh, "the sixteen as a fresh map holds them");
    }

    /// A raster page that grows takes no other source's cached tiles,
    /// which hold none of its layers: back at the view it left, the map
    /// still shows a source whose tiles can no longer load.
    #[test]
    fn a_growing_page_keeps_other_sources_cached_tiles() {
        let dir = crate::common::Scratch::new("page-growth");
        let imagery = solid_tile(&dir);
        let overlay = solid_png(&dir, "overlay.png", [255, 0, 0, 255]);
        let path = dir.join("style.json");
        std::fs::write(
            &path,
            format!(
                r#"{{"version":8,"sources":{{
                    "imagery":{{"type":"raster","tiles":["file://{}"],"tileSize":256}},
                    "overlay":{{"type":"raster","tiles":["file://{}"],"tileSize":256}}}},
                  "layers":[{{"id":"imagery","type":"raster","source":"imagery"}},
                    {{"id":"overlay","type":"raster","source":"overlay","maxzoom":2}}]}}"#,
                imagery.display(),
                overlay.display()
            ),
        )
        .unwrap();
        let at = |zoom: f64, pitch: f64| {
            let mut camera = orb_runtime::CameraState::new(
                orb_crs::Geodetic::new(0.0, 0.0),
                zoom,
                &crate::common::web_mercator(),
            );
            camera.pitch = pitch;
            camera
        };
        let options = crate::common::Options {
            style: Some(format!("file://{}", path.display())),
            at: Some((orb_crs::Geodetic::new(0.0, 0.0), 1.0, 0.0, 0.0)),
            ..tokyo(1.0)
        };
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
        let first = settle(&mut offscreen).memory.tile_textures;
        let center = |offscreen: &crate::common::Offscreen| {
            let (w, h) = (offscreen.viewport.width, offscreen.viewport.height);
            let i = ((h / 2.0) as usize * w as usize + (w / 2.0) as usize) * 4;
            offscreen.pixels()[i..i + 3].to_vec()
        };
        assert_eq!(center(&offscreen), [255, 0, 0], "the overlay draws");
        // The overlay's tiles can't load again; past its layer's maxzoom
        // they stay cached while the imagery's page grows for a tilted
        // view of many tiles.
        std::fs::remove_file(&overlay).unwrap();
        offscreen.map.jump_to_state(at(4.5, 60.0));
        let grown = settle(&mut offscreen).memory.tile_textures;
        assert!(grown > first, "the page grew: {first} -> {grown} bytes");
        offscreen.map.jump_to_state(at(1.0, 0.0));
        let started = std::time::Instant::now();
        while !offscreen.frame().idle {
            assert!(started.elapsed().as_secs() < 60, "tiles did not settle");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        offscreen.frame();
        assert_eq!(center(&offscreen), [255, 0, 0], "the overlay's tiles kept");
    }

    /// Tiles out of view stay cached, two views' worth, the least recently
    /// used giving way first: panned off the extract's data, the
    /// view still keeps the city it came from.
    #[test]
    fn tile_cache_keeps_a_view_of_tiles() {
        let (mut offscreen, _, _) = crate::common::Offscreen::load(&tokyo(14.0)).unwrap();
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
            assert!(memory.cached_tiles <= memory.cache_limit, "{memory:?}");
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
        let map = crate::common::Map::new(services, crate::common::camera(&tokyo(15.0)));
        let mut offscreen = crate::common::Offscreen::with_map(map).unwrap();
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
        let executor = Arc::new(HeldExecutor::default());
        let wakes = Arc::new(AtomicUsize::new(0));
        let wake_count = wakes.clone();
        let services = crate::common::services_on(
            executor.clone(),
            Arc::new(move || {
                wake_count.fetch_add(1, Ordering::SeqCst);
            }),
        );
        let camera = crate::common::null_island();
        let mut offscreen =
            crate::common::Offscreen::with_map(crate::common::Map::new(services, Some(camera)))
                .unwrap();
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
            assert!(!gpu_driven() || !report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
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
            assert!(!gpu_driven() || !report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            // Only placements: a tile's rows are written again where it
            // or the tiles drawn around it changed (`TileRow::neighbours`).
            assert!(report.frame.placements_written <= report.tiles_drawn);
            assert!(report.tiles_drawn > 0);
        }
        // Switching 1×/4× rebuilds dependent targets and bundles once; the
        // next camera-only frame still replays them without paint writes.
        for enabled in [true, false] {
            offscreen.map.set_antialias_3d(enabled);
            let changed = offscreen.frame();
            assert!(changed.frame.bundle_recorded);
            let replay = offscreen.frame();
            assert!(!replay.frame.bundle_recorded);
            assert_eq!(replay.frame.paints_written, 0);
        }
        assert!(
            !crate::common::take_warnings(&mut offscreen.map)
                .iter()
                .any(|w| w.contains("fill-extrusion"))
        );
    }

    /// Antialiasing reaches the terrain, which draws in the map's own
    /// passes: the hills' silhouettes draw otherwise with it on (once,
    /// it multisampled the 3D layers alone). Switched either way on a
    /// settled map, over the hills alone and with buildings splitting
    /// the passes, the map draws what a fresh one with that setting does:
    /// nothing kept from the passes' other sample count shows, and labels
    /// multisampled read their own bitmaps alone, wherever the atlas
    /// packed them (once, a label's edge pixels read the next bitmap over,
    /// and fresh maps differed by the order their glyphs arrived in).
    #[test]
    fn antialiasing_on_terrain_draws_as_a_fresh_map_either_way() {
        for name in ["terrain", "terrain-buildings"] {
            let scene = crate::common::scenes::scenes(&[name]).remove(0);
            let (zoom, bearing, pitch) = scene.at;
            let options = crate::common::Options {
                style: Some(scene.url.clone()),
                encoding: "mvt".into(),
                at: Some((orb_crs::Geodetic::new(139.74, 35.685), zoom, bearing, pitch)),
            };
            let load = |antialias: bool| {
                let mut offscreen =
                    crate::common::Offscreen::new(&options, orb_runtime::SceneCapacity::default())
                        .unwrap();
                offscreen.map.set_antialias_3d(antialias);
                settle(&mut offscreen);
                offscreen
            };
            let fresh = [false, true].map(|antialias| load(antialias).pixels());
            assert_ne!(
                crate::common::differing_pixels(&fresh[0], &fresh[1]),
                0,
                "{name}: antialiasing changed nothing"
            );
            let mut switched = load(true);
            for antialias in [false, true] {
                switched.map.set_antialias_3d(antialias);
                settle(&mut switched);
                assert_eq!(
                    crate::common::differing_pixels(
                        &switched.pixels(),
                        &fresh[usize::from(antialias)]
                    ),
                    0,
                    "{name}: switched to antialias {antialias}"
                );
            }
        }
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

    /// A map at rest covers with a new tile level of detail at once, as
    /// one loaded with it does: the plan that frames keep while nothing
    /// changes reads it.
    #[test]
    fn a_map_at_rest_covers_with_a_new_tile_lod() {
        let lod = orb_camera::TileLod {
            zoom_shift: -1.0,
            ..orb_camera::TileLod::default()
        };
        let (mut moved, _, _) = crate::common::Offscreen::load(&tokyo(14.0)).unwrap();
        moved.map.set_tile_lod(lod);
        settle(&mut moved);
        let mut fresh =
            crate::common::Offscreen::new(&tokyo(14.0), orb_runtime::SceneCapacity::default())
                .unwrap();
        fresh.map.set_tile_lod(lod);
        settle(&mut fresh);
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
            let services =
                crate::common::services_on(Arc::new(orb_host::jobs::Inline), Arc::new(|| {}));
            let camera = crate::common::null_island();
            let mut offscreen =
                crate::common::Offscreen::with_map(crate::common::Map::new(services, Some(camera)))
                    .unwrap();
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
        let mut offscreen = null_island(
            r#"{"version":8,
            "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
            "layers":[{"id":"label","type":"symbol","source":"s","layout":{"text-field":"Null Island"}}]}"#,
        );
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

    /// Placement paces itself: whenever the lane is free and the camera
    /// changed, the frame asks for one, moving or not, and never while one
    /// is out.
    #[test]
    fn labels_place_whenever_the_lane_is_free() {
        let mut offscreen = null_island(
            r#"{"version":8,
            "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
            "layers":[{"id":"label","type":"symbol","source":"s","layout":{"text-field":"Null Island"}}]}"#,
        );
        let timeout = std::time::Duration::from_secs(10);
        let started = std::time::Instant::now();
        while !offscreen.frame().idle && started.elapsed() < timeout {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let mut camera = *offscreen.map.camera();
        for _ in 0..10 {
            camera.bearing += 1.0;
            offscreen.map.jump_to_state(camera);
            // The last placement committed: the first frame that moves asks
            // for the next at once.
            assert_eq!(offscreen.frame().placements, 1);
            // Until it commits, frames ask for none.
            let started = std::time::Instant::now();
            loop {
                let report = offscreen.frame();
                if report.loaded || started.elapsed() > timeout {
                    break;
                }
                assert_eq!(report.placements, 0, "a placement is out");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// Until the map is first idle, labels show at once rather than fade,
    /// as GL JS skips fades before its first `idle`: a map's first view is
    /// complete as soon as its data is. Here a raster tile that never
    /// arrives keeps the map from idling, and frames all show one instant, so a
    /// fade would never end. Two labels collide until the map turns a
    /// quarter, and then both show in full at once.
    #[test]
    fn labels_skip_fades_until_first_idle() {
        // Takes connections and never answers.
        let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let point = |name: &str, lon: f64| {
            format!(
                r#"{{"type":"Feature","properties":{{"name":"{name}"}},"geometry":{{"type":"Point","coordinates":[{lon},20]}}}}"#
            )
        };
        let style = format!(
            r#"{{"version":8,
            "sources":{{"r":{{"type":"raster","tiles":["http://{}/{{z}}/{{x}}/{{y}}.png"]}},
                "s":{{"type":"geojson","data":{{"type":"FeatureCollection","features":[{},{}]}}}}}},
            "layers":[{{"id":"r","type":"raster","source":"r"}},
                {{"id":"label","type":"symbol","source":"s","layout":{{"text-field":["get","name"]}}}}]}}"#,
            silent.local_addr().unwrap(),
            point("Null Island", 20.0),
            point("Gulf of Guinea", 27.0),
        );
        let mut offscreen = null_island(&style);
        offscreen.map.jump_to_state(orb_runtime::CameraState::new(
            orb_crs::Geodetic::new(23.5, 20.0),
            1.5,
            &offscreen.map.projection(),
        ));
        // Frames until `shown` labels show; their opacities.
        let frames_until = |offscreen: &mut crate::common::Offscreen, shown: usize| {
            let started = std::time::Instant::now();
            while started.elapsed() < std::time::Duration::from_secs(10) {
                let report = offscreen.frame();
                offscreen.time -= 1000.0 / 60.0;
                assert!(!report.idle, "the raster keeps the map from idling");
                let labels = offscreen.map.shown_labels();
                let opacities: Vec<f32> = labels.values().copied().filter(|&o| o > 0.0).collect();
                if opacities.len() == shown {
                    return opacities;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            panic!("{shown} labels didn't show");
        };
        assert_eq!(frames_until(&mut offscreen, 1), [1.0], "one has room");
        let mut turned = *offscreen.map.camera();
        turned.bearing = 90.0;
        offscreen.map.jump_to_state(turned);
        assert_eq!(
            frames_until(&mut offscreen, 2),
            [1.0, 1.0],
            "both have room"
        );
    }

    /// Between frames, `prepare` applies what arrives and sends what comes
    /// next: here the tile, then the placement it asks for, so the one
    /// frame after draws the label and is idle.
    #[test]
    fn prepare_applies_arrivals_between_frames() {
        let mut offscreen = null_island(
            r#"{"version":8,
            "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
            "layers":[{"id":"label","type":"symbol","source":"s","layout":{"text-field":"Null Island"}}]}"#,
        );
        assert!(!offscreen.frame().idle);
        // Until prepare has the label shown, not for a fixed time, which
        // a loaded machine can overrun.
        let started = std::time::Instant::now();
        let shown = |map: &orb_runtime::Map| map.shown_labels().get("Null Island").copied();
        while shown(&offscreen.map) != Some(1.0)
            && started.elapsed() < std::time::Duration::from_secs(30)
        {
            let (map, gpu) = (&mut offscreen.map, &mut offscreen.gpu);
            let renderer = offscreen.renderer.as_mut().expect("made by the frame");
            map.prepare(gpu, renderer, offscreen.viewport, 1.0);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let report = offscreen.frame();
        let shown = offscreen.map.shown_labels().get("Null Island").copied();
        assert_eq!((report.idle, shown), (true, Some(1.0)));
    }

    /// Before the first frame, `prepare` takes the style in and asks for
    /// what it shows, so the first frame draws it all, labels placed.
    #[test]
    fn prepare_before_the_first_frame_asks_for_the_view() {
        // The map's clock starts after this, so its readings trail ours.
        let origin = std::time::Instant::now();
        let mut offscreen = null_island(
            r#"{"version":8,
            "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
            "layers":[{"id":"label","type":"symbol","source":"s","layout":{"text-field":"Null Island"}}]}"#,
        );
        // Until prepare has the label shown, not for a fixed time, which
        // a loaded machine can overrun.
        let started = std::time::Instant::now();
        let shown = |map: &orb_runtime::Map| map.shown_labels().get("Null Island").copied();
        while shown(&offscreen.map) != Some(1.0)
            && started.elapsed() < std::time::Duration::from_secs(30)
        {
            let (map, gpu, format) = (&mut offscreen.map, &mut offscreen.gpu, offscreen.format);
            let renderer = offscreen
                .renderer
                .get_or_insert_with(|| map.create_renderer(gpu, format));
            map.prepare(gpu, renderer, offscreen.viewport, 1.0);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // The first frame comes after prepare's on prepare's clock, as a
        // host's does: from the offscreen's own time, which starts at 0,
        // it would come before them whenever setting up the GPU took a
        // frame's time (the frame then wasn't idle).
        offscreen.time = origin.elapsed().as_secs_f64() * 1000.0;
        let report = offscreen.frame();
        let shown = offscreen.map.shown_labels().get("Null Island").copied();
        assert_eq!((report.idle, shown), (true, Some(1.0)));
    }

    /// A canvas's (or video's) tile that `prepare` asks for before any
    /// frame waits for the host's frame, as a frame's request does: the
    /// source has no URL to fetch (#175).
    #[test]
    fn prepare_before_the_first_frame_waits_for_a_canvas_frame() {
        let mut offscreen = null_island(
            r##"{"version":8,
            "sources":{"c":{"type":"canvas","canvas":"drawing",
                "coordinates":[[-120,70],[120,70],[120,-70],[-120,-70]]}},
            "layers":[{"id":"c","type":"raster","source":"c","paint":{"raster-fade-duration":0}}]}"##,
        );
        for _ in 0..3 {
            let (map, gpu, format) = (&mut offscreen.map, &mut offscreen.gpu, offscreen.format);
            let renderer = offscreen
                .renderer
                .get_or_insert_with(|| map.create_renderer(gpu, format));
            map.prepare(gpu, renderer, offscreen.viewport, 1.0);
        }
        let rgba = [0, 255, 0, 255];
        let frame = orb_runtime::TileImage {
            width: 4,
            height: 4,
            pixels: orb_runtime::ImagePixels::Core(rgba.repeat(16)),
        };
        offscreen.map.source_frame("c", frame).unwrap();
        settle(&mut offscreen);
        let pixels = offscreen.pixels();
        let (w, h) = (
            offscreen.viewport.width as usize,
            offscreen.viewport.height as usize,
        );
        let center = (h / 2 * w + w / 2) * 4;
        assert_eq!(pixels[center..center + 4], rgba);
        // Nor does a reload fetch it: only the app's sources reload.
        assert!(offscreen.map.reload_host_tiles("c").is_err());
        settle(&mut offscreen);
    }

    /// A source with symbol layers loads its tiles while the sprites load:
    /// only their layout waits for the icons.
    #[test]
    fn symbol_tiles_load_while_the_sprites_do() {
        // Takes connections and never answers.
        let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut offscreen = null_island(&format!(
            r#"{{"version":8,"sprite":"http://{}/sprite",
            "sources":{{"s":{{"type":"geojson","data":{{"type":"Point","coordinates":[0,0]}}}}}},
            "layers":[{{"id":"label","type":"symbol","source":"s","layout":{{"text-field":"Null Island"}}}}]}}"#,
            silent.local_addr().unwrap()
        ));
        let loading = (0..20).any(|_| {
            let report = offscreen.frame();
            std::thread::sleep(std::time::Duration::from_millis(2));
            report.tiles_loading > 0
        });
        assert!(loading, "the source's tiles load");
        assert!(
            !offscreen.frame().loaded,
            "the sprites keep the map loading"
        );
    }

    /// Icon layout takes a sprite's images from its JSON, ahead of its
    /// pixels; a sprite whose image then fails takes them back, as one
    /// whose JSON failed never gave them.
    #[test]
    fn a_sprite_whose_image_fails_takes_its_icons_back() {
        let dir = crate::common::Scratch::new("sprite-image");
        std::fs::write(
            dir.join("s.json"),
            r#"{"dot":{"x":0,"y":0,"width":1,"height":1}}"#,
        )
        .unwrap();
        let mut offscreen = null_island(&format!(
            r#"{{"version":8,"sprite":"file://{}/s",
            "sources":{{"s":{{"type":"geojson","data":{{"type":"Point","coordinates":[0,0]}}}}}},
            "layers":[{{"id":"icon","type":"symbol","source":"s","layout":{{"icon-image":"dot"}}}}]}}"#,
            dir.display()
        ));
        let started = std::time::Instant::now();
        let mut errors = Vec::new();
        while started.elapsed() < std::time::Duration::from_secs(10) {
            let loaded = offscreen.frame().loaded;
            errors.extend(
                offscreen
                    .map
                    .take_events()
                    .into_iter()
                    .filter_map(|e| match e {
                        orb_runtime::MapEvent::Error { message } => Some(message),
                        _ => None,
                    }),
            );
            if loaded {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(errors.iter().any(|e| e.starts_with("sprite")), "{errors:?}");
        assert!(!offscreen.map.has_image("dot"));
    }

    /// Raster tiles show at once until the map is first idle, as in GL JS,
    /// whose first view has no tiles to fade from; after that, new tiles
    /// fade in. Frames all show one instant, so a fade would never end.
    #[test]
    fn raster_tiles_skip_fades_until_first_idle() {
        let dir = crate::common::Scratch::new("raster-fade");
        let tile = solid_tile(&dir);
        let mut offscreen = null_island(&format!(
            r#"{{"version":8,
            "sources":{{"imagery":{{"type":"raster","tiles":["file://{}"],"tileSize":256}}}},
            "layers":[{{"id":"raster","type":"raster","source":"imagery"}}]}}"#,
            tile.display()
        ));
        // Frames at one instant until idle, or until a few after the
        // tiles are in; whether it idled.
        let idles = |offscreen: &mut crate::common::Offscreen| {
            let started = std::time::Instant::now();
            let mut settled = 0;
            while started.elapsed() < std::time::Duration::from_secs(10) && settled < 10 {
                let report = offscreen.frame();
                offscreen.time -= 1000.0 / 60.0;
                if report.idle {
                    return true;
                }
                if report.tiles_loading == 0 && report.tiles_drawn > 0 {
                    settled += 1;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            false
        };
        let first = idles(&mut offscreen);
        offscreen.map.jump_to_state(orb_runtime::CameraState::new(
            orb_crs::Geodetic::new(150.0, 40.0),
            6.0,
            &offscreen.map.projection(),
        ));
        let later = idles(&mut offscreen);
        assert!(first, "the first view's tiles show at once");
        assert!(!later, "later tiles fade in");
    }

    /// A map of `style` at Null Island, with labels faded over GL JS's
    /// default 300 ms and placed on the placement thread with local fonts.
    fn null_island(style: &str) -> crate::common::Offscreen {
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
        let camera = crate::common::null_island();
        let mut offscreen =
            crate::common::Offscreen::with_map(crate::common::Map::new(services, Some(camera)))
                .unwrap();
        offscreen.map.set_style_json(style).unwrap();
        offscreen
    }

    #[test]
    fn styles_with_symbol_layers_render() {
        let services =
            crate::common::services_on(Arc::new(orb_host::jobs::Inline), Arc::new(|| {}));
        let camera = crate::common::null_island();
        let mut offscreen =
            crate::common::Offscreen::with_map(crate::common::Map::new(services, Some(camera)))
                .unwrap();
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

    /// A blended layer costs its composite only while it shows: a raster
    /// multiplied over the map draws its composite (and splits the main
    /// pass for it) at the zooms it shows at, recording its bundles once,
    /// and none once a zoom past its range or its visibility hides it.
    #[test]
    fn a_hidden_blended_layer_composites_nothing() {
        let dir = crate::common::Scratch::new("hidden-blend");
        let tile = solid_tile(&dir);
        let draws = |blend: &str| {
            let style = format!(
                r##"{{"version":8,"sources":{{
                    "imagery":{{"type":"raster","tiles":["file://{tile}"],"tileSize":256}}}},
                  "layers":[{{"id":"ground","type":"background","paint":{{"background-color":"#88aa66"}}}},
                    {{"id":"raster","type":"raster","source":"imagery","maxzoom":5,
                      "paint":{{"raster-layer-blend":"{blend}"}}}}]}}"##,
                tile = tile.display(),
            );
            let path = dir.join(format!("{blend}.json"));
            std::fs::write(&path, style).unwrap();
            let options = crate::common::Options {
                style: Some(format!("file://{}", path.display())),
                at: Some((orb_crs::Geodetic::new(0.0, 0.0), 3.0, 0.0, 0.0)),
                ..tokyo(3.0)
            };
            let (mut offscreen, _, _) = crate::common::Offscreen::load(&options).unwrap();
            let frame = |offscreen: &mut crate::common::Offscreen| {
                settle(offscreen);
                let report = offscreen.frame();
                assert!(!report.frame.bundle_recorded, "{blend}");
                report.frame.draws
            };
            let shown = frame(&mut offscreen);
            let projection = offscreen.map.projection();
            crate::common::move_camera(&mut offscreen.map, |c| c.set_zoom(&projection, 6.0));
            let past = frame(&mut offscreen);
            crate::common::move_camera(&mut offscreen.map, |c| c.set_zoom(&projection, 3.0));
            assert_eq!(frame(&mut offscreen), shown, "{blend}: shown again");
            let hide = r#"[["setLayoutProperty", "raster", "visibility", "none"]]"#;
            let hide = orb_runtime::Transaction::from_commands(hide).unwrap();
            offscreen.map.edit(&hide).unwrap();
            [shown, past, frame(&mut offscreen)]
        };
        let [shown, past, hidden] = draws("multiply");
        let normal = draws("normal");
        assert_eq!(shown, normal[0] + 1, "the blend's composite");
        // Hidden, no more than drawn normally (whose bundle, recorded while
        // the layer showed, keeps its draw for the cull to skip).
        assert!(
            past <= normal[1],
            "past its zooms: {past} draws, {} normally",
            normal[1]
        );
        assert!(
            hidden <= normal[2],
            "hidden: {hidden} draws, {} normally",
            normal[2]
        );
    }

    /// A 256-pixel tile of one colour, as `tile.png` in `dir`.
    fn solid_tile(dir: &std::path::Path) -> std::path::PathBuf {
        solid_png(dir, "tile.png", [64, 128, 255, 255])
    }

    /// A 256-pixel tile of colour `rgba`, as `name` in `dir`.
    fn solid_png(dir: &std::path::Path, name: &str, rgba: [u8; 4]) -> std::path::PathBuf {
        let tile = dir.join(name);
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 256, 256);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&rgba.repeat(256 * 256)).unwrap();
        }
        std::fs::write(&tile, png).unwrap();
        tile
    }

    /// Terrain, its color relief and its hillshade over one DEM source
    /// the app feeds, 256 heights a side to a 512-pixel tile, as the
    /// demo's Earthquake has it.
    const FED: &str = r##"{"version": 8,
        "sources": {"ground": {"type": "raster-dem", "encoding": "terrarium", "tileSize": 512,
            "maxzoom": 11, "tiles": ["orbis-host:"]}},
        "terrain": {"source": "ground", "exaggeration": 1.5},
        "layers": [
          {"id": "land", "type": "background", "paint": {"background-color": "#d8e0c8"}},
          {"id": "colour", "type": "color-relief", "source": "ground", "paint": {
            "color-relief-color": ["interpolate", ["linear"], ["elevation"],
              0, "#3a7d44", 1500, "#c8b88a", 3000, "#ffffff"],
            "color-relief-opacity": 0.6}},
          {"id": "shade", "type": "hillshade", "source": "ground"}
        ]}"##;

    const FED_SIDE: u32 = 256;

    /// The ground of step `step`, `swing` metres: hills over the Valais,
    /// and rings rippling out from under the camera.
    fn ground(step: u32, swing: f64) -> impl Fn(f64, f64) -> f32 {
        move |lon, lat| {
            let km = (lon - 7.75).hypot(lat - 46.02) * 111.0;
            let phase = km * 0.8 - f64::from(step);
            (1200.0 + 900.0 * (lon * 30.0).sin() * (lat * 40.0).cos() + swing * phase.sin()) as f32
        }
    }

    /// Tile `tile`'s heights of `ground`, at its texels' centres.
    fn heights(tile: orb_crs::TileIndex, ground: &dyn Fn(f64, f64) -> f32) -> Vec<f32> {
        let (n, side) = (f64::from(1u32 << tile.z), f64::from(FED_SIDE));
        (0..FED_SIDE * FED_SIDE)
            .map(|i| {
                let x = (f64::from(tile.col) + (f64::from(i % FED_SIDE) + 0.5) / side) / n;
                let y = (f64::from(tile.row) + (f64::from(i / FED_SIDE) + 0.5) / side) / n;
                let lat = (std::f64::consts::PI * (1.0 - 2.0 * y))
                    .sinh()
                    .atan()
                    .to_degrees();
                ground(x * 360.0 - 180.0, lat)
            })
            .collect()
    }

    /// A map of [`FED`] whose tile requests it answers with heights.
    struct Fed {
        offscreen: crate::common::Offscreen,
        /// The tiles it gave heights, less those the map dropped.
        given: std::collections::BTreeSet<orb_crs::TileIndex>,
    }

    impl Fed {
        fn new(pitch: f64, budget: Option<orb_runtime::Budget>) -> Self {
            let camera = orb_camera::CameraState {
                bearing: 20.0,
                pitch,
                ..orb_camera::CameraState::new(
                    orb_crs::Geodetic::new(7.75, 46.02),
                    11.0,
                    &crate::common::web_mercator(),
                )
            };
            Self::with(FED, camera, budget)
        }

        /// A map of `style`, whose `ground` source the app feeds, at
        /// `camera`.
        fn with(
            style: &str,
            camera: orb_camera::CameraState,
            budget: Option<orb_runtime::Budget>,
        ) -> Self {
            let services = crate::common::native_services(std::sync::Arc::new(|| {}));
            let mut map = orb_runtime::Map::new(services, Some(camera));
            map.set_clear_color([1.0; 4]);
            map.set_stable_labels(false);
            if let Some(budget) = budget {
                map.set_frame_budget(budget);
            }
            map.set_style_json(style).unwrap();
            Self {
                offscreen: crate::common::Offscreen::with_map(map).unwrap(),
                given: Default::default(),
            }
        }

        /// A frame; the tiles it asked for get `ground`'s heights. Its events.
        fn frame(
            &mut self,
            ground: &dyn Fn(f64, f64) -> f32,
        ) -> (orb_runtime::FrameReport, Vec<orb_runtime::MapEvent>) {
            let report = self.offscreen.frame();
            let events = self.offscreen.map.take_events();
            let asked: Vec<_> = (events.iter())
                .filter_map(|e| match e {
                    orb_runtime::MapEvent::TileRequest { tile, .. } => Some(*tile),
                    _ => None,
                })
                .collect();
            if !asked.is_empty() {
                let tiles = asked.iter().map(|&t| (t, heights(t, ground))).collect();
                let done = (self.offscreen.map)
                    .set_host_elevations("ground", FED_SIDE, tiles)
                    .unwrap();
                assert!(
                    done.dropped.is_empty(),
                    "the map dropped tiles it asked for"
                );
                self.given.extend(asked);
            }
            for event in &events {
                if let orb_runtime::MapEvent::Error { message } = event {
                    panic!("{message}");
                }
            }
            (report, events)
        }

        /// Frames until two in a row are idle; the last's report.
        fn settle(&mut self, ground: &dyn Fn(f64, f64) -> f32) -> orb_runtime::FrameReport {
            let started = std::time::Instant::now();
            let mut was_idle = false;
            loop {
                let (report, _) = self.frame(ground);
                if report.idle && was_idle {
                    return report;
                }
                was_idle = report.idle;
                assert!(started.elapsed().as_secs() < 120, "did not settle");
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }

        /// `ground`'s heights over every tile given before: one step.
        fn step(&mut self, ground: &dyn Fn(f64, f64) -> f32) -> u64 {
            let tiles = self
                .given
                .iter()
                .map(|&t| (t, heights(t, ground)))
                .collect();
            let done = (self.offscreen.map)
                .set_host_elevations("ground", FED_SIDE, tiles)
                .unwrap();
            for tile in done.dropped {
                self.given.remove(&tile);
            }
            done.batch
        }
    }

    /// Heights given over the tiles a map draws, terrain and the hillshade
    /// and color relief on it, draw what a fresh map given those heights
    /// draws: the same ground, shading, drape and cover, though the tiles
    /// kept their layers and arrived with other heights. The camera stands
    /// where a fresh one does.
    #[test]
    fn heights_given_over_drawn_tiles_draw_what_a_fresh_map_does() {
        let (before, after) = (ground(0, 600.0), ground(3, 600.0));
        let mut moved = Fed::new(60.0, None);
        moved.settle(&before);
        moved.step(&after);
        moved.settle(&after);
        let mut fresh = Fed::new(60.0, None);
        fresh.settle(&after);
        assert_eq!(moved.offscreen.map.camera(), fresh.offscreen.map.camera());
        let differing =
            crate::common::differing_pixels(&moved.offscreen.pixels(), &fresh.offscreen.pixels());
        assert_eq!(differing, 0, "pixels differ from a fresh map's");
    }

    /// A batch of heights larger than a frame takes goes in over frames
    /// and shows whole in one: every frame shows one step's heights on
    /// every tile, never the tiles of one beside those of the next. A
    /// batch given while one goes in replaces it, which never shows. No
    /// GPU memory moves meanwhile.
    #[test]
    fn no_frame_shows_two_batches_of_heights() {
        // A tile's heights a frame, and every stale drape slot: the drape
        // follows a batch within the frame's budget like any other change,
        // which is another test's.
        let budget = orb_runtime::Budget {
            ms: 1000.0,
            bytes: 1,
            drape_texels: u64::MAX,
            drape_slots: usize::MAX,
        };
        let steps = [ground(0, 25.0), ground(1, 25.0), ground(2, 25.0)];
        let mut fresh = Fed::new(0.0, None);
        fresh.settle(&steps[2]);
        let last = fresh.offscreen.pixels();
        let mut moved = Fed::new(0.0, Some(budget));
        // GPU memory: the CPU driver's copies of its tables are the core's,
        // and what the pools hold of it is what the frame draws.
        let gpu_memory = |report: orb_runtime::FrameReport| {
            let mut memory = report.memory;
            memory.scene.cpu_copies = 0;
            let s = &mut memory.scene;
            for pool in [
                &mut s.vertices,
                &mut s.indices,
                &mut s.compacted,
                &mut s.props,
            ] {
                pool.used = 0;
            }
            memory
        };
        let memory = gpu_memory(moved.settle(&steps[0]));
        let first = moved.offscreen.pixels();
        assert_ne!(first, last, "the steps draw alike: the test shows nothing");
        assert!(moved.given.len() >= 3, "{:?}", moved.given);
        let replaced = moved.step(&steps[1]);
        let (_, events) = moved.frame(&steps[1]);
        assert!(events.is_empty(), "{events:?}");
        assert_eq!(moved.offscreen.pixels(), first, "a part of a batch showed");
        let batch = moved.step(&steps[2]);
        let mut frames = 0;
        loop {
            let (report, events) = moved.frame(&steps[2]);
            assert_eq!(gpu_memory(report), memory, "memory moved in frame {frames}");
            frames += 1;
            let shown: Vec<u64> = (events.iter())
                .filter_map(|e| match e {
                    orb_runtime::MapEvent::ElevationsShown { batch, .. } => Some(*batch),
                    _ => None,
                })
                .collect();
            assert!(!shown.contains(&replaced), "the replaced batch showed");
            let pixels = moved.offscreen.pixels();
            if shown.contains(&batch) {
                let differing = crate::common::differing_pixels(&pixels, &last);
                assert_eq!(differing, 0, "the batch's frame isn't wholly its own");
                break;
            }
            assert_eq!(pixels, first, "frame {frames} showed a part of a batch");
            assert!(frames < 500, "the batch never showed");
        }
        assert!(frames >= 3, "the batch went in whole in {frames} frames");
        moved.settle(&steps[2]);
        let differing = crate::common::differing_pixels(&moved.offscreen.pixels(), &last);
        assert_eq!(differing, 0, "at rest, pixels differ from a fresh map's");
    }

    /// A symbol pitched with the map lies level on the terrain at the
    /// height under it, so it reads as it does on flat ground wherever it
    /// stands: on a crest over a drop that faces away from the eye, its
    /// corners raised each by the ground under them folded its top below
    /// its bottom, drawing it mirrored.
    #[test]
    fn pitched_symbols_on_a_drop_facing_away_are_not_mirrored() {
        // A crest 20 m north of the camera's target, past which the ground
        // falls 3 m a metre.
        let crest = 46.02 + 20.0 / 111_000.0;
        let ground = move |_lon: f64, lat: f64| {
            (1000.0 - 3.0 * ((lat - crest) * 111_000.0).max(0.0)).max(0.0) as f32
        };
        let style = r##"{"version": 8,
            "sources": {
              "ground": {"type": "raster-dem", "encoding": "terrarium", "tileSize": 512,
                "maxzoom": 11, "tiles": ["orbis-host:"]},
              "mark": {"type": "geojson", "data": {"type": "Point", "coordinates": [7.75, 46.02]}}},
            "terrain": {"source": "ground", "exaggeration": 1.5},
            "layers": [
              {"id": "land", "type": "background", "paint": {"background-color": "#d8e0c8"}},
              {"id": "mark", "type": "symbol", "source": "mark", "layout": {
                "icon-image": "mark", "icon-size": 2, "icon-allow-overlap": true,
                "icon-ignore-placement": true, "icon-pitch-alignment": "map",
                "icon-rotation-alignment": "map"}}
            ]}"##;
        let camera = orb_camera::CameraState {
            pitch: 70.0,
            ..orb_camera::CameraState::new(
                orb_crs::Geodetic::new(7.75, 46.02),
                14.0,
                &crate::common::web_mercator(),
            )
        };
        let mut map = Fed::with(style, camera, None);
        // Red over blue.
        let side = 32u32;
        let pixels = (0..side * side)
            .flat_map(|i| match i / side < side / 2 {
                true => [255, 0, 0, 255],
                false => [0, 0, 255, 255],
            })
            .collect();
        let image = orb_runtime::StyleImage::with_options(side, side, pixels, "").unwrap();
        map.offscreen.map.add_image("mark", image).unwrap();
        map.settle(&ground);
        let pixels = map.offscreen.pixels();
        let width = crate::common::Offscreen::WIDTH as usize;
        // Each colour's pixels: how many, and their mean row.
        let rows = |pick: fn(&[u8; 4]) -> bool| {
            let rows: Vec<usize> = (pixels.as_chunks::<4>().0.iter().enumerate())
                .filter(|(_, p)| pick(p))
                .map(|(i, _)| i / width)
                .collect();
            let mean = rows.iter().sum::<usize>() as f64 / rows.len().max(1) as f64;
            (rows.len(), mean)
        };
        let (reds, red) = rows(|p| p[0] > 200 && p[1] < 60 && p[2] < 60);
        let (blues, blue) = rows(|p| p[2] > 200 && p[0] < 60 && p[1] < 60);
        assert!(
            reds > 50 && blues > 50,
            "{reds} red and {blues} blue pixels"
        );
        assert!(
            red < blue,
            "red's mean row {red:.1} lies under blue's {blue:.1}"
        );
    }
}

//! Offline scenes of the Tokyo extract that between them draw every
//! family and pass: Liberty's layers flat and pitched, the globe, terrain
//! with its drape and a hillshade, buildings on it, a heatmap, raster
//! imagery, and 3D buildings, single-sampled and multisampled, on the
//! ground and on terrain.

use serde_json::{Value, json};

use super::{DEMO_DATA, Scratch};

/// A scene: its style, camera (zoom, bearing, pitch over Tokyo), an edit
/// a frame of the settled map makes, and whether its 3D layers
/// multisample (off by default; on in the lit buildings, so that path
/// keeps a scene).
pub(crate) struct Scene {
    pub name: &'static str,
    pub url: String,
    pub at: (f64, f64, f64),
    pub edit: Option<&'static str>,
    pub antialias: bool,
    /// Where its files are, while any of the scenes made with it lives.
    pub dir: std::sync::Arc<Scratch>,
}

/// The Tokyo extract's style with its URLs absolute, changed by `edit`,
/// written to `dir`: its URL.
pub(crate) fn style(dir: &std::path::Path, name: &str, edit: impl FnOnce(&mut Value)) -> String {
    let data = std::path::Path::new(DEMO_DATA).canonicalize().unwrap();
    let text = std::fs::read_to_string(data.join("tokyo-mvt.json")).unwrap();
    let mut style: Value = serde_json::from_str(&text).unwrap();
    let data = data.display();
    style["glyphs"] = json!(format!("file://{data}/glyphs/{{fontstack}}/{{range}}.pbf"));
    style["sprite"] = json!(format!("file://{data}/sprites/ofm"));
    style["sources"]["openmaptiles"]["url"] =
        json!(format!("pmtiles://file://{data}/tokyo-mvt.pmtiles"));
    edit(&mut style);
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, style.to_string()).unwrap();
    format!("file://{}", path.display())
}

/// A 256 px PNG of `texel(x, y)`, written to `dir`: its path.
pub(crate) fn png(
    dir: &std::path::Path,
    name: &str,
    texel: impl Fn(u32, u32) -> [u8; 4],
) -> String {
    let mut pixels = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256 {
        for x in 0..256 {
            pixels.extend(texel(x, y));
        }
    }
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 256, 256);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&pixels).unwrap();
    }
    let path = dir.join(format!("{name}.png"));
    std::fs::write(&path, bytes).unwrap();
    path.display().to_string()
}

/// Inserts `layer` before the first symbol layer.
pub(crate) fn add_layer(style: &mut Value, layer: Value) {
    let layers = style["layers"].as_array_mut().unwrap();
    let at = layers
        .iter()
        .position(|l| l["type"] == "symbol")
        .unwrap_or(layers.len());
    layers.insert(at, layer);
}

/// Liberty's 3D buildings at `opacity`.
fn extrusion(opacity: f64) -> Value {
    json!({"id": "building-3d", "type": "fill-extrusion",
        "source": "openmaptiles", "source-layer": "building", "minzoom": 14,
        "paint": {"fill-extrusion-color": "hsl(35,8%,85%)",
            "fill-extrusion-height": ["get", "render_height"],
            "fill-extrusion-base": ["get", "render_min_height"],
            "fill-extrusion-opacity": opacity}})
}

/// The scenes named `names`, their files in a scratch directory.
pub(crate) fn scenes(names: &[&str]) -> Vec<Scene> {
    let first = names.first().copied().unwrap_or("none");
    let dir = std::sync::Arc::new(Scratch::new(&format!("scenes-{first}")));
    let all = |name: &str| names.contains(&name);
    let mut out = Vec::new();
    let mut add = |name, url: &dyn Fn() -> String, at, edit| {
        if all(name) {
            out.push(Scene {
                name,
                url: url(),
                at,
                edit,
                antialias: matches!(name, "extrusions-lit" | "terrain-buildings-antialiased"),
                dir: dir.clone(),
            });
        }
    };
    // Liberty's vector layers and labels, flat and pitched.
    let liberty = || style(&dir, "liberty", |_| {});
    add("liberty-z12", &liberty, (12.0, 0.0, 0.0), None);
    add("liberty-z15-pitched", &liberty, (15.0, 30.0, 60.0), None);
    // The globe from afar, with its sky.
    let globe = || {
        style(&dir, "globe", |s| {
            s["projection"] = json!({"type": "globe"});
        })
    };
    add("globe", &globe, (3.0, 0.0, 20.0), None);
    // Terrain from a DEM, the vector layers draped on it, and a
    // hillshade; then a paint change, which the drape redraws.
    let terrain = || {
        // Terrarium: elevation = R·256 + G + B/256 − 32768, here hills
        // 200–800 m high.
        let dem = png(&dir, "dem", |x, y| {
            let h = 500.0 + 300.0 * (f64::from(x) / 40.0).sin() * (f64::from(y) / 40.0).cos();
            let v = (h + 32768.0).round() as u32;
            [(v >> 8) as u8, v as u8, 0, 255]
        });
        style(&dir, "terrain", |s| {
            s["sources"]["dem"] = json!({"type": "raster-dem", "tiles": [format!("file://{dem}")],
                "tileSize": 256, "encoding": "terrarium", "maxzoom": 12});
            s["terrain"] = json!({"source": "dem", "exaggeration": 1.5});
            add_layer(
                s,
                json!({"id": "hills", "type": "hillshade", "source": "dem"}),
            );
        })
    };
    let repaint = r##"[["setPaintProperty", "water", "fill-color", "#3070c0"]]"##;
    add("terrain", &terrain, (13.0, 20.0, 60.0), None);
    add("terrain-drape", &terrain, (13.0, 20.0, 60.0), Some(repaint));
    // Buildings on the terrain under roads draped over them, multiplied
    // into the map: the map's pass splits at the buildings, which hide
    // the roads behind them, and again where the roads read the map.
    let buildings = || {
        let terrain = terrain();
        let path = terrain.strip_prefix("file://").unwrap();
        let mut s: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        add_layer(&mut s, extrusion(0.8));
        add_layer(
            &mut s,
            json!({"id": "roads-over", "type": "line", "source": "openmaptiles",
            "source-layer": "transportation",
            "paint": {"line-color": "#c06040", "line-width": 4,
                "line-layer-blend": "multiply"}}),
        );
        let path = dir.join("terrain-buildings.json");
        std::fs::write(&path, s.to_string()).unwrap();
        format!("file://{}", path.display())
    };
    add("terrain-buildings", &buildings, (16.0, -20.0, 60.0), None);
    // Those multisampled: the map's passes then take four samples, kept
    // between the passes split at the buildings.
    add(
        "terrain-buildings-antialiased",
        &buildings,
        (16.0, -20.0, 60.0),
        None,
    );
    // A heatmap of the points of interest, and circles.
    let heatmap = || {
        style(&dir, "heatmap", |s| {
            add_layer(
                s,
                json!({"id": "heat", "type": "heatmap", "source": "openmaptiles",
                "source-layer": "poi", "paint": {"heatmap-radius": 20}}),
            );
        })
    };
    add("heatmap", &heatmap, (15.0, 0.0, 0.0), None);
    // Raster imagery under the vector layers.
    let raster = || {
        let tile = png(&dir, "tile", |x, y| [x as u8, y as u8, 128, 255]);
        style(&dir, "raster", |s| {
            s["sources"]["imagery"] =
                json!({"type": "raster", "tiles": [format!("file://{tile}")], "tileSize": 256});
            let layers = s["layers"].as_array_mut().unwrap();
            layers.insert(
                1,
                json!({"id": "imagery", "type": "raster", "source": "imagery",
                "paint": {"raster-opacity": 0.5}}),
            );
        })
    };
    add("raster", &raster, (14.0, 0.0, 0.0), None);
    // Liberty's 3D buildings; then multisampled, under a light of their
    // own.
    let extrusions = || style(&dir, "extrusions", |s| add_layer(s, extrusion(0.8)));
    add("extrusions", &extrusions, (16.0, -20.0, 60.0), None);
    let lit = || {
        style(&dir, "extrusions-lit", |s| {
            s["light"] = json!({"anchor": "viewport", "position": [1.15, 210, 30],
                "intensity": 0.5});
            add_layer(
                s,
                json!({"id": "building-3d", "type": "fill-extrusion",
                "source": "openmaptiles", "source-layer": "building", "minzoom": 14,
                "paint": {"fill-extrusion-color": "hsl(35,8%,85%)",
                    "fill-extrusion-height": ["get", "render_height"],
                    "fill-extrusion-base": ["get", "render_min_height"]}}),
            );
        })
    };
    add("extrusions-lit", &lit, (16.0, -20.0, 60.0), None);
    assert_eq!(out.len(), names.len(), "unknown scenes among {names:?}");
    out
}

/// The names of every scene.
pub(crate) const ALL: &[&str] = &[
    "liberty-z12",
    "liberty-z15-pitched",
    "globe",
    "terrain",
    "terrain-drape",
    "terrain-buildings",
    "terrain-buildings-antialiased",
    "heatmap",
    "raster",
    "extrusions",
    "extrusions-lit",
];

const SIZE: (u32, u32) = (512, 384);

/// `scene`'s map on `gpu`, settled, then one frame more after the scene's
/// edit if it has one (which a frame of a settled map may not draw
/// otherwise: the terrain's drape redraws on a paint change).
pub(crate) fn settle<G: orb_hal::Gpu>(gpu: &mut G, scene: &Scene) {
    use super::{Geodetic, Map, TargetFormat, Viewport};
    let url = &scene.url;
    let format = TargetFormat {
        color: orb_hal::TextureFormat::Rgba8Unorm,
        sample_count: 1,
    };
    let target = gpu.create_texture(&orb_hal::TextureDesc {
        label: "target",
        width: SIZE.0,
        height: SIZE.1,
        array_layers: 1,
        dimension: orb_hal::TextureViewDimension::D2,
        format: format.color,
        usage: orb_hal::TextureUsages::RENDER_ATTACHMENT | orb_hal::TextureUsages::COPY_SRC,
        sample_count: 1,
        mip_levels: 1,
    });
    let (zoom, bearing, pitch) = scene.at;
    let camera = super::camera(&super::Options {
        style: None,
        encoding: "mvt".into(),
        at: Some((Geodetic::new(139.74, 35.685), zoom, bearing, pitch)),
    });
    let mut map = Map::new(super::native_services(std::sync::Arc::new(|| {})), camera);
    map.set_clear_color([1.0; 4]);
    map.set_stable_labels(false);
    map.set_antialias_3d(scene.antialias);
    map.load_style_url(url);
    let viewport = Viewport {
        width: f64::from(SIZE.0),
        height: f64::from(SIZE.1),
    };
    let mut renderer = map.create_renderer(gpu, format);
    let mut time = 0.0;
    let mut frame = |map: &mut Map, gpu: &mut G| {
        time += 1000.0 / 60.0;
        map.render(gpu, &mut renderer, target, viewport, 1.0, time)
    };
    let started = std::time::Instant::now();
    let mut idle = 0;
    while idle < 3 {
        let report = frame(&mut map, gpu);
        for warning in super::take_warnings(&mut map) {
            eprintln!("{url}: {warning}");
        }
        idle = if report.idle { idle + 1 } else { 0 };
        assert!(
            started.elapsed().as_secs() < 60,
            "{url}: timed out settling"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    if let Some(edit) = scene.edit {
        let tx = orb_runtime::Transaction::from_commands(edit).unwrap();
        map.edit(&tx).unwrap();
    }
    frame(&mut map, gpu);
}

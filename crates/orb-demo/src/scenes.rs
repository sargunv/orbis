//! The map catalogue the web demo shows too (`web/demo/data/scenes.json`,
//! whose fields `web/demo/styles.js` documents), and what a scene does to
//! its style once it loads: the projection and sky it puts in, or for the
//! Globe, the restyle `protomaps3d` in `styles.js` makes.

use orb_api::{
    Bounds, CameraState, CameraTarget, Constraints, FlyOptions, Geodetic, TileMatrixSet,
};
use serde::Deserialize;
use serde_json::{Value, json};

/// The web demo's data directory, where the catalogue's local files are.
const DEMO_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data");

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Catalogue {
    /// Where a basemap opens without a camera of its own.
    pub(crate) basemap_camera: Camera,
    pub(crate) groups: Vec<Group>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Group {
    pub(crate) name: String,
    pub(crate) scenes: Vec<Scene>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Scene {
    pub(crate) key: String,
    pub(crate) name: String,
    pub(crate) detail: Option<String>,
    pub(crate) url: String,
    /// The group's name, filled in after reading.
    #[serde(default)]
    pub(crate) provider: String,
    /// Land, water and a line, for the picker's chip.
    pub(crate) swatch: [String; 3],
    #[serde(default)]
    pub(crate) dark: bool,
    #[serde(default)]
    pub(crate) basemap: bool,
    pub(crate) camera: Option<Camera>,
    pub(crate) constraints: Option<SceneConstraints>,
    pub(crate) projection: Option<Value>,
    pub(crate) sky: Option<Value>,
    pub(crate) restyle: Option<String>,
    /// The terrain the options' switch puts in, or `true` for the style's own.
    pub(crate) terrain: Option<Value>,
    /// Switches the style designs that start off.
    #[serde(default)]
    pub(crate) off: Vec<String>,
    /// Whether the options offer the light's anchor.
    #[serde(default)]
    pub(crate) light: bool,
    /// A looping flight, relative to the data directory.
    pub(crate) flight: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct Camera {
    pub(crate) center: [f64; 2],
    pub(crate) zoom: f64,
    #[serde(default)]
    pub(crate) bearing: f64,
    #[serde(default)]
    pub(crate) pitch: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SceneConstraints {
    pub(crate) min_zoom: Option<f64>,
    pub(crate) max_zoom: Option<f64>,
    pub(crate) min_pitch: Option<f64>,
    pub(crate) max_pitch: Option<f64>,
    pub(crate) bounds: Option<[f64; 4]>,
}

impl Catalogue {
    pub(crate) fn read() -> Self {
        let mut catalogue: Self =
            serde_json::from_str(include_str!("../../../web/demo/data/scenes.json"))
                .expect("web/demo/data/scenes.json");
        for group in &mut catalogue.groups {
            for scene in &mut group.scenes {
                scene.provider.clone_from(&group.name);
            }
        }
        catalogue
    }

    pub(crate) fn scenes(&self) -> impl Iterator<Item = &Scene> {
        self.groups.iter().flat_map(|g| &g.scenes)
    }

    /// The scene a command line names: a catalogue key, a style URL an
    /// entry draws as it is, or any other style URL as a scene of its own.
    pub(crate) fn scene_for(&self, name: &str) -> Scene {
        if let Some(scene) = self.scenes().find(|s| {
            s.key == name
                || (s.url == name
                    && s.projection.is_none()
                    && s.sky.is_none()
                    && s.restyle.is_none())
        }) {
            return scene.clone();
        }
        Scene {
            key: name.to_string(),
            name: name
                .split("://")
                .nth(1)
                .and_then(|rest| rest.split('/').next())
                .unwrap_or(name)
                .to_string(),
            detail: Some(name.to_string()),
            url: name.to_string(),
            provider: "Custom style".into(),
            swatch: ["#e8e8e8", "#c8c8c8", "#9a9a9a"].map(Into::into),
            dark: false,
            basemap: true,
            camera: None,
            constraints: None,
            projection: None,
            sky: None,
            restyle: None,
            terrain: None,
            off: Vec::new(),
            light: false,
            flight: None,
        }
    }

    /// Where `scene` opens without a camera given: its own, central Tokyo
    /// for a basemap, else the style's (`None`).
    pub(crate) fn opening_camera(&self, scene: &Scene) -> Option<Camera> {
        scene
            .camera
            .or_else(|| scene.basemap.then_some(self.basemap_camera))
    }
}

impl Scene {
    /// The style's URL: local entries are files in the data directory.
    pub(crate) fn style_url(&self) -> String {
        if self.url.contains("://") {
            self.url.clone()
        } else {
            local(&self.url)
        }
    }

    pub(crate) fn constraints(&self) -> Constraints {
        let defaults = Constraints::default();
        let Some(c) = self.constraints else {
            return defaults;
        };
        Constraints {
            min_zoom: c.min_zoom.unwrap_or(defaults.min_zoom),
            max_zoom: c.max_zoom.unwrap_or(defaults.max_zoom),
            min_pitch: c.min_pitch.unwrap_or(defaults.min_pitch),
            max_pitch: c.max_pitch.unwrap_or(defaults.max_pitch),
            bounds: c.bounds.map(|[west, south, east, north]| Bounds {
                west,
                south,
                east,
                north,
            }),
        }
    }

    /// The projection the scene draws on, `mercator` when it sets none.
    pub(crate) fn projection_or_mercator(&self) -> Value {
        self.projection
            .clone()
            .unwrap_or_else(|| json!({ "type": "mercator" }))
    }

    /// `style` as the scene designs it: restyled, with its projection and
    /// sky, and with its terrain when `terrain` asks for it.
    pub(crate) fn designed(&self, style: &Value, terrain: bool) -> Value {
        let mut style = match self.restyle.as_deref() {
            Some("protomaps-3d") => protomaps3d(style),
            _ => style.clone(),
        };
        if let Some(projection) = &self.projection {
            style["projection"] = projection.clone();
        }
        if let Some(sky) = &self.sky {
            style["sky"] = sky.clone();
        }
        if terrain && let Some(terrain @ Value::Object(_)) = &self.terrain {
            style["terrain"] = terrain.clone();
        }
        style
    }

    pub(crate) fn read_flight(&self) -> Option<Result<Flight, String>> {
        self.flight.as_deref().map(Flight::read)
    }
}

impl Camera {
    pub(crate) fn state(&self) -> CameraState {
        let mercator = *TileMatrixSet::by_id("WebMercatorQuad")
            .expect("built in")
            .projection();
        CameraState {
            bearing: self.bearing,
            pitch: self.pitch,
            ..CameraState::new(
                Geodetic::new(self.center[0], self.center[1]),
                self.zoom,
                &mercator,
            )
        }
    }
}

fn local(path: &str) -> String {
    let path = std::path::Path::new(DEMO_DATA).join(path);
    let path = path.canonicalize().unwrap_or(path);
    format!("file://{}", path.display())
}

/// A loop of camera flights: legs of `flyTo` options (center, zoom,
/// bearing, pitch and duration in milliseconds), as the web demo flies them.
pub(crate) struct Flight {
    pub(crate) legs: Vec<(CameraTarget, FlyOptions)>,
    pub(crate) next: usize,
}

impl Flight {
    fn read(name: &str) -> Result<Self, String> {
        let path = std::path::Path::new(DEMO_DATA).join(name);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{name}: {e}"))?;
        let json: Value = serde_json::from_str(&text).map_err(|e| format!("{name}: {e}"))?;
        let legs = json["legs"]
            .as_array()
            .ok_or_else(|| format!("{name}: no legs"))?
            .iter()
            .map(|leg| {
                let number = |key: &str| leg[key].as_f64();
                let center = leg["center"]
                    .as_array()
                    .and_then(|c| Some(Geodetic::new(c.first()?.as_f64()?, c.get(1)?.as_f64()?)));
                let target = CameraTarget {
                    center,
                    zoom: number("zoom"),
                    bearing: number("bearing"),
                    pitch: number("pitch"),
                    ..Default::default()
                };
                let fly = FlyOptions {
                    duration: number("duration"),
                    ..Default::default()
                };
                (target, fly)
            })
            .collect::<Vec<_>>();
        if legs.is_empty() {
            return Err(format!("{name}: no legs"));
        }
        Ok(Self { legs, next: 0 })
    }
}

/// Mapterhorn's global elevation: Terrarium-encoded WebP (data/README.md).
fn mapterhorn() -> Value {
    json!({
        "type": "raster-dem",
        "tiles": ["https://tiles.mapterhorn.com/{z}/{x}/{y}.webp"],
        "encoding": "terrarium",
        "tileSize": 512,
        "maxzoom": 17,
    })
}

/// Protomaps Light with what it lacks for a 3D map, as `protomaps3d` in
/// `web/demo/styles.js` adds it (which says why each part is as it is):
/// Mapterhorn's DEM for the scene's terrain, a hillshade over the land and
/// water, the buildings extruded under the labels with rounded footprints,
/// shadows and ambient occlusion, a viewport-anchored light, and colours
/// mixed in linear light. Each addition is placed by the layers it finds.
fn protomaps3d(style: &Value) -> Value {
    let mut style = style.clone();
    let mut layers = style["layers"].as_array().cloned().unwrap_or_default();
    let labels = |layers: &[Value]| {
        layers
            .iter()
            .position(|l| l["type"] == "symbol")
            .unwrap_or(layers.len())
    };
    let water = layers.iter().position(|l| l["id"] == "water");
    let at = water.map_or_else(|| labels(&layers), |i| i + 1);
    layers.insert(
        at,
        json!({
            "id": "hillshade",
            "type": "hillshade",
            "source": "hillshade",
            "paint": {
                "hillshade-illumination-anchor": "viewport",
                "hillshade-exaggeration": ["interpolate", ["linear"], ["zoom"], 3, 0.6, 12, 0.35],
                "hillshade-shadow-color": "#6f6a63",
                "hillshade-highlight-color": "#fbfaf7",
                "hillshade-accent-color": "#8d877e",
            },
        }),
    );
    if let Some(at) = layers
        .iter()
        .position(|l| l["source-layer"] == "buildings" && l["type"] == "fill")
    {
        let source = layers[at]["source"].clone();
        layers[at]["layout"]["fill-rounded-corner-distance"] = json!(2);
        let height = json!(["coalesce", ["get", "height"], 5]);
        let at = labels(&layers);
        layers.insert(
            at,
            json!({
                "id": "buildings_3d",
                "type": "fill-extrusion",
                "source": source,
                "source-layer": "buildings",
                "minzoom": 14,
                "filter": ["==", ["geometry-type"], "Polygon"],
                "layout": { "fill-extrusion-rounded-corner-distance": 2 },
                "paint": {
                    "fill-extrusion-color": "#d7d6d3",
                    "fill-extrusion-height": ["interpolate", ["linear"], ["zoom"], 14, 0, 15, height],
                    "fill-extrusion-base": [
                        "interpolate", ["linear"], ["zoom"], 14, 0, 15,
                        ["min", ["coalesce", ["get", "min_height"], 0], height],
                    ],
                    "fill-extrusion-opacity": 0.9,
                    "fill-extrusion-ambient-occlusion-wall-intensity": 0.6,
                    "fill-extrusion-ambient-occlusion-ground-intensity": 0.5,
                },
            }),
        );
    }
    let mut terrain = mapterhorn();
    terrain["attribution"] =
        json!(r#"<a href="https://mapterhorn.com/attribution">© Mapterhorn</a>"#);
    style["sources"]["terrain"] = terrain;
    style["sources"]["hillshade"] = mapterhorn();
    style["light"] = json!({
        "anchor": "viewport",
        "position": [1.15, 210, 30],
        "intensity": 0.35,
        "shadow-intensity": 0.5,
    });
    style["metadata"]["dev.sargunv.orbis:color-space"] = json!("linear");
    style["layers"] = Value::Array(layers);
    style
}

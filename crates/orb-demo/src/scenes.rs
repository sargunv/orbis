//! The map catalogue the web demo shows too (`web/demo/data/scenes.json`,
//! whose fields `web/demo/styles.js` documents), and what a scene does to
//! its style once it loads: the projection and sky it puts in, or for the
//! Globe, the restyle of `web/demo/data/globe.json` that `globeStyle` in
//! `styles.js` makes, and for the Earthquake, `crate::quake`'s.

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
    /// Whether the scene exists to show an Orbis extension (a projection
    /// or CRS beyond the spec), which the picker tags.
    #[serde(default)]
    pub(crate) orbis_only: bool,
    #[serde(default)]
    pub(crate) basemap: bool,
    pub(crate) camera: Option<Camera>,
    pub(crate) constraints: Option<SceneConstraints>,
    pub(crate) projection: Option<Value>,
    pub(crate) sky: Option<Value>,
    pub(crate) restyle: Option<String>,
    /// Whether the options offer a switch for the style's own terrain,
    /// on by default.
    #[serde(default)]
    pub(crate) terrain: bool,
    /// Switches the style designs that start off.
    #[serde(default)]
    pub(crate) off: Vec<String>,
    /// Whether the options offer the light's anchor.
    #[serde(default)]
    pub(crate) light: bool,
    /// A looping flight, relative to the data directory.
    pub(crate) flight: Option<String>,
    /// The Earthquake's shock: its host-fed ground ripples from here
    /// (`crate::quake`).
    pub(crate) quake: Option<QuakeScene>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct QuakeScene {
    pub(crate) epicentre: [f64; 2],
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
            orbis_only: false,
            basemap: true,
            camera: None,
            constraints: None,
            projection: None,
            sky: None,
            restyle: None,
            terrain: false,
            off: Vec::new(),
            light: false,
            flight: None,
            quake: None,
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
    /// sky.
    pub(crate) fn designed(&self, style: &Value) -> Value {
        let mut style = match self.restyle.as_deref() {
            Some("globe") => globe(style),
            Some("quake") => crate::quake::style(style),
            _ => style.clone(),
        };
        if let Some(projection) = &self.projection {
            style["projection"] = projection.clone();
        }
        if let Some(sky) = &self.sky {
            style["sky"] = sky.clone();
        }
        style
    }

    pub(crate) fn read_flight(&self) -> Option<Result<Flight, String>> {
        self.flight.as_deref().map(Flight::read)
    }
}

impl Camera {
    pub(crate) fn state(&self) -> CameraState {
        let mercator = *TileMatrixSet::web_mercator().projection();
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

/// Protomaps Light as the Globe draws it, from `web/demo/data/globe.json`
/// as `globeStyle` in `web/demo/styles.js` applies it (which says why each
/// part is as it is): the palette over each layer's paint by id, the sea's
/// depth and relief under the land to zoom 6, and from there the ocean
/// polygons over the landuse with the depth over them, the land below sea
/// level and the Caspian put back over that, a hillshade over the land and
/// lakes, the buildings extruded under the labels, the DEM sources, the
/// terrain and the light, and the hillshade multiplied into the land (a
/// preview) for its option to switch on, off by default. Each addition is
/// placed by the layers it finds.
fn globe(style: &Value) -> Value {
    let globe: Value = serde_json::from_str(include_str!("../../../web/demo/data/globe.json"))
        .expect("web/demo/data/globe.json");
    let mut style = style.clone();
    let mut layers = style["layers"].as_array().cloned().unwrap_or_default();
    for layer in &mut layers {
        if let Some(Value::Object(paint)) = layer["id"].as_str().map(|id| &globe["paint"][id]) {
            for (name, value) in paint {
                layer["paint"][name] = value.clone();
            }
        }
        // In the filter's own syntax: Protomaps' is legacy.
        if layer["id"] == "water" {
            let legacy = layer["filter"].is_null() || is_legacy_filter(&layer["filter"]);
            let kind = if legacy {
                json!("kind")
            } else {
                json!(["get", "kind"])
            };
            let ocean = json!(["!=", kind, "ocean"]);
            layer["filter"] = match layer["filter"].take() {
                Value::Null => ocean,
                filter => json!(["all", filter, ocean]),
            };
        }
    }
    let labels = |layers: &[Value]| {
        layers
            .iter()
            .position(|l| l["type"] == "symbol")
            .unwrap_or(layers.len())
    };
    let earth = layers.iter().position(|l| l["id"] == "earth").unwrap_or(1);
    let sea = globe["sea"].as_array().cloned().unwrap_or_default();
    layers.splice(earth..earth, sea);
    // The ocean polygons, from z6 over the landuse, and the sources the
    // Globe derives from the water layer's: those with no `type` of their own.
    let water = layers.iter().position(|l| l["id"] == "water");
    let vector = water.map(|i| layers[i]["source"].clone());
    if let Value::Object(sources) = &globe["sources"] {
        for (name, source) in sources {
            if !source["type"].is_null() {
                style["sources"][name] = source.clone();
            } else if let Some(Value::Object(base)) = vector
                .as_ref()
                .and_then(Value::as_str)
                .map(|v| &style["sources"][v])
            {
                let mut derived = base.clone();
                if let Value::Object(own) = source {
                    derived.extend(own.clone());
                }
                style["sources"][name] = Value::Object(derived);
            }
        }
    }
    if let (Some(at), Some(vector)) = (water, &vector) {
        let ocean = globe["ocean"].as_array().into_iter().flatten().map(|l| {
            let mut l = l.clone();
            if l["source"].is_null() {
                l["source"] = vector.clone();
            }
            l
        });
        layers.splice(at..at, ocean);
    }
    let lakes = layers.iter().position(|l| l["id"] == "water");
    let at = lakes.map_or_else(|| labels(&layers), |i| i + 1);
    layers.insert(at, globe["hillshade"].clone());
    if let Some(at) = layers
        .iter()
        .position(|l| l["source-layer"] == "buildings" && l["type"] == "fill")
    {
        let mut buildings = globe["buildings"].clone();
        buildings["source"] = layers[at]["source"].clone();
        let at = labels(&layers);
        layers.insert(at, buildings);
    }
    style["terrain"] = globe["terrain"].clone();
    style["light"] = globe["light"].clone();
    style["layers"] = Value::Array(layers);
    style
}

/// Whether a filter is in v8's legacy syntax rather than an expression.
fn is_legacy_filter(filter: &Value) -> bool {
    let Some([op, args @ ..]) = filter.as_array().map(Vec::as_slice) else {
        return false;
    };
    match op.as_str() {
        Some("all" | "any" | "none") => args.iter().all(is_legacy_filter),
        Some("==" | "!=" | "<" | ">" | "<=" | ">=" | "in" | "!in" | "has" | "!has") => {
            args.first().is_some_and(Value::is_string)
        }
        _ => false,
    }
}

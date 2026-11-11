//! The map options panel's switches, as `web/demo/app/options.js` has
//! them (which says why each is as it is): what the scene's style draws
//! (terrain, the light's anchor, buildings and their shadows, the globe
//! and its sky, labels, the colour space), and only those the style can
//! take. Each choice edits the style JSON; the map takes it as one diffed
//! `setStyle`, so nothing reloads. An Orbis-only option starts off,
//! whatever the scene designs, its keys out of the style, so that by
//! default every scene draws what GL JS and MapLibre Native could.

use serde_json::{Value, json};

use crate::scenes::Scene;

const COLOR_SPACE: &str = "dev.sargunv.orbis:color-space";
const GROUND_LIGHTING: &str = "dev.sargunv.orbis:ground-lighting";
/// The light's keys that go with the night side: the ground lighting, the
/// dusk it passes through, and the ambient light the night side keeps.
const NIGHT: [&str; 4] = [
    GROUND_LIGHTING,
    "dev.sargunv.orbis:twilight",
    "dev.sargunv.orbis:ambient-color",
    "dev.sargunv.orbis:ambient-intensity",
];
const DAYLIGHT: &str = "dev.sargunv.orbis:daylight";
const SHADOWS: &str = "dev.sargunv.orbis:shadow-intensity";
const ROUNDED: &str = "dev.sargunv.orbis:fill-rounded-corner-distance";
/// The stars, and the backdrop they shine on.
const STARRY: [&str; 2] = [
    "dev.sargunv.orbis:star-opacity",
    "dev.sargunv.orbis:backdrop-color",
];
const HILLSHADE_ANCHOR: &str = "hillshade-illumination-anchor";
const AMBIENT_OCCLUSION: [&str; 2] = [
    "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity",
    "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-intensity",
];

/// The style as the scene designs it, and the terrain its switch puts in:
/// what each option's support, default and "on" read from.
pub(crate) struct Context {
    pub(crate) scene: Scene,
    pub(crate) designed: Value,
    pub(crate) terrain: Option<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Opt {
    Terrain,
    Hillshade,
    Light,
    Night,
    Buildings,
    Shadows,
    Ao,
    Rounded,
    Projection,
    Atmosphere,
    Fog,
    Stars,
    Labels,
    Colors,
}

/// A switch, or one of a few choices: `(value, label)`.
pub(crate) enum Kind {
    Switch,
    Choices(&'static [(&'static str, &'static str)]),
}

/// The panel's order.
pub(crate) const ALL: [Opt; 14] = [
    Opt::Terrain,
    Opt::Hillshade,
    Opt::Light,
    Opt::Night,
    Opt::Buildings,
    Opt::Shadows,
    Opt::Ao,
    Opt::Rounded,
    Opt::Projection,
    Opt::Atmosphere,
    Opt::Fog,
    Opt::Stars,
    Opt::Labels,
    Opt::Colors,
];

fn layers<'a>(style: &'a Value, kind: &'a str) -> impl Iterator<Item = &'a Value> {
    style["layers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(move |l| l["type"] == kind)
}

fn visible(layer: &Value) -> bool {
    layer["layout"]["visibility"] != "none"
}

/// Whether a number as written (a number or an expression) is on: not
/// zero, `fallback` when unset.
fn nonzero(value: &Value, fallback: f64) -> bool {
    match value {
        Value::Null => fallback != 0.0,
        Value::Number(n) => n.as_f64() != Some(0.0),
        _ => true,
    }
}

fn number_or(value: &Value, fallback: f64) -> Option<f64> {
    match value {
        Value::Null => Some(fallback),
        Value::Number(n) => n.as_f64(),
        // An expression: on, whatever it gives.
        _ => None,
    }
}

fn projection_type(style: &Value) -> String {
    match &style["projection"]["type"] {
        Value::Null => "mercator".into(),
        Value::String(s) => s.clone(),
        _ => "globe".into(),
    }
}

fn occluded(style: &Value) -> Vec<&Value> {
    layers(style, "fill-extrusion")
        .filter(|l| {
            AMBIENT_OCCLUSION
                .iter()
                .any(|p| nonzero(&l["paint"][*p], 0.0))
        })
        .collect()
}

fn rounded(style: &Value) -> Vec<&Value> {
    layers(style, "fill")
        .filter(|l| nonzero(&l["layout"][ROUNDED], 0.0))
        .collect()
}

/// `value` as the day side draws it: each `step` or `interpolate` over
/// `["dev.sargunv.orbis:daylight"]` replaced by its last output, its value
/// at full daylight.
fn by_day(value: &Value) -> Value {
    let Some(array) = value.as_array() else {
        return value.clone();
    };
    let op = array.first().and_then(Value::as_str);
    let input = if op == Some("step") {
        array.get(1)
    } else {
        array.get(2)
    };
    let daylight = input
        .and_then(Value::as_array)
        .and_then(|i| i.first())
        .and_then(Value::as_str)
        == Some(DAYLIGHT);
    if daylight
        && (op == Some("step") || op.is_some_and(|o| o.starts_with("interpolate")))
        && let Some(last) = array.last()
    {
        return by_day(last);
    }
    Value::Array(array.iter().map(by_day).collect())
}

/// Whether a property's value reads the daylight.
fn reads_daylight(value: &Value) -> bool {
    value.as_array().is_some_and(|a| {
        a.first().and_then(Value::as_str) == Some(DAYLIGHT) || a.iter().any(reads_daylight)
    })
}

/// Sets `keys` in `parent[field]` as `from` has them when `on`, else
/// removes them, leaving an absent `field` absent.
fn with_keys(parent: &mut Value, field: &str, keys: &[&str], on: bool, from: &Value) {
    for &key in keys {
        if on && !from[key].is_null() {
            parent[field][key] = from[key].clone();
        } else if let Some(object) = parent.get_mut(field).and_then(Value::as_object_mut) {
            object.remove(key);
        }
    }
}

/// A number in the style's `sky`: `off` for off, the designed value (or
/// unset) for on.
struct SkyNumber {
    name: &'static str,
    fallback: f64,
    off: f64,
    /// Fog is on below 1, not at any value but its off.
    below: bool,
}

impl SkyNumber {
    fn is_on(&self, value: &Value) -> bool {
        match number_or(value, self.fallback) {
            Some(v) if self.below => v < self.off,
            Some(v) => v != self.off,
            None => true,
        }
    }
}

impl Opt {
    pub(crate) fn group(self) -> &'static str {
        match self {
            Opt::Terrain | Opt::Hillshade | Opt::Light | Opt::Night => "Terrain and light",
            Opt::Buildings | Opt::Shadows | Opt::Ao | Opt::Rounded => "Buildings",
            Opt::Projection | Opt::Atmosphere | Opt::Fog | Opt::Stars => "Globe and sky",
            Opt::Labels | Opt::Colors => "Map",
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Opt::Terrain => "Terrain",
            Opt::Hillshade => "Hillshade",
            Opt::Light => "Light",
            Opt::Night => "Night side",
            Opt::Buildings => "3D buildings",
            Opt::Shadows => "Shadows",
            Opt::Ao => "Ambient occlusion",
            Opt::Rounded => "Rounded footprints",
            Opt::Projection => "Projection",
            Opt::Atmosphere => "Atmosphere",
            Opt::Fog => "Fog",
            Opt::Stars => "Stars",
            Opt::Labels => "Labels",
            Opt::Colors => "Colours",
        }
    }

    pub(crate) fn note(self) -> Option<&'static str> {
        match self {
            Opt::Light => Some("Fixed to the ground, or turning with the view"),
            Opt::Night => Some("The ground darkened where the sun is down, labels lit against it"),
            Opt::Rounded => Some("Flat footprints rounded as the 3D buildings are"),
            Opt::Stars => Some("On a night-sky backdrop"),
            Opt::Colors => Some("Mixed in linear light, or in sRGB as GL JS mixes them"),
            _ => None,
        }
    }

    /// A preview of a style-spec proposal that GL JS and MapLibre Native
    /// don't draw: off by default, and off takes its keys out of the style.
    pub(crate) fn orbis_only(self) -> bool {
        matches!(
            self,
            Opt::Night | Opt::Shadows | Opt::Ao | Opt::Rounded | Opt::Stars | Opt::Colors
        )
    }

    pub(crate) fn kind(self) -> Kind {
        match self {
            Opt::Light => Kind::Choices(&[("map", "Map"), ("viewport", "Viewport")]),
            Opt::Projection => Kind::Choices(&[("globe", "Globe"), ("mercator", "Mercator")]),
            Opt::Colors => Kind::Choices(&[("linear", "Linear"), ("srgb", "sRGB")]),
            _ => Kind::Switch,
        }
    }

    /// The option, and its value, without which this one draws nothing,
    /// and how the panel says so.
    pub(crate) fn needs(self) -> Option<(Opt, Value, &'static str)> {
        match self {
            Opt::Shadows | Opt::Ao => Some((Opt::Buildings, json!(true), "With 3D buildings on")),
            Opt::Atmosphere | Opt::Stars => Some((Opt::Projection, json!("globe"), "On the globe")),
            Opt::Fog => Some((Opt::Terrain, json!(true), "With terrain on")),
            _ => None,
        }
    }

    fn layer_type(self) -> Option<&'static str> {
        match self {
            Opt::Hillshade => Some("hillshade"),
            Opt::Buildings => Some("fill-extrusion"),
            Opt::Labels => Some("symbol"),
            _ => None,
        }
    }

    fn sky(self) -> Option<SkyNumber> {
        let (name, fallback, off, below) = match self {
            // The spec's defaults: atmosphere 0.8, fog over the far half.
            Opt::Atmosphere => ("atmosphere-blend", 0.8, 0.0, false),
            // Fog draws only over terrain; a blend of 1 puts it past the horizon.
            Opt::Fog => ("fog-ground-blend", 0.5, 1.0, true),
            _ => return None,
        };
        Some(SkyNumber {
            name,
            fallback,
            off,
            below,
        })
    }

    fn projection_by_design(cx: &Context) -> String {
        cx.scene
            .projection
            .as_ref()
            .and_then(|p| p["type"].as_str())
            .map_or_else(|| projection_type(&cx.designed), Into::into)
    }

    /// Whether the scene's style can take this option.
    pub(crate) fn supported(self, cx: &Context) -> bool {
        let designed = &cx.designed;
        if let Some(kind) = self.layer_type() {
            return layers(designed, kind).any(visible);
        }
        if let Some(sky) = self.sky() {
            return designed["sky"].is_object() && sky.is_on(&designed["sky"][sky.name]);
        }
        match self {
            Opt::Terrain => cx.scene.terrain.is_some() && cx.terrain.is_some(),
            Opt::Light => cx.scene.light,
            Opt::Night => nonzero(&designed["light"][GROUND_LIGHTING], 0.0),
            Opt::Shadows => {
                nonzero(&designed["light"][SHADOWS], 0.0)
                    && layers(designed, "fill-extrusion").next().is_some()
            }
            Opt::Ao => !occluded(designed).is_empty(),
            Opt::Rounded => !rounded(designed).is_empty(),
            Opt::Stars => nonzero(&designed["sky"][STARRY[0]], 0.0),
            Opt::Projection => {
                matches!(
                    Self::projection_by_design(cx).as_str(),
                    "globe" | "mercator"
                )
            }
            Opt::Colors => matches!(
                designed["metadata"][COLOR_SPACE].as_str(),
                Some("linear" | "srgb")
            ),
            _ => unreachable!("layer and sky options return above"),
        }
    }

    /// Its value in `style`.
    pub(crate) fn read(self, style: &Value) -> Value {
        if let Some(kind) = self.layer_type() {
            return json!(layers(style, kind).any(visible));
        }
        if let Some(sky) = self.sky() {
            return json!(style["sky"].is_object() && sky.is_on(&style["sky"][sky.name]));
        }
        match self {
            Opt::Terrain => json!(!style["terrain"].is_null()),
            Opt::Light => {
                let hillshade = layers(style, "hillshade")
                    .next()
                    .map(|l| l["paint"][HILLSHADE_ANCHOR].clone());
                match hillshade {
                    Some(Value::String(anchor)) => json!(anchor),
                    _ => json!(style["light"]["anchor"].as_str().unwrap_or("viewport")),
                }
            }
            Opt::Night => json!(nonzero(&style["light"][GROUND_LIGHTING], 0.0)),
            Opt::Shadows => json!(nonzero(&style["light"][SHADOWS], 0.0)),
            Opt::Ao => json!(!occluded(style).is_empty()),
            Opt::Rounded => json!(!rounded(style).is_empty()),
            Opt::Stars => json!(nonzero(&style["sky"][STARRY[0]], 0.0)),
            Opt::Projection => json!(projection_type(style)),
            Opt::Colors => json!(if style["metadata"][COLOR_SPACE] == "linear" {
                "linear"
            } else {
                "srgb"
            }),
            _ => unreachable!("layer and sky options return above"),
        }
    }

    /// The scene's default: off where it says so or for an Orbis-only
    /// option (sRGB for the colours), else the option's own, else as
    /// designed.
    pub(crate) fn initial(self, cx: &Context) -> Value {
        if cx.scene.off.iter().any(|o| o == self.key()) {
            return json!(false);
        }
        match self {
            // A scene with its style's own terrain draws it by default; one
            // with terrain of its own puts it in only when asked.
            Opt::Terrain => json!(cx.scene.terrain == Some(Value::Bool(true))),
            Opt::Projection => json!(Self::projection_by_design(cx)),
            Opt::Colors => json!("srgb"),
            _ if self.orbis_only() => json!(false),
            _ => self.read(&cx.designed),
        }
    }

    /// The name the web demo's URL and the scene's `off` use.
    fn key(self) -> &'static str {
        match self {
            Opt::Terrain => "terrain",
            Opt::Hillshade => "hillshade",
            Opt::Light => "light",
            Opt::Night => "night",
            Opt::Buildings => "buildings",
            Opt::Shadows => "shadows",
            Opt::Ao => "ao",
            Opt::Rounded => "rounded",
            Opt::Projection => "projection",
            Opt::Atmosphere => "atmosphere",
            Opt::Fog => "fog",
            Opt::Stars => "stars",
            Opt::Labels => "labels",
            Opt::Colors => "colors",
        }
    }

    /// Edits `style` to `value`.
    pub(crate) fn apply(self, style: &mut Value, value: &Value, cx: &Context) {
        let on = value.as_bool() == Some(true);
        if let Some(kind) = self.layer_type() {
            // Back to each layer's visibility as designed.
            let shown: Vec<&Value> = layers(&cx.designed, kind)
                .filter(|l| visible(l))
                .map(|l| &l["id"])
                .collect();
            for layer in style["layers"].as_array_mut().into_iter().flatten() {
                if layer["type"] == kind {
                    let show = on && shown.contains(&&layer["id"]);
                    layer["layout"]["visibility"] = json!(if show { "visible" } else { "none" });
                }
            }
            return;
        }
        if let Some(sky) = self.sky() {
            style["sky"][sky.name] = if on {
                cx.designed["sky"][sky.name].clone()
            } else {
                json!(sky.off)
            };
            return;
        }
        match self {
            Opt::Terrain => match (&cx.terrain, on) {
                (Some(terrain), true) => style["terrain"] = terrain.clone(),
                _ => remove(style, "terrain"),
            },
            Opt::Light => {
                for layer in style["layers"].as_array_mut().into_iter().flatten() {
                    if layer["type"] == "hillshade" {
                        layer["paint"][HILLSHADE_ANCHOR] = value.clone();
                    }
                }
                style["light"]["anchor"] = value.clone();
            }
            // The ambient light and the dusk with it, and the labels'
            // colours that follow the daylight, or their day side's.
            Opt::Night => {
                with_keys(style, "light", &NIGHT, on, &cx.designed["light"]);
                for designed in layers(&cx.designed, "symbol") {
                    let Some(layer) = layer_mut(style, &designed["id"]) else {
                        continue;
                    };
                    for (name, value) in designed["paint"].as_object().into_iter().flatten() {
                        if reads_daylight(value) {
                            layer["paint"][name] = if on { value.clone() } else { by_day(value) };
                        }
                    }
                }
            }
            Opt::Shadows => with_keys(style, "light", &[SHADOWS], on, &cx.designed["light"]),
            Opt::Ao => {
                for designed in occluded(&cx.designed) {
                    if let Some(layer) = layer_mut(style, &designed["id"]) {
                        with_keys(layer, "paint", &AMBIENT_OCCLUSION, on, &designed["paint"]);
                    }
                }
            }
            Opt::Rounded => {
                for designed in rounded(&cx.designed) {
                    if let Some(layer) = layer_mut(style, &designed["id"]) {
                        with_keys(layer, "layout", &[ROUNDED], on, &designed["layout"]);
                    }
                }
            }
            Opt::Stars => with_keys(style, "sky", &STARRY, on, &cx.designed["sky"]),
            Opt::Projection => style["projection"]["type"] = value.clone(),
            // sRGB is the spec's, without the key.
            Opt::Colors => with_keys(
                style,
                "metadata",
                &[COLOR_SPACE],
                value != "srgb",
                &json!({ COLOR_SPACE: value }),
            ),
            _ => unreachable!("layer and sky options return above"),
        }
    }
}

/// The layer `id` in `style`.
fn layer_mut<'a>(style: &'a mut Value, id: &Value) -> Option<&'a mut Value> {
    style["layers"]
        .as_array_mut()?
        .iter_mut()
        .find(|l| l["id"] == *id)
}

fn remove(style: &mut Value, key: &str) {
    if let Some(object) = style.as_object_mut() {
        object.remove(key);
    }
}

/// The panel's state for the style shown.
#[derive(Default)]
pub(crate) struct Panel {
    pub(crate) context: Option<Context>,
    pub(crate) available: Vec<Opt>,
    pub(crate) values: Vec<(Opt, Value)>,
}

impl Panel {
    /// Another style is on its way.
    pub(crate) fn loading(&mut self) {
        *self = Self::default();
    }

    /// The scene's style is in: offers what it can take, and edits `style`
    /// to each option's default.
    pub(crate) fn attach(&mut self, context: Context, style: &mut Value) {
        self.available = ALL.into_iter().filter(|o| o.supported(&context)).collect();
        for &option in &self.available {
            let value = option.initial(&context);
            if option.read(style) != value {
                option.apply(style, &value, &context);
            }
        }
        self.values = self.available.iter().map(|o| (*o, o.read(style))).collect();
        self.context = Some(context);
    }

    pub(crate) fn value(&self, option: Opt) -> Option<&Value> {
        self.values
            .iter()
            .find(|(o, _)| *o == option)
            .map(|(_, v)| v)
    }

    /// Why `option` is disabled, if another it needs is off.
    pub(crate) fn disabled(&self, option: Opt) -> Option<&'static str> {
        let (needed, value, why) = option.needs()?;
        let current = self.value(needed)?;
        (*current != value).then_some(why)
    }

    /// Picks `value` for `option`, editing `style`; whether it changed.
    pub(crate) fn pick(&mut self, option: Opt, value: Value, style: &mut Value) -> bool {
        let Some(context) = &self.context else {
            return false;
        };
        if self.value(option) == Some(&value) {
            return false;
        }
        option.apply(style, &value, context);
        self.values = self.available.iter().map(|o| (*o, o.read(style))).collect();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenes::Catalogue;

    /// The Orbis-only keys, values and expressions in `style`.
    fn orbis_keys(style: &Value) -> Vec<String> {
        let text = style.to_string();
        let mut keys: Vec<String> = text
            .match_indices("dev.sargunv.orbis:")
            .map(|(at, _)| {
                text[at..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || ":.-".contains(*c))
                    .collect()
            })
            .collect();
        keys.sort();
        keys.dedup();
        keys
    }

    /// The Globe opens with none of Orbis's previews, and its switches
    /// bring back all it designs, the night side with its labels.
    #[test]
    fn globe_opens_spec_only() {
        let scene = Catalogue::read().scene_for("globe");
        let protomaps = json!({
            "version": 8,
            "sources": { "protomaps": { "type": "vector", "url": "x" } },
            "layers": [
                { "id": "background", "type": "background" },
                { "id": "earth", "type": "fill", "source": "protomaps", "source-layer": "earth" },
                { "id": "water", "type": "fill", "source": "protomaps", "source-layer": "water" },
                { "id": "buildings", "type": "fill", "source": "protomaps", "source-layer": "buildings" },
                { "id": "address_label", "type": "symbol", "source": "protomaps", "source-layer": "buildings" },
            ],
        });
        let designed = scene.designed(&protomaps, false);
        assert!(orbis_keys(&designed).len() > 10, "{designed}");
        let mut style = designed.clone();
        let mut panel = Panel::default();
        let context = Context {
            terrain: scene.terrain.clone(),
            scene,
            designed: designed.clone(),
        };
        panel.attach(context, &mut style);
        assert_eq!(orbis_keys(&style), Vec::<String>::new());
        // The label's day colour.
        let label = layer_mut(&mut style, &json!("address_label")).expect("label");
        assert!(label["paint"]["text-color"].is_string(), "{label}");
        for option in ALL.into_iter().filter(|o| o.orbis_only()) {
            let on = if option == Opt::Colors {
                json!("linear")
            } else {
                json!(true)
            };
            assert!(panel.pick(option, on, &mut style), "{option:?}");
        }
        assert_eq!(orbis_keys(&style), orbis_keys(&designed));
    }
}

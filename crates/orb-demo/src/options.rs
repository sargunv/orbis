//! The map options panel's switches, as `web/demo/app/options.js` has
//! them (which says why each is as it is): what the scene's style draws
//! (terrain and its relief, the light's anchor, buildings, the globe and
//! its sky, labels), and only those the style can take. Each choice edits the style JSON; the map takes it as one diffed
//! `setStyle`, so nothing reloads. An Orbis-only option starts off,
//! whatever the scene designs, its keys out of the style, so that by
//! default every scene draws what GL JS and MapLibre Native could.

use serde_json::{Value, json};

use crate::scenes::Scene;

const HILLSHADE_ANCHOR: &str = "hillshade-illumination-anchor";
/// Style-spec#1661's blends for the layer types the spec gives none.
const LAYER_BLENDS: [(&str, &str); 2] = [
    ("hillshade", "dev.sargunv.orbis:hillshade-layer-blend"),
    ("color-relief", "dev.sargunv.orbis:color-relief-layer-blend"),
];
/// A blended layer's `metadata` key holding the paint it takes with its
/// blend off: colours made for the layer drawn normally.
const UNBLENDED: &str = "demo:unblended";
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
    Fog,
    Hillshade,
    Blend,
    Light,
    Buildings,
    Projection,
    Atmosphere,
    Labels,
}

/// A switch, or one of a few choices: `(value, label)`.
pub(crate) enum Kind {
    Switch,
    Choices(&'static [(&'static str, &'static str)]),
}

/// The panel's order.
pub(crate) const ALL: [Opt; 9] = [
    Opt::Terrain,
    Opt::Fog,
    Opt::Hillshade,
    Opt::Blend,
    Opt::Light,
    Opt::Buildings,
    Opt::Projection,
    Opt::Atmosphere,
    Opt::Labels,
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

/// The blend key of a layer's type, where style-spec#1661's preview has one.
fn layer_blend(layer: &Value) -> Option<&'static str> {
    LAYER_BLENDS
        .iter()
        .find(|(ty, _)| layer["type"] == *ty)
        .map(|&(_, key)| key)
}

/// The layers the style blends by style-spec#1661's preview.
fn blended(style: &Value) -> Vec<&Value> {
    style["layers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|l| {
            layer_blend(l).is_some_and(|key| {
                !matches!(&l["paint"][key], Value::Null) && l["paint"][key] != "normal"
            })
        })
        .collect()
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
            Opt::Terrain | Opt::Fog | Opt::Hillshade | Opt::Blend | Opt::Light => {
                "Terrain and light"
            }
            Opt::Buildings => "Buildings",
            Opt::Projection | Opt::Atmosphere => "Globe and sky",
            Opt::Labels => "Map",
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Opt::Terrain => "Terrain",
            Opt::Hillshade => "Hillshade",
            Opt::Blend => "Blended relief",
            Opt::Light => "Light",
            Opt::Buildings => "3D buildings",
            Opt::Projection => "Projection",
            Opt::Atmosphere => "Atmosphere",
            Opt::Fog => "Fog",
            Opt::Labels => "Labels",
        }
    }

    pub(crate) fn note(self) -> Option<&'static str> {
        match self {
            Opt::Blend => {
                Some("Shading multiplied into the colours beneath, which keep their saturation")
            }
            Opt::Light => Some("Fixed to the ground, or turning with the view"),
            _ => None,
        }
    }

    /// A preview of a style-spec proposal that GL JS and MapLibre Native
    /// don't draw: off by default, and off takes its keys out of the style.
    pub(crate) fn orbis_only(self) -> bool {
        self == Opt::Blend
    }

    pub(crate) fn kind(self) -> Kind {
        match self {
            Opt::Light => Kind::Choices(&[("map", "Map"), ("viewport", "Viewport")]),
            Opt::Projection => Kind::Choices(&[("globe", "Globe"), ("mercator", "Mercator")]),
            _ => Kind::Switch,
        }
    }

    /// The option, and its value, without which this one draws nothing,
    /// and how the panel says so.
    pub(crate) fn needs(self) -> Option<(Opt, Value, &'static str)> {
        match self {
            Opt::Blend => Some((Opt::Hillshade, json!(true), "With the hillshade on")),
            Opt::Atmosphere => Some((Opt::Projection, json!("globe"), "On the globe")),
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
            Opt::Blend => !blended(designed).is_empty(),
            Opt::Light => cx.scene.light,
            Opt::Projection => {
                matches!(
                    Self::projection_by_design(cx).as_str(),
                    "globe" | "mercator"
                )
            }
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
            Opt::Blend => json!(!blended(style).is_empty()),
            Opt::Light => {
                let hillshade = layers(style, "hillshade")
                    .next()
                    .map(|l| l["paint"][HILLSHADE_ANCHOR].clone());
                match hillshade {
                    Some(Value::String(anchor)) => json!(anchor),
                    _ => json!(style["light"]["anchor"].as_str().unwrap_or("viewport")),
                }
            }
            Opt::Projection => json!(projection_type(style)),
            _ => unreachable!("layer and sky options return above"),
        }
    }

    /// The scene's default: off where it says so or for an Orbis-only
    /// option, else the option's own, else as designed.
    pub(crate) fn initial(self, cx: &Context) -> Value {
        if cx.scene.off.iter().any(|o| o == self.key()) {
            return json!(false);
        }
        match self {
            // A scene with its style's own terrain draws it by default; one
            // with terrain of its own puts it in only when asked.
            Opt::Terrain => json!(cx.scene.terrain == Some(Value::Bool(true))),
            Opt::Projection => json!(Self::projection_by_design(cx)),
            _ if self.orbis_only() => json!(false),
            _ => self.read(&cx.designed),
        }
    }

    /// The name the web demo's URL and the scene's `off` use.
    fn key(self) -> &'static str {
        match self {
            Opt::Terrain => "terrain",
            Opt::Hillshade => "hillshade",
            Opt::Blend => "blend",
            Opt::Light => "light",
            Opt::Buildings => "buildings",
            Opt::Projection => "projection",
            Opt::Atmosphere => "atmosphere",
            Opt::Fog => "fog",
            Opt::Labels => "labels",
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
            // Each layer's blend with the paint designed for it, or neither
            // and the paint it takes drawn normally.
            Opt::Blend => {
                for designed in blended(&cx.designed) {
                    let Some(layer) = layer_mut(style, &designed["id"]) else {
                        continue;
                    };
                    let unblended = &designed["metadata"][UNBLENDED];
                    let names: Vec<&str> = (unblended.as_object().into_iter().flatten())
                        .map(|(name, _)| name.as_str())
                        .collect();
                    let from = if on { &designed["paint"] } else { unblended };
                    with_keys(layer, "paint", &names, true, from);
                    let key = layer_blend(designed).expect("blended");
                    with_keys(layer, "paint", &[key], on, &designed["paint"]);
                }
            }
            Opt::Light => {
                for layer in style["layers"].as_array_mut().into_iter().flatten() {
                    if layer["type"] == "hillshade" {
                        layer["paint"][HILLSHADE_ANCHOR] = value.clone();
                    }
                }
                style["light"]["anchor"] = value.clone();
            }
            Opt::Projection => style["projection"]["type"] = value.clone(),
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
    /// bring back all it designs.
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
        assert!(!orbis_keys(&designed).is_empty(), "{designed}");
        let mut style = designed.clone();
        let mut panel = Panel::default();
        let context = Context {
            terrain: scene.terrain.clone(),
            scene,
            designed: designed.clone(),
        };
        panel.attach(context, &mut style);
        assert_eq!(orbis_keys(&style), Vec::<String>::new());
        // The hillshade in the colours it takes drawn normally.
        let unblended = &(designed["layers"].as_array().unwrap().iter())
            .find(|l| l["id"] == "hillshade")
            .expect("the hillshade")["metadata"][UNBLENDED];
        let hillshade = layer_mut(&mut style, &json!("hillshade")).expect("hillshade");
        assert!(unblended.is_object(), "{designed}");
        for (name, value) in unblended.as_object().into_iter().flatten() {
            assert_eq!(&hillshade["paint"][name], value);
        }
        for option in ALL.into_iter().filter(|o| o.orbis_only()) {
            assert!(panel.pick(option, json!(true), &mut style), "{option:?}");
        }
        assert_eq!(orbis_keys(&style), orbis_keys(&designed));
    }
}

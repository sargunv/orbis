//! The map options panel's switches, from `web/demo/data/options.json`,
//! which the web demo's `web/demo/app/options.js` reads too (and which says
//! what each field means and why each option is as it is): what the scene's
//! style draws (terrain and its relief, the light's anchor, buildings, the
//! globe and its sky, labels), and only those the style can take. Each
//! choice edits the style JSON; the map takes it as one diffed `setStyle`,
//! so nothing reloads. An Orbis-only option starts off, whatever the scene
//! designs, its keys out of the style, so that by default every scene draws
//! what GL JS and MapLibre Native could.

use std::sync::LazyLock;

use serde::Deserialize;
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

/// The options, in the panel's order.
pub(crate) static ALL: LazyLock<Vec<Opt>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../web/demo/data/options.json"))
        .expect("web/demo/data/options.json")
});

/// One option of the panel.
#[derive(Debug, Deserialize)]
#[serde(try_from = "Entry")]
pub(crate) struct Opt {
    /// The name the web demo's URL and the scene's `off` use.
    key: String,
    pub(crate) group: String,
    pub(crate) label: String,
    pub(crate) note: Option<String>,
    /// A preview of a style-spec proposal that GL JS and MapLibre Native
    /// don't draw: off by default, and off takes its keys out of the style.
    pub(crate) orbis_only: bool,
    /// The option, and its value, without which this one draws nothing,
    /// and how the panel says so.
    needs: Option<Needs>,
    /// `(value, label)`, or none for a switch.
    pub(crate) choices: Option<Vec<(String, String)>>,
    does: Does,
}

/// An entry of `options.json`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Entry {
    key: String,
    group: String,
    label: String,
    note: Option<String>,
    #[serde(default)]
    orbis_only: bool,
    needs: Option<Needs>,
    choices: Option<Vec<(String, String)>>,
    layers: Option<String>,
    sky: Option<SkyNumber>,
}

#[derive(Debug, Deserialize)]
struct Needs {
    key: String,
    value: Value,
    why: String,
}

/// What an option does to the style.
#[derive(Debug)]
enum Does {
    /// Shows or hides every layer of this type.
    Layers(String),
    Sky(SkyNumber),
    Terrain,
    Blend,
    Light,
    Projection,
}

impl TryFrom<Entry> for Opt {
    type Error = String;

    fn try_from(entry: Entry) -> Result<Self, String> {
        let does = match (entry.layers, entry.sky, entry.key.as_str()) {
            (Some(kind), None, _) => Does::Layers(kind),
            (None, Some(sky), _) => Does::Sky(sky),
            (None, None, "terrain") => Does::Terrain,
            (None, None, "blend") => Does::Blend,
            (None, None, "light") => Does::Light,
            (None, None, "projection") => Does::Projection,
            (None, None, key) => return Err(format!("{key}: give it code in options.rs")),
            (Some(_), Some(_), key) => return Err(format!("{key}: both layers and sky")),
        };
        Ok(Self {
            key: entry.key,
            group: entry.group,
            label: entry.label,
            note: entry.note,
            orbis_only: entry.orbis_only,
            needs: entry.needs,
            choices: entry.choices,
            does,
        })
    }
}

/// A number in the style's `sky`, on at any value but `off` (an
/// expression included), with `default` standing in for an unset one; on
/// puts back the designed value (or unsets it).
#[derive(Debug, Deserialize)]
struct SkyNumber {
    name: String,
    default: f64,
    off: f64,
}

impl SkyNumber {
    fn is_on(&self, sky: &Value) -> bool {
        sky.is_object()
            && match &sky[&self.name] {
                Value::Null => self.default != self.off,
                Value::Number(n) => n.as_f64() != Some(self.off),
                _ => true,
            }
    }
}

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

fn projection_by_design(cx: &Context) -> String {
    cx.scene
        .projection
        .as_ref()
        .and_then(|p| p["type"].as_str())
        .map_or_else(|| projection_type(&cx.designed), Into::into)
}

impl Opt {
    /// Whether the scene's style can take this option.
    pub(crate) fn supported(&self, cx: &Context) -> bool {
        let designed = &cx.designed;
        match &self.does {
            Does::Layers(kind) => layers(designed, kind).any(visible),
            Does::Sky(sky) => sky.is_on(&designed["sky"]),
            Does::Terrain => cx.scene.terrain && cx.terrain.is_some(),
            Does::Blend => !blended(designed).is_empty(),
            Does::Light => cx.scene.light,
            Does::Projection => matches!(projection_by_design(cx).as_str(), "globe" | "mercator"),
        }
    }

    /// Its value in `style`.
    pub(crate) fn read(&self, style: &Value) -> Value {
        match &self.does {
            Does::Layers(kind) => json!(layers(style, kind).any(visible)),
            Does::Sky(sky) => json!(sky.is_on(&style["sky"])),
            Does::Terrain => json!(!style["terrain"].is_null()),
            Does::Blend => json!(!blended(style).is_empty()),
            // The first hillshade layer's anchor, else the light's, else
            // the spec's default.
            Does::Light => {
                let hillshade = layers(style, "hillshade")
                    .next()
                    .map(|l| l["paint"][HILLSHADE_ANCHOR].clone());
                match hillshade {
                    Some(Value::String(anchor)) => json!(anchor),
                    _ => json!(style["light"]["anchor"].as_str().unwrap_or("viewport")),
                }
            }
            Does::Projection => json!(projection_type(style)),
        }
    }

    /// The scene's default: off where it says so or for an Orbis-only
    /// option, else the option's own, else as designed.
    pub(crate) fn initial(&self, cx: &Context) -> Value {
        if cx.scene.off.contains(&self.key) {
            return json!(false);
        }
        match self.does {
            Does::Projection => json!(projection_by_design(cx)),
            _ if self.orbis_only => json!(false),
            _ => self.read(&cx.designed),
        }
    }

    /// Edits `style` to `value`.
    pub(crate) fn apply(&self, style: &mut Value, value: &Value, cx: &Context) {
        let on = value.as_bool() == Some(true);
        match &self.does {
            Does::Layers(kind) => {
                // Back to each layer's visibility as designed.
                let shown: Vec<&Value> = layers(&cx.designed, kind)
                    .filter(|l| visible(l))
                    .map(|l| &l["id"])
                    .collect();
                for layer in style["layers"].as_array_mut().into_iter().flatten() {
                    if layer["type"] == *kind {
                        let show = on && shown.contains(&&layer["id"]);
                        layer["layout"]["visibility"] =
                            json!(if show { "visible" } else { "none" });
                    }
                }
            }
            Does::Sky(sky) => {
                let off = json!({ sky.name.as_str(): sky.off });
                let from = if on { &cx.designed["sky"] } else { &off };
                with_keys(style, "sky", &[&sky.name], true, from);
            }
            Does::Terrain => match (&cx.terrain, on) {
                (Some(terrain), true) => style["terrain"] = terrain.clone(),
                _ => remove(style, "terrain"),
            },
            // Each layer's blend with the paint designed for it, or neither
            // and the paint it takes drawn normally.
            Does::Blend => {
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
            Does::Light => {
                for layer in style["layers"].as_array_mut().into_iter().flatten() {
                    if layer["type"] == "hillshade" {
                        layer["paint"][HILLSHADE_ANCHOR] = value.clone();
                    }
                }
                style["light"]["anchor"] = value.clone();
            }
            Does::Projection => style["projection"]["type"] = value.clone(),
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
    pub(crate) available: Vec<&'static Opt>,
    pub(crate) values: Vec<(&'static Opt, Value)>,
}

impl Panel {
    /// Another style is on its way.
    pub(crate) fn loading(&mut self) {
        *self = Self::default();
    }

    /// The scene's style is in: offers what it can take, and edits `style`
    /// to each option's default.
    pub(crate) fn attach(&mut self, context: Context, style: &mut Value) {
        self.available = ALL.iter().filter(|o| o.supported(&context)).collect();
        for option in &self.available {
            let value = option.initial(&context);
            if option.read(style) != value {
                option.apply(style, &value, &context);
            }
        }
        self.read(style);
        self.context = Some(context);
    }

    fn read(&mut self, style: &Value) {
        self.values = self.available.iter().map(|&o| (o, o.read(style))).collect();
    }

    /// The value of the option named `key`, if the style offers it.
    fn value_of(&self, key: &str) -> Option<&Value> {
        self.values
            .iter()
            .find(|(o, _)| o.key == key)
            .map(|(_, v)| v)
    }

    pub(crate) fn value(&self, option: &Opt) -> Option<&Value> {
        self.value_of(&option.key)
    }

    /// Why `option` is disabled, if another it needs is off.
    pub(crate) fn disabled(&self, option: &'static Opt) -> Option<&'static str> {
        let needs = option.needs.as_ref()?;
        let current = self.value_of(&needs.key)?;
        (*current != needs.value).then_some(needs.why.as_str())
    }

    /// Picks `value` for `option`, editing `style`; whether it changed.
    pub(crate) fn pick(&mut self, option: &Opt, value: Value, style: &mut Value) -> bool {
        let Some(context) = &self.context else {
            return false;
        };
        if self.value(option) == Some(&value) {
            return false;
        }
        option.apply(style, &value, context);
        self.read(style);
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

    /// Every option the data names reads, and every option a `needs` or a
    /// scene's `off` names is one.
    #[test]
    fn options_name_options() {
        let named = |key: &str| ALL.iter().any(|o| o.key == key);
        for option in ALL.iter() {
            if let Some(needs) = &option.needs {
                assert!(named(&needs.key), "{}: needs {}", option.key, needs.key);
            }
        }
        for scene in Catalogue::read().scenes() {
            for key in &scene.off {
                assert!(named(key), "{}: off {key}", scene.key);
            }
        }
    }

    /// The Globe opens with none of Orbis's previews, and its switches
    /// bring back all it designs.
    #[test]
    fn globe_opens_spec_only() {
        let scene = Catalogue::read().scene_for("globe");
        let globe: Value =
            serde_json::from_str(include_str!("../../../web/demo/data/globe.json")).unwrap();
        let designed = scene.designed(&globe);
        assert!(!orbis_keys(&designed).is_empty(), "{designed}");
        let mut style = designed.clone();
        let mut panel = Panel::default();
        let context = Context {
            terrain: designed.get("terrain").cloned(),
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
        for option in ALL.iter().filter(|o| o.orbis_only) {
            assert!(panel.pick(option, json!(true), &mut style), "{option:?}");
        }
        assert_eq!(orbis_keys(&style), orbis_keys(&designed));
    }
}

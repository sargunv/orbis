//! `cargo xtask demo-data styles`: the demo's styles that restyle another,
//! written once here so that the web demo and orb-demo load the same file.
//! The Globe is Protomaps Light, pinned in `web/demo/data/protomaps-light.json`
//! (`demo-data protomaps` pins it again), restyled by `globe-restyle.json`
//! into `globe.json`; the Earthquake is the Alps style with its terrain fed
//! by the app, `earthquake.json`. Both bake offline from committed files.

use serde_json::{Map, Value, json};

const OUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/demo/data");
/// Protomaps Light in English, as its API serves it, which
/// `demo-data protomaps` pins as `protomaps-light.json`. The key is the
/// demo's public one; the tiles' URLs carry it too.
const PROTOMAPS: &str = "https://api.protomaps.com/styles/v5/light/en.json?key=73c45a97eddd43fb";
/// The pinned Protomaps style `globe.json` restyles.
const PROTOMAPS_PIN: &str = "protomaps-light.json";
/// The source the Earthquake's app feeds heights to: orb-demo's
/// `quake::SOURCE` and the web demo's `QUAKE_SOURCE`.
const QUAKE_SOURCE: &str = "quake";
/// The Matterhorn, which the quake starts under and the marker marks.
const MATTERHORN: [f64; 2] = [7.6586, 45.9763];

/// Bakes `globe.json` and `earthquake.json` from the committed files they
/// restyle, offline.
pub(crate) fn generate() -> Result<(), String> {
    write("earthquake.json", &quake(&read("alps.json")?))?;
    let globe = globe(&read(PROTOMAPS_PIN)?, &read("globe-restyle.json")?);
    write("globe.json", &globe)
}

/// Pins Protomaps Light as its API serves it now in
/// `protomaps-light.json`, for the Globe to be baked from (`git diff`
/// shows what moved upstream), then bakes.
pub(crate) fn pin_protomaps() -> Result<(), String> {
    let upstream = ureq::Agent::new_with_defaults()
        .get(PROTOMAPS)
        .call()
        .map_err(|e| format!("{PROTOMAPS}: {e}"))?
        .body_mut()
        .read_to_vec()
        .map_err(|e| format!("{PROTOMAPS}: {e}"))?;
    let upstream = parse(&upstream, PROTOMAPS)?;
    eprintln!("demo styles: Protomaps serves {}", upstream["name"]);
    write(PROTOMAPS_PIN, &upstream)?;
    generate()
}

fn read(name: &str) -> Result<Value, String> {
    parse(
        &std::fs::read(format!("{OUT}/{name}")).map_err(|e| format!("{name}: {e}"))?,
        name,
    )
}

fn parse(bytes: &[u8], name: &str) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("{name}: {e}"))
}

fn write(name: &str, style: &Value) -> Result<(), String> {
    let text = style_json(style);
    eprintln!("demo styles: {name}, {} bytes", text.len());
    std::fs::write(format!("{OUT}/{name}"), text).map_err(|e| format!("{name}: {e}"))
}

/// Protomaps Light as the Globe draws it, from `globe-restyle.json` (whose
/// `$comment` says why each part is as it is): its palette set over each
/// layer's paint by id, keeping Protomaps' widths, filters and layout, so
/// the road hierarchy and label placement follow upstream; below zoom 6 the
/// sea's depth and relief over the background and under `earth`, the water
/// fill leaving out Protomaps' ocean polygons; from zoom 6 those polygons
/// over the landuse with the depth over them, the land below sea level and
/// the Caspian put back over that; a hillshade over the land and lakes; the
/// buildings extruded under the labels; the DEM sources (and those derived
/// from the water layer's, which have no `type` of their own), the
/// terrain, the light, the projection and the sky. Each addition is placed
/// by the layers it finds.
fn globe(style: &Value, restyle: &Value) -> Value {
    let mut style = style.clone();
    let mut layers = style["layers"].as_array().cloned().unwrap_or_default();
    for layer in &mut layers {
        if let Some(Value::Object(paint)) = layer["id"].as_str().map(|id| &restyle["paint"][id]) {
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
    let sea = restyle["sea"].as_array().cloned().unwrap_or_default();
    layers.splice(earth..earth, sea);
    let water = layers.iter().position(|l| l["id"] == "water");
    let vector = water.map(|i| layers[i]["source"].clone());
    if let Value::Object(sources) = &restyle["sources"] {
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
        let ocean = restyle["ocean"].as_array().into_iter().flatten().map(|l| {
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
    layers.insert(at, restyle["hillshade"].clone());
    if let Some(at) = layers
        .iter()
        .position(|l| l["source-layer"] == "buildings" && l["type"] == "fill")
    {
        // Protomaps carries `height` on only some footprints (about half in
        // Tokyo and Manhattan, a tenth in central Paris) and no level
        // counts: the rest stand at OpenMapTiles' default `render_height`,
        // 5 m, as Liberty's do where OpenStreetMap has neither.
        let mut buildings = restyle["buildings"].clone();
        buildings["source"] = layers[at]["source"].clone();
        let at = labels(&layers);
        layers.insert(at, buildings);
    }
    for key in ["terrain", "light", "projection", "sky"] {
        style[key] = restyle[key].clone();
    }
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

/// The Alps style with its Mapterhorn sources made one the app feeds
/// (`"tiles": ["orbis-host:"]`): the terrain stands on it and the
/// hillshade shades it, so both follow the moving ground. Its tiles stop at
/// zoom 11, which keeps the heights a step writes to what a view of the
/// valleys needs: a few dozen tiles of 256² heights. A marker on the
/// Matterhorn, which the quake starts under, lets a viewer who wandered off
/// click back to it (the catalogue's `marker`).
fn quake(style: &Value) -> Value {
    let mut style = style.clone();
    let dem: Vec<String> = (style["sources"].as_object().into_iter().flatten())
        .filter(|(_, s)| s["type"] == "raster-dem")
        .map(|(id, _)| id.clone())
        .collect();
    if let Some(sources) = style["sources"].as_object_mut() {
        sources.retain(|id, _| !dem.contains(id));
        sources.insert(
            QUAKE_SOURCE.into(),
            json!({
                "type": "raster-dem",
                "tiles": ["orbis-host:"],
                "encoding": "terrarium",
                "tileSize": 512,
                "maxzoom": 11,
                "attribution": "<a href=\"https://mapterhorn.com/attribution\">© Mapterhorn</a>",
            }),
        );
        sources.insert(
            "marker".into(),
            json!({
                "type": "geojson",
                "data": {"type": "Feature", "properties": {},
                         "geometry": {"type": "Point", "coordinates": MATTERHORN}},
            }),
        );
    }
    for layer in style["layers"].as_array_mut().into_iter().flatten() {
        if layer["source"]
            .as_str()
            .is_some_and(|s| dem.iter().any(|d| d == s))
        {
            layer["source"] = QUAKE_SOURCE.into();
        }
    }
    if let Some(layers) = style["layers"].as_array_mut() {
        layers.extend([
            json!({
                "id": "marker", "type": "circle", "source": "marker",
                "paint": {"circle-radius": 7, "circle-color": "#d7263d",
                          "circle-stroke-color": "#ffffff", "circle-stroke-width": 2.5},
            }),
            json!({
                "id": "marker-label", "type": "symbol", "source": "marker",
                "layout": {"text-field": "Matterhorn", "text-font": ["Noto Sans Bold"],
                           "text-size": 13, "text-anchor": "bottom", "text-offset": [0, -0.9]},
                "paint": {"text-color": "#7a1020", "text-halo-color": "#ffffff",
                          "text-halo-width": 1.5},
            }),
        ]);
    }
    style["terrain"] = json!({"source": QUAKE_SOURCE, "exaggeration": 1.6});
    style
}

/// A style as the baked files are written: the root's keys and each
/// source and layer on a line of its own, each compact, so a diff shows
/// what changed by layer at about the size of the compact style. Keys run
/// in the order a style is read in (`ROOT`, `LAYER`), then alphabetically.
fn style_json(style: &Value) -> String {
    const ROOT: &[&str] = &[
        "version",
        "name",
        "metadata",
        "glyphs",
        "sprite",
        "projection",
        "sky",
        "light",
        "terrain",
        "sources",
        "layers",
    ];
    let Value::Object(root) = style else {
        return compact(style, &[]);
    };
    let entries = ordered(root, ROOT).into_iter().map(|(key, value)| {
        let value = match (key.as_str(), value) {
            ("sources", Value::Object(sources)) => {
                let lines = sources
                    .iter()
                    .map(|(id, source)| format!("    {}: {}", string(id), compact(source, &[])));
                format!("{{\n{}\n  }}", lines.collect::<Vec<_>>().join(",\n"))
            }
            ("layers", Value::Array(layers)) => {
                let lines = layers
                    .iter()
                    .map(|layer| format!("    {}", compact(layer, LAYER)));
                format!("[\n{}\n  ]", lines.collect::<Vec<_>>().join(",\n"))
            }
            _ => compact(value, &[]),
        };
        format!("  {}: {value}", string(key))
    });
    format!("{{\n{}\n}}\n", entries.collect::<Vec<_>>().join(",\n"))
}

/// A layer's keys, in the order the spec lists them.
const LAYER: &[&str] = &[
    "id",
    "type",
    "source",
    "source-layer",
    "minzoom",
    "maxzoom",
    "filter",
    "layout",
    "paint",
    "metadata",
];

/// `object`'s entries, those `first` names first in its order.
fn ordered<'a>(object: &'a Map<String, Value>, first: &[&str]) -> Vec<(&'a String, &'a Value)> {
    let mut entries: Vec<_> = object.iter().collect();
    entries.sort_by_key(|(key, _)| first.iter().position(|k| k == key).unwrap_or(first.len()));
    entries
}

/// `value` on one line, an object's keys `first` names first.
fn compact(value: &Value, first: &[&str]) -> String {
    match value {
        Value::Object(object) => {
            let entries = ordered(object, first)
                .into_iter()
                .map(|(key, value)| format!("{}:{}", string(key), compact(value, &[])));
            format!("{{{}}}", entries.collect::<Vec<_>>().join(","))
        }
        Value::Array(items) => {
            let items = items.iter().map(|item| compact(item, &[]));
            format!("[{}]", items.collect::<Vec<_>>().join(","))
        }
        _ => value.to_string(),
    }
}

fn string(text: &str) -> String {
    Value::from(text).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed Earthquake style is the committed Alps style's.
    #[test]
    fn earthquake_is_the_alps_restyled() {
        let alps = read("alps.json").unwrap();
        let committed = std::fs::read_to_string(format!("{OUT}/earthquake.json")).unwrap();
        assert!(
            committed == style_json(&quake(&alps)),
            "earthquake.json is stale: run `cargo xtask demo-data styles`"
        );
    }

    /// The committed Globe style is the pinned Protomaps style restyled by
    /// the committed restyle.
    #[test]
    fn globe_is_protomaps_restyled() {
        let globe = globe(
            &read(PROTOMAPS_PIN).unwrap(),
            &read("globe-restyle.json").unwrap(),
        );
        let committed = std::fs::read_to_string(format!("{OUT}/globe.json")).unwrap();
        assert!(
            committed == style_json(&globe),
            "globe.json is stale: run `cargo xtask demo-data styles`"
        );
    }

    /// Protomaps' legacy water filter keeps its syntax as the ocean leaves.
    #[test]
    fn water_leaves_the_ocean_in_its_own_syntax() {
        let restyle = json!({"paint": {}, "hillshade": {"id": "hillshade"}});
        let filter = |filter: Value| {
            let style = json!({"sources": {}, "layers": [
                {"id": "water", "type": "fill", "source": "v", "filter": filter},
            ]});
            let globe = globe(&style, &restyle);
            let water = globe["layers"].as_array().unwrap().iter();
            water.clone().find(|l| l["id"] == "water").unwrap()["filter"].clone()
        };
        assert_eq!(
            filter(json!(["==", "$type", "Polygon"])),
            json!(["all", ["==", "$type", "Polygon"], ["!=", "kind", "ocean"]])
        );
        assert_eq!(
            filter(json!(["==", ["geometry-type"], "Polygon"])),
            json!([
                "all",
                ["==", ["geometry-type"], "Polygon"],
                ["!=", ["get", "kind"], "ocean"]
            ])
        );
    }
}

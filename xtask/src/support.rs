//! `cargo xtask support`: the style support matrix, `docs/style-support.md`,
//! generated from the style spec's `v8.json` (types, and GL JS's and
//! Native's support), orb-style's property table (each property's phase
//! and the inputs Orbis reads) and the latest fixture results under
//! `docs/scratch/rt`. Every `summary.json` there with a `tests` list from
//! an unfiltered run is a fixture column, named by its directory
//! (`native`, `chrome`, `query/native`, and any suite added later), so new
//! suites join the matrix as they report. A test counts for a property when a directory
//! of its id is named for it (`fill-color/default`, or a suite's
//! `render-tests/fill-color/default`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use orb_style::{Inputs, LayerType, Phase, Preview, Prop};
use serde_json::Value;

use crate::root;

const SPEC: &str = "web/node_modules/@maplibre/maplibre-gl-style-spec";
const OUTPUT: &str = "docs/style-support.md";

pub(crate) fn run() -> Result<(), String> {
    let root = root();
    let read = |path: &Path| -> Result<Value, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("{}: {e} (run `pnpm -C web install`)", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    };
    let spec = read(&root.join(SPEC).join("src/reference/v8.json"))?;
    let version = read(&root.join(SPEC).join("package.json"))?["version"]
        .as_str()
        .unwrap_or("?")
        .to_string();
    let suites = fixture_suites(&root.join("docs/scratch/rt"))?;
    let text = render(&spec, &version, &suites);
    let out = root.join(OUTPUT);
    std::fs::create_dir_all(out.parent().expect("in docs")).map_err(|e| e.to_string())?;
    std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    println!(
        "support: {OUTPUT} from style-spec {version}, the property table and {} fixture result{}",
        suites.len(),
        if suites.len() == 1 { "" } else { "s" }
    );
    Ok(())
}

/// A fixture run: its name and each test's id and whether it passed.
struct Suite {
    name: String,
    tests: Vec<(String, &'static str)>,
}

/// Every fixture summary under `dir`, one or two levels down.
fn fixture_suites(dir: &Path) -> Result<Vec<Suite>, String> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut walk = vec![(dir.to_path_buf(), 0)];
    while let Some((at, depth)) = walk.pop() {
        let Ok(entries) = std::fs::read_dir(&at) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && depth < 2 {
                walk.push((path, depth + 1));
            } else if depth > 0 && path.file_name().is_some_and(|n| n == "summary.json") {
                found.push(path);
            }
        }
    }
    found.sort();
    let mut suites = Vec::new();
    for path in found {
        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let json: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let Some(tests) = json["tests"].as_array() else {
            continue;
        };
        let name = path
            .parent()
            .and_then(|p| p.strip_prefix(dir).ok())
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        // A filtered run would show every property outside its filter as
        // untested.
        if let Some(filter) = json["filter"].as_str().filter(|f| !f.is_empty()) {
            eprintln!(
                "support: skipping `{name}`, a run filtered to {filter:?}; run the whole suite"
            );
            continue;
        }
        let tests = tests
            .iter()
            .filter_map(|t| {
                let status = match t["status"].as_str()? {
                    "pass" => "pass",
                    "skip" => "skip",
                    _ => "fail",
                };
                Some((t["id"].as_str()?.to_string(), status))
            })
            .collect();
        suites.push(Suite { name, tests });
    }
    Ok(suites)
}

/// What a phase means for a value evaluated in it.
fn phase_row(phase: Phase) -> [&'static str; 4] {
    match phase {
        Phase::Tile => [
            "Once per tile at the tile's zoom, on the job workers",
            "When the tile is laid out again: an edit, or crossing a whole zoom",
            "The layer's layout in every tile it draws",
            "Highest: job time and uploads per tile",
        ],
        Phase::Placement => [
            "Per camera snapshot, in placement",
            "At the next placement",
            "Collision and placement of the layer's labels",
            "Placement time; no layout",
        ],
        Phase::Frame => [
            "Every frame in the shaders, per feature: property rows at the tile's zoom and the next, or the feature-state table",
            "The next frame",
            "Row or state-table uploads for data; node paint for the rest",
            "Low: a uniform, or rows written once per tile",
        ],
        Phase::Layer => [
            "Every frame, once for the whole layer",
            "The next frame",
            "The node's paint",
            "Lowest",
        ],
    }
}

/// A link to the style-spec proposal `preview` follows, if any.
fn proposal_link(preview: Preview) -> Option<String> {
    let n = preview.proposal()?;
    Some(format!(
        "[style-spec#{n}](https://github.com/maplibre/maplibre-style-spec/issues/{n})"
    ))
}

fn inputs(inputs: Inputs) -> String {
    let names: Vec<&str> = inputs.names().collect();
    if names.is_empty() {
        "constant".into()
    } else {
        names.join(", ")
    }
}

/// An `sdk-support` entry for one platform: its version, or `no` linking
/// the issue that tracks it.
fn support(entry: &Value, platform: &str) -> Option<String> {
    let value = entry.get(platform)?.as_str()?;
    Some(if value.starts_with("http") {
        format!("[no]({value})")
    } else {
        value.to_string()
    })
}

/// GL JS's or Native's support: the version with basic support, and with
/// the data-driven form where the spec has one.
fn platform_support(spec: &Value, platforms: &[&str]) -> String {
    let sdk = &spec["sdk-support"];
    let pick = |key: &str| {
        platforms
            .iter()
            .find_map(|p| support(&sdk[key], p))
            .unwrap_or_else(|| "—".into())
    };
    let basic = pick("basic functionality");
    match sdk.get("data-driven styling") {
        Some(_) => format!("{basic}; data {}", pick("data-driven styling")),
        None => basic,
    }
}

/// The spec's own inputs for a property: its expression parameters.
fn spec_inputs(spec: &Value) -> String {
    let params: Vec<&str> = spec["expression"]["parameters"]
        .as_array()
        .map(|p| p.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let names: Vec<&str> = params
        .iter()
        .map(|p| match *p {
            "feature" => "data",
            "feature-state" => "state",
            other => other,
        })
        .filter(|p| *p != "global-state")
        .collect();
    if names.is_empty() {
        "constant".into()
    } else {
        names.join(", ")
    }
}

/// A fixture run's column header, from its directory under
/// `docs/scratch/rt`: the suite, then the platform it ran on.
fn label(name: &str) -> String {
    let (suite, platform) = match name.split_once('/') {
        Some(parts) => parts,
        // GL JS's render suite reports at the top level.
        None => ("render", name),
    };
    let suite = match suite {
        "render" => "GL JS render",
        "query" => "GL JS query",
        "native-render" => "Native render",
        "native-query" => "Native query",
        other => other,
    };
    format!("{suite} ({platform})")
}

/// Whether test `id` is named for property `name`: a directory of its id.
fn names(id: &str, name: &str) -> bool {
    let mut parts: Vec<&str> = id.split('/').collect();
    parts.pop();
    parts.contains(&name)
}

fn fixtures(name: &str, suite: &Suite) -> String {
    let (mut pass, mut fail, mut skip) = (0, 0, 0);
    for (id, status) in &suite.tests {
        if names(id, name) {
            match *status {
                "pass" => pass += 1,
                "skip" => skip += 1,
                _ => fail += 1,
            }
        }
    }
    match (pass + fail, skip) {
        (0, 0) => "—".into(),
        (0, s) => format!("{s} skipped"),
        (run, 0) => format!("{pass}/{run}"),
        (run, s) => format!("{pass}/{run}, {s} skipped"),
    }
}

fn render(spec: &Value, version: &str, suites: &[Suite]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Style support\n\n\
         Generated by `cargo xtask support` from style-spec {version}'s `v8.json`,\n\
         orb-style's property table and the latest fixture results in\n\
         `docs/scratch/rt`{}. Don't edit it by hand: run the fixtures, then the\n\
         task.\n",
        if suites.is_empty() {
            " (none were found)".to_string()
        } else {
            format!(
                " ({})",
                suites
                    .iter()
                    .map(|s| format!("`{}`", s.name))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );

    out.push_str(
        "## Previews\n\n\
         New style syntax beyond the approved spec (keys, values and\n\
         expressions from unapproved proposals or our own, coordinate\n\
         reference systems) is a preview, off unless a map names it:\n\
         `MapOptions::previews` in orb-api, `previews: [...]` (or `\"all\"`) in\n\
         the web package, `--previews` on the CLI. With a preview off, its\n\
         inputs warn and fall back to their defaults. A preview may change or\n\
         disappear at any time. Inputs the spec's syntax already has, read\n\
         where the spec doesn't allow them, need no preview (see Properties).\n\n\
         Every such key, property, value and expression is named\n\
         `dev.sargunv.orbis:<name>`, with the preview of the same name\n\
         (`dev.sargunv.orbis:crs` gates the CRS projection types), whether\n\
         it is our own or from a style-spec proposal; once the spec approves a\n\
         proposal, the spec's name replaces ours. CRS identifiers (`EPSG:3413`)\n\
         and PROJ parameters keep their standard names.\n\n\
         | Preview | Proposal | What it enables |\n|---|---|---|\n",
    );
    for p in Preview::ALL {
        let proposal = proposal_link(p).unwrap_or_else(|| "—".to_string());
        let _ = writeln!(out, "| `{}` | {proposal} | {} |", p.name(), p.summary());
    }

    out.push_str(
        "\n## Light and atmosphere\n\n\
         `light.position` places the light relative to lit geometry: its\n\
         azimuth clockwise from the top of the viewport (`viewport` anchor) or\n\
         from north (`map`), its polar angle from the zenith. GL JS's and\n\
         MapLibre Native's extrusions agree on that direction. Orbis lights\n\
         the extrusions by it, either anchor's light standing over the\n\
         camera's target, as on a flat map. On a globe that light is one direction\n\
         across the sphere: a face far from the target is lit by how far it\n\
         turns to it there, as a wall at the limb facing it is lit.\n\n\
         GL JS's globe extrusions turn their normals and the light onto the\n\
         sphere by axes that keep the horizontal parts the extrusion axes\n\
         negate. That cancels at the target, but a face an arc away from it\n\
         is lit as if the light had swung twice that arc: roofs on the\n\
         light's side of the globe dim, and walls facing the light near the\n\
         limb go dark.\n\n\
         GL JS's globe atmosphere (`getSunPos` in `draw_sky.ts`) negates the\n\
         same vector but reads it on the view's or the globe's axes, which\n\
         turns its sun half a turn about the east axis: the north–south and\n\
         up–down parts reverse. At the spec's default light it stands behind\n\
         the globe, and the haze rims its far side, while the extrusions are\n\
         lit from the front. Native has no globe, sky or atmosphere, so\n\
         GL JS's is the atmosphere's only reference and its fixtures hold it:\n\
         Orbis's scattering takes GL JS's sun.\n",
    );

    out.push_str(
        "\n## Phases\n\n\
         Orbis has no paint and layout: each property is evaluated in one\n\
         phase, which bounds the inputs it can follow.\n\n\
         | Phase | Evaluated | A change shows | What runs again | Cost | Can follow |\n\
         |---|---|---|---|---|---|\n",
    );
    for phase in Phase::ALL {
        let [when, shows, reruns, cost] = phase_row(phase);
        let _ = writeln!(
            out,
            "| {} | {when} | {shows} | {reruns} | {cost} | {} |",
            phase.name(),
            inputs(phase.honours())
        );
    }

    out.push_str(
        "\n## Properties\n\n\
         Inputs are `zoom`; `camera`, the camera's pitch and bearing, in\n\
         values the same for every feature; `data`, the feature's\n\
         properties, id and geometry; and `state`, its feature state.\n\
         `camera × data` is the camera and data in one number:\n\
         `clamp(feature × factor, min, max)`, where the feature part reads\n\
         data, state and the zoom but no camera, and the factor and bounds\n\
         read the camera and the zoom but no data. Property rows hold each\n\
         feature's part, laid out with its tile, and the camera's part\n\
         scales it every frame, so `[\"*\", [\"get\", \"width\"],\n\
         [\"/\", [\"pitch\"], 60]]` widens each feature's line as the map\n\
         tilts. Parts combine through `*` and `/`;\n\
         a `step`, `interpolate`, `match` or `case` the camera chooses by\n\
         takes outputs that are constant multiples of one feature part (or\n\
         0), one the feature chooses by outputs sharing one camera part;\n\
         `min` and `max` of camera values may bound the whole. Other mixes\n\
         (`[\"+\", [\"get\", \"w\"], [\"pitch\"]]`, a colour) are errors.\n\
         Where a feature's part gives no number, the default stands in for\n\
         it, scaled.\n\
         **Spec** lists the spec's expression parameters; **GL JS** and **Native**\n\
         the first versions with the property and its data-driven form (the\n\
         spec's `sdk-support`; Native is Android's); **Orbis** what Orbis reads:\n\
         the spec's inputs, then those it honours beyond the spec, marked\n\
         *accepted (not portable)*: always on, with a warning, once per\n\
         property, that GL JS and Native reject the style; then, per fixture\n\
         run (its suite and platform), the tests named for the property that\n\
         pass.\n",
    );
    // Runs whose tests are named for properties: the render suites.
    let named = |s: &Suite| {
        LayerType::V8
            .iter()
            .flat_map(|t| t.properties())
            .any(|id| s.tests.iter().any(|(test, _)| names(test, id.name())))
    };
    let mut fixture_suites: Vec<&Suite> = suites.iter().filter(|s| named(s)).collect();
    fixture_suites.sort_by_key(|s| label(&s.name));
    for ty in LayerType::V8 {
        let _ = write!(
            out,
            "\n### {}\n\n| Property | Section | Phase | Spec | GL JS | Native | Orbis |",
            ty.name()
        );
        for s in &fixture_suites {
            let _ = write!(out, " {} |", label(&s.name));
        }
        out.push_str("\n|---|---|---|---|---|---|---|");
        out.push_str(&"---|".repeat(fixture_suites.len()));
        out.push('\n');
        // `resampling` is one property under several layer types; the
        // table lists it once per type.
        let mut seen = BTreeMap::new();
        for id in ty.properties() {
            let prop: &Prop = id.prop();
            if seen.insert(prop.name, ()).is_some() {
                continue;
            }
            let section = if prop.paint { "paint" } else { "layout" };
            let entry = &spec[format!("{section}_{}", ty.name())][prop.name];
            let mut orbis = inputs(prop.spec_inputs());
            if !prop.extended.is_empty() {
                let _ = write!(orbis, "; {} accepted (not portable)", inputs(prop.extended));
            }
            // A proposal's property isn't in the spec yet.
            let spec_inputs = match prop.preview {
                Some(preview) => {
                    orbis.push_str("; behind its preview");
                    if let Some(link) = proposal_link(preview) {
                        let _ = write!(orbis, " ({link})");
                    }
                    "—".to_string()
                }
                None => spec_inputs(entry),
            };
            let _ = write!(
                out,
                "| `{}` | {section} | {} | {spec_inputs} | {} | {} | {orbis} |",
                prop.name,
                prop.phase().name(),
                platform_support(entry, &["js"]),
                platform_support(entry, &["android", "ios"]),
            );
            for s in &fixture_suites {
                let _ = write!(out, " {} |", fixtures(prop.name, s));
            }
            out.push('\n');
        }
    }
    out
}

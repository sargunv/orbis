//! Diff of a transaction's draft against the document, into classified
//! changes.

use orb_style::*;
use std::collections::{HashMap, HashSet};

use crate::{Diagnostic, Diagnostics};
use orb_expr::json::{Json, JsonValue, Member};

use crate::doc::{Draft, StyleDoc, member, synthetic};
use crate::doc::{same, same_opt};

/// Whether `json` reads any of `keys` through `global-state`.
fn reads_global(json: &Json, keys: &HashSet<&str>) -> bool {
    match &json.value {
        JsonValue::Array(items) => {
            matches!(items.as_slice(), [op, key] if op.as_str() == Some("global-state")
                && key.as_str().is_some_and(|k| keys.contains(k)))
                || items.iter().any(|i| reads_global(i, keys))
        }
        JsonValue::Object(members) => members.iter().any(|m| reads_global(&m.value, keys)),
        _ => false,
    }
}

fn section<'a>(json: &'a Json, key: &str) -> &'a [Member] {
    json.get(key).and_then(Json::as_object).unwrap_or_default()
}

/// The changes between a layer before and after, by its JSON (given the
/// global state keys whose values changed) and its parsed slots.
fn layer_changes(
    key: LayerKey,
    (old_json, old): (&Json, &Layer),
    (new_json, new): (&Json, &Layer),
    changed: &HashSet<&str>,
    out: &mut Vec<Change>,
) {
    let differs = |a: Option<&Json>, b: Option<&Json>| {
        !same_opt(a, b) || b.is_some_and(|b| reads_global(b, changed))
    };
    let member_of = |json: &Json, id: PropertyId| {
        let section = section(json, if id.is_paint() { "paint" } else { "layout" });
        member(section, id.name()).cloned()
    };
    for id in new.ty.properties() {
        if !differs(
            member_of(old_json, id).as_ref(),
            member_of(new_json, id).as_ref(),
        ) {
            continue;
        }
        out.push(if id.is_paint() {
            Change::Paint {
                layer: key,
                id,
                from: Binding::of_slot(old.slot(id)),
                to: Binding::of_slot(new.slot(id)),
            }
        } else {
            Change::Layout {
                layer: key,
                change: LayoutChange::Property(id),
            }
        });
    }
    let visibility = |json: &Json| member(section(json, "layout"), "visibility").cloned();
    if differs(visibility(old_json).as_ref(), visibility(new_json).as_ref()) {
        out.push(Change::Visibility(key));
    }
    if differs(old_json.get("filter"), new_json.get("filter")) {
        out.push(Change::Layout {
            layer: key,
            change: LayoutChange::Filter,
        });
    }
    if ["minzoom", "maxzoom"]
        .iter()
        .any(|k| !same_opt(old_json.get(k), new_json.get(k)))
    {
        out.push(Change::Layout {
            layer: key,
            change: LayoutChange::ZoomRange,
        });
    }
}

/// The root keys whose change needs a full reload.
const RESTYLE_ROOT: &[&str] = &["sprite", "glyphs", "font-faces", "light", "sky"];

/// The new snapshot and the patch from `doc` to `draft`.
pub(crate) fn diff(doc: &StyleDoc, draft: &Draft) -> Result<(Scene, Patch), Diagnostics> {
    let old = doc.style();
    let mut changes = Vec::new();
    let mut warnings = Vec::new();

    // The root: camera and state defaults, sprites.
    let mut style = Scene {
        name: old.name.clone(),
        center: old.center,
        zoom: old.zoom,
        bearing: old.bearing,
        pitch: old.pitch,
        roll: old.roll,
        transition: old.transition,
        sprites: old.sprites.clone(),
        glyphs: old.glyphs.clone(),
        font_faces: old.font_faces.clone(),
        lights: old.lights.clone(),
        atmosphere: old.atmosphere.clone(),
        terrain: None,
        globe: old.globe,
        projection: old.projection,
        state: old.state.clone(),
        sources: Vec::new(),
        layers: Vec::new(),
        warnings: Vec::new(),
    };
    if draft.new_root {
        // Terrain names a source: it is parsed with the sources below.
        let mut members: Vec<Member> = draft
            .root
            .iter()
            .filter(|m| m.key != "terrain")
            .cloned()
            .collect();
        for (key, value) in [
            ("sources", JsonValue::Object(Vec::new())),
            ("layers", JsonValue::Array(Vec::new())),
        ] {
            members.push(Member {
                key: key.into(),
                key_span: Default::default(),
                value: synthetic(value),
            });
        }
        let root = crate::parse(&synthetic(JsonValue::Object(members)).to_string())?;
        if RESTYLE_ROOT
            .iter()
            .any(|k| !same_opt(member(&doc.root, k), member(&draft.root, k)))
        {
            changes.push(Change::Restyle);
        }
        if !same_opt(
            member(&doc.root, "projection"),
            member(&draft.root, "projection"),
        ) {
            changes.push(Change::Projection);
        }
        warnings.extend(root.warnings.iter().cloned());
        style = Scene {
            warnings: Vec::new(),
            ..root
        };
    }

    let old_state = doc.global_state();
    let new_state = crate::resolve_state(&style.state, &draft.overrides);
    let changed: HashSet<&str> = old_state
        .iter()
        .chain(&new_state)
        .map(|(k, _)| k.as_str())
        .filter(|k| {
            let value = |state: &[(String, Value<'static>)]| {
                state.iter().find(|(s, _)| s == k).map(|(_, v)| v.clone())
            };
            value(&old_state) != value(&new_state)
        })
        .collect();

    if let Some(json) = member(&draft.root, "sky")
        && reads_global(json, &changed)
    {
        let (sky, w) = crate::parse_sky(json, &new_state)?;
        style.atmosphere = Some(sky);
        warnings.extend(w);
        if !changes.contains(&Change::Restyle) {
            changes.push(Change::Restyle);
        }
    }

    // A projection that reads the global state follows it.
    if let Some(json) = member(&draft.root, "projection")
        && (draft.new_root || reads_global(json, &changed))
    {
        let (plane, globe, w) = crate::parse_projection(json, &new_state);
        warnings.extend(w);
        if globe != old.globe && !changes.contains(&Change::Projection) {
            changes.push(Change::Projection);
        }
        style.projection = plane;
        style.globe = globe;
    }

    // Sources.
    let mut source_diffs = Vec::new();
    let old_sources: HashMap<SourceKey, &std::sync::Arc<Source>> =
        doc.source_keys().zip(&old.sources).collect();
    for entry in &draft.sources {
        match old_sources.get(&entry.key) {
            Some(_) if draft.new_data.contains(&entry.key) => {
                let (source, w) = crate::parse_source(&entry.id, &entry.json)?;
                warnings.extend(w);
                style.sources.push(std::sync::Arc::new(source));
                changes.push(Change::SourceData(entry.key));
            }
            Some(source)
                if let Some((_, diffs)) = draft.diffs.iter().find(|(k, _)| *k == entry.key) =>
            {
                // The parsed features change by the diffs; the JSON already did.
                let source =
                    crate::geojson::apply_diffs(source, diffs).map_err(crate::doc::error)?;
                style.sources.push(std::sync::Arc::new(source));
                changes.push(Change::SourceData(entry.key));
                source_diffs.push((entry.key, diffs.clone()));
            }
            Some(source) => style.sources.push(std::sync::Arc::clone(source)),
            None => {
                let (source, w) = crate::parse_source(&entry.id, &entry.json)?;
                warnings.extend(w);
                style.sources.push(std::sync::Arc::new(source));
                changes.push(Change::SourceAdded(entry.key));
            }
        }
    }
    for key in doc.source_keys() {
        if !draft.sources.iter().any(|s| s.key == key) {
            changes.push(Change::SourceRemoved(key));
        }
    }

    // Another plane, or sources in other tile matrix sets: the surface is
    // planned again and every tile loads on it.
    if (style.projection != old.projection || style.tile_matrix_sets() != old.tile_matrix_sets())
        && !changes.contains(&Change::Restyle)
    {
        changes.push(Change::Restyle);
    }

    // Terrain reads the new sources; switching it needs no reload.
    if let Some(json) = member(&draft.root, "terrain") {
        let (terrain, w) = crate::parse_terrain(json, &style.sources)?;
        warnings.extend(w);
        style.terrain = Some(terrain);
    }

    // Layers.
    let old_layers: HashMap<LayerKey, (usize, &std::sync::Arc<Layer>)> = doc
        .layer_keys()
        .zip(&old.layers)
        .enumerate()
        .map(|(i, (k, l))| (k, (i, l)))
        .collect();
    let mut ids = HashSet::new();
    for (index, entry) in draft.layers.iter().enumerate() {
        let id = entry
            .json
            .get("id")
            .and_then(Json::as_str)
            .unwrap_or_default();
        if !ids.insert(id) {
            return Err(Diagnostics {
                errors: vec![Diagnostic {
                    path: format!("layers[{index}].id"),
                    span: Default::default(),
                    message: format!("duplicate layer id \"{id}\""),
                }],
                warnings: Vec::new(),
            });
        }
        let old_entry = old_layers.get(&entry.key);
        let global = reads_global(&entry.json, &changed);
        let reparse = old_entry.is_none() || draft.dirty.contains(&entry.key) || global;
        if !reparse {
            style
                .layers
                .push(std::sync::Arc::clone(old_entry.expect("kept").1));
            continue;
        }
        let (layer, w) = crate::parse_layer(&entry.json, index, &style.sources, &new_state)?;
        warnings.extend(w);
        match old_entry {
            None => changes.push(Change::LayerAdded(entry.key)),
            Some(&(old_index, old)) => layer_changes(
                entry.key,
                (doc.layer_json(old_index), old),
                (&entry.json, &layer),
                &changed,
                &mut changes,
            ),
        }
        style.layers.push(std::sync::Arc::new(layer));
    }
    for key in doc.layer_keys() {
        if !draft.layers.iter().any(|l| l.key == key) {
            changes.push(Change::LayerRemoved(key));
        }
    }
    let kept: Vec<LayerKey> = draft
        .layers
        .iter()
        .map(|l| l.key)
        .filter(|k| old_layers.contains_key(k))
        .collect();
    let before: Vec<LayerKey> = doc
        .layer_keys()
        .filter(|k| draft.layers.iter().any(|l| l.key == *k))
        .collect();
    if kept != before {
        changes.push(Change::LayersReordered);
    }
    // Unchanged layers' JSON is shared; only edited ones were compared.
    debug_assert!(draft.layers.iter().all(|l| {
        old_layers
            .get(&l.key)
            .is_none_or(|&(i, _)| draft.dirty.contains(&l.key) || same(doc.layer_json(i), &l.json))
    }));
    style.warnings = warnings.clone();
    Ok((
        style,
        Patch {
            version: 0,
            changes,
            feature_state: Vec::new(),
            warnings,
            source_diffs,
        },
    ))
}

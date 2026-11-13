//! Diff of a transaction's draft against the document, into classified
//! changes.

use crate::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::Arc;

use crate::v8::{Diagnostic, Diagnostics};
use orb_expr::json::{Json, JsonValue, Member};

use crate::v8::doc::{Draft, StyleDoc, member, synthetic};
use crate::v8::doc::{same, same_opt};

/// Whether `json` reads any of `keys` through `global-state`.
fn reads_global(json: &Json, keys: &HashSet<&str>) -> bool {
    if keys.is_empty() {
        return false;
    }
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
    if old.transitions != new.transitions {
        out.push(Change::Transitions(key));
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

/// The root keys whose change needs a full reload: every label's glyphs
/// load again.
const RESTYLE_ROOT: &[&str] = &["glyphs", "font-faces"];

/// Adds `change` unless an earlier step did.
fn add_once(changes: &mut Vec<Change>, change: Change) {
    if !changes.contains(&change) {
        changes.push(change);
    }
}

/// The new snapshot and the patch from `doc` to `draft`.
pub(crate) fn diff(doc: &StyleDoc, draft: &Draft) -> Result<(Style, Patch), Diagnostics> {
    let old = doc.style();
    let previews = old.previews;
    let mut changes = Vec::new();
    let mut warnings = Vec::new();

    // The root: camera and state defaults, sprites.
    let mut style = Style {
        terrain: None,
        sources: Vec::new(),
        layers: Vec::new(),
        warnings: Vec::new(),
        ..(**old).clone()
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
        let root = crate::v8::parse_json(&synthetic(JsonValue::Object(members)), previews, &[])?;
        if RESTYLE_ROOT
            .iter()
            .any(|k| !same_opt(member(&doc.root, k), member(&draft.root, k)))
        {
            changes.push(Change::Restyle);
        }
        if !same_opt(member(&doc.root, "sprite"), member(&draft.root, "sprite")) {
            changes.push(Change::Sprite);
        }
        if !same_opt(
            member(&doc.root, "projection"),
            member(&draft.root, "projection"),
        ) {
            changes.push(Change::Projection);
        }
        // Sky, light, terrain, transition, the camera defaults: frames read
        // them from the new snapshot.
        if (doc.root.iter().chain(draft.root.iter()))
            .any(|m| !same_opt(member(&doc.root, &m.key), member(&draft.root, &m.key)))
        {
            changes.push(Change::Root);
        }
        warnings.extend(root.warnings.iter().cloned());
        style = Style {
            warnings: Vec::new(),
            ..root
        };
    }

    let old_state = doc.global_state();
    let new_state = crate::v8::resolve_state(&style.state, &draft.overrides);
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
        let (sky, w) = crate::v8::parse_sky(json, &new_state, previews)?;
        style.atmosphere = Some(sky);
        warnings.extend(w);
        add_once(&mut changes, Change::Root);
    }

    if let Some(json) = member(&draft.root, "light")
        && reads_global(json, &changed)
    {
        style.light = crate::v8::parse_light(json, &new_state, previews)?;
        add_once(&mut changes, Change::Root);
    }

    // A projection that reads the global state follows it.
    if let Some(json) = member(&draft.root, "projection")
        && (draft.new_root || reads_global(json, &changed))
    {
        let (plane, globe, w) = crate::v8::parse_projection(json, &new_state, previews);
        warnings.extend(w);
        if globe != old.globe {
            add_once(&mut changes, Change::Projection);
        }
        style.projection = plane;
        style.globe = globe;
    }

    // Sources.
    let mut source_diffs = Vec::new();
    let old_sources: HashMap<SourceKey, &Arc<Source>> =
        doc.source_keys().zip(&old.sources).collect();
    for entry in &draft.sources {
        let old = old_sources.get(&entry.key);
        let kept = old.filter(|_| !draft.new_data.contains(&entry.key));
        match (kept, draft.diffs.iter().find(|(k, _)| *k == entry.key)) {
            (Some(source), Some((_, diffs))) => {
                // The parsed features change by the diffs; the JSON already did.
                let source = crate::v8::geojson::apply_diffs(source, diffs)
                    .map_err(crate::v8::doc::error)?;
                style.sources.push(Arc::new(source));
                changes.push(Change::SourceData(entry.key));
                source_diffs.push((entry.key, diffs.clone()));
            }
            (Some(source), None) => style.sources.push(Arc::clone(source)),
            (None, _) => {
                let (source, w) = crate::v8::parse_source(&entry.id, &entry.json, previews)?;
                warnings.extend(w);
                style.sources.push(Arc::new(source));
                changes.push(match old {
                    Some(_) => Change::SourceData(entry.key),
                    None => Change::SourceAdded(entry.key),
                });
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
    if style.projection != old.projection || style.tile_matrix_sets() != old.tile_matrix_sets() {
        add_once(&mut changes, Change::Restyle);
    }

    // Terrain reads the new sources; switching it needs no reload.
    if let Some(json) = member(&draft.root, "terrain") {
        let (terrain, w) = crate::v8::parse_terrain(json, &style.sources)?;
        warnings.extend(w);
        style.terrain = Some(terrain);
    }

    // Layers.
    let old_layers: HashMap<LayerKey, (usize, &Arc<Layer>)> = doc
        .layer_keys()
        .zip(&old.layers)
        .enumerate()
        .map(|(i, (k, l))| (k, (i, l)))
        .collect();
    // Which old layers the draft keeps, by index.
    let mut kept = vec![false; old.layers.len()];
    let mut dirty = draft.dirty.clone();
    dirty.sort_unstable();
    let dirty = |key| dirty.binary_search(key).is_ok();
    let mut ids = HashSet::default();
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
        if let Some(&(i, _)) = old_entry {
            kept[i] = true;
        }
        if let Some(&(_, old)) = old_entry
            && !dirty(&entry.key)
            && !reads_global(&entry.json, &changed)
        {
            style.layers.push(Arc::clone(old));
            continue;
        }
        let (layer, w) =
            crate::v8::parse_layer(&entry.json, index, &style.sources, &new_state, previews)?;
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
        style.layers.push(Arc::new(layer));
    }
    let old_keys = || doc.layer_keys().zip(&kept);
    changes.extend(
        old_keys()
            .filter(|(_, kept)| !**kept)
            .map(|(key, _)| Change::LayerRemoved(key)),
    );
    // The layers both have, in their old order and in their new.
    let before = old_keys().filter(|(_, kept)| **kept).map(|(key, _)| key);
    let after = (draft.layers.iter().map(|l| l.key)).filter(|k| old_layers.contains_key(k));
    if !before.eq(after) {
        changes.push(Change::LayersReordered);
    }
    // Unchanged layers' JSON is shared; only edited ones were compared.
    debug_assert!(draft.layers.iter().all(|l| {
        old_layers
            .get(&l.key)
            .is_none_or(|&(i, _)| dirty(&l.key) || same(doc.layer_json(i), &l.json))
    }));
    style.warnings.clone_from(&warnings);
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

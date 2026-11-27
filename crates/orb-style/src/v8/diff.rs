//! Diff of a transaction's draft against the document, into classified
//! changes.

use crate::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::Arc;

use crate::v8::{Diagnostic, Diagnostics, RootKey};
use orb_expr::json::{Json, JsonValue, Member};

use crate::v8::doc::{Draft, StyleDoc, Written, member, synthetic};
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
    if [
        crate::v8::target::DESTINATION,
        crate::v8::target::SOURCE,
        crate::v8::target::OPERATIONS,
    ]
    .iter()
    .any(|k| differs(old_json.get(k), new_json.get(k)))
    {
        out.push(Change::Targets(key));
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

    // The root without its sources, layers and terrain (which names a
    // source: it is parsed with the sources below), under the draft's
    // global state overrides, which light, sky and projection read.
    let parse_root = || {
        let mut members: Vec<Member> = draft
            .root
            .iter()
            .filter(|m| m.key != RootKey::Terrain.name())
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
        let written = draft
            .written
            .iter()
            .filter_map(Written::root_path)
            .collect();
        crate::v8::parse_json(
            &synthetic(JsonValue::Object(members)),
            previews,
            &draft.overrides,
            written,
        )
    };
    let mut root = draft.new_root.then(parse_root).transpose()?;
    if draft.new_root {
        let changed = |key: RootKey| !same_opt(doc.root(key), member(&draft.root, key.name()));
        // Every label's glyphs load again, or the render targets are
        // planned again.
        if [RootKey::Glyphs, RootKey::FontFaces, RootKey::Targets]
            .into_iter()
            .any(changed)
        {
            changes.push(Change::Restyle);
        }
        if changed(RootKey::Sprite) {
            changes.push(Change::Sprite);
        }
        // Sky, light, terrain, transition, the camera defaults: frames read
        // them from the new snapshot.
        if (doc.root.iter().chain(draft.root.iter()))
            .any(|m| !same_opt(member(&doc.root, &m.key), member(&draft.root, &m.key)))
        {
            changes.push(Change::Root);
        }
    }

    let old_state = doc.global_state();
    let defaults = root.as_ref().map_or(&old.state, |root| &root.state);
    let new_state = crate::v8::resolve_state(defaults, &draft.overrides);
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

    // A root value that reads changed global state (light, sky,
    // projection) follows it.
    if draft.root.iter().any(|m| reads_global(&m.value, &changed)) {
        if root.is_none() {
            root = Some(parse_root()?);
        }
        add_once(&mut changes, Change::Root);
    }

    // The root: camera and state defaults, sprites, light, sky, projection.
    let root = match root {
        Some(mut root) => {
            warnings.append(&mut root.warnings);
            root
        }
        None => (**old).clone(),
    };
    let mut style = Style {
        terrain: None,
        sources: Vec::new(),
        layers: Vec::new(),
        warnings: Vec::new(),
        ..root
    };
    // A new surface: tiles lay out again for it.
    if style.projection != old.projection || style.globe != old.globe {
        changes.push(Change::Projection);
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
    if let Some(json) = member(&draft.root, RootKey::Terrain.name()) {
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
        let written = (draft.written.iter())
            .filter_map(|w| w.layer_path(entry.key, index))
            .collect();
        let (layer, w) = crate::v8::parse_edited_layer(
            &entry.json,
            index,
            &style.sources,
            &new_state,
            previews,
            written,
        )?;
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

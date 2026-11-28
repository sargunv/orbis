//! Diff of a transaction's draft against the document, into classified
//! changes.

use crate::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::Arc;

use crate::v8::{Diagnostics, RootKey};
use orb_expr::json::{Json, JsonValue, Member};

use crate::v8::doc::{Draft, StyleDoc, Written, member, set_member, synthetic};
use crate::v8::doc::{same, same_opt};
use crate::v8::{layer, parse_edited, source, terrain};

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
/// `draft` loses what it held that failed to parse and the document drops:
/// broken layers, sources and terrain of a new document.
pub(crate) fn diff(doc: &StyleDoc, draft: &mut Draft) -> Result<(Style, Patch), Diagnostics> {
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
        old.sources.iter().map(|s| (s.key, s)).collect();
    let mut dropped = Vec::new();
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
                let path = format!("sources.{}", entry.id);
                let droppable =
                    draft.replaced && !draft.written.contains(&Written::Source(entry.key));
                let (source, w) =
                    parse_edited("the source", &[], previews, Vec::new(), droppable, |cx| {
                        source::parse_source(cx, entry.key, &entry.id, &entry.json, &path)
                    })?;
                warnings.extend(w);
                let Some(source) = source else {
                    dropped.push(entry.key);
                    continue;
                };
                style.sources.push(Arc::new(source));
                changes.push(match old {
                    Some(_) => Change::SourceData(entry.key),
                    None => Change::SourceAdded(entry.key),
                });
            }
        }
    }
    draft.sources.retain(|s| !dropped.contains(&s.key));
    for source in &old.sources {
        if !style.sources.iter().any(|s| s.key == source.key) {
            changes.push(Change::SourceRemoved(source.key));
        }
    }

    // Another plane, or sources in other tile matrix sets: the surface is
    // planned again and every tile loads on it.
    if style.projection != old.projection || style.tile_matrix_sets() != old.tile_matrix_sets() {
        add_once(&mut changes, Change::Restyle);
    }

    // Terrain reads the new sources; switching it needs no reload.
    if let Some(json) = member(&draft.root, RootKey::Terrain.name()) {
        let terrain = Written::Root(RootKey::Terrain);
        let droppable = draft.replaced && !draft.written.contains(&terrain);
        let (parsed, w) =
            parse_edited("the terrain", &[], previews, Vec::new(), droppable, |cx| {
                terrain::parse(cx, json, &style.sources)
            })?;
        warnings.extend(w);
        style.terrain = parsed;
        if style.terrain.is_none() {
            let mut root = draft.root.as_ref().clone();
            set_member(&mut root, RootKey::Terrain.name(), None);
            draft.root = Arc::new(root);
        }
    }

    // Layers.
    let old_layers: HashMap<LayerKey, &Arc<Layer>> =
        old.layers.iter().map(|l| (l.key, l)).collect();
    let mut dirty = draft.dirty.clone();
    dirty.sort_unstable();
    let dirty = |key: &LayerKey| dirty.binary_search(key).is_ok();
    let mut ids = HashSet::default();
    let mut dropped = Vec::new();
    for (index, entry) in draft.layers.iter().enumerate() {
        let old = old_layers.get(&entry.key);
        let id = entry
            .json
            .get("id")
            .and_then(Json::as_str)
            .unwrap_or_default();
        let duplicate = !ids.insert(id);
        if let Some(old) = old
            && !duplicate
            && !dirty(&entry.key)
            && !reads_global(&entry.json, &changed)
        {
            style.layers.push(Arc::clone(old));
            continue;
        }
        let written: Vec<String> = (draft.written.iter())
            .filter_map(|w| w.layer_path(entry.key, index))
            .collect();
        let droppable = draft.replaced && written.is_empty();
        let (layer, w) = parse_edited(
            "the layer",
            &new_state,
            previews,
            written,
            droppable,
            |cx| {
                if duplicate {
                    let message = format!("duplicate layer id \"{id}\"");
                    cx.error(format!("layers[{index}].id"), Default::default(), message);
                    return None;
                }
                let (json, sources) = (&entry.json, &style.sources);
                layer::parse_layer(cx, json, index, entry.key, sources, &mut Default::default())
            },
        )?;
        warnings.extend(w);
        let Some(layer) = layer else {
            dropped.push(entry.key);
            continue;
        };
        match old {
            None => changes.push(Change::LayerAdded(entry.key)),
            Some(old) => layer_changes(
                entry.key,
                (doc.layer_json(entry.key).expect("an old layer"), old),
                (&entry.json, &layer),
                &changed,
                &mut changes,
            ),
        }
        style.layers.push(Arc::new(layer));
    }
    draft.layers.retain(|l| !dropped.contains(&l.key));
    let kept = |key: &LayerKey| style.layers.iter().any(|l| l.key == *key);
    changes.extend(
        (old.layers.iter())
            .filter(|l| !kept(&l.key))
            .map(|l| Change::LayerRemoved(l.key)),
    );
    // The layers both have, in their old order and in their new.
    let before = (old.layers.iter().map(|l| l.key)).filter(kept);
    let after = (style.layers.iter().map(|l| l.key)).filter(|k| old_layers.contains_key(k));
    if !before.eq(after) {
        changes.push(Change::LayersReordered);
    }
    // Unchanged layers' JSON is shared; only edited ones were compared.
    debug_assert!(draft.layers.iter().all(|l| {
        doc.layer_json(l.key)
            .is_none_or(|old| dirty(&l.key) || same(old, &l.json))
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

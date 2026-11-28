//! Feature state as the app set it, by source id, source layer and
//! feature id: the one home of feature state. Reads come from here, the
//! GPU state table's rows ([`crate::state`]) among them.
//!
//! State lasts until it is removed, its source leaves the style (or is
//! defined anew, which makes it another source), or a style loads
//! ([`Map::set_style_json`] or a style URL's arrival, even one the loaded
//! style is diffed to), as in MapLibre Native, whose loaded style takes
//! every source's state with it. State kept for a source the style doesn't
//! have applies once a transaction adds the source, as mln-rs's map store
//! does; state set before any style reads back until the first style
//! loads, which clears it. Transactions check their
//! targets as GL JS does (the source exists, a vector source's feature
//! names its source layer); [`Map::set_feature_state`] takes any target,
//! as MapLibre Native does, whose state a feature never carries draws
//! nothing.

use std::sync::LazyLock;

use rustc_hash::FxHashMap as HashMap;

use orb_style::v8::{FeatureTarget, StyleDoc};
use orb_style::{FeatureStateOp, Patch, SourceKey, Value};

use crate::map::{Map, WaitingStyle};

/// A feature's state: its keys in the order first set, never empty.
type State = Vec<(String, Value<'static>)>;

#[derive(Debug, Default)]
pub(crate) struct FeatureStateStore {
    sources: HashMap<String, SourceStates>,
}

#[derive(Debug, Default)]
struct SourceStates {
    /// The style's source by this id that the state applies to, once it
    /// has one.
    key: Option<SourceKey>,
    /// Each feature's state, by source layer and feature id; never an
    /// empty layer.
    features: HashMap<Option<String>, HashMap<String, State>>,
}

impl FeatureStateStore {
    /// What the drawn style reads: no state while a new style waits for
    /// its modules, the store's source keys being that style's, which
    /// number their sources afresh.
    pub(crate) fn drawn<'a>(&'a self, waiting: &Option<WaitingStyle>) -> &'a Self {
        static NONE: LazyLock<FeatureStateStore> = LazyLock::new(FeatureStateStore::default);
        match waiting {
            Some(WaitingStyle { fresh: true, .. }) => &NONE,
            _ => self,
        }
    }

    /// The state of a feature of the style's source `key`.
    pub(crate) fn state(
        &self,
        key: u32,
        layer: Option<&str>,
        id: &str,
    ) -> &[(String, Value<'static>)] {
        (self.sources.values())
            .find(|s| s.key == Some(SourceKey(key)))
            .and_then(|s| {
                let (_, features) = s.features.iter().find(|(l, _)| l.as_deref() == layer)?;
                features.get(id)
            })
            .map_or(&[], Vec::as_slice)
    }

    /// The states of features of the style's `sources` (keys), as JSON
    /// objects by (source, source layer, id): what queried features
    /// report.
    pub(crate) fn snapshot(
        &self,
        sources: &[u32],
    ) -> HashMap<(u32, Option<String>, String), String> {
        let bound = (self.sources.values())
            .filter_map(|s| Some((s.key?.0, s)))
            .filter(|(key, _)| sources.contains(key));
        bound
            .flat_map(|(key, s)| {
                (s.features.iter()).flat_map(move |(layer, features)| {
                    features.iter().map(move |(id, state)| {
                        let object = Value::Object(std::borrow::Cow::Borrowed(state));
                        ((key, layer.clone(), id.clone()), object.to_json_string())
                    })
                })
            })
            .collect()
    }

    /// The state of `target`'s feature; none without an id.
    fn get(&self, target: &FeatureTarget) -> &[(String, Value<'static>)] {
        let Some(id) = &target.id else {
            return &[];
        };
        (self.sources.get(&target.source))
            .and_then(|s| s.features.get(&target.source_layer)?.get(id))
            .map_or(&[], Vec::as_slice)
    }

    /// Merges `state` into a feature's state, `key` naming the style's
    /// source by its id if it has one; whether it changed anything.
    fn set(
        &mut self,
        source: &str,
        key: Option<SourceKey>,
        layer: Option<&str>,
        id: &str,
        state: &[(String, Value<'static>)],
    ) -> bool {
        if state.is_empty() {
            return false;
        }
        let source = self.sources.entry(source.to_string()).or_default();
        source.key = key;
        let kept = (source
            .features
            .entry(layer.map(str::to_string))
            .or_default())
        .entry(id.to_string())
        .or_default();
        let mut changed = false;
        for (name, value) in state {
            match kept.iter_mut().find(|(k, _)| k == name) {
                Some((_, old)) if old == value => {}
                Some((_, old)) => {
                    *old = value.clone();
                    changed = true;
                }
                None => {
                    kept.push((name.clone(), value.clone()));
                    changed = true;
                }
            }
        }
        changed
    }

    /// Removes `key` of a feature's state, or its whole state; without an
    /// id, the whole state of every feature of the source layer (or of the
    /// source's features without one). Whether it removed anything.
    fn remove(&mut self, target: &FeatureTarget, key: Option<&str>) -> bool {
        let Some(source) = self.sources.get_mut(&target.source) else {
            return false;
        };
        let Some(layer) = source.features.get_mut(&target.source_layer) else {
            return false;
        };
        // Takes the key, or everything; whether it took anything, and
        // whether the state is gone.
        let prune = |state: &mut State| {
            let before = state.len();
            match key {
                Some(key) => state.retain(|(k, _)| k != key),
                None => state.clear(),
            }
            (state.len() != before, state.is_empty())
        };
        let mut changed = false;
        match &target.id {
            Some(id) => {
                if let Some(state) = layer.get_mut(id) {
                    let (took, empty) = prune(state);
                    changed = took;
                    if empty {
                        layer.remove(id);
                    }
                }
            }
            None => {
                changed = !layer.is_empty();
                layer.clear();
            }
        }
        if layer.is_empty() {
            source.features.remove(&target.source_layer);
        }
        if source.features.is_empty() {
            self.sources.remove(&target.source);
        }
        changed
    }

    /// Follows `doc`'s sources: the state of a source that left it or was
    /// defined anew goes, and state kept for a source it now has applies
    /// to it.
    pub(crate) fn follow(&mut self, doc: &StyleDoc) {
        self.sources.retain(|id, source| {
            let now = doc.source_key(id);
            match (source.key, now) {
                (Some(was), now) => now == Some(was),
                (None, now) => {
                    source.key = now;
                    true
                }
            }
        });
    }

    /// Drops every state, returned as the edits that take it from the
    /// style's sources.
    fn clear(&mut self) -> Vec<FeatureStateOp> {
        (self.sources.drain())
            .filter_map(|(_, s)| Some((s.key?, s.features)))
            .flat_map(|(source, features)| {
                features
                    .into_keys()
                    .map(move |layer| FeatureStateOp::Remove {
                        source,
                        source_layer: layer,
                        id: None,
                        key: None,
                    })
            })
            .collect()
    }
}

impl Map {
    /// The state of `target`'s feature as set ([`set_feature_state`](Self::set_feature_state)
    /// or a transaction's `setFeatureState`); empty when it has none, or
    /// `target` names no feature.
    pub fn feature_state(&self, target: &FeatureTarget) -> &[(String, Value<'static>)] {
        self.feature_state.get(target)
    }

    /// Merges `state` into the state of `target`'s feature, whatever its
    /// source: the map keeps state for a source the style lacks until the
    /// source is added, and state set before any style until the first
    /// style loads ([module docs](crate::feature_state)).
    /// Returns whether it changed anything: state the feature already
    /// holds draws nothing new. A target without an id sets nothing.
    pub fn set_feature_state(
        &mut self,
        target: &FeatureTarget,
        state: &[(String, Value<'static>)],
    ) -> bool {
        let Some(id) = &target.id else {
            return false;
        };
        let key = self.style_doc().and_then(|d| d.source_key(&target.source));
        let layer = target.source_layer.as_deref();
        let changed = (self.feature_state).set(&target.source, key, layer, id, state);
        if let (true, Some(key)) = (changed, key) {
            self.mirror_states(&[FeatureStateOp::Set {
                source: key,
                source_layer: target.source_layer.clone(),
                id: id.clone(),
                state: state.to_vec(),
            }]);
        }
        changed
    }

    /// Removes `key` of the state of `target`'s feature, or its whole
    /// state; without an id, the whole state of every feature of its
    /// source layer (or of its source's features without one), whatever
    /// `key`, as MapLibre Native does. Returns whether it removed anything.
    pub fn remove_feature_state(&mut self, target: &FeatureTarget, key: Option<&str>) -> bool {
        let key = key.filter(|_| target.id.is_some());
        let changed = self.feature_state.remove(target, key);
        let source = self.style_doc().and_then(|d| d.source_key(&target.source));
        if let (true, Some(source)) = (changed, source) {
            self.mirror_states(&[FeatureStateOp::Remove {
                source,
                source_layer: target.source_layer.clone(),
                id: target.id.clone(),
                key: key.map(str::to_string),
            }]);
        }
        changed
    }

    /// Drops every feature state, as a style load does: from the style's
    /// table too, which a style diffed to keeps.
    pub(crate) fn clear_feature_state(&mut self) {
        let removals = self.feature_state.clear();
        self.mirror_states(&removals);
    }

    /// Takes in an edit's patch of the document [`style_doc`](Self::style_doc)
    /// has now: state follows its sources, then takes its feature-state
    /// edits, each to the source it named if the edit kept it.
    pub(crate) fn take_state_edits(&mut self, patch: &Patch) {
        // The fields themselves, which the store's borrow doesn't overlap.
        let doc =
            (self.waiting_style.as_ref().map(|w| &w.doc)).or(self.style.as_ref().map(|s| &s.doc));
        let Some(doc) = doc else {
            return;
        };
        let mut mirror = Vec::new();
        if !patch.changes.is_empty() {
            self.feature_state.follow(doc);
        }
        // Each edit's source by id, if a later edit of the transaction
        // didn't remove or redefine it.
        let edits: Vec<(String, &FeatureStateOp)> = (patch.feature_state.iter())
            .filter_map(|op| {
                let (FeatureStateOp::Set { source, .. } | FeatureStateOp::Remove { source, .. }) =
                    op;
                let index = doc.source_index(*source)?;
                Some((doc.source_ids().nth(index)?.to_string(), op))
            })
            .collect();
        for (id, op) in edits {
            let changed = match op {
                FeatureStateOp::Set {
                    source,
                    source_layer,
                    id: feature,
                    state,
                } => (self.feature_state).set(
                    &id,
                    Some(*source),
                    source_layer.as_deref(),
                    feature,
                    state,
                ),
                FeatureStateOp::Remove {
                    source_layer,
                    id: feature,
                    key,
                    ..
                } => self.feature_state.remove(
                    &FeatureTarget {
                        source: id,
                        source_layer: source_layer.clone(),
                        id: feature.clone(),
                    },
                    key.as_deref(),
                ),
            };
            if changed {
                mirror.push(op.clone());
            }
        }
        self.mirror_states(&mirror);
    }

    /// Tells the installed style's state table of `ops`, changes to the
    /// state of sources of [`style_doc`](Self::style_doc): none while
    /// that is a new style waiting for its modules, whose keys aren't the
    /// drawn style's.
    fn mirror_states(&mut self, ops: &[FeatureStateOp]) {
        if let Some(style) = &mut self.style
            && !ops.is_empty()
            && !matches!(self.waiting_style, Some(WaitingStyle { fresh: true, .. }))
        {
            for op in ops {
                style.states.changed(op);
            }
            self.events.wake();
        }
    }
}
